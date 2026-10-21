"""
Studi Kasus 6 - Klasifikasi Driver Peringatan Deforestasi dari Citra Sentinel-1 (radar) + Sentinel-2 (optik)

Materi: Dr. Bart Slachter (Wageningen University) - deforestation alerts & AI driver attribution.

Skrip ini membuat SIMULASI sederhana (data sintetis, bukan citra satelit asli) dari ide inti presentasi:
  * Setiap alert = patch 10x10 piksel dengan 6 band: 2 band Sentinel-1 (VV, VH) + 4 band Sentinel-2 (B2,B3,B4,B8)
  * 7 kelas driver: smallholder_ag, large_ag, road, selective_logging, mining, wildfire, natural
  * Kelas tidak seimbang (pertanian mendominasi) -> accuracy tinggi tapi MACRO-F1 lebih rendah
  * Awan menutup optik -> model gabungan S1+S2 lebih tahan daripada S2 saja
  * Label "unconfident" jika probabilitas tertinggi < ambang
  * Benchmark: model konvensional (Random Forest) vs jaringan saraf (MLP). Catatan: MLP bukan CNN; di dunia nyata
    dipakai CNN pada citra asli -- di sini kita fokus pada logika evaluasi tanpa dependensi deep learning berat.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
RNG = np.random.default_rng(SEED)

CLASSES = ["smallholder_ag", "large_ag", "road", "selective_logging", "mining", "wildfire", "natural"]
PRIOR = np.array([0.30, 0.28, 0.06, 0.09, 0.05, 0.12, 0.10])  # pertanian ~58% -> dominan
H = W = 10
BANDS = ["S1_VV", "S1_VH", "S2_B2", "S2_B3", "S2_B4", "S2_B8"]
FOREST = np.array([-7.5, -13.5, 0.03, 0.05, 0.03, 0.35])
# tanda spektral rata-rata tiap driver pada area gangguan: [VV dB, VH dB, B2, B3, B4, B8]
SIG = {
    "smallholder_ag": [-9.5, -16.0, 0.05, 0.08, 0.09, 0.25],
    "large_ag": [-11.0, -18.0, 0.06, 0.10, 0.12, 0.28],
    "road": [-12.0, -19.0, 0.08, 0.11, 0.14, 0.22],
    "selective_logging": [-9.0, -15.0, 0.04, 0.06, 0.06, 0.30],
    "mining": [-13.0, -20.0, 0.10, 0.13, 0.16, 0.18],
    "wildfire": [-9.5, -15.5, 0.04, 0.05, 0.07, 0.14],  # NIR turun (bekas terbakar)
    "natural": [-12.0, -19.0, 0.04, 0.06, 0.05, 0.19],  # mirip wildfire -> sengaja membingungkan
}
YY, XX = np.mgrid[0:H, 0:W]


def blob(r, jitter=0.0, a=1.0, b=1.0, theta=0.0):
    cy, cx = H / 2 + RNG.normal(0, 0.8), W / 2 + RNG.normal(0, 0.8)
    dy, dx = YY - cy, XX - cx
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    d = np.sqrt((u / a) ** 2 + (v / b) ** 2) + RNG.normal(0, jitter, (H, W))
    return d <= r


def make_mask(cls):
    if cls == "smallholder_ag":
        return blob(RNG.uniform(1.4, 2.6))
    if cls == "large_ag":
        return blob(RNG.uniform(3.8, 5.2))
    if cls == "road":
        th = RNG.uniform(0, np.pi)
        cy, cx = H / 2, W / 2
        return np.abs((XX - cx) * np.sin(th) - (YY - cy) * np.cos(th)) <= 0.7
    if cls == "selective_logging":
        m = RNG.random((H, W)) < 0.10
        th = RNG.uniform(0, np.pi)
        m |= np.abs((XX - W / 2) * np.sin(th) - (YY - H / 2) * np.cos(th)) <= 0.45
        return m
    if cls == "mining":
        return blob(RNG.uniform(2.2, 3.8), jitter=0.4)
    if cls == "wildfire":
        return blob(RNG.uniform(1.8, 4.8), jitter=0.5)
    return blob(RNG.uniform(1.5, 2.2), jitter=0.3, a=2.2, b=0.8, theta=RNG.uniform(0, np.pi))  # natural: memanjang (sungai)


def make_sample(cls_idx, p_cloud):
    cls = CLASSES[cls_idx]
    sample_shift = RNG.normal(0, [1.4, 1.4, 0.012, 0.015, 0.015, 0.04])  # variasi antar-lokasi
    mask = make_mask(cls)
    base = FOREST + sample_shift
    patch = np.zeros((H, W, 6))
    for k in range(6):
        sd = 1.2 if k < 2 else 0.02
        patch[..., k] = np.where(mask, np.array(SIG[cls])[k] + sample_shift[k], base[k]) + RNG.normal(0, sd, (H, W))
    cloudy = RNG.random() < p_cloud
    if cloudy:  # optik tertutup awan
        patch[..., 2:] = RNG.normal(0.55, 0.05, (H, W, 4))
    return patch, cloudy, float(mask.mean())


def make_dataset(n, p_cloud):
    y = RNG.choice(len(CLASSES), n, p=PRIOR)
    X, cloud = [], []
    for c in y:
        p, cl, _ = make_sample(c, p_cloud)
        X.append(p)
        cloud.append(cl)
    return np.array(X), y, np.array(cloud)


def context_features(X):
    """Fitur 'konteks': rata-rata tengah, rata-rata tepi, selisih, dan std per band (24 fitur)."""
    c = X[:, 3:7, 3:7, :].reshape(len(X), -1, 6).mean(1)
    ring = X.copy()
    ring[:, 2:8, 2:8, :] = np.nan
    e = np.nanmean(ring.reshape(len(X), -1, 6), axis=1)
    s = X.reshape(len(X), -1, 6).std(1)
    return np.hstack([c, e, c - e, s])


def band_idx(which):
    return {"S1": [0, 1], "S2": [2, 3, 4, 5], "S1+S2": [0, 1, 2, 3, 4, 5]}[which]


def evaluate(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred), 3), round(f1_score(y_true, y_pred, average="macro"), 3)


def main(n=5000, p_cloud=0.5):
    print(f"Membuat {n} alert sintetis (peluang tertutup awan = {p_cloud:.0%}) ...")
    X, y, cloud = make_dataset(n, p_cloud)
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    print("Distribusi kelas:", dict(zip(CLASSES, map(int, np.bincount(y)))))

    results = []

    def rf():
        return RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)

    # --- ablasi sensor: S1 saja / S2 saja / S1+S2 (fitur konteks + RF) ---
    abl_models = {}
    for which in ["S1", "S2", "S1+S2"]:
        sub = X[..., band_idx(which)]
        nb = sub.shape[-1]
        c = sub[:, 3:7, 3:7, :].reshape(n, -1, nb).mean(1)
        ring = sub.copy()
        ring[:, 2:8, 2:8, :] = np.nan
        e = np.nanmean(ring.reshape(n, -1, nb), axis=1)
        f = np.hstack([c, e, c - e, sub.reshape(n, -1, nb).std(1)])
        m = rf().fit(f[tr], y[tr])
        abl_models[which] = (m, f)
        acc, mf1 = evaluate(y[te], m.predict(f[te]))
        acc_cloud, _ = evaluate(y[te][cloud[te]], m.predict(f[te][cloud[te]]))
        results.append(dict(model=f"RF konteks | {which}", accuracy=acc, macro_f1=mf1, acc_saat_berawan=acc_cloud))

    # --- benchmark konvensional vs jaringan saraf pada S1+S2 ---
    Fraw = X.reshape(n, -1)
    rf_raw = rf().fit(Fraw[tr], y[tr])
    mlp = make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=300, early_stopping=True, random_state=SEED)).fit(Fraw[tr], y[tr])
    bench = {
        "RF konteks | S1+S2": abl_models["S1+S2"],
        "RF piksel mentah | S1+S2": (rf_raw, Fraw),
        "MLP piksel mentah | S1+S2": (mlp, Fraw),
    }
    preds = {}
    for name, (m, F) in bench.items():
        p = m.predict(F[te])
        if name == "RF konteks | S1+S2":
            preds[name] = p
            continue
        preds[name] = p
        acc, mf1 = evaluate(y[te], p)
        acc_cloud, _ = evaluate(y[te][cloud[te]], p[cloud[te]])
        results.append(dict(model=name, accuracy=acc, macro_f1=mf1, acc_saat_berawan=acc_cloud))
    res = pd.DataFrame(results)
    print("\n== Hasil (data uji) ==")
    print(res.to_string(index=False))

    # --- model terbaik berdasar macro-F1 ---
    best_name = res[res.model.str.contains("S1\\+S2")].sort_values("macro_f1", ascending=False).iloc[0].model
    best_pred = preds[best_name]
    print(f"\nModel terbaik (macro-F1): {best_name}")
    print(classification_report(y[te], best_pred, target_names=CLASSES, digits=3))
    cm = confusion_matrix(y[te], best_pred, normalize="true")

    # --- label 'unconfident' (probabilitas tertinggi < 0.5) ---
    best_model, best_F = bench[best_name]
    proba = best_model.predict_proba(best_F[te])
    conf = proba.max(1)
    confident = conf >= 0.5
    acc_conf = accuracy_score(y[te][confident], proba.argmax(1)[confident])
    print(f"Unconfident (<0.5): {1 - confident.mean():.1%} alert | akurasi pada alert yang yakin: {acc_conf:.3f}")

    # --- plot ---
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    a = ax[0].imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax[0].set_xticks(range(7), CLASSES, rotation=60, ha="right")
    ax[0].set_yticks(range(7), CLASSES)
    for i in range(7):
        for j in range(7):
            ax[0].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.5 else "black")
    ax[0].set_title(f"Confusion matrix (recall) - {best_name}", fontsize=10)
    f1s = f1_score(y[te], best_pred, average=None)
    ax[1].barh(CLASSES, f1s, color=["#B0B0B0"] * 7)
    for i, v in enumerate(f1s):
        if v == f1s.min() or CLASSES[i] in ("wildfire", "natural"):
            ax[1].patches[i].set_color("#D55E00")
    ax[1].axvline(f1s.mean(), ls="--", color="k")
    ax[1].set_title(f"F1 per kelas (garis = macro-F1 {f1s.mean():.2f})")
    ax[1].set_xlim(0, 1)
    abl = res.iloc[:3]
    xs = np.arange(3)
    ax[2].bar(xs - 0.2, abl.accuracy, 0.4, label="akurasi semua", color="#0072B2")
    ax[2].bar(xs + 0.2, abl.acc_saat_berawan, 0.4, label="akurasi saat berawan", color="#E69F00")
    ax[2].set_xticks(xs, ["S1 (radar)", "S2 (optik)", "S1+S2"])
    ax[2].set_ylim(0, 1)
    ax[2].legend(frameon=False)
    ax[2].set_title("Ablasi sensor: radar menolong saat awan")
    plt.tight_layout()
    plt.savefig(OUT / "evaluasi_driver.png", dpi=130)

    # --- contoh patch ---
    fig, axs = plt.subplots(2, 7, figsize=(17, 5))
    for j, cname in enumerate(CLASSES):
        p, _, _ = make_sample(j, 0.0)
        axs[0, j].imshow(p[..., 1], cmap="gray")
        axs[0, j].set_title(cname, fontsize=9)
        rgb = np.clip(p[..., [4, 3, 2]] * 5, 0, 1)
        axs[1, j].imshow(rgb)
        for r in (0, 1):
            axs[r, j].axis("off")
    axs[0, 0].text(-3, 5, "S1 VH", rotation=90, va="center")
    axs[1, 0].text(-3, 5, "S2 RGB", rotation=90, va="center")
    plt.tight_layout()
    plt.savefig(OUT / "contoh_patch.png", dpi=110)

    res.to_csv(OUT / "hasil_model.csv", index=False)
    (OUT / "ringkasan.json").write_text(
        json.dumps(
            {
                "model_terbaik": best_name,
                "accuracy": float(accuracy_score(y[te], best_pred)),
                "macro_f1": float(f1s.mean()),
                "f1_per_kelas": dict(zip(CLASSES, map(lambda v: round(float(v), 3), f1s))),
                "persen_unconfident": round(float(1 - confident.mean()), 3),
                "akurasi_yang_yakin": round(float(acc_conf), 3),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("Output tersimpan di", OUT)


if __name__ == "__main__":
    main()
