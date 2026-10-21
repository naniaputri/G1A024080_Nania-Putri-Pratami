"""
Studi Kasus 3 - Workflow Machine Learning end-to-end: Persetujuan Kredit, Harga Rumah, Segmentasi Pelanggan

Materi: Pengenalan Machine Learning
  - Rule-based vs ML
  - 7 langkah workflow (collect -> preprocess -> split -> pilih algoritma -> train -> evaluate -> deploy)
  - Supervised (klasifikasi + regresi) dan Unsupervised (clustering)
  - Overfitting vs underfitting
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)


# ----------------------------- 1. Data collection ----------------------------- #
def make_loans(n=2000) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "income_juta": RNG.gamma(4, 2.2, n),
            "credit_score": RNG.normal(650, 80, n).clip(300, 850),
            "debt_ratio": RNG.beta(2, 5, n),
            "employment_years": RNG.integers(0, 25, n).astype(float),
            "loan_amount_juta": RNG.gamma(3, 25, n),
        }
    )
    z = (
        -1.0
        + 0.18 * df.income_juta
        + 0.012 * (df.credit_score - 650)
        - 5 * df.debt_ratio
        + 0.05 * df.employment_years
        - 0.006 * df.loan_amount_juta
    )
    df["approved"] = (RNG.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def make_houses(n=1500) -> pd.DataFrame:
    city = RNG.choice(["Bengkulu", "Padang", "Jambi"], n, p=[0.4, 0.3, 0.3])
    area = RNG.uniform(30, 250, n)
    beds = np.clip((area / 45 + RNG.normal(0, 0.7, n)).round(), 1, 6)
    age = RNG.integers(0, 40, n).astype(float)
    base = pd.Series(city).map({"Bengkulu": 6.0, "Padang": 7.5, "Jambi": 6.5}).to_numpy()  # juta/m2
    price = (base * area + 40 * beds - 4 * age) * RNG.normal(1, 0.06, n)  # juta rupiah
    return pd.DataFrame({"luas_m2": area, "kamar_tidur": beds, "usia_tahun": age, "kota": city, "harga_juta": price})


def make_customers(n=900) -> pd.DataFrame:
    seg = RNG.choice(3, n, p=[0.5, 0.3, 0.2])
    rec = np.array([40, 10, 5])[seg] + RNG.normal(0, 4, n)
    freq = np.array([2, 8, 20])[seg] + RNG.normal(0, 1.5, n)
    mon = np.array([150, 600, 2500])[seg] + RNG.normal(0, 80, n)
    return pd.DataFrame({"recency_hari": rec.clip(1), "frekuensi": freq.clip(1), "monetary_ribu": mon.clip(10)})


def metrics_cls(y, p):
    return {
        "accuracy": round(accuracy_score(y, p), 3),
        "precision": round(precision_score(y, p), 3),
        "recall": round(recall_score(y, p), 3),
        "f1": round(f1_score(y, p), 3),
    }


def main():
    summary = {}

    # =========================== A. KLASIFIKASI: kredit =========================== #
    loans = make_loans()
    X, y = loans.drop(columns="approved"), loans.approved
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)  # split 80/20

    rule = ((Xte.income_juta >= 8) & (Xte.credit_score >= 650)).astype(int)  # pemrograman tradisional (rule-based)
    models = {
        "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
        "Decision Tree (depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    cls = {"Rule-based (manual)": metrics_cls(yte, rule)}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        cls[name] = metrics_cls(yte, m.predict(Xte))
    print("== [Klasifikasi kredit] ==")
    print(pd.DataFrame(cls).T.to_string())
    best = max(models, key=lambda k: cls[k]["f1"])
    print("Model terbaik (F1):", best)
    print("Confusion matrix:\n", confusion_matrix(yte, models[best].predict(Xte)))
    summary["klasifikasi"] = cls

    # --- overfitting vs underfitting: kedalaman pohon ---
    noisy = ytr.copy()
    flip = RNG.random(len(noisy)) < 0.10  # 10% label salah (noise nyata)
    noisy[flip] = 1 - noisy[flip]
    depths = range(1, 21)
    tr_acc, te_acc = [], []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Xtr, noisy)
        tr_acc.append(t.score(Xtr, noisy))
        te_acc.append(t.score(Xte, yte))
    best_d = list(depths)[int(np.argmax(te_acc))]
    print(f"Depth terbaik={best_d} | depth=20: train={tr_acc[-1]:.2f} test={te_acc[-1]:.2f} (overfitting)")
    summary["overfitting"] = {"depth_terbaik": best_d, "train_acc_depth20": round(tr_acc[-1], 3), "test_acc_depth20": round(te_acc[-1], 3)}

    # ============================ B. REGRESI: rumah ============================= #
    houses = make_houses()
    Xh, yh = houses.drop(columns="harga_juta"), houses.harga_juta
    Xhtr, Xhte, yhtr, yhte = train_test_split(Xh, yh, test_size=0.2, random_state=42)
    pre = ColumnTransformer([("cat", OneHotEncoder(), ["kota"])], remainder="passthrough")
    reg = {
        "Linear Regression": Pipeline([("pre", pre), ("m", LinearRegression())]),
        "Random Forest": Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]),
    }
    reg_res = {}
    for name, m in reg.items():
        m.fit(Xhtr, yhtr)
        pr = m.predict(Xhte)
        reg_res[name] = {"MAE_juta": round(mean_absolute_error(yhte, pr), 1), "R2": round(r2_score(yhte, pr), 3)}
    print("\n== [Regresi harga rumah] ==")
    print(pd.DataFrame(reg_res).T.to_string())
    summary["regresi"] = reg_res

    # --- deploy for inference: simpan model + prediksi data baru ---
    joblib.dump(reg["Linear Regression"], OUT / "model_harga_rumah.joblib")
    loaded = joblib.load(OUT / "model_harga_rumah.joblib")
    new_house = pd.DataFrame([{"luas_m2": 120, "kamar_tidur": 3, "usia_tahun": 5, "kota": "Bengkulu"}])
    pred_new = float(loaded.predict(new_house)[0])
    print(f"Prediksi rumah baru (120 m2, 3 KT, 5 th, Bengkulu): Rp {pred_new:,.0f} juta")
    summary["deploy_contoh"] = {"prediksi_juta": round(pred_new, 1)}

    # ========================= C. UNSUPERVISED: clustering ======================== #
    cust = make_customers()
    Xc = StandardScaler().fit_transform(cust)
    sil = {k: silhouette_score(Xc, KMeans(k, n_init=10, random_state=42).fit_predict(Xc)) for k in range(2, 7)}
    k_best = max(sil, key=sil.get)
    labels = KMeans(k_best, n_init=10, random_state=42).fit_predict(Xc)
    cust["segmen"] = labels
    prof = cust.groupby("segmen").mean().round(1)
    print(f"\n== [Clustering pelanggan] k terbaik (silhouette) = {k_best} ==")
    print(prof.to_string())
    summary["clustering"] = {"k_terbaik": int(k_best), "silhouette": {str(k): round(v, 3) for k, v in sil.items()}}

    # ================================ plot ================================= #
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(list(depths), tr_acc, "o-", label="train")
    ax[0].plot(list(depths), te_acc, "s-", label="test")
    ax[0].axvline(best_d, ls="--", color="gray")
    ax[0].set_xlabel("kedalaman pohon")
    ax[0].set_ylabel("akurasi")
    ax[0].set_title("Overfitting: train naik, test turun")
    ax[0].legend()
    pr = reg["Linear Regression"].predict(Xhte)
    ax[1].scatter(yhte, pr, s=8, alpha=0.5)
    lim = [yhte.min(), yhte.max()]
    ax[1].plot(lim, lim, "r--")
    ax[1].set_xlabel("harga aktual (juta)")
    ax[1].set_ylabel("prediksi (juta)")
    ax[1].set_title("Regresi harga rumah")
    xy = PCA(2).fit_transform(Xc)
    ax[2].scatter(xy[:, 0], xy[:, 1], c=labels, cmap="viridis", s=8)
    ax[2].set_title(f"K-Means segmentasi pelanggan (k={k_best}, PCA 2D)")
    plt.tight_layout()
    plt.savefig(OUT / "ringkasan_ml.png", dpi=130)
    (OUT / "ringkasan.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nOutput tersimpan di", OUT)


if __name__ == "__main__":
    main()
