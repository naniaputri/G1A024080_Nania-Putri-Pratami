"""
Studi Kasus 1 - Prediksi Keterlambatan Pembayaran Tagihan (CRISP-DM + 4 tingkat analitik)

Materi: Kuliah praktisi "Data & AI di dunia profesional" (Andria Arisal, BRIN)

Alur:
  1. Business & data understanding  -> data sintetis yang SENGAJA berantakan
  2. Data preparation               -> cleaning (duplikat, typo, anomali, missing)
  3. Analitik deskriptif            -> "apa yang terjadi?"
  4. Analitik diagnostik            -> "mengapa terjadi?"
  5. Analitik prediktif             -> "apa yang akan terjadi?" + demo DATA LEAKAGE
  6. Analitik preskriptif           -> "tindakan apa yang terbaik dengan batasan sumber daya?"
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

NUM_COLS = ["customer_age", "tenure_months", "num_prev_late", "invoice_amount", "payment_term_days"]
CAT_COLS = ["region", "industry"]


# --------------------------------------------------------------------------- #
# 1. Data understanding: data mentah yang berantakan
# --------------------------------------------------------------------------- #
def generate_raw(n: int = 3000) -> pd.DataFrame:
    region = RNG.choice(["Bengkulu", "Rejang Lebong", "Kepahiang", "Seluma", "Mukomuko"], n)
    industry = RNG.choice(["Retail", "Konstruksi", "Pertanian", "Jasa"], n, p=[0.35, 0.2, 0.25, 0.2])
    age = RNG.normal(40, 10, n).round()
    tenure = RNG.integers(1, 120, n).astype(float)
    prev_late = RNG.poisson(1.2, n).astype(float)
    amount = RNG.lognormal(15, 0.6, n).round(-3)
    term = RNG.choice([14, 30, 45, 60], n).astype(float)

    logit = (
        -2.0
        + 0.7 * prev_late
        - 0.02 * tenure
        + 0.25 * (term == 60)
        + 0.4 * (industry == "Konstruksi")
        + 0.6 * (amount > 8e6)
    )
    p_late = 1 / (1 + np.exp(-logit))
    late = (RNG.random(n) < p_late).astype(int)
    # kolom BOCOR: hanya diketahui SETELAH tagihan jatuh tempo (berasal dari target)
    days_late = np.where(late == 1, RNG.poisson(7, n) + 1, 0)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "customer_age": age,
            "tenure_months": tenure,
            "num_prev_late": prev_late,
            "invoice_amount": amount,
            "payment_term_days": term,
            "region": region,
            "industry": industry,
            "days_late": days_late,
            "late": late,
        }
    )

    # --- kotori data (seperti data nyata) ---
    for col, frac in [("tenure_months", 0.08), ("invoice_amount", 0.04), ("customer_age", 0.03)]:
        df.loc[df.sample(frac=frac, random_state=1).index, col] = np.nan
    df.loc[df.sample(frac=0.01, random_state=2).index, "customer_age"] = RNG.choice([-1, 0, 199, 250], size=int(n * 0.01))
    df.loc[df.sample(frac=0.005, random_state=3).index, "invoice_amount"] = -5_000_000
    typo = df.sample(frac=0.08, random_state=4).index
    df.loc[typo, "region"] = df.loc[typo, "region"].map(lambda s: s.upper() if RNG.random() < 0.5 else f" {s} ")
    dup = df.sample(frac=0.03, random_state=5)
    return pd.concat([df, dup], ignore_index=True)


def quality_report(df: pd.DataFrame) -> dict:
    return {
        "baris": int(len(df)),
        "duplikat": int(df.duplicated().sum()),
        "missing_total": int(df[NUM_COLS].isna().sum().sum()),
        "umur_tidak_wajar": int(((df.customer_age < 17) | (df.customer_age > 90)).sum()),
        "nominal_tidak_valid": int((df.invoice_amount <= 0).sum()),
        "label_region_unik": int(df.region.nunique()),
    }


# --------------------------------------------------------------------------- #
# 2. Data preparation
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df["region"] = df["region"].str.strip().str.title()
    # aturan bisnis: umur 17-90, nominal > 0; yang melanggar dijadikan NaN (diimputasi nanti di pipeline)
    df.loc[(df.customer_age < 17) | (df.customer_age > 90), "customer_age"] = np.nan
    df.loc[df.invoice_amount <= 0, "invoice_amount"] = np.nan
    return df.reset_index(drop=True)


def build_pipeline(features_num, features_cat) -> Pipeline:
    # imputer berada DI DALAM pipeline -> statistik hanya dihitung dari data latih (anti-leakage)
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ]
    )
    return Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1))])


def main():
    raw = generate_raw()
    before = quality_report(raw)
    df = clean(raw)
    after = quality_report(df)
    print("== Data quality sebelum :", before)
    print("== Data quality sesudah :", after)

    # ------------------ 3. Deskriptif ------------------
    desc = {
        "late_rate_total": round(df.late.mean(), 3),
        "late_rate_per_region": df.groupby("region").late.mean().round(3).to_dict(),
        "late_rate_per_industri": df.groupby("industry").late.mean().round(3).to_dict(),
    }
    print("\n== [Deskriptif] Tingkat keterlambatan:", desc["late_rate_total"])

    # ------------------ 4. Diagnostik ------------------
    df["prev_late_bucket"] = pd.cut(df.num_prev_late, [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
    diag_prev = df.groupby("prev_late_bucket", observed=True).late.mean().round(3)
    diag_term = df.groupby("payment_term_days").late.mean().round(3)
    print("== [Diagnostik] Late-rate menurut riwayat telat sebelumnya:\n", diag_prev.to_string())

    # ------------------ 5. Prediktif ------------------
    train, test = train_test_split(df, test_size=0.2, stratify=df.late, random_state=42)
    feats = NUM_COLS + CAT_COLS
    model = build_pipeline(NUM_COLS, CAT_COLS).fit(train[feats], train.late)
    auc_clean = roc_auc_score(test.late, model.predict_proba(test[feats])[:, 1])

    # demo kebocoran data: memasukkan days_late (informasi masa depan)
    leak_feats = feats + ["days_late"]
    leak_model = build_pipeline(NUM_COLS + ["days_late"], CAT_COLS).fit(train[leak_feats], train.late)
    auc_leak = roc_auc_score(test.late, leak_model.predict_proba(test[leak_feats])[:, 1])
    print(f"\n== [Prediktif] AUC tanpa leakage = {auc_clean:.3f} | dengan leakage = {auc_leak:.3f} (terlalu bagus -> curiga!)")

    # ------------------ 6. Preskriptif ------------------
    test = test.copy()
    test["p_late"] = model.predict_proba(test[feats])[:, 1]
    test["amount_f"] = test.invoice_amount.fillna(test.invoice_amount.median())
    test["benefit"] = test.p_late * test.amount_f * 0.35  # asumsi: pengingat memulihkan 35% nilai berisiko
    test["cost"] = np.where(test.amount_f > 8e6, 2, 1)  # tagihan besar butuh 2 unit jam kerja
    BUDGET = 150
    ranked = test.assign(ratio=test.benefit / test.cost).sort_values("ratio", ascending=False)
    chosen = ranked[ranked.cost.cumsum() <= BUDGET]
    naive = test.sample(len(chosen), random_state=0)
    presc = {
        "budget_unit": BUDGET,
        "pelanggan_dihubungi": int(len(chosen)),
        "ekspektasi_nilai_terselamatkan_rp": round(float(chosen.benefit.sum())),
        "pembanding_acak_rp": round(float(naive.benefit.sum())),
    }
    print("== [Preskriptif]", presc)

    # ------------------ visualisasi ------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    df.groupby("region").late.mean().sort_values().plot.barh(ax=ax[0, 0], color="#4C78A8")
    ax[0, 0].set_title("Deskriptif: late-rate per wilayah")
    diag_prev.plot.bar(ax=ax[0, 1], color="#F58518", rot=0)
    ax[0, 1].set_title("Diagnostik: riwayat telat -> peluang telat")
    ax[1, 0].bar(["Tanpa leakage", "Dengan leakage"], [auc_clean, auc_leak], color=["#54A24B", "#E45756"])
    ax[1, 0].set_ylim(0.5, 1.05)
    ax[1, 0].set_title("Prediktif: AUC (leakage = terlalu sempurna)")
    ax[1, 1].bar(["Acak", "Optimasi (preskriptif)"], [presc["pembanding_acak_rp"] / 1e6, presc["ekspektasi_nilai_terselamatkan_rp"] / 1e6], color=["#9D9D9D", "#54A24B"])
    ax[1, 1].set_title("Preskriptif: nilai terselamatkan (juta Rp)")
    plt.tight_layout()
    plt.savefig(OUT / "empat_tingkat_analitik.png", dpi=130)

    summary = {
        "kualitas_data_sebelum": before,
        "kualitas_data_sesudah": after,
        "deskriptif": desc,
        "diagnostik_late_rate_riwayat_telat": {str(k): float(v) for k, v in diag_prev.items()},
        "diagnostik_late_rate_tenor": {str(k): float(v) for k, v in diag_term.items()},
        "prediktif": {"auc_tanpa_leakage": round(auc_clean, 3), "auc_dengan_leakage": round(auc_leak, 3)},
        "preskriptif": presc,
    }
    (OUT / "ringkasan.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    df.drop(columns=["prev_late_bucket"]).to_csv(OUT / "data_bersih.csv", index=False)
    print("\nOutput tersimpan di", OUT)


if __name__ == "__main__":
    main()
