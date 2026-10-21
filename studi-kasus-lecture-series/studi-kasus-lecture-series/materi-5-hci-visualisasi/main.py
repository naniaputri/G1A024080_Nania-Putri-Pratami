"""
Studi Kasus 5 - Dashboard Penjualan: HCI + Visualisasi Data + AI yang Transparan

Materi: Human-Computer Interaction & Data Visualization "Make Sense of Data with Analytics and AI" (Arafat Febriandirza)

Prinsip yang diterapkan:
  * Ringkasan (KPI) di bagian paling atas -> detail di bawah         (overview first)
  * Jenis grafik sesuai tujuan: bar = bandingkan, line = tren, scatter = hubungan, donut = porsi sederhana
  * Warna: abu-abu netral + SATU warna aksen untuk menarik perhatian (pre-attentive); palet ramah buta warna
  * Tanpa hiasan yang tidak berfungsi (chartjunk)
  * AI "kotak hitam" dibuat transparan: model what-if yang bisa dibaca (elastisitas diskon) + tabel sensitivitas
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

CITIES = ["Bengkulu", "Curup", "Manna", "Mukomuko", "Arga Makmur"]
PRODUCTS = {"Kopi": (45, 0.35), "Beras": (14, 0.55), "Minyak Sawit": (18, 0.45), "Gula": (16, 0.65)}  # (harga ribu, rasio biaya)
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]  # palet Okabe-Ito (ramah buta warna)
GRAY, ACCENT = "#B0B0B0", "#D55E00"
B_TRUE = 3.0  # elastisitas "sebenarnya" pada data sintetis (log-unit per 100% diskon)


def make_sales() -> pd.DataFrame:
    rows = []
    for m in pd.period_range("2024-01", "2025-12", freq="M"):
        season = 1 + 0.25 * np.sin(2 * np.pi * (m.month - 3) / 12)
        trend = 1 + 0.015 * (m.year - 2024) * 12 + 0.01 * m.month
        for city, cf in zip(CITIES, [1.6, 1.2, 0.8, 0.7, 0.9]):
            for prod, (price, cost) in PRODUCTS.items():
                disc = float(RNG.choice([0, 0.05, 0.10, 0.15, 0.20, 0.30]))
                units = 120 * cf * season * trend * np.exp(B_TRUE * disc) * RNG.normal(1, 0.08)
                rows.append(dict(bulan=m.to_timestamp(), kota=city, produk=prod, harga=price, diskon=disc, unit=units, biaya_rasio=cost))
    df = pd.DataFrame(rows)
    df["omzet"] = df.unit * df.harga * (1 - df.diskon) / 1000  # juta rupiah
    return df


def kpi(ax, label, value, sub, color="#222"):
    ax.axis("off")
    ax.text(0.0, 0.78, label, fontsize=11, color="#666", transform=ax.transAxes)
    ax.text(0.0, 0.30, value, fontsize=24, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.0, 0.02, sub, fontsize=9, color="#888", transform=ax.transAxes)


def clean_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def dashboard(df):
    y25, y24 = df[df.bulan.dt.year == 2025], df[df.bulan.dt.year == 2024]
    growth = y25.omzet.sum() / y24.omzet.sum() - 1
    by_city = df.groupby("kota").omzet.sum().sort_values()
    top_city = by_city.index[-1]
    monthly = df.groupby("bulan").omzet.sum()

    fig = plt.figure(figsize=(15, 9.5))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.7, 2, 2], hspace=0.5, wspace=0.35)
    # ---- 1) RINGKASAN DI ATAS ----
    kpi(fig.add_subplot(gs[0, 0]), "Total omzet 2025", f"Rp {y25.omzet.sum():,.0f} jt", "2 tahun: 2024-2025")
    kpi(fig.add_subplot(gs[0, 1]), "Pertumbuhan YoY", f"{growth:+.1%}", "2025 vs 2024", ACCENT if growth < 0 else "#009E73")
    kpi(fig.add_subplot(gs[0, 2]), "Kota teratas", top_city, f"{by_city.iloc[-1] / by_city.sum():.0%} dari total omzet")
    kpi(fig.add_subplot(gs[0, 3]), "Rata-rata diskon", f"{df.diskon.mean():.1%}", "semua transaksi")

    # ---- 2) BAR: perbandingan antar-kota; hanya yang tertinggi diberi warna aksen ----
    ax = fig.add_subplot(gs[1, :2])
    ax.barh(by_city.index, by_city.values, color=[GRAY] * (len(by_city) - 1) + [ACCENT])
    ax.set_title("Omzet per kota (juta Rp) - bandingkan", loc="left", fontweight="bold")
    clean_axes(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    # ---- 3) LINE: tren waktu ----
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(monthly.index, monthly.values, color=OKABE[0], lw=2)
    ax.scatter(monthly.idxmax(), monthly.max(), color=ACCENT, zorder=5)
    ax.annotate("puncak", (monthly.idxmax(), monthly.max()), xytext=(8, 6), textcoords="offset points", color=ACCENT)
    ax.set_title("Omzet bulanan (juta Rp) - tren", loc="left", fontweight="bold")
    clean_axes(ax)

    # ---- 4) SCATTER: hubungan diskon vs unit ----
    ax = fig.add_subplot(gs[2, :2])
    for (prod, g), c in zip(df.groupby("produk"), OKABE):
        ax.scatter(g.diskon * 100, g.unit, s=10, alpha=0.5, color=c, label=prod)
    ax.set_xlabel("diskon (%)")
    ax.set_ylabel("unit terjual")
    ax.set_title("Diskon vs unit terjual - hubungan", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    clean_axes(ax)

    # ---- 5) DONUT: porsi sederhana (4 kategori saja) ----
    ax = fig.add_subplot(gs[2, 2:])
    share = df.groupby("produk").omzet.sum()
    ax.pie(share, labels=share.index, colors=OKABE, startangle=90, wedgeprops=dict(width=0.4, edgecolor="white"), autopct="%1.0f%%", pctdistance=0.8, textprops={"fontsize": 9})
    ax.set_title("Porsi omzet per produk - bagian dari keseluruhan", loc="left", fontweight="bold")
    fig.suptitle("Dashboard Penjualan Sembako & Komoditas - Bengkulu (data sintetis)", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "dashboard.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fit_elasticity(df):
    """Model transparan: log(unit) = a + b*diskon + efek produk + efek kota. Koefisien b bisa dibaca manusia."""
    X = pd.get_dummies(df[["diskon", "produk", "kota"]], drop_first=True).astype(float)
    m = LinearRegression().fit(X, np.log(df.unit))
    return m, X.columns, float(m.coef_[list(X.columns).index("diskon")])


def what_if(df, b):
    """Profit(d) = unit(d) * (harga*(1-d) - biaya). Unit(d) = unit0 * exp(b*d). Cari diskon terbaik 0-40%."""
    grid = np.linspace(0, 0.4, 41)
    rec, curves = [], {}
    for prod, (price, cost_ratio) in PRODUCTS.items():
        unit0 = df[(df.produk == prod) & (df.diskon == 0)].unit.mean()
        cost = price * cost_ratio
        profit = unit0 * np.exp(b * grid) * (price * (1 - grid) - cost) / 1000  # juta Rp/bulan/kota
        curves[prod] = profit
        i = int(profit.argmax())
        rec.append(dict(produk=prod, diskon_optimal_pct=round(grid[i] * 100), profit_optimal_jt=round(profit[i], 2), profit_tanpa_diskon_jt=round(profit[0], 2), profit_diskon_20pct_jt=round(profit[20], 2)))
    return grid, curves, pd.DataFrame(rec)


def whatif_figure(grid, curves, rec):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (prod, p), c in zip(curves.items(), OKABE):
        ax.plot(grid * 100, p, color=c, lw=2, label=prod)
        i = int(p.argmax())
        ax.scatter(grid[i] * 100, p[i], color=c, s=60, zorder=5)
    ax.set_xlabel("diskon (%)")
    ax.set_ylabel("profit (juta Rp / bulan / kota)")
    ax.set_title("Simulasi what-if: diskon mana yang memaksimalkan profit?\n(titik = rekomendasi model; garis bisa digeser pengguna di aplikasi interaktif)", loc="left", fontsize=11)
    ax.legend(frameon=False)
    clean_axes(ax)
    fig.savefig(OUT / "whatif_diskon.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    df = make_sales()
    dashboard(df)
    model, cols, b = fit_elasticity(df)
    print(f"== Model transparan: kenaikan diskon 10 poin -> unit naik ~{(np.exp(b * 0.10) - 1):.0%} (estimasi b={b:.2f}, sebenarnya {B_TRUE})")
    grid, curves, rec = what_if(df, b)
    whatif_figure(grid, curves, rec)
    print("\n== Rekomendasi diskon (what-if) ==")
    print(rec.to_string(index=False))
    rec.to_csv(OUT / "rekomendasi_diskon.csv", index=False)
    df.to_csv(OUT / "data_penjualan_sintetis.csv", index=False)
    print("\nOutput tersimpan di", OUT)


if __name__ == "__main__":
    main()
