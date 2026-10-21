"""
Studi Kasus 4 - GIS untuk Keputusan Bisnis & Kota Pintar (Bengkulu & model "Snap Send Solve")

Materi: Kuliah praktisi GIS (Widi Prima Deva) - GIS sebagai alat data-driven decision making,
        anak IT = "GIS Developer" yang membuat tools sendiri dengan Python & data spasial.

Bagian A (ritel, seperti Indomaret): pilih lokasi gerai baru yang memaksimalkan populasi terlayani,
         dengan aturan jarak minimum antar gerai agar tidak saling mematikan pasar.
Bagian B (smart city, seperti Snap Send Solve): laporan warga -> prioritas -> penugasan ke pos petugas.
Semua data SINTETIS di sekitar Kota Bengkulu. Tanpa library GIS berat: hanya numpy/pandas (haversine).
Hasil: GeoJSON (bisa dibuka di QGIS / geojson.io) + peta PNG.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)
CENTER = (-3.80, 102.26)  # (lat, lon) sekitar Kota Bengkulu
R_EARTH = 6371.0


def haversine_matrix(lat1, lon1, lat2, lon2):
    """Jarak (km) antara setiap titik set-1 (n) dan set-2 (m) -> matriks n x m."""
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dphi = p2 - p1
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def jitter(n, scale=0.004):
    return RNG.normal(0, scale, n)


# ------------------------------- Bagian A ------------------------------------ #
def part_a(radius_km=0.8, min_gap_km=0.5, n_new=5):
    # populasi: 5 kelurahan sintetis (titik rumah tangga berbobot jumlah jiwa)
    centers = np.array([[CENTER[0] + RNG.uniform(-0.04, 0.04), CENTER[1] + RNG.uniform(-0.04, 0.04)] for _ in range(5)])
    pts = []
    for c in centers:
        k = 500
        pts.append(np.c_[c[0] + jitter(k, 0.010), c[1] + jitter(k, 0.010), RNG.integers(2, 7, k)])
    pop = pd.DataFrame(np.vstack(pts), columns=["lat", "lon", "jiwa"])

    stores = pop.sample(18, random_state=1)[["lat", "lon"]].reset_index(drop=True)
    stores[["lat", "lon"]] += RNG.normal(0, 0.002, (18, 2))
    cand = pop.sample(80, random_state=2)[["lat", "lon"]].reset_index(drop=True)
    cand[["lat", "lon"]] += RNG.normal(0, 0.002, (80, 2))

    d_pop_store = haversine_matrix(pop.lat.values, pop.lon.values, stores.lat.values, stores.lon.values)
    covered = (d_pop_store <= radius_km).any(axis=1)  # sudah terlayani gerai eksisting
    d_pop_cand = haversine_matrix(pop.lat.values, pop.lon.values, cand.lat.values, cand.lon.values)
    d_cand_store = haversine_matrix(cand.lat.values, cand.lon.values, stores.lat.values, stores.lon.values)
    d_cand_cand = haversine_matrix(cand.lat.values, cand.lon.values, cand.lat.values, cand.lon.values)

    valid = d_cand_store.min(axis=1) >= min_gap_km  # aturan bisnis: jarak minimum ke gerai eksisting
    chosen, gains = [], []
    for _ in range(n_new):  # algoritma greedy (maximal coverage)
        gain = np.array([(pop.jiwa.values * ((d_pop_cand[:, j] <= radius_km) & ~covered)).sum() if valid[j] else -1 for j in range(len(cand))])
        j = int(gain.argmax())
        if gain[j] <= 0:
            break
        chosen.append(j)
        gains.append(int(gain[j]))
        covered |= d_pop_cand[:, j] <= radius_km
        valid &= d_cand_cand[j] >= min_gap_km  # jangan terlalu dekat dengan gerai baru lainnya

    total = pop.jiwa.sum()
    base_cov = (pop.jiwa * (d_pop_store <= radius_km).any(axis=1)).sum() / total
    new_cov = (pop.jiwa * covered).sum() / total
    print("== [A] Analisis lokasi gerai ==")
    print(f"Populasi total: {int(total):,} jiwa | cakupan awal: {base_cov:.1%} -> setelah {len(chosen)} gerai baru: {new_cov:.1%}")
    for rank, (j, g) in enumerate(zip(chosen, gains), 1):
        print(f"  #{rank}: lat={cand.lat[j]:.4f}, lon={cand.lon[j]:.4f} | tambahan jiwa terlayani: {g:,}")

    sel = cand.loc[chosen].assign(gain_jiwa=gains, urutan=range(1, len(chosen) + 1))
    return pop, stores, cand, sel, {"cakupan_awal": round(base_cov, 3), "cakupan_akhir": round(new_cov, 3), "lokasi_terpilih": len(chosen), "tambahan_jiwa": int(sum(gains))}


# ------------------------------- Bagian B ------------------------------------ #
def part_b(n_reports=60, top_k=12):
    kinds = {"pohon_miring": 5, "jalan_berlubang": 4, "lampu_mati": 2, "rumput_berm": 1}  # bobot urgensi
    rep = pd.DataFrame(
        {
            "lat": CENTER[0] + RNG.normal(0, 0.02, n_reports),
            "lon": CENTER[1] + RNG.normal(0, 0.02, n_reports),
            "jenis": RNG.choice(list(kinds), n_reports),
            "hari_menunggu": RNG.integers(0, 14, n_reports),
        }
    )
    rep["skor"] = rep.jenis.map(kinds) * (1 + rep.hari_menunggu / 7)
    bases = pd.DataFrame({"nama": ["Pos Utara", "Pos Tengah", "Pos Selatan"], "lat": [CENTER[0] + 0.03, CENTER[0], CENTER[0] - 0.03], "lon": [CENTER[1] - 0.01, CENTER[1] + 0.01, CENTER[1]]})
    d = haversine_matrix(rep.lat.values, rep.lon.values, bases.lat.values, bases.lon.values)
    rep["pos"] = bases.nama.values[d.argmin(axis=1)]
    rep["jarak_pos_km"] = d.min(axis=1)

    routes, total_km = {}, 0.0
    top = rep.sort_values("skor", ascending=False).head(top_k)
    for _, b in bases.iterrows():  # rute nearest-neighbour per pos untuk laporan prioritas
        todo = top[top.pos == b.nama].copy()
        cur, order, dist = (b.lat, b.lon), [], 0.0
        while len(todo):
            dd = haversine_matrix(np.array([cur[0]]), np.array([cur[1]]), todo.lat.values, todo.lon.values)[0]
            i = int(dd.argmin())
            dist += dd[i]
            row = todo.iloc[i]
            order.append(row.name)
            cur = (row.lat, row.lon)
            todo = todo.drop(row.name)
        routes[b.nama] = {"laporan": [int(o) for o in order], "jarak_km": round(float(dist), 2)}
        total_km += dist
    print("\n== [B] Penugasan laporan warga (Snap Send Solve) ==")
    print(f"{n_reports} laporan | {top_k} prioritas tertinggi dijadwalkan | total jarak rute: {total_km:.1f} km")
    for k, v in routes.items():
        print(f"  {k}: {len(v['laporan'])} laporan, {v['jarak_km']} km")
    return rep, bases, routes, {"laporan": n_reports, "dijadwalkan": top_k, "total_km": round(total_km, 2)}


def to_geojson(df, props):
    feats = [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]}, "properties": {p: (r[p].item() if hasattr(r[p], "item") else r[p]) for p in props}}
        for _, r in df.iterrows()
    ]
    return {"type": "FeatureCollection", "features": feats}


def main():
    pop, stores, cand, sel, sum_a = part_a()
    rep, bases, routes, sum_b = part_b()

    (OUT / "gerai_baru_terpilih.geojson").write_text(json.dumps(to_geojson(sel, ["urutan", "gain_jiwa"]), indent=2))
    (OUT / "gerai_eksisting.geojson").write_text(json.dumps(to_geojson(stores, []), indent=2))
    (OUT / "laporan_warga.geojson").write_text(json.dumps(to_geojson(rep, ["jenis", "skor", "pos"]), indent=2))
    sel.to_csv(OUT / "gerai_baru_terpilih.csv", index=False)
    (OUT / "ringkasan.json").write_text(json.dumps({"bagian_A": sum_a, "bagian_B": sum_b, "rute": routes}, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].scatter(pop.lon, pop.lat, s=pop.jiwa, c="#BBBBBB", alpha=0.4, label="populasi")
    ax[0].scatter(stores.lon, stores.lat, marker="s", c="#4C78A8", s=50, label="gerai eksisting")
    ax[0].scatter(sel.lon, sel.lat, marker="*", c="#E45756", s=260, label="lokasi baru", zorder=5)
    for _, r in sel.iterrows():
        ax[0].annotate(f"#{int(r.urutan)}", (r.lon, r.lat), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax[0].set_title(f"A. Lokasi gerai baru (cakupan {sum_a['cakupan_awal']:.0%} -> {sum_a['cakupan_akhir']:.0%})")
    ax[0].legend()
    cm = {"Pos Utara": "#4C78A8", "Pos Tengah": "#F58518", "Pos Selatan": "#54A24B"}
    ax[1].scatter(rep.lon, rep.lat, c=rep.pos.map(cm), s=rep.skor * 8, alpha=0.6)
    ax[1].scatter(bases.lon, bases.lat, marker="P", c="k", s=160, label="pos petugas")
    for nama, v in routes.items():
        b = bases[bases.nama == nama].iloc[0]
        path = [(b.lon, b.lat)] + [(rep.loc[i, "lon"], rep.loc[i, "lat"]) for i in v["laporan"]]
        ax[1].plot(*zip(*path), "-", color=cm[nama], lw=1.5)
    ax[1].set_title(f"B. Laporan warga & rute prioritas ({sum_b['total_km']} km)")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("lon")
        a.set_ylabel("lat")
    plt.tight_layout()
    plt.savefig(OUT / "peta_gis.png", dpi=130)
    print("\nOutput tersimpan di", OUT)


if __name__ == "__main__":
    main()
