"""
Studi Kasus 2 - Simulasi Sebaran Asap (Kabut Asap) Paralel: peran "Integrator" HPC

Materi: Kuliah umum HPC & AI (Dr. Worawan) - masalah lokal (kabut asap PM2.5) dijawab dengan komputasi paralel.

Model: advection-diffusion 2D sederhana. Satu titik api mengemisi partikel, angin membawanya ke
kota. Ratusan skenario angin dijalankan (Monte Carlo) -> tugas "embarrassingly parallel" yang cocok
untuk HPC. Skrip ini membandingkan serial vs paralel dan menghitung speedup (Amdahl).
Skrip SLURM untuk dijalankan di cluster ada di `job.sbatch`.
"""
import multiprocessing as mp
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

N = 200  # ukuran grid (N x N), 1 sel = 1 km
STEPS = 600  # langkah waktu (jam)
SRC = (140, 30)  # (baris, kolom) titik api
CITY = (100, 150)  # lokasi kota
THRESH = 25.0  # ambang konsentrasi (skala simulasi) dianggap "tidak sehat"


def simulate(scenario, return_field=False):
    seed, u0, v0, q = scenario
    rng = np.random.default_rng(seed)
    c = np.zeros((N, N))
    D, dt = 0.15, 1.0
    exceed_hours, peak = 0, 0.0
    for _ in range(STEPS):
        u = u0 * (1 + 0.15 * rng.standard_normal())  # turbulensi angin
        v = v0 * (1 + 0.15 * rng.standard_normal())
        # advection (upwind) + difusi
        dcdx = c - np.roll(c, 1, axis=1) if u >= 0 else np.roll(c, -1, axis=1) - c
        dcdy = c - np.roll(c, 1, axis=0) if v >= 0 else np.roll(c, -1, axis=0) - c
        lap = np.roll(c, 1, 0) + np.roll(c, -1, 0) + np.roll(c, 1, 1) + np.roll(c, -1, 1) - 4 * c
        c = c + dt * (-abs(u) * dcdx - abs(v) * dcdy + D * lap)
        c[SRC[0] - 1 : SRC[0] + 2, SRC[1] - 1 : SRC[1] + 2] += q * dt
        c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = 0.0  # batas terbuka
        val = c[CITY]
        peak = max(peak, val)
        exceed_hours += val > THRESH
    result = (float(c[CITY]), float(peak), int(exceed_hours))
    return (result, c) if return_field else result


def make_scenarios(n=48):
    rng = np.random.default_rng(7)
    return [(i, rng.uniform(0.15, 0.45), rng.uniform(-0.35, -0.05), rng.uniform(1.0, 3.0)) for i in range(n)]


def run(scenarios, workers):
    t0 = time.perf_counter()
    if workers == 1:
        res = [simulate(s) for s in scenarios]
    else:
        with mp.Pool(workers) as pool:
            res = pool.map(simulate, scenarios, chunksize=2)
    return res, time.perf_counter() - t0


def main():
    scenarios = make_scenarios()
    cpu = os.cpu_count() or 1
    worker_list = sorted({1, 2, cpu, min(4, cpu * 2)})
    print(f"CPU tersedia: {cpu} | jumlah skenario: {len(scenarios)} | uji worker: {worker_list}")

    times, base_res = {}, None
    for w in worker_list:
        res, t = run(scenarios, w)
        times[w] = t
        base_res = base_res or res
        assert res == base_res, "hasil paralel harus sama dengan serial"
        print(f"  worker={w}: {t:6.2f} dtk | speedup {times[1] / t:4.2f}x")

    # Estimasi bagian paralel (Amdahl) dari p terbesar yang masih <= jumlah core
    p = max(w for w in worker_list if w <= cpu)
    s = times[1] / times[p]
    f = (1 - 1 / s) / (1 - 1 / p) if p > 1 else 1.0
    print(f"Estimasi fraksi paralel (Amdahl) ~ {f:.2f} (p={p}, speedup={s:.2f})")

    # Analisis risiko: berapa persen skenario membuat kota melewati ambang?
    peaks = np.array([r[1] for r in base_res])
    hours = np.array([r[2] for r in base_res])
    risk = float((hours > 0).mean())
    print(f"Skenario dengan kota melewati ambang: {risk:.0%} | median jam melebihi ambang: {np.median(hours):.0f}")

    (_, field) = simulate(scenarios[0], return_field=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].bar([str(w) for w in worker_list], [times[w] for w in worker_list], color="#4C78A8")
    ax[0].set_xlabel("jumlah worker")
    ax[0].set_ylabel("waktu (detik)")
    ax[0].set_title("Waktu eksekusi 48 skenario")
    ax[1].plot(worker_list, [times[1] / times[w] for w in worker_list], "o-", label="aktual")
    ax[1].plot(worker_list, worker_list, "--", color="gray", label="ideal")
    ax[1].set_xlabel("jumlah worker")
    ax[1].set_ylabel("speedup")
    ax[1].set_title(f"Speedup (CPU fisik = {cpu})")
    ax[1].legend()
    im = ax[2].imshow(field, cmap="magma", origin="lower")
    ax[2].plot(SRC[1], SRC[0], "c^", label="titik api")
    ax[2].plot(CITY[1], CITY[0], "wo", label="kota")
    ax[2].legend(loc="lower right")
    ax[2].set_title("Contoh sebaran asap (skenario 0)")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    plt.tight_layout()
    plt.savefig(OUT / "hpc_speedup_dan_sebaran.png", dpi=130)

    with open(OUT / "hasil.csv", "w") as fh:
        fh.write("worker,waktu_detik,speedup\n")
        for w in worker_list:
            fh.write(f"{w},{times[w]:.3f},{times[1] / times[w]:.3f}\n")
        fh.write(f"# risiko_kota_melewati_ambang,{risk:.3f}\n")
    print("Output tersimpan di", OUT)


if __name__ == "__main__":
    main()
