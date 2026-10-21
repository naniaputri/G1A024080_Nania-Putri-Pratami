# Studi Kasus 2 - Simulasi Sebaran Kabut Asap Paralel (peran "Integrator" HPC)

**Materi:** Kuliah umum AI, IoT dan High Performance Computing (Dr. Worawan)

## Ringkasan materi yang dipakai
- AI modern ditopang HPC/superkomputer (diukur dalam FLOPS, kini skala Exaflops).
- Infrastruktur global harus bermuara pada **masalah lokal**; mahasiswa informatika berperan sebagai **Integrator** yang menjembatani masalah nyata dan komputasi cepat.
- Contoh dari materi: pemantauan kualitas udara/kabut asap PM2.5 (Thailand), peramalan banjir (Bangladesh), dll.
- Roadmap: Learner -> Practitioner (Linux, SSH, batch scripting) -> Facilitator -> Innovator.

## Masalah
Kabut asap dari kebakaran lahan mengancam kota di sekitarnya. Arah dan kecepatan angin tidak pasti, sehingga satu simulasi tidak cukup:
perlu **ratusan skenario angin** (Monte Carlo) untuk menilai risiko. Satu skenario cukup ringan, tetapi banyak skenario butuh komputasi paralel.

## Solusi (`main.py`)
- Model **advection-diffusion 2D** (grid 200x200 km, 600 jam): satu titik api, angin turbulen, satu kota target.
- 48 skenario angin/emisi dijalankan dengan `multiprocessing.Pool` dengan 1, 2, dan 4 worker.
- Dihitung **speedup** dan estimasi **fraksi paralel (hukum Amdahl)**.
- Output risiko: persentase skenario yang membuat konsentrasi di kota melewati ambang.
- `job.sbatch`: contoh **batch script SLURM** untuk menjalankan di cluster (`sbatch job.sbatch`).

## Hasil (mesin uji 2 core, seed tetap)
| Worker | Waktu | Speedup |
|---|---|---|
| 1 | 12,0 dtk | 1,00x |
| 2 | 6,3 dtk | 1,92x |
| 4 | 6,0 dtk | 2,02x (mentok, hanya 2 core fisik) |

- Estimasi fraksi paralel ~0,96. Hasil paralel dicek **identik** dengan hasil serial (`assert`).
- Pada skenario simulasi, 15% skenario membuat kota melewati ambang konsentrasi.
- Grafik: `outputs/hpc_speedup_dan_sebaran.png`, tabel: `outputs/hasil.csv`.

## Keterbatasan
Model difusi sangat disederhanakan (bukan model kimia-transport atmosfer sungguhan) dan ambang `THRESH` bersifat skala simulasi.
Nilai speedup bergantung pada jumlah core mesin Anda; di cluster dengan 16+ core hasilnya akan lebih besar.
