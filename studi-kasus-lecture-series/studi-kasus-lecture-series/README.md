# Studi Kasus Informatics Practitioner Lecture Series 2026

Kumpulan 6 studi kasus beserta implementasi kode (Python) berdasarkan 6 materi kuliah praktisi
Program Studi Informatika, Universitas Bengkulu.

| # | Materi (Pemateri) | Studi kasus | Folder |
|---|-------------------|-------------|--------|
| 1 | Analitik data & AI di dunia profesional, CRISP-DM (Andria Arisal, BRIN) | Prediksi keterlambatan pembayaran tagihan | [`materi-1-crispdm-analitik`](materi-1-crispdm-analitik) |
| 2 | HPC & AI, peran "Integrator" (Dr. Worawan) | Simulasi sebaran kabut asap paralel + SLURM | [`materi-2-hpc-integrator`](materi-2-hpc-integrator) |
| 3 | Pengenalan Machine Learning | Workflow ML: kredit, harga rumah, segmentasi pelanggan | [`materi-3-machine-learning`](materi-3-machine-learning) |
| 4 | GIS di industri & kota pintar (Widi Prima Deva) | Lokasi gerai baru + penugasan laporan warga | [`materi-4-gis-industri`](materi-4-gis-industri) |
| 5 | HCI & Data Visualization (Arafat Febriandirza) | Dashboard penjualan + AI what-if yang transparan | [`materi-5-hci-visualisasi`](materi-5-hci-visualisasi) |
| 6 | Deforestation alerts & AI (Dr. Bart Slachter, Wageningen) | Klasifikasi driver deforestasi Sentinel-1 + Sentinel-2 | [`materi-6-deforestasi-ai`](materi-6-deforestasi-ai) |

## Cara menjalankan

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python materi-1-crispdm-analitik/main.py
python materi-2-hpc-integrator/main.py
python materi-3-machine-learning/main.py
python materi-4-gis-industri/main.py
python materi-5-hci-visualisasi/main.py
python materi-6-deforestasi-ai/main.py
```

Setiap skrip menyimpan grafik/CSV/JSON hasil ke folder `outputs/` di dalam folder materinya.
Seed acak dikunci (`42`) sehingga hasilnya bisa direproduksi. Diuji dengan Python 3.13.

## Catatan penting

- **Semua data bersifat sintetis** (dibangkitkan oleh skrip) supaya repositori bisa dijalankan tanpa mengunduh dataset.
  Angka hasil menggambarkan perilaku metode, bukan kondisi dunia nyata.
- Setiap folder punya `README.md` berisi: ringkasan materi, masalah studi kasus, alur solusi, hasil, dan keterbatasan.
- Studi kasus ini terinspirasi dari isi materi pemateri; implementasinya adalah versi sederhana untuk tujuan belajar.
