# Studi Kasus 1 - Prediksi Keterlambatan Pembayaran Tagihan

**Materi:** Kuliah praktisi analitik data & AI di dunia profesional (Andria Arisal, BRIN)

## Ringkasan materi yang dipakai
- Data dunia nyata itu berantakan; praktisi harus menerjemahkan masalah bisnis menjadi pertanyaan analitik.
- Siklus **CRISP-DM** (iteratif): business understanding -> data understanding -> data preparation -> modeling -> evaluation.
- Data preparation = tahap paling menyita waktu: missing value, duplikasi, anomali aturan bisnis (*garbage in, garbage out*) dan **data leakage**.
- Empat tingkat analitik: **deskriptif** (apa yang terjadi), **diagnostik** (mengapa), **prediktif** (apa yang akan terjadi), **preskriptif** (apa yang harus dilakukan, dengan batasan sumber daya).
- AI tidak punya konteks bisnis utuh dan bisa berhalusinasi -> validasi manusia tetap kunci.

## Masalah
Sebuah perusahaan di Bengkulu ingin menekan tagihan yang terlambat dibayar. Data tagihan 3.000 pelanggan berantakan
(duplikat, typo nama wilayah, umur 199 tahun, nominal negatif, nilai kosong). Tim penagihan hanya punya **150 unit jam kerja**
untuk menghubungi pelanggan. Siapa yang harus dihubungi lebih dulu?

## Alur solusi (`main.py`)
| Tahap CRISP-DM | Implementasi |
|---|---|
| Data understanding | Data sintetis sengaja dikotori + laporan kualitas data |
| Data preparation | Hapus duplikat, standarkan label wilayah, anomali -> NaN, imputasi median **di dalam pipeline** |
| Deskriptif | Late-rate total, per wilayah, per industri |
| Diagnostik | Late-rate menurut riwayat telat dan tenor |
| Prediktif | Random Forest (AUC) + demonstrasi **leakage** (kolom `days_late`) |
| Preskriptif | Pilih pelanggan dengan `benefit/cost` tertinggi hingga budget 150 unit (greedy knapsack) |

## Hasil (seed 42)
- Kualitas data: 3.090 baris -> 3.000 baris, duplikat 90 -> 0, 15 variasi label wilayah -> 5, anomali umur/nominal -> 0.
- Late-rate keseluruhan **13,3%**; pelanggan dengan >=3 keterlambatan sebelumnya punya late-rate **34,1%** vs **6,1%** untuk yang tidak pernah telat.
- AUC model tanpa leakage **0,705**; dengan kolom bocor **1,000** -> terlalu sempurna, tanda ada kebocoran data.
- Preskriptif: menghubungi 127 pelanggan hasil optimasi diperkirakan menyelamatkan Rp 69,6 juta vs Rp 26,2 juta jika dipilih acak.

Grafik: `outputs/empat_tingkat_analitik.png` | Ringkasan: `outputs/ringkasan.json` | Data bersih: `outputs/data_bersih.csv`

## Keterbatasan
Data sintetis; asumsi "pengingat memulihkan 35% nilai berisiko" dan biaya 1-2 unit adalah asumsi ilustrasi, bukan hasil estimasi.
