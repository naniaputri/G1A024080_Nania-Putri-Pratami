# Studi Kasus 6 - Klasifikasi Driver Peringatan Deforestasi (Sentinel-1 + Sentinel-2)

**Materi:** Presentasi Dr. Bart Slachter (Wageningen University) tentang *deforestation alerts* dan atribusi driver berbasis AI.

## Ringkasan materi yang dipakai
- Citra **optik Sentinel-2** kaya informasi vegetasi tetapi sering tertutup awan di tropis; **radar Sentinel-1** tembus awan dan konsisten tetapi sulit diinterpretasi. Keduanya ditumpuk sebagai citra multi-band (2 band S1 + 4 band S2).
- Alert hanya mendeteksi gangguan hutan; **CNN** dipakai mengklasifikasikan *driver langsung*-nya: smallholder agriculture, large-scale agriculture, jalan, selective logging, mining, wildfire, natural disturbance.
- Overall accuracy tinggi tetapi didominasi kelas pertanian; **macro-F1** lebih jujur karena semua kelas berbobot sama. Wildfire dan gangguan alami (banjir) paling sulit dan sering tertukar.
- Ada label *unconfident* ketika model tidak yakin; klasifikasi hanya driver langsung, data diperbarui bulanan, detail seperti arang/penebangan artisanal butuh data lapangan.
- Saran Q&A: mulai dari metode konvensional sebagai **benchmark**, baru uji deep learning bila masalah kompleks dan data berlabel banyak.

## Masalah
Bangun prototipe yang menerima *patch* citra di sekitar sebuah alert dan menentukan penyebabnya, lalu evaluasi dengan cara yang jujur
(macro-F1, per kelas, ketahanan terhadap awan, tingkat keyakinan).

## Simulasi (`main.py`)
> **Penting:** ini simulasi **data sintetis**, bukan citra satelit asli, dan bukan CNN sungguhan.
> Patch 10x10 piksel x 6 band dibangkitkan dengan tanda spektral tiap driver (bentuk area gangguan, nilai backscatter radar, pantulan optik),
> 50% patch sengaja tertutup awan, kelas dibuat tidak seimbang (pertanian ~58%), wildfire dan natural sengaja dibuat mirip.
> Model yang dipakai Random Forest dan MLP dari scikit-learn agar repositori ringan; untuk citra asli gunakan CNN (PyTorch/TensorFlow) dan label lapangan.

Eksperimen:
1. **Ablasi sensor:** S1 saja vs S2 saja vs S1+S2 (fitur konteks: rata-rata tengah, tepi, selisih, std).
2. **Benchmark:** Random Forest (fitur konteks) vs Random Forest (piksel mentah) vs MLP (piksel mentah).
3. **Evaluasi:** accuracy vs **macro-F1**, F1 per kelas, confusion matrix, akurasi khusus saat berawan, ambang *unconfident* (probabilitas < 0,5).

## Hasil (seed 42, 1.250 data uji)
| Model | Accuracy | Macro-F1 | Akurasi saat berawan |
|---|---|---|---|
| RF konteks, S1 saja | 0,779 | 0,742 | 0,795 |
| RF konteks, S2 saja | 0,646 | 0,666 | **0,321** |
| RF konteks, S1+S2 | **0,886** | **0,864** | 0,782 |
| RF piksel mentah, S1+S2 | 0,787 | 0,714 | 0,703 |
| MLP piksel mentah, S1+S2 | 0,880 | 0,858 | 0,839 |

- **Radar menolong saat awan:** optik saja jatuh ke 32% akurasi di patch berawan, sedangkan S1 saja tetap ~80%; gabungan terbaik secara keseluruhan.
- Kelas tersulit di model terbaik: **wildfire (F1 0,745)** dan **natural (F1 0,798)**, sesuai temuan di presentasi; accuracy 0,886 > macro-F1 0,864.
- 4,3% alert berlabel *unconfident*; akurasi pada alert yang yakin naik menjadi 0,903.
- Pada data sintetis ini fitur konteks yang dirancang manual mengungguli piksel mentah untuk Random Forest, dan MLP menyusul - pesan benchmark: bandingkan metode, jangan langsung memilih yang paling kompleks.

Grafik: `outputs/evaluasi_driver.png`, `outputs/contoh_patch.png` | Tabel: `outputs/hasil_model.csv`, `outputs/ringkasan.json`

## Keterbatasan dan langkah lanjut
Tidak ada citra asli (bisa diunduh dari Copernicus/Google Earth Engine), tidak ada validasi lapangan. Langkah lanjut: ambil alert RADD/GLAD nyata,
label manual sampel, latih CNN, tambahkan data hotspot termal (FIRMS) untuk wildfire, dan evaluasi per wilayah (misalnya Sumatra/Kalimantan).
