# Studi Kasus 5 - Dashboard Penjualan dengan HCI, Visualisasi Data, dan AI yang Transparan

**Materi:** Human-Computer Interaction & Data Visualization - "Make Sense of Data with Analytics and AI" (Arafat Febriandirza)

## Ringkasan materi yang dipakai
Tiga pilar yang harus digabung: **Analytics & AI** (mesin di balik layar), **Data Visualization** (menerjemahkan angka jadi gambar), dan **HCI** (nyaman, mudah dipakai, sesuai cara kerja otak).
- Pilih grafik sesuai tujuan: bar = membandingkan, line = tren, scatter = hubungan, donut = porsi sederhana.
- Susunan: **ringkasan dulu di atas**, lalu filter, lalu detail; hindari hiasan tak berfungsi; warna mencolok hanya untuk hal penting.
- AI sering "kotak hitam"; visualisasi + HCI yang baik (what-if interaktif) menumbuhkan kepercayaan.
- Banyak proyek AI gagal dipakai karena hasilnya tidak mudah dipahami klien/pimpinan.

## Masalah
Manajer toko komoditas di lima kota Bengkulu punya data 2 tahun (2.400 baris transaksi bulanan) dan bertanya:
*bagaimana kondisi penjualan, dan berapa diskon yang sebaiknya diberikan?*

## Solusi (`main.py`)
**1. Dashboard (`outputs/dashboard.png`)** - penerapan prinsip HCI:
- 4 KPI di atas (omzet, pertumbuhan, kota teratas, rata-rata diskon)
- Bar horizontal per kota: abu-abu semua, **satu warna aksen** untuk yang tertinggi
- Line tren bulanan dengan penanda puncak, scatter diskon vs unit, donut hanya 4 kategori
- Palet Okabe-Ito (ramah buta warna), tanpa garis/hiasan berlebih

**2. AI transparan (`outputs/whatif_diskon.png`, `outputs/rekomendasi_diskon.csv`)**
- Model yang dapat dibaca: `log(unit) = a + b*diskon + efek produk + efek kota` -> koefisien `b` = elastisitas ("tiap diskon 10 poin, unit naik ~35%").
- Simulasi *what-if*: profit(d) = unit(d) x (harga x (1-d) - biaya), dicari diskon optimal per produk dalam rentang 0-40%.

## Hasil (seed 42)
- Estimasi `b = 3,04` (nilai sebenarnya data sintetis 3,0): model berhasil menemukan pola.
- Rekomendasi diskon optimal: Kopi **32%**, Minyak Sawit **22%**, Beras **12%**, Gula **2%** (margin tipis -> hampir tanpa diskon).
- Insight bisnis: diskon seragam 20% justru **menurunkan** profit Gula (0,58 jt vs 0,73 jt tanpa diskon).

## Pengembangan
Versi interaktif (slider diskon, filter kota) dapat dibuat dengan Streamlit/Plotly Dash di atas fungsi `fit_elasticity` dan `what_if` pada skrip ini.
