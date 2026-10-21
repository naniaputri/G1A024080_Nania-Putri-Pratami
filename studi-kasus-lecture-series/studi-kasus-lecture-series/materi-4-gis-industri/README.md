# Studi Kasus 4 - GIS untuk Keputusan Bisnis dan Kota Pintar (Bengkulu)

**Materi:** Kuliah praktisi GIS dan karier global (Widi Prima Deva, alumni Teknik Informatika UNIB, GIS Analyst di Selandia Baru)

## Ringkasan materi yang dipakai
- GIS adalah alat *data-driven decision making* lintas industri, bukan hanya pemetaan/kehutanan/tambang.
- Contoh ritel (Indomaret) & logistik (JNE): analisis radius antar-gerai agar tidak saling mematikan pasar, populasi di sekitar lokasi, jalur distribusi.
- Lulusan Informatika punya keunggulan sebagai **GIS Developer** (logika pemrograman + basis data), bukan sekadar *end-user*.
- Kota pintar: aplikasi *Snap Send Solve* - warga memotret masalah + koordinat GPS, GIS Analyst mensinkronkan ke basis data kota dan mengirim petugas.
- Rekomendasi tools: open source (QGIS, PostGIS, Python); data spasial standar & terbuka.

## Bagian A - Lokasi gerai ritel baru
**Masalah:** Di sekitar Kota Bengkulu ada 18 gerai dan ~10.000 penduduk (data sintetis). Di mana 5 gerai baru sebaiknya dibuka?

**Metode:** jarak **haversine**, radius layanan 0,8 km, jarak minimum 0,5 km antar gerai, algoritma **greedy maximal coverage**
(pilih kandidat yang menambah penduduk terlayani terbanyak, perbarui area tercakup, ulangi).

**Hasil:** cakupan populasi naik dari **51,4% menjadi 66,3%** (+1.501 jiwa) dengan 5 lokasi baru.

## Bagian B - Laporan warga (model Snap Send Solve)
**Masalah:** 60 laporan (pohon miring, jalan berlubang, lampu mati, rumput berm) masuk; petugas terbatas.

**Metode:** skor prioritas = bobot urgensi x (1 + hari menunggu/7), tiap laporan diarahkan ke pos terdekat,
12 laporan teratas diurutkan dengan rute *nearest neighbour*.

**Hasil:** total rute 20,1 km untuk 3 pos (Utara 5 laporan, Tengah 4, Selatan 3).

## Output
- `outputs/peta_gis.png` (kedua peta)
- `outputs/*.geojson` - dapat dibuka langsung di **QGIS** atau [geojson.io](https://geojson.io)
- `outputs/ringkasan.json`, `outputs/gerai_baru_terpilih.csv`

## Keterbatasan
Jarak garis lurus (bukan jarak jalan), populasi dan lokasi sintetis, tidak ada data kompetitor/harga lahan. Pengembangan lanjut: GeoPandas + PostGIS + OSRM untuk rute jalan.
