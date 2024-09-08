## SA7-Shipping
This repository contains our work for Final Project: SA7 - Rakamin Data Science Bootcamp Batch 46.

Final Project ini dibentuk untuk memenuhi syarat dalam Final Project Rakamin Data Scientist Batch 46
Stage 1 - Pre-Processing

(Kelompok 1 - SA7)
Database: https://www.kaggle.com/datasets/prachi13/customer-analytics/data
1. Data Cleansing (50 poin)
Lakukan pembersihan data, sesuai yang diajarkan di kelas, seperti:
A. Handle missing values

Dengan dilakukan pengecekan nilai
kosong, database yang digunakan tidak
memiliki missing values di semua
kolom, sehingga tidak dilakukan handle
missing values.

B. Handle duplicate data
Tidak ditemukan adanya nilai duplikat, sehingga tidak perlu dilakukan handle duplicate
data.

C. Handle outliers
Pada database shipping data ditemukan outlier, untuk menangani outlier akan
digunakan metode z-score. Metode z-score ini dipilih karena penanganan outlier oleh
z-score tidak semata-mata menghapus data outlier yang ada, sehingga data yang
dihapus dinilai lebih efektif karena tidak menghapus banyak data yang dibutuhkan.

No Metode Penemuan
1 Pengecekan distribusi Data e-commerce ini memiliki
distribusi tidak normal
khususnya prior_purchase,
discount_offered dan
weight_in_gms condong
skewed ke kanan.
Hal ini akan berdampak pada
uji statistik yang tidak valid,
algoritma Machine Learning
overfit dengan data latih,
menyebabkan kesimpulan yang
salah jika hanya bergantung
pada rata-rata, dll.

2 Log transformation Penerapan log transformation
dilakukan supaya bentuk
distribusi menjadi normal.
Setelah dilakukan transformasi
logaritma, pada violet plot dan
density plot menunjukkan
bahwa kolom prior_purchase,
discount_offered dan
weight_in_gms sudah
mendekati distribusi normal.

3 Handling outlier dengan z-score Dari hasil diatas, ada sekitar
178 baris berisi outliers yang
dihapus berdasarkan z-score.
4 Split Data Tujuan utama dari pembagian
ini adalah untuk melatih
model pada satu bagian data
(training set) dan menguji
kinerjanya pada bagian lain
yang tidak digunakan selama
pelatihan (test set).
Pada kasus ini, 80% data
didistribusikan untuk Data
Latih, sedangkan 20% data
digunakan untuk Data Uji

D. Feature transformation
Featured transformation (transformasi fitur) adalah teknik dalam machine learning
dan statistik untuk mengubah fitur data sehingga lebih mudah dianalisis atau
digunakan oleh model prediktif. Metode yang digunakan adalah Min-Max Scaling
yaitu mengubah data sehingga berada dalam rentang 0-1.
E. Feature encoding
Pada data ini, terdapat beberapa kolom yang memiliki jenis data kategori, yaitu
Warehouse_block, Mode_of_Shipment, Product_importance dan Gender. Tentu ini
akan menyulitkan algoritma Machine Learning karena algoritma hanya dapat bekerja
melalui data numerikal. Maka dari itu, setiap kolom tersebut dilakukan "One-Hot
Encoding" agar lebih mudah dalam pengerjaannya.
F. Handle class imbalance
Handling class imbalance (menangani ketidakseimbangan kelas) adalah langkah
penting dalam machine learning ketika satu kelas pada dataset jauh lebih dominan

daripada kelas lain. Tujuannya adalah Meningkatkan Akurasi Model pada Kelas
Minoritas, Menghindari Prediksi Bias, Meningkatkan Performansi Evaluasi,
Mengurangi Overfitting pada Kelas Mayoritas dan Mendukung Keadilan dan Etika
dalam Prediks

Jika dilihat, maka perbandingan antara Class 0 dan Class 1 adalah 1:1,4. Artinya
data tidak terlalu banyak ke salah satu Class. Sehingga tidak perlu menangani Class
Imbalance
NOTE U/ LAPORAN: Di laporan homework, tuliskan apa saja yang telah dilakukan dan
metode yang digunakan.
* Tetap tuliskan jika memang ada tidak yang perlu di-handle (contoh: “Tidak perlu feature
encoding karena semua feature sudah numerical” atau “Outlier tidak di-handle karena akan
fokus menggunakan model yang robust terhadap outlier”).
2. Feature Engineering (35 poin)
A. Feature selection (membuang feature yang kurang relevan atau redundan)
Dalam penerapan feature selection, kita akan memilih features yang akan dibuang.
Pembuangan features ini didasari oleh tingkat korelasi sebuah features yang rendah dengan
variabel target (Reached.on.Time_Y.N).
Korelasi dilakukan dengan dua cara:
1. Korelasi fitur numerik dengan menggunakan heatmap. Penentuan korelasi fitur
numerik disesuaikan dengan angka korelasi terhadap variabel target (bernilai 0).
2. Korelasi fitur kategorikal dengan menggunakan metode kendall. Fitur yang
berkorelasi ditunjukkan pada nilai p-value >0.5. Jika nilai p-value mendekati 0 akan
menunjukkan korelasi yang tidak signifikan terhadap variabel target sehingga
keputusannya dilakukan feature selection.

Fitur Numerik Fitur Kategorikal

Berdasarkan hasil heatmap, tidak ada korelasi
yang bernilai 0 sempurna. Sehingga tidak ada
fitur yang akan dihapus selain ID.

Hasil kendalltau di atas menunjukkan
bahwa ada fitur yang memiliki nilai
p-value <0.5 yang menandakan tidak
berkorelasi.
Maka feature pada tabel yang akan dihapus adalah sebagai berikut ini.
Feature Redundan Alasan Pembuangan Feature
ID ID biasanya hanya digunakan sebagai identifier dan
tidak memiliki korelasi signifikan dengan variabel target
(Reached.on.Time_Y.N).

Product_Importance Fitur ini menunjukkan nilai p-value = 0.016979588~ atau
p-value mendekati nol. Sehingga fitur ini menunjukkan
tidak ada korelasi signifikan dengan variabel target.
B. Feature extraction (membuat feature baru dari feature yang sudah ada)
No Feature Baru Formula Keterangan
1 Weight Category - Ringan = Berat Barang

< 2615 gr
- Sedang = 2615 gr <
Berat Barang < 5.230
gr
- Berat = Berat Barang >
5230 gr

Kategorikan berat ke dalam kategori
(ringan, sedang, berat).

Insight:
1. Mayoritas produk masuk kedalam kategori medium kemudian diikuti dengan kategori
low dan terakhir high. Dari data ini dapat disimpulkan bahwa perusahaan lebih
banyak mengirim produk dengan kategori medium yaitu rentang >2615 hingga 5230.
2. Produk dengan kategori low menerima rata-rata diskon paling besar, sedangkan
produk kategori high menerima diskon paling kecil.
3. Produk dengan kategori low memiliki presentase pengiriman tepat waktu tertinggi
yaitu >70% sedangkan kategori high memiliki presentase pengiriman tepat waktu
terendah yaitu <50%.
4. Semua kategori berat memiliki jumlah mode of shipment yang hampir sama, mode
paling banyak menggunakan Ship mode kemudian diikuti dengan Flight mode dan
terakhi Road mode. Dengan demikian, weight category tidak menentukan jenis
pengiriman yang digunakan.

2 Total Customer
Service
Involvement

Customer_care_calls /
Prior_purchases

Fitur ini mengukur seberapa sering
pelanggan menghubungi layanan
pelanggan dibandingkan dengan jumlah
pembelian yang sudah mereka lakukan.
Pelanggan yang sering menghubungi
layanan pelanggan mungkin lebih mungkin
mengalami masalah dengan pengiriman.

Insight:
1. Segmen Pelanggan: Perusahaan dapat mengidentifikasi beberapa segmen
pelanggan berdasarkan tingkat keterlibatan mereka.

2. Penyebab Keterlibatan Tinggi: Perusahaan perlu menyelidiki lebih lanjut mengapa
ada kelompok pelanggan dengan nilai keterlibatan yang tinggi.
3. Peningkatan Kepuasan Pelanggan: Dengan memahami pola keterlibatan
pelanggan, perusahaan dapat mengambil langkah-langkah untuk meningkatkan
kepuasan pelanggan.
4. Program Loyalitas: Perusahaan dapat merancang program loyalitas yang lebih
efektif dengan mempertimbangkan tingkat keterlibatan pelanggan.

3 Product Value
(Cost Of The
Product) & Product
Importance

Korelasi (Cost of The
Product) & (Product
Importance)

Fitur ini mengukur rasio biaya produk
terhadap product importance. Dengan fitur
ini kita dapat menentukan hubungan biaya
produk dengan kepentingan produk
sehingga dapat menentukan segmentasi
pasar.

Insight:
1. Harga bukan satu-satunya penentu: Harga hanyalah salah satu dari banyak faktor
yang dipertimbangkan oleh konsumen dalam membuat keputusan pembelian.
Faktor-faktor non-moneter seringkali memiliki bobot yang lebih besar
5. Pentingnya segmentasi pasar: Mungkin ada segmen pasar tertentu yang lebih
sensitif terhadap harga dibandingkan segmen lainnya. Analisis lebih lanjut dapat
dilakukan untuk mengidentifikasi segmen-segmen ini.

4 Average Cost per
Gram

Cost_of_the_Product /
Weight_in_gms

Fitur ini mengukur harga rata-rata per gram
produk. Ini bisa berguna untuk melihat
apakah produk dengan harga per gram
yang lebih tinggi memiliki hubungan
dengan keterlambatan pengiriman.

Insight:
1. Data rata-rata biaya per gram memiliki distribusi yang miring ke kanan, dengan
sebagian besar data terkonsentrasi pada rentang berat yang lebih rendah.
2. Rata-rata biaya per gram kemungkinan lebih besar daripada median karena
pengaruh nilai-nilai ekstrem yang lebih tinggi.
3. Data tidak sepenuhnya mengikuti distribusi normal, seperti yang ditunjukkan oleh
perbedaan antara histogram dan kurva normal.

5 Discount
Percentage

(Discount_offered /
Cost_of_the_Product) *
100

Fitur ini memberikan insight tentang
seberapa besar diskon yang diberikan
sebagai persentase dari harga produk.
Produk dengan diskon besar mungkin
memiliki pola pengiriman yang berbeda
dengan ketepatan waktu yang berbeda
pula.

Insight:
1. Pengiriman tepat waktu dan tidak tepat waktu: Pengiriman tepat waktu dengan
diskon tinggi dapat meningkatkan kepuasan pelanggan dan mendorong loyalitas,
strategi ini bisa dipertahankan atau ditingkatkan. Variasi persentase diskon yang
diberikan dapat menunjukkan bahwa pengiriman tidak tepat waktu bisa disebabkan
faktor lain seperti faktor logistik atau geografis.
2. Prioritas pemberian diskon untuk pengiriman tepat waktu: Prioritas pemberian
diskon bisa dipertimbangkan kembali dengan mempertimbangkan faktor lainnya
seperti faktor logistik maupun geografis karena mempengaruhi biaya pengiriman.
Prioritas diskon ini diperlukan supaya pemberian diskon memberikan keuntungan
yang sebanding dengan pengeluaran biaya pengiriman yang dilakukan.

6 Mode of Shipment
and Importance
Interaction

Mode_of_Shipment +
Product_importance

Interaksi antara mode pengiriman dan
pentingnya produk bisa memberi insight
apakah produk penting yang dikirim
melalui metode tertentu (misalnya,
pesawat) lebih cepat atau lebih lambat
mencapai tujuan.

Insight:
1. Pertimbangkan untuk mengoptimalkan rute dan pengelolaan sumber daya:
untuk pengiriman melalui jalan guna meningkatkan tingkat ketepatan waktu.
2. Pemanfaatan moda: Dalam hal ini pengiriman kepentingan tingkat tinggi dapat
memanfaatkan moda yang tidak memiliki kuota yang padat dan waktu pengiriman
yang cepat, seperti moda pesawat.
3. Evaluasi strategi untuk kepuasan pelanggan: Sebaiknya dilakukan evaluasi
kembali untuk pengiriman dengan kepentingan rendah melalui kapal, karena rasio
tidak tepat waktu lebih tinggi dibandingkan yang tepat waktu. Hal ini akan
mempengaruhi kepuasan pelanggan juga.

C. Tuliskan minimal 4 feature tambahan (selain yang sudah tersedia di dataset) yang
mungkin akan sangat membantu membuat performansi model semakin bagus (ini hanya ide
saja, untuk menguji kreativitas teman-teman, tidak perlu benar-benar dicari datanya dan
tidak perlu diimplementasikan)
No Feature Tambahan Keterangan
1 Customer_satisfaction Skor yang menggabungkan customer rating dan
customer care calls untuk mencerminkan sentimen
pelanggan.

2 Shipment_priority Berasal dari kombinasi Product_importance dan
Mode_of_Shipment untuk mengetahui apakah produk
penting dikirim menggunakan metode yang lebih cepat.
3 Delivery Distance Jarak antara gudang dan lokasi pelanggan. Ini bisa
signifikan dalam menentukan waktu pengiriman.

4 Shipping Duration
(ETA)

Waktu estimasi pengiriman berdasarkan moda
pengiriman yang digunakan (misalnya: via udara lebih
cepat dibanding darat).

5 Product Category Menambahkan kategori produk untuk melihat apakah
jenis produk (misalnya, elektronik vs pakaian)
mempengaruhi kecepatan pengiriman.

6 Promotional Period Indikator apakah pesanan dilakukan selama periode
promosi atau diskon, karena tingginya volume pesanan
mungkin menyebabkan keterlambatan.

NOTE U/ LAPORAN Untuk 2A & 2B, tetap tuliskan jika memang tidak bisa dilakukan
(contoh: “Semua feature digunakan untuk modelling (tidak ada yang dihapus), karena
semua feature relevan”)
