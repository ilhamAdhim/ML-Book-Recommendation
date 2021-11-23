# Laporan Proyek Machine Learning - Muhammad Ilham Adhim

## Project Overview
Domain proyek yang dipilih pada proyek ini adalah  `Sistem Rekomendasi Buku`
<br>
<br>
Dari tahun ke tahun, semakin banyak buku yang diterbitkan dan dipublikasikan. Di era teknologi informasi seperti sekarang, untuk mendapatkan konten buku tidak perlu pergi ke perpustakaan atau ke toko buku, melainkan bisa melalui laptop dan perangkat lain dan mulai membaca softcopy dari buku tersebut. 

Banyak orang menganggap buku sebagai sarana entertainment dan edukasi. Tetapi karena banyaknya buku yang ada dan keterbatasan informasi, terkadang kita jadi ragu memilih buku yang sesuai dengan minat dan keinginan kita. Jika menanyakan orang tentang rekomendasi buku yang ingin kita cari pun akan memakan waktu yang lama, dan kurang efektif. Oleh karena itu, perlu adanya sistem untuk merekomendasikan buku yang ingin dibaca selanjutnya.

Dengan adanya sistem rekomendasi, user experience dari pembaca akan meningkat karena user mendapatkan insight macam macam buku dengan genre yang mirip dari buku yang telah dibaca. Data yang digunakan dalam projek ini cukup lengkap. Mulai dari seri, jumlah rating, genre, judul buku, sampai penghargaan yang diperoleh buku tersebut.

<br>

## Business Understanding
Sistem rekomendasi adalah suatu aplikasi yang digunakan untuk memberikan rekomendasi dalam membuat suatu keputusan yang diinginkan pengguna. Untuk meningkatkan user experience dalam menemukan judul buku yang menarik dan yang sesuai dengan yang user inginkan, maka sistem rekomendasi adalah pilihan yang tepat untuk diterapkan. Dengan adanya sistem rekomendasi, user experience tentu akan lebih baik karena pengguna bisa mendapatkan rekomendasi judul buku yang ingin dibaca.

### Problem Statement
> Bagaimana cara meningkatkan *user experience* dengan mencari judul buku yang dipersonalisasi sesuai genre bacaan pembaca ?


### Goal
> Meningkatkan *user experience* pembaca dengan rekomendasi buku yang dipersonalisasi sesuai genre bacaan pembaca


### Solution
> Karena dataset terkait hanya berisi tentang detail buku dan genre , maka solusi yang tepat untuk masalah ini adalah dengan menggunakan pendekatan Content-Based Filtering untuk meningkatkan *user experience* pembaca dengan cara memberikan rekomendasi buku dengan genre serupa

* Content-Based Filtering : Merupakan cara untuk memberi rekomendasi bedasarkan genre atau fitur pada item yang disukai oleh pengguna. Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna.
  
* Cosine Similarity : Sebuah model yang digunakan dalam penerapan Content Based Filtering. Dengan cara kalkulasi similaritas antara satu item dengan item lain, sistem dapat menyatakan nilai kemiripan antara item satu dengan lainnya. Formula yang digunakan di cosine similarity adalah sebagai berikut:
  
![Formula Cosine Similarity](https://user-images.githubusercontent.com/82896196/137344839-c770d89e-0109-4f91-9691-813d818d0b64.png)

## Data Understanding
Untuk submission ini, saya mengambil data dari Kaggle yang bernama Best Books of The 21st Century Dataset. Berikut adalah daftar kolom di file CSV yang tersedia:
* id
* title: judul buku
* series: Seri buku 
* author: penulis
* book_link: URL buku tersebut di GoodReads
* genre: genre / kategori dari buku tersebut
* date_published: tanggal diterbitkan buku
* publisher: penerbit buku
* num_of_page: jumlah halaman
* lang: bahasa yang digunakan dalam buku
* review_count: jumlah reviews
* rating_count: jumlah ratings
* rate: rating (skala 0 - 5)
* award: penghargaan yang didapat oleh buku tersebut

Untuk mengenali data lebih lanjut, saya melihat fitur apa saja yang tersedia, apa tipe datanya, serta ada berapa null values yang ada dalam dataset ini.

![info_csv](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/info_csv.png?raw=true)

> Dilihat dari persebaran null values yang begitu bervariasi. Tentu kita perlu menghapus data tersebut. Namun karena tujuan kita adalah meningkatkan user experience dalam rekomendasi buku berdasarkan genre dan judul. Maka kedepannya kita hanya akan menghapus null values dari 2 fitur tersebut.

Berikut ilustrasi top 10 publisher dengan jumlah buku diterbitkan terbanyak:

![top_10_publisher](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/top_10_publisher.png?raw=true)

> Terlihat pada grafik diatas bahwa publisher dengan buku diterbitkan terbanyak di dataset ini adalah `Vintage`.

Berikut ilustrasi top 10 genre dengan jumlah buku terbanyak:

![top_10_genres](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/top_10_genres.png?raw=true)

## Data Preparation

Dalam tahap ini, saya menyiapkan dataframe yang telah menyimpan data dari CSV tersebut untuk dilakukan beberapa pengecekan, pertama kita perlu memeriksa adanya null values. Ini perlu dilakukan untuk menjaga akurasi dari prediksi model yang akan kita lakukan di proses pelatihan data sebelum melanjutkan ke proses cosine similarity.
   
![penghapusan_null_values](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/penghapusan_null_values.png?raw=true)


<br> Setelah itu, kita perlu melakukan penghapusan data duplikat di kolom `title` untuk mencegah terjadinya bias dalam implementasi cosine similarity.

![penghapusan_duplikat_data](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/penghapusan_duplikat_data.png?raw=true)

 Karena genre yang ada dalam dataset mempunyai value yang banyak dan untuk kemudahan dalam pemrosesan data lebih lanjut, saya memilih satu genre pertama sebagai representasi kategori dari tiap buku yuang ada.

![pemilihan-genre](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/pemilihan-genre.png?raw=true)


Sebagai rangkuman, langkah yang telah saya lakukan untuk tahap ini adalah:
* Penghapusan missing values
* Penghapusan duplikat data
* Pengambilan genre pertama dari banyak genre dalam satu buku

## Modeling Content-Based Filtering
Dalam tahap modelling, saya menerapkan **cosine similarity** untuk Content Based Filtering. Ini berguna untuk kalkulasi kemiripan antar judul buku dengan menggunakan fitur genre. Berikut untuk tahapan model lebih detailnya:

### Content Based Filtering
Saya menerapkan TF-IDF Vectorizer untuk menemukan representasi fitur penting dari setiap genre yang telah diproses dari buku yang ada. Berikut sampel dari outputnya :

![tf-idf-vectorizer](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/tf-idf-vectorizer.png?raw=true)

Setelah itu, saya melakukan fit dan transformasi ke dalam bentuk matriks. Outputnya adalah matriks berukuran (8140, 119). Nilai 8140 merupakan jumlah buku yang ada dan 119 merupakan matrik genre.

![hasil-transformasi-data](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/hasil-transformasi-data.png?raw=true)


Untuk melakukan perhitungan derajat kesamaan (similarity degree) berdasarkan genre buku, saya mengimplementasikan fungsi cosine_similarity dari library sklearn. Output yang didapat adalah berupa matrix.

![cosine_sim_matrix](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/cosine_sim_matrix.png?raw=true)

Output yang didapat adalah berupa dataframe dengan size (8140,8140). Size nya yang begitu besar dikarenakan ini merupakan menampung nilai kemiripan antara satu buku dengan buku lain. Sebagai ilustrasi, saya memutuskan untuk hanya mengambil 10 sampel secara acak dari dataframe tersebut

![cosine_sim_df_res](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/cosine_sim_df_res.png?raw=true)

## Evaluation

Setelah pembuatan model, saya menguji akurasi dari sistem rekomendasi ini. Sebagai percobaan, saya ingin  menemukan rekomendasi buku yang mirip dengan 'War Time'. Berikut adalah detail informasi buku yang akan saya uji:

![coba-akurasi](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/coba-akurasi.png?raw=true)


> Mencari rekomendasi buku yang mirip dengan 'War Time', tentunya harus memiliki genre yang sejenis (Genre History). Berikut hasil rekomendasi yang diberikan oleh sistem:

![hasil-rekomendasi](https://github.com/ilhamAdhim/ML-Book-Recommendation/blob/master/assets/hasil-rekomendasi.png?raw=true)

> Berdasarkan hasil pada gambar diatas, sistem berhasil memberikan sepuluh rekomendasi buku yang mirip dengan 'War Time' dengan genre sama (History)