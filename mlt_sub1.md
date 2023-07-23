# Laporan Proyek Machine Learning - Muhammad Faturachman Atthaariq

## Domain Proyek
Stroke merupakan penyakit mematikan dengan tingkat mortalitas tinggi hingga saat ini. Di Amerika Serikat, 1 dari 6 kematian akibat penyakit kardiovaskular disebabkan oleh stroke di tahun 2021 dan setiap 3 menit dan 14 detik sekali, seseorang meninggal karena stroke[[1]](https://wonder.cdc.gov/mcd.html). Laporan oleh American Heart Association / AHA memaparkan banyak fakta yang lebih mengerikan, yaitu setiap 40 detik sekali, akan ada seseorang yang terkena stroke[[2]](https://pubmed.ncbi.nlm.nih.gov/36695182/). Setiap tahun, sekitar 795.000 orang menderita stroke, di mana hampir seperempat / 185.000 kasus yang dicatat diderita orang yang pernah menderita stroke juga sebelumnya, serta 610.000 kasus stroke lainnya merupakan kasus stroke baru / pertama kali[[2]](https://pubmed.ncbi.nlm.nih.gov/36695182/).

Selain itu, penanganan kasus yang terkait dengan stroke menelan biaya hingga $56.5 miliar dollar pada tahun 2018-2019[[2]](https://pubmed.ncbi.nlm.nih.gov/36695182/), yang mana ini merupakan biaya layanan kesehatan, pengobatan, serta kerugian akibat kehilangan hari kerja akibat stroke.

Stroke merupakan penyebab utama dari kecacatan jangka panjang yang sangat serius & mengurangi mobilitas lebih dari setengah penderitanya yang berusia di atas 65 tahun. Mortalitas akibat stroke tercatat meningkat dari 38.8 menjadi 41.1 per 100.000 orang pada 2020-2021[[1]](https://wonder.cdc.gov/mcd.html). Fakta tersebut menjadikan stroke sebagai penyakit yang butuh penanganan intensif, dan lebih baik di deteksi segera sebelum terjadi *bleeding* / pendarahan atau tersumbatnya pembuluh darah yang mengakibatkan mati rasa / kelumpuhan. 

Survey oleh Fang *et al*. pada penderita stroke menunjukkan bahwa sebagian besar responden (93%) menyadari mati rasa mendadak di satu sisi sebagai gejala stroke, tetapi hanya 38% saja yang mengetahui semua gejala utama dan tahu untuk menelepon panggilan darurat (9-1-1) ketika seseorang mengalami stroke[[3]](https://www.cdc.gov/mmwr/preview/mmwrhtml/mm5718a2.htm).

Langkah preventif untuk mencegah terjadinya stroke adalah dengan memprediksi apakah seseorang berisiko besar terkena stroke dengan melihat berbagai parameter kesehatan yang berpengaruh tinggi pada risiko terkena stroke, seperti hipertensi / darah tinggi, kolesterol tinggi, perokok, obesitas, dan diabetes sebagai penyebab utama stroke[[2]](https://pubmed.ncbi.nlm.nih.gov/36695182/).
Dengan memprediksi risiko terkena stroke, seseorang dapat mempertimbangkan langkah pencegahan lebih dini sehingga mengurangi risiko terkena stroke.

Dari uraian di atas, sebuah program yang dapat melakukan prediksi risiko seseorang terkena stroke sangat diperlukan dengan harapan sebagai langkah pencegahan dini terhadap penyakit stroke.

## Business Understanding
### Problem Statements
Berdasarkan permasalahan dari latar belakang, masalah yang akan dijawab adalah:

 1. Bagaimana berbagai paramater kesehatan mempengaruhi seseorang terkena stroke?
 2. Bagaimana cara mengetahui seseorang berisiko terkena stroke melalui pendekatan machine learning?

### Goals
Tujuan yang hendak dicapai dari proyek ini adalah menghasilkan prediksi risiko terkena stroke melalui pendekatan machine learning.

### Solution Statements

 1. Membuat EDA untuk melihat pengaruh antar parameter kesehatan terhadap risiko terkena stroke.
 2. Membuat model machine learning dengan pendekatan klasifikasi menggunakan algoritma berikut:
    - Logistic Regression (Logit / LR)
    - Support Vector Machine (SVM)
   
## Data Understanding
Data yang digunakan pada proyek ini didapatkan dari Kaggle pada tautan berikut ini *[Brain stroke prediction dataset](https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset)*. Variabel-variabel pada data *Brain stroke prediction dataset* adalah sebagai berikut:
* gender: Jenis kelamin pasien, "Male" atau "Female"
* age: Usia pasien
* hypertension: Apakah pasien menderita hipertensi atau tidak
* heart_disease: Apakah pasien menderita penyakit jantung atau tidak
* ever_married: Apakah pasien sudah atau belum menikah
* work_type: Tipe pekerjaan pasien, apakah "Private", "Self-employed", atau "Other"
* Residence_type: Apakah pasien tinggal di daerah perkotaan atau pedesaan
* avg_glucosa_level: Level gula darah pasien, berkaitan dengan penyakit diabetes
* bmi: *Body Mass Index* (BMI), yaitu nilai yang menentukan pasien mengalami obesitas atau tidak
* smoking_status: Apakah pasien merokok atau tidak, yaitu "never smoked", "Unknown", atau "Other"
* stroke: variabel target, yaitu apakah pasien terkena stroke atau tidak.

*Exploratory Data Analysis* / EDA dilakukan pada tahap ini untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

### EDA
1. Terdapat 4981 data dalam *dataset* dengan 11 kolom, termasuk kolom target yaitu 'stroke'.
2. Terdapat 6 kolom numerik dari awal, yaitu 3 kolom numerik kontinyu seperti 'age', 'avg_glucose_level', dan 'bmi', serta 3 kolom numerik *binary* seperti 'hypertension', 'heart_disease', dan 'stroke'.
3. Sebanyak 5 kolom data lainnya awalnya merupakan kolom kategorikal, seperti 'gender', 'work_type', 'Residence_type', 'smoking_status', dan 'ever_married'.
4. Tidak ada kolom yang duplikat dan Null / <NaN>pada *dataset* ini.
5. 3 kolom numerik kontinyu seperti 'age' memiliki 104 ragam data, kolom 'avg_glucose_level' sebanyak 3895 ragam data, dan kolom 'bmi' sebanyak 342 ragam data.
6. Korelasi antar data sebagai berikut: 
![Korelasi](https://i.ibb.co/Pj20wVJ/corr-sub1.png)
7. Rasio penderita stroke lebih banyak dialami oleh wanita dengan 140 data dibanding pria dengan 108 data: 
![Rasio penderita stroke berdasarkan gender](https://i.ibb.co/M58kDpN/stroke-gender-ratio-sub1.png)
8. *Dataset* ini memiliki data yang *imbalanced*, yaitu 4733 data non-stroke dan hanya 248 data penderita stroke: 
 ![Jumlah data non-stroke dan penderita stroke](https://i.ibb.co/NK1T6Rz/stroke-data-sub1.png)
9. Kemudian akan dilihat kecenderungan risiko stroke dari 3 data numerika kontinyu sebagai berikut: 
	- age  
	![age vs stroke](https://i.ibb.co/KzT4WVj/age-stroke-sub1.png)  
	- avg_glucose _level 
	![age vs stroke](https://i.ibb.co/XDtJDCq/glucose-stroke-sub1.png)
	- bmi
	![age vs stroke](https://i.ibb.co/K5t7ZxQ/bmi-stroke-sub1.png)

## Data Preparation
Tahapan pada Data Perparation adalah sebagai berikut:
1. Penanganan *Missing Values*
Machine Learning perlu data yang bersih dari nilai Null / `NaN`, sehingga proses ini dilakukan untuk memastikan tidak ada Null dalam data. *Dataset* ini tidak memiliki *missing value* sehingga tidak perlu ada penanganan khusus.
2. *Label Encoding*
Untuk variabel kategorikal seperti gender, ever_married, work_type, smoking_status, dan Residence_type akan di encode terlebih dahulu agar dapat digunakan machine learning
3. *Oversampling* dengan SMOTE
Terjadi ketimpngan *imbalance* pada kelas yang akan diprediksi pada data ini, yaitu jumlah data penderita stroke mencapai 4733 orang (95%), sedangkan data non-stroke hanya 248 orang (5%). Oleh karena itu, data ini perlu diseimbangkan. Saya menggunakan *Synthetic Minority Over-sampling Technique* (SMOTE) untuk menyeimbangkan data : 
	- *imbalanced data*: ![imbalanced data](https://i.ibb.co/kKcfYZQ/imbalanced-sub1.png)
	- Setalah SMOTE: ![imbalanced data](https://i.ibb.co/gM2sz4R/smote-sub1.png)
4. Standarisasi data
Tahap ini dilakukan karena algoritma *machine learning* memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala yang relatif sama, yaitu dengan MinMaxScaler. MinMaxScaler dipilih karena data yang dihasilkan antara 0 dan 1 sehingga komputasinya lebih cepat.
5. *Train-Test Split*
Untuk keperluan validasi model, data perlu dipisahkan menjadi data *training* dan data *testing* dengan rasio *train* : *test* sebesar 80 : 20.

## Modelling
Pada tahap ini akan dibuat model untuk menjawab pertanyaan mengenai prediksi stroke dengan pendekatan *machine learning*. Prediksi ini akan membandingkan dua algoritma, yaitu *Logistic Regression* (Logit / LR) & *Support Vector Machine* (SVM).

*Logistic Regression* (Logit / LR) atau Regresi logistik adalah pendekatan analisis regresi yang diadopsi dari *Linear Regression* / Regresi linier. Regresi linier yaitu teknik untuk memprediksi hubungan antara fitur *input* (diskrit/kontinyu) dengan *output* diskrit melalui garis yang paling cocok dari titik data dengan garis lurus. Perbedaan Regresi logistik dan linier adalah pada regresi logistik, data akan dicocokkan pada fungsi logistik (sigmoid) yang melengkung. Hal ini membuat regresi logistik dapat menangani prediksi pada data *outlier*. Selain itu regresi linier dapat menghasilkan prediksi yang salah karena kesalahan pemilihan variabel yang digunakan sehingga kita perlu menganalisis secara rinci mengenai korelasi variabel (independen dan dependen) yang digunakan, berbeda dengan regresi logistik yang tidak terpengaruh dengan korelasi variabel tersebut. Pada proyek ini, saya akan menggunakan *Binary Logistic Regression* karena prediksi akan menghasilkan *output* antara positif (1) dan negatif (0).

*Support Vector Machine* (SVM) merupakan pendekatan *supervised* pada ML yang dapat digunakan pada klasifikasi dan regresi. SVM adalah algoritma yang memanfaatkan konsep *hyperplane* untuk memisahkan dua kelas data dengan margin yang maksimal. Tujuan utama SVM adalah untuk menemukan *hyperplane* yang memaksimalkan jarak antara dua kelas data terdekat (*support vectors*) dalam ruang fitur.

*Logistic Regression* yang saya gunakan pada proyek ini menggunakan `LogisticRegression()`dengan parameter:
 - solver: "liblinear", yaitu menggunakan algoritma *coordinate descent* dari LIBLINEAR[[4]](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).
 
 *Support Vector Machine* yang saya gunakan pada proyek ini menggunakan `SVC()` dengan parameter:
 - kernel: "rbf", yaitu *Radial Basis Function*. RBF adalah jenis kernel yang menggunakan fungsi basis radial yang mengandalkan distribusi *gaussian*.
 - random_state: 3, yaitu pemilihan inisialisasi pembangkit bilangan *random* pada tahapan yang memerlukan bilangan *random* dalam SVM, dengan nilai = 3.

Pemodelan dilakukan dengan membagi data *train* dan *test* ke dalam rasio 80:20 dalam dua tipe, yaitu data yang akan di-*feed* pada model pada tahap pelatihan dan untuk pengujian. Tahapan pelatihan menggunakan input `x_train` yang di latih untuk menghasilkan prediksi pada `y_train`.  Hasil pelatihan tersebut akan diuji oleh input `x_test` yang menghasilkan prediksi `y_pred_LR` dan `y_pred_SVM`. Prediksi pada kedua model ini dibandingkan dengan prediksi yang benar (*ground truth*) yaitu `y_test` untuk melihat apakah prediksi positif / P dan negatif / N yang dihasilkan bernilai benar / T atau salah / F. 

## Evaluation

### *Confusion Matrix*
Proses *training* dan *test* pada dua model ini menghasilkan prediksi positif berisiko stroke / P dan negatif / N pada data yang benar / *true* (TP & TN) serta pada prediksi yang salah / *false* (FP & FN). Hasil ini dilihat dengan *confusion matrix* sebagai berikut:
- *Logistic Regression* 
 ![Logit Confusion Matrix](https://i.ibb.co/0njwPwP/lr-cm-sub1.png)
- *Support Vector Machine* 
![SVM Confusion Matrix](https://i.ibb.co/6HfBbd0/svm-cm-sub1.png)

### ROC (*Receiver Operating Characteristic*) LR vs SVM
ROC adalah kurva yang digunakan untuk melihat performa suatu model klasifikasi yang umumnya biner. ROC menggunakan nilai *sensitivity* / *true positive rate* (TPR) dan 1-*specificity* / *false positive rate* (FPR): 
$$TPR = \frac{TP} {TP + FN} $$
$$FPR = \frac{FP} {FP + TN}$$
yang menjadi *threshold* pengambilan keputusan, di mana garis lurus yang ditarik dari sudut kiri-bawah ke kanan-atas merupakan keputusan *random guess*. Karena kasus ini merupakan klasifikasi biner, maka ROC dapat dilihat sebagai berikut:
- *Logistic Regression* 
![Logit ROC](https://i.ibb.co/R9QD5fm/roc-lr-sub1.png)
- *Support Vector Machine* 
![SVM ROC](https://i.ibb.co/9h16DVv/roc-svm-sub1.png)

Semakin mendekati pojok kiri-atas (nilai 1), maka area dibawah kurva (AUC) semakin besar dan model akan dinilai lebih sempurna. Pada kasus ini, AUC pada SVM sebesar 0.91, lebih besar dari LR yang sebesar 0.89.
### Metrik Evaluasi
Untuk kasus klasifikasi, terdapat 4 metrik umum yang dapat digunakan, yaitu *accuracy*, *precision*, *recall*, dan *f1-score*. Adapun untuk formula dari metrik tersebut adalah sebagai berikut:
$$accuracy = \frac{TP + TN}{TP + FN + TN + FP}$$

$$precision = \frac{TP}{TP + FP}$$

$$recall = \frac{TP}{TP + FN}$$

$$F1 = \frac{2 \times precision \times recall}{precision + recall}$$

#### Akurasi *Training* & *Test*
| Model                    | *Train Acc* | *Test Acc* |
|--------------------------|-------------|------------|
| *Logistic Regression*    | 80.84%      | 80.04%     |
| *Support Vector Machine* | 85.08%      | 83.58%     |

#### Perbandingan Metrik LR vs SVM
| Model                    | *accuracy* | *precision* | *recall* | *f1-score* |
|--------------------------|------------|-------------|----------|------------|
| *Logistic Regression*    | 80.04%     | 78.27%      | 83.53%   | 80.81%     |
| *Support Vector Machine* | 83.58%     | 80.28%      | 89.30%   | 84.55%     |

### Pembahasan
Prediksi ini melihat apakah seseorang berisiko terkena penyakit stroke dengan tujuan untuk pencegahan sedini mungkin. 

Berdasarkan hal ini, *False Positive* (FP) atau prediksi positif "stroke" pada data yang sebenarnya bernilai negatif memiliki pertimbangan yang kecil, karena seseorang yang sebenarnya tidak berisiko terkena stroke akan diprediksi berisiko stroke. Sebaliknya, *False Negative* (FN) atau prediksi negatif pada data yang bernilai positif akan berbahaya, karena hasil tersebut menunjukkan bahwa seseorang tidak berisiko terkena stroke, padahal ia sebenarnya berisiko.

Dari sini, saya melihat model dengan nilai FN yang harus minimum, sementara nilai FP masih bisa ditolerir. Jika dilihat dari *confusion matrix*, *False negative* pada SVM (102) lebih rendah dibandingkan LR (157), begitu pula nilai *False positive* pada SVM (209) yang lebih rendah dari LR yaitu (221).

Jika melihat 4 metrik evaluasi sebelumnya, maka metrik yang dapat dipakai adalah *recall* dan *f1-score*.
Jika kita melihat pengaruh FN, nilai *recall* dapat dipakai karena semakin kecil FN, maka *recall* akan semakin tinggi. Jika kita melihat model pada kasus kesehatan, maka model yang baik sebenarnya harus memiliki nilai FP dan FN yang minimum[[5]](https://www.nature.com/articles/nmeth.3945), yang mana nilai ini akan menaikkan nilai *f1-score*.

### Kesimpulan
Berdasarkan metrik yang telah ditentukan, saya memilih model dengan *Support Vector Machine* (SVM) dibanding *Logistic Regression* (LR). Hal ini karena nilai *recall* pada model SVM mencapai 89.30% dibanding model dengan LR yaitu 83.53%. Kemudian nilai *f1-score* model dengan SVM sebesar 84.55%, lebih tinggi dari LR yaitu 80.81%. 

## Referensi

[1]  “Multiple Cause of Death Data on CDC WONDER.” https://wonder.cdc.gov/mcd.html (accessed May 25, 2023).

[2]  C. W. Tsao _et al._, “Heart Disease and Stroke Statistics-2023 Update: A Report From the American Heart Association,” _Circulation_, vol. 147, no. 8, pp. E93–E621, Feb. 2023, doi: 10.1161/CIR.0000000000001123.

[3]  “Awareness of Stroke Warning Symptoms --- 13 States and the District of Columbia, 2005.” https://www.cdc.gov/mmwr/preview/mmwrhtml/mm5718a2.htm (accessed May 25, 2023).

[4]  "LIBLINEAR -- A Library for Large Linear Classification." https://www.csie.ntu.edu.tw/~cjlin/liblinear/ (accessed May 27, 2023).

[5]  J. Lever, M. Krzywinski, and N. Altman, “Points of Significance: Classification evaluation,” _Nat Methods_, vol. 13, no. 8, pp. 603–604, Jul. 2016, doi: 10.1038/NMETH.3945.
