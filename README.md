# 🫀 Heart Attack Risk Analysis

## 📌 Proje Açıklaması

Bu proje, kalp krizi riskini etkileyen faktörleri analiz etmek ve farklı makine öğrenmesi modellerini karşılaştırarak en yüksek doğrulukla tahmin yapan algoritmayı belirlemek amacıyla hazırlanmıştır. Modelleme sürecinde eksik veri temizliği, görselleştirme, öznitelik kodlama ve hiperparametre optimizasyonları yapılmıştır.

---

## 📊 Kullanılan Modeller
Aşağıdaki sınıflandırma algoritmaları eğitilip karşılaştırılmıştır:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes

Her model için:
- **GridSearchCV** ve **RandomizedSearchCV** ile hiperparametre optimizasyonu yapılmıştır.
- **Test doğruluğu** ölçülmüştür.

---

## 🧪 Kullanılan Kütüphaneler

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Scipy (loguniform dağılım için)

---

## 📁 Veri Kümesi

Veri dosyası: `data.csv`  
- Eksik değerler `?` ile işaretlenmişti, temizlendi.
- Kategorik değişkenler one-hot encoding ile dönüştürüldü.
- Sütunlar: yaş, cinsiyet, kolesterol, kan basıncı, göğüs ağrısı tipi, maksimum kalp atım hızı, ST segment eğimi, vs.
- Hedef sütun: `num` (kalp krizi riski: 0 = risk yok, 1+ = risk var)

---

## 📊 Görselleştirme

Aşağıdaki grafikler oluşturulmuştur:
- `sns.countplot`: Kalp krizi riski dağılımı
- `sns.boxplot`: Yaş - risk ilişkisi
- `sns.violinplot`: Kolesterol - risk ilişkisi
- `sns.heatmap`: Korelasyon matrisi

---

