# 🫀 Heart Attack Risk Analysis

# 📌 Proje Açıklaması
Bu proje,Heart Attack Prediction veri seti (https://www.kaggle.com/datasets/imnikhilanand/heart-attack-prediction) kullanılarak kalp krizi riskini tahmin etmeyi amaçlayan bir makine öğrenmesi modelini geliştirmektedir. Veri seti Kaggle platformundan alınmıştır ve çeşitli demografik ve klinik özelliklere dayalı olarak kalp hastalığı riskini sınıflandırmak için kullanılmıştır.

## 📊 Kullanılan Modeller ve Sonuçlar

KNN (K-Nearest Neighbors)

Grid Search ile en iyi parametreler: metric=euclidean, n_neighbors=16, weights=uniform

Grid Search CV Score: 0.839

Random Search ile en iyi parametreler: metric=manhattan, n_neighbors=35, weights=uniform

Random Search CV Score: 0.834

Test Accuracy: 0.831

---------------------------------------------------------------------------------------------------------------------------------------------------------------

Decision Tree

Grid Search ile en iyi parametreler: criterion=gini, max_depth=5, max_features=sqrt, min_samples_leaf=1, min_samples_split=2

Grid Search CV Score: 0.805

Random Search ile en iyi parametreler: min_samples_split=10, min_samples_leaf=4, max_features=None, max_depth=5, criterion=gini

Random Search CV Score: 0.791

Test Accuracy: 0.814 

--------------------------------------------------------------------------------------------------------------------------------------------------------------


Random Forest

Grid Search ile en iyi parametreler: n_estimators=300, max_depth=5, max_features=sqrt, min_samples_leaf=1, min_samples_split=10, bootstrap=True, criterion=gini

Grid Search CV Score: 0.834

Random Search ile en iyi parametreler: n_estimators=500, max_depth=20, max_features=None, min_samples_leaf=4, min_samples_split=5, bootstrap=True, criterion=entropy

Random Search CV Score: 0.826

Test Accuracy: 0.831

------------------------------------------------------------------------------------------------------------------------------------------------------------------

Logistic Regression

Grid Search ile en iyi parametreler: C=0.01, penalty=l2, solver=liblinear, class_weight=None

Grid Search CV Score: 0.843

Random Search ile en iyi parametreler: C≈0.00996, penalty=l2, solver=liblinear, class_weight=None

Random Search CV Score: 0.843

Test Accuracy: 0.864

-------------------------------------------------------------------------------------------------------------------------------------------------------------

SVM (Support Vector Machine)

Grid Search ile en iyi parametreler: C=0.01, kernel=linear, gamma=scale, class_weight=balanced

Grid Search CV Score: 0.843

Random Search ile en iyi parametreler: C≈0.054, kernel=rbf, gamma=auto, class_weight=balanced

Random Search CV Score: 0.838

Test Accuracy: 0.847

------------------------------------------------------------------------------------------------------------------------------------------------------------------

Naive Bayes

Grid Search ile en iyi parametreler: var_smoothing=1e-09

Grid Search CV Score: 0.821

Random Search ile en iyi parametreler: var_smoothing≈1.07e-07

Random Search CV Score: 0.821

Test Accuracy: 0.831

---------------------------------------------------------------------------------------------------------------------------------------------------------------

En yüksek test doğruluğu Logistic Regression ile elde edilmiştir (~86.4%).

KNN, SVM ve Random Forest modelleri de iyi performans göstermiştir (~83–85%).
Her model için:
- **GridSearchCV** ve **RandomizedSearchCV** ile hiperparametre optimizasyonu yapılmıştır.
- **Test doğruluğu** ölçülmüştür.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

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

