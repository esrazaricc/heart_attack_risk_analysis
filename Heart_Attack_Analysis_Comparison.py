import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


df=pd.read_csv("C:\\Users\\Esra\\Downloads\\data.csv",na_values='?')
print(df.isnull().sum())
df.drop(['slope','ca','thal'],axis=1,inplace=True)   
df.fillna(df.mean(numeric_only=True),inplace=True)  
df.columns=df.columns.str.strip()
num_value=df['num'].value_counts()
print(num_value)
categoric_columns=[col for col in df.columns if df[col].dtype in ['int64','float64'] and df[col].nunique()<10 and col!="num"]
df_encoded=pd.get_dummies(df,columns=categoric_columns,drop_first=True)
print("df columns:",df.columns)
plt.figure()
sns.countplot(x='num',data=df)
plt.figure()
sns.boxplot(x='num',y='age',data=df)
plt.figure()
sns.violinplot(x='num',y='chol',data=df)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.show()


X=df.drop("num",axis=1)
y=df["num"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#KNN
knn=KNeighborsClassifier(n_neighbors=3)
max_neighbors = min(50, X_train.shape[0])
knn_param={'n_neighbors': np.arange(2, max_neighbors+1),
                'weights':['uniform','distance'],
                'metric':["euclidean","manhattan"]}
knn_grid_search=GridSearchCV(knn,knn_param,cv=10)
knn_grid_search.fit(X_train,y_train)
print("knn grid search best parameters:",knn_grid_search.best_params_)
print("knn grid search best score:",knn_grid_search.best_score_)
knn_random_search=RandomizedSearchCV(knn,knn_param,n_iter=5)
knn_random_search.fit(X_train,y_train)
print("knn random search best parameters:",knn_random_search.best_params_)
print("knn random search best results:",knn_random_search.best_score_)
knn_model = knn_grid_search.best_estimator_
test_accuracy = knn_model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)



#DECİSİON TREE CLASSİFİER
tree_clf=DecisionTreeClassifier(criterion="gini",max_depth=10,random_state=42)
tree_clf_param={
            "max_depth":[5,20,35,70,None],
            "min_samples_split":[2,5,10],
            "min_samples_leaf":[1,2,4],
            "criterion":["gini","entropy"],
            "max_features":[None,"sqrt","log2"]
    
    
    }
kf=KFold(n_splits=10)
tree_grid_search_kf=GridSearchCV(tree_clf,tree_clf_param,cv=kf)
tree_grid_search_kf.fit(X_train,y_train)
print("decision tree grid search best parameters:",tree_grid_search_kf.best_params_)
print("decision tree grid search best score:",tree_grid_search_kf.best_score_)

tree_random_search_kf=RandomizedSearchCV(tree_clf,tree_clf_param,n_iter=50)
tree_random_search_kf.fit(X_train,y_train)
print("decision tree random search best parameters:",tree_random_search_kf.best_params_)
print("decision tree random search best score:",tree_random_search_kf.best_score_)
tree_model = tree_grid_search_kf.best_estimator_
test_accuracy = tree_model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)


#RANDOM FOREST CLASSİFİER
rf_clf=RandomForestClassifier(random_state=42)
rf_clf_param = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],        
    'min_samples_split': [2, 5, 10],       
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],            
    'criterion': ['gini', 'entropy']
}
rf_grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=rf_clf_param,
    cv=5,               
    scoring='accuracy', 
    n_jobs=-1
)
rf_grid_search.fit(X_train,y_train)
print("random forest grid search best parameters:",rf_grid_search.best_params_)
print("random forest frid search best score:",rf_grid_search.best_score_)

rf_random_search=RandomizedSearchCV(
    estimator=rf_clf,
    param_distributions=rf_clf_param,
    n_iter=50,
    scoring="accuracy",
    n_jobs=-1
    
     )
rf_random_search.fit(X_train,y_train)
print("random forest random search best parameters:",rf_random_search.best_params_)
print("random forest random search best score:",rf_random_search.best_score_)   
rf_model = rf_grid_search.best_estimator_
test_accuracy = rf_model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)  


#LOGİSTİC REGRESSİON
log_reg=LogisticRegression(penalty="l2",C=1,solver="liblinear",max_iter=100)
log_reg_param_grid = {
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced']
}


log_reg_grid_search=GridSearchCV(
    estimator=log_reg,
    param_grid=log_reg_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,                 
    return_train_score=True  
    
    )
log_reg_grid_search.fit(X_train,y_train)
print("logistic regression grid search best parameters:",log_reg_grid_search.best_params_)
print("logistic regression grid search best score:",log_reg_grid_search.best_score_)

log_reg_param_random = {
    'solver': ['liblinear'],              
    'penalty': ['l1', 'l2'],
    'C': loguniform(0.001, 100),          
    'class_weight': [None, 'balanced']
}

log_reg_random_search=RandomizedSearchCV(
    estimator=log_reg,
    param_distributions=log_reg_param_random,
    n_iter=20,
    cv=5,                         
    scoring='accuracy',
    n_jobs=-1,                   
    verbose=1,
    random_state=42,
    return_train_score=True
    
    )
log_reg_random_search.fit(X_train,y_train)
print("logistic regression random search best parameters:",log_reg_random_search.best_params_)
print("logistic regression random search best score:",log_reg_random_search.best_score_)
logreg_model = log_reg_grid_search.best_estimator_
test_accuracy = logreg_model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)
#SVM
svm_clf=SVC()
svm_param_random = {
    'C': loguniform(1e-2, 1e2),
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}
svm_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}
svm_grid_search=GridSearchCV(svm_clf,
                             svm_param_grid,
                             cv=5,
                             scoring="accuracy",
                             verbose=1,
                             n_jobs=-1)
svm_grid_search.fit(X_train,y_train)
print("SVM grid search best parameters:",svm_grid_search.best_params_)
print("SVM grid search best score:",svm_grid_search.best_score_)

svm_random_search=RandomizedSearchCV(estimator=svm_clf,
                                     param_distributions=svm_param_random,
                                     n_iter=30,
                                     cv=5,
                                     scoring="accuracy",
                                     verbose=1,
                                     n_jobs=-1,
                                     random_state=42)
svm_random_search.fit(X_train,y_train)
print("SVM random search best parameters:",svm_random_search.best_params_)
print("SVM random search best score:",svm_random_search.best_score_)
svm_model = svm_grid_search.best_estimator_
test_accuracy = svm_model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)

#NAİVE BAYES
nb_clf=GaussianNB()
nb_param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}
nb_grid_search=GridSearchCV(nb_clf,
                            nb_param_grid,
                            cv=5,
                            scoring="accuracy",
                            verbose=1,
                            n_jobs=-1)
nb_grid_search.fit(X_train,y_train)
print("NAİVE BAYES grid search best parameters:",nb_grid_search.best_params_)
print("NAİVE BAYES grid search best score:",nb_grid_search.best_score_)

nb_param_random = {
    'var_smoothing': np.logspace(-12, -6, 100)  
}
nb_random_search = RandomizedSearchCV(
    estimator=nb_clf,
    param_distributions=nb_param_random,
    n_iter=20,           
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)
nb_random_search.fit(X_train,y_train)
print("NAİVE BAYES random search best parameters:",nb_random_search.best_params_)
print("NAİVE BAYES random search best score:",nb_random_search.best_score_)
nb_model = nb_grid_search.best_estimator_
test_accuracy = nb_model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)
