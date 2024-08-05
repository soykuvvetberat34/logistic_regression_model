from sklearn.metrics import mean_squared_error,roc_auc_score,accuracy_score,confusion_matrix,classification_report,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score#test hatası hesaplama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df["Outcome"]#.value_counts() dersen y nin eleman sayısı hakkında bilgi veriğr
df=df.drop(["Outcome"],axis=1).astype("float64")
infos=df.describe().T#columns değerlerinin eleman sayısı ortalama max gibi değerlerini döndürür
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.25,random_state=200)
loj_model=LogisticRegression(solver="liblinear")
loj_model.fit(x_train,y_train)
pred=loj_model.predict(x_test)
#classification report
#sınıflandırma raporu döndürür
cr=classification_report(y_test,pred)
#accuracy_score
#doğruluk oranını verir 
ascore=accuracy_score(y_test,pred)
#confusion matrix
#karmaşıklık matrisini döndürür
cm=confusion_matrix(y_test,pred)

cvs=cross_val_score(loj_model,x_test,y_test,cv=10).mean()#10 adet test hatası hesapla ve ortalamalarını al
print(cvs)

























