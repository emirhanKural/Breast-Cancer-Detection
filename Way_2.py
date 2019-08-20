import numpy as np
from sklearn.impute import SimpleImputer #Soru işareti olan değerleri görmek için
from sklearn.neighbors import KNeighborsClassifier #En yakın komşular sınıflandırıcısını kullanacağız
from sklearn.metrics import accuracy_score #Yüzdesel olarak gerçek verilerle nekadar örtüşdüğünü anlayacağız
import pandas as pd #Veri setini almak için.

veri = pd.read_csv("breast-cancer-wisconsin.data")

#Verilerimizin içiresinde soru işareti olan yerler var.Onlar için uygulanacak işlem:
veri.replace("?",-99999,inplace=True) # ?lerini -9999ile değiştirdik.Tam önemli değil.

veri = veri.drop(["id"],axis=1) #id kısmını silmek istiyoruz.

#Tahmin etmek istediğimiz kısım son sütun benormal sütunu. 2 ise iyi,4 ise kötü huylu tümör.
y = veri.benormal #Tahmin etmek istediğimiz sütunu atadık y'ye.
x = veri.drop(["benormal"],axis=1) #Burayı tahmin edeceğiz zaten o yüzden siliiyoruz.

imp = SimpleImputer(missing_values=-99999,strategy="mean") #Sklearn'ün bir özelliği var burada.-99999 yaptığımız sütunun ortalamsını alıp
                                                            #ona göre bir değerlerle yer değiştirecek.
x = imp.fit_transform((x))

# #kaç komşu kullanmak iyii olacak. Bunu deniyoruz:
# for z in range(25):
#     z = 2*z+1 #k değeri tek olmalı. Onu garanti altına alıyoruz.
#
#     tahmin = KNeighborsClassifier(n_neighbors=z, weights="uniform", algorithm="auto", leaf_size=30, p=2,
#                                   metric="euclidean", metric_params=None, n_jobs=1)
#     # 3 Komşuya bakacağız,ağırlığımız yok,algoritmayı kendi belirlesin,  ,  ,euclidean karesini alarak yapacak,
#     tahmin.fit(x, y)  # x ve y değerlerini oturtuyoruz.
#     yTahmin = tahmin.predict(x)  # Xleri tahmin ediyoruz y'lere göre
#
#     basari = accuracy_score(y, yTahmin, normalize=True, sample_weight=None)
#
#     print(z," kadar komşu da başarı :",basari)
# #Bu for döngüsünden çıkan başarı oranlarına göre en iyi komşu sayısı 3 . O yüzden onu kullanacağız.



tahmin = KNeighborsClassifier(n_neighbors=3, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="euclidean", metric_params=None, n_jobs=1)
                              # 3 Komşuya bakacağız,ağırlığımız yok,algoritmayı kendi belirlesin,  ,  ,euclidean karesini alarak yapacak,
tahmin.fit(x, y)  # x ve y değerlerini oturtuyoruz.
yTahmin = tahmin.predict(x)  # Yleri tahmin ediyoruz x'lere göre

basari = accuracy_score(y, yTahmin, normalize=True, sample_weight=None)

print("%",basari*100,"oranında :",tahmin.predict([[1,1,1,1,2,10,3,1,1]]))

# [[1,1,1,1,2,10,3,1,1]] == np.array([1,1,1,1,2,10,3,1,1]).reshape(1,-1) bu şekilde de yazılabilir.

