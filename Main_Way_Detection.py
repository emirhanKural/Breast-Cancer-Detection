import numpy as np
from sklearn.impute import SimpleImputer #Soru işareti olan değerleri görmek için
from sklearn.neighbors import KNeighborsClassifier #En yakın komşular sınıflandırıcısını kullanacağız
from sklearn.metrics import accuracy_score #Yüzdesel olarak gerçek verilerle nekadar örtüşdüğünü anlayacağız
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd #Veri setini almak için.

veri = pd.read_csv("breast-cancer-wisconsin.data")

#Verilerimizin içiresinde soru işareti olan yerler var.Onlar için uygulanacak işlem:
veri.replace("?",-99999,inplace=True) # ?lerini -9999ile değiştirdik.Tam önemli değil.

veri = veri.drop(["id"],axis=1) #id kısmını silmek istiyoruz.

#Tahmin etmek istediğimiz kısım son sütun benormal sütunu. 2 ise iyi,4 ise kötü huylu tümör.
y = veri.benormal #Tahmin etmek istediğimiz sütunu atadık y'ye.
x = veri.drop(["benormal"],axis=1) #Burayı tahmin edeceğiz zaten o yüzden siliiyoruz.

# y = np.array(veri.benormal) #Bu şekilde de yazılabilir.Sadece matrissel olarak yapıyor bu işlemi.
# x = np.array(veri.drop(["benormal"],axis=1))


imp = SimpleImputer(missing_values=-99999,strategy="mean") #Sklearn'ün bir özelliği var burada.-99999 yaptığımız sütunun ortalamsını alıp
                                                            #ona göre bir değerlerle yer değiştirecek.
x = imp.fit_transform((x))

#Kross Validation ile verilerimizi eğitim ve test kümelerine ayırıyoruz.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
#x'deki eğitim,x'deki test,y'deki eğitim,y'deki test = Bunları 0.33 oranında ayır.

tahmin = KNeighborsClassifier() # Parantez içini boş bıraktığımızda kendi bir model oluşturuyor.
tahmin.fit(x_train, y_train)  # Eğim kümlerimizi oturuyoruz.
basari = tahmin.score(x_test,y_test) # Test kümelerimizle işleme tutarak başırımızı görüyoruz.
#her derlediğimizde oranımız farklı gelecek çünkü her dafasında rastegele eğitim ve test kümesi alıyor.
print("%",basari*100,"oranında :",tahmin.predict([[1,1,1,1,2,10,3,1,1]]))

# [[1,1,1,1,2,10,3,1,1]] == np.array([1,1,1,1,2,10,3,1,1]).reshape(1,-1) bu şekilde de yazılabilir.


