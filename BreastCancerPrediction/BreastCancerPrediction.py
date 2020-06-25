import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

veri = pd.read_csv("breast-cancer-wisconsin.data")#Reading data from .csv file

veri.replace('?', np.nan, inplace=True)#Replace empty values with a numPy's NaN value
veri = veri.drop(['id'], axis=1)#Delete "id" column because it doesn't effect machine learning

y = np.array(veri.benormal)#Define y value as Result(benormal column)
x = np.array(veri.drop(['benormal'], axis=1))#Define parameters of machine learning as x value(out of benormal column)

imp = Imputer(missing_values=np.nan, strategy="mean",axis=0)#Put mean values of that values in columns instead of numPy's NaN values
x = imp.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)#cross-validation method to train and test data
prediction = KNeighborsClassifier(n_neighbors=11)#Perform KNN method
prediction.fit(X_train, y_train)

y_predict = prediction.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

success_ratio = prediction.score(X_test, y_test)#Show success ratio, using test values
print("Success: %"+str(success_ratio*100))

#To try predicting a sample, you can use the code below and see the result
"""new_sample = np.array([6,1,2,3,5,4,8,10,2]).reshape(1,-1)
print(prediction.predict(new_sample))"""