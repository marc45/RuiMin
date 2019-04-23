#  高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error


filename = '../Datasets/maweiweather.csv'
data = pd.read_csv(filename).drop(['日期'], axis=1)

X = np.array(data.drop(['成品率'], axis=1))
y = np.array(data['成品率'])

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
num = y.shape[0]
for i in range(num):
    if y[i] < 0.9:
        y[i] = 0
    else:
        y[i] = 1

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
mae = mean_absolute_error(y_pred, y_test)
print(y_pred, y_test)
accuracy = np.mean(np.equal(y_pred, y_test))
print('accuracy:', accuracy)