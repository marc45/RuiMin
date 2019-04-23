import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(filename):
    data = pd.read_csv(filename)
    data = pd.DataFrame(data)
    x = data.drop(['成品率'],axis=1).drop(['日期'],axis=1)
    y = data['成品率']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test


def try_different_method(model):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()


filename='../Datasets/maweiweather.csv'
x_train, x_test, y_train, y_test = load_data(filename)

##决策树回归
from sklearn import tree
model = tree.DecisionTreeRegressor()
try_different_method(model)

##线性回归
from sklearn import linear_model
model = linear_model.LinearRegression()
try_different_method(model)

##SVR回归
from sklearn import svm
model = svm.SVR()
try_different_method(model)

##KNN回归
from sklearn import neighbors
model = neighbors.KNeighborsRegressor()
try_different_method(model)

##Adaboost回归
from sklearn import ensemble
model = ensemble.AdaBoostRegressor(n_estimators=50)
try_different_method(model)

##GBRT回归
from sklearn import ensemble
model = ensemble.GradientBoostingRegressor(n_estimators=100)
try_different_method(model)

##ExtraTree极端随机树回归
from sklearn.tree import ExtraTreeRegressor
model = ExtraTreeRegressor()
try_different_method(model)