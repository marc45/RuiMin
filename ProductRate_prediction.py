import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def load_data(filename):
    data = pd.read_excel(filename)
    data = pd.DataFrame(data)
    x = data.drop(['成品率'], axis=1).drop(['卷号'],axis=1).drop(['时间'],axis=1)
    y = data['成品率']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x_train, x_test, y_train, y_test


def try_different_method(model, method):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    dt = pd.DataFrame()
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'g-',label='true value')
    plt.plot(np.arange(len(result)),result,'r-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()
    dt['true'] = y_test
    dt['predict'] = result
    dt.to_excel('Datasets/' + method +'.xls')
    print(method, ':', mean_absolute_error(y_test, result))


filename='Datasets/成品率预测数据集--数字化后.xlsx'
x_train, x_test, y_train, y_test = load_data(filename)

from sklearn import neighbors
model = neighbors.KNeighborsRegressor()
try_different_method(model, 'KNeighborsRegressor')

from sklearn import svm
model = svm.SVR()
try_different_method(model, 'SVR')

from sklearn import ensemble
model = ensemble.AdaBoostRegressor(n_estimators=50)
try_different_method(model, 'Adaboost')

from sklearn import tree
model = tree.DecisionTreeRegressor()
try_different_method(model, 'DecisionTree')