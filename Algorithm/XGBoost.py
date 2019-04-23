import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data(filename):
    data = pd.read_excel(filename)
    data = pd.DataFrame(data)
    x = data.drop(['成品率'], axis=1).drop(['卷号'],axis=1).drop(['时间'],axis=1)
    y = data['成品率']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x_train, x_test, y_train, y_test


filename='../Datasets/成品率预测数据集--数字化后.xlsx'
X_train, X_test, y_train, y_test = load_data(filename)

data_train = xgb.DMatrix(X_train, label=y_train)
data_test = xgb.DMatrix(X_test)
params = {'booster': 'gbtree',
          'max_depth': 7,
          'lambda': 10,
          'subsample': 0.75,
          'colsample_bytree': 0.75,
          'min_child_weight': 1,
          'eta': 0.025,
          'seed': 0,
          'silent': 0,
          'gamma': 0.15,
          'learning_rate': 0.01}

watchlist = [(data_train, 'train')]
bst = xgb.train(params, data_train, num_boost_round=100, evals=watchlist)
ypred = bst.predict(data_test)

print(mean_absolute_error(y_test, ypred))

# plt.figure()
# plt.plot(np.arange(len(ypred)), y_test, label='true value')
# plt.plot(np.arange(len(ypred)), ypred, label='predict value')
# plt.legend()
# plt.show()
#  属性重要度排序
# feat_imp = pd.Series(bst.get_fscore()).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')
# plt.show()