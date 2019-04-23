from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def sigmoid(x):  # 激活函数
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):  # sigmoid的倒数
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers):  # 这里是三层网络，列表[64,100,10]表示输入，隐藏，输出层的单元个数
        # 初始化权值，范围1~-1
        self.V = np.random.random((layers[0] + 1, layers[1])) * 2 - 1  # 隐藏层权值(65,100)，之所以是65，因为有偏置W0
        self.W = np.random.random((layers[1], layers[2])) * 2 - 1  # (100,10)

    def train(self, X, y, lr=0.005, epochs=20000):
        # lr为学习率，epochs为迭代的次数
        # 为数据集添加偏置
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp  # 这里最后一列为偏置

        # 进行权值训练更新
        for n in range(epochs + 1):
            i = np.random.randint(X.shape[0])  # 随机选取一行数据(一个样本)进行更新
            x = X[i]
            x = np.atleast_2d(x)  # 转为二维数据

            L1 = sigmoid(np.dot(x, self.V))  # 隐层输出(1,100)
            L2 = sigmoid(np.dot(L1, self.W))  # 输出层输出(1,10)

            # delta
            L2_delta = (y[i] - L2) * dsigmoid(L2)  # (1,10)
            L1_delta = L2_delta.dot(self.W.T) * dsigmoid(L1)  # (1,100)，这里是数组的乘法，对应元素相乘

            # 更新
            self.W += lr * L1.T.dot(L2_delta)  # (100,10)
            self.V += lr * x.T.dot(L1_delta)  #

            # 每训练1000次预测准确率
            if n % 1000 == 0:
                predictions = []
                for j in range(X_test.shape[0]):
                    out = self.predict(X_test[j])  # 用验证集去测试
                    predictions.append(np.argmax(out))  # 返回预测结果
                accuracy = np.mean(np.equal(predictions, y_test))  # 求平均值
                print('epoch:', n, 'accuracy:', accuracy)

    def predict(self, x):
        # 添加转置,这里是一维的
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        x = temp
        x = np.atleast_2d(x)

        L1 = sigmoid(np.dot(x, self.V))  # 隐层输出
        L2 = sigmoid(np.dot(L1, self.W))  # 输出层输出
        return L2


filename = '../Datasets/成品率预测数据集--数字化后.xlsx'
data = pd.read_excel(filename).drop(['时间'], axis=1).drop(['卷号'], axis=1)

X = np.array(data.drop(['成品率'], axis=1))  # 数据
y = np.array(data['成品率'])  #标签
num = y.shape[0]
for i in range(num):
    if y[i] < 0.8:
        y[i] = 0
    elif y[i] < 0.85:
        y[i] = 1
    elif y[i] < 0.9:
        y[i] = 2
    elif y[i] < 0.95:
        y[i] = 3
    else:
        y[i] = 4
# 数据归一化,一般是x=(x-x.min)/x.max-x.min
# X -= X.min()
# X /= X.max()
# min_max_scaler = preprocessing.MinMaxScaler()
# X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)  # 默认分割：3:1

# 创建神经网络
nm = NeuralNetwork([7, 100, 5])

print('start')

nm.train(X_train, y_train, epochs=20000)

print('end')