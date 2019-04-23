import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


# 超参数
infer_seq_length = 50  # 设定的历史序列长度
split_rate = 0.9


def create_model():
    model = Sequential()
    # 输入数据的shape为(n_samples, timestamps, features)
    # 隐藏层设置为256, input_shape元组第二个参数1意指features为1
    # 下面还有个lstm，故return_sequences设置为True
    model.add(LSTM(units=256, activation='tanh', input_shape=(None,1), return_sequences=True))
    model.add(LSTM(units=256))
    # 后接全连接层，直接输出单个值，故units为1
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


# 数据归一化
df = pd.read_excel('../Datasets/成品率预测数据集--数字化后.xlsx')['成品率']
df = np.array(df).reshape(-1, 1)
scaler_minmax = MinMaxScaler()
data = scaler_minmax.fit_transform(df)

# 将原始(,1)序列转化为(,infer_seq_length)
d = []
for i in range(data.shape[0]-infer_seq_length):
    d.append(data[i:i+infer_seq_length+1].tolist())
d = np.array(d)

#  划分x和y，截取前90%的记录作为训练集
X_train, y_train = d[:int(d.shape[0]*split_rate), :-1], d[:int(d.shape[0]*split_rate), -1]

model = create_model()
model.fit(X_train, y_train, batch_size=100, epochs=20, validation_split=0.1)

y_true = scaler_minmax.inverse_transform(d[int(len(d)*split_rate):, -1])
y_predict = scaler_minmax.inverse_transform(model.predict(d[int(len(d)*split_rate):, :-1]))

dt = {'true': y_true.reshape(-1), 'predict': y_predict.reshape(-1)}
dt = pd.DataFrame(dt)
dt.to_excel('../Datasets/LSTM.xls')

plt.plot()
plt.plot(y_true, label='true data')
plt.plot(y_predict, 'r:', label='predict')
plt.legend()
plt.show()