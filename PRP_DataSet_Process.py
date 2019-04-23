import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


dt = pd.read_excel('Datasets/成品率预测数据集.xlsx')
print(dt.shape[0], dt.shape[1])
dt = dt.reset_index(drop=True)
# a = LabelEncoder().fit_transform(dt['前机组'])
# b = OneHotEncoder(sparse=False).fit_transform(a.reshape(-1,1))
# print(b)
# print(a)
# c = LabelBinarizer().fit_transform(dt['合金'])
# print(c)
#最大最小归一化
def min_max(df):
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

for i in range(dt.shape[0]):
    weather = dt.loc[i, '天气']
    self_test = dt.loc[i, '前机组自检']
    product_final = dt.loc[i, '最终产品']

    if pd.isnull(weather):
        dt.set_value(i, '天气', 0)
    elif '雨' in str(weather):
        dt.set_value(i, '天气', 1)
    else:
        dt.set_value(i, '天气', 2)

    if pd.isnull(self_test):
        dt.set_value(i, '前机组自检', 0)
    elif (str(self_test).strip('/').strip(' ').isalnum()):
        dt.set_value(i, '前机组自检', 1)
    elif '正常' in str(self_test):
        dt.set_value(i, '前机组自检', 1)
    else:
        dt.set_value(i, '前机组自检', 2)

    if pd.isnull(product_final):
        dt.set_value(i, '最终产品', dt.loc[i, '产品'])

a = ['班组', '班次', '前工序', '前机组', '合金']
for j in a:
    dt[j].fillna('空值', inplace=True)
    dt[j] = LabelEncoder().fit_transform(dt[j])
dt[['产品', '最终产品']] = dt[['产品', '最终产品']].apply(LabelEncoder().fit_transform)

b = ['最高气温', '温差', '投料量', '厚度mm', '宽度mm', '长度m', '重量kg', '外径', '测量入口温度', '厚度超限长度', '测量凸度', '凸度超限长度', '测量乳化液温度', '测量楔形', '楔形超限长度', '测量温度', 'F1-F2间张力', 'F2-F3间张力', '卷取机张力']
for k in b:
    dt[k].fillna(0, inplace=True)
    dt[k] = min_max(dt[k])
print(dt.shape[0], dt.shape[1])
dt.to_excel('Datasets/成品率预测数据集--数字化后.xlsx')