from sklearn.externals import joblib
import pandas as pd

#load model

X = pd.read_excel('Datasets/测试.xlsx')
print(X)
rfc2 = joblib.load('Model/ProductRate_Rezha.pkl')
print(rfc2.predict(X))