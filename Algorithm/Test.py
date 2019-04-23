import pandas as pd
import numpy as np
import datetime
import time

# t = '2018-01-01 15:09:09'
# timeStruct = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
# t1 = str(timeStruct.year)+'/'+str(timeStruct.month)+'/'+str(timeStruct.day)
# timeStruct = datetime.datetime.strptime(t1, "%Y/%m/%d")
# timeStruct =timeStruct.strftime('%Y/%m/%d')
#
# print(t1)
# t2 = '2018/1/1'
# print(t1==t2)

# region 按照时间段划分班组
data1 = pd.read_excel('../Datasets/热轧/热轧机生产汇报_产出数据14-18.xlsx')
data2 = pd.read_excel('../Datasets/热轧/热精轧机2014-2018班组汇总.xlsx')
for i in range(data1.shape[0]):
    t = data1.iat[i, 6]
    t1 = str(t.year)+'/'+str(t.month)+'/'+str(t.day)
    print(t1)
    if t.hour < 8:
        banci = '大夜'
    elif t.hour > 16:
        banci = '小夜'
    else:
        banci = '白班'
    data3 = data2[data2['fdate']== t1]
    if(data3.shape[0]>0):
        data4 = data3[data3['shift'] == banci]
        if(data4.shape[0]>0):
             data1.set_value(i, '班组', data4.iat[0, 5])

data1.to_excel('../Datasets/热轧/222.xlsx')
# end

# data2 = data1[data1['banci'] == 'D班']
# # print(data4['finishrate'].mean())
# # print(data4[data4['finishrate'] < 0.8].shape)
# print(data2.shape[0])
# print(data2['finishrate'].mean())
# print(data2[data2['finishrate']<0.8].shape[0]/data2.shape[0])
# print(data2['finishrate'].std())

# sum = 0
# j = 0
# for i in range(data2.shape[0]):
#     t = data2.iat[i, 0]
#     if (t>=datetime.datetime.strptime('2018-01-01 00:00:00','%Y-%m-%d %H:%M:%S') and t< datetime.datetime.strptime('2019-01-01 00:00:00','%Y-%m-%d %H:%M:%S')):
#         sum += data2.iat[i, 11]
#         j += 1
#
# print(sum/j)


# region 统计各班组不良品的个数
# data1 = pd.read_excel('../Datasets/2018年热轧机不良品.xlsx')
# data2 = pd.read_excel('../Datasets/工艺段成平率明细最终.xlsx')
# a = 0
# b = 0
# c = 0
# d = 0
# print(data1.shape[0])
# for i in range(data1.shape[0]):
#     id = data1.iat[i, 6]
#     data3 = data2[data2['coilid'] == id]
#     if(data3.shape[0]>0):
#         banci = data3.iat[0, 7]
#         if (banci == 'A班'):
#             a += 1
#         elif(banci == 'B班'):
#             b += 1
#         elif(banci == 'C班'):
#             c += 1
#         elif(banci == 'D班'):
#             d += 1
# print(a, b, c, d)
# end

# 计算每个月的平均成品率
# year = [2014, 2015, 2016, 2017, 2018]
# banzu = ['A班', 'B班', 'C班', 'D班']
# data1 = pd.read_excel('../Datasets/热轧/111.xlsx')
# data2 = data1[data1['fname']=='热轧卷']
# data2 = data2.set_index('date')
# for i in year:
#     for j in range(1, 13):
#         t = str(i) + '/' + str(j)
#         data3 = data2[t]
#         print('总:', t, ':')
#         print(data3['finishrate'].mean())
#         for k in banzu:
#             data4 = data3[data3['banci']==k]
#             print(k, ':')
#             print(data4['finishrate'].mean())
# end