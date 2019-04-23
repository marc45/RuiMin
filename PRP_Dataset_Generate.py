import pandas as pd
import numpy as np
import datetime


#  读文件
df = pd.DataFrame(columns=("卷号", "时间", "天气", "最高气温", "温差", "班组", "班次", "前工序", "前机组自检", "前机组", "投料量", "合金", "厚度mm", "宽度mm", "长度m", "重量kg", "外径", "测量入口温度", "厚度超限长度", "测量凸度", "凸度超限长度", "测量乳化液温度", "测量楔形", "楔形超限长度", "测量温度", "F1-F2间张力", "F2-F3间张力", "卷取机张力", "产品", "最终产品", "成品率"))
dt_produce = pd.read_csv('Datasets/热轧/热轧机产出数据.csv')
dt_team = pd.read_excel('Datasets/热轧/热精轧机2014-2018班组汇总.xlsx')
dt_weather = pd.read_excel('Datasets/maweiweather.xlsx')
dt_record = pd.read_csv('Datasets/热轧/热轧机生产汇报_已完成作业.csv')
dt_ratio = pd.read_excel('Datasets/热轧/2012-2018热轧成品率汇总.xlsx')

#  设置变量
coilid = produce_date = weather = T_max = T_diff = banzu = banci = procedure_pre = self_test = jizu_pre = feedqty = alloy = thickness = width = length = weight = Outer_diameter = T_inlet = thickness_out = crown = crown_out = T_emulsion = wedge = wedgr_out = T_measure = F1F2Tension = F2F3Tension = WindingMachine = product = product_final = finishrate = np.NAN

#  匹配数据
for i in range(dt_produce.shape[0]):
    coilid = dt_produce.loc[i, '入口料']  # 卷号
    produce_date = datetime.datetime.strptime(dt_produce.loc[i, '开工时间'], "%Y/%m/%d %H:%M")  # 生产日期
    print(produce_date)
    alloy = dt_produce.loc[i, '合金']
    thickness = dt_produce.loc[i, '厚度mm']
    width = dt_produce.loc[i, '宽度mm']
    length = dt_produce.loc[i, '长度m']
    weight = dt_produce.loc[i, '重量kg']
    Outer_diameter = dt_produce.loc[i, '外径']
    T_inlet = dt_produce.loc[i, '测量入口温度']
    thickness_out = dt_produce.loc[i, '厚度超限长度']
    crown = dt_produce.loc[i, '测量凸度']
    crown_out = dt_produce.loc[i, '凸度超限长度']
    T_emulsion = dt_produce.loc[i, '测量乳化液温度']
    wedge = dt_produce.loc[i, '测量楔形']
    wedgr_out = dt_produce.loc[i, '楔形超限长度']
    T_measure = dt_produce.loc[i, '测量温度']
    F1F2Tension = dt_produce.loc[i, 'F1-F2间张力']
    F2F3Tension = dt_produce.loc[i, 'F2-F3间张力']
    WindingMachine = dt_produce.loc[i, '卷取机张力']
    product_final = dt_produce.loc[i, '最终产品']

    if produce_date.hour < 8:  # 班次
        banci = '大夜'
    elif produce_date.hour > 16:
        banci = '小夜'
    else:
        banci = '白班'
    t = str(produce_date.year) + '/' + str(produce_date.month) + '/' + str(produce_date.day)
    data1 = dt_team[dt_team["fdate"] == t]
    if(data1.shape[0]>0):
        data2 = data1[data1["shift"] == banci]
        if(data2.shape[0]>0):
            data2 = data2.reset_index(drop=True)
            banzu = data2.loc[0, 'crew']  # 班组


    data1 = dt_weather[dt_weather["日期"] == t]
    if(data1.shape[0]>0):
        data1 = data1.reset_index(drop=True)
        weather = data1.loc[0, '天气']  # 天气
        T_max = data1.loc[0, '最高气温']  # 最高温度
        T_diff = data1.loc[0, '温差']  # 温差

    data1 = dt_ratio[dt_ratio["coilid"] == coilid]
    if(data1.shape[0]>0):
        data1 = data1.reset_index(drop=True)
        feedqty = data1.loc[0, 'feedqty']  # 投料量
        finishrate = data1.loc[0, 'finishrate']  # 成品率

    data1 = dt_record[dt_record["来料号"] == coilid]
    if(data1.shape[0]>0):
        data1 = data1.reset_index(drop=True)
        self_test = data1.loc[0, '前机组自检信息']  # 前机组自检信息
        jizu_pre = data1.loc[0, '前机组']  # 前机组
        procedure_pre = data1.loc[0, '前工序']  # 前工序
        product = data1.loc[0, '产品']  # 产品
    new = pd.DataFrame([[coilid, produce_date, weather, T_max, T_diff, banzu, banci, procedure_pre, self_test, jizu_pre, feedqty, alloy, thickness, width, length, weight, Outer_diameter, T_inlet, thickness_out, crown, crown_out, T_emulsion, wedge, wedgr_out, T_measure, F1F2Tension, F2F3Tension, WindingMachine, product, product_final, finishrate]], columns=("卷号", "时间", "天气", "最高气温", "温差", "班组", "班次", "前工序", "前机组自检", "前机组", "投料量", "合金", "厚度mm", "宽度mm", "长度m", "重量kg", "外径", "测量入口温度", "厚度超限长度", "测量凸度", "凸度超限长度", "测量乳化液温度", "测量楔形", "楔形超限长度", "测量温度", "F1-F2间张力", "F2-F3间张力", "卷取机张力", "产品", "最终产品", "成品率"))
    print(new)
    df = df.append(new, ignore_index=True)
df.to_excel('Datasets/成品率预测数据集.xlsx')
print(df)