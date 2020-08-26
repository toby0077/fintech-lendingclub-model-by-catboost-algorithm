# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 20:15:17 2018
@author: 231469242@qq.com
python风控建模实战lendingClub》视频教程：http://dwz.date/b626


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import catboost as cb
import pandas as pd
import numpy as np

#字符串转换为数值型，删除空缺值100%变量
readFileName="data1.xlsx"
#删除信息增益低或0的变量，单一占比高变量，入模型变量99个
#读取excel
data=pd.read_excel(readFileName)
#字符串转换为数值型，删除空缺值100%变量
X=data.ix[:,"loan_amnt":"debt_settlement_flag"]
y=data["target"]
train_x, test_x, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)
 
cb = cb.CatBoostClassifier()
cb.fit(train_x, y_train)

print("accuracy on the training subset:{:.3f}".format(cb.score(train_x,y_train)))
print("accuracy on the test subset:{:.3f}".format(cb.score(test_x,y_test)))


feature_importances=cb.feature_importances_
names=X.columns
list_feature_importances=list(zip(feature_importances,names))
df_feature_importances=pd.DataFrame(list_feature_importances)

df_feature_importances.to_excel("catboost_变量重要性.xlsx")

n_features=X.shape[1]
plt.barh(range(n_features),cb.feature_importances_,align='center')
plt.yticks(np.arange(n_features),X.columns)
plt.title("catboost")
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()













