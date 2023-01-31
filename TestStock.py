import urllib.request
import re
import glob
import time
import numpy as np
import tushare as ts
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
reg = linear_model.LinearRegression()
datalist=[200011,
200012,
200016,
200017,
200019,
200020,
200025,
200026,
200028,
200029,
200030,
200037,
200045,
200054,
200055,
200056,
200058,
200152,
200413,
200429,
200468,
200488,
200505,
200512,
200521,
200530,
200539,
200541,
200550,
200553,
200570,
200581,
200596,
200625,
200706,
200725,
200726,
200761,
200771,
200869,
200992,
201872,
]
df = ts.get_hist_data('399003', start='2020-6-01', end='2021-12-30')
#print(df.index)
X=np.zeros((len(datalist),len(df.index)))
Y=np.zeros((len(df.index),1))

totalDF=df.index
for i in range(0,len(datalist)):
    df = ts.get_hist_data(str(datalist[i]), start='2020-6-01', end='2021-12-30')
    df.to_csv(str(i)+'.csv')
    print(len(df.index))
    tempDf=[]
    for m in df.index:
        if(m in totalDF):
            tempDf.append(m)
    totalDF=tempDf
    #X[i,:]=np.array(df["open"]).reshape(1,X.shape[1])
X=np.zeros((len(totalDF),len(datalist)))
Y=np.zeros((len(totalDF),1))
df = ts.get_hist_data('399003', start='2020-6-01', end='2021-12-30')
for j in range(0,len(totalDF)):
        Y[j]=df['open'][totalDF[j]]
for i in range(0,len(datalist)):
    df = ts.get_hist_data(str(datalist[i]), start='2020-6-01', end='2021-12-30')
    df.to_csv(str(i)+'.csv')
    print(len(df.index))
    
    for j in range(0,len(totalDF)):
        X[j,i]=df['open'][totalDF[j]]
#print(totalDF)
#print(X)
#划分数据集
reg.fit(X[0:200,:],Y[0:200])
Ytest=reg.predict(X[200:,:])
destScore=mean_squared_error(Ytest,Y[200:])
score=destScore
brand=[]
for i in range(0,X.shape[1]):
    brand.append(i)
preX=X
while(score<destScore+1):
    
    scoreNum=[]
    tempBrand=[]
    tempX=np.zeros((X.shape[0],(len(brand)-1)))
    for i in range(0,len(brand)):
        counter=0
        for j in range(0,len(brand)):
            if(j==i):
                continue
            tempX[:,counter]=X[:,brand[j]];
            counter=counter+1
        reg.fit(tempX[0:200,:],Y[0:200])
        Ytest=reg.predict(tempX[200:,:])
        destScore=mean_squared_error(Ytest,Y[200:])
        scoreNum.append(destScore)
    selectM=min(scoreNum)
    score=selectM
    reg.fit(preX[0:200,:],Y[0:200])
    Ytest=reg.predict(preX[200:,:])
    preScore=mean_squared_error(Ytest,Y[200:])
    if(preScore>=score):
        preX=tempX
    if(preScore<score):
        break
    for i in range(0,len(scoreNum)):
        if(selectM==scoreNum[i]):
            continue
        tempBrand.append(brand[i])
    brand=tempBrand
print('代表股票'+str(len(brand))+'只：')
for i in brand:
    print(datalist[i])

