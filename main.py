# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:57:28 2020

@author: RADHIKA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cardata = pd.read_csv("D:\\DS ALL Projets\\CarPredictionProj\\cardata.csv",encoding = "ISO-8859-1")
cardata.head()
cardata.shape
print(cardata['Seller_Type'].unique())
print(cardata['Transmission'].unique())
print(cardata['Owner'].unique())
cardata.isnull().sum()
cardata=cardata.drop(['Car_Name'],axis=1)
cardata.columns
cardata['Current_year']=2020
cardata.head
cardata['no_year']=cardata['Current_year']-cardata['Year']
cardata.head()
cardata=cardata.drop(['Year'],axis=1)
cardata=cardata.drop(['Current_year'],axis=1)
cardata.head
cardata=pd.get_dummies(cardata,drop_first=True)
cardata.head()
cardata.corr()
import seaborn as sns
sns.pairplot(cardata)
corrmat=cardata.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(cardata[top_corr_features].corr(),annot=True,cmap="RdYlGn")
##Independent and dependent variables
x=cardata.iloc[:,1:]
y=cardata.iloc[:,0]


#Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)
feature_importances=pd.Series(model.feature_importances_,index=x.columns)
feature_importances.nlargest(5).plot(kind="barh")
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_train

from sklearn.ensemble import RandomForestRegressor
n_estimators=[int(x) for x in np.linspace(start=100, stop=1200,num=12)]
print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(5, 30,num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]
random_grid={"n_estimators":n_estimators,'max_features':max_features,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}
rf=RandomForestRegressor()

rf_random=RandomizedSearchCV(estimator=rf, param_distributions=random_grid,n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)
rf_random.fit(X_train,y_train)
predictions=rf_random.predict(X_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)

import pickle
file=open('random_forest.pkl','wb')
pickle.dump(rf_random,file)
