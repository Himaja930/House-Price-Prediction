import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#
dataset=pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
dataset.shape
dataset.columns
dataset["availability"].value_counts()
a=pd.get_dummies(dataset['availability'])
dataset=pd.concat([a,dataset],axis=1)
dataset.shape
del dataset['availability']
dataset["society"].value_counts()
dataset["area_type"].value_counts()
b=pd.get_dummies(dataset['area_type'])
dataset=pd.concat([b,dataset],axis=1)
del dataset['area_type']
q=dataset["location"].value_counts()
c=pd.get_dummies(dataset['location'])
wanted_location=['Whitefield','Electronic City','Kanakpura Road','Thanisandra', 'Yelahanka','Uttarahalli','Hebbal','Marathahalli', 'Hennur Road','Bannerghatta Road','7th Phase JP Nagar','Haralur Road']
e=c[wanted_location]
dataset=pd.concat([e,dataset],axis=1)
del dataset['location']
exdata=dataset[['total_sqft', 'bath', 'balcony','size']]

from sklearn.preprocessing import StandardScaler
scaling=StandardScaler() 
for i in range(0,exdata.shape[0]):
    z=exdata['total_sqft'].iloc[i]
    if type(z)==str:
        z=int(z[0:4])
        exdata['total_sqft'].iloc[i]=z
d1=scaling.fit_transform(exdata)
d1=pd.DataFrame(d1)
dataset=pd.concat([d1,dataset],axis=1)
del dataset['total_sqft']
del dataset['bath']
del dataset['balcony']
del dataset['size']
del dataset['society']

from sklearn.model_selection import train_test_split
X=dataset[[0,1,2,3,'Whitefield','Electronic City','Kanakpura Road','Ready To Move']]
y=dataset[['price']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
clf.score(X_test,y_test)
clf.score(X_train,y_train)

from sklearn.ensemble import GradientBoostingRegressor
clf1 = GradientBoostingRegressor(n_estimators=500, max_depth = 15, min_samples_split = 30, learning_rate = 0.1, max_features=14, subsample = 0.94,  random_state=3, max_leaf_nodes=41)
clf1.fit(X_train,y_train)
y_pred=clf1.predict(X_test)
clf1.score(X_test,y_test)
clf1.score(X_train,y_train)

from sklearn.ensemble import RandomForestRegressor
clf2 = RandomForestRegressor(n_estimators=100,criterion='mse', max_depth=None,
                             min_samples_split=2, min_samples_leaf=1,
                             max_features='auto', max_leaf_nodes=None,
                             bootstrap=True, oob_score=False, n_jobs=None, 
                             random_state=None,verbose=0,
warm_start=False, ccp_alpha=1,
                             max_samples=None)
clf2.fit(X_train,y_train)
y_pred=clf2.predict(X_test)
clf2.score(X_test,y_test)
clf2.score(X_train,y_train)

from sklearn.ensemble import AdaBoostRegressor
clf3= AdaBoostRegressor(base_estimator=clf2, n_estimators=50, learning_rate=1.0, 
                        loss='linear', random_state=None)
clf3.fit(X_train,y_train)
y_pred=clf3.predict(X_test)
clf3.score(X_test,y_test)
clf3.score(X_train,y_train) 
