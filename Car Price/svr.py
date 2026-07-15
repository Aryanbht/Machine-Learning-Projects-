import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , root_mean_squared_error , mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('data2.csv')

dataset = dataset.drop(['Unnamed: 0', 'name', 'Mileage Unit'], axis=1)
x = dataset.drop(['selling_price'] , axis=1)
y = dataset['selling_price'].values

cat_dat = ['fuel' , 'seller_type' , 'transmission' , 'owner']

ct = ColumnTransformer(transformers= [('encoder' , OneHotEncoder() , cat_dat )],remainder='passthrough')
x = np.array(ct.fit_transform(x))

y = y.reshape(len(y) , 1) 

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0) 

sc_x = StandardScaler()
sc_y = StandardScaler()

x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

regressor = SVR()
regressor.fit(x_train,y_train)

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.inverse_transform(x_test)).reshape(-1,1))

np.set_printoptions(precision=2)
print(np.concatenate((y_pred, y_test), 1))

print(r2_score(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))


#Bad Model 
