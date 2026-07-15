import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , root_mean_squared_error , mean_absolute_error

dataset = pd.read_csv('data2.csv')

dataset = dataset.drop(['Unnamed: 0', 'name', 'Mileage Unit'], axis=1)
x = dataset.drop(['selling_price'] , axis=1)
y = dataset['selling_price'].values

cat_dat = ['fuel' , 'seller_type' , 'transmission' , 'owner']

ct = ColumnTransformer(transformers= [('encoder' , OneHotEncoder() , cat_dat )],remainder='passthrough')
x = np.array(ct.fit_transform(x))


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0) 

pol_reg = PolynomialFeatures(degree=4)
x_train = pol_reg.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_train , y_train)

y_pred = regressor.predict(pol_reg.transform(x_test))
print(np.concatenate((y_pred.reshape(len(y_pred) , 1) , y_test.reshape(len(y_test) , 1)) , 1))

print(r2_score(y_test , y_pred))
print(root_mean_squared_error(y_test , y_pred))
print(mean_absolute_error(y_test , y_pred))