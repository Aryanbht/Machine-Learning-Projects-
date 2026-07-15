import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import seaborn as sns

dataset = pd.read_csv('housing.csv/housing.csv')


x = dataset.drop('median_house_value' , axis=1)
y = dataset['median_house_value'].values

imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
x[['total_bedrooms']] = imputer.fit_transform(x[['total_bedrooms']])

ct = ColumnTransformer(transformers= [('encoder' , OneHotEncoder() , [8] )] , remainder='passthrough')
x = np.array(ct.fit_transform(x))


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

regressor = RandomForestRegressor(n_estimators=10 , random_state=0)

regressor.fit(x_train , y_train)
y_pred = regressor.predict(x_test)

print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("RMSE:    ", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("MAE:     ", round(mean_absolute_error(y_test, y_pred), 2))

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred) , 1) , y_test.reshape(len(y_test) , 1)) , 1))
