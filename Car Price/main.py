import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score , root_mean_squared_error , mean_absolute_error

dataset = pd.read_csv('data2.csv')

dataset = dataset.drop(['Unnamed: 0', 'name', 'Mileage Unit'], axis=1)
x = dataset.drop(['selling_price'] , axis=1)
y = dataset['selling_price'].values

cat_dat = ['fuel' , 'seller_type' , 'transmission' , 'owner']

ct = ColumnTransformer(transformers= [('encoder' , OneHotEncoder() , cat_dat )],remainder='passthrough')
x = np.array(ct.fit_transform(x))


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0) 

models = {
    'Linear Reg ' : LinearRegression(),
    'Decision Tree Reg ' : DecisionTreeRegressor(),
    'Random Forest Reg ' : RandomForestRegressor(),#Best Model 
}

np.set_printoptions(precision=2)
for name,model in models.items() :
    model.fit(x_train , y_train)
    y_pred = model.predict(x_test)
    print(f'{name}')
    print(f'The R2 value is : {r2_score(y_test , y_pred)}')
    print(f'The RMSE value is : {root_mean_squared_error(y_test , y_pred)}')
    print(f'The MAE value is : {mean_absolute_error(y_test , y_pred)}\n')
    print(np.concatenate((y_pred.reshape(len(y_pred) , 1) , y_test.reshape(len(y_test) , 1)) , 1))



