import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

dataset = pd.read_csv('student_performance_dataset.csv')

dataset = dataset.drop(['Student_ID', 'Pass_Fail'], axis=1)

x = dataset.drop('Final_Exam_Score', axis=1)
y = dataset['Final_Exam_Score'].values 

cat_data = ['Gender', 'Parental_Education_Level',
            'Internet_Access_at_Home', 'Extracurricular_Activities']

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), cat_data)
], remainder='passthrough')

x = np.array(ct.fit_transform(x))  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

models = {
    'MultiLinear Reg':   LinearRegression(),
    'Decision Tree Reg': DecisionTreeRegressor(random_state=0),
    'Random Forest Reg': RandomForestRegressor(n_estimators=100, random_state=0),
}

np.set_printoptions(precision=2)
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'{name}')
    print(f'  R²:   {round(r2_score(y_test, y_pred), 4)}')
    print(f'  RMSE: {round(root_mean_squared_error(y_test, y_pred), 4)}')
    print(f'  MAE:  {round(mean_absolute_error(y_test, y_pred), 4)}\n')
    # print(np.concatenate((
    #     y_pred.reshape(len(y_pred), 1),
    #     y_test.reshape(len(y_test), 1)
    # ), axis=1))
    print()