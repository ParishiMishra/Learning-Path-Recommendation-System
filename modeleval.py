import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

data = pd.read_csv('data/research_paper_data.csv')

X = data.drop('target', axis=1)
y = data['target']

model = joblib.load('model.pkl')

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse}')
