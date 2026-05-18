import pandas as pd
import matplotlib.pyplot as plt 
from predict import load_params

data = pd.read_csv('data.csv')
params = load_params()

if not params or params["max_mileage"] == 0:
    print("Train the model first!")
    exit()

theta0 = params["theta0"]
theta1 = params["theta1"]
min_km = params["min_mileage"]
max_km = params["max_mileage"]
min_p = params["min_price"]
max_p = params["max_price"]

y_start_norm = theta0 + (theta1 * 0) 
y_start = (y_start_norm * (max_p - min_p)) + min_p

# For the end of the line (max mileage)
y_end_norm = theta0 + (theta1 * 1)
y_end = (y_end_norm * (max_p - min_p)) + min_p

plt.scatter(data['km'], data['price'], color='blue', label='Data Points')


plt.plot([min_km, max_km], [y_start, y_end], color='red', label='Regression Line')

plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.legend()
plt.show()