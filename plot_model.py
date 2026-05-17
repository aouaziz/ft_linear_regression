import pandas as pd
import matplotlib.pyplot as plt # Fix import
from predict import load_params

data = pd.read_csv('data.csv')
params = load_params()

if not params or params["max_mileage"] == 0:
    print("Train the model first!")
    exit()

# 1. Get the parameters
theta0 = params["theta0"]
theta1 = params["theta1"]
min_km = params["min_mileage"]
max_km = params["max_mileage"]
min_p = params["min_price"]
max_p = params["max_price"]

# 2. Calculate the "Y" points for your line using your model
# For the start of the line (min mileage)
# Since min_km maps to 0 in normalized form:
y_start_norm = theta0 + (theta1 * 0) 
y_start = (y_start_norm * (max_p - min_p)) + min_p

# For the end of the line (max mileage)
# Since max_km maps to 1 in normalized form:
y_end_norm = theta0 + (theta1 * 1)
y_end = (y_end_norm * (max_p - min_p)) + min_p

# 3. Plot the original data points
plt.scatter(data['km'], data['price'], color='blue', label='Data Points')

# 4. Plot the line your model created
# X-coordinates: [min_km, max_km]
# Y-coordinates: [y_start, y_end]
plt.plot([min_km, max_km], [y_start, y_end], color='red', label='Regression Line')

plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.legend()
plt.show()