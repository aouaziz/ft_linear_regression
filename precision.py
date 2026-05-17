import pandas as pd
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

def estimate_price(km):
    normalized_km = (km - min_km) / (max_km - min_km)
    normalized_price = theta0 + (theta1 * normalized_km)
    return (normalized_price * (max_p - min_p)) + min_p

actual_prices = data['price'].tolist()
mean_price = sum(actual_prices) / len(actual_prices)

ss_res = sum((actual - estimate_price(km)) ** 2
             for km, actual in zip(data['km'], actual_prices))

ss_tot = sum((actual - mean_price) ** 2
             for actual in actual_prices)

r2 = 1 - (ss_res / ss_tot)

print(f"R² Score: {r2:.4f}")
print(f"Model precision: {r2 * 100:.2f}%")