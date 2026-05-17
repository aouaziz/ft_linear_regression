import pandas as pd
import sys

def normalize(x, min_val, max_val):
    if max_val == min_val:
        return 0.0  
    return (x - min_val) / (max_val - min_val)

def data_prep():
    try:
        data = pd.read_csv("data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("The file 'data.csv' was not found.")

    if len(data) == 0:
        raise ValueError("Error: check data file. It is empty.")

    try:
        data['km'] = pd.to_numeric(data['km'], errors='raise')
        data['price'] = pd.to_numeric(data['price'], errors='raise')
    except ValueError as e:
        # This will catch strings like 'aafda' and route them to your main error handler
        raise ValueError(f"Data corruption detected: Non-numeric values found in dataset. Original error: {e}")
    
    data = data.dropna()

    if len(data) == 0:
        raise ValueError("Error: No valid numeric data left after dropping NaNs.")

    min_mileage = data['km'].min()
    max_mileage = data['km'].max()
    min_price = data['price'].min()
    max_price = data['price'].max()

    if max_mileage == min_mileage:
        raise ValueError("Invalid data: all mileage values are identical.")

    if max_price == min_price:
        raise ValueError("Invalid data: all price values are identical.")

    # Apply normalization
    data['km'] = data['km'].apply(lambda x: normalize(x, min_mileage, max_mileage))
    data['price'] = data['price'].apply(lambda x: normalize(x, min_price, max_price))

    return data, min_mileage, max_mileage, min_price, max_price


def trainingLoop(data):
    theta0 = 0.0
    theta1 = 0.0
    L = 0.1
    m = len(data)
    iterations = 1000

    # OPTIMIZATION: Vectorized operations run 100x faster than data.iloc loops
    km_series = data['km']
    price_series = data['price']

    for i in range(iterations): 
        # Calculate all predictions at once
        predictions = theta0 + (theta1 * km_series)
        errors = predictions - price_series
        
        # Calculate gradients using vector math sums
        gradient0 = errors.sum() / m
        gradient1 = (errors * km_series).sum() / m
        
        # Update parameters simultaneously
        theta0 -= L * gradient0
        theta1 -= L * gradient1

    return theta0, theta1 


if __name__ == "__main__":
    try:
        data, min_mileage, max_mileage, min_price, max_price = data_prep()
        theta0, theta1 = trainingLoop(data)

        with open("params.txt", "w") as f:
            f.write(f"theta0={theta0}\n")
            f.write(f"theta1={theta1}\n")
            f.write(f"min_mileage={min_mileage}\n")
            f.write(f"max_mileage={max_mileage}\n")
            f.write(f"min_price={min_price}\n")
            f.write(f"max_price={max_price}\n")
        print(f"data  {data}")
        print("Training complete. Parameters saved successfully.")

    except Exception as e:                 
        print(f"Caught an error: {e}")
