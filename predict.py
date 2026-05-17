import os

def load_params(filename="params.txt"):
    params = {}

    if not os.path.exists(filename):
        print("Warning: params.txt file is missing.")
        return params

    try:
        with open(filename, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=")
                    params[key] = float(value)
    except Exception as e:
        print(f"Error loading file: {e}. Using defaults.")
    
    return params

if __name__ == "__main__":
    params = load_params()
    if(len(params) == 0):
        print("Predicted price: 0")
        exit()
    theta0 = params["theta0"]
    theta1 = params["theta1"]
    min_mileage = params["min_mileage"]
    max_mileage = params["max_mileage"]
    min_price = params["min_price"]
    max_price = params["max_price"]

    try:
        user_input = input("Enter mileage: ")
        km = float(user_input)
    except ValueError:
        print("Please enter a valid number for mileage.")
        exit()


    if max_mileage == min_mileage:
        predicted_price = theta0 + (theta1 * km)
    else:
        # 1. Normalize KM
        normalized_km = (km - min_mileage) / (max_mileage - min_mileage)
        
        # 2. Apply Hypothesis (Math from PDF page 9)
        normalized_price = theta0 + (theta1 * normalized_km)
        
        # 3. De-normalize Price
        predicted_price = (normalized_price * (max_price - min_price)) + min_price

    if predicted_price < 0:
        predicted_price = 0.0

    print(f"Predicted price: {predicted_price:.2f}")