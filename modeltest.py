import numpy as np
import pandas as pd
import joblib

# Load the trained Random Forest model
rf_model = joblib.load('finalmodel.pkl')

# Function to get input features from user
def get_input_features():
    ear = float(input("Enter EAR value: "))
    mar = float(input("Enter MAR value: "))
    eyes_output = input("Enter Eyes Output (Open/Closed): ").strip().capitalize()  # Convert to title case
    mouth_output = input("Enter Mouth Output (Yawn/NY): ").strip().capitalize()  # Convert to title case

    # Encode categorical features
    if eyes_output == "Closed":
        eyes_output_closed = 1
        eyes_output_open = 0
    elif eyes_output == "Open":
        eyes_output_closed = 0
        eyes_output_open = 1
    else:
        print("Invalid input for Eyes Output. Please enter 'Open' or 'Closed'.")
        return None

    if mouth_output == "Yawn":
        mouth_output_yawn = 1
        mouth_output_no_yawn = 0
    elif mouth_output == "Ny":
        mouth_output_yawn = 0
        mouth_output_no_yawn = 1
    else:
        print("Invalid input for Mouth Output. Please enter 'Yawn' or 'NY'.")
        return None

    return np.array([[ear, mar, eyes_output_closed, eyes_output_open, mouth_output_no_yawn, mouth_output_yawn]])

# Get input features from user
features = get_input_features()

# If input features are valid, make predictions
if features is not None:
    print("Input features:", features)

    # Predict drowsiness using the trained Random Forest model
    prediction = rf_model.predict(features)

    # Display prediction
    print("Predicted class:", prediction)
