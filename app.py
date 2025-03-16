from flask import Flask, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route that returns a default prediction with a descriptive message
@app.route('/')
def home():
    # Define a sample input with 8 numeric features and 43 one-hot encoded features (total 51 features)
    # Adjust these sample values to match realistic normalized data for your application.
    sample_input = [
        -0.4,    # normalized value for "bedrooms"
        -0.3,    # normalized value for "bathrooms"
        0.2,     # normalized value for "guests"
        0.1,     # normalized value for "openness"
        0.0,     # normalized value for "occupancy"
        0.05,    # normalized value for "nightly rate"
        -0.1,    # normalized value for "lead time"
        0.0      # normalized value for "length stay"
    ] + [0] * 43  # One-hot encoded categorical features, defaulted to 0
    
    # Ensure the input has the proper shape (1, number_of_features)
    input_features = np.array(sample_input).reshape(1, -1)
    
    # Get prediction from the model (which predicts property revenue)
    prediction = model.predict(input_features)
    
    # Return a JSON response with a descriptive message and the prediction
    return jsonify({
        'message': (
            "Prediction of property revenue based on the input values. "
            "The input includes normalized features for bedrooms, bathrooms, guests, openness, occupancy, "
            "nightly rate, lead time, and length stay, plus one-hot encoded categorical features."
        ),
        'prediction': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
