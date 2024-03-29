from flask import Flask, render_template, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pickled model
model = joblib.load("model/your_model.pkl")  # Replace "your_model.pkl" with your model file

# Define route for rendering the form
@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")

# Define route for handling form submission
@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    features = [float(request.form.get(field)) for field in request.form]

    # Make prediction
    prediction = model.predict([features])[0]  # Assuming your model accepts a list of features

    # Return prediction result
    return jsonify({"prediction": prediction})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

    