from flask import Flask, render_template, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load the pickled model
model = joblib.load("model/boston_model.pkl")  # Replace "boston_house_price_prediction_model.pkl" with your model file

# Define route for rendering the form
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        features = [float(request.form.get(field)) for field in request.form]
        print("Form data:", features)
        try:
            prediction = model.predict([features])[0]
            print("Prediction:", prediction)
        except Exception as e:
            print("Error occurred during prediction:", e)
    return render_template("index.html", prediction=prediction)

# # Define route for handling form submission
# @app.route("/predict", methods=["POST"])
# def predict():
#     # Extract form data
#     features = [float(request.form.get(field)) for field in request.form]

#     # Make prediction
#     prediction = model.predict([features])[0]  # Assuming your model accepts a list of features
#     # Render the form template with the prediction result
#     return render_template("index.html", prediction=prediction)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
