import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the data from CSV
data = pd.read_csv("boston_house_data.csv")
print(data.head())
# Separate features and target variable
X = data.drop("medv", axis=1)  # Assuming "MEDV" is the target variable
y = data["medv"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train Score:", train_score)
print("Test Score:", test_score)

# Save the trained model as a pickled object
joblib.dump(model, "model/boston_model.pkl")
