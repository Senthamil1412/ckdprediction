# ckdprediction

!pip install flask-ngrok
from flask import Flask, request, render_template_string
import pickle
import numpy as np

# Load the logistic regression model
model_path = "/content/logistic_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Kidney Disease Prediction</title>
</head>
<body>
    <h1>Kidney Disease Prediction</h1>
    <form method="POST">
        <label for="features">Enter Features (comma-separated):</label><br>
        <input type="text" id="features" name="features" placeholder="e.g., 48,80,1.020,1,..."><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction is not none %}
    <h2>Prediction: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_features = request.form.get("features")
        try:
            features = np.array([float(x) for x in input_features.split(",")]).reshape(1, -1)
            prediction = model.predict(features)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template_string(html_template, prediction=prediction)

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000)
