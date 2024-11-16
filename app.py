from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
from model.model import prepare_data, evaluate_model

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = load_model("model/kidney_disease_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = ""
    if request.method == 'POST':
        try:
            input_features = [float(x) for x in request.form.values()]
            if len(input_features) != 24:
                return "Incorrect number of inputs. It should be 24 features.", 400
            final_features = np.array(input_features).reshape(1, -1)
            prediction = model.predict(final_features)
            output = prediction[0][0]

            # Interpretation of the result
            if output > 0.5:
                prediction_text = 'The person is diagnosed with kidney disease.'
            else:
                prediction_text = 'The person is not diagnosed with kidney disease.'
        except Exception as e:
            return f"Error: {e}", 500
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
