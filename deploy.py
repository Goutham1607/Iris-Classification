from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the model and scaler
model = joblib.load('saved_model1.pkl')
scaler = joblib.load('scaler.save')

app = Flask(__name__)

# Configure the folder for static images
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def home():
    if request.method == 'POST':
        try:
            # Get form data
            sl = float(request.form['SepalLength'])
            sw = float(request.form['SepalWidth'])
            pl = float(request.form['PetalLength'])
            pw = float(request.form['PetalWidth'])
            
            # Prepare data for prediction
            data = np.array([[sl, sw, pl, pw]])
            x = scaler.transform(data)
            prediction = model.predict(x)
            
            # Generate image path
            image_filename = f"{prediction[0]}.png"
            image = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            
            return render_template('index.html', prediction=prediction[0], image=image)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
