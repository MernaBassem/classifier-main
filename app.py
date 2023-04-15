from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2



app = Flask(__name__)

cancer_cells = ['malignant','benine']
model=load_model('F:/intel-Classfier-main/MLmodellast.h5')

@app.route('/')
def index():

    return render_template('index.html', appName="MLmodellast")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request

        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        # convert string data to numpy array


        img_data = request.files['fileup'].read()

# Convert binary data to numpy array
        nparr = np.frombuffer(img_data, np.uint8)

# Decode image from numpy array
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_np = cv2.resize(img_np, (70, 70))
        img_np = img_np / 255.0
        img_np = np.expand_dims(img_np, axis=0)
# make prediction using the loaded model
        pred_probs = model.predict(img_np)[0]
        pred_class = np.argmax(pred_probs)
        pred_label = cancer_cells[pred_class]




        print("Model predicting ...")
        result = model.predict(img_np)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = cancer_cells[ind]
        print(result)
        print(prediction)
        return jsonify({'prediction': prediction})



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
    
        print("image loaded....")
        img_data = request.files['fileup'].read()

        # Convert binary data to numpy array
        nparr = np.frombuffer(img_data, np.uint8)

        # Decode image from numpy array
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_np = cv2.resize(img_np, (70, 70))
        img_np = img_np / 255.0
        img_np = np.expand_dims(img_np, axis=0)
        # make prediction using the loaded model
        pred_probs = model.predict(img_np)[0]
        pred_class = np.argmax(pred_probs)
        pred_label = cancer_cells[pred_class]
        # return predictions as JSON response
        print("predicting ...")
        result = model.predict(img_np)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = cancer_cells[ind]

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="MLmodellast")
    else:
        return render_template('index.html',appName="MLmodellast")


if __name__ == '__main__':
    app.run(debug=True)