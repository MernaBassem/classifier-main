from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2



app = Flask(__name__)



classes = ['malignant' ,'benine']
model=load_model("MLmodellast.h5")

@app.route('/')
def index():

    return render_template('index.html', appName="MLmodellast")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request

        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"


        img_data = request.files['fileup'].read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_np = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_np = cv2.resize(img_np, (70, 70))
        img_np = img_np / 255.0
        img_np = np.expand_dims(img_np, axis=0)
        print("Model predicting ...")
        result = model.predict(img_np)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(result)
        print(prediction)
        return jsonify({'prediction': prediction})



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        img_data = request.files.get('fileup')
        nparr = np.frombuffer(img_data, np.uint8)
        # Decode image from numpy array
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_np = cv2.resize(img_np, (70, 70))
        img_np = img_np / 255.0
        img_np = np.expand_dims(img_np, axis=0)
        print("predicting ...")
        result = model.predict(img_np)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="MLmodellast")
    else:
        return render_template('index.html',appName="MLmodellast.")


if __name__ == '__main__':
    app.run(debug=True)
