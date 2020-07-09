#importing libraries for image data handling
import base64
import numpy as np
import io
from PIL import Image

#to ignore all the tensorflow related GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#importing libraries for handling the ML Model
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

#importing standard libraries for serving API
from flask import request
from flask import jsonify
from flask import Flask

#to prevent any CORS policy error
from flask_cors import CORS

app = Flask(__name__)

#to prevent any CORS policy error
CORS(app)

#function to load the model into memory
def get_model():
	global model
	model = load_model('model.h5')
	print(" * Model Loaded!")

#function to convert the image to a suitable format 
#so that it can be plugged into the model
def preprocess_image(image, target_size):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis = 0)

	return image
			
print(" * Loading Keras Model...")
graph = tf.compat.v1.get_default_graph()
get_model()

#Our API endpoint that will service the frontend requests
@app.route("/predict", methods=["POST"])	
def predict():

	#getting the encoded image from frontend 
	message = request.get_json(force=True)
	encoded = message['image']

	#decoding the image and preprocessing it
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image, target_size=(64, 64))
	test_datagen = ImageDataGenerator()

	#preparing the input format for our Model
	test_generator = test_datagen.flow(processed_image, [1], batch_size = 1)
	
	#predicting the presence of malaria
	prediction = model.evaluate_generator(test_generator, steps = len(test_generator))

	response = {
		'prediction': prediction[1]
	}

	#sending the prediction results back to frontend
	return jsonify(response) 
