import tensorflow as tf
import base64
import numpy as np
import io
from PIL import Image
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

from flask import request
from flask import jsonify
from flask import Flask

session = tf.compat.v1.keras.backend.get_session()
init = tf.compat.v1.global_variables_initializer()


app = Flask(__name__)

def get_model():
	global model
	model = load_model('cnn_model.h5')
	print("* Model Loaded!")

def preprocess_image(image, target_size):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis = 0)

	return image

def pred(num):
	if num == 1:
		return "INFECTED"
	return "NOT INFECTED"
		
tf.compat.v1.disable_eager_execution()

print("* Loading Keras Model...")
graph = tf.compat.v1.get_default_graph()
get_model()

@app.route("/predict", methods=["POST"])	
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	print("DECODED")
	print(type(decoded))
	byteimg = io.BytesIO(decoded);
	print("BYTES")
	print(byteimg)
	image = Image.open(byteimg)
	processed_image = preprocess_image(image, target_size=(64, 64))

	global graph
	global session
	with graph.as_default():
		session.run(init)
		prediction = model.predict(processed_image)

		response = {
			'prediction': pred(prediction[0][0])
		}

		return jsonify(response) 