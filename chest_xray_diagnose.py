from flask import Flask, render_template, request, send_from_directory

import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
import skimage

from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from  datetime import datetime
import re


app = Flask(__name__)

@app.route('/')
def features():
	return render_template('upload_chest_xray_image.html')

UPLOAD_FOLDER = 'chest_xray_uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

## Variables for writing  the classification result into a file.s
filenames = []
pred_results = []
probabilities = []


@app.route('/predict' , methods =  ['GET' , 'POST'])
def upload_and_predict():
	if request.method == 'POST':
		##
		## read in the image and format it for model prediction
		f = request.files['file']
		file = f.filename
		## save the chest xray image 
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], file))

		## get the file extension so that we can read in the image as jpeg or png.
		extension = os.path.splitext(file)[1]
		if extension in {'.jpeg', '.jpg'}:
			img = skimage.io.imread(os.path.join(app.config['UPLOAD_FOLDER'], file))
		elif extension == '.png':
			## 
			img = skimage.io.imread(os.path.join(app.config['UPLOAD_FOLDER'], file), plugin = 'matplotlib')

		img_resized = resize(img, output_shape=(150, 150), anti_aliasing=True, mode='reflect')
		## 
		X_test = img_resized.reshape(-1,150*150)
      		##
		## probabilty predictions
		probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

		predictions = probability_model.predict(X_test)
		## based on the max value of probability , predict the class.
		if len(predictions) > 0:

			test_predicted_labels = []
			for i in range(len(predictions)):
				pred_label = np.argmax(predictions[i])
				test_predicted_labels.append(pred_label)
			
			## set the value for result
			if test_predicted_labels ==[0]:
				result = 'Pneumonia.'
				prob = np.round(predictions[0][0],2)
				result_1 = result + " " + 'Probability : ' +  str(prob)
				
			elif test_predicted_labels == [1]:
				result = 'Normal.'
				prob = np.round(predictions[0][1],2)
				result_1 = result + " " + 'Probability : ' + str(prob)
			else:
				return render_template("chest_xray_result.html", result = 'Prediction Failure')
      			##
			
			## assign the value to the variables.
			filenames.append(file)
			pred_results.append(result)
			probabilities.append(prob)
		
			##
			return render_template("chest_xray_result.html" , result = result_1 )   ## Display the result
		else:
			return render_template("chest_xray_result.html", result = 'Prediction Failure') #Error in model prediction
	return	


@app.route('/download_result' , methods = ['GET', 'POST'])
def download_file():
	try:
		## Save the results into csv file in the 'Results' folder.
		result_df = pd.DataFrame({'file_name' : filenames , 'predicted_result' : pred_results , 'probability' : probabilities })
      		## 
		now = datetime.now()
		dt = now.strftime("%d-%m-%Y ")
		time = now.strftime("%H-%M-%S")
		##
		## create unique file name based on date and time.
		result_filename = 'result' + "_" + dt.strip() + "_" + time +  ".csv"
		

		## set the download folder 
		file_path = os.getcwd()+'/Results/'+ result_filename
		## Save the file as csv.
		result_df.to_csv(file_path, index = False)
		
		## download the file
		return send_from_directory(os.getcwd()+'/Results/', result_filename , as_attachment = True, cache_timeout = 0 )
		
		return
	except Exception as e:
		return str(e)


if __name__ == '__main__':
	model_name = 'chest_xray_classify.h5'
	model = tf.keras.models.load_model(model_name)
	print("model loaded")
	app.run(debug = True,host = '0.0.0.0' , port = 5000 )
