# Pneumonia_diagnosis
  
Chest X-Ray is one of the diagnosis tool for Pneumonia. The doctor scrutinizes the X-ray to look for the location and extent of inflammation in the lungs.  
There  is an attempt to use artificial intelligence for the purpose of chest xray classification with the main goal to improve the accuracy of diagnosis.  
  
A neural net is trained using tensor flow on a training set of 5216 images and test set of 624 images.  
Data source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. 

After checking the test performance of various models along with hyper parameter tuning , the model selected is a neural network model containing two layers of 128 neurons , Adam optimizer , learning rate of 0.0001 , batch size of 32 . A test set accuracy of 0.85 is obtained .  This pre trained model is accessed by the flask application for classifying new chest xray images. 
The user can upload chest xray images in jpeg/png format.  
The NN model then predicts the chest condition  as normal or pneumonia along with it's associated probability.
The user can download the result , which can be further verified by physician. 
The user can upload images multiple times and download the results at once. 
The results are saved as a .csv file in '/Results' folder which resides with in the app folder. 

Further Improvement: 
1. Attempt to improve accuracy.   
2. Use pretrained models.   


