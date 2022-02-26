# pgcp_project1_group4_image_captioning
image captioning project implementation from Group 4 for PG Certification Program from IIIT Hyderabad.

Problem Statement:

          Objective is to build an image captioning model using CNN to generate captions which describe the image

Dataset:
        Flicker 8k dataset has been used for this project. 
        
        Please refer to Jason Brownlee's GITHUB link to Download Flickr_8k dataset

        Flickr8k_Dataset.zip  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

        Flickr8k_text.zip   https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
        
        Flick8k/
        Flick8k_Dataset/ :- contains the 8000 images
        Flick8k_Text/
        Flickr8k.token.txt:- contains the image id along with the 5 captions
        Flickr8k.trainImages.txt:- contains the training image id’s
        Flickr8k.testImages.txt:- contains the test image id’s
 
 Model:
 
        Word Vector : Glove (glove.6B.200d.txt)
        Image feature extraction : Transfer learning using InceptionV3
        model : LSTM
        
Evaluation techniques:

        greedy search & beam search with different K values
        
Test results:

        ![kid1](https://user-images.githubusercontent.com/42133005/155842929-7312878c-a5b3-4219-820d-d4df4930670f.PNG)

      
Description of files:

        v7_Keras Model.ipynb : This is the colab file contains the modle training and testing
        
        v5_image_caption_attention.ipynb : This is the experiment done out side this project. this file can be ignored
        
         /templates/index.html : this is the default html file gets loaded when you deploy the modle. This file continas option to load an image from your system and submit the          image for processing
        
         /templates/predict.html : this is the html file gets loaded with all predicted captions when you submit the image
         
         requirements.txt : this file contains all the list of python dependent packages 
         
         runtime.txt : python run time
         
         app.py : this is the main file where we load our trained model weights and run our application.
         
Steps to run/deploy the model:
  
          1. run this colab notebook v7_Keras Model.ipynb , this will take some time to run all ephocs and train model
          
          2. once this file runs , it will save all the weights into google drive
          
          3. Now we have all the model weights & word vector npy files
          
          4. install all the dependent packages mentioned in requirments.txt into your Server (EC2/Heroku/you local machine/Azure instance)
          
          5.run the app.py and test the model
         
         

        
