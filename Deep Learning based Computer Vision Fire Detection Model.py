#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True




train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
            
training_set = train_datagen.flow_from_directory(
                    'C:/Users/708788/OneDrive - Cognizant/Desktop/ML Datasets/Fire Detection/Fire Detection Dataset/Train',
                    target_size=(320, 240),
                    batch_size=32,
                    class_mode='binary')
            
test_set = test_datagen.flow_from_directory(
                    'C:/Users/708788/OneDrive - Cognizant/Desktop/ML Datasets/Fire Detection/Fire Detection Dataset/Test',
                    target_size=(320, 240),
                    batch_size=512,
                    class_mode='binary')

#dense_layers = [0,1,2]
#layer_sizes = [32,64,128]
#conv_layers = [1,2,3]
dense_layers = [2]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            

            classifier = Sequential()

            classifier.add(Convolution2D(32,3,3,input_shape=(320,240,3),activation='relu'))
            
            classifier.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                classifier.add(Convolution2D(32,3,3,activation='relu'))
                classifier.add(MaxPooling2D(pool_size=(2, 2)))

            classifier.add(Flatten())
            for _ in range(dense_layer):
                classifier.add(Dense(layer_size,activation='relu'))
                
                

            classifier.add(Dense(units=1,activation='sigmoid'))
            
            

            classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
            
            classifier.fit_generator(
                    training_set,
                    steps_per_epoch=62,
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=5)


# In[199]:


classifier.save('ip.h5')


# In[200]:


fire_cascade = cv2.CascadeClassifier('fire_detection_cascade_model.xml')


# In[201]:


import warnings
warnings.filterwarnings("ignore")
import cv2         # Library for openCV
import threading   # Library for threading -- which allows code to run in backend
import playsound   # Library for alarm sound
import smtplib     # Library for email sending


# In[202]:


# importing geopy library
from geopy.geocoders import Nominatim
  
# calling the Nominatim tool
loc = Nominatim(user_agent="GetLoc")
  
# entering the location name
getLoc = loc.geocode("Nighwa")
  
# printing address
print(getLoc.address)
print("Latitude = ", getLoc.latitude)
print("Longitude = ", getLoc.longitude)


# In[203]:


runOnce = False # created boolean


def play_alarm_sound_function(): # defined function to play alarm post fire detection using threading
    playsound.playsound('fire_alarm.mp3',True) # to play alarm # mp3 audio file is also provided with the code.
    print("Fire alarm end") # to print in consol

def send_mail_function(): # defined function to send mail post fire detection using threading
    
    recipientmail = "vkalyane6895@gmail.com" # recipients mail
    recipientmail = recipientmail.lower() # To lower case mail
    subject = "Fire Incident Occured"
    text = "Hi Team," + "\n" + "\n" + "Fire Incident has been reported at following place:" + "\n" + "\n" + "Nighwa, Kubeer mandal, Nirmal, Telangana, India" + "\n" + "19.185649" + "\n" + "77.8590162" + "\n" + "\n" + "Please Reach out as early as possible." + "\n" + "\n" + "Regards," +"\n" + "Venkatesh"
    
    message = 'Subject: {}\n\n{}'.format(subject, text)
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("vsk5895@gmail.com", '**********') # Senders mail ID and password
        server.sendmail('vsk5895@gmail.com', 'vkalyane6895@gmail.com', message) # recipients mail with mail message
        print("Alert mail sent sucesfully to {}".format(recipientmail)) # to print in consol to whome mail is sent
        server.close() ## To close server
        
    except Exception as e:
        print(e) # To print error if any


# In[204]:


import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

#Load the saved model
model = keras.models.load_model('ip.h5')
vid = cv2.VideoCapture(0)

#width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
#height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
#size = (width, height)
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#result = cv2.VideoWriter('C:/Users/708788/OneDrive - Cognizant/Desktop/videos/Fire Detection.mp4', fourcc,20.0, (width, height))

while(True):
    ret, frame = vid.read() # Value in ret is True # To read video frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # To convert frame into gray color
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5) # to provide frame resolution
    im = Image.fromarray(frame, 'RGB')
    im = im.resize((240,320))
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255
    probabilities = model.predict(img_array)[0]
    prediction = np.argmax(probabilities)
    label = 'FIRE: '+str(probabilities)

    ## to highlight fire with square
    if prediction == 0:
        for (x,y,w,h) in fire:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, label, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            print("Fire alarm initiated")
            threading.Thread(target=play_alarm_sound_function).start()  # To call alarm thread
            
            if runOnce == False:
                print("Mail send initiated")
                threading.Thread(target=send_mail_function).start() # To call alarm thread
                runOnce = True
                
            if runOnce == True:
                break

    cv2.imshow('frame', frame)
    key=cv2.waitKey(1)
    
    if(key==27):
        break

vid.release()
#result.release()
cv2.destroyAllWindows()

