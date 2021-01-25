import cv2
import numpy as np
import tensorflow as tf
import os
from playsound import playsound
from pygame import mixer


l_eye = cv2.CascadeClassifier(r'C:\Users\amitn\Downloads\left_eye.xml')
r_eye = cv2.CascadeClassifier(r'C:\Users\amitn\Downloads\right_eye.xml')

score = 0
model = tf.keras.models.load_model(r'C:\Users\amitn\Downloads\ddv0.5.h5')
mixer.init()
sound = mixer.Sound(r'C:\Users\amitn\Downloads\emergency_alert.mp3')
cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    height,width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    right_eye =  r_eye.detectMultiScale(gray)
    
    for (x,y,w,h) in right_eye:

        right_eye_=frame[y:y+h,x:x+w]
        right_eye_= cv2.cvtColor(right_eye_,cv2.COLOR_BGR2GRAY)
        right_eye_= cv2.resize(right_eye_,(40,40))
        right_eye_= right_eye_/255
        right_eye_=  right_eye_.reshape(40,40,-1)
        right_eye_= np.expand_dims(right_eye_,axis=0)
        right_eye_prediction = model.predict_classes(right_eye_)
        # print(right_eye_prediction)

    left_eye = l_eye.detectMultiScale(gray)
    
    for (x,y,w,h) in left_eye:

        left_eye_=frame[y:y+h,x:x+w]
        left_eye_= cv2.cvtColor(left_eye_,cv2.COLOR_BGR2GRAY)  
        left_eye_= cv2.resize(left_eye_,(40,40))
        left_eye_= left_eye_/255
        left_eye_=left_eye_.reshape(40,40,-1)
        left_eye_= np.expand_dims(left_eye_,axis=0)
        left_eye_prediction = model.predict_classes(left_eye_)
        # print(left_eye_prediction)
    

    if (right_eye_prediction == 0 and left_eye_prediction == 0):
        score = score + 1
        cv2.putText(frame,"Closed",(10,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
        

    elif (right_eye_prediction == 1 and left_eye_prediction == 1):
        score = 0
        cv2.putText(frame,"Open",(10,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
    
    else:
        score = 0
    if(score>15):
        print(score)
        if mixer.get_busy() == False:
            sound.play()
        elif mixer.get_busy() ==True and (score<14):
            sound.stop()
   
    elif(score<14):
        print(score)
        try:
            sound.stop()
        except:
            pass 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()