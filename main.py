# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:07:55 2020

@author: My Laptop
"""

import numpy as np
import cv2
from math import sqrt, pow

def getDistance(point1, point2):
    diffX = point1[0] - point2[0]
    diffY = point1[1] - point2[1]
    
    return sqrt(pow(diffX,2) + pow(diffY,2))
    
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

vid = cv2.VideoCapture(0) 

while(True):
    ret, frame = vid.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        cx_list = []
        cy_list = []
        
        for (ex,ey,ew,eh) in eyes:
            cx_list.append(int(ex+ew/2))
            cy_list.append(int(ey+eh/2))
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        if (len(cx_list) == 2):
            point1 = (cx_list[0], cy_list[0])
            point2 = (cx_list[1], cy_list[1])
            cv2.line(roi_color, point1, point2, (0,0,255),2)
            
            pixelDistance = getDistance(point1, point2)
            
            #the estimated distance is obtained from measurement and interpolation
            estimatedDistance = -0.7249*pixelDistance + 103.85
            
            print(str(estimatedDistance) + " cm")
            
    cv2.imshow('img',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break       
    
vid.release()
cv2.destroyAllWindows()