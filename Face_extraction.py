#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Acemyzoe'

import os
import sys
import cv2
import numpy as np

def resize(pic_name): 
    image = cv2.imread(pic_name)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #轮询所有分类器直至获取到人脸
    cascPaths = ["./model/haarcascade_frontalface_alt2.xml","./model/haarcascade_frontalface_alt.xml","./model/haarcascade_frontalface_default.xml"]   
    for cascPath in cascPaths:
        faceCascade = cv2.CascadeClassifier(cascPath)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
            )
        if len(faces):
            print("got face")
            break
        print("can't get faces")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = image[y:y+h,x:x+w]
        image = cv2.resize(image,(64,64))
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',image)
        image_path = pic_name+'_resize.jpg'
        pic = cv2.imwrite(image_path,image)
    return image_path

if __name__=='__main__':
    path = resize('./data/cg1.jpg')
    print(path)
    


