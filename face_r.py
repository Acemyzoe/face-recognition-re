
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
from face_train import CNN
import cv2
import sys
from PIL import Image, ImageDraw, ImageFont
import math
 
if __name__ == '__main__':
        
    #加载模型
    model = CNN()
    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)   
    #人脸识别分类器本地存储路径
    cascade_path ="./haarcascades/haarcascade_frontalface_alt2.xml"    
    
    #循环检测识别人脸
    while True:
        ret, frame = cap.read()   #读取一帧视频       
        if ret is True:           
            #图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                
        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect               
                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                face_probe = model.face_predict(image)   #获得预测值
                #print("GJ:{:0.2%} unknown:{:0.2%}".format(face_probe[0],face_probe[1]))
               
                if face_probe[0] >=0.98:                                                                           
                    #文字提示是谁
                    name = 'GJ'
                else:
                    name = 'unknown'
                cv2.putText(frame,name, 
                            (x - 5, y + h + 5),                      #坐标
                            cv2.FONT_HERSHEY_SIMPLEX,              #字体
                            1,                                     #字号
                            (0,0,255),                             #颜色
                            2)                                     #字的线宽
                # cv2和PIL中颜色的RGB码的储存顺序不同
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10),(205,0,0), thickness = 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                pilimg = Image.fromarray(frame)
                draw = ImageDraw.Draw(pilimg) # 图片上打印 出所有人的预测值
                font = ImageFont.load_default()
                draw.text((x+10,y-30), 'GJ:{:.2%}'.format(face_probe[0]), (255, 250, 250), font=font)
                draw.text((x+10,y-20), 'unknown:{:.2%}'.format(face_probe[1]), (255, 250, 250), font=font)
                frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

        cv2.imshow("Show", frame)       
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break
 
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
 
 
