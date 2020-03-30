#from __future__ import absolute_import, division, print_function, unicode_literals
import random
import numpy as np
from sklearn.model_selection import  train_test_split
from face_data import load_dataset, resize_image, IMAGE_SIZE
import tensorflow as tf
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
import cv2

 
 
class Dataset:
    def __init__(self, path_name):
        #训练集
        self.train_images = None
        self.train_labels = None       
        #测试集
        self.test_images  = None            
        self.test_labels  = None       
        #数据集加载路径
        self.path_name    = path_name        
        #当前库采用的维度顺序
        self.input_shape = None
        self.nb_classes=None     
    #加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, 
             img_channels = 1, nb_classes = 2): #灰度图 所以通道数为1 ; 2个类别 所以分组数为2
        #加载数据集到内存
        images, labels = load_dataset(self.path_name)        
        
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))   #将总数据按0.3比重随机分配给训练集和测试集    
        
        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        #TensorFlow需要通道数，我们上一步设置为灰度图，所以这里为1，否则彩色图为3
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        self.input_shape = (img_rows, img_cols, img_channels)            
        
        #输出训练集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')                       
    
        #像素数据浮点化以便归一化
        train_images = train_images.astype('float32')            
        test_images = test_images.astype('float32')
        
        #将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        test_images /= 255
 
        self.train_images = train_images
        self.test_images  = test_images
        self.train_labels = train_labels
        self.test_labels  = test_labels
        self.nb_classes   = nb_classes
  
 
 #建立CNN模型
class CNN():
    #模型初始化
    def __init__(self):
        self.model = None
        self.pre_model = tf.keras.models.load_model('./model/face_model.h5')
    
    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1)))
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.model.summary()

    def train_model(self,path_name):
        images, labels = load_dataset(path_name)         
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))   #将总数据按0.3比重随机分配给训练集和测试集           
        train_images = train_images.reshape(train_images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
        #TensorFlow需要通道数，我们上一步设置为灰度图，所以这里为1，否则彩色图为3
        test_images = test_images.reshape(test_images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)               
        #输出训练集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')                           
        #像素数据浮点化以便归一化
        train_images = train_images.astype('float32')            
        test_images = test_images.astype('float32')       
        #将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        test_images /= 255
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(train_images, train_labels, batch_size=32,epochs=5)
        score = self.model.evaluate(test_images,test_labels, verbose=2)
        print('loss:',score[0])
        print('accuracy:',score[1])

    def save_model(self):
        self.model.save('.model/face_model.h5')
        self.model.save('./model/face_model',save_format = 'tf')

    #识别人脸
    def face_predict(self,image):    
        #载入picture
        image = resize_image(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))                    
        
        #浮点并归一化
        image = image.astype('float32')
        image /= 255
        #给出输入属于各个类别的概率  
        result = self.pre_model.predict(image)
        #print('result:', result[0][0])
        #返回类别预测结果
        return result[0]


if __name__ == '__main__': 
    model = CNN()#模型初始化
    model.build_model()

   # model.train_model('./face_data')
    #model.save_model()

    image = cv2.imread('./1.jpg')
    model.face_predict(image)
