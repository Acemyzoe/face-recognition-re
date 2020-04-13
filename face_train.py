#from __future__ import absolute_import, division, print_function, unicode_literals
import random
import numpy as np
from sklearn.model_selection import  train_test_split
from face_data import load_dataset, resize_image, IMAGE_SIZE
import tensorflow as tf
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

 #建立CNN模型
class CNN():
    #模型初始化
    def __init__(self):
        self.model = None
        self.pre_model = tf.keras.models.load_model('./model/face_model0.h5')
    
    def build_model(self):
        self.model = tf.keras.models.Sequential() #将图像格式从二维数组转换为一维数组。可以将这一层看作是堆叠图像中的像素行并将它们排成一行。该层没有学习参数。它只会重新格式化数据。
        self.model.add(tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1)))
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))  #是Google提出的一种正则化技术，作用是在神经网络中丢弃部分神经元，用以对抗过拟合。
        self.model.add(tf.keras.layers.Dense(2, activation='softmax')) #像素展平后，网络由tf.keras.layers.Dense两层序列组成。这些是紧密连接或完全连接的神经层。第一Dense层有512个节点（或神经元）。第二层（也是最后一层）返回长度为2的logits数组。每个节点包含一个得分，该得分指示当前图像属于2个类之一。
        self.model.summary() #预览网络结构

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
        self.model.save('./model/face_model0.h5')
        self.model.save('./model/face_model0',save_format = 'tf')

    #识别人脸
    def face_predict(self,image):    
        #载入picture
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(64,64))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        image /= 255
        #给出输入属于各个类别的概率  
        result = self.pre_model.predict(image)
        #print('result:', result[0][0])
        #返回类别预测结果
        return result[0]


if __name__ == '__main__': 
    model = CNN()#模型初始化
    model.build_model()

    #model.train_model('./face_data')
    #model.save_model()

    image = cv2.imread('./1.jpg')
    model.face_predict(image)
