
# 基本图像分类：人脸识别  
    本指南训练了一个神经网络来进行人脸的识别，即对本人和其他人进行分类,下面是完整的tensorflow程序的快速概括，使用了opencv进行人脸检测，使用tf.keras(高级API)在tensorflow中构建和训练网络。  
## 数据集准备和处理 
针对不同格式的数据集（例如CSV、Numpy、Text、Images)，tensorflow有多种不同的处理方式。这里我们自行采集本人的图像进行处理，构建自己的数据集并进行图像处理和标注.其他人脸数据集采用的是lfw数据集。  
> #使用opencv分类器获取人脸  
    faceCascade = cv2.CascadeClassifier(cascPath) #路径为./haarcascades  

>#Detect faces in the image  
         faces = faceCascade.detectMultiScale(grayscaleFactor=1.3,minNeighbors=5,minSize=(30, 30) )  

> #Draw a rectangle around the faces  
     for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)   
        image = image[y:y+h,x:x+w]  
        image = cv2.resize(image,(64,64))  
        cv2.imshow('image',image)  
        cv2.imwrite(output_dir+'/'+str(index)+'.jpg',image)

图像送到网络前，将图像格式化为经过适当预处理的浮点张量:   
1.  从磁盘读取图像。  
2. 解码这些图像的内容，并根据其RGB内容将其转换为正确的网格格式。
3. 将它们转换为浮点张量。
4. 归一化，将张量从0到255之间的值重新缩放为0到1之间的值。
> #标注数据，将图片转成张量。构建的函数如下，详细代码见face_data.py  
>+ resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE) #按照指定图像大小调整尺寸  
>+ read_path(path_name) #读取训练数据  
>+ load_dataset(path_name) #标注数据，并从指定路径读取训练数据  

>#数据预处理，详细代码见model类中的train_model函数   
>+ train_images = train_images.reshape(train_images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)  #TensorFlow需要通道数，我们上一步设置为灰度图，所以这里为1，否则彩色图为3  
>+ train_images = train_images.astype('float32') #像素数据浮点化以便归一化  
>+ train_images /= 255   #将其归一化,图像的各像素值归一化到0~1区间    

之后的项目会用keras提供的ImageDataGenerator类来快速完成。它可以从磁盘读取图像并将其预处理为适当的张量。它还将设置将这些图像转换成张量的生成器，这对于训练网络很有帮助。   

## 模型构建、训练、保存
建立模型需要配置模型的各层，然后编译模型。这里我将模型的构建、训练、保存、预测封装成类。之后的项目会用到迁移学习，借助一些有名的CNN模型如VGG、ResNet等训练一些大型的公开数据集。
> #深度学习的大部分内容是将简单的层链接在一起。大多数层（例如tf.keras.layers.Dense）具有在训练期间学习的参数。  
>+ self.model = tf.keras.models.Sequential()   
        self.model.add(tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1))) #将图像格式从二维数组转换为一维数组。可以将这一层看作是堆叠图像中的像素行并将它们排成一行。该层没有学习参数。它只会重新格式化数据。   
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))   
        self.model.add(tf.keras.layers.Dropout(0.2)) #是Google提出的一种正则化技术，作用是在神经网络中丢弃部分神经元，用以对抗过拟合。    
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))  #像素展平后，网络由tf.keras.layers.Dense两层序列组成。这些是紧密连接或完全连接的神经层。第一Dense层有512个节点（或神经元）。第二层（也是最后一层）返回长度为2的logits数组。每个节点包含一个得分，该得分指示当前图像属于2个类之一。  
        self.model.summary() #预览网络结构  

>  #模型的编译使用model.compile，需要添加一些其他设置：  
>+ Loss function*损失函数* ：衡量训练过程中模型的准确性。以在正确的方向上“引导”模型。  
>+ Optimizer*优化器* ： 基于模型看到的数据及其损失函数来更新模型的方式。  
>+ Metrics*指标* :  用于监视培训和测试步骤。  
    >>model.compile (optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])   

> #开始训练，使用model.fit，参数一般包括图像和标签,迭代次数。模型训练时会显示损失和准确性指标。    
>>  model.fit(train_images, train_labels, batch_size=32,epochs=5)  

> #评估准确性，比较模型在测试数据集上的表现:  
>>model.evaluate(test_images,test_labels, verbose=2)   

> #保存模型，可以只保存模型权重参数，也可以保存整个模型（网络结构和权重参数）。不同的保存形式，后续的模型部署方式不同。     
>> model.save('./model/face_model0.h5')   

> #作出预测，使用预训练的模型预测某些图像   
>> result = pre_model.predict(image) #给出输入属于各个类别的概率,结果是2个类别组成的数组，可以np.argmax(result[0])看哪个标签的置信度最高。  

## 模型部署：通过摄像头实时识别人脸
程序的流程如下：  
>1. 加载模型、人脸分类器、摄像头设备;   
>>  model = tf.keras.models.load_model('./model/face_model_h5.h5')  
>>  cascade_path ="./haarcascades/haarcascade_frontalface_alt2.xml"  
>>  cap = cv2.VideoCapture(0)   
>2. 通过摄像头获取图像，通过分类器识别出人脸;   
>3. 将人脸图像处理后交给模型识别;  
>4. 根据预测值设置置信区间并作出判断，打印出预测数据。  
        

