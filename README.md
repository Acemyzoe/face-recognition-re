# 基于opencv、tensorflow的人脸识别2.0
  **BY**  [GJ](https://github.com/Acemyzoe/face-recognition)
## 使用指南
1. 运行catch_face.py获取自己的脸  
>根据需要修改opencv人脸分类器，分类器文件可见haarcascades文件夹 。  
默认图片将保存在myface文件夹，获取数量5000张。      

2. 可使用other_face_cv.py或者other_face_dlib.py处理其他人的图片(默认处理lfw图片集）。  
或者重复上一步采集另一个人的脸用作other_faces。
>例如处理lfw图片集(lf文件夹）
>
>处理完成的myface文件夹和other_faces文件夹一起放入face_data文件夹作为数据集。

3. face_data.py作为模块用于后续图片的预处理。

4. 运行face_train.py来训练神经网络。  
>载入的数据集即为face_data文件夹，包含myface文件夹和other_faces文件夹。  *训练过程，终端显示每一波训练时训练和测试的损失和准确率，训练结束将模型保存至model文件夹(h5格式和tf原型格式)。*   

5. 运行face_r.py可以打开摄像头开始识别我的脸了。  

## 环境配置
  * ubuntu18.04+Anaconda
  * 推荐使用Anaconda（一个提供包管理和环境管理的python版本）。  [官网下载](https://www.anaconda.com/distribution/)
  * 推荐修改镜像地址：

  >pip install pip -U  
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

* 安装需要的python库：(缺少相应的库可用conda或者pip自行安装)
>conda install scikit-learn  
pip install opencv-python    
conda install -c conda-forge tensorflow  
pip install --ignore-installed --upgrade   https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/cpu/tensorflow-1.7.0-cp36-cp36m-linux_x86_64.whl
