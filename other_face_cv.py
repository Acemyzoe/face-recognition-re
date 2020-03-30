
import os
import cv2

cascPath = "/home/ace/anaconda3/pkgs/libopencv-3.4.2-hb342d67_1/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
input_dir = './lfw'
output_dir = './other_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# classifiers
faceCascade = cv2.CascadeClassifier(cascPath)

index = 1
for (path,dirnames,filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('处理picture %s'%index)
            image = cv2.imread(path + '/' + filename)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
                )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
                image = image[y:y+h,x:x+w]
                image = cv2.resize(image,(64,64))
                cv2.imshow('image',image)
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg',image)
                index +=1
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break    
cv2.destroyAllWindows()
