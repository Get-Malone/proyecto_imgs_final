import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
#from keras.optimizers import adam
from keras.preprocessing import image
import cv2
import datetime

# IMPLEMENTING LIVE DETECTION OF FACE MASK

#load model es para cargar el modelo que se ha entrenado
# el modelo entrenado consiste en una red neuronal con una capa de convoluciones,
# una capa de pooling, una capa de flatten, una capa de dropout, una capa de dense y una capa de softmax
# para que el modelo pueda predecir la probabilidad de que una imagen sea un cubrebocas o no
mymodel=load_model('../Automatizacion-de-Camara-Inteligente-main/mymodel.h5')

cap=cv2.VideoCapture(0)
#cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') es para detectar la cara en la imagen 
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
        _,img=cap.read()
        #detect multiscale es para detectar multiples caras en una imagen 
        face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        for(x,y,w,h) in face:
                face_img = img[y:y+h, x:x+w]
                cv2.imwrite('temp.jpg',face_img)
                #image.load image es para cargar la imagen 
                test_image=image.image_utils.load_img('temp.jpg',target_size=(150,150,3))
                #image to array es para convertir la imagen a un array 
                test_image=image.image_utils.img_to_array(test_image)
                # expand_dims es para agregar una dimension a la imagen que se esta procesando 
                test_image=np.expand_dims(test_image,axis=0)
                #pred=mymodel.predict_classes(test_image)[0][0]
                #predict es para predecir la clase de la imagen 
                pred=mymodel.predict(test_image)[0][0]
                # si la clase es 1 es que la persona no esta usando el cubrebocas
                if pred==1:
                        #dibujar un rectangulo en la cara detectada
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                        #coloca el texto en la imagen sin el cubrebocas
                        cv2.putText(img,'SIN CUBREBOCAS',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                        #de lo contrario si la clase es 0 es que la persona esta usando el cubrebocas
                else:
                        #dibujar un rectangulo en la cara detectada
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                        #coloca el texto en la imagen con el cubrebocas
                        cv2.putText(img,'CON CUBREBOCAS',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                        #muestra la hora actual
                datet=str(datetime.datetime.now())
                #put text es para colocar el texto en la imagen 
                cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        # muestra la imagen en pantalla
        cv2.imshow('img',img)

        # si se presiona la tecla q se cierra el programa
        if cv2.waitKey(1)==ord('q'):
                break

# se cierra la camara
cap.release()
# se cierra la ventana
cv2.destroyAllWindows()
