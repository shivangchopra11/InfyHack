from keras.preprocessing import image
import numpy as np
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.models import Sequential,load_model
import time
import imutils


clas=load_model('bank_1.h5') 

def resize(frame1):
    img=cv2.resize(frame1,(64,64))       
    img = image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=clas.predict(img)
    return pred
obj={0: "knife" ,
     1: "Guns",
     2: "Nothing"}
def windows(image):
    
    (winW, winH) = (512, 512)
    def pyramid(image, scale=1.5, minSize=(30, 30)):
        yield image
        while True:
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
              	break
            yield image
    
    
    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    for resized in pyramid(image, scale=1.5):
        sums_tot=0
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
                sums=0
                if window.shape[0] != winH or window.shape[1] != winW: continue
                predict=resize(window)
                clone = resized.copy()  
                index=predict.argmax()
            
                if index==0:
                    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (253, 2, 0), 2)
                    x=x+10
                    y=y+10
                    sums=sums+predict/10
                    cv2.putText(clone, "knife", (x, y+winH), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 200), 2)
                if index==1:
                    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (253, 2, 0), 2)
                    x=x+10
                    y=y+10
                    sums=sums+predict/10
                    cv2.putText(clone, "guns", (x, y+winH), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 200), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.1)  
        sums_tot=sums_tot+sums + sums


video=cv2.VideoCapture(0)
while True:
    _,frame=video.read()
    cv2.waitKey(100)
    windows(frame)
    if cv2.waitKey(1) and 0xFF ==ord('q'):
        break
video.release()
cv2.destroyAllWindows()