
# coding: utf-8

# In[1]:


from keras import backend as K
from keras.models import Model
from keras import optimizers
from keras.models import load_model
import numpy as np
import cv2 as cv


# In[2]:


# external loss function for training Unet
def dice_coef(y_true, y_pred, smooth=0.9):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class Unet:
    model = None
    inpWidth = 640
    inpHeight = 480
    model_name = None
    mask_color = (64,0,128)
    mask_contours = None
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = load_model(model_name, custom_objects={'dice_coef_loss': dice_coef_loss})
    
    # get predicted mask from Unet
    def getMask(self, frame):       
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = np.array([frame/255.])
        frame = frame[0,:,:,:]
        
        # resize the input images to match Unet model
        frame = cv.resize(frame,(640,480))
        Test_image = np.ndarray((1 ,480, 640, 3), dtype=np.float)
        Test_image[0] = frame
        
        # predict mask using keras, here only one frame could be processed
        results = self.model.predict(Test_image, batch_size=1, verbose=1, steps=None)
        results_iamge = []
        for i in range(len(results)):
            a = results[i,:,:,0]*255.
            results_iamge.append(a.astype(np.uint8)) 
        img = results_iamge[0]
        ret,mask = cv.threshold(img,0,255,cv.THRESH_BINARY)
        
        # find contours in binary mask images
        _, contours, _= cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.mask_contours = contours
    
    def drawContours(self, frame):
        frame = cv.resize(frame,(640, 480))
        
        # draw mask in frames with mask_color
        frame = cv.drawContours(frame,self.mask_contours,-1,self.mask_color,-1)
        return frame

