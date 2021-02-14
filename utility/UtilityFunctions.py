import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from utility.hyperparameters import hyperparameters

class Utility():

    def __init__(self):
        pass
    
    def Preprocess_Image(self, state):
        # Crop and resize the image
        img = np.uint8(resize(rgb2gray(state), hyperparameters.RESIZE_IMAGE_SIZE, mode='constant') * 255)

        return img


    
