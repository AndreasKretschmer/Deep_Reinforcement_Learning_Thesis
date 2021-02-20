import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from utility.hyperparameters import hyperparameters

class Utility():

    def __init__(self):
        pass
    
    def Preprocess_Image(self, state, lastState):
        # Crop and resize the image
        img = np.maximum(state, lastState)
        img = np.uint8(resize(rgb2gray(img), hyperparameters.RESIZE_IMAGE_SIZE) * 255)
        return np.reshape(img, (1, hyperparameters.RESIZE_IMAGE_SIZE[0], hyperparameters.RESIZE_IMAGE_SIZE[1]))

    def GetInitialStateForEpisode(self, state, lastState):
        img = np.maximum(state, lastState)
        img = np.uint8(resize(rgb2gray(img), hyperparameters.RESIZE_IMAGE_SIZE) * 255)
        stackedstate = [img for _ in range(hyperparameters.STACKED_FRAME_SIZE)]
        return np.stack(stackedstate, axis=0)



    
