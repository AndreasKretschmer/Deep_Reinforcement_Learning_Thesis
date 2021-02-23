import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from utility.hyperparameters import hyperparameters
from collections import deque

class Utility():

    def __init__(self):
        pass

    def StackFrames(self, StackedFrames, obs):
        img = self.PreprocessImage2(obs)
        img = np.maximum(StackedFrames[-1], img)
        StackedFrames.append(img)

        StackedState = np.stack(StackedFrames, axis=2)

        return StackedState, StackedFrames

    def GetInitialStackedState(self, StackedFrames, obs):
        StackedFrames = deque([np.zeros((hyperparameters.RESIZE_IMAGE_SIZE[0], hyperparameters.RESIZE_IMAGE_SIZE[1]), dtype=np.uint8) for i in range(hyperparameters.STACKED_FRAME_SIZE)], maxlen=4)
        
        img = self.PreprocessImage2(obs)
        img = np.maximum(img, img)
        StackedFrames.append(img)
        StackedFrames.append(img)
        StackedFrames.append(img)
        StackedFrames.append(img)    

        StackedState = np.stack(StackedFrames, axis=2)

        return StackedState, StackedFrames

    def PreprocessImage2(self, obs):
        img = np.uint8(resize(rgb2gray(obs), hyperparameters.RESIZE_IMAGE_SIZE) * 255)
        return img



    
