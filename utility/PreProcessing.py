import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from utility.hyperparameters import hyperparameters

class PreProcessing():

    def __init__(self):
        self.FrameBuffer = None

    def StackState(self, newObs, obs):
        #StackedFrames stores the last 4 frames
        #StackedState is a stacked image of the last 4 frames

        img = self.PreprocessImage(newObs, obs)
        img = np.reshape([img], (1, 84, 84, 1))
        newFrameBuffer = np.append(img, self.FrameBuffer[:, :, :, :3], axis=3)
    
        return newFrameBuffer

    def GetInitialStackedState(self, newObs, obs):
        #StackedFrames stores the last 4 frames
        #StackedState is a stacked image of the last 4 frames
        img = self.PreprocessImage(newObs, obs)
        self.FrameBuffer = np.stack((img, img, img, img), axis=2)
        self.FrameBuffer = np.reshape([self.FrameBuffer], (1, 84, 84, 4))
        return self.FrameBuffer    


    def PreprocessImage(self, newObs, obs):
        #cut off the edges and gamesscore
        obs = obs[25:201:]
        newObs = newObs[25:201:]

        #changed the size of the image from 210x160x3 to 84x84
        img = np.maximum(newObs, obs)
        img = np.uint8(resize(rgb2gray(img), (84, 84), mode='constant') * 255)
        return img






    
