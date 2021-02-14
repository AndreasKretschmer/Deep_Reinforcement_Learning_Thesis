import numpy as np
import random
from collections import deque

class ExpReplay:

    def __init__(self, Size):
        self.Size = Size
        self.Step = 0
        self.Que = deque()

    def GetLen(self):
        return self.Step

    def addExperience(self, experience):

        if self.Step < self.Size:
            self.Que.append(experience)
            self.Step += 1
        else:
            self.Que.popleft()
            self.Que.append(experience)

    def ResetExpReplay(self):
        self.Que.clear()
        self.Step = 0

    def GetSampleExpierences(self, SampleSize):
        ExpSamples = []

        if self.Step < SampleSize:
            ExpSamples = random.sample(self.Que, self.Step)
        else:
            ExpSamples = random.sample(self.Que, SampleSize)

        oldS_Sample, a_Sample, r_Sample, d_Sample ,newS_Sample = list(map(np.array, list(zip(*ExpSamples))))

        return oldS_Sample, a_Sample, r_Sample, d_Sample ,newS_Sample