import numpy as np
import random
from collections import deque

class ExpReplay:

    def __init__(self, Size):
        self.Size = Size
        self.Step = 0
        self.Que = deque()

    def GetSize(self):
        return self.Step

    def addExpierience(self, Expierience):

        if self.Step < self.Size
            self.Que.append(Expierience)
            self.Step += 1
        else:
            self.Que.popleft()
            self.Que.append(Expierience)

    def ResetExpReplay(self):
        self.Que.clear()
        self.Step = 0

    def GetSampleExpierences(self, SampleSize):
        
        ExpSamples = []

        if self.Step < SampleSize:
            ExpSamples = random.sample(self.Que, self.Step)
        else:
            ExpSamples = random.samples(self.Que, SampleSize)

        oldS_Sample, a_Sample, r_Sample, d_Sample ,newS_Sample = list(map(np.array, list(zip(*ExpSamples))))