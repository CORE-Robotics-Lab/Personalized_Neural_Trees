import random
from scheduling.create_scheduling_data.constants import *

class Task:
    def __init__(self, c=None, loc=None, name = ""):
        if c == None:
            self.c = random.randint(1,10)  # duration
        else:
            self.c = c
        if loc == None:
            self.loc = (random.randint(0, grid_size_x-1), random.randint(0, grid_size_y-1))
        else:
            self.loc = loc
        self.isTaskFinished = False
        self.isTaskScheduled = False
        self.name = name

    def getc(self):
        return self.c
    def getloc(self):
        return self.loc
    def getisTaskFinished(self):
        return self.isTaskFinished
    def getisTaskScheduled(self):
        return self.isTaskScheduled
    def changeTaskCompletionStatus(self):
        self.isTaskFinished = True
    def changeTaskScheduleStatus(self):
        if self == None:
            return
        self.isTaskScheduled = True
    def getName(self):
        return self.name

    # TODO: add method to print all traits
