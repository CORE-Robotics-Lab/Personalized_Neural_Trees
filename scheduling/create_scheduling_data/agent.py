import random
import numpy as np
from scheduling.create_scheduling_data.constants import *


class Agent:
    def __init__(self, v = None, z = None, name = ""):
        if v == None:
            self.v = random.randint(0,10) # velocity
        else:
            self.v = v
        if z == None:
            self.z = (random.randint(0, grid_size_x-1),random.randint(0,grid_size_y-1))
            self.orig_location = self.z
        else:
            self.z = z
            self.orig_location = z
        self.isBusy = False
        self.name = name
        self.curr_finish_time = 0
        self.curr_task = None
        self.task_list = []
        self.orientation =  np.random.uniform(0, np.pi)
        self.task_event_dict = {} # task_num: [start_time, expected_finish_time]

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def getv(self):
        return self.v

    def getz(self):
        return self.z

    def getisBusy(self):
        return self.isBusy

    def changebusy(self,b):
        self.isBusy = b

    def updateAgentLocation(self, new_location):
        self.z = new_location

    def getOrientation(self):
        return self.orientation

    def getName(self):
        return self.name
    def setFinishTime(self, finish_time):
        self.curr_finish_time = finish_time
    def getFinishTime(self):
        return self.curr_finish_time
    def setCurrTask(self, task):
        self.curr_task = task
        self.task_list.append(task)
    def getCurrTask(self):
        return self.curr_task


    # TODO: add method to print all traits
