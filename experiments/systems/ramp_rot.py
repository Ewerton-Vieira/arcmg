import numpy as np 
from arcmg.system import BaseSystem


class Ramp_rot(BaseSystem):
    def __init__(self, slope, **kwargs):
        super().__init__(**kwargs)
        self.name = "ramp_rot"
        self.slope = slope
        self.dim = 2
        self.state_bounds = np.array([[-1, 1]]*self.dim)
        self.state_bounds = np.array([[-2, 2]]*self.dim)

        # theta = np.radians(90)
        # c, s = np.cos(theta), np.sin(theta)
        # self.R = np.array(((c, -s), (s, c)))
        
    def R(self, theta=np.radians(90)):
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))
    
    def f(self,s):
        # s=s.tolist()[0]
        X = np.array([min(max(-1, self.slope * s[0]), 1)] + [s[i]/4 for i in range(1, self.dim)])
        # X = [np.arctan(2*s[0])] + [s[i]/2 for i in range(1, len(s))]
        return self.transform(X)

    # def f(self,s):
    #     return [np.arctan(2*s[0])] + [s[i]/2 for i in range(1, len(s))]

    def transform(self,x):
        "clockwise rotation"
        return np.matmul(x, self.R(np.pi*np.linalg.norm(x)))
        # x=np.array(x)
        # for i in range(len(x)):
        #     x[i,:] = np.matmul(x[i,:], self.R(np.linalg.norm(x[i,:])))
        # return x


        # return np.matmul(x, self.R)

    def attractors(self):
        return [[-1]+[0]*(self.dim - 1), [1]+[0]*(self.dim - 1)]

    def which_attracting_region(self, s):
        """defaut -1: attracting region not specified"""
        if s[0] > 1/self.slope and np.linalg.norm(s[1:-1])<0.05:
            return 1
            return self.attractors()[1]
        elif s[0] < - 1/self.slope and np.linalg.norm(s[1:-1])<0.05:
            return 0
            return self.attractors()[0]
        else:
            return False