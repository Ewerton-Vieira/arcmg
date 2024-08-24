import numpy as np 
from arcmg.system import BaseSystem

class Pendulum(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "pendulum"
        self.state_bounds = np.array([[-3.14, 3.14], [-6.28, 6.28]])

    # def f(self,s):
    #     # s=s.tolist()[0]
    #     return np.array([min(max(-1, self.slope * s[0]), 1)] + [s[i]/4 for i in range(1, self.dim)])

    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError
    
    def attractors(self):
        return [np.array([0,0]), np.array([2.1,0]), np.array([-2.1,0])]

    def which_attracting_region(self, s, rad):
        """defaut -1: attracting region not specified"""
        s=np.array(s)
        att_neighborh_radius = rad
        attractors = self.attractors()
        if np.linalg.norm(s - attractors[0]) < att_neighborh_radius:
            return 0, attractors[0]
        elif np.linalg.norm(s - attractors[1]) < att_neighborh_radius: 
            return 1, attractors[1]
        elif np.linalg.norm(s - attractors[2]) < att_neighborh_radius:
            return 2, attractors[2]
        else:
            return -1, None