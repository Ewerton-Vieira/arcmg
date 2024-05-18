import numpy as np 
from arcmg.system import BaseSystem

class Ramp(BaseSystem):
    def __init__(self, slope, dim=10, **kwargs):
        super().__init__(**kwargs)
        self.slope = slope
        self.name = "ramp"
        self.dim = dim
        self.state_bounds = np.array([[-1, 1]]*self.dim)

    def f(self,s):
        # s=s.tolist()[0]
        return np.array([min(max(-1, self.slope * s[0]), 1)] + [s[i]/4 for i in range(1, self.dim)])

    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError
    
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
        
    
    #def is_point_in_ramp_attractor(self, x, attractor):
    #    return (attractor * x) > 1/self.slope
    
    #def is_pair_in_ramp_attractor(self, pair, attractor):
    #    return self.is_pt_in_ramp_attractor(pair[0], attractor) and self.is_pt_in_ramp_attractor(pair[1], attractor)

