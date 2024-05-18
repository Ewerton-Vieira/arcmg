import numpy as np

"""For euclidian: space get_bounds=get_true_bounds. For manifold: get_true_bounds=parametrization 
and get_bounds have to be defined as a box that contain the manifold 
(the box can be defined or obtained from data)"""

class BaseSystem:
    def __init__(self, **kwargs):
        self.name = "base_system"

        self.state_bounds = NotImplementedError
    
    def f(self,s):
        return s

    # def sample_state(self):
    #     return np.random.uniform(self.state_bounds[:,0], self.state_bounds[:,1])

    def sample_state(self, num_pts=1, region = False):
        if region is False:
            region = self.get_true_bounds()
        sample_ = np.random.uniform(region[:,0], region[:,1], size=(num_pts, self.dimension()))
        return self.transform(sample_)[0]
    
    def sample_trajectory(self, size=4, region = False):
        if region is False:
            region = self.get_true_bounds()
        initial_point = self.sample_state(num_pts=1, region=region)
        trajectory = [initial_point]
        temp_ =  initial_point
        for i in range(1, size):
            temp_ = self.f(temp_)
            trajectory.append(temp_)
        return trajectory
    
    def label_trajectory(self, size=4, region = False):
        if region is False:
            region = self.get_true_bounds()

        trajectory = self.sample_trajectory(size, region=region)
        labeled_traj = []
        label = -1
        while trajectory:
            end_point = trajectory.pop()
            if self.which_attracting_region(end_point) is not False:
                label = self.which_attracting_region(end_point)
            else:
                if label != -1:
                    label += 2
            labeled_traj.append(end_point.tolist() + [label])
        return labeled_traj[::-1]

    def which_attracting_region(self, s):
        """defaut -1: attracting region not specified"""
        return -1
        


    
    def get_bounds(self): # bounds on the embedded space
        return self.state_bounds

    def get_true_bounds(self): # bounds of the parametrization
        return self.get_bounds()
    
    def dimension(self): # dimension of the manifold
        return self.get_true_bounds().shape[0]
    
    def transform(self, s):
        return s
    
    def inverse_transform(self, s):
        return s