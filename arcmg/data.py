from torch.utils.data import Dataset
import csv
import torch
from tqdm import tqdm
import os
import numpy as np

# class RampFn:
#     def __init__(self, slope):
#         self.slope = slope
    
#     def map(self, x):
#         return min(max(-1, self.slope * x), 1)
    
    #def is_pt_in_ramp_attractor(self, x, attractor):
    #    return (attractor * x) > 1/self.slope
    
    #def is_pair_in_ramp_attractor(self, pair, attractor):
    #    return self.is_pt_in_ramp_attractor(pair[0], attractor) and self.is_pt_in_ramp_attractor(pair[1], attractor)

# def is_pt_in_attractor(x, attractor, tolerance):
#     return abs(x - attractor) < tolerance

# def is_pair_in_attractor(pair, attractor, tolerance):
#     return is_pt_in_attractor(pair[0], attractor, tolerance) and is_pt_in_attractor(pair[1], attractor, tolerance)

# def label_pt_by_attractors(pair, attractor_list, label_threshold):
#     label = -1
#     for i, attractor in enumerate(attractor_list):
#         # The following line of code is specific to 1D
#         if is_pt_in_attractor(pair[1], attractor[0], label_threshold):
#             label = i
#     return label

class Dataset(Dataset):
    def __init__(self, config):

        # Should these variables be moved into __getitem__? 
        
        self.d = config.input_dimension
        # self.attractor_list = config.attractor_list
        X=[]
        Y=[]


        for f in tqdm(os.listdir(config.data_file)):
            data = np.loadtxt(os.path.join(config.data_file, f), delimiter=',')
            X.append(data[:-1])
            Y.append(data[1:])

        self.X = np.vstack(X)
        self.Y = np.vstack(Y)
        assert len(self.X) == len(self.Y), "X and Y must have the same length"

        self.data = np.concatenate((self.X,self.Y), axis=1)
        self.data = torch.from_numpy(self.data).float()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        labeled_point_pair = [data_point[:self.d], 
                              data_point[self.d+1:-1], 
                              int(data_point[-1])]
        # labeled_point_pair = [torch.tensor(data_point[:self.d]), 
        #                       torch.tensor(data_point[self.d+1:-1]), 
        #                       int(data_point[-1])]

        # labeled_point_pair has the form [tensor([x]), tensor([y]), y_label]
        return labeled_point_pair
    
