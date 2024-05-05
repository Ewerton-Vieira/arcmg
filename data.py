from torch.utils.data import Dataset
import csv
import torch

class RampFn:
    def __init__(self, slope):
        self.slope = slope
    
    def map(self, x):
        return min(max(-1, self.slope * x), 1)
    
    #def is_pt_in_ramp_attractor(self, x, attractor):
    #    return (attractor * x) > 1/self.slope
    
    #def is_pair_in_ramp_attractor(self, pair, attractor):
    #    return self.is_pt_in_ramp_attractor(pair[0], attractor) and self.is_pt_in_ramp_attractor(pair[1], attractor)

def is_pt_in_attractor(x, attractor, tolerance):
    return abs(x - attractor) < tolerance

def is_pair_in_attractor(pair, attractor, tolerance):
    return is_pt_in_attractor(pair[0], attractor, tolerance) and is_pt_in_attractor(pair[1], attractor, tolerance)

def label_pt_by_attractors(pair, attractor_list, label_threshold):
    label = -1
    for i, attractor in enumerate(attractor_list):
        # The following line of code is specific to 1D
        if is_pt_in_attractor(pair[1], attractor[0], label_threshold):
            label = i
    return label

class Dataset(Dataset):
    def __init__(self, config):

        # Should these variables be moved into __getitem__? 
        
        self.d = config.input_dimension
        data_file = config.data_file
        self.attractor_list = config.attractor_list

        with open(data_file, newline='') as f:
            reader = csv.reader(f)
            self.data = list(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        labeled_point_pair = []

        for k in range(2):
            point = torch.zeros(self.d)
            for i in range(0, self.d):
                point[i] = float(data_point[i + k])

            labeled_point_pair.append(point)

        label = int(data_point[-1])

        labeled_point_pair.append(label)

        # labeled_point_pair has the form [tensor([x]), tensor([xnext]), xnext_label]
        return labeled_point_pair
    
