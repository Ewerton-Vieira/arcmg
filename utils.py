import numpy as np
import torch

def rampfn_special_data_sample(regions, total_num_pts):
    pts_per_region = total_num_pts//len(regions)
    total_sample = np.empty(0)
    total_sample = sample_points([regions[0]], pts_per_region)
    flag = True

    for region in regions:
        if flag:
            flag = False
        else:
            total_sample = np.concatenate((sample_points([region], pts_per_region), total_sample), axis = 0)
    return total_sample

def sample_points(domain, num_pts):
    # Sample num_pts in dimension dim, where each
    # component of the sampled points are in the
    # ranges given by lower_bounds and upper_bounds
    dim = len(domain)
    lower_bounds = []
    upper_bounds = []
    for i in domain:
        lower_bounds.append(i[0])
        upper_bounds.append(i[1])
    X = np.random.uniform(lower_bounds, upper_bounds, size=(num_pts, dim))
    return X

def find_class_order(model, config, num_points_in_mesh = 1000):
    X = sample_points(config.domain, num_points_in_mesh)
    X_sorted = sorted(X, key=lambda x: x[0])

    classes_found_list = []
    list_is_empty = True
    point = torch.zeros(config.input_dimension, requires_grad = False)
    for i in range(0, num_points_in_mesh):
        point = torch.FloatTensor(X_sorted[i])
        _, classification = model.classification_of_point(point)
        if list_is_empty:
            classes_found_list.append(classification)
            list_is_empty = False
        elif classification != classes_found_list[-1]:
            classes_found_list.append(classification)
    
    return classes_found_list

def find_classes(model, config, num_points_in_mesh):
    X = sample_points(config.domain, num_points_in_mesh)

    classes_found_set = set()
    point = torch.zeros(config.input_dimension, requires_grad = False)
    for i in range(0, num_points_in_mesh):
        point = torch.FloatTensor(X[i])
        _, classification = model.classification_of_point(point)
        classes_found_set.add(classification)
    return classes_found_set

def is_bistable(class_set):
    triple_conditions = [False, False, False]
    for i in class_set:
        if i == 0:
            triple_conditions[0] = True
        elif i % 2 == 0:
            triple_conditions[1] = True
        else:
            triple_conditions[2] = True
        
    return all(triple_conditions)

def get_bistable_CG(num_labels):
    CG = dict()
    num_nodes = num_labels + 2
    CG["num_nodes"] = num_nodes
    CG["edge_list"] = []

    for i in range(num_nodes//2):
        CG["edge_list"].append([2*i+2, 2*i])
        CG["edge_list"].append([min(2*(i+1)+1,num_labels+1), 2*i+1])
        # CG["edge_list"].append([max(2*i-1,0), 2*(i+1)-1])
        # CG["edge_list"].append([2*i, 2*(i+1)])
    CG["edge_list"] = str(CG["edge_list"])
    return CG

def penalty_submatrix(config, matrix, num_labels):
    """"Input: penalty matrix and a num_labels<= len(penalty matrix) 
    Return: a penalty submatrix for CG with num_labels+2 nodes"""
    if matrix.shape[0] < num_labels + config.num_attractors or num_labels%2 == 0:
        raise ValueError("number of labels is even or greater than expected")
    return matrix[config.num_labels - num_labels:config.num_labels + config.num_attractors, config.num_labels - num_labels:config.num_labels + config.num_attractors]