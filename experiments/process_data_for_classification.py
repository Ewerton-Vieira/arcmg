import argparse
from tqdm import tqdm
import os
import numpy as np
import csv
import cv2
from systems.pendulum import Pendulum
from matplotlib import pyplot as plt


def get_label(system, end_point):
    if system == "pendulum":
        pendulum_ = Pendulum()
        return pendulum_.which_attracting_region(end_point)
    else:
        return NotImplementedError

def main(args, kwargs):

    save_base_data = kwargs.get('save_base_data', False)
    # save_final_data = kwargs.get('save_final_data', False)
    # save_images = kwargs.get('save_images', False)
    # make_video = kwargs.get('make_video', False)
    radius = kwargs.get('radius', 0.2)
    # folder = kwargs.get('folder', "levels")

    # level_interval = kwargs.get('level_interval', 1)
    dataset_size = kwargs.get('dataset_size', '1k')
    save_dir = kwargs.get('save_dir', f"{os.getcwd()}/data/pendulum/{dataset_size}")

    # os.makedirs(folder, exist_ok=True)

    
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # data_att = dict()
    # if args.att_file != '':
    #     with open(args.att_file) as csv_file:
    #         csv_reader = csv.reader(csv_file, delimiter=',')
    #         for row in csv_reader:
    #             data_att[row[0]]=int(row[1])

    system = Pendulum()

    
    # discretize the state space
    x = np.linspace(-3.14, 3.14, 200)
    y = np.linspace(-6.28, 6.28, 400)
    X, Y = np.meshgrid(x, y)
    

    # system.attrac
    attractors = system.attractors()
    # plot circles
    # for i in attractors:
    #     circle = plt.Circle((i[0], i[1]), radius, color='r', fill=False)
    #     plt.gca().add_artist(circle)

    full_dataset = []
    dataset = []
    trajs = 0
    ctr = 0
    missed = 0
    if save_base_data:
        # counting-based # 
        for f in tqdm(os.listdir(args.data_dir)):
            trajs += 1
            data = np.loadtxt(os.path.join(args.data_dir, f), delimiter=',')
            attractor_class, attractor = system.which_attracting_region(data[-1, :], rad=radius)
            if attractor_class == -1:
                first_attractor_idx = data.shape[0]
                # missed += 1
                # continue

            else:
                # continue

                first_attractor_idx = (np.linalg.norm(data[:, :] - attractor, axis=1) < radius).nonzero()[0][0] + 1 # get the first attractor index
            full_dataset_tmp = np.zeros((first_attractor_idx, data.shape[1]))
            full_dataset_tmp[:, :2] = data[:first_attractor_idx, :]
            # print(first_attractor_idx, data.shape)
            
            ctr += (first_attractor_idx - 1)
            # print(first_attractor_idx)
            # dataset.extend(np.hstack([data[:first_attractor_idx-1, :], data[1:first_attractor_idx, :], np.ones((first_attractor_idx-1, 1))]).tolist())

            # print(len(dataset))

            for _ in range(kwargs['samples']):
                idx1, idx2 = tuple(sorted(np.random.randint(0, first_attractor_idx, 2).tolist()))
                dataset.append([*(data[idx1, :].tolist()), *(data[idx2,:].tolist()), 1])

            full_dataset.extend(full_dataset_tmp.tolist())
                  
        # print(np.array(dataset).shape, np.array(full_dataset).shape)
        # 
        
        pos_dataset = np.array(dataset)
        full_dataset = np.array(full_dataset)
        
        neg_dataset = np.zeros((pos_dataset.shape[0], pos_dataset.shape[1]))
        for neg_ix in range(pos_dataset.shape[0]):
            idx1, idx2 = tuple(np.random.randint(0, full_dataset.shape[0], 2).tolist())
            neg_dataset[neg_ix, :pos_dataset.shape[1]-1] = np.hstack((full_dataset[idx1, :], full_dataset[idx2, :]))
    
        dataset = np.vstack((pos_dataset, neg_dataset))
        print(ctr, trajs)
        print("Writing dataset...")
        np.savetxt(os.path.join(save_dir, f"dataset_{dataset_size}.csv"), dataset, delimiter=',')
        print("Done", dataset.shape)

    else:
        print("Loading dataset...")
        dataset = np.loadtxt(os.path.join(save_dir, f"dataset_{dataset_size}.csv"), delimiter=',')
        print("Done", dataset.shape)

    
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(pos_dataset[:, 0], pos_dataset[:, 1], c='green', s=1)
    ax[1].scatter(neg_dataset[:, 0], neg_dataset[:, 1], c='red', s=1)
    plt.savefig(f"dataset_{dataset_size}.png")

if __name__ == "__main__":

    step = 5
    # TODO: use config file instead of args and kwargs
    kwargs = dict()
    kwargs['save_base_data'] = True # change this everytime you change the dataset size and/or the attractor_radius
    kwargs['save_final_data'] = True
    kwargs['save_images'] = True
    kwargs['make_video'] = False
    kwargs['radius'] = 0.05
    kwargs['folder'] = "levels"
    kwargs['dataset_size'] = '1k'
    kwargs['level_interval'] = 20
    kwargs['samples'] = 100
    kwargs['cwd'] = f"/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/arcmg"
    kwargs['save_dir'] = f"{kwargs['cwd']}/data/pendulum_clf/{kwargs['dataset_size']}"

    cwd = "/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/arcmg"

    data_dir = cwd + f"/data/pendulum_lqr/pendulum_lqr{kwargs['dataset_size']}/pendulum_lqr{kwargs['dataset_size']}"
    # args.att_file = os.getcwd()+ "/pendulum_lqr1k_success.txt"
    att_file = ''
    system = "pendulum"

    parser = argparse.ArgumentParser()
    #  parser.add_argument('--job_index',help='Job index',type=int,default=0)
    parser.add_argument('--data_dir',help='Directory of data files',type=str,default=data_dir)
    parser.add_argument('--att_file',help='attractors of the data files',type=str,default=att_file)
    parser.add_argument('--save_dir',help='Path to save the data',type=str,default=f"{cwd}/data/pendulum/{kwargs['dataset_size']}")
    parser.add_argument('--system',help='Name of the system',type=str,default=system)
    parser.add_argument('--step',help='Increase the step size',type=int,default=step)
    #  parser.add_argument('--condensation_graph',help='Condensation graph file inside config_dir',type=str,default='condensation_graph.txt')


    args = parser.parse_args()
    
    # dataset collection
    dataset_sizes = ['1k'] # ['1k', '10k', '50k']
    level_intervals = [20]

    for dataset_size in dataset_sizes:
        for level_interval in level_intervals:
            kwargs['dataset_size'] = dataset_size
            kwargs['level_interval'] = level_interval
            kwargs['save_dir'] = f"{kwargs['cwd']}/data/pendulum/clf_{kwargs['dataset_size']}"
            print(f"Dataset size: {dataset_size}, Level interval: {level_interval}")
            main(args, kwargs)

# grid-based #
# ctr_dict = dict()
# ctr_contains_dict = dict()
# ctr_filler = dict()
# ctr_to_stop = 0
# for f in tqdm(os.listdir(args.data_dir)):
#     data = np.loadtxt(os.path.join(args.data_dir, f), delimiter=',')
#     attractor_class, attractor = system.which_attracting_region(data[-1, :], rad=radius)
#     if attractor_class == -1:
#         continue
#     first_attractor_idx = (np.linalg.norm(data[:, :] - attractor, axis=1) < radius).nonzero()[0][0] + 1
#     temp = np.zeros((first_attractor_idx, 3))
#     temp[:, :2] = data[:first_attractor_idx, :]
#     temp[:, 2] = np.arange(first_attractor_idx).tolist()[::-1]

#     # find nearest point in X,Y grid
#     for i in range(first_attractor_idx):
#         x = temp[i, 0]
#         y = temp[i, 1]
#         idx = np.argmin((X - x)**2 + (Y - y)**2)
#         nearest_point = (X.flatten()[idx], Y.flatten()[idx])
#         # check if the nearest point is already in the dictionary
#         if ctr_contains_dict.get(nearest_point, None) is None:
#             # check if the labels are different
#             ctr_contains_dict[nearest_point] = True
#             ctr_dict[nearest_point] = []
#             ctr_filler[nearest_point] = []
#         ctr_dict[nearest_point].append(temp[i, 2])
#         ctr_filler[nearest_point].append(f)

# # get the maximum level for each point
# max_levels = 0
# new_dataset = dict()
# for i in list(ctr_dict.keys()):
#     new_dataset[i] = max(ctr_dict[i])
#     if max(ctr_dict[i]) > max_levels:
#         max_levels = max(ctr_dict[i])
# for i in range(first_attractor_idx):
#     x = temp[i, 0]
#     y = temp[i, 1]
#     idx = np.argmin((X - x)**2 + (Y - y)**2)
#     nearest_point = (X.flatten()[idx], Y.flatten()[idx])
#     if new_dataset.get(nearest_point, None) is not None:
#         dataset.append(temp[i, :2].tolist() + [new_dataset[nearest_point], attractor_class])