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


def main(args):

    radius = 0.2
    folder = "levels9"
    os.makedirs(folder, exist_ok=True)
    save_data = False
    save_images = False
    make_video = False


    save_dir = os.getcwd() + args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_att = dict()
    if args.att_file != '':
        with open(args.att_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                data_att[row[0]]=int(row[1])

    system = Pendulum()

    
    # discretize the state space
    x = np.linspace(-3.14, 3.14, 200)
    y = np.linspace(-6.28, 6.28, 400)
    X, Y = np.meshgrid(x, y)
    

    # system.attrac
    attractors = system.attractors()
    # plot circles
    for i in attractors:
        circle = plt.Circle((i[0], i[1]), radius, color='r', fill=False)
        plt.gca().add_artist(circle)
    dataset = []
    if save_data:

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
    
        # counting-based # 
        for f in tqdm(os.listdir(args.data_dir)):
            data = np.loadtxt(os.path.join(args.data_dir, f), delimiter=',')
            attractor_class, attractor = system.which_attracting_region(data[-1, :], rad=radius)
            if attractor_class == -1:
                continue
            first_attractor_idx = (np.linalg.norm(data[:, :] - attractor, axis=1) < radius).nonzero()[0][0] + 1
            temp = np.zeros((first_attractor_idx, 5))
            temp[:, :2] = data[:first_attractor_idx, :]
            temp[:, 2] = np.arange(first_attractor_idx).tolist()[::-1]
            for i in range(first_attractor_idx):
                dataset.append(temp[i, :].tolist() + [attractor_class])
        dataset = np.array(dataset)

        print("Writing dataset...")
        np.savetxt(os.path.join(save_dir, "dataset_50k.csv"), dataset, delimiter=',')
        print("Done", dataset.shape)

    else:
        print("Loading dataset...")
        dataset = np.loadtxt(os.path.join(save_dir, "dataset_50k.csv"), delimiter=',')
        print("Done", dataset.shape)

    # data normalization
    level_interval = 20
    dataset[:, 3] = dataset[:, 2] // level_interval
    max_levels = np.max(dataset[:, 3])
    dataset[:, 4] = (max_levels - dataset[:, 3]) + 1
    dataset[:, 3] = (max_levels - dataset[:, 3])
    dataset[:, 4] = dataset[:, 4] / (max_levels+1)
    mul = np.ones_like(dataset[:, 4])
    mul[dataset[:, 5] > 0] = -1
    dataset[:, 4] = dataset[:, 4] * mul
    
    if save_images:
        # plt.figure(figsize=(10, 20))
        # plt.grid() 
        colors = plt.cm.rainbow(np.linspace(0, 1, max_levels+1))
        # rainbow(np.linspace(0, 1, max_levels+1))
        for i in tqdm(range(1, max_levels+1)):
            levels = dataset[dataset[:, 2] < i*level_interval]
            levels = levels[levels[:, 2] >= (i-1)*level_interval]
            if levels.shape[0] == 0:
                print("No more levels", i*level_interval)
                break
            plt.scatter(levels[:, 0], levels[:, 1], s=0.1,  color=colors[i], label="Level {}".format(i-1))
            # plt.legend()
            plt.xlim(-3.14, 3.14)
            plt.ylim(-6.28, 6.28)
            
            plt.savefig(f'{folder}/level_{i-1}.png')
        plt.close()

    # from all the saved images, create a video using opencv
    if make_video:
        img_array = []
        for filename in tqdm(sorted(os.listdir(folder), key=lambda x: int(x.split("_")[1].split(".")[0]))):
            img = cv2.imread(os.path.join(folder, filename))
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        # save the video as mp4 
        out = cv2.VideoWriter(f'{folder}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    

    # plot dataset here with dataset[:, :2] as x and y and dataset[:, 4] as color
    print('plotting now...')
    plt.figure(figsize=(10, 10))
    plt.grid()
    # random sample points from the dataset
    # idx = np.random.choice(dataset.shape[0], 7000000, replace=False)
    # plt.scatter(dataset[idx, 0], dataset[idx, 1], s=0.1, c=dataset[idx, 4], cmap='nipy_spectral')
    plt.scatter(dataset[:, 0], dataset[:, 1], s=0.1, c=dataset[:, 4], cmap='nipy_spectral')
    plt.colorbar()
    plt.xlim(-3.14, 3.14)
    plt.ylim(-6.28, 6.28)
    plt.savefig(f'level_all.png')


if __name__ == "__main__":

    step = 5
    data_dir = os.getcwd()+ "/data/pendulum_lqr50k"
    # args.att_file = os.getcwd()+ "/pendulum_lqr1k_success.txt"
    att_file = ''
    system = "pendulum"


    parser = argparse.ArgumentParser()
    #  parser.add_argument('--job_index',help='Job index',type=int,default=0)
    parser.add_argument('--data_dir',help='Directory of data files',type=str,default=data_dir)
    parser.add_argument('--att_file',help='attractors of the data files',type=str,default=att_file)
    parser.add_argument('--save_dir',help='Path to save the data',type=str,default="/data/pendulum/50k")
    parser.add_argument('--system',help='Name of the system',type=str,default=system)
    parser.add_argument('--step',help='Increase the step size',type=int,default=step)
    #  parser.add_argument('--condensation_graph',help='Condensation graph file inside config_dir',type=str,default='condensation_graph.txt')

    args = parser.parse_args()
    main(args)