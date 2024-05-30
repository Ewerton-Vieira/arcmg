from arcmg.config import Config
from arcmg.data import Dataset
from torch.utils.data import DataLoader
import argparse
import yaml
import os
import numpy as np


from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import argparse
import os

def main(args, yaml_file):
    yaml_file_path = args.config_dir

    with open(os.path.join(yaml_file_path, yaml_file), mode="rb") as yaml_reader:
        configuration_file = yaml.unsafe_load(yaml_reader)

    config = Config(configuration_file)
    # config.check_types()

    config.output_dir = os.path.join(os.getcwd(),config.output_dir)

    dataset = Dataset(config)

    # dataset = TrajectoryDataset(config)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    assert len(dataset) >= args.num_trajs, "Not enough trajectories in dataset"
    idxes = np.random.choice(len(dataset), args.num_trajs, replace=False)




    fig, ax = plt.subplots(figsize=(10,5))


    ax.set_title("Data")

    count = 0
    for idx in tqdm(idxes):
        traj = dataset[idx]
        z = np.array([traj[0].detach().numpy(),traj[1].detach().numpy()])
        
        ax.plot(z[:,0], z[:,1],color='black')
        # ax.scatter(z[0][0], z[0][1], color='r', marker='.')
        # ax.scatter(z[-1][0], z[-1][1], color='g', marker='x')

        
        
    plt.show()

    # plt.savefig(os.path.join(config.output_dir, "trajectories.png"))
    # plt.close()

        
if __name__ == "__main__":

# 
    yaml_file_path = "/Users/ewerton/Dropbox/Codes/arcmg/experiments/output/pendulum"

    parser = argparse.ArgumentParser()
    #  parser.add_argument('--job_index',help='Job index',type=int,default=0)
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default=yaml_file_path)
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default="config.yaml")
    #  parser.add_argument('--verbose',help='Print training output',action='store_true')
    parser.add_argument('--output_dir',type=str, default="")
    parser.add_argument('--num_trajs', type=int, default=300)
    args = parser.parse_args()

    main(args, args.config) 