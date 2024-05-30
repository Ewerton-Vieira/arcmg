import argparse
from tqdm import tqdm
from systems.ramp import Ramp
from arcmg.config import Config
import os
import yaml
import numpy as np

slope = 2

# Right now this file is creating data with a uniform distribution on specified set of intervals

def main(args):
    yaml_file_path = args.config_dir
    yaml_file = args.config


    with open(os.path.join(yaml_file_path, yaml_file), mode="rb") as yaml_reader:
        configuration_file = yaml.unsafe_load(yaml_reader)

    config = Config(configuration_file)


    Ramp_system = Ramp(slope, config.input_dimension)

    save_dir = os.getcwd() + args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    for i in range(args.num_traj):
        traj = Ramp_system.label_trajectory(region=np.array([[-0.4, 0.4]]+[[-1, 1]]*(config.input_dimension-1)))

        traj = np.array(traj)
        np.savetxt(f"{save_dir}/{counter}.txt",traj,delimiter=",")
        counter += 1

if __name__ == "__main__":

    yaml_file_path = os.getcwd() + "/output/ramp/"
    yaml_file = "config.yaml"

    parser = argparse.ArgumentParser()
    #  parser.add_argument('--job_index',help='Job index',type=int,default=0)
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default=yaml_file_path)
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default=yaml_file)
    parser.add_argument('--num_traj',help='Number of trajectories',type=int,default=1000)
    parser.add_argument('--save_dir',help='Path to save the data',type=str,default="/data/ramp")
    #  parser.add_argument('--condensation_graph',help='Condensation graph file inside config_dir',type=str,default='condensation_graph.txt')

    args = parser.parse_args()

    main(args)