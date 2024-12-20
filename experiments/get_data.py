import argparse
from tqdm import tqdm
from systems.ramp import Ramp
from systems.ramp_rot import Ramp_rot
from systems.homoclinic import Homoclinic
from arcmg.config import Config
import os
import yaml
import numpy as np

slope = 1.75

# Right now this file is creating data with a uniform distribution on specified set of intervals

def main(args):
    yaml_file_path = args.config_dir
    yaml_file = args.config
    length = args.length


    with open(os.path.join(yaml_file_path, yaml_file), mode="rb") as yaml_reader:
        configuration_file = yaml.unsafe_load(yaml_reader)

    config = Config(configuration_file)


    if config.name == "rampfn":
        current_system = Ramp(slope, config.input_dimension)
        region=current_system.get_bounds()
    elif config.name == "ramp_rot":
        current_system = Ramp_rot(slope)
        region=current_system.get_bounds()
    elif config.name == "homoclinic":
        current_system = Homoclinic()
        region = np.array([[-1, 1], [-0.25, 0.25]])
    else:
        NotImplemented


    # a = Ramp_system.f(np.array([1,0]))
    # b = Ramp_system.f(np.array([-1,0]))
    # exit()

        

    save_dir = os.getcwd() + args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    for i in range(args.num_traj):
        # traj = current_system.label_trajectory(length=length, region=np.array([[-1, 1]]+[[-1, 1]]*(config.input_dimension-1)))
        traj = current_system.label_trajectory(length=length, region=region)
        traj = np.array(traj)
        np.savetxt(f"{save_dir}/{counter}.txt",traj,delimiter=",")
        counter += 1

if __name__ == "__main__":

    # yaml_file_path = os.getcwd() + "/output/ramp/"
    yaml_file_path = os.getcwd() + "/output/ramp_rot/"
    yaml_file_path = os.getcwd() + "/output/homoclinic/"
    yaml_file = "config.yaml"
    # save_dir = "/data/ramp"
    save_dir = "/data/ramp_rot"
    save_dir = "/data/homoclinic"
    num_traj = 5000
    length = 200

    parser = argparse.ArgumentParser()
    #  parser.add_argument('--job_index',help='Job index',type=int,default=0)
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default=yaml_file_path)
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default=yaml_file)
    parser.add_argument('--num_traj',help='Number of trajectories',type=int,default=num_traj)
    parser.add_argument('--save_dir',help='Path to save the data',type=str,default=save_dir)
    parser.add_argument('--length',help='Length of the trajectory',type=str,default=length)
    #  parser.add_argument('--condensation_graph',help='Condensation graph file inside config_dir',type=str,default='condensation_graph.txt')

    args = parser.parse_args()

    main(args)
