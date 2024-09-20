import argparse
from tqdm import tqdm
import os
import numpy as np
import csv
from systems.pendulum import Pendulum


def get_label(system, end_point):
    if system == "pendulum":
        pendulum_ = Pendulum()
        return pendulum_.which_attracting_region(end_point)
    else:
        return NotImplementedError


    

def main(args):

    save_dir = os.path.join(os.getcwd(),args.save_dir)
    save_dir = os.getcwd() + args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_att = dict()
    if args.att_file != '':
        with open(args.att_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                data_att[row[0]]=int(row[1])

    counter = 0
    for f in tqdm(os.listdir(args.data_dir)):
        data = np.loadtxt(os.path.join(args.data_dir, f), delimiter=',')
        data = data.tolist()
        if args.att_file != '':
            label=data_att[str(f)]
        else:
            label = get_label(args.system, data[-1])
        new_data = []

        new_data.append(data[-1] + [label])
        while data:
            end_point = data.pop()
            if label == -1:
                new_data.append(end_point + [label])
            else:
                if get_label(args.system, end_point) is False:
                    label += 2
                    new_data.append(end_point + [label])
                    
            step_counter = args.step - 1
            while data and step_counter > 0:
                data.pop()
                step_counter -= 1


            
        new_data = new_data[::-1]
        name_out =  save_dir + "/" + str(counter) +".txt"
        np.savetxt(name_out,new_data,delimiter=",")
        counter += 1










if __name__ == "__main__":

    step = 4
    data_dir = os.getcwd()+ "/data/ramp_rot_base"
    # args.att_file = os.getcwd()+ "/pendulum_lqr1k_success.txt"
    att_file = ''
    system = "pendulum"
    system = "ramp_rot"
    save_dir = '/data/ramp_rot'



    parser = argparse.ArgumentParser()
    #  parser.add_argument('--job_index',help='Job index',type=int,default=0)
    parser.add_argument('--data_dir',help='Directory of data files',type=str,default=data_dir)
    parser.add_argument('--att_file',help='attractors of the data files',type=str,default=att_file)
    parser.add_argument('--save_dir',help='Path to save the data',type=str,default=save_dir)
    parser.add_argument('--system',help='Name of the system',type=str,default=system)
    parser.add_argument('--step',help='Increase the step size',type=int,default=step)
    #  parser.add_argument('--condensation_graph',help='Condensation graph file inside config_dir',type=str,default='condensation_graph.txt')

    args = parser.parse_args()



    main(args)