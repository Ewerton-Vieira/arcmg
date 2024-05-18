import os
import argparse 
import uuid
import yaml
import numpy as np
import datetime
import itertools
from arcmg.utils import get_bistable_CG


class Experiment:
    def __init__(self, yaml_file, CG_type = "bistable"):
        with open(yaml_file, mode="rb") as yaml_reader:
            self.yaml_dict = yaml.unsafe_load(yaml_reader)
        # print(self.yaml_dict["parameters"])
        # print(self.yaml_dict["generate_parameters"])
        self.name = self.yaml_dict['parameters']['name']

        self.CG_type = CG_type

        

        parser = argparse.ArgumentParser()
        parser.add_argument('--config',help='Base yaml file inside config/',type=str,default='text.yaml')
        parser.add_argument('--dir', help='Directory to save generated config files', type=str, default='tmp_config')
        # parser.add_argument('--name', help='Name of the experiment', type=str,required=True)
        parser.add_argument('--max_exps',help='Max num of exp for each folder',type=int,default=1)
        parser.add_argument('--shell',help='Generate shell script to send job',type=str,default="")
        parser.add_argument('--out_extra',help='Add extra name to output',type=str,default=datetime.datetime.today().strftime("%Y_%m_%d_%H%M"))

        self.args = parser.parse_args()

        self.save_folder_sh = os.path.join(os.getcwd(),f"tmp_{self.name}{self.args.out_extra}")

    def generate_config(self):

        uuid_generator = uuid.uuid4

        # copy parameters
        config = self.yaml_dict["parameters"]

        new_config = config.copy()

        # gen_par = self.yaml_dict["parameters"] | self.yaml_dict["generate_parameters"]
         
        gen_par = self.yaml_dict["generate_parameters"].copy()

        counter = 0
        self.dir_counter = 0

        
        parameters2change = list(self.yaml_dict["generate_parameters"].keys())
        # parameters2change = ["step","network_width","num_labels","label_threshold","seed","data_file","verbose"]
        # converting value to list for future for loop
        for key in parameters2change:
            if isinstance(gen_par[key],(str,int,float,bool)):
                gen_par[key] = [gen_par[key]]
            elif isinstance(gen_par[key],range):
                gen_par[key] = list(gen_par[key])
            elif isinstance(gen_par[key],list):
                continue
            else:
                raise NotImplementedError


        for element in itertools.product(*[gen_par[par] for par in parameters2change]):
            id = uuid_generator().hex
            for index in range(len(parameters2change)):
                new_config[parameters2change[index]]= element[index]

            # use num_label to create bistable CG
            if self.CG_type == 'bistable':
                new_config["condensation_graph"] = get_bistable_CG(new_config["num_labels"])


            
            # output path
            
            output_temp = f"output/{new_config['name']}{self.args.out_extra}/{id}/"

            new_config["log_dir"] = output_temp + "logs/"
            new_config["model_dir"] = output_temp + "models/"
            new_config["output_dir"] = output_temp

            new_yaml_file = {"parameters":new_config}

            # counter to break into multiple folders
            counter += 1
            if counter % self.args.max_exps == 0 and counter != 1: 
                self.dir_counter += 1
            
            # save temp config to run experiments
            temp_dir_exp = f'{self.save_folder_sh}/{self.dir_counter}'
            if not os.path.exists(temp_dir_exp):
                os.makedirs(temp_dir_exp)
            with open(f'{temp_dir_exp}/{id}.yaml', 'w') as f:
                yaml.dump(new_yaml_file, f)



            # save yaml file with new parameters
            if not os.path.exists(output_temp):
                os.makedirs(output_temp)
            with open(f'{output_temp}/config.yaml', mode="wt") as file:
                # adding "parameters" header
                yaml.safe_dump(new_yaml_file, file)

            # save all ids
            with open(f"output/{new_config['name']}{self.args.out_extra}/all_exps.yaml", 'a') as f:
                yaml.safe_dump({id:{"parameters":new_config}}, f)
    
    def exp_cluster_array(self, name_sh = "/arcmg_array.sh"):
        name_sh = os.getcwd() + name_sh

        with open(name_sh, "r") as reader:
            lines = reader.readlines()
            lines.pop(1)
        return lines

    def generate_job_array(self):
        
        if not os.path.exists(self.save_folder_sh):
            os.makedirs(self.save_folder_sh)

        lines = self.exp_cluster_array()

        shell_name = f"{self.save_folder_sh}/{self.name}.sh"
        with open(shell_name, "w") as file:
            file.write(lines[0])
            file.write(f"#SBATCH --array=0-{self.dir_counter}\n")

            for line in lines[1::]:
                
                if line[0:11] == "search_dir=":
                    write_path = f"\nsearch_dir=tmp_{self.name}{self.args.out_extra}/$SLURM_ARRAY_TASK_ID/\n"
                    file.write(write_path)
                    continue

                file.write(line)

    # def get_bistable_CG(self, num_labels):
    #     CG = dict()
    #     num_nodes = num_labels + 2
    #     CG["num_nodes"] = num_nodes
    #     CG["edge_list"] = []

    #     for i in range(num_nodes//2):
    #         CG["edge_list"].append([max(2*i-1,0), 2*(i+1)-1])
    #         CG["edge_list"].append([2*i, 2*(i+1)])
    #     CG["edge_list"] = str(CG["edge_list"])

    #     return CG


            
if __name__ == "__main__":
    exp = Experiment(os.path.join(os.getcwd(),"rampfn_exp.yaml"))
    exp.generate_config()
    exp.generate_job_array()

    



