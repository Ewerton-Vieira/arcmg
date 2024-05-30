from arcmg.training import ClassifierTraining
from arcmg.config import Config
from arcmg.data import Dataset
from arcmg.plot import plot_classes, plot_loss, plot_classes_2D
from torch.utils.data import DataLoader
from arcmg.utils import find_classes, is_bistable, penalty_submatrix
import argparse
import csv
import torch
import yaml
import os
from torch import nn

torch.autograd.set_detect_anomaly(True)

num_points_in_mesh = 1000

def transfer_learning_training(trainer, config):

    transfer_steps = config.num_labels//2

    for i in range(transfer_steps):
        temp_num_labels = 3 + 2*i
        trainer.num_labels = temp_num_labels

        trainer.classifier.update_out_layer(temp_num_labels)

        print(trainer.classifier)

        trainer.train()

        name = f"_classes{temp_num_labels}"
        plot_classes(trainer.classifier, config, name)

        trainer.classifier.freeze_layers()

    trainer.classifier.unfreeze_layers()
    trainer.train()



        # trainer.num_clabels = 5

        # trainer.classifier.update_out_layer(5)
        # ###

        # exit()

        # trainer.classifier.freeze_layers()

        # trainer.classifier.add_layers(1)

        # # trainer.classifier.add_layers(2)
        
        # trainer.classifier.update_out_layer(7)

        # print(trainer.classifier)

        # trainer.classifier.unfreeze_layers()


        # maxpools = [k for k, m in trainer.classifier.named_modules()]

        # print(maxpools)

        # # maxpools = [k for k, m in trainer.classifier.named_modules() 
        # #             if type(m).__name__ == 'MaxPool2d']

        # trainer.classifier.hidden_layers = nn.Sequential(trainer.classifier.hidden_layers, trainer.classifier.hidden_layers)
        # # trainer.classifier.hidden_layers.fc2 = nn.Sequential()

        # # trainer.classifier.hidden_layers.linear_2 = nn.Sequential()

        # trainer.classifier.hidden_layers.linear_2 = nn.Sequential(
        #     nn.Linear(32, 32),  # New layer
        #     nn.ReLU(),
        #     nn.Linear(32, 7),  # New layer
        #     nn.ReLU()
        # )
        # print(trainer.classifier)
    return 

def main(args, yaml_file):
    ###
    # yaml_file = "config/rampfn.yaml"
    yaml_file_path = args.config_dir

    with open(os.path.join(yaml_file_path, yaml_file), mode="rb") as yaml_reader:
        configuration_file = yaml.unsafe_load(yaml_reader)

    config = Config(configuration_file)
    # config.check_types()

    config.output_dir = os.path.join(os.getcwd(),config.output_dir)

    dynamics_dataset = Dataset(config)

    dynamics_train_size = int(0.8*len(dynamics_dataset))
    dynamics_test_size = len(dynamics_dataset) - dynamics_train_size
    dynamics_train_dataset, dynamics_test_dataset = torch.utils.data.random_split(dynamics_dataset, [dynamics_train_size, dynamics_test_size])
    
    dynamics_train_loader = DataLoader(dynamics_train_dataset, batch_size=config.batch_size, shuffle=True)
    dynamics_test_loader = DataLoader(dynamics_test_dataset, batch_size=config.batch_size, shuffle=True)

    if config.verbose:
        print("Train size: ", len(dynamics_train_dataset))
        print("Test size: ", len(dynamics_test_dataset))

    loaders = {
        'train_dynamics': dynamics_train_loader,
        'test_dynamics': dynamics_test_loader,
    }

    # save test_loss to a csv file
    with open(config.output_dir + 'result.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        trainer = ClassifierTraining(loaders, config)

        print(trainer.classifier)
        # print(penalty_matrix)        
        if args.transfer_learning:
            transfer_learning_training(trainer, config)

        else:
            trainer.train()

        # if config.dropout != 0:  # after training with dropout, set it to zero and train more
        #     name_before_dropout_zero = "_dropout_not_0"
        #     plot_classes(trainer.classifier, config, name_before_dropout_zero)

        #     for _, module_in_hidden_out in trainer.classifier.named_children():
        #         for name, module in module_in_hidden_out.named_children():
        #             if isinstance(module,torch.nn.Dropout):
        #                 setattr(module_in_hidden_out, name, nn.Dropout(0))
 
        #     print(trainer.classifier)
        #     trainer.train()



        trainer.save_model('classifier') 

        # plot_classes(trainer.classifier, config)  
        plot_classes_2D(trainer.classifier, config)          

        train_losses = trainer.train_losses['loss_total']
        test_losses = trainer.test_losses['loss_total']
        plot_loss(config, train_losses, test_losses)

        # class_set = find_classes(trainer.classifier, config, num_points_in_mesh)
        # num_classes_found = len(class_set)

        # writer.writerow(["class_set", "num_classes_found", "is_bistable", "final_train_loss", "final_test_loss"])
        # writer.writerow([class_set, num_classes_found, is_bistable(class_set), train_losses[-1], test_losses[-1]])

    if config.analyze_train_dynamics:
        with open(config.output_dir + 'train_dynamics.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            class_order_data = trainer.class_order_data

            #print('class order data: ', class_order_data)

            for ordering in class_order_data:
                writer.writerow(ordering)


if __name__ == "__main__":

# 
    yaml_file_path = "/Users/ewerton/Dropbox/Codes/arcmg/experiments/output/pendulum"

    parser = argparse.ArgumentParser()
    #  parser.add_argument('--job_index',help='Job index',type=int,default=0)
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default=yaml_file_path)
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default="config.yaml")
    parser.add_argument('--transfer_learning',help='Config file inside config_dir',action='store_true')
    #  parser.add_argument('--verbose',help='Print training output',action='store_true')

    args = parser.parse_args()

    # args.transfer_learning = True

    main(args, args.config) 