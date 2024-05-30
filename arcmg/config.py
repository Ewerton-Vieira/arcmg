
class Config:
    def __init__(self, yalm_dict):
        for key, value in list(yalm_dict["parameters"].items()):
            setattr(self, str(key), value)
        self.auto_add_features()
        

    def auto_add_features(self):
        """automatically add num_label, optimizer, scheduler"""
        self.num_attractors = len(self.attractor_list)
        if not hasattr(self, 'dropout'):
            self.dropout = 0
        if not hasattr(self, 'optimizer'):
            self.optimizer = 'Adam'
        if not hasattr(self, 'scheduler'):
            self.scheduler = 'ReduceLROnPlateau'
        if not hasattr(self, 'analyze_train_dynamics'):
            self.analyze_train_dynamics = False
        if not hasattr(self, 'trajectory_length'):
            self.trajectory_length = 1

    # needs to be updated
    def check_types(self):
        if type(self.input_dimension) is not int:
            print("Input dimension has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.input_dimension)))
            exit()
        if type(self.network_width) is not int:
            print("Network width has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.network_width)))
            exit()
        if type(self.learning_rate) is not float:
            print("Learning rate has the incorrect type. Must be " + str(float) + ". Found a " + str(type(self.learning_rate)))
            exit()
        if type(self.model_dir) is not str:
            print("Model directory has the incorrect type. Must be " + str(str) + ". Found a " + str(type(self.model_dir)))
            exit()
        if type(self.log_dir) is not str:
            print("Log directory has the incorrect type. Must be " + str(str) + ". Found a " + str(type(self.log_dir)))
            exit()
        if type(self.verbose) is not bool:
            print("Verbose has the incorrect type. Must be " + str(bool) + ". Found a " + str(type(self.verbose)))
            exit()
        if type(self.batch_size) is not int:
            print("Batch size has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.batch_size)))
            exit()
        if type(self.patience) is not int:
            print("Patience has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.patience)))
            exit()
        if type(self.epochs) is not int:
            print("Epochs has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.epochs)))
            exit()
        if type(self.data_file) is not str:
            print("Data file has the incorrect type. Must be " + str(str) + ". Found a " + str(type(self.data_file)))
            exit()
        if type(self.input_dimension) is not int:
            print("Input dimension has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.input_dimension)))
            exit()
        if type(self.attractor_list) is not list:
            print("Attractor list has the incorrect type. Must be " + str(list) + ". Found a " + str(type(self.attractor_list)))
            exit()
        if type(self.label_threshold) is not float:
            print("Label threshold has the incorrect type. Must be " + str(float) + ". Found a " + str(type(self.label_threshold)))
            exit()
        # if type(self.num_attractors) is not int:
        #     print("Number of attractors has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.num_attractors)))
        #     exit()
        if type(self.num_data_points) is not int:
            print("Number of data points has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.num_data_points)))
        if type(self.domain) is not list:
            print("Domain has the incorrect type. Must be " + str(list) + ". Found a " + str(type(self.domain)))
