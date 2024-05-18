from torch import nn
import torch


"""todo:
forward function should be the output of the netword

"""
class PhaseSpaceClassifier(nn.Module):
    def __init__(self, config):
        super(PhaseSpaceClassifier, self).__init__()

        self.config = config
        self.dropout = config.dropout

        num_layers = config.num_layers if hasattr(config, 'num_layers') else 2
        hidden_shape = num_layers        

        # self.input_layer = nn.Sequential(
        #     nn.Linear(self.config.input_dimension, self.config.network_width), nn.ReLU(True))

        self.input_layer = nn.Sequential()
        self.input_layer.add_module('1', nn.Linear(self.config.input_dimension, self.config.network_width))
        self.input_layer.add_module('2', nn.Dropout(self.dropout))
        self.input_layer.add_module('3', nn.ReLU(True))

        self.hidden_layers = nn.Sequential()
        for i in range(hidden_shape-1):
            self.hidden_layers.add_module(f"linear_{i}", nn.Linear(config.network_width, config.network_width))
            self.hidden_layers.add_module(f"dropout_{i}", nn.Dropout(self.dropout))
            self.hidden_layers.add_module(f"relu_{i}", nn.ReLU(True))
        
        # self.output_layer = nn.Sequential(
        #     nn.Linear(config.network_width, config.num_labels), nn.Softmax(dim=1))

        self.output_layer = nn.Sequential()
        self.output_layer.add_module('1', nn.Linear(config.network_width, config.num_labels))
        self.output_layer.add_module('2', nn.Dropout(self.dropout))
        self.output_layer.add_module('3', nn.Softmax(dim=1))

        
    def classification_of_point(self, point):
        point = point.unsqueeze(0)
        output = self.vector_of_probabilities(point)
        classification = int(torch.argmax(output))
        return output[0][classification], classification + self.config.num_attractors
    
    def add_layers(self, new_hidden_layers):
        "Adding layers at the beginning"
        new_layers = nn.Sequential()
        for i in range(new_hidden_layers):
            new_layers.add_module(f"linear_{i}", nn.Linear(self.config.network_width, self.config.network_width))
            new_layers.add_module(f"relu_{i}", nn.ReLU(True))

        self.hidden_layers = nn.Sequential(new_layers, self.hidden_layers)

    def update_out_layer(self, new_num_labels):
        self.output_layer[0] = nn.Linear(self.config.network_width, new_num_labels)

    def freeze_layers(self, freeze_out = False):
        for param in self.hidden_layers.parameters():
            param.requires_grad = False
        for param in self.input_layer.parameters():
            param.requires_grad = False
        if freeze_out:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    def unfreeze_layers(self, unfreeze_out = False):
        for param in self.hidden_layers.parameters():
            param.requires_grad = True
        for param in self.input_layer.parameters():
            param.requires_grad = True
        if unfreeze_out:
            for param in self.output_layer.parameters():
                param.requires_grad = True

    def vector_of_probabilities(self, point): 
        return self.output_layer(self.hidden_layers(
            self.input_layer(point))
                                 )

    def forward(self, xt, xnext):
        probs_x = self.vector_of_probabilities(xt)
        probs_xnext = self.vector_of_probabilities(xnext)
        return {'probs_x': probs_x, 'probs_xnext': probs_xnext}
    
    def remove_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module = nn.Dropout(0)

