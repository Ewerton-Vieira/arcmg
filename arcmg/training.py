import torch 
import os
import numpy as np
from tqdm import tqdm
from .utils import find_class_order
from .network import PhaseSpaceClassifier
from scipy.stats import norm

# to do: make this just depend on config
# start with dropout after patience turn off dropout
class ClassifierTraining:
    def __init__(self, loaders, config):
        self.config = config
        self.num_labels = self.config.num_labels
        self.lr = self.config.learning_rate
        self.classifier = PhaseSpaceClassifier(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
        self.train_loader = loaders['train_dynamics']
        self.test_loader = loaders['test_dynamics']
        self.reset_losses()
        if self.config.analyze_train_dynamics:
            self.class_order_data = [find_class_order(self.classifier, self.config)]

    def reset_losses(self):
        self.train_losses = {'loss_total': []}
        self.test_losses = {'loss_total': []}
    
    def save_model(self, name):
        model_path = os.path.join(os.getcwd(),self.config.model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.classifier, os.path.join(model_path, f'{name}.pt'))

    def load_model(self, name):
        model_path = os.path.join(os.getcwd(),self.config.model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.classifier = torch.load(os.path.join(model_path, f'{name}.pt'))

    def normalize_prob_vector(self, v):
        sum_v = sum(v)
        return [element/sum_v for element in v]
    
    def scale_q_vector(self, q, label):
        # label = label // 2 # height
        # sigma = 0.1
        # mean = label * (self.num_labels - 1) // self.config.trajectory_length
        # for index, element in enumerate(q):  # index//2 + 1 (plus one because we do a shift)
        #     q[index] = element * norm.pdf(index//2+1,mean,sigma)
        return self.normalize_prob_vector(q)

    def q_probability_vector(self, y, data_label):

        d = self.num_labels - 2  # denominator

        Q_ = list()
        for i, y_single_prob in enumerate(y):

            if data_label[i] == 1 or data_label[i] == 0:
                q_ = [0]*self.num_labels
                q_[data_label[i]] = 1
                Q_ += [q_]
            else:

                if data_label[i] != -1:
                    q_ = y_single_prob.tolist()
                    last = q_.pop()
                    for index in range(len(q_)):
                        if index%2 != data_label[i]%2:
                            q_[index] = 0
                    q_.append(1.5*last)
                    q_ = self.scale_q_vector(q_, data_label[i])
                
                else:
                    p_separatrix = sum(y_single_prob[self.num_labels-3:self.num_labels])
                    p_correction = 1/3 * p_separatrix / d
                    q_ = [0,0] + (y_single_prob[0:self.num_labels-3] + p_correction).tolist() 
                    q_ += [2*p_separatrix/3 + p_correction]


                # p_separatrix = sum(y_single_prob[self.num_labels-3:self.num_labels])
                # p_correction = 1/3 * p_separatrix / d
                # q_ = [0,0] + (y_single_prob[0:self.num_labels-3] + p_correction).tolist() 
                # q_ += [2*p_separatrix/3 + p_correction]

                # if data_label[i] != -1:
                #     last = q_.pop()
                #     for index in range(len(q_)):
                #         if index%2 != data_label[i]%2:
                #             q_[index] = 0
                #     q_.append(last)
                #     sum_q = sum(q_)
                #     q_=[element/sum_q for element in q_]
                    
                Q_ += [q_]

                            


            # if data_label[i] != 1 and data_label[i] != 0:
                
            #     p_separatrix = sum(y_single_prob[self.num_labels-3:self.num_labels])
            #     p_correction = 1/3 * p_separatrix / d
            #     q_ = [0,0] + (y_single_prob[0:self.num_labels-3] + p_correction).tolist() 
            #     q_ += [2*p_separatrix/3 + p_correction]
            #     Q_ += [q_]

            # # if data_label[i] == -1:
            # #     p_separatrix = y_single_prob[self.num_labels-1]/d
            # #     q_ = [0,0] + (y_single_prob[0:self.num_labels-3] + p_separatrix).tolist() 
            # #     q_ += [sum(y_single_prob[self.num_labels-3:self.num_labels-1]) + p_separatrix]
            # #     Q_ += [q_]

            # # if data_label[i] == -1:
            # #     q_ = [0,0] + y_single_prob[0:self.num_labels-3].tolist() 
            # #     q_ += [sum(y_single_prob[self.num_labels-3:self.num_labels]).tolist()]
            # #     Q_ += [q_]

            # else:
            #     q_ = [0]*self.num_labels
            #     q_[data_label[i]] = 1
            #     Q_ += [q_]
        
        return torch.Tensor(Q_)

    def loss_function(self, forward_dict):
        probs_x = forward_dict['probs_x']
        probs_xnext = forward_dict['probs_xnext']
        labels_xnext = forward_dict['labels_xnext']
        return torch.nn.CrossEntropyLoss()(probs_x, self.q_probability_vector(probs_xnext,labels_xnext))
    
    def get_optimizer(self, list_parameters):
        if self.config.optimizer == 'Adam':
            return torch.optim.Adam(list_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'SGD':
            return torch.optim.SGD(list_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'Adagrad':
            return torch.optim.Adagrad(list_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'AdamW':
            return torch.optim.AdamW(list_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'RMSprop':
            return torch.optim.RMSprop(list_parameters, lr=self.config.learning_rate) # not compatible with cyclic LR
        elif self.config.optimizer == 'Adadelta':
            return torch.optim.Adadelta(list_parameters, lr=self.config.learning_rate) # not compatible with cyclic LR

    def get_scheduler(self, optimizer):
        if self.config.scheduler == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.config.learning_rate, max_lr=0.1)
        elif self.config.scheduler == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.config.patience, verbose=self.config.verbose)

    def train(self):
        epochs = self.config.epochs
        patience = self.config.patience
        analyze_td = self.config.analyze_train_dynamics
        list_parameters = list(self.classifier.parameters())
        optimizer = self.get_optimizer(list_parameters)
        scheduler = self.get_scheduler(optimizer)
        #optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=self.verbose)

        if analyze_td:
            self.class_order_data.append(find_class_order(self.classifier, self.config))

        for epoch in tqdm(range(epochs)):
            epoch_train_loss = 0

            self.classifier.train()

            # loop over the batches in the train_loadaer 
            for i, (x, xnext, xnext_label) in enumerate(self.train_loader):
                
                optimizer.zero_grad()

                forward_dict = self.classifier(x.to(self.device), xnext.to(self.device))
                forward_dict['labels_xnext'] = xnext_label

                loss = self.loss_function(forward_dict)

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                
                if analyze_td:
                    self.class_order_data.append(find_class_order(self.classifier, self.config))
            
            epoch_train_loss /= len(self.train_loader)
            self.train_losses['loss_total'].append(epoch_train_loss)

            epoch_test_loss = 0

            self.classifier.eval()

            with torch.no_grad():
                for i, (x, xnext, xnext_label) in enumerate(self.test_loader):
                    forward_dict = self.classifier(x.to(self.device), xnext.to(self.device))
                    forward_dict['labels_xnext'] = xnext_label
                    loss = self.loss_function(forward_dict)
                    epoch_test_loss += loss.item()
                
                epoch_test_loss /= len(self.test_loader)
                self.test_losses['loss_total'].append(epoch_test_loss)

            if self.config.scheduler == 'ReduceLROnPlateau':
                scheduler.step(epoch_test_loss)
            else:
                scheduler.step()

            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    return epoch_test_loss
            
            if self.config.verbose:
                print(f"Epoch: {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}")

        #return self.train_losses['loss_total'], self.test_losses['loss_total']
