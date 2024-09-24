import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from systems.pendulum import Pendulum
from matplotlib import pyplot as plt
from tqdm import tqdm

torch.seed = 42


def create_batch_around_attractor(attractor, radius, num_samples):
    samples = np.random.normal(0, 0.1, (num_samples, 2))
    samples = samples / np.linalg.norm(samples, axis=1).reshape(-1, 1)
    samples = samples * radius
    samples = samples + attractor
    return samples

def process_data(data):
    d = np.zeros((data.shape[0], 5))

    d[:, 0] = data[:, 0]/np.pi
    d[:, 1] = data[:, 1]/(2*np.pi)
    d[:, 2] = data[:, 2]/np.pi
    d[:, 3] = data[:, 3]/(2*np.pi)
    d[:, 4] = data[:, 4]
    return d

class PendulumDataset(Dataset):
    def __init__(self, data):
        
        self.data = process_data(data)
        # process_data(self.data)
        # self.data[:, 0] = self.data[:, 0] / np.pi
        # self.data[:, 1] = self.data[:, 1] / (2*np.pi)
        # self.data[:, 2] = self.data[:, 2] / np.pi
        # self.data[:, 3] = self.data[:, 3] / (2*np.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return torch.tensor(self.data[idx]).float()

class PendulumNet(nn.Module):
    def __init__(self):
        super(PendulumNet, self).__init__()
        width = 256
        self.dropout1 = nn.Dropout(0.2)  
        self.dropout2 = nn.Dropout(0.2)  
        self.dropout3 = nn.Dropout(0.2)  
        self.fc1 = nn.Linear(4, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        # x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    

def train_classifier(net, criterion, optimizer, train_loader, val_loader, device='cpu', epochs=10):
    
    for epoch in range(epochs):
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data[:, :4], data[:, 4]
            # print(inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            # print(inputs)
            outputs = net(inputs)
            # print(outputs)
            # exit()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        net.eval()
        val_loss = 0.0
        for i, data in enumerate(val_loader):
            inputs, labels = data[:, :4], data[:, 4]
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {val_loss/len(val_loader)}")
    return net

def main(args, kwargs):
    # load data

    data = np.loadtxt(f"{kwargs['save_dir']}/dataset_{kwargs['dataset_size']}.csv", delimiter=',')
    np.random.shuffle(data)
    data = data[:]
    train_data = data[:int(0.8*len(data))]
    val_data = data[int(0.8*len(data)):]

    train_dataset = PendulumDataset(train_data)
    val_dataset = PendulumDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kwargs['batch_size'], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PendulumNet()
    net = net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    net = train_classifier(net, criterion, optimizer, train_loader, val_loader, device=device, epochs=kwargs['epochs'])

    torch.save(net.state_dict(), f"{kwargs['save_dir']}/pendulum_{kwargs['dataset_size']}_classifier.pth")


def plot_clf_results(kwargs):
    
    # discretize the state space
    x = np.linspace(-3.14, 3.14, 200) 
    y = np.linspace(-6.28, 6.28, 400)

    system = Pendulum()
    attractors = system.attractors()
    clf = PendulumNet()
    clf = clf.to('cuda')
    clf.load_state_dict(torch.load(f"{kwargs['save_dir']}/pendulum_{kwargs['dataset_size']}_classifier.pth"))
    clf.eval()

    plotting_data = []
    with torch.no_grad():
        for i in tqdm(x):
            for j in y:
                scores = []
                for att in attractors:
                    # print(i, j)
                    # state = np.hstack([np.array([i / np.pi, j/(2*np.pi)]), att[0]/np.pi, att[1]/(2*np.pi), 0])
                    samples = create_batch_around_attractor(att, 0.05, 32)
                    # print(samples, att)
                    batch = np.zeros((samples.shape[0]+1, 5))
                    batch[:, 0] = i
                    batch[:, 1] = j
                    batch[:-1, 2] = samples[:, 0]
                    batch[:-1, 3] = samples[:, 1]
                    batch[-1, 2] = att[0]
                    batch[-1, 3] = att[1]

                    state = batch # np.array([[i, j, att[0], att[1], 0]])
                    state = process_data(state)
                    # print
                    state = torch.tensor(state[:, :-1]).float()
                    state = state.to('cuda')
                    scores.append((np.sum(torch.sigmoid(clf(state)).cpu().numpy() > 0.5) > 0) * 1.0)

                scores = np.array(scores)
                final_class = np.argmax(scores)
                
                if scores.sum() == 0:
                    final_class = 3
                if scores.sum() >= 2:
                    if scores[0] == 1:
                        final_class = 3
                    # else:
                    #     final_class = 2
                plotting_data.append([i, j, final_class])
    
    colors = ['r', 'g', 'b', 'yellow']
    labels = ['Success', 'Failure 1', 'Failure 2', 'Separatrix']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    plotting_data = np.array(plotting_data)
    for i in range(4):
        if plotting_data[plotting_data[:, 2] == i].shape[0] > 0:
            ax.scatter(*zip(*[(x[0], x[1]) for x in plotting_data if x[2] == i]), color=colors[i], s=1, label=labels[i])
    ax.set_xlabel("Theta")
    ax.set_ylabel("Theta_dot")
    ax.set_title(f"{kwargs['dataset_size']} Classification")
    ax.legend()
    fig.savefig(f"clf_{kwargs['dataset_size']}_256.png")

if __name__ == "__main__":

    train = True
    kwargs = dict()
    kwargs['dataset_size'] = '5k'
    kwargs['batch_size'] = 8
    kwargs['epochs'] = 20
    kwargs['cwd'] = f"/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/arcmg"
    kwargs['save_dir'] = f"{kwargs['cwd']}/data/pendulum/clf_{kwargs['dataset_size']}"
    if train:
        main([], kwargs)
    plot_clf_results(kwargs)