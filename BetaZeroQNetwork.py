import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

CHANNELS = 512


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(CHANNELS, CHANNELS, 3, stride=1)
        self.conv4 = nn.Conv2d(CHANNELS, CHANNELS, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(CHANNELS)
        self.bn2 = nn.BatchNorm2d(CHANNELS)
        self.bn3 = nn.BatchNorm2d(CHANNELS)
        self.bn4 = nn.BatchNorm2d(CHANNELS)

        fc_input_dims = self.calculate_input_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 1024)

        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Actions
        self.A = nn.Linear(512, n_actions)

        # Value
        self.V = nn.Linear(512, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_input_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.bn1(self.conv1(state))
        dims = self.bn2(self.conv2(dims))
        dims = self.bn3(self.conv3(dims))
        dims = self.bn4(self.conv4(dims))

        return int(np.prod(dims.size()))

    def forward(self, state):

        s = F.relu(self.bn1(self.conv1(state)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))

        conv_state = s.view(s.size()[0], -1)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(conv_state))), p=0.3)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=0.3)

        v = self.V(s)
        a = self.A(s)

        return v, a

    def save_checkpoint(self):
        print("Saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
