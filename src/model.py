import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self, input_channels, actions, res_blocks= 10, filters= 256):
        super(Net, self).__init__()

        #Initial convolution
        self.conv_input = nn.Conv2d(input_channels, filters, kernel_size= 3, padding= 1)
        self.bn_input = nn.BatchNorm2d(filters)

        #Residual blocks
        self.res_blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(filters)
        ) for _ in range(res_blocks)
        ])

        #Policy head
        self.conv_policy = nn.Conv2d(filters, 2, kernel_size= 1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2*8*8, actions)

        #Value head
        self.conv_value = nn.Conv2d(filters, 1, kernel_size= 1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1*8*8, 256)
        self.fc_value2 = nn.Linear(256, 1)

        def forward(self, x):
            x = torch.relu(self.bn_input(self.conv_input(x)))

            #Residual blocks
            for res_block in self.res_blocks:
                residual = x
                x = res_block(x)
                x += residual
                x = torch.relu(x)

            #Policy head
            policy = torch.relu(self.bn_policy(self.conv_poluicy(x)))
            policy = policy.view(policy.size(0), -1)
            policy = self.fc_policy(policy)
            policy = torch.sofmax(policy, dim= 1)

            #value head
            value = torch.relu(self.bn_value(self.conv_value(x)))
            value = value.view(value.size(0), -1)
            value = torch.relu(self.fc_value1(value))
            value = torch.relu(self.fc_value2(value))

            return policy, value

class Model:
    def __init__(self, input_channels, actions, device= 'gpu', lr= 0.01, weight_decay= 1e-4):
        self.device = device
        self.net = Net(input_channels, actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr= lr, weight_decay= weight_decay)

    def predict(self, state):
        state_tensor = self._state_to_tensor(state).to(self.device)
        with torch.no_grad():
            policy, value = self.net(state_tensor)

        return policy.cpu().numpy()[0], value.cpu().numpy()[0][0]

    def _state_to_tensor(self, state):
        board_tensor = np.zeros((18, 8, 8), dtype= np.float32)
        piece_channels = {
            'pawn': 0, 'knight': 1, 'bishop': 2, 'rook': 3, 'queen': 4, 'king': 5
        }
        for x in range(8):
            for y in range(8):
                piece = state.board.getPiece(x, y)
                if piece:
                    type = str(piece.__class__.__name__).lower()
                    color = 0 if piece.color == 'W' else 1
                    channel = piece_channels[type] + color*6
                    board_tensor[channel, x, y] = 1
        board_tensor[12, :, :] = 1 if state.board.curPlayer == 'W' else 0
        return torch.from_numpy(board_tensor).unsqueeze(0)
    def train(self, states, target_policies, target_values):
        states= torch.stack([self._state_to_tensor(s) for s in states]).to(self.device)
        target_policies = torch.tensor(target_policies).to(self.device)
        target_values = torch.tensor(target_values).float().to(self.device)

        policies, values = self.net(states)
        policy_probs = torch.softmax(policies, dim=1)

        #Calculate losses
        value_loss = torch.mean((target_values - values.squeeze()) ** 2)
        policy_loss = -torch.mean(torch.sum(target_policies * torch.log(policy_probs + 1e-10), dim= 1))
        regularization_loss = torch.sum(torch.tensor([torch.sum(p ** 2) for p in self.net.parameters()]))
        total = value_loss + policy_loss + 1e-4*regularization_loss

        #Backward pass
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        return total.item()

    def save(self, path):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

