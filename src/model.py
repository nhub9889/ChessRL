import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self, input_channels, actions, res_blocks=20, filters=512):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(filters)

        self.res_blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters)
        ) for _ in range(res_blocks)])

        self.conv_policy = nn.Conv2d(filters, 4, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(4)
        self.fc_policy1 = nn.Linear(4 * 8 * 8, 512)
        self.fc_policy2 = nn.Linear(512, actions)

        # Value head - tăng kích thước
        self.conv_value = nn.Conv2d(filters, 2, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(2)
        self.fc_value1 = nn.Linear(2 * 8 * 8, 512)
        self.fc_value2 = nn.Linear(512, 256)
        self.fc_value3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.bn_input(self.conv_input(x)))

        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x += residual
            x = torch.relu(x)

        policy = torch.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(policy.size(0), -1)
        policy = torch.relu(self.fc_policy1(policy))
        policy = self.fc_policy2(policy)
        policy = torch.softmax(policy, dim=1)

        value = torch.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.fc_value1(value))
        value = torch.relu(self.fc_value2(value))
        value = torch.tanh(self.fc_value3(value))
        return policy, value

class Model:
    def __init__(self, input_channels, actions, device='cuda', lr=1e-3, weight_decay=1e-4):
        self.device = torch.device(device)
        self.net = Net(input_channels, actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        self.accumulation_steps = 2
        self.optimizer_step = 0

    def predict(self, state):
        if not isinstance(state, torch.Tensor):
            state_tensor = self.state_to_tensor(state)
        else:
            state_tensor = state

        with torch.no_grad():
            policy, value = self.net(state_tensor)

        return policy.cpu().numpy()[0], value.cpu().numpy()[0][0]

    def state_to_tensor(self, chess_state):
            board_tensor = np.zeros((18, 8, 8), dtype=np.float32)
            piece_channels = {
                'pawn': 0, 'knight': 1, 'bishop': 2, 'rook': 3, 'queen': 4, 'king': 5
            }

            for x in range(8):
                for y in range(8):
                    piece = chess_state.board.getPiece(x, y)
                    if piece:
                        piece_type = str(piece.__class__.__name__).lower()
                        color_idx = 0 if piece.color == 'W' else 1
                        if piece_type in piece_channels:
                            channel = piece_channels[piece_type] + color_idx * 6
                            board_tensor[channel, x, y] = 1

            board_tensor[12, :, :] = 1 if chess_state.board.curPlayer == 'W' else 0
            return torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)

    def batch_predict(self, states):
            if not states:
                return [], []
            state_tensors = []
            for state in states:
                if isinstance(state, torch.Tensor):
                    state_tensors.append(state)
                else:
                    state_tensor = self.state_to_tensor(state)
                    state_tensors.append(state_tensor)
            state_batch = torch.cat(state_tensors, dim=0).to(self.device)

            with torch.no_grad():
                policies, values = self.net(state_batch)

            return policies.cpu().numpy(), values.cpu().numpy().flatten()

    def train(self, states, target_policies, target_values):
        state_tensors = []
        for state in states:
            if not isinstance(state, torch.Tensor):
                state_tensor = self.state_to_tensor(state)
            else:
                state_tensor = state
            state_tensors.append(state_tensor)

        state_tensors = torch.cat(state_tensors, dim=0)
        state_tensors = state_tensors.to(self.device, non_blocking=True)

        if not isinstance(target_policies[0], torch.Tensor):
            target_policies = torch.tensor(np.array(target_policies), dtype=torch.float32)
        else:
            target_policies = torch.stack(target_policies)
        target_policies = target_policies.to(self.device, non_blocking=True)

        if not isinstance(target_values[0], torch.Tensor):
            target_values = torch.tensor(target_values, dtype=torch.float32)
        else:
            target_values = torch.stack(target_values)
        target_values = target_values.to(self.device, non_blocking=True)

        # Mixed precision training
        with torch.cuda.amp.autocast():
            policies, values = self.net(state_tensors)
            value_loss = torch.mean((target_values - values.squeeze()) ** 2)
            policy_loss = -torch.mean(torch.sum(target_policies * torch.log(policies + 1e-8), dim=1))
            total_loss = (value_loss + policy_loss) / self.accumulation_steps

        # Gradient accumulation
        self.scaler.scale(total_loss).backward()

        if (self.optimizer_step + 1) % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.optimizer_step = (self.optimizer_step + 1) % self.accumulation_steps

        return total_loss.item() * self.accumulation_steps

    def save(self, path):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path, device= 'cuda'):
        if device == 'cuda':
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])