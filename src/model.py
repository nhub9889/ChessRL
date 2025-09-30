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
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net = Net(input_channels, actions).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.accumulation_steps = 2
        self.optimizer_step = 0

    def predict(self, state):
        if not isinstance(state, torch.Tensor):
            state_tensor = self.state_to_tensor(state)
        else:
            state_tensor = state.to(self.device)

        with torch.no_grad():
            policy_logits, value = self.net(state_tensor)
            policy = torch.softmax(policy_logits, dim=1)

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
        state_tensors = [self.state_to_tensor(s) if not isinstance(s, torch.Tensor) else s.to(self.device) for s in states]
        state_batch = torch.cat(state_tensors, dim=0)

        with torch.no_grad():
            policy_logits, values = self.net(state_batch)
            policies = torch.softmax(policy_logits, dim=1)

        return policies.cpu().numpy(), values.cpu().numpy().flatten()

    def _prepare_targets(self, target_policies, target_values):
        # Policies → đảm bảo tensor 2D float32
        if isinstance(target_policies[0], torch.Tensor):
            target_policies = torch.stack(target_policies, dim=0)
        else:
            target_policies = torch.tensor(np.stack(target_policies, axis=0), dtype=torch.float32)
        target_policies = target_policies.to(self.device, non_blocking=True)

        # Values → đảm bảo tensor 1D float32
        if isinstance(target_values[0], torch.Tensor):
            target_values = torch.stack(target_values).float()
        else:
            target_values = torch.tensor(np.array(target_values), dtype=torch.float32)
        target_values = target_values.to(self.device, non_blocking=True)

        return target_policies, target_values

    def train(self, states, target_policies, target_values):
        # Chuẩn hoá target trước khi đưa vào train gốc
        target_policies, target_values = self._prepare_targets(target_policies, target_values)
        return self._train_impl(states, target_policies, target_values)

    def _train_impl(self, states, target_policies, target_values):
        # Nếu đã là tensor batch sẵn (supervised)
        if isinstance(states, torch.Tensor) and states.ndim == 4:
            states_tensors = states.to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            # Tạo batch từ list (reinforcement/self-play)
            states_tensors = []
            for state in states:
                if not isinstance(state, torch.Tensor):
                    state_tensor = self.state_to_tensor(state)  # (1,18,8,8)
                else:
                    # nếu là (18,8,8) thì thêm batch dim
                    state_tensor = state.unsqueeze(0) if state.ndim == 3 else state
                states_tensors.append(state_tensor)
            states_tensors = torch.cat(states_tensors, dim=0).to(self.device, dtype=torch.float32, non_blocking=True)

        target_policies = target_policies.to(self.device, dtype=torch.float32)
        target_values = target_values.to(self.device, dtype=torch.float32)

        with torch.cuda.amp.autocast(True):
            policies, values = self.net(states_tensors)
            value_loss = torch.mean((target_values - values.squeeze()) ** 2)
            policy_loss = -torch.mean(torch.sum(target_policies * torch.log(policies + 1e-8), dim=1))
            total_loss = (value_loss + policy_loss) / self.accumulation_steps

        self.scaler.scale(total_loss).backward()

        self.optimizer_step += 1
        if self.optimizer_step % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return total_loss.item() * self.accumulation_steps


    def save(self, path):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path, device='cuda'):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
