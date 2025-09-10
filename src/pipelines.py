from MCTS import MCTS
import numpy as np
import random
import torch
from UI.pieces import Board, Pawn, Queen
from collections import deque
import torch.nn as nn
import torch.optim as optim
import io
import chess.pgn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
from datetime import datetime
import csv
import os
import math

class ChessState:
    def __init__(self, board):
        self.board = board
        self.current_player = board.curPlayer

    def legal_moves(self):
        legal_moves = []
        for x in range(8):
            for y in range(8):
                piece = self.board.getPiece(x, y)
                if piece and piece.color == self.current_player:
                    moves = self.board.validMoves(x, y)
                    for move in moves:
                        legal_moves.append(((x, y), move))
        return legal_moves


    def isTerminal(self):
        return self.board.checkmate('W') or self.board.checkmate('B') or \
                self.board.stalemate('W') or self.board.stalemate('B')
    def get_reward(self):
        if self.board.checkmake('W'):
            return 1.0 if self.current_player == 'B' else -1.0
        elif self.board.checkmate('B'):
            return 1.0 if self.current_player == 'W' else -1.0
        else:
            return 0.0

    def apply(self, action):
        new_board = self.board.copy()
        from_pos, to_pos = action
        new_board.Move(from_pos, to_pos)

        piece = new_board.getPiece(to_pos[0], to_pos[1])
        if isinstance(piece, Pawn) and (to_pos[1] == 0 or to_pos[1] == 7):
            new_board.promote(to_pos[0], to_pos[1], Queen)

        return ChessState(new_board)

class Visualizer:
    def __init__(self, dir = "training_log"):
        self.dir = dir
        os.makedirs(dir, exist_ok= True)
        self.metrics = {
            'iteration': [],
            'supervised_loss': [],
            'reinforcement_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'eval_wins': [],
            'eval_losses': [],
            'game_length': [],
            'exploration_rate': [],
            'timestamp': []
        }

        self.csv = os.path.join(dir, 'metrics.csv')
        with open(self.csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.metrics.keys())

        plt.ion()
        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Training progress', fontsize= 16)

        def update_plots(self):
            for ax in self.axs.flat:
                ax.clear()
            valid_indices = [i for i, loss in enumerate(self.metrics['reinforcement_loss']) if loss is not None]
            if valid_indices:
                iterations = [self.metrics['iteration'][i] for i in valid_indices]

                # Loss curves
                if any(loss is not None for loss in self.metrics['reinforcement_loss']):
                    self.axs[0, 0].plot(iterations, [self.metrics['reinforcement_loss'][i] for i in valid_indices],'b-',
                                        label= 'Reinforcement Loss')
                if any(loss is not None for loss in self.metrics['supervised_loss']):
                    self.axs[0, 0].plot(iterations, [self.metrics['supervised_loss'][i] for i in valid_indices],'b-',
                                        label= 'Supervised Loss')
                if any(loss is not None for loss in self.metrics['value_loss']):
                    self.axs[0, 0].plot(iterations, [self.metrics['value_loss'][i] for i in valid_indices], '-g', label= 'Value loss')
                if any(loss is not None for loss in self.metrics['policy_loss']):
                    self.axs[0, 0].plot(iterations, [self.metrics['policy_loss'][i] for i in valid_indices], '-m', label= 'Policy Loss')
                self.axs[0, 0].set_title('Training loss')
                self.axz[0, 0].set_xlabel('Iterations')
                self.axz[0, 0].set_ylabel('Loss')
                self.axs[0, 0].legend()
                self.axs[0, 0].grid(True)

                # Evaluation results
                eval_indices = [i for i, wins in enumerate(self.metrics['eval_wins']) if wins is not None]
                if eval_indices:
                    eval_iterations = [self.metrics['iteration'][i] for i in eval_indices]
                    wins = [self.metrics['eval_wins'][i] for i in eval_indices]
                    losses = [self.metrics['eval_losses'][i] for i in eval_indices]
                    draws = [self.metrics['eval_losses'][i] for i in eval_indices]

                    self.axs[0, 1].plot(eval_iterations, wins, '-g', label= 'Wins')
                    self.axs[0, 1].plot(eval_iterations, draws, '-h', labels= 'Draws')
                    self.axs[0, 1].plot(eval_iterations, losses, '-r', labels= 'Losses')
                    self.axs[0, 1].set_title('Evaluation loss')
                    self.axs[0, 1].set_xlabel('Iterations')
                    self.axs[0, 1].set_ylabel('Games')
                    self.axs[0, 1].legend()
                    self.axs[0, 1].grid(True)

                    # Plot win rate
                    total_games = [w + d + l for w, d, l in zip(wins, draws, losses)]
                    win_rates = [w/t if t > 0 else 0 for w, t in zip(wins, total_games)]
                    self.axs[0, 2].plot(eval_iterations, win_rates, 'purple', label= 'Win Rate')
                    self.axs[0, 2].set_title('Win rate over time')
                    self.axs[0, 2].set_xlabel('Iteration')
                    self.axs[0, 2].set_ylabel('Win Rate')
                    self.axs[0, 2].set_ylim(0, 1)
                    self.axs[0, 2].legend()
                    self.axs[0, 2].grid(True)

                    # Elo rating estimate
                    elo_ratings = []
                    baseline_elo = 1200
                    for i in eval_indices:
                        wins = self.metrics['eval_wins'][i]
                        losses = self.metrics['eval_losses'][i]
                        draws = self.metrics['eval_draws'][i]
                        total_games = wins + losses + draws

                        if total_games > 0:
                            win_rate = wins / total_games
                            draw_rate = draws / total_games

                            expected_score = win_rate + 0.5 * draw_rate
                            elo_difference = 400 * math.log10(
                                expected_score / (1 - expected_score)) if expected_score < 1 else 400
                            elo_ratings.append(baseline_elo + elo_difference)
                        else:
                            elo_ratings.append(baseline_elo)
                    self.axs[1, 2].plot(eval_iterations, elo_ratings, 'red')
                    self.axs[1, 2].set_title('Estimated Elo ratings')
                    self.axs[1, 2].set_xlabel('Iteration')
                    self.axs[1, 2].set_ylabel('Elo')
                    self.axs[1, 2].grid(True)

                # Game length
                game_length = [i for i, length in enumerate(self.metrics['game_length']) if length is not None]
                if game_length:
                    game_iterations = [self.metrics['iteration'][i] for i in game_length]
                    game_lengths = [self.metrics['game_length'][i] for i in game_length]
                    self.axs[1, 0].plot(game_iterations, game_lengths, 'orange')
                    self.axs[1, 0].set_title('Average game length')
                    self.axs[1, 0].set_xlabel('Iterations')
                    self.axs[1, 0].set_ylabel('Moves')
                    self.axs[1, 0].grid(True)

                # Exploration rate
                exploration_indices = [i for i, rate in enumerate(self.metrics['exploration_rate']) if rate is not None]
                if exploration_indices:
                    exploration_iterations = [self.metrics['iteration'][i] for i in exploration_indices]
                    exploration_rates = [self.metrics['exploration_rate'][i] for i in exploration_indices]
                    self.axs[1, 1].plot(exploration_iterations, exploration_rates, 'brown')
                    self.axs[1, 1].set_title('Exploration rate')
                    self.axs[1, 1].set_xlabel('Iteration')
                    self.axs[1, 1].set_ylabel('Temperature')
                    self.axs[1, 1].grid(True)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        def save_plots(self):
            path = os.path.join(self.dir, f"training_plots_{datetime.now().strftime('%Y%m%d-%H%M')}.png")
            self.fig.savefig(path)
            print(f"Plots saved to {path}")

        def generate_report(self):
            path = os.path.join(self.dir, f"training_reports_{datetime.now().strftime('%Y%m%d-%H')}.txt")
            with open(path, 'w') as file:
                file.write("Training report \n")
                file.write("==============================\n\n")
                file.write(f"Generated on: {datetime.now().strftime('%Y%m%d-%H')}\n \n")
                file.write("Training statistics: \n")
                file.write(f"Total iterations: {len(self.metrics['iteration'])} \n")

                # Loss statistics
                valid_losses = [loss for loss in self.metrics['reinforcement_losses'] if loss is not None]
                if valid_losses:
                    file.write(f"Final reinforcement loss: {valid_losses[-1]:.4f} \n")
                    file.write(f"Minimum RL loss: {min(valid_losses):.4f} \n")

                # Evaluation statistics
                eval_wins = [w for w in self.metrics['eval_wins'] if w is not None]
                if eval_wins:
                    total_wins = sum(eval_wins)
                    total_draws = sum([d for d in self.metrics['eval_draws'] if d is not None])
                    total_losses = sum([l for l in self.metrics['eval_losses'] if l is not None])
                    total_games = total_wins + total_draws + total_losses

                    file.write(f"\nEvaluations results: \n")
                    file.write(f"Total games: {total_games} \n")
                    file.write(f"Wins: {total_wins} ({total_wins/total_games*100:.1f}%)\n")
                    file.write(f"Draws: {total_draws} ({total_draws/total_games*100:.1f}\n")
                    file.write(f"Losses: {total_losses} ({total_losses/total_games*100:.1f}\n")

                if len(self.metrics['timestamp']) > 1:
                    start_time = datetime.strptime(self.metrics['timestamp'][0], "%Y-%m-%d %H:%M:%S.%f")
                    end_time = datetime.strptime(self.metrics['timestamp'][-1], "%Y-%m-%d %H:%M:%S.%f")
                    duration = end_time - start_time

                    file.write(f"\n Training duration: {duration} \n")

            print(f"Report saved to {path}")


        def update_metrics(self, iteration, supervised_loss= None, reinforcement_loss= None,
                           value_loss= None, policy_loss= None, eval_results= None,
                           game_length= None, exploration_rate= None):
            self.metrics['iteration'].append(iteration)
            self.metrics['supervised_loss'].append(supervised_loss)
            self.metrics['reinforcement_loss'].append(reinforcement_loss)
            self.metrics['value_loss'].append(value_loss)
            self.metrics['policy_loss'].append(policy_loss)
            self.metrics['game_length'].append(game_length)
            self.metrics['exploration_rate'].append(exploration_rate)
            self.metrics['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            if eval_results:
                self.metrics['eval_wins'].append(eval_results['wins'])
                self.metrics['eval_losses'].append(eval_results['losses'])
                self.metrics['eval_draws'].append(eval_results['losses'])
            else:
                self.metrics['eval_wins'].append(None)
                self.metrics['eval_draws'].append(None)
                self.metrics['eval_losses'].append(None)

            with open(self.csv, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([iteration, supervised_loss, reinforcement_loss, value_loss, policy_loss,
                                    eval_results['wins'] if eval_results else None, eval_results['losses'] if eval_results else None,
                                    eval_results['draws'] if eval_results else None, game_length, exploration_rate,
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            self.update_plots()
class SelfPlay:
    def __init__(self, model, simulations= 800, temperature= 1.0, max_moves= 512):
        self.model = model
        self.mcts = MCTS(model, simulations)
        self.temperature = temperature
        self.max_moves = max_moves

    def _action_to_idx(self, action):
        from_square, to_square = action
        from_idx = from_square[0]*8 + from_square[1]
        to_idx = to_square[0]*8 + to_square[1]
        return from_idx*64 + to_idx
    def play(self):
        state = ChessState(Board())
        states = []
        mcts_policies = []
        currentPlayer = state.board.curPlayer
        count = 0

        while not state.isTerminal() and count < self.max_moves:
            action_probs, _ = self.mcts.run(state)

            if self.temperature > 0:
                visits = np.array([count for count in action_probs.values()])
                probs = visits**(1/self.temperature)
                probs /= probs.sum()
            else:
                probs = np.array([prob for prob in action_probs.values()])
                probs /= probs.sum()

            states.append(state)
            vector_policy = np.zeros(64*64)
            for (from_pos, to_pos), prob in action_probs.items():
                idx = from_pos[0]*8*64 + from_pos[1]*63 + to_pos[0]*8 + to_pos[1]
                vector_policy[idx] = prob
            mcts_policies.append(vector_policy)

            actions = list(action_probs.keys())
            action = random.choices(actions, weights= probs)[0]

            state = state.apply_action(action)

            count += 1

        outcome = state.get_reward()
        training = []
        for i, (state, policy) in enumerate(zip(states, mcts_policies)):
            player_outcome = outcome if (i % 2 == 0 and currentPlayer == 'W') or (i % 2 == 1 and currentPlayer == 'B') else  -outcome
            training.append((state, policy, player_outcome))

        return training

def LoadPGN(path, num_games= 100000):
    games = []
    with open(path) as pgn:
        for _ in range(num_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games

def board_to_state(board):
    state = np.zeros((18, 8, 8), dtype= np.float32)
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            channel_offset = 0 if piece.color == chess.WHITE else 6
            channel = piece_to_channel[piece.piece_type] + channel_offset
            state[channel, 7 - row, col] = 1
    state[12, :, :] = 1 if board.turn == chess.WHITE else 0
    return state
def ProcessPGN(games):
    states = []
    policies = []
    values = []

    for game in games:
        board = game.board()
        res = game.headers.get("Result", "1/2-1/2")

        if res == "1-0":
            outcome = 1.0
        elif res == "0-1":
            outcome = -1.0
        else:
            outcome = 0.0
        for node in game.mainlines():
            board.push(node.move)
            state = board_to_state(board)
            states.append(state)

            move = node.move
            from_square = move.from_square
            to_square = move.to_square

            policy_vector = np.zeros(64*64)
            idx = from_square*64 + to_square
            policy_vector[idx] = 1.0
            policies.append(policy_vector)
            values.append(outcome if board.turn == chess.WHITE else -outcome)
        return states, policies,values

class TrainingPipeline:
    def __init__(self, model, num_workers= 4, games_per_iterations= 100, iterations= 1000, buffer= 100000, batch_size= 1024,
                 supervised_epochs= 10, supervised_batch_size= 512):
        self.model = model
        self.num_workers = num_workers
        self.games_per_iterations = games_per_iterations
        self.buffer = buffer
        self.batch_size = batch_size
        self.supervised_epochs = supervised_epochs
        self.supervised_batch_size = supervised_batch_size
        self.iterations = iterations
        self.replay = deque(maxlen=buffer)
        self.workers = [SelfPlay(model) for _ in range(num_workers)]

        self.iteration = 0
        self.best_win_rate = 0
        self.best_model_path = None

        self.visualizer = Visualizer()

    def evaluate(self):
        wins = 0
        draws = 0
        losses = 0
        for _ in range(10):
            state = ChessState(Board())
            count = 0
            while not state.isTerminal() and count < 100:
                if state.board.curPlayer == 'B':
                    action_probs = self.model.mcts.run(state)
                    action = max(action_probs, key = action_probs.get)

                else:
                    legal = state.legal_moves()
                    action = random.choice(legal) if legal else None

                if action:
                    state = state.apply(action)
                    count += 1
                else:
                    break
            outcome = state.get_reward()
            if outcome == 1:
                wins += 1
            elif outcome == 0:
                draws = 1
            else:
                losses += 1

        print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")

    def supervised_train(self, path, num_games= 10000):
        print("Loading supervised data...")
        games = LoadPGN(path, num_games)
        print(f"Load {len(games)} games")
        print("Processing supervised data...")
        states, policies, values = ProcessPGN(games)
        print(f"Processing {len(states)} positions")
        states_tensor = torch.stack([torch.from_numpy(s).unsqueeze(0) for s in states])
        policies_tensor = torch.tensor(policies)
        values_tensor = torch.tensor(values).float()

        dataset = torch.utils.data.TensorDataset(states_tensor, policies_tensor, values_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size= self.supervised_batch_size, shuffle= True)

        print("Starting supervised training...")
        for epoch in range(self.supervised_epochs):
            total = 0.0
            epoch_loss = 0.0
            num_batches = 0
            for idx, (batch_states, batch_policies, batch_values) in enumerate(dataloader):
                batch_states = batch_states.to(self.model.device)
                batch_policies = batch_policies.to(self.model.device)
                batch_values = batch_values.to(self.model.device)

                loss = self.model.train(batch_states, batch_policies, batch_values)
                epoch_loss += loss
                num_batches += 1

                if idx % (10 * self.supervised_batch_size) == 0:
                    self.visualizer.update_metrics(
                        iteration = self.iteration,
                        supervised_loss = loss,
                        exploration_rate = 1.0
                    )

                policy_logits, value_pred = self.model.net(batch_states)
                policy_loss = nn.CrossEntropyLoss()(policy_logits, batch_policies.argmax(dim=1))
                value_loss = nn.MSELoss()(value_pred.squeeze(), batch_values)
                regularization_loss = torch.sum(torch.tensor
                                                ([torch.sum(p**2) for p in self.model.net.parameters()]))
                batch_loss = policy_loss + value_loss + 1e-4*regularization_loss
                self.model.optimizer.zero_grad()
                batch_loss.backward()
                self.model.optimizer.step()

                total += batch_loss.item()

                if idx % 100 == 0:
                    print(f"Epoch: {epoch + 1}/{self.supervised_epochs}, Batch {idx}, Loss: {batch_loss.item()}")

                avg_loss = total / len(dataloader)
                print(f"Epoch: {epoch + 1}/{self.supervised_epochs}, Average Loss: {avg_loss:.4f}")
                if (epoch + 1) % 5 == 0:
                    self.model.save(f"Supervised_{epoch + 1}.pth")
        print("Supervised training completed!")



    def train(self):
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            all_data = []
            loss = None
            game_lengths = []
            for worker in self.workers:
                for _ in range(self.games_per_iterations // self.num_workers):
                    game_data = worker.play()
                    all_data.extend(game_data)

                    if game_data:
                        game_lengths.append(len(game_data))
            self.replay.extend(all_data)

            if len(self.replay) >= self.batch_size:
                batch = random.sample(self.replay, self.batch_size)
                states, target_policies, target_values = zip(*batch)

                loss = self.model.train(states, target_policies, target_values)
                print(f"Training loss: {loss:.4f}")

            temperature = max(0.1, 1.0 - (iteration / self.iterations)*0.9)
            for worker in self.workers:
                worker.temperature = temperature

            eval_results = None

            if (iteration + 1) % 50 == 0:
                eval_results = self.evaluate()

                win_rate = eval_results['wins']/(eval_results['wins'] + eval_results['draws'] + eval_results['losses'])
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self.best_model_path = f"chessRL_{iteration + 1}.pth"
                    self.model.save(self.best_model_path)
                    print(f"Best model saved win rate: {win_rate:.2f}")


                # Update visualizer:
                avg_length = np.mean(game_lengths) if game_lengths else None
                self.visualizer.update_metrics(
                    iteration=self.iteration,
                    reinforcement_loss = loss,
                    eval_results = eval_results,
                    game_lengths = avg_length,
                    exploration_rate = temperature
                )

                if (iteration + 1) % 10 == 0:
                    self.model.save(f"ChessRL_{iteration + 1}.pth")

                self.iterations += 1
        print("Reinforcement training completed")