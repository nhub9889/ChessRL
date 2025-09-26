from src.MCTS import MCTS
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
from src.visualizer import Visualizer


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
        if self.board.checkmate('W'):
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

class SelfPlay:
    def __init__(self, model, simulations=800, temperature=1.0, max_moves=512):
        self.model = model
        self.mcts = MCTS(model, simulations)
        self.temperature = temperature
        self.max_moves = max_moves

    def _action_to_idx(self, action):
        from_square, to_square = action
        from_idx = from_square[0] * 8 + from_square[1]
        to_idx = to_square[0] * 8 + to_square[1]
        return from_idx * 64 + to_idx

    def play(self):
        state = ChessState(Board())
        states = []
        mcts_policies = []
        currentPlayer = state.board.curPlayer
        count = 0

        while not state.isTerminal() and count < self.max_moves:
            action_probs = self.mcts.run(state)

            if self.temperature > 0:
                visits = np.array([count for count in action_probs.values()])
                probs = visits ** (1 / self.temperature)
                probs /= probs.sum()
            else:
                probs = np.array([prob for prob in action_probs.values()])
                probs /= probs.sum()

            states.append(state)
            vector_policy = np.zeros(64 * 64)
            for (from_pos, to_pos), prob in action_probs.items():
                idx = from_pos[0] * 8 * 64 + from_pos[1] * 64 + to_pos[0] * 8 + to_pos[1]
                vector_policy[idx] = prob
            mcts_policies.append(vector_policy)

            actions = list(action_probs.keys())
            action = random.choices(actions, weights=probs)[0]

            state = state.apply(action)
            count += 1

        outcome = state.get_reward()
        training = []
        for i, (state, policy) in enumerate(zip(states, mcts_policies)):
            player_outcome = outcome if (i % 2 == 0 and currentPlayer == 'W') or (
                        i % 2 == 1 and currentPlayer == 'B') else -outcome
            training.append((state, policy, player_outcome))

        return training


def LoadPGN(path, num_games=100000):
    games = []
    with open(path) as pgn:
        for _ in range(num_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games


def board_to_state(board):
    state = np.zeros((18, 8, 8), dtype=np.float32)
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

        for node in game.mainline():
            board.push(node.move)
            state = board_to_state(board)
            states.append(state)

            move = node.move
            from_square = move.from_square
            to_square = move.to_square

            policy_vector = np.zeros(64 * 64)
            idx = from_square * 64 + to_square
            policy_vector[idx] = 1.0
            policies.append(policy_vector)
            values.append(outcome if board.turn == chess.WHITE else -outcome)

    return states, policies, values


class TrainingPipeline:
    def __init__(self, model, num_workers=4, games_per_iteration=100, iterations=1000, buffer=100000, batch_size=1024,
                 supervised_epochs=10, supervised_batch_size=512, visualizer=None):
        self.model = model
        self.num_workers = num_workers
        self.games_per_iteration = games_per_iteration
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

        self.visualizer = visualizer if visualizer else Visualizer()

    def evaluate(self):
        wins = 0
        draws = 0
        losses = 0
        for _ in range(10):
            state = ChessState(Board())
            count = 0
            while not state.isTerminal() and count < 100:
                if state.board.curPlayer == 'W':  # AI's turn
                    action_probs = self.workers[0].mcts.run(state)
                    action = max(action_probs, key=action_probs.get)
                else:  # Random opponent's turn
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
                draws += 1
            else:
                losses += 1

        print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
        return {'wins': wins, 'draws': draws, 'losses': losses}

    def supervised_train(self, path, num_games=10000):
        print("Loading supervised data...")
        games = LoadPGN(path, num_games)
        print(f"Loaded {len(games)} games")
        print("Processing supervised data...")
        states, policies, values = ProcessPGN(games)
        print(f"Processed {len(states)} positions")

        # Convert to tensors
        states_tensor = torch.tensor(np.array(states))
        policies_tensor = torch.tensor(np.array(policies))
        values_tensor = torch.tensor(np.array(values)).float()

        dataset = torch.utils.data.TensorDataset(states_tensor, policies_tensor, values_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.supervised_batch_size, shuffle=True)

        print("Starting supervised training...")
        for epoch in range(self.supervised_epochs):
            total_loss = 0.0
            num_batches = 0

            for idx, (batch_states, batch_policies, batch_values) in enumerate(dataloader):
                batch_states_list = [batch_states[i] for i in range(batch_states.size(0))]
                batch_policies_list = [batch_policies[i] for i in range(batch_policies.size(0))]
                batch_values_list = [batch_values[i] for i in range(batch_values.size(0))]

                # Train on batch
                loss = self.model.train(batch_states_list, batch_policies_list, batch_values_list)
                total_loss += loss
                num_batches += 1

                if idx % 100 == 0:
                    print(f"Epoch: {epoch + 1}/{self.supervised_epochs}, Batch {idx}, Loss: {loss:.4f}")

                    # Update visualization
                    self.visualizer.update_metrics(
                        iteration=self.iteration,
                        supervised_loss=loss,
                        exploration_rate=1.0
                    )

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch: {epoch + 1}/{self.supervised_epochs}, Average Loss: {avg_loss:.4f}")

            if (epoch + 1) % 5 == 0:
                self.model.save(f"Supervised_{epoch + 1}.pth")

        print("Supervised training completed!")

    def train(self):
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            all_data = []
            game_lengths = []

            for worker in self.workers:
                for _ in range(self.games_per_iteration // self.num_workers):
                    game_data = worker.play()
                    all_data.extend(game_data)

                    if game_data:
                        game_lengths.append(len(game_data))

            self.replay.extend(all_data)

            loss = None
            if len(self.replay) >= self.batch_size:
                batch = random.sample(self.replay, self.batch_size)
                states, target_policies, target_values = zip(*batch)

                loss = self.model.train(states, target_policies, target_values)
                print(f"Training loss: {loss:.4f}")

            temperature = max(0.1, 1.0 - (iteration / self.iterations) * 0.9)
            for worker in self.workers:
                worker.temperature = temperature

            eval_results = None
            if (iteration + 1) % 50 == 0:
                eval_results = self.evaluate()

                win_rate = eval_results['wins'] / (
                            eval_results['wins'] + eval_results['draws'] + eval_results['losses'])
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self.best_model_path = f"chessRL_{iteration + 1}.pth"
                    self.model.save(self.best_model_path)
                    print(f"Best model saved with win rate: {win_rate:.2f}")

            # Update visualizer
            avg_length = np.mean(game_lengths) if game_lengths else None
            self.visualizer.update_metrics(
                iteration=self.iteration,
                reinforcement_loss=loss,
                eval_results=eval_results,
                game_length=avg_length,
                exploration_rate=temperature
            )

            if (iteration + 1) % 10 == 0:
                self.model.save(f"ChessRL_{iteration + 1}.pth")

            self.iteration += 1

        print("Reinforcement training completed")