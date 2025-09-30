# File: pipelines.py
# Mục đích: Training pipeline with curriculum scheduling for max_moves (and optional simulations).
# TL;DR: schedule max_moves theo linear/exp, cập nhật workers mỗi iteration.

from src.MCTS import MCTS
import numpy as np
import random
import torch
from UI.pieces import Board, Pawn, Queen
from collections import deque
import io
import chess.pgn
import time
from datetime import datetime
import csv
import os
import math
from src.visualizer import Visualizer
import concurrent.futures

# --- ChessState, SelfPlay, PGN helpers (unchanged logic except small fixes) ---
class ChessState:
    def __init__(self, board):
        self.board = board
        self.current_player = board.curPlayer
        self._legal_moves_cache = None

    def legal_moves(self):
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache

        legal_moves = []
        def get_moves_for_square(x, y):
            piece = self.board.getPiece(x, y)
            if piece and piece.color == self.current_player:
                moves = self.board.validMoves(x, y)
                return [((x, y), move) for move in moves]
            return []

        tasks = [(x, y) for x in range(8) for y in range(8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda args: get_moves_for_square(*args), tasks))

        for result in results:
            if result:
                legal_moves.extend(result)

        self._legal_moves_cache = legal_moves
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
        self.mcts = MCTS(model, simulations=simulations)
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
        start_player = state.board.curPlayer
        count = 0

        while not state.isTerminal() and count < self.max_moves:
            action_probs = self.mcts.run(state)
            if not action_probs:
                break

            actions = list(action_probs.keys())
            visits = np.array([action_probs[a] for a in actions], dtype=np.float64)
            if visits.sum() == 0:
                probs = np.ones_like(visits) / len(visits)
            else:
                if self.temperature > 0:
                    probs = visits ** (1.0 / self.temperature)
                    probs = probs / probs.sum()
                else:
                    probs = visits / visits.sum()

            states.append(state)
            vector_policy = np.zeros(64 * 64, dtype=np.float32)
            for a, p in zip(actions, probs):
                idx = self._action_to_idx(a)
                if 0 <= idx < vector_policy.size:
                    vector_policy[idx] = float(p)
            ssum = vector_policy.sum()
            if ssum > 0:
                vector_policy /= ssum
            mcts_policies.append(vector_policy)

            action = random.choices(actions, weights=probs, k=1)[0]
            state = state.apply(action)
            count += 1

        outcome = state.get_reward()
        training_data = []
        for i, (s, policy) in enumerate(zip(states, mcts_policies)):
            mover = start_player if (i % 2 == 0) else ('B' if start_player == 'W' else 'W')
            perspective = outcome if mover == 'W' else -outcome
            training_data.append((s, policy, perspective))
        return training_data


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
            policy_vector = np.zeros(64 * 64, dtype=np.float32)
            idx = from_square * 64 + to_square
            policy_vector[int(idx)] = 1.0
            policies.append(policy_vector)
            values.append(outcome if board.turn == chess.WHITE else -outcome)
    return states, policies, values

# --- Scheduler helper ---
def schedule_value(iteration, start, end, steps, schedule_type='linear', mode='decrease'):
    if steps <= 0:
        return end if mode == 'decrease' else start
    frac = min(max(iteration / steps, 0.0), 1.0)
    if schedule_type == 'linear':
        if mode == 'decrease':
            return int(round(start + (end - start) * frac))
        else:
            return int(round(start + (end - start) * frac))
    elif schedule_type == 'exp':
        # exponential interpolation (smooth)
        if mode == 'decrease':
            return int(round(start * (end / (start if start != 0 else 1)) ** frac))
        else:
            return int(round(start * (end / (start if start != 0 else 1)) ** frac))
    else:
        return int(round(start + (end - start) * frac))

class TrainingPipeline:
    def __init__(self, model,
                 num_workers=4,
                 games_per_iteration=50,
                 iterations=1000,
                 buffer=2000,
                 batch_size=8192,
                 simulations_start=100,
                 simulations_end=100,
                 max_moves_start=300,
                 max_moves_end=80,
                 schedule_steps=800,
                 schedule_type='linear',
                 schedule_mode='decrease',
                 supervised_epochs=10,
                 supervised_batch_size=2048,
                 visualizer=None):
        self.model = model
        self.num_workers = num_workers
        self.games_per_iteration = games_per_iteration
        self.buffer = buffer
        self.batch_size = batch_size
        self.supervised_epochs = supervised_epochs
        self.supervised_batch_size = supervised_batch_size
        self.iterations = iterations
        self.replay = deque(maxlen=buffer)

        # initial worker list using start values
        self.workers = [SelfPlay(model, simulations=simulations_start, max_moves=max_moves_start)
                        for _ in range(num_workers)]

        # curriculum params
        self.simulations_start = simulations_start
        self.simulations_end = simulations_end
        self.max_moves_start = max_moves_start
        self.max_moves_end = max_moves_end
        self.schedule_steps = schedule_steps
        self.schedule_type = schedule_type
        self.schedule_mode = schedule_mode

        self.visualizer = visualizer if visualizer else Visualizer()
        self.iteration = 0
        self.best_win_rate = 0
        self.best_model_path = None

    def _apply_curriculum(self, iteration):
        cur_max_moves = schedule_value(iteration, self.max_moves_start, self.max_moves_end,
                                       self.schedule_steps, self.schedule_type, self.schedule_mode)
        cur_simulations = schedule_value(iteration, self.simulations_start, self.simulations_end,
                                         self.schedule_steps, self.schedule_type, self.schedule_mode)
        # clip to sensible bounds
        cur_max_moves = max(10, min(2000, cur_max_moves))
        cur_simulations = max(1, min(5000, cur_simulations))
        # apply to workers
        for w in self.workers:
            w.max_moves = cur_max_moves
            # update underlying MCTS if attribute exists
            if hasattr(w, 'mcts') and hasattr(w.mcts, 'simulations'):
                w.mcts.simulations = cur_simulations
        return cur_max_moves, cur_simulations

    def evaluate(self):
        wins = 0
        draws = 0
        losses = 0
        for _ in range(10):
            state = ChessState(Board())
            count = 0
            while not state.isTerminal() and count < 200:
                if state.board.curPlayer == 'W':  # AI's turn
                    action_probs = self.workers[0].mcts.run(state)
                    if not action_probs:
                        break
                    action = max(action_probs, key=action_probs.get)
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

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32)
        values_tensor = torch.tensor(np.array(values), dtype=torch.float32)

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
                loss = self.model.train(batch_states_list, batch_policies_list, batch_values_list)
                total_loss += loss
                num_batches += 1
                if idx % 100 == 0:
                    print(f"Epoch: {epoch + 1}/{self.supervised_epochs}, Batch {idx}, Loss: {loss:.4f}")
                    self.visualizer.update_metrics(iteration=self.iteration, supervised_loss=loss, exploration_rate=1.0)
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch: {epoch + 1}/{self.supervised_epochs}, Average Loss: {avg_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                self.model.save(f"Supervised_{epoch + 1}.pth")
        print("Supervised training completed!")

    def train(self):
        for iteration in range(self.iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.iterations} ===")
            # apply curriculum before self-play
            cur_max_moves, cur_simulations = self._apply_curriculum(iteration)
            print(f"Curriculum: max_moves={cur_max_moves}, simulations={cur_simulations}")

            all_data = []
            game_lengths = []
            per_worker_games = max(1, self.games_per_iteration // max(1, self.num_workers))

            for worker in self.workers:
                for _ in range(per_worker_games):
                    game_data = worker.play()
                    if game_data:
                        all_data.extend(game_data)
                        game_lengths.append(len(game_data))

            self.replay.extend(all_data)

            loss = None
            if len(self.replay) >= self.batch_size:
                batch = random.sample(self.replay, self.batch_size)
                states, target_policies, target_values = zip(*batch)
                param_pre = [p.detach().cpu().clone() for p in self.model.net.parameters()]
                loss = self.model.train(states, target_policies, target_values)
                print(f"Training loss: {loss:.4f}")
                param_post = [p.detach().cpu().clone() for p in self.model.net.parameters()]
                max_delta = 0.0
                for a, b in zip(param_pre, param_post):
                    delta = (b - a).norm().item()
                    if delta > max_delta:
                        max_delta = delta
                print(f"Max parameter L2-change this batch: {max_delta:.6e}")

            temperature = max(0.1, 1.0 - (iteration / max(1, self.iterations)) * 0.9)
            for worker in self.workers:
                worker.temperature = temperature

            eval_results = None
            if (iteration + 1) % 50 == 0:
                eval_results = self.evaluate()
                total = (eval_results['wins'] + eval_results['draws'] + eval_results['losses'])
                win_rate = eval_results['wins'] / total if total > 0 else 0.0
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self.best_model_path = f"chessRL_{iteration + 1}.pth"
                    self.model.save(self.best_model_path)
                    print(f"Best model saved with win rate: {win_rate:.2f}")

            avg_length = np.mean(game_lengths) if game_lengths else None
            self.visualizer.update_metrics(iteration=self.iteration,
                                           reinforcement_loss=loss,
                                           eval_results=eval_results,
                                           game_length=avg_length,
                                           exploration_rate=temperature)

            if (iteration + 1) % 10 == 0:
                self.model.save(f"ChessRL_{iteration + 1}.pth")

            self.iteration += 1

        print("Reinforcement training completed")

