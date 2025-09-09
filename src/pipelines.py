from MCTS import MCTS
import numpy as np
import random
from UI.pieces import Board, Pawn, Queen
from collections import deque

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


class TrainingPipeline:
    def __init__(self, model, num_workers= 4, games_per_iterations= 100, iterations= 1000, buffer= 100000, batch_size= 1024):
        self.model = model
        self.num_workers = num_workers
        self.games_per_iterations = games_per_iterations
        self.buffer = buffer
        self.batch_size = batch_size
        self.iterations = iterations
        self.replay = deque(maxlen=buffer)
        self.workers = [SelfPlay(model) for _ in range(num_workers)]

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
    def train(self):
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            all_data = []
            for worker in self.workers:
                for _ in range(self.games_per_iterations//self.num_workers):
                    game_data = worker.play()
                    all_data.extend(game_data)
            self.replay.extend(all_data)

            if len(self.replay) >= self.batch_size:
                batch = random.sample(self.replay, self.batch_size)
                states, target_policies, target_values = zip(*batch)

                loss = self.model.train(states, target_policies, target_values)
                print(f"Training loss: {loss}")

            if (iteration + 1) % 10 == 0:
                self.model.save(f"ChessRL_{iteration + 1}.pth")

            temperature = max(0.1, 1.0 - (iteration / self.iterations)*0.9)
            for worker in self.workers:
                worker.temperature = temperature

            if (iteration + 1) % 50 == 0:
                self.evaluate()
