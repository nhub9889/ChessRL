from MCTS import MCTS
import numpy as np
import random
from UI.pieces import Board, Pawn, Queen

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
    def __init__(self, model, simulations= 800, temperature= 1.0):
        self.model = model
        self.mcts = MCTS(model, simulations)
        self.temperature = temperature
        self.data = []

    def _action_to_idx(self, action):
        from_square, to_square = action
        from_idx = from_square[0]*8 + from_square[1]
        to_idx = to_square[0]*8 + to_square[1]
        return from_idx*64 + to_idx
    def play(self, initial_state):
        state = initial_state
        game_data = []

        while not state.isTerminal():
            action_probs, _ = self.mcts.run(state)

            if self.temperature > 0:
                visits = np.array([count for count in action_probs.values()])
                probs = visits**(1/self.temperature)
                probs /= probs.sum()
                action = list(action_probs.keys())
                action = random.choices(action, weights=probs)[0]
            else:
                action = max(action_probs, key= action_probs.get)

            game_data.append((state, action_probs))
            state = state.apply_action(action)
        outcome = state.get_reward()
        training_data = []
        for state, action_probs in game_data:
            action_probs_vec = np.zeros(64*64)
            for action, prob in action_probs.items():
                idx = self._action_to_idx(action)
                action_probs_vec[idx] = prob

            training_data.append((state, action_probs_vec, outcome))
            outcome = -outcome

        self.data.extend(training_data)
        return training_data

class TrainingPipeline:
    def __init__(self, model, num_workers= 5, games_per_iterations= 100, iterations= 1000):
        self.model = model
        self.num_workers = num_workers
        self.games_per_iterations = games_per_iterations
        self.iterations = iterations
        self.workers = [SelfPlay(model) for _ in range(num_workers)]

    def train(self):
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            all_data = []
            for worker in self.workers:
                for _ in range(self.games_per_iterations//self.num_workers):
                    initial_state = ChessState(Board())
                    game_data = worker.play(initial_state)
                    all_data.extend(game_data)

            states, target_policies, target_values = zip(*all_data)

            loss = self.model.train(states, target_policies, target_values)
            print(f"Training loss: {loss}")

            if (iteration + 1) % 10 == 0:
                self.model.save(f"ChessRL_{iteration + 1}.pth")

            for worker in self.workers:
                worker.temperature = max(0.1, worker.temperature*0.99)
