
import math
import torch

class MCTSNode():
    def __init__(self, state, parent= None, action= None, prior= 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visits = 0
        self.value = 0.0
        self.children = []
        self.outcome = None

    def isExpanded(self):
        return len(self.children) > 0

    def isTerminal(self):
        return self.outcome is not None

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state = None,
                    parent = self,
                    action = action,
                    prior = prob
                )

    def select(self, exploration_weight= 1.0):
        bscore = -float('inf')
        baction = None
        bchild = None

        for action, child in self.children.items():
            if child.visits == 0:
                score = exploration_weight*child.prior*math.sqrt(self.visits)/ 1
            else:
                score = child.value() + exploration_weight*child.prior * math.sqrt(self.visits) / (1 + child.visit)
            if score > bscore:
                bscore = score
                baction = action
                bchild = child
        return baction, bchild

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value/self.visits

    def update(self, value):
        self.visits += 1
        self.value += value

    def update_recursive(self, value):
        if self.parent:
            self.parent.update_recursive(-value)

        self.update(value)

class MCTS:
    def __init__(self, model, exploration_weight= 1.0, simulations= 800):
        self.model = model
        self.exploration_weight = exploration_weight
        self.simulations = simulations

    def run(self, state):
        root = MCTSNode(state)
        for _ in range(self.simulations):
            node = root
            path = [node]

            while node.isExpanded() and not node.isTerminal():
                action, node = node.select(self.exploration_weight)
                path.append(node)

            if not node.isTerminal():
                policy, value = self.model.predict(node.state)
                action = self._get_action_probs(node.state, policy)
                node.expand(action)
                node.update_recursive(value)
            else:
                value = node.outcome
                node.update_recursive(value)

            action_probs = {
                action: child.visits
                for action, child in root.children.items()
            }
            total = sum(action_probs.values())

            if total > 0:
                action_probs = {a: c/total for a, c in action_probs.items()}
            return action_probs, root.children[max(action_probs, key= action_probs.get)].action

    def _get_action_probs(self, state, policy):
        legal_moves = state.getLegalMoves()
        action_probs = []
        policy = torch.softmax(torch.tensor(policy), dim= 1).numpy()
        for move in legal_moves:
            idx = self._move_to_index(move)
            action_probs.append((move, policy[idx]))

        total = sum(prob for _, prob in action_probs)
        if total > 0:
            action_probs = [(move, prob/total) for move, prob in action_probs]
        return action_probs

    def _move_to_index(self, move):
        from_square, to_square = move
        from_idx = from_square[0]*8 + from_square[1]
        to_idx = to_square[0]*8 + to_square[1]

        return from_idx*64 + to_idx
