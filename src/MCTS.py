import math
import torch


class MCTSNode():
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0  # Renamed from 'value' to avoid conflict
        self.children = {}
        self.outcome = None

    def isExpanded(self):
        return len(self.children) > 0

    def isTerminal(self):
        if self.state is None:
            return False
        return self.state.isTerminal()

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                # Create child node with the new state after applying the action
                if self.state is not None:
                    new_state = self.state.apply(action)
                else:
                    new_state = None

                self.children[action] = MCTSNode(
                    state=new_state,
                    parent=self,
                    action=action,
                    prior=prob
                )

    def select(self, exploration_weight=1.0):
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            # FIXED: Use get_value() method instead of value() call
            if child.visits == 0:
                ucb_score = exploration_weight * child.prior * math.sqrt(self.visits + 1e-8)
            else:
                ucb_score = child.get_value() + exploration_weight * child.prior * math.sqrt(self.visits) / (
                            1 + child.visits)

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def get_value(self):  # Renamed from value() to avoid conflict
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def update(self, value):
        self.visits += 1
        self.value_sum += value

    def update_recursive(self, value):
        self.update(value)
        if self.parent:
            # Negate value for the parent (alternating turns)
            self.parent.update_recursive(-value)


class MCTS:
    def __init__(self, model, exploration_weight=1.0, simulations=800):
        self.model = model
        self.exploration_weight = exploration_weight
        self.simulations = simulations

    def run(self, state):
        root = MCTSNode(state)

        for _ in range(self.simulations):
            node = root
            path = [node]

            # Selection
            while node.isExpanded() and not node.isTerminal():
                action, node = node.select(self.exploration_weight)
                if node is None:  # No children available
                    break
                path.append(node)

            if node is None or node.isTerminal():
                # Terminal node or no moves available
                if node and node.state:
                    leaf_value = node.state.get_reward()
                else:
                    leaf_value = 0.0
            else:
                # Expansion - get policy and value from model
                policy, value = self.model.predict(node.state)
                action_probs = self._get_action_probs(node.state, policy)
                node.expand(action_probs)
                leaf_value = value

            # Backpropagation
            for i, node in enumerate(reversed(path)):
                # Alternate perspective for opponent
                perspective_value = leaf_value if i % 2 == 0 else -leaf_value
                node.update_recursive(perspective_value)

        # Return action probabilities
        action_probs = {
            action: child.visits
            for action, child in root.children.items()
        }

        total_visits = sum(action_probs.values())
        if total_visits > 0:
            action_probs = {action: count / total_visits for action, count in action_probs.items()}

        return action_probs

    def _get_action_probs(self, state, policy):
        legal_moves = state.legal_moves()
        action_probs = []

        policy_tensor = torch.tensor(policy)
        if policy_tensor.dim() == 1:
            policy_probs = torch.softmax(policy_tensor, dim=0).numpy()
        else:
            policy_probs = torch.softmax(policy_tensor, dim=1).numpy()[0]

        for move in legal_moves:
            idx = self._move_to_index(move)
            if idx < len(policy_probs):
                action_probs.append((move, policy_probs[idx]))
            else:
                action_probs.append((move, 1e-8))

        total = sum(prob for _, prob in action_probs)
        if total > 0:
            action_probs = [(move, prob / total) for move, prob in action_probs]
        else:
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
            action_probs = [(move, uniform_prob) for move in legal_moves]

        return action_probs

    def _move_to_index(self, move):
        from_square, to_square = move
        from_idx = from_square[0] * 8 + from_square[1]
        to_idx = to_square[0] * 8 + to_square[1]
        return from_idx * 64 + to_idx