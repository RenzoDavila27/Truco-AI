import numpy as np

class RandomAgent:
    def choose_action(self, action_mask):
        """
        Chooses a random valid action from the action mask.
        """
        valid_actions = np.where(action_mask)[0]
        return np.random.choice(valid_actions)
