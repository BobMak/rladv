import numpy as np


class EnvStats:
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.stats = {
            "eval_reward_auc": 0,
            "number_of_runs": 0,
            "max_eval_reward": 0,
            "number_of_steps": 0,
        }

    def add(self, data: np.ndarray):
        self.data.append(data)

    def get(self):
        return np.array(self.data)

    def __str__(self):
        return f"{self.name}: {self.get()}"