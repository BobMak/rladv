import os
import pickle

import numpy as np
import wandb
import tensorflow as tf

import tqdm


class Handler:
    def get_runs(self, *args):
        return NotImplemented()

    def get_history(self, *args):
        return NotImplemented()


class Filter:
    def __init__(self, key, value, type="set"):
        assert type in {"range", "set"}
        if type == "range":
            assert len(value) == 2
        if type == "set":
            assert isinstance(value, set) or isinstance(value, list)
        self.key = key
        self.value = value
        self.type = type

    def __call__(self, run_path) -> bool:
        raise NotImplemented()


class WandbHandler(Handler):
    def __init__(self, path, filters):
        self.filters = filters
        self.wandb = wandb
        self.path = path

    def get_runs(self, use_cached=True, cache=True):
        print("getting runs")
        api = wandb.Api()
        if use_cached and os.path.exists(f"envs_{self.path}.pkl"):
            with open(f"envs_{self.path}.pkl", "rb") as f:
                runs = pickle.load(f)
            return runs
        else:
            self.filters['state'] = "finished"
            runs = api.runs(path=self.path, filters=self.filters)
        if cache:
            with open(f"envs_{self.path}.pkl", "wb") as f:
                pickle.dump(runs, f)
        return runs

    def get_history(self, run, value_variable, comparison_variable):
        key = run.summary.get(comparison_variable)
        env_id = run.summary.get('env_id')
        return np.array(run.history(keys=[value_variable])).astype(float), key, env_id


class TBFilter:
    def __init__(self, key, value, type="set"):
        assert type in {"range", "set"}
        if type == "range":
            assert len(value) == 2
        if type == "set":
            assert isinstance(value, set) or isinstance(value, list)
        self.key = key
        self.value = value
        self.type = type

    def __call__(self, run_path):
        """Seeks the first value of the filtered key and checks if its value is in the set"""
        for e in tf.compat.v1.train.summary_iterator(run_path):
            if e.summary.value.tag == self.key:
                val = e.summary.value.simple_value
                if self.type == "range":
                    return self.value[0] <= val <= self.value[1]
                elif self.type == "set":
                    return val in self.value
        # didn't find the filtered key in the run
        return False


class TensorboardHandler(Handler):
    """Handles local tensorboard files"""
    def __init__(self, logdir, filters:dict[str, any]):
        self.logdir = logdir
        self.filters = [TBFilter(key, val, type) for key, (val, type) in filters.items()]

    def filter_run(self, run):
        for filter in self.filters:
            if not filter(run):
                return False
        return True

    def get_runs(self, **kwargs):
        for key, value in kwargs.items():
            print(f"ignoring {key}={value}")
        tb_run_paths = []
        for root, dirs, files in os.walk(self.logdir):
            for file in files:
                if file.startswith("events.out.tfevents."):
                    tb_run_paths.append(os.path.join(root,file))
                    break
        filtered_run_paths = []
        for run_path in tb_run_paths:
            print(run_path)
            if not self.filter_run(run_path):
                continue
            filtered_run_paths.append(run_path)
        return filtered_run_paths

    def get_history(self, run, value_variable, comparison_variable):
        eval_rewards = []
        key = None
        env_id = None
        for e in tf.compat.v1.train.summary_iterator(run):
            for v in e.summary.value:
                if v.tag == value_variable:
                    eval_rewards.append(v.simple_value)
                elif v.tag == "env_id":
                    env_id = v.simple_value
                elif v.tag == comparison_variable:
                    key = v.simple_value
        return np.array(eval_rewards).astype(float), key, env_id


if __name__ == "__main__":
    logdir = 'runs'
    handler = TensorboardHandler(logdir)
    handler.get_runs("runs", "eta")