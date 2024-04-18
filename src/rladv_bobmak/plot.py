"""retrieve runs from wandb project in a specified group,
integrate the evaluation reward and calculate the percentage advantage/disadvantage
of our method compared to the baseline for each available environment
"""
import copy
import os

import wandb
import numpy as np
import tqdm
import pickle


def plot_advantage(project, comparison_variable, cache=True, use_cached=True):
    api = wandb.Api()  # eval/mean_reward
    print("getting runs")
    runs = api.runs(project)
    if use_cached and os.path.exists(f"envs_{project}.pkl"):
        with open(f"envs_{project}.pkl", "rb") as f:
            envs = pickle.load(f)
    else:
        envs = {}
        for run in tqdm.tqdm(runs):
            try:
                if run.config["env_id"] not in envs:
                    empty_stats = {
                        "eval_reward_auc": 0,
                        "number_of_runs": 0,
                        "max_eval_reward": 0,
                        "number_of_steps": 0,
                    }
                    envs[run.config["env_id"]] = {
                        f"{comparison_variable}": copy.deepcopy(empty_stats),
                        "baseline": copy.deepcopy(empty_stats),
                    }
                key = f"{comparison_variable}" if run.config[comparison_variable]=='True' else "baseline"
                # calculate reward auc
                eval_reward = run.history(keys=["eval/mean_reward"])
                eval_reward = np.array(eval_reward).astype(float)
                # skip if there is less than 20% of the expected data
                if len(eval_reward) < 50:
                    print(f"skipping {run.name} due to insufficient data {len(eval_reward)}")
                    continue
                envs[run.config["env_id"]][key]['eval_reward_auc'] += np.sum(eval_reward)
                envs[run.config["env_id"]][key]['number_of_runs'] += 1
                envs[run.config["env_id"]][key]['number_of_steps'] += len(eval_reward)
                envs[run.config["env_id"]][key]['max_eval_reward'] = max(
                    envs[run.config["env_id"]][key]['max_eval_reward'], np.max(eval_reward)
                )
            except Exception as e:
                print("error in run", run.name, e)
                continue
        if cache:
            with open(f"envs_{project}.pkl", "wb") as f:
                pickle.dump(envs, f)
    # calculate percentage advantage/disadvantage
    compared_envs = {}
    for env_id, env_stat in envs.items():
        cont=False
        for key in env_stat:
            n_runs = env_stat[key]['number_of_runs']
            if n_runs < 1:
                print(f"Not enough runs for {env_id} ({n_runs})")
                cont = True
                break
            env_stat[key]['eval_reward_auc'] /= env_stat[key]['number_of_steps']
        if cont:
            env_stat['percentage_advantage'] = 0
            continue
        compared_envs[env_id[:-len("NoFrameskip-v4")]] = env_stat
        rew_targ = env_stat[f"{comparison_variable}"]['eval_reward_auc']
        rew_base = env_stat['baseline']['eval_reward_auc']
        adv = rew_targ / rew_base
        adv = 1 / adv if rew_base<0 and rew_targ<0 else adv
        # todo: add a baseline human performance for normalization
        env_stat['percentage_advantage'] = adv * 100 - 100
    # sort the environments by percentage advantage
    compared_envs = dict(
        sorted(compared_envs.items(), key=lambda item: item[1]['percentage_advantage'])
    )
    # plot the results
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    # use poster settings:
    sns.set_context("poster")
    # Make font color black:
    plt.rcParams['text.color'] = 'black'
    # plot the results
    plt.figure(figsize=(12, 8))
    plt.bar(compared_envs.keys(), [compared_envs[env]['percentage_advantage'] for env in compared_envs])
    plt.title(f"Percentage advantage of {comparison_variable} compared to baseline")
    plt.xlabel("Environment")
    plt.yscale("symlog")
    plt.ylabel("Percentage advantage")
    plt.xticks(rotation=90)
    # make the x-axis labels fit in the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="shaping")
    parser.add_argument("--comparison_variable", type=str, default="do_shape")
    args = parser.parse_args()
    plot_advantage(args.project, args.comparison_variable, use_cached=True)