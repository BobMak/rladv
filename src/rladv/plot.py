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
import matplotlib.pyplot as plt
import seaborn as sns

from Handlers import WandbHandler, TensorboardHandler, Handler
from atari_utils import get_human_normalized_score


empty_stats = {
    "eval_reward_auc": 0,
    "number_of_runs": 0,
    "max_eval_reward": -np.inf,
    "number_of_steps": 0,
    "human_normalized": [],
}
x_axis = np.arange(0,10_000_000, 50_000)

aggreation_to_fn = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
}

handler_to_cls = {
    "tb": TensorboardHandler,
    "wandb": WandbHandler,
}

def parase_env_reward_data(
        path, comparison_variable,
        value_variable="eval/mean_reward",
        cache=True,
        use_cached=True,
        handler_type:Handler=None,
        filters=None,
):
    handler = handler_to_cls[handler_type](path, filters=filters)
    runs = handler.get_runs(cache=cache, use_cached=use_cached)
    if use_cached and handler_type == "wandb":
        return runs
    # parse eval reward data
    envs = {}
    for run in tqdm.tqdm(runs):
        eval_reward, key, env_id = handler.get_history(run, value_variable, comparison_variable)
        envs[env_id][key]['eval_reward_auc'] += np.sum(eval_reward)
        envs[env_id][key]['number_of_runs'] += 1
        envs[env_id][key]['number_of_steps'] += len(eval_reward)
        envs[env_id][key]['max_eval_reward'] = max(
            envs[env_id][key]['max_eval_reward'], np.max(eval_reward)
        )
        # record the best model performance in an episode
        envs[env_id][key][f'human_normalized'].append(get_human_normalized_score(env_id, eval_reward))
        # except Exception as e:
        #     print("error in run", run, e)
        #     continue
    return envs


def calculate_advangate(envs, aggregation="mean", comparison_variable:str= "do_shape", baseline_value=False):
    aggr_fn = aggreation_to_fn[aggregation]
    # calculate percentage advantage/disadvantage
    compared_envs = {}
    for env_id, env_stat in envs.items():
        cont = False
        for key in env_stat:
            n_runs = env_stat[key]['number_of_runs']
            if n_runs < 1:
                print(f"Not enough runs for {env_id} ({n_runs})")
                cont = True
                break
            # record either max, mean, or median best human normalized score
            # over all episodes for the given environment and comparison variable
            env_stat[key][f'human_normalized_{aggregation}'] = aggr_fn(np.array(env_stat[key]['human_normalized']),
                                                                       axis=0)
        if cont:
            env_stat['percentage_advantage'] = 0
            continue
        compared_envs[env_id[:-len("NoFrameskip-v4")]] = env_stat
        rew_targ = env_stat[f"{comparison_variable}"]['eval_reward_auc']
        rew_base = env_stat[baseline_value]['eval_reward_auc']
        adv = rew_targ / rew_base
        adv = 1 / adv if rew_base < 0 and rew_targ < 0 else adv
        env_stat['percentage_advantage'] = adv * 100 - 100
    # sort the environments by percentage advantage
    compared_envs = dict(
        sorted(compared_envs.items(), key=lambda item: item[1]['percentage_advantage'])
    )
    return compared_envs


def plot_advantage(envs, aggregation="median", comparison_variable:str= "do_shape", baseline_value=False):
    compared_envs = calculate_advangate(envs, aggregation=aggregation, comparison_variable=comparison_variable, baseline_value=baseline_value)
    # plot the results
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


def plot_advantage_comp_aggr(envs, comp_var_aggregation='max', aggregation="median", comparison_variable:str= "do_shape", baseline_value=False):
    """plot_advantage but aggregated over the comparison variable.
    Eg max, mean, or median for each environment"""
    compared_envs = calculate_advangate(envs, aggregation=aggregation, comparison_variable=comparison_variable, baseline_value=baseline_value)
    comp_var_aggr_fn = aggreation_to_fn[comp_var_aggregation]
    aggr_envs = {}
    for env_id, env_stat in compared_envs.items():
        hmn_norm_scores = np.zeros(len(env_stat))
        comp_vars = []
        for i, key in enumerate(env_stat):
            hmn_norm_scores[i] = env_stat[key][f'percentage_advantage']
            comp_vars.append(key)
        aggr_score = comp_var_aggr_fn(hmn_norm_scores, axis=0)
        best_comp_var = comp_vars[np.argmax(hmn_norm_scores)]
        aggr_envs[env_id] = {
            'percentage_advantage': aggr_score,
            'best_comp_var': best_comp_var
        }

    # plot the results
    sns.set_theme(style="whitegrid")
    # use poster settings:
    sns.set_context("poster")
    # Make font color black:
    plt.rcParams['text.color'] = 'black'
    # plot the results
    plt.figure(figsize=(12, 8))
    plt.bar(compared_envs.keys(), [aggr_envs[env]['percentage_advantage'] for env in aggr_envs])
    plt.title(f"Percentage advantage of {comparison_variable} compared to baseline")
    plt.xlabel("Environment")
    plt.yscale("symlog")
    plt.ylabel("Percentage advantage")
    plt.xticks(rotation=90)
    # make the x-axis labels fit in the plot
    plt.tight_layout()
    plt.show()


def plot_human_normalized_env_histories(envs, comp_var_aggregation='', comparison_variable:str= "do_shape", baseline_value=False):
    """historical plots for human normalized performance comparison
    variables aggregated for all environments"""
    compared_envs = calculate_advangate(envs, aggregation=comp_var_aggregation, comparison_variable=comparison_variable,
                                        baseline_value=baseline_value)
    # Now plot the median human normalized scores
    plt.figure(figsize=(12, 8))
    # get all env eval rewards:
    for env_id, env_stat in compared_envs.items():
        # plot the results
        for key in env_stat:
            plt.plot(x_axis, env_stat[key][f'human_normalized_{comp_var_aggregation}'], label=key)
    plt.legend()
    plt.title(f"Aggregated Reward Curve")
    plt.xlabel('Env. Steps')
    plt.ylabel(f'{comp_var_aggregation} Human Normalized Score')
    plt.savefig(f"human_norm_{comparison_variable}.png")


# todo how do you normalize over uneven episode lengths?
# locally compress/average local values?
# def human_normalied_aggr_history(envs, env_aggregation='', comp_var_aggregation='', comparison_variable:str= "do_shape", baseline_value=False):
#     """plot_human_normalized_env_histories but aggregated over all environments"""
#     compared_envs = calculate_advangate(envs, aggregation=comp_var_aggregation, comparison_variable=comparison_variable,
#                                         baseline_value=baseline_value)
#     env_aggr_fn = aggreation_to_fn[env_aggregation]
#     plt.figure(figsize=(12, 8))
#
#     vals = np.array(list(compared_envs.values()))
#     aggr_hist = env_aggr_fn(np.array(list(compared_envs.values())), axis=0)
#     aggr_hist_idxs =
#     plt.plot(x_axis, aggr_hist, label='baseline')


# def human_normalized_best_history(compared_envs, aggreation='', comparison_variable:str="do_shape", baseline_value=False):
#     # Now pick the best shape scale value for each environment before aggregating:
#     finetuned_data = {}
#     for env_id, env_stat in compared_envs.items():
#         best_shape_scale = None
#         best_adv = float('-inf')
#         for key in env_stat:
#             try:
#                 human_norm = env_stat[key]['human_normalized']
#                 adv = np.nanmean(human_norm)
#                 if adv > best_adv:
#                     best_adv = adv
#                     best_shape_scale = key
#                     finetuned_data[env_id] = human_norm
#             except KeyError:
#                 print(f"Skipping {env_id} for {key}")
#                 continue
#
#         if best_shape_scale is not None:
#             print(f"Best shape scale for {env_id} is {best_shape_scale}")
#         else:
#             print(f"No valid shape scale found for {env_id}")
#
#     # Calculate median human normalized score
#     median_human_normalized = np.nanmedian(np.array(list(finetuned_data.values())), axis=0)
#
#     plt.plot(x_axis, median_human_normalized, label=r'Finetuned $\eta$ for each env.')
#     plt.legend(loc='upper left')
#     plt.title("Finetuned Aggregated Reward Curve")
#     plt.xlabel('Env. Steps')
#     plt.ylabel(f'{aggreation} Human Normalized Score')
#     plt.tight_layout()
#     plt.savefig(f"finetuned_{path}_{comparison_variable}.png")
#     plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="atari10m")  # logs/tb/runs
    parser.add_argument("--comparison_variable", type=str, default="shape_scale")
    parser.add_argument("--handler_type", type=str, default="wandb")
    parser.add_argument("--aggregation", type=str, default="max")
    args = parser.parse_args()
    filters = {}
    value_variable = "eval/mean_reward"
    baseline_value = False
    cache = False
    use_cached = True
    compared_envs = parase_env_reward_data(
        args.project, args.comparison_variable,
        value_variable=value_variable,
        cache=cache,
        use_cached=use_cached,
        handler_type=args.handler_type,
        filters=filters
    )
    plot_advantage(compared_envs)