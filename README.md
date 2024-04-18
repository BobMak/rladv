# rladv

A simple plotting utility for algorithm performance comparison in reinforcement learning benchmarks.
Plot the relative advantage of your algorithm against your baseline.

## Installation

`pip install wandb tqdm rladv`

## Usage

1. Run your algorithm and a baseline on a benchmark, logging the target performance metric to wandb.

2. Generate the plot:
```python
import rladv
rladv.plot_advantage(project="my_project", comparison_variable="my_metric", cache=True, use_cached=True)
```

Example Plot: 

![Example Plot](http://imgur.com/uY0EuU5.png)

