# LWO

This repository is the official implementation for the paper **Leave a Window Out: Modifying the Jackknife for Predictive Inference in Time Series**.

It contains experiment scripts for comparing conformal prediction methods on simulated and real time-series data.

The implemented methods are:

- Split conformal prediction (Split CP)
- Jackknife
- Leave-window-out Jackknife (LWO)

The scripts print coverage and interval width summaries to the terminal. They
do not save files or generate plots.

## Files

- `utils.py`: shared helper functions for conformal prediction, model fitting,
  prediction, quantiles, and summary statistics.
- `ma1_LWO.py`: multivariate MA(1) simulation with vector-valued responses.
- `real_data_LWO.py`: real-data time-series experiment using chunked one-step
  prediction tasks.
- `sticky_markov_chain_LWO.py`: sticky Markov-chain counterexample.

## Installation

Use Python 3.9 or newer. Install the required packages with:

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

`matplotlib` and `scipy` are included because `utils.py` contains some legacy
utility code. The cleaned experiment scripts themselves only print summaries.

## Running the Experiments

Run each script directly from the repository folder.

```bash
python ma1_LWO.py
```

```bash
python real_data_LWO.py
```

```bash
python sticky_markov_chain_LWO.py
```

Each script has an editable configuration block inside `main()`. Change those
values to adjust sample size, lag, predictor, number of trials, alpha level, or
LWO window size.

## Real Data Layout

`real_data_LWO.py` expects datasets inside a `real_data/` folder next to the
script. For example:

```text
.
├── real_data_LWO.py
├── utils.py
└── real_data/
    ├── traffic.txt
    └── solar_AL.txt
```

The default dataset setting is:

```python
dataset_name = "traffic.txt"
```

You can change the dataset, columns, lag, history length, and chunk gap in
`RealDataConfig` inside `real_data_LWO.py`.

Source of real data: 

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. 2018. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. In SIGIR '18: The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval, July 8–12, 2018, Ann Arbor, MI, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3209978.3210006

https://github.com/laiguokun/multivariate-time-series-data


## Predictors

All experiment scripts support the same five predictors:

- `ridge`
- `knn`
- `kernel`
- `mlp`
- `dt`

## Output

The scripts report empirical coverage and interval width. For the multivariate
MA(1) experiment, the interval is a Euclidean ball and the reported width is the
ball diameter.

Example output format:

```text
Summary
Method                     Coverage Mean    Coverage SE     Width Mean    Width SE
Split CP                   0.9000           0.0100          1.2345        0.0200
Jackknife                  0.9100           0.0090          1.3456        0.0180
LWO jackknife              0.9200           0.0080          1.4567        0.0170
```

## Notes

The real-data and MA(1) scripts share generic CP utilities from `utils.py`.
The sticky Markov-chain experiment is slightly different because its CP methods
rebuild repeat-count features inside each leave-one-out or leave-window-out
subset.
