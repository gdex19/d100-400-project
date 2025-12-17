# D100/D400 Final Project

# Setup

## Environment

Install the required packages by running:

```bash
conda env create -f environment.yml
```

Activate the environment by running:

```bash
conda activate final-project
```

## Pre-commit

This repository uses pre-commit to enforce consistent code. To install
pre-commit, run:

```bash
pre-commit install
```

To run the checks manually, run:

```bash
pre-commit run --all-files
```

# Getting the data
We have written a script to pull the relevant data from Binance's API into the ```data``` directory. Run the following command
in the command line to download data:
```bash
download-data --start_date START_DATE
```
where ```START_DATE``` is of the form YYYY/MM/DD. For our analysis, we used data from 2023/11/01 onwards. This may take a few minutes.

# Reproducing the analysis
After pulling the data as above, run the eda_cleaning.ipynb to see the exploratory data analysis and save the cleaned
data. Next, run model_training.py to select hyperparameters and save the best results. Finally, run mode_evaluation.ipynb to train the models with these parameters and evaluate the models on the validation set.

If you want to run the rpo outside of the root folder, first run the following from the root folder:
```bash
pip install -e .
```
