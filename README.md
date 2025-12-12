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
We have written a script to pull the relevant data from Binance's API into the ```data``` directory. Run the following command in the
terminal being at the root of the repository.

```bash
python ???/get_data.py 
```
