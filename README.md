# Simulation of charging and discharging EV recommendations with predictions

This repository contains the codebase developed for a Master's Thesis focused on optimizing smart charging behavior of electric vehicles (EVs) within renewable energy communities (RECs). The project models EV energy flows, predicts energy needs and charging behaviors, and implements smart charging strategies to enhance community self-sufficiency. The final phase involves simulating various charging and discharging scenarios to maximize the self-consumption of renewable energy.
## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Simulations](#simulations)
- [Running Simulations](#run-the-simulations)
- [Tests](#tests)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/victoriavrl/optimizingC2V2C-predictions_and_simulations.git
    cd optimizingC2V2C-predictions_and_simulations
    ```

2. Create a virtual environment and activate it:
   - **On Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
Note : a previous version of Pandas needs to be used as the current one is not compatible with the package `emobpy`. 
This third step is then very important.
## Project structure 

- `datasets_generation/`: jupyter notebooks and modified data files from `emobpy` package used for the generation of the datasets
- `predictions/` : source code for the prediction of the energy needs and charging behaviors of the EVs
- `simulations/`: source code for the simulation of the charging and discharging recommendations
- `tests/`: unit tests for the source code

## Predictions 

- `clustering_1/`: alternative clustering method
- `data/`: all datasets used for predictions
- `preprocessing.py`: dataset preprocessing script
- `models/` : models used for the predictions
- `next_CBS/` : prediction of the next consumption between sessions
- `next_dest/` : prediction of the next charging destination
- `plug-out-time/` : prediction of the plug-out time
- `preprocessing.py` : preprocessing of the datasets
- `sessions_clustering.py` : clustering of the sessions
- `users_clustering.py` : clustering of the users

## Simulations

### Directory structure
- `data_ev/` : contains the data of the EVs used for the simulations 
- `data_REC/` : contains the data of the REC used for the simulations
- `models/` : contains the EV objects and charging recommendations algorithm 
- `utils/` : contains the utility functions used for the simulations (predictions, metrics, initialization of the EVs, etc.)
- `main.py` : main file to run the simulations

### Description
4 different simulations were implemented :
- **Smart**: EVs are charged and discharged based on the predictions of their energy needs and charging behaviors. The predictions are made using a machine learning model trained on historical data.
- **Non-smart**: EVs are charged as soon as they arrive to a charge station.
- **Non-smart no public**: EVs are charged as soon as they arrive to a charge station, but they do not charge at public places. 
- **Smart oracle**: EVs are charged and discharged based on the same recommendation algorithm than the **Smart** but the predictions are considered perfect.

### Run the simulations

First, you need to navigate to the `simulations` directory. You can do this by running the following command in your terminal:


```bash 
cd simulations
```

Then, you can run the simulations with the following command:

```bash

python main.py --mode smart  # Only run smart simulations

python main.py --mode non_smart  # Only run non-smart simulations

python main.py --mode both  # Run both smart and non-smart (default if you omit --mode)

python main.py --mode non_smart_no_public  # Run non-smart simulations when the car do not charge at public places

python main.py --mode smart_oracle  # Run smart simulations with oracle i.e. perfect predictions
```

## Tests

To run the tests, you can use the following command:

```bash
cd tests
pytest tests.py
```

Tests validate the results of previous simulations.
Therefore, you must run at least one simulation before executing the tests.
Running tests before any simulation will result in failures.