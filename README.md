# Stochastic-SEIR-model-with-SMC
Implementation of a stochastic SEIR model using a kernel density particle filter for  oline state and parameter estimation. This repository includes detailed simulations using synthetics and real data.

## About
This repository contains the implementation of a stochastic model to simulate and forecast the spread of COVID-19 in Ireland. The model is based on compartmental epidemiological structures and uses a sequeltial monte carlo to estimate and update state trajectories and model parameters over time.

## Table of Contents
- [Introduction](#introduction)
- [Model Description](#model-description)
- [Installation](#installation)
- [Model Parameters](#model-parameters)
- [Model Outputs](#model-outputs)
- [Example Usage](#example-usage)

## Introduction
The COVID-19 Stochastic Model aims to provide a detailed simulation of the virus's spread within the population of Ireland. By incorporating real data and using a particle filter, the model can produce accurate state estimates and time-varying reproduction number

## Model Description
For a simple SEIR model, the modelled states are: susceptibles (S), exposed (E), infected (I) and removed (R).

$$
\begin{aligned}
S_{t+ \delta t} &= S_{t} - Y_{SE}(t), & Y_{SE}(t) &\sim \mathrm{Bin}\left(S_{t}, 1-e^{-\beta \frac{ I_{t}}{N} \delta t}\right) \\
E_{t+ \delta t} &= E_{t} + Y_{SE}(t) - Y_{EI}(t), & Y_{EI}(t) &\sim \mathrm{Bin}\left(E_{t}, 1-e^{-\sigma \delta t}\right) \\
I_{t+ \delta t} &= I_{t} + Y_{EI}(t) -  Y_{IR}(t), & Y_{IR}(t) &\sim \mathrm{Bin}\left(I_{t}, 1-e^{-\gamma \delta t}\right) \\
R_{t+ \delta t} &= R_{t} + Y_{IR}(t). &
\end{aligned}
$$

When there is not many vraiation in the data, the observation of daily new infections is modeled using the Poisson distribution, which offers an intuitive interpretation for generating daily count events on a given day, denoted as $y_{t}|x_{t}\sim\mathrm{Poisson}(Y_{EI}(t))$.

## Installation
To install and set up the environment for running this model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Dhorasso/Stochastic-SEIR-model-with-SMC.git
    cd covid19-stochastic-model-ireland
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


##  Kernel_Smoothing_Filter Inputs

Perform Sequential Monte Carlo (Particle Filter) for state-space models.

- `model`: Model function (e.g., SIR, SEIR, extended-SEIR stochastic model)
- `initial_state_info`: Information about the initial state of the system  (dictionary)
- `initial_theta_info`: Initial parameters information  (dictionary)
- `observed_data`: Observed data (a DataFrame)
- `num_particles`: Number of particles 
- `resampling_threshold`: Threshold for effective sample size in resampling  
- `delta`: Parameter for updating theta during resampling  
- `population_size`: Total population size  
- `resampling_method`: Method for particle resampling ('stratified' by default)  
- `observation_distribution`: Distribution of observations ('poisson by default) 
- `forecast_days`: Number of days to forecast
- `num_cores`: Number of processor to be used in parallel ( defaut all available -1) 
- `show_progress`: Whether to display a progress bar during computation  (default is TRue)

#### Initial State Information (`initial_state_info`)

The `initial_state_info` dictionary should contain the initial state variables of the model. Each state variable should be defined with the following information:
- `state name `and  `prior distribution`: A list specifying `[lower_bound, upper_bound, mean, std_deviation, distribution_type]`. The distribution type can be 'fixed' if the value is known or 'uniform' if it follows a uniform distribution.
#### Initial Parameters Information (`initial_theta_info`)

 The initial_theta_info dictionary should contain the initial parameters of the model. Each parameter should be defined with the following information:

- `parameter name` and `prior distribution`: A list specifying `[lower_bound/shape, upper_bound/scale, mean, std_deviation, distribution_type]`. The distribution can be 'uniform',  'normal', 'lognormal', 'gamma', 'invgamma'.

##  Model outputs 
- `margLogLike`: Marginal log-likelihood of the observed data given the model.
- `trajState`: Trajectories of the state variables over time.
- `trajtheta`: Trajectories of the model parameters over time.

## Example Usage
Below is an example of how to use the Particle Filter with the stochastic model for the COVID-19 Case in Ireland:

```python
# Import necessary modules
import numpy as np
import pandas as pd
from joblib import Parallel, delayed  # For parallel computing
from tqdm import tqdm                 # For Display the progress
from stochastic_epidemic_model import seir_model_const, seir_model_var, stochastic_model_covid
from particle_filter import Kernel_Smoothing_Filter
from trace_plot import trace_smc, trace_smc_covid, plot_smc, plot_smc_covid

# Example data
data = pd.read_csv('covid19_ireland_data.csv')  # Replace with actual data file

# Define initial state information
state_info = {
    'S': {'prior': [0, 0, 0,0, 'fixed']},  # 'fixed' indicates that it's calculated based on the others
    'E': {'prior': [0, 0, 0,0, 'uniform']},
    'I': {'prior': [0,3, 0,0, 'uniform']},
    'R': {'prior': [0, 0, 0,0, 'uniform']},
     'NI': {'prior': [0, 0, 0,0, 'uniform']}
}

# Define initial parameters information
theta_info = {
    'beta': {'prior': [0.1, 0.6,0,0, 'uniform']},
    'sigma': {'prior': [1/14, 1/2,0,0, 'uniform']},
    'gamma': {'prior': [1/15, 1/2,0,0, 'uniform']}
}



# Specify the seed for reproductibility
np.random.seed(123)

## Generate simulated data
theta_example = [0.45, 1/3, 1/5]
InitialState_example = [5999, 0, 1, 0, 0]
state_names = ['S', 'E', 'I', 'R', 'NI']
t_start_example = 0
t_end_example = 120
dt_example = 1


np.random.seed(123)
results_example = solve_seir_const_beta(seir_const_beta, theta_example, InitialState_example, t_start_example, t_end_example, dt_example)
simulated_data = pd.DataFrame({'time': results_example['time'], 'obs': results_example['NI']})

results_filter = Kernel_Smoothing_Filter(
    model=seir_model_const, 
    initial_state_info=state_info , 
    initial_theta_info=theta_info , 
    observed_data=simulated_data, 
    num_particles=10000, 
    observation_distribution='poisson',
    num_cores =-1 
    show_progress=True
)

```
