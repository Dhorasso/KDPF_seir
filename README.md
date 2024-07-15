# Stochastic-SEIR-model-with-SMC
Implementation of a stochastic SEIR model using a kernel density particle filter for state and parameter estimation. Includes detailed simulations using synthetics and real data.

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


##  Kernel_Smoothing_Filter Parameters

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
 -`show_progress`: Whether to display a progress bar during computation  (default is TRue)

#### Initial State Information (`initial_state_info`)

The `initial_state_info` dictionary should contain the initial state variables of the model. Each state variable should be defined with the following information:
- `state name `and  `prior distribution`: A list specifying `[lower_bound, upper_bound, mean, std_deviation, distribution_type]`. The distribution type can be 'fixed' if the value is known or 'uniform' if it follows a uniform distribution.
#### Initial Parameters Information (`initial_theta_info`)

 The initial_theta_info dictionary should contain the initial parameters of the model. Each parameter should be defined with the following information:

-`parameter name` and `prior distribution`: A list specifying `[lower_bound/shape, upper_bound/scale, mean, std_deviation, distribution_type]`. The distribution can be 'uniform',  'normal', 'lognormal', 'gamma', 'invgamma'.

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
from joblib import Parallel, delayed
from tqdm import tqdm
from stochastic_epidemic_model import seir_model_const, seir_model_var, stochastic_model_covid
from filter_preprocessing import initialization_state_theta, solve_model
from weight_processing import resampling_style, compute_log_weight
from particle_filter import Kernel_Smoothing_Filter
from trace_plot import trace_smc, trace_smc_covid,  plot_smc,  plot_smc_covid

# Example data
data = pd.read_csv('covid19_ireland_data.csv')  # Replace with actual data file

# Define initial state information
state_info = {
    'S': {'prior': [0, 0, 0, 0, 'fixed']},
    'E': {'prior': [1, 1, 0, 0, 'uniform']},
    'Ips': {'prior': [15, 100, 0, 0, 'uniform']},
    'Ias': {'prior': [0, 0, 0, 0, 'uniform']},
    'Isi': {'prior': [0, 0, 0, 0, 'uniform']},
    'Ist': {'prior': [0, 0, 0, 0, 'uniform']},
    'Ipi': {'prior': [0, 0, 0, 0, 'uniform']},
    'Isn': {'prior': [0, 0, 0, 0, 'uniform']},
    'R': {'prior': [0, 0, 0, 0, 'uniform']},
    'CD': {'prior': [0, 0, 0, 0, 'uniform']},
    'NI': {'prior': [0, 0, 0, 0, 'uniform']},
    'B': {'prior': [0.7, 0.8, 0, 0, 'uniform']},
    'd': {'prior': [0.001, 0.002, 0, 0, 'uniform']}
}

# Define initial parameters information
theta_info = {
    'nu_beta_2':  {'prior': [80, 0.02, 0, 0, 'invgamma']},
    'nu_d_2': {'prior': [80, 0.01, 0, 0, 'invgamma']},
    'phi': {'prior': [30, 0.2, 0, 0, 'invgamma']}
}

# specify the seed for reproductibility
np.random.seed(123)
results_filter = Kernel_Smoothing_Filter(
    model=stochastic_model_covid, 
    initial_state_info=state_info, 
    initial_theta_info=theta_info,  
    observed_data=data, 
    num_state_particles=20000, 
    resampling_threshold=1, 
    delta=0.99, 
    population_size=4965439, 
    resampling_method='stratified', 
    observation_distribution='normal_approx_negative_binomial',
    forecast_days=28, 
    show_progress=True
)
```
