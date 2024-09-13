#  This file contain the code to draw initial particle and solve the SEIR model

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm, truncnorm, gamma, invgamma

def draw_value(lower, upper, mean, std, distribution):
    """
    Draw a random value from a specified distribution.

    Parameters:
    - lower: Lower bound for uniform, gamma, and invgamma distributions
    - upper: Upper bound for uniform, gamma, and invgamma distributions
    - mean: Mean for normal and lognormal distributions
    - std: Standard deviation for normal and lognormal distributions
    - distribution: Type of distribution ('uniform', 'uniform_logit', 
                    'normal', 'trunorm' 'lognormal', 'gamma', 'invgamma')

    Returns:
    - value: Drawn value from the specified distribution
    """

    if distribution == 'uniform':
        return np.random.uniform(lower, upper)
    elif distribution == 'uniform_logit':
        x=np.random.uniform(lower, upper)
        return x/(1-x)
    elif distribution == 'normal':
        return  np.random.normal(mean,std) 
    elif distribution == 'lognormal':
        return np.random.lognormal(mean,std) # to adjust later
    elif distribution == 'gamma':
        return gamma.rvs(lower,upper)
    elif distribution == 'invgamma':
        return invgamma.rvs(lower,upper)
    elif distribution == 'truncnorm':
        # Calculate the a and b values for truncnorm
        a, b = (lower - mean) / std, (upper - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)
    else:
        raise ValueError("Invalid distribution")
        

def initialization_state_theta(state_info, theta_info, num_particles, population_size=1000):
    """
    Initialize the state and parameter particles for the particle filter.

    Parameters:
    - state_info: Dictionary containing information about initial states
    - theta_info: Dictionary containing information about parameters
    - num_particles: Number of particles
    - population_size: Total population size (default is 1000)

    Returns:
    - result_dict: Dictionary containing initial particles, state names, and parameter names
    """
    state_names = list(state_info.keys())
    theta_names= list(theta_info.keys())
    current_particles = []

    for _ in range(num_particles):
        # Initialize state values, ensuring 'S' compartment accounts for the total population size
        init_state_values = [draw_value(*state_info[state]['prior']) for state in state_names[1:]]
        # Initialize parameter values
        param_values = [np.log(draw_value(*theta_info[param]['prior'])) for param in theta_names]

        # Combine state and parameter values into a single particle
        current_particles.append(np.array(init_state_values + param_values))

    result_dict = {
        'currentStateParticles': current_particles,
        'stateName': state_names,
        'thetaName': theta_names
    }

    return result_dict


def solve_model(model, theta, initial_state, state_names, theta_names, t_start, t_end, dt=1):
    """
    Solve a stochastic disease model using the Euler method.

    Parameters:
    - model: Model function (e.g., SIR stochastic model)
    - parameters: Set of parameters for the model
    - initial_state: Initial conditions for the model
    - t_start: Start time
    - t_end: End time
    - dt: Time step (default is 1)

    Returns:
    - results_df: DataFrame containing time points and model solutions 
    """
    theta = np.exp(theta)  #  orginal value of  the parameters
    t_points = np.arange(t_start, t_end + dt, dt)  # Time points
    num_steps = len(t_points)

    # Initialize array to store results
    results = np.zeros((num_steps, len(initial_state)))
    results[0, :] = initial_state

    # Simulate the model using the Euler method
    for i in range(1, num_steps):
        current_state = np.array(results[i - 1, :])  # Current state
        next_state = model(current_state, theta, theta_names, dt)  # Compute next state
        results[i, :] = next_state  # Store next state

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results, columns=state_names)
    results_df['time'] = t_points

    # If simulating for a single time step, extract the last row
    if t_end - t_start == 0 or t_end - t_start == dt:
        results_df = results_df.iloc[-1]  # Extract the solution at time t

    return results_df
