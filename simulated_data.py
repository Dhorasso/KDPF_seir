import numpy as np
import pandas as pd
################################################################
##### Generate data with constant transmission rate ############
################################################################
def seir_const_beta(y, theta, theta_names, dt=1):
    
    """
    Discrete-time stochastic SEIR model.

    Parameters:
    - y: Vector of variables [S, E, I, R]
    - theta: Set of parameters

    Returns:
    - y_next: Updated vector of variables [S, E, I, R]
    """

    # Unpack variables
    S, E, I, R, NI = y
    N = S + E + I + R

    # Unpack parameters
    beta, sigma, gamma = theta

    # Transition probabilities
    P_SE = 1 - np.exp(-beta * I/N * dt)       # Probability of transition from S to E
    P_EI = 1 - np.exp(-sigma * dt)            # Probability of transition from E to I
    P_IR = 1 - np.exp(-gamma * dt)            # Probability of transition from I to R

    # Binomial distributions for transitions
    B_SE = np.random.binomial(S, P_SE)
    B_EI = np.random.binomial(E, P_EI)
    B_IR = np.random.binomial(I, P_IR)

    # Update the compartments
    S -= B_SE
    E += B_SE - B_EI
    I += B_EI - B_IR
    R += B_IR
    NI = B_EI



    y_next = [max(0, compartment) for compartment in [S, E, I, R,  NI]] # Ensure non-negative elements

    return y_next


def solve_seir_const_beta(model, theta, InitialState, t_start, t_end, dt=1):

    """
    Solve the stochastic SEIR model using the Euler method.

    Parameters:
    - model: Model function (e.g., seirStoch_model2)
    - theta: Set of parameters for the model
    - InitialState: Initial conditions for the model
    - t_start: Start time
    - t_end: End time
    - dt: Time step

    Returns:
    - results_df: Dataframe containing time points and model solutions
    """

    # Initialize arrays to store results
    t_values = np.arange(t_start, t_end + dt, dt)
    S_values = np.zeros(len(t_values))
    E_values = np.zeros(len(t_values))
    I_values = np.zeros(len(t_values))
    R_values = np.zeros(len(t_values))
    NI_values = np.zeros(len(t_values))


    # Set initial conditions
    S_values[0], E_values[0], I_values[0], R_values[0], NI_values[0] = InitialState

    # Iterate over time steps using Euler method
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = [S_values[i - 1], E_values[i - 1], I_values[i - 1], R_values[i - 1], NI_values[i - 1]]
        y_next = model(y, theta, dt)
        S_values[i], E_values[i], I_values[i], R_values[i] , NI_values[i] = y_next

    # Create a dataframe to store results
    results_dict = {
        'time': t_values,
        'S': S_values,
        'E': E_values,
        'I': I_values,
        'R': R_values,
        'NI': NI_values
    }

    results_df = pd.DataFrame(results_dict)

    return results_df



################################################################
##### Generate data with time- varying transmission rate #######
################################################################


def seir_var_beta(y, theta, dt=1):

    """
    Discrete-time stochastic SEIR model.

    Parameters:
    - y: Vector of variables [S, E, I, R]
    - theta: Set of parameters
    
    Returns:
    - y_next: Updated vector of variables [S, E, I, R]
    """

   

    # Unpack variables
    t, S, E, I, R, NI, B = y
    N = S + E + I + R

    # Unpack parameters
    beta_0, sigma, gamma = theta

    # Binomial distributions for transitions
    P_SE = 1 - np.exp(-B * I / N * dt)  # Probability of transition from S to E
    P_EI = 1 - np.exp(-sigma * dt)      # Probability of transition from E to I
    P_IR = 1 - np.exp(-gamma * dt)      # Probability of transition from I to R

    # Binomial distributions for transitions
    B_SE = np.random.binomial(S, P_SE)
    B_EI = np.random.binomial(E, P_EI)
    B_IR = np.random.binomial(I, P_IR)

    # Update the compartments

    S -= B_SE
    E += B_SE - B_EI
    I += B_EI - B_IR
    R += B_IR
    NI = B_EI

    B = 0.5*(1+0.75*np.sin(0.2*t))
    t+=1

    y_next = [max(0, compartment) for compartment in [t, S, E, I, R,  NI,B]] # Ensure non-negative elements

    return y_next



def solve_seir_var_beta(model, theta, InitialState, t_start, t_end, dt=1):
    
    """
    Solve the stochastic SEIR model using the Euler method.

    Parameters:
    - model: Model function (e.g., seirStoch_model2)
    - theta: Set of parameters for the model
    - InitialState: Initial conditions for the model
    - t_start: Start time
    - t_end: End time
    - dt: Time step

    Returns:
    - results_df: Dataframe containing time points and model solutions
    """

    # Initialize arrays to store results
    t_values = np.arange(t_start, t_end + dt, dt)
    S_values = np.zeros(len(t_values))
    E_values = np.zeros(len(t_values))
    I_values = np.zeros(len(t_values))
    R_values = np.zeros(len(t_values))
    NI_values = np.zeros(len(t_values))
    Bt_values = np.zeros(len(t_values))

    # Set initial conditions
    t, S_values[0], E_values[0], I_values[0], R_values[0], NI_values[0] , Bt_values[0]= InitialState

    # Iterate over time steps using Euler method
    for i in range(1, len(t_values)):
        y = [t_values[i - 1], S_values[i - 1], E_values[i - 1], I_values[i - 1], R_values[i - 1], NI_values[i - 1], Bt_values[i - 1]]
        y_next = model(y, theta, dt)
        t_values[i], S_values[i], E_values[i], I_values[i], R_values[i] , NI_values[i], Bt_values[i] = y_next

    # Create a dataframe to store results
    results_dict = {
        'time': t_values,
        'S': S_values,
        'E': E_values,
        'I': I_values,
        'R': R_values,
        'NI': NI_values,
        'Bt': Bt_values
    }

    results_df = pd.DataFrame(results_dict)

    return results_df

