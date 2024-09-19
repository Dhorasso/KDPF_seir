# This file contain the code for a SEIR modelto track epidemic with
# constant transmissio rate,
# time-varying transmission rate 
# and SEIR model for COVID-19 data of Ireland 

import numpy as np

def seir_model_const(y, theta, theta_names, dt=1):
    """
    Discrete-time stochastic SEIR model.

    Parameters:
    - y: Vector of variables [S, E, I, R, NI]
            S: susceptible
            E: Exposed
            I: Infected
            R: Recovered
            NI: New infected  (used to link with the observations)
    - theta: Set of parameters
    - theta_names: Name of the parameters:
            beta: Transmission rate
            sigma: Latency rate
            gamma: Recovery rate

    Returns:
    - y_next: Updated vector of variables [S, E, I, R]
    """

    # Unpack variables
    S, E, I, R, NI= y
    N = S + E + I + R

    # Unpack parameters
    param= dict(zip(theta_names,theta))

    # Binomial distributions for transitions
    P_SE = 1 - np.exp(-param['beta'] * I / N * dt)  # Probability of transition from S to E
    P_EI = 1 - np.exp(-param['sigma'] * dt)      # Probability of transition from E to I
    P_IR = 1 - np.exp(-param['gamma'] * dt)      # Probability of transition from I to R


    # Binomial distributions for transitions
    B_SE = np.random.binomial(S, P_SE)
    B_EI = np.random.binomial(E, P_EI)
    B_IR = np.random.binomial(I, P_IR)

    # Update the compartments
    S += -B_SE
    E += B_SE - B_EI
    I += B_EI - B_IR
    R += B_IR
    NI= B_EI

    y_next = [max(0, compartment) for compartment in [S, E, I, R,  NI]] # Ensure non-negative elements
    return y_next



def seir_model_var(y, theta, theta_names, dt=1):
    """
    Discrete-time stochastic SEIR model.

    Parameters:
    - y: Vector of variables [S, E, I, R, NI, B]
            S: susceptible
            E: Exposed
            I: Infected
            R: Recovered
            NI: New infected  (used to link with the observations)
            B: Time-varying transmission rate
    - theta: Set of parameters
    - theta_names: Name of the parameters:
            sigma: Latency rate
            gamma: Recovery rate
            v_beta: volatility transmission rate ( default square value)

    Returns:
    - y_next: Updated vector of variables [S, E, I, R, NI, B]
    """

    # Unpack variables
    S, E, I, R, NI, B= y
    N = S + E + I + R

    # Unpack parameters
    param= dict(zip(theta_names,theta))
   

    # Binomial distributions for transitions
    P_SE = 1 - np.exp(-B * I / N * dt)  # Probability of transition from S to E
    P_EI = 1 - np.exp(-param['sigma'] * dt)      # Probability of transition from E to I
    P_IR = 1 - np.exp(-param['gamma'] * dt)      # Probability of transition from I to R

    # Binomial distributions for transitions
    B_SE = np.random.binomial(S, P_SE)
    B_EI = np.random.binomial(E, P_EI)
    B_IR = np.random.binomial(I, P_IR)

    # Update the compartments
    S += -B_SE
    E += B_SE - B_EI
    I += B_EI - B_IR
    R += B_IR
    B *= np.exp( param['v_beta']**0.5* np.random.normal(0, 1) * dt) 
    # or np.exp( param['v_beta']* np.random.normal(0, 1) * dt)  depending on prior of v_beta
    
    NI= B_EI

    y_next = [max(0, compartment) for compartment in [S, E, I, R,  NI, B]] # Ensure non-negative elements

    return y_next



def stochastic_model_covid(y, theta, theta_names, dt=1):
    """
    Discrete-time stochastic compartmental model for COVID-19 Ireland.

    Parameters:
    - y: Vector of compartments [S, E, Ips, Ias, Isi, Ist, Ipi, Isn, R, D, NI, B, d]
     (We ue D and NI variables to link with the observations)
    - parameters: Set of parameters [vb_2, v_d]
     
    Returns:
    - y_next: Updated vector of compartments [S, E, Ips, Ias, Isi, Ist, Ipi, Isn, R, D, NI, B, d]
    """

    # Unpack variables
    S, E, Ips, Ias, Isi, Ist, Ipi, Isn, R, CD, NI, B, d = y
    N = S + E + Ips + Ias + Isi + Ist + Ipi + Isn + R 

    # Convert d from logit to probability
    d = logit(d)

    # Unpack parameters
    param= dict(zip(theta_names, theta))
    

    # Model parameters from table
    r_as = 0.55
    r_si = 0.05
    r_pi = 0.05
    tau_c = 5.85
    tau_l = 4.9
    tau_d = 7
    tau_r = 3.52
    f_as = 0.2
    f_si = 0.1
    f_st = 0.8

    # Probabilities of transitions using binomial distributions
    P_Y1 = 1 - np.exp(-B * (Ips + r_as * Ias + r_si * Isi + Ist + r_pi * Ipi + Isn) * dt / N)
    P_Y2 = 1 - np.exp(-1 / tau_l * dt)
    P_Y3 = 1 - f_as
    P_Y4 = 1 - np.exp(-1 / (tau_c - tau_l) * dt)
    P_Y5 = f_si
    P_Y6 = f_st
    P_Y7 = 1 - np.exp(-1 / tau_d * dt)
    P_Y8 = 1 - np.exp(-1 / (tau_d - tau_c + tau_l) * dt)
    P_Y9 = 1 - np.exp(-1 / tau_r * dt)
    P_Y10 = 1 - np.exp(-1 / (tau_d - tau_c + tau_l - tau_r) * dt)
    P_Y11 = 1 - np.exp(-1 / (tau_d - tau_c + tau_l) * dt)

    # Binomial distributions for transitions
    Y1 = np.random.binomial(S, P_Y1)
    Y2 = np.random.binomial(E, P_Y2)
    Y3 = np.random.binomial(Y2, P_Y3)
    Y4 = np.random.binomial(Ips, P_Y4)
    Y5 = np.random.binomial(Y4, P_Y5)
    Y6 = np.random.binomial(Y4, P_Y6)
    Y7 = np.random.binomial(Ias, P_Y7)
    Y8 = np.random.binomial(Isi, P_Y8)
    Y9 = np.random.binomial(Ist, P_Y9)
    Y10 = np.random.binomial(Ipi, P_Y10)
    Y11 = np.random.binomial(Isn, P_Y11)

    # Update the compartments
    S -= Y1
    E += Y1 - Y2
    Ips += Y3 - Y4
    Ias += Y2 - Y3 - Y7
    Isi += Y5 - Y8
    Ist += Y6 - Y9
    Ipi += Y9 - Y10
    Isn += Y4 - Y5 - Y6 - Y11
    R += Y7 + Y8 + Y10 + Y11
    CD += np.random.binomial(R, invlogit(d))
    NI = Y9

    # Update parameters B and d with stochastic variation
    B = B * np.exp(param['nu_beta_2'] ** 0.5 * np.random.normal(0, 1) * dt)
    d = invlogit(d + param['nu_d_2'] ** 0.5 * np.random.normal(0, 1) * dt)

    # Ensure all compartments are non-negative
    y_next = [max(0, compartment) for compartment in [S, E, Ips, Ias, Isi, Ist, Ipi, Isn, R, CD, NI, B, d]]

    return y_next
