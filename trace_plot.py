

import matplotlib.pyplot as plt
from plotnine import*
import numpy as np
def replace_outliers(matrix):
    # Calculate the IQR for each column
    q1 = np.percentile(matrix, 25, axis=0)
    q3 = np.percentile(matrix, 75, axis=0)
    iqr = q3 - q1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Replace outliers with appropriate values
    for col in range(matrix.shape[1]):
        col_values = matrix[:, col]
        col_non_outliers = col_values[(col_values >= lower_bound[col]) & (col_values <= upper_bound[col])]
        max_non_outlier = np.max(col_non_outliers)
        min_non_outlier = np.min(col_non_outliers)
        
        # Replace outliers above the upper bound
        outliers_above = col_values[col_values > upper_bound[col]]
        if len(outliers_above) > 0:
            matrix[col_values > upper_bound[col], col] = max_non_outlier
        
        # Replace outliers below the lower bound
        outliers_below = col_values[col_values < lower_bound[col]]
        if len(outliers_below) > 0:
            matrix[col_values < lower_bound[col], col] = min_non_outlier
    
    return matrix



def trace_smc(Traject):

    matrix_dict = {}
    stateName=list(Traject[0].columns[1:])
    # Iterate through each state name
    for state in stateName:
      # Extract matrices for each state from all dataframes
      state_matrices = [df[state].values.reshape(1, -1) for df in Traject]
      # Concatenate matrices horizontally
      combined_matrix = np.concatenate(state_matrices, axis=1)
      # Reshape the combined matrix based on the shape of the original dataframe
      reshaped_matrix = combined_matrix.reshape(-1,Traject[0].shape[0])
      # Store the reshaped matrix in the dictionary with state name as the key
      matrix_dict[state] = reshaped_matrix 
    return matrix_dict


def plot_smc(matrix, color='dodgerblue', med=False):
    # matrix=replace_outliers(matrix)
    T = matrix.shape[1]

    # Calculate the median along the columns (axis=0)
    median_values = np.median(matrix, axis=0)
    # Calculate the 95% and 50% credible intervals
    credible_interval_95 = np.percentile(matrix, [2.5, 97.5], axis=0)
    credible_interval_50 = np.percentile(matrix, [25, 75], axis=0)
    # Plotting the time evolution of the median
    time_steps = np.arange(T)  # Assuming time steps are represented by the columns
    # Create a ggplot object
    p = ggplot() 
    # Add a ribbon layer for the 50% credible interval
    p += geom_ribbon(aes(x=time_steps, ymin=credible_interval_50[0], ymax=credible_interval_50[1]), fill=color, alpha=1)
    # Add a ribbon layer for the 95% credible interval
    p += geom_ribbon(aes(x=time_steps, ymin=credible_interval_95[0], ymax=credible_interval_95[1]), fill=color, alpha=0.35)
    p += geom_line(aes(x=time_steps, y=median_values), color='k',size=1.2)
    return p

def trace_smc_covid(Traject):
    matrix_dict = {}
    stateName = list(Traject[0].columns[1:])  # Assuming Traject is a list of DataFrames with state variables
    # Iterate through each state name
    for state in stateName:
        # Extract matrices for each state from all dataframes
        state_matrices = [df[state].values.reshape(1, -1) for df in Traject]
        # Concatenate matrices horizontally
        combined_matrix = np.concatenate(state_matrices, axis=0)
        # Reshape the combined matrix based on the shape of the original dataframe
        reshaped_matrix = combined_matrix.reshape(-1, Traject[0].shape[0])
        # Store the reshaped matrix in the dictionary with state name as the key
        matrix_dict[state] = reshaped_matrix
    # Calculate variables from the model parameters
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
    Npop = 4965439

    # Calculate Rt using the formula provided
    Rt = matrix_dict['B'] * ((f_as - 1) * ((r_si - 1) * f_si * (tau_c - tau_l) + (r_pi - 1) * f_st * (tau_c - tau_l + tau_r)) \
                             + tau_d * (f_as * (r_as - r_si * f_si - r_pi * f_st + f_si + f_st - 1) \
                                        + (r_si - 1) * f_si + (r_pi - 1) * f_st + 1)) \
         * matrix_dict['S'] / Npop
    # Add Rt to the matrix_dict
    matrix_dict['Rt'] = Rt

    return matrix_dict

def  plot_smc_covid(matrix,Date, color='dodgerblue', med=False):
    matrix=replace_outliers(matrix)
    T = matrix.shape[1]

    # Calculate the median along the columns (axis=0)
    median_values = np.median(matrix, axis=0)

    # Calculate the 95% and 50% credible intervals
    credible_interval_95 = np.percentile(matrix, [2.5, 97.5], axis=0)
    credible_interval_50 = np.percentile(matrix, [25, 75], axis=0)
    time_steps=Date
    # Create a ggplot object
    p = ggplot() 
    # Add a ribbon layer for the 50% credible interval
    p += geom_ribbon(aes(x=time_steps, ymin=credible_interval_50[0], ymax=credible_interval_50[1]), fill=color, alpha=1)

    # Add a ribbon layer for the 95% credible interval
    p += geom_ribbon(aes(x=time_steps, ymin=credible_interval_95[0], ymax=credible_interval_95[1]), fill=color, alpha=0.35)
    p +=scale_x_date(date_breaks='1 months', date_labels='%b')
    p += theme(figure_size=(10, 4.5),  # Adjust the figure size as needed
               axis_text_x=element_text(size=12),  # Increase x-axis label size
               axis_text_y=element_text(size=12),  # Increase y-axis label size
               axis_title=element_text(size=14))  
    # p+= theme(axis_text_x=element_text(rotation=90, hjust=1))

    p += geom_line(aes(x=time_steps, y=median_values), color='k',size=1.2)
    return p
