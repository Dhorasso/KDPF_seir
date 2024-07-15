

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