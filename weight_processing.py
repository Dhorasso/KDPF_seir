import numpy as np
from scipy.stats import poisson, norm, nbinom
import warnings



def resampling_style(weights, name_method):
    """ Performs resampling algorithms used by particle filters based on the chosen method.

    Parameters
    ----------
    weights : list-like of float
        List of weights as floats.
    nParticles : int
        Number of particles.
    name_method : str
        Name of the resampling method to be used. Should be one of: 'residual', 'stratified', 'systematic', 'multinomial'.

    Returns
    -------
    indexes : ndarray of ints
        Array of indexes into the weights defining the resample. i.e. the index of the zeroth resample is indexes[0], etc.

    Raises
    ------
    ValueError
        If the specified resampling method is not recognized.

    References
    ----------
    [1] J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
       systems. Journal of the American Statistical Association,
       93(443):1032–1044, 1998.
    [2] https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo
    """

    if name_method == 'residual':
        N = len(weights)
        indexes = np.zeros(N, 'i')

        # take int(N*w) copies of each weight, which ensures particles with the
        # same weight are drawn uniformly
        num_copies = (np.floor(nParticles * np.asarray(weights))).astype(int)
        k = 0
        for i in range(N):
            for _ in range(num_copies[i]):  # make n copies
                indexes[k] = i
                k += 1
                if k>N:
                    warnings.warn("Resampling failed: Index k exceeds N. Adjusting indexes.")
                    return np.arange(N) # Reset j to 0 to avoid out-of-bounds error

        # use multinormal resample on the residual to fill up the rest. This
        # maximizes the variance of the samples
        residual = weights - num_copies  # get fractional part
        residual /= sum(residual)  # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
        indexes[k:N] = np.searchsorted(cumulative_sum, np.random.rand(N - k))

        return indexes

    elif name_method == 'stratified':
        N = len(weights)
        # make N subdivisions, and chose a random position within each one
        positions = (np.random.rand(N) + range(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                if j >= N:
                    warnings.warn("Resampling failed: Index j exceeds N. Adjusting indexes.")
                    return np.arange(N) # Reset j to 0 to avoid out-of-bounds error
        return indexes

    elif name_method == 'systematic':
        N = len(weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.rand() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                if j >= N:
                    warnings.warn("Resampling failed: Index j exceeds N. Adjusting indexes.")
                    return np.arange(N) # Reset j to 0 to avoid out-of-bounds error
        return indexes

    elif name_method == 'multinomial':
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
        return np.searchsorted(cumulative_sum, np.random.rand(len(weights)))

    else:
        raise ValueError("Unknown resampling method. Please choose one of: 'residual', 'stratified', 'systematic', 'multinomial'.")




def compute_log_weight(observed_data_point, model_data_point, theta, theta_names, distribution_type):
    """

    Compute the log_likelihood of observing a single data point given the model's prediction.
    ***IMPORTANT*** user may need to change the column name of observed new case to 'obs'
    the computation of the weights can be modify depending on all the data sources aviable 
    Parameters:
    - observed_data_point: Observed number of infected individuals
    - model_data_point: Predicted number of infected individuals from the model
    - theta_params: Parameters of the model
    - theta_names: Names of the theta parameters
    - distribution_type: Type of distribution ('poisson' by default, also supports 'normal', 
                         'normal_approx_negative_binomial', or 'negative_binomial')

    Returns:
    - log_likelihood: log_likelihood of the observation
    """

    param = dict(zip(theta_names, np.exp(theta)))
    y=observed_data_point['obs']
 
    model_est_case =  model_data_point['NI']

    if distribution_type == 'poisson':
        log_likelihood = poisson.logpmf(y, mu= model_est_case)

    elif distribution_type == 'normal':
        epsi = 1e-4
        sigma_normal=param['phi']
        log_likelihood = norm.logpdf(np.log(epsi + y), 
                                     loc=np.log( model_est_case), 
                                     scale=sigma_normal)

    elif distribution_type == 'normal_approx_negative_binomial':
        y_death=observed_data_point['Death']
        model_est_death = model_data_point['CD']
        overdisperssion = param['phi']
        variance = model_est_case * (1 + overdisperssion * model_est_case)
        if variance < 1:
            variance = 1
        
        log_likelihood = norm.logpdf(y, loc=model_est_case, scale=np.sqrt(variance))+poisson.logpmf(y_death, mu= model_est_death)

    elif distribution_type == 'negative_binomial':
        overdisperssion = param['phi']
        p =1/ (1 + overdisperssion * model_est_case)
        n = 1 / overdisperssion
        log_likelihood = nbinom.logpmf(y, n, p)

    else:
        raise ValueError("Invalid distribution type")

    if np.isfinite(log_likelihood):

        return log_likelihood
    else:

        return -np.inf


