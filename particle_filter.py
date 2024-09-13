import numpy as np
from filter_preprocessing import initialization_state_theta, solve_model
from weight_processing import resampling_style, compute_log_weight

def Kernel_Smoothing_Filter(model, initial_state_info, initial_theta_info, observed_data, num_particles, 
                    resampling_threshold=1, delta=0.99, population_size=6000, 
                    resampling_method='stratified', observation_distribution='poisson', 
                    forecast_days=0, num_core=-1, show_progress=True):
    """
    Perform Sequential Monte Carlo (Particle Filter) for state-space models.

    INPUTS:
    - model: Model function (e.g., SIR stochastic model)
    - initial_state_info: Information about the initial state of the system
    - initial_theta_info: Initial parameters information
    - observed_data: Observed data (a DataFrame with 'time' column)
    - num_particles: Number of particles
    - resampling_threshold: Threshold for effective sample size (default is 1)
    - delta: Parameter for updating theta (default is 0.99)
    - population_size: Population size (default is 6000)
    - resampling_method: Resampling method ('stratified' by default)
    - observation_distribution: Distribution of observation ('poisson' by default)
    - forecast_days: Number of forecasting days (default is 14)
    - num_core: Number of processor to be used in parallel ( defaut all available -1) 
    - show_progress: Display progress bar (default is True)

    OUTPUTS:
    - margLogLike: Marginal log_likelihood
    - trajState: State trajectories
    - trajtheta: Parameters trajectories
    - ESS: Effective sample size
    """
    # Initialize marginal log_likelihood
    marginal_log_likelihood = 0

    # Initial state of particles
    initialization = initialization_state_theta(initial_state_info, initial_theta_info, num_particles, population_size)
    current_particles = np.array(initialization['currentStateParticles'])
    state_names = initialization['stateName']
    theta_names = initialization['thetaName']

    # Initialize filtered trajectories
    traj_theta = [{key: [] for key in ['time'] + theta_names} for k in range(num_particles)]
    traj_state = [{key: [] for key in ['time'] + state_names} for k in range(num_particles)]
    

    # Initialize particle weights
    particle_weights = np.ones(num_particles) / num_particles

    num_timesteps = len(observed_data)
    if show_progress:
        # Display a progress bar
        progress_bar = tqdm(total=num_timesteps + forecast_days, desc="Particle Filter Progress")

    
    # Kernel smoothing parameters
    h = np.sqrt(1 - ((3 * delta - 1) / (2 * delta)) ** 2)
    a = np.sqrt(1 - h ** 2)

    for t in range(num_timesteps + forecast_days):
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)
        if t < num_timesteps:
            current_data_point = observed_data.iloc[t]

        sample_theta = np.array([current_particles[j][len(state_names):] for j in range(num_particles)])
        theta_mean = np.mean(sample_theta, axis=0)
        theta_covariance = np.cov(sample_theta.T)

        # Update parcticles and weights
        def process_particle(j):
            state = current_particles[j][:len(state_names)]
            theta = current_particles[j][len(state_names):]
          
            # Kernel density ( θ(i_{t} ∼ N (α θ(i)_{t−1} + (1 − α) ̄θ_{t−1}, h**2*Vt−1) )
            if len(theta_mean) > 0 and t < num_timesteps:
                m = a * theta + (1 - a) * theta_mean
                if len(theta) > 1:
                    theta = np.random.multivariate_normal(m, h ** 2 * theta_covariance)
                else:
                    theta = np.random.normal(m, h * theta_covariance**0.5)

       
            trajectory = solve_model(model,theta, state, state_names, theta_names, t_start, t_end)
            model_point = trajectory[state_names]

            weight = particle_weights[j]
            if t < num_timesteps:
                weight = compute_log_weight(current_data_point, model_point, theta, theta_names, observation_distribution)

            return {'state': list(model_point), 'theta': list(theta), 'weight': weight}

        
        # Parrallel computing using all available worker (n_job=-1 user can choose to define the number of workers) 
        particles_update = Parallel(n_jobs=num_core)(delayed(process_particle)(j) for j in range(num_particles))

        current_particles = np.array([np.array(p['state'] + p['theta']) for p in particles_update])
        particle_weights = np.array([p['weight'] for p in particles_update])
      
        if t < num_timesteps:
            zt = max(np.mean(np.exp(particle_weights)), 1e-12)
            marginal_log_likelihood += np.log(zt)
            A = np.max(particle_weights)
            particle_weights_mod = np.ones(num_state_particles) if A < -1e2 else np.exp(particle_weights - A)

        
        normalized_weights =  particle_weights_mod / np.sum(particle_weights_mod)
      
        # Compute effective sample size
        effective_sample_size = 1 / np.sum(normalized_weights ** 2)

        # Resampling step
        if effective_sample_size < resampling_threshold * num_particles:
            resampled_indices = resampling_style(normalized_weights, resampling_method)
            current_particles = current_particles[resampled_indices]

        traj_state = [
            pd.DataFrame({'time': list(traj['time']) + [t], 
                          **{name: list(traj[name]) + [current_particles[j][:len(state_names)][i]] for i, name in enumerate(state_names)}})
            for j, traj in enumerate(traj_state)
        ]
        traj_theta = [
            pd.DataFrame({'time': list(traj['time']) + [t], 
                          **{name: list(traj[name]) + [np.exp(current_particles[j][len(state_names):][i])] for i, name in enumerate(theta_names)}})
            for j, traj in enumerate(traj_theta)
        ]

        if show_progress:
            progress_bar.update(1)

    if show_progress:
        progress_bar.close()

    return {'margLogLike': marginal_log_likelihood, 'trajState': traj_state, 'trajtheta': traj_theta, 'ESS': effective_sample_size, 'weights':  particle_weights}
    
