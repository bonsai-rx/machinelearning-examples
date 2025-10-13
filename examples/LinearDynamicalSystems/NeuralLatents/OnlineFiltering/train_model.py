import numpy as np
import ssm.learning
import os

base_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "..", "..", "..", "datasets")

print("Loading training data...")

# load training data
training_data_file = os.path.join(base_dir, "sub-Jenkins_ses-small_desc-train_behavior+ecephys.bin")

n_clusters = 142
transformed_binned_spikes = np.fromfile(training_data_file, dtype=float).reshape(-1, n_clusters).T

print("Initializing model...")

# model
n_latents = 10

# estimation initial conditions
sigma_B = 0.1
sigma_Z = 0.1
sigma_Q = 0.1
sigma_R = 0.1
sigma_m0 = 0.1
sigma_V0 = 0.1

# estimation parameters
max_iter = 100
tol = 0.1
vars_to_estimate = {"B": True, "Q": True, "Z": True, "R": True,
                    "m0": True, "V0": True, }

np.random.seed(0)

B0 = np.diag(np.random.normal(loc=0, scale=sigma_B, size=n_latents))
Z0 = np.random.normal(loc=0, scale=sigma_Z, size=(n_clusters, n_latents))
Q0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_Q, size=n_latents)))
R0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_R, size=n_clusters)))
m0_0 = np.random.normal(loc=0, scale=sigma_m0, size=n_latents)
V0_0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_V0, size=n_latents)))

print("Estimating parameters...")

optim_res = ssm.learning.em_SS_LDS(
    y=transformed_binned_spikes, B0=B0, Q0=Q0, Z0=Z0, R0=R0,
    m0_0=m0_0, V0_0=V0_0, max_iter=max_iter, tol=tol,
    vars_to_estimate=vars_to_estimate,
)

# Save estimated parameters
optim_res["B"].astype(float).tofile(os.path.join(base_dir, "transition_matrix.bin"))
optim_res["Z"].astype(float).tofile(os.path.join(base_dir, "measurement_function.bin"))
optim_res["Q"].astype(float).tofile(os.path.join(base_dir, "process_noise_covariance.bin"))
optim_res["R"].astype(float).tofile(os.path.join(base_dir, "measurement_noise_covariance.bin"))
optim_res["m0"].astype(float).tofile(os.path.join(base_dir, "initial_state_mean.bin"))
optim_res["V0"].astype(float).tofile(os.path.join(base_dir, "initial_state_covariance.bin"))

print(f"Estimated parameters saved to {os.path.realpath(base_dir)}")