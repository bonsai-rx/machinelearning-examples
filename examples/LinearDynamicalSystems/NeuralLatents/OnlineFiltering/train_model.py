import numpy as np
import ssm.learning
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_latents", type=int, default=10,
                        help="Number of latent states")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of EM iterations")
    parser.add_argument("--tol", type=float, default=0.1,
                        help="Convergence tolerance for EM")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory to save estimated parameters. If not specified, uses the datasets directory.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--vars_to_estimate", type=str, default="B,Q,Z,R,m0,V0",
                        help="Comma-separated list of variables to estimate. Options are B, Q, Z, R, m0, V0.")
    args = parser.parse_args()

    n_latents = args.n_latents
    max_iter = args.max_iter
    tol = args.tol
    output_dir = args.output_dir
    seed = args.seed
    vars_to_estimate_list = [var.strip() for var in args.vars_to_estimate.split(",")]
    vars_to_estimate = {var: (var in vars_to_estimate_list) for var in ["B", "Q", "Z", "R", "m0", "V0"]}

    if output_dir is None:
        output_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "..", "..", "..", "datasets")

    print("Loading training data...")

    # load training data
    training_data_file = os.path.join(output_dir, "sub-Jenkins_ses-small_desc-train_behavior+ecephys.bin")

    n_clusters = 142
    transformed_binned_spikes = np.fromfile(training_data_file, dtype=float).reshape(-1, n_clusters).T

    print("Initializing model...")

    # initialize model
    np.random.seed(seed)

    B0 = np.diag(np.random.normal(loc=0, scale=0.1, size=n_latents))
    Z0 = np.random.normal(loc=0, scale=0.1, size=(n_clusters, n_latents))
    Q0 = np.diag(np.abs(np.random.normal(loc=0, scale=0.1, size=n_latents)))
    R0 = np.diag(np.abs(np.random.normal(loc=0, scale=0.1, size=n_clusters)))
    m0_0 = np.random.normal(loc=0, scale=0.1, size=n_latents)
    V0_0 = np.diag(np.abs(np.random.normal(loc=0, scale=0.1, size=n_latents)))

    print("Estimating parameters...")

    optim_res = ssm.learning.em_SS_LDS(
        y=transformed_binned_spikes, B0=B0, Q0=Q0, Z0=Z0, R0=R0,
        m0_0=m0_0, V0_0=V0_0, max_iter=max_iter, tol=tol,
        vars_to_estimate=vars_to_estimate,
    )

    # Save estimated parameters
    optim_res["B"].astype(float).tofile(os.path.join(output_dir, "transition_matrix.bin"))
    optim_res["Z"].astype(float).tofile(os.path.join(output_dir, "measurement_function.bin"))
    optim_res["Q"].astype(float).tofile(os.path.join(output_dir, "process_noise_covariance.bin"))
    optim_res["R"].astype(float).tofile(os.path.join(output_dir, "measurement_noise_covariance.bin"))
    optim_res["m0"].astype(float).tofile(os.path.join(output_dir, "initial_state_mean.bin"))
    optim_res["V0"].astype(float).tofile(os.path.join(output_dir, "initial_state_covariance.bin"))

    print(f"Estimated parameters saved to {os.path.realpath(output_dir)}")