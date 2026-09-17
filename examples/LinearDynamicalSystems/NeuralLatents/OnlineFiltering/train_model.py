import numpy as np
import ssm.learning
import os
import argparse
import leb128

def convert_type(array):
    match array.dtype:
        case np.uint8:
            return 0
        case np.int8:
            return 1
        case np.int16:
            return 2
        case np.int32:
            return 3
        case np.int64:
            return 4
        case np.float16:
            return 5
        case np.float32:
            return 6
        case np.float64:
            return 7
        case np.bool_:
            return 11
        case _:
            return 4711

def save_to_tensor(array, filename):
    with open(filename, "wb") as f:
        f.write(leb128.u.encode(convert_type(array)))
        f.write(leb128.u.encode(len(array.shape)))
        for size in array.shape:
            f.write(leb128.u.encode(size))
        f.write(array.tobytes())

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
        output_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "..", "..", "..", "datasets", "NeuralLatents")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading training data...")

    # load training data
    training_data_file = os.path.join(os.path.dirname(output_dir), "sub-Jenkins_ses-small_desc-train_behavior+ecephys.bin")

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
    save_to_tensor(optim_res["B"], os.path.join(output_dir, "TransitionMatrix.bin"))
    save_to_tensor(optim_res["Z"], os.path.join(output_dir, "MeasurementFunction.bin"))
    save_to_tensor(optim_res["Q"], os.path.join(output_dir, "ProcessNoiseCovariance.bin"))
    save_to_tensor(optim_res["R"], os.path.join(output_dir, "MeasurementNoiseCovariance.bin"))
    save_to_tensor(optim_res["m0"], os.path.join(output_dir, "InitialMean.bin"))
    save_to_tensor(optim_res["V0"], os.path.join(output_dir, "InitialCovariance.bin"))

    print(f"Estimated parameters saved to {os.path.realpath(output_dir)}")