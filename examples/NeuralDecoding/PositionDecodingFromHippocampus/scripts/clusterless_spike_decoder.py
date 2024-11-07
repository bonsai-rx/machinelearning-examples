import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

import replay_trajectory_classification as rtc
import track_linearization as tl

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions_filename", type=str,
                        default="../../../../datasets/decoder_data/position_info.pkl",
                        help="positions filename")
    parser.add_argument("--spike_times_filename", type=str,
                        default="../../../../datasets/decoder_data/clusterless_spike_times.pkl",
                        help="spikes filename")
    parser.add_argument("--features_filename", type=str,
                        default="../../../../datasets/decoder_data/clusterless_spike_features.pkl",
                        help="features filename")
    parser.add_argument("--model_filename", type=str,
                        default="../../../../datasets/decoder_data/clusterless_spike_decoder.pkl",
                        help="model filename")
    parser.add_argument("--decoding_results_filename", type=str,
                        default="../../../../datasets/decoder_data/clusterless_spike_decoding_results.pkl",
                        help="decoding results filename")
    args = parser.parse_args()

    positions_filename = args.positions_filename
    spikes_filename = args.spikes_filename
    features_filename = args.features_filename
    model_filename = args.model_filename
    decoding_results_filename = args.decoding_results_filename

    positions_df = pd.read_pickle(positions_filename)
    timestamps = positions_df.index.to_numpy()
    dt = timestamps[1] - timestamps[0]
    Fs = 1.0 / dt
    spikes_bins = np.append(timestamps-dt/2, timestamps[-1]+dt/2)

    x = positions_df["nose_x"].to_numpy()
    y = positions_df["nose_y"].to_numpy()
    positions = np.column_stack((x, y))
    node_positions = [(120.0, 100.0),
                        (  5.0, 100.0),
                        (  5.0,  55.0),
                        (120.0,  55.0),
                        (  5.0,   8.5),
                        (120.0,   8.5),
                        ]
    edges = [
                (3, 2),
                (0, 1),
                (1, 2),
                (5, 4),
                (4, 2),
            ]
    track_graph = rtc.make_track_graph(node_positions, edges)

    edge_order = [
                (3, 2),
                (0, 1),
                (1, 2),
                (5, 4),
                (4, 2),
                ]

    edge_spacing = [16, 0, 16, 0]

    linearized_positions = tl.get_linearized_position(positions, track_graph, edge_order=edge_order, edge_spacing=edge_spacing, use_HMM=False)

    with open(features_filename, "rb") as f:
        clusterless_spike_features = pkl.load(f)

    with open(spikes_filename, "rb") as f:
        clusterless_spike_times = pkl.load(f)

    features = np.ones((len(timestamps), len(clusterless_spike_features[0][0]), len(clusterless_spike_times)), dtype=float) * np.nan
    for n in range(len(clusterless_spike_times)):
        in_spikes_window = np.digitize(clusterless_spike_times[n], spikes_bins)
        features[in_spikes_window, :, n] = clusterless_spike_features[n]

    place_bin_size = 0.5
    movement_var = 0.25

    environment = rtc.Environment(place_bin_size=place_bin_size,
                                    track_graph=track_graph,
                                    edge_order=edge_order,
                                    edge_spacing=edge_spacing)

    transition_type = rtc.RandomWalk(movement_var=movement_var)

    decoder = rtc.ClusterlessDecoder(
        environment=environment,
        transition_type=transition_type,
        clusterless_algorithm="multiunit_likelihood"
    )

    print("Learning model parameters")
    decoder.fit(linearized_positions.linear_position, features)

    print(f"Saving model to {model_filename}")

    results = dict(decoder=decoder, linearized_positions=linearized_positions,
                    clusterless_spike_times=clusterless_spike_times, features=features, Fs=Fs)

    with open(model_filename, "wb") as f:
        pkl.dump(results, f)

    decoding_start_secs = 0
    decoding_duration_secs = 100

    with open(model_filename, "rb") as f:
        model_results = pkl.load(f)
        
    decoder = model_results["decoder"]
    Fs = model_results["Fs"]
    clusterless_spike_times = model_results["clusterless_spike_times"]
    features = model_results["features"]
    linearized_positions = model_results["linearized_positions"]

    print("Decoding positions from features")
    decoding_start_samples = int(decoding_start_secs * Fs)
    decoding_duration_samples = int(decoding_duration_secs * Fs)
    time_ind = slice(decoding_start_samples, decoding_start_samples + decoding_duration_samples)
    time = np.arange(linearized_positions.linear_position.size) / Fs
    decoding_results = decoder.predict(features[time_ind], time=time[time_ind])

    print(f"Saving decoding results to {decoding_results_filename}")

    results = dict(decoding_results=decoding_results, time=time[time_ind],
                    linearized_positions=linearized_positions.iloc[time_ind],
                    features=features[time_ind])

    with open(decoding_results_filename, "wb") as f:
        pkl.dump(results, f)