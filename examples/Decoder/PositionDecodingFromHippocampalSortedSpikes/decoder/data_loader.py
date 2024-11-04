import pickle
import pandas as pd
import numpy as np
import track_linearization as tl
import replay_trajectory_classification as rtc
import os

class DataLoader:
    def __init__(self):
        super().__init__()

    @classmethod
    def load_data(cls, 
                  dataset_path: str = "../../../datasets/decoder_data",
                  bin_spikes: bool = True) -> dict:

        if len([file for file in os.listdir(dataset_path) if file == "position_info.pkl" or file == "sorted_spike_times.pkl" or file == "decoding_results.pkl"]) != 3:
            raise Exception("Dataset incorrect. Missing at least one of the following files: 'position_info.pkl', 'sorted_spike_times.pkl', 'decoding_results.pkl'")

        position_data = pd.read_pickle(os.path.join(dataset_path, "position_info.pkl"))
        position_index = position_data.index.to_numpy()
        position_index = np.insert(position_index, 0, position_index[0] - (position_index[1] - position_index[0]))
        position_data = position_data[["nose_x", "nose_y"]].to_numpy()

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

        linearized_positions = tl.get_linearized_position(position_data, track_graph, edge_order=edge_order, edge_spacing=edge_spacing, use_HMM=False)
        position_data = linearized_positions.linear_position

        with open(os.path.join(dataset_path, "sorted_spike_times.pkl"), "rb") as f:
            spike_times = pickle.load(f)

        if bin_spikes:
            spike_mat = np.zeros((len(position_data), len(spike_times)))
            for neuron in range(len(spike_times)):
                spike_mat[:, neuron] = np.histogram(spike_times[neuron], position_index)[0]
            spike_times = spike_mat

        with open(os.path.join(dataset_path, "decoding_results.pkl"), "rb") as f:
            results = pickle.load(f)["decoding_results"]
            position_bins = results.position.to_numpy()
            decoding_results = results.acausal_posterior.to_numpy()

        return {
            "position_data": position_data,
            "spike_times": spike_times,
            "decoding_results": decoding_results,
            "position_bins": position_bins
        }
