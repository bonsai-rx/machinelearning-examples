import replay_trajectory_classification as rtc
from replay_trajectory_classification.core import scaled_likelihood, get_centers
from replay_trajectory_classification.likelihoods import _SORTED_SPIKES_ALGORITHMS, _ClUSTERLESS_ALGORITHMS
from replay_trajectory_classification.likelihoods.spiking_likelihood_kde import combined_likelihood, poisson_log_likelihood
from replay_trajectory_classification.likelihoods.multiunit_likelihood import estimate_position_distance, estimate_log_joint_mark_intensity

from .likelihood import LIKELIHOOD_FUNCTION

import numpy as np
import pickle as pkl

class Decoder():
    def __init__(self):
        super().__init__()

    def decode(self):
        raise NotImplementedError

class ClusterlessSpikeDecoder(Decoder):
    def __init__(self, model_dict: dict):
        self.decoder = model_dict["decoder"]
        self.Fs = model_dict["Fs"]
        self.features = model_dict["features"]

        encoding_model = self.decoder.encoding_model_
        self.encoding_marks = encoding_model["encoding_marks"]
        self.mark_std = encoding_model["mark_std"]
        self.encoding_positions = encoding_model["encoding_positions"]
        self.position_std = encoding_model["position_std"]
        self.occupancy = encoding_model["occupancy"]
        self.mean_rates = encoding_model["mean_rates"]
        self.summed_ground_process_intensity = encoding_model["summed_ground_process_intensity"]
        self.block_size = encoding_model["block_size"]
        self.bin_diffusion_distances = encoding_model["bin_diffusion_distances"]
        self.edges = encoding_model["edges"]

        self.place_bin_centers = self.decoder.environment.place_bin_centers_
        self.is_track_interior = self.decoder.environment.is_track_interior_.ravel(order="F")
        self.st_interior_ind = np.ix_(self.is_track_interior, self.is_track_interior)
        self.interior_place_bin_centers = np.asarray(
            self.place_bin_centers[self.is_track_interior], dtype=np.float32
        )
        self.interior_occupancy = np.asarray(
            self.occupancy[self.is_track_interior], dtype=np.float32
        )
        self.n_position_bins = self.is_track_interior.shape[0]
        self.n_track_bins = self.is_track_interior.sum()

        self.initial_conditions = self.decoder.initial_conditions_[self.is_track_interior].astype(float)
        self.state_transition = self.decoder.state_transition_[self.st_interior_ind].astype(float)

        self.likelihood_funcion = LIKELIHOOD_FUNCTION[self.decoder.clusterless_algorithm]

        self.posterior = None
        super().__init__()
    
    def decode(self,
               multiunits: np.ndarray):

        likelihood = self.likelihood_function(
            multiunits, 
            self.summed_ground_process_intensity, 
            self.encoding_marks, 
            self.encoding_positions, 
            self.mean_rates, 
            self.is_track_interior, 
            self.interior_place_bin_centers, 
            self.position_std, 
            self.mark_std, 
            self.interior_occupancy, 
            self.n_track_bins
        )

        if self.posterior is None:
            self.posterior = np.full((1, self.n_position_bins), np.nan, dtype=float)
            self.posterior[0, self.is_track_interior] = self.initial_conditions * likelihood[0, self.is_track_interior]

        else:
            self.posterior[0, self.is_track_interior] = self.state_transition.T @ self.posterior[0, self.is_track_interior] * likelihood[0, self.is_track_interior]

        norm = np.nansum(self.posterior[0])
        self.posterior[0] /= norm

        return self.posterior

class SortedSpikeDecoder(Decoder):
    def __init__(self, model_dict: dict):
        self.decoder = model_dict["decoder"]
        self.Fs = model_dict["Fs"]
        self.spikes = model_dict["binned_spikes_times"]
        self.is_track_interior = self.decoder.environment.is_track_interior_.ravel(order="F")
        self.st_interior_ind = np.ix_(self.is_track_interior, self.is_track_interior)
        self.n_position_bins = self.is_track_interior.shape[0]

        self.initial_conditions = self.decoder.initial_conditions_[self.is_track_interior].astype(float)
        self.state_transition = self.decoder.state_transition_[self.st_interior_ind].astype(float)
        self.place_fields = np.asarray(self.decoder.place_fields_)
        self.position_centers = get_centers(self.decoder.environment.edges_[0])
        self.conditional_intensity = np.clip(self.place_fields, a_min=1e-15, a_max=None)

        self.likelihood_function = LIKELIHOOD_FUNCTION[self.decoder.sorted_spikes_algorithm]

        self.posterior = None
        super().__init__()

    def decode(
            self,
            spikes: np.ndarray
        ):

        likelihood = self.likelihood_function(spikes, self.conditional_intensity, self.is_track_interior)

        if self.posterior is None:
            self.posterior = np.full((1, self.n_position_bins), np.nan, dtype=float)
            self.posterior[0, self.is_track_interior] = self.initial_conditions * likelihood[0]

        else:
            self.posterior[0, self.is_track_interior] = self.state_transition.T @ self.posterior[0, self.is_track_interior] * likelihood[0]

        norm = np.nansum(self.posterior[0])
        self.posterior[0] /= norm

        return self.posterior