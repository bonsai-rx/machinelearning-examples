import replay_trajectory_classification as rtc
from replay_trajectory_classification.core import scaled_likelihood, get_centers
from replay_trajectory_classification.likelihoods import _SORTED_SPIKES_ALGORITHMS, _ClUSTERLESS_ALGORITHMS
from replay_trajectory_classification.likelihoods.spiking_likelihood_kde import combined_likelihood, poisson_log_likelihood
from replay_trajectory_classification.likelihoods.multiunit_likelihood import estimate_position_distance, estimate_log_joint_mark_intensity

import numpy as np

class ClusterlessSpikeDecoder:
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

        self.posterior = None
        super().__init__()
    
    def decode(self,
               multiunits: np.ndarray):

        log_likelihood = -self.summed_ground_process_intensity * np.ones((1,1), dtype=np.float32)

        if not np.isnan(multiunits).all():
            multiunit_idxs = np.where(~np.isnan(multiunits, axis=0))[0]


            for multiunit, enc_marks, enc_pos, mean_rate in zip(
                    multiunits.T,
                    self.encoding_marks,
                    self.encoding_positions,
                    self.mean_rates,
                ):
                is_spike = np.any(~np.isnan(multiunit))
                if is_spike:
                    decoding_marks = np.asarray(
                        multiunit, dtype=np.float32
                    )[np.newaxis]
                    log_joint_mark_intensity = np.zeros(
                        (1, self.n_track_bins), dtype=np.float32
                    )
                    position_distance = estimate_position_distance(
                        self.interior_place_bin_centers,
                        np.asarray(enc_pos, dtype=np.float32),
                        self.position_std,
                    ).astype(np.float32)
                    log_joint_mark_intensity[0] = estimate_log_joint_mark_intensity(
                        decoding_marks,
                        enc_marks,
                        self.mark_std,
                        self.interior_occupancy,
                        mean_rate,
                        position_distance=position_distance,
                    )
                    log_likelihood[:, self.is_track_interior] += np.nan_to_num(
                        log_joint_mark_intensity
                    )
        
        log_likelihood[:, ~self.is_track_interior] = np.nan
        likelihood = scaled_likelihood(log_likelihood)

        if self.posterior is None:
            self.posterior = np.full((1, self.n_position_bins), np.nan, dtype=float)
            self.posterior[0, self.is_track_interior] = self.initial_conditions * likelihood[0, self.is_track_interior]

        else:
            self.posterior[0, self.is_track_interior] = self.state_transition.T @ self.posterior[0, self.is_track_interior] * likelihood[0, self.is_track_interior]

        norm = np.nansum(self.posterior[0])
        self.posterior[0] /= norm

        return self.posterior

class SortedSpikeDecoder:
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
        self.position_centers = get_centers(self.decoder.environment.edges_[0]),

        self.posterior = None
        super().__init__()

    def decode(
            self,
            spikes: np.ndarray
        ):
        conditional_intensity = np.clip(self.place_fields, a_min=1e-15, a_max=None)

        log_likelihood = 0
        for spike, ci in zip(spikes, conditional_intensity.T):
            log_likelihood += poisson_log_likelihood(spike[np.newaxis], ci)

        mask = np.ones_like(self.is_track_interior, dtype=float)
        mask[~self.is_track_interior] = np.nan
        
        likelihood = scaled_likelihood(log_likelihood * mask)
        likelihood = likelihood[:, self.is_track_interior].astype(float)

        if self.posterior is None:
            self.posterior = np.full((1, self.n_position_bins), np.nan, dtype=float)
            self.posterior[0, self.is_track_interior] = self.initial_conditions * likelihood[0]

        else:
            self.posterior[0, self.is_track_interior] = self.state_transition.T @ self.posterior[0, self.is_track_interior] * likelihood[0]

        norm = np.nansum(self.posterior[0])
        self.posterior[0] /= norm

        return self.posterior