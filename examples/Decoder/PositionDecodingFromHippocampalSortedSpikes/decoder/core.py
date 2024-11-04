import replay_trajectory_classification as rtc
from replay_trajectory_classification.core import scaled_likelihood, get_centers
from replay_trajectory_classification.likelihoods import _SORTED_SPIKES_ALGORITHMS
# from likelihoods import _SORTED_SPIKES_ALGORITHMS
from replay_trajectory_classification.likelihoods.spiking_likelihood_kde import combined_likelihood, poisson_log_likelihood

import numpy as np

class SortedSpikeDecoder:
    def __init__(self, model_dict: dict):
        self.model = model_dict["decoder"]
        self.Fs = model_dict["Fs"]
        self.spikes = model_dict["binned_spikes_times"]
        self.is_track_interior = self.model.environment.is_track_interior_.ravel(order="F")
        self.st_interior_ind = np.ix_(self.is_track_interior, self.is_track_interior)
        self.n_position_bins = self.is_track_interior.shape[0]

        self.initial_conditions = self.model.initial_conditions_[self.is_track_interior].astype(float)
        self.state_transition = self.model.state_transition_[self.st_interior_ind].astype(float)
        self.place_fields = np.asarray(self.model.place_fields_)
        self.position_centers = get_centers(self.model.environment.edges_[0]),

        self.posterior = None
        super().__init__()

    def decode_spikes(
            self,
            spikes: np.ndarray
        ):
        # likelihood = scaled_likelihood(_SORTED_SPIKES_ALGORITHMS[self.model.sorted_spikes_algorithm][1](spikes[np.newaxis], self.place_fields))
        # likelihood = likelihood[:, self.is_track_interior].astype(float)
        
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

    # def decode_spikes(
    #         self,
    #         data: tuple[list, list]
    #     ) -> tuple[np.ndarray, float]:

    #     likelihood = scaled_likelihood(_SORTED_SPIKES_ALGORITHMS[model.sorted_spikes_algorithm][1](spikes, np.asarray(model.place_fields_)))
    #     likelihood = likelihood[:, is_track_interior].astype(float)

    #     n_time = likelihood.shape[0]
    #     posterior = np.zeros_like(likelihood)

    #     posterior[0] = initial_conditions.copy() * likelihood[0]
    #     norm = np.nansum(posterior[0])
    #     log_data_likelihood = np.log(norm)
    #     posterior[0] /= norm

    #     for k in np.arange(1, n_time):
    #         posterior[k] = state_transition.T @ posterior[k - 1] * likelihood[k]
    #         norm = np.nansum(posterior[k])
    #         log_data_likelihood += np.log(norm)
    #         posterior[k] /= norm

    #     return posterior, log_data_likelihood