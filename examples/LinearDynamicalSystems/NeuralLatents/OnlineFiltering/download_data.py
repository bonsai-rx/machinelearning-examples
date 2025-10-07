# Import libraries
import numpy as np
import remfile, h5py
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
import os

# set base directory to save data
base_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "..", "..", "..", "datasets")

# dandi dataset info
dandiset_ID = "000140"
dandi_filepath = "sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"

# download data using remfile and dandi
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_ID,
                                "draft").get_asset_by_path(dandi_filepath)
    s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
    cache = remfile.DiskCache(os.path.join(base_dir, "remfile_cache"))
    rf = remfile.File(s3_path, disk_cache=cache)
    with h5py.File(rf, "r") as h:
        with NWBHDF5IO(file=h, mode="r") as io:
            nwbfile = io.read()
            # extract spike sorted units dataframe
            units_df = nwbfile.units.to_dataframe()

# bin spikes
n_clusters = units_df.shape[0]
bin_size = 0.02

spike_times = [units_df.iloc[n]['spike_times'] for n in range(n_clusters)]
t_max = max(max(st) for st in spike_times if len(st) > 0)

bin_edges = np.arange(0, t_max + bin_size, bin_size)
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
n_bins = len(bin_edges) - 1

spike_counts = np.zeros((n_clusters, n_bins), dtype=int)

for n, spikes in enumerate(spike_times):
    spike_counts[n], _ = np.histogram(spikes, bins=bin_edges)

# transform binned spikes using square-root transform
transformed_binned_spikes = np.sqrt(spike_counts + 0.5)
transformed_binned_spikes.astype(float).tofile(os.path.join(base_dir, "transformed_binned_spikes.bin"))