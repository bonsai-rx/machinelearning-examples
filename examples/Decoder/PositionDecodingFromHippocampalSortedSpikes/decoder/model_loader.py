import pickle
from .decoder import SortedSpikeDecoder, ClusterlessSpikeDecoder

class ModelLoader:
    def __init__(self):
        super().__init__()

    @classmethod
    def load_sorted_spike_decoder(cls,
                   model_path: str = "../../../datasets/decoder_data/sorted_spike_decoder.pkl") -> SortedSpikeDecoder:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return SortedSpikeDecoder(model)
    
    @classmethod
    def load_clusterless_spike_decoder(cls,
                   model_path: str = "../../../datasets/decoder_data/clusterless_spike_decoder.pkl") -> ClusterlessSpikeDecoder:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return ClusterlessSpikeDecoder(model)