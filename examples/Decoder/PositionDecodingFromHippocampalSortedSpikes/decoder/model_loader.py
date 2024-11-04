import pickle
from .core import SortedSpikeDecoder

class ModelLoader:
    def __init__(self):
        super().__init__()

    @classmethod
    def load_model(cls,
                   model_path: str = "../../../datasets/decoder_data/model.pkl") -> SortedSpikeDecoder:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return SortedSpikeDecoder(model)