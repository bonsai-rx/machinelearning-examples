import numpy as np

class DataIterator:

    def __init__(self,
                 data: dict,
                 start_index: int = 0):
        self.data = data
        self.position_bins = self.data["position_bins"]
        self.index = start_index
        super().__init__()
    
    def next(self,
             loop: bool = True) -> tuple[list, list]:
            
        output = None
        position_data = self.data["position_data"]
        spike_times = self.data["spike_times"]
        decoding_results = self.data["decoding_results"]

        if self.index > len(position_data) and loop:
            self.index = 0

        if self.index < len(position_data):
            position = position_data[self.index]
            spikes = spike_times[self.index]
            decoding = decoding_results[self.index][np.newaxis]

            self.index += 1

            output = (position, spikes, decoding, self.position_bins)
        
        return output