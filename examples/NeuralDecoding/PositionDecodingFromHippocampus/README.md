# Position Decoding from Hippocampal Sorted Spikes

In the following example, you can find how to use the decoder from [here](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification/tree/master?tab=readme-ov-file) to decode an animals position from sorted hippocampal units sampled with tetrodes.

### Dataset

We thank Eric Denovellis for sharing his data and for his help with the decoder. Please cite his work: Eric L Denovellis, Anna K Gillespie, Michael E Coulter, Marielena Sosa, Jason E Chung, Uri T Eden, Loren M Frank (2021). Hippocampal replay of experience at real-world speeds. eLife 10:e64505.

You can download the data [here](https://drive.google.com/file/d/1ddRC28w0U4_q3pcGfY-1vPHjO9mjEaJb/view?usp=sharing). The workflow expects the zip file to be extracted into the `datasets/decoder_data` folder. The workflow also expects the files to be renamed to just `sorted_spike_times.pkl` and `position_info.pkl`. 

### Python

You need to install the [replay_trajectory_classification](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) package into your python virtual environment. 
