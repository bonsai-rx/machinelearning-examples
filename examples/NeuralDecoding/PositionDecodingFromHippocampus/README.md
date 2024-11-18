# Position Decoding from Hippocampal Sorted Spikes

In the following example, you can find how to use the spike sorted decoder or clusterless spike decoder from [here](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification/tree/master?tab=readme-ov-file) to decode position from hippocampal activity.

## Dataset

We thank Eric Denovellis for sharing his data and for his help with the decoder. If you use this example dataset, please consider citing the work: Joshi, A., Denovellis, E.L., Mankili, A. et al. Dynamic synchronization between hippocampal representations and stepping. Nature 617, 125–131 (2023). https://doi.org/10.1038/s41586-023-05928-6.

## Algorithm

The neural decoder consists of a bayesian state-space model and point-processes to decode a latent variable (position) from neural spiking activity. To read more about the theory behind the model and how the algorithm works, we refer the reader to: Denovellis, E.L., Gillespie, A.K., Coulter, M.E., et al. Hippocampal replay of experience at real-world speeds. eLife 10, e64505 (2021). https://doi.org/10.7554/eLife.64505.

## Installation

### Python

To install the package, run:

```
cd \path\to\examples\NeuralDecoding\PositionDecodingFromHippocampus
python -m venv .venv
.\.venv\Scripts\activate
pip install git+https://github.com/ncguilbeault/bayesian-neural-decoder.git
```

You can test whether the installation was successful by launching python and running

```python
import bayesian_neural_decoder
```

### Bonsai

You can bootstrap the bonsai environment using:

```
cd \path\to\examples\NeuralDecoding\PositionDecodingFromHippocampus
dotnet new bonsaienv --allow-scripts yes
```

Alternatively, you can copy the `.bonsai\Bonsai.config` file into your Bonsai installation folder. You can test if it worked by openning bonsai and searching for the `CreateRuntime` node, which should appear in the toolbox.

## Usage

### Training the decoder offline

The package contains 2 different models, one which takes as input sorted spike activity, and another which uses clusterless spike activity taken from raw ephys recordings. 

You first need to train the decoder model and save it to disk. Open up the `notebooks` folder and select either `SortedSpikeDecoder.ipynb` or `ClusterlessDecoder.ipynb` depending on which model type you would like to use. Run the notebook. Once completed, this will create 2 new files: 
1) `datasets\decoder_data\[ModelType]_decoder.pkl` for the trained decoder model
2) `datasets\decoder_data\[ModelType]_decoding_results.pkl` for the predictions.

Both of these files are needed to be run the decoder example in Bonsai.

### Running the decoder online with Bonsai

Launch the Bonsai.exe file inside of the .bonsai folder and open the workflow corresponding to the model type you used in the previous step. Press the `Start Workflow` button. The workflow may take some time to initialize and load the data. Once the workflow is running, open the `VisualizeDecoder` node to see the model's inference online with respect to the true position.
