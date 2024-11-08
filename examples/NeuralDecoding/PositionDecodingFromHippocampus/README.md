# Position Decoding from Hippocampal Sorted Spikes

In the following example, you can find how to use the decoder from [here](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification/tree/master?tab=readme-ov-file) to decode an position from hippocampal activity.

## Dataset

We thank Eric Denovellis for sharing his data and for his help with the decoder. If you use this example dataset, please cite: Eric L Denovellis, Anna K Gillespie, Michael E Coulter, Marielena Sosa, Jason E Chung, Uri T Eden, Loren M Frank (2021). Hippocampal replay of experience at real-world speeds. eLife 10:e64505.

## Installation

### Python

You can bootstrap the python environment by running:

```python
cd \path\to\examples\NeuralDecoding\PositionDecodingFromHippocampus
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

You can test whether the installation was successful by launching python and running `import replay_trajectory_classification`.

### Bonsai

You can bootstrap the bonsai environment using:

```
cd \path\to\examples\NeuralDecoding\PositionDecodingFromHippocampus
dotnet new bonsaienv --allow-scripts yes
```

Alternatively, you can copy the `.bonsai\Bonsai.config` file into your Bonsai installation folder. You can test if it worked by openning bonsai and searching for the `CreateRuntime` node, which should appear in the toolbox.

### Training the decoder offline

You first need to train the decoder model and save it to disk. Open up the `notebooks` folder and select either `SortedSpikeDecoder.ipynb` or `ClusterlessDecoder.ipynb` depending on which model type you would like to use. Run the notebook. Once completed, this should create 2 new files: 1) `datasets\decoder_data\[ModelType]_decoder.pkl` for the trained decoder model; and 2) `datasets\decoder_data\[ModelType]_decoding_results.pkl` for the predictions.

### Running the decoder online

Launch the Bonsai.exe file inside of the .bonsai folder and open the workflow corresponding to the model type you used in the previous step. Press the `Start Workflow` button. The workflow may take some time to initialize and load the data. Once the workflow is running, open the `VisualizeDecoder` node to bring up the following:
1. Online prediction of position
2. Offline acausal prediction of position
3. True position
4. Latest online posterior distribution
5. Latest offline acausal posterior distribution
