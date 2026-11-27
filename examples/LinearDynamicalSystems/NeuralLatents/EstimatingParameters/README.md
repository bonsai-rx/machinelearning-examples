# Estimating Parameters of Kalman Filter

In the following example, you will see how the `Bonsai.ML.Lds.Torch` package can be used to estimate the parameters of a Kalman filter and perform acausal smoothing of neural spike counts.

### Download the Dataset

To run the example, you need to download the dataset from DANDI and transform the raw spike data into binned spike counts. We provide a convenient python script and `uv` environment to easily download and transform the raw data. First, make sure `uv` is installed on your computer, which you can do by following the guide [here](https://docs.astral.sh/uv/getting-started/installation/).

Next, open up a terminal and run the following commands.

```bash
cd examples/LinearDynamicalSystems/NeuralLatents/EstimatingParameters
uv run download_data.py
```

When the script finishes, you should see 2 different datasets saved, one for training and the other for testing.

### Running the Workflow

The example workflow is shown here:

:::workflow
![Estimate Parameters](EstimateParameters.bonsai)
:::

The first group node, `LoadData`, loads the training dataset and converts it into a `Tensor` object. It then reshapes it so that the dimensions are `time` x `neurons` and passes it to a `ReplaySubject` called `SpikeCounts` to be used in the downstream processing pipeline. In this example, the entire batch of data is loaded at once. 

In the `LearnParameters` group node, the `SpikeCounts` are first fed into the `StochasticSubspaceIdentification` node, which estimates the parameters of a model with `10` latent states directly from the covariance between past and future spike counts, without iterating. The `EstimateObservationOffset` property is enabled, so the mean spike count of each neuron is estimated as the observation offset of the model. Without the offset, the mean of the spike counts would have to be explained by the latent states, and the first latent would have a large constant value instead of being centered on zero. The result is passed to an `AsyncSubject` called `SubspaceIdentificationResult`.

The parameters from subspace identification are then mapped to the `ModelParameters` property of the `ExpectationMaximization` node, which uses the EM algorithm to iteratively optimize the selected parameters of the model given the spike count data, starting from the subspace identification estimate. The EM algorithm will iterate until it reaches the `MaxIterations` count or until the algorithm converges to less than the `Tolerance`. After some time, the algorithm will finish and the estimated parameters will be passed to the `CreateKalmanFilter` node, which creates the model and passes it to a `BehaviorSubject` called `KalmanFilterModel` to trigger the inference procedure.

The last group node, `InferNeuralLatents`, contains the inference pipeline and starts when the `KalmanFilterModel` is emitted. The `Filter` step runs causal filtering over the entire batch of `SpikeCounts`. The `UpdatedFilterResult` is then run through the `Smooth` step, which runs acausal smoothing from the last data point all the way back to the initial data point. This `UpdatedSmoothedResult` is then fed into the `Orthogonalize` node, which essentially node extracts the principal, orthogonal components of the measurement function and projects the state mean and covariance into the orthogonalized space to focus only on the most informative directions. The `OrthogonalizedResult` is then decomposed into individual observations over time and gated to produce values at a frequency of a ~100Hz.

When you start the workflow, you should see a window pop up with the title `NeuralLatents`. As subspace identification and the EM algorithm run, nothing will be displayed at first. After 10 iterations, the EM algorithm will finish, and once the data are filtered, smoothed, and orthogonalized, you will start to see the inferred latents displayed in the chart.
