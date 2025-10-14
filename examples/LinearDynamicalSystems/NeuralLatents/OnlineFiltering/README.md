# Inferring Neural Latents using Online Kalman Filtering

In the following example, you will see how the `Bonsai.ML.Lds.Torch` package can be used for inferring neural latents online using a Kalman filter from binned spike counts.

### Download the Dataset

To run the example, you need to download the dataset from DANDI and transform the raw spike data into binned spike counts. We provide a convenient python script and `uv` environment to easily download and transform the raw data. First, make sure `uv` is installed on your computer, which you can do by following the guide [here](https://docs.astral.sh/uv/getting-started/installation/).

Next, open up a terminal and run the following commands.

```bash
cd examples/Torch/NeuralLatents/OnlineFiltering
uv run download_data.py
```

When the script finishes, you should see 2 different datasets saved, one for training and the other for testing.

Next, run the python script to train the model on the training dataset and save the parameters of the model to disk.

```bash
uv run train_model.py
```

You can change the training procedure by passing parameters to the script. By default, this script will run 100 iterations, which can take several hours. You may want to consider changing this amount depending on your needs.

### Running the Workflow

The example workflow is shown here:

:::workflow
![Online Filtering of Neural Latents](OnlineFiltering.bonsai)
:::

The first group node, `LoadData`, loads the testing dataset and produces a sequence of binned spike counts, one spike count vector for each time point. Each spike count vector gets converted into a `Tensor` object and then reshaped into a column vector of `time` x `neurons` (in this case, time is `1`). This data then feeds into a `PublishSubject` called `SpikeCounts` to be used in the downstream processing pipeline.

The `LoadModel` group node uses the `CreateKalmanFilter` node to specify the model. We leave most of the optional parameters blank, but use `0.01` and `0.1` as our initial guesses of the `MeasurementNoiseVariance` and `ProcessNoiseVariance`, respectively. We set the `NumStates` to `10` and the `NumObservations` to `142` to match the number of spiking neurons contained in the dataset.

In the `LearnParameters` group node, the `SpikeCounts` are fed into `Buffer`, which will collect the spike counts into a small batch of `1000`. Realistically, you would want to increase the size of this batch to include more data when running the EM algorithm. The sequence completes once the first batch is fed into the `ExpectationMaximization` node, which uses the EM algorithm to iteratively optimize the models selected parameters given the batch of spike count data. In this example, all the model parameters are estimated, but we set a very small number of iterations and large tolerance. Again, in a real experiment, you would likely need to increase the number of iterations to get accurate estimates. The EM algorithm will iterate until it reaches the `MaxIterations` count or until the algorithm converges to less than the `Tolerance`. After some time, the algorithm will finish and the output will be passed to a `BehaviorSubject` called `ExpectationMaximizationResult` to trigger the inference procedure.

The last group node, `InferNeuralLatents`, contains the inference pipeline and starts when the `ExpectationMaximizationResult` is emitted. The `Filter` step runs causal filtering over each new data point it observes from the `SpikeCounts` subject. This output is fed into the `Orthogonalize` node, which extracts the principal components of the measurement function and projects the state mean and covariance into the orthogonalized space.

When you start the workflow, a window will pop up with the title `NeuralLatents`. As the EM algorithm runs, nothing will be displayed at first. After some time (~1 min), the EM algorithm will finish, and you will start to see the filtered latents displayed in the chart.