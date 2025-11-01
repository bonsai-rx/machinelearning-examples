# Saving and Loading Model Parameters

In the following example, you will use the `Bonsai.ML.Lds.Torch` package to train a Kalman filter model for estimating neural latents in Bonsai, save the parameters of the model to disk, and subsequently load the parameters and create a model in a separate Bonsai workflow.

### Download the dataset

To run the example, you need to download the dataset from DANDI and transform the raw spike data into binned spike counts. We provide a convenient python script and `uv` environment to easily download and transform the raw data. First, make sure `uv` is installed on your computer, which you can do by following the guide [here](https://docs.astral.sh/uv/getting-started/installation/).

Next, open up a terminal and run the following commands.

```bash
cd examples/Torch/NeuralLatents/SavingAndLoadingParameters
uv run download_data.py
```

When the script finishes, you should see 2 different datasets saved, one for training and the other for testing.

### Saving model parameters in Bonsai

The example workflow is shown here:

:::workflow
![Saving Model Parameters](SaveModelParameters.bonsai)
:::

The first group node, `LoadData`, loads the training dataset and converts it into a `Tensor` object. It then reshapes it so that the dimensions are `time` x `neurons` and passes it to a `Subject` called `SpikeCounts`. In this example, the entire batch of data is loaded. 

The `LoadModel` group node uses the `CreateKalmanFilter` node to specify the model. We leave the optional parameters blank. We set the `NumStates` to `10` and the `NumObservations` to `142` to match the number of spiking neurons contained in the dataset. This then feeds into a `BehaviorSubject` node called `KalmanFilterModel` for use in other parts of the workflow.

In the `LearnParameters` group node, the `SpikeCounts` are fed into the `ExpectationMaximization` node which uses the EM algorithm to iteratively optimize the models selected parameters given the spike count data. The EM algorithm will iterate until it reaches the `MaxIterations` count or until the algorithm converges to less than the `Tolerance`. After some time, the algorithm will finish and the output will be passed to a `BehaviorSubject` called `ExpectationMaximizationResult`.

In the `SaveModelParameters` group node, the emission of a value from the `ExpectationMaximizationResult` subject triggers the `SaveModelParameters` node to write the model parameters to binary data files.

### Loading model parameters in Bonsai

The example workflow is shown here:

:::workflow
![Loading Model Parameters](LoadModelParameters.bonsai)
:::

The first group node, `LoadData`, loads the test dataset and converts it into a `Tensor` object.

Inside of the `LoadModel` group node, the `LoadKalmanFilterParameters` operator reads in the data files for the different parameters of the model that were created in the previous workflow. These parameters are then used by the `CreateKalmanFilter` operator. 

In the `InferNeuralLatents` group node, the `SpikeCounts` are passed to the model and online filtering is performed, followed by orthogonalization of the values. The inferrence stream can then be visualized online.