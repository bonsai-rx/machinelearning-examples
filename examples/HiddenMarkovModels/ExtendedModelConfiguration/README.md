# HMM - Extended Configuration Example

The code for this repo can be found [here](https://github.com/bonsai-rx/machinelearning-examples/tree/main/examples/HiddenMarkovModels/ExtendedModelConfiguration).

In the following example, you can see how to construct a Hidden Markov Model (HMM) with a custom configuration for the model's observation and transition distributions. This example extends the basic HMM configuration to allow for more complex scenarios, and includes 2 workflows, one for saving a custom model to disk and another for loading the model in a seperate workflow.

### Instructions

Ensure you have Bonsai 2.9 installed on your system. You will also need to have the [uv Python environment manager](https://docs.astral.sh/uv/) installed.

> [!NOTE]
> The `ssm` package uses an old version of `setuptools` that is incompatible with modern package management tools. Because of this, you first need to create a new Python virtual environment using `uv venv`, install the required packages using `uv pip install -r requirements.txt`, and then synchronize the Python environment using `uv sync --no-build-isolation`. 

### Save a Custom HMM Configuration

This example workflow demonstrates how to create an HMM with custom parameters. When the workflow is run, it will instantiate a custom HMM model with the specified parameters and save it to disk.

:::workflow
![Hidden Markov Models - Saving Custom Model To Disk](SaveModelConfig.bonsai)
:::

The default parameters when using the `CreateHMM` operator is a `Gaussian` observation model and `Stationary` transition model. However, these options don't allow for fine tuning the parameters of our model. To address this, the `CreateHMM` node allows us to fully parameterize the model's `StateParameters`, where we can fully specify the `Observations`, `Transitions`, and `InitialState` distributions, which will supersede the parameters set directly inside the `CreateHMM` operator.

In this example, the `ConfigureHMM` group node encapsulates this functionality. The `ConfigureHMM` node contains externalized properties which allow us to fully specify the `Observations` and `Transitions` model. Here, we use `AutoRegressiveObservations`, and can specify the number of lags, etc. The `Transitions` model is set to `ConstrainedStationary`, which allows us to mask certain transitions from being used in the model.

When the workflow is run, the output of the `ConfigureHMM` node is a custom HMM model that is saved to disk using the `SerializeToJson` and `WriteAllText` operators. The model will be saved to a JSON file named `hmm_config.json`.

### Load a Custom HMM Configuration

This example workflow demonstrates how to load the custom HMM model saved in the previous workflow. 

:::workflow
![Hidden Markov Models - Loading Custom Model From Disk](LoadModelConfig.bonsai)
:::

The `LoadStateParameters` is a `SelectMany` operator which will load the JSON file specified in the `Path` property, and outputs a `ModelParameters` object. The `Dimensions`, `NumStates`, and `StateParameters` properties of the `ModelParameters` object are then used as input parameters to the `CreateHMM` operator, which creates an HMM model with the parameters loaded from the JSON file.