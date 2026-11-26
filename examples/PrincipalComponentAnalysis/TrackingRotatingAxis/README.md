# Tracking a rotating axis with online PCA

The following example shows how the `OnlinePcaGha` model from the `Bonsai.ML.Pca.Torch` package follows data whose principal axis changes over time, and how a standard `Pca` model fitted once to the start of the same data goes out of date.

## Dataset

The data is generated inside the workflow, so nothing needs to be downloaded. Each batch contains 10 points drawn from a 2D Gaussian with a standard deviation of 3 along its long axis and 0.5 along its short axis. The long axis rotates at a constant speed and completes one turn every 60 seconds.

### Workflow

Below is the example workflow.

:::workflow
![](TrackingRotatingAxis.bonsai)
:::

The workflow can be broken down into the following sections.

1. `ManualSeed` - Sets the seed of the random number generator, so that the same data is generated every time the workflow runs. The generator is shared as `Generator`.
2. `GenerateData` - A `Timer` emits a new batch 30 times per second. The current angle of the long axis, `Theta`, is computed from the timer count, and its cosine and sine are shared as `X` and `Y`. These are combined into the long and short axes of the distribution, each scaled by its standard deviation. Two `Normal` nodes draw the coordinates of each point along those axes, and the result is shared as `Data`.
3. `CreateModels` - Creates the two models that are compared. The `OnlineModel` is an `OnlinePcaGha` model with 1 component and a `LearningRate` of 0.1, which uses the `Generator` to initialize its component. The `BatchModel` is a standard `Pca` model with 1 component.
4. `FitOnline` - Passes every batch to the `Fit` operator together with the `OnlineModel`, so the model is updated 30 times per second. The component of the model after each update is published as `OnlineComponent`.
5. `FitBatchOnce` - Collects the first 150 batches, which is the first 5 seconds of data, concatenates them and fits the `BatchModel` to them once. The component of this model is published as `BatchComponent` and does not change afterwards.
6. `Visualizer` - Displays the most recent data points together with a line along the component of each model, in the `Rotating axis` window.

When the workflow starts, the blue line of the online model aligns with the long axis of the data within the first few seconds and keeps following it as it rotates. The red line of the batch model appears after 5 seconds and stays fixed at the average direction of the data it was fitted to, so it points away from the data as soon as the axis has moved on.

The `LearningRate` of the `OnlineModel` sets how quickly the model follows the data. With a value of 0.1 the component stays within a few degrees of the true axis. Lower values, such as 0.02, give a steadier estimate that visibly lags behind the rotation, and higher values follow more closely but fluctuate more from batch to batch.
