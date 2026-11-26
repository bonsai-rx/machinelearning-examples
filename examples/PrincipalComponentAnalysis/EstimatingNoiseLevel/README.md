# Estimating the noise level with probabilistic PCA

The following example shows how the `ProbabilisticPca` model from the `Bonsai.ML.Pca.Torch` package separates a signal shared across channels from independent noise on each channel, and reports how much noise there is.

## Dataset

The data is generated inside the workflow, so nothing needs to be downloaded. It generates 2000 samples of 16 channels. Each channel is a weighted sum of the same two sine waves, with different weights for each channel, plus independent Gaussian noise. The noise has a standard deviation of 0.5, which corresponds to a variance of 0.25.

### Workflow

Below is the example workflow.

:::workflow
![](EstimatingNoiseLevel.bonsai)
:::

The workflow can be broken down into the following sections.

1. `ManualSeed` - Sets the seed of the random number generator, so that the same noise is generated every time the workflow runs. The generator is shared as `Generator`.
2. `GenerateData` - Computes the time of each sample, `X`, and the two sine waves at that time, and multiplies them by the weights of the 16 channels to give the noise-free value of each channel, `Y`. A `Normal` node draws the noise for each sample, which is added to `Y`. The noise-free samples are collected into the `Signal` and the noisy samples into the `Data`, each as a tensor with one row per sample.
3. `CreateModel` - Creates a `ProbabilisticPca` model with 2 components, which uses the `Generator` to initialize its components. The model is fitted with the expectation-maximization (EM) algorithm, which runs for up to `Iterations` steps and stops early once the parameters change by less than the `Tolerance`.
4. `FitModel` - Fits the model to the `Data`. The fitted model is published as `FittedModel`, its noise variance as `EstimatedVariance`, and the log likelihood of the data after each step of the algorithm as `LogLikelihood`.
5. `ReconstructData` - Transforms the `Data` into the 2 `Latents` of the model and reconstructs all 16 channels from them.
6. `VisualizeData` - Displays the first channel of the `Data`, the `Signal` and the `Reconstruction` in the `Simulated data` window.
7. `VisualizeLogLikelihood` - Displays the log likelihood after each step of the algorithm in the `Log likelihood` window.

The value of `EstimatedVariance` is close to 0.25, the variance of the noise that was added to the signal. This value is not available from a standard `Pca` model, which describes how the variance is divided among components but not how much of it is noise. The log likelihood increases at every step and flattens as the algorithm converges.

In the `Simulated data` window, the reconstruction follows the signal much more closely than the data does, because the noise on each channel is independent of the other channels and is not captured by the 2 components.

To change the amount of noise, edit the `Variance` property of the `Normal` node inside `GenerateData`, which is set to 0.5. The node multiplies each sample by this value, so it acts as the standard deviation of the noise, and the estimated variance should be close to its square.
