# Online PCA of striatal population activity

The following example shows how the online models from the `Bonsai.ML.Pca.Torch` package can be fitted to neural population activity as it arrives, using spike counts from Neuropixel recordings of striatum.

## Dataset

The data for this example was generously provided by the authors of "Thompson, E., Rollik, L., Waked, B. et al. (2024). Replay of procedural experience is independent of the hippocampus. bioRxiv, 2024.06.05.597547." If you found this example useful, please consider citing their work.

You can download the `StriatalNeuropixelRecordings.zip` dataset here: [https://doi.org/10.5281/zenodo.10629221](https://doi.org/10.5281/zenodo.10629221). This example uses the `spike_counts.bin` file, which contains the spike counts from 564 spike sorted units detected using [KiloSort](https://github.com/MouseLand/Kilosort), sampled at 60 Hz. Once downloaded, you will need to extract the contents of the zip file into the `datasets` folder. The workflow expects the file to be in the folder path `datasets/StriatalNeuropixelRecordings/`.

### Workflow

Below is the example workflow.

:::workflow
![](OnlineLatentsFromStriatum.bonsai)
:::

The workflow can be broken down into the following sections.

1. `ManualSeed` - Sets the seed of the random number generator, so that the models start from the same components every time the workflow runs. The generator is shared as `Generator`.
2. `LoadData` - Reads the spike counts from disk with the `MatrixReader` node, 12 samples at a time, and sums them to give the number of spikes of each unit in bins of 200 ms. The square root of each count plus 0.5 is then taken, which keeps the units with the highest firing rates from dominating the components. Every 5 bins, which is 1 second of the recording, are concatenated into a batch and shared as `SpikeCounts`. The `SampleRate` of the `MatrixReader` is set to 600, which plays the recording at 10 times its original speed. Set it to 60 to play the recording in real time.
3. `CreateModels` - Creates an `OnlineProbabilisticPca` model and an `OnlinePcaGha` model, each with 3 components. Both use the `Generator` to initialize their components.
4. `FitProbabilistic` - Passes each batch to the `FitAndTransform` operator together with the `OnlineProbabilisticPca` model, which updates the model and transforms the batch in a single step. The transformed data is published as `ProbabilisticLatents`, the noise variance of the model as `NoiseVariance`, and the mean of the model for a single unit, unit 288, as `ProbabilisticMean`.
5. `FitGha` - Does the same with the `OnlinePcaGha` model. The mean of the model for the same unit is published as `GhaMean`.
6. `UnitActivity` - Selects the transformed spike counts of that unit and averages them over each batch.
7. `VisualizeLatents` - Displays the 3 latents of the `OnlineProbabilisticPca` model over time in the `Latents` window.
8. `VisualizeBaseline` - Displays the activity of the selected unit together with the mean that each model has learned for it, in the `Baseline of unit 288` window.
9. `VisualizeNoiseVariance` - Displays the noise variance of the `OnlineProbabilisticPca` model over time in the `Noise variance` window.

Both models are updated once per batch, in a single pass over the recording, without holding the recording in memory.

The values of the latents during the first part of the recording should be interpreted with care. While a model is still learning, its components change from one batch to the next, so a latent does not have a fixed meaning over time. The latents become comparable over time once the model has settled, which can be seen in the `Noise variance` window as the noise variance stops decreasing quickly.

The `Baseline of unit 288` window shows a difference between the two models. Unit 288 is the unit whose average firing rate changes the most over the recording relative to its variability, which makes the difference easy to see. Most units change less, and for those the means of the two models stay closer together. To compare a different unit, change the index in the `Index` nodes of `FitProbabilistic`, `FitGha` and `UnitActivity`.

The firing rate of unit 288 decreases over the course of the recording, from about 9 spikes per bin in the first quarter to about 1 in the last. After the square root is taken, this is a decrease from about 3 to about 1.3, which is what the window shows. The `OnlineProbabilisticPca` model weighs recent data more heavily, at a rate set by its `Rho` property, so its mean follows the firing rate down. The `OnlinePcaGha` model uses the average of all the data it has seen, so its mean stays well above the current firing rate. The `OnlineProbabilisticPca` model is therefore the better choice when the baseline of the data is expected to drift.
