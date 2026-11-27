# Forecasting Position with a Kalman Filter

In the following example, the `Bonsai.ML.Lds.Torch` package is used to estimate the velocity of an animal from its tracked position and to forecast where the animal will be 100 ms later. The model runs entirely in Bonsai and does not require a Python environment.

### Dataset

The data for this example was generously provided by the authors of "Joshi, A., Denovellis, E.L., Mankili, A. et al. (2023). Dynamic synchronization between hippocampal representations and stepping. Nature, 617, 125–131." [https://doi.org/10.1038/s41586-023-05928-6](https://doi.org/10.1038/s41586-023-05928-6). If you found this example useful, please consider citing their work.

You can download the `HippocampalTetrodeRecordings.zip` dataset here: [https://doi.org/10.5281/zenodo.10629221](https://doi.org/10.5281/zenodo.10629221). Only the `position.bin` file is needed for this example. It contains 2D tracking coordinates of the animal nose position detected using [DeepLabCut](https://deeplabcut.github.io/DeepLabCut/README.html), sampled at 500 Hz, while the animal navigates a W-shaped track.

Once downloaded, you will need to extract the contents of the zip file into the `datasets` folder. The workflow expects `position.bin` to be in the folder path `datasets/HippocampalTetrodeRecordings/`.

### Model

The latent state of the model contains the position and velocity of the animal along each axis, $x(t) = [p_x, p_y, v_x, v_y]^T$. Between samples, the position advances by the velocity multiplied by the time step $\Delta t$, and the velocity stays the same apart from the process noise:

$$
x(t+1) = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} x(t) + w(t)
$$

Only the position is observed:

$$
y(t) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} x(t) + \epsilon(t)
$$

The process noise $w(t)$ represents random changes in velocity, and the measurement noise $\epsilon(t)$ represents the error of the tracking.

### Running the Workflow

The example workflow is shown here:

:::workflow
![Forecasting Position](ForecastingPosition.bonsai)
:::

The `CreateModel` group node uses the `CreateKalmanFilter` operator to specify the model. The `TransitionMatrix` and `MeasurementFunction` properties are set to the matrices above, with a time step of `0.002` seconds to match the 500 Hz sample rate. The `ProcessNoiseVariance` property is set to the covariance of a random acceleration with an intensity of 100 units²/s³, and the `MeasurementNoiseVariance` is set to `0.1`. The model is shared as `Model`.

The `LoadData` group node reads the position from disk with the `MatrixReader` operator, one sample at a time, and converts each sample into a `Tensor` with dimensions `time` x `position` (in this case, time is `1`). The samples are shared as `Position`.

In the `EstimateKinematics` group node, each position is passed to the `Filter` operator together with the `Model`. The mean of the filtered state is shared as `State`. Its first two elements are shared as `FilteredPosition`, its last two elements as `Velocity`, and the norm of the velocity as `Speed`.

The `Forecast` group node multiplies the `Velocity` by the forecast horizon of `0.1` seconds and adds the result to the `FilteredPosition`. This is the position the model expects the animal to reach 100 ms later, and it is shared as `Forecast`.

The `EvaluateForecast` group node compares each forecast with the position that is measured 50 samples, or 100 ms, later. The distance between them is shared as `ForecastError`. For reference, the distance between the current position and the position 100 ms later is shared as `LastPositionError`, which is the error of assuming that the animal stays where it is.

The `Visualizer` group node shows 3 graphs. The first graph shows the last 2 seconds of the animal's position along with the current forecast. The second graph shows the `ForecastError` and the `LastPositionError` over time. The third graph shows the `Speed` of the animal.

When the animal is running along the track, at a speed above 20 units per second, the median `ForecastError` is about 2 units and the median `LastPositionError` is about 6 units. When the animal is stationary, there is no movement to forecast and the two errors are similar, with the last position being slightly more accurate. The forecast horizon can be changed by setting the value of the `CreateTensor` operator in the `Forecast` group node and the `Count` of the `Skip` operators in the `EvaluateForecast` group node to the same duration.

The `ProcessNoiseVariance` controls how quickly the velocity estimate follows changes in the movement of the animal. Larger values follow changes in direction more closely but make the `Speed` estimate noisier, and smaller values give a smoother estimate that lags behind.
