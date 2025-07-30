# Zebrafish Centroid Tracking

The code for this repo can be found [here](https://github.com/bonsai-rx/machinelearning-examples/tree/main/examples/LinearDynamicalSystems/Kinematics/ZebrafishCentroidTracking).

In the following example, you can find how the Kalman Filter can be used to infer the kinematics of a freely swimming juvenille zebrafish.

### Instructions

Ensure you have Bonsai 2.9 installed on your system. You will also need to have the [uv Python environment manager](https://docs.astral.sh/uv/) installed. You can bootstrap the Python environment by running `uv sync` in the example directory.

### Dependencies

If you used the bootstrapping method, you dont have to worry about the package dependencies, as these should be already installed. However, if creating a new environment or integrating into an existing one, you will need to install the following packages:

* Bonsai - Core v2.8.1
* Bonsai - Design v2.8.0
* Bonsai - Editor v2.8.0
* Bonsai - ML v0.1.0
* Bonsai - ML LinearDynamicalSystems v0.1.0
* Bonsai - ML Visualizers v0.1.0
* Bonsai - Scripting v2.8.0
* Bonsai - Scripting Python v0.2.0
* Bonsai - Vision v2.8.1
* Bonsai - Vision Design v2.8.1

### Dataset

The video in this dataset was provided by Nicholas Guilbeault in the Thiele lab at the University of Toronto and published as part of Guilbeault, N.C., Guerguiev, J., Martin, M. et al. BonZeb: open-source, modular software tools for high-resolution zebrafish tracking and analysis. Sci Rep 11, 8148 (2021). [https://doi.org/10.1038/s41598-021-85896-x](https://doi.org/10.1038/s41598-021-85896-x).

You can download the `ZebrafishExampleVid.avi` video file here: [https://doi.org/10.5281/zenodo.10629221](https://doi.org/10.5281/zenodo.10629221). The workflow expects the video to be placed into the `datasets` folder but if you prefer to keep the video elsewhere, simply change the `Filename` property of the `ZebrafishTracking` group node to point to the correct location.

### Workflow

Below is the workflow for inferring kinematics of zebrafish swimming.

:::workflow
![Kinematics - Zebrafish Tracking](ZebrafishTracking.bonsai)
:::

In this example, a Kalman Filter is used to infer the position, velocity, and acceleration of a freely swimming zebrafish. The original video was collected at 200 Hz, so the `Fps` property of the `CreateKFModel` node is set to 200. The workflow works by performing centroid tracking inside of the `ZebrafishTracking` node, which performs image analyses to extract the `Centroid` of the zebrafish. The `Centroid` data is then converted into a type of `Observation2D` that the model then uses to perform inference using the  `PerformInference` node.

Since the tracking pipeline takes some time to initialize, we wait to create the kalman filter using the `SubscribeWhen` node only once `TrackingStarted` has produced a value, which is conditioned on centroid != NaN.

To visualize the inferred position, velocity, and acceleration kinematics, double click on the `ZebrafishKinematicsVisualizer` node at the bottom of the workflow while it is running to open up the visualizer. On the left, you should see the inferred position, velocity and acceleration (top to bottom), with the ability to select which state component (X or Y) you want to visualize from the dropdown menu on the top right corner of the graph. Both the mean (dark blue) and variance (light blue shading) of the inferred state component are visualized. On the right, you should see the video of the zebrafish playing, with the original tracking data (blue) and inferred position (red) overlaid.

The window should look similar to this:

![Zebrafish Tracking](ZebrafishTracking.gif)
