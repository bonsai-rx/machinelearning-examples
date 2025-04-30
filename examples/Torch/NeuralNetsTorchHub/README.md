# Running Inference using a Torch Hub Model

In the following example, you will see how to run inference using a model downloaded directly from [torch hub](https://pytorch.org/hub/). The model we will be using is called the `MiDaS` model, which takes 2D images in RGB space and converts them into 1D images representing depth - more information about this model can be found [here](https://pytorch.org/hub/intelisl_midas_v2/).

### Instructions

This example uses the [uv manager](https://docs.astral.sh/uv/) to manage package dependencies and boostrap the environment. To install `uv`, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/). 

The folder contains a Python script which uses the [PyTorch](https://pytorch.org) library. The python script, `torchhub_download.py`, is designed to automatically download the `MiDaS` PyTorch model from the web and place it into the correct folder for the Bonsai workflow. You can bootstrap the Python environment and run the script using uv. For this, open a terminal, change to the example directory, and run the script using uv:

```cmd
cd examples/Torch/Neural
uv run torchhub_download.py
```

Once the script is finished, the model will be saved as `MiDaS_small.pt` inside of the `models` directory.

### Workflow

Below is the example workflow.

:::workflow
![Torch Hub Model](TorchHubModel.bonsai)
:::

The `CameraCapture` node grabs frames from the camera. These frames are then processed and converted into `Tensor` format. The model is loaded using the `LoadScriptModule` node and used as the `Model` property for the `Forward` node, which will perform forward inference of the image using the model. The remaining nodes process and convert the tensor back into an image.

