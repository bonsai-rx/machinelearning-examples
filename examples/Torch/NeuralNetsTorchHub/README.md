# Running Inference using a Torch Hub Model

In the following example, you will see how to run inference using a model downloaded directly from [torch hub](https://pytorch.org/hub/). The model we will be using is called the `MiDaS` model, which takes 2D images in RGB space and converts them into 1D images representing depth - more information about this model can be found [here](https://pytorch.org/hub/intelisl_midas_v2/).

### Python Script

This example contains a python script called `torchhub_download.py`. To run this, you need to have a python environment with the pytorch package installed. To install pytorch in python, you can follow the [instructions on the pytorch website](https://pytorch.org/get-started/locally/). Afterwards, you need to change to the example directory and run:

```cmd
python torchhub_download.py
```

This will download the model from torch hub and save it to a `.pt` file. The file will be saved inside of the `models` directory.

### Workflow

Below is the example workflow.

:::workflow
![Torch Hub Model](TorchHubModel.bonsai)
:::

The `CameraCapture` node grabs frames from the camera. These frames are then processed and converted into `Tensor` format. The model is loaded using the `LoadScriptModule` node and used as the `Model` property for the `Forward` node, which will perform forward inference of the image using the model. The remaining nodes process and convert the tensor back into an image.

