# Custom PyTorch Model Example

In this example, we demonstrate how to create a simple PyTorch model in Python, train it on the MNIST dataset, and then use Bonsai to load the model and run inference online.

### Python Guide

This example works best with the `uv` Python environment manager. Open up a terminal and run the following commands:

```bash
cd examples/Torch/CustomPyTorchModel
uv run main.py
```

The script will download the MNIST dataset into the `datasets` folder at the root of the examples directory and then train a simple convolutional neural network (CNN) to classify digits from the MNIST dataset. After training the model, the script will save the model to a file named `models/custom_pytorch_model.pt`.

### Workflow

The workflow for this example looks like this:

:::workflow
![Custom PyTorch Model](RunCustomPyTorchModel.bonsai)
:::

The workflow first loads the custom PyTorch model from the file, `models/custom_pytorch_model.pt`. It then subscribes to the `MnistLoader`, which will read the MNIST dataset and generate images and labels. The images are then processed by normalizing them and then passing them through the model's forward method to get the log probabilities for each label given the image. The `argmax` of the output is then used to determine the predicted label for each image.