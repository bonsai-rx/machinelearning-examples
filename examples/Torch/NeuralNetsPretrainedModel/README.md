# Image Classification using a Neural Network with Pretrained Model Weights

In the following example, you will see how to use a pretrained torch model to perform online image classification using the fashion MNIST dataset.

### Dataset

The dataset used in this example can be obtained by going to [this url](https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file#get-the-data). Only the files `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz` are needed for this example. The workflow expects the datasets to be placed into a folder called `fashion-mnist` inside of the `datasets` folder. The paths should be `datasets/fashion-mnist/t10k-images-idx3-ubyte.gz` and `datasets/fashion-mnist/t10k-labels-idx1-ubyte.gz`.

This example uses the model weights contained in the `fashion-mnist.model.bin` file inside of the [Bonsai.ML - Datasets repository](https://doi.org/10.5281/zenodo.10629221). The workflow expects the model weights to be placed inside of the `datasets` folder. The path should be `datasets/fashion-mnist.model.bin`.

### Workflow

Below is the example workflow.

:::workflow
![Image Classification - Pretrained](NeuralNetsPretrainedModel.bonsai)
:::

The workflow can be broken down into the following sections.
1. `LoadData` - Uses a custom source node called `FashionMnistLoader` to load the dataset files and output a `FashionMnistData` class which contains the testing images and testing labels. The images and labels are iterated to process individual images and labels.
2. `LoadPretrainedModel` - Loads the pretrained model and saves it to a subject. Since TorchSharp cannot infer the model's architecture from a binary weights file, it is necessary to specify the correct model architecture corresponding to the pretrained weights, otherwise it will fail. The weights in this example are based on the `Mnist` model architecture, a convolutional neural network with 2 convolutional layers followed by 2 fully connected layers.
3. `ProcessImage` - Converts the `IplImage` object and label into `Tensor` objects.
4. `RunInference` - Runs a forward pass on the `ProcessedImage`. The argmax of the output is taken as the predicted class label.
5. `Visualizer` - Displays the most recent image along with the history of observed target labels and the models predicted labels.

The demo should look like this when running:

![]()