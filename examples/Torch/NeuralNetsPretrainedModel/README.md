# Image Classification using a Neural Network with Pretrained Model Weights

In the following example, you will see how to use a pretrained torch model to perform online image classification using the fashion MNIST dataset.

### Dataset

The dataset used in this example can be obtained by going to [this url](https://github.com/zalandoresearch/fashion-mnist). Only the datasets `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz` are needed for this example. The files need to be converted into images and text files and placed into the `datasets/fashion-mnist` folder. You can convert the compressed files into the correct formats using [this tool](https://github.com/ncguilbeault/fashion-mnist-dataset-export).

### Workflow

Below is the example workflow.

:::workflow
![Image Classification - Pretrained](FashionMnistPretrainedModel.bonsai)
:::

The workflow can be broken down into the following sections.
1. `LoadData` - This group node first enumerates and sorts all of the files inside the `datasets/fashion-mnist` directory matching the `t10k*` pattern and loads each image and label iteratively at a 1 Hz sampling rate.
2. `LoadPretrainedModel` - Loads the pretrained model and saves it to a subject. In this example, only the model weights are saved. Since TorchSharp cannot infer the models architecture from the binary weights file alone, we need to specify what model architecture the model weights correspond to. The weights in this example are based on the `MNIST` model architecture.
3. `ProcessImage` - Takes the original image in `IplImage` format, converts it to a `Tensor` format of the appropriate shape and data type, and normalizes the image.
4. `RunInference` - The model runs a forward pass on the `ProcessedImage`, and the argmax of the output is taken as our prediction.
5. `Visualizer` - Displays the most recent image along with the history of observed target labels and the models predicted labels.