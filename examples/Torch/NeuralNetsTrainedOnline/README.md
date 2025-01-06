# Image Classification using a Neural Network Trained Online

In the following example, you will see how to train a neural network online to perform image classification using the fashion MNIST dataset. This example relies on the GPU, so please read the instructions for how to set up GPU support for `Bonsai.ML.Torch`.

### Dataset

The dataset used in this example can be obtained by going to [this url](https://github.com/zalandoresearch/fashion-mnist). This example uses both the training and testing datasets. The files need to be converted into images and text files and placed into the `datasets/fashion-mnist` folder. You can convert the compressed files into the correct formats using [this tool](https://github.com/ncguilbeault/fashion-mnist-dataset-export).

### Workflow

Below is the example workflow.

:::workflow
![Image Classification - Online](NeuralNetsTrainedOnline.bonsai)
:::

The workflow can be broken down into the following sections.
1. `LoadData` - First, the training dataset is loaded and as quickly as possible passes the images and labels to the model to learn. Once the `stop training` button has been pressed, it will stop iterating through the training data. Finally, once the `start testing` button has been pressed, the testing data will be sampled at 1 Hz and the model will run inference only.
2. `LoadModel` - Loads the model and saves it to a subject. In this example, the model is initialized with completely random weights.
3. `ProcessInputs` - Takes the original image in `IplImage` format, converts it to a `Tensor` format of the appropriate shape and data type, and normalizes the image. Also takes the string label and converts it to a `Tensor` format.
4. `RunInference` - The model runs a forward pass on the `ProcessedImage`, and the argmax of the output is taken as the prediction.
5. `OnlineLearning` - Inputs into the model are collected in batches. Once a batch is filled, the model updates its weights by performing stochastic gradient descent. The loss is calculated as the mean squared error between the ground truth label and the prediction.
6. `Visualizer` - Displays the most recent image along with the history of observed target labels and the models predicted labels.
7. `Controller` - A panel with 2 buttons to control when the training will stop and when the test images will start.

