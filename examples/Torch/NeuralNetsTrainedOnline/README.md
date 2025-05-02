# Image Classification using a Neural Network Trained Online

In the following example, you will see how to train a neural network online to perform image classification using the fashion MNIST dataset.

### Dataset

The dataset used in this example can be obtained by going to [this url](https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file#get-the-data). This example uses both the training and testing datasets, so download all 4 of the urls. The workflow expects the datasets to be placed into the `datasets/fashion-mnist` folder. The path should be `datasets/fashion-mnist/*.gz`.

### Workflow

Below is the example workflow.

:::workflow
![Image Classification - Online](NeuralNetsTrainedOnline.bonsai)
:::

The workflow can be broken down into the following sections.
1. `LoadData` - Uses a custom source node called `FashionMnistLoader` to load the dataset files and output a `FashionMnistData` class which contains the training images, training labels, testing images, and testing labels. The training dataset is iterated until the button is pressed to stop training and switch to the testing dataset. 
2. `LoadModel` - Loads the model and saves it to a subject. The model is initialized with random weights and will be trained online.
3. `ProcessInputs` - Converts the image, which is a type of `IplImage`, into a `Tensor` object. The tensor is normalized and reshaped. The training label is also converted into a `Tensor` object.
4. `RunInference` - Runs forward inference on the `ProcessedImage`, and the argmax of the output is taken as the predicted class label.
5. `OnlineLearning` - Image tensors are collected in batches. Once a batch is filled, the model trains against the loss between its prediction and the provided label by performing stochastic gradient optimization to update its weights. It uses the `Adam` optimizer and computes the `NegativeLogLikelihood`.
6. `Visualizer` - Displays the most recent image along with the history of observed target labels and the models predicted labels.
7. `Controller` - A button to control when the training will stop and when the test images will start.

Currently, this example runs on the CPU. To run this example on the GPU, follow the instructions for GPU installation. Navigate to the `LoadModel` tab, select the `InitializeTorchDevice` node upstream of the `CUDA` subject, and set the `DeviceType` to CUDA. 
