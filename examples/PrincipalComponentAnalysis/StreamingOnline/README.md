# Fitting a PCA model online

The following example shows how a PCA model from the `Bonsai.ML.Pca.Torch` package can be fit online using the generalized Hebbian algorithm (GHA) from streaming data (MNIST).

## Dataset

The dataset used in this example can be obtained by going to [this url](http://yann.lecun.com/exdb/mnist/). This example uses both the training and testing datasets, so download all 4 of the urls. The workflow expects the datasets to be placed into the `datasets/MNIST/raw` folder. The path should be `datasets/MNIST/raw/*.gz`.

### Workflow

Below is the example workflow.

:::workflow
![](MnistStreamingPca.bonsai)
:::

The workflow can be broken down into the following sections.

1. `LoadData` - Uses a custom extension for loading the MNIST dataset from the `*.tar.gz` files contained in the `datasets/` folder. It loads the training images and labels as well as the testing images and labels for testing the results after fitting.
2. `CreateModel` - This is where we create the PCA model. In this case, we select the `OnlinePcaGha` model type and set the number of components that we want to fit to 2. We set the learning rate to 0.1. The `CreateGenerator` node with the seed set to 0 allows us to create a reproducible RNG for initializing the weights for this particular model.
3. `Run` - Inside this group workflow, there are 2 main data streams. The top stream sets up the branch that will feed the `TestImages` through the `Transform` node after the `Fitting` procedure has finished. In the second branch, the model is fed with `TrainImages`, which are passed to the `FitAndTransform` operator that simultaneously updates the models weights and transforms each element in the input sequence.