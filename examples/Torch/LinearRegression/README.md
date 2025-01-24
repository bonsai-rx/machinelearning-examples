# Linear Regression using Torch

In the following example, you will see how the Bonsai.ML.Torch package can be used to perform linear regression in Bonsai using the ordinary least squares method.

### Workflow

Below is the example workflow.

:::workflow
![Linear Regression using Torch](Simulation.bonsai)
:::

We use a linear model to generate simulated data. The linear model takes the form:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

In the `SyntheticDataset` group node, we define the parameters of the linear model that are used to generate our data. The properties `a0` and `a1` correspond to the values $\beta_0$ and $\beta_1$ in our linear model, and the `sigma` property corresponds to the variance in our gaussian noise parameter $\epsilon$. The `SampleRate` parameter determines the rate at which data points are generated.

Ordinary least squares is a method used to obtain the unknown parameters of an underlying linear model from observations of data. It minimizes the least squares difference between the observed input data and the associated output data. Provided that the parameters in $\mathbf{\beta}$ are linearly independent, and that $\epsilon$ is gaussian with a conditional mean of 0, then the parameters $\mathbf{\beta}$ can be solved using the normal equation, a closed-form solution to the minimization problem:

$$
\hat{\beta} = \beta + {(X^TX)}^{-1}X^Ty
$$

In the workflow example, we can compute this by concatenate each new data point to the existing collection of datapoints. Then, we extract the columns of the matrix corresponding to the input matrix $\mathbf{X}$, and extract the column vector corresponding to the output $\mathbf{y}$. The final step is to combine the matrices and compute the normal equation, giving our final output matrix $\hat{\beta}$ containing the estimated parameters.  