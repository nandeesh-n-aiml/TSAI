# PART 1
## Backpropagation  

The below image refers to the backpropagation calculated using excel. The [`backpropagation.xlsx`](./Backpropagation.xlsx) excel file contains the calculations.

<img src="./images/Backpropagation.png" />

### Steps to calculate gradients:
- The `Formulae` section refers to the forward pass. Here, the outputs are updated based on the inputs and weights. At last, the total loss is calculated.
- The `Output layer` and the `Hidden layer` blocks refer to the gradient calculation of the network loss w.r.t output and the hidden layer weights. This block provides the formulae for the gradient calculation.
- The `Derivatives` block contain helper derivatives to compute the final derivative for every weight.
- The `Derivatives` section (highlighted) in the table calculates the gradient for each weight in the network.
- Based on the gradient, the corresponding weights are updated.
- The total loss is calculated everytime after all the weights are updated. The calculations are performed ~100 times to reduce the loss which can also be noted from the plot.

### Visualise loss for different learning rates:
Learning rate: 0.1
<img src="./images/LR_0.1.png" />

Learning rate: 0.2
<img src="./images/LR_0.2.png" />

Learning rate: 0.5
<img src="./images/LR_0.5.png" />

Learning rate: 0.8
<img src="./images/LR_0.8.png" />

Learning rate: 1
<img src="./images/LR_1.png" />

Learning rate: 2
<img src="./images/LR_2.png" />

# PART 2
**Requirement:** Training a CNN model having less than 20K parameters. The network should be able to achieve 99.4% validation accuracy within 20 epochs.

**Solution:**  
Parameters: 18,418  
Validation accuracy: 99.26%  
Used 1x1 convolution + GAP

The notebook can be found [here](./S6.ipynb).