## Automatic Differentiation


### Team Information:
Team ID: 8 

Members:
- Rajiv Koliwad (tct3rx)
- Malik Kurtz (kra8ku)
- Sasha (Alexander) Arditi (ukk8zb)

## Overview

### Task:

- Build a PyTorch-like automatic differentiation reverse mode package from scratch using only NumPy
- Train + Test Neural Network with this package
- Compare our package with pytorch only implementation

### Proposed Solution:

We can build a model with custom classes containing numpy arrays: a neural network class instance stores layer class instances, which in turn stores a “weights” tensor and a “bias” tensor, along with an activation function. these tensors are analogous to nodes in our pitch visualisation, but represent arrays rather than individual values. Tensors will contain numpy arrays, and we will use sklearn to facilitate easier testing.



## Usage:
* Note, use a kernel of Python 3.10. At submission time, this worked well
* Be sure to run `pip install -r requirements.txt` in order for the code to run

For simplicity, we transformed our code base into an ipynb file called "Ml_Final.ipynb"

From top to bottom, the code will add the nessecary packages, build the model, and compare it to a pytorch implementation.


In the "Dataset options" section of the code, chaning the variable in `set options()` will change the data that gets loaded into the models


Once you get to the evaluation and visualization part of the code, you will get visualizations about training error and testing performance. You will also get times it took to run.

At the end, you will be able to compare our results with that of a pytorch model.


## Video:

[Machine Learning Final Project - UVA CS 4774](https://youtu.be/1IJK0WWCo3E)


