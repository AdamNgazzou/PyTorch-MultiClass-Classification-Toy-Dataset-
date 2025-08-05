# Multi-class Classification with PyTorch

This notebook demonstrates how to build and train a multi-class classification model using PyTorch.

## Dataset

The model is trained on a toy multi-class dataset generated using `sklearn.datasets.make_blobs`.

- Number of samples: 1000
- Number of features: 2
- Number of classes: 4
- Cluster standard deviation: 1.5
- Random state: 42

The dataset is split into training (80%) and testing (20%) sets.

## Model Architecture

A simple feed-forward neural network is used for classification.

- Input features: 2
- Output features: 4 (corresponding to the 4 classes)
- Hidden units: 8

The model uses a sequential linear layer stack.

## Training

- Loss function: `nn.CrossEntropyLoss()`
- Optimizer: `torch.optim.SGD` with a learning rate of 0.1
- Number of epochs: 100

The training loop includes steps for:
- Forward pass to get logits
- Converting logits to prediction probabilities using `torch.softmax`
- Converting prediction probabilities to labels using `torch.argmax`
- Calculating loss and accuracy
- Backpropagation and optimizer step

## Evaluation

The model is evaluated on the test set using:
- `nn.CrossEntropyLoss()` for test loss
- Custom `accuracy_fn` for test accuracy

The model's decision boundary is visualized for both the training and test sets.

## Results

The training and testing results are printed every 10 epochs, showing the loss and accuracy. The final test accuracy is also reported.

## Dependencies

- torch
- matplotlib
- sklearn
- requests
- pathlib

A helper function `plot_decision_boundary` is used from `helper_functions.py`, which is downloaded if not present.
