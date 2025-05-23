# MNISTNet-C

A high-performance, pure-C implementation of a feed-forward neural network for MNIST digit classification, achieving over **99%** test accuracy with advanced training and regularization techniques.

## Purpose

MNISTNet-C demonstrates how to build, train, persist, and interact with a state-of-the-art digit classifier **entirely in C**, without external ML libraries. Ideal for:

- Learning the internals of neural-network training  
- Embedding a compact model in resource-constrained environments  
- Rapid experimentation with network and training hyperparameters  

## Features

- **Two hidden layers** (512 → 256) with ReLU  
- **Data augmentation**: on-the-fly ±2 px shifts  
- **Regularization**: dropout (p=0.5), label-smoothing (ε=0.1), L2 weight decay  
- **One-cycle learning-rate schedule** (base_lr=0.001 → max_lr=0.02)  
- **Momentum** (0.9) and **OpenMP**-accelerated batched training  
- **Model persistence**: save/load to \`model.bin\`  
- **Interactive CLI**: ASCII-rendered test images, user guess + model prediction  

## Repository Structure

    /
    ├── mnist/*                 # MNIST IDX files (included)
    ├── src/
    │   ├── mnist_dataset.c/.h  # IDX loader
    │   ├── nn_optimized.c/.h   # Network implementation
    │   └── main.c              # Train/load/predict + interactive loop
    ├── run.sh                  # Build & run script
    ├── Makefile                # Build instructions
    ├── models/*                # Saved model weights (after training)
    └── README.md               # This file

## Prerequisites

- C compiler with **C99** support (e.g. \`gcc\`, \`clang\`)  
- **Optional**: OpenMP (\`-fopenmp\`) for speed  
- MNIST dataset files in \`data/\`:
  - \`train-images-idx3-ubyte\`
  - \`train-labels-idx1-ubyte\`
  - \`t10k-images-idx3-ubyte\`
  - \`t10k-labels-idx1-ubyte\`

## Building

Simply run:

\`\`\`bash
make
\`\`\`

This compiles everything into the \`mnist_simple\` executable.

## Running

\`\`\`bash
./mnist_simple
\`\`\`

- On first run you’ll be prompted to **load** an existing \`model.bin\` or **train** a new model.  
- After training/loading, you’ll see the overall test accuracy, then enter an **interactive** mode where you can guess individual digits.

## Customization

- **Hyperparameters**: edit \`BASE_LR\`, \`MAX_LR\`, \`EPOCHS\`, \`WEIGHT_DECAY\`, \`AUG_MAX_SHIFT\` in \`main.c\`.  
- **Architecture**: adjust \`NN_HIDDEN1\` and \`NN_HIDDEN2\` in \`nn_optimized.h\`.  
- **Extensions**: swap in convolutional layers or other optimizers in \`nn_optimized.c/.h\`.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

