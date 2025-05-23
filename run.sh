gcc -std=c99 -O2 mnist_dataset.c nn_optimized.c main.c -o mnist_simple -lm -fopenmp -march=native
./mnist_simple