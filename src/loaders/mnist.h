#ifndef MNIST_DATASET_H
#define MNIST_DATASET_H
#include <stdint.h>

#include "../nn_dataset.h"

int mnist_load_dataset(const char *img_path,
                 const char *lbl_path,
                 Dataset *out);

#endif // MNIST_DATASET_H