// cifar10_dataset.h
#ifndef CIFAR10_DATASET_H
#define CIFAR10_DATASET_H

#include <stdint.h>
#include "../nn_dataset.h"
/**
 * Load one or more CIFAR-10 binary batch files (.bin) into `out`.
 *
 * @param batch_paths   array of paths to CIFAR-10 batch files (e.g. data_batch_1.bin, …, test_batch.bin)
 * @param num_batches   number of files in batch_paths
 * @param out           pointer to an uninitialized Dataset; on success it will be filled in
 *
 * @return 0 on success, -1 on I/O error, -2 on malloc failure.
 */
int load_cifar10_batches(const char *batch_paths[], int num_batches, Dataset *out);

#endif // CIFAR10_DATASET_H
