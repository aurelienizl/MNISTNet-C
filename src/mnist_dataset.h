#ifndef MNIST_DATASET_H
#define MNIST_DATASET_H
#include <stdint.h>

// That's remove the warning and handle errors
#define SAFE_FREAD(ptr, size, count, stream)            \
    do                                                  \
    {                                                   \
        if (fread(ptr, size, count, stream) != (count)) \
        {                                               \
            perror("fread");                            \
            return -1;                                  \
        }                                               \
    } while (0)

typedef struct
{
    uint32_t count;
    uint32_t rows, cols;
    uint8_t **images; // [count][rows*cols]
    uint8_t *labels;  // [count]
} Dataset;

int load_dataset(const char *img_path,
                 const char *lbl_path,
                 Dataset *out);

void free_dataset(Dataset *d);
#endif // MNIST_DATASET_H