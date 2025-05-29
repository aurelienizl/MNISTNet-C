#ifndef NN_DATASET_H
#define NN_DATASET_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// That's remove the warning and handle errors
#define SAFE_FREAD(ptr, size, count, stream)            \
    do                                                  \
    {                                                   \
        if (fread(ptr, size, count, stream) != (count)) \
        {                                               \
            perror("fread");                            \
            return EXIT_FAILURE;                        \
        }                                               \
    } while (0)

typedef struct
{
    uint32_t count;     
    uint32_t rows;      
    uint32_t cols;      
    uint8_t **images;  
    uint8_t *labels;    
} Dataset;

static inline void free_dataset(Dataset *ds)
{
    if (ds->images) {
        for (uint32_t i = 0; i < ds->count; i++)
            free(ds->images[i]);
        free(ds->images);
    }
    if (ds->labels)
        free(ds->labels);
    ds->count = 0;
    ds->rows = 0;
    ds->cols = 0;
    ds->images = NULL;
    ds->labels = NULL;
}

#endif // NN_DATASET_H