// cifar10_dataset.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "cifar10.h"

/* Grayscale conversion weights */
#define W_R 0.299f
#define W_G 0.587f
#define W_B 0.114f

int load_cifar10_batches(const char *batch_paths[], int num_batches, Dataset *out)
{
    const uint32_t IMG_SIZE = 32 * 32;
    const uint32_t RECORD_SIZE = 1 + 3 * IMG_SIZE;  // 1 label + 3072 pixels
    const uint32_t IMAGES_PER_BATCH = 10000;

    out->count = IMAGES_PER_BATCH * num_batches;
    out->rows  = 32;
    out->cols  = 32;

    /* allocate top-level arrays */
    out->images = malloc(out->count * sizeof(uint8_t *));
    out->labels = malloc(out->count * sizeof(uint8_t));
    if (!out->images || !out->labels) {
        perror("malloc");
        return -2;
    }

    uint8_t *raw = malloc(3 * IMG_SIZE);
    if (!raw) {
        perror("malloc");
        return -2;
    }

    uint32_t idx = 0;
    for (int b = 0; b < num_batches; ++b)
    {
        FILE *fp = fopen(batch_paths[b], "rb");
        if (!fp) {
            perror(batch_paths[b]);
            free(raw);
            return -1;
        }

        for (uint32_t i = 0; i < IMAGES_PER_BATCH; ++i, ++idx)
        {
            /* allocate storage for this grayscale image */
            out->images[idx] = malloc(IMG_SIZE);
            if (!out->images[idx]) {
                perror("malloc");
                fclose(fp);
                free(raw);
                return -2;
            }

            /* read label */
            SAFE_FREAD(&out->labels[idx], 1, 1, fp);

            /* read raw RGB */
            SAFE_FREAD(raw, 1, 3 * IMG_SIZE, fp);

            /* convert to grayscale: linear matrix */
            for (uint32_t p = 0; p < IMG_SIZE; ++p) {
                uint8_t r = raw[p];
                uint8_t g = raw[p + IMG_SIZE];
                uint8_t b = raw[p + 2*IMG_SIZE];
                float gray_f = W_R * r + W_G * g + W_B * b;
                out->images[idx][p] = (uint8_t)(gray_f + 0.5f);
            }
        }

        fclose(fp);
    }

    free(raw);
    return 0;
}