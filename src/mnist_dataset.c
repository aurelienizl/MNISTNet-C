#include "mnist_dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint32_t be32(uint32_t x)
{
    return (x << 24) | ((x << 8) & 0x00FF0000) |
           ((x >> 8) & 0x0000FF00) | (x >> 24);
}

int load_dataset(const char *img_path, const char *lbl_path, Dataset *out)
{
    FILE *fi = fopen(img_path, "rb");
    FILE *fl = fopen(lbl_path, "rb");
    if (!fi || !fl)
    {
        perror("fopen");
        return -1;
    }
    uint32_t magic, n, rows, cols;
    SAFE_FREAD(&magic, 4, 1, fi);
    SAFE_FREAD(&n, 4, 1, fi);
    SAFE_FREAD(&rows, 4, 1, fi);
    SAFE_FREAD(&cols, 4, 1, fi);
    magic = be32(magic);
    n = be32(n);
    rows = be32(rows);
    cols = be32(cols);
    if (magic != 0x00000803)
        return -2;
    SAFE_FREAD(&magic, 4, 1, fl);
    SAFE_FREAD(&n, 4, 1, fl);
    magic = be32(magic);
    n = be32(n);
    if (magic != 0x00000801)
        return -3;

    out->count = n;
    out->rows = rows;
    out->cols = cols;
    size_t sz = rows * cols;
    out->images = malloc(n * sizeof(uint8_t *));
    out->labels = malloc(n * sizeof(uint8_t));
    for (uint32_t i = 0; i < n; ++i)
    {
        out->images[i] = malloc(sz);
        SAFE_FREAD(out->images[i], 1, sz, fi);
        SAFE_FREAD(&out->labels[i], 1, 1, fl);
    }
    fclose(fi);
    fclose(fl);
    return 0;
}

void free_dataset(Dataset *d)
{
    size_t sz = d->rows * d->cols;
    for (uint32_t i = 0; i < d->count; ++i)
        free(d->images[i]);
    free(d->images);
    free(d->labels);
}