#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <ctype.h>
#include "mnist_dataset.h"
#include "nn_optimized.h"

// On-the-fly ±2px shift augmentation (like this we can get more data
// from the same image and get 1% better accuracy)
static void augment_shift(float *dst, const float *src)
{
    int dx = rand() % (2 * AUG_MAX_SHIFT + 1) - AUG_MAX_SHIFT;
    int dy = rand() % (2 * AUG_MAX_SHIFT + 1) - AUG_MAX_SHIFT;
    for (int r = 0; r < 28; r++)
    {
        for (int c = 0; c < 28; c++)
        {
            int rr = r - dy, cc = c - dx;
            if (rr >= 0 && rr < 28 && cc >= 0 && cc < 28)
                dst[r * 28 + c] = src[rr * 28 + cc];
            else
                dst[r * 28 + c] = 0.0f;
        }
    }
}

int main(void)
{
    srand((unsigned)time(NULL));

    Dataset train, test;
    if (load_dataset("mnist/fashion/train-images-idx3-ubyte",
                     "mnist/fashion/train-labels-idx1-ubyte", &train))
        return 1;
    if (load_dataset("mnist/fashion/t10k-images-idx3-ubyte",
                     "mnist/fashion/t10k-labels-idx1-ubyte", &test))
        return 1;

    // Normalize to [0,1]
    float *x_train = malloc(train.count * NN_INPUT * sizeof(float));
    float *x_test = malloc(test.count * NN_INPUT * sizeof(float));
    for (uint32_t i = 0; i < train.count; i++)
        for (int j = 0; j < NN_INPUT; j++)
            x_train[i * NN_INPUT + j] = train.images[i][j] / 255.0f;
    for (uint32_t i = 0; i < test.count; i++)
        for (int j = 0; j < NN_INPUT; j++)
            x_test[i * NN_INPUT + j] = test.images[i][j] / 255.0f;

    uint8_t *y_train = train.labels;
    uint8_t *y_test = test.labels;

    // Workspaces
    float *x_batch = malloc(BATCH_SIZE * NN_INPUT * sizeof(float));
    float *h1b = malloc(BATCH_SIZE * NN_HIDDEN1 * sizeof(float));
    float *h2b = malloc(BATCH_SIZE * NN_HIDDEN2 * sizeof(float));
    float *yb = malloc(BATCH_SIZE * NN_OUTPUT * sizeof(float));

    // TRAIN
    nn_init();
    float total_steps = (EPOCHS * (float)train.count) / BATCH_SIZE;
    float step = 0.0f;

    for (int ep = 1; ep <= EPOCHS; ep++)
    {
        printf("Epoch %d/%d\n", ep, EPOCHS);
        for (uint32_t s = 0; s < train.count; s += BATCH_SIZE)
        {
            uint32_t bs = (s + BATCH_SIZE <= train.count)
                              ? BATCH_SIZE
                              : (train.count - s);
            // one-cycle LR
            float pct = step++ / total_steps;
            float lr = (pct < 0.4f)
                           ? BASE_LR + (MAX_LR - BASE_LR) * (pct / 0.4f)
                           : MAX_LR * (1.0f - (pct - 0.4f) / 0.6f);
            nn_set_hyper(lr, WEIGHT_DECAY);

            // augment + batch copy
            for (uint32_t b = 0; b < bs; b++)
            {
                augment_shift(
                    x_batch + b * NN_INPUT,
                    x_train + (s + b) * NN_INPUT);
            }

            nn_forward_batch(x_batch, h1b, h2b, yb, bs);
            nn_backward_batch(x_batch, h1b, h2b, yb, y_train + s, bs);
        }
        double tr_acc = nn_evaluate(x_train, y_train, train.count);
        printf("Epoch %2d/%d — Train Acc: %.2f%%\n",
               ep, EPOCHS, tr_acc * 100);
    }

    // CLEANUP
    free(x_train);
    free(x_test);
    free(x_batch);
    free(h1b);
    free(h2b);
    free(yb);
    free_dataset(&train);
    free_dataset(&test);
    return 0;
}