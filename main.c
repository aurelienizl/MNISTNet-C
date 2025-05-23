// File: main.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <ctype.h>
#include "mnist_dataset.h"
#include "nn_optimized.h"

#define EPOCHS        20
#define BASE_LR       0.001f
#define MAX_LR        0.02f
#define WEIGHT_DECAY  1e-4f
#define AUG_MAX_SHIFT 2
#define MODEL_PATH    "model.bin"

// On-the-fly ±2px shift augmentation
static void augment_shift(float *dst, const float *src) {
    int dx = rand() % (2*AUG_MAX_SHIFT + 1) - AUG_MAX_SHIFT;
    int dy = rand() % (2*AUG_MAX_SHIFT + 1) - AUG_MAX_SHIFT;
    for(int r = 0; r < 28; r++){
        for(int c = 0; c < 28; c++){
            int rr = r - dy, cc = c - dx;
            if(rr >= 0 && rr < 28 && cc >= 0 && cc < 28)
                dst[r*28 + c] = src[rr*28 + cc];
            else
                dst[r*28 + c] = 0.0f;
        }
    }
}

// Render a 28×28 float image in ASCII
static void print_image(const float *x) {
    for(int r = 0; r < 28; r++){
        for(int c = 0; c < 28; c++){
            float v = x[r*28 + c];
            char ch = (v > 0.8f ? '#' :
                       v > 0.5f ? '*' :
                       v > 0.2f ? '.' : ' ');
            putchar(ch);
        }
        putchar('\n');
    }
}

int main(void) {
    srand((unsigned)time(NULL));

    Dataset train, test;
    if(load_dataset("data/mnist/train-images-idx3-ubyte",
                    "data/mnist/train-labels-idx1-ubyte",&train)) return 1;
    if(load_dataset("data/mnist/t10k-images-idx3-ubyte",
                    "data/mnist/t10k-labels-idx1-ubyte",&test)) return 1;

    // Normalize to [0,1]
    float *x_train = malloc(train.count * NN_INPUT * sizeof(float));
    float *x_test  = malloc(test.count  * NN_INPUT * sizeof(float));
    for(uint32_t i = 0; i < train.count; i++)
      for(int j = 0; j < NN_INPUT; j++)
        x_train[i*NN_INPUT + j] = train.images[i][j] / 255.0f;
    for(uint32_t i = 0; i < test.count; i++)
      for(int j = 0; j < NN_INPUT; j++)
        x_test[i*NN_INPUT + j]  = test.images[i][j]  / 255.0f;

    uint8_t *y_train = train.labels;
    uint8_t *y_test  = test.labels;

    // Workspaces
    float *x_batch = malloc(BATCH_SIZE * NN_INPUT  * sizeof(float));
    float *h1b     = malloc(BATCH_SIZE * NN_HIDDEN1 * sizeof(float));
    float *h2b     = malloc(BATCH_SIZE * NN_HIDDEN2 * sizeof(float));
    float *yb      = malloc(BATCH_SIZE * NN_OUTPUT  * sizeof(float));

    // Ask user whether to load or train
    printf("Load pretrained model from \"%s\"? (y/n): ", MODEL_PATH);
    char choice = getchar();
    while(getchar()!='\n');  // flush rest of line

    if(tolower(choice) == 'y') {
        if(nn_load(MODEL_PATH) != 0) {
            fprintf(stderr, "Failed to load \"%s\", will train a new model.\n", MODEL_PATH);
        } else {
            printf("Model loaded. Skipping training.\n");
            goto SKIP_TRAIN;
        }
    }

    // TRAIN
    nn_init();
    float total_steps = (EPOCHS * (float)train.count) / BATCH_SIZE;
    float step = 0.0f;

    for(int ep = 1; ep <= EPOCHS; ep++) {
        for(uint32_t s = 0; s < train.count; s += BATCH_SIZE) {
            uint32_t bs = (s + BATCH_SIZE <= train.count)
                        ? BATCH_SIZE
                        : (train.count - s);
            // one-cycle LR
            float pct = step++ / total_steps;
            float lr  = (pct < 0.4f)
                      ? BASE_LR + (MAX_LR-BASE_LR)*(pct/0.4f)
                      : MAX_LR * (1.0f - (pct-0.4f)/0.6f);
            nn_set_hyper(lr, WEIGHT_DECAY);

            // augment + batch copy
            for(uint32_t b = 0; b < bs; b++){
                augment_shift(
                  x_batch + b*NN_INPUT,
                  x_train + (s + b)*NN_INPUT
                );
            }

            nn_forward_batch (x_batch, h1b, h2b, yb, bs);
            nn_backward_batch(x_batch, h1b, h2b, yb, y_train + s, bs);
        }
        double tr_acc = nn_evaluate(x_train, y_train, train.count);
        printf("Epoch %2d/%d — Train Acc: %.2f%%\n",
               ep, EPOCHS, tr_acc*100);
    }

    if(nn_save(MODEL_PATH) != 0)
        fprintf(stderr, "Warning: could not save model to \"%s\"\n", MODEL_PATH);
    else
        printf("Trained model saved to \"%s\"\n", MODEL_PATH);

SKIP_TRAIN:;
    // FINAL TEST ACCURACY
    double test_acc = nn_evaluate(x_test, y_test, test.count);
    printf("Test Acc: %.2f%%\n\n", test_acc*100);

    // INTERACTIVE TEST
    printf("Interactive test (enter 0–9 to guess, q to quit)\n");
    char buf[16];
    for(uint32_t i = 0; i < test.count; i++){
        print_image(x_test + i*NN_INPUT);
        printf("Your guess: ");
        if(!fgets(buf, sizeof(buf), stdin)) break;
        if(buf[0]=='q' || buf[0]=='Q') break;
        int user_guess = buf[0] - '0';
        int model_pred = nn_predict(x_test + i*NN_INPUT);
        uint8_t actual = y_test[i];
        printf(" Model: %d — Actual: %d — You: %d — %s\n\n",
               model_pred, actual, user_guess,
               (user_guess == actual) ? "✓ correct":"✗ wrong");
    }

    // CLEANUP
    free(x_train); free(x_test);
    free(x_batch); free(h1b);
    free(h2b);     free(yb);
    free_dataset(&train);
    free_dataset(&test);
    return 0;
}