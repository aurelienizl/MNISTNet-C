#ifndef NN_IMLP_H
#define NN_IMLP_H

#include <stdint.h>
#include <string.h>

#define NN_INPUT 784
#define NN_HIDDEN1 128
#define NN_HIDDEN2 64
#define NN_OUTPUT 10
#define BATCH_SIZE 64
#define EPOCHS 50
#define BASE_LR 0.001f
#define MAX_LR 0.02f
#define WEIGHT_DECAY 1e-4f
#define AUG_MAX_SHIFT 2
#define MODEL_PATH "models/default.bin"

// initialize weights/biases
void nn_init(void);

// do a forward pass on a batch:
//  - x:  bs × NN_INPUT floats
//  - h1: bs × NN_HIDDEN1 workspace
//  - h2: bs × NN_HIDDEN2 workspace
//  - y:  bs × NN_OUTPUT outputs (softmax probs)
void nn_forward_batch(const float *x,
                      float *h1, float *h2,
                      float *y, uint32_t bs);

// do a backward pass + weight update on one batch:
//  - x, h1, h2, y as above
//  - lbls: bs labels in [0..9]
//  - bs: batch size
void nn_backward_batch(const float *x,
                       const float *h1, const float *h2,
                       const float *y, const uint8_t *lbls,
                       uint32_t bs);

// evaluate accuracy on N examples:
//  - x: N × NN_INPUT
//  - lbls: length-N
double nn_evaluate(const float *x,
                   const uint8_t *lbls,
                   uint32_t N);

// set global learning‐rate & weight‐decay
void nn_set_hyper(float lr, float weight_decay);

int nn_predict(const float *x);

#endif // NN_IMLP_H
