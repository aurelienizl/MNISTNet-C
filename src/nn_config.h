#define NN_INPUT 784
#define NN_HIDDEN1 512
#define NN_HIDDEN2 1024
#define NN_OUTPUT 10
#define BATCH_SIZE 96
#define EPOCHS 3
#define BASE_LR 0.001f
#define MAX_LR 0.02f
#define WEIGHT_DECAY 1e-4f
#define AUG_MAX_SHIFT 2
#define MODEL_PATH "models/default.bin"

#include <math.h>
#include <stdlib.h>