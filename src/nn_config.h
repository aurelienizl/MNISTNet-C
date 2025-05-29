#define NN_INPUT 784
#define NN_HIDDEN1 16
#define NN_HIDDEN2 8
#define NN_OUTPUT 10
#define BATCH_SIZE 64
#define EPOCHS 3
#define BASE_LR 0.001f
#define MAX_LR 0.02f
#define WEIGHT_DECAY 1e-4f
#define AUG_MAX_SHIFT 2
#define MODEL_PATH "models/default.bin"

#include <math.h>
#include <stdlib.h>

static inline float xavier(int in, int out) { return ((rand() / (float)RAND_MAX) * 2.f - 1.f) * sqrtf(6.f / (in + out)); }

static inline float relu(float x) { return x > 0 ? x : 0; }
static inline float drelu(float y) { return y > 0 ? 1 : 0; }