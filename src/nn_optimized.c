/* nn_optimized.c (factorized) */
#include "nn_optimized.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>

// Hyperparameters
static float LR = 0.01f;
static float WD = 1e-4f;
static const float MOM = 0.9f;

// Parameters and momentum buffers
static float W1[NN_HIDDEN1][NN_INPUT], B1[NN_HIDDEN1];
static float W2[NN_HIDDEN2][NN_HIDDEN1], B2[NN_HIDDEN2];
static float W3[NN_OUTPUT][NN_HIDDEN2], B3[NN_OUTPUT];
static float VdW1[NN_HIDDEN1][NN_INPUT], VdB1[NN_HIDDEN1];
static float VdW2[NN_HIDDEN2][NN_HIDDEN1], VdB2[NN_HIDDEN2];
static float VdW3[NN_OUTPUT][NN_HIDDEN2], VdB3[NN_OUTPUT];

// Activation
static inline float relu(float x) { return x > 0 ? x : 0; }
static inline float drelu(float y) { return y > 0 ? 1 : 0; }

// Xavier initialization
static float xavier(int in, int out)
{
    return ((rand() / (float)RAND_MAX) * 2.f - 1.f) * sqrtf(6.f / (in + out));
}

// Generic fully-connected forward
static void layer_forward(const float *restrict in, int in_dim,
                          const float *bias, const float *restrict W,
                          int out_dim, float (*act)(float),
                          float *restrict out)
{
    for (int i = 0; i < out_dim; i++)
    {
        float s = bias[i];
#pragma omp simd
        for (int j = 0; j < in_dim; j++)
            s += W[i * in_dim + j] * in[j];
        out[i] = act ? act(s) : s;
    }
}

// Max-stable softmax
static void softmax(const float *z, int n, float *out)
{
    float m = z[0];
#pragma omp simd reduction(max : m)
    for (int i = 1; i < n; i++)
        m = fmaxf(m, z[i]);
    float sum = 0;
#pragma omp simd reduction(+ : sum)
    for (int i = 0; i < n; i++)
        sum += (out[i] = expf(z[i] - m));
#pragma omp simd
    for (int i = 0; i < n; i++)
        out[i] /= sum;
}

// Backprop through one relu layer
static void layer_back_relu(const float *delta_next, const float *W_next,
                            int out_dim, int in_dim,
                            const float *act_prev, float *delta)
{
#pragma omp simd
    for (int i = 0; i < in_dim; i++)
    {
        float s = 0;
#pragma omp simd
        for (int k = 0; k < out_dim; k++)
            s += W_next[k * in_dim + i] * delta_next[k];
        delta[i] = s * drelu(act_prev[i]);
    }
}

// Update weights & biases with momentum & weight decay
typedef struct
{
    float *W, *B, *dW, *dB, *vW, *vB;
    int in, out;
} Layer;
static void update_layer(Layer L)
{
    for (int i = 0; i < L.out; i++)
    {
        float gB = L.dB[i] / L.out;
        L.vB[i] = MOM * L.vB[i] - LR * gB;
        L.B[i] += L.vB[i];
#pragma omp simd
        for (int j = 0; j < L.in; j++)
        {

            int idx = i * L.in + j;
            float gW = L.dW[idx] / L.out + WD * L.W[idx];
            L.vW[idx] = MOM * L.vW[idx] - LR * gW;
            L.W[idx] += L.vW[idx];
        }
    }
}

void nn_set_hyper(float lr, float weight_decay)
{
    LR = lr;
    WD = weight_decay;
}

void nn_init(void)
{
    srand((unsigned)time(NULL));
    // init weights + zero momentum
    for (int i = 0; i < NN_HIDDEN1; i++)
        for (int j = 0; j < NN_INPUT; j++)
        {
            W1[i][j] = xavier(NN_INPUT, NN_HIDDEN1);
            VdW1[i][j] = 0;
        }
    for (int i = 0; i < NN_HIDDEN2; i++)
        for (int j = 0; j < NN_HIDDEN1; j++)
        {
            W2[i][j] = xavier(NN_HIDDEN1, NN_HIDDEN2);
            VdW2[i][j] = 0;
        }
    for (int i = 0; i < NN_OUTPUT; i++)
        for (int j = 0; j < NN_HIDDEN2; j++)
        {
            W3[i][j] = xavier(NN_HIDDEN2, NN_OUTPUT);
            VdW3[i][j] = 0;
        }
    memset(B1, 0, sizeof B1);
    memset(VdB1, 0, sizeof VdB1);
    memset(B2, 0, sizeof B2);
    memset(VdB2, 0, sizeof VdB2);
    memset(B3, 0, sizeof B3);
    memset(VdB3, 0, sizeof VdB3);
}

void nn_forward_batch(const float *x,
                      float *h1, float *h2,
                      float *y, uint32_t bs)
{
    for (uint32_t b = 0; b < bs; b++)
    {
        const float *in = x + b * NN_INPUT;
        float *a1 = h1 + b * NN_HIDDEN1;
        float *a2 = h2 + b * NN_HIDDEN2;
        float *out = y + b * NN_OUTPUT;
        layer_forward(in, NN_INPUT, B1, &W1[0][0], NN_HIDDEN1, relu, a1);
        layer_forward(a1, NN_HIDDEN1, B2, &W2[0][0], NN_HIDDEN2, relu, a2);
        float z[NN_OUTPUT];
        layer_forward(a2, NN_HIDDEN2, B3, &W3[0][0], NN_OUTPUT, NULL, z);
        softmax(z, NN_OUTPUT, out);
    }
}

void nn_backward_batch(const float *x,
                       const float *h1, const float *h2,
                       const float *y, const uint8_t *lbls,
                       uint32_t bs)
{
    static float dW1[NN_HIDDEN1][NN_INPUT], dB1_[NN_HIDDEN1];
    static float dW2[NN_HIDDEN2][NN_HIDDEN1], dB2_[NN_HIDDEN2];
    static float dW3[NN_OUTPUT][NN_HIDDEN2], dB3_[NN_OUTPUT];
    memset(dW1, 0, sizeof dW1);
    memset(dB1_, 0, sizeof dB1_);
    memset(dW2, 0, sizeof dW2);
    memset(dB2_, 0, sizeof dB2_);
    memset(dW3, 0, sizeof dW3);
    memset(dB3_, 0, sizeof dB3_);
    for (uint32_t b = 0; b < bs; b++)
    {
        const float *in = x + b * NN_INPUT;
        const float *a1 = h1 + b * NN_HIDDEN1;
        const float *a2 = h2 + b * NN_HIDDEN2;
        const float *out = y + b * NN_OUTPUT;
        int lbl = lbls[b];
        // delta3 = out - one_hot(lbl)
        float delta3[NN_OUTPUT];
        for (int k = 0; k < NN_OUTPUT; k++)
        {
            delta3[k] = out[k] - (k == lbl);
            dB3_[k] += delta3[k];
            for (int i = 0; i < NN_HIDDEN2; i++)
                dW3[k][i] += delta3[k] * a2[i];
        }
        // delta2
        float delta2[NN_HIDDEN2];
        layer_back_relu(delta3, &W3[0][0], NN_OUTPUT, NN_HIDDEN2, a2, delta2);
        for (int i = 0; i < NN_HIDDEN2; i++)
        {
            dB2_[i] += delta2[i];
            for (int j = 0; j < NN_HIDDEN1; j++)
                dW2[i][j] += delta2[i] * a1[j];
        }
        // delta1
        float delta1[NN_HIDDEN1];
        layer_back_relu(delta2, &W2[0][0], NN_HIDDEN2, NN_HIDDEN1, a1, delta1);
        for (int i = 0; i < NN_HIDDEN1; i++)
        {
            dB1_[i] += delta1[i];
            for (int j = 0; j < NN_INPUT; j++)
                dW1[i][j] += delta1[i] * in[j];
        }
    }
    // update layers
    Layer L1 = {(float *)&W1[0][0], B1, (float *)&dW1[0][0], dB1_, (float *)&VdW1[0][0], VdB1, NN_INPUT, NN_HIDDEN1};
    Layer L2 = {(float *)&W2[0][0], B2, (float *)&dW2[0][0], dB2_, (float *)&VdW2[0][0], VdB2, NN_HIDDEN1, NN_HIDDEN2};
    Layer L3 = {(float *)&W3[0][0], B3, (float *)&dW3[0][0], dB3_, (float *)&VdW3[0][0], VdB3, NN_HIDDEN2, NN_OUTPUT};
    update_layer(L1);
    update_layer(L2);
    update_layer(L3);
}

// evaluate for N samples

double nn_evaluate(const float *x, const uint8_t *lbls, uint32_t N)
{
    uint32_t correct = 0;
    for (uint32_t i = 0; i < N; i++)
    {
        const float *in = x + i * NN_INPUT;
        float h1[NN_HIDDEN1], h2[NN_HIDDEN2], z[NN_OUTPUT];
        layer_forward(in, NN_INPUT, B1, &W1[0][0], NN_HIDDEN1, relu, h1);
        layer_forward(h1, NN_HIDDEN1, B2, &W2[0][0], NN_HIDDEN2, relu, h2);
        layer_forward(h2, NN_HIDDEN2, B3, &W3[0][0], NN_OUTPUT, NULL, z);
        // softmax & argmax
        float maxz = z[0], sum = 0, p[NN_OUTPUT];
        int pred = 0;
#pragma omp simd
        for (int k = 0; k < NN_OUTPUT; k++)
        {
            p[k] = expf(z[k] - maxz);
            sum += p[k];
        }
#pragma omp simd
        for (int k = 0; k < NN_OUTPUT; k++)
        {
            p[k] /= sum;
            if (p[k] > p[pred])
                pred = k;
        }
        if (pred == lbls[i])
            correct++;
    }
    return (double)correct / N;
}

int nn_predict(const float *x)
{
    float h1[NN_HIDDEN1], h2[NN_HIDDEN2], z[NN_OUTPUT];
    layer_forward(x, NN_INPUT, B1, &W1[0][0], NN_HIDDEN1, relu, h1);
    layer_forward(h1, NN_HIDDEN1, B2, &W2[0][0], NN_HIDDEN2, relu, h2);
    layer_forward(h2, NN_HIDDEN2, B3, &W3[0][0], NN_OUTPUT, NULL, z);
    float maxz = z[0], pbest = 0;
    int pred = 0;
    for (int k = 0; k < NN_OUTPUT; k++)
    {
        float p = expf(z[k] - maxz);
        if (p > pbest)
        {
            pbest = p;
            pred = k;
        }
    }
    return pred;
}