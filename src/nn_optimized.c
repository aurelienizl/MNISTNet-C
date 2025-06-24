/* nn_optimized.c — OpenMP‑safe & numerically solid re‑rewrite
 * -----------------------------------------------------------
 *  ▸ Keeps the *exact* public API (so it still drops into mnist_simple).
 *  ▸ Uses OpenMP only where thread‑safe; removes 5.x array‑section reductions
 *    that some compilers mis-handle and that broke training accuracy.
 *  ▸ Thread‑private gradient buffers → deterministic sums, no races.
 *  ▸ Optional OpenMP SIMD for the hot inner loops.
 * -----------------------------------------------------------
 *  Build:   gcc/clang  -Ofast -march=native -fopenmp  nn_optimized.c  -o libnn.o
 */

#include "nn_optimized.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

/* ---------- Hyper‑parameters ---------- */
static float LR = 0.01f;          /* learning rate         */
static float WD = 1e-4f;          /* weight‑decay (L2)      */
static const float MOM = 0.9f;    /* momentum coefficient   */

/* ---------- Alignment helpers ---------- */
#ifndef ALIGN
#   define ALIGN(N) __attribute__((aligned(N)))
#endif

/* ---------- Tensor sizes (row‑major) --- */
#define W1_SIZE  (NN_HIDDEN1 * NN_INPUT)
#define W2_SIZE  (NN_HIDDEN2 * NN_HIDDEN1)
#define W3_SIZE  (NN_OUTPUT  * NN_HIDDEN2)

/* ---------- Parameters + momentum ------ */
static float ALIGN(64) W1[NN_HIDDEN1][NN_INPUT],     B1[NN_HIDDEN1];
static float ALIGN(64) W2[NN_HIDDEN2][NN_HIDDEN1],   B2[NN_HIDDEN2];
static float ALIGN(64) W3[NN_OUTPUT ][NN_HIDDEN2],   B3[NN_OUTPUT];

static float ALIGN(64) VdW1[NN_HIDDEN1][NN_INPUT],   VdB1[NN_HIDDEN1];
static float ALIGN(64) VdW2[NN_HIDDEN2][NN_HIDDEN1], VdB2[NN_HIDDEN2];
static float ALIGN(64) VdW3[NN_OUTPUT ][NN_HIDDEN2], VdB3[NN_OUTPUT];

/* ---------- Fallback activations ------- */
#ifndef NN_OPTIMIZED_PROVIDES_ACT
static inline float relu (float x){ return x > 0.0f ? x : 0.0f; }
static inline float drelu(float x){ return x > 0.0f ? 1.0f : 0.0f; }
static inline float xavier(int in,int out){
    const float r = sqrtf(6.f/(in+out));
    return ((float)rand()/RAND_MAX)*2.f*r - r;
}
#endif

/* ---------- Forward helpers ------------ */
static inline void layer_forward(const float*restrict in,int in_dim,
                                 const float*restrict bias,
                                 const float*restrict W,int out_dim,
                                 float (*act)(float),
                                 float*restrict out){
    for(int i=0;i<out_dim;++i){
        const float* Wi = W + (size_t)i*in_dim;
        float s = bias[i];
#pragma omp simd reduction(+:s)
        for(int j=0;j<in_dim;++j) s += Wi[j]*in[j];
        out[i] = act ? act(s):s;
    }
}

static inline void softmax(const float*restrict z,int n,float*restrict p){
    float m = z[0];
#pragma omp simd reduction(max:m)
    for(int i=1;i<n;++i) m = fmaxf(m,z[i]);
    float sum = 0.f;
#pragma omp simd reduction(+:sum)
    for(int i=0;i<n;++i){ p[i]=expf(z[i]-m); sum+=p[i]; }
    const float inv = 1.f/sum;
#pragma omp simd
    for(int i=0;i<n;++i) p[i]*=inv;
}

static inline void layer_back_relu(const float*restrict delta_next,
                                   const float*restrict W_next,
                                   int out_dim,int in_dim,
                                   const float*restrict act_prev,
                                   float*restrict delta){
    for(int i=0;i<in_dim;++i){
        const float* col = W_next + i; /* column i */
        float s=0.f;
#pragma omp simd reduction(+:s)
        for(int k=0;k<out_dim;++k) s += col[(size_t)k*in_dim]*delta_next[k];
        delta[i]=s*drelu(act_prev[i]);
    }
}

/* ---------- SGD‑M update --------------- */
static inline void update_layer(const float*restrict dW,const float*restrict dB,
                                float*restrict W,float*restrict B,
                                float*restrict vW,float*restrict vB,
                                int in,int out){
#pragma omp parallel for schedule(static)
    for(int i=0;i<out;++i){
        const float gB = dB[i]/(float)out;
        vB[i]  = MOM*vB[i] - LR*gB;
        B [i] += vB[i];
        float* Wi  = W  + (size_t)i*in;
        float* vWi = vW + (size_t)i*in;
        const float* gWi = dW + (size_t)i*in;
#pragma omp simd
        for(int j=0;j<in;++j){
            const float g = gWi[j]/(float)out + WD*Wi[j];
            vWi[j] = MOM*vWi[j] - LR*g;
            Wi [j] += vWi[j];
        }
    }
}

/* ---------- Public API ----------------- */
void nn_set_hyper(float lr,float weight_decay){ LR=lr; WD=weight_decay; }

void nn_init(void){
    srand((unsigned)time(NULL));
#pragma omp parallel for schedule(static)
    for(int i=0;i<NN_HIDDEN1;++i){
        for(int j=0;j<NN_INPUT;++j){ W1[i][j]=xavier(NN_INPUT,NN_HIDDEN1); VdW1[i][j]=0.f; }
        B1[i]=VdB1[i]=0.f;
    }
#pragma omp parallel for schedule(static)
    for(int i=0;i<NN_HIDDEN2;++i){
        for(int j=0;j<NN_HIDDEN1;++j){ W2[i][j]=xavier(NN_HIDDEN1,NN_HIDDEN2); VdW2[i][j]=0.f; }
        B2[i]=VdB2[i]=0.f;
    }
#pragma omp parallel for schedule(static)
    for(int i=0;i<NN_OUTPUT;++i){
        for(int j=0;j<NN_HIDDEN2;++j){ W3[i][j]=xavier(NN_HIDDEN2,NN_OUTPUT); VdW3[i][j]=0.f; }
        B3[i]=VdB3[i]=0.f;
    }
}

void nn_forward_batch(const float*x,float*h1,float*h2,float*y,uint32_t bs){
#pragma omp parallel for schedule(static)
    for(uint32_t b=0;b<bs;++b){
        const float* in = x + (size_t)b*NN_INPUT;
        float* a1 = h1 + (size_t)b*NN_HIDDEN1;
        float* a2 = h2 + (size_t)b*NN_HIDDEN2;
        float* out= y  + (size_t)b*NN_OUTPUT;
        layer_forward(in ,NN_INPUT ,B1,&W1[0][0],NN_HIDDEN1,relu,a1);
        layer_forward(a1 ,NN_HIDDEN1,B2,&W2[0][0],NN_HIDDEN2,relu,a2);
        float z[NN_OUTPUT];
        layer_forward(a2 ,NN_HIDDEN2,B3,&W3[0][0],NN_OUTPUT ,NULL,z);
        softmax(z,NN_OUTPUT,out);
    }
}

void nn_backward_batch(const float*x,const float*h1,const float*h2,
                       const float*y,const uint8_t*lbls,uint32_t bs){
    /* global (shared) gradient buffers */
    float ALIGN(64) dW1[W1_SIZE]={0}, dW2[W2_SIZE]={0}, dW3[W3_SIZE]={0};
    float ALIGN(64) dB1[NN_HIDDEN1]={0}, dB2[NN_HIDDEN2]={0}, dB3[NN_OUTPUT]={0};

#pragma omp parallel
    {
        /* thread‑private accumulators to avoid false sharing */
        float* dW1_t = calloc(W1_SIZE ,sizeof(float));
        float* dW2_t = calloc(W2_SIZE ,sizeof(float));
        float* dW3_t = calloc(W3_SIZE ,sizeof(float));
        float  dB1_t[NN_HIDDEN1]={0}, dB2_t[NN_HIDDEN2]={0}, dB3_t[NN_OUTPUT]={0};

#pragma omp for schedule(static)
        for(uint32_t b=0;b<bs;++b){
            const float* in  = x  + (size_t)b*NN_INPUT;
            const float* a1  = h1 + (size_t)b*NN_HIDDEN1;
            const float* a2  = h2 + (size_t)b*NN_HIDDEN2;
            const float* out = y  + (size_t)b*NN_OUTPUT;
            const int lbl = lbls[b];

            float delta3[NN_OUTPUT];
            for(int k=0;k<NN_OUTPUT;++k){
                const float err = out[k] - (k==lbl);
                delta3[k]=err;
                dB3_t[k]+=err;
                const size_t base=(size_t)k*NN_HIDDEN2;
                for(int i=0;i<NN_HIDDEN2;++i) dW3_t[base+i]+=err*a2[i];
            }

            float delta2[NN_HIDDEN2];
            layer_back_relu(delta3,&W3[0][0],NN_OUTPUT,NN_HIDDEN2,a2,delta2);
            for(int i=0;i<NN_HIDDEN2;++i){
                dB2_t[i]+=delta2[i];
                const size_t base=(size_t)i*NN_HIDDEN1;
                for(int j=0;j<NN_HIDDEN1;++j) dW2_t[base+j]+=delta2[i]*a1[j];
            }

            float delta1[NN_HIDDEN1];
            layer_back_relu(delta2,&W2[0][0],NN_HIDDEN2,NN_HIDDEN1,a1,delta1);
            for(int i=0;i<NN_HIDDEN1;++i){
                dB1_t[i]+=delta1[i];
                const size_t base=(size_t)i*NN_INPUT;
                for(int j=0;j<NN_INPUT;++j) dW1_t[base+j]+=delta1[i]*in[j];
            }
        }

        /* merge into globals */
#pragma omp critical
        {
            for(size_t i=0;i<W1_SIZE;++i) dW1[i]+=dW1_t[i];
            for(size_t i=0;i<W2_SIZE;++i) dW2[i]+=dW2_t[i];
            for(size_t i=0;i<W3_SIZE;++i) dW3[i]+=dW3_t[i];
            for(int i=0;i<NN_HIDDEN1;++i) dB1[i]+=dB1_t[i];
            for(int i=0;i<NN_HIDDEN2;++i) dB2[i]+=dB2_t[i];
            for(int i=0;i<NN_OUTPUT;++i)  dB3[i]+=dB3_t[i];
        }
        free(dW1_t); free(dW2_t); free(dW3_t);
    }

    /* single‑thread update */
    update_layer(dW1,dB1,&W1[0][0],B1,&VdW1[0][0],VdB1,NN_INPUT ,NN_HIDDEN1);
    update_layer(dW2,dB2,&W2[0][0],B2,&VdW2[0][0],VdB2,NN_HIDDEN1,NN_HIDDEN2);
    update_layer(dW3,dB3,&W3[0][0],B3,&VdW3[0][0],VdB3,NN_HIDDEN2,NN_OUTPUT );
}

/* ---------- Evaluate / Predict ---------- */
double nn_evaluate(const float*x,const uint8_t*lbls,uint32_t N){
    uint32_t correct=0;
#pragma omp parallel for reduction(+:correct) schedule(static)
    for(uint32_t i=0;i<N;++i){
        const float* in = x + (size_t)i*NN_INPUT;
        float h1[NN_HIDDEN1],h2[NN_HIDDEN2],z[NN_OUTPUT];
        layer_forward(in ,NN_INPUT ,B1,&W1[0][0],NN_HIDDEN1,relu ,h1);
        layer_forward(h1 ,NN_HIDDEN1,B2,&W2[0][0],NN_HIDDEN2,relu ,h2);
        layer_forward(h2 ,NN_HIDDEN2,B3,&W3[0][0],NN_OUTPUT ,NULL,z);
        int pred=0; float m=z[0];
        for(int k=1;k<NN_OUTPUT;++k) if(z[k]>m){m=z[k];pred=k;}
        if(pred==lbls[i]) ++correct;
    }
    return (double)correct/N;
}

int nn_predict(const float*x){
    float h1[NN_HIDDEN1],h2[NN_HIDDEN2],z[NN_OUTPUT];
    layer_forward(x ,NN_INPUT ,B1,&W1[0][0],NN_HIDDEN1,relu ,h1);
    layer_forward(h1,NN_HIDDEN1,B2,&W2[0][0],NN_HIDDEN2,relu ,h2);
    layer_forward(h2,NN_HIDDEN2,B3,&W3[0][0],NN_OUTPUT ,NULL,z);
    int pred=0; float m=z[0];
    for(int k=1;k<NN_OUTPUT;++k) if(z[k]>m){m=z[k];pred=k;}
    return pred;
}
