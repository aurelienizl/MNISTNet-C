// File: nn_imlp.c
#include "nn_optimized.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Parameters
static float W1[NN_HIDDEN1][NN_INPUT], B1[NN_HIDDEN1];
static float W2[NN_HIDDEN2][NN_HIDDEN1], B2[NN_HIDDEN2];    
static float W3[NN_OUTPUT][NN_HIDDEN2], B3[NN_OUTPUT];

// Momentum terms
static float VdW1[NN_HIDDEN1][NN_INPUT], VdB1[NN_HIDDEN1];
static float VdW2[NN_HIDDEN2][NN_HIDDEN1], VdB2[NN_HIDDEN2];
static float VdW3[NN_OUTPUT][NN_HIDDEN2], VdB3[NN_OUTPUT];

// Hyperparams
static float LR = 0.01f;
static float WD = 1e-4f;    // L2 weight decay
static const float MOM = 0.9f;

// helpers
static inline float relu(float x) { return x > 0 ? x : 0; }
static inline float drelu(float x) { return x > 0 ? 1 : 0; }

// uniform random in [-a,a]
static float urand(float a) {
    return ((rand()/(float)RAND_MAX)*2 - 1)*a;
}

void nn_set_hyper(float lr, float weight_decay) {
    LR = lr;
    WD = weight_decay;
}

void nn_init(void) {
    srand((unsigned)time(NULL));
    float s1 = sqrtf(6.0f/(NN_INPUT+NN_HIDDEN1));
    float s2 = sqrtf(6.0f/(NN_HIDDEN1+NN_HIDDEN2));
    float s3 = sqrtf(6.0f/(NN_HIDDEN2+NN_OUTPUT));
    // init W/B and zero V*
    for(int i=0;i<NN_HIDDEN1;i++){
        B1[i]=0; VdB1[i]=0;
        for(int j=0;j<NN_INPUT;j++){
            W1[i][j] = urand(s1);
            VdW1[i][j] = 0;
        }
    }
    for(int i=0;i<NN_HIDDEN2;i++){
        B2[i]=0; VdB2[i]=0;
        for(int j=0;j<NN_HIDDEN1;j++){
            W2[i][j] = urand(s2);
            VdW2[i][j] = 0;
        }
    }
    for(int k=0;k<NN_OUTPUT;k++){
        B3[k]=0; VdB3[k]=0;
        for(int i=0;i<NN_HIDDEN2;i++){
            W3[k][i] = urand(s3);
            VdW3[k][i] = 0;
        }
    }
}

void nn_forward_batch(const float *x,
                      float *h1, float *h2,
                      float *y, uint32_t bs) {
    #pragma omp parallel for
    for(uint32_t b=0;b<bs;b++){
        const float *xb = x + b*NN_INPUT;
        float *hh1 = h1 + b*NN_HIDDEN1;
        float *hh2 = h2 + b*NN_HIDDEN2;
        float *yy  = y  + b*NN_OUTPUT;
        // Layer 1
        for(int i=0;i<NN_HIDDEN1;i++){
            float s = B1[i];
            #pragma omp simd
            for(int j=0;j<NN_INPUT;j++) s += W1[i][j]*xb[j];
            hh1[i] = relu(s);
        }
        // Layer 2
        for(int i=0;i<NN_HIDDEN2;i++){
            float s = B2[i];
            #pragma omp simd
            for(int j=0;j<NN_HIDDEN1;j++) s += W2[i][j]*hh1[j];
            hh2[i] = relu(s);
        }
        // Output + softmax
        float maxz = -INFINITY;
        float z[NN_OUTPUT];
        for(int k=0;k<NN_OUTPUT;k++){
            float s = B3[k];
            #pragma omp simd
            for(int i=0;i<NN_HIDDEN2;i++) s += W3[k][i]*hh2[i];
            z[k] = s;
            if(s>maxz) maxz=s;
        }
        float sum=0;
        for(int k=0;k<NN_OUTPUT;k++){
            yy[k] = expf(z[k]-maxz);
            sum += yy[k];
        }
        for(int k=0;k<NN_OUTPUT;k++) yy[k] /= sum;
    }
}

void nn_backward_batch(const float *x,
                       const float *h1, const float *h2,
                       const float *y, const uint8_t *lbls,
                       uint32_t bs) {
    // accumulate grads
    static float dW1[NN_HIDDEN1][NN_INPUT], dB1[NN_HIDDEN1];
    static float dW2[NN_HIDDEN2][NN_HIDDEN1], dB2[NN_HIDDEN2];
    static float dW3[NN_OUTPUT][NN_HIDDEN2], dB3[NN_OUTPUT];
    // zero
    for(int i=0;i<NN_HIDDEN1;i++) { dB1[i]=0; memset(dW1[i],0,NN_INPUT*sizeof(float)); }
    for(int i=0;i<NN_HIDDEN2;i++) { dB2[i]=0; memset(dW2[i],0,NN_HIDDEN1*sizeof(float)); }
    for(int k=0;k<NN_OUTPUT;k++) { dB3[k]=0; memset(dW3[k],0,NN_HIDDEN2*sizeof(float)); }

    for(uint32_t b=0;b<bs;b++){
        const float *xb = x + b*NN_INPUT;
        const float *hh1 = h1 + b*NN_HIDDEN1;
        const float *hh2 = h2 + b*NN_HIDDEN2;
        const float *yy  = y  + b*NN_OUTPUT;
        int lbl = lbls[b];
        // output delta
        float delta3[NN_OUTPUT];
        for(int k=0;k<NN_OUTPUT;k++){
            delta3[k] = yy[k] - (k==lbl);
            dB3[k] += delta3[k];
            for(int i=0;i<NN_HIDDEN2;i++)
                dW3[k][i] += delta3[k]*hh2[i];
        }
        // layer2 delta
        float delta2[NN_HIDDEN2];
        for(int i=0;i<NN_HIDDEN2;i++){
            float s=0;
            for(int k=0;k<NN_OUTPUT;k++) s += W3[k][i]*delta3[k];
            delta2[i] = s * drelu(hh2[i]);
            dB2[i] += delta2[i];
            for(int j=0;j<NN_HIDDEN1;j++)
                dW2[i][j] += delta2[i]*hh1[j];
        }
        // layer1 delta
        for(int i=0;i<NN_HIDDEN1;i++){
            float s=0;
            for(int j=0;j<NN_HIDDEN2;j++) s += W2[j][i]*delta2[j];
            float d = s * drelu(hh1[i]);
            dB1[i] += d;
            for(int j=0;j<NN_INPUT;j++)
                dW1[i][j] += d*xb[j];
        }
    }

    // update with momentum + weight decay
    for(int i=0;i<NN_HIDDEN1;i++){
        VdB1[i] = MOM*VdB1[i] - (LR/bs)*dB1[i];
        B1[i] += VdB1[i];
        for(int j=0;j<NN_INPUT;j++){
            float grad = (dW1[i][j]/bs) + WD*W1[i][j];
            VdW1[i][j] = MOM*VdW1[i][j] - LR*grad;
            W1[i][j] += VdW1[i][j];
        }
    }
    for(int i=0;i<NN_HIDDEN2;i++){
        VdB2[i] = MOM*VdB2[i] - (LR/bs)*dB2[i];
        B2[i] += VdB2[i];
        for(int j=0;j<NN_HIDDEN1;j++){
            float grad = (dW2[i][j]/bs) + WD*W2[i][j];
            VdW2[i][j] = MOM*VdW2[i][j] - LR*grad;
            W2[i][j] += VdW2[i][j];
        }
    }
    for(int k=0;k<NN_OUTPUT;k++){
        VdB3[k] = MOM*VdB3[k] - (LR/bs)*dB3[k];
        B3[k] += VdB3[k];
        for(int i=0;i<NN_HIDDEN2;i++){
            float grad = (dW3[k][i]/bs) + WD*W3[k][i];
            VdW3[k][i] = MOM*VdW3[k][i] - LR*grad;
            W3[k][i] += VdW3[k][i];
        }
    }
}

double nn_evaluate(const float *x,
                   const uint8_t *lbls,
                   uint32_t N) {
    uint32_t correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for(uint32_t i=0;i<N;i++){
        // forward single
        const float *xb = x + i*NN_INPUT;
        float h1[NN_HIDDEN1], h2[NN_HIDDEN2], z[NN_OUTPUT];
        for(int ii=0;ii<NN_HIDDEN1;ii++){
            float s = B1[ii];
            for(int jj=0;jj<NN_INPUT;jj++) s += W1[ii][jj]*xb[jj];
            h1[ii] = relu(s);
        }
        for(int ii=0;ii<NN_HIDDEN2;ii++){
            float s = B2[ii];
            for(int jj=0;jj<NN_HIDDEN1;jj++) s += W2[ii][jj]*h1[jj];
            h2[ii] = relu(s);
        }
        float maxz = -INFINITY;
        for(int k=0;k<NN_OUTPUT;k++){
            float s = B3[k];
            for(int j=0;j<NN_HIDDEN2;j++) s += W3[k][j]*h2[j];
            z[k]=s;
            if(s>maxz) maxz=s;
        }
        int pred=0;
        float bestp=-INFINITY, sum=0;
        for(int k=0;k<NN_OUTPUT;k++){
            float p = expf(z[k]-maxz);
            sum += p;
            if(p>bestp){ bestp=p; pred=k; }
        }
        if(pred==lbls[i]) correct++;
    }
    return correct/(double)N;
}

int nn_save(const char *path){
    FILE *f = fopen(path,"wb");
    if(!f){ perror("fopen"); return -1; }
    fwrite(W1, sizeof(W1),1,f);
    fwrite(B1, sizeof(B1),1,f);
    fwrite(W2, sizeof(W2),1,f);
    fwrite(B2, sizeof(B2),1,f);
    fwrite(W3, sizeof(W3),1,f);
    fwrite(B3, sizeof(B3),1,f);
    fclose(f);
    return 0;
}

int nn_load(const char *path){
    FILE *f = fopen(path,"rb");
    if(!f){ perror("fopen"); return -1; }
    if(fread(W1, sizeof(W1),1,f)!=1) return -1;
    if(fread(B1, sizeof(B1),1,f)!=1) return -1;
    if(fread(W2, sizeof(W2),1,f)!=1) return -1;
    if(fread(B2, sizeof(B2),1,f)!=1) return -1;
    if(fread(W3, sizeof(W3),1,f)!=1) return -1;
    if(fread(B3, sizeof(B3),1,f)!=1) return -1;
    fclose(f);
    return 0;
}

int nn_predict(const float *x){
    float h1[NN_HIDDEN1], h2[NN_HIDDEN2], z[NN_OUTPUT];
    // forward single sample (no dropout)
    for(int i=0;i<NN_HIDDEN1;i++){
        float s = B1[i];
        for(int j=0;j<NN_INPUT;j++) s += W1[i][j]*x[j];
        h1[i] = relu(s);
    }
    for(int i=0;i<NN_HIDDEN2;i++){
        float s = B2[i];
        for(int j=0;j<NN_HIDDEN1;j++) s += W2[i][j]*h1[j];
        h2[i] = relu(s);
    }
    float maxz=-INFINITY;
    for(int k=0;k<NN_OUTPUT;k++){
        float s = B3[k];
        for(int i=0;i<NN_HIDDEN2;i++) s += W3[k][i]*h2[i];
        z[k] = s; if(s>maxz) maxz=s;
    }
    int pred = 0;
    float best = -INFINITY;
    for(int k=0;k<NN_OUTPUT;k++){
        float p = expf(z[k]-maxz);
        if(p > best){ best = p; pred = k; }
    }
    return pred;
}