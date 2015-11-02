#ifndef _PSO_CUH
#define _PSO_CUH
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

class CudaPSO{
private:
    int n;
    float2 *par_dPos;
    float2 *par_dVel;
    float3 *par_dFit; // Best x, Best y, fit, Best fit
    float  *best_Fits;
    cublasHandle_t handle;
    curandState *curand_state;
    void Init();

public:  
    float3 gBest;    //x, y, fit
    float *RME;
    int iters;
    CudaPSO(int n);
    ~CudaPSO();
    float Solve(int m, float eps);
};

#endif
