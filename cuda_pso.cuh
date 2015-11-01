#ifndef _PSO_CUH
#define _PSO_CUH
#include <cuda_runtime.h>
#include <curand_kernel.h>

class CudaPSO{
private:
    int n;
    float2 *par_dPos;
    float2 *par_dVel;
    float4 *par_dFit; // Best x, Best y, fit, Best fit
    float  *best_Fit;
    curandState *curand_state;
    void Init();
public:  
    float3 gBest;    //x, y, fit
    CudaPSO(int n);
    float Solve(int m, float eps);
};

#endif
