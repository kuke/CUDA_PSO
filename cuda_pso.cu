#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_pso.cuh"
#define PI 3.14159265898

using namespace std;

__constant__ double  data[9] = {
   0.004450342097284,   0.017440060625698,   0.025142479360125,
   0.018345616768773,   0.071893050392351,   0.103644681886180,
   0.027821257493749,   0.109026319048659,   0.157177892624538
    };

__device__ float compute_fit(float x, float y)
{
    float ret;
    if(x<-1 || x>1 || y<-1 || y>1) {
        ret = 9.0;
    } else {
        float square_sum = 0;
        for (int j=-1; j<=1; j++) {
            for (int i=-1; i<=1; i++) {
                int index = 3*(j+1)+(i+1);
                float diff = 1/(2*PI)*exp(-(pow(i-x,2)+pow(j-y, 2))/2);
                square_sum += pow(diff-data[index], 2);
            }
        }
        ret = sqrt(square_sum/9);
    }
    return ret;
}

__global__ void curand_setup(curandState *state, int n){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index < n)
        curand_init(1234, index, 0, &state[index]);
}


__global__ static void Init_kernel(float2 *par_dPos, float2 *par_dVel, float3 *par_dFit,  curandState *state,  int n)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < n) {
      curandState localState = state[index];
      float2 Pos = make_float2(-1+2*curand_uniform(&localState), -1+2*curand_uniform(&localState));
      float2 Vel = make_float2(-0.01+0.02*curand_uniform(&localState), -0.01+0.02*curand_uniform(&localState));
      state[index] = localState;
      float fit = compute_fit(Pos.x, Pos.y);
      float3 Fit = make_float3(Pos.x, Pos.y, fit);
      par_dPos[index] = Pos;
      par_dVel[index] = Vel;
      par_dFit[index] = Fit;
  }
    
}

__global__ static void Solve_kernel(float2 *par_dPos, float2 *par_dVel, float3 *par_dFit,  float *best_Fits, int best_index, curandState *state, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float3 gBest[1];
    if (index < n) {
        float2 Pos = par_dPos[index];
        float2 Vel = par_dVel[index];
        float3 Fit = par_dFit[index];
        if (threadIdx.x == 0) {
            gBest[0] = par_dFit[best_index];
        }
        __syncthreads();
        float3 best_fit = gBest[0];
        curandState localState = state[index];
        float fit = compute_fit(Pos.x, Pos.y);
        if (fit < Fit.z) {
            Fit.z = fit;
            Fit.x = Pos.x;
            Fit.y = Pos.y;
        }
        if (Fit.z < best_fit.z) {
            best_fit = Fit;
        }
        int c1 = 1;
        int c2 = 1;
        Vel.x += c1*curand_uniform(&localState)*(best_fit.x - Pos.x)+c2*curand_uniform(&localState)*(Fit.x - Pos.x);
        Vel.y += c1*curand_uniform(&localState)*(best_fit.y - Pos.y)+c2*curand_uniform(&localState)*(Fit.y - Pos.y);
        Pos.x += Vel.x;
        Pos.y += Vel.y;
        best_Fits[index] = Fit.z;
        par_dFit[index] = Fit;
        state[index] = localState;
        par_dPos[index] = Pos;
        par_dVel[index] = Vel;
    }
}

CudaPSO::CudaPSO(int n){
    this->n = n;
    cublasCreate(&handle);
    cudaMalloc((void **)&par_dPos, sizeof(float2)*n);
    cudaMalloc((void **)&par_dVel, sizeof(float2)*n);
    cudaMalloc((void **)&par_dFit, sizeof(float3)*n);
    cudaMalloc((void **)&best_Fits, sizeof(float)*n);
    cudaMalloc((void**)&curand_state, sizeof(curandState)*n);
    Init();    
}

void CudaPSO::Init() {
    int nThreadsPerBlock = 32;
    int nBlocks = (n+nThreadsPerBlock-1)/nThreadsPerBlock;
    curand_setup<<<nBlocks, nThreadsPerBlock>>>(curand_state, n);
    Init_kernel<<<nBlocks, nThreadsPerBlock>>>(par_dPos, par_dVel, par_dFit,curand_state, n);
}

float CudaPSO::Solve(int m, float eps){
    int nThreadsPerBlock = 32;
    int nBlocks = (n+nThreadsPerBlock-1)/nThreadsPerBlock;
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    float3 gBest;
    int best_index = 0;
    for (int k=0; k<m; k++) {
        Solve_kernel<<<nBlocks, nThreadsPerBlock>>>(par_dPos, par_dVel, par_dFit, best_Fits,  best_index, curand_state, n);
        cudaDeviceSynchronize();
        cublasIsamin(handle, n, best_Fits, 1, &best_index); 
//        cublasGetVector(1, sizeof(float), best_Fits, best_index, &best_fit, 0);
        cudaMemcpy(&gBest, par_dFit+best_index, 1*sizeof(float3), cudaMemcpyDeviceToHost);
        cout<<best_index<<"\t"<<gBest.z<<"\t";
        if (gBest.z < eps) break;   
    }
    cout<<endl;
    cout<<gBest.x<<"\t"<<gBest.y<<endl;
   
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return time;    
}
