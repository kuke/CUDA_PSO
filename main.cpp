#include <iostream>
#include "pso.hpp"
#include "cuda_pso.cuh"

int main()
{
   PSO pso(1024);
   float time = pso.Solve(50, 10e-6);
   std::cout<<"CPU result: "<<std::endl;
   std::cout<<"x: "<<pso.gBest.x<<" y: "<<pso.gBest.y<<" iters: "<<pso.iters<<" time: "<<time<<"ms"<<std::endl;
   
   CudaPSO cuda_pso(1024);
   double start = cpu_time();
   float cuda_time = cuda_pso.Solve(50, 10^-6);
   cudaDeviceSynchronize();
   double end = cpu_time();
   std::cout<<"GPU result: "<<std::endl;
   std::cout<<" x: "<<cuda_pso.gBest.x<<" y: "<<cuda_pso.gBest.y<<" iters: "<<cuda_pso.iters<<" time: "<<cuda_time<<"ms"<<" cpu time "<<end-start<<std::endl;
}
