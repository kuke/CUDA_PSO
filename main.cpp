#include <iostream>
#include "pso.hpp"
#include "cuda_pso.cuh"

int main()
{
   int numParticles = 1024;
   int maxIters = 50;
   PSO pso(numParticles);
   float time = pso.Solve(maxIters, 10e-6);
   std::cout<<"CPU result: "<<std::endl;
   std::cout<<"a: "<<pso.gBest.x<<" b: "<<pso.gBest.y<<" iters: "<<pso.iters<<" time: "<<time<<"ms"<<std::endl;
   
   CudaPSO cuda_pso(numParticles);
   float cuda_time = cuda_pso.Solve(maxIters, 10^-6);
   std::cout<<"GPU result: "<<std::endl;
   std::cout<<" a: "<<cuda_pso.gBest.x<<" b: "<<cuda_pso.gBest.y<<" iters: "<<cuda_pso.iters<<" time: "<<cuda_time<<"ms"<<std::endl;
   
   std::cout<<"GPU perf./CPU perf. = "<<time/cuda_time<<std::endl;
   return 0;
}
