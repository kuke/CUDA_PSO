#include <iostream>
#include "pso.hpp"
#include "cuda_pso.cuh"

int main()
{
   PSO pso(1024);
   float time = pso.Solve(50, 10e-6);
   std::cout<<"CPU result: "<<std::endl;
   std::cout<<"x: "<<pso.gBest.x<<"y: "<<pso.gBest.y<<"iters: "<<pso.iters<<"time: "<<time<<std::endl;
   
   CudaPSO cuda_pso(1024);
   float cuda_time = cuda_pso.Solve(150, 10^-6);
   std::cout<<"GPU result: "<<std::endl;
   std::cout<<"x: "<<cuda_pso.gBest.x<<"y: "<<cuda_pso.gBest.y<<"iters: "<<cuda_pso.iters<<"time: "<<cuda_time<<std::endl;
}
