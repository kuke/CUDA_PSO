#include <iostream>
#include <fstream>
#include "pso.hpp"
#include "cuda_pso.cuh"
#include <string>

int main()
{
   int numParticles = 1024;
   int maxIters = 50;
   float eps = 10^-6;
   PSO pso(numParticles);
   float cpu_time = pso.Solve(maxIters, eps);
   std::cout<<"CPU result: "<<std::endl;
   std::cout<<"a: "<<pso.gBest.x<<" b: "<<pso.gBest.y<<" iters: "<<pso.iters<<" time: "<<cpu_time<<"ms"<<std::endl;
   
   CudaPSO cuda_pso(numParticles);
   float cuda_time = cuda_pso.Solve(maxIters, eps);
   std::cout<<"GPU result: "<<std::endl;
   std::cout<<" a: "<<cuda_pso.gBest.x<<" b: "<<cuda_pso.gBest.y<<" iters: "<<cuda_pso.iters<<" time: "<<cuda_time<<"ms"<<std::endl;
   
   std::cout<<std::endl<<"GPU perf./CPU perf. = "<<cpu_time/cuda_time<<std::endl;

   time_t now = time(NULL);
   struct tm timeinfo = *localtime(&now);
   char name[50];
   strftime(name, sizeof(name),"%Y%m%d%H%M%S", &timeinfo);   
   std::string filename = std::string(name)+std::string(".log");
   std::ofstream fout(filename.c_str(), std::ios::app);
   fout<<"CPU RME\t\t"<<"GPU RME"<<std::endl;
   for (int i=0; i<std::min(pso.iters, cuda_pso.iters); i++) {
       fout<<pso.RME[i]<<"\t\t"<<cuda_pso.RME[i]<<std::endl;
   }
   fout.close();
   std::cout<<std::endl<<"RMEs have been written into ./"<<filename<<std::endl;
   /*
   std::cout<<std::endl;
   std::cout<<"CPU RME:"<<std::endl;
   for (int i=0; i<pso.iters; i++){
       std::cout<<pso.RME[i]<<"\t";
   }
   std::cout<<std::endl;
   std::cout<<"GPU RME:"<<std::endl;
   for (int i=0; i<cuda_pso.iters; i++){
       std::cout<<cuda_pso.RME[i]<<"\t";
   }
   */
   return 0;
}
