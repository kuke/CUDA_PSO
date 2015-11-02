#include <iostream>
#include <fstream>
#include "pso.hpp"
#include "cuda_pso.cuh"
#include <string>
#include <helper_string.h>

void helper()
{
    std::cout<<"Usage: pso.bin -n=<uint> -m=<uint> -threads=<uint>"<<std::endl;
    std::cout<<"      -n, optional, num of particles, 1024 default;"<<std::endl;
    std::cout<<"      -m, optional, max iterations, 50 default;"<<std::endl;
    std::cout<<"-threads, optional, num of threads per block, 32 default."<<std::endl;
}

int main(int argc, char **argv)
{
   int numParticles = 1024;
   int maxIters = 50;
   int numThreads = 32;
   float eps = 10^-6;
   if (checkCmdLineFlag(argc,(const char **)argv,"h") || checkCmdLineFlag(argc,(const char **)argv,"help"))
   {
       helper();
       exit(0);
   } 
   if (checkCmdLineFlag(argc, (const char **) argv, "n"))
   {
       numParticles = getCmdLineArgumentFloat(argc, (const char **)argv, "n");
   }
   if (checkCmdLineFlag(argc, (const char **) argv, "m"))
   {    
       maxIters = getCmdLineArgumentFloat(argc, (const char **)argv, "m");
   }
   if (checkCmdLineFlag(argc, (const char **) argv, "threads"))
   {
       numThreads = getCmdLineArgumentFloat(argc, (const char **)argv, "threads");
   }
   std::cout<<"PSO Algorithm: "<<" n= "<<numParticles<<", m= "<<maxIters<<", threads= "<<numThreads<<std::endl;
   PSO pso(numParticles);
   float cpu_time = pso.Solve(maxIters, eps);
   std::cout<<"CPU result: "<<std::endl;
   std::cout<<"a: "<<pso.gBest.x<<" b: "<<pso.gBest.y<<" iters: "<<pso.iters<<" time: "<<cpu_time<<"ms"<<std::endl;
   
   CudaPSO cuda_pso(numParticles);
   float cuda_time = cuda_pso.Solve(maxIters, numThreads, eps);
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
