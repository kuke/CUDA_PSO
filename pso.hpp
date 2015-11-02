#ifndef _PSO_HPP_
#define _PSO_HPP_

struct particle {
    float x;
    float y;
    float vx;
    float vy;
    float fit;
    float bestfit;
    float bestx;
    float besty;
};

inline double cpu_time()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC,&ts);
  return (ts.tv_sec*1000 + ts.tv_nsec/(1000*1000.0));
}

class PSO{
private:
    int n;
    particle *par;
    void Init();
 
public:
    particle gBest;
    float *rme
    int iters;
    PSO(int n);
    ~PSO();
    float Solve(int m, float eps);
};

#endif
