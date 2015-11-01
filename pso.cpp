#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include "pso.hpp"
using namespace std;

const double pi = 3.1415926535898;
double compute_fit(double x, double y)
{
   double re = 0;
   double  data[9] = {
   0.004450342097284,   0.017440060625698,   0.025142479360125,
   0.018345616768773,   0.071893050392351,   0.103644681886180,
   0.027821257493749,   0.109026319048659,   0.157177892624538
    };

   if (x<-1 || x>1 || y<-1 || y>1) {
       re = 9;
   } else {
       vector<vector<int> > grid;
       for (int j=-1; j<=1; j++) {
           for (int i=-1; i<=1; i++) {
               vector<int> point;
               point.push_back(i);
               point.push_back(j);
               grid.push_back(point);
           }
       }
       vector<double> dd;
       double square_sum = 0;
       for (int i=0; i<grid.size(); i++) {
           double tmp = 1/(2*pi)*exp(-((grid[i][0]-x)*(grid[i][0]-x)+(grid[i][1]-y)*(grid[i][1]-y))/2);
           dd.push_back(tmp);
       }
       for (int i=0; i<dd.size(); i++) {
           square_sum += (dd[i]-data[i])*(dd[i]-data[i]);
       }
       re = sqrt(square_sum/9);
   }
   return re;
}

PSO::PSO(int n) 
{
    this->n = n;
    par = new particle[n];
    Init();
    gBest = par[0];
}

inline float uniform_rand()
{
    return ((float)rand())/RAND_MAX;
}

void PSO::Init()
{
    srand(time(NULL));
    for (int i=0; i<n; i++) {
        par[i].x = -1+2*uniform_rand();
        par[i].y = -1+2*uniform_rand();
        par[i].vx = -0.01+0.02*uniform_rand();
        par[i].vy = -0.01+0.02*uniform_rand();
        par[i].fit = compute_fit(par[i].x, par[i].y);
        par[i].bestfit = par[i].fit;
        par[i].bestx = par[i].x;
        par[i].besty = par[i].y;
    }
}

float PSO::Solve(int m, float eps) 
{
    int k;
    for (k=0; k<m; k++) {
        for (int i=0; i<n; i++) {
            par[i].fit = compute_fit(par[i].x, par[i].y);
            if (par[i].fit < par[i].bestfit) {
                par[i].bestfit = par[i].fit;
                par[i].bestx = par[i].x;
                par[i].besty = par[i].y;
                if (par[i].bestfit < gBest.fit) {
                   gBest.fit = par[i].bestfit;
                   gBest.x = par[i].x;
                   gBest.y = par[i].y;
                   cout<<i<<"\t"<<gBest.fit<<"\t";
                }
           }

          int c1 = 1;
          int c2 = 1;
          par[i].vx = par[i].vx + c1*uniform_rand()*(gBest.x - par[i].x)+c2*uniform_rand()*(par[i].bestx-par[i].x);
          par[i].vy = par[i].vy + c1*uniform_rand()*(gBest.y - par[i].y)+c2*uniform_rand()*(par[i].besty-par[i].y);
          par[i].x = par[i].x+par[i].vx;
          par[i].y = par[i].y+par[i].vy;
       }
       if (gBest.fit < eps) {
           break;
       }
       //rme(k) = gBest.fit;
    }
    iters = k;
    return 0.0;
}

