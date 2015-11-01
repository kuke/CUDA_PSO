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

class PSO{
private:
    int n;
    particle *par;
    void Init();
 
public:
    particle gBest;
    int iters;
    PSO(int n);
    float Solve(int m, float eps);
};
