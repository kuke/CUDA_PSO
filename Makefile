TARGET = pso.bin
OBJS = main.o pso.o 
CUOBJS = cuda_pso.o
INCS := -I/usr/local/cuda-7.5/include 
LIBS := -L/usr/local/cuda-7.5/lib64 -lcuda  -lcudart -lcurand

$(TARGET): $(OBJS) $(CUOBJS)
	g++ -o $@ $(OBJS) $(CUOBJS) $(LIBS) 

$(OBJS): %.o: %.cpp
	g++ $(INCS) -c  $< -o $@

$(CUOBJS): %.o: %.cu
	nvcc  $(INCS) -c -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  $< -o $@

clean:
	rm $(OBJS) $(CUOBJS) $(TARGET)
