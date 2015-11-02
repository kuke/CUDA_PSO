TARGET = pso.bin
OBJS = main.o pso.o 
CUOBJS = cuda_pso.o
CUDA_DIR := /usr/local/cuda-7.5
INCS := -I$(CUDA_DIR)/include -I$(CUDA_DIR)/samples/common/inc
LIBS := -L$(CUDA_DIR)/lib64 -lcuda  -lcudart -lcurand

$(TARGET): $(OBJS) $(CUOBJS)
	g++ -o $@ $(OBJS) $(CUOBJS) $(LIBS) 

$(OBJS): %.o: %.cpp
	g++ $(INCS) -c  $< -o $@

$(CUOBJS): %.o: %.cu
	nvcc  $(INCS) -c -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  $< -o $@

clean:
	rm $(OBJS) $(CUOBJS) $(TARGET)
