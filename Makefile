CXX := nvcc
TARGET := main
CUDNN_PATH := /usr/local/cuda-9.0
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/user/local/lib `pkg-config opencv --cflags --libs`

CXXFLAGS := -arch=sm_35 -std=c++11 -O2

all: main

main: $(TARGET).cu
		$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
		-lcudnn -lcublas -lopencv_imgproc -lopencv_core

.phony: clean

clean:
		rm $(TARGET) || echo -n ""
