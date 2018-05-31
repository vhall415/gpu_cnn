CXX := nvcc
TARGET := main
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/user/local/lib
CXXFLAGS := -arch=sm_35 -std=c++11 -O2

all: conv

main: $(TARGET).cu
		$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
		-lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

.phony: clean

clean:
		rm $(TARGET) || echo -n ""
