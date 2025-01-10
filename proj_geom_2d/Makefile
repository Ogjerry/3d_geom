# Compiler
CXX := g++
NVCC := /usr/local/cuda-12/bin/nvcc

# Compiler flags
CXXFLAGS = -std=c++20 -Wno-deprecated-enum-enum-conversion
NVCCFLAGS = -std=c++20 -arch=sm_86 -O2 -g -diag-suppress=549 -diag-suppress=20096

# Include and library paths for OpenCV & CUDA
CUDA_INCLUDE = -I/usr/local/cuda/include
OPENCV_INCLUDE = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_calib3d
CUBLAS_LIBS = -lineinfo -lcublas -lcusolver -lcuda -lcudart


# Source files
CAM_SRCS = cam.cc
DLT_SRCS = DLT.cc
PLUCKER_SRCS = plucker.cc
NORM_DLT_SRCS = nDLT.cu

# Object files
CAM_OBJS = $(CAM_SRCS:.cc=.o)
DLT_OBJS = $(DLT_SRCS:.cc=.o)
PLUCKER_OBJS = $(PLUCKER_SRCS:.cc=.o)
NORM_DLT_OBJS = $(NORM_DLT_SRCS:.cu=.o)

# Executables
CAM_TARGET := cam
DLT_TARGET := dlt
PLUCKER_TARGET := pluck
NORM_DLT_TARGET := norm_dlt

# Default target
all: $(CAM_TARGET) $(DLT_TARGET) $(PLUCKER_TARGET) $(NORM_DLT_TARGET)

# Link the object files to create the final executables


# default input: 4 0 0 100 0 0 100 100 100 0 33 85.86 24.92 15.15 70.37 100 63
$(NORM_DLT_TARGET): $(NORM_DLT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(NORM_DLT_TARGET) $(NORM_DLT_OBJS) $(CUBLAS_LIBS) $(OPENCV_LIBS)

$(PLUCKER_TARGET): $(PLUCKER_OBJS)
	$(CXX) $(CXXFLAGS) -o $(PLUCKER_TARGET) $(PLUCKER_OBJS) $(OPENCV_LIBS)

$(CAM_TARGET): $(CAM_OBJS)
	$(CXX) $(CXXFLAGS) -o $(CAM_TARGET) $(CAM_OBJS) $(OPENCV_LIBS)

$(DLT_TARGET): $(DLT_OBJS)
	$(CXX) $(CXXFLAGS) -o $(DLT_TARGET) $(DLT_OBJS) $(OPENCV_LIBS)

# Compile the source files into object files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDE) -c $< -o $@
# Compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(OPENCV_INCLUDE) $(CUDA_INCLUDE) -c $< -o $@


# RUN

run_plucker:
	./$(PLUCKER_TARGET)

run_cam:
	./$(CAM_TARGET)

run_dlt:
	./$(DLT_TARGET)

# Clean up build files
clean:
	rm -f $(CAM_OBJS) $(DLT_OBJS) $(PLUCKER_OBJS) $(NORM_DLT_OBJS) \
	 $(CAM_TARGET) $(DLT_TARGET) $(PLUCKER_TARGET) $(NORM_DLT_TARGET)

.PHONY: all clean run_cam run_dlt plucker run_plucker
