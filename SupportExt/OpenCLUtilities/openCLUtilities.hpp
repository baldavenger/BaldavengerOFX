#ifndef OPENCL_UTILITIES_H
#define OPENCL_UTILITIES_H

//#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS


#if defined(__APPLE__) || defined(__MACOSX)
    #include "OpenCL/cl.hpp"
#else
    #include <CL/cl.hpp>
#endif


#include <string>
#include <iostream>
#include <fstream>
#include <set>


enum cl_vendor {
    VENDOR_ANY,
    VENDOR_NVIDIA,
    VENDOR_AMD,
    VENDOR_INTEL
};

class GarbageCollector {
    public:
        void addMemObject(cl::Memory * mem);
        void deleteMemObject(cl::Memory * mem);
        void deleteAllMemObjects();
        ~GarbageCollector();
    private:
        std::set<cl::Memory *> memObjects;
};
typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Device device;
    cl::Platform platform;
    GarbageCollector GC;
} OpenCL;

cl::Context createCLContextFromArguments(int argc, char ** argv);

cl::Context createCLContext(cl_device_type type = CL_DEVICE_TYPE_ALL, cl_vendor vendor = VENDOR_ANY);

cl::Platform getPlatform(cl_device_type = CL_DEVICE_TYPE_ALL, cl_vendor vendor = VENDOR_ANY); 

cl::Program buildProgramFromSource(cl::Context context, std::string filename, std::string buildOptions = "");

cl::Program buildProgramFromBinary(cl::Context context, std::string filename, std::string buildOptions = "");

char *getCLErrorString(cl_int err);

#endif
