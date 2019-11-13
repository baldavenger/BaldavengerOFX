#ifndef OPENCLGL_UTILITIES_H
#define OPENCLGL_UTILITIES_H

#if defined (__APPLE__) || defined(MACOSX)
   #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
   #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif
        
#include "openCLUtilities.hpp"

#if defined(__APPLE__) || defined(__MACOSX)
   #include <OpenCL/cl_gl.h>
   #include <OpenGL/OpenGL.h>
#else
#if _WIN32
#else
   #include <GL/glx.h>
   #include <CL/cl_gl.h>
#endif 
#endif

cl::Context createCLGLContext(cl_device_type type = CL_DEVICE_TYPE_ALL, cl_vendor vendor = VENDOR_ANY);
#if !(defined(__APPLE__) || defined(__MACOSX))
cl::Device getValidGLCLInteropDevice(cl::Platform platform, cl_context_properties* properties);
#endif

#endif
