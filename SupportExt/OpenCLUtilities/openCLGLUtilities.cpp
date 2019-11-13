#include "openCLGLUtilities.hpp"


cl::Context createCLGLContext(cl_device_type type, cl_vendor vendor) {

    cl::Platform platform = getPlatform(type, vendor);

    //Creating the context
#if defined(__APPLE__) || defined(__MACOSX)
    // Apple (untested)
    cl_context_properties cps[] = {
       CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
       (cl_context_properties)CGLGetShareGroup(CGLGetCurrentContext()),
       0};


#else
  #ifdef _WIN32
      // Windows
      cl_context_properties cps[] = {
          CL_GL_CONTEXT_KHR,
          (cl_context_properties)wglGetCurrentContext(),
          CL_WGL_HDC_KHR,
          (cl_context_properties)wglGetCurrentDC(),
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)(platform)(),
          0
      };
  #else
      // Linux
      cl_context_properties cps[] = {
          CL_GL_CONTEXT_KHR,
          (cl_context_properties)glXGetCurrentContext(),
          CL_GLX_DISPLAY_KHR,
          (cl_context_properties)glXGetCurrentDisplay(),
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)(platform)(),
          0
      };
  #endif
#endif

    try {

		// We need to check if there is more than one device first
      VECTOR_CLASS<cl::Device> devices;
      VECTOR_CLASS<cl::Device> singleDevice;
		platform.getDevices(type, &devices);
		cl::Context context;

		// If more than one CL device find out which one is associated with GL context
		if(devices.size() > 1) {
#if !(defined(__APPLE__) || defined(__MACOSX))
			cl::Device interopDevice = getValidGLCLInteropDevice(platform, cps);
			singleDevice.push_back(interopDevice);
			context = cl::Context(singleDevice, cps);
#else
			context = cl::Context(type,cps);
#endif
		} else {
			context = cl::Context(type, cps);
		}

        return context;
    } catch(cl::Error error) {
        throw error;
    }
}

#if !(defined(__APPLE__) || defined(__MACOSX))
cl::Device getValidGLCLInteropDevice(cl::Platform platform, cl_context_properties* properties) {
    // Function for finding a valid device for CL-GL context. 
    // Thanks to Jim Vaughn for this contribution
	cl::Device displayDevice;
	
	cl_device_id interopDeviceId;

	int status;
	size_t deviceSize = 0;

	// Load extension function call
	clGetGLContextInfoKHR_fn glGetGLContextInfo_func = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");

	// Ask for the CL device associated with the GL context
	status = glGetGLContextInfo_func( properties, 
                                    CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                                    sizeof(cl_device_id), 
                                    &interopDeviceId, 
                                    &deviceSize);
	
	if(deviceSize == 0) {
        throw cl::Error(1,"No GLGL devices found for current platform");
	}

	if(status != CL_SUCCESS) {
		throw cl::Error(1, "Could not get CLGL interop device for the current platform. Failure occured during call to clGetGLContextInfoKHR.");
	}

	return cl::Device(interopDeviceId);
}
#endif
