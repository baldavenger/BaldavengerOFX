#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
#include <stdio.h>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char *KernelSource = "\n" \
"#define  BLOCKSIZE 4 \n" \
"__kernel void ReplaceAdjustKernel(	\n" \
"   int p_Width,	\n" \
"   int p_Height,	\n" \
"   float hueRangeA, \n" \
"   float hueRangeB, \n" \
"   float hueRangeWithRollOffA, \n" \
"   float hueRangeWithRollOffB, \n" \
"   float hueRotation, \n" \
"   float hueMean, \n" \
"   float hueRotationGain, \n" \
"   float hueRolloff, \n" \
"   float satRangeA,\n" \
"   float satRangeB, \n" \
"   float satAdjust, \n" \
"   float satAdjustGain, \n" \
"   float satRolloff, \n" \
"   float valRangeA, \n" \
"   float valRangeB, \n" \
"   float valAdjust, \n" \
"   float valAdjustGain, \n" \
"   float valRolloff, \n" \
"   int OutputAlpha, \n" \
"   int DisplayAlpha, \n" \
"   float mix,\n" \
"   __global const float* p_Input,\n" \
"   __global float* p_Output)\n" \
"{\n" \
"   const int x = get_global_id(0);                                     \n" \
"   const int y = get_global_id(1);                                     \n" \
"                                                                       \n" \
"   if ((x < p_Width) && (y < p_Height))                                \n" \
"   {                                                                   \n" \
"       const int index = ((y * p_Width) + x) * BLOCKSIZE;             \n" \
"    float hcoeff, scoeff, vcoeff; \n" \
"	float r, g, b, h, s, v; \n" \
"             \n" \
"	r = p_Input[index + 0]; \n" \
"	g = p_Input[index + 1]; \n" \
"	b = p_Input[index + 2]; \n" \
"	 \n" \
"    float min = fmin(fmin(r, g), b); \n" \
"    float max = fmax(fmax(r, g), b); \n" \
"    v = max; \n" \
"    float delta = max - min; \n" \
" \n" \
"    if (max != 0.0f) { \n" \
"        s = delta / max; \n" \
"    } else { \n" \
"        s = 0.0f; \n" \
"        h = 0.0f; \n" \
"    } \n" \
" \n" \
"    if (delta == 0.0f) { \n" \
"        h = 0.0f; \n" \
"    } else if (r == max) { \n" \
"        h = (g - b) / delta; \n" \
"    } else if (g == max) { \n" \
"        h = 2 + (b - r) / delta; \n" \
"    } else { \n" \
"        h = 4 + (r - g) / delta; \n" \
"    } \n" \
"    h *= 1 / 6.0f; \n" \
"    if (h < 0.0f) { \n" \
"        h += 1.0f; \n" \
"    } \n" \
"	 \n" \
"	float R, G, B; \n" \
" \n" \
"	h *= 360.0f; \n" \
"	float h0 = hueRangeA; \n" \
"	float h1 = hueRangeB; \n" \
"	float h0mrolloff = hueRangeWithRollOffA; \n" \
"	float h1prolloff = hueRangeWithRollOffB; \n" \
"	 \n" \
"	if ( ( h1 < h0 && (h <= h1 || h0 <= h) ) || (h0 <= h && h <= h1) ) { \n" \
"		hcoeff = 1.0f; \n" \
"	} else { \n" \
"		float c0 = 0.0f; \n" \
"		float c1 = 0.0f; \n" \
"		 \n" \
"		if ( ( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0) ) { \n" \
"		c0 = h0 == (h0mrolloff + 360.0f) || h0 == h0mrolloff ? 1.0f : !(( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0)) ? 0.0f :  \n" \
"		((h < h0mrolloff ? h + 360.0f : h) - h0mrolloff) / ((h0 < h0mrolloff ? h0 + 360.0f : h0) - h0mrolloff);		 \n" \
"		} \n" \
"		 \n" \
"		if ( ( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff) ) { \n" \
"		c1 = !(( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff)) ? 0.0f : h1prolloff == h1 ? 1.0f : \n" \
"		((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - (h < h1 ? h + 360.0f : h)) / ((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - h1);	 \n" \
"		} \n" \
"		 \n" \
"		hcoeff = fmax(c0, c1); \n" \
"	} \n" \
"	float s0 = satRangeA; \n" \
"	float s1 = satRangeB; \n" \
"	float s0mrolloff = s0 - satRolloff; \n" \
"	float s1prolloff = s1 + satRolloff; \n" \
"	if ( (s0 <= s) && (s <= s1) ) { \n" \
"		scoeff = 1.0f; \n" \
"	} else if ( (s0mrolloff <= s) && (s <= s0) ) { \n" \
"		scoeff = (s - s0mrolloff) / satRolloff; \n" \
"	} else if ( (s1 <= s) && (s <= s1prolloff) ) { \n" \
"		scoeff = (s1prolloff - s) / satRolloff; \n" \
"	} else { \n" \
"		scoeff = 0.0f; \n" \
"	} \n" \
"	float v0 = valRangeA; \n" \
"	float v1 = valRangeB; \n" \
"	float v0mrolloff = v0 - valRolloff; \n" \
"	float v1prolloff = v1 + valRolloff; \n" \
"	if ( (v0 <= v) && (v <= v1) ) { \n" \
"		vcoeff = 1.0f; \n" \
"	} else if ( (v0mrolloff <= v) && (v <= v0) ) { \n" \
"		vcoeff = (v - v0mrolloff) / valRolloff; \n" \
"	} else if ( (v1 <= v) && (v <= v1prolloff) ) { \n" \
"		vcoeff = (v1prolloff - v) / valRolloff; \n" \
"	} else { \n" \
"		vcoeff = 0.0f; \n" \
"	} \n" \
"	float coeff = fmin(fmin(hcoeff, scoeff), vcoeff); \n" \
"	if (coeff <= 0.0f) { \n" \
"		R = p_Input[index + 0]; \n" \
"		G = p_Input[index + 1]; \n" \
"		B = p_Input[index + 2]; \n" \
"	} else { \n" \
"	 \n" \
"		float H = (h - hueMean + 180) - (int)(floor((h - hueMean + 180) / 360) * 360) - 180; \n" \
"		h += coeff * ( hueRotation + (hueRotationGain - 1.0f) * H ); \n" \
"		s += coeff * ( satAdjust + (satAdjustGain - 1.0f) * (s - (s0 + s1) / 2) ); \n" \
"		if (s < 0.0f) { \n" \
"			s = 0.0f; \n" \
"		} \n" \
"		v += coeff * ( valAdjust + (valAdjustGain - 1.0f) * (v - (v0 + v1) / 2) ); \n" \
"		h *= 1 / 360.0f; \n" \
"	 \n" \
"    if (s == 0.0f) { \n" \
"        R = G = B = v; \n" \
"    } \n" \
"    h *= 6.0f; \n" \
"    int i = floor(h); \n" \
"    float f = h - i; \n" \
"    i = (i >= 0) ? (i % 6) : (i % 6) + 6; \n" \
"    float p = v * ( 1.0f - s ); \n" \
"    float q = v * ( 1.0f - s * f ); \n" \
"    float t = v * ( 1.0f - s * ( 1.0f - f )); \n" \
" \n" \
"    if (i == 0){ \n" \
"        R = v; \n" \
"        G = t; \n" \
"        B = p;} \n" \
"    else if (i == 1){ \n" \
"        R = q; \n" \
"        G = v; \n" \
"        B = p;} \n" \
"    else if (i == 2){ \n" \
"        R = p; \n" \
"        G = v; \n" \
"        B = t;} \n" \
"    else if (i == 3){ \n" \
"        R = p; \n" \
"        G = q; \n" \
"        B = v;} \n" \
"	else if (i == 4){ \n" \
"        R = t; \n" \
"        G = p; \n" \
"        B = v;} \n" \
"	else{ \n" \
"		R = v; \n" \
"        G = p; \n" \
"        B = q;} \n" \
"        } \n" \
"	   	 \n" \
"		float a = OutputAlpha == 0 ? 1.0f : OutputAlpha == 1 ? hcoeff : OutputAlpha == 2 ? scoeff : \n" \
"		OutputAlpha == 3 ? vcoeff : OutputAlpha == 4 ? fmin(hcoeff, scoeff) : OutputAlpha == 5 ?  \n" \
"		fmin(hcoeff, vcoeff) : OutputAlpha == 6 ? fmin(scoeff, vcoeff) : fmin(fmin(hcoeff, scoeff), vcoeff); \n" \
"		p_Output[index + 0] = DisplayAlpha == 1 ? a : R * (1.0f - mix) + p_Input[index + 0] * mix; \n" \
"		p_Output[index + 1] = DisplayAlpha == 1 ? a : G * (1.0f - mix) + p_Input[index + 1] * mix; \n" \
"		p_Output[index + 2] = DisplayAlpha == 1 ? a : B * (1.0f - mix) + p_Input[index + 2] * mix; \n" \
"		p_Output[index + 3] = OutputAlpha != 0 ? a : p_Input[index + 3]; \n" \
"   }                                                                   \n" \
"}                                                                      \n" \
"\n";

class Locker
{
public:
	Locker()
	{
#ifdef _WIN64
		InitializeCriticalSection(&mutex);
#else
		pthread_mutex_init(&mutex, NULL);
#endif
	}

	~Locker()
	{
#ifdef _WIN64
		DeleteCriticalSection(&mutex);
#else
		pthread_mutex_destroy(&mutex);
#endif
	}

	void Lock()
	{
#ifdef _WIN64
		EnterCriticalSection(&mutex);
#else
		pthread_mutex_lock(&mutex);
#endif
	}

	void Unlock()
	{
#ifdef _WIN64
		LeaveCriticalSection(&mutex);
#else
		pthread_mutex_unlock(&mutex);
#endif
	}

private:
#ifdef _WIN64
	CRITICAL_SECTION mutex;
#else
	pthread_mutex_t mutex;
#endif
};

void CheckError(cl_int p_Error, const char* p_Msg)
{
	if (p_Error != CL_SUCCESS)
	{
		fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
	}
}


void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* hueRange, float* hueRangeWithRollOff, 
	float hueRotation, float hueMean, float hueRotationGain, float hueRolloff, float* satRange, 
	float satAdjust, float satAdjustGain, float satRolloff, float* valRange, float valAdjust, 
	float valAdjustGain, float valRolloff, int OutputAlpha, int DisplayAlpha, float mix, 
    const float* p_Input, float* p_Output)
{	
	cl_int error;

	cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

	// store device id and kernel per command queue (required for multi-GPU systems)
	static std::map<cl_command_queue, cl_device_id> deviceIdMap;
	static std::map<cl_command_queue, cl_kernel> kernelMap;

	static Locker locker; // simple lock to control access to the above maps from multiple threads

	locker.Lock();

	// find the device id corresponding to the command queue
	cl_device_id deviceId = NULL;
	if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
	{
		error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
		CheckError(error, "Unable to get the device");

		deviceIdMap[cmdQ] = deviceId;
	}
	else
	{
		deviceId = deviceIdMap[cmdQ];
	}

//#define _DEBUG

	// find the program kernel corresponding to the command queue
	cl_kernel kernel = NULL;
	if (kernelMap.find(cmdQ) == kernelMap.end())
	{
		cl_context clContext = NULL;
		error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
		CheckError(error, "Unable to get the context");

		cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
		CheckError(error, "Unable to create program");

		error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#ifdef _DEBUG
		error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			char buffer[4096];
			size_t length;
			clGetProgramBuildInfo
				(
				program,
				// valid program object
				deviceId,
				// valid device_id that executable was built
				CL_PROGRAM_BUILD_LOG,
				// indicate to retrieve build log
				sizeof(buffer),
				// size of the buffer to write log to
				buffer,
				// the actual buffer to write log to
				&length);
			// the actual size in bytes of data copied to buffer
			FILE * pFile;
			pFile = fopen("/", "w");
			if (pFile != NULL)
			{
				fprintf(pFile, "%s\n", buffer);
				//fprintf(pFile, "%s [%lu]\n", "localWorkSize 0 =", szWorkSize);
			}
			fclose(pFile);
		}
#else
		CheckError(error, "Unable to build program");
#endif

		kernel = clCreateKernel(program, "ReplaceAdjustKernel", &error);
		CheckError(error, "Unable to create kernel");

		kernelMap[cmdQ] = kernel;
	}
	else
	{
		kernel = kernelMap[cmdQ];
	}

	locker.Unlock();

    int count = 0;
    error  = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &hueRange[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &hueRange[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &hueRangeWithRollOff[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &hueRangeWithRollOff[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &hueRotation);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &hueMean);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &hueRotationGain);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &hueRolloff);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &satRange[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &satRange[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &satAdjust);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &satAdjustGain);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &satRolloff);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &valRange[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &valRange[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &valAdjust);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &valAdjustGain);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &valRolloff);
	error |= clSetKernelArg(kernel, count++, sizeof(int), &OutputAlpha);
	error |= clSetKernelArg(kernel, count++, sizeof(int), &DisplayAlpha);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &mix);
	error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Output);
	CheckError(error, "Unable to set kernel arguments");

	size_t localWorkSize[2], globalWorkSize[2];
	clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
	localWorkSize[1] = 1;
	globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
	globalWorkSize[1] = p_Height;

	clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
