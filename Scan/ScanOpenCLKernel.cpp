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
"__kernel void ScanAdjustKernel(	\n" \
"   int p_Width,	\n" \
"   int p_Height,	\n" \
"   float balGainR,  \n" \
"   float balGainB,  \n" \
"   float balOffsetR,  \n" \
"   float balOffsetB,  \n" \
"   float balLiftR,  \n" \
"   float balLiftB,  \n" \
"   float lumaMath,  \n" \
"   float lumaLimit,  \n" \
"   float GainBalance,  \n" \
"   float OffsetBalance,  \n" \
"   float WhiteBalance,  \n" \
"   float PreserveLuma,  \n" \
"   float DisplayAlpha,  \n" \
"   float LumaRec709,  \n" \
"   float LumaRec2020,  \n" \
"   float LumaDCIP3,  \n" \
"   float LumaACESAP0,  \n" \
"   float LumaACESAP1,  \n" \
"   float LumaAvg, \n" \
"   __global const float* p_Input,\n" \
"   __global float* p_Output)\n" \
"{\n" \
"   float SRC[BLOCKSIZE]; \n" \
"   float w_balGainR;  \n" \
"   float w_balGainB;  \n" \
"   float w_balOffsetR;  \n" \
"   float w_balOffsetB;  \n" \
"   float w_balLiftR;  \n" \
"   float w_balLiftB;  \n" \
"   float w_lumaMath;  \n" \
"   float w_lumaLimit;  \n" \
"   float w_GainBalance;  \n" \
"   float w_OffsetBalance;  \n" \
"   float w_WhiteBalance;  \n" \
"   float w_PreserveLuma;  \n" \
"   float w_DisplayAlpha;  \n" \
"   float w_LumaRec709;  \n" \
"   float w_LumaRec2020;  \n" \
"   float w_LumaDCIP3;  \n" \
"   float w_LumaACESAP0;  \n" \
"   float w_LumaACESAP1;  \n" \
"   float w_LumaAvg; \n" \
"   float lumaRec709; \n" \
"   float lumaRec2020; \n" \
"   float lumaDCIP3;  \n" \
"   float lumaACESAP0; \n" \
"   float lumaACESAP1; \n" \
"   float lumaAvg; \n" \
"   float lumaMax; \n" \
"   float luma; \n" \
"   float alpha; \n" \
"   float Alpha; \n" \
"   float BalR; \n" \
"   float BalB; \n" \
"   float Red; \n" \
"   float Green; \n" \
"   float Blue; \n" \
"   const int x = get_global_id(0);                                     \n" \
"   const int y = get_global_id(1);                                     \n" \
"                                                                       \n" \
"   if ((x < p_Width) && (y < p_Height))                                \n" \
"   {                                                                   \n" \
"       const int index = ((y * p_Width) + x) * BLOCKSIZE;             \n" \
"                                                                       \n" \
"       SRC[0] = p_Input[index + 0] ;    \n" \
"       SRC[1] = p_Input[index + 1] ;    \n" \
"       SRC[2] = p_Input[index + 2] ;    \n" \
"       SRC[3] = p_Input[index + 3] ;    \n" \
"    	w_balGainR = balGainR;  \n" \
"    	w_balGainB = balGainB;  \n" \
"    	w_balOffsetR = balOffsetR;  \n" \
"    	w_balOffsetB = balOffsetB;  \n" \
"    	w_balLiftR = balLiftR;  \n" \
"    	w_balLiftB = balLiftB;  \n" \
"    	w_lumaMath = lumaMath;  \n" \
"    	w_lumaLimit = lumaLimit;  \n" \
"    	w_GainBalance = GainBalance;  \n" \
"    	w_OffsetBalance = OffsetBalance;  \n" \
"    	w_WhiteBalance = WhiteBalance;  \n" \
"    	w_PreserveLuma = PreserveLuma;  \n" \
"    	w_DisplayAlpha = DisplayAlpha;  \n" \
"    	w_LumaRec709 = LumaRec709;  \n" \
"    	w_LumaRec2020 = LumaRec2020;  \n" \
"    	w_LumaDCIP3 = LumaDCIP3;  \n" \
"    	w_LumaACESAP0 = LumaACESAP0;  \n" \
"    	w_LumaACESAP1 = LumaACESAP1;  \n" \
"    	w_LumaAvg = LumaAvg; \n" \
"                                                                       \n" \
"      float lumaRec709 = SRC[0] * 0.2126f + SRC[1] * 0.7152f + SRC[2] * 0.0722f;\n" \
"	   float lumaRec2020 = SRC[0] * 0.2627f + SRC[1] * 0.6780f + SRC[2] * 0.0593f;\n" \
"	   float lumaDCIP3 = SRC[0] * 0.209492f + SRC[1] * 0.721595f + SRC[2] * 0.0689131f;\n" \
"	   float lumaACESAP0 = SRC[0] * 0.3439664498f + SRC[1] * 0.7281660966f + SRC[2] * -0.0721325464f;\n" \
"      float lumaACESAP1 = SRC[0] * 0.2722287168f + SRC[1] * 0.6740817658f + SRC[2] * 0.0536895174f;\n" \
"      float lumaAvg = (SRC[0] + SRC[1] + SRC[2]) / 3.0f;\n" \
"      float lumaMax = fmax(SRC[2], fmax(SRC[0], SRC[1]));\n" \
"      float luma = w_LumaRec709 == 1.0f ? lumaRec709 : w_LumaRec2020 == 1.0f ? lumaRec2020 : w_LumaDCIP3 == 1.0f ? lumaDCIP3 : w_LumaACESAP0 == 1.0f ? lumaACESAP0 : w_LumaACESAP1 == 1.0f ? lumaACESAP1 : w_LumaAvg == 1.0f ? lumaAvg : lumaMax;\n" \
"      \n" \
"      float alpha = w_lumaLimit > 1.0f ? luma + (1.0f - w_lumaLimit) * (1.0f - luma) : w_lumaLimit >= 0.0f ? (luma >= w_lumaLimit ? \n" \
"      1.0f : luma / w_lumaLimit) : w_lumaLimit < -1.0f ? (1.0f - luma) + (w_lumaLimit + 1.0f) * luma : luma <= (1.0f + w_lumaLimit) ? 1.0f : \n" \
"      (1.0f - luma) / (1.0f - (w_lumaLimit + 1.0f));\n" \
"      float Alpha = alpha > 1.0f ? 1.0f : alpha < 0.0f ? 0.0f : alpha;\n" \
"      \n" \
"      float BalR = w_GainBalance == 1.0f ? SRC[0] * w_balGainR : w_OffsetBalance == 1.0f ? SRC[0] + w_balOffsetR : SRC[0] + (w_balLiftR * (1.0f - SRC[0]));\n" \
"      float BalB = w_GainBalance == 1.0f ? SRC[2] * w_balGainB : w_OffsetBalance == 1.0f ? SRC[2] + w_balOffsetB : SRC[2] + (w_balLiftB * (1.0f - SRC[2]));\n" \
"      float Red = w_WhiteBalance == 1.0f ? ( w_PreserveLuma == 1.0f ? BalR * w_lumaMath : BalR) : SRC[0];\n" \
"      float Green = w_WhiteBalance && w_PreserveLuma == 1.0f ? SRC[1] * w_lumaMath : SRC[1]; \n" \
"      float Blue = w_WhiteBalance == 1.0f ? (w_PreserveLuma == 1.0f ? BalB * w_lumaMath : BalB) : SRC[2];\n" \
"      \n" \
"      SRC[0] = w_DisplayAlpha == 1.0f ? Alpha : Red * Alpha + SRC[0] * (1.0f - Alpha);\n" \
"      SRC[1] = w_DisplayAlpha == 1.0f ? Alpha : Green * Alpha + SRC[1] * (1.0f - Alpha);\n" \
"      SRC[2] = w_DisplayAlpha == 1.0f ? Alpha : Blue * Alpha + SRC[2] * (1.0f - Alpha);\n" \
"      SRC[3] = w_DisplayAlpha == 1.0f ? SRC[3] : Alpha;\n" \
"      p_Output[index + 0] = SRC[0];  \n" \
"      p_Output[index + 1] = SRC[1];  \n" \
"      p_Output[index + 2] = SRC[2];  \n" \
"      p_Output[index + 3] = SRC[3];  \n" \
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


void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* balGain, float* balOffset, float* balLift, 
float* lumaMath, float* lumaLimit, float* GainBalance, float* OffsetBalance, float* WhiteBalance, float* PreserveLuma, 
float* DisplayAlpha, float* LumaRec709, float* LumaRec2020, float* LumaDCIP3, float* LumaACESAP0, 
float* LumaACESAP1, float* LumaAvg, const float* p_Input, float* p_Output)
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

		kernel = clCreateKernel(program, "ScanAdjustKernel", &error);
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
    error |= clSetKernelArg(kernel, count++, sizeof(float), &balGain[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &balGain[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &balOffset[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &balOffset[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &balLift[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &balLift[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &lumaMath[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &lumaLimit[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &GainBalance[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &OffsetBalance[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &WhiteBalance[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &PreserveLuma[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &DisplayAlpha[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &LumaRec709[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &LumaRec2020[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &LumaDCIP3[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &LumaACESAP0[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &LumaACESAP1[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &LumaAvg[0]);
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
