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

size_t localWorkSize[2], globalWorkSize[2];
cl_int error;

const char *KernelSource = \
"float Luma(float R, float G, float B, int L) { \n" \
"float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f; \n" \
"float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f; \n" \
"float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f; \n" \
"float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f; \n" \
"float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f; \n" \
"float lumaAvg = (R + G + B) / 3.0f; \n" \
"float lumaMax = fmax(fmax(R, G), B); \n" \
"float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax; \n" \
"return Lu; \n" \
"} \n" \
"float Sat(float r, float g, float b){ \n" \
"float min = fmin(fmin(r, g), b); \n" \
"float max = fmax(fmax(r, g), b); \n" \
"float delta = max - min; \n" \
"float S = max != 0.0f ? delta / max : 0.0f; \n" \
"return S; \n" \
"} \n" \
"__kernel void MatrixKernel(__global const float* p_Input, __global float* p_Output, int p_Width, int p_Height, float p_MatrixRR, float p_MatrixRG, float p_MatrixRB, \n" \
"float p_MatrixGR, float p_MatrixGG, float p_MatrixGB, float p_MatrixBR, float p_MatrixBG, float p_MatrixBB, int p_Luma, int p_Sat, int p_LumaMath) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float red = p_Input[index] * p_MatrixRR + p_Input[index + 1] * p_MatrixRG + p_Input[index + 2] * p_MatrixRB; \n" \
"float green = p_Input[index] * p_MatrixGR + p_Input[index + 1] * p_MatrixGG + p_Input[index + 2] * p_MatrixGB; \n" \
"float blue = p_Input[index] * p_MatrixBR + p_Input[index + 1] * p_MatrixBG + p_Input[index + 2] * p_MatrixBB; \n" \
"if (p_Luma == 1) { \n" \
"float inLuma = Luma(p_Input[index], p_Input[index + 1], p_Input[index + 2], p_LumaMath); \n" \
"float outLuma = Luma(red, green, blue, p_LumaMath); \n" \
"red = red * (inLuma / outLuma); \n" \
"green = green * (inLuma / outLuma); \n" \
"blue = blue * (inLuma / outLuma); \n" \
"} \n" \
"if (p_Sat == 1) { \n" \
"float inSat = Sat(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float outSat = Sat(red, green, blue); \n" \
"float satgap = inSat / outSat; \n" \
"float sLuma = Luma(red, green, blue, p_LumaMath); \n" \
"float sr = (1.0f - satgap) * sLuma + red * satgap; \n" \
"float sg = (1.0f - satgap) * sLuma + green * satgap; \n" \
"float sb = (1.0f - satgap) * sLuma + blue * satgap; \n" \
"red = inSat == 0.0f ? sLuma : sr; \n" \
"green = inSat == 0.0f ? sLuma : sg; \n" \
"blue = inSat == 0.0f ? sLuma : sb; \n" \
"} \n" \
"p_Output[index] = red; \n" \
"p_Output[index + 1] = green; \n" \
"p_Output[index + 2] = blue; \n" \
"p_Output[index + 3] = p_Input[index + 3];  \n" \
"}} \n" \
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

int clDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

void CheckError(cl_int p_Error, const char* p_Msg)
{
if (p_Error != CL_SUCCESS) {
fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
}}

void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Matrix, int p_Luma, int p_Sat, int p_LumaMath)
{
cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);
static std::map<cl_command_queue, cl_device_id> deviceIdMap;
static std::map<cl_command_queue, cl_kernel> kernelMap;
static Locker locker;
locker.Lock();
cl_device_id deviceId = NULL;
if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
{
error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
CheckError(error, "Unable to get the device");
deviceIdMap[cmdQ] = deviceId;
} else {
deviceId = deviceIdMap[cmdQ];
}

cl_context clContext = NULL;
error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
CheckError(error, "Unable to get the context");

cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
CheckError(error, "Unable to create program");

error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
CheckError(error, "Unable to build program");

cl_kernel Matrix = NULL;

Matrix = clCreateKernel(program, "MatrixKernel", &error);
CheckError(error, "Unable to create kernel");

locker.Unlock();

localWorkSize[0] = 128;
localWorkSize[1] = 1;
globalWorkSize[0] = localWorkSize[0] * clDivUp(p_Width, localWorkSize[0]);
globalWorkSize[1] = p_Height;

error = clSetKernelArg(Matrix, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Matrix, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Matrix, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Matrix, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Matrix, 4, sizeof(float), &p_Matrix[0]);
error |= clSetKernelArg(Matrix, 5, sizeof(float), &p_Matrix[1]);
error |= clSetKernelArg(Matrix, 6, sizeof(float), &p_Matrix[2]);
error |= clSetKernelArg(Matrix, 7, sizeof(float), &p_Matrix[3]);
error |= clSetKernelArg(Matrix, 8, sizeof(float), &p_Matrix[4]);
error |= clSetKernelArg(Matrix, 9, sizeof(float), &p_Matrix[5]);
error |= clSetKernelArg(Matrix, 10, sizeof(float), &p_Matrix[6]);
error |= clSetKernelArg(Matrix, 11, sizeof(float), &p_Matrix[7]);
error |= clSetKernelArg(Matrix, 12, sizeof(float), &p_Matrix[8]);
error |= clSetKernelArg(Matrix, 13, sizeof(int), &p_Luma);
error |= clSetKernelArg(Matrix, 14, sizeof(int), &p_Sat);
error |= clSetKernelArg(Matrix, 15, sizeof(int), &p_LumaMath);
CheckError(error, "Unable to set kernel arguments");

clEnqueueNDRangeKernel(cmdQ, Matrix, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}