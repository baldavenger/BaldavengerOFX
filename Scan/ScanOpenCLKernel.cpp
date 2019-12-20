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

#define iTransposeBlockDim 32

size_t localWorkSize[2], globalWorkSize[2];
size_t gausLocalWorkSize[2], gausGlobalWorkSizeA[2], gausGlobalWorkSizeB[2];
size_t TransLocalWorkSize[2], TransGlobalWorkSizeA[2], TransGlobalWorkSizeB[2];

cl_mem tempBuffer;
size_t szBuffBytes;
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
"float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 :  \n" \
"L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax; \n" \
"return Lu; \n" \
"} \n" \
"__kernel void k_gaussian(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float blur) { \n" \
"float nsigma = blur < 0.1f ? 0.1f : blur; \n" \
"float alpha = 1.695f / nsigma; \n" \
"float ema = exp(-alpha); \n" \
"float ema2 = exp(-2.0f * alpha); \n" \
"float b1 = -2.0f * ema; \n" \
"float b2 = ema2; \n" \
"float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, coefp = 0.0f, coefn = 0.0f; \n" \
"float k = (1.0f - ema) * (1.0f - ema) / (1.0f + 2.0f * alpha * ema - ema2); \n" \
"a0 = k; \n" \
"a1 = k * (alpha - 1.0f) * ema; \n" \
"a2 = k * (alpha + 1.0f) * ema; \n" \
"a3 = -k * ema2; \n" \
"coefp = (a0 + a1) / (1.0f + b1 + b2); \n" \
"coefn = (a2 + a3) / (1.0f + b1 + b2); \n" \
"int x = get_group_id(0) * get_local_size(0) + get_local_id(0); \n" \
"if (x >= p_Width) return; \n" \
"p_Input += x * 4 + 3; \n" \
"p_Output += x; \n" \
"float xp, yp, yb; \n" \
"xp = *p_Input; \n" \
"yb = coefp * xp; \n" \
"yp = yb; \n" \
"for (int y = 0; y < p_Height; y++) { \n" \
"float xc = *p_Input; \n" \
"float yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb; \n" \
"*p_Output = yc; \n" \
"p_Input += p_Width * 4; \n" \
"p_Output += p_Width; \n" \
"xp = xc; \n" \
"yb = yp; \n" \
"yp = yc; \n" \
"} \n" \
"p_Input -= p_Width * 4; \n" \
"p_Output -= p_Width; \n" \
"float xn, xa, yn, ya; \n" \
"xn = xa = *p_Input; \n" \
"yn = coefn * xn; \n" \
"ya = yn; \n" \
"for (int y = p_Height - 1; y >= 0; y--) { \n" \
"float xc = *p_Input; \n" \
"float yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya; \n" \
"xa = xn; \n" \
"xn = xc; \n" \
"ya = yn; \n" \
"yn = yc; \n" \
"*p_Output = *p_Output + yc; \n" \
"p_Input -= p_Width * 4; \n" \
"p_Output -= p_Width; \n" \
"}} \n" \
"__kernel void k_transpose( __global float* p_Input, __global float* p_Output, int p_Width, int p_Height, __local float* buffer) { \n" \
"int xIndex = get_global_id(0); \n" \
"int yIndex = get_global_id(1); \n" \
"if (xIndex < p_Width && yIndex < p_Height) { \n" \
"buffer[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = p_Input[(yIndex * p_Width + xIndex)]; \n" \
"} \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"xIndex = get_group_id(1) * get_local_size(1) + get_local_id(0); \n" \
"yIndex = get_group_id(0) * get_local_size(0) + get_local_id(1); \n" \
"if (xIndex < p_Height && yIndex < p_Width) { \n" \
"p_Output[(yIndex * p_Height + xIndex) * 4 + 3] = buffer[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)]; \n" \
"}} \n" \
"__kernel void k_alphaKernel( __global const float* p_Input, __global float* p_Output, int p_Width, int p_Height, float lumaLimit, int LumaMath) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float luma = Luma(p_Input[index], p_Input[index + 1], p_Input[index + 2], LumaMath); \n" \
"float alpha = lumaLimit > 1.0f ? luma + (1.0f - lumaLimit) * (1.0f - luma) : lumaLimit >= 0.0f ? (luma >= lumaLimit ?  \n" \
"1.0f : luma / lumaLimit) : lumaLimit < -1.0f ? (1.0f - luma) + (lumaLimit + 1.0f) * luma : luma <= (1.0f + lumaLimit) ? 1.0f :  \n" \
"(1.0f - luma) / (1.0f - (lumaLimit + 1.0f)); \n" \
"float Alpha = alpha > 1.0f ? 1.0f : alpha < 0.0f ? 0.0f : alpha; \n" \
"p_Output[index + 3] = Alpha; \n" \
"}} \n" \
"__kernel void k_scanKernel( __global const float* p_Input, __global float* p_Output, int p_Width, int p_Height, float balGainR,  \n" \
"float balGainB, float balOffsetR, float balOffsetB,float balLiftR, float balLiftB, float lumaBalance,  \n" \
"int GainBalance, int OffsetBalance, int WhiteBalance, int PreserveLuma, int DisplayAlpha) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float Alpha = p_Output[index + 3]; \n" \
"float BalR = GainBalance == 1 ? p_Input[index] * balGainR : OffsetBalance == 1 ? p_Input[index] + balOffsetR : p_Input[index] + (balLiftR * (1.0f - p_Input[index])); \n" \
"float BalB = GainBalance == 1 ? p_Input[index + 2] * balGainB : OffsetBalance == 1 ? p_Input[index + 2] + balOffsetB : p_Input[index + 2] + (balLiftB * (1.0f - p_Input[index + 2])); \n" \
"float Red = WhiteBalance == 1 ? ( PreserveLuma == 1 ? BalR * lumaBalance : BalR) : p_Input[index ]; \n" \
"float Green = WhiteBalance == 1 && PreserveLuma == 1 ? p_Input[index + 1] * lumaBalance : p_Input[index + 1]; \n" \
"float Blue = WhiteBalance == 1 ? ( PreserveLuma == 1 ? BalB * lumaBalance : BalB) : p_Input[index + 2]; \n" \
"p_Output[index] = DisplayAlpha == 1 ? Alpha : Red * Alpha + p_Input[index] * (1.0f - Alpha); \n" \
"p_Output[index + 1] = DisplayAlpha == 1 ? Alpha : Green * Alpha + p_Input[index + 1] * (1.0f - Alpha); \n" \
"p_Output[index + 2] = DisplayAlpha == 1 ? Alpha : Blue * Alpha + p_Input[index + 2] * (1.0f - Alpha); \n" \
"p_Output[index + 3] = 1.0f; \n" \
"}} \n" \
"__kernel void k_markerKernel( __global float* p_Input, int p_Width, int p_Height, int LumaMaxX,  \n" \
"int LumaMaxY, int LumaMinX, int LumaMinY, int Radius, int DisplayMax, int DisplayMin) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"bool MIN = DisplayMin == 1 && (x >= LumaMinX - Radius && x <= LumaMinX + Radius && y >= LumaMinY - Radius && y <= LumaMinY + Radius); \n" \
"bool MAX = DisplayMax == 1 && (x >= LumaMaxX - Radius && x <= LumaMaxX + Radius && y >= LumaMaxY - Radius && y <= LumaMaxY + Radius); \n" \
"p_Input[index] = MIN ? 0.0f : MAX ? 1.0f : p_Input[index]; \n" \
"p_Input[index + 1] = MIN || MAX ? 1.0f : p_Input[index + 1]; \n" \
"p_Input[index + 2] = MAX ? 0.0f : MIN ? 1.0f : p_Input[index + 2]; \n" \
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

void CheckError(cl_int p_Error, const char* p_Msg) {
if (p_Error != CL_SUCCESS) {
fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
}}

int clDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

int shrRoundUp(size_t localWorkSize, int numItems) {
int result = localWorkSize;
while (result < numItems)
result += localWorkSize;
return result;
}

void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Gain, 
float* p_Offset, float* p_Lift, float p_LumaBalance, float p_LumaLimit, float p_Blur, int* p_Switch, 
int p_LumaMath, int* p_LumaMaxXY, int* p_LumaMinXY, int* p_DisplayXY, int Radius)
{	
szBuffBytes = p_Width * p_Height * sizeof(float);

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

tempBuffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &error);
CheckError(error, "Unable to create buffer");

cl_kernel AlphaKernel = NULL;
cl_kernel ScanKernel = NULL;
cl_kernel MarkerKernel = NULL;
cl_kernel Gausfilter = NULL;
cl_kernel Transpose = NULL;

AlphaKernel = clCreateKernel(program, "k_alphaKernel", &error);
CheckError(error, "Unable to create kernel");

ScanKernel = clCreateKernel(program, "k_scanKernel", &error);
CheckError(error, "Unable to create kernel");

MarkerKernel = clCreateKernel(program, "k_markerKernel", &error);
CheckError(error, "Unable to create kernel");

Gausfilter = clCreateKernel(program, "k_gaussian", &error);
CheckError(error, "Unable to create kernel");

Transpose = clCreateKernel(program, "k_transpose", &error);
CheckError(error, "Unable to create kernel");

locker.Unlock();

localWorkSize[0] = gausLocalWorkSize[0] = 128;
localWorkSize[1] = gausLocalWorkSize[1] = gausGlobalWorkSizeA[1] = gausGlobalWorkSizeB[1] = 1;

globalWorkSize[0] = localWorkSize[0] * clDivUp(p_Width, localWorkSize[0]);
globalWorkSize[1] = p_Height;

gausGlobalWorkSizeA[0] = gausLocalWorkSize[0] * clDivUp(p_Width, gausLocalWorkSize[0]);
gausGlobalWorkSizeB[0] = gausLocalWorkSize[0] * clDivUp(p_Height, gausLocalWorkSize[0]);

TransLocalWorkSize[0] = TransLocalWorkSize[1] = iTransposeBlockDim;
TransGlobalWorkSizeA[0] = shrRoundUp(TransLocalWorkSize[0], p_Width);
TransGlobalWorkSizeA[1] = shrRoundUp(TransLocalWorkSize[1], p_Height);
TransGlobalWorkSizeB[0] = TransGlobalWorkSizeA[1];
TransGlobalWorkSizeB[1] = TransGlobalWorkSizeA[0];

error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Transpose, 0, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Transpose, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Transpose, 4, sizeof(float) * iTransposeBlockDim * (iTransposeBlockDim + 1), NULL);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(AlphaKernel, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(AlphaKernel, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(AlphaKernel, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(AlphaKernel, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(AlphaKernel, 4, sizeof(float), &p_LumaLimit);
error |= clSetKernelArg(AlphaKernel, 5, sizeof(int), &p_LumaMath);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, AlphaKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Blur > 0.0f) {
p_Blur *= 10.0f;
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

error = clSetKernelArg(ScanKernel, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(ScanKernel, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(ScanKernel, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(ScanKernel, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(ScanKernel, 4, sizeof(float), &p_Gain[0]);
error |= clSetKernelArg(ScanKernel, 5, sizeof(float), &p_Gain[1]);
error |= clSetKernelArg(ScanKernel, 6, sizeof(float), &p_Offset[0]);
error |= clSetKernelArg(ScanKernel, 7, sizeof(float), &p_Offset[1]);
error |= clSetKernelArg(ScanKernel, 8, sizeof(float), &p_Lift[0]);
error |= clSetKernelArg(ScanKernel, 9, sizeof(float), &p_Lift[1]);
error |= clSetKernelArg(ScanKernel, 10, sizeof(float), &p_LumaBalance);
error |= clSetKernelArg(ScanKernel, 11, sizeof(int), &p_Switch[0]);
error |= clSetKernelArg(ScanKernel, 12, sizeof(int), &p_Switch[1]);
error |= clSetKernelArg(ScanKernel, 13, sizeof(int), &p_Switch[2]);
error |= clSetKernelArg(ScanKernel, 14, sizeof(int), &p_Switch[3]);
error |= clSetKernelArg(ScanKernel, 15, sizeof(int), &p_Switch[4]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, ScanKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_DisplayXY[0] == 1 || p_DisplayXY[1] == 1) {
error = clSetKernelArg(MarkerKernel, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(MarkerKernel, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(MarkerKernel, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(MarkerKernel, 3, sizeof(int), &p_LumaMaxXY[0]);
error |= clSetKernelArg(MarkerKernel, 4, sizeof(int), &p_LumaMaxXY[1]);
error |= clSetKernelArg(MarkerKernel, 5, sizeof(int), &p_LumaMinXY[0]);
error |= clSetKernelArg(MarkerKernel, 6, sizeof(int), &p_LumaMinXY[1]);
error |= clSetKernelArg(MarkerKernel, 7, sizeof(int), &Radius);
error |= clSetKernelArg(MarkerKernel, 8, sizeof(int), &p_DisplayXY[0]);
error |= clSetKernelArg(MarkerKernel, 9, sizeof(int), &p_DisplayXY[1]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, MarkerKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}

clReleaseMemObject(tempBuffer);
}
