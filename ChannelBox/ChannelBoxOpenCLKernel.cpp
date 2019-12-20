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

const char *KernelSource = "\n" \
"float Luma(float R, float G, float B, int L); \n" \
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
"__kernel void k_gaussian(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float blur, int c) { \n" \
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
"p_Input += x * 4 + c; \n" \
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
"__kernel void k_transpose(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, __local float* buffer, int c) { \n" \
"int xIndex = get_global_id(0); \n" \
"int yIndex = get_global_id(1); \n" \
"if ((xIndex < p_Width) && (yIndex < p_Height)) \n" \
"{ \n" \
"buffer[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = p_Input[(yIndex * p_Width + xIndex)]; \n" \
"} \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"xIndex = get_group_id(1) * get_local_size(1) + get_local_id(0); \n" \
"yIndex = get_group_id(0) * get_local_size(0) + get_local_id(1); \n" \
"if((xIndex < p_Height) && (yIndex < p_Width)) \n" \
"{ \n" \
"p_Output[(yIndex * p_Height + xIndex) * 4 + c] = buffer[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)]; \n" \
"}} \n" \
"__kernel void k_garbageCore(__global float* p_Input, int p_Width, int p_Height, float p_Garbage, float p_Core) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = ((y * p_Width) + x) * 4; \n" \
"float A = p_Input[index + 3]; \n" \
"if (p_Core > 0.0f) { \n" \
"float CoreA = fmin(A * (1.0f + p_Core), 1.0f); \n" \
"CoreA = fmax(CoreA + (0.0f - p_Core * 3.0f) * (1.0f - CoreA), 0.0f); \n" \
"A = fmax(A, CoreA); \n" \
"} \n" \
"if (p_Garbage > 0.0f) { \n" \
"float GarA = fmax(A + (0.0f - p_Garbage * 3.0f) * (1.0f - A), 0.0f); \n" \
"GarA = fmin(GarA * (1.0f + p_Garbage* 3.0f), 1.0f); \n" \
"A = fmin(A, GarA); \n" \
"} \n" \
"p_Input[index + 3] = A; \n" \
"}} \n" \
"__kernel void k_channelBoxKernelA(__global const float* p_Input, __global float* p_Output, int p_Width, int p_Height,  \n" \
"int p_Choice, int p_ChannelBox, float p_ChannelSwap0, float p_ChannelSwap1, float p_ChannelSwap2,  \n" \
"float p_ChannelSwap3, float p_ChannelSwap4, float p_ChannelSwap5, float p_ChannelSwap6,  \n" \
"float p_ChannelSwap7, float p_ChannelSwap8, int p_LumaMath, int p_Preserve, float p_Mask) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float R = p_Input[index];		 \n" \
"float G = p_Input[index + 1]; \n" \
"float B = p_Input[index + 2]; \n" \
"float inLuma = Luma(p_Input[index], p_Input[index + 1], p_Input[index + 2], p_LumaMath); \n" \
"float red, green, blue; \n" \
"float mask = 1.0f; \n" \
"switch (p_Choice) \n" \
"{ \n" \
"case 0: \n" \
"{ \n" \
"float BR = B > R ? R : B;     \n" \
"float BG = B > G ? G : B;     \n" \
"float BGR = B > fmin(G, R) ? fmin(G, R) : B;     \n" \
"float BGRX = B > fmax(G, R) ? fmax(G, R) : B;     \n" \
"blue = p_ChannelBox == 0 ? BR : p_ChannelBox == 1 ? BG : p_ChannelBox == 2 ? BGR : p_ChannelBox == 3 ? BGRX : B; 													    \n" \
"float GR = G > R ? R : G;     \n" \
"float GB = G > B ? B : G;     \n" \
"float GBR = G > fmin(B, R) ? fmin(B, R) : G;     \n" \
"float GBRX = G > fmax(B, R) ? fmax(B, R) : G;     \n" \
"green = p_ChannelBox == 4 ? GR : p_ChannelBox == 5 ? GB : p_ChannelBox == 6 ? GBR : p_ChannelBox == 7 ? GBRX : G; 													    \n" \
"float RG = R > G ? G : R;     \n" \
"float RB = R > B ? B : R;     \n" \
"float RBG = R > fmin(B, G) ? fmin(B, G) : R;     \n" \
"float RBGX = R > fmax(B, G) ? fmax(B, G) : R;     \n" \
"red = p_ChannelBox == 8 ? RG : p_ChannelBox == 9 ? RB : p_ChannelBox == 10 ? RBG : p_ChannelBox == 11 ? RBGX : R; \n" \
"} \n" \
"break; \n" \
"case 1: \n" \
"{ \n" \
"red = R * (1.0f + p_ChannelSwap0 + p_ChannelSwap1 + p_ChannelSwap2) + G * (0.0f - p_ChannelSwap0 - (p_ChannelSwap2 / 2.0f)) + B * (0.0f - p_ChannelSwap1 - (p_ChannelSwap2 / 2.0f)); \n" \
"green = R * (0.0f - p_ChannelSwap3 - (p_ChannelSwap5 / 2.0f)) + G * (1.0f + p_ChannelSwap3 + p_ChannelSwap4 + p_ChannelSwap5) + B * (0.0f - p_ChannelSwap4 - (p_ChannelSwap5 / 2.0f)); \n" \
"blue = R * (0.0f - p_ChannelSwap6 - (p_ChannelSwap8 / 2.0f)) + G * (0.0f - p_ChannelSwap7 - (p_ChannelSwap8 / 2.0f)) + B * (1.0f + p_ChannelSwap6 + p_ChannelSwap7 + p_ChannelSwap8); \n" \
"} \n" \
"break; \n" \
"default:  \n" \
"red = R; \n" \
"green = G; \n" \
"blue = B; \n" \
"} \n" \
"if (p_Preserve == 1) { \n" \
"float outLuma = Luma(red, green, blue, p_LumaMath); \n" \
"red = red * (inLuma / outLuma); \n" \
"green = green * (inLuma / outLuma); \n" \
"blue = blue * (inLuma / outLuma); \n" \
"} \n" \
"if(p_Mask != 0.0f) { \n" \
"mask = p_Mask > 1.0f ? inLuma + (1.0f - p_Mask) * (1.0f - inLuma) : p_Mask >= 0.0f ? (inLuma >= p_Mask ? 1.0f :  \n" \
"inLuma / p_Mask) : p_Mask < -1.0f ? (1.0f - inLuma) + (p_Mask + 1.0f) * inLuma : inLuma <= (1.0f + p_Mask) ? 1.0f :  \n" \
"(1.0f - inLuma) / (1.0f - (p_Mask + 1.0f)); \n" \
"mask = mask > 1.0f ? 1.0f : mask < 0.0f ? 0.0f : mask; \n" \
"}																								    \n" \
"p_Output[index] = red; \n" \
"p_Output[index + 1] = green; \n" \
"p_Output[index + 2] = blue; \n" \
"p_Output[index + 3] = mask; \n" \
"}} \n" \
"__kernel void k_channelBoxKernelB(__global const float* p_Input, __global float* p_Output, int p_Width, int p_Height, int p_Display) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"p_Output[index] = p_Display == 1 ? p_Output[index + 3] : p_Output[index] * p_Output[index + 3] + p_Input[index] * (1.0f - p_Output[index + 3]); \n" \
"p_Output[index + 1] = p_Display == 1 ? p_Output[index + 3] : p_Output[index + 1] * p_Output[index + 3] + p_Input[index + 1] * (1.0f - p_Output[index + 3]); \n" \
"p_Output[index + 2] = p_Display == 1 ? p_Output[index + 3] : p_Output[index + 2] * p_Output[index + 3] + p_Input[index + 2] * (1.0f - p_Output[index + 3]); \n" \
"p_Output[index + 3] = 1.0f; \n" \
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


void CheckError(cl_int p_Error, const char* p_Msg)
{
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

void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Choice, 
int p_ChannelBox, float* p_ChannelSwap, int p_LumaMath, int* p_Switch, float* p_Mask)
{
szBuffBytes = p_Width * p_Height * sizeof(float);
p_Mask[1] *= 10.0f;
int alpha = 3;

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
}
else
{
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

cl_kernel Gausfilter = NULL;
cl_kernel Transpose = NULL;
cl_kernel Core = NULL;
cl_kernel KernelA = NULL;
cl_kernel KernelB = NULL;

Gausfilter = clCreateKernel(program, "k_gaussian", &error);
CheckError(error, "Unable to create kernel");

Transpose = clCreateKernel(program, "k_transpose", &error);
CheckError(error, "Unable to create kernel");

Core = clCreateKernel(program, "k_garbageCore", &error);
CheckError(error, "Unable to create kernel");

KernelA = clCreateKernel(program, "k_channelBoxKernelA", &error);
CheckError(error, "Unable to create kernel");

KernelB = clCreateKernel(program, "k_channelBoxKernelB", &error);
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
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Mask[1]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &alpha);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Transpose, 0, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Transpose, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Transpose, 4, sizeof(float) * iTransposeBlockDim * (iTransposeBlockDim + 1), NULL);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &alpha);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(KernelA, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(KernelA, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(KernelA, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(KernelA, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(KernelA, 4, sizeof(int), &p_Choice);
error |= clSetKernelArg(KernelA, 5, sizeof(int), &p_ChannelBox);
error |= clSetKernelArg(KernelA, 6, sizeof(float), &p_ChannelSwap[0]);
error |= clSetKernelArg(KernelA, 7, sizeof(float), &p_ChannelSwap[1]);
error |= clSetKernelArg(KernelA, 8, sizeof(float), &p_ChannelSwap[2]);
error |= clSetKernelArg(KernelA, 9, sizeof(float), &p_ChannelSwap[3]);
error |= clSetKernelArg(KernelA, 10, sizeof(float), &p_ChannelSwap[4]);
error |= clSetKernelArg(KernelA, 11, sizeof(float), &p_ChannelSwap[5]);
error |= clSetKernelArg(KernelA, 12, sizeof(float), &p_ChannelSwap[6]);
error |= clSetKernelArg(KernelA, 13, sizeof(float), &p_ChannelSwap[7]);
error |= clSetKernelArg(KernelA, 14, sizeof(float), &p_ChannelSwap[8]);
error |= clSetKernelArg(KernelA, 15, sizeof(int), &p_LumaMath);
error |= clSetKernelArg(KernelA, 16, sizeof(int), &p_Switch[0]);
error |= clSetKernelArg(KernelA, 17, sizeof(float), &p_Mask[0]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, KernelA, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Mask[1] > 0.0f) {
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

if (p_Mask[2] > 0.0f || p_Mask[3] > 0.0f) {
error = clSetKernelArg(Core, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Core, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(Core, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Core, 3, sizeof(float), &p_Mask[2]);
error |= clSetKernelArg(Core, 4, sizeof(float), &p_Mask[3]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, Core, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}

error = clSetKernelArg(KernelB, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(KernelB, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(KernelB, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(KernelB, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(KernelB, 4, sizeof(int), &p_Switch[1]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, KernelB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

clReleaseMemObject(tempBuffer);
}
