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
size_t erodeLocalWorkSize[2], dilateLocalWorkSize[2];
size_t erodeGlobalWorkSizeA[2], erodeGlobalWorkSizeB[2];
size_t dilateGlobalWorkSizeA[2], dilateGlobalWorkSizeB[2];

cl_mem tempBuffer;
size_t szBuffBytes;
cl_int error;

const char *KernelSource = \
"float Mix( float A, float B, float mix) { \n" \
"float C = A * (1.0f - mix) + B * mix; \n" \
"return C; \n" \
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
"__kernel void k_garbageCore( __global float* p_Input, int p_Width, int p_Height, float p_Garbage, float p_Core) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
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
"__kernel void k_erode( __global float* p_Input, __global float* p_Output,  \n" \
"int radio, int width, int height, int tile_w, int tile_h, __local float *smem) { \n" \
"int tx = get_local_id(0); \n" \
"int ty = get_local_id(1); \n" \
"int bx = get_group_id(0); \n" \
"int by = get_group_id(1); \n" \
"int x = bx * tile_w + tx - radio; \n" \
"int y = by * tile_h + ty; \n" \
"smem[ty * get_local_size(0) + tx] = 1.0f; \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"if (x < 0 || x >= width || y >= height) { \n" \
"return; \n" \
"} \n" \
"smem[ty * get_local_size(0) + tx] = p_Input[(y * width + x) * 4 + 3]; \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) { \n" \
"return; \n" \
"} \n" \
"__local float* smem_thread = &smem[ty * get_local_size(0) + tx - radio]; \n" \
"float val = smem_thread[0]; \n" \
"for (int xx = 1; xx <= 2 * radio; xx++) { \n" \
"val = fmin(val, smem_thread[xx]); \n" \
"} \n" \
"p_Output[y * width + x] = val; \n" \
"} \n" \
"__kernel void k_dilate( __global float* p_Input, __global float* p_Output,  \n" \
"int radio, int width, int height, int tile_w, int tile_h, __local float *smem) { \n" \
"int tx = get_local_id(0); \n" \
"int ty = get_local_id(1); \n" \
"int bx = get_group_id(0); \n" \
"int by = get_group_id(1); \n" \
"int x = bx * tile_w + tx - radio; \n" \
"int y = by * tile_h + ty; \n" \
"smem[ty * get_local_size(0) + tx] = 0.0f; \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"if (x < 0 || x >= width || y >= height) { \n" \
"return; \n" \
"} \n" \
"smem[ty * get_local_size(0) + tx] = p_Input[(y * width + x) * 4 + 3]; \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"if (x < bx * tile_w || x >= (bx + 1) * tile_w) { \n" \
"return; \n" \
"} \n" \
"__local float* smem_thread = &smem[ty * get_local_size(0) + tx - radio]; \n" \
"float val = smem_thread[0]; \n" \
"for (int xx = 1; xx <= 2 * radio; xx++) { \n" \
"val = fmax(val, smem_thread[xx]); \n" \
"} \n" \
"p_Output[y * width + x] = val; \n" \
"} \n" \
"__kernel void k_replaceKernelA( __global const float* p_Input, __global float* p_Output, int p_Width, int p_Height, float hueRangeA, float hueRangeB,  \n" \
"float hueRangeWithRollOffA, float hueRangeWithRollOffB, float satRangeA, float satRangeB, float satRolloff, float valRangeA,  \n" \
"float valRangeB, float valRolloff, int OutputAlpha, int DisplayAlpha, float p_Black, float p_White) { \n" \
"const int x = get_global_id(0);                                     \n" \
"const int y = get_global_id(1);                                     \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float hcoeff, scoeff, vcoeff; \n" \
"float r, g, b, h, s, v; \n" \
"r = p_Input[index]; \n" \
"g = p_Input[index + 1]; \n" \
"b = p_Input[index + 2]; \n" \
"float min = fmin(fmin(r, g), b); \n" \
"float max = fmax(fmax(r, g), b); \n" \
"v = max; \n" \
"float delta = max - min; \n" \
"if (max != 0.0f) { \n" \
"s = delta / max; \n" \
"} else { \n" \
"s = 0.0f; \n" \
"h = 0.0f; \n" \
"} \n" \
"if (delta == 0.0f) { \n" \
"h = 0.0f; \n" \
"} else if (r == max) { \n" \
"h = (g - b) / delta; \n" \
"} else if (g == max) { \n" \
"h = 2.0f + (b - r) / delta; \n" \
"} else { \n" \
"h = 4.0f + (r - g) / delta; \n" \
"} \n" \
"h *= 1.0f / 6.0f; \n" \
"if (h < 0.0f) { \n" \
"h += 1.0f; \n" \
"} \n" \
"h *= 360.0f; \n" \
"float h0 = hueRangeA; \n" \
"float h1 = hueRangeB; \n" \
"float h0mrolloff = hueRangeWithRollOffA; \n" \
"float h1prolloff = hueRangeWithRollOffB; \n" \
"if ( ( h1 < h0 && (h <= h1 || h0 <= h) ) || (h0 <= h && h <= h1) ) { \n" \
"hcoeff = 1.0f; \n" \
"} else { \n" \
"float c0 = 0.0f; \n" \
"float c1 = 0.0f; \n" \
"if ( ( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0) ) { \n" \
"c0 = h0 == (h0mrolloff + 360.0f) || h0 == h0mrolloff ? 1.0f : !(( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0)) ? 0.0f :  \n" \
"((h < h0mrolloff ? h + 360.0f : h) - h0mrolloff) / ((h0 < h0mrolloff ? h0 + 360.0f : h0) - h0mrolloff);		 \n" \
"} \n" \
"if ( ( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff) ) { \n" \
"c1 = !(( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff)) ? 0.0f : h1prolloff == h1 ? 1.0f : \n" \
"((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - (h < h1 ? h + 360.0f : h)) / ((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - h1);	 \n" \
"} \n" \
"hcoeff = fmax(c0, c1); \n" \
"} \n" \
"float s0 = satRangeA; \n" \
"float s1 = satRangeB; \n" \
"float s0mrolloff = s0 - satRolloff; \n" \
"float s1prolloff = s1 + satRolloff; \n" \
"if ( s0 <= s && s <= s1 ) { \n" \
"scoeff = 1.0f; \n" \
"} else if ( s0mrolloff <= s && s <= s0 ) { \n" \
"scoeff = (s - s0mrolloff) / satRolloff; \n" \
"} else if ( s1 <= s && s <= s1prolloff ) { \n" \
"scoeff = (s1prolloff - s) / satRolloff; \n" \
"} else { \n" \
"scoeff = 0.0f; \n" \
"} \n" \
"float v0 = valRangeA; \n" \
"float v1 = valRangeB; \n" \
"float v0mrolloff = v0 - valRolloff; \n" \
"float v1prolloff = v1 + valRolloff; \n" \
"if ( (v0 <= v) && (v <= v1) ) { \n" \
"vcoeff = 1.0f; \n" \
"} else if ( v0mrolloff <= v && v <= v0 ) { \n" \
"vcoeff = (v - v0mrolloff) / valRolloff; \n" \
"} else if ( v1 <= v && v <= v1prolloff ) { \n" \
"vcoeff = (v1prolloff - v) / valRolloff; \n" \
"} else { \n" \
"vcoeff = 0.0f; \n" \
"} \n" \
"float coeff = fmin(fmin(hcoeff, scoeff), vcoeff); \n" \
"float A = OutputAlpha == 0 ? 1.0f : OutputAlpha == 1 ? hcoeff : OutputAlpha == 2 ? scoeff : \n" \
"OutputAlpha == 3 ? vcoeff : OutputAlpha == 4 ? fmin(hcoeff, scoeff) : OutputAlpha == 5 ?  \n" \
"fmin(hcoeff, vcoeff) : OutputAlpha == 6 ? fmin(scoeff, vcoeff) : fmin(fmin(hcoeff, scoeff), vcoeff); \n" \
"if (DisplayAlpha == 0) \n" \
"A = coeff; \n" \
"if (p_Black > 0.0f) \n" \
"A = fmax(A - (p_Black * 4.0f) * (1.0f - A), 0.0f); \n" \
"if (p_White > 0.0f) \n" \
"A = fmin(A * (1.0f + p_White * 4.0f), 1.0f); \n" \
"p_Output[index] = h; \n" \
"p_Output[index + 1] = s; \n" \
"p_Output[index + 2] = v; \n" \
"p_Output[index + 3] = A; \n" \
"}} \n" \
"__kernel void k_replaceKernelB( __global const float* p_Input, __global float* p_Output, int p_Width, int p_Height, float hueRotation,  \n" \
"float hueRotationGain, float hueMean, float satRangeA, float satRangeB, float satAdjust, float satAdjustGain,  \n" \
"float valRangeA, float valRangeB, float valAdjust, float valAdjustGain, int DisplayAlpha, float mix) { \n" \
"const int x = get_global_id(0);                                     \n" \
"const int y = get_global_id(1);                                     \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float h, s, v, R, G, B, coeff; \n" \
"h = p_Output[index]; \n" \
"s = p_Output[index + 1]; \n" \
"v = p_Output[index + 2]; \n" \
"coeff = p_Output[index + 3]; \n" \
"float s0 = satRangeA; \n" \
"float s1 = satRangeB; \n" \
"float v0 = valRangeA; \n" \
"float v1 = valRangeB; \n" \
"float H = (h - hueMean + 180.0f) - (int)(floor((h - hueMean + 180.0f) / 360.0f) * 360.0f) - 180.0f; \n" \
"h += coeff * ( hueRotation + (hueRotationGain - 1.0f) * H ); \n" \
"s += coeff * ( satAdjust + (satAdjustGain - 1.0f) * (s - (s0 + s1) / 2.0f) ); \n" \
"if (s < 0.0f) { \n" \
"s = 0.0f; \n" \
"} \n" \
"v += coeff * ( valAdjust + (valAdjustGain - 1.0f) * (v - (v0 + v1) / 2.0f) ); \n" \
"h *= 1.0f / 360.0f; \n" \
"if (s == 0.0f) \n" \
"R = G = B = v; \n" \
"h *= 6.0f; \n" \
"int i = floor(h); \n" \
"float f = h - i; \n" \
"i = (i >= 0) ? (i % 6) : (i % 6) + 6; \n" \
"float p = v * ( 1.0f - s ); \n" \
"float q = v * ( 1.0f - s * f ); \n" \
"float t = v * ( 1.0f - s * ( 1.0f - f )); \n" \
"if (i == 0){ \n" \
"R = v; \n" \
"G = t; \n" \
"B = p;} \n" \
"else if (i == 1){ \n" \
"R = q; \n" \
"G = v; \n" \
"B = p;} \n" \
"else if (i == 2){ \n" \
"R = p; \n" \
"G = v; \n" \
"B = t;} \n" \
"else if (i == 3){ \n" \
"R = p; \n" \
"G = q; \n" \
"B = v;} \n" \
"else if (i == 4){ \n" \
"R = t; \n" \
"G = p; \n" \
"B = v;} \n" \
"else{ \n" \
"R = v; \n" \
"G = p; \n" \
"B = q; \n" \
"} \n" \
"p_Output[index] = DisplayAlpha == 1 ? coeff : Mix(R, p_Input[index], mix); \n" \
"p_Output[index + 1] = DisplayAlpha == 1 ? coeff : Mix(G, p_Input[index + 1], mix); \n" \
"p_Output[index + 2] = DisplayAlpha == 1 ? coeff : Mix(B, p_Input[index + 2], mix); \n" \
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

void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Hue, 
float* p_Sat, float* p_Val, int OutputAlpha, int DisplayAlpha, float mix, float* p_Blur)
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

cl_kernel ReplaceKernelA = NULL;
cl_kernel ReplaceKernelB = NULL;
cl_kernel Gausfilter = NULL;
cl_kernel Transpose = NULL;
cl_kernel Core = NULL;
cl_kernel Erode = NULL;
cl_kernel Dilate = NULL;

ReplaceKernelA = clCreateKernel(program, "k_replaceKernelA", &error);
CheckError(error, "Unable to create kernel");

ReplaceKernelB = clCreateKernel(program, "k_replaceKernelB", &error);
CheckError(error, "Unable to create kernel");

Gausfilter = clCreateKernel(program, "k_gaussian", &error);
CheckError(error, "Unable to create kernel");

Transpose = clCreateKernel(program, "k_transpose", &error);
CheckError(error, "Unable to create kernel");

Core = clCreateKernel(program, "k_garbageCore", &error);
CheckError(error, "Unable to create kernel");

Erode = clCreateKernel(program, "k_erode", &error);
CheckError(error, "Unable to create kernel");

Dilate = clCreateKernel(program, "k_dilate", &error);
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
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[2]);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Transpose, 0, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Transpose, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Transpose, 4, sizeof(float) * iTransposeBlockDim * (iTransposeBlockDim + 1), NULL);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(ReplaceKernelA, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(ReplaceKernelA, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(ReplaceKernelA, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(ReplaceKernelA, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(ReplaceKernelA, 4, sizeof(float), &p_Hue[0]);
error |= clSetKernelArg(ReplaceKernelA, 5, sizeof(float), &p_Hue[1]);
error |= clSetKernelArg(ReplaceKernelA, 6, sizeof(float), &p_Hue[2]);
error |= clSetKernelArg(ReplaceKernelA, 7, sizeof(float), &p_Hue[3]);
error |= clSetKernelArg(ReplaceKernelA, 8, sizeof(float), &p_Sat[0]);
error |= clSetKernelArg(ReplaceKernelA, 9, sizeof(float), &p_Sat[1]);
error |= clSetKernelArg(ReplaceKernelA, 10, sizeof(float), &p_Sat[4]);
error |= clSetKernelArg(ReplaceKernelA, 11, sizeof(float), &p_Val[0]);
error |= clSetKernelArg(ReplaceKernelA, 12, sizeof(float), &p_Val[1]);
error |= clSetKernelArg(ReplaceKernelA, 13, sizeof(float), &p_Val[4]);
error |= clSetKernelArg(ReplaceKernelA, 14, sizeof(int), &OutputAlpha);
error |= clSetKernelArg(ReplaceKernelA, 15, sizeof(int), &DisplayAlpha);
error |= clSetKernelArg(ReplaceKernelA, 16, sizeof(float), &p_Blur[0]);
error |= clSetKernelArg(ReplaceKernelA, 17, sizeof(float), &p_Blur[1]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, ReplaceKernelA, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Blur[2] > 0.0f) {
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

if (p_Blur[3] > 0.0f || p_Blur[4] > 0.0f) {
error = clSetKernelArg(Core, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Core, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(Core, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Core, 3, sizeof(float), &p_Blur[3]);
error |= clSetKernelArg(Core, 4, sizeof(float), &p_Blur[4]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, Core, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}

int radE = ceil(p_Blur[5] * 15.0f);
int radD = ceil(p_Blur[6] * 15.0f);
int tile_w = 640;
int tile_h = 1;
erodeLocalWorkSize[0] = tile_w + (2 * radE);
erodeLocalWorkSize[1] = tile_h;
dilateLocalWorkSize[0] = tile_w + (2 * radD);
dilateLocalWorkSize[1] = tile_h;

erodeGlobalWorkSizeA[0] = shrRoundUp(erodeLocalWorkSize[0], p_Width);
erodeGlobalWorkSizeA[1] = shrRoundUp(erodeLocalWorkSize[1], p_Height);
erodeGlobalWorkSizeB[0] = shrRoundUp(erodeLocalWorkSize[0], p_Height);
erodeGlobalWorkSizeB[1] = shrRoundUp(erodeLocalWorkSize[1], p_Width);

dilateGlobalWorkSizeA[0] = shrRoundUp(dilateLocalWorkSize[0], p_Width);
dilateGlobalWorkSizeA[1] = shrRoundUp(dilateLocalWorkSize[1], p_Height);
dilateGlobalWorkSizeB[0] = shrRoundUp(dilateLocalWorkSize[0], p_Height);
dilateGlobalWorkSizeB[1] = shrRoundUp(dilateLocalWorkSize[1], p_Width);

error = clSetKernelArg(Erode, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Erode, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Erode, 2, sizeof(float), &radE);
error |= clSetKernelArg(Erode, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Erode, 4, sizeof(int), &p_Height);
error |= clSetKernelArg(Erode, 5, sizeof(int), &tile_w);
error |= clSetKernelArg(Erode, 6, sizeof(int), &tile_h);
error |= clSetKernelArg(Erode, 7, sizeof(float) * erodeLocalWorkSize[1] * erodeLocalWorkSize[0], NULL);
CheckError(error, "Unable to set kernel arguments");
error = clSetKernelArg(Dilate, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Dilate, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Dilate, 2, sizeof(float), &radD);
error |= clSetKernelArg(Dilate, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Dilate, 4, sizeof(int), &p_Height);
error |= clSetKernelArg(Dilate, 5, sizeof(int), &tile_w);
error |= clSetKernelArg(Dilate, 6, sizeof(int), &tile_h);
error |= clSetKernelArg(Dilate, 7, sizeof(float) * dilateLocalWorkSize[1] * dilateLocalWorkSize[0], NULL);
CheckError(error, "Unable to set kernel arguments");

if (p_Blur[5] > 0.0f) {
clEnqueueNDRangeKernel(cmdQ, Erode, 2, NULL, erodeGlobalWorkSizeA, erodeLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Erode, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Erode, 4, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Erode, 2, NULL, erodeGlobalWorkSizeB, erodeLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[6] > 0.0f) {
clEnqueueNDRangeKernel(cmdQ, Dilate, 2, NULL, dilateGlobalWorkSizeA, dilateLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Dilate, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Dilate, 4, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Dilate, 2, NULL, dilateGlobalWorkSizeB, dilateLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

error = clSetKernelArg(ReplaceKernelB, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(ReplaceKernelB, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(ReplaceKernelB, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(ReplaceKernelB, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(ReplaceKernelB, 4, sizeof(float), &p_Hue[4]);
error |= clSetKernelArg(ReplaceKernelB, 5, sizeof(float), &p_Hue[5]);
error |= clSetKernelArg(ReplaceKernelB, 6, sizeof(float), &p_Hue[6]);
error |= clSetKernelArg(ReplaceKernelB, 7, sizeof(float), &p_Sat[0]);
error |= clSetKernelArg(ReplaceKernelB, 8, sizeof(float), &p_Sat[1]);
error |= clSetKernelArg(ReplaceKernelB, 9, sizeof(float), &p_Sat[2]);
error |= clSetKernelArg(ReplaceKernelB, 10, sizeof(float), &p_Sat[3]);
error |= clSetKernelArg(ReplaceKernelB, 11, sizeof(float), &p_Val[0]);
error |= clSetKernelArg(ReplaceKernelB, 12, sizeof(float), &p_Val[1]);
error |= clSetKernelArg(ReplaceKernelB, 13, sizeof(float), &p_Val[2]);
error |= clSetKernelArg(ReplaceKernelB, 14, sizeof(float), &p_Val[3]);
error |= clSetKernelArg(ReplaceKernelB, 15, sizeof(int), &DisplayAlpha);
error |= clSetKernelArg(ReplaceKernelB, 16, sizeof(float), &mix);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, ReplaceKernelB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

clReleaseMemObject(tempBuffer);
}