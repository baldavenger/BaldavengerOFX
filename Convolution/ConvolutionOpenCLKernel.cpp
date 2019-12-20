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
size_t erodeLocalWorkSize[2], erodeGlobalWorkSizeA[2], erodeGlobalWorkSizeB[2];

cl_mem tempBuffer;
size_t szBuffBytes;
cl_int error;

const char *KernelSource = \
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
"__kernel void k_boxfilter(__global float *id, __global float *od, int w, int h, int r, int c) \n" \
"{ \n" \
"id += c; \n" \
"int x = get_group_id(0) * get_local_size(0) + get_local_id(0); \n" \
"if (x >= w) return; \n" \
"float scale = 1.0f / (float)((r << 1) + 1); \n" \
"float t; \n" \
"t = id[x * 4] * r; \n" \
"for (int y = 0; y < (r + 1); y++) { \n" \
"t += id[(y * w + x) * 4]; \n" \
"} \n" \
"od[x] = t * scale; \n" \
"for (int y = 1; y < (r + 1); y++) { \n" \
"t += id[((y + r) * w + x) * 4]; \n" \
"t -= id[x * 4]; \n" \
"od[y * w + x] = t * scale; \n" \
"} \n" \
"for (int y = (r + 1); y < (h - r); y++) { \n" \
"t += id[((y + r) * w + x) * 4]; \n" \
"t -= id[(((y - r) * w + x) - w) * 4]; \n" \
"od[y * w + x] = t * scale; \n" \
"} \n" \
"for (int y = h - r; y < h; y++) { \n" \
"t += id[((h-1) * w + x) * 4]; \n" \
"t -= id[(((y - r) * w + x) - w) * 4]; \n" \
"od[y * w + x] = t * scale; \n" \
"}} \n" \
"__kernel void k_simpleRecursive(__global float *id, __global float *od, int w, int h, float blur, int c) \n" \
"{ \n" \
"int x = get_group_id(0) * get_local_size(0) + get_local_id(0); \n" \
"const float nsigma = blur < 0.1f ? 0.1f : blur; \n" \
"float alpha = 1.695f / nsigma; \n" \
"float ema = exp(-alpha); \n" \
"if (x >= w) return; \n" \
"id += x * 4 + c; \n" \
"od += x; \n" \
"float yp = *id; \n" \
"for (int y = 0; y < h; y++) { \n" \
"float xc = *id; \n" \
"float yc = xc + ema * (yp - xc); \n" \
"*od = yc; \n" \
"id += w * 4; \n" \
"od += w; \n" \
"yp = yc; \n" \
"} \n" \
"id -= w * 4; \n" \
"od -= w; \n" \
"yp = *id; \n" \
"for (int y = h - 1; y >= 0; y--) { \n" \
"float xc = *id; \n" \
"float yc = xc + ema * (yp - xc); \n" \
"*od = (*od + yc) * 0.5f; \n" \
"id -= w * 4; \n" \
"od -= w; \n" \
"yp = yc; \n" \
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
"__kernel void k_erode(__global float *src, __global float *dst, int radio, int width, int height, int tile_w, int tile_h, __local float *smem, int c) { \n" \
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
"smem[ty * get_local_size(0) + tx] = src[(y * width + x) * 4 + c]; \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) { \n" \
"return; \n" \
"} \n" \
"__local float* smem_thread = &smem[ty * get_local_size(0) + tx - radio]; \n" \
"float val = smem_thread[0]; \n" \
"for (int xx = 1; xx <= 2 * radio; xx++) { \n" \
"val = fmin(val, smem_thread[xx]); \n" \
"} \n" \
"dst[y * width + x] = val; \n" \
"} \n" \
"__kernel void k_dilate(__global float *src, __global float *dst, int radio, int width, int height, int tile_w, int tile_h, __local float *smem, int c) { \n" \
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
"smem[ty * get_local_size(0) + tx] = src[(y * width + x) * 4 + c]; \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) { \n" \
"return; \n" \
"} \n" \
"__local float* smem_thread = &smem[ty * get_local_size(0) + tx - radio]; \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"float val = smem_thread[0]; \n" \
"for (int xx = 1; xx <= 2 * radio; xx++) { \n" \
"val = fmax(val, smem_thread[xx]); \n" \
"} \n" \
"dst[y * width + x] = val; \n" \
"} \n" \
"__kernel void k_simple(__global float *id, __global float *od, int w, int h)      \n" \
"{                                 				    \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);				    \n" \
"if ((x < w) && (y < h)) \n" \
"{   \n" \
"const int index = (y * w + x) * 4; \n" \
"od[index + 0] = id[index + 0]; \n" \
"od[index + 1] = id[index + 1]; \n" \
"od[index + 2] = id[index + 2]; \n" \
"od[index + 3] = id[index + 3]; \n" \
"}} \n" \
"__kernel void k_freqSharpen(__global float *id, __global float *od, int w, int h, float sharpen, int display) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);	 \n" \
"float offset = display == 1 ? 0.5f : 0.0f; \n" \
"if ((x < w) && (y < h)) \n" \
"{   \n" \
"const int index = (y * w + x) * 4; \n" \
"for(int c = 0; c < 3; c++) {										 \n" \
"id[index + c] = (id[index + c] - od[index + c]) * sharpen + offset; \n" \
"} \n" \
"if (display == 1) { \n" \
"for(int c = 0; c < 3; c++) { \n" \
"od[index + c] = id[index + c]; \n" \
"}}}} \n" \
"__kernel void k_freqAdd(__global float *id, __global float *od, int w, int h) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);				    \n" \
"if ((x < w) && (y < h)) \n" \
"{   \n" \
"const int index = (y * w + x) * 4; \n" \
"for(int c = 0; c < 3; c++) {																								 \n" \
"od[index + c] = id[index + c] + od[index + c]; \n" \
"}}} \n" \
"__kernel void k_edgeDetect(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float p_Threshold) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);	 \n" \
"p_Threshold += 3.0f; \n" \
"p_Threshold *= 3.0f; \n" \
" \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = ((y * p_Width) + x) * 4; \n" \
"for(int c = 0; c < 3; c++) {																								    \n" \
"p_Output[index + c] = (p_Input[index + c] - p_Output[index + c]) * p_Threshold; \n" \
"}}} \n" \
"__kernel void k_edgeEnhance(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float p_Enhance) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"int X = x == 0 ? 1 : x; \n" \
"int indexE = (y * p_Width + X) * 4; \n" \
"p_Output[index + 0] = (p_Input[index + 0] - p_Input[indexE - 4]) * p_Enhance; \n" \
"p_Output[index + 1] = (p_Input[index + 1] - p_Input[indexE - 3]) * p_Enhance; \n" \
"p_Output[index + 2] = (p_Input[index + 2] - p_Input[indexE - 2]) * p_Enhance; \n" \
"}} \n" \
"__kernel void k_customMatrix(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float p_Scale,  \n" \
"int p_Normalise, float p_Matrix11, float p_Matrix12, float p_Matrix13, float p_Matrix21, float p_Matrix22,  \n" \
"float p_Matrix23, float p_Matrix31, float p_Matrix32, float p_Matrix33) {  \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"p_Scale += 1.0f; \n" \
"float normalise = 1.0f; \n" \
"if (p_Scale > 1.0f) { \n" \
"p_Matrix11 *= p_Scale; \n" \
"p_Matrix12 *= p_Scale; \n" \
"p_Matrix13 *= p_Scale; \n" \
"p_Matrix21 *= p_Scale; \n" \
"p_Matrix22 *= p_Scale; \n" \
"p_Matrix23 *= p_Scale; \n" \
"p_Matrix31 *= p_Scale; \n" \
"p_Matrix32 *= p_Scale; \n" \
"p_Matrix33 *= p_Scale; \n" \
"} \n" \
"float total = p_Matrix11 + p_Matrix12 + p_Matrix13 + p_Matrix21 + p_Matrix22 + p_Matrix23 + p_Matrix31 + p_Matrix32 + p_Matrix33; \n" \
"if (p_Normalise == 1 && total > 1.0f) \n" \
"normalise /= total; \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = ((y * p_Width) + x) * 4; \n" \
"int start_y = max(y - 1, 0); \n" \
"int end_y = min(p_Height - 1, y + 1); \n" \
"int start_x = max(x - 1, 0); \n" \
"int end_x = min(p_Width - 1, x + 1); \n" \
"for(int c = 0; c < 3; c++) { \n" \
"p_Output[index + c] = (p_Input[(end_y * p_Width + start_x) * 4 + c] * p_Matrix11 + \n" \
"p_Input[(end_y * p_Width + x) * 4 + c] * p_Matrix12 + \n" \
"p_Input[(end_y * p_Width + end_x) * 4 + c] * p_Matrix13 + \n" \
"p_Input[(y * p_Width + start_x) * 4 + c] * p_Matrix21 + \n" \
"p_Input[(y * p_Width + x) * 4 + c] * p_Matrix22 + \n" \
"p_Input[(y * p_Width + end_x) * 4 + c] * p_Matrix23 + \n" \
"p_Input[(start_y * p_Width + start_x) * 4 + c] * p_Matrix31 + \n" \
"p_Input[(start_y * p_Width + x) * 4 + c] * p_Matrix32 + \n" \
"p_Input[(start_y * p_Width + end_x) * 4 + c] * p_Matrix33) * normalise; \n" \
"}}} \n" \
"__kernel void k_scatter(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, int p_Range, float p_Mix) \n" \
"{                                  \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);                                   \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"int rg = p_Range + 1; \n" \
"int totA = round((p_Input[index + 0] + p_Input[index + 1] + p_Input[index + 2]) * 1111) + x; \n" \
"int totB = round((p_Input[index + 0] + p_Input[index + 1]) * 1111) + y; \n" \
"int polarityA = totA % 2 > 0.0f ? -1.0f : 1.0f; \n" \
"int polarityB = totB % 2 > 0.0f ? -1.0f : 1.0f; \n" \
"int scatterA = (totA % rg) * polarityA; \n" \
"int scatterB = (totB % rg) * polarityB; \n" \
"int X = (x + scatterA) < 0 ? abs(x + scatterA) : ((x + scatterA) > (p_Width - 1) ? (2 * (p_Width - 1)) - (x + scatterA) : (x + scatterA)); \n" \
"int Y = (y + scatterB) < 0 ? abs(y + scatterB) : ((y + scatterB) > (p_Height - 1) ? (2 * (p_Height - 1)) - (y + scatterB) : (y + scatterB)); \n" \
"p_Output[index + 0] = p_Input[((Y * p_Width) + X) * 4 + 0] * (1.0f - p_Mix) + p_Mix * p_Input[index + 0]; \n" \
"p_Output[index + 1] = p_Input[((Y * p_Width) + X) * 4 + 1] * (1.0f - p_Mix) + p_Mix * p_Input[index + 1]; \n" \
"p_Output[index + 2] = p_Input[((Y * p_Width) + X) * 4 + 2] * (1.0f - p_Mix) + p_Mix * p_Input[index + 2]; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
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

void RunOpenCLKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Convolve, int p_Display, float* p_Adjust, float* p_Matrix)
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

cl_kernel Gausfilter = NULL;
cl_kernel Recursive = NULL;
cl_kernel Boxfilter = NULL;
cl_kernel Transpose = NULL;
cl_kernel Erode = NULL;
cl_kernel Dilate = NULL;
cl_kernel Simple = NULL;
cl_kernel FreqSharpen = NULL;
cl_kernel FreqAdd = NULL;
cl_kernel EdgeDetect = NULL;
cl_kernel EdgeEnhance = NULL;
cl_kernel Scatter = NULL;
cl_kernel CustomMatrix = NULL;

Gausfilter = clCreateKernel(program, "k_gaussian", &error);
CheckError(error, "Unable to create kernel");

Recursive = clCreateKernel(program, "k_simpleRecursive", &error);
CheckError(error, "Unable to create kernel");

Boxfilter = clCreateKernel(program, "k_boxfilter", &error);
CheckError(error, "Unable to create kernel");

Transpose = clCreateKernel(program, "k_transpose", &error);
CheckError(error, "Unable to create kernel");

Simple = clCreateKernel(program, "k_simple", &error);
CheckError(error, "Unable to create kernel");

Erode = clCreateKernel(program, "k_erode", &error);
CheckError(error, "Unable to create kernel");

Dilate = clCreateKernel(program, "k_dilate", &error);
CheckError(error, "Unable to create kernel");

FreqSharpen = clCreateKernel(program, "k_freqSharpen", &error);
CheckError(error, "Unable to create kernel");

FreqAdd = clCreateKernel(program, "k_freqAdd", &error);
CheckError(error, "Unable to create kernel");

EdgeDetect = clCreateKernel(program, "k_edgeDetect", &error);
CheckError(error, "Unable to create kernel");

EdgeEnhance = clCreateKernel(program, "k_edgeEnhance", &error);
CheckError(error, "Unable to create kernel");

Scatter = clCreateKernel(program, "k_scatter", &error);
CheckError(error, "Unable to create kernel");

CustomMatrix = clCreateKernel(program, "k_customMatrix", &error);
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

error |= clSetKernelArg(Gausfilter, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Recursive, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Boxfilter, 1, sizeof(cl_mem), &tempBuffer);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Transpose, 0, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Transpose, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Transpose, 4, sizeof(float) * iTransposeBlockDim * (iTransposeBlockDim + 1), NULL);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Simple, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Simple, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Simple, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Simple, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

clEnqueueNDRangeKernel(cmdQ, Simple, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

switch (p_Convolve)
{
case 0:
{
p_Adjust[0] *= 100.0f;
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Adjust[0]);
if (p_Adjust[0] > 0.0f) {
for(int c = 0; c < 3; c++) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}}}
break;

case 1:
{
p_Adjust[0] *= 100.0f;
error |= clSetKernelArg(Recursive, 4, sizeof(float), &p_Adjust[0]);
if (p_Adjust[0] > 0.0f) {
for(int c = 0; c < 3; c++) {
error = clSetKernelArg(Recursive, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Recursive, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Recursive, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Recursive, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Recursive, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Recursive, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Recursive, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Recursive, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Recursive, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}}}
break;

case 2:
{
int R = p_Adjust[0] * 100;
error |= clSetKernelArg(Boxfilter, 4, sizeof(int), &R);
if (R > 0) {
for(int c = 0; c < 3; c++) {
error = clSetKernelArg(Boxfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Boxfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Boxfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Boxfilter, 5, sizeof(float), &c);
clEnqueueNDRangeKernel(cmdQ, Boxfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Boxfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Boxfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Boxfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Boxfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}}}
break;

case 3:
{
p_Adjust[2] *= 5.0f;
p_Adjust[0] *= 10.0f;
float sharpen = (2 * p_Adjust[1]) + 1;
if (p_Adjust[2] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Adjust[2]);
for(int c = 0; c < 3; c++) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}}
error = clSetKernelArg(FreqSharpen, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FreqSharpen, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FreqSharpen, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FreqSharpen, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(FreqSharpen, 4, sizeof(float), &sharpen);
error |= clSetKernelArg(FreqSharpen, 5, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, FreqSharpen, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if (p_Display != 1) {
if (p_Adjust[0] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Adjust[0]);
for(int c = 0; c < 3; c++) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}}
error = clSetKernelArg(FreqAdd, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FreqAdd, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FreqAdd, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FreqAdd, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, FreqAdd, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}
break;

case 4:
{
p_Adjust[2] *= 3.0f;
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Adjust[2]);
if (p_Adjust[2] > 0.0f) {
for(int c = 0; c < 3; c++) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}}
error = clSetKernelArg(EdgeDetect, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(EdgeDetect, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(EdgeDetect, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(EdgeDetect, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(EdgeDetect, 4, sizeof(float), &p_Adjust[2]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, EdgeDetect, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
break;

case 5:
{
p_Adjust[0] *= 20.0f;
error = clSetKernelArg(EdgeEnhance, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(EdgeEnhance, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(EdgeEnhance, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(EdgeEnhance, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(EdgeEnhance, 4, sizeof(float), &p_Adjust[0]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, EdgeEnhance, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
break;

case 6:
{
int radio = ceil(p_Adjust[0] * 15);
int tile_w = 640;
int tile_h = 1;
erodeLocalWorkSize[0] = tile_w + (2 * radio);
erodeLocalWorkSize[1] = tile_h;
erodeGlobalWorkSizeA[0] = shrRoundUp(erodeLocalWorkSize[0], p_Width);
erodeGlobalWorkSizeA[1] = shrRoundUp(erodeLocalWorkSize[1], p_Height);
erodeGlobalWorkSizeB[0] = shrRoundUp(erodeLocalWorkSize[0], p_Height);
erodeGlobalWorkSizeB[1] = shrRoundUp(erodeLocalWorkSize[1], p_Width);
error |= clSetKernelArg(Erode, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Erode, 5, sizeof(int), &tile_w);
error |= clSetKernelArg(Erode, 6, sizeof(int), &tile_h);
error |= clSetKernelArg(Erode, 7, sizeof(float) * erodeLocalWorkSize[1] * erodeLocalWorkSize[0], NULL);
CheckError(error, "Unable to set kernel arguments");
if (p_Adjust[0] > 0.0f) {
for(int c = 0; c < 3; c++) {
error = clSetKernelArg(Erode, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Erode, 2, sizeof(int), &radio);
error |= clSetKernelArg(Erode, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Erode, 4, sizeof(int), &p_Height);
error |= clSetKernelArg(Erode, 8, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Erode, 2, NULL, erodeGlobalWorkSizeA, erodeLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Erode, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Erode, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Erode, 4, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Erode, 2, NULL, erodeGlobalWorkSizeB, erodeLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}}}
break;

case 7:
{

int radio = ceil(p_Adjust[0] * 15);
int tile_w = 640;
int tile_h = 1;
erodeLocalWorkSize[0] = tile_w + (2 * radio);
erodeLocalWorkSize[1] = tile_h;
erodeGlobalWorkSizeA[0] = shrRoundUp(erodeLocalWorkSize[0], p_Width);
erodeGlobalWorkSizeA[1] = shrRoundUp(erodeLocalWorkSize[1], p_Height);
erodeGlobalWorkSizeB[0] = shrRoundUp(erodeLocalWorkSize[0], p_Height);
erodeGlobalWorkSizeB[1] = shrRoundUp(erodeLocalWorkSize[1], p_Width);
error |= clSetKernelArg(Dilate, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Dilate, 5, sizeof(int), &tile_w);
error |= clSetKernelArg(Dilate, 6, sizeof(int), &tile_h);
error |= clSetKernelArg(Dilate, 7, sizeof(float) * erodeLocalWorkSize[1] * erodeLocalWorkSize[0], NULL);
CheckError(error, "Unable to set kernel arguments");
if (p_Adjust[0] > 0.0f) {
for(int c = 0; c < 3; c++) {
error = clSetKernelArg(Dilate, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Dilate, 2, sizeof(int), &radio);
error |= clSetKernelArg(Dilate, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Dilate, 4, sizeof(int), &p_Height);
error |= clSetKernelArg(Dilate, 8, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Dilate, 2, NULL, erodeGlobalWorkSizeA, erodeLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Dilate, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Dilate, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Dilate, 4, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Dilate, 2, NULL, erodeGlobalWorkSizeB, erodeLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}} else {
clEnqueueNDRangeKernel(cmdQ, Simple, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}
break;

case 8:
{
int radio = ceil(p_Adjust[0] * 15);
error = clSetKernelArg(Scatter, 0, sizeof(cl_mem), &p_Input);
error = clSetKernelArg(Scatter, 1, sizeof(cl_mem), &p_Output);
error = clSetKernelArg(Scatter, 2, sizeof(int), &p_Width);
error = clSetKernelArg(Scatter, 3, sizeof(int), &p_Height);
error = clSetKernelArg(Scatter, 4, sizeof(int), &radio);
error = clSetKernelArg(Scatter, 5, sizeof(float), &p_Adjust[1]);
CheckError(error, "Unable to set kernel arguments");
if (p_Adjust[0] > 0.0f) {
clEnqueueNDRangeKernel(cmdQ, Scatter, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}
break;

case 9:
{
p_Adjust[0] *= 10.0f;
error = clSetKernelArg(CustomMatrix, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(CustomMatrix, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(CustomMatrix, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(CustomMatrix, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(CustomMatrix, 4, sizeof(float), &p_Adjust[0]);
error |= clSetKernelArg(CustomMatrix, 5, sizeof(int), &p_Display);
error |= clSetKernelArg(CustomMatrix, 6, sizeof(float), &p_Matrix[0]);
error |= clSetKernelArg(CustomMatrix, 7, sizeof(float), &p_Matrix[1]);
error |= clSetKernelArg(CustomMatrix, 8, sizeof(float), &p_Matrix[2]);
error |= clSetKernelArg(CustomMatrix, 9, sizeof(float), &p_Matrix[3]);
error |= clSetKernelArg(CustomMatrix, 10, sizeof(float), &p_Matrix[4]);
error |= clSetKernelArg(CustomMatrix, 11, sizeof(float), &p_Matrix[5]);
error |= clSetKernelArg(CustomMatrix, 12, sizeof(float), &p_Matrix[6]);
error |= clSetKernelArg(CustomMatrix, 13, sizeof(float), &p_Matrix[7]);
error |= clSetKernelArg(CustomMatrix, 14, sizeof(float), &p_Matrix[8]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, CustomMatrix, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}
clReleaseMemObject(tempBuffer);
}