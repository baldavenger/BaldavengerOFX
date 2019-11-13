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
"__kernel void k_rec709toLAB(__global float *id, __global float *od, int w, int h) \n" \
"{ \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < w) && (y < h)) \n" \
"{ \n" \
"const int index = (y * w + x) * 4; \n" \
"float linR = id[index + 0] < 0.08145f ? (id[index + 0] < 0.0f ? 0.0f : id[index + 0] * (1.0f / 4.5f)) : pow((id[index + 0] + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f)); \n" \
"float linG = id[index + 1] < 0.08145f ? (id[index + 1] < 0.0f ? 0.0f : id[index + 1] * (1.0f / 4.5f)) : pow((id[index + 1] + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f)); \n" \
"float linB = id[index + 2] < 0.08145f ? (id[index + 2] < 0.0f ? 0.0f : id[index + 2] * (1.0f / 4.5f)) : pow((id[index + 2] + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f)); \n" \
"float xyzR = 0.4124564f * linR + 0.3575761f * linG + 0.1804375f * linB; \n" \
"float xyzG = 0.2126729f * linR + 0.7151522f * linG + 0.0721750f * linB; \n" \
"float xyzB = 0.0193339f * linR + 0.1191920f * linG + 0.9503041f * linB; \n" \
"xyzR /= (0.412453f + 0.357580f + 0.180423f); \n" \
"xyzG /= (0.212671f + 0.715160f + 0.072169f); \n" \
"xyzB /= (0.019334f + 0.119193f + 0.950227f); \n" \
"float fx = xyzR >= 0.008856f ? pow(xyzR, 1.0f / 3.0f) : 7.787f * xyzR + 16.0f / 116.0f; \n" \
"float fy = xyzG >= 0.008856f ? pow(xyzG, 1.0f / 3.0f) : 7.787f * xyzG + 16.0f / 116.0f; \n" \
"float fz = xyzB >= 0.008856f ? pow(xyzB, 1.0f / 3.0f) : 7.787f * xyzB + 16.0f / 116.0f;     \n" \
"float L = (116.0f * fy - 16.0f) / 100.0f; \n" \
"od[index + 0] = L; \n" \
"od[index + 1] = (500.0f * (fx - fy)) / 200.0f + 0.5f; \n" \
"od[index + 2] = (200.0f * (fy - fz)) / 200.0f + 0.5f; \n" \
"id[index + 0] = L; \n" \
"} \n" \
"} \n" \
"__kernel void k_LABtoRec709(__global float *id, int w, int h) \n" \
"{ \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < w) && (y < h)) \n" \
"{ \n" \
"const int index = (y * w + x) * 4; \n" \
"float l = id[index + 0] * 100.0f; \n" \
"float a = (id[index + 1] - 0.5f) * 200.0f; \n" \
"float b = (id[index + 2] - 0.5f) * 200.0f; \n" \
"float cy = (l + 16.0f) / 116.0f; \n" \
"float CY = cy >= 0.206893f ? (cy * cy * cy) : (cy - 16.0f / 116.0f) / 7.787f; \n" \
"float y = (0.212671f + 0.715160f + 0.072169f) * CY; \n" \
"float cx = a / 500.0f + cy; \n" \
"float CX = cx >= 0.206893f ? (cx * cx * cx) : (cx - 16.0f / 116.0f) / 7.787f; \n" \
"float x = (0.412453f + 0.357580f + 0.180423f) * CX; \n" \
"float cz = cy - b / 200.0f; \n" \
"float CZ = cz >= 0.206893f ? (cz * cz * cz) : (cz - 16.0f / 116.0f) / 7.787f; \n" \
"float z = (0.019334f + 0.119193f + 0.950227f) * CZ; \n" \
"float r =  3.2404542f * x + -1.5371385f * y + -0.4985314f * z; \n" \
"float g = -0.9692660f * x +  1.8760108f * y +  0.0415560f * z; \n" \
"float _b =  0.0556434f * x + -0.2040259f * y +  1.0572252f * z; \n" \
"float R = r < 0.0181f ? (r < 0.0f ? 0.0f : r * 4.5f) : 1.0993f * pow(r, 0.45f) - (1.0993f - 1.0f); \n" \
"float G = g < 0.0181f ? (g < 0.0f ? 0.0f : g * 4.5f) : 1.0993f * pow(g, 0.45f) - (1.0993f - 1.0f); \n" \
"float B = _b < 0.0181f ? (_b < 0.0f ? 0.0f : _b * 4.5f) : 1.0993f * pow(_b, 0.45f) - (1.0993f - 1.0f); \n" \
"id[index + 0] = R; \n" \
"id[index + 1] = G; \n" \
"id[index + 2] = B; \n" \
"} \n" \
"} \n" \
"__kernel void k_rec709toYUV(__global float *id, __global float *od, int w, int h) \n" \
"{ \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < w) && (y < h)) \n" \
"{ \n" \
"const int index = (y * w + x) * 4; \n" \
"float Y = 0.2126f * id[index + 0] + 0.7152f * id[index + 1] + 0.0722f * id[index + 2]; \n" \
"od[index + 0] = Y; \n" \
"od[index + 1] = -0.09991f * id[index + 0] - 0.33609f * id[index + 1] + 0.436f * id[index + 2]; \n" \
"od[index + 2] = 0.615f * id[index + 0] - 0.55861f * id[index + 1] - 0.05639f * id[index + 2]; \n" \
"id[index + 0] = Y; \n" \
"} \n" \
"} \n" \
"__kernel void k_YUVtoRec709(__global float *id, int w, int h) \n" \
"{ \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < w) && (y < h)) \n" \
"{ \n" \
"const int index = (y * w + x) * 4; \n" \
"float r = id[index + 0] + 1.28033f * id[index + 2]; \n" \
"float g = id[index + 0] - 0.21482f * id[index + 1] - 0.38059f * id[index + 2]; \n" \
"float b = id[index + 0] + 2.12798f * id[index + 1]; \n" \
"id[index + 0] = r; \n" \
"id[index + 1] = g; \n" \
"id[index + 2] = b; \n" \
"} \n" \
"} \n" \
"__kernel void k_gaussian(__global float *id, __global float *od, int w, int h, float blur, int c) \n" \
"{ \n" \
"float nsigma = blur < 0.1f ? 0.1f : blur; \n" \
"float alpha = 1.695f / nsigma; \n" \
"float ema = exp(-alpha); \n" \
"float ema2 = exp(-2.0f * alpha), \n" \
"b1 = -2.0f * ema, \n" \
"b2 = ema2; \n" \
"float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, coefp = 0.0f, coefn = 0.0f; \n" \
"float k = (1.0f - ema) * (1.0f - ema) / (1.0f + 2.0f * alpha * ema - ema2); \n" \
"a0 = k; \n" \
"a1 = k * (alpha - 1.0f) * ema; \n" \
"a2 = k * (alpha + 1.0f) * ema; \n" \
"a3 = -k * ema2; \n" \
"coefp = (a0 + a1) / (1.0f + b1 + b2); \n" \
"coefn = (a2 + a3) / (1.0f + b1 + b2); \n" \
"int x = get_group_id(0) * get_local_size(0) + get_local_id(0); \n" \
"if (x >= w) return; \n" \
"id += x * 4 + c; \n" \
"od += x; \n" \
"float xp, yp, yb; \n" \
"xp = *id; \n" \
"yb = coefp * xp; \n" \
"yp = yb; \n" \
"for (int y = 0; y < h; y++) \n" \
"{ \n" \
"float xc = *id; \n" \
"float yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb; \n" \
"*od = yc; \n" \
"id += w * 4; \n" \
"od += w; \n" \
"xp = xc; \n" \
"yb = yp; \n" \
"yp = yc; \n" \
"} \n" \
"id -= w * 4; \n" \
"od -= w; \n" \
"float xn, xa, yn, ya; \n" \
"xn = xa = *id; \n" \
"yn = coefn * xn; \n" \
"ya = yn; \n" \
"for (int y = h - 1; y >= 0; y--) \n" \
"{ \n" \
"float xc = *id; \n" \
"float yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya; \n" \
"xa = xn; \n" \
"xn = xc; \n" \
"ya = yn; \n" \
"yn = yc; \n" \
"*od = *od + yc; \n" \
"id -= w * 4; \n" \
"od -= w; \n" \
"} \n" \
"} \n" \
"__kernel void k_transpose(__global float *id, __global float *od, int w, int h, __local float *buffer, int c) \n" \
"{ \n" \
"int xIndex = get_global_id(0); \n" \
"int yIndex = get_global_id(1); \n" \
"if((xIndex < w) && (yIndex < h)) \n" \
"{ \n" \
"buffer[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = id[(yIndex * w + xIndex)]; \n" \
"} \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"xIndex = get_group_id(1) * get_local_size(1) + get_local_id(0); \n" \
"yIndex = get_group_id(0) * get_local_size(0) + get_local_id(1); \n" \
"if((xIndex < h) && (yIndex < w)) \n" \
"{ \n" \
"od[(yIndex * h + xIndex) * 4 + c] = buffer[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)]; \n" \
"} \n" \
"} \n" \
"__kernel void k_simple(__global float *id, __global float *od, int w, int h, int c)      \n" \
"{                                 				    \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);				    \n" \
"if ((x < w) && (y < h)) \n" \
"{   \n" \
"const int index = (y * w + x) * 4; \n" \
"od[index + c] = id[index + c];  \n" \
"} \n" \
"} \n" \
"__kernel void k_freqSharpen(__global float *id, __global float *od, int w, int h, float sharpen, int p_Display, int c) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);	 \n" \
"float offset = p_Display == 1 ? 0.5f : 0.0f; \n" \
"if ((x < w) && (y < h)) \n" \
"{   \n" \
"const int index = (y * w + x) * 4;										 \n" \
"id[index + c] = (id[index + c] - od[index + c]) * sharpen + offset; \n" \
"if (p_Display == 1) { \n" \
"od[index + c] = id[index + c]; \n" \
"} \n" \
"} \n" \
"} \n" \
"__kernel void k_freqSharpenLuma(__global float *id, __global float *od, int w, int h, float sharpen, int p_Display) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);	 \n" \
"float offset = p_Display == 1 ? 0.5f : 0.0f; \n" \
"if ((x < w) && (y < h)) \n" \
"{   \n" \
"const int index = (y * w + x) * 4;										 \n" \
"id[index] = (id[index] - od[index]) * sharpen + offset; \n" \
"if (p_Display == 1) \n" \
"od[index] = od[index + 1] = od[index + 2] = id[index]; \n" \
"} \n" \
"} \n" \
"__kernel void k_lowFreqCont(__global float *id, int w, int h, float contrast, float pivot, int curve, int p_Display, int c) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if ((x < w) && (y < h)) \n" \
"{ \n" \
"const int index = (y * w + x) * 4; \n" \
"float graph = 0.0f; \n" \
"if(curve == 1) \n" \
"id[index + c] = id[index + c] <= pivot ? pow(id[index + c] / pivot, contrast) * pivot : (1.0f - pow(1.0f - (id[index + c] - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"else \n" \
"id[index + c] = (id[index + c] - pivot) * contrast + pivot; \n" \
"if(p_Display == 3){ \n" \
"float width = w; \n" \
"float height = h; \n" \
"float X = x; \n" \
"float Y = y; \n" \
"float ramp = X / (width - 1.0f); \n" \
"if(curve == 1) \n" \
"ramp = ramp <= pivot ? pow(ramp / pivot, contrast) * pivot : (1.0f - pow(1.0f - (ramp - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"else \n" \
"ramp = (ramp - pivot) * contrast + pivot; \n" \
"graph = ramp >= (Y - 5.0f) / height && ramp <= (Y + 5.0f) / height ? 1.0f : 0.0f; \n" \
"id[index + c] = graph == 0.0f ? id[index + c] : graph; \n" \
"} \n" \
"}  \n" \
"} \n" \
"__kernel void k_lowFreqContLuma(__global float *id, int w, int h, float contrast, float pivot, int curve, int p_Display) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if ((x < w) && (y < h)) \n" \
"{ \n" \
"const int index = (y * w + x) * 4; \n" \
"float graph = 0.0f; \n" \
"if(curve == 1) \n" \
"id[index] = id[index] <= pivot ? pow(id[index] / pivot, contrast) * pivot : (1.0f - pow(1.0f - (id[index] - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"else \n" \
"id[index] = (id[index] - pivot) * contrast + pivot; \n" \
"if(p_Display == 2) \n" \
"id[index + 2] = id[index + 1] = id[index]; \n" \
"if(p_Display == 3){ \n" \
"float width = w; \n" \
"float height = h; \n" \
"float X = x; \n" \
"float Y = y; \n" \
"float ramp = X / (width - 1.0f); \n" \
"if(curve == 1) \n" \
"ramp = ramp <= pivot ? pow(ramp / pivot, contrast) * pivot : (1.0f - pow(1.0f - (ramp - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"else \n" \
"ramp = (ramp - pivot) * contrast + pivot; \n" \
"graph = ramp >= (Y - 5.0f) / height && ramp <= (Y + 5.0f) / height ? 1.0f : 0.0f; \n" \
"id[index] = graph == 0.0f ? id[index] : graph; \n" \
"id[index + 2] = id[index + 1] = id[index]; \n" \
"} \n" \
"}  \n" \
"} \n" \
"__kernel void k_freqAdd(__global float *id, __global float *od, int w, int h, int c) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1);				    \n" \
"if ((x < w) && (y < h)) \n" \
"{   \n" \
"const int index = (y * w + x) * 4;																									 \n" \
"od[index + c] = id[index + c] + od[index + c]; \n" \
"} \n" \
"} \n" \
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

int clDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

int shrRoundUp(size_t localWorkSize, int numItems) {
int result = localWorkSize;
while (result < numItems)
result += localWorkSize;
return result;
}

void RunOpenCLKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Space, int p_Display, float* p_Blur, float* p_Sharpen, float* p_Cont, int* p_Switch)
{
int red = 0;
int green = 1;
int blue = 2;

szBuffBytes = p_Width * p_Height * sizeof(float);
cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

// store device id and kernel per command queue (required for multi-GPU systems)
static std::map<cl_command_queue, cl_device_id> deviceIdMap;
static std::map<cl_command_queue, cl_kernel> kernelMap;

static Locker locker; // simple lock to control access to the above maps from multiple threads
locker.Lock();

// find the device id corresponding to the command queue
cl_device_id deviceId = NULL;
if (deviceIdMap.find(cmdQ) == deviceIdMap.end()) {
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
cl_kernel Simple = NULL;
cl_kernel FreqSharpen = NULL;
cl_kernel FreqSharpenLuma = NULL;
cl_kernel LowFreqCont = NULL;
cl_kernel LowFreqContLuma = NULL;
cl_kernel FreqAdd = NULL;
cl_kernel Rec709toYUV = NULL;
cl_kernel YUVtoRec709 = NULL;
cl_kernel Rec709toLAB = NULL;
cl_kernel LABtoRec709 = NULL;

Gausfilter = clCreateKernel(program, "k_gaussian", &error);
CheckError(error, "Unable to create kernel");

Transpose = clCreateKernel(program, "k_transpose", &error);
CheckError(error, "Unable to create kernel");

Simple = clCreateKernel(program, "k_simple", &error);
CheckError(error, "Unable to create kernel");

FreqSharpen = clCreateKernel(program, "k_freqSharpen", &error);
CheckError(error, "Unable to create kernel");

FreqSharpenLuma = clCreateKernel(program, "k_freqSharpenLuma", &error);
CheckError(error, "Unable to create kernel");

LowFreqCont = clCreateKernel(program, "k_lowFreqCont", &error);
CheckError(error, "Unable to create kernel");

LowFreqContLuma = clCreateKernel(program, "k_lowFreqContLuma", &error);
CheckError(error, "Unable to create kernel");

FreqAdd = clCreateKernel(program, "k_freqAdd", &error);
CheckError(error, "Unable to create kernel");

Rec709toYUV = clCreateKernel(program, "k_rec709toYUV", &error);
CheckError(error, "Unable to create kernel");

YUVtoRec709 = clCreateKernel(program, "k_YUVtoRec709", &error);
CheckError(error, "Unable to create kernel");

Rec709toLAB = clCreateKernel(program, "k_rec709toLAB", &error);
CheckError(error, "Unable to create kernel");

LABtoRec709 = clCreateKernel(program, "k_LABtoRec709", &error);
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

error = clSetKernelArg(LowFreqCont, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LowFreqCont, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(LowFreqCont, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(LowFreqCont, 3, sizeof(float), &p_Cont[0]);
error |= clSetKernelArg(LowFreqCont, 4, sizeof(float), &p_Cont[1]);
error |= clSetKernelArg(LowFreqCont, 5, sizeof(int), &p_Switch[2]);
error |= clSetKernelArg(LowFreqCont, 6, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(LowFreqContLuma, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LowFreqContLuma, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(LowFreqContLuma, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(LowFreqContLuma, 3, sizeof(float), &p_Cont[0]);
error |= clSetKernelArg(LowFreqContLuma, 4, sizeof(float), &p_Cont[1]);
error |= clSetKernelArg(LowFreqContLuma, 5, sizeof(int), &p_Switch[2]);
error |= clSetKernelArg(LowFreqContLuma, 6, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FreqSharpen, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FreqSharpen, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FreqSharpen, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FreqSharpen, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(FreqSharpen, 5, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FreqSharpenLuma, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FreqSharpenLuma, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FreqSharpenLuma, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FreqSharpenLuma, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(FreqSharpenLuma, 4, sizeof(float), &p_Sharpen[0]);
error |= clSetKernelArg(FreqSharpenLuma, 5, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FreqAdd, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FreqAdd, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FreqAdd, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FreqAdd, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Rec709toYUV, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Rec709toYUV, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Rec709toYUV, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Rec709toYUV, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(YUVtoRec709, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(YUVtoRec709, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(YUVtoRec709, 2, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Rec709toLAB, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Rec709toLAB, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Rec709toLAB, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Rec709toLAB, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(LABtoRec709, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LABtoRec709, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(LABtoRec709, 2, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

for(int c = 0; c < 4; c++) {
error |= clSetKernelArg(Simple, 4, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Simple, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}

switch (p_Space) {
case 0:
{    
if (p_Switch[0] == 1)
p_Blur[2] = p_Blur[1] = p_Blur[0];

if (p_Blur[0] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[0]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[1] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[1]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[2] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[2]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Switch[0] == 1)
p_Sharpen[2] = p_Sharpen[1] = p_Sharpen[0];

for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(FreqSharpen, 4, sizeof(float), &p_Sharpen[c]);
error |= clSetKernelArg(FreqSharpen, 6, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FreqSharpen, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}

if (p_Display != 1) {
if (p_Switch[0] == 1)
p_Blur[5] = p_Blur[4] = p_Blur[3];

if (p_Blur[3] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[3]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[4] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[4]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[5] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[5]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(LowFreqCont, 7, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, LowFreqCont, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
}

if (p_Display == 0) {
for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(FreqAdd, 4, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FreqAdd, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
}
}
break;

case 1:
{
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
CheckError(error, "Unable to set kernel arguments");

clEnqueueNDRangeKernel(cmdQ, Rec709toYUV, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Blur[0] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[0]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

clEnqueueNDRangeKernel(cmdQ, FreqSharpenLuma, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Display != 1){
if (p_Switch[1] == 1)
p_Blur[5] = p_Blur[4];

if (p_Blur[3] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[3]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[4] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[4]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[5] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[5]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

clEnqueueNDRangeKernel(cmdQ, LowFreqContLuma, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}

if (p_Display == 0) {
error |= clSetKernelArg(FreqAdd, 4, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, FreqAdd, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, YUVtoRec709, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
}
break;

case 2:
{
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
CheckError(error, "Unable to set kernel arguments");

clEnqueueNDRangeKernel(cmdQ, Rec709toLAB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Blur[0] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[0]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

clEnqueueNDRangeKernel(cmdQ, FreqSharpenLuma, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Display != 1){
if (p_Switch[1] == 1)
p_Blur[5] = p_Blur[4];

if (p_Blur[3] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[3]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[4] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[4]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

if (p_Blur[5] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[5]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}

clEnqueueNDRangeKernel(cmdQ, LowFreqContLuma, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}

if (p_Display == 0) {
error |= clSetKernelArg(FreqAdd, 4, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, FreqAdd, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, LABtoRec709, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
}
}

clReleaseMemObject(tempBuffer);
}
