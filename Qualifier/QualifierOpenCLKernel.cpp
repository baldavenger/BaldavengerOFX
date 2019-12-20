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
"float Luma(float R, float G, float B, int L); \n" \
"float Alpha(float p_ScaleA, float p_ScaleB, float p_ScaleC, float p_ScaleD, float p_ScaleE, float p_ScaleF, float N, int p_Switch); \n" \
"void RGB_to_HSV( float r, float g, float b, float *h, float *s, float *v ); \n" \
"void HSV_to_RGB(float H, float S, float V, float *r, float *g, float *b); \n" \
"float Luma(float R, float G, float B, int L) { \n" \
"float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f; \n" \
"float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f; \n" \
"float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f; \n" \
"float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f; \n" \
"float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f; \n" \
"float lumaAvg = (R + G + B) / 3.0f; \n" \
"float lumaMax = fmax(fmax(R, G), B); \n" \
"float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : \n" \
"L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax; \n" \
"return Lu; \n" \
"} \n" \
"float Alpha(float p_ScaleA, float p_ScaleB, float p_ScaleC, float p_ScaleD, \n" \
"float p_ScaleE, float p_ScaleF, float N, int p_Switch) { \n" \
"float r = p_ScaleA; \n" \
"float g = p_ScaleB; \n" \
"float b = p_ScaleC; \n" \
"float a = p_ScaleD; \n" \
"float d = 1.0f / p_ScaleE; \n" \
"float e = 1.0f / p_ScaleF; \n" \
"float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= N ? 1.0f : (r >= N ? pow((r - N) / (1.0f - g), d) : 0.0f)); \n" \
"float k = a == 1.0f ? 0.0f : (a + b <= N ? 1.0f : (a <= N ? pow((N - a) / b, e) : 0.0f)); \n" \
"float alpha = k * w; \n" \
"float alphaV = p_Switch==1 ? 1.0f - alpha : alpha; \n" \
"return alphaV; \n" \
"} \n" \
"void RGB_to_HSV( float r, float g, float b, float *h, float *s, float *v ) { \n" \
"float min = fmin(fmin(r, g), b); \n" \
"float max = fmax(fmax(r, g), b); \n" \
"*v = max; \n" \
"float delta = max - min; \n" \
"if (max != 0.0f) { \n" \
"*s = delta / max; \n" \
"} else { \n" \
"*s = 0.0f; \n" \
"*h = 0.0f; \n" \
"return; \n" \
"} \n" \
"if (delta == 0.0f) { \n" \
"*h = 0.0f; \n" \
"} else if (r == max) { \n" \
"*h = (g - b) / delta; \n" \
"} else if (g == max) { \n" \
"*h = 2.0f + (b - r) / delta; \n" \
"} else { \n" \
"*h = 4.0f + (r - g) / delta; \n" \
"} \n" \
"*h *= 1.0f / 6.0f; \n" \
"if (*h < 0.0f) { \n" \
"*h += 1.0f; \n" \
"}} \n" \
"void HSV_to_RGB(float H, float S, float V, float *r, float *g, float *b) { \n" \
"if (S == 0.0f) { \n" \
"*r = *g = *b = V; \n" \
"return; \n" \
"} \n" \
"H *= 6.0f; \n" \
"int i = floor(H); \n" \
"float f = H - i; \n" \
"i = (i >= 0) ? (i % 6) : (i % 6) + 6; \n" \
"float p = V * (1.0f - S); \n" \
"float q = V * (1.0f - S * f); \n" \
"float t = V * (1.0f - S * (1.0f - f)); \n" \
"*r = i == 0 ? V : i == 1 ? q : i == 2 ? p : i == 3 ? p : i == 4 ? t : V; \n" \
"*g = i == 0 ? t : i == 1 ? V : i == 2 ? V : i == 3 ? q : i == 4 ? p : p; \n" \
"*b = i == 0 ? p : i == 1 ? p : i == 2 ? t : i == 3 ? V : i == 4 ? V : q; \n" \
"} \n" \
"__kernel void k_gaussian(__global float *id, __global float *od, int w, int h, float blur) { \n" \
"const float nsigma = blur < 0.1f ? 0.1f : blur, \n" \
"alpha = 1.695f / nsigma, \n" \
"ema = exp(-alpha); \n" \
"float ema2 = exp(-2.0f * alpha), \n" \
"b1 = -2*ema, \n" \
"b2 = ema2; \n" \
"float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0; \n" \
"const float k = (1.0f - ema) * (1.0f - ema) / (1.0f + 2.0f * alpha * ema - ema2); \n" \
"a0 = k; \n" \
"a1 = k * (alpha - 1.0f) * ema; \n" \
"a2 = k * (alpha + 1.0f) * ema; \n" \
"a3 = -k * ema2; \n" \
"coefp = (a0 + a1) / (1.0f + b1 + b2); \n" \
"coefn = (a2 + a3) / (1.0f + b1 + b2); \n" \
"int x = get_group_id(0) * get_local_size(0) + get_local_id(0); \n" \
"if (x >= w) return; \n" \
"id += x * 4 + 3; \n" \
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
"}} \n" \
"__kernel void d_garbageCore(__global float* p_Input, int p_Width, int p_Height, float p_Garbage, float p_Core) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) \n" \
"{ \n" \
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
"__kernel void k_transpose(__global float *id, __global float *od, int w, int h, __local float *buffer) \n" \
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
"if (xIndex < h && yIndex < w) \n" \
"{ \n" \
"od[(yIndex * h + xIndex) * 4 + 3] = buffer[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)]; \n" \
"}} \n" \
"__kernel void k_erode(__global float *src, __global float *dst, int radio, int width, int height, int tile_w, int tile_h, __local float *smem) { \n" \
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
"smem[ty * get_local_size(0) + tx] = src[(y * width + x) * 4 + 3]; \n" \
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
" \n" \
"__kernel void k_dilate(__global float *src, __global float *dst, int radio, int width, int height, int tile_w, int tile_h, __local float *smem) { \n" \
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
"smem[ty * get_local_size(0) + tx] = src[(y * width + x) * 4 + 3]; \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"if (x < bx * tile_w || x >= (bx + 1) * tile_w) { \n" \
"return; \n" \
"} \n" \
"__local float* smem_thread = &smem[ty * get_local_size(0) + tx - radio]; \n" \
"float val = smem_thread[0]; \n" \
"for (int xx = 1; xx <= 2 * radio; xx++) { \n" \
"val = fmax(val, smem_thread[xx]); \n" \
"} \n" \
"dst[y * width + x] = val; \n" \
"} \n" \
"__kernel void QualifierA(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float p_AlphaHA, float p_AlphaHB,  \n" \
"float p_AlphaHC, float p_AlphaHD, float p_AlphaHE, float p_AlphaHF, float p_AlphaHO, float p_AlphaSA, float p_AlphaSB, float p_AlphaSC,  \n" \
"float p_AlphaSD, float p_AlphaSE, float p_AlphaSF, float p_AlphaSO, float p_AlphaLA, float p_AlphaLB, float p_AlphaLC, float p_AlphaLD,  \n" \
"float p_AlphaLE, float p_AlphaLF, float p_AlphaLO, int p_InvertH, int p_InvertS, int p_InvertL, int p_Math, int p_OutputAlpha, \n" \
"float p_Black, float p_White, float p_HsvA, float p_HsvB, float p_HsvC) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float RGB[3]; \n" \
"RGB[0] = p_Input[index]; \n" \
"RGB[1] = p_Input[index + 1]; \n" \
"RGB[2] = p_Input[index + 2]; \n" \
" \n" \
"float h, s, v, H; \n" \
"RGB_to_HSV( RGB[0], RGB[1], RGB[2], &h, &s, &v ); \n" \
"float lum = Luma(RGB[0], RGB[1], RGB[2], p_Math); \n" \
"if ( s > 0.1f && v > 0.1f ) { \n" \
"float hh = h + p_AlphaHO; \n" \
"H = hh < 0.0f ? hh + 1.0f : hh >= 1.0f ? hh - 1.0f : hh; \n" \
"} else { H = 0.0f;} \n" \
"float Hue = Alpha(p_AlphaHA, p_AlphaHB, p_AlphaHC, p_AlphaHD, p_AlphaHE, p_AlphaHF, H, p_InvertH); \n" \
"float ss = s + p_AlphaSO; \n" \
"float S = ss < 0.0f ? 0.0f : ss >= 1.0f ? 1.0f : ss; \n" \
"float Sat = Alpha(p_AlphaSA, p_AlphaSB, p_AlphaSC, p_AlphaSD, p_AlphaSE, p_AlphaSF, S, p_InvertS); \n" \
"float l = lum + p_AlphaLO; \n" \
"float L = l < 0.0f ? 0.0f : l >= 1.0f ? 1.0f : l; \n" \
"float Lum = Alpha(p_AlphaLA, p_AlphaLB, p_AlphaLC, p_AlphaLD, p_AlphaLE, p_AlphaLF, L, p_InvertL); \n" \
"float A = p_OutputAlpha == 0 ? fmin(fmin(Hue, Sat), Lum) : p_OutputAlpha == 1 ? Hue : p_OutputAlpha == 2 ? Sat : \n" \
"p_OutputAlpha == 3 ? Lum : p_OutputAlpha == 4 ? fmin(Hue, Sat) : p_OutputAlpha == 5 ?  \n" \
"fmin(Hue, Lum) : p_OutputAlpha == 6 ? fmin(Sat, Lum) : 1.0f; \n" \
"if (p_Black > 0.0f) \n" \
"A = fmax(A + (0.0f - p_Black * 4.0f) * (1.0f - A), 0.0f); \n" \
"if (p_White > 0.0f) \n" \
"A = fmin(A * (1.0f + p_White * 4.0f), 1.0f); \n" \
"if (p_HsvA != 0.0f || p_HsvB != 0.0f || p_HsvC != 0.0f) { \n" \
"float h2 = h + p_HsvA; \n" \
"float H2 = h2 < 0.0f ? h2 + 1.0f : h2 >= 1.0f ? h2 - 1.0f : h2; \n" \
"float S = s * (1.0f + p_HsvB); \n" \
"float V = v * (1.0f + p_HsvC); \n" \
"HSV_to_RGB(H2, S, V, &RGB[0], &RGB[1], &RGB[2]); \n" \
"} \n" \
"p_Output[index] = RGB[0]; \n" \
"p_Output[index + 1] = RGB[1]; \n" \
"p_Output[index + 2] = RGB[2]; \n" \
"p_Output[index + 3] = A; \n" \
"}} \n" \
"__kernel void QualifierB(__global float* p_Input, __global float* p_Output, int p_Width, \n" \
"int p_Height, int p_Display, int p_Invert, int p_Warning, float p_Mix) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float A = p_Output[index + 3]; \n" \
"if (p_Invert == 1) \n" \
"A = 1.0f - A; \n" \
"if (p_Mix != 0.0f) { \n" \
"if (p_Mix > 0.0f) { \n" \
"A = A + (1.0f - A) * p_Mix; \n" \
"} else { \n" \
"A *= 1.0f + p_Mix; \n" \
"}} \n" \
"A = fmax(fmin(A, 1.0f), 0.0f); \n" \
"float RA, GA, BA; \n" \
"RA = GA = BA = A; \n" \
"if (p_Warning == 1 && p_Display == 1) { \n" \
"RA = A > 0.0f && A < 0.2f ? 0.0f : A < 1.0f && A > 0.8f ? 1.0f : A; \n" \
"GA = A > 0.0f && A < 0.2f ? 0.0f : A < 1.0f && A > 0.8f ? 1.0f : A; \n" \
"BA = A > 0.0f && A < 0.2f ? 1.0f : A < 1.0f && A > 0.8f ? 0.0f : A; \n" \
"} \n" \
"p_Output[index] = p_Display == 1 ? RA : p_Output[index] * A + p_Input[index] * (1.0f - A); \n" \
"p_Output[index + 1] = p_Display == 1 ? GA : p_Output[index + 1] * A + p_Input[index + 1] * (1.0f - A); \n" \
"p_Output[index + 2] = p_Display == 1 ? BA : p_Output[index + 2] * A + p_Input[index + 2] * (1.0f - A); \n" \
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
if (p_Error != CL_SUCCESS)
{
fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
}
}

int clDivUp(int a, int b)
{
return (a % b != 0) ? (a / b + 1) : (a / b);
}

int shrRoundUp(size_t localWorkSize, int numItems) {
int result = localWorkSize;
while (result < numItems)
result += localWorkSize;

return result;
}

void RunOpenCLKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, 
int* p_Switch, float* p_AlphaH, float* p_AlphaS, float* p_AlphaL, float p_Mix, int p_Math, int p_OutputAlpha, 
float p_Black, float p_White, float p_Blur, float p_Garbage, float p_Core, float p_Erode, float p_Dilate, float* p_HSV)
{
szBuffBytes = p_Width * p_Height * sizeof(float);
p_Blur *= 10.0f;

cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);
static std::map<cl_command_queue, cl_device_id> deviceIdMap;
static std::map<cl_command_queue, cl_kernel> kernelMap;
static Locker locker; // simple lock to control access to the above maps from multiple threads
locker.Lock();

cl_device_id deviceId = NULL;
if (deviceIdMap.find(cmdQ) == deviceIdMap.end()) {
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

cl_kernel KernelA = NULL;
cl_kernel Gausfilter = NULL;
cl_kernel Transpose = NULL;
cl_kernel Core = NULL;
cl_kernel Erode = NULL;
cl_kernel Dilate = NULL;
cl_kernel KernelB = NULL;

KernelA = clCreateKernel(program, "QualifierA", &error);
CheckError(error, "Unable to create kernel");

Gausfilter = clCreateKernel(program, "k_gaussian", &error);
CheckError(error, "Unable to create kernel");

Transpose = clCreateKernel(program, "k_transpose", &error);
CheckError(error, "Unable to create kernel");

Core = clCreateKernel(program, "d_garbageCore", &error);
CheckError(error, "Unable to create kernel");

Erode = clCreateKernel(program, "k_erode", &error);
CheckError(error, "Unable to create kernel");

Dilate = clCreateKernel(program, "k_dilate", &error);
CheckError(error, "Unable to create kernel");

KernelB = clCreateKernel(program, "QualifierB", &error);
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

error = clSetKernelArg(KernelA, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(KernelA, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(KernelA, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(KernelA, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(KernelA, 4, sizeof(float), &p_AlphaH[0]);
error |= clSetKernelArg(KernelA, 5, sizeof(float), &p_AlphaH[1]);
error |= clSetKernelArg(KernelA, 6, sizeof(float), &p_AlphaH[2]);
error |= clSetKernelArg(KernelA, 7, sizeof(float), &p_AlphaH[3]);
error |= clSetKernelArg(KernelA, 8, sizeof(float), &p_AlphaH[4]);
error |= clSetKernelArg(KernelA, 9, sizeof(float), &p_AlphaH[5]);
error |= clSetKernelArg(KernelA, 10, sizeof(float), &p_AlphaH[6]);
error |= clSetKernelArg(KernelA, 11, sizeof(float), &p_AlphaS[0]);
error |= clSetKernelArg(KernelA, 12, sizeof(float), &p_AlphaS[1]);
error |= clSetKernelArg(KernelA, 13, sizeof(float), &p_AlphaS[2]);
error |= clSetKernelArg(KernelA, 14, sizeof(float), &p_AlphaS[3]);
error |= clSetKernelArg(KernelA, 15, sizeof(float), &p_AlphaS[4]);
error |= clSetKernelArg(KernelA, 16, sizeof(float), &p_AlphaS[5]);
error |= clSetKernelArg(KernelA, 17, sizeof(float), &p_AlphaS[6]);
error |= clSetKernelArg(KernelA, 18, sizeof(float), &p_AlphaL[0]);
error |= clSetKernelArg(KernelA, 19, sizeof(float), &p_AlphaL[1]);
error |= clSetKernelArg(KernelA, 20, sizeof(float), &p_AlphaL[2]);
error |= clSetKernelArg(KernelA, 21, sizeof(float), &p_AlphaL[3]);
error |= clSetKernelArg(KernelA, 22, sizeof(float), &p_AlphaL[4]);
error |= clSetKernelArg(KernelA, 23, sizeof(float), &p_AlphaL[5]);
error |= clSetKernelArg(KernelA, 24, sizeof(float), &p_AlphaL[6]);
error |= clSetKernelArg(KernelA, 25, sizeof(int), &p_Switch[2]);
error |= clSetKernelArg(KernelA, 26, sizeof(int), &p_Switch[3]);
error |= clSetKernelArg(KernelA, 27, sizeof(int), &p_Switch[4]);
error |= clSetKernelArg(KernelA, 28, sizeof(int), &p_Math);
error |= clSetKernelArg(KernelA, 29, sizeof(int), &p_OutputAlpha);
error |= clSetKernelArg(KernelA, 30, sizeof(float), &p_Black);
error |= clSetKernelArg(KernelA, 31, sizeof(float), &p_White);
error |= clSetKernelArg(KernelA, 32, sizeof(float), &p_HSV[0]);
error |= clSetKernelArg(KernelA, 33, sizeof(float), &p_HSV[1]);
error |= clSetKernelArg(KernelA, 34, sizeof(float), &p_HSV[2]);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, KernelA, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Blur > 0.0f) {
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

if (p_Garbage > 0.0f || p_Core > 0.0f) {
error = clSetKernelArg(Core, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Core, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(Core, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Core, 3, sizeof(float), &p_Garbage);
error |= clSetKernelArg(Core, 4, sizeof(float), &p_Core);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, Core, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}

int radE = ceil(p_Erode * 15);
int radD = ceil(p_Dilate * 15);
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

if (p_Erode > 0.0f) {
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

if (p_Dilate > 0.0f) {
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

error = clSetKernelArg(KernelB, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(KernelB, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(KernelB, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(KernelB, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(KernelB, 4, sizeof(int), &p_Switch[0]);
error |= clSetKernelArg(KernelB, 5, sizeof(int), &p_Switch[1]);
error |= clSetKernelArg(KernelB, 6, sizeof(int), &p_Switch[5]);
error |= clSetKernelArg(KernelB, 7, sizeof(float), &p_Mix);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, KernelB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

clReleaseMemObject(tempBuffer);
}