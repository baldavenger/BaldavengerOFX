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
"__constant float eu = 2.718281828459045f; \n" \
"__constant float pie = 3.141592653589793f; \n" \
"__kernel void k_Prepare(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, int p_Display) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"float ramp = (float)x / (float)(p_Width - 1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"p_Output[index] = p_Display == 1 ? ramp : p_Input[index]; \n" \
"p_Output[index + 1] = p_Display == 1 ? ramp : p_Input[index + 1]; \n" \
"p_Output[index + 2] = p_Display == 1 ? ramp : p_Input[index + 2]; \n" \
"p_Output[index + 3] = 1.0f; \n" \
"if (p_Display == 2) { \n" \
"p_Input[index] = ramp; \n" \
"p_Input[index + 1] = ramp; \n" \
"p_Input[index + 2] = ramp; \n" \
"}}} \n" \
"__kernel void k_FilmGradeKernelA(__global float* p_Input, int p_Width, int p_Height, float p_Exp, int ch_in) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if(x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"p_Input[index + ch_in] = p_Input[index + ch_in] + p_Exp * 0.01f; \n" \
"}} \n" \
"__kernel void k_FilmGradeKernelB(__global float* p_Input, int p_Width, int p_Height, float p_Shad, float p_Mid, float p_High, float p_ShadP, float p_HighP, float p_ContP, int ch_in) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float exp = p_Input[index + ch_in]; \n" \
"float expr1 = (p_ShadP / 2.0f) - (1.0f - p_HighP) / 4.0f; \n" \
"float expr2 = (1.0f - (1.0f - p_HighP) / 2.0f) + (p_ShadP / 4.0f); \n" \
"float expr3 = (exp - expr1) / (expr2 - expr1); \n" \
"float expr4 =  p_ContP < 0.5f ? 0.5f - (0.5f - p_ContP) / 2.0f : 0.5f + (p_ContP - 0.5f) / 2.0f; \n" \
"float expr5 = expr3 > expr4 ? (expr3 - expr4) / (2.0f - 2.0f * expr4) + 0.5f : expr3 / (2.0f * expr4); \n" \
"float expr6 = (((sin(2.0f * pie * (expr5 -1.0f / 4.0f)) + 1.0f) / 20.0f) * p_Mid * 4.0f) + expr3; \n" \
"float mid = exp >= expr1 && exp <= expr2 ? expr6 * (expr2 - expr1) + expr1 : exp; \n" \
"float shadup1 = mid > 0.0f ? 2.0f * (mid / p_ShadP) - log((mid / p_ShadP) * (eu * p_Shad * 2.0f) + 1.0f) / log(eu * p_Shad * 2.0f + 1.0f) : mid; \n" \
"float shadup = mid < p_ShadP && p_Shad > 0.0f ? (shadup1 + p_Shad * (1.0f - shadup1)) * p_ShadP : mid; \n" \
"float shaddown1 = shadup / p_ShadP + p_Shad * 2.0f * (1.0f - shadup / p_ShadP); \n" \
"float shaddown = shadup < p_ShadP && p_Shad < 0.0f ? (shaddown1 >= 0.0f ? log(shaddown1 * (eu * p_Shad * -2.0f) + 1.0f) / log(eu * p_Shad * -2.0f + 1.0f) : shaddown1) * p_ShadP : shadup; \n" \
"float highup1 = ((shaddown - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_High * 2.0f)); \n" \
"float highup = shaddown > p_HighP && p_HighP < 1.0f && p_High > 0.0f ? (2.0f * highup1 - log(highup1 * eu * p_High + 1.0f) / log(eu * p_High + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddown; \n" \
"float highdown1 = (highup - p_HighP) / (1.0f - p_HighP); \n" \
"float highdown = highup > p_HighP && p_HighP < 1.0f && p_High < 0.0f ? log(highdown1 * (eu * p_High * -2.0f) + 1.0f) / log(eu * p_High * -2.0f + 1.0f) * (1.0f + p_High) * (1.0f - p_HighP) + p_HighP : highup; \n" \
"p_Input[index + ch_in] = highdown; \n" \
"}} \n" \
"__kernel void k_FilmGradeKernelC(__global float* p_Input, int p_Width, int p_Height, float p_ContR, float p_ContG, float p_ContB, float p_SatR, float p_SatG, float p_SatB, float p_ContP) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float contR = (p_Input[index] - p_ContP) * p_ContR + p_ContP; \n" \
"float contG = (p_Input[index + 1] - p_ContP) * p_ContG + p_ContP; \n" \
"float contB = (p_Input[index + 2] - p_ContP) * p_ContB + p_ContP; \n" \
"float luma = contR * 0.2126f + contG * 0.7152f + contB * 0.0722f; \n" \
"float outR = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contR * p_SatR; \n" \
"float outG = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contG * p_SatG; \n" \
"float outB = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contB * p_SatB; \n" \
"p_Input[index] = outR; \n" \
"p_Input[index + 1] = outG; \n" \
"p_Input[index + 2] = outB; \n" \
"}} \n" \
"__kernel void k_FilmGradeKernelD(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float p_Pivot, int p_Display, int ch_in) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"float height = p_Height; \n" \
"float width = p_Width; \n" \
"float X = x; \n" \
"float Y = y; \n" \
"const float RES = width / 1920.0f; \n" \
"float overlay = 0.0f; \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"if (p_Display == 1) { \n" \
"overlay = Y / height >= p_Pivot && Y / height <= p_Pivot + 0.005f * RES ? (fmod(X, 2.0f) != 0.0f ? 1.0f : 0.0f) : \n" \
"p_Output[index + ch_in] >= (Y - 5.0f * RES) / height && p_Output[index + ch_in] <= (Y + 5.0f * RES) / height ? 1.0f : 0.0f; \n" \
"p_Output[index + ch_in] = overlay; \n" \
"} \n" \
"if (p_Display == 2) { \n" \
"overlay = Y / height >= p_Pivot && Y / height <= p_Pivot + 0.005f ? (fmod(X, 2.0f) != 0.0f ? 1.0f : 0.0f) : \n" \
"p_Input[index + ch_in] >= (Y - 5.0f * RES) / height && p_Input[index + ch_in] <= (Y + 5.0f * RES) / height ? 1.0f : 0.0f; \n" \
"p_Output[index + ch_in] = overlay == 0.0f ? p_Output[index + ch_in] : overlay; \n" \
"}}} \n" \
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

void CheckError(cl_int p_Error, const char* p_Msg) {
if (p_Error != CL_SUCCESS) {
fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
}}

void RunOpenCLKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Exp, 
float* p_Cont, float* p_Sat, float* p_Shad, float* p_Mid, float* p_High, float* p_Pivot, int p_Display)
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

cl_kernel Prepare = NULL;
cl_kernel FilmGradeKernelA = NULL;
cl_kernel FilmGradeKernelB = NULL;
cl_kernel FilmGradeKernelC = NULL;
cl_kernel FilmGradeKernelD = NULL;

Prepare = clCreateKernel(program, "k_Prepare", &error);
CheckError(error, "Unable to create kernel");

FilmGradeKernelA = clCreateKernel(program, "k_FilmGradeKernelA", &error);
CheckError(error, "Unable to create kernel");

FilmGradeKernelB = clCreateKernel(program, "k_FilmGradeKernelB", &error);
CheckError(error, "Unable to create kernel");

FilmGradeKernelC = clCreateKernel(program, "k_FilmGradeKernelC", &error);
CheckError(error, "Unable to create kernel");

FilmGradeKernelD = clCreateKernel(program, "k_FilmGradeKernelD", &error);
CheckError(error, "Unable to create kernel");

locker.Unlock();

localWorkSize[0] = 128;
localWorkSize[1] = 1;
globalWorkSize[0] = localWorkSize[0] * clDivUp(p_Width, localWorkSize[0]);
globalWorkSize[1] = p_Height;

error = clSetKernelArg(Prepare, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Prepare, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Prepare, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Prepare, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Prepare, 4, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FilmGradeKernelA, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FilmGradeKernelA, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(FilmGradeKernelA, 2, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FilmGradeKernelB, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FilmGradeKernelB, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(FilmGradeKernelB, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(FilmGradeKernelB, 6, sizeof(float), &p_Pivot[0]);
error |= clSetKernelArg(FilmGradeKernelB, 7, sizeof(float), &p_Pivot[1]);
error |= clSetKernelArg(FilmGradeKernelB, 8, sizeof(float), &p_Pivot[2]);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FilmGradeKernelC, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FilmGradeKernelC, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(FilmGradeKernelC, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(FilmGradeKernelC, 3, sizeof(float), &p_Cont[0]);
error |= clSetKernelArg(FilmGradeKernelC, 4, sizeof(float), &p_Cont[1]);
error |= clSetKernelArg(FilmGradeKernelC, 5, sizeof(float), &p_Cont[2]);
error |= clSetKernelArg(FilmGradeKernelC, 6, sizeof(float), &p_Sat[0]);
error |= clSetKernelArg(FilmGradeKernelC, 7, sizeof(float), &p_Sat[1]);
error |= clSetKernelArg(FilmGradeKernelC, 8, sizeof(float), &p_Sat[2]);
error |= clSetKernelArg(FilmGradeKernelC, 9, sizeof(float), &p_Pivot[2]);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FilmGradeKernelD, 0, sizeof(cl_mem), &p_Input);
error = clSetKernelArg(FilmGradeKernelD, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FilmGradeKernelD, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FilmGradeKernelD, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(FilmGradeKernelD, 5, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

clEnqueueNDRangeKernel(cmdQ, Prepare, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(FilmGradeKernelA, 3, sizeof(float), &p_Exp[c]);
error |= clSetKernelArg(FilmGradeKernelA, 4, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FilmGradeKernelA, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(FilmGradeKernelB, 3, sizeof(float), &p_Shad[c]);
error |= clSetKernelArg(FilmGradeKernelB, 4, sizeof(float), &p_Mid[c]);
error |= clSetKernelArg(FilmGradeKernelB, 5, sizeof(float), &p_High[c]);
error |= clSetKernelArg(FilmGradeKernelB, 9, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FilmGradeKernelB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
clEnqueueNDRangeKernel(cmdQ, FilmGradeKernelC, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

if (p_Display > 0) {
if(p_Display == 2) {
error |= clSetKernelArg(FilmGradeKernelA, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FilmGradeKernelB, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FilmGradeKernelC, 0, sizeof(cl_mem), &p_Input);
for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(FilmGradeKernelA, 3, sizeof(float), &p_Exp[c]);
error |= clSetKernelArg(FilmGradeKernelA, 4, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FilmGradeKernelA, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(FilmGradeKernelB, 3, sizeof(float), &p_Shad[c]);
error |= clSetKernelArg(FilmGradeKernelB, 4, sizeof(float), &p_Mid[c]);
error |= clSetKernelArg(FilmGradeKernelB, 5, sizeof(float), &p_High[c]);
error |= clSetKernelArg(FilmGradeKernelB, 9, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FilmGradeKernelB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
clEnqueueNDRangeKernel(cmdQ, FilmGradeKernelC, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(FilmGradeKernelD, 4, sizeof(float), &p_Pivot[c]);
error |= clSetKernelArg(FilmGradeKernelD, 6, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FilmGradeKernelD, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}}