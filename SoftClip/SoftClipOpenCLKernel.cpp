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
"__kernel void k_softclipKernel( __global const float* p_Input, __global float* p_Output, int p_Width, int p_Height,  \n" \
"float p_SoftClipA, float p_SoftClipB, float p_SoftClipC, float p_SoftClipD, float p_SoftClipE, float p_SoftClipF,  \n" \
"int p_SwitchA, int p_SwitchB, int p_Source) { \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if (x < p_Width && y < p_Height) { \n" \
"const int index = (y * p_Width + x) * 4;   \n" \
"float r = p_Input[index];    \n" \
"float g = p_Input[index + 1];    \n" \
"float b = p_Input[index + 2]; \n" \
"float cr = (pow(10.0f, (1023.0f * r - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"float cg = (pow(10.0f, (1023.0f * g - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"float cb = (pow(10.0f, (1023.0f * b - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"float lr = r > 0.1496582f ? (pow(10.0f, (r - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (r - 0.092809f) / 5.367655f; \n" \
"float lg = g > 0.1496582f ? (pow(10.0f, (g - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (g - 0.092809f) / 5.367655f; \n" \
"float lb = b > 0.1496582f ? (pow(10.0f, (b - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (b - 0.092809f) / 5.367655f; \n" \
"float mr = lr * 1.617523f  + lg * -0.537287f + lb * -0.080237f; \n" \
"float mg = lr * -0.070573f + lg * 1.334613f  + lb * -0.26404f; \n" \
"float mb = lr * -0.021102f + lg * -0.226954f + lb * 1.248056f; \n" \
"float sr = p_Source == 0 ? r : p_Source == 1 ? cr : mr; \n" \
"float sg = p_Source == 0 ? g : p_Source == 1 ? cg : mg; \n" \
"float sb = p_Source == 0 ? b : p_Source == 1 ? cb : mb; \n" \
"float Lr = sr > 1.0f ? 1.0f : sr; \n" \
"float Lg = sg > 1.0f ? 1.0f : sg; \n" \
"float Lb = sb > 1.0f ? 1.0f : sb; \n" \
"float Hr = (sr < 1.0f ? 1.0f : sr) - 1.0f; \n" \
"float Hg = (sg < 1.0f ? 1.0f : sg) - 1.0f; \n" \
"float Hb = (sb < 1.0f ? 1.0f : sb) - 1.0f; \n" \
"float rr = p_SoftClipA; \n" \
"float gg = p_SoftClipB; \n" \
"float aa = p_SoftClipC; \n" \
"float bb = p_SoftClipD; \n" \
"float ss = 1.0f - (p_SoftClipE / 10.0f); \n" \
"float sf = 1.0f - p_SoftClipF; \n" \
"float Hrr = Hr * pow(2.0f, rr); \n" \
"float Hgg = Hg * pow(2.0f, rr); \n" \
"float Hbb = Hb * pow(2.0f, rr); \n" \
"float HR = Hrr <= 1.0f ? 1.0f - pow(1.0f - Hrr, gg) : Hrr; \n" \
"float HG = Hgg <= 1.0f ? 1.0f - pow(1.0f - Hgg, gg) : Hgg; \n" \
"float HB = Hbb <= 1.0f ? 1.0f - pow(1.0f - Hbb, gg) : Hbb; \n" \
"float R = Lr + HR; \n" \
"float G = Lg + HG; \n" \
"float B = Lb + HB; \n" \
"float softr = aa == 1.0f ? R : (R > aa ? (-1.0f / ((R - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : R); \n" \
"float softR = bb == 1.0f ? softr : softr > 1.0f - (bb / 50.0f) ? (-1.0f / ((softr - (1.0f - (bb / 50.0f))) /  \n" \
"(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softr; \n" \
"float softg = (aa == 1.0f) ? G : (G > aa ? (-1.0f / ((G - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : G); \n" \
"float softG = bb == 1.0f ? softg : softg > 1.0f - (bb / 50.0f) ? (-1.0f / ((softg - (1.0f - (bb / 50.0f))) /  \n" \
"(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softg; \n" \
"float softb = (aa == 1.0f) ? B : (B > aa ? (-1.0f / ((B - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : B); \n" \
"float softB = bb == 1.0f ? softb : softb > 1.0f - (bb / 50.0f) ? (-1.0f / ((softb - (1.0f - (bb / 50.0f))) /  \n" \
"(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softb; \n" \
"float Cr = (softR * -1.0f) + 1.0f; \n" \
"float Cg = (softG * -1.0f) + 1.0f; \n" \
"float Cb = (softB * -1.0f) + 1.0f; \n" \
"float cR = ss == 1.0f ? Cr : Cr > ss ? (-1.0f / ((Cr - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cr; \n" \
"float CR = sf == 1.0f ? (cR - 1.0f) * -1.0f : ((cR > 1.0f - (-p_SoftClipF / 50.0f) ? (-1.0f / ((cR - (1.0f - (-p_SoftClipF / 50.0f))) /  \n" \
"(1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + (1.0f - (-p_SoftClipF / 50.0f)) : cR) - 1.0f) * -1.0f; \n" \
"float cG = ss == 1.0f ? Cg : Cg > ss ? (-1.0f / ((Cg - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cg; \n" \
"float CG = sf == 1.0f ? (cG - 1.0f) * -1.0f : ((cG > 1.0f - (-p_SoftClipF / 50.0f) ? (-1.0f / ((cG - (1.0f - (-p_SoftClipF / 50.0f))) /  \n" \
"(1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + (1.0f - (-p_SoftClipF / 50.0f)) : cG) - 1.0f) * -1.0f; \n" \
"float cB = ss == 1.0f ? Cb : Cb > ss ? (-1.0f / ((Cb - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cb; \n" \
"float CB = sf == 1.0f ? (cB - 1.0f) * -1.0f : ((cB > 1.0f - (-p_SoftClipF / 50.0f) ? (-1.0f / ((cB - (1.0f - (-p_SoftClipF / 50.0f))) /  \n" \
"(1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + (1.0f - (-p_SoftClipF / 50.0f)) : cB) - 1.0f) * -1.0f; \n" \
"float SR = p_Source == 0 ? CR : CR >= 0.0f && CR <= 1.0f ? (CR < 0.0181f ? (CR * 4.5f) : 1.0993f * pow(CR, 0.45f) - (1.0993f - 1.0f)) : CR; \n" \
"float SG = p_Source == 0 ? CG : CG >= 0.0f && CG <= 1.0f ? (CG < 0.0181f ? (CG * 4.5f) : 1.0993f * pow(CG, 0.45f) - (1.0993f - 1.0f)) : CG; \n" \
"float SB = p_Source == 0 ? CB : CB >= 0.0f && CB <= 1.0f ? (CB < 0.0181f ? (CB * 4.5f) : 1.0993f * pow(CB, 0.45f) - (1.0993f - 1.0f)) : CB; \n" \
"p_Output[index] = p_SwitchA == 1 ? (SR < 1.0f ? 1.0f : SR) - 1.0f : p_SwitchB == 1 ? (SR >= 0.0f ? 0.0f : SR + 1.0f) : SR; \n" \
"p_Output[index + 1] = p_SwitchA == 1 ? (SG < 1.0f ? 1.0f : SG) - 1.0f : p_SwitchB == 1 ? (SG >= 0.0f ? 0.0f : SG + 1.0f) : SG; \n" \
"p_Output[index + 2] = p_SwitchA == 1 ? (SB < 1.0f ? 1.0f : SB) - 1.0f : p_SwitchB == 1 ? (SB >= 0.0f ? 0.0f : SB + 1.0f) : SB; \n" \
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

int clDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

void CheckError(cl_int p_Error, const char* p_Msg) {
if (p_Error != CL_SUCCESS) {
fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
}}

void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, 
int p_Height, float* p_SoftClip, int* p_Switch, int p_Source)
{
cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);
static std::map<cl_command_queue, cl_device_id> deviceIdMap;
static std::map<cl_command_queue, cl_kernel> kernelMap;
static Locker locker;
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

cl_kernel SoftClipKernel = NULL;

SoftClipKernel = clCreateKernel(program, "k_softclipKernel", &error);
CheckError(error, "Unable to create kernel");

locker.Unlock();

localWorkSize[0] = 128;
localWorkSize[1] = 1;
globalWorkSize[0] = localWorkSize[0] * clDivUp(p_Width, localWorkSize[0]);
globalWorkSize[1] = p_Height;

error = clSetKernelArg(SoftClipKernel, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(SoftClipKernel, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(SoftClipKernel, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(SoftClipKernel, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(SoftClipKernel, 4, sizeof(float), &p_SoftClip[0]);
error |= clSetKernelArg(SoftClipKernel, 5, sizeof(float), &p_SoftClip[1]);
error |= clSetKernelArg(SoftClipKernel, 6, sizeof(float), &p_SoftClip[2]);
error |= clSetKernelArg(SoftClipKernel, 7, sizeof(float), &p_SoftClip[3]);
error |= clSetKernelArg(SoftClipKernel, 8, sizeof(float), &p_SoftClip[4]);
error |= clSetKernelArg(SoftClipKernel, 9, sizeof(float), &p_SoftClip[5]);
error |= clSetKernelArg(SoftClipKernel, 10, sizeof(int), &p_Switch[0]);
error |= clSetKernelArg(SoftClipKernel, 11, sizeof(int), &p_Switch[1]);
error |= clSetKernelArg(SoftClipKernel, 12, sizeof(int), &p_Source);
CheckError(error, "Unable to set kernel arguments");

clEnqueueNDRangeKernel(cmdQ, SoftClipKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}