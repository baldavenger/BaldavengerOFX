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
"float Luma(float R, float G, float B, int L); \n" \
"void RGB_to_HSV( float r, float g, float b, float *h, float *s, float *v ); \n" \
"void HSV_to_RGB(float H, float S, float V, float *r, float *g, float *b); \n" \
"void Temperature(float *R, float *G, float *B, float Temp); \n" \
"void VideoGradeKernelA(float R, float G, float B, float *RR, float *GG, float *BB, int p_LumaMath, int p_Gang,  \n" \
"int p_GammaBias, float p_Exposure, float p_Temp, float p_Tint, float p_Hue, float p_Sat, float p_GainR, float p_GainG, float p_GainB,  \n" \
"float p_GainAnchor, float p_LiftR, float p_LiftG, float p_LiftB, float p_LiftAnchor, float p_OffsetR, float p_OffsetG, float p_OffsetB,  \n" \
"float p_GammaR, float p_GammaG, float p_GammaB, float p_GammaStart, float p_GammaEnd); \n" \
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
"void RGB_to_HSV( float r, float g, float b, float *h, float *s, float *v ) \n" \
"{ \n" \
"float min = fmin(fmin(r, g), b); \n" \
"float max = fmax(fmax(r, g), b); \n" \
"*v = max; \n" \
"float delta = max - min; \n" \
"if (max != 0.) { \n" \
"*s = delta / max; \n" \
"} else { \n" \
"*s = 0.f; \n" \
"*h = 0.f; \n" \
"return; \n" \
"} \n" \
"if (delta == 0.) { \n" \
"*h = 0.f; \n" \
"} else if (r == max) { \n" \
"*h = (g - b) / delta; \n" \
"} else if (g == max) { \n" \
"*h = 2 + (b - r) / delta; \n" \
"} else { \n" \
"*h = 4 + (r - g) / delta; \n" \
"} \n" \
"*h *= 1.0f / 6.; \n" \
"if (*h < 0) { \n" \
"*h += 1.0f; \n" \
"} \n" \
"} \n" \
" \n" \
"void HSV_to_RGB(float H, float S, float V, float *r, float *g, float *b) \n" \
"{ \n" \
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
"void Temperature(float *R, float *G, float *B, float Temp){ \n" \
"float r, g, b; \n" \
"if (Temp <= 66.0f){ \n" \
"r = 255.0f; \n" \
"} else { \n" \
"r = Temp - 60.0f; \n" \
"r = 329.698727446f * pow(r, -0.1332047592f); \n" \
"if(r < 0.0f){r = 0.0f;} \n" \
"if(r > 255.0f){r = 255.0f;} \n" \
"} \n" \
"if (Temp <= 66.0f){ \n" \
"g = Temp; \n" \
"g = 99.4708025861f * log(g) - 161.1195681661f; \n" \
"if(g < 0.0f){g = 0.0f;} \n" \
"if(g > 255.0f){g = 255.0f;} \n" \
"} else { \n" \
"g = Temp - 60.0f; \n" \
"g = 288.1221695283f * pow(g, -0.0755148492f); \n" \
"if(g < 0.0f){g = 0.0f;} \n" \
"if(g > 255.0f){g = 255.0f;} \n" \
"} \n" \
"if(Temp >= 66.0f){ \n" \
"b = 255.0f; \n" \
"} else { \n" \
"if(Temp <= 19.0f){ \n" \
"b = 0.0f; \n" \
"} else { \n" \
"b = Temp - 10.0f; \n" \
"b = 138.5177312231f * log(b) - 305.0447927307f; \n" \
"if(b < 0.0f){b = 0.0f;} \n" \
"if(b > 255.0f){b = 255.0f;} \n" \
"} \n" \
"} \n" \
"*R = r / 255.0f; \n" \
"*G = g / 255.0f; \n" \
"*B = b / 255.0f; \n" \
"} \n" \
"void VideoGradeKernelA(float R, float G, float B, float *RR, float *GG, float *BB, int p_LumaMath, int p_Gang,  \n" \
"int p_GammaBias, float p_Exposure, float p_Temp, float p_Tint, float p_Hue, float p_Sat, float p_GainR, float p_GainG, float p_GainB,  \n" \
"float p_GainAnchor, float p_LiftR, float p_LiftG, float p_LiftB, float p_LiftAnchor, float p_OffsetR, float p_OffsetG, float p_OffsetB,  \n" \
"float p_GammaR, float p_GammaG, float p_GammaB, float p_GammaStart, float p_GammaEnd){ \n" \
"if(p_Gang == 1){ \n" \
"p_GainB = p_GainG = p_GainR; \n" \
"p_LiftB = p_LiftG = p_LiftR; \n" \
"p_OffsetB = p_OffsetG = p_OffsetR; \n" \
"p_GammaB = p_GammaG = p_GammaR; \n" \
"} \n" \
"float Temp1 = (p_Temp / 100.0f) + 1.0f; \n" \
"if(p_Exposure != 0.0f){ \n" \
"R = R * pow(2.0f, p_Exposure); \n" \
"G = G * pow(2.0f, p_Exposure); \n" \
"B = B * pow(2.0f, p_Exposure); \n" \
"} \n" \
"if(Temp1 != 66.0f){ \n" \
"float r, g, b, R1, G1, B1, templuma1, templuma2; \n" \
"templuma1 = Luma(R, G, B, p_LumaMath); \n" \
"Temperature(&r, &g, &b, Temp1); \n" \
"R1 = R * r; \n" \
"G1 = G * g; \n" \
"B1 = B * b; \n" \
"templuma2 = Luma(R1, G1, B1, p_LumaMath); \n" \
"R = R1 / (templuma2 / templuma1); \n" \
"G = G1 / (templuma2 / templuma1); \n" \
"B = B1 / (templuma2 / templuma1); \n" \
"} \n" \
"if(p_Tint != 0.0f){ \n" \
"float tintluma1 = Luma(R, G, B, p_LumaMath); \n" \
"float R1 = R * (1 + (p_Tint / 2)); \n" \
"float B1 = B * (1 + (p_Tint / 2)); \n" \
"float tintluma2 = Luma(R1, G, B1, p_LumaMath); \n" \
"float tintluma3 = tintluma2 / tintluma1; \n" \
"R = R1 / tintluma3; \n" \
"G = G / tintluma3; \n" \
"B = B1 / tintluma3; \n" \
"} \n" \
"if(p_Hue != 0.0f || p_Sat != 0.0f){ \n" \
"p_Hue /= 360.0f; \n" \
"float h, s, v; \n" \
"RGB_to_HSV(R, G, B, &h, &s, &v); \n" \
"float h2 = h + p_Hue; \n" \
"float H2 = h2 < 0.0f ? h2 + 1.0f : h2 >= 1.0f ? h2 - 1.0f : h2; \n" \
"float S = s * (1.0f + p_Sat); \n" \
"HSV_to_RGB(H2, S, v, &R, &G, &B); \n" \
"} \n" \
"if(p_GainR != 1.0f){ \n" \
"R = R >= p_GainAnchor ? (R - p_GainAnchor) * p_GainR  + p_GainAnchor: R; \n" \
"} \n" \
"if(p_GainG != 1.0f){ \n" \
"G = G >= p_GainAnchor ? (G - p_GainAnchor) * p_GainG  + p_GainAnchor: G; \n" \
"} \n" \
"if(p_GainB != 1.0f){ \n" \
"B = B >= p_GainAnchor ? (B - p_GainAnchor) * p_GainB  + p_GainAnchor: B; \n" \
"} \n" \
"if(p_LiftR != 0.0f){ \n" \
"R = R <= p_LiftAnchor ? ((R / p_LiftAnchor) + p_LiftR * (1.0f - (R / p_LiftAnchor))) * p_LiftAnchor : R; \n" \
"} \n" \
"if(p_LiftG != 0.0f){ \n" \
"G = G <= p_LiftAnchor ? ((G / p_LiftAnchor) + p_LiftG * (1.0f - (G / p_LiftAnchor))) * p_LiftAnchor : G; \n" \
"} \n" \
"if(p_LiftB != 0.0f){ \n" \
"B = B <= p_LiftAnchor ? ((B / p_LiftAnchor) + p_LiftB * (1.0f - (B / p_LiftAnchor))) * p_LiftAnchor : B; \n" \
"} \n" \
"if(p_OffsetR != 0.0f){ \n" \
"R += p_OffsetR; \n" \
"} \n" \
"if(p_OffsetG != 0.0f){ \n" \
"G += p_OffsetG; \n" \
"} \n" \
"if(p_OffsetB != 0.0f){ \n" \
"B += p_OffsetB; \n" \
"} \n" \
"if(p_GammaR != 0.0f){ \n" \
"float Prl = R >= p_GammaStart && R <= p_GammaEnd ? pow((R - p_GammaStart) / (p_GammaEnd - p_GammaStart), 1.0f / p_GammaR) * (p_GammaEnd - p_GammaStart) + p_GammaStart : R; \n" \
"float Pru = R >= p_GammaStart && R <= p_GammaEnd ? (1.0f - pow(1.0f - (R - p_GammaStart) / (p_GammaEnd - p_GammaStart), p_GammaR)) * (p_GammaEnd - p_GammaStart) + p_GammaStart : R; \n" \
"R = p_GammaBias == 1 ? Pru : Prl; \n" \
"} \n" \
"if(p_GammaG != 0.0f){ \n" \
"float Pgl = G >= p_GammaStart && G <= p_GammaEnd ? pow((G - p_GammaStart) / (p_GammaEnd - p_GammaStart), 1.0f / p_GammaG) * (p_GammaEnd - p_GammaStart) + p_GammaStart : G; \n" \
"float Pgu = G >= p_GammaStart && G <= p_GammaEnd ? (1.0f - pow(1.0f - (G - p_GammaStart) / (p_GammaEnd - p_GammaStart), p_GammaG)) * (p_GammaEnd - p_GammaStart) + p_GammaStart : G; \n" \
"G = p_GammaBias == 1 ? Pgu : Pgl; \n" \
"} \n" \
"if(p_GammaB != 0.0f){ \n" \
"float Pbl = B >= p_GammaStart && B <= p_GammaEnd ? pow((B - p_GammaStart) / (p_GammaEnd - p_GammaStart), 1.0f / p_GammaB) * (p_GammaEnd - p_GammaStart) + p_GammaStart : B; \n" \
"float Pbu = B >= p_GammaStart && B <= p_GammaEnd ? (1.0f - pow(1.0f - (B - p_GammaStart) / (p_GammaEnd - p_GammaStart), p_GammaB)) * (p_GammaEnd - p_GammaStart) + p_GammaStart : B; \n" \
"B = p_GammaBias == 1 ? Pbu : Pbl; \n" \
"}						 \n" \
"*RR = R; \n" \
"*GG = G; \n" \
"*BB = B; \n" \
"} \n" \
"__kernel void VideoGradeKernel(__global const float* p_Input, __global float* p_Output, int p_Width, int p_Height, int p_LumaMath, int p_Gang,  \n" \
"int p_GammaBias, int p_DisplayA, int p_DisplayB, float p_Exposure, float p_Temp, float p_Tint, float p_Hue, float p_Sat,  \n" \
"float p_GainR, float p_GainG, float p_GainB, float p_GainAnchor, float p_LiftR, float p_LiftG, float p_LiftB, float p_LiftAnchor,  \n" \
"float p_OffsetR, float p_OffsetG, float p_OffsetB, float p_GammaR, float p_GammaG, float p_GammaB, float p_GammaStart, float p_GammaEnd) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float r1, g1, b1, R, G, B, r2, g2, b2, R1, G1, B1; \n" \
"R = p_Input[index + 0]; \n" \
"G = p_Input[index + 1]; \n" \
"B = p_Input[index + 2]; \n" \
"float width = p_Width; \n" \
"float height = p_Height; \n" \
"R1 = x / (width - 1); \n" \
"B1 = G1 = R1; \n" \
"if(p_DisplayA == 0){ \n" \
"VideoGradeKernelA(R, G, B, &r1, &g1, &b1, p_LumaMath, p_Gang, p_GammaBias, p_Exposure, p_Temp, p_Tint, p_Hue, p_Sat,  \n" \
"p_GainR, p_GainG, p_GainB, p_GainAnchor, p_LiftR, p_LiftG, p_LiftB, p_LiftAnchor, p_OffsetR, p_OffsetG, p_OffsetB,  \n" \
"p_GammaR, p_GammaG, p_GammaB, p_GammaStart, p_GammaEnd); \n" \
"} \n" \
"if(p_DisplayA == 1 && p_DisplayB == 0){ \n" \
"VideoGradeKernelA(R1, G1, B1, &r2, &g2, &b2, p_LumaMath, p_Gang, p_GammaBias, p_Exposure, p_Temp, p_Tint, p_Hue, p_Sat,  \n" \
"p_GainR, p_GainG, p_GainB, p_GainAnchor, p_LiftR, p_LiftG, p_LiftB, p_LiftAnchor, p_OffsetR, p_OffsetG, p_OffsetB,  \n" \
"p_GammaR, p_GammaG, p_GammaB, p_GammaStart, p_GammaEnd); \n" \
"r1 = r2 >= (y - 5)/(height) && r2 <= (y + 5)/(height) ? 1.0f : 0.0f; \n" \
"g1 = g2 >= (y - 5)/(height) && g2 <= (y + 5)/(height) ? 1.0f : 0.0f; \n" \
"b1 = b2 >= (y - 5)/(height) && b2 <= (y + 5)/(height) ? 1.0f : 0.0f; \n" \
"} \n" \
"if(p_DisplayA == 1 && p_DisplayB == 1){ \n" \
"VideoGradeKernelA(R, G, B, &r1, &g1, &b1, p_LumaMath, p_Gang, p_GammaBias, p_Exposure, p_Temp, p_Tint, p_Hue, p_Sat,  \n" \
"p_GainR, p_GainG, p_GainB, p_GainAnchor, p_LiftR, p_LiftG, p_LiftB, p_LiftAnchor, p_OffsetR, p_OffsetG, p_OffsetB,  \n" \
"p_GammaR, p_GammaG, p_GammaB, p_GammaStart, p_GammaEnd); \n" \
"VideoGradeKernelA(R1, G1, B1, &r2, &g2, &b2, p_LumaMath, p_Gang, p_GammaBias, p_Exposure, p_Temp, p_Tint, p_Hue, p_Sat,  \n" \
"p_GainR, p_GainG, p_GainB, p_GainAnchor, p_LiftR, p_LiftG, p_LiftB, p_LiftAnchor, p_OffsetR, p_OffsetG, p_OffsetB,  \n" \
"p_GammaR, p_GammaG, p_GammaB, p_GammaStart, p_GammaEnd); \n" \
"r2 = r2 >= (y - 5)/(height) && r2 <= (y + 5)/(height) ? 1.0f : 0.0f; \n" \
"g2 = g2 >= (y - 5)/(height) && g2 <= (y + 5)/(height) ? 1.0f : 0.0f; \n" \
"b2 = b2 >= (y - 5)/(height) && b2 <= (y + 5)/(height) ? 1.0f : 0.0f; \n" \
"r1 = r2 == 0.0f ? r1 : r2; \n" \
"g1 = g2 == 0.0f ? g1 : g2; \n" \
"b1 = b2 == 0.0f ? b1 : b2; \n" \
"} \n" \
"p_Output[index + 0] = r1; \n" \
"p_Output[index + 1] = g1; \n" \
"p_Output[index + 2] = b1; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
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

void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_LumaMath, int* p_Switch, float* p_Scale)
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

	  cl_context clContext = NULL;
	  error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
	  CheckError(error, "Unable to get the context");

	  cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
	  CheckError(error, "Unable to create program");

	  error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	  CheckError(error, "Unable to build program");
	  
	  cl_kernel kernel = NULL;

	  kernel = clCreateKernel(program, "VideoGradeKernel", &error);
	  CheckError(error, "Unable to create kernel");

	locker.Unlock();

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &p_Output);
    error |= clSetKernelArg(kernel, 2, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, 3, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, 4, sizeof(int), &p_LumaMath);
    error |= clSetKernelArg(kernel, 5, sizeof(int), &p_Switch[0]);
    error |= clSetKernelArg(kernel, 6, sizeof(int), &p_Switch[1]);
    error |= clSetKernelArg(kernel, 7, sizeof(int), &p_Switch[2]);
    error |= clSetKernelArg(kernel, 8, sizeof(int), &p_Switch[3]);
    error |= clSetKernelArg(kernel, 9, sizeof(float), &p_Scale[0]);
    error |= clSetKernelArg(kernel, 10, sizeof(float), &p_Scale[1]);
    error |= clSetKernelArg(kernel, 11, sizeof(float), &p_Scale[2]);
    error |= clSetKernelArg(kernel, 12, sizeof(float), &p_Scale[3]);
    error |= clSetKernelArg(kernel, 13, sizeof(float), &p_Scale[4]);
    error |= clSetKernelArg(kernel, 14, sizeof(float), &p_Scale[5]);
    error |= clSetKernelArg(kernel, 15, sizeof(float), &p_Scale[6]);
    error |= clSetKernelArg(kernel, 16, sizeof(float), &p_Scale[7]);
    error |= clSetKernelArg(kernel, 17, sizeof(float), &p_Scale[8]);
    error |= clSetKernelArg(kernel, 18, sizeof(float), &p_Scale[9]);
    error |= clSetKernelArg(kernel, 19, sizeof(float), &p_Scale[10]);
    error |= clSetKernelArg(kernel, 20, sizeof(float), &p_Scale[11]);
    error |= clSetKernelArg(kernel, 21, sizeof(float), &p_Scale[12]);
    error |= clSetKernelArg(kernel, 22, sizeof(float), &p_Scale[13]);
    error |= clSetKernelArg(kernel, 23, sizeof(float), &p_Scale[14]);
    error |= clSetKernelArg(kernel, 24, sizeof(float), &p_Scale[15]);
    error |= clSetKernelArg(kernel, 25, sizeof(float), &p_Scale[16]);
    error |= clSetKernelArg(kernel, 26, sizeof(float), &p_Scale[17]);
    error |= clSetKernelArg(kernel, 27, sizeof(float), &p_Scale[18]);
    error |= clSetKernelArg(kernel, 28, sizeof(float), &p_Scale[19]);
    error |= clSetKernelArg(kernel, 29, sizeof(float), &p_Scale[20]);
    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
