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
"#define  BLOCKSIZE 4 \n" \
"__kernel void SoftClipAdjustKernel(  \n" \
"   int p_Width,        \n" \
"   int p_Height,       \n" \
"   float p_SoftClipA,  \n" \
"   float p_SoftClipB,  \n" \
"   float p_SoftClipC,  \n" \
"   float p_SoftClipD,  \n" \
"   float p_SoftClipE,  \n" \
"   float p_SoftClipF,  \n" \
"   float p_SwitchA,    \n" \
"   float p_SwitchB,    \n" \
"   float p_SourceA,    \n" \
"   float p_SourceB,    \n" \
"   float p_SourceC,    \n" \
"   __global const float* p_Input,  \n" \
"   __global float* p_Output)       \n" \
"{      \n" \
"   float SRC[BLOCKSIZE]; \n" \
"   float w_SoftClipA;  \n" \
"   float w_SoftClipB;  \n" \
"   float w_SoftClipC;  \n" \
"   float w_SoftClipD;  \n" \
"   float w_SoftClipE;  \n" \
"   float w_SoftClipF;  \n" \
"   float w_SwitchA;    \n" \
"   float w_SwitchB;    \n" \
"   float w_SourceA;    \n" \
"   float w_SourceB;    \n" \
"   float w_SourceC;    \n" \
"   float r;   \n" \
"   float g;   \n" \
"   float b;   \n" \
"   float cr;  \n" \
"   float cg;  \n" \
"   float cb;  \n" \
"   float lr;  \n" \
"   float lg;  \n" \
"   float lb;  \n" \
"   float mr;  \n" \
"   float mg;  \n" \
"   float mb;  \n" \
"   float sr;  \n" \
"   float sg;  \n" \
"   float sb;  \n" \
"   float Lr;  \n" \
"   float Lg;  \n" \
"   float Lb;  \n" \
"   float Hr;  \n" \
"   float Hg;  \n" \
"   float Hb;  \n" \
"   float rr;  \n" \
"   float gg;  \n" \
"   float aa;  \n" \
"   float bb;  \n" \
"   float ss;  \n" \
"   float sf;  \n" \
"   float Hrr; \n" \
"   float Hgg; \n" \
"   float Hbb; \n" \
"   float HR;  \n" \
"   float HG;  \n" \
"   float HB;  \n" \
"   float R;   \n" \
"   float G;   \n" \
"   float B;   \n" \
"   float softr;  \n" \
"   float softR;  \n" \
"   float softg;  \n" \
"   float softG;  \n" \
"   float softb;  \n" \
"   float softB;  \n" \
"   float Cr;  \n" \
"   float cR;  \n" \
"   float CR;  \n" \
"   float Cg;  \n" \
"   float cG;  \n" \
"   float CG;  \n" \
"   float Cb;  \n" \
"   float cB;  \n" \
"   float CB;  \n" \
"   float SR;  \n" \
"   float SG;  \n" \
"   float SB;  \n" \
"   const int x = get_global_id(0); \n" \
"   const int y = get_global_id(1); \n" \
"   if ((x < p_Width) && (y < p_Height))                                \n" \
"   {   \n" \
"      const int index = ((y * p_Width) + x) * BLOCKSIZE;   \n" \
"      SRC[0] = p_Input[index + 0];    \n" \
"      SRC[1] = p_Input[index + 1];    \n" \
"      SRC[2] = p_Input[index + 2];    \n" \
"      SRC[3] = p_Input[index + 3];    \n" \
"      w_SoftClipA = p_SoftClipA;  \n" \
"      w_SoftClipB = p_SoftClipB;  \n" \
"      w_SoftClipC = p_SoftClipC;  \n" \
"      w_SoftClipD = p_SoftClipD;  \n" \
"      w_SoftClipE = p_SoftClipE;  \n" \
"      w_SoftClipF = p_SoftClipF;  \n" \
"      w_SwitchA   = p_SwitchA;    \n" \
"      w_SwitchB   = p_SwitchB;    \n" \
"      w_SourceA   = p_SourceA;    \n" \
"      w_SourceB   = p_SourceB;    \n" \
"      w_SourceC   = p_SourceC;    \n" \
"      r = SRC[0];  \n" \
"      g = SRC[1];  \n" \
"      b = SRC[2];  \n" \
"                   \n" \
"      cr = (pow(10.0f, (1023.0f * r - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"      cg = (pow(10.0f, (1023.0f * g - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"      cb = (pow(10.0f, (1023.0f * b - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"                   \n" \
"      lr = r > 0.1496582f ? (pow(10.0f, (r - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (r - 0.092809f) / 5.367655f;  \n" \
"      lg = g > 0.1496582f ? (pow(10.0f, (g - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (g - 0.092809f) / 5.367655f;  \n" \
"      lb = b > 0.1496582f ? (pow(10.0f, (b - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (b - 0.092809f) / 5.367655f;  \n" \
"                   \n" \
"      mr = (lr * 1.617523f)  + (lg * -0.537287f) + (lb * -0.080237f); \n" \
"      mg = (lr * -0.070573f) + (lg * 1.334613f)  + (lb * -0.26404f);  \n" \
"      mb = (lr * -0.021102f) + (lg * -0.226954f) + (lb * 1.248056f);  \n" \
"                   \n" \
"      sr = w_SourceA == 1.0f ? r : w_SourceB == 1.0f ? cr : mr; \n" \
"      sg = w_SourceA == 1.0f ? g : w_SourceB == 1.0f ? cg : mg; \n" \
"      sb = w_SourceA == 1.0f ? b : w_SourceB == 1.0f ? cb : mb; \n" \
"                   \n" \
"      Lr = sr > 1.0f ? 1.0f : sr;  \n"\
"      Lg = sg > 1.0f ? 1.0f : sg;  \n"\
"      Lb = sb > 1.0f ? 1.0f : sb;  \n"\
"                   \n" \
"      Hr = (sr < 1.0f ? 1.0f : sr) - 1.0f; \n" \
"      Hg = (sg < 1.0f ? 1.0f : sg) - 1.0f; \n" \
"      Hb = (sb < 1.0f ? 1.0f : sb) - 1.0f; \n" \
"                   \n" \
"      rr = w_SoftClipA; \n" \
"      gg = w_SoftClipB; \n" \
"      aa = w_SoftClipC; \n" \
"      bb = w_SoftClipD; \n" \
"	   ss = 1.0f - (w_SoftClipE / 10.0f); \n" \
"	   sf = 1.0f - w_SoftClipF; \n" \
"                   \n" \
"      Hrr = Hr * pow(2.0f, rr); \n" \
"      Hgg = Hg * pow(2.0f, rr); \n" \
"      Hbb = Hb * pow(2.0f, rr); \n" \
"                   \n" \
"	   HR = Hrr <= 1.0f ? 1.0f - pow(1.0f - Hrr, gg) : Hrr;	\n" \
"	   HG = Hgg <= 1.0f ? 1.0f - pow(1.0f - Hgg, gg) : Hgg;	\n" \
"	   HB = Hbb <= 1.0f ? 1.0f - pow(1.0f - Hbb, gg) : Hbb;	\n" \
"                   \n" \
"      R = Lr + HR; \n" \
"      G = Lg + HG; \n" \
"      B = Lb + HB; \n" \
"                   \n" \
"	   softr = aa == 1.0f ? R : (R > aa ? (-1.0f / ((R - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : R); \n" \
"	   softR = bb == 1.0f ? softr : softr > 1.0f - (bb / 50.0f) ? (-1.0f / ((softr - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) +  \n" \
"	   1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softr; \n" \
"	   softg = (aa == 1.0f) ? G : (G > aa ? (-1.0f / ((G - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : G); \n" \
"	   softG = bb == 1.0f ? softg : softg > 1.0f - (bb / 50.0f) ? (-1.0f / ((softg - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) +  \n" \
"	   1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softg; \n" \
"	   softb = (aa == 1.0f) ? B : (B > aa ? (-1.0f / ((B - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : B); \n" \
"	   softB = bb == 1.0f ? softb : softb > 1.0f - (bb / 50.0f) ? (-1.0f / ((softb - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) +  \n" \
"	   1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softb; \n" \
" \n" \
"	   Cr = (softR * -1.0f) + 1.0f; \n" \
"	   Cg = (softG * -1.0f) + 1.0f; \n" \
"	   Cb = (softB * -1.0f) + 1.0f; \n" \
" \n" \
"	   cR = ss == 1.0f ? Cr : Cr > ss ? (-1.0f / ((Cr - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cr; \n" \
"	   CR = sf == 1.0f ? (cR - 1.0f) * -1.0f : ((cR > 1.0f - (-w_SoftClipF / 50.0f) ? (-1.0f / ((cR - (1.0f - (-w_SoftClipF / 50.0f))) /  \n" \
"	   (1.0f - (1.0f - (-w_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-w_SoftClipF / 50.0f))) + (1.0f - (-w_SoftClipF / 50.0f)) : cR) - 1.0f) * -1.0f; \n" \
"	   cG = ss == 1.0f ? Cg : Cg > ss ? (-1.0f / ((Cg - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cg; \n" \
"	   CG = sf == 1.0f ? (cG - 1.0f) * -1.0f : ((cG > 1.0f - (-w_SoftClipF / 50.0f) ? (-1.0f / ((cG - (1.0f - (-w_SoftClipF / 50.0f))) /  \n" \
"	   (1.0f - (1.0f - (-w_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-w_SoftClipF / 50.0f))) + (1.0f - (-w_SoftClipF / 50.0f)) : cG) - 1.0f) * -1.0f; \n" \
"	   cB = ss == 1.0f ? Cb : Cb > ss ? (-1.0f / ((Cb - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cb; \n" \
"	   CB = sf == 1.0f ? (cB - 1.0f) * -1.0f : ((cB > 1.0f - (-w_SoftClipF / 50.0f) ? (-1.0f / ((cB - (1.0f - (-w_SoftClipF / 50.0f))) /  \n" \
"	   (1.0f - (1.0f - (-w_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-w_SoftClipF / 50.0f))) + (1.0f - (-w_SoftClipF / 50.0f)) : cB) - 1.0f) * -1.0f; \n" \
"		   \n" \
"	   SR = w_SourceA == 1.0f ? CR : CR >= 0.0f && CR <= 1.0f ? (CR < 0.0181f ? (CR * 4.5f) : 1.0993f * pow(CR, 0.45f) - (1.0993f - 1.0f)) : CR; \n" \
"	   SG = w_SourceA == 1.0f ? CG : CG >= 0.0f && CG <= 1.0f ? (CG < 0.0181f ? (CG * 4.5f) : 1.0993f * pow(CG, 0.45f) - (1.0993f - 1.0f)) : CG; \n" \
"	   SB = w_SourceA == 1.0f ? CB : CB >= 0.0f && CB <= 1.0f ? (CB < 0.0181f ? (CB * 4.5f) : 1.0993f * pow(CB, 0.45f) - (1.0993f - 1.0f)) : CB; \n" \
" \n" \
"	   SRC[0] = w_SwitchA == 1.0f ? (SR < 1.0f ? 1.0f : SR) - 1.0f : w_SwitchB == 1.0f ? (SR >= 0.0f ? 0.0f : SR + 1.0f) : SR; \n" \
"	   SRC[1] = w_SwitchA == 1.0f ? (SG < 1.0f ? 1.0f : SG) - 1.0f : w_SwitchB == 1.0f ? (SG >= 0.0f ? 0.0f : SG + 1.0f) : SG; \n" \
"	   SRC[2] = w_SwitchA == 1.0f ? (SB < 1.0f ? 1.0f : SB) - 1.0f : w_SwitchB == 1.0f ? (SB >= 0.0f ? 0.0f : SB + 1.0f) : SB; \n" \
" \n" \
"       p_Output[index + 0] = SRC[0];  \n" \
"       p_Output[index + 1] = SRC[1];  \n" \
"       p_Output[index + 2] = SRC[2];  \n" \
"       p_Output[index + 3] = SRC[3];  \n" \
"   }  \n" \
"}     \n" \
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

void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_SoftClip, float* p_Switch, float* p_Source, const float* p_Input, float* p_Output)
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

    //#define _DEBUG


	// find the program kernel corresponding to the command queue
	cl_kernel kernel = NULL;
	if (kernelMap.find(cmdQ) == kernelMap.end())
	{
		cl_context clContext = NULL;
		error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
		CheckError(error, "Unable to get the context");

		cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
		CheckError(error, "Unable to create program");

		error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#ifdef _DEBUG
		if (error != CL_SUCCESS)
		{
			char buffer[4096];
			size_t length;
			clGetProgramBuildInfo
				(
				program,
				// valid program object
				deviceId,
				// valid device_id that executable was built
				CL_PROGRAM_BUILD_LOG,
				// indicate to retrieve build log
				sizeof(buffer),
				// size of the buffer to write log to
				buffer,
				// the actual buffer to write log to
				&length);
			// the actual size in bytes of data copied to buffer
			FILE * pFile;
			pFile = fopen("/", "w");
			if (pFile != NULL)
			{
				fprintf(pFile, "%s\n", buffer);
				//fprintf(pFile, "%s [%lu]\n", "localWorkSize 0 =", szWorkSize);
			}
			fclose(pFile);
		}
#else
		CheckError(error, "Unable to build program");
#endif

		kernel = clCreateKernel(program, "SoftClipAdjustKernel", &error);
		CheckError(error, "Unable to create kernel");

		kernelMap[cmdQ] = kernel;
	}
	else
	{
		kernel = kernelMap[cmdQ];
	}

	locker.Unlock();

    int count = 0;
    error  = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_SoftClip[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_SoftClip[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_SoftClip[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_SoftClip[3]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_SoftClip[4]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_SoftClip[5]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Source[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Source[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Source[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Output);
    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
