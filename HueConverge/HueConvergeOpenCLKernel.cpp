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
"__kernel void HueConvergeKernel(  \n" \
"   int p_Width,      \n" \
"   int p_Height,     \n" \
"   float p_SwitchA,  \n" \
"   float p_SwitchB,  \n" \
"   float p_SwitchC,  \n" \
"   float p_SwitchD,  \n" \
"   float p_SwitchE,  \n" \
"   float p_SwitchL,  \n" \
"   float p_Alpha1,  \n" \
"   float p_Alpha2,  \n" \
"   float p_LogA,     \n" \
"   float p_LogB,     \n" \
"   float p_LogC,     \n" \
"   float p_LogD,     \n" \
"   float p_SatA,     \n" \
"   float p_SatB,     \n" \
"   float p_SatC,     \n" \
"   float p_HueA,     \n" \
"   float p_HueB,     \n" \
"   float p_HueC,     \n" \
"   float p_HueR1,    \n" \
"   float p_HueP1,    \n" \
"   float p_HueSH1,   \n" \
"   float p_HueSHP1,  \n" \
"   float p_HueR2,    \n" \
"   float p_HueP2,    \n" \
"   float p_HueSH2,   \n" \
"   float p_HueSHP2,  \n" \
"   float p_HueR3,    \n" \
"   float p_HueP3,    \n" \
"   float p_HueSH3,   \n" \
"   float p_HueSHP3,  \n" \
"   float p_LumaA,  \n" \
"   float p_LumaB,  \n" \
"   float p_LumaC,  \n" \
"   __global const float* p_Input, \n" \
"   __global float* p_Output)      \n" \
"{                                 \n" \
"   float SRC[BLOCKSIZE]; \n" \
"   float w_SwitchA;  \n" \
"   float w_SwitchB;  \n" \
"   float w_SwitchC;  \n" \
"   float w_SwitchD;  \n" \
"   float w_SwitchE;  \n" \
"   float w_SwitchL;  \n" \
"   float w_Alpha1;  \n" \
"   float w_Alpha2;  \n" \
"   float w_LogA;     \n" \
"   float w_LogB;     \n" \
"   float w_LogC;     \n" \
"   float w_LogD;     \n" \
"   float w_SatA;     \n" \
"   float w_SatB;     \n" \
"   float w_SatC;     \n" \
"   float w_HueA;     \n" \
"   float w_HueB;     \n" \
"   float w_HueC;     \n" \
"   float w_HueR1;    \n" \
"   float w_HueP1;    \n" \
"   float w_HueSH1;   \n" \
"   float w_HueSHP1;  \n" \
"   float w_HueR2;    \n" \
"   float w_HueP2;    \n" \
"   float w_HueSH2;   \n" \
"   float w_HueSHP2;  \n" \
"   float w_HueR3;    \n" \
"   float w_HueP3;    \n" \
"   float w_HueSH3;   \n" \
"   float w_HueSHP3;  \n" \
"   float w_LumaA;  \n" \
"   float w_LumaB;  \n" \
"   float w_LumaC;  \n" \
"   float Lr;        \n" \
"   float Lg;        \n" \
"   float Lb;        \n" \
"   float mn;        \n" \
"   float Mx;        \n" \
"   float del_Mx;    \n" \
"   float L;         \n" \
"   float luma;      \n" \
"   float Sat;       \n" \
"   float del_R;     \n" \
"   float del_G;     \n" \
"   float del_B;     \n" \
"   float h;         \n" \
"   float Hh;        \n" \
"   float s;         \n" \
"   float ss;        \n" \
"   float satAlphaA; \n" \
"   float satAlpha;  \n" \
"   float S;         \n" \
"   float h1;        \n" \
"   float Hs1;       \n" \
"   float cv1;       \n" \
"   float curve1;    \n" \
"   float H1;        \n" \
"   float h2;        \n" \
"   float Hs2;       \n" \
"   float cv2;       \n" \
"   float curve2;    \n" \
"   float H2;        \n" \
"   float h3;        \n" \
"   float Hs3;       \n" \
"   float cv3;       \n" \
"   float curve3;    \n" \
"   float H;         \n" \
"   float Q;         \n" \
"   float P;         \n" \
"   float RH;        \n" \
"   float RR;        \n" \
"   float GH;        \n" \
"   float GG;        \n" \
"   float BH;        \n" \
"   float BB;        \n" \
"   float e;         \n" \
"	float r1;		 \n" \
"	float g1;		 \n" \
"	float b1;		 \n" \
"	float luma601;		 \n" \
"	float meanluma;		 \n" \
"	float del_luma;		 \n" \
"	float satluma;		 \n" \
"	float satlumaA;		 \n" \
"	float satlumaB;		 \n" \
"	float satalphaL;		 \n" \
"   const int x = get_global_id(0);      \n" \
"   const int y = get_global_id(1);      \n" \
"                                                                    \n" \
"   if ((x < p_Width) && (y < p_Height))                             \n" \
"   {                                                                \n" \
"       const int index = ((y * p_Width) + x) * BLOCKSIZE;           \n" \
"                                    \n" \
"       SRC[0] = p_Input[index + 0] ;    \n" \
"       SRC[1] = p_Input[index + 1] ;    \n" \
"       SRC[2] = p_Input[index + 2] ;    \n" \
"       SRC[3] = p_Input[index + 3] ;    \n" \
"       w_SwitchA  = p_SwitchA; \n" \
"       w_SwitchB  = p_SwitchB; \n" \
"       w_SwitchC  = p_SwitchC; \n" \
"       w_SwitchD  = p_SwitchD; \n" \
"       w_SwitchE  = p_SwitchE; \n" \
"       w_SwitchL  = p_SwitchL; \n" \
"       w_Alpha1   = p_Alpha1; \n" \
"       w_Alpha2   = p_Alpha2; \n" \
"       w_LogA     = p_LogA;    \n" \
"       w_LogB     = p_LogB;    \n" \
"       w_LogC     = p_LogC;    \n" \
"       w_LogD     = p_LogD;    \n" \
"       w_SatA     = p_SatA;    \n" \
"       w_SatB     = p_SatB;    \n" \
"       w_SatC     = p_SatC;    \n" \
"       w_HueA     = p_HueA;    \n" \
"       w_HueB     = p_HueB;    \n" \
"       w_HueC     = p_HueC;    \n" \
"       w_HueR1    = p_HueR1;   \n" \
"       w_HueP1    = p_HueP1;   \n" \
"       w_HueSH1   = p_HueSH1;  \n" \
"       w_HueSHP1  = p_HueSHP1; \n" \
"       w_HueR2    = p_HueR2;   \n" \
"       w_HueP2    = p_HueP2;   \n" \
"       w_HueSH2   = p_HueSH2;  \n" \
"       w_HueSHP2  = p_HueSHP2; \n" \
"       w_HueR3    = p_HueR3;   \n" \
"       w_HueP3    = p_HueP3;   \n" \
"       w_HueSH3   = p_HueSH3;  \n" \
"       w_HueSHP3  = p_HueSHP3; \n" \
"       w_LumaA    = p_LumaA;    \n" \
"       w_LumaB    = p_LumaB;    \n" \
"       w_LumaC    = p_LumaC;    \n" \
"       e = 2.718281828459045;   \n" \
"                                \n" \
"       Lr = w_SwitchA == 1.0f ? w_LogA / (1.0f + pow(e, (-8.9f * w_LogB) * (SRC[0] - w_LogC))) + w_LogD : SRC[0];             \n" \
"       Lg = w_SwitchA == 1.0f ? w_LogA / (1.0f + pow(e, (-8.9f * w_LogB) * (SRC[1] - w_LogC))) + w_LogD : SRC[1];             \n" \
"       Lb = w_SwitchA == 1.0f ? w_LogA / (1.0f + pow(e, (-8.9f * w_LogB) * (SRC[2] - w_LogC))) + w_LogD : SRC[2];             \n" \
"                                                                                                                            \n" \
"       mn = fmin(Lr, fmin(Lg, Lb));                                                                                         \n" \
"       Mx = fmax(Lr, fmax(Lg, Lb));                                                                                         \n" \
"       del_Mx = Mx - mn;                                                                                                    \n" \
"                                                                                                                            \n" \
"       L = (Mx + mn) / 2.0f;                                                                                                 \n" \
"       luma = SRC[0] * 0.2126f + SRC[1] * 0.7152f + SRC[2] * 0.0722f;                                                          \n" \
"       Sat = del_Mx == 0.0f ? 0.0f : L < 0.5f ? del_Mx / (Mx + mn) : del_Mx / (2.0f - Mx - mn);                                 \n" \
"                                                                                                                            \n" \
"       del_R = ((Mx - Lr) / 6.0f + del_Mx / 2.0f) / del_Mx;                                                                   \n" \
"       del_G = ((Mx - Lg) / 6.0f + del_Mx / 2.0f) / del_Mx;                                                                   \n" \
"       del_B = ((Mx - Lb) / 6.0f + del_Mx / 2.0f) / del_Mx;                                                                   \n" \
"                                                                                                                            \n" \
"       h = del_Mx == 0.0f ? 0.0f : Lr == Mx ? del_B - del_G : Lg == Mx ? 1.0f / 3.0f +                                          \n" \
"       del_R - del_B : 2.0f / 3.0f + del_G - del_R;                                                                          \n" \
"                                                                                                                            \n" \
"       Hh = h < 0.0f ? h + 1.0f : h > 1.0f ? h - 1.0f : h;                                                                    \n" \
"                                                                                                                            \n" \
"       s = Sat * w_SatA > 1.0f ? 1.0f : Sat * w_SatA;                                                                       \n" \
"       ss = s > w_SatB ? (-1.0f / ((s - w_SatB) / (1.0f - w_SatB) + 1.0f) + 1.0f) * (1.0f - w_SatB) + w_SatB : s;           \n" \
"       satAlphaA = w_SatC > 1.0f ? luma + (1.0f - w_SatC) * (1.0f - luma) : w_SatC >= 0.0f ? (luma >= w_SatC ? 1.0f : \n" \
"       luma / w_SatC) : w_SatC < -1.0f ? (1.0f - luma) + (w_SatC + 1.0f) * luma : luma <= (1.0f + w_SatC) ? 1.0f :    \n" \
"       (1.0f - luma) / (1.0f - (w_SatC + 1.0f));                                                                        \n" \
"       satAlpha = satAlphaA > 1.0f ? 1.0f : satAlphaA;                                                                      \n" \
"                                                                                                                            \n" \
"       S = w_SwitchB == 1.0f ? ss * satAlpha + s * (1.0f - satAlpha) : Sat;                                                 \n" \
"                                                                                                                            \n" \
"       h1 = Hh - (w_HueA - 0.5f) < 0.0f ? Hh - (w_HueA - 0.5f) + 1.0f : Hh - (w_HueA - 0.5f) >                              \n" \
"       1.0f ? Hh - (w_HueA - 0.5f) - 1.0f : Hh - (w_HueA - 0.5f);                                                 			\n" \
"                                                                                                                            \n" \
"       Hs1 = w_HueSHP1 >= 1.0f ? 1.0f - pow(1.0f - S, w_HueSHP1) : pow(S, 1.0f/w_HueSHP1);                                  \n" \
"       cv1 = 1.0f + (w_HueP1 - 1.0f) * Hs1;                                                                                 \n" \
"       curve1 = w_HueP1 + (cv1 - w_HueP1) * w_HueSH1;                                                                       \n" \
"                                                                                                                            \n" \
"       H1 = w_SwitchC != 1.0f ? Hh : h1 > 0.5f - w_HueR1 && h1 < 0.5f ? (1.0f - pow(1.0f - (h1 - (0.5f - w_HueR1)) *       \n" \
"       (1.0f/w_HueR1), curve1)) * w_HueR1 + (0.5f - w_HueR1) + (w_HueA - 0.5f) : h1 > 0.5f && h1 < 0.5f +             		\n" \
"       w_HueR1 ? pow((h1 - 0.5f) * (1.0f/w_HueR1), curve1) * w_HueR1 + 0.5f + (w_HueA - 0.5f) : Hh;                  		\n" \
"                                                                                                                            \n" \
"       h2 = H1 - (w_HueB - 0.5f) < 0.0f ? H1 - (w_HueB - 0.5f) + 1.0f : H1 - (w_HueB - 0.5f) >                              \n" \
"       1.0f ? H1 - (w_HueB - 0.5f) - 1.0f : H1 - (w_HueB - 0.5f);                                           	           \n" \
"                                                                                                                            \n" \
"       Hs2 = w_HueSHP2 >= 1.0f ? 1.0f - pow(1.0f - S, w_HueSHP2) : pow(S, 1.0f/w_HueSHP2);                                  \n" \
"       cv2 = 1.0f + (w_HueP2 - 1.0f) * Hs2;                                                                                 \n" \
"       curve2 = w_HueP2 + (cv2 - w_HueP2) * w_HueSH2;                                                                       \n" \
"                                                                                                                            \n" \
"       H2 = w_SwitchD != 1.0f ? H1 : h2 > 0.5f - w_HueR2 && h2 < 0.5f ? (1.0f - pow(1.0f - (h2 - (0.5f - w_HueR2)) *       \n" \
"       (1.0f/w_HueR2), curve2)) * w_HueR2 + (0.5f - w_HueR2) + (w_HueB - 0.5f) : h2 > 0.5f && h2 < 0.5f +          	    \n" \
"       w_HueR2 ? pow((h2 - 0.5f) * (1.0f/w_HueR2), curve2) * w_HueR2 + 0.5f + (w_HueB - 0.5f) : H1;           		         \n" \
"                                                                                                                            \n" \
"       h3 = H2 - (w_HueC - 0.5f) < 0.0f ? H2 - (w_HueC - 0.5f) + 1.0f : H2 - (w_HueC - 0.5f) >                              \n" \
"       1.0f ? H2 - (w_HueC - 0.5f) - 1.0f : H2 - (w_HueC - 0.5f);                                                  	    \n" \
"                                                                                                                            \n" \
"       Hs3 = w_HueSHP3 >= 1.0f ? 1.0f - pow(1.0f - S, w_HueSHP3) : pow(S, 1.0f/w_HueSHP3);                                  \n" \
"       cv3 = 1.0f + (w_HueP3 - 1.0f) * Hs3;                                                                                 \n" \
"       curve3 = w_HueP3 + (cv3 - w_HueP3) * w_HueSH3;                                                                       \n" \
"                                                                                                                            \n" \
"       H = w_SwitchE != 1.0f ? H2 : h3 > 0.5f - w_HueR3 && h3 < 0.5f ? (1.0f - pow(1.0f - (h3 - (0.5f - w_HueR3)) *        \n" \
"       (1.0f/w_HueR3), curve3)) * w_HueR3 + (0.5f - w_HueR3) + (w_HueC - 0.5f) : h3 > 0.5f && h3 < 0.5f +          	      \n" \
"       w_HueR3 ? pow((h3 - 0.5f) * (1.0f/w_HueR3), curve3) * w_HueR3 + 0.5f + (w_HueC - 0.5f) : H2;                    	  \n" \
"                                                                                                                            \n" \
"       Q = L < 0.5f ? L * (1.0f + S) : L + S - L * S;                                                                       \n" \
"       P = 2.0f * L - Q;                                                                                                    \n" \
"                                                                                                                            \n" \
"       RH = H + 1.0f / 3.0f < 0.0f ? H + 1.0f / 3.0f + 1.0f :                                                               \n" \
"       H + 1.0f / 3.0f > 1.0f ? H + 1.0f / 3.0f - 1.0f : H + 1.0f / 3.0f;                                               \n" \
"                                                                                                                            \n" \
"       RR = RH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * RH :                                                                    \n" \
"       RH < 1.0f / 2.0f ? Q : RH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - RH) * 6.0f : P;                               \n" \
"                                                                                                                            \n" \
"       GH = H < 0.0f ? H + 1.0f : H > 1.0f ? H - 1.0f : H;                                                                  \n" \
"                                                                                                                            \n" \
"       GG = GH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * GH :                                                                    \n" \
"       GH < 1.0f / 2.0f ? Q : GH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - GH) * 6.0f : P;                            \n" \
"                                                                                                                            \n" \
"       BH = H - 1.0f / 3.0f < 0.0f ? H - 1.0f / 3.0f + 1.0f :                                                               \n" \
"       H - 1.0f / 3.0f > 1.0f ? H - 1.0f / 3.0f - 1.0f : H - 1.0f / 3.0f;                                                 \n" \
"                                                                                                                            \n" \
"       BB = BH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * BH :                                                                    \n" \
"       BH < 1.0f / 2.0f ? Q : BH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - BH) * 6.0f : P;                            \n" \
"                                                                                                                            \n" \
"       r1 = S == 0.0f ? L : RR;        \n" \
"       g1 = S == 0.0f ? L : GG;        \n" \
"       b1 = S == 0.0f ? L : BB;        \n" \
"										\n" \
"	   luma601 = 0.299f * r1 + 0.587f * g1 + 0.114f * b1;	\n" \
"	   meanluma = (r1 + g1 + b1) / 3.0f;					\n" \
"	   del_luma = luma601 / meanluma;						\n" \
"	  														\n" \
"	   satluma = w_LumaA >= 1.0f ? meanluma + ((1.0f - w_LumaA) * (1.0f - meanluma)) : w_LumaA >= 0.0f ? (meanluma >= w_LumaA ? \n" \
"	   1.0f : meanluma / w_LumaA) : w_LumaA < -1.0f ? (1.0f - meanluma) + (w_LumaA + 1.0f) * meanluma : meanluma <= (1.0f + w_LumaA) ? 1.0f : 	\n" \
"	   (1.0f - meanluma) / (1.0f - (w_LumaA + 1.0f));	\n" \
"	   satlumaA = satluma <= 0.0f ? 0.0f : satluma >= 1.0f ? 1.0f : satluma; 	\n" \
"	   satlumaB = (w_LumaB > 1.0f ? 1.0f - pow(1.0f - S, w_LumaB) : pow(S, 1.0f / w_LumaB)) * satlumaA * del_luma;	\n" \
"	   satalphaL = satlumaB > 1.0f ? 1.0f : satlumaB;	\n" \
"	  				\n" \
"	   SRC[0] = w_SwitchL == 1.0f ? (r1 * w_LumaC * satalphaL) + (r1 * (1.0f - satalphaL)) : r1;	\n" \
"	   SRC[1] = w_SwitchL == 1.0f ? (g1 * w_LumaC * satalphaL) + (g1 * (1.0f - satalphaL)) : g1;	\n" \
"	   SRC[2] = w_SwitchL == 1.0f ? (b1 * w_LumaC * satalphaL) + (b1 * (1.0f - satalphaL)) : b1;	\n" \
"                                      \n" \
"       p_Output[index + 0] = w_Alpha1 == 1.0f ? satAlpha : w_Alpha2 == 1.0f ? satalphaL : SRC[0];  \n" \
"       p_Output[index + 1] = w_Alpha1 == 1.0f ? satAlpha : w_Alpha2 == 1.0f ? satalphaL : SRC[1];  \n" \
"       p_Output[index + 2] = w_Alpha1 == 1.0f ? satAlpha : w_Alpha2 == 1.0f ? satalphaL : SRC[2];  \n" \
"       p_Output[index + 3] = SRC[3];  \n" \
"                                      \n" \
"   }                                  \n" \
"}                                     \n" \
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

void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Switch, float* p_Alpha, float* p_Log, float* p_Sat,
	float* p_Hue, float* p_Luma, const float* p_Input, float* p_Output)
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


		kernel = clCreateKernel(program, "HueConvergeKernel", &error);
		CheckError(error, "Unable to create kernel");

		kernelMap[cmdQ] = kernel;
	}
	else
	{
		kernel = kernelMap[cmdQ];
	}

	locker.Unlock();

	int count = 0;
	error = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
	error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[2]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[3]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[4]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[5]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Alpha[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Alpha[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Log[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Log[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Log[2]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Log[3]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Sat[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Sat[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Sat[2]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[2]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[3]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[4]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[5]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[6]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[7]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[8]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[9]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[10]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[11]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[12]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[13]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Hue[14]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Luma[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Luma[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Luma[2]);
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
