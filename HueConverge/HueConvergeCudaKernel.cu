#include "HueConvergeCudaKernel.h"
#include "HueConvergeCudaBlur.h"

cudaError_t cudaError;

__global__ void LogStageKernel( float* p_Input, float* p_Output, int p_Width, int p_Height, int p_SwitchLog, int p_SwitchHue,
float p_LogA, float p_LogB, float p_LogC, float p_LogD, float p_SatA, float p_LumaLimit, float p_SatLimit, int p_Math)
{	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < p_Width) && (y < p_Height))
	{
	const int index = (y * p_Width + x) * 4;
	
	float3 RGB = make_float3(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2]);
	
	if(p_SwitchLog == 1)
	RGB = Sigmoid( RGB, p_LogA, p_LogB, p_LogC, p_LogD);
	
	if(p_SatA != 1.0f){
	float3 HSV;
	RGB_to_HSV( RGB.x, RGB.y, RGB.z, &HSV.x, &HSV.y, &HSV.z);
	HSV.y *= p_SatA;
	HSV_to_RGB( HSV.x, HSV.y, HSV.z, &RGB.x, &RGB.y, &RGB.z);
	}
	
	float luma = Luma(RGB.x, RGB.y, RGB.z, p_Math);
	float lumaAlpha = 1.0f;
	float satAlpha = 1.0f;
	
	if(p_LumaLimit != 0.0f && p_SwitchHue == 1)
	lumaAlpha = Limiter(luma, p_LumaLimit);
	
	if(p_SatLimit != 0.0f && p_SwitchHue == 1)
	{
	float3 ych = rgb_2_ych( RGB);
	satAlpha = Limiter(ych.y * 10.0f, p_SatLimit);
	}
	
	p_Output[index + 0] = RGB.x;
	p_Output[index + 1] = RGB.y;
	p_Output[index + 2] = RGB.z;
	p_Output[index + 3] = luma;
	
	p_Input[index + 0] = 1.0f * lumaAlpha * satAlpha;
	
	}
}

__global__ void HueStageKernel(float* p_In, float* p_ALPHA, int p_Width, int p_Height, int p_SwitchHue, int p_SwitchHue1, 
float p_Hue1, float p_Hue2, float p_Hue3, float p_Hue4, float p_Hue5, float p_LumaLimit, float p_SatLimit, int ALPHA)
{	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < p_Width) && (y < p_Height))
	{
	const int index = (y * p_Width + x) * 4;
	
	float3 RGB = make_float3(p_In[index + 0], p_In[index + 1], p_In[index + 2]);
	float3 ych = rgb_2_ych( RGB);
	
	float lumaAlpha = 1.0f;
	float satAlpha = 1.0f;
	
	if(p_LumaLimit != 0.0f && p_SwitchHue == 1)
	lumaAlpha = Limiter(p_In[index + 3], p_LumaLimit);
	
	if(p_SatLimit != 0.0f && p_SwitchHue == 1)
	satAlpha = Limiter(ych.y * 10.0f, p_SatLimit);
	
	float3 new_ych = modify_hue( ych, p_Hue1, p_Hue2, p_Hue3, p_Hue4, p_Hue5);
	RGB = ych_2_rgb( new_ych);
	
	float alpha = p_ALPHA[index + ALPHA];
	if(alpha != 1.0f)
	{
	RGB.x = RGB.x * alpha + p_In[index + 0] * (1.0f - alpha);
	RGB.y = RGB.y * alpha + p_In[index + 1] * (1.0f - alpha);
	RGB.z = RGB.z * alpha + p_In[index + 2] * (1.0f - alpha);
	}
	
    p_In[index + 0] = p_SwitchHue == 1 ? RGB.x : p_In[index + 0];
	p_In[index + 1] = p_SwitchHue == 1 ? RGB.y : p_In[index + 1];
	p_In[index + 2] = p_SwitchHue == 1 ? RGB.z : p_In[index + 2];
	
	p_ALPHA[index + ALPHA + 1] = 1.0f * lumaAlpha * satAlpha;
	
	}
}

__global__ void FinalStageKernel(float* p_In, float* p_ALPHA, int p_Width, int p_Height, float p_SatSoft, int p_Display)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < p_Width) && (y < p_Height))
	{
	const int index = (y * p_Width + x) * 4;
	
	float3 RGB = make_float3(p_In[index + 0], p_In[index + 1], p_In[index + 2]);
	float3 ych = rgb_2_ych( RGB);
	
	float alpha = p_ALPHA[index + 3];
	float soft = Sat_Soft_Clip(ych.y, p_SatSoft);
	ych.y = soft * alpha + ych.y * (1.0f - alpha);
	
	float3 rgb = ych_2_rgb( ych);
	
	float luma = p_In[index + 3];
	
	if(p_Display != 0)
	{
	float displayAlpha = p_Display == 1 ? p_ALPHA[index + 0] : p_Display == 2 ? 
	p_ALPHA[index + 1] : p_Display == 3 ? p_ALPHA[index + 2] :  p_Display == 4 ? alpha : luma;  
	p_In[index + 0] = displayAlpha;
	p_In[index + 1] = displayAlpha;
	p_In[index + 2] = displayAlpha;
	p_In[index + 3] = 1.0f;
	} else {
	p_In[index + 0] = rgb.x;
	p_In[index + 1] = rgb.y;
	p_In[index + 2] = rgb.z;
	p_In[index + 3] = 1.0f;
	}
	
	}
}

void  RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, int* p_Switch, 
float* p_Log, float* p_Sat, float *p_Hue1, float *p_Hue2, float *p_Hue3, int p_Display, float *p_Blur, int p_Math)
{
	int nthreads = 128;
	float* tempBuffer;
	cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);
	
	dim3 threadsT(128, 1, 1);
	dim3 blocks(((p_Width + threadsT.x - 1) / threadsT.x), p_Height, 1);
	
	dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
	dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	
    LogStageKernel<<<blocks, threadsT>>>(p_Input, p_Output, p_Width, p_Height, p_Switch[0], p_Switch[1],
	p_Log[0], p_Log[1], p_Log[2], p_Log[3], p_Sat[0], p_Hue1[5], p_Hue1[6], p_Math);
	
	if (p_Blur[0] > 0.0f && p_Switch[1] == 1) {
    d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input, tempBuffer, p_Width, p_Height, p_Blur[0]);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Input, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input, tempBuffer, p_Height, p_Width, p_Blur[0]);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Input, p_Height, p_Width);
    }
	
	HueStageKernel<<<blocks, threadsT>>>(p_Output, p_Input, p_Width, p_Height, p_Switch[1], p_Switch[2], p_Hue1[0], p_Hue1[1], 
	p_Hue1[2], p_Hue1[3], p_Hue1[4], p_Hue2[5], p_Hue2[6], 0);
	
	if (p_Blur[1] > 0.0f && p_Switch[2] == 1) {
    d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 1, tempBuffer, p_Width, p_Height, p_Blur[1]);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Input + 1, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input + 1, tempBuffer, p_Height, p_Width, p_Blur[1]);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Input + 1, p_Height, p_Width);
    }
	
	HueStageKernel<<<blocks, threadsT>>>(p_Output, p_Input, p_Width, p_Height, p_Switch[2], p_Switch[3], p_Hue2[0], p_Hue2[1], 
	p_Hue2[2], p_Hue2[3], p_Hue2[4], p_Hue3[5], p_Hue3[6], 1);
	
	if (p_Blur[2] > 0.0f && p_Switch[3] == 1) {
    d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 2, tempBuffer, p_Width, p_Height, p_Blur[2]);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Input + 2, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input + 2, tempBuffer, p_Height, p_Width, p_Blur[2]);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Input + 2, p_Height, p_Width);
    }
	
	HueStageKernel<<<blocks, threadsT>>>(p_Output, p_Input, p_Width, p_Height, p_Switch[3], 1, p_Hue3[0], p_Hue3[1], 
	p_Hue3[2], p_Hue3[3], p_Hue3[4], p_Sat[2], 0.0f, 2);
	
	if (p_Blur[3] > 0.0f) {
    d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 3, tempBuffer, p_Width, p_Height, p_Blur[3]);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Input + 3, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input + 3, tempBuffer, p_Height, p_Width, p_Blur[3]);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Input + 3, p_Height, p_Width);
    }
	
	FinalStageKernel<<<blocks, threadsT>>>(p_Output, p_Input, p_Width, p_Height, p_Sat[1], p_Display);
	
	cudaError = cudaFree(tempBuffer);
	
}
