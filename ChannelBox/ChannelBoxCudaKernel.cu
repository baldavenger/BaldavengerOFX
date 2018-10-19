#include "ChannelBoxCudaKernel.cuh"

cudaError_t cudaError;

__global__ void ChannelBoxKernelA(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Choice, int p_ChannelBox, float p_ChannelSwap0, float p_ChannelSwap1, float p_ChannelSwap2, 
float p_ChannelSwap3, float p_ChannelSwap4, float p_ChannelSwap5, float p_ChannelSwap6, 
float p_ChannelSwap7, float p_ChannelSwap8, int p_LumaMath, int p_Preserve, float p_Mask)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
    const int index = (y * p_Width + x) * 4;
    
	const float R = p_Input[index + 0];		
	const float G = p_Input[index + 1];
	const float B = p_Input[index + 2];
	float inLuma = Luma(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2], p_LumaMath);
	float red, green, blue;
	float mask = 1.0f;
	
	switch (p_Choice)
    {
    case 0:
    {
	float BR = B > R ? R : B;    
	float BG = B > G ? G : B;    
	float BGR = B > fmin(G, R) ? fmin(G, R) : B;    
	float BGRX = B > fmax(G, R) ? fmax(G, R) : B;    
	blue = p_ChannelBox==0 ? BR : p_ChannelBox==1 ? BG : p_ChannelBox==2 ? BGR : p_ChannelBox==3 ? BGRX : B; 
																   
	float GR = G > R ? R : G;    
	float GB = G > B ? B : G;    
	float GBR = G > fmin(B, R) ? fmin(B, R) : G;    
	float GBRX = G > fmax(B, R) ? fmax(B, R) : G;    
	green = p_ChannelBox==4 ? GR : p_ChannelBox==5 ? GB : p_ChannelBox==6 ? GBR : p_ChannelBox==7 ? GBRX : G; 
																   
	float RG = R > G ? G : R;    
	float RB = R > B ? B : R;    
	float RBG = R > fmin(B, G) ? fmin(B, G) : R;    
	float RBGX = R > fmax(B, G) ? fmax(B, G) : R;    
	red = p_ChannelBox==8 ? RG : p_ChannelBox==9 ? RB : p_ChannelBox==10 ? RBG : p_ChannelBox==11 ? RBGX : R; 					
	}
    	break;
    
    case 1:
    {
	red = R * (1.0f + p_ChannelSwap0 + p_ChannelSwap1 + p_ChannelSwap2) + G * (0.0f - p_ChannelSwap0 - (p_ChannelSwap2 / 2)) + B * (0.0f - p_ChannelSwap1 - (p_ChannelSwap2 / 2));
	green = R * (0.0f - p_ChannelSwap3 - (p_ChannelSwap5 / 2)) + G * (1.0f + p_ChannelSwap3 + p_ChannelSwap4 + p_ChannelSwap5) + B * (0.0f - p_ChannelSwap4 - (p_ChannelSwap5 / 2));
	blue = R * (0.0f - p_ChannelSwap6 - (p_ChannelSwap8 / 2)) + G * (0.0f - p_ChannelSwap7 - (p_ChannelSwap8 / 2)) + B * (1.0f + p_ChannelSwap6 + p_ChannelSwap7 + p_ChannelSwap8);
	}
    	break;
    	
    default: 
			red = R;
			green = G;
			blue = B;
    }
	
	if (p_Preserve == 1) {
	float outLuma = Luma(red, green, blue, p_LumaMath);
	red = red * (inLuma / outLuma);
	green = green * (inLuma / outLuma);
	blue = blue * (inLuma / outLuma);
	}
	
	if(p_Mask != 0.0f) {
	mask = p_Mask > 1.0f ? inLuma + (1.0f - p_Mask) * (1.0f - inLuma) : p_Mask >= 0.0f ? (inLuma >= p_Mask ? 1.0f : 
	inLuma / p_Mask) : p_Mask < -1.0f ? (1.0f - inLuma) + (p_Mask + 1.0f) * inLuma : inLuma <= (1.0f + p_Mask) ? 1.0f : 
	(1.0 - inLuma) / (1.0f - (p_Mask + 1.0f));
	mask = mask > 1.0f ? 1.0f : mask < 0.0f ? 0.0f : mask;
	}
																											   
	p_Output[index + 0] = red;
	p_Output[index + 1] = green;
	p_Output[index + 2] = blue;
	p_Output[index + 3] = mask;
   }
}

__global__ void ChannelBoxKernelB(const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Display)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
    const int index = (y * p_Width + x) * 4;
    
    p_Output[index + 0] = p_Display == 1 ? p_Output[index + 3] : p_Output[index + 0] * p_Output[index + 3] + p_Input[index + 0] * (1.0f - p_Output[index + 3]);
    p_Output[index + 1] = p_Display == 1 ? p_Output[index + 3] : p_Output[index + 1] * p_Output[index + 3] + p_Input[index + 1] * (1.0f - p_Output[index + 3]);
    p_Output[index + 2] = p_Display == 1 ? p_Output[index + 3] : p_Output[index + 2] * p_Output[index + 3] + p_Input[index + 2] * (1.0f - p_Output[index + 3]);
	p_Output[index + 3] = p_Display == 1 ? 1.0f : p_Output[index + 3];
   }
}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Choice, 
int p_ChannelBox, float* p_ChannelSwap, int p_LumaMath, int* p_Switch, float* p_Mask)
{
	dim3 threads(128, 1, 1);
	dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    ChannelBoxKernelA<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Choice, p_ChannelBox, 
    p_ChannelSwap[0], p_ChannelSwap[1], p_ChannelSwap[2], p_ChannelSwap[3], p_ChannelSwap[4], p_ChannelSwap[5], 
    p_ChannelSwap[6], p_ChannelSwap[7], p_ChannelSwap[8], p_LumaMath, p_Switch[0], p_Mask[0]);
    
    int nthreads = 128;
	float* tempBuffer;
	cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);
	
	dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
	dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
	dim3 threadsT(BLOCK_DIM, BLOCK_DIM);

    if(p_Mask[1] > 0.0f) {
    p_Mask[1] *= 10.0f;
    d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Width, p_Height, p_Mask[1]);
    d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, p_Mask[1]);
	d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + 3, p_Height, p_Width);
	
	d_garbageCore<<<blocks, threads>>>(p_Output + 3, p_Width, p_Height, p_Mask[2], p_Mask[3]);
    }
    
    ChannelBoxKernelB<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Switch[1]);
    
    cudaError = cudaFree(tempBuffer);
}