#include "QualifierCudaKernel.cuh"

cudaError_t cudaError;
  
__global__ void QualifierKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, float p_AlphaHA, float p_AlphaHB, 
float p_AlphaHC, float p_AlphaHD, float p_AlphaHE, float p_AlphaHF, float p_AlphaHO, float p_AlphaSA, float p_AlphaSB, float p_AlphaSC, 
float p_AlphaSD, float p_AlphaSE, float p_AlphaSF, float p_AlphaSO, float p_AlphaLA, float p_AlphaLB, float p_AlphaLC, float p_AlphaLD, 
float p_AlphaLE, float p_AlphaLF, float p_AlphaLO, int p_InvertH, int p_InvertS, int p_InvertL, int p_Math, int p_OutputAlpha,
float p_Black, float p_White, float p_HsvA, float p_HsvB, float p_HsvC)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < p_Width) && (y < p_Height))
	{

	const int index = ((y * p_Width) + x) * 4;
	
	float RGB[3];
	RGB[0] = p_Input[index + 0];
	RGB[1] = p_Input[index + 1];
	RGB[2] = p_Input[index + 2];
	
	float h, s, v, H;
	RGB_to_HSV( RGB[0], RGB[1], RGB[2], &h, &s, &v );
	float lum = Luma(RGB[0], RGB[1], RGB[2], p_Math);
	
	if ( (s > 0.1f) && (v > 0.1f) ) {
	float hh = h + p_AlphaHO;
	H = hh < 0.0f ? hh + 1.0f : hh >= 1.0f ? hh - 1.0f : hh;
	} else { H = 0.0f;}
	float Hue = Alpha(p_AlphaHA, p_AlphaHB, p_AlphaHC, p_AlphaHD, p_AlphaHE, p_AlphaHF, H, p_InvertH);
	
	float ss = s + p_AlphaSO;
	float S = ss < 0.0f ? 0.0f : ss >= 1.0f ? 1.0f : ss;
	float Sat = Alpha(p_AlphaSA, p_AlphaSB, p_AlphaSC, p_AlphaSD, p_AlphaSE, p_AlphaSF, S, p_InvertS);
	
	float l = lum + p_AlphaLO;
	float L = l < 0.0f ? 0.0f : l >= 1.0f ? 1.0f : l;
	float Lum = Alpha(p_AlphaLA, p_AlphaLB, p_AlphaLC, p_AlphaLD, p_AlphaLE, p_AlphaLF, L, p_InvertL);
	
	float A = p_OutputAlpha == 0 ? fmin(fmin(Hue, Sat), Lum) : p_OutputAlpha == 1 ? Hue : p_OutputAlpha == 2 ? Sat :
	p_OutputAlpha == 3 ? Lum : p_OutputAlpha == 4 ? fmin(Hue, Sat) : p_OutputAlpha == 5 ? 
	fmin(Hue, Lum) : p_OutputAlpha == 6 ? fmin(Sat, Lum) : 1.0f;
	
	if (p_Black > 0.0f) {
	A = fmax(A + (0.0f - p_Black * 4.0f) * (1.0f - A), 0.0f);
	}
	
	if (p_White > 0.0f) {
	A = fmin(A * (1.0f + p_White * 4.0f), 1.0f);
	}
	
	if (p_HsvA != 0.0f || p_HsvB != 0.0f || p_HsvC != 0.0f) {
	float h2 = h + p_HsvA;
	float H2 = h2 < 0.0f ? h2 + 1.0f : h2 >= 1.0f ? h2 - 1.0f : h2;
	float S = s * (1.0f + p_HsvB);
	float V = v * (1.0f + p_HsvC);
	HSV_to_RGB(H2, S, V, &RGB[0], &RGB[1], &RGB[2]);
	}
	
	p_Output[index + 0] = RGB[0];
	p_Output[index + 1] = RGB[1];
	p_Output[index + 2] = RGB[2];
	p_Output[index + 3] = A;
	
	}
}

__global__ void QualifierEnd(float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Display, int p_Invert, int p_Warning, float p_Mix)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < p_Width) && (y < p_Height))
	{

	const int index = ((y * p_Width) + x) * 4;
	
	float A = p_Output[index + 3];
	
	if (p_Invert == 1) {
	A = 1.0f - A;
	}
	
	if (p_Mix != 0.0f) {
	if (p_Mix > 0.0f) {
	A = A + (1.0f - A) * p_Mix;
	} else {
	A *= 1.0f + p_Mix;
	}}
	
	A = fmax(fmin(A, 1.0f), 0.0f);
	
	float RA, GA, BA;
	RA = GA = BA = A;
	
	if (p_Warning == 1 && p_Display == 1) {
	RA = A > 0.0f && A < 0.2f ? 0.0f : A < 1.0f && A > 0.8f ? 1.0f : A;
	GA = A > 0.0f && A < 0.2f ? 0.0f : A < 1.0f && A > 0.8f ? 1.0f : A;
	BA = A > 0.0f && A < 0.2f ? 1.0f : A < 1.0f && A > 0.8f ? 0.0f : A;
	}
	
	p_Output[index + 0] = p_Display == 1 ? RA : p_Output[index + 0] * A + p_Input[index + 0] * (1.0f - A);
	p_Output[index + 1] = p_Display == 1 ? GA : p_Output[index + 1] * A + p_Input[index + 1] * (1.0f - A);
	p_Output[index + 2] = p_Display == 1 ? BA : p_Output[index + 2] * A + p_Input[index + 2] * (1.0f - A);
	p_Output[index + 3] = p_Display == 1 ? p_Input[index + 3] : A;
	
	}
}

void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, int* p_Switch, 
float* p_AlphaH, float* p_AlphaS, float* p_AlphaL, float p_Mix, int p_Math, int p_OutputAlpha, 
float p_Black, float p_White, float p_Blur, float p_Garbage, float p_Core, float p_Erode, float p_Dilate, float* p_HSV)
{
	dim3 Threads(128, 1, 1);
	dim3 blocks(((p_Width + Threads.x - 1) / Threads.x), p_Height, 1);
	
	int nthreads = 128;
	float* tempBuffer;
	cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);
	
	dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
	dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	
	QualifierKernel<<<blocks, Threads>>>(p_Input, p_Output, p_Width, p_Height, p_AlphaH[0], p_AlphaH[1], p_AlphaH[2], p_AlphaH[3], p_AlphaH[4], 
	p_AlphaH[5], p_AlphaH[6], p_AlphaS[0], p_AlphaS[1], p_AlphaS[2], p_AlphaS[3], p_AlphaS[4], p_AlphaS[5], p_AlphaS[6],
	p_AlphaL[0], p_AlphaL[1], p_AlphaL[2], p_AlphaL[3], p_AlphaL[4], p_AlphaL[5], p_AlphaL[6],
	p_Switch[2], p_Switch[3], p_Switch[4], p_Math, p_OutputAlpha, p_Black, p_White, p_HSV[0], p_HSV[1], p_HSV[2]);
	
	if (p_Blur > 0.0f) {
	p_Blur *= 10.0f;
    const float nsigma = p_Blur < 0.1f ? 0.1f : p_Blur,
	alpha = 1.695f / nsigma,
	ema = (float)exp(-alpha);
	float ema2 = (float)exp(-2*alpha),
	b1 = -2*ema,
	b2 = ema2;
	float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;
	const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
	a0 = k;
	a1 = k*(alpha-1)*ema;
	a2 = k*(alpha+1)*ema;
	a3 = -k*ema2;
	coefp = (a0+a1)/(1+b1+b2);
	coefn = (a2+a3)/(1+b1+b2);
				
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Width, p_Height, a0, a1, a2, a3, b1, b2, coefp, coefn);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, a0, a1, a2, a3, b1, b2, coefp, coefn);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 3, p_Height, p_Width);
	
	d_garbageCore<<<blocks, Threads>>>(p_Output + 3, p_Width, p_Height, p_Garbage, p_Core);
	}
	
	int radE = ceil(p_Erode * 15);
	int radD = ceil(p_Dilate * 15);
    int tile_w = 640;
    int tile_h = 1;
    dim3 blockE2(tile_w + (2 * radE), tile_h);
    dim3 blockD2(tile_w + (2 * radD), tile_h);
    dim3 grid2(iDivUp(p_Width, tile_w), iDivUp(p_Height, tile_h));
    
    int tile_w2 = 8;
    int tile_h2 = 64;
    dim3 blockE3(tile_w2, tile_h2 + (2 * radE));
    dim3 blockD3(tile_w2, tile_h2 + (2 * radD));
    dim3 grid3(iDivUp(p_Width, tile_w2), iDivUp(p_Height, tile_h2));
    
    if (p_Erode > 0.0f) {
    ErosionSharedStep1<<<grid2,blockE2,blockE2.y*blockE2.x*sizeof(float)>>>(p_Output + 3, tempBuffer, radE, p_Width, p_Height, tile_w, tile_h);
    cudaError = cudaDeviceSynchronize();
    ErosionSharedStep2<<<grid3,blockE3,blockE3.y*blockE3.x*sizeof(float)>>>(tempBuffer, p_Output + 3, radE, p_Width, p_Height, tile_w2, tile_h2);
    }
    
    if (p_Dilate > 0.0f) {
    DilateSharedStep1<<<grid2,blockD2,blockD2.y*blockD2.x*sizeof(float)>>>(p_Output + 3, tempBuffer, radD, p_Width, p_Height, tile_w, tile_h);
    cudaError = cudaDeviceSynchronize();
    DilateSharedStep2<<<grid3,blockD3,blockD3.y*blockD3.x*sizeof(float)>>>(tempBuffer, p_Output + 3, radD, p_Width, p_Height, tile_w2, tile_h2);
    }
    
    QualifierEnd<<<blocks, Threads>>>(p_Input, p_Output, p_Width, p_Height, p_Switch[0], p_Switch[1], p_Switch[5], p_Mix);
    
	cudaError = cudaFree(tempBuffer);
	
}
