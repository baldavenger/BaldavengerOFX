#include "ConvolutionCudaKernel.cuh"

cudaError_t cudaError;

void RunCudaKernel(int p_Width, int p_Height, int p_Convolve, float p_Adjust1, float p_Adjust2, 
			float p_Threshold, int p_Display, float* p_Matrix, float* p_Input, float* p_Output)
{   	
	int nthreads = 128;
	float* tempBuffer;
	cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);
	
	dim3 threadsT(128, 1, 1);
	dim3 blocks(((p_Width + threadsT.x - 1) / threadsT.x), p_Height, 1);
	
	dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
	dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	
	int radio = ceil(p_Adjust1 * 15);
    int tile_w = 640;
    int tile_h = 1;
    dim3 blockE2(tile_w + (2 * radio), tile_h);
    dim3 blockD2(tile_w + (2 * radio), tile_h);
    dim3 grid2(iDivUp(p_Width, tile_w), iDivUp(p_Height, tile_h));
    
    int tile_w2 = 8;
    int tile_h2 = 64;
    dim3 blockE3(tile_w2, tile_h2 + (2 * radio));
    dim3 blockD3(tile_w2, tile_h2 + (2 * radio));
    dim3 grid3(iDivUp(p_Width, tile_w2), iDivUp(p_Height, tile_h2));
	
	switch (p_Convolve)
    {
    case 0:
    {
    p_Adjust1 *= 100.0f;
    
    if (p_Adjust1 > 0.0f) {	
	for(int c = 0; c < 3; c++) {
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, p_Adjust1);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Adjust1);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
    	}
        break;

    case 1:
    {
    p_Adjust1 *= 100.0f;
    if (p_Adjust1 > 0.0f) {	
    for(int c = 0; c < 3; c++) {
	d_simpleRecursive<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, p_Adjust1);
	d_transpose<<< grid, threads >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_simpleRecursive<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Adjust1);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
        }
        break;

    case 2:
    {
    int R = p_Adjust1 * 100;
    if (R > 0) {	
	for(int c = 0; c < 3; c++) {
	d_boxfilter<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, R);
	d_transpose<<< grid, threads >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_boxfilter<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, R);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
        }
        break;
       
    case 3:
    {
    
    p_Threshold *= 5.0f;
    p_Adjust1 *= 10.0f;
    float sharpen = (2 * p_Adjust2) + 1;
    
    if (p_Threshold > 0.0f) {
    for(int c = 0; c < 3; c++) {
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, p_Threshold);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Threshold);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
	
	for(int c = 0; c < 3; c++) {
	FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c, sharpen, p_Display);
	}
	
	if (p_Display != 1) {
	
	if (p_Adjust1 > 0.0f) {
    for(int c = 0; c < 3; c++) {
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Width, p_Height, p_Adjust1);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Adjust1);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	}
	
	for(int c = 0; c < 3; c++) {
	FrequencyAdd<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
    	}
    	}
        break;
       
    case 4:
    {
    p_Threshold *= 3.0f;
    if (p_Threshold > 0.0f) {
    for(int c = 0; c < 3; c++) {
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, p_Threshold);
    d_transpose<<< grid, threads >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Threshold);
	d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
	for(int c = 0; c < 3; c++) {
	EdgeDetectAdd<<< blocks, threadsT >>>(p_Width, p_Height, p_Input + c, p_Output + c, p_Threshold);
	}
    	}
        break;
        
    case 5:
    {
    
    p_Adjust1 *= 20.0f;
    EdgeEnhance<<< blocks, threadsT >>>(p_Width, p_Height, p_Input, p_Output, p_Adjust1);
    
    	}
    	break;
    
    case 6:
    {
    
    if (p_Adjust1 > 0.0f) {
    for(int c = 0; c < 3; c++) {
    ErosionSharedStep1<<<grid2,blockE2,blockE2.y*blockE2.x*sizeof(float)>>>(p_Input + c, tempBuffer, radio, p_Width, p_Height, tile_w, tile_h);
    cudaError = cudaDeviceSynchronize();
    ErosionSharedStep2<<<grid3,blockE3,blockE3.y*blockE3.x*sizeof(float)>>>(tempBuffer, p_Output + c, radio, p_Width, p_Height, tile_w2, tile_h2);
    }
    } else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
    	}
    	break;
    	
    case 7:
    {
    
    if (p_Adjust1 > 0.0f) {
    for(int c = 0; c < 3; c++) {
    DilateSharedStep1<<<grid2,blockD2,blockD2.y*blockD2.x*sizeof(float)>>>(p_Input + c, tempBuffer, radio, p_Width, p_Height, tile_w, tile_h);
    cudaError = cudaDeviceSynchronize();
    DilateSharedStep2<<<grid3,blockD3,blockD3.y*blockD3.x*sizeof(float)>>>(tempBuffer, p_Output + c, radio, p_Width, p_Height, tile_w2, tile_h2);
    }
    } else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
    
    	}
    	break;
    	
    case 8:
    {
	
	p_Adjust1 *= 10.0f;
	for(int c = 0; c < 3; c++) {
	CustomMatrix<<< blocks, threadsT >>>(p_Width, p_Height, p_Input + c, p_Output + c, p_Adjust1, p_Display, 
	p_Matrix[0], p_Matrix[1], p_Matrix[2], p_Matrix[3], p_Matrix[4], p_Matrix[5], p_Matrix[6], p_Matrix[7], p_Matrix[8]);
    }
    	}
    	break;
    
    default: 
			for(int c = 0; c < 3; c++) {
			SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
			}
    }

	cudaError = cudaFree(tempBuffer);
}
