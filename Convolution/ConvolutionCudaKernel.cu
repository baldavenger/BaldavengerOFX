#include "ConvolutionCudaKernel.cuh"

cudaError_t cudaError;

void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Convolve, int p_Display, float* p_Adjust, float* p_Matrix)
{   	
	int nthreads = 128;
	float* tempBuffer;
	cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);
	
	dim3 threads(128, 1, 1);
	dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);
	
	dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
	dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
	dim3 threadsT(BLOCK_DIM, BLOCK_DIM);
	
	int radio = ceil(p_Adjust[0] * 15);
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
    p_Adjust[0] *= 100.0f;
    
    if (p_Adjust[0] > 0.0f) {	
	for(int c = 0; c < 3; c++) {
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, p_Adjust[0]);
    d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Adjust[0]);
	d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
    	}
        break;

    case 1:
    {
    p_Adjust[0] *= 100.0f;
    if (p_Adjust[0] > 0.0f) {	
    for(int c = 0; c < 3; c++) {
	d_simpleRecursive<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, p_Adjust[0]);
	d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_simpleRecursive<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Adjust[0]);
	d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
        }
        break;

    case 2:
    {
    int R = p_Adjust[0] * 100;
    if (R > 0) {	
	for(int c = 0; c < 3; c++) {
	d_boxfilter<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, R);
	d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_boxfilter<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, R);
	d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
        }
        break;
       
    case 3:
    {
    
    p_Adjust[2] *= 5.0f;
    p_Adjust[0] *= 10.0f;
    float sharpen = (2 * p_Adjust[1]) + 1;
    
    if (p_Adjust[2] > 0.0f) {
    for(int c = 0; c < 3; c++) {
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, p_Adjust[2]);
    d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Adjust[2]);
	d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
	
	for(int c = 0; c < 3; c++) {
	FrequencySharpen<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c, sharpen, p_Display);
	}
	
	if (p_Display != 1) {
	
	if (p_Adjust[0] > 0.0f) {
    for(int c = 0; c < 3; c++) {
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Width, p_Height, p_Adjust[0]);
    d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Adjust[0]);
	d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	}
	
	for(int c = 0; c < 3; c++) {
	FrequencyAdd<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
    	}
    	}
        break;
       
    case 4:
    {
    p_Adjust[2] *= 3.0f;
    if (p_Adjust[2] > 0.0f) {
    for(int c = 0; c < 3; c++) {
	d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + c, tempBuffer, p_Width, p_Height, p_Adjust[2]);
    d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + c, p_Width, p_Height);
	d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + c, tempBuffer, p_Height, p_Width, p_Adjust[2]);
	d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + c, p_Height, p_Width);
	}
	} else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
	for(int c = 0; c < 3; c++) {
	EdgeDetectAdd<<< blocks, threads >>>(p_Width, p_Height, p_Input + c, p_Output + c, p_Adjust[2]);
	}
    	}
        break;
        
    case 5:
    {
    
    p_Adjust[0] *= 20.0f;
    EdgeEnhance<<< blocks, threads >>>(p_Width, p_Height, p_Input, p_Output, p_Adjust[0]);
    
    	}
    	break;
    
    case 6:
    {
    
    if (p_Adjust[0] > 0.0f) {
    for(int c = 0; c < 3; c++) {
    ErosionSharedStep1<<<grid2,blockE2,blockE2.y*blockE2.x*sizeof(float)>>>(p_Input + c, tempBuffer, radio, p_Width, p_Height, tile_w, tile_h);
    cudaError = cudaDeviceSynchronize();
    ErosionSharedStep2<<<grid3,blockE3,blockE3.y*blockE3.x*sizeof(float)>>>(tempBuffer, p_Output + c, radio, p_Width, p_Height, tile_w2, tile_h2);
    }
    } else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
    	}
    	break;
    	
    case 7:
    {
    
    if (p_Adjust[0] > 0.0f) {
    for(int c = 0; c < 3; c++) {
    DilateSharedStep1<<<grid2,blockD2,blockD2.y*blockD2.x*sizeof(float)>>>(p_Input + c, tempBuffer, radio, p_Width, p_Height, tile_w, tile_h);
    cudaError = cudaDeviceSynchronize();
    DilateSharedStep2<<<grid3,blockD3,blockD3.y*blockD3.x*sizeof(float)>>>(tempBuffer, p_Output + c, radio, p_Width, p_Height, tile_w2, tile_h2);
    }
    } else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
    
    	}
    	break;
    	
    case 8:
    {
    
    if (p_Adjust[0] > 0.0f) {
    Scatter<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, radio, p_Adjust[1]);
    } else {
	for(int c = 0; c < 3; c++) {
	SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
	}
	}
    
    	}
    	break;
    	
    case 9:
    {
	
	p_Adjust[0] *= 10.0f;
	for(int c = 0; c < 3; c++) {
	CustomMatrix<<< blocks, threads >>>(p_Width, p_Height, p_Input + c, p_Output + c, p_Adjust[0], p_Display, 
	p_Matrix[0], p_Matrix[1], p_Matrix[2], p_Matrix[3], p_Matrix[4], p_Matrix[5], p_Matrix[6], p_Matrix[7], p_Matrix[8]);
    }
    	}
    	break;
    
    default: 
			for(int c = 0; c < 3; c++) {
			SimpleKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input + c, p_Output + c);
			}
    }

	cudaError = cudaFree(tempBuffer);
}
