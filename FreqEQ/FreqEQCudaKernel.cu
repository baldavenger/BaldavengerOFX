#include "FreqEQCudaKernel.cuh"

cudaError_t cudaError;

void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_EQ, int p_Switch, int p_Grey)
{   	
int nthreads = 128;
float* tempBuffer;
cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);

dim3 threadsT(128, 1, 1);
dim3 blocks(((p_Width + threadsT.x - 1) / threadsT.x), p_Height, 1);

dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
dim3 threads(BLOCK_DIM, BLOCK_DIM);

float Res_Scale = (float)p_Width / 1920.0f;
float p_Blur[6];
p_Blur[0] = (0.4f * Res_Scale) * p_EQ[6]; 
for(int c = 0; c < 5; c++) {
p_Blur[c + 1] = p_Blur[c] * 2.0f; 
}

if(p_Switch == 0 && p_EQ[0] == 1.0f && p_EQ[1] == 1.0f && p_EQ[2] == 1.0f && p_EQ[3] == 1.0f && p_EQ[4] == 1.0f && p_EQ[5] == 1.0f){
for(int c = 0; c < 4; c++) {
SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
}
cudaError = cudaFree(tempBuffer);
return;
}

d_rec709_to_lab<<<blocks, threadsT>>>(p_Input, p_Input, p_Width, p_Height);
SimpleKernelALPHA<<<blocks, threadsT>>>(p_Width, p_Height, 0.0f, p_Output + 0);

d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 0, tempBuffer, p_Width, p_Height, p_Blur[0]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 2, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Height, p_Width, p_Blur[0]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 2, p_Height, p_Width);

FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + 0, p_Output + 2, p_Output + 0, p_EQ[0], p_Switch);

if(p_Switch == 1){
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 0, p_Output, p_Grey);
cudaError = cudaFree(tempBuffer);
return;}

if(p_EQ[1] != 1.0f || p_EQ[2] != 1.0f || p_EQ[3] != 1.0f || p_EQ[4] != 1.0f || p_EQ[5] != 1.0f || p_Switch > 0){
SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 2, p_Output + 3);
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Width, p_Height, p_Blur[1]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, p_Blur[1]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 3, p_Height, p_Width);

FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 2, p_Output + 3, p_Output + 0, p_EQ[1], p_Switch);
SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 3, p_Output + 2);
}
if(p_Switch == 2){
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 0, p_Output, p_Grey);
cudaError = cudaFree(tempBuffer);
return;}

if(p_EQ[2] != 1.0f || p_EQ[3] != 1.0f || p_EQ[4] != 1.0f || p_EQ[5] != 1.0f || p_Switch > 0){
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Width, p_Height, p_Blur[2]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, p_Blur[2]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 3, p_Height, p_Width);

FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 2, p_Output + 3, p_Output + 0, p_EQ[2], p_Switch);
SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 3, p_Output + 2);
}
if(p_Switch == 3){
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 0, p_Output, p_Grey);
cudaError = cudaFree(tempBuffer);
return;}

if(p_EQ[3] != 1.0f || p_EQ[4] != 1.0f || p_EQ[5] != 1.0f || p_Switch > 0){
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Width, p_Height, p_Blur[3]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, p_Blur[3]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 3, p_Height, p_Width);

FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 2, p_Output + 3, p_Output + 0, p_EQ[3], p_Switch);
SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 3, p_Output + 2);
}
if(p_Switch == 4){
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 0, p_Output, p_Grey);
cudaError = cudaFree(tempBuffer);
return;}

if(p_EQ[4] != 1.0f || p_EQ[5] != 1.0f || p_Switch > 0){
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Width, p_Height, p_Blur[4]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, p_Blur[4]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 3, p_Height, p_Width);

FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 2, p_Output + 3, p_Output + 0, p_EQ[4], p_Switch);
SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 3, p_Output + 2);
}
if(p_Switch == 5){
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 0, p_Output, p_Grey);
cudaError = cudaFree(tempBuffer);
return;}

if(p_EQ[5] != 1.0f || p_Switch > 0){
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Width, p_Height, p_Blur[5]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, p_Blur[5]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 3, p_Height, p_Width);

FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 2, p_Output + 3, p_Output + 0, p_EQ[5], p_Switch);
SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 3, p_Output + 2);
}
if(p_Switch == 6){
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 0, p_Output, p_Grey);
cudaError = cudaFree(tempBuffer);
return;}

if(p_Switch == 7){
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + 0, p_Output, p_Grey);
cudaError = cudaFree(tempBuffer);
return;}

FrequencyAdd<<<blocks, threadsT>>>(p_Width, p_Height, p_Input, p_Output, p_EQ[7]);
d_lab_to_rec709<<<blocks, threadsT>>>(p_Output, p_Width, p_Height);

cudaError = cudaFree(tempBuffer);

}