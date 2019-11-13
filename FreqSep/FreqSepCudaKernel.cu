#include "FreqSepCudaKernel.cuh"

cudaError_t cudaError;

void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Space, int p_Display, float* p_Blur, float* p_Sharpen, float* p_Cont, int* p_Switch)
{   	
int nthreads = 128;
float* tempBuffer;
cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);

dim3 threadsT(128, 1, 1);
dim3 blocks(((p_Width + threadsT.x - 1) / threadsT.x), p_Height, 1);

dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
dim3 threads(BLOCK_DIM, BLOCK_DIM);

for(int c = 0; c < 4; c++) {
SimpleKernel<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
}

switch (p_Space)
{
case 0:
{
if (p_Switch[0] == 1) {
p_Blur[2] = p_Blur[1] = p_Blur[0];
}

if (p_Blur[0] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 0, tempBuffer, p_Width, p_Height, p_Blur[0]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 0, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 0, tempBuffer, p_Height, p_Width, p_Blur[0]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 0, p_Height, p_Width);
}
if (p_Blur[1] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 1, tempBuffer, p_Width, p_Height, p_Blur[1]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 1, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 1, tempBuffer, p_Height, p_Width, p_Blur[1]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 1, p_Height, p_Width);
}
if (p_Blur[2] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 2, tempBuffer, p_Width, p_Height, p_Blur[2]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 2, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Height, p_Width, p_Blur[2]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 2, p_Height, p_Width);
}

if (p_Switch[0] == 1) {
p_Sharpen[2] = p_Sharpen[1] = p_Sharpen[0];
}
for(int c = 0; c < 3; c++) {
FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c, p_Sharpen[c], p_Display);
}

if (p_Display != 1) {

if (p_Switch[0] == 1) {
p_Blur[5] = p_Blur[4] = p_Blur[3];
}

if (p_Blur[3] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 0, tempBuffer, p_Width, p_Height, p_Blur[3]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 0, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 0, tempBuffer, p_Height, p_Width, p_Blur[3]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 0, p_Height, p_Width);
}
if (p_Blur[4] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 1, tempBuffer, p_Width, p_Height, p_Blur[4]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 1, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 1, tempBuffer, p_Height, p_Width, p_Blur[4]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 1, p_Height, p_Width);
}
if (p_Blur[5] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Width, p_Height, p_Blur[5]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 2, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Height, p_Width, p_Blur[5]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 2, p_Height, p_Width);
}

for(int c = 0; c < 3; c++) {
LowFreqCont<<<blocks, threadsT>>>(p_Width, p_Height, p_Output + c, p_Cont[0], p_Cont[1], p_Switch[2], p_Display, 0);
}

if (p_Display == 0) {
for(int c = 0; c < 3; c++) {
FrequencyAdd<<<blocks, threadsT>>>(p_Width, p_Height, p_Input + c, p_Output + c);
}
}
}
}
break;

case 1:
{
d_rec709_to_yuv<<<blocks, threadsT>>>(p_Input, p_Output, p_Width, p_Height);

if (p_Blur[0] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output, tempBuffer, p_Width, p_Height, p_Blur[0]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output, tempBuffer, p_Height, p_Width, p_Blur[0]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output, p_Height, p_Width);
}

FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Input, p_Output, p_Sharpen[0], p_Display);

if (p_Display == 1) {
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output);
} else {
if (p_Switch[1] == 1) {
p_Blur[5] = p_Blur[4];
}

if (p_Blur[3] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 0, tempBuffer, p_Width, p_Height, p_Blur[3]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 0, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 0, tempBuffer, p_Height, p_Width, p_Blur[3]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 0, p_Height, p_Width);
}
if (p_Blur[4] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 1, tempBuffer, p_Width, p_Height, p_Blur[4]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 1, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 1, tempBuffer, p_Height, p_Width, p_Blur[4]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 1, p_Height, p_Width);
}
if (p_Blur[5] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Width, p_Height, p_Blur[5]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 2, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Height, p_Width, p_Blur[5]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 2, p_Height, p_Width);
}

LowFreqCont<<<blocks, threadsT>>>(p_Width, p_Height, p_Output, p_Cont[0], p_Cont[1], p_Switch[2], p_Display, 1);

if (p_Display == 0) {
FrequencyAdd<<<blocks, threadsT>>>(p_Width, p_Height, p_Input, p_Output);
d_yuv_to_rec709<<<blocks, threadsT>>>(p_Output, p_Width, p_Height);
}
}
}
break;

case 2:
{
d_rec709_to_lab<<<blocks, threadsT>>>(p_Input, p_Output, p_Width, p_Height);

if (p_Blur[0] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output, tempBuffer, p_Width, p_Height, p_Blur[0]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output, tempBuffer, p_Height, p_Width, p_Blur[0]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output, p_Height, p_Width);
}

FrequencySharpen<<<blocks, threadsT>>>(p_Width, p_Height, p_Input, p_Output, p_Sharpen[0], p_Display);

if (p_Display == 1) {
DisplayThreshold<<<blocks, threadsT>>>(p_Width, p_Height, p_Output);
} else {
if (p_Switch[1] == 1)
p_Blur[5] = p_Blur[4];

if (p_Blur[3] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 0, tempBuffer, p_Width, p_Height, p_Blur[3]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 0, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 0, tempBuffer, p_Height, p_Width, p_Blur[3]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 0, p_Height, p_Width);
}
if (p_Blur[4] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 1, tempBuffer, p_Width, p_Height, p_Blur[4]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 1, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 1, tempBuffer, p_Height, p_Width, p_Blur[4]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 1, p_Height, p_Width);
}
if (p_Blur[5] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Width, p_Height, p_Blur[5]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Output + 2, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 2, tempBuffer, p_Height, p_Width, p_Blur[5]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Output + 2, p_Height, p_Width);
}

LowFreqCont<<<blocks, threadsT>>>(p_Width, p_Height, p_Output, p_Cont[0], p_Cont[1], p_Switch[2], p_Display, 1);

if (p_Display == 0) {
FrequencyAdd<<<blocks, threadsT>>>(p_Width, p_Height, p_Input, p_Output);
d_lab_to_rec709<<<blocks, threadsT>>>(p_Output, p_Width, p_Height);
}
}
}
}

cudaError = cudaFree(tempBuffer);
}
