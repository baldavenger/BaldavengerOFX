#ifndef _SCANCUDAKERNEL_CUH_
#define _SCANCUDAKERNEL_CUH_

#define BLOCK_DIM 32

int iDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void d_transpose( float* Input, float* Output, int p_Width, int p_Height) {
__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];
unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
if (xIndex < p_Width && yIndex < p_Height) {
unsigned int index_in = (yIndex * p_Width + xIndex);
block[threadIdx.y][threadIdx.x] = Input[index_in];
}
__syncthreads();
xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
if (xIndex < p_Height && yIndex < p_Width) {
unsigned int index_out = (yIndex * p_Height + xIndex) * 4;
Output[index_out] = block[threadIdx.x][threadIdx.y];
}}

__global__ void d_recursiveGaussian( float* Input, float* Output, int p_Width, int p_Height, float p_Blur) {
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
const float nsigma = p_Blur < 0.1f ? 0.1f : p_Blur;
float alpha = 1.695f / nsigma;
float ema = exp(-alpha);
float ema2 = exp(-2.0f * alpha);
float b1 = -2.0f * ema;
float b2 = ema2;
float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, coefp = 0.0f, coefn = 0.0f;
const float k = (1.0f - ema) * (1.0f - ema) / (1.0f + 2.0f * alpha * ema - ema2);
a0 = k;
a1 = k * (alpha - 1.0f) * ema;
a2 = k * (alpha + 1.0f) * ema;
a3 = -k * ema2;
coefp = (a0 + a1) / (1.0f + b1 + b2);
coefn = (a2 + a3) / (1.0f + b1 + b2);
if (x >= p_Width) return;
Input += x * 4;
Output += x;
float xp, yp, yb;
xp = *Input;
yb = coefp * xp;
yp = yb;
for (int y = 0; y < p_Height; y++) {
float xc = *Input;
float yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb;
*Output = yc;
Input += p_Width * 4;
Output += p_Width;
xp = xc;
yb = yp;
yp = yc;
}
Input -= p_Width * 4;
Output -= p_Width;
float xn, xa, yn, ya;
xn = xa = *Input;
yn = coefn * xn;
ya = yn;
for (int y = p_Height - 1; y >= 0; y--) {
float xc = *Input;
float yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya;
xa = xn;
xn = xc;
ya = yn;
yn = yc;
*Output = *Output + yc;
Input -= p_Width * 4;
Output -= p_Width;
}}

#endif // #ifndef _SCANCUDAKERNEL_CUH_