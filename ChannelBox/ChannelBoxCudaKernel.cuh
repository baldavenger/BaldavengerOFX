#ifndef _CHANNELBOXCUDAKERNEL_CUH_
#define _CHANNELBOXCUDAKERNEL_CUH_

#define BLOCK_DIM 32

int iDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ float Luma(float R, float G, float B, int L) {
float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f;
float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f;
float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f;
float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f;
float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f;
float lumaAvg = (R + G + B) / 3.0f;
float lumaMax = fmax(fmax(R, G), B);
float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax;
return Lu;
}

__global__ void d_garbageCore(float* p_Input, int p_Width, int p_Height, float p_Garbage, float p_Core) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = ((y * p_Width) + x) * 4;
float A = p_Input[index];
if (p_Core > 0.0f) {
float CoreA = fmin(A * (1.0f + p_Core), 1.0f);
CoreA = fmax(CoreA + (0.0f - p_Core * 3.0f) * (1.0f - CoreA), 0.0f);
A = fmax(A, CoreA);
}
if (p_Garbage > 0.0f) {
float GarA = fmax(A + (0.0f - p_Garbage * 3.0f) * (1.0f - A), 0.0f);
GarA = fmin(GarA * (1.0f + p_Garbage* 3.0f), 1.0f);
A = fmin(A, GarA);
}
p_Input[index] = A;
}}

__global__ void d_transpose(float *idata, float *odata, int width, int height) {
__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
if ((xIndex < width) && (yIndex < height))
{
unsigned int index_in = (yIndex * width + xIndex);
block[threadIdx.y][threadIdx.x] = idata[index_in];
}
__syncthreads();
xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
if ((xIndex < height) && (yIndex < width))
{
unsigned int index_out = (yIndex * height + xIndex) * 4;
odata[index_out] = block[threadIdx.x][threadIdx.y];
}}

__global__ void d_recursiveGaussian(float *id, float *od, int w, int h, float blur) {
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
const float nsigma = blur < 0.1f ? 0.1f : blur;
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
if (x >= w) return;
id += x * 4;
od += x;
float xp, yp, yb;
xp = *id;
yb = coefp*xp;
yp = yb;
for (int y = 0; y < h; y++) {
float xc = *id;
float yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb;
*od = yc;
id += w * 4;
od += w;
xp = xc;
yb = yp;
yp = yc;
}
id -= w * 4;
od -= w;
float xn, xa, yn, ya;
xn = xa = *id;
yn = coefn * xn;
ya = yn;
for (int y = h - 1; y >= 0; y--) {
float xc = *id;
float yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya;
xa = xn;
xn = xc;
ya = yn;
yn = yc;
*od = *od + yc;
id -= w * 4;
od -= w;
}}

__global__ void SimpleKernel(float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
p_Output[index] = p_Input[index];
p_Output[index + 1] = p_Input[index + 1];
p_Output[index + 2] = p_Input[index + 2];
p_Output[index + 3] = p_Input[index + 3];
}}

#endif // #ifndef _CHANNELBOXCUDAKERNEL_CUH_