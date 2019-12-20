#ifndef _QUALIFIERCUDAKERNEL_CUH_
#define _QUALIFIERCUDAKERNEL_CUH_

#define BLOCK_DIM 32

int iDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ float clamp( float A, float B, float C) {
float D = fmax(A, B);
D = fmin(D, C);
return D;
}

__device__ float Luma( float R, float G, float B, int L) {
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

__device__ float Alpha( float p_ScaleA, float p_ScaleB, float p_ScaleC, 
float p_ScaleD, float p_ScaleE, float p_ScaleF, float N, int p_Switch) {
float r = p_ScaleA;
float g = p_ScaleB;
float b = p_ScaleC;
float a = p_ScaleD;				
float d = 1.0f / p_ScaleE;							
float e = 1.0f / p_ScaleF;											 
float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= N ? 1.0f : (r >= N ? powf((r - N) / (1.0f - g), d) : 0.0f));		
float k = a == 1.0f ? 0.0f : (a + b <= N ? 1.0f : (a <= N ? powf((N - a) / b, e) : 0.0f));						
float alpha = k * w;											 
float alphaV = p_Switch == 1 ? 1.0f - alpha : alpha;
return alphaV;
}

__device__ float3 RGB_to_HSV( float3 RGB) {
float3 HSV;
float min = fmin(fmin(RGB.x, RGB.y), RGB.z);
float max = fmax(fmax(RGB.x, RGB.y), RGB.z);
HSV.z = max;
float delta = max - min;
if (max != 0.0f) {
HSV.y = delta / max;
} else {
HSV.y = 0.0f;
HSV.x = 0.0f;
return HSV;
}
if (delta == 0.0f) {
HSV.x = 0.0f;
} else if (RGB.x == max) {
HSV.x = (RGB.y - RGB.z) / delta;
} else if (RGB.y == max) {
HSV.x = 2.0f + (RGB.z - RGB.x) / delta;
} else {
HSV.x = 4.0f + (RGB.x - RGB.y) / delta;
}
HSV.x *= 1.0f / 6.0f;
if (HSV.x < 0.0f) {
HSV.x += 1.0f;
}
return HSV;
}

__device__ float3 HSV_to_RGB( float3 HSV) {
float3 RGB;
if (HSV.y == 0.0f) {
RGB.x = RGB.y = RGB.z = HSV.z;
return RGB;
}
HSV.x *= 6.0f;
int i = floor(HSV.x);
float f = HSV.x - i;
i = (i >= 0) ? (i % 6) : (i % 6) + 6;
float p = HSV.z * (1.0f - HSV.y);
float q = HSV.z * (1.0f - HSV.y * f);
float t = HSV.z * (1.0f - HSV.y * (1.0f - f));
RGB.x = i == 0 ? HSV.z : i == 1 ? q : i == 2 ? p : i == 3 ? p : i == 4 ? t : HSV.z;
RGB.y = i == 0 ? t : i == 1 ? HSV.z : i == 2 ? HSV.z : i == 3 ? q : i == 4 ? p : p;
RGB.z = i == 0 ? p : i == 1 ? p : i == 2 ? t : i == 3 ? HSV.z : i == 4 ? HSV.z : q;
return RGB;
}

__global__ void d_garbageCore( float* p_Input, int p_Width, int p_Height, float p_Garbage, float p_Core) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height)
{
const int index = (y * p_Width + x) * 4;
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

__global__ void d_transpose( float* Input, float* Output, int p_Width, int p_Height) {
__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];
unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
if (xIndex < p_Width && yIndex < p_Height)
{
unsigned int index_in = (yIndex * p_Width + xIndex);
block[threadIdx.y][threadIdx.x] = Input[index_in];
}
__syncthreads();
xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
if (xIndex < p_Height && yIndex < p_Width)
{
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

__global__ void ErosionSharedStep1( float* Input, float* Output, int radio, int width, int height, int tile_w, int tile_h) {
extern __shared__ float smem[];
int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x;
int by = blockIdx.y;
int x = bx * tile_w + tx - radio;
int y = by * tile_h + ty;
smem[ty * blockDim.x + tx] = 1.0f;
__syncthreads();
if (x < 0 || x >= width || y >= height) {
return;
}
smem[ty * blockDim.x + tx] = Input[(y * width + x) * 4];
__syncthreads();
if (x < bx * tile_w || x >= (bx + 1) * tile_w) {
return;
}
float* smem_thread = &smem[ty * blockDim.x + tx - radio];
float val = smem_thread[0];
for (int xx = 1; xx <= 2 * radio; xx++) {
val = fmin(val, smem_thread[xx]);
}
Output[y * width + x] = val;
}

__global__ void ErosionSharedStep2( float* Input, float* Output, int radio, int width, int height, int tile_w, int tile_h) {
extern __shared__ float smem[];
int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x;
int by = blockIdx.y;
int x = bx * tile_w + tx;
int y = by * tile_h + ty - radio;
smem[ty * blockDim.x + tx] = 1.0f;
__syncthreads();
if (x >= width || y < 0 || y >= height) {
return;
}
smem[ty * blockDim.x + tx] = Input[y * width + x];
__syncthreads();
if (y < by * tile_h || y >= (by + 1) * tile_h) {
return;
}
float * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
float val = smem_thread[0];
for (int yy = 1; yy <= 2 * radio; yy++) {
val = fmin(val, smem_thread[yy * blockDim.x]);
}
Output[(y * width + x) * 4] = val;
}

__global__ void DilateSharedStep1( float* Input, float* Output, int radio, int width, int height, int tile_w, int tile_h) {
extern __shared__ float smem[];
int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x;
int by = blockIdx.y;
int x = bx * tile_w + tx - radio;
int y = by * tile_h + ty;
smem[ty * blockDim.x + tx] = 0.0f;
__syncthreads();
if (x < 0 || x >= width || y >= height) {
return;
}
smem[ty * blockDim.x + tx] = Input[(y * width + x) * 4];
__syncthreads();
if (x < bx * tile_w || x >= (bx + 1) * tile_w) {
return;
}
float* smem_thread = &smem[ty * blockDim.x + tx - radio];
float val = smem_thread[0];
for (int xx = 1; xx <= 2 * radio; xx++) {
val = fmax(val, smem_thread[xx]);
}
Output[y * width + x] = val;
}

__global__ void DilateSharedStep2( float* Input, float* Output, int radio, int width, int height, int tile_w, int tile_h) {
extern __shared__ float smem[];
int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x;
int by = blockIdx.y;
int x = bx * tile_w + tx;
int y = by * tile_h + ty - radio;
smem[ty * blockDim.x + tx] = 0.0f;
__syncthreads();
if (x >= width || y < 0 || y >= height) {
return;
}
smem[ty * blockDim.x + tx] = Input[y * width + x];
__syncthreads();
if (y < by * tile_h || y >= (by + 1) * tile_h) {
return;
}
float * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
float val = smem_thread[0];
for (int yy = 1; yy <= 2 * radio; yy++) {
val = fmax(val, smem_thread[yy * blockDim.x]);
}
Output[(y * width + x) * 4] = val;
}

#endif // #ifndef _QUALIFIERCUDAKERNEL_CUH_