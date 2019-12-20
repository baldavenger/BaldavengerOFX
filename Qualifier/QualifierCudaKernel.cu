#include "QualifierCudaKernel.cuh"

cudaError_t cudaError;

__global__ void QualifierKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, float p_AlphaHA, float p_AlphaHB, 
float p_AlphaHC, float p_AlphaHD, float p_AlphaHE, float p_AlphaHF, float p_AlphaHO, float p_AlphaSA, float p_AlphaSB, float p_AlphaSC, 
float p_AlphaSD, float p_AlphaSE, float p_AlphaSF, float p_AlphaSO, float p_AlphaLA, float p_AlphaLB, float p_AlphaLC, float p_AlphaLD, 
float p_AlphaLE, float p_AlphaLF, float p_AlphaLO, int p_InvertH, int p_InvertS, int p_InvertL, int p_Math, int p_OutputAlpha,
float p_Black, float p_White, float p_HsvA, float p_HsvB, float p_HsvC) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height)
{
const int index = (y * p_Width + x) * 4;
float3 RGB = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float3 HSV = RGB_to_HSV(RGB);
float lum = Luma(RGB.x, RGB.y, RGB.z, p_Math);
if ( HSV.y > 0.1f && HSV.z > 0.1f ) {
float hh = HSV.x + p_AlphaHO;
HSV.x = hh < 0.0f ? hh + 1.0f : hh >= 1.0f ? hh - 1.0f : hh;
} else {
HSV.x = 0.0f;
}
float Hue = Alpha(p_AlphaHA, p_AlphaHB, p_AlphaHC, p_AlphaHD, p_AlphaHE, p_AlphaHF, HSV.x, p_InvertH);
float S = clamp(HSV.y + p_AlphaSO, 0.0f, 1.0f);
float Sat = Alpha(p_AlphaSA, p_AlphaSB, p_AlphaSC, p_AlphaSD, p_AlphaSE, p_AlphaSF, S, p_InvertS);
float L = clamp(lum + p_AlphaLO, 0.0f, 1.0f);
float Lum = Alpha(p_AlphaLA, p_AlphaLB, p_AlphaLC, p_AlphaLD, p_AlphaLE, p_AlphaLF, L, p_InvertL);
float A = p_OutputAlpha == 0 ? fmin(fmin(Hue, Sat), Lum) : p_OutputAlpha == 1 ? Hue : p_OutputAlpha == 2 ? Sat :
p_OutputAlpha == 3 ? Lum : p_OutputAlpha == 4 ? fmin(Hue, Sat) : p_OutputAlpha == 5 ? 
fmin(Hue, Lum) : p_OutputAlpha == 6 ? fmin(Sat, Lum) : 1.0f;
if (p_Black > 0.0f) {
A = fmax(A - (p_Black * 4.0f) * (1.0f - A), 0.0f);
}
if (p_White > 0.0f) {
A = fmin(A * (1.0f + p_White * 4.0f), 1.0f);
}
if (p_HsvA != 0.0f || p_HsvB != 0.0f || p_HsvC != 0.0f) {
float h2 = HSV.x + p_HsvA;
float H2 = h2 < 0.0f ? h2 + 1.0f : h2 >= 1.0f ? h2 - 1.0f : h2;
float S = HSV.y * (1.0f + p_HsvB);
float V = HSV.z * (1.0f + p_HsvC);
RGB = HSV_to_RGB(make_float3(H2, S, V));
}
p_Output[index] = RGB.x;
p_Output[index + 1] = RGB.y;
p_Output[index + 2] = RGB.z;
p_Output[index + 3] = 1.0f;
p_Input[index + 3] = clamp(A, 0.0f, 1.0f);
}}

__global__ void QualifierEnd( float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Display, int p_Invert, int p_Warning, float p_Mix) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height)
{
const int index = (y * p_Width + x) * 4;
float3 RGBin = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float3 RGBout = make_float3(p_Output[index], p_Output[index + 1], p_Output[index + 2]);
float A = p_Input[index + 3];
if (p_Invert == 1)
A = 1.0f - A;
if (p_Mix != 0.0f) {
if (p_Mix > 0.0f) {
A = A + (1.0f - A) * p_Mix;
} else {
A *= 1.0f + p_Mix;
}}
A = clamp(A, 0.0f, 1.0f);
float RA, GA, BA;
RA = GA = BA = A;
if (p_Warning == 1 && p_Display == 1) {
RA = A > 0.0f && A < 0.2f ? 0.0f : A < 1.0f && A > 0.8f ? 1.0f : A;
GA = A > 0.0f && A < 0.2f ? 0.0f : A < 1.0f && A > 0.8f ? 1.0f : A;
BA = A > 0.0f && A < 0.2f ? 1.0f : A < 1.0f && A > 0.8f ? 0.0f : A;
}
p_Output[index] = p_Display == 1 ? RA : RGBout.x * A + RGBin.x * (1.0f - A);
p_Output[index + 1] = p_Display == 1 ? GA : RGBout.y * A + RGBin.y * (1.0f - A);
p_Output[index + 2] = p_Display == 1 ? BA : RGBout.z * A + RGBin.z * (1.0f - A);
p_Output[index + 3] = 1.0f;
}}

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

QualifierKernel<<<blocks, Threads>>>(p_Input, p_Output, p_Width, p_Height, 
p_AlphaH[0], p_AlphaH[1], p_AlphaH[2], p_AlphaH[3], p_AlphaH[4], p_AlphaH[5], p_AlphaH[6], 
p_AlphaS[0], p_AlphaS[1], p_AlphaS[2], p_AlphaS[3], p_AlphaS[4], p_AlphaS[5], p_AlphaS[6],
p_AlphaL[0], p_AlphaL[1], p_AlphaL[2], p_AlphaL[3], p_AlphaL[4], p_AlphaL[5], p_AlphaL[6],
p_Switch[2], p_Switch[3], p_Switch[4], p_Math, p_OutputAlpha, p_Black, p_White, p_HSV[0], p_HSV[1], p_HSV[2]);
if (p_Blur > 0.0f) {
p_Blur *= 10.0f;
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 3, tempBuffer, p_Width, p_Height, p_Blur);
d_transpose<<< grid, threads >>>(tempBuffer, p_Input + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input + 3, tempBuffer, p_Height, p_Width, p_Blur);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Input + 3, p_Height, p_Width);
d_garbageCore<<<blocks, Threads>>>(p_Input + 3, p_Width, p_Height, p_Garbage, p_Core);
}
int radE = ceil(p_Erode * 15.0f);
int radD = ceil(p_Dilate * 15.0f);
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
ErosionSharedStep1<<<grid2,blockE2,blockE2.y*blockE2.x*sizeof(float)>>>(p_Input + 3, tempBuffer, radE, p_Width, p_Height, tile_w, tile_h);
cudaError = cudaDeviceSynchronize();
ErosionSharedStep2<<<grid3,blockE3,blockE3.y*blockE3.x*sizeof(float)>>>(tempBuffer, p_Input + 3, radE, p_Width, p_Height, tile_w2, tile_h2);
}

if (p_Dilate > 0.0f) {
DilateSharedStep1<<<grid2,blockD2,blockD2.y*blockD2.x*sizeof(float)>>>(p_Input + 3, tempBuffer, radD, p_Width, p_Height, tile_w, tile_h);
cudaError = cudaDeviceSynchronize();
DilateSharedStep2<<<grid3,blockD3,blockD3.y*blockD3.x*sizeof(float)>>>(tempBuffer, p_Input + 3, radD, p_Width, p_Height, tile_w2, tile_h2);
}

QualifierEnd<<<blocks, Threads>>>(p_Input, p_Output, p_Width, p_Height, p_Switch[0], p_Switch[1], p_Switch[5], p_Mix);

cudaError = cudaFree(tempBuffer);
}