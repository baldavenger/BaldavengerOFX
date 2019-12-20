#include "ScanCudaKernel.cuh"

cudaError_t cudaError;

__device__ float Luma(float R, float G, float B, int L) {
float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f;
float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f;
float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f;
float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f;
float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f;
float lumaAvg = (R + G + B) / 3.0f;
float lumaMax = fmax(fmax(R, G), B);
float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : 
L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax;
return Lu;
}

__global__ void AlphaKernel( const float* p_Input, float* p_Output, int p_Width, int p_Height, float lumaLimit, int LumaMath) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float luma = Luma(p_Input[index], p_Input[index + 1], p_Input[index + 2], LumaMath);
float alpha = lumaLimit > 1.0f ? luma + (1.0f - lumaLimit) * (1.0f - luma) : lumaLimit >= 0.0f ? (luma >= lumaLimit ? 
1.0f : luma / lumaLimit) : lumaLimit < -1.0f ? (1.0f - luma) + (lumaLimit + 1.0f) * luma : luma <= (1.0f + lumaLimit) ? 1.0f : 
(1.0f - luma) / (1.0f - (lumaLimit + 1.0f));
float Alpha = alpha > 1.0f ? 1.0f : alpha < 0.0f ? 0.0f : alpha;
p_Output[index + 3] = Alpha;
}}

__global__ void ScanKernel( const float* p_Input, float* p_Output, int p_Width, int p_Height, float balGainR, 
float balGainB, float balOffsetR, float balOffsetB,float balLiftR, float balLiftB, float lumaBalance, 
float GainBalance, float OffsetBalance, int WhiteBalance, int PreserveLuma, int DisplayAlpha) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float Alpha = p_Output[index + 3];
float BalR = GainBalance == 1 ? p_Input[index] * balGainR : OffsetBalance == 1 ? p_Input[index] + balOffsetR : p_Input[index] + (balLiftR * (1.0f - p_Input[index]));
float BalB = GainBalance == 1 ? p_Input[index + 2] * balGainB : OffsetBalance == 1 ? p_Input[index + 2] + balOffsetB : p_Input[index + 2] + (balLiftB * (1.0f - p_Input[index + 2]));
float Red = WhiteBalance == 1 ? ( PreserveLuma == 1 ? BalR * lumaBalance : BalR) : p_Input[index ];
float Green = WhiteBalance == 1 && PreserveLuma == 1 ? p_Input[index + 1] * lumaBalance : p_Input[index + 1];
float Blue = WhiteBalance == 1 ? ( PreserveLuma == 1 ? BalB * lumaBalance : BalB) : p_Input[index + 2];
p_Output[index] = DisplayAlpha == 1 ? Alpha : Red * Alpha + p_Input[index] * (1.0f - Alpha);
p_Output[index + 1] = DisplayAlpha == 1 ? Alpha : Green * Alpha + p_Input[index + 1] * (1.0f - Alpha);
p_Output[index + 2] = DisplayAlpha == 1 ? Alpha : Blue * Alpha + p_Input[index + 2] * (1.0f - Alpha);
p_Output[index + 3] = 1.0f;
}}

__global__ void MarkerKernel( float* p_Input, int p_Width, int p_Height, int LumaMaxX, 
int LumaMaxY, int LumaMinX, int LumaMinY, int Radius, int DisplayMax, int DisplayMin) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
bool MIN = DisplayMin == 1 && (x >= LumaMinX - Radius && x <= LumaMinX + Radius && y >= LumaMinY - Radius && y <= LumaMinY + Radius);
bool MAX = DisplayMax == 1 && (x >= LumaMaxX - Radius && x <= LumaMaxX + Radius && y >= LumaMaxY - Radius && y <= LumaMaxY + Radius);
p_Input[index] = MIN ? 0.0f : MAX ? 1.0f : p_Input[index];
p_Input[index + 1] = MIN || MAX ? 1.0f : p_Input[index + 1];
p_Input[index + 2] = MAX ? 0.0f : MIN ? 1.0f : p_Input[index + 2];
}}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Gain, 
float* p_Offset, float* p_Lift, float p_LumaBalance, float p_LumaLimit, float p_Blur, int* p_Switch, 
int p_LumaMath, int* p_LumaMaxXY, int* p_LumaMinXY, int* p_DisplayXY, int Radius)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

int nthreads = 128;
float* tempBuffer;
cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);

dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
dim3 threadsT(BLOCK_DIM, BLOCK_DIM);

AlphaKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_LumaLimit, p_LumaMath);

if (p_Blur > 0.0f) {
p_Blur *= 10.0f;
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Width, p_Height, p_Blur);
d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, p_Blur);
d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + 3, p_Height, p_Width);
}

ScanKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Gain[0], p_Gain[1], p_Offset[0], p_Offset[1],
p_Lift[0], p_Lift[1], p_LumaBalance, p_Switch[0], p_Switch[1], p_Switch[2], p_Switch[3], p_Switch[4]);

if (p_DisplayXY[0] == 1 || p_DisplayXY[1] == 1)
MarkerKernel<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_LumaMaxXY[0], p_LumaMaxXY[1], 
p_LumaMinXY[0], p_LumaMinXY[1], Radius, p_DisplayXY[0], p_DisplayXY[1]);

cudaError = cudaFree(tempBuffer);
}