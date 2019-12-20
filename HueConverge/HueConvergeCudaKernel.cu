#include "HueConvergeCudaKernel.cuh"

cudaError_t cudaError;

__global__ void LogStageKernel( float* p_Input, float* p_Output, int p_Width, int p_Height, int p_SwitchLog, int p_SwitchHue,
float p_LogA, float p_LogB, float p_LogC, float p_LogD, float p_SatA, float p_SatB, float p_LumaLimit, float p_SatLimit, int p_Math, int p_Chart)
{	
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 RGB = make_float3(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2]);
if(p_Chart == 1)
RGB = Hue_Chart( p_Width, p_Height, x, y);
if(p_SwitchLog == 1)
RGB = Sigmoid( RGB, p_LogA, p_LogB, p_LogC, p_LogD);
float luma = Luma(RGB.x, RGB.y, RGB.z, p_Math);
if(p_SatA != 1.0f)
{
float minluma = fminf(RGB.x, fminf(RGB.y, RGB.z));
RGB = saturation_f3( RGB, minluma, p_SatA);
}
if(p_SatB != 1.0f)
{
float maxluma = fmaxf(RGB.x, fmaxf(RGB.y, RGB.z));
RGB = saturation_f3( RGB, maxluma, p_SatB);
}
float lumaAlpha = 1.0f;
float satAlpha = 1.0f;
if(p_LumaLimit != 0.0f && p_SwitchHue == 1)
lumaAlpha = Limiter(luma, p_LumaLimit);
float3 ych = rgb_2_ych( RGB);
if(p_SatLimit != 0.0f && p_SwitchHue == 1)
{
satAlpha = Limiter(ych.y * 10.0f, p_SatLimit);
}
p_Output[index] = ych.x;
p_Output[index + 1] = ych.y;
p_Output[index + 2] = ych.z;
p_Output[index + 3] = luma;
p_Input[index] = 1.0f * lumaAlpha * satAlpha;
}}
/*
__global__ void HueMedian(float* p_In, float* p_Out, int p_Width, int p_Height, int p_Median)
{	
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
int length = 3;
int area = 9;
int offset = 1;
if ((x < p_Width) && (y < p_Height))
{
switch (p_Median)
{
case 0:
{
}
break;
case 1:
{
float TableA[9];
for (int i = 0; i < area; ++i) {
int xx = (i >= length ? fmod(i, (float)length) : i) - offset;
int yy = floor(i / length) - offset;
TableA[i] = p_In[(((y + yy) * p_Width + x + xx) * 4) + 2];
}
p_Out[y * p_Width + x] = median(TableA, area);
}
break;
case 2:
{
length = 5;
area = 25;
offset = 2;
float TableB[25];
for (int i = 0; i < area; ++i) {
int xx = (i >= length ? fmod(i, (float)length) : i) - offset;
int yy = floor(i / length) - offset;
TableB[i] = p_In[(((y + yy) * p_Width + x + xx) * 4) + 2];
}
p_Out[y * p_Width + x] = median(TableB, area);
}}}}
*/
__global__ void HueMedian9( float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
int i, j;
float Table[9];
if ((x < p_Width) && (y < p_Height))
{
for(i = -1; i <= 1; i++) {
for(j = -1; j <= 1; j++) {
Table[(i + 1) * 3 + j + 1] = p_Input[( min(max(y + i, 0), p_Height) * p_Width + min(max(x + j, 0), p_Width) ) * 4 + 2];
}}
__syncthreads();
p_Output[y * p_Width + x] = median(Table, 9);
}}

__global__ void HueMedian25( float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
int i, j;
float Table[25];
if ((x < p_Width) && (y < p_Height))
{
for(i = -2; i <= 2; i++) {
for(j = -2; j <= 2; j++) {
Table[(i + 2) * 5 + j + 2] = p_Input[( min(max(y + i, 0), p_Height) * p_Width + min(max(x + j, 0), p_Width) ) * 4 + 2];
}}
__syncthreads();
p_Output[y * p_Width + x] = median(Table, 25);
}}

__global__ void TempReturn(float* p_Input, float* p_Temp, int p_Width, int p_Height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
p_Input[index + 2] = p_Temp[y * p_Width + x];
}}

__global__ void HueStageKernel(float* p_In, float* p_ALPHA, int p_Width, int p_Height, int p_SwitchHue, int p_SwitchHue1, 
float p_Hue1, float p_Hue2, float p_Hue3, float p_Hue4, float p_Hue5, float p_LumaLimit, float p_SatLimit, int ALPHA, int p_Isolate)
{	
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 ych = make_float3(p_In[index + 0], p_In[index + 1], p_In[index + 2]);
float lumaAlpha = 1.0f;
float satAlpha = 1.0f;
if(p_LumaLimit != 0.0f && p_SwitchHue1 == 1)
lumaAlpha = Limiter(p_In[index + 3], p_LumaLimit);
if(p_SatLimit != 0.0f && p_SwitchHue1 == 1)
satAlpha = Limiter(ych.y * 10.0f, p_SatLimit);
if(p_Isolate == ALPHA + 1)
{
float offset = 180.0f - p_Hue1;
float outside = ych.z + offset;
outside = outside > 360.0f ? outside - 360.0f : outside < 0.0f ? outside + 360.0f : outside;
if(outside > 180.0f + (p_Hue2 / 2) || outside < 180.0f - (p_Hue2 / 2))
ych.y = 0.0f;
}
float3 new_ych = modify_hue( ych, p_Hue1, p_Hue2, p_Hue3, p_Hue4, p_Hue5);
float alpha = p_ALPHA[index + ALPHA];
if(alpha != 1.0f)
{
float3 RGB = ych_2_rgb( ych);
float3 new_RGB = ych_2_rgb( new_ych);
new_RGB.x = new_RGB.x * alpha + RGB.x * (1.0f - alpha);
new_RGB.y = new_RGB.y * alpha + RGB.y * (1.0f - alpha);
new_RGB.z = new_RGB.z * alpha + RGB.z * (1.0f - alpha);
new_ych = rgb_2_ych( new_RGB);
}
p_In[index] = p_SwitchHue == 1 ? new_ych.x : p_In[index];
p_In[index + 1] = p_SwitchHue == 1 ? new_ych.y : p_In[index + 1];
p_In[index + 2] = p_SwitchHue == 1 ? new_ych.z : p_In[index + 2];
p_ALPHA[index + ALPHA + 1] = 1.0f * lumaAlpha * satAlpha;
}}

__global__ void FinalStageKernel(float* p_In, float* p_ALPHA, int p_Width, int p_Height, float p_SatSoft, int p_Display)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 ych = make_float3(p_In[index + 0], p_In[index + 1], p_In[index + 2]);
float alpha = p_ALPHA[index + 3];
if(p_SatSoft != 1.0f)
{
float soft = Sat_Soft_Clip(ych.y, p_SatSoft);
ych.y = soft * alpha + ych.y * (1.0f - alpha);
}
float luma = p_In[index + 3];
float3 rgb = ych_2_rgb( ych);
if(p_Display != 0)
{
float displayAlpha = p_Display == 1 ? p_ALPHA[index + 0] : p_Display == 2 ? 
p_ALPHA[index + 1] : p_Display == 3 ? p_ALPHA[index + 2] :  p_Display == 4 ? 
alpha : p_Display == 5 ? (ych.z / 360.0f) : p_Display == 6 ? clamp(ych.y * 10.0f, 0.0f, 1.0f) : luma;  
p_In[index] = displayAlpha;
p_In[index + 1] = displayAlpha;
p_In[index + 2] = displayAlpha;
p_In[index + 3] = 1.0f;
} else {
p_In[index] = rgb.x;
p_In[index + 1] = rgb.y;
p_In[index + 2] = rgb.z;
p_In[index + 3] = 1.0f;
}}}

void  RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, int* p_Switch, float* p_Log, float* p_Sat, 
float *p_Hue1, float *p_Hue2, float *p_Hue3, int p_Display, float *p_Blur, int p_Math, int p_HueMedian, int p_Isolate)
{
int nthreads = 128;
float* tempBuffer;
cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);

dim3 threadsT(128, 1, 1);
dim3 blocks(((p_Width + threadsT.x - 1) / threadsT.x), p_Height, 1);
dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
dim3 threads(BLOCK_DIM, BLOCK_DIM);

LogStageKernel<<<blocks, threadsT>>>(p_Input, p_Output, p_Width, p_Height, p_Switch[0], p_Switch[1],
p_Log[0], p_Log[1], p_Log[2], p_Log[3], p_Sat[0], p_Sat[1], p_Hue1[5], p_Hue1[6], p_Math, p_Switch[4]);
if(p_HueMedian == 1 || p_HueMedian == 2) {
if(p_HueMedian == 1)
HueMedian9<<<blocks, threadsT>>>(p_Output, tempBuffer, p_Width, p_Height);
if(p_HueMedian == 2)
HueMedian25<<<blocks, threadsT>>>(p_Output, tempBuffer, p_Width, p_Height);
TempReturn<<<blocks, threadsT>>>(p_Output, tempBuffer, p_Width, p_Height);
}
if (p_Blur[0] > 0.0f && p_Switch[1] == 1) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input, tempBuffer, p_Width, p_Height, p_Blur[0]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Input, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input, tempBuffer, p_Height, p_Width, p_Blur[0]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Input, p_Height, p_Width);
}
HueStageKernel<<<blocks, threadsT>>>(p_Output, p_Input, p_Width, p_Height, p_Switch[1], p_Switch[2], p_Hue1[0], p_Hue1[1], 
p_Hue1[2], p_Hue1[3], p_Hue1[4], p_Hue2[5], p_Hue2[6], 0, p_Isolate);
if (p_Blur[1] > 0.0f && p_Switch[2] == 1) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 1, tempBuffer, p_Width, p_Height, p_Blur[1]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Input + 1, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input + 1, tempBuffer, p_Height, p_Width, p_Blur[1]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Input + 1, p_Height, p_Width);
}
HueStageKernel<<<blocks, threadsT>>>(p_Output, p_Input, p_Width, p_Height, p_Switch[2], p_Switch[3], p_Hue2[0], p_Hue2[1], 
p_Hue2[2], p_Hue2[3], p_Hue2[4], p_Hue3[5], p_Hue3[6], 1, p_Isolate);
if (p_Blur[2] > 0.0f && p_Switch[3] == 1) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 2, tempBuffer, p_Width, p_Height, p_Blur[2]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Input + 2, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input + 2, tempBuffer, p_Height, p_Width, p_Blur[2]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Input + 2, p_Height, p_Width);
}
HueStageKernel<<<blocks, threadsT>>>(p_Output, p_Input, p_Width, p_Height, p_Switch[3], 1, p_Hue3[0], p_Hue3[1], 
p_Hue3[2], p_Hue3[3], p_Hue3[4], p_Sat[3], 0.0f, 2, p_Isolate);
if (p_Blur[3] > 0.0f) {
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Input + 3, tempBuffer, p_Width, p_Height, p_Blur[3]);
d_transpose<<< grid, threads >>>(tempBuffer, p_Input + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Input + 3, tempBuffer, p_Height, p_Width, p_Blur[3]);
d_transpose<<< gridT, threads >>>(tempBuffer, p_Input + 3, p_Height, p_Width);
}
FinalStageKernel<<<blocks, threadsT>>>(p_Output, p_Input, p_Width, p_Height, p_Sat[2], p_Display);

cudaError = cudaFree(tempBuffer);
}