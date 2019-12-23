#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

#define BLOCK_DIM	32
#define bufferLength	(p_Width * p_Height * sizeof(float))

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"float3 make_float3( float A, float B, float C) { \n" \
"float3 out; \n" \
"out.x = A; out.y = B; out.z = C; \n" \
"return out; \n" \
"} \n" \
"float Luma(float R, float G, float B, int L) { \n" \
"float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f; \n" \
"float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f; \n" \
"float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f; \n" \
"float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f; \n" \
"float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f; \n" \
"float lumaAvg = (R + G + B) / 3.0f; \n" \
"float lumaMax = fmax(fmax(R, G), B); \n" \
"float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : \n" \
"L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax; \n" \
"return Lu; \n" \
"} \n" \
"float Alpha(float p_ScaleA, float p_ScaleB, float p_ScaleC, float p_ScaleD, \n" \
"float p_ScaleE, float p_ScaleF, float N, int p_Switch) { \n" \
"float r = p_ScaleA; \n" \
"float g = p_ScaleB; \n" \
"float b = p_ScaleC; \n" \
"float a = p_ScaleD; \n" \
"float d = 1.0f / p_ScaleE; \n" \
"float e = 1.0f / p_ScaleF; \n" \
"float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= N ? 1.0f : (r >= N ? pow((r - N) / (1.0f - g), d) : 0.0f)); \n" \
"float k = a == 1.0f ? 0.0f : (a + b <= N ? 1.0f : (a <= N ? pow((N - a) / b, e) : 0.0f)); \n" \
"float alpha = k * w; \n" \
"float alphaV = p_Switch == 1 ? 1.0f - alpha : alpha; \n" \
"return alphaV; \n" \
"} \n" \
"float3 RGB_to_HSV( float3 RGB) { \n" \
"float3 HSV; \n" \
"float min = fmin(fmin(RGB.x, RGB.y), RGB.z); \n" \
"float max = fmax(fmax(RGB.x, RGB.y), RGB.z); \n" \
"HSV.z = max; \n" \
"float delta = max - min; \n" \
"if (max != 0.0f) { \n" \
"HSV.y = delta / max; \n" \
"} else { \n" \
"HSV.y = 0.0f; \n" \
"HSV.x = 0.0f; \n" \
"return HSV; \n" \
"} \n" \
"if (delta == 0.0f) { \n" \
"HSV.x = 0.0f; \n" \
"} else if (RGB.x == max) { \n" \
"HSV.x = (RGB.y - RGB.z) / delta; \n" \
"} else if (RGB.y == max) { \n" \
"HSV.x = 2.0f + (RGB.z - RGB.x) / delta; \n" \
"} else { \n" \
"HSV.x = 4.0f + (RGB.x - RGB.y) / delta; \n" \
"} \n" \
"HSV.x *= 1.0f / 6.0f; \n" \
"if (HSV.x < 0.0f) { \n" \
"HSV.x += 1.0f; \n" \
"} \n" \
"return HSV; \n" \
"} \n" \
"float3 HSV_to_RGB( float3 HSV) { \n" \
"float3 RGB; \n" \
"if (HSV.y == 0.0f) { \n" \
"RGB.x = RGB.y = RGB.z = HSV.z; \n" \
"return RGB; \n" \
"} \n" \
"HSV.x *= 6.0f; \n" \
"int i = floor(HSV.x); \n" \
"float f = HSV.x - i; \n" \
"i = (i >= 0) ? (i % 6) : (i % 6) + 6; \n" \
"float p = HSV.z * (1.0f - HSV.y); \n" \
"float q = HSV.z * (1.0f - HSV.y * f); \n" \
"float t = HSV.z * (1.0f - HSV.y * (1.0f - f)); \n" \
"RGB.x = i == 0 ? HSV.z : i == 1 ? q : i == 2 ? p : i == 3 ? p : i == 4 ? t : HSV.z; \n" \
"RGB.y = i == 0 ? t : i == 1 ? HSV.z : i == 2 ? HSV.z : i == 3 ? q : i == 4 ? p : p; \n" \
"RGB.z = i == 0 ? p : i == 1 ? p : i == 2 ? t : i == 3 ? HSV.z : i == 4 ? HSV.z : q; \n" \
"return RGB; \n" \
"} \n" \
"kernel void k_transpose( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_out [[buffer (5)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) { \n" \
"const int BLOCK_D = 32; \n" \
"threadgroup float sblock[BLOCK_D * (BLOCK_D + 1)]; \n" \
"int xIndex = blockIdx.x * BLOCK_D + threadIdx.x; \n" \
"int yIndex = blockIdx.y * BLOCK_D + threadIdx.y; \n" \
"if( (xIndex < p_Width) && (yIndex < p_Height) ) \n" \
"{ \n" \
"sblock[threadIdx.y * (BLOCK_D + 1) + threadIdx.x] = p_Input[(yIndex * p_Width + xIndex)]; \n" \
"} \n" \
"threadgroup_barrier(mem_flags::mem_threadgroup); \n" \
"xIndex = blockIdx.y * BLOCK_D + threadIdx.x; \n" \
"yIndex = blockIdx.x * BLOCK_D + threadIdx.y; \n" \
"if( (xIndex < p_Height) && (yIndex < p_Width) ) \n" \
"{ \n" \
"p_Output[(yIndex * p_Height + xIndex) * 4 + ch_out] = sblock[threadIdx.x * (BLOCK_D + 1) + threadIdx.y]; \n" \
"}} \n" \
"kernel void k_gaussian( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]],  \n" \
"constant float& blur [[buffer (15)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) { \n" \
"float nsigma = blur < 0.1f ? 0.1f : blur; \n" \
"float alpha = 1.695f / nsigma; \n" \
"float ema = exp(-alpha); \n" \
"float ema2 = exp(-2.0f * alpha), \n" \
"b1 = -2.0f * ema, \n" \
"b2 = ema2; \n" \
"float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, coefp = 0.0f, coefn = 0.0f; \n" \
"float k = (1.0f - ema) * (1.0f - ema) / (1.0f + 2.0f * alpha * ema - ema2); \n" \
"a0 = k; \n" \
"a1 = k * (alpha - 1.0f) * ema; \n" \
"a2 = k * (alpha + 1.0f) * ema; \n" \
"a3 = -k * ema2; \n" \
"coefp = (a0 + a1) / (1.0f + b1 + b2); \n" \
"coefn = (a2 + a3) / (1.0f + b1 + b2); \n" \
"int x = blockIdx.x * blockDim.x + threadIdx.x; \n" \
"if (x >= p_Width) return; \n" \
"p_Input += x * 4 + ch_in; \n" \
"p_Output += x; \n" \
"float xp, yp, yb; \n" \
"xp = *p_Input; \n" \
"yb = coefp * xp; \n" \
"yp = yb; \n" \
"for (int y = 0; y < p_Height; y++) { \n" \
"float xc = *p_Input; \n" \
"float yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb; \n" \
"*p_Output = yc; \n" \
"p_Input += p_Width * 4; \n" \
"p_Output += p_Width; \n" \
"xp = xc; \n" \
"yb = yp; \n" \
"yp = yc; \n" \
"} \n" \
"p_Input -= p_Width * 4; \n" \
"p_Output -= p_Width; \n" \
"float xn, xa, yn, ya; \n" \
"xn = xa = *p_Input; \n" \
"yn = coefn * xn; \n" \
"ya = yn; \n" \
"for (int y = p_Height - 1; y >= 0; y--) { \n" \
"float xc = *p_Input; \n" \
"float yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya; \n" \
"xa = xn; \n" \
"xn = xc; \n" \
"ya = yn; \n" \
"yn = yc; \n" \
"*p_Output = *p_Output + yc; \n" \
"p_Input -= p_Width * 4; \n" \
"p_Output -= p_Width; \n" \
"}} \n" \
"kernel void k_garbageCore( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]],  \n" \
"constant int& ch_in [[buffer (5)]], constant float& p_Garbage [[buffer (16)]], constant float& p_Core [[buffer (17)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float A = p_Input[index + ch_in]; \n" \
"if (p_Core > 0.0f) { \n" \
"float CoreA = fmin(A * (1.0f + p_Core), 1.0f); \n" \
"CoreA = fmax(CoreA + (0.0f - p_Core * 3.0f) * (1.0f - CoreA), 0.0f); \n" \
"A = fmax(A, CoreA); \n" \
"} \n" \
"if (p_Garbage > 0.0f) { \n" \
"float GarA = fmax(A + (0.0f - p_Garbage * 3.0f) * (1.0f - A), 0.0f); \n" \
"GarA = fmin(GarA * (1.0f + p_Garbage* 3.0f), 1.0f); \n" \
"A = fmin(A, GarA); \n" \
"} \n" \
"p_Input[index + ch_in] = A; \n" \
"}} \n" \
"kernel void k_erode1( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (21)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"float t = 1.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmin(t, p_Input[(clamp((int)id.y + i, 0, p_Height) * p_Width + id.x) * 4 + ch_in]); \n" \
"} \n" \
"p_Output[id.y * p_Width + id.x] = t;  \n" \
"}} \n" \
"kernel void k_erode2( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (21)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"float t = 1.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmin(t, p_Input[id.y * p_Width + clamp((int)id.x + i, 0, p_Width)]); \n" \
"} \n" \
"p_Output[(id.y * p_Width + id.x) * 4 + ch_in] = t;  \n" \
"}} \n" \
"kernel void k_dilate1( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (22)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"float t = 0.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmax(t, p_Input[(clamp((int)id.y + i, 0, p_Height) * p_Width + id.x) * 4 + ch_in]); \n" \
"} \n" \
"p_Output[id.y * p_Width + id.x] = t;  \n" \
"}} \n" \
"kernel void k_dilate2( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (22)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"float t = 0.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmax(t, p_Input[id.y * p_Width + clamp((int)id.x + i, 0, p_Width)]); \n" \
"} \n" \
"p_Output[(id.y * p_Width + id.x) * 4 + ch_in] = t;  \n" \
"}} \n" \
"kernel void k_QualifierA( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int* p_Switch [[buffer (6)]],  \n" \
"constant float* p_AlphaH [[buffer (7)]], constant float* p_AlphaS [[buffer (8)]], constant float* p_AlphaL [[buffer (9)]],  \n" \
"constant int& p_Math [[buffer (11)]], constant int& p_OutputAlpha [[buffer (12)]], constant float& p_Black [[buffer (13)]],  \n" \
"constant float& p_White [[buffer (14)]], constant float* p_Hsv [[buffer (20)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 RGB = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 HSV = RGB_to_HSV(RGB); \n" \
"float lum = Luma(RGB.x, RGB.y, RGB.z, p_Math); \n" \
"if ( HSV.y > 0.1f && HSV.z > 0.1f ) { \n" \
"float hh = HSV.x + p_AlphaH[6]; \n" \
"HSV.x = hh < 0.0f ? hh + 1.0f : hh >= 1.0f ? hh - 1.0f : hh; \n" \
"} else { HSV.x = 0.0f;} \n" \
"float Hue = Alpha(p_AlphaH[0], p_AlphaH[1], p_AlphaH[2], p_AlphaH[3], p_AlphaH[4], p_AlphaH[5], HSV.x, p_Switch[2]); \n" \
"float S = clamp(HSV.y + p_AlphaS[6], 0.0f, 1.0f); \n" \
"float Sat = Alpha(p_AlphaS[0], p_AlphaS[1], p_AlphaS[2], p_AlphaS[3], p_AlphaS[4], p_AlphaS[5], S, p_Switch[3]); \n" \
"float L = clamp(lum + p_AlphaL[6], 0.0f, 1.0f); \n" \
"float Lum = Alpha(p_AlphaL[0], p_AlphaL[1], p_AlphaL[2], p_AlphaL[3], p_AlphaL[4], p_AlphaL[5], L, p_Switch[4]); \n" \
"float A = p_OutputAlpha == 0 ? fmin(fmin(Hue, Sat), Lum) : p_OutputAlpha == 1 ? Hue : p_OutputAlpha == 2 ? Sat : \n" \
"p_OutputAlpha == 3 ? Lum : p_OutputAlpha == 4 ? fmin(Hue, Sat) : p_OutputAlpha == 5 ?  \n" \
"fmin(Hue, Lum) : p_OutputAlpha == 6 ? fmin(Sat, Lum) : 1.0f; \n" \
"if (p_Black > 0.0f) \n" \
"A = fmax(A + (0.0f - p_Black * 4.0f) * (1.0f - A), 0.0f); \n" \
"if (p_White > 0.0f) \n" \
"A = fmin(A * (1.0f + p_White * 4.0f), 1.0f); \n" \
"if (p_Hsv[0] != 0.0f || p_Hsv[1] != 0.0f || p_Hsv[2] != 0.0f) { \n" \
"float h2 = HSV.x + p_Hsv[0]; \n" \
"float H2 = h2 < 0.0f ? h2 + 1.0f : h2 >= 1.0f ? h2 - 1.0f : h2; \n" \
"float S = HSV.y * (1.0f + p_Hsv[1]); \n" \
"float V = HSV.z * (1.0f + p_Hsv[2]); \n" \
"RGB = HSV_to_RGB(make_float3(H2, S, V)); \n" \
"} \n" \
"p_Output[index] = RGB.x; \n" \
"p_Output[index + 1] = RGB.y; \n" \
"p_Output[index + 2] = RGB.z; \n" \
"p_Output[index + 3] = A; \n" \
"}} \n" \
"kernel void k_QualifierB( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int* p_Switch [[buffer (6)]],  \n" \
"constant float& p_Mix [[buffer (10)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float A = p_Output[index + 3]; \n" \
"if (p_Switch[1] == 1) \n" \
"A = 1.0f - A; \n" \
"if (p_Mix != 0.0f) { \n" \
"if (p_Mix > 0.0f) { \n" \
"A = A + (1.0f - A) * p_Mix; \n" \
"} else { \n" \
"A *= 1.0f + p_Mix; \n" \
"}} \n" \
"A = fmax(fmin(A, 1.0f), 0.0f); \n" \
"float RA, GA, BA; \n" \
"RA = GA = BA = A; \n" \
"if (p_Switch[5] == 1 && p_Switch[0] == 1) { \n" \
"RA = A > 0.0f && A < 0.2f ? 0.0f : A < 1.0f && A > 0.8f ? 1.0f : A; \n" \
"GA = A > 0.0f && A < 0.2f ? 0.0f : A < 1.0f && A > 0.8f ? 1.0f : A; \n" \
"BA = A > 0.0f && A < 0.2f ? 1.0f : A < 1.0f && A > 0.8f ? 0.0f : A; \n" \
"} \n" \
"p_Output[index] = p_Switch[0] == 1 ? RA : p_Output[index] * A + p_Input[index] * (1.0f - A); \n" \
"p_Output[index + 1] = p_Switch[0] == 1 ? GA : p_Output[index + 1] * A + p_Input[index + 1] * (1.0f - A); \n" \
"p_Output[index + 2] = p_Switch[0] == 1 ? BA : p_Output[index + 2] * A + p_Input[index + 2] * (1.0f - A); \n" \
"p_Output[index + 3] = 1.0f; \n" \
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, int* p_Switch, 
float* p_AlphaH, float* p_AlphaS, float* p_AlphaL, float p_Mix, int p_Math, int p_OutputAlpha, float p_Black, 
float p_White, float p_Blur, float p_Garbage, float p_Core, float p_Erode, float p_Dilate, float* p_HSV)
{
int red = 0;
int green = 1;
int blue = 2;
int alpha = 3;

const char* gaussian			= "k_gaussian";
const char* transpose			= "k_transpose";
const char* erode1				= "k_erode1";
const char* erode2				= "k_erode2";
const char* dilate1				= "k_dilate1";
const char* dilate2				= "k_dilate2";
const char* garbageCore			= "k_garbageCore";
const char* QualifierA			= "k_QualifierA";
const char* QualifierB			= "k_QualifierB";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLBuffer>					tempBuffer;
id<MTLComputePipelineState>    	_gaussian;
id<MTLComputePipelineState>    	_transpose;
id<MTLComputePipelineState>    	_erode1;
id<MTLComputePipelineState>    	_erode2;
id<MTLComputePipelineState>     _dilate1;
id<MTLComputePipelineState>     _dilate2;
id<MTLComputePipelineState>     _garbageCore;
id<MTLComputePipelineState>     _QualifierA;
id<MTLComputePipelineState>     _QualifierB;

NSError* err;
std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

const auto it = s_PipelineQueueMap.find(queue);
if (it == s_PipelineQueueMap.end()) {
s_PipelineQueueMap[queue] = pipelineState;
} else {
pipelineState = it->second;
}   

MTLCompileOptions* options	=	[MTLCompileOptions new];
options.fastMathEnabled		=	YES;

if (!(metalLibrary = [device newLibraryWithSource:@(kernelSource) options:options error:&err])) {
fprintf(stderr, "Failed to load metal library, %s\n", err.localizedDescription.UTF8String);
return;
}
[options release];

tempBuffer = [device newBufferWithLength:bufferLength options:0];

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:gaussian]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_gaussian = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:transpose]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_transpose = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:erode1]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_erode1 = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:erode2]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_erode2 = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:dilate1]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_dilate1 = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:dilate2]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_dilate2 = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:garbageCore]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_garbageCore = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:QualifierA]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_QualifierA = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:QualifierB]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_QualifierB = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"RunMetalKernel"];
id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
[computeEncoder setComputePipelineState:_QualifierA];
int exeWidth = [_QualifierA threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

MTLSize gausThreadGroupsA		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsB		= MTLSizeMake((p_Height + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsCount	= MTLSizeMake(exeWidth, 1, 1);

MTLSize transThreadGroupsA		= MTLSizeMake((p_Width + BLOCK_DIM - 1)/BLOCK_DIM, (p_Height + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsB		= MTLSizeMake((p_Height + BLOCK_DIM - 1)/BLOCK_DIM, (p_Width + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsCount	= MTLSizeMake(BLOCK_DIM, BLOCK_DIM, 1);

int radE = ceil(p_Erode * 15.0f);
int radD = ceil(p_Dilate * 15.0f);
float Blur = p_Blur * 10.0f;

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:p_Switch length:(sizeof(int) * 6) atIndex: 6];
[computeEncoder setBytes:p_AlphaH length:(sizeof(float) * 7) atIndex: 7];
[computeEncoder setBytes:p_AlphaS length:(sizeof(float) * 7) atIndex: 8];
[computeEncoder setBytes:p_AlphaL length:(sizeof(float) * 7) atIndex: 9];
[computeEncoder setBytes:&p_Mix length:sizeof(float) atIndex: 10];
[computeEncoder setBytes:&p_Math length:sizeof(int) atIndex: 11];
[computeEncoder setBytes:&p_OutputAlpha length:sizeof(int) atIndex: 12];
[computeEncoder setBytes:&p_Black length:sizeof(float) atIndex: 13];
[computeEncoder setBytes:&p_White length:sizeof(float) atIndex: 14];
[computeEncoder setBytes:&Blur length:sizeof(float) atIndex: 15];
[computeEncoder setBytes:&p_Garbage length:sizeof(float) atIndex: 16];
[computeEncoder setBytes:&p_Core length:sizeof(float) atIndex: 17];
[computeEncoder setBytes:&p_Erode length:sizeof(float) atIndex: 18];
[computeEncoder setBytes:&p_Dilate length:sizeof(float) atIndex: 19];
[computeEncoder setBytes:p_HSV length:(sizeof(float) * 3) atIndex: 20];
[computeEncoder setBytes:&radE length:sizeof(float) atIndex: 21];
[computeEncoder setBytes:&radD length:sizeof(float) atIndex: 22];

[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_Blur > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
if (p_Garbage > 0.0f || p_Core > 0.0f) {
[computeEncoder setComputePipelineState:_garbageCore];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}

if (radE > 0) {
[computeEncoder setComputePipelineState:_erode1];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder setComputePipelineState:_erode2];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (radD > 0) {
[computeEncoder setComputePipelineState:_dilate1];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder setComputePipelineState:_dilate2];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

[computeEncoder setComputePipelineState:_QualifierB];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
}