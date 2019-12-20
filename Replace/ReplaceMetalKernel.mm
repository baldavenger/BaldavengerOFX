#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

#define BLOCK_DIM	32
#define bufferLength	(p_Width * p_Height * sizeof(float))

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"float Mix( float A, float B, float mix) { \n" \
"float C = A * (1.0f - mix) + B * mix; \n" \
"return C; \n" \
"} \n" \
"kernel void k_replaceKernelA( device const float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant float* p_Hue [[buffer (5)]],  \n" \
"constant float* p_Sat [[buffer (6)]], constant float* p_Val [[buffer (7)]], constant float* p_Blur [[buffer (8)]],  \n" \
"constant int& OutputAlpha [[buffer (10)]], constant int& DisplayAlpha [[buffer (11)]], uint2 id [[ thread_position_in_grid ]]) {  \n" \
"if (id.x < p_Width && id.y < p_Height ) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float hcoeff, scoeff, vcoeff; \n" \
"float r, g, b, h, s, v; \n" \
"r = p_Input[index]; \n" \
"g = p_Input[index + 1]; \n" \
"b = p_Input[index + 2]; \n" \
"float min = fmin(fmin(r, g), b); \n" \
"float max = fmax(fmax(r, g), b); \n" \
"v = max; \n" \
"float delta = max - min; \n" \
"if (max != 0.0f) { \n" \
"s = delta / max; \n" \
"} else { \n" \
"s = 0.0f; \n" \
"h = 0.0f; \n" \
"} \n" \
"if (delta == 0.0f) { \n" \
"h = 0.0f; \n" \
"} else if (r == max) { \n" \
"h = (g - b) / delta; \n" \
"} else if (g == max) { \n" \
"h = 2.0f + (b - r) / delta; \n" \
"} else { \n" \
"h = 4.0f + (r - g) / delta; \n" \
"} \n" \
"h *= 1.0f / 6.0f; \n" \
"if (h < 0.0f) { \n" \
"h += 1.0f; \n" \
"} \n" \
"h *= 360.0f; \n" \
"float h0 = p_Hue[0]; \n" \
"float h1 = p_Hue[1]; \n" \
"float h0mrolloff = p_Hue[2]; \n" \
"float h1prolloff = p_Hue[3]; \n" \
"if ( ( h1 < h0 && (h <= h1 || h0 <= h) ) || (h0 <= h && h <= h1) ) { \n" \
"hcoeff = 1.0f; \n" \
"} else { \n" \
"float c0 = 0.0f; \n" \
"float c1 = 0.0f; \n" \
"if ( ( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0) ) { \n" \
"c0 = h0 == (h0mrolloff + 360.0f) || h0 == h0mrolloff ? 1.0f : !(( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0)) ? 0.0f :  \n" \
"((h < h0mrolloff ? h + 360.0f : h) - h0mrolloff) / ((h0 < h0mrolloff ? h0 + 360.0f : h0) - h0mrolloff);		 \n" \
"} \n" \
"if ( ( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff) ) { \n" \
"c1 = !(( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff)) ? 0.0f : h1prolloff == h1 ? 1.0f : \n" \
"((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - (h < h1 ? h + 360.0f : h)) / ((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - h1);	 \n" \
"} \n" \
"hcoeff = fmax(c0, c1); \n" \
"} \n" \
"float s0 = p_Sat[0]; \n" \
"float s1 = p_Sat[1]; \n" \
"float s0mrolloff = s0 - p_Sat[4]; \n" \
"float s1prolloff = s1 + p_Sat[4]; \n" \
"if ( s0 <= s && s <= s1 ) { \n" \
"scoeff = 1.0f; \n" \
"} else if ( s0mrolloff <= s && s <= s0 ) { \n" \
"scoeff = (s - s0mrolloff) / p_Sat[4]; \n" \
"} else if ( s1 <= s && s <= s1prolloff ) { \n" \
"scoeff = (s1prolloff - s) / p_Sat[4]; \n" \
"} else { \n" \
"scoeff = 0.0f; \n" \
"} \n" \
"float v0 = p_Val[0]; \n" \
"float v1 = p_Val[1]; \n" \
"float v0mrolloff = v0 - p_Val[4]; \n" \
"float v1prolloff = v1 + p_Val[4]; \n" \
"if ( (v0 <= v) && (v <= v1) ) { \n" \
"vcoeff = 1.0f; \n" \
"} else if ( v0mrolloff <= v && v <= v0 ) { \n" \
"vcoeff = (v - v0mrolloff) / p_Val[4]; \n" \
"} else if ( v1 <= v && v <= v1prolloff ) { \n" \
"vcoeff = (v1prolloff - v) / p_Val[4]; \n" \
"} else { \n" \
"vcoeff = 0.0f; \n" \
"} \n" \
"float coeff = fmin(fmin(hcoeff, scoeff), vcoeff); \n" \
"float A = OutputAlpha == 0 ? 1.0f : OutputAlpha == 1 ? hcoeff : OutputAlpha == 2 ? scoeff : \n" \
"OutputAlpha == 3 ? vcoeff : OutputAlpha == 4 ? fmin(hcoeff, scoeff) : OutputAlpha == 5 ?  \n" \
"fmin(hcoeff, vcoeff) : OutputAlpha == 6 ? fmin(scoeff, vcoeff) : fmin(fmin(hcoeff, scoeff), vcoeff); \n" \
"if (DisplayAlpha == 0) \n" \
"A = coeff; \n" \
"if (p_Blur[0] > 0.0f) \n" \
"A = fmax(A - (p_Blur[0] * 4.0f) * (1.0f - A), 0.0f); \n" \
"if (p_Blur[1] > 0.0f) \n" \
"A = fmin(A * (1.0f + p_Blur[0] * 4.0f), 1.0f); \n" \
"p_Output[index] = h; \n" \
"p_Output[index + 1] = s; \n" \
"p_Output[index + 2] = v; \n" \
"p_Output[index + 3] = A; \n" \
"}} \n" \
"kernel void k_replaceKernelB( device const float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant float* p_Hue [[buffer (5)]],  \n" \
"constant float* p_Sat [[buffer (6)]], constant float* p_Val [[buffer (7)]], constant float& mix [[buffer (9)]],  \n" \
"constant int& DisplayAlpha [[buffer (11)]], uint2 id [[ thread_position_in_grid ]]) {  \n" \
"if (id.x < p_Width && id.y < p_Height ) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float h, s, v, R, G, B, coeff; \n" \
"h = p_Output[index]; \n" \
"s = p_Output[index + 1]; \n" \
"v = p_Output[index + 2]; \n" \
"coeff = p_Output[index + 3]; \n" \
"float s0 = p_Sat[0]; \n" \
"float s1 = p_Sat[1]; \n" \
"float v0 = p_Val[0]; \n" \
"float v1 = p_Val[1]; \n" \
"float H = (h - p_Hue[6] + 180.0f) - (int)(floor((h - p_Hue[6] + 180.0f) / 360.0f) * 360.0f) - 180.0f; \n" \
"h += coeff * ( p_Hue[4] + (p_Hue[5] - 1.0f) * H ); \n" \
"s += coeff * ( p_Sat[2] + (p_Sat[3] - 1.0f) * (s - (s0 + s1) / 2.0f) ); \n" \
"if (s < 0.0f) { \n" \
"s = 0.0f; \n" \
"} \n" \
"v += coeff * ( p_Val[2] + (p_Val[3] - 1.0f) * (v - (v0 + v1) / 2.0f) ); \n" \
"h *= 1.0f / 360.0f; \n" \
"if (s == 0.0f) \n" \
"R = G = B = v; \n" \
"h *= 6.0f; \n" \
"int i = floor(h); \n" \
"float f = h - i; \n" \
"i = (i >= 0) ? (i % 6) : (i % 6) + 6; \n" \
"float p = v * ( 1.0f - s ); \n" \
"float q = v * ( 1.0f - s * f ); \n" \
"float t = v * ( 1.0f - s * ( 1.0f - f )); \n" \
"if (i == 0){ \n" \
"R = v; \n" \
"G = t; \n" \
"B = p;} \n" \
"else if (i == 1){ \n" \
"R = q; \n" \
"G = v; \n" \
"B = p;} \n" \
"else if (i == 2){ \n" \
"R = p; \n" \
"G = v; \n" \
"B = t;} \n" \
"else if (i == 3){ \n" \
"R = p; \n" \
"G = q; \n" \
"B = v;} \n" \
"else if (i == 4){ \n" \
"R = t; \n" \
"G = p; \n" \
"B = v;} \n" \
"else{ \n" \
"R = v; \n" \
"G = p; \n" \
"B = q; \n" \
"} \n" \
"p_Output[index] = DisplayAlpha == 1 ? coeff : Mix(R, p_Input[index], mix); \n" \
"p_Output[index + 1] = DisplayAlpha == 1 ? coeff : Mix(G, p_Input[index + 1], mix); \n" \
"p_Output[index + 2] = DisplayAlpha == 1 ? coeff : Mix(B, p_Input[index + 2], mix); \n" \
"p_Output[index + 3] = 1.0f; \n" \
"}} \n" \
"kernel void k_gaussian( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant float* p_Blur [[buffer (8)]], uint2 threadIdx [[ thread_position_in_threadgroup ]],  \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) { \n" \
"float nsigma = p_Blur[2] < 0.1f ? 0.1f : p_Blur[2]; \n" \
"float alpha = 1.695f / nsigma; \n" \
"float ema = exp(-alpha); \n" \
"float ema2 = exp(-2.0f * alpha); \n" \
"float b1 = -2.0f * ema; \n" \
"float b2 = ema2; \n" \
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
"p_Input += x * 4 + 3; \n" \
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
"kernel void k_transpose( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]],  \n" \
"uint2 threadIdx [[ thread_position_in_threadgroup ]], uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) { \n" \
"const int BLOCK_D = 32; \n" \
"threadgroup float sblock[BLOCK_D * (BLOCK_D + 1)]; \n" \
"int xIndex = blockIdx.x * BLOCK_D + threadIdx.x; \n" \
"int yIndex = blockIdx.y * BLOCK_D + threadIdx.y; \n" \
"if (xIndex < p_Width && yIndex < p_Height) { \n" \
"sblock[threadIdx.y * (BLOCK_D + 1) + threadIdx.x] = p_Input[(yIndex * p_Width + xIndex)]; \n" \
"} \n" \
"threadgroup_barrier(mem_flags::mem_threadgroup); \n" \
"xIndex = blockIdx.y * BLOCK_D + threadIdx.x; \n" \
"yIndex = blockIdx.x * BLOCK_D + threadIdx.y; \n" \
"if( (xIndex < p_Height) && (yIndex < p_Width) ) { \n" \
"p_Output[(yIndex * p_Height + xIndex) * 4 + 3] = sblock[threadIdx.x * (BLOCK_D + 1) + threadIdx.y]; \n" \
"}} \n" \
"kernel void k_garbageCore( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]],  \n" \
"constant float* p_Blur [[buffer (8)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float A = p_Input[index + 3]; \n" \
"if (p_Blur[4] > 0.0f) { \n" \
"float CoreA = fmin(A * (1.0f + p_Blur[4]), 1.0f); \n" \
"CoreA = fmax(CoreA + (0.0f - p_Blur[4] * 3.0f) * (1.0f - CoreA), 0.0f); \n" \
"A = fmax(A, CoreA); \n" \
"} \n" \
"if (p_Blur[3] > 0.0f) { \n" \
"float GarA = fmax(A + (0.0f - p_Blur[3] * 3.0f) * (1.0f - A), 0.0f); \n" \
"GarA = fmin(GarA * (1.0f + p_Blur[3] * 3.0f), 1.0f); \n" \
"A = fmin(A, GarA); \n" \
"} \n" \
"p_Input[index + 3] = A; \n" \
"}} \n" \
"kernel void k_erode1( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& Radius [[buffer (12)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height ) { \n" \
"float t = 1.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmin(t, p_Input[(clamp((int)id.y + i, 0, p_Height) * p_Width + id.x) * 4 + 3]); \n" \
"} \n" \
"p_Output[id.y * p_Width + id.x] = t;  \n" \
"}} \n" \
"kernel void k_erode2( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& Radius [[buffer (12)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"float t = 1.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmin(t, p_Input[id.y * p_Width + clamp((int)id.x + i, 0, p_Width)]); \n" \
"} \n" \
"p_Output[(id.y * p_Width + id.x) * 4 + 3] = t;  \n" \
"}} \n" \
"kernel void k_dilate1( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& Radius [[buffer (13)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"float t = 0.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmax(t, p_Input[(clamp((int)id.y + i, 0, p_Height) * p_Width + id.x) * 4 + 3]); \n" \
"} \n" \
"p_Output[id.y * p_Width + id.x] = t;  \n" \
"}} \n" \
"kernel void k_dilate2( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& Radius [[buffer (13)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"float t = 0.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmax(t, p_Input[id.y * p_Width + clamp((int)id.x + i, 0, p_Width)]); \n" \
"} \n" \
"p_Output[(id.y * p_Width + id.x) * 4 + 3] = t;  \n" \
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Hue, 
float* p_Sat, float* p_Val, int OutputAlpha, int DisplayAlpha, float mix, float* p_Blur)
{
const char* ReplaceKernelA		= "k_replaceKernelA";
const char* ReplaceKernelB		= "k_replaceKernelB";
const char* gaussian			= "k_gaussian";
const char* transpose			= "k_transpose";
const char* garbageCore			= "k_garbageCore";
const char* erode1				= "k_erode1";
const char* erode2				= "k_erode2";
const char* dilate1				= "k_dilate1";
const char* dilate2				= "k_dilate2";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLBuffer>					tempBuffer;
id<MTLComputePipelineState>     _ReplaceKernelA;
id<MTLComputePipelineState>     _ReplaceKernelB;
id<MTLComputePipelineState>    	_gaussian;
id<MTLComputePipelineState>    	_transpose;
id<MTLComputePipelineState>     _garbageCore;
id<MTLComputePipelineState>    	_erode1;
id<MTLComputePipelineState>    	_erode2;
id<MTLComputePipelineState>     _dilate1;
id<MTLComputePipelineState>     _dilate2;

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

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:ReplaceKernelA]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_ReplaceKernelA = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:ReplaceKernelB]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_ReplaceKernelB = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

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

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"RunMetalKernel"];

id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

[computeEncoder setComputePipelineState:_ReplaceKernelA];

int exeWidth = [_ReplaceKernelA threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

MTLSize gausThreadGroupsA		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsB		= MTLSizeMake((p_Height + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsCount	= MTLSizeMake(exeWidth, 1, 1);

MTLSize transThreadGroupsA		= MTLSizeMake((p_Width + BLOCK_DIM - 1)/BLOCK_DIM, (p_Height + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsB		= MTLSizeMake((p_Height + BLOCK_DIM - 1)/BLOCK_DIM, (p_Width + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsCount	= MTLSizeMake(BLOCK_DIM, BLOCK_DIM, 1);

int radE = ceil(p_Blur[5] * 15.0f);
int radD = ceil(p_Blur[6] * 15.0f);
p_Blur[2] = p_Blur[2] * 10.0f;

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:p_Hue length:(sizeof(int) * 8) atIndex: 5];
[computeEncoder setBytes:p_Sat length:(sizeof(float) * 5) atIndex: 6];
[computeEncoder setBytes:p_Val length:(sizeof(float) * 5) atIndex: 7];
[computeEncoder setBytes:p_Blur length:(sizeof(float) * 7) atIndex: 8];
[computeEncoder setBytes:&mix length:sizeof(float) atIndex: 9];
[computeEncoder setBytes:&OutputAlpha length:sizeof(int) atIndex: 10];
[computeEncoder setBytes:&DisplayAlpha length:sizeof(int) atIndex: 11];
[computeEncoder setBytes:&radE length:sizeof(float) atIndex: 12];
[computeEncoder setBytes:&radD length:sizeof(float) atIndex: 13];

[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_Blur[2] > 0.0f) {
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
if (p_Blur[3] > 0.0f || p_Blur[4] > 0.0f) {
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

[computeEncoder setComputePipelineState:_ReplaceKernelB];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder endEncoding];
[commandBuffer commit];
}