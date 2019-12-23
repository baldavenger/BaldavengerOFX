#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

#define BLOCK_DIM	32
#define bufferLength	(p_Width * p_Height * sizeof(float))

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"kernel void k_simple( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{                                 				    \n" \
"if ((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Output[index + ch_in] = p_Input[index + ch_in];  \n" \
"}} \n" \
"kernel void k_gaussian( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]],  \n" \
"constant float& blur [[buffer (11)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) \n" \
"{ \n" \
"float nsigma = blur < 0.1f ? 0.1f : blur; \n" \
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
"kernel void k_simpleRecursive( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]],  \n" \
"constant float& blur [[buffer (11)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) \n" \
"{ \n" \
"int x = blockIdx.x * blockDim.x + threadIdx.x; \n" \
"const float nsigma = blur < 0.1f ? 0.1f : blur; \n" \
"float alpha = 1.695f / nsigma; \n" \
"float ema = exp(-alpha); \n" \
"if (x >= p_Width) return; \n" \
"p_Input += x * 4 + ch_in; \n" \
"p_Output += x; \n" \
"float yp = *p_Input; \n" \
"for (int y = 0; y < p_Height; y++) { \n" \
"float xc = *p_Input; \n" \
"float yc = xc + ema * (yp - xc); \n" \
"*p_Output = yc; \n" \
"p_Input += p_Width * 4; \n" \
"p_Output += p_Width; \n" \
"yp = yc; \n" \
"} \n" \
"p_Input -= p_Width * 4; \n" \
"p_Output -= p_Width; \n" \
"yp = *p_Input; \n" \
"for (int y = p_Height - 1; y >= 0; y--) { \n" \
"float xc = *p_Input; \n" \
"float yc = xc + ema * (yp - xc); \n" \
"*p_Output = (*p_Output + yc) * 0.5f; \n" \
"p_Input -= p_Width * 4; \n" \
"p_Output -= p_Width; \n" \
"yp = yc; \n" \
"}} \n" \
"kernel void k_boxfilter( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (6)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) \n" \
"{ \n" \
"p_Input += ch_in; \n" \
"int x = blockIdx.x * blockDim.x + threadIdx.x; \n" \
"if (x >= p_Width) return; \n" \
"float scale = 1.0f / (float)((Radius << 1) + 1); \n" \
"float t; \n" \
"t = p_Input[x * 4] * Radius; \n" \
"for (int y = 0; y < (Radius + 1); y++) { \n" \
"t += p_Input[(y * p_Width + x) * 4]; \n" \
"} \n" \
"p_Output[x] = t * scale; \n" \
"for (int y = 1; y < (Radius + 1); y++) { \n" \
"t += p_Input[((y + Radius) * p_Width + x) * 4]; \n" \
"t -= p_Input[x * 4]; \n" \
"p_Output[y * p_Width + x] = t * scale; \n" \
"} \n" \
"for (int y = (Radius + 1); y < (p_Height - Radius); y++) { \n" \
"t += p_Input[((y + Radius) * p_Width + x) * 4]; \n" \
"t -= p_Input[(((y - Radius) * p_Width + x) - p_Width) * 4]; \n" \
"p_Output[y * p_Width + x] = t * scale; \n" \
"} \n" \
"for (int y = p_Height - Radius; y < p_Height; y++) { \n" \
"t += p_Input[((p_Height - 1) * p_Width + x) * 4]; \n" \
"t -= p_Input[(((y - Radius) * p_Width + x) - p_Width) * 4]; \n" \
"p_Output[y * p_Width + x] = t * scale; \n" \
"}} \n" \
"kernel void k_transpose( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_out [[buffer (5)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) \n" \
"{ \n" \
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
"kernel void k_erode1( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (6)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{                                 				    \n" \
"if ((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"float t = 1.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmin(t, p_Input[(clamp((int)id.y + i, 0, p_Height) * p_Width + id.x) * 4 + ch_in]); \n" \
"} \n" \
"p_Output[id.y * p_Width + id.x] = t;  \n" \
"}} \n" \
"kernel void k_erode2( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (6)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{                                 				    \n" \
"if ((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"float t = 1.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmin(t, p_Input[id.y * p_Width + clamp((int)id.x + i, 0, p_Width)]); \n" \
"} \n" \
"p_Output[(id.y * p_Width + id.x) * 4 + ch_in] = t;  \n" \
"}} \n" \
"kernel void k_dilate1( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (6)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{                                 				    \n" \
"if ((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"float t = 0.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmax(t, p_Input[(clamp((int)id.y + i, 0, p_Height) * p_Width + id.x) * 4 + ch_in]); \n" \
"} \n" \
"p_Output[id.y * p_Width + id.x] = t;  \n" \
"}} \n" \
"kernel void k_dilate2( device float* p_Input [[buffer (2)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& Radius [[buffer (6)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{                                 				    \n" \
"if ((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"float t = 0.0f; \n" \
"for (int i = -Radius; i < (Radius + 1); i++) { \n" \
"t = fmax(t, p_Input[id.y * p_Width + clamp((int)id.x + i, 0, p_Width)]); \n" \
"} \n" \
"p_Output[(id.y * p_Width + id.x) * 4 + ch_in] = t;  \n" \
"}} \n" \
"kernel void k_freqSharpen( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& p_Display [[buffer (8)]],  \n" \
"constant float& sharpen [[buffer (12)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"float offset = p_Display == 1 ? 0.5f : 0.0f; \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Input[index + ch_in] = (p_Input[index + ch_in] - p_Output[index + ch_in]) * sharpen + offset; \n" \
"if (p_Display == 1) { \n" \
"p_Output[index + ch_in] = p_Input[index + ch_in]; \n" \
"}}} \n" \
"kernel void k_freqAdd( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Output[index + ch_in] = p_Input[index + ch_in] + p_Output[index + ch_in];												   \n" \
"}} \n" \
"kernel void k_edgeDetect( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant float& p_Threshold [[buffer (11)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"float Threshold = p_Threshold + 3.0f; \n" \
"Threshold *= 3.0f; \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"for(int c = 0; c < 3; c++) {																								    \n" \
"p_Output[index + c] = (p_Input[index + c] - p_Output[index + c]) * Threshold; \n" \
"}}} \n" \
"kernel void k_edgeEnhance( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant float& p_Enhance [[buffer (11)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"int X = id.x == 0 ? 1 : id.x; \n" \
"int indexE = (id.y * p_Width + X) * 4; \n" \
"p_Output[index] = (p_Input[index] - p_Input[indexE - 4]) * p_Enhance; \n" \
"p_Output[index + 1] = (p_Input[index + 1] - p_Input[indexE - 3]) * p_Enhance; \n" \
"p_Output[index + 2] = (p_Input[index + 2] - p_Input[indexE - 2]) * p_Enhance; \n" \
"}} \n" \
"kernel void k_customMatrix( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& p_Normalise [[buffer (8)]], constant float& p_Scale [[buffer (11)]], \n" \
"constant float& p_Matrix11 [[buffer (16)]], constant float& p_Matrix12 [[buffer (17)]], constant float& p_Matrix13 [[buffer (18)]],  \n" \
"constant float& p_Matrix21 [[buffer (19)]], constant float& p_Matrix22 [[buffer (20)]], constant float& p_Matrix23 [[buffer (21)]],  \n" \
"constant float& p_Matrix31 [[buffer (22)]], constant float& p_Matrix32 [[buffer (23)]], constant float& p_Matrix33 [[buffer (24)]], \n" \
"uint2 id [[ thread_position_in_grid ]]) { \n" \
"float Scale = p_Scale + 1.0f; \n" \
"float normalise = 1.0f; \n" \
"float Matrix11 = p_Matrix11 * Scale; \n" \
"float Matrix12 = p_Matrix12 * Scale; \n" \
"float Matrix13 = p_Matrix13 * Scale; \n" \
"float Matrix21 = p_Matrix21 * Scale; \n" \
"float Matrix22 = p_Matrix22 * Scale - p_Scale; \n" \
"float Matrix23 = p_Matrix23 * Scale; \n" \
"float Matrix31 = p_Matrix31 * Scale; \n" \
"float Matrix32 = p_Matrix32 * Scale; \n" \
"float Matrix33 = p_Matrix33 * Scale; \n" \
"float total = Matrix11 + Matrix12 + Matrix13 + Matrix21 + Matrix22 + Matrix23 + Matrix31 + Matrix32 + Matrix33; \n" \
"if (p_Normalise == 1 && total > 1.0f) \n" \
"normalise /= total; \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"int start_y = max((int)id.y - 1, 0); \n" \
"int end_y = min(p_Height - 1, (int)id.y + 1); \n" \
"int start_x = max((int)id.x - 1, 0); \n" \
"int end_x = min(p_Width - 1, (int)id.x + 1); \n" \
"for(int c = 0; c < 3; c++) { \n" \
"p_Output[index + c] = (p_Input[(end_y * p_Width + start_x) * 4 + c] * Matrix11 + \n" \
"p_Input[(end_y * p_Width + id.x) * 4 + c] * Matrix12 + \n" \
"p_Input[(end_y * p_Width + end_x) * 4 + c] * Matrix13 + \n" \
"p_Input[(id.y * p_Width + start_x) * 4 + c] * Matrix21 + \n" \
"p_Input[(id.y * p_Width + id.x) * 4 + c] * Matrix22 + \n" \
"p_Input[(id.y * p_Width + end_x) * 4 + c] * Matrix23 + \n" \
"p_Input[(start_y * p_Width + start_x) * 4 + c] * Matrix31 + \n" \
"p_Input[(start_y * p_Width + id.x) * 4 + c] * Matrix32 + \n" \
"p_Input[(start_y * p_Width + end_x) * 4 + c] * Matrix33) * normalise; \n" \
"}}} \n" \
"kernel void k_scatter( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& p_Range [[buffer (6)]], constant float& p_Mix [[buffer (12)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"int rg = p_Range + 1; \n" \
"int totA = round((p_Input[index] + p_Input[index + 1] + p_Input[index + 2]) * 1111) + id.x; \n" \
"int totB = round((p_Input[index] + p_Input[index + 1]) * 1111) + id.y; \n" \
"int polarityA = totA % 2 > 0 ? -1 : 1; \n" \
"int polarityB = totB % 2 > 0 ? -1 : 1; \n" \
"int scatterA = (totA % rg) * polarityA; \n" \
"int scatterB = (totB % rg) * polarityB; \n" \
"int X = (id.x + scatterA) < 0 ? abs(id.x + scatterA) : ((id.x + scatterA) > (p_Width - 1) ? (2 * (p_Width - 1)) - (id.x + scatterA) : (id.x + scatterA)); \n" \
"int Y = (id.y + scatterB) < 0 ? abs(id.y + scatterB) : ((id.y + scatterB) > (p_Height - 1) ? (2 * (p_Height - 1)) - (id.y + scatterB) : (id.y + scatterB)); \n" \
"p_Output[index] = p_Input[((Y * p_Width) + X) * 4 + 0] * (1.0f - p_Mix) + p_Mix * p_Input[index]; \n" \
"p_Output[index + 1] = p_Input[((Y * p_Width) + X) * 4 + 1] * (1.0f - p_Mix) + p_Mix * p_Input[index + 1]; \n" \
"p_Output[index + 2] = p_Input[((Y * p_Width) + X) * 4 + 2] * (1.0f - p_Mix) + p_Mix * p_Input[index + 2]; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Convolve, int p_Display, float* p_Adjust, float* p_Matrix)
{
const char* simple				= "k_simple";
const char* gaussian			= "k_gaussian";
const char* transpose			= "k_transpose";
const char* simpleRecursive		= "k_simpleRecursive";
const char* boxfilter			= "k_boxfilter";
const char* freqSharpen			= "k_freqSharpen";
const char* freqAdd				= "k_freqAdd";
const char* edgeDetect			= "k_edgeDetect";
const char* edgeEnhance			= "k_edgeEnhance";
const char* erode1				= "k_erode1";
const char* erode2				= "k_erode2";
const char* dilate1				= "k_dilate1";
const char* dilate2				= "k_dilate2";
const char* customMatrix		= "k_customMatrix";
const char* scatter				= "k_scatter";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLBuffer>					tempBuffer;
id<MTLComputePipelineState>    	_simple;
id<MTLComputePipelineState>    	_gaussian;
id<MTLComputePipelineState>    	_transpose;
id<MTLComputePipelineState>    	_simpleRecursive;
id<MTLComputePipelineState>    	_boxfilter;
id<MTLComputePipelineState>    	_freqSharpen;
id<MTLComputePipelineState>    	_freqAdd;
id<MTLComputePipelineState>    	_edgeDetect;
id<MTLComputePipelineState>    	_edgeEnhance;
id<MTLComputePipelineState>    	_erode1;
id<MTLComputePipelineState>    	_erode2;
id<MTLComputePipelineState>    	_dilate1;
id<MTLComputePipelineState>    	_dilate2;
id<MTLComputePipelineState>    	_customMatrix;
id<MTLComputePipelineState>    	_scatter;

NSError* err;
std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

const auto it = s_PipelineQueueMap.find(queue);
if (it == s_PipelineQueueMap.end()) {
s_PipelineQueueMap[queue] = pipelineState;
} else {
pipelineState = it->second;
}   

MTLCompileOptions* options	=	[MTLCompileOptions new];
options.fastMathEnabled	=		YES;

if (!(metalLibrary    = [device newLibraryWithSource:@(kernelSource) options:options error:&err])) {
fprintf(stderr, "Failed to load metal library, %s\n", err.localizedDescription.UTF8String);
return;
}
[options release];

tempBuffer 			= 			[device newBufferWithLength:bufferLength options:0];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:simple] ];

_simple				=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:gaussian] ];

_gaussian			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:transpose] ];

_transpose			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:simpleRecursive] ];

_simpleRecursive	=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:boxfilter] ];

_boxfilter			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:freqSharpen] ];

_freqSharpen		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:freqAdd] ];

_freqAdd			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:edgeDetect] ];

_edgeDetect			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:edgeEnhance] ];

_edgeEnhance		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:erode1] ];

_erode1				=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:dilate1] ];

_dilate1			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:erode2] ];

_erode2				=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:dilate2] ];

_dilate2			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:customMatrix] ];

_customMatrix		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:scatter] ];

_scatter			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"RunMetalKernel"];
id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
[computeEncoder setComputePipelineState:_simple];
int exeWidth = [_simple threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

MTLSize gausThreadGroupsA		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsB		= MTLSizeMake((p_Height + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsCount	= MTLSizeMake(exeWidth, 1, 1);

MTLSize transThreadGroupsA		= MTLSizeMake((p_Width + BLOCK_DIM - 1)/BLOCK_DIM, (p_Height + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsB		= MTLSizeMake((p_Height + BLOCK_DIM - 1)/BLOCK_DIM, (p_Width + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsCount	= MTLSizeMake(BLOCK_DIM, BLOCK_DIM, 1);

int Radius = ceil(p_Adjust[0] * 15.0f);
float thresh = p_Adjust[2] * 5.0f;
float fblur = p_Adjust[0] * 10.0f;
float fBlur = p_Adjust[0] * 100.0f;
int iBlur = fBlur;
float sharpen = (2.0f * p_Adjust[1]) + 1.0f;
float dThresh = p_Adjust[2] * 3.0f;
float enhance = p_Adjust[0] * 20.0f;

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&Radius length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:&p_Display length:sizeof(int) atIndex: 8];
[computeEncoder setBytes:&fBlur length:sizeof(float) atIndex: 11];
[computeEncoder setBytes:&p_Matrix[0] length:sizeof(float) atIndex: 16];
[computeEncoder setBytes:&p_Matrix[1] length:sizeof(float) atIndex: 17];
[computeEncoder setBytes:&p_Matrix[2] length:sizeof(float) atIndex: 18];
[computeEncoder setBytes:&p_Matrix[3] length:sizeof(float) atIndex: 19];
[computeEncoder setBytes:&p_Matrix[4] length:sizeof(float) atIndex: 20];
[computeEncoder setBytes:&p_Matrix[5] length:sizeof(float) atIndex: 21];
[computeEncoder setBytes:&p_Matrix[6] length:sizeof(float) atIndex: 22];
[computeEncoder setBytes:&p_Matrix[7] length:sizeof(float) atIndex: 23];
[computeEncoder setBytes:&p_Matrix[8] length:sizeof(float) atIndex: 24];

for(int c = 0; c < 4; c++) {
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

switch (p_Convolve)
{
case 0:
{
if (p_Adjust[0] > 0.0f) {
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
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
}}}
break;

case 1:
{
if (p_Adjust[0] > 0.0f) {
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_simpleRecursive];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_simpleRecursive];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
}}}
break;

case 2:
{
[computeEncoder setBytes:&iBlur length:sizeof(int) atIndex: 6];
if (iBlur > 0) {
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_boxfilter];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_boxfilter];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
}}}
break;

case 3:
{
[computeEncoder setBytes:&thresh length:sizeof(float) atIndex: 11];
[computeEncoder setBytes:&sharpen length:sizeof(float) atIndex: 12];
if (p_Adjust[2] > 0.0f) {
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
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
}}
[computeEncoder setComputePipelineState:_freqSharpen];
for(int c = 0; c < 3; c++) {
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
if (p_Display != 1) {
[computeEncoder setBytes:&fblur length:sizeof(float) atIndex: 11];
if (fblur > 0.0f) {
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
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
}}
[computeEncoder setComputePipelineState:_freqAdd];
for(int c = 0; c < 3; c++) {
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}}
break;

case 4:
{
[computeEncoder setBytes:&dThresh length:sizeof(float) atIndex: 11];
if (p_Adjust[2] > 0.0f) {
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
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
}}
[computeEncoder setComputePipelineState:_edgeDetect];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
break;

case 5:
{
[computeEncoder setBytes:&enhance length:sizeof(float) atIndex: 11];
[computeEncoder setComputePipelineState:_edgeEnhance];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
break;

case 6:
{
if (Radius > 0) {
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_erode1];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder setComputePipelineState:_erode2];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}}
break;

case 7:
{
if (Radius > 0) {
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_dilate1];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder setComputePipelineState:_dilate2];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}}
break;

case 8:
{
[computeEncoder setComputePipelineState:_scatter];
[computeEncoder setBytes:&p_Adjust[1] length:sizeof(float) atIndex: 12];
if (p_Adjust[0] > 0.0f)
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
break;

case 9:
{
[computeEncoder setComputePipelineState:_customMatrix];
[computeEncoder setBytes:&p_Adjust[0] length:sizeof(float) atIndex: 11];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}

[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
}