#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

#define BLOCK_DIM	32
#define bufferLength	(p_Width * p_Height * sizeof(float))

const char* kernelSource =  \
"#include <metal_stdlib>\n" \
"using namespace metal; \n" \
"kernel void k_rec709toLAB(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = ((id.y * p_Width) + id.x) * 4; \n" \
"float linR = p_Input[index + 0] < 0.08145f ? (p_Input[index + 0] < 0.0f ? 0.0f : p_Input[index + 0] * (1.0f / 4.5f)) : pow((p_Input[index + 0] + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f)); \n" \
"float linG = p_Input[index + 1] < 0.08145f ? (p_Input[index + 1] < 0.0f ? 0.0f : p_Input[index + 1] * (1.0f / 4.5f)) : pow((p_Input[index + 1] + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f)); \n" \
"float linB = p_Input[index + 2] < 0.08145f ? (p_Input[index + 2] < 0.0f ? 0.0f : p_Input[index + 2] * (1.0f / 4.5f)) : pow((p_Input[index + 2] + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f)); \n" \
"float xyzR = 0.4124564f * linR + 0.3575761f * linG + 0.1804375f * linB; \n" \
"float xyzG = 0.2126729f * linR + 0.7151522f * linG + 0.0721750f * linB; \n" \
"float xyzB = 0.0193339f * linR + 0.1191920f * linG + 0.9503041f * linB; \n" \
"xyzR /= (0.412453f + 0.357580f + 0.180423f); \n" \
"xyzG /= (0.212671f + 0.715160f + 0.072169f); \n" \
"xyzB /= (0.019334f + 0.119193f + 0.950227f); \n" \
"float fx = xyzR >= 0.008856f ? pow(xyzR, 1.0f / 3.0f) : 7.787f * xyzR + 16.0f / 116.0f; \n" \
"float fy = xyzG >= 0.008856f ? pow(xyzG, 1.0f / 3.0f) : 7.787f * xyzG + 16.0f / 116.0f; \n" \
"float fz = xyzB >= 0.008856f ? pow(xyzB, 1.0f / 3.0f) : 7.787f * xyzB + 16.0f / 116.0f;  \n" \
"float L = (116.0f * fy - 16.0f) / 100.0f; \n" \
"p_Output[index + 0] = L; \n" \
"p_Output[index + 1] = (500 * (fx - fy)) / 200.0f + 0.5f; \n" \
"p_Output[index + 2] = (200 * (fy - fz)) / 200.0f + 0.5f; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
"} \n" \
"} \n" \
"kernel void k_LABtoRec709(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = ((id.y * p_Width) + id.x) * 4; \n" \
"float l = p_Input[index + 0] * 100.0f; \n" \
"float a = (p_Input[index + 1] - 0.5f) * 200.0f; \n" \
"float b = (p_Input[index + 2] - 0.5f) * 200.0f; \n" \
"float cy = (l + 16.0f) / 116.0f; \n" \
"float CY = cy >= 0.206893f ? (cy * cy * cy) : (cy - 16.0f / 116) / 7.787f; \n" \
"float y = (0.212671f + 0.715160f + 0.072169f) * CY; \n" \
"float cx = a / 500 + cy; \n" \
"float CX = cx >= 0.206893f ? (cx * cx * cx) : (cx - 16.0f / 116) / 7.787f; \n" \
"float x = (0.412453f + 0.357580f + 0.180423f) * CX; \n" \
"float cz = cy - b / 200; \n" \
"float CZ = cz >= 0.206893f ? (cz * cz * cz) : (cz - 16.0f / 116) / 7.787f; \n" \
"float z = (0.019334f + 0.119193f + 0.950227f) * CZ; \n" \
"float r =  3.2404542f * x + -1.5371385f * y + -0.4985314f * z; \n" \
"float g = -0.9692660f * x +  1.8760108f * y +  0.0415560f * z; \n" \
"float _b =  0.0556434f * x + -0.2040259f * y +  1.0572252f * z; \n" \
"float R = r < 0.0181f ? (r < 0.0f ? 0.0f : r * 4.5f) : 1.0993f * pow(r, 0.45f) - (1.0993f - 1.0f); \n" \
"float G = g < 0.0181f ? (g < 0.0f ? 0.0f : g * 4.5f) : 1.0993f * pow(g, 0.45f) - (1.0993f - 1.0f); \n" \
"float B = _b < 0.0181f ? (_b < 0.0f ? 0.0f : _b * 4.5f) : 1.0993f * pow(_b, 0.45f) - (1.0993f - 1.0f); \n" \
"p_Output[index + 0] = R; \n" \
"p_Output[index + 1] = G; \n" \
"p_Output[index + 2] = B; \n" \
"} \n" \
"} \n" \
"kernel void k_gaussian(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]],  \n" \
"constant float& blur [[buffer (7)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) \n" \
"{ \n" \
"const float nsigma = blur < 0.1f ? 0.1f : blur, \n" \
"alpha = 1.695f / nsigma, \n" \
"ema = (float)exp(-alpha); \n" \
"float ema2 = (float)exp(-2.0 * alpha), \n" \
"b1 = -2.0 * ema, \n" \
"b2 = ema2; \n" \
"float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, coefp = 0.0f, coefn = 0.0f; \n" \
"const float k = (1.0f - ema) * (1.0f - ema) / (1.0f + 2.0f * alpha * ema - ema2); \n" \
"a0 = k; \n" \
"a1 = k * (alpha - 1.0f) * ema; \n" \
"a2 = k * (alpha + 1.0f) * ema; \n" \
"a3 = -k * ema2; \n" \
"coefp = (a0 + a1) / (1.0f + b1 + b2); \n" \
"coefn = (a2 + a3)/(1.0f + b1 + b2); \n" \
"int x = (blockIdx.x * blockDim.x + threadIdx.x); \n" \
"if (x >= p_Width) return; \n" \
"p_Input += x * 4 + ch_in; \n" \
"p_Output += x; \n" \
"float xp, yp, yb; \n" \
"xp = *p_Input; \n" \
"yb = coefp*xp; \n" \
"yp = yb; \n" \
"for (int y = 0; y < p_Height; y++) \n" \
"{ \n" \
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
"for (int y = p_Height - 1; y >= 0; y--) \n" \
"{ \n" \
"float xc = *p_Input; \n" \
"float yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya; \n" \
"xa = xn; \n" \
"xn = xc; \n" \
"ya = yn; \n" \
"yn = yc; \n" \
"*p_Output = *p_Output + yc; \n" \
"p_Input -= p_Width * 4; \n" \
"p_Output -= p_Width; \n" \
"} \n" \
"} \n" \
"kernel void k_simple(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]],  \n" \
"constant int& ch_out [[buffer (6)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{                                 				    \n" \
"if ((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = ((id.y * p_Width) + id.x) * 4; \n" \
"p_Output[index + ch_out] = p_Input[index + ch_in];  \n" \
"} \n" \
"} \n" \
"kernel void k_simpleALPHA(device float* p_Input [[buffer (0)]], constant int& p_Width [[buffer (3)]], \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], \n" \
"constant float& reset [[buffer (7)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{	\n" \
"if ((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = ((id.y * p_Width) + id.x) * 4; \n" \
"p_Input[index + ch_in] = reset;  \n" \
"} \n" \
"} \n" \
"kernel void k_transpose(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& ch_out [[buffer (6)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) \n" \
"{ \n" \
"const int BLOCK_D = 32; \n" \
"threadgroup float sblock[BLOCK_D * (BLOCK_D + 1)]; \n" \
"int xIndex = blockIdx.x * BLOCK_D + threadIdx.x; \n" \
"int yIndex = blockIdx.y * BLOCK_D + threadIdx.y; \n" \
"if((xIndex < p_Width) && (yIndex < p_Height)) \n" \
"{ \n" \
"sblock[threadIdx.y * (BLOCK_D + 1) + threadIdx.x] = p_Input[(yIndex * p_Width + xIndex)]; \n" \
"} \n" \
"threadgroup_barrier(mem_flags::mem_threadgroup); \n" \
"xIndex = blockIdx.y * BLOCK_D + threadIdx.x; \n" \
"yIndex = blockIdx.x * BLOCK_D + threadIdx.y; \n" \
"if((xIndex < p_Height) && (yIndex < p_Width)) \n" \
"{ \n" \
"p_Output[(yIndex * p_Height + xIndex) * 4 + ch_out] = sblock[threadIdx.x * (BLOCK_D + 1) + threadIdx.y]; \n" \
"} \n" \
"} \n" \
"kernel void k_displayThreshold(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], \n" \
"constant int& Grey [[buffer (10)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = ((id.y * p_Width) + id.x) * 4; \n" \
"float freq = p_Input[index + ch_in]; \n" \
"float bkground = Grey == 1 ? 0.5f : 0.0f; \n" \
"p_Output[index + 0] = freq + bkground; \n" \
"p_Output[index + 1] = freq + bkground; \n" \
"p_Output[index + 2] = freq + bkground; \n" \
"p_Output[index + 3] = 1.0f; \n" \
"} \n" \
"} \n" \
"kernel void k_freqSharpen(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], device float* p_HighFreq [[buffer (2)]], \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]], constant int& ch_out [[buffer (6)]], \n" \
"constant float& p_Sharpen [[buffer (8)]], constant int& p_Switch [[buffer (9)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = ((id.y * p_Width) + id.x) * 4; \n" \
"if(p_Switch > 0 && p_Switch != 7){																								   \n" \
"p_HighFreq[index] = (p_Input[index + ch_in] - p_Output[index + ch_out]) * p_Sharpen; \n" \
"} else { \n" \
"p_HighFreq[index] += (p_Input[index + ch_in] - p_Output[index + ch_out]) * p_Sharpen; \n" \
"} \n" \
"} \n" \
"} \n" \
"kernel void k_freqAdd(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]],  \n" \
"constant float& blend [[buffer (8)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = ((id.y * p_Width) + id.x) * 4; \n" \
"float combine = p_Output[index + 2] + p_Output[index + 0];												   \n" \
"p_Output[index + 0] = combine * (1.0f - blend) + p_Input[index + 0] * blend; \n" \
"p_Output[index + 1] = p_Input[index + 1]; \n" \
"p_Output[index + 2] = p_Input[index + 2]; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
"} \n" \
"} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_EQ, int p_Switch, int p_Grey)
{
int red = 0;
int green = 1;
int blue = 2;
int alpha = 3;
float zero = 0.0f;

const char* gaussian			= "k_gaussian";
const char* simple				= "k_simple";
const char* simpleALPHA			= "k_simpleALPHA";
const char* transpose			= "k_transpose";
const char* rec709toLAB			= "k_rec709toLAB";
const char* LABtoRec709			= "k_LABtoRec709";
const char* displayThreshold	= "k_displayThreshold";
const char* freqSharpen			= "k_freqSharpen";
const char* freqAdd				= "k_freqAdd";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLBuffer>					tempBuffer;

id<MTLComputePipelineState>    _gaussian;
id<MTLComputePipelineState>    _simple;
id<MTLComputePipelineState>    _simpleALPHA;
id<MTLComputePipelineState>    _transpose;
id<MTLComputePipelineState>    _rec709toLAB;
id<MTLComputePipelineState>    _LABtoRec709;
id<MTLComputePipelineState>    _displayThreshold;
id<MTLComputePipelineState>    _freqSharpen;
id<MTLComputePipelineState>    _freqAdd;

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
metalLibrary		=			[device newLibraryWithSource:@(kernelSource) options:options error:&err];
[options release];

tempBuffer 			= 			[device newBufferWithLength:bufferLength options:0];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:gaussian] ];

_gaussian			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:simpleALPHA] ];

_simpleALPHA		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:simple] ];

_simple				=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:transpose] ];

_transpose			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:rec709toLAB] ];

_rec709toLAB		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:LABtoRec709] ];

_LABtoRec709		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:displayThreshold] ];

_displayThreshold	=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:freqSharpen] ];

_freqSharpen		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:freqAdd] ];

_freqAdd			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"RunMetalKernel"];

id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

int exeWidth = [_rec709toLAB threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

MTLSize gausThreadGroupsA		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsB		= MTLSizeMake((p_Height + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsCount	= MTLSizeMake(exeWidth, 1, 1);

MTLSize transThreadGroupsA		= MTLSizeMake((p_Width + BLOCK_DIM - 1)/BLOCK_DIM, (p_Height + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsB		= MTLSizeMake((p_Height + BLOCK_DIM - 1)/BLOCK_DIM, (p_Width + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsCount	= MTLSizeMake(BLOCK_DIM, BLOCK_DIM, 1);

float Res_Scale = (float)p_Width / 1920.0f;
float p_Blur[6];
p_Blur[0] = (0.4f * Res_Scale) * p_EQ[6]; 
for(int c = 0; c < 5; c++) {
p_Blur[c + 1] = p_Blur[c] * 2.0f; 
}

if (p_Switch == 0 && p_EQ[0] == 1.0f && p_EQ[1] == 1.0f && p_EQ[2] == 1.0f && p_EQ[3] == 1.0f && p_EQ[4] == 1.0f && p_EQ[5] == 1.0f) {
[computeEncoder setComputePipelineState:_simple];
[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
for (int c = 0; c < 4; c++) {
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
return;
}

[computeEncoder setComputePipelineState:_rec709toLAB];
[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_simpleALPHA];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBytes:&zero length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Blur[0] length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];    
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_freqSharpen];
[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:&p_EQ[0] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Switch length:sizeof(int) atIndex: 9];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_simple];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if(p_Switch == 1){
[computeEncoder setComputePipelineState:_displayThreshold];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Grey length:sizeof(int) atIndex: 10];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
return;
}

if (p_EQ[1] != 1.0f || p_EQ[2] != 1.0f || p_EQ[3] != 1.0f || p_EQ[4] != 1.0f || p_EQ[5] != 1.0f || p_Switch > 0) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Blur[1] length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];    
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_freqSharpen];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:&p_EQ[1] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Switch length:sizeof(int) atIndex: 9];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_simple];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_Switch == 2) {
[computeEncoder setComputePipelineState:_displayThreshold];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Grey length:sizeof(int) atIndex: 10];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
return;
}

if (p_EQ[2] != 1.0f || p_EQ[3] != 1.0f || p_EQ[4] != 1.0f || p_EQ[5] != 1.0f || p_Switch > 0) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Blur[2] length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];    
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_freqSharpen];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:&p_EQ[2] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Switch length:sizeof(int) atIndex: 9];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_simple];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_Switch == 3) {
[computeEncoder setComputePipelineState:_displayThreshold];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Grey length:sizeof(int) atIndex: 10];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
return;
}

if (p_EQ[3] != 1.0f || p_EQ[4] != 1.0f || p_EQ[5] != 1.0f || p_Switch > 0) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Blur[3] length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];    
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_freqSharpen];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:&p_EQ[3] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Switch length:sizeof(int) atIndex: 9];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_simple];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_Switch == 4) {
[computeEncoder setComputePipelineState:_displayThreshold];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Grey length:sizeof(int) atIndex: 10];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
return;
}

if (p_EQ[4] != 1.0f || p_EQ[5] != 1.0f || p_Switch > 0) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Blur[4] length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];    
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_freqSharpen];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:&p_EQ[4] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Switch length:sizeof(int) atIndex: 9];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_simple];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_Switch == 5) {
[computeEncoder setComputePipelineState:_displayThreshold];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Grey length:sizeof(int) atIndex: 10];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
return;
}

if (p_EQ[5] != 1.0f || p_Switch > 0) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Blur[5] length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];

[computeEncoder setComputePipelineState:_transpose];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];    
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];

[computeEncoder setComputePipelineState:_freqSharpen];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:&p_EQ[5] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Switch length:sizeof(int) atIndex: 9];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_simple];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_Switch == 6) {
[computeEncoder setComputePipelineState:_displayThreshold];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Grey length:sizeof(int) atIndex: 10];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
return;
}

if (p_Switch == 7) {
[computeEncoder setComputePipelineState:_displayThreshold];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Grey length:sizeof(int) atIndex: 10];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
return;
}

[computeEncoder setComputePipelineState:_freqAdd];
[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_EQ[7] length:sizeof(float) atIndex: 8];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setComputePipelineState:_LABtoRec709];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
}