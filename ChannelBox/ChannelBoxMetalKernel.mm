#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

#define BLOCK_DIM	32
#define bufferLength	(p_Width * p_Height * sizeof(float))

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"float Luma(float R, float G, float B, int L); \n" \
"float Luma(float R, float G, float B, int L) { \n" \
"float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f; \n" \
"float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f; \n" \
"float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f; \n" \
"float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f; \n" \
"float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f; \n" \
"float lumaAvg = (R + G + B) / 3.0f; \n" \
"float lumaMax = fmax(fmax(R, G), B); \n" \
"float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax; \n" \
"return Lu; \n" \
"} \n" \
"kernel void k_channelBoxKernelA1(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]],  \n" \
"constant int& p_ChannelBox [[buffer (7)]], constant int& p_LumaMath [[buffer (8)]], constant int& p_Preserve [[buffer (9)]], constant float& p_Mask [[buffer (20)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float R = p_Input[index]; \n" \
"float G = p_Input[index + 1]; \n" \
"float B = p_Input[index + 2]; \n" \
"float BR = B > R ? R : B;     \n" \
"float BG = B > G ? G : B;     \n" \
"float BGR = B > fmin(G, R) ? fmin(G, R) : B;     \n" \
"float BGRX = B > fmax(G, R) ? fmax(G, R) : B;     \n" \
"float blue = p_ChannelBox == 0 ? BR : p_ChannelBox == 1 ? BG : p_ChannelBox == 2 ? BGR : p_ChannelBox == 3 ? BGRX : B; 													    \n" \
"float GR = G > R ? R : G;     \n" \
"float GB = G > B ? B : G;     \n" \
"float GBR = G > fmin(B, R) ? fmin(B, R) : G;     \n" \
"float GBRX = G > fmax(B, R) ? fmax(B, R) : G;     \n" \
"float green = p_ChannelBox == 4 ? GR : p_ChannelBox == 5 ? GB : p_ChannelBox == 6 ? GBR : p_ChannelBox == 7 ? GBRX : G; 													    \n" \
"float RG = R > G ? G : R;     \n" \
"float RB = R > B ? B : R;     \n" \
"float RBG = R > fmin(B, G) ? fmin(B, G) : R;     \n" \
"float RBGX = R > fmax(B, G) ? fmax(B, G) : R;     \n" \
"float red = p_ChannelBox == 8 ? RG : p_ChannelBox == 9 ? RB : p_ChannelBox == 10 ? RBG : p_ChannelBox == 11 ? RBGX : R;																								    \n" \
"p_Output[index] = red; \n" \
"p_Output[index + 1] = green; \n" \
"p_Output[index + 2] = blue; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
"}} \n" \
"kernel void k_channelBoxKernelA2(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]],  \n" \
"constant float& p_ChannelSwap0 [[buffer (11)]], constant float& p_ChannelSwap1 [[buffer (12)]], constant float& p_ChannelSwap2 [[buffer (13)]],  \n" \
"constant float& p_ChannelSwap3 [[buffer (14)]], constant float& p_ChannelSwap4 [[buffer (15)]], constant float& p_ChannelSwap5 [[buffer (16)]],  \n" \
"constant float& p_ChannelSwap6 [[buffer (17)]], constant float& p_ChannelSwap7 [[buffer (18)]], constant float& p_ChannelSwap8 [[buffer (19)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float R = p_Input[index]; \n" \
"float G = p_Input[index + 1]; \n" \
"float B = p_Input[index + 2]; \n" \
"float red = R * (1.0f + p_ChannelSwap0 + p_ChannelSwap1 + p_ChannelSwap2) + G * (0.0f - p_ChannelSwap0 - (p_ChannelSwap2 / 2.0f)) + B * (0.0f - p_ChannelSwap1 - (p_ChannelSwap2 / 2.0f)); \n" \
"float green = R * (0.0f - p_ChannelSwap3 - (p_ChannelSwap5 / 2.0f)) + G * (1.0f + p_ChannelSwap3 + p_ChannelSwap4 + p_ChannelSwap5) + B * (0.0f - p_ChannelSwap4 - (p_ChannelSwap5 / 2.0f)); \n" \
"float blue = R * (0.0f - p_ChannelSwap6 - (p_ChannelSwap8 / 2.0f)) + G * (0.0f - p_ChannelSwap7 - (p_ChannelSwap8 / 2.0f)) + B * (1.0f + p_ChannelSwap6 + p_ChannelSwap7 + p_ChannelSwap8); \n" \
"p_Output[index] = red; \n" \
"p_Output[index + 1] = green; \n" \
"p_Output[index + 2] = blue; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
"}} \n" \
"kernel void k_channelBoxKernelA3(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]],  \n" \
"constant int& p_LumaMath [[buffer (8)]], constant int& p_Preserve [[buffer (9)]], constant float& p_Mask [[buffer (20)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float R = p_Input[index]; \n" \
"float G = p_Input[index + 1]; \n" \
"float B = p_Input[index + 2]; \n" \
"float inLuma = Luma(R, G, B, p_LumaMath); \n" \
"float red, green, blue; \n" \
"red = p_Output[index] ; \n" \
"green = p_Output[index + 1] ; \n" \
"blue = p_Output[index + 2] ; \n" \
"float mask = 1.0f; \n" \
"if (p_Preserve == 1) { \n" \
"float outLuma = Luma(red, green, blue, p_LumaMath); \n" \
"red = red * (inLuma / outLuma); \n" \
"green = green * (inLuma / outLuma); \n" \
"blue = blue * (inLuma / outLuma); \n" \
"} \n" \
"if(p_Mask != 0.0f) { \n" \
"mask = p_Mask > 1.0f ? inLuma + (1.0f - p_Mask) * (1.0f - inLuma) : p_Mask >= 0.0f ? (inLuma >= p_Mask ? 1.0f :  \n" \
"inLuma / p_Mask) : p_Mask < -1.0f ? (1.0f - inLuma) + (p_Mask + 1.0f) * inLuma : inLuma <= (1.0f + p_Mask) ? 1.0f :  \n" \
"(1.0f - inLuma) / (1.0f - (p_Mask + 1.0f)); \n" \
"mask = mask > 1.0f ? 1.0f : mask < 0.0f ? 0.0f : mask; \n" \
"}																								    \n" \
"p_Output[index] = red; \n" \
"p_Output[index + 1] = green; \n" \
"p_Output[index + 2] = blue; \n" \
"p_Output[index + 3] = mask; \n" \
"}} \n" \
"kernel void k_channelBoxKernelB(device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant int& p_Display [[buffer (10)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Output[index] = p_Display == 1 ? p_Output[index + 3] : p_Output[index] * p_Output[index + 3] + p_Input[index] * (1.0f - p_Output[index + 3]); \n" \
"p_Output[index + 1] = p_Display == 1 ? p_Output[index + 3] : p_Output[index + 1] * p_Output[index + 3] + p_Input[index + 1] * (1.0f - p_Output[index + 3]); \n" \
"p_Output[index + 2] = p_Display == 1 ? p_Output[index + 3] : p_Output[index + 2] * p_Output[index + 3] + p_Input[index + 2] * (1.0f - p_Output[index + 3]); \n" \
"p_Output[index + 3] = 1.0f; \n" \
"}} \n" \
"kernel void k_garbageCore(device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]],  \n" \
"constant int& ch_out [[buffer (5)]], constant float& p_Garbage [[buffer (22)]], constant float& p_Core [[buffer (23)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float A = p_Input[index + ch_out]; \n" \
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
"p_Input[index + ch_out] = A; \n" \
"}} \n" \
"kernel void k_gaussian( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]],  \n" \
"constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int& ch_in [[buffer (5)]],  \n" \
"constant float& blur [[buffer (21)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
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
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Choice, 
int p_ChannelBox, float* p_ChannelSwap, int p_LumaMath, int* p_Switch, float* p_Mask)
{
int alpha = 3;
p_Mask[1] *= 10.0f;

const char* channelBoxKernelA1	= "k_channelBoxKernelA1";
const char* channelBoxKernelA2	= "k_channelBoxKernelA2";
const char* channelBoxKernelA3	= "k_channelBoxKernelA3";
const char* channelBoxKernelB	= "k_channelBoxKernelB";
const char* garbageCore			= "k_garbageCore";
const char* gaussian			= "k_gaussian";
const char* transpose			= "k_transpose";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLBuffer>					tempBuffer;
id<MTLComputePipelineState>    	_channelBoxKernelA1;
id<MTLComputePipelineState>    	_channelBoxKernelA2;
id<MTLComputePipelineState>    	_channelBoxKernelA3;
id<MTLComputePipelineState>    	_channelBoxKernelB;
id<MTLComputePipelineState>    	_garbageCore;
id<MTLComputePipelineState>    	_gaussian;
id<MTLComputePipelineState>    	_transpose;

NSError* err;

std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

const auto it = s_PipelineQueueMap.find(queue);
if (it == s_PipelineQueueMap.end())
{
s_PipelineQueueMap[queue] = pipelineState;
}
else
{
pipelineState = it->second;
}   

MTLCompileOptions* options	=	[MTLCompileOptions new];
options.fastMathEnabled	=		YES;

if (!(metalLibrary    = [device newLibraryWithSource:@(kernelSource) options:options error:&err]))
{
fprintf(stderr, "Failed to load metal library, %s\n", err.localizedDescription.UTF8String);
return;
}
[options release];

tempBuffer 			= 			[device newBufferWithLength:bufferLength options:0];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:channelBoxKernelA1] ];

_channelBoxKernelA1	=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:channelBoxKernelA2] ];

_channelBoxKernelA2	=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:channelBoxKernelA3] ];

_channelBoxKernelA3	=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:channelBoxKernelB] ];

_channelBoxKernelB	=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:garbageCore] ];

_garbageCore		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:gaussian] ];

_gaussian			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:transpose] ];

_transpose			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"RunMetalKernel"];

id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

[computeEncoder setComputePipelineState:_channelBoxKernelA1];

int exeWidth = [_channelBoxKernelA1 threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

MTLSize gausThreadGroupsA		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsB		= MTLSizeMake((p_Height + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsCount	= MTLSizeMake(exeWidth, 1, 1);

MTLSize transThreadGroupsA		= MTLSizeMake((p_Width + BLOCK_DIM - 1)/BLOCK_DIM, (p_Height + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsB		= MTLSizeMake((p_Height + BLOCK_DIM - 1)/BLOCK_DIM, (p_Width + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsCount	= MTLSizeMake(BLOCK_DIM, BLOCK_DIM, 1);

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Choice length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:&p_ChannelBox length:sizeof(int) atIndex: 7];
[computeEncoder setBytes:&p_LumaMath length:sizeof(int) atIndex: 8];
[computeEncoder setBytes:&p_Switch[0] length:sizeof(int) atIndex: 9];
[computeEncoder setBytes:&p_Switch[1] length:sizeof(int) atIndex: 10];

[computeEncoder setBytes:&p_ChannelSwap[0] length:sizeof(float) atIndex: 11];
[computeEncoder setBytes:&p_ChannelSwap[1] length:sizeof(float) atIndex: 12];
[computeEncoder setBytes:&p_ChannelSwap[2] length:sizeof(float) atIndex: 13];
[computeEncoder setBytes:&p_ChannelSwap[3] length:sizeof(float) atIndex: 14];
[computeEncoder setBytes:&p_ChannelSwap[4] length:sizeof(float) atIndex: 15];
[computeEncoder setBytes:&p_ChannelSwap[5] length:sizeof(float) atIndex: 16];
[computeEncoder setBytes:&p_ChannelSwap[6] length:sizeof(float) atIndex: 17];
[computeEncoder setBytes:&p_ChannelSwap[7] length:sizeof(float) atIndex: 18];
[computeEncoder setBytes:&p_ChannelSwap[8] length:sizeof(float) atIndex: 19];

[computeEncoder setBytes:&p_Mask[0] length:sizeof(float) atIndex: 20];
[computeEncoder setBytes:&p_Mask[1] length:sizeof(float) atIndex: 21];
[computeEncoder setBytes:&p_Mask[2] length:sizeof(float) atIndex: 22];
[computeEncoder setBytes:&p_Mask[3] length:sizeof(float) atIndex: 23];

if (p_Choice == 0)
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
if (p_Choice == 1){
[computeEncoder setComputePipelineState:_channelBoxKernelA2];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
[computeEncoder setComputePipelineState:_channelBoxKernelA3];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_Mask[1] > 0.0f) {
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
if (p_Mask[2] > 0.0f || p_Mask[3] > 0.0f) {
[computeEncoder setComputePipelineState:_garbageCore];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}

[computeEncoder setComputePipelineState:_channelBoxKernelB];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder endEncoding];
[commandBuffer commit];
}