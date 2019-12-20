#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

#define BLOCK_DIM	32
#define bufferLength	(p_Width * p_Height * sizeof(float))

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"float Luma(float R, float G, float B, int L) { \n" \
"float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f; \n" \
"float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f; \n" \
"float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f; \n" \
"float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f; \n" \
"float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f; \n" \
"float lumaAvg = (R + G + B) / 3.0f; \n" \
"float lumaMax = fmax(fmax(R, G), B); \n" \
"float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 :  \n" \
"L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax; \n" \
"return Lu; \n" \
"} \n" \
"kernel void k_gaussian( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (2)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant float& p_Blur [[buffer (10)]], uint2 threadIdx [[ thread_position_in_threadgroup ]],  \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) { \n" \
"float nsigma = p_Blur < 0.1f ? 0.1f : p_Blur; \n" \
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
"kernel void k_alphaKernel( device const float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant float& lumaLimit [[buffer (9)]], constant int& LumaMath [[buffer (12)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height ) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float luma = Luma(p_Input[index], p_Input[index + 1], p_Input[index + 2], LumaMath); \n" \
"float alpha = lumaLimit > 1.0f ? luma + (1.0f - lumaLimit) * (1.0f - luma) : lumaLimit >= 0.0f ? (luma >= lumaLimit ?  \n" \
"1.0f : luma / lumaLimit) : lumaLimit < -1.0f ? (1.0f - luma) + (lumaLimit + 1.0f) * luma : luma <= (1.0f + lumaLimit) ? 1.0f :  \n" \
"(1.0f - luma) / (1.0f - (lumaLimit + 1.0f)); \n" \
"float Alpha = alpha > 1.0f ? 1.0f : alpha < 0.0f ? 0.0f : alpha; \n" \
"p_Output[index + 3] = Alpha; \n" \
"}} \n" \
"kernel void k_scanKernel( device const float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (3)]],  \n" \
"constant int& p_Height [[buffer (4)]], constant float* balGain [[buffer (5)]], constant float* balOffset [[buffer (6)]], constant float* balLift [[buffer (7)]],  \n" \
"constant float& lumaBalance [[buffer (8)]], constant int* p_Switch [[buffer (11)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height ) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float Alpha = p_Output[index + 3]; \n" \
"float BalR = p_Switch[0] == 1 ? p_Input[index] * balGain[0] : p_Switch[1] == 1 ? p_Input[index] + balOffset[0] : p_Input[index] + (balLift[0] * (1.0f - p_Input[index])); \n" \
"float BalB = p_Switch[0] == 1 ? p_Input[index + 2] * balGain[1] : p_Switch[1] == 1 ? p_Input[index + 2] + balOffset[1] : p_Input[index + 2] + (balLift[1] * (1.0f - p_Input[index + 2])); \n" \
"float Red = p_Switch[2] == 1 ? ( p_Switch[3] == 1 ? BalR * lumaBalance : BalR) : p_Input[index ]; \n" \
"float Green = p_Switch[2] == 1 && p_Switch[3] == 1 ? p_Input[index + 1] * lumaBalance : p_Input[index + 1]; \n" \
"float Blue = p_Switch[2] == 1 ? ( p_Switch[3] == 1 ? BalB * lumaBalance : BalB) : p_Input[index + 2]; \n" \
"p_Output[index] = p_Switch[4] == 1 ? Alpha : Red * Alpha + p_Input[index] * (1.0f - Alpha); \n" \
"p_Output[index + 1] = p_Switch[4] == 1 ? Alpha : Green * Alpha + p_Input[index + 1] * (1.0f - Alpha); \n" \
"p_Output[index + 2] = p_Switch[4] == 1 ? Alpha : Blue * Alpha + p_Input[index + 2] * (1.0f - Alpha); \n" \
"p_Output[index + 3] = 1.0f; \n" \
"}} \n" \
"kernel void k_markerKernel( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (3)]], constant int& p_Height [[buffer (4)]], constant int* LumaMax [[buffer (13)]],  \n" \
"constant int* LumaMin [[buffer (14)]], constant int* Display [[buffer (15)]], constant int& Radius [[buffer (16)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height ) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"bool MIN = Display[1] == 1 && (id.x >= LumaMin[0] - Radius && id.x <= LumaMin[0] + Radius && id.y >= LumaMin[1] - Radius && id.y <= LumaMin[1] + Radius); \n" \
"bool MAX = Display[0] == 1 && (id.x >= LumaMax[0] - Radius && id.x <= LumaMax[0] + Radius && id.y >= LumaMax[1] - Radius && id.y <= LumaMax[1] + Radius); \n" \
"p_Input[index] = MIN ? 0.0f : MAX ? 1.0f : p_Input[index]; \n" \
"p_Input[index + 1] = MIN || MAX ? 1.0f : p_Input[index + 1]; \n" \
"p_Input[index + 2] = MAX ? 0.0f : MIN ? 1.0f : p_Input[index + 2]; \n" \
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Gain, 
float* p_Offset, float* p_Lift, float p_LumaBalance, float p_LumaLimit, float p_Blur, int* p_Switch, 
int p_LumaMath, int* p_LumaMaxXY, int* p_LumaMinXY, int* p_DisplayXY, int Radius)
{
const char* AlphaKernel			= "k_alphaKernel";
const char* ScanKernel			= "k_scanKernel";
const char* MarkerKernel		= "k_markerKernel";
const char* gaussian			= "k_gaussian";
const char* transpose			= "k_transpose";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLBuffer>					tempBuffer;
id<MTLComputePipelineState>     _AlphaKernel;
id<MTLComputePipelineState>     _ScanKernel;
id<MTLComputePipelineState>     _MarkerKernel;
id<MTLComputePipelineState>    	_gaussian;
id<MTLComputePipelineState>    	_transpose;

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

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:AlphaKernel]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_AlphaKernel = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:ScanKernel]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_ScanKernel = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:MarkerKernel]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_MarkerKernel = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
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

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"RunMetalKernel"];

id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

[computeEncoder setComputePipelineState:_AlphaKernel];

int exeWidth = [_AlphaKernel threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

MTLSize gausThreadGroupsA		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsB		= MTLSizeMake((p_Height + exeWidth - 1)/exeWidth, 1, 1);
MTLSize gausThreadGroupsCount	= MTLSizeMake(exeWidth, 1, 1);

MTLSize transThreadGroupsA		= MTLSizeMake((p_Width + BLOCK_DIM - 1)/BLOCK_DIM, (p_Height + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsB		= MTLSizeMake((p_Height + BLOCK_DIM - 1)/BLOCK_DIM, (p_Width + BLOCK_DIM - 1)/BLOCK_DIM, 1);
MTLSize transThreadGroupsCount	= MTLSizeMake(BLOCK_DIM, BLOCK_DIM, 1);

float Blur = p_Blur * 10.0f;

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:p_Gain length:(sizeof(int) * 2) atIndex: 5];
[computeEncoder setBytes:p_Offset length:(sizeof(float) * 2) atIndex: 6];
[computeEncoder setBytes:p_Lift length:(sizeof(float) * 2) atIndex: 7];
[computeEncoder setBytes:&p_LumaBalance length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_LumaLimit length:sizeof(float) atIndex: 9];
[computeEncoder setBytes:&Blur length:sizeof(int) atIndex: 10];
[computeEncoder setBytes:p_Switch length:(sizeof(int) * 5) atIndex: 11];
[computeEncoder setBytes:&p_LumaMath length:sizeof(int) atIndex: 12];
[computeEncoder setBytes:p_LumaMaxXY length:(sizeof(int) * 2) atIndex: 13];
[computeEncoder setBytes:p_LumaMinXY length:(sizeof(int) * 2) atIndex: 14];
[computeEncoder setBytes:p_DisplayXY length:(sizeof(int) * 2) atIndex: 15];
[computeEncoder setBytes:&Radius length:sizeof(int) atIndex: 16];

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
}

[computeEncoder setComputePipelineState:_ScanKernel];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_DisplayXY[0] == 1 || p_DisplayXY[1] == 1) {
[computeEncoder setComputePipelineState:_MarkerKernel];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

[computeEncoder endEncoding];
[commandBuffer commit];
}