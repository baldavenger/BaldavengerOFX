#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"float Luma( float R, float G, float B, int L) { \n" \
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
"float Sat( float r, float g, float b) { \n" \
"float min = fmin(fmin(r, g), b); \n" \
"float max = fmax(fmax(r, g), b); \n" \
"float delta = max - min; \n" \
"float S = max != 0.0f ? delta / max : 0.0f; \n" \
"return S; \n" \
"} \n" \
"kernel void k_matrix( device const float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant float& p_MatrixRR [[buffer (4)]], constant float& p_MatrixRG [[buffer (5)]],  \n" \
"constant float& p_MatrixRB [[buffer (6)]], constant float& p_MatrixGR [[buffer (7)]], constant float& p_MatrixGG [[buffer (8)]],  \n" \
"constant float& p_MatrixGB [[buffer (9)]], constant float& p_MatrixBR [[buffer (10)]], constant float& p_MatrixBG [[buffer (11)]],  \n" \
"constant float& p_MatrixBB [[buffer (12)]], constant int& p_Luma [[buffer (13)]], constant int& p_Sat [[buffer (14)]],  \n" \
"constant int& p_LumaMath [[buffer (15)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float red = p_Input[index] * p_MatrixRR + p_Input[index + 1] * p_MatrixRG + p_Input[index + 2] * p_MatrixRB; \n" \
"float green = p_Input[index] * p_MatrixGR + p_Input[index + 1] * p_MatrixGG + p_Input[index + 2] * p_MatrixGB; \n" \
"float blue = p_Input[index] * p_MatrixBR + p_Input[index + 1] * p_MatrixBG + p_Input[index + 2] * p_MatrixBB; \n" \
"if (p_Luma == 1) { \n" \
"float inLuma = Luma(p_Input[index], p_Input[index + 1], p_Input[index + 2], p_LumaMath); \n" \
"float outLuma = Luma(red, green, blue, p_LumaMath); \n" \
"red = red * (inLuma / outLuma); \n" \
"green = green * (inLuma / outLuma); \n" \
"blue = blue * (inLuma / outLuma); \n" \
"} \n" \
"if (p_Sat == 1) { \n" \
"float inSat = Sat(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float outSat = Sat(red, green, blue); \n" \
"float satgap = inSat / outSat; \n" \
"float sLuma = Luma(red, green, blue, p_LumaMath); \n" \
"float sr = (1.0f - satgap) * sLuma + red * satgap; \n" \
"float sg = (1.0f - satgap) * sLuma + green * satgap; \n" \
"float sb = (1.0f - satgap) * sLuma + blue * satgap; \n" \
"red = inSat == 0.0f ? sLuma : sr; \n" \
"green = inSat == 0.0f ? sLuma : sg; \n" \
"blue = inSat == 0.0f ? sLuma : sb; \n" \
"} \n" \
"p_Output[index] = red; \n" \
"p_Output[index + 1] = green; \n" \
"p_Output[index + 2] = blue; \n" \
"p_Output[index + 3] = p_Input[index + 3];  \n" \
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, 
int p_Height, float* p_Matrix, int p_Luma, int p_Sat, int p_LumaMath)
{
const char* Matrix				= "k_matrix";
id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLComputePipelineState>     _Matrix;

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

if (!(metalLibrary    = [device newLibraryWithSource:@(kernelSource) options:options error:&err])) {
fprintf(stderr, "Failed to load metal library, %s\n", err.localizedDescription.UTF8String);
return;
}
[options release];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:Matrix] ];

_Matrix						=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"k_matrix"];
id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
[computeEncoder setComputePipelineState:_Matrix];
int exeWidth = [_Matrix threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

[computeEncoder setBuffer:srcDeviceBuf 	offset: 0 				atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf 	offset: 0 				atIndex: 1];
[computeEncoder setBytes:&p_Width 		length:sizeof(int) 		atIndex: 2];
[computeEncoder setBytes:&p_Height 		length:sizeof(int) 		atIndex: 3];
[computeEncoder setBytes:&p_Matrix[0] 	length:sizeof(float) 	atIndex: 4];
[computeEncoder setBytes:&p_Matrix[1] 	length:sizeof(float) 	atIndex: 5];
[computeEncoder setBytes:&p_Matrix[2] 	length:sizeof(float) 	atIndex: 6];
[computeEncoder setBytes:&p_Matrix[3] 	length:sizeof(float) 	atIndex: 7];
[computeEncoder setBytes:&p_Matrix[4] 	length:sizeof(float) 	atIndex: 8];
[computeEncoder setBytes:&p_Matrix[5] 	length:sizeof(float) 	atIndex: 9];
[computeEncoder setBytes:&p_Matrix[6] 	length:sizeof(float) 	atIndex: 10];
[computeEncoder setBytes:&p_Matrix[7] 	length:sizeof(float) 	atIndex: 11];
[computeEncoder setBytes:&p_Matrix[8] 	length:sizeof(float) 	atIndex: 12];
[computeEncoder setBytes:&p_Luma 		length:sizeof(int) 		atIndex: 13];
[computeEncoder setBytes:&p_Sat 		length:sizeof(int) 		atIndex: 14];
[computeEncoder setBytes:&p_LumaMath 	length:sizeof(int) 		atIndex: 15];

[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder endEncoding];
[commandBuffer commit];
}