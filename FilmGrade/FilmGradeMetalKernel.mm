#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"constant float eu = 2.718281828459045f; \n" \
"constant float pie = 3.141592653589793f; \n" \
"kernel void k_Prepare( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant int& p_Display [[buffer (17)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"float ramp = (float)id.x / (float)(p_Width - 1); \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Output[index] = p_Display == 1 ? ramp : p_Input[index]; \n" \
"p_Output[index + 1] = p_Display == 1 ? ramp : p_Input[index + 1]; \n" \
"p_Output[index + 2] = p_Display == 1 ? ramp : p_Input[index + 2]; \n" \
"p_Output[index + 3] = 1.0f; \n" \
"if (p_Display == 2) { \n" \
"p_Input[index] = ramp; \n" \
"p_Input[index + 1] = ramp; \n" \
"p_Input[index + 2] = ramp; \n" \
"}}} \n" \
"kernel void k_FilmGradeKernelA( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant float& p_Exp [[buffer (4)]], constant int& ch_in [[buffer (19)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if(id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Input[index + ch_in] = p_Input[index + ch_in] + p_Exp * 0.01f; \n" \
"}} \n" \
"kernel void k_FilmGradeKernelB( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant float& p_Shad [[buffer (5)]], constant float& p_Mid [[buffer (6)]], constant float& p_High [[buffer (7)]], constant float& p_ShadP [[buffer (8)]],  \n" \
"constant float& p_HighP [[buffer (9)]], constant float& p_ContP [[buffer (10)]], constant int& ch_in [[buffer (19)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float exp = p_Input[index + ch_in]; \n" \
"float expr1 = (p_ShadP / 2.0f) - (1.0f - p_HighP) / 4.0f; \n" \
"float expr2 = (1.0f - (1.0f - p_HighP) / 2.0f) + (p_ShadP / 4.0f); \n" \
"float expr3 = (exp - expr1) / (expr2 - expr1); \n" \
"float expr4 =  p_ContP < 0.5f ? 0.5f - (0.5f - p_ContP) / 2.0f : 0.5f + (p_ContP - 0.5f) / 2.0f; \n" \
"float expr5 = expr3 > expr4 ? (expr3 - expr4) / (2.0f - 2.0f * expr4) + 0.5f : expr3 / (2.0f * expr4); \n" \
"float expr6 = (((sin(2.0f * pie * (expr5 -1.0f / 4.0f)) + 1.0f) / 20.0f) * p_Mid * 4.0f) + expr3; \n" \
"float mid = exp >= expr1 && exp <= expr2 ? expr6 * (expr2 - expr1) + expr1 : exp; \n" \
"float shadup1 = mid > 0.0f ? 2.0f * (mid / p_ShadP) - log((mid / p_ShadP) * (eu * p_Shad * 2.0f) + 1.0f) / log(eu * p_Shad * 2.0f + 1.0f) : mid; \n" \
"float shadup = mid < p_ShadP && p_Shad > 0.0f ? (shadup1 + p_Shad * (1.0f - shadup1)) * p_ShadP : mid; \n" \
"float shaddown1 = shadup / p_ShadP + p_Shad * 2.0f * (1.0f - shadup / p_ShadP); \n" \
"float shaddown = shadup < p_ShadP && p_Shad < 0.0f ? (shaddown1 >= 0.0f ? log(shaddown1 * (eu * p_Shad * -2.0f) + 1.0f) / log(eu * p_Shad * -2.0f + 1.0f) : shaddown1) * p_ShadP : shadup; \n" \
"float highup1 = ((shaddown - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_High * 2.0f)); \n" \
"float highup = shaddown > p_HighP && p_HighP < 1.0f && p_High > 0.0f ? (2.0f * highup1 - log(highup1 * eu * p_High + 1.0f) / log(eu * p_High + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddown; \n" \
"float highdown1 = (highup - p_HighP) / (1.0f - p_HighP); \n" \
"float highdown = highup > p_HighP && p_HighP < 1.0f && p_High < 0.0f ? log(highdown1 * (eu * p_High * -2.0f) + 1.0f) / log(eu * p_High * -2.0f + 1.0f) * (1.0f + p_High) * (1.0f - p_HighP) + p_HighP : highup; \n" \
"p_Input[index + ch_in] = highdown; \n" \
"}} \n" \
"kernel void k_FilmGradeKernelC( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant float& p_ContR [[buffer (11)]], constant float& p_ContG [[buffer (12)]], constant float& p_ContB [[buffer (13)]], constant float& p_SatR [[buffer (14)]],  \n" \
"constant float& p_SatG [[buffer (15)]], constant float& p_SatB [[buffer (16)]], constant float& p_ContP [[buffer (10)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float contR = (p_Input[index] - p_ContP) * p_ContR + p_ContP; \n" \
"float contG = (p_Input[index + 1] - p_ContP) * p_ContG + p_ContP; \n" \
"float contB = (p_Input[index + 2] - p_ContP) * p_ContB + p_ContP; \n" \
"float luma = contR * 0.2126f + contG * 0.7152f + contB * 0.0722f; \n" \
"float outR = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contR * p_SatR; \n" \
"float outG = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contG * p_SatG; \n" \
"float outB = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contB * p_SatB; \n" \
"p_Input[index] = outR; \n" \
"p_Input[index + 1] = outG; \n" \
"p_Input[index + 2] = outB; \n" \
"}} \n" \
"kernel void k_FilmGradeKernelD( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant float& p_Pivot [[buffer (8)]], constant int& p_Display [[buffer (17)]],  \n" \
"constant int& ch_in [[buffer (19)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"float height = p_Height; \n" \
"float width = p_Width; \n" \
"float X = id.x; \n" \
"float Y = id.y; \n" \
"const float RES = width / 1920.0f; \n" \
"float overlay = 0.0f; \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"if (p_Display == 1) { \n" \
"overlay = Y / height >= p_Pivot && Y / height <= p_Pivot + 0.005f * RES ? (fmod(X, 2.0f) != 0.0f ? 1.0f : 0.0f) : \n" \
"p_Output[index + ch_in] >= (Y - 5.0f * RES) / height && p_Output[index + ch_in] <= (Y + 5.0f * RES) / height ? 1.0f : 0.0f; \n" \
"p_Output[index + ch_in] = overlay; \n" \
"} \n" \
"if (p_Display == 2) { \n" \
"overlay = Y / height >= p_Pivot && Y / height <= p_Pivot + 0.005f ? (fmod(X, 2.0f) != 0.0f ? 1.0f : 0.0f) : p_Input[index + ch_in] >= (Y - 5.0f) / height && p_Input[index + ch_in] <= (Y + 5.0f) / height ? 1.0f : 0.0f; \n" \
"p_Output[index + ch_in] = overlay == 0.0f ? p_Output[index + ch_in] : overlay; \n" \
"}}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Exp, 
float* p_Cont, float* p_Sat, float* p_Shad, float* p_Mid, float* p_High, float* p_Pivot, int p_Display)
{
const char* Prepare				= "k_Prepare";
const char* FilmGradeKernelA	= "k_FilmGradeKernelA";
const char* FilmGradeKernelB	= "k_FilmGradeKernelB";
const char* FilmGradeKernelC	= "k_FilmGradeKernelC";
const char* FilmGradeKernelD	= "k_FilmGradeKernelD";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLComputePipelineState>     _Prepare;
id<MTLComputePipelineState>     _FilmGradeKernelA;
id<MTLComputePipelineState>     _FilmGradeKernelB;
id<MTLComputePipelineState>     _FilmGradeKernelC;
id<MTLComputePipelineState>     _FilmGradeKernelD;

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

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:Prepare] ];

_Prepare					=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:FilmGradeKernelA] ];

_FilmGradeKernelA			=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:FilmGradeKernelB] ];

_FilmGradeKernelB			=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:FilmGradeKernelC] ];

_FilmGradeKernelC			=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:FilmGradeKernelD] ];

_FilmGradeKernelD			=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"RunMetalKernel"];

id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
[computeEncoder setComputePipelineState:_Prepare];

int exeWidth = [_Prepare threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Pivot[0] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Pivot[1] length:sizeof(float) atIndex: 9];
[computeEncoder setBytes:&p_Pivot[2] length:sizeof(float) atIndex: 10];
[computeEncoder setBytes:&p_Cont[0] length:sizeof(float) atIndex: 11];
[computeEncoder setBytes:&p_Cont[1] length:sizeof(float) atIndex: 12];
[computeEncoder setBytes:&p_Cont[2] length:sizeof(float) atIndex: 13];
[computeEncoder setBytes:&p_Sat[0] length:sizeof(float) atIndex: 14];
[computeEncoder setBytes:&p_Sat[1] length:sizeof(float) atIndex: 15];
[computeEncoder setBytes:&p_Sat[2] length:sizeof(float) atIndex: 16];
[computeEncoder setBytes:&p_Display length:sizeof(int) atIndex: 17];

[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_FilmGradeKernelA];
[computeEncoder setBytes:&p_Exp[c] length:sizeof(float) atIndex: 4];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 19];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder setComputePipelineState:_FilmGradeKernelB];
[computeEncoder setBytes:&p_Shad[c] length:sizeof(float) atIndex: 5];
[computeEncoder setBytes:&p_Mid[c] length:sizeof(float) atIndex: 6];
[computeEncoder setBytes:&p_High[c] length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
[computeEncoder setComputePipelineState:_FilmGradeKernelC];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_Display > 0) {
if(p_Display == 2) {
[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 1];
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_FilmGradeKernelA];
[computeEncoder setBytes:&p_Exp[c] length:sizeof(float) atIndex: 4];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 19];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder setComputePipelineState:_FilmGradeKernelB];
[computeEncoder setBytes:&p_Shad[c] length:sizeof(float) atIndex: 5];
[computeEncoder setBytes:&p_Mid[c] length:sizeof(float) atIndex: 6];
[computeEncoder setBytes:&p_High[c] length:sizeof(float) atIndex: 7];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
[computeEncoder setComputePipelineState:_FilmGradeKernelC];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
}
for(int c = 0; c < 3; c++) {
[computeEncoder setComputePipelineState:_FilmGradeKernelD];
[computeEncoder setBytes:&p_Pivot[c] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 19];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}

[computeEncoder endEncoding];
[commandBuffer commit];
}