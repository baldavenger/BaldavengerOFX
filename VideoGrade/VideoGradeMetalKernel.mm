#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"float3 make_float3( float A, float B, float C) { \n" \
"float3 out; \n" \
"out.x = A; out.y = B; out.z = C; \n" \
"return out; \n" \
"} \n" \
"float Luma( float R, float G, float B, int L) { \n" \
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
"float3 RGB_to_HSV( float3 RGB) { \n" \
"float min = fmin(fmin(RGB.x, RGB.y), RGB.z); \n" \
"float max = fmax(fmax(RGB.x, RGB.y), RGB.z); \n" \
"float3 HSV; \n" \
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
"if (HSV.x < 0.0f) \n" \
"HSV.x += 1.0f; \n" \
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
"float3 Temperature( float3 RGB, float Temp){ \n" \
"float r, g, b; \n" \
"if (Temp <= 66.0f) { \n" \
"r = 255.0f; \n" \
"} else { \n" \
"r = Temp - 60.0f; \n" \
"r = 329.698727446f * pow(r, -0.1332047592f); \n" \
"if(r < 0.0f) r = 0.0f; \n" \
"if(r > 255.0f) r = 255.0f; \n" \
"} \n" \
"if (Temp <= 66.0f) { \n" \
"g = Temp; \n" \
"g = 99.4708025861f * log(g) - 161.1195681661f; \n" \
"if (g < 0.0f) g = 0.0f; \n" \
"if (g > 255.0f) g = 255.0f; \n" \
"} else { \n" \
"g = Temp - 60.0f; \n" \
"g = 288.1221695283f * pow(g, -0.0755148492f); \n" \
"if (g < 0.0f) g = 0.0f; \n" \
"if (g > 255.0f) g = 255.0f; \n" \
"} \n" \
"if (Temp >= 66.0f) { \n" \
"b = 255.0f; \n" \
"} else { \n" \
"if (Temp <= 19.0f) { \n" \
"b = 0.0f; \n" \
"} else { \n" \
"b = Temp - 10.0f; \n" \
"b = 138.5177312231f * log(b) - 305.0447927307f; \n" \
"if (b < 0.0f) b = 0.0f; \n" \
"if (b > 255.0f) b = 255.0f; \n" \
"}} \n" \
"RGB.x *= r / 255.0f; \n" \
"RGB.y *= g / 255.0f; \n" \
"RGB.z *= b / 255.0f; \n" \
"return RGB; \n" \
"} \n" \
"device float3 VideoGradeKernelA( float3 RGB, int p_LumaMath, constant int* p_Switch, constant float* p_Scales) { \n" \
"float Temp1 = (p_Scales[1] / 100.0f) + 1.0f; \n" \
"if (p_Scales[0] != 0.0f) { \n" \
"RGB.x *= exp(p_Scales[0]); \n" \
"RGB.y *= exp(p_Scales[0]); \n" \
"RGB.z *= exp(p_Scales[0]); \n" \
"} \n" \
"if (Temp1 != 66.0f) { \n" \
"float3 RGB1; \n" \
"float templuma1 = Luma(RGB.x, RGB.y, RGB.z, p_LumaMath); \n" \
"RGB1 = Temperature(RGB, Temp1); \n" \
"float templuma2 = Luma(RGB1.x, RGB1.y, RGB1.z, p_LumaMath); \n" \
"RGB.x = RGB1.x / (templuma2 / templuma1); \n" \
"RGB.y = RGB1.y / (templuma2 / templuma1); \n" \
"RGB.z = RGB1.z / (templuma2 / templuma1); \n" \
"} \n" \
"if (p_Scales[2] != 0.0f) { \n" \
"float tintluma1 = Luma(RGB.x, RGB.y, RGB.z, p_LumaMath); \n" \
"float R1 = RGB.x * (1.0f + p_Scales[2] / 2.0f); \n" \
"float B1 = RGB.z * (1.0f + p_Scales[2] / 2.0f); \n" \
"float tintluma2 = Luma(R1, RGB.y, B1, p_LumaMath); \n" \
"float tintluma3 = tintluma2 / tintluma1; \n" \
"RGB.x = R1 / tintluma3; \n" \
"RGB.y /= tintluma3; \n" \
"RGB.z = B1 / tintluma3; \n" \
"} \n" \
"if (p_Scales[3] != 0.0f || p_Scales[4] != 0.0f) { \n" \
"float p_Hue = p_Scales[3] / 360.0f; \n" \
"RGB = RGB_to_HSV(RGB); \n" \
"float h2 = RGB.x + p_Hue; \n" \
"float H2 = h2 < 0.0f ? h2 + 1.0f : h2 >= 1.0f ? h2 - 1.0f : h2; \n" \
"float S = RGB.y * (1.0f + p_Scales[4]); \n" \
"RGB = HSV_to_RGB(make_float3(H2, S, RGB.z)); \n" \
"} \n" \
"if (p_Scales[5] != 1.0f) \n" \
"RGB.x = RGB.x >= p_Scales[8] ? (RGB.x - p_Scales[8]) * p_Scales[5]  + p_Scales[8]: RGB.x; \n" \
"if (p_Scales[6] != 1.0f) \n" \
"RGB.y = RGB.y >= p_Scales[8] ? (RGB.y - p_Scales[8]) * p_Scales[6]  + p_Scales[8]: RGB.y; \n" \
"if (p_Scales[7] != 1.0f) \n" \
"RGB.z = RGB.z >= p_Scales[8] ? (RGB.z - p_Scales[8]) * p_Scales[7]  + p_Scales[8]: RGB.z; \n" \
"if (p_Scales[9] != 0.0f) \n" \
"RGB.x = RGB.x <= p_Scales[12] ? ((RGB.x / p_Scales[12]) + p_Scales[9] * (1.0f - (RGB.x / p_Scales[12]))) * p_Scales[12] : RGB.x; \n" \
"if (p_Scales[10] != 0.0f) \n" \
"RGB.y = RGB.y <= p_Scales[12] ? ((RGB.y / p_Scales[12]) + p_Scales[10] * (1.0f - (RGB.y / p_Scales[12]))) * p_Scales[12] : RGB.y; \n" \
"if (p_Scales[11] != 0.0f) \n" \
"RGB.z = RGB.z <= p_Scales[12] ? ((RGB.z / p_Scales[12]) + p_Scales[11] * (1.0f - (RGB.z / p_Scales[12]))) * p_Scales[12] : RGB.z; \n" \
"if (p_Scales[13] != 0.0f) \n" \
"RGB.x += p_Scales[13]; \n" \
"if (p_Scales[14] != 0.0f) \n" \
"RGB.y += p_Scales[14]; \n" \
"if (p_Scales[15] != 0.0f) \n" \
"RGB.z += p_Scales[15]; \n" \
"if (p_Scales[16] != 0.0f) { \n" \
"float Prl = RGB.x >= p_Scales[19] && RGB.x <= p_Scales[20] ? pow((RGB.x - p_Scales[19]) / (p_Scales[20] - p_Scales[19]), 1.0f / p_Scales[16]) * (p_Scales[20] - p_Scales[19]) + p_Scales[19] : RGB.x; \n" \
"float Pru = RGB.x >= p_Scales[19] && RGB.x <= p_Scales[20] ? (1.0f - pow(1.0f - (RGB.x - p_Scales[19]) / (p_Scales[20] - p_Scales[19]), p_Scales[16])) * (p_Scales[20] - p_Scales[19]) + p_Scales[19] : RGB.x; \n" \
"RGB.x = p_Switch[1] == 1 ? Pru : Prl; \n" \
"} \n" \
"if (p_Scales[17] != 0.0f) { \n" \
"float Pgl = RGB.y >= p_Scales[19] && RGB.y <= p_Scales[20] ? pow((RGB.y - p_Scales[19]) / (p_Scales[20] - p_Scales[19]), 1.0f / p_Scales[17]) * (p_Scales[20] - p_Scales[19]) + p_Scales[19] : RGB.y; \n" \
"float Pgu = RGB.y >= p_Scales[19] && RGB.y <= p_Scales[20] ? (1.0f - pow(1.0f - (RGB.y - p_Scales[19]) / (p_Scales[20] - p_Scales[19]), p_Scales[17])) * (p_Scales[20] - p_Scales[19]) + p_Scales[19] : RGB.y; \n" \
"RGB.y = p_Switch[1] == 1 ? Pgu : Pgl; \n" \
"} \n" \
"if (p_Scales[18] != 0.0f) { \n" \
"float Pbl = RGB.z >= p_Scales[19] && RGB.z <= p_Scales[20] ? pow((RGB.z - p_Scales[19]) / (p_Scales[20] - p_Scales[19]), 1.0f / p_Scales[16]) * (p_Scales[20] - p_Scales[19]) + p_Scales[19] : RGB.z; \n" \
"float Pbu = RGB.z >= p_Scales[19] && RGB.z <= p_Scales[20] ? (1.0f - pow(1.0f - (RGB.z - p_Scales[19]) / (p_Scales[20] - p_Scales[19]), p_Scales[16])) * (p_Scales[20] - p_Scales[19]) + p_Scales[19] : RGB.z; \n" \
"RGB.z = p_Switch[1] == 1 ? Pbu : Pbl; \n" \
"}						 \n" \
"return RGB; \n" \
"} \n" \
"kernel void k_videoGradeKernel(device const float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], constant int& p_LumaMath [[buffer (4)]],  \n" \
"constant int* p_Switch [[buffer (5)]], constant int& p_Display [[buffer (6)]], constant float* p_Scales [[buffer (7)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 RGB, RGB1; \n" \
"RGB.x = p_Input[index]; \n" \
"RGB.y = p_Input[index + 1]; \n" \
"RGB.z = p_Input[index + 2]; \n" \
"float width = p_Width; \n" \
"float height = p_Height; \n" \
"float X = id.x; \n" \
"float Y = id.y; \n" \
"const float THICK = 5.0f * (width / 1920.0f); \n" \
"RGB1.x = X / (width - 1.0f); \n" \
"RGB1.z = RGB1.y = RGB1.x; \n" \
"if (p_Display == 0) \n" \
"RGB = VideoGradeKernelA(RGB, p_LumaMath, p_Switch, p_Scales); \n" \
"if (p_Display == 1) { \n" \
"RGB1 = VideoGradeKernelA(RGB1, p_LumaMath, p_Switch, p_Scales); \n" \
"RGB.x = RGB1.x >= (Y - THICK) / height && RGB1.x <= (Y + THICK) / height ? 1.0f : 0.0f; \n" \
"RGB.y = RGB1.y >= (Y - THICK) / height && RGB1.y <= (Y + THICK) / height ? 1.0f : 0.0f; \n" \
"RGB.z = RGB1.z >= (Y - THICK) / height && RGB1.z <= (Y + THICK) / height ? 1.0f : 0.0f; \n" \
"} \n" \
"if (p_Display == 2) { \n" \
"RGB = VideoGradeKernelA(RGB, p_LumaMath, p_Switch, p_Scales); \n" \
"RGB1 = VideoGradeKernelA(RGB1, p_LumaMath, p_Switch, p_Scales); \n" \
"RGB1.x = RGB1.x >= (Y - THICK) / height && RGB1.x <= (Y + THICK) / height ? 1.0f : 0.0f; \n" \
"RGB1.y = RGB1.y >= (Y - THICK) / height && RGB1.y <= (Y + THICK) / height ? 1.0f : 0.0f; \n" \
"RGB1.z = RGB1.z >= (Y - THICK) / height && RGB1.z <= (Y + THICK) / height ? 1.0f : 0.0f; \n" \
"RGB.x = RGB1.x == 0.0f ? RGB.x : RGB1.x; \n" \
"RGB.y = RGB1.y == 0.0f ? RGB.y : RGB1.y; \n" \
"RGB.z = RGB1.z == 0.0f ? RGB.z : RGB1.z; \n" \
"} \n" \
"p_Output[index] = RGB.x; \n" \
"p_Output[index + 1] = RGB.y; \n" \
"p_Output[index + 2] = RGB.z; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, 
int p_Height, int p_LumaMath, int* p_Switch, int p_Display, float* p_Scales)
{
const char* VideoGradeKernel	= "k_videoGradeKernel";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLComputePipelineState>     _VideoGradeKernel;

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

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:VideoGradeKernel]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_VideoGradeKernel = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
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
[computeEncoder setComputePipelineState:_VideoGradeKernel];
int exeWidth = [_VideoGradeKernel threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

if (p_Switch[0] == 1) {
p_Scales[7] = p_Scales[6] = p_Scales[5];
p_Scales[11] = p_Scales[10] = p_Scales[9];
p_Scales[15] = p_Scales[14] = p_Scales[13];
p_Scales[18] = p_Scales[17] = p_Scales[16];
}

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_LumaMath length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:p_Switch length:(sizeof(int) * 4) atIndex: 5];
[computeEncoder setBytes:&p_Display length:sizeof(int) atIndex: 6];
[computeEncoder setBytes:p_Scales length:(sizeof(float) * 21) atIndex: 7];

[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder endEncoding];
[commandBuffer commit];
}