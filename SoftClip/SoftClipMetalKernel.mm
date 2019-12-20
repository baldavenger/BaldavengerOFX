#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"kernel void k_softclipKernel( device const float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], constant float* p_SoftClip [[buffer (4)]],  \n" \
"constant int* p_Switch [[buffer (5)]], constant int& p_Source [[buffer (6)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float r = p_Input[index];    \n" \
"float g = p_Input[index + 1];    \n" \
"float b = p_Input[index + 2]; \n" \
"float cr = (pow(10.0f, (1023.0f * r - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"float cg = (pow(10.0f, (1023.0f * g - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"float cb = (pow(10.0f, (1023.0f * b - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f); \n" \
"float lr = r > 0.1496582f ? (pow(10.0f, (r - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (r - 0.092809f) / 5.367655f; \n" \
"float lg = g > 0.1496582f ? (pow(10.0f, (g - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (g - 0.092809f) / 5.367655f; \n" \
"float lb = b > 0.1496582f ? (pow(10.0f, (b - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (b - 0.092809f) / 5.367655f; \n" \
"float mr = lr * 1.617523f  + lg * -0.537287f + lb * -0.080237f; \n" \
"float mg = lr * -0.070573f + lg * 1.334613f  + lb * -0.26404f; \n" \
"float mb = lr * -0.021102f + lg * -0.226954f + lb * 1.248056f; \n" \
"float sr = p_Source == 0 ? r : p_Source == 1 ? cr : mr; \n" \
"float sg = p_Source == 0 ? g : p_Source == 1 ? cg : mg; \n" \
"float sb = p_Source == 0 ? b : p_Source == 1 ? cb : mb; \n" \
"float Lr = sr > 1.0f ? 1.0f : sr; \n" \
"float Lg = sg > 1.0f ? 1.0f : sg; \n" \
"float Lb = sb > 1.0f ? 1.0f : sb; \n" \
"float Hr = (sr < 1.0f ? 1.0f : sr) - 1.0f; \n" \
"float Hg = (sg < 1.0f ? 1.0f : sg) - 1.0f; \n" \
"float Hb = (sb < 1.0f ? 1.0f : sb) - 1.0f; \n" \
"float rr = p_SoftClip[0]; \n" \
"float gg = p_SoftClip[1]; \n" \
"float aa = p_SoftClip[2]; \n" \
"float bb = p_SoftClip[3]; \n" \
"float ss = 1.0f - (p_SoftClip[4] / 10.0f); \n" \
"float sf = 1.0f - p_SoftClip[5]; \n" \
"float Hrr = Hr * pow(2.0f, rr); \n" \
"float Hgg = Hg * pow(2.0f, rr); \n" \
"float Hbb = Hb * pow(2.0f, rr); \n" \
"float HR = Hrr <= 1.0f ? 1.0f - pow(1.0f - Hrr, gg) : Hrr; \n" \
"float HG = Hgg <= 1.0f ? 1.0f - pow(1.0f - Hgg, gg) : Hgg; \n" \
"float HB = Hbb <= 1.0f ? 1.0f - pow(1.0f - Hbb, gg) : Hbb; \n" \
"float R = Lr + HR; \n" \
"float G = Lg + HG; \n" \
"float B = Lb + HB; \n" \
"float softr = aa == 1.0f ? R : (R > aa ? (-1.0f / ((R - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : R); \n" \
"float softR = bb == 1.0f ? softr : softr > 1.0f - (bb / 50.0f) ? (-1.0f / ((softr - (1.0f - (bb / 50.0f))) /  \n" \
"(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softr; \n" \
"float softg = (aa == 1.0f) ? G : (G > aa ? (-1.0f / ((G - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : G); \n" \
"float softG = bb == 1.0f ? softg : softg > 1.0f - (bb / 50.0f) ? (-1.0f / ((softg - (1.0f - (bb / 50.0f))) /  \n" \
"(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softg; \n" \
"float softb = (aa == 1.0f) ? B : (B > aa ? (-1.0f / ((B - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : B); \n" \
"float softB = bb == 1.0f ? softb : softb > 1.0f - (bb / 50.0f) ? (-1.0f / ((softb - (1.0f - (bb / 50.0f))) /  \n" \
"(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softb; \n" \
"float Cr = (softR * -1.0f) + 1.0f; \n" \
"float Cg = (softG * -1.0f) + 1.0f; \n" \
"float Cb = (softB * -1.0f) + 1.0f; \n" \
"float cR = ss == 1.0f ? Cr : Cr > ss ? (-1.0f / ((Cr - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cr; \n" \
"float CR = sf == 1.0f ? (cR - 1.0f) * -1.0f : ((cR > 1.0f - (-p_SoftClip[5] / 50.0f) ? (-1.0f / ((cR - (1.0f - (-p_SoftClip[5] / 50.0f))) /  \n" \
"(1.0f - (1.0f - (-p_SoftClip[5] / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClip[5] / 50.0f))) + (1.0f - (-p_SoftClip[5] / 50.0f)) : cR) - 1.0f) * -1.0f; \n" \
"float cG = ss == 1.0f ? Cg : Cg > ss ? (-1.0f / ((Cg - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cg; \n" \
"float CG = sf == 1.0f ? (cG - 1.0f) * -1.0f : ((cG > 1.0f - (-p_SoftClip[5] / 50.0f) ? (-1.0f / ((cG - (1.0f - (-p_SoftClip[5] / 50.0f))) /  \n" \
"(1.0f - (1.0f - (-p_SoftClip[5] / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClip[5] / 50.0f))) + (1.0f - (-p_SoftClip[5] / 50.0f)) : cG) - 1.0f) * -1.0f; \n" \
"float cB = ss == 1.0f ? Cb : Cb > ss ? (-1.0f / ((Cb - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cb; \n" \
"float CB = sf == 1.0f ? (cB - 1.0f) * -1.0f : ((cB > 1.0f - (-p_SoftClip[5] / 50.0f) ? (-1.0f / ((cB - (1.0f - (-p_SoftClip[5] / 50.0f))) /  \n" \
"(1.0f - (1.0f - (-p_SoftClip[5] / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClip[5] / 50.0f))) + (1.0f - (-p_SoftClip[5] / 50.0f)) : cB) - 1.0f) * -1.0f; \n" \
"float SR = p_Source == 0 ? CR : CR >= 0.0f && CR <= 1.0f ? (CR < 0.0181f ? (CR * 4.5f) : 1.0993f * pow(CR, 0.45f) - (1.0993f - 1.0f)) : CR; \n" \
"float SG = p_Source == 0 ? CG : CG >= 0.0f && CG <= 1.0f ? (CG < 0.0181f ? (CG * 4.5f) : 1.0993f * pow(CG, 0.45f) - (1.0993f - 1.0f)) : CG; \n" \
"float SB = p_Source == 0 ? CB : CB >= 0.0f && CB <= 1.0f ? (CB < 0.0181f ? (CB * 4.5f) : 1.0993f * pow(CB, 0.45f) - (1.0993f - 1.0f)) : CB; \n" \
"p_Output[index] = p_Switch[0] == 1 ? (SR < 1.0f ? 1.0f : SR) - 1.0f : p_Switch[1] == 1 ? (SR >= 0.0f ? 0.0f : SR + 1.0f) : SR; \n" \
"p_Output[index + 1] = p_Switch[0] == 1 ? (SG < 1.0f ? 1.0f : SG) - 1.0f : p_Switch[1] == 1 ? (SG >= 0.0f ? 0.0f : SG + 1.0f) : SG; \n" \
"p_Output[index + 2] = p_Switch[0] == 1 ? (SB < 1.0f ? 1.0f : SB) - 1.0f : p_Switch[1] == 1 ? (SB >= 0.0f ? 0.0f : SB + 1.0f) : SB; \n" \
"p_Output[index + 3] = 1.0f; \n" \
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, 
int p_Width, int p_Height, float* p_Scales, int* p_Switch, int p_Source)
{
const char* SoftClipKernel		= "k_softclipKernel";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLComputePipelineState>     _SoftClipKernel;

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

if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:SoftClipKernel]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_SoftClipKernel = [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
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

[computeEncoder setComputePipelineState:_SoftClipKernel];

int exeWidth = [_SoftClipKernel threadExecutionWidth];

MTLSize threadGroupCount 		= MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     		= MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:p_Scales length:(sizeof(float) * 6) atIndex: 4];
[computeEncoder setBytes:p_Switch length:(sizeof(int) * 2) atIndex: 5];
[computeEncoder setBytes:&p_Source length:sizeof(int) atIndex: 6];

[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];


[computeEncoder endEncoding];
[commandBuffer commit];
}