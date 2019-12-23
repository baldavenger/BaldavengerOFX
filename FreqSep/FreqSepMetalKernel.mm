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
"out.x = A; \n" \
"out.y = B; \n" \
"out.z = C; \n" \
"return out; \n" \
"} \n" \
"float from_func_Rec709( float v) { \n" \
"if (v < 0.08145f) \n" \
"return (v < 0.0f) ? 0.0f : v * (1.0f / 4.5f); \n" \
"else \n" \
"return pow( (v + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f) ); \n" \
"} \n" \
"float to_func_Rec709( float v) { \n" \
"if (v < 0.0181f) \n" \
"return (v < 0.0f) ? 0.0f : v * 4.5f; \n" \
"else \n" \
"return 1.0993f * pow(v, 0.45f) - (1.0993f - 1.0f); \n" \
"} \n" \
"float normalizedLogCToRelativeExposure( float x) { \n" \
"if (x > 0.149659f) \n" \
"return (pow(10.0f, (x - 0.385537f) / 0.247189f) - 0.052272f) / 5.555556f; \n" \
"else \n" \
"return (x - 0.092809f) / 5.367650f; \n" \
"} \n" \
"float relativeExposureToLogC( float x) { \n" \
"if (x > 0.010591f) \n" \
"return 0.247190f * log10(5.555556f * x + 0.052272f) + 0.385537f; \n" \
"else \n" \
"return 5.367655f * x + 0.092809f; \n" \
"} \n" \
"float3 Rec709_to_XYZ( float3 rgb) { \n" \
"rgb.x = from_func_Rec709(rgb.x); \n" \
"rgb.y = from_func_Rec709(rgb.y); \n" \
"rgb.z = from_func_Rec709(rgb.z); \n" \
"float3 xyz; \n" \
"xyz.x = 0.4124564f * rgb.x + 0.3575761f * rgb.y + 0.1804375f * rgb.z; \n" \
"xyz.y = 0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f * rgb.z; \n" \
"xyz.z = 0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z; \n" \
"return xyz; \n" \
"} \n" \
"float3 XYZ_to_Rec709(float3 xyz) { \n" \
"float3 rgb; \n" \
"rgb.x =  3.2404542f * xyz.x + -1.5371385f * xyz.y + -0.4985314f * xyz.z; \n" \
"rgb.y = -0.9692660f * xyz.x +  1.8760108f * xyz.y +  0.0415560f * xyz.z; \n" \
"rgb.z =  0.0556434f * xyz.x + -0.2040259f * xyz.y +  1.0572252f * xyz.z; \n" \
"rgb.x = to_func_Rec709(rgb.x); \n" \
"rgb.y = to_func_Rec709(rgb.y); \n" \
"rgb.z = to_func_Rec709(rgb.z); \n" \
"return rgb; \n" \
"} \n" \
"float3 ArrilogC_to_XYZ( float3 Alexa) { \n" \
"float r_lin = normalizedLogCToRelativeExposure(Alexa.x); \n" \
"float g_lin = normalizedLogCToRelativeExposure(Alexa.y); \n" \
"float b_lin = normalizedLogCToRelativeExposure(Alexa.z); \n" \
"float3 XYZ; \n" \
"XYZ.x = r_lin * 0.638008f + g_lin * 0.214704f + b_lin * 0.097744f; \n" \
"XYZ.y = r_lin * 0.291954f + g_lin * 0.823841f - b_lin * 0.115795f; \n" \
"XYZ.z = r_lin * 0.002798f - g_lin * 0.067034f + b_lin * 1.153294f; \n" \
"return XYZ; \n" \
"} \n" \
"float3 XYZ_to_ArrilogC( float3 XYZ) { \n" \
"float3 Alexa; \n" \
"Alexa.x = XYZ.x * 1.789066f - XYZ.y * 0.482534f - XYZ.z * 0.200076f; \n" \
"Alexa.y = XYZ.x * -0.639849f + XYZ.y * 1.3964f + XYZ.z * 0.194432f; \n" \
"Alexa.z = XYZ.x * -0.041532f + XYZ.y * 0.082335f + XYZ.z * 0.878868f; \n" \
"Alexa.x = relativeExposureToLogC(Alexa.x); \n" \
"Alexa.y = relativeExposureToLogC(Alexa.y); \n" \
"Alexa.z = relativeExposureToLogC(Alexa.z); \n" \
"return Alexa; \n" \
"} \n" \
"float3 ACEScct_to_XYZ( float3 in) { \n" \
"float Y_BRK = 0.155251141552511f; \n" \
"float A = 10.5402377416545f; \n" \
"float B = 0.0729055341958355f; \n" \
"float3 out; \n" \
"in.x = in.x > Y_BRK ? pow( 2.0f, in.x * 17.52f - 9.72f) : (in.x - B) / A; \n" \
"in.y = in.y > Y_BRK ? pow( 2.0f, in.y * 17.52f - 9.72f) : (in.y - B) / A; \n" \
"in.z = in.z > Y_BRK ? pow( 2.0f, in.z * 17.52f - 9.72f) : (in.z - B) / A; \n" \
"out.x = 0.6624541811f * in.x + 0.1340042065f * in.y + 0.156187687f * in.z; \n" \
"out.y = 0.2722287168f * in.x + 0.6740817658f * in.y + 0.0536895174f * in.z; \n" \
"out.z = -0.0055746495f * in.x + 0.0040607335f * in.y + 1.0103391003f * in.z; \n" \
"return out; \n" \
"} \n" \
"float3 XYZ_to_ACEScct( float3 in) { \n" \
"float X_BRK = 0.0078125f; \n" \
"float A = 10.5402377416545f; \n" \
"float B = 0.0729055341958355f; \n" \
"float3 out; \n" \
"out.x = 1.6410233797f * in.x + -0.3248032942f * in.y + -0.2364246952f * in.z; \n" \
"out.y = -0.6636628587f * in.x + 1.6153315917f * in.y + 0.0167563477f * in.z; \n" \
"out.z = 0.0117218943f * in.x + -0.008284442f * in.y + 0.9883948585f * in.z; \n" \
"out.x = out.x <= X_BRK ? A * out.x + B : (log2(out.x) + 9.72f) / 17.52f; \n" \
"out.y = out.y <= X_BRK ? A * out.y + B : (log2(out.y) + 9.72f) / 17.52f; \n" \
"out.z = out.z <= X_BRK ? A * out.z + B : (log2(out.z) + 9.72f) / 17.52f; \n" \
"return out; \n" \
"} \n" \
"float3 XYZ_to_LAB( float3 XYZ) { \n" \
"float fx, fy, fz; \n" \
"float Xn = 0.950489f; \n" \
"float Zn = 1.08884f; \n" \
"if(XYZ.x / Xn > 0.008856f) \n" \
"fx = pow(XYZ.x / Xn, 1.0f / 3.0f); \n" \
"else \n" \
"fx = 7.787f * (XYZ.x / Xn) + 0.137931f; \n" \
"if(XYZ.y > 0.008856f) \n" \
"fy = pow(XYZ.y, 1.0f / 3.0f); \n" \
"else \n" \
"fy = 7.787f * XYZ.y + 0.137931f; \n" \
"if(XYZ.z / Zn > 0.008856f) \n" \
"fz = pow(XYZ.z / Zn, 1.0f / 3.0f); \n" \
"else \n" \
"fz = 7.787f * (XYZ.z / Zn) + 0.137931f; \n" \
"float3 Lab; \n" \
"Lab.x = 1.16f * fy - 0.16f; \n" \
"Lab.y = 2.5f * (fx - fy) + 0.5f; \n" \
"Lab.z = 1.0f * (fy - fz) + 0.5f; \n" \
"return Lab; \n" \
"} \n" \
"float3 LAB_to_XYZ( float3 LAB) { \n" \
"float3 XYZ; \n" \
"float Xn = 0.950489f; \n" \
"float Zn = 1.08884f; \n" \
"float cy = (LAB.x + 0.16f) / 1.16f; \n" \
"if(cy >= 0.206893f) \n" \
"XYZ.y = cy * cy * cy; \n" \
"else \n" \
"XYZ.y = (cy - 0.137931f) / 7.787f; \n" \
"float cx = (LAB.y - 0.5f) / 2.5f + cy; \n" \
"if(cx >= 0.206893f) \n" \
"XYZ.x = Xn * cx * cx * cx; \n" \
"else \n" \
"XYZ.x = Xn * (cx - 0.137931f) / 7.787f; \n" \
"float cz = cy - (LAB.z - 0.5f); \n" \
"if(cz >= 0.206893f) \n" \
"XYZ.z = Zn * cz * cz * cz; \n" \
"else \n" \
"XYZ.z = Zn * (cz - 0.137931f) / 7.787f; \n" \
"return XYZ; \n" \
"} \n" \
"kernel void k_simple( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant int& ch_in [[buffer (4)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{                                 				    \n" \
"if ((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Output[index + ch_in] = p_Input[index + ch_in];  \n" \
"} \n" \
"} \n" \
"kernel void k_rec709_to_lab( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 rgb = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 lab = Rec709_to_XYZ(rgb); \n" \
"lab = XYZ_to_LAB(lab); \n" \
"p_Output[index] = lab.x; \n" \
"p_Output[index + 1] = lab.y; \n" \
"p_Output[index + 2] = lab.z; \n" \
"p_Input[index] = lab.x; \n" \
"} \n" \
"} \n" \
"kernel void k_lab_to_rec709( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 rgb = LAB_to_XYZ(lab); \n" \
"rgb = XYZ_to_Rec709(rgb); \n" \
"p_Input[index] = rgb.x; \n" \
"p_Input[index + 1] = rgb.y; \n" \
"p_Input[index + 2] = rgb.z; \n" \
"} \n" \
"} \n" \
"kernel void k_arri_to_lab( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 Alexa = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 lab = ArrilogC_to_XYZ(Alexa); \n" \
"lab = XYZ_to_LAB(lab); \n" \
"p_Output[index] = lab.x; \n" \
"p_Output[index + 1] = lab.y; \n" \
"p_Output[index + 2] = lab.z; \n" \
"p_Input[index] = lab.x; \n" \
"} \n" \
"} \n" \
"kernel void k_lab_to_arri( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 Alexa; \n" \
"Alexa = LAB_to_XYZ(lab); \n" \
"Alexa = XYZ_to_ArrilogC(Alexa); \n" \
"p_Input[index] = Alexa.x; \n" \
"p_Input[index + 1] = Alexa.y; \n" \
"p_Input[index + 2] = Alexa.z; \n" \
"} \n" \
"} \n" \
"kernel void k_acescct_to_lab( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 lab = ACEScct_to_XYZ(aces); \n" \
"lab = XYZ_to_LAB(lab); \n" \
"p_Output[index] = lab.x; \n" \
"p_Output[index + 1] = lab.y; \n" \
"p_Output[index + 2] = lab.z; \n" \
"p_Input[index] = lab.x; \n" \
"} \n" \
"} \n" \
"kernel void k_lab_to_acescct( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 aces; \n" \
"aces = LAB_to_XYZ(lab); \n" \
"aces = XYZ_to_ACEScct(aces); \n" \
"p_Input[index] = aces.x; \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"} \n" \
"} \n" \
" \n" \
"kernel void k_transpose( device float* p_Input [[buffer (11)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant int& ch_out [[buffer (4)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
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
"} \n" \
"} \n" \
"kernel void k_gaussian( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (11)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], constant int& ch_in [[buffer (4)]],  \n" \
"constant float& blur [[buffer (5)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
"uint2 blockIdx [[ threadgroup_position_in_grid ]], uint2 blockDim [[ threads_per_threadgroup ]]) \n" \
"{ \n" \
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
"} \n" \
"} \n" \
"kernel void k_freqSharpen( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant int& ch_in [[buffer (4)]], constant float& sharpen [[buffer (6)]],  \n" \
"constant int& p_Display [[buffer (7)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"float offset = p_Display == 1 ? 0.5f : 0.0f; \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Input[index + ch_in] = (p_Input[index + ch_in] - p_Output[index + ch_in]) * sharpen + offset; \n" \
"if (p_Display == 1) { \n" \
"p_Output[index + ch_in] = p_Input[index + ch_in]; \n" \
"} \n" \
"} \n" \
"} \n" \
"kernel void k_freqSharpenLuma( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant float& sharpen [[buffer (6)]], constant int& p_Display [[buffer (7)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"float offset = p_Display == 1 ? 0.5f : 0.0f; \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Input[index] = (p_Input[index] - p_Output[index]) * sharpen + offset; \n" \
"if (p_Display == 1) \n" \
"p_Output[index] = p_Output[index + 1] = p_Output[index + 2] = p_Input[index]; \n" \
"} \n" \
"} \n" \
"kernel void k_freqAdd( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant int& ch_in [[buffer (4)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Output[index + ch_in] = p_Input[index + ch_in] + p_Output[index + ch_in];												   \n" \
"} \n" \
"} \n" \
"kernel void k_lowFreqCont( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant int& ch_in [[buffer (4)]], constant int& p_Display [[buffer (7)]], constant int& curve [[buffer (8)]],  \n" \
"constant float& contrast [[buffer (9)]], constant float& pivot [[buffer (10)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float graph = 0.0f; \n" \
"if(curve == 1){ \n" \
"if(p_Input[index + ch_in] > 0.0f && p_Input[index + ch_in] < 1.0f) \n" \
"p_Input[index + ch_in] = p_Input[index + ch_in] <= pivot ? pow(p_Input[index + ch_in] / pivot, contrast) * pivot : (1.0f - pow(1.0f - (p_Input[index + ch_in] - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"} else { \n" \
"p_Input[index + ch_in] = (p_Input[index + ch_in] - pivot) * contrast + pivot; \n" \
"} \n" \
"if(p_Display == 3){ \n" \
"float width = p_Width; \n" \
"float height = p_Height; \n" \
"float X = id.x; \n" \
"float Y = id.y; \n" \
"float ramp = X / (width - 1.0f); \n" \
"if(curve == 1) \n" \
"ramp = ramp <= pivot ? pow(ramp / pivot, contrast) * pivot : (1.0f - pow(1.0f - (ramp - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"else \n" \
"ramp = (ramp - pivot) * contrast + pivot; \n" \
"graph = ramp >= (Y - 5.0f) / height && ramp <= (Y + 5.0f) / height ? 1.0f : 0.0f; \n" \
"p_Input[index + ch_in] = graph == 0.0f ? p_Input[index + ch_in] : graph; \n" \
"} \n" \
"} \n" \
"} \n" \
"kernel void k_lowFreqContLuma( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant int& p_Display [[buffer (7)]], constant int& curve [[buffer (8)]],  \n" \
"constant float& contrast [[buffer (9)]], constant float& pivot [[buffer (10)]], uint2 id [[ thread_position_in_grid ]]) \n" \
"{ \n" \
"if((id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float graph = 0.0f; \n" \
"if(curve == 1){ \n" \
"if(p_Input[index] > 0.0f && p_Input[index] < 1.0f) \n" \
"p_Input[index] = p_Input[index] <= pivot ? pow(p_Input[index] / pivot, contrast) * pivot : (1.0f - pow(1.0f - (p_Input[index] - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"} else { \n" \
"p_Input[index] = (p_Input[index] - pivot) * contrast + pivot; \n" \
"} \n" \
"if(p_Display == 2) \n" \
"p_Input[index + 2] = p_Input[index + 1] = p_Input[index]; \n" \
"if(p_Display == 3){ \n" \
"float width = p_Width; \n" \
"float height = p_Height; \n" \
"float X = id.x; \n" \
"float Y = id.y; \n" \
"float ramp = X / (width - 1.0f); \n" \
"if(curve == 1) \n" \
"ramp = ramp <= pivot ? pow(ramp / pivot, contrast) * pivot : (1.0f - pow(1.0f - (ramp - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"else \n" \
"ramp = (ramp - pivot) * contrast + pivot; \n" \
"graph = ramp >= (Y - 5.0f) / height && ramp <= (Y + 5.0f) / height ? 1.0f : 0.0f; \n" \
"p_Input[index] = graph == 0.0f ? p_Input[index] : graph; \n" \
"p_Input[index + 2] = p_Input[index + 1] = p_Input[index]; \n" \
"} \n" \
"} \n" \
"} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Space, int p_Display, float* p_Blur, float* p_Sharpen, float* p_Cont, int* p_Switch)
{
int red = 0;
int green = 1;
int blue = 2;

const char* simple				= "k_simple";
const char* gaussian			= "k_gaussian";
const char* transpose			= "k_transpose";
const char* Rec709toLAB			= "k_rec709_to_lab";
const char* LABtoRec709			= "k_lab_to_rec709";
const char* ARRItoLAB			= "k_arri_to_lab";
const char* LABtoARRI			= "k_lab_to_arri";
const char* ACEStoLAB			= "k_acescct_to_lab";
const char* LABtoACES			= "k_lab_to_acescct";
const char* freqSharpen			= "k_freqSharpen";
const char* freqSharpenLuma		= "k_freqSharpenLuma";
const char* lowFreqCont			= "k_lowFreqCont";
const char* lowFreqContLuma		= "k_lowFreqContLuma";
const char* freqAdd				= "k_freqAdd";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLBuffer>					tempBuffer;
id<MTLComputePipelineState>    _simple;
id<MTLComputePipelineState>    _gaussian;
id<MTLComputePipelineState>    _transpose;
id<MTLComputePipelineState>    _Rec709toLAB;
id<MTLComputePipelineState>    _LABtoRec709;
id<MTLComputePipelineState>    _ARRItoLAB;
id<MTLComputePipelineState>    _LABtoARRI;
id<MTLComputePipelineState>    _ACEStoLAB;
id<MTLComputePipelineState>    _LABtoACES;
id<MTLComputePipelineState>    _freqSharpen;
id<MTLComputePipelineState>    _freqSharpenLuma;
id<MTLComputePipelineState>    _lowFreqCont;
id<MTLComputePipelineState>    _lowFreqContLuma;
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

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:simple] ];

_simple				=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:gaussian] ];

_gaussian			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:transpose] ];

_transpose			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:Rec709toLAB] ];

_Rec709toLAB		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:LABtoRec709] ];

_LABtoRec709		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:ARRItoLAB] ];

_ARRItoLAB			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:LABtoARRI] ];

_LABtoARRI			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:ACEStoLAB] ];

_ACEStoLAB			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:LABtoACES] ];

_LABtoACES			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:freqSharpen] ];

_freqSharpen		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:freqSharpenLuma] ];

_freqSharpenLuma	=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:lowFreqCont] ];

_lowFreqCont		=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:lowFreqContLuma] ];

_lowFreqContLuma	=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  	=			[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:freqAdd] ];

_freqAdd			=			[device newComputePipelineStateWithFunction:kernelFunction error:&err];

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

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
for(int c = 0; c < 4; c++) {
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 11];
[computeEncoder setBytes:&p_Display length:sizeof(int) atIndex: 7];
[computeEncoder setBytes:&p_Switch[2] length:sizeof(int) atIndex: 8];
[computeEncoder setBytes:&p_Cont[0] length:sizeof(float) atIndex: 9];
[computeEncoder setBytes:&p_Cont[1] length:sizeof(float) atIndex: 10];

if(p_Space == 0) {
if (p_Switch[0] == 1)
p_Blur[2] = p_Blur[1] = p_Blur[0];
if (p_Blur[0] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[0] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
if (p_Blur[1] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&green length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[1] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
if (p_Blur[2] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[2] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
if (p_Switch[0] == 1)
p_Sharpen[2] = p_Sharpen[1] = p_Sharpen[0];
[computeEncoder setComputePipelineState:_freqSharpen];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Display length:sizeof(int) atIndex: 7];
for(int c = 0; c < 3; c++) {
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Sharpen[c] length:sizeof(float) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
if (p_Display != 1) {
if (p_Switch[0] == 1)
p_Blur[5] = p_Blur[4] = p_Blur[3];
if (p_Blur[3] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[3] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
if (p_Blur[4] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&green length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[4] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
if (p_Blur[5] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[5] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
[computeEncoder setComputePipelineState:_lowFreqCont];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
for(int c = 0; c < 3; c++) {
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
if (p_Display == 0) {
[computeEncoder setComputePipelineState:_freqAdd];
for(int c = 0; c < 3; c++) {
[computeEncoder setBytes:&c length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}}}

if(p_Space != 0) {
if(p_Space == 1){
[computeEncoder setComputePipelineState:_Rec709toLAB];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
if(p_Space == 2){
[computeEncoder setComputePipelineState:_ARRItoLAB];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
if(p_Space == 3){
[computeEncoder setComputePipelineState:_ACEStoLAB];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
if (p_Blur[0] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[0] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
[computeEncoder setComputePipelineState:_freqSharpenLuma];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&p_Sharpen[0] length:sizeof(float) atIndex: 6];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
if (p_Display != 1){
if (p_Switch[1] == 1)
p_Blur[5] = p_Blur[4];
if (p_Blur[3] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[3] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
if (p_Blur[4] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&green length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[4] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
if (p_Blur[5] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Blur[5] length:sizeof(float) atIndex: 5];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
}
[computeEncoder setComputePipelineState:_lowFreqContLuma];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
if (p_Display == 0) {
[computeEncoder setComputePipelineState:_freqAdd];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 4];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
if(p_Space == 1){
[computeEncoder setComputePipelineState:_LABtoRec709];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
if(p_Space == 2){
[computeEncoder setComputePipelineState:_LABtoARRI];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
if(p_Space == 3){
[computeEncoder setComputePipelineState:_LABtoACES];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}
}}}

[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
}