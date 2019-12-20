#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
#include <stdio.h>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define iTransposeBlockDim 32

size_t localWorkSize[2], globalWorkSize[2];
size_t gausLocalWorkSize[2], gausGlobalWorkSizeA[2], gausGlobalWorkSizeB[2];
size_t TransLocalWorkSize[2], TransGlobalWorkSizeA[2], TransGlobalWorkSizeB[2];

cl_mem tempBuffer;
size_t szBuffBytes;
cl_int error;

const char *KernelSource = \
"float normalizedLogCToRelativeExposure( float x); \n" \
"float relativeExposureToLogC( float x); \n" \
"float3 ArrilogC_to_XYZ( float3 Alexa); \n" \
"float3 XYZ_to_ArrilogC( float3 XYZ); \n" \
"float from_func_Rec709(float v); \n" \
"float to_func_Rec709(float v); \n" \
"float3 Rec709_to_XYZ( float3 rgb); \n" \
"float3 XYZ_to_Rec709(float3 xyz);\n"
"float3 XYZ_to_LAB( float3 XYZ); \n" \
"float3 LAB_to_XYZ( float3 LAB); \n" \
"float3 Rec709_to_LAB( float3 rgb); \n" \
"float3 LAB_to_Rec709( float3 lab); \n" \
"float3 ACEScct_to_XYZ( float3 in); \n" \
"float3 XYZ_to_ACEScct( float3 in); \n" \
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
"float from_func_Rec709(float v) { \n" \
"if (v < 0.08145f) \n" \
"return (v < 0.0f) ? 0.0f : v * (1.0f / 4.5f); \n" \
"else \n" \
"return pow( (v + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f) ); \n" \
"} \n" \
"float to_func_Rec709(float v) { \n" \
"if (v < 0.0181f) \n" \
"return (v < 0.0f) ? 0.0f : v * 4.5f; \n" \
"else \n" \
"return 1.0993f * pow(v, 0.45f) - (1.0993f - 1.0f); \n" \
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
"float3 Rec709_to_LAB( float3 rgb) { \n" \
"float3 lab; \n" \
"lab = Rec709_to_XYZ(rgb); \n" \
"lab = XYZ_to_LAB(lab); \n" \
"return lab; \n" \
"} \n" \
"float3 LAB_to_Rec709( float3 lab) { \n" \
"float3 rgb; \n" \
"rgb = LAB_to_XYZ(lab); \n" \
"rgb = XYZ_to_Rec709(rgb); \n" \
"return rgb; \n" \
"} \n" \
"float3 ACEScct_to_XYZ( float3 in) { \n" \
"const float Y_BRK = 0.155251141552511f; \n" \
"const float A = 10.5402377416545f; \n" \
"const float B = 0.0729055341958355f; \n" \
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
"const float X_BRK = 0.0078125f; \n" \
"const float A = 10.5402377416545f; \n" \
"const float B = 0.0729055341958355f; \n" \
"float3 out; \n" \
"out.x = 1.6410233797f * in.x + -0.3248032942f * in.y + -0.2364246952f * in.z; \n" \
"out.y = -0.6636628587f * in.x + 1.6153315917f * in.y + 0.0167563477f * in.z; \n" \
"out.z = 0.0117218943f * in.x + -0.008284442f * in.y + 0.9883948585f * in.z; \n" \
"out.x = out.x <= X_BRK ? A * out.x + B : (log2(out.x) + 9.72f) / 17.52f; \n" \
"out.y = out.y <= X_BRK ? A * out.y + B : (log2(out.y) + 9.72f) / 17.52f; \n" \
"out.z = out.z <= X_BRK ? A * out.z + B : (log2(out.z) + 9.72f) / 17.52f; \n" \
"return out; \n" \
"} \n" \
"__kernel void k_arri_to_lab(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 Alexa = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 lab = ArrilogC_to_XYZ(Alexa); \n" \
"lab = XYZ_to_LAB(lab); \n" \
"p_Output[index] = lab.x; \n" \
"p_Output[index + 1] = lab.y; \n" \
"p_Output[index + 2] = lab.z; \n" \
"p_Input[index] = lab.x; \n" \
"} \n" \
"} \n" \
"__kernel void k_lab_to_arri(__global float* p_Input, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 Alexa; \n" \
"Alexa = LAB_to_XYZ(lab); \n" \
"Alexa = XYZ_to_ArrilogC(Alexa); \n" \
"p_Input[index] = Alexa.x; \n" \
"p_Input[index + 1] = Alexa.y; \n" \
"p_Input[index + 2] = Alexa.z; \n" \
"} \n" \
"} \n" \
"__kernel void k_acescct_to_lab(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 aces = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 lab = ACEScct_to_XYZ(aces); \n" \
"lab = XYZ_to_LAB(lab); \n" \
"p_Output[index] = lab.x; \n" \
"p_Output[index + 1] = lab.y; \n" \
"p_Output[index + 2] = lab.z; \n" \
"p_Input[index] = lab.x; \n" \
"} \n" \
"} \n" \
"__kernel void k_lab_to_acescct(__global float* p_Input, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 aces; \n" \
"aces = LAB_to_XYZ(lab); \n" \
"aces = XYZ_to_ACEScct(aces); \n" \
"p_Input[index] = aces.x; \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"} \n" \
"} \n" \
"__kernel void k_rec709_to_lab(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 rgb = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 lab = Rec709_to_LAB(rgb); \n" \
"p_Output[index] = lab.x; \n" \
"p_Output[index + 1] = lab.y; \n" \
"p_Output[index + 2] = lab.z; \n" \
"p_Input[index] = lab.x; \n" \
"} \n" \
"} \n" \
"__kernel void k_lab_to_rec709(__global float* p_Input, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float3 rgb = LAB_to_Rec709(lab); \n" \
"p_Input[index] = rgb.x; \n" \
"p_Input[index + 1] = rgb.y; \n" \
"p_Input[index + 2] = rgb.z; \n" \
"} \n" \
"} \n" \
"__kernel void k_transpose(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, __local float* buffer, int c) { \n" \
"int xIndex = get_global_id(0); \n" \
"int yIndex = get_global_id(1); \n" \
"if ((xIndex < p_Width) && (yIndex < p_Height)) \n" \
"{ \n" \
"buffer[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = p_Input[(yIndex * p_Width + xIndex)]; \n" \
"} \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"xIndex = get_group_id(1) * get_local_size(1) + get_local_id(0); \n" \
"yIndex = get_group_id(0) * get_local_size(0) + get_local_id(1); \n" \
"if((xIndex < p_Height) && (yIndex < p_Width)) \n" \
"{ \n" \
"p_Output[(yIndex * p_Height + xIndex) * 4 + c] = buffer[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)]; \n" \
"} \n" \
"} \n" \
"__kernel void k_gaussian(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float blur, int c) { \n" \
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
"int x = get_group_id(0) * get_local_size(0) + get_local_id(0); \n" \
"if (x >= p_Width) return; \n" \
"p_Input += x * 4 + c; \n" \
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
"__kernel void k_freqSharpen(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float sharpen, int p_Display, int c) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"float offset = p_Display == 1 ? 0.5f : 0.0f; \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"p_Input[index + c] = (p_Input[index + c] - p_Output[index + c]) * sharpen + offset; \n" \
"if (p_Display == 1) \n" \
"p_Output[index + c] = p_Input[index + c]; \n" \
"}  \n" \
"} \n" \
"__kernel void k_freqSharpenLuma(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float sharpen, int p_Display) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"float offset = p_Display == 1 ? 0.5f : 0.0f; \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4;									 \n" \
"p_Input[index] = (p_Input[index] - p_Output[index]) * sharpen + offset; \n" \
"if (p_Display == 1) \n" \
"p_Output[index] = p_Output[index + 1] = p_Output[index + 2] = p_Input[index]; \n" \
"} \n" \
"} \n" \
"__kernel void k_lowFreqCont(__global float* p_Input, int p_Width, int p_Height, float contrast, float pivot, int curve, int p_Display, int c) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float graph = 0.0f; \n" \
"if(p_Display == 3){ \n" \
"float width = p_Width; \n" \
"float height = p_Height; \n" \
"float X = x; \n" \
"float Y = y; \n" \
"float ramp = X / (width - 1.0f); \n" \
"if(curve == 1) \n" \
"ramp = ramp <= pivot ? pow(ramp / pivot, contrast) * pivot : (1.0f - pow(1.0f - (ramp - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"else \n" \
"ramp = (ramp - pivot) * contrast + pivot; \n" \
"graph = ramp >= (Y - 5.0f) / height && ramp <= (Y + 5.0f) / height ? 1.0f : 0.0f; \n" \
"} \n" \
"if(curve == 1){ \n" \
"if(p_Input[index + c] > 0.0f && p_Input[index + c] < 1.0f) \n" \
"p_Input[index + c] = p_Input[index + c] <= pivot ? pow(p_Input[index + c] / pivot, contrast) * pivot : (1.0f - pow(1.0f - (p_Input[index + c] - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"} else { \n" \
"p_Input[index + c] = (p_Input[index + c] - pivot) * contrast + pivot; \n" \
"} \n" \
"if(p_Display == 3) \n" \
"p_Input[index + c] = graph == 0.0f ? p_Input[index + c] : graph; \n" \
"}  \n" \
"} \n" \
"__kernel void k_lowFreqContLuma(__global float* p_Input, int p_Width, int p_Height, float contrast, float pivot, int curve, int p_Display) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
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
"float X = x; \n" \
"float Y = y; \n" \
"float ramp = X / (width - 1.0f); \n" \
"if(curve == 1) \n" \
"ramp = ramp <= pivot ? pow(ramp / pivot, contrast) * pivot : (1.0f - pow(1.0f - (ramp - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot; \n" \
"else \n" \
"ramp = (ramp - pivot) * contrast + pivot; \n" \
"graph = ramp >= (Y - 5.0f) / height && ramp <= (Y + 5.0f) / height ? 1.0f : 0.0f; \n" \
"p_Input[index] = graph == 0.0f ? p_Input[index] : graph; \n" \
"p_Input[index + 2] = p_Input[index + 1] = p_Input[index]; \n" \
"} \n" \
"}  \n" \
"} \n" \
"__kernel void k_freqAdd(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, int c) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4;											   \n" \
"p_Output[index + c] = p_Input[index + c] + p_Output[index + c]; \n" \
"} \n" \
"} \n" \
"__kernel void k_simple(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, int c) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"p_Output[index + c] = p_Input[index + c]; \n" \
"} \n" \
"} \n" \
"\n";

class Locker
{
public:
Locker()
{
#ifdef _WIN64
InitializeCriticalSection(&mutex);
#else
pthread_mutex_init(&mutex, NULL);
#endif
}

~Locker()
{
#ifdef _WIN64
DeleteCriticalSection(&mutex);
#else
pthread_mutex_destroy(&mutex);
#endif
}

void Lock()
{
#ifdef _WIN64
EnterCriticalSection(&mutex);
#else
pthread_mutex_lock(&mutex);
#endif
}

void Unlock()
{
#ifdef _WIN64
LeaveCriticalSection(&mutex);
#else
pthread_mutex_unlock(&mutex);
#endif
}

private:
#ifdef _WIN64
CRITICAL_SECTION mutex;
#else
pthread_mutex_t mutex;
#endif
};


void CheckError(cl_int p_Error, const char* p_Msg) {
if (p_Error != CL_SUCCESS) {
fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
}}

int clDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

int shrRoundUp(size_t localWorkSize, int numItems) {
int result = localWorkSize;
while (result < numItems)
result += localWorkSize;
return result;
}

void RunOpenCLKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Space, int p_Display, float* p_Blur, float* p_Sharpen, float* p_Cont, int* p_Switch)
{
int red = 0;
int green = 1;
int blue = 2;

szBuffBytes = p_Width * p_Height * sizeof(float);
cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

static std::map<cl_command_queue, cl_device_id> deviceIdMap;
static std::map<cl_command_queue, cl_kernel> kernelMap;

static Locker locker;
locker.Lock();

cl_device_id deviceId = NULL;
if (deviceIdMap.find(cmdQ) == deviceIdMap.end()) {
error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
CheckError(error, "Unable to get the device");
deviceIdMap[cmdQ] = deviceId;
} else {
deviceId = deviceIdMap[cmdQ];
}

cl_context clContext = NULL;
error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
CheckError(error, "Unable to get the context");

cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
CheckError(error, "Unable to create program");

error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
CheckError(error, "Unable to build program");

tempBuffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &error);
CheckError(error, "Unable to create buffer");

cl_kernel Gausfilter = NULL;
cl_kernel Transpose = NULL;
cl_kernel Simple = NULL;
cl_kernel FreqSharpen = NULL;
cl_kernel FreqSharpenLuma = NULL;
cl_kernel LowFreqCont = NULL;
cl_kernel LowFreqContLuma = NULL;
cl_kernel FreqAdd = NULL;
cl_kernel Rec709toLAB = NULL;
cl_kernel LABtoRec709 = NULL;
cl_kernel ARRItoLAB = NULL;
cl_kernel LABtoARRI = NULL;
cl_kernel ACEStoLAB = NULL;
cl_kernel LABtoACES = NULL;

Gausfilter = clCreateKernel(program, "k_gaussian", &error);
CheckError(error, "Unable to create kernel");

Transpose = clCreateKernel(program, "k_transpose", &error);
CheckError(error, "Unable to create kernel");

Simple = clCreateKernel(program, "k_simple", &error);
CheckError(error, "Unable to create kernel");

FreqSharpen = clCreateKernel(program, "k_freqSharpen", &error);
CheckError(error, "Unable to create kernel");

FreqSharpenLuma = clCreateKernel(program, "k_freqSharpenLuma", &error);
CheckError(error, "Unable to create kernel");

LowFreqCont = clCreateKernel(program, "k_lowFreqCont", &error);
CheckError(error, "Unable to create kernel");

LowFreqContLuma = clCreateKernel(program, "k_lowFreqContLuma", &error);
CheckError(error, "Unable to create kernel");

FreqAdd = clCreateKernel(program, "k_freqAdd", &error);
CheckError(error, "Unable to create kernel");

Rec709toLAB = clCreateKernel(program, "k_rec709_to_lab", &error);
CheckError(error, "Unable to create kernel");

LABtoRec709 = clCreateKernel(program, "k_lab_to_rec709", &error);
CheckError(error, "Unable to create kernel");

ARRItoLAB = clCreateKernel(program, "k_arri_to_lab", &error);
CheckError(error, "Unable to create kernel");

LABtoARRI = clCreateKernel(program, "k_lab_to_arri", &error);
CheckError(error, "Unable to create kernel");

ACEStoLAB = clCreateKernel(program, "k_acescct_to_lab", &error);
CheckError(error, "Unable to create kernel");

LABtoACES = clCreateKernel(program, "k_lab_to_acescct", &error);
CheckError(error, "Unable to create kernel");

locker.Unlock();

localWorkSize[0] = gausLocalWorkSize[0] = 128;
localWorkSize[1] = gausLocalWorkSize[1] = gausGlobalWorkSizeA[1] = gausGlobalWorkSizeB[1] = 1;
globalWorkSize[0] = localWorkSize[0] * clDivUp(p_Width, localWorkSize[0]);
globalWorkSize[1] = p_Height;

gausGlobalWorkSizeA[0] = gausLocalWorkSize[0] * clDivUp(p_Width, gausLocalWorkSize[0]);
gausGlobalWorkSizeB[0] = gausLocalWorkSize[0] * clDivUp(p_Height, gausLocalWorkSize[0]);

TransLocalWorkSize[0] = TransLocalWorkSize[1] = iTransposeBlockDim;
TransGlobalWorkSizeA[0] = shrRoundUp(TransLocalWorkSize[0], p_Width);
TransGlobalWorkSizeA[1] = shrRoundUp(TransLocalWorkSize[1], p_Height);
TransGlobalWorkSizeB[0] = TransGlobalWorkSizeA[1];
TransGlobalWorkSizeB[1] = TransGlobalWorkSizeA[0];

error |= clSetKernelArg(Gausfilter, 1, sizeof(cl_mem), &tempBuffer);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Transpose, 0, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Transpose, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Transpose, 4, sizeof(float) * iTransposeBlockDim * (iTransposeBlockDim + 1), NULL);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Simple, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Simple, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Simple, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Simple, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(LowFreqCont, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LowFreqCont, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(LowFreqCont, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(LowFreqCont, 3, sizeof(float), &p_Cont[0]);
error |= clSetKernelArg(LowFreqCont, 4, sizeof(float), &p_Cont[1]);
error |= clSetKernelArg(LowFreqCont, 5, sizeof(int), &p_Switch[2]);
error |= clSetKernelArg(LowFreqCont, 6, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(LowFreqContLuma, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LowFreqContLuma, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(LowFreqContLuma, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(LowFreqContLuma, 3, sizeof(float), &p_Cont[0]);
error |= clSetKernelArg(LowFreqContLuma, 4, sizeof(float), &p_Cont[1]);
error |= clSetKernelArg(LowFreqContLuma, 5, sizeof(int), &p_Switch[2]);
error |= clSetKernelArg(LowFreqContLuma, 6, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FreqSharpen, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FreqSharpen, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FreqSharpen, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FreqSharpen, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(FreqSharpen, 5, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FreqSharpenLuma, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FreqSharpenLuma, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FreqSharpenLuma, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FreqSharpenLuma, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(FreqSharpenLuma, 4, sizeof(float), &p_Sharpen[0]);
error |= clSetKernelArg(FreqSharpenLuma, 5, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FreqAdd, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FreqAdd, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FreqAdd, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FreqAdd, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Rec709toLAB, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Rec709toLAB, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Rec709toLAB, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Rec709toLAB, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(LABtoRec709, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LABtoRec709, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(LABtoRec709, 2, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(ARRItoLAB, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(ARRItoLAB, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(ARRItoLAB, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(ARRItoLAB, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(LABtoARRI, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LABtoARRI, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(LABtoARRI, 2, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(ACEStoLAB, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(ACEStoLAB, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(ACEStoLAB, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(ACEStoLAB, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(LABtoACES, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LABtoACES, 1, sizeof(int), &p_Width);
error |= clSetKernelArg(LABtoACES, 2, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

for(int c = 0; c < 4; c++) {
error |= clSetKernelArg(Simple, 4, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, Simple, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}

if(p_Space == 0) {
if (p_Switch[0] == 1)
p_Blur[2] = p_Blur[1] = p_Blur[0];
if (p_Blur[0] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[0]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
if (p_Blur[1] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[1]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
if (p_Blur[2] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[2]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
if (p_Switch[0] == 1)
p_Sharpen[2] = p_Sharpen[1] = p_Sharpen[0];
for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(FreqSharpen, 4, sizeof(float), &p_Sharpen[c]);
error |= clSetKernelArg(FreqSharpen, 6, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FreqSharpen, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
if (p_Display != 1) {
if (p_Switch[0] == 1)
p_Blur[5] = p_Blur[4] = p_Blur[3];
if (p_Blur[3] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[3]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
if (p_Blur[4] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[4]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
if (p_Blur[5] > 0.0f) {
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[5]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(LowFreqCont, 7, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, LowFreqCont, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
if (p_Display == 0)
for(int c = 0; c < 3; c++) {
error |= clSetKernelArg(FreqAdd, 4, sizeof(int), &c);
clEnqueueNDRangeKernel(cmdQ, FreqAdd, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}}

if(p_Space != 0) {
if(p_Space == 1)
clEnqueueNDRangeKernel(cmdQ, Rec709toLAB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if(p_Space == 2)
clEnqueueNDRangeKernel(cmdQ, ARRItoLAB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if(p_Space == 3)
clEnqueueNDRangeKernel(cmdQ, ACEStoLAB, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Output);
CheckError(error, "Unable to set kernel arguments");
if (p_Blur[0] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[0]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
CheckError(error, "Unable to set kernel arguments");
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
clEnqueueNDRangeKernel(cmdQ, FreqSharpenLuma, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if (p_Display != 1){
if (p_Switch[1] == 1)
p_Blur[5] = p_Blur[4];
if (p_Blur[3] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[3]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
if (p_Blur[4] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[4]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
if (p_Blur[5] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[5]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
clEnqueueNDRangeKernel(cmdQ, LowFreqContLuma, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if (p_Display == 0) {
error |= clSetKernelArg(FreqAdd, 4, sizeof(int), &red);
clEnqueueNDRangeKernel(cmdQ, FreqAdd, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if(p_Space == 1)
clEnqueueNDRangeKernel(cmdQ, LABtoRec709, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if(p_Space == 2)
clEnqueueNDRangeKernel(cmdQ, LABtoARRI, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if(p_Space == 3)
clEnqueueNDRangeKernel(cmdQ, LABtoACES, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}}}
clReleaseMemObject(tempBuffer);
}