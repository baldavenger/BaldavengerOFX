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

const char *KernelSource = "\n" \
"__constant float pie = 3.1415926535898f; \n" \
"__constant float eu = 2.718281828459045f; \n" \
"__constant float sqrt3over4 = 0.433012701892219f; \n" \
"typedef struct { \n" \
"float3 c0, c1, c2; \n" \
"} mat3; \n" \
"float3 Make_float3(float A, float B, float C); \n" \
"mat3 make_mat3(float3 A, float3 B, float3 C); \n" \
"mat3 mult_f_f33(float f, mat3 A); \n" \
"float3 mult_f3_f33(float3 X, mat3 A); \n" \
"mat3 invert_f33(mat3 A); \n" \
"float Luma(float R, float G, float B, int L); \n" \
"float3 Sigmoid(float3 In, float peak, float curve, float pivot, float offset); \n" \
"float3 HSV_to_RGB(float3 HSV); \n" \
"float3 Hue_Chart(int width, int height, int X, int Y); \n" \
"float3 saturation_f3(float3 In, float luma, float sat); \n" \
"float Sat_Soft_Clip(float S, float softclip); \n" \
"float Limiter(float val, float limiter); \n" \
"float interpolate1D(float2 table[], float p, int Size); \n" \
"float cubic_basis_shaper(float x,  float w); \n" \
"float center_hue(float hue, float centerH); \n" \
"float uncenter_hue(float hueCentered, float centerH); \n" \
"float3 rgb_2_yab(float3 rgb); \n" \
"float3 yab_2_rgb(float3 yab); \n" \
"float3 yab_2_ych(float3 yab); \n" \
"float3 ych_2_yab(float3 ych); \n" \
"float3 rgb_2_ych(float3 rgb); \n" \
"float3 ych_2_rgb(float3 ych); \n" \
"float3 modify_hue(float3 ych, float Hue, float Range,  \n" \
"float Shift, float Converge, float SatScale); \n" \
"float median(float p_Table[], int m); \n" \
"float3 Make_float3(float A, float B, float C) { \n" \
"float3 out; \n" \
"out.x = A; out.y = B; out.z = C; \n" \
"return out; \n" \
"} \n" \
"mat3 make_mat3(float3 A, float3 B, float3 C) { \n" \
"mat3 D; \n" \
"D.c0 = A; D.c1 = B; D.c2 = C; \n" \
"return D; \n" \
"} \n" \
"mat3 mult_f_f33 (float f, mat3 A) { \n" \
"float r[3][3]; \n" \
"float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z}, \n" \
"{A.c1.x, A.c1.y, A.c1.z}, \n" \
"{A.c2.x, A.c2.y, A.c2.z}}; \n" \
"for( int i = 0; i < 3; ++i ) { \n" \
"for( int j = 0; j < 3; ++j ) { \n" \
"r[i][j] = f * a[i][j]; \n" \
"}} \n" \
"mat3 R = make_mat3(Make_float3(r[0][0], r[0][1], r[0][2]),  \n" \
"Make_float3(r[1][0], r[1][1], r[1][2]), Make_float3(r[2][0], r[2][1], r[2][2])); \n" \
"return R; \n" \
"} \n" \
"float3 mult_f3_f33 (float3 X, mat3 A) { \n" \
"float r[3]; \n" \
"float x[3] = {X.x, X.y, X.z}; \n" \
"float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z}, \n" \
"{A.c1.x, A.c1.y, A.c1.z}, \n" \
"{A.c2.x, A.c2.y, A.c2.z}}; \n" \
"for( int i = 0; i < 3; ++i) { \n" \
"r[i] = 0.0f; \n" \
"for( int j = 0; j < 3; ++j) { \n" \
"r[i] = r[i] + x[j] * a[j][i]; \n" \
"}} \n" \
"return Make_float3(r[0], r[1], r[2]); \n" \
"} \n" \
"mat3 invert_f33 (mat3 A) { \n" \
"mat3 R; \n" \
"float result[3][3]; \n" \
"float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z}, \n" \
"{A.c1.x, A.c1.y, A.c1.z}, \n" \
"{A.c2.x, A.c2.y, A.c2.z}}; \n" \
"float det =   a[0][0] * a[1][1] * a[2][2] \n" \
"+ a[0][1] * a[1][2] * a[2][0] \n" \
"+ a[0][2] * a[1][0] * a[2][1] \n" \
"- a[2][0] * a[1][1] * a[0][2] \n" \
"- a[2][1] * a[1][2] * a[0][0] \n" \
"- a[2][2] * a[1][0] * a[0][1]; \n" \
"if( det != 0.0f ) { \n" \
"result[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1]; \n" \
"result[0][1] = a[2][1] * a[0][2] - a[2][2] * a[0][1]; \n" \
"result[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1]; \n" \
"result[1][0] = a[2][0] * a[1][2] - a[1][0] * a[2][2]; \n" \
"result[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2]; \n" \
"result[1][2] = a[1][0] * a[0][2] - a[0][0] * a[1][2]; \n" \
"result[2][0] = a[1][0] * a[2][1] - a[2][0] * a[1][1]; \n" \
"result[2][1] = a[2][0] * a[0][1] - a[0][0] * a[2][1]; \n" \
"result[2][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1]; \n" \
"R = make_mat3(Make_float3(result[0][0], result[0][1], result[0][2]),  \n" \
"Make_float3(result[1][0], result[1][1], result[1][2]), Make_float3(result[2][0], result[2][1], result[2][2])); \n" \
"return mult_f_f33( 1.0f / det, R); \n" \
"} \n" \
"R = make_mat3(Make_float3(1.0f, 0.0f, 0.0f),  \n" \
"Make_float3(0.0f, 1.0f, 0.0f), Make_float3(0.0f, 0.0f, 1.0f)); \n" \
"return R; \n" \
"} \n" \
"float Luma(float R, float G, float B, int L) { \n" \
"float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f; \n" \
"float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f; \n" \
"float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f; \n" \
"float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f; \n" \
"float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f; \n" \
"float lumaAvg = (R + G + B) / 3.0f; \n" \
"float lumaMax = max(max(R, G), B); \n" \
"float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 :  \n" \
"L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax; \n" \
"return Lu; \n" \
"} \n" \
"float3 Sigmoid( float3 In, float peak, float curve, float pivot, float offset) { \n" \
"float3 out; \n" \
"out.x = peak / (1.0f + pow(eu, (-8.9f * curve) * (In.x - pivot))) + offset; \n" \
"out.y = peak / (1.0f + pow(eu, (-8.9f * curve) * (In.y - pivot))) + offset; \n" \
"out.z = peak / (1.0f + pow(eu, (-8.9f * curve) * (In.z - pivot))) + offset; \n" \
"return out; \n" \
"} \n" \
"float3 HSV_to_RGB(float3 HSV) { \n" \
"float3 RGB = Make_float3(0.0f, 0.0f, 0.0f); \n" \
"if (HSV.y == 0.0f) { \n" \
"RGB.x = RGB.y = RGB.z = HSV.z; \n" \
"} else { \n" \
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
"} \n" \
"return RGB; \n" \
"} \n" \
"float3 Hue_Chart( int width, int height, int X, int Y) { \n" \
"float3 HSV = Make_float3(0.0f, 0.0f, 0.0f); \n" \
"HSV.x = (float)X / ((float)width - 1.0f); \n" \
"HSV.y = (float)Y / ((float)height - 1.0f); \n" \
"HSV.y = pow(HSV.y, 0.4f); \n" \
"HSV.z = 0.75f; \n" \
"float3 RGB = HSV_to_RGB(HSV); \n" \
"return RGB; \n" \
"} \n" \
"float3 saturation_f3( float3 In, float luma, float sat) { \n" \
"float3 out = Make_float3(0.0f, 0.0f, 0.0f); \n" \
"out.x = (1.0f - sat) * luma + sat * In.x; \n" \
"out.y = (1.0f - sat) * luma + sat * In.y; \n" \
"out.z = (1.0f - sat) * luma + sat * In.z; \n" \
"return out; \n" \
"} \n" \
"float Sat_Soft_Clip(float S, float softclip) { \n" \
"softclip *= 0.3f; \n" \
"float ss = S > softclip ? (-1.0f / ((S - softclip) / (1.0f - softclip) + 1.0f) + 1.0f) * (1.0f - softclip) + softclip : S; \n" \
"return ss; \n" \
"} \n" \
"float Limiter(float val, float limiter) { \n" \
"float Alpha = limiter > 1.0f ? val + (1.0f - limiter) * (1.0f - val) : limiter >= 0.0f ? (val >= limiter ? 1.0f :  \n" \
"val / limiter) : limiter < -1.0f ? (1.0f - val) + (limiter + 1.0f) * val : val <= (1.0f + limiter) ? 1.0f :  \n" \
"(1.0 - val) / (1.0f - (limiter + 1.0f)); \n" \
"Alpha = clamp(Alpha, 0.0f, 1.0f); \n" \
"return Alpha; \n" \
"} \n" \
"float interpolate1D( float2 table[], float p, int Size) { \n" \
"if( p <= table[0].x ) return table[0].y; \n" \
"if( p >= table[Size - 1].x ) return table[Size - 1].y; \n" \
"for( int i = 0; i < Size - 1; ++i ){ \n" \
"if( table[i].x <= p && p < table[i + 1].x ){ \n" \
"float s = (p - table[i].x) / (table[i + 1].x - table[i].x); \n" \
"return table[i].y * ( 1.0f - s ) + table[i+1].y * s;}} \n" \
"return 0.0f; \n" \
"} \n" \
"float cubic_basis_shaper ( float x,  float w) { \n" \
"float M[4][4] = { { -1.0f/6.0f,  1.0f/2.0f, -1.0f/2.0f,  1.0f/6.0f }, \n" \
"{  1.0f/2.0f, -1.0f,  1.0f/2.0f,  0.0f }, \n" \
"{ -1.0f/2.0f,  0.0f,  1.0f/2.0f,  0.0f }, \n" \
"{  1.0f/6.0f,  2.0f/3.0f,  1.0f/6.0f,  0.0f } }; \n" \
"float knots[5] = { -w/2.0f, -w/4.0f, 0.0f, w/4.0f, w/2.0f }; \n" \
"float y = 0.0f; \n" \
"if ((x > knots[0]) && (x < knots[4])) {   \n" \
"float knot_coord = (x - knots[0]) * 4.0f/w;   \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float monomials[4] = { t*t*t, t*t, t, 1.0f }; \n" \
"if ( j == 3) { \n" \
"y = monomials[0] * M[0][0] + monomials[1] * M[1][0] +  \n" \
"monomials[2] * M[2][0] + monomials[3] * M[3][0]; \n" \
"} else if ( j == 2) { \n" \
"y = monomials[0] * M[0][1] + monomials[1] * M[1][1] +  \n" \
"monomials[2] * M[2][1] + monomials[3] * M[3][1]; \n" \
"} else if ( j == 1) { \n" \
"y = monomials[0] * M[0][2] + monomials[1] * M[1][2] +  \n" \
"monomials[2] * M[2][2] + monomials[3] * M[3][2]; \n" \
"} else if ( j == 0) { \n" \
"y = monomials[0] * M[0][3] + monomials[1] * M[1][3] +  \n" \
"monomials[2] * M[2][3] + monomials[3] * M[3][3]; \n" \
"} else { \n" \
"y = 0.0f; \n" \
"}} \n" \
"return y * 3.0f/2.0f; \n" \
"} \n" \
"float center_hue( float hue, float centerH) { \n" \
"float hueCentered = hue - centerH; \n" \
"if (hueCentered < -180.0f) hueCentered = hueCentered + 360.0f; \n" \
"else if (hueCentered > 180.0f) hueCentered = hueCentered - 360.0f; \n" \
"return hueCentered; \n" \
"} \n" \
"float uncenter_hue( float hueCentered, float centerH) { \n" \
"float hue = hueCentered + centerH; \n" \
"if (hue < 0.0f) hue = hue + 360.0f; \n" \
"else if (hue > 360.0f) hue = hue - 360.0f; \n" \
"return hue; \n" \
"} \n" \
"float3 rgb_2_yab( float3 rgb) { \n" \
"mat3 RGB_2_YAB_MAT = make_mat3(Make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f),  \n" \
"Make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), Make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4)); \n" \
"float3 yab = mult_f3_f33( rgb, RGB_2_YAB_MAT); \n" \
"return yab; \n" \
"} \n" \
"float3 yab_2_rgb( float3 yab) { \n" \
"mat3 RGB_2_YAB_MAT = make_mat3(Make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f),  \n" \
"Make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), Make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4)); \n" \
"float3 rgb = mult_f3_f33( yab, invert_f33(RGB_2_YAB_MAT)); \n" \
"return rgb; \n" \
"} \n" \
"float3 yab_2_ych(float3 yab) { \n" \
"float3 ych = yab; \n" \
"float yo = yab.y * yab.y + yab.z * yab.z; \n" \
"ych.y = sqrt(yo); \n" \
"ych.z = atan2(yab.z, yab.y) * (180.0f / pie); \n" \
"if (ych.z < 0.0f) ych.z += 360.0f; \n" \
"return ych; \n" \
"} \n" \
"float3 ych_2_yab( float3 ych )  { \n" \
"float3 yab; \n" \
"yab.x = ych.x; \n" \
"float h = ych.z * (pie / 180.0f); \n" \
"yab.y = ych.y * cos(h); \n" \
"yab.z = ych.y * sin(h); \n" \
"return yab; \n" \
"} \n" \
"float3 rgb_2_ych( float3 rgb) { \n" \
"return yab_2_ych( rgb_2_yab( rgb)); \n" \
"} \n" \
"float3 ych_2_rgb( float3 ych) { \n" \
"return yab_2_rgb( ych_2_yab( ych)); \n" \
"} \n" \
"float3 modify_hue( float3 ych, float Hue, float Range,  \n" \
"float Shift, float Converge, float SatScale) { \n" \
"float3 new_ych = ych; \n" \
"float centeredHue = center_hue( ych.z, Hue); \n" \
"float f_H = cubic_basis_shaper( centeredHue, Range); \n" \
"float old_hue = centeredHue; \n" \
"float new_hue = centeredHue + Shift; \n" \
"float2 table[2] = { {0.0f, old_hue}, {1.0f, new_hue} }; \n" \
"float blended_hue = interpolate1D( table, f_H, 2); \n" \
"if (f_H > 0.0f)  { \n" \
"new_ych.z = uncenter_hue(blended_hue, Hue); \n" \
"if (ych.y > 0.0f) { \n" \
"new_ych.y = ych.y * (f_H * (SatScale - 1.0f) + 1.0f); \n" \
"}} \n" \
"float h, H, hue, range; \n" \
"h = new_ych.z / 360.0f; \n" \
"H = h; \n" \
"hue = (Hue + Shift) / 360.0f; \n" \
"hue = hue > 1.0f ? hue - 1.0f : hue < 0.0f ? hue + 1.0f : hue; \n" \
"range = Range / 720.0f; \n" \
"h = h - (hue - 0.5f) < 0.0f ? h - (hue - 0.5f) + 1.0f : h - (hue - 0.5f) > \n" \
"1.0f ? h - (hue - 0.5f) - 1.0f : h - (hue - 0.5f); \n" \
"H = h > 0.5f - range && h < 0.5f ? (1.0f - pow(1.0f - (h - (0.5f - range)) * \n" \
"(1.0f/range), Converge)) * range + (0.5f - range) + (hue - 0.5f) : h > 0.5f && h < 0.5f +  \n" \
"range ? pow((h - 0.5f) * (1.0f/range), Converge) * range + 0.5f + (hue - 0.5f) : H; \n" \
"new_ych.z = H * 360.0f; \n" \
"return new_ych; \n" \
"} \n" \
"float median(float p_Table[], int m) { \n" \
"float temp; \n" \
"int i, j; \n" \
"for(i = 0; i < m - 1; i++) { \n" \
"for(j = i + 1; j < m; j++) { \n" \
"if(p_Table[j] < p_Table[i]) { \n" \
"temp = p_Table[i]; \n" \
"p_Table[i] = p_Table[j]; \n" \
"p_Table[j] = temp; }}} \n" \
"return p_Table[(m - 1) / 2]; \n" \
"} \n" \
"__kernel void k_logStageKernel( __global float* p_Input, __global float* p_Output, int p_Width, int p_Height, int p_SwitchLog, int p_SwitchHue, \n" \
"float p_LogA, float p_LogB, float p_LogC, float p_LogD, float p_SatA, float p_SatB, float p_LumaLimit, float p_SatLimit, int p_Math, int p_Chart) {	 \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 RGB = Make_float3(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2]); \n" \
"if(p_Chart == 1) \n" \
"RGB = Hue_Chart( p_Width, p_Height, x, y); \n" \
"if(p_SwitchLog == 1) \n" \
"RGB = Sigmoid( RGB, p_LogA, p_LogB, p_LogC, p_LogD); \n" \
"float luma = Luma(RGB.x, RGB.y, RGB.z, p_Math); \n" \
"if(p_SatA != 1.0f) { \n" \
"float minluma = min(RGB.x, min(RGB.y, RGB.z)); \n" \
"RGB = saturation_f3( RGB, minluma, p_SatA); \n" \
"} \n" \
"if(p_SatB != 1.0f) { \n" \
"float maxluma = max(RGB.x, max(RGB.y, RGB.z)); \n" \
"RGB = saturation_f3( RGB, maxluma, p_SatB); \n" \
"} \n" \
"float lumaAlpha = 1.0f; \n" \
"float satAlpha = 1.0f; \n" \
"if(p_LumaLimit != 0.0f && p_SwitchHue == 1) \n" \
"lumaAlpha = Limiter(luma, p_LumaLimit); \n" \
"float3 ych = rgb_2_ych( RGB); \n" \
"if(p_SatLimit != 0.0f && p_SwitchHue == 1) { \n" \
"satAlpha = Limiter(ych.y * 10.0f, p_SatLimit); \n" \
"} \n" \
"p_Output[index] = ych.x; \n" \
"p_Output[index + 1] = ych.y; \n" \
"p_Output[index + 2] = ych.z; \n" \
"p_Output[index + 3] = luma; \n" \
"p_Input[index] = 1.0f * lumaAlpha * satAlpha; \n" \
"}} \n" \
"__kernel void k_hueMedian9(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"int i, j; \n" \
"float Table[9]; \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"float temp = 0.0f; \n" \
"for(i = -1; i <= 1; i++) { \n" \
"for(j = -1; j <= 1; j++) { \n" \
"Table[(i + 1) * 3 + j + 1] = p_Input[( min(max(y + i, 0), p_Height) * p_Width + min(max(x + j, 0), p_Width) ) * 4 + 2]; \n" \
"}} \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"p_Output[y * p_Width + x] = median(Table, 9); \n" \
"}} \n" \
"__kernel void k_hueMedian25(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"int i, j; \n" \
"float Table[25]; \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"float temp = 0.0f; \n" \
"for(i = -2; i <= 2; i++) { \n" \
"for(j = -2; j <= 2; j++) { \n" \
"Table[(i + 2) * 5 + j + 2] = p_Input[( min(max(y + i, 0), p_Height) * p_Width + min(max(x + j, 0), p_Width) ) * 4 + 2]; \n" \
"}} \n" \
"barrier(CLK_LOCAL_MEM_FENCE); \n" \
"p_Output[y * p_Width + x] = median(Table, 25); \n" \
"}} \n" \
"__kernel void k_tempReturn(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"p_Input[index + 2] = p_Output[y * p_Width + x]; \n" \
"}} \n" \
"__kernel void k_hueStageKernel(__global float* p_In, __global float* p_ALPHA, int p_Width, int p_Height, int p_SwitchHue, int p_SwitchHue1,  \n" \
"float p_Hue1, float p_Hue2, float p_Hue3, float p_Hue4, float p_Hue5, float p_LumaLimit, float p_SatLimit, int ALPHA, int p_Isolate) {	 \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 ych = Make_float3(p_In[index], p_In[index + 1], p_In[index + 2]); \n" \
"float lumaAlpha = 1.0f; \n" \
"float satAlpha = 1.0f; \n" \
"if(p_LumaLimit != 0.0f && p_SwitchHue1 == 1) \n" \
"lumaAlpha = Limiter(p_In[index + 3], p_LumaLimit); \n" \
"if(p_SatLimit != 0.0f && p_SwitchHue1 == 1) \n" \
"satAlpha = Limiter(ych.y * 10.0f, p_SatLimit); \n" \
"if(p_Isolate == ALPHA + 1) \n" \
"{ \n" \
"float offset = 180.0f - p_Hue1; \n" \
"float outside = ych.z + offset; \n" \
"outside = outside > 360.0f ? outside - 360.0f : outside < 0.0f ? outside + 360.0f : outside; \n" \
"if(outside > 180.0f + (p_Hue2 / 2) || outside < 180.0f - (p_Hue2 / 2)) \n" \
"ych.y = 0.0f; \n" \
"} \n" \
"float3 new_ych = modify_hue( ych, p_Hue1, p_Hue2, p_Hue3, p_Hue4, p_Hue5); \n" \
"float alpha = p_ALPHA[index + ALPHA]; \n" \
"if(alpha != 1.0f) \n" \
"{ \n" \
"float3 RGB = ych_2_rgb( ych); \n" \
"float3 new_RGB = ych_2_rgb( new_ych); \n" \
"new_RGB.x = new_RGB.x * alpha + RGB.x * (1.0f - alpha); \n" \
"new_RGB.y = new_RGB.y * alpha + RGB.y * (1.0f - alpha); \n" \
"new_RGB.z = new_RGB.z * alpha + RGB.z * (1.0f - alpha); \n" \
"new_ych = rgb_2_ych( new_RGB); \n" \
"} \n" \
"p_In[index] = p_SwitchHue == 1 ? new_ych.x : p_In[index]; \n" \
"p_In[index + 1] = p_SwitchHue == 1 ? new_ych.y : p_In[index + 1]; \n" \
"p_In[index + 2] = p_SwitchHue == 1 ? new_ych.z : p_In[index + 2]; \n" \
"p_ALPHA[index + ALPHA + 1] = 1.0f * lumaAlpha * satAlpha; \n" \
"}} \n" \
"__kernel void k_finalStageKernel(__global float* p_In, __global float* p_ALPHA, int p_Width, int p_Height, float p_SatSoft, int p_Display) { \n" \
"int x = get_global_id(0); \n" \
"int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{ \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 ych = Make_float3(p_In[index + 0], p_In[index + 1], p_In[index + 2]); \n" \
"float alpha = p_ALPHA[index + 3]; \n" \
"if(p_SatSoft != 1.0f) \n" \
"{ \n" \
"float soft = Sat_Soft_Clip(ych.y, p_SatSoft); \n" \
"ych.y = soft * alpha + ych.y * (1.0f - alpha); \n" \
"} \n" \
"float luma = p_In[index + 3]; \n" \
"float3 rgb = ych_2_rgb( ych); \n" \
"if(p_Display != 0) \n" \
"{ \n" \
"float displayAlpha = p_Display == 1 ? p_ALPHA[index + 0] : p_Display == 2 ?  \n" \
"p_ALPHA[index + 1] : p_Display == 3 ? p_ALPHA[index + 2] :  p_Display == 4 ?  \n" \
"alpha : p_Display == 5 ? (ych.z / 360.0f) : p_Display == 6 ? clamp(ych.y * 10.0f, 0.0f, 1.0f) : luma;   \n" \
"p_In[index] = displayAlpha; \n" \
"p_In[index + 1] = displayAlpha; \n" \
"p_In[index + 2] = displayAlpha; \n" \
"p_In[index + 3] = 1.0f; \n" \
"} else { \n" \
"p_In[index] = rgb.x; \n" \
"p_In[index + 1] = rgb.y; \n" \
"p_In[index + 2] = rgb.z; \n" \
"p_In[index + 3] = 1.0f; \n" \
"}}} \n" \
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
"}} \n" \
"__kernel void k_gaussian(__global float* p_Input, __global float* p_Output, int p_Width, int p_Height, float blur, int c) { \n" \
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
"}} \n" \
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


void CheckError(cl_int p_Error, const char* p_Msg)
{
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

void RunOpenCLKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, int* p_Switch, float* p_Log, float* p_Sat, 
float *p_Hue1, float *p_Hue2, float *p_Hue3, int p_Display, float* p_Blur, int p_Math, int p_HueMedian, int p_Isolate)
{
int red = 0;
int green = 1;
int blue = 2;
int alpha = 3;
float zero = 0.0f;

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
}
else
{
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
cl_kernel LogStageKernel = NULL;
cl_kernel HueMedian9 = NULL;
cl_kernel HueMedian25 = NULL;
cl_kernel TempReturn = NULL;
cl_kernel HueStageKernel = NULL;
cl_kernel FinalStageKernel = NULL;

Gausfilter = clCreateKernel(program, "k_gaussian", &error);
CheckError(error, "Unable to create kernel");

Transpose = clCreateKernel(program, "k_transpose", &error);
CheckError(error, "Unable to create kernel");

LogStageKernel = clCreateKernel(program, "k_logStageKernel", &error);
CheckError(error, "Unable to create kernel");

HueMedian9 = clCreateKernel(program, "k_hueMedian9", &error);
CheckError(error, "Unable to create kernel");

HueMedian25 = clCreateKernel(program, "k_hueMedian25", &error);
CheckError(error, "Unable to create kernel");

TempReturn = clCreateKernel(program, "k_tempReturn", &error);
CheckError(error, "Unable to create kernel");

HueStageKernel = clCreateKernel(program, "k_hueStageKernel", &error);
CheckError(error, "Unable to create kernel");

FinalStageKernel = clCreateKernel(program, "k_finalStageKernel", &error);
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

error = clSetKernelArg(Gausfilter, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Gausfilter, 1, sizeof(cl_mem), &tempBuffer);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(Transpose, 0, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(Transpose, 1, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(Transpose, 4, sizeof(float) * iTransposeBlockDim * (iTransposeBlockDim + 1), NULL);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(LogStageKernel, 0, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(LogStageKernel, 1, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(LogStageKernel, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(LogStageKernel, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(LogStageKernel, 4, sizeof(int), &p_Switch[0]);
error |= clSetKernelArg(LogStageKernel, 5, sizeof(int), &p_Switch[1]);
error |= clSetKernelArg(LogStageKernel, 6, sizeof(float), &p_Log[0]);
error |= clSetKernelArg(LogStageKernel, 7, sizeof(float), &p_Log[1]);
error |= clSetKernelArg(LogStageKernel, 8, sizeof(float), &p_Log[2]);
error |= clSetKernelArg(LogStageKernel, 9, sizeof(float), &p_Log[3]);
error |= clSetKernelArg(LogStageKernel, 10, sizeof(float), &p_Sat[0]);
error |= clSetKernelArg(LogStageKernel, 11, sizeof(float), &p_Sat[1]);
error |= clSetKernelArg(LogStageKernel, 12, sizeof(float), &p_Hue1[5]);
error |= clSetKernelArg(LogStageKernel, 13, sizeof(float), &p_Hue1[6]);
error |= clSetKernelArg(LogStageKernel, 14, sizeof(int), &p_Math);
error |= clSetKernelArg(LogStageKernel, 15, sizeof(int), &p_Switch[4]);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(HueMedian9, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(HueMedian9, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(HueMedian9, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(HueMedian9, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(HueMedian25, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(HueMedian25, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(HueMedian25, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(HueMedian25, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(TempReturn, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(TempReturn, 1, sizeof(cl_mem), &tempBuffer);
error |= clSetKernelArg(TempReturn, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(TempReturn, 3, sizeof(int), &p_Height);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(HueStageKernel, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(HueStageKernel, 1, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(HueStageKernel, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(HueStageKernel, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(HueStageKernel, 4, sizeof(int), &p_Switch[1]);
error |= clSetKernelArg(HueStageKernel, 5, sizeof(int), &p_Switch[2]);
error |= clSetKernelArg(HueStageKernel, 6, sizeof(float), &p_Hue1[0]);
error |= clSetKernelArg(HueStageKernel, 7, sizeof(float), &p_Hue1[1]);
error |= clSetKernelArg(HueStageKernel, 8, sizeof(float), &p_Hue1[2]);
error |= clSetKernelArg(HueStageKernel, 9, sizeof(float), &p_Hue1[3]);
error |= clSetKernelArg(HueStageKernel, 10, sizeof(float), &p_Hue1[4]);
error |= clSetKernelArg(HueStageKernel, 11, sizeof(float), &p_Hue2[5]);
error |= clSetKernelArg(HueStageKernel, 12, sizeof(float), &p_Hue2[6]);
error |= clSetKernelArg(HueStageKernel, 13, sizeof(int), &red);
error |= clSetKernelArg(HueStageKernel, 14, sizeof(int), &p_Isolate);
CheckError(error, "Unable to set kernel arguments");

error = clSetKernelArg(FinalStageKernel, 0, sizeof(cl_mem), &p_Output);
error |= clSetKernelArg(FinalStageKernel, 1, sizeof(cl_mem), &p_Input);
error |= clSetKernelArg(FinalStageKernel, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(FinalStageKernel, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(FinalStageKernel, 4, sizeof(float), &p_Sat[2]);
error |= clSetKernelArg(FinalStageKernel, 5, sizeof(int), &p_Display);
CheckError(error, "Unable to set kernel arguments");

clEnqueueNDRangeKernel(cmdQ, LogStageKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);


if(p_HueMedian == 1 || p_HueMedian == 2)
{
if(p_HueMedian == 1)
clEnqueueNDRangeKernel(cmdQ, HueMedian9, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if(p_HueMedian == 2)
clEnqueueNDRangeKernel(cmdQ, HueMedian25, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, TempReturn, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}

if (p_Blur[0] > 0.0f && p_Switch[1] == 1) {
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
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
clEnqueueNDRangeKernel(cmdQ, HueStageKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if (p_Blur[1] > 0.0f && p_Switch[2] == 1) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[1]);
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
error |= clSetKernelArg(HueStageKernel, 4, sizeof(int), &p_Switch[2]);
error |= clSetKernelArg(HueStageKernel, 5, sizeof(int), &p_Switch[3]);
error |= clSetKernelArg(HueStageKernel, 6, sizeof(float), &p_Hue2[0]);
error |= clSetKernelArg(HueStageKernel, 7, sizeof(float), &p_Hue2[1]);
error |= clSetKernelArg(HueStageKernel, 8, sizeof(float), &p_Hue2[2]);
error |= clSetKernelArg(HueStageKernel, 9, sizeof(float), &p_Hue2[3]);
error |= clSetKernelArg(HueStageKernel, 10, sizeof(float), &p_Hue2[4]);
error |= clSetKernelArg(HueStageKernel, 11, sizeof(float), &p_Hue3[5]);
error |= clSetKernelArg(HueStageKernel, 12, sizeof(float), &p_Hue3[6]);
error |= clSetKernelArg(HueStageKernel, 13, sizeof(int), &green);
clEnqueueNDRangeKernel(cmdQ, HueStageKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if (p_Blur[2] > 0.0f && p_Switch[3] == 1) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[2]);
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
error |= clSetKernelArg(HueStageKernel, 4, sizeof(int), &p_Switch[3]);
error |= clSetKernelArg(HueStageKernel, 5, sizeof(int), &green);
error |= clSetKernelArg(HueStageKernel, 6, sizeof(float), &p_Hue3[0]);
error |= clSetKernelArg(HueStageKernel, 7, sizeof(float), &p_Hue3[1]);
error |= clSetKernelArg(HueStageKernel, 8, sizeof(float), &p_Hue3[2]);
error |= clSetKernelArg(HueStageKernel, 9, sizeof(float), &p_Hue3[3]);
error |= clSetKernelArg(HueStageKernel, 10, sizeof(float), &p_Hue3[4]);
error |= clSetKernelArg(HueStageKernel, 11, sizeof(float), &p_Sat[3]);
error |= clSetKernelArg(HueStageKernel, 12, sizeof(float), &zero);
error |= clSetKernelArg(HueStageKernel, 13, sizeof(int), &blue);
clEnqueueNDRangeKernel(cmdQ, HueStageKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
if (p_Blur[3] > 0.0f) {
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 4, sizeof(float), &p_Blur[3]);
error |= clSetKernelArg(Gausfilter, 5, sizeof(int), &alpha);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeA, gausLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 5, sizeof(int), &alpha);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeA, TransLocalWorkSize, 0, NULL, NULL);
error |= clSetKernelArg(Gausfilter, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Gausfilter, 3, sizeof(int), &p_Width);
error |= clSetKernelArg(Transpose, 2, sizeof(int), &p_Height);
error |= clSetKernelArg(Transpose, 3, sizeof(int), &p_Width);
clEnqueueNDRangeKernel(cmdQ, Gausfilter, 1, NULL, gausGlobalWorkSizeB, gausLocalWorkSize, 0, NULL, NULL);
clEnqueueNDRangeKernel(cmdQ, Transpose, 2, NULL, TransGlobalWorkSizeB, TransLocalWorkSize, 0, NULL, NULL);
}
clEnqueueNDRangeKernel(cmdQ, FinalStageKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

clReleaseMemObject(tempBuffer);
}