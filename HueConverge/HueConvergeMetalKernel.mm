#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

#define BLOCK_DIM	32
#define bufferLength	(p_Width * p_Height * sizeof(float))

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"typedef struct { \n" \
"float3 c0, c1, c2; \n" \
"} mat3; \n" \
"float3 make_float3( float A, float B, float C) { \n" \
"float3 out; \n" \
"out.x = A; out.y = B; out.z = C; \n" \
"return out; \n" \
"} \n" \
"constant float pie = 3.1415926535898f; \n" \
"constant float eu = 2.718281828459045f; \n" \
"constant float sqrt3over4 = 0.433012701892219f; \n" \
"mat3 make_mat3(float3 A, float3 B, float3 C) { \n" \
"mat3 D; \n" \
"D.c0 = A; D.c1 = B; D.c2 = C; \n" \
"return D; \n" \
"} \n" \
"mat3 mult_f_f33 (float f, mat3 A) { \n" \
"float r[3][3]; \n" \
"float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, \n" \
"{A.c1.x, A.c1.y, A.c1.z}, \n" \
"{A.c2.x, A.c2.y, A.c2.z}}; \n" \
"for( int i = 0; i < 3; ++i ) { \n" \
"for( int j = 0; j < 3; ++j ) { \n" \
"r[i][j] = f * a[i][j]; \n" \
"}} \n" \
"mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]),  \n" \
"make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2])); \n" \
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
"return make_float3(r[0], r[1], r[2]); \n" \
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
"R = make_mat3(make_float3(result[0][0], result[0][1], result[0][2]),  \n" \
"make_float3(result[1][0], result[1][1], result[1][2]), make_float3(result[2][0], result[2][1], result[2][2])); \n" \
"return mult_f_f33( 1.0f / det, R); \n" \
"} \n" \
"R = make_mat3(make_float3(1.0f, 0.0f, 0.0f),  \n" \
"make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f)); \n" \
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
"float3 RGB = make_float3(0.0f, 0.0f, 0.0f); \n" \
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
"float3 HSV = make_float3(0.0f, 0.0f, 0.0f); \n" \
"HSV.x = (float)X / ((float)width - 1.0f); \n" \
"HSV.y = (float)Y / ((float)height - 1.0f); \n" \
"HSV.y = pow(HSV.y, 0.4f); \n" \
"HSV.z = 0.75f; \n" \
"float3 RGB = HSV_to_RGB(HSV); \n" \
"return RGB; \n" \
"} \n" \
"float3 saturation_f3( float3 In, float luma, float sat) { \n" \
"float3 out = make_float3(0.0f, 0.0f, 0.0f); \n" \
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
"mat3 RGB_2_YAB_MAT = make_mat3(make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f),  \n" \
"make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4)); \n" \
"float3 yab = mult_f3_f33( rgb, RGB_2_YAB_MAT); \n" \
"return yab; \n" \
"} \n" \
"float3 yab_2_rgb( float3 yab) { \n" \
"mat3 RGB_2_YAB_MAT = make_mat3(make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f),  \n" \
"make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4)); \n" \
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
"kernel void k_logStageKernel( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], constant int& p_SwitchLog [[buffer (4)]],  \n" \
"constant int& p_SwitchHue [[buffer (5)]], constant float& p_LogA [[buffer (6)]], constant float& p_LogB [[buffer (7)]],  \n" \
"constant float& p_LogC [[buffer (8)]], constant float& p_LogD [[buffer (9)]], constant float& p_SatA [[buffer (10)]],  \n" \
"constant float& p_SatB [[buffer (11)]], constant float& p_LumaLimit [[buffer (12)]], constant float& p_SatLimit [[buffer (13)]],  \n" \
"constant int& p_Math [[buffer (14)]], constant int& p_Chart [[buffer (15)]], uint2 id [[ thread_position_in_grid ]]) {	 \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 RGB = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"if(p_Chart == 1) \n" \
"RGB = Hue_Chart( p_Width, p_Height, id.x, id.y); \n" \
"if(p_SwitchLog == 1) \n" \
"RGB = Sigmoid( RGB, p_LogA, p_LogB, p_LogC, p_LogD); \n" \
"float luma = Luma(RGB.x, RGB.y, RGB.z, p_Math); \n" \
"if(p_SatA != 1.0f) { \n" \
"float minluma = fmin(RGB.x, fmin(RGB.y, RGB.z)); \n" \
"RGB = saturation_f3( RGB, minluma, p_SatA); \n" \
"} \n" \
"if(p_SatB != 1.0f) { \n" \
"float maxluma = fmax(RGB.x, fmax(RGB.y, RGB.z)); \n" \
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
"kernel void k_hueMedian( device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (20)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant int& p_Median [[buffer (18)]], uint2 id [[ thread_position_in_grid ]]) {	 \n" \
"int length = 3; \n" \
"int area = 9; \n" \
"int offset = 1; \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"if(p_Median == 1) { \n" \
"float TableA[9]; \n" \
"for (int i = 0; i < area; ++i) { \n" \
"int xx = (i >= length ? fmod(i, (float)length) : i) - offset; \n" \
"int yy = floor(i / (float)length) - offset; \n" \
"TableA[i] = p_Input[(((id.y + yy) * p_Width + id.x + xx) * 4) + 2]; \n" \
"} \n" \
"p_Output[id.y * p_Width + id.x] = median(TableA, area); \n" \
"} \n" \
"if(p_Median == 2) { \n" \
"length = 5; \n" \
"area = 25; \n" \
"offset = 2; \n" \
"float TableB[25]; \n" \
"for (int i = 0; i < area; ++i) { \n" \
"int xx = (i >= length ? fmod(i, (float)length) : i) - offset; \n" \
"int yy = floor(i / (float)length) - offset; \n" \
"TableB[i] = p_Input[(((id.y + yy) * p_Width + id.x + xx) * 4) + 2]; \n" \
"} \n" \
"p_Output[id.y * p_Width + id.x] = median(TableB, area); \n" \
"}}} \n" \
"kernel void k_tempReturn(device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (20)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if ((id.x < p_Width) && (id.y < p_Height)) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Input[index + 2] = p_Output[id.y * p_Width + id.x]; \n" \
"}} \n" \
"kernel void k_hueStageKernel(device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (0)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], constant int& p_SwitchHue [[buffer (4)]],  \n" \
"constant int& p_SwitchHue1 [[buffer (5)]], constant float& p_Hue1 [[buffer (6)]], constant float& p_Hue2 [[buffer (7)]],  \n" \
"constant float& p_Hue3 [[buffer (8)]], constant float& p_Hue4 [[buffer (9)]], constant float& p_Hue5 [[buffer (10)]],  \n" \
"constant float& p_LumaLimit [[buffer (11)]], constant float& p_SatLimit [[buffer (12)]], constant int& ALPHA [[buffer (14)]],  \n" \
"constant int& p_Isolate [[buffer (15)]], uint2 id [[ thread_position_in_grid ]]) {	 \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 ych = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float lumaAlpha = 1.0f; \n" \
"float satAlpha = 1.0f; \n" \
"if(p_LumaLimit != 0.0f && p_SwitchHue1 == 1) \n" \
"lumaAlpha = Limiter(p_Input[index + 3], p_LumaLimit); \n" \
"if(p_SatLimit != 0.0f && p_SwitchHue1 == 1) \n" \
"satAlpha = Limiter(ych.y * 10.0f, p_SatLimit); \n" \
"if(p_Isolate == ALPHA + 1) { \n" \
"float offset = 180.0f - p_Hue1; \n" \
"float outside = ych.z + offset; \n" \
"outside = outside > 360.0f ? outside - 360.0f : outside < 0.0f ? outside + 360.0f : outside; \n" \
"if(outside > 180.0f + (p_Hue2 / 2) || outside < 180.0f - (p_Hue2 / 2)) \n" \
"ych.y = 0.0f; \n" \
"} \n" \
"float3 new_ych = modify_hue( ych, p_Hue1, p_Hue2, p_Hue3, p_Hue4, p_Hue5); \n" \
"float alpha = p_Output[index + ALPHA]; \n" \
"if(alpha != 1.0f) { \n" \
"float3 RGB = ych_2_rgb( ych); \n" \
"float3 new_RGB = ych_2_rgb( new_ych); \n" \
"new_RGB.x = new_RGB.x * alpha + RGB.x * (1.0f - alpha); \n" \
"new_RGB.y = new_RGB.y * alpha + RGB.y * (1.0f - alpha); \n" \
"new_RGB.z = new_RGB.z * alpha + RGB.z * (1.0f - alpha); \n" \
"new_ych = rgb_2_ych( new_RGB); \n" \
"} \n" \
"p_Input[index] = p_SwitchHue == 1 ? new_ych.x : p_Input[index]; \n" \
"p_Input[index + 1] = p_SwitchHue == 1 ? new_ych.y : p_Input[index + 1]; \n" \
"p_Input[index + 2] = p_SwitchHue == 1 ? new_ych.z : p_Input[index + 2]; \n" \
"p_Output[index + ALPHA + 1] = 1.0f * lumaAlpha * satAlpha; \n" \
"}} \n" \
"kernel void k_finalStageKernel(device float* p_Input [[buffer (1)]], device float* p_Output [[buffer (0)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], constant float& p_SatSoft [[buffer (12)]],  \n" \
"constant int& p_Display [[buffer (15)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if( (id.x < p_Width) && (id.y < p_Height) ) \n" \
"{ \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 ych = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]); \n" \
"float alpha = p_Output[index + 3]; \n" \
"if(p_SatSoft != 1.0f) \n" \
"{ \n" \
"float soft = Sat_Soft_Clip(ych.y, p_SatSoft); \n" \
"ych.y = soft * alpha + ych.y * (1.0f - alpha); \n" \
"} \n" \
"float luma = p_Input[index + 3]; \n" \
"float3 rgb = ych_2_rgb( ych); \n" \
"if(p_Display != 0) \n" \
"{ \n" \
"float displayAlpha = p_Display == 1 ? p_Output[index] : p_Display == 2 ?  \n" \
"p_Output[index + 1] : p_Display == 3 ? p_Output[index + 2] :  p_Display == 4 ?  \n" \
"alpha : p_Display == 5 ? (ych.z / 360.0f) : p_Display == 6 ? clamp(ych.y * 10.0f, 0.0f, 1.0f) : luma;   \n" \
"p_Input[index] = displayAlpha; \n" \
"p_Input[index + 1] = displayAlpha; \n" \
"p_Input[index + 2] = displayAlpha; \n" \
"p_Input[index + 3] = 1.0f; \n" \
"} else { \n" \
"p_Input[index] = rgb.x; \n" \
"p_Input[index + 1] = rgb.y; \n" \
"p_Input[index + 2] = rgb.z; \n" \
"p_Input[index + 3] = 1.0f; \n" \
"}}} \n" \
"kernel void k_transpose( device float* p_Input [[buffer (20)]], device float* p_Output [[buffer (0)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant int& ch_out [[buffer (16)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
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
"kernel void k_gaussian( device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (20)]],  \n" \
"constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], constant int& ch_in [[buffer (16)]],  \n" \
"constant float& blur [[buffer (17)]], uint2 threadIdx [[ thread_position_in_threadgroup ]], \n" \
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
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, int* p_Switch, float* p_Log, float* p_Sat, 
float *p_Hue1, float *p_Hue2, float *p_Hue3, int p_Display, float* p_Blur, int p_Math, int p_HueMedian, int p_Isolate)
{
int red = 0;
int green = 1;
int blue = 2;
int alpha = 3;
float zero = 0.0f;

const char* logStageKernel		= "k_logStageKernel";
const char* hueMedian			= "k_hueMedian";
const char* tempReturn			= "k_tempReturn";
const char* gaussian			= "k_gaussian";
const char* transpose			= "k_transpose";
const char* hueStageKernel		= "k_hueStageKernel";
const char* finalStageKernel	= "k_finalStageKernel";

id<MTLCommandQueue>				queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>					device = queue.device;
id<MTLLibrary>					metalLibrary;
id<MTLFunction>					kernelFunction;
id<MTLComputePipelineState>		pipelineState;
id<MTLBuffer>					tempBuffer;
id<MTLComputePipelineState>    _logStageKernel;
id<MTLComputePipelineState>    _hueMedian;
id<MTLComputePipelineState>    _tempReturn;
id<MTLComputePipelineState>    _gaussian;
id<MTLComputePipelineState>    _transpose;
id<MTLComputePipelineState>    _hueStageKernel;
id<MTLComputePipelineState>    _finalStageKernel;

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
metalLibrary				=	[device newLibraryWithSource:@(kernelSource) options:options error:&err];
[options release];

tempBuffer 					= 	[device newBufferWithLength:bufferLength options:0];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:logStageKernel] ];

_logStageKernel				=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:hueMedian] ];

_hueMedian					=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:tempReturn] ];

_tempReturn					=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:gaussian] ];

_gaussian					=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:transpose] ];

_transpose					=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:hueStageKernel] ];

_hueStageKernel				=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

kernelFunction  			=	[metalLibrary newFunctionWithName:[NSString stringWithUTF8String:finalStageKernel] ];

_finalStageKernel			=	[device newComputePipelineStateWithFunction:kernelFunction error:&err];

[metalLibrary release];
[kernelFunction release];

id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
commandBuffer.label = [NSString stringWithFormat:@"RunMetalKernel"];
id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
[computeEncoder setComputePipelineState:_logStageKernel];
int exeWidth = [_logStageKernel threadExecutionWidth];

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
[computeEncoder setBytes:&p_Switch[0] length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Switch[1] length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Log[0] length:sizeof(float) atIndex: 6];
[computeEncoder setBytes:&p_Log[1] length:sizeof(float) atIndex: 7];
[computeEncoder setBytes:&p_Log[2] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Log[3] length:sizeof(float) atIndex: 9];
[computeEncoder setBytes:&p_Sat[0] length:sizeof(float) atIndex: 10];
[computeEncoder setBytes:&p_Sat[1] length:sizeof(float) atIndex: 11];
[computeEncoder setBytes:&p_Hue1[5] length:sizeof(float) atIndex: 12];
[computeEncoder setBytes:&p_Hue1[6] length:sizeof(float) atIndex: 13];
[computeEncoder setBytes:&p_Math length:sizeof(int) atIndex: 14];
[computeEncoder setBytes:&p_Switch[4] length:sizeof(int) atIndex: 15];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder setBuffer:tempBuffer offset: 0 atIndex: 20];
[computeEncoder setBytes:&p_HueMedian length:sizeof(int) atIndex: 18];

if(p_HueMedian == 1 || p_HueMedian == 2)
{
[computeEncoder setComputePipelineState:_hueMedian];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
[computeEncoder setComputePipelineState:_tempReturn];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_Blur[0] > 0.0f && p_Switch[1] == 1) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 16];
[computeEncoder setBytes:&p_Blur[0] length:sizeof(float) atIndex: 17];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
}

[computeEncoder setComputePipelineState:_hueStageKernel];
[computeEncoder setBytes:&p_Switch[1] length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Switch[2] length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Hue1[0] length:sizeof(float) atIndex: 6];
[computeEncoder setBytes:&p_Hue1[1] length:sizeof(float) atIndex: 7];
[computeEncoder setBytes:&p_Hue1[2] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Hue1[3] length:sizeof(float) atIndex: 9];
[computeEncoder setBytes:&p_Hue1[4] length:sizeof(float) atIndex: 10];
[computeEncoder setBytes:&p_Hue2[5] length:sizeof(float) atIndex: 11];
[computeEncoder setBytes:&p_Hue2[6] length:sizeof(float) atIndex: 12];
[computeEncoder setBytes:&red length:sizeof(int) atIndex: 14];
[computeEncoder setBytes:&p_Isolate length:sizeof(int) atIndex: 15];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_Blur[1] > 0.0f && p_Switch[2] == 1) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&green length:sizeof(int) atIndex: 16];
[computeEncoder setBytes:&p_Blur[1] length:sizeof(float) atIndex: 17];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
}

[computeEncoder setComputePipelineState:_hueStageKernel];
[computeEncoder setBytes:&p_Switch[2] length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&p_Switch[3] length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Hue2[0] length:sizeof(float) atIndex: 6];
[computeEncoder setBytes:&p_Hue2[1] length:sizeof(float) atIndex: 7];
[computeEncoder setBytes:&p_Hue2[2] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Hue2[3] length:sizeof(float) atIndex: 9];
[computeEncoder setBytes:&p_Hue2[4] length:sizeof(float) atIndex: 10];
[computeEncoder setBytes:&p_Hue3[5] length:sizeof(float) atIndex: 11];
[computeEncoder setBytes:&p_Hue3[6] length:sizeof(float) atIndex: 12];
[computeEncoder setBytes:&green length:sizeof(int) atIndex: 14];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_Blur[2] > 0.0f && p_Switch[3] == 1) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 16];
[computeEncoder setBytes:&p_Blur[2] length:sizeof(float) atIndex: 17];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
}

[computeEncoder setComputePipelineState:_hueStageKernel];
[computeEncoder setBytes:&p_Switch[3] length:sizeof(int) atIndex: 4];
[computeEncoder setBytes:&green length:sizeof(int) atIndex: 5];
[computeEncoder setBytes:&p_Hue3[0] length:sizeof(float) atIndex: 6];
[computeEncoder setBytes:&p_Hue3[1] length:sizeof(float) atIndex: 7];
[computeEncoder setBytes:&p_Hue3[2] length:sizeof(float) atIndex: 8];
[computeEncoder setBytes:&p_Hue3[3] length:sizeof(float) atIndex: 9];
[computeEncoder setBytes:&p_Hue3[4] length:sizeof(float) atIndex: 10];
[computeEncoder setBytes:&p_Sat[3] length:sizeof(float) atIndex: 11];
[computeEncoder setBytes:&zero length:sizeof(float) atIndex: 12];
[computeEncoder setBytes:&blue length:sizeof(int) atIndex: 14];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_Blur[3] > 0.0f) {
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&alpha length:sizeof(int) atIndex: 16];
[computeEncoder setBytes:&p_Blur[3] length:sizeof(float) atIndex: 17];
[computeEncoder dispatchThreadgroups:gausThreadGroupsA threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsA threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setComputePipelineState:_gaussian];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 3];
[computeEncoder dispatchThreadgroups:gausThreadGroupsB threadsPerThreadgroup: gausThreadGroupsCount];
[computeEncoder setComputePipelineState:_transpose];
[computeEncoder dispatchThreadgroups:transThreadGroupsB threadsPerThreadgroup: transThreadGroupsCount];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex: 2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex: 3];
}

[computeEncoder setComputePipelineState:_finalStageKernel];
[computeEncoder setBytes:&p_Sat[2] length:sizeof(float) atIndex: 12];
[computeEncoder setBytes:&p_Display length:sizeof(int) atIndex: 15];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

[computeEncoder endEncoding];
[commandBuffer commit];
[tempBuffer release];
}