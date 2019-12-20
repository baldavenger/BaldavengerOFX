#ifndef _HUECONVERGECUDAKERNEL_CUH_
#define _HUECONVERGECUDAKERNEL_CUH_

#ifndef M_PI
#define M_PI	3.14159265358979323846264338327950288
#endif

#define BLOCK_DIM 32

#define RGB_2_YAB_MAT  make_mat3(make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f), make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4))

__constant__ float e = 2.718281828459045f;
__constant__ float sqrt3over4 = 0.433012701892219f;

int iDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void d_transpose(float *idata, float *odata, int width, int height)
{
__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
if ((xIndex < width) && (yIndex < height))
{
unsigned int index_in = (yIndex * width + xIndex);
block[threadIdx.y][threadIdx.x] = idata[index_in];
}
__syncthreads();
xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
if ((xIndex < height) && (yIndex < width))
{
unsigned int index_out = (yIndex * height + xIndex) * 4;
odata[index_out] = block[threadIdx.x][threadIdx.y];
}}

__global__ void d_recursiveGaussian(float *id, float *od, int w, int h, float blur)
{
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
const float nsigma = blur < 0.1f ? 0.1f : blur;
float alpha = 1.695f / nsigma;
float ema = exp(-alpha);
float ema2 = exp(-2.0f * alpha);
float b1 = -2.0f * ema;
float b2 = ema2;
float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, coefp = 0.0f, coefn = 0.0f;
const float k = (1.0f - ema) * (1.0f - ema)/(1.0f + 2.0f * alpha * ema - ema2);
a0 = k;
a1 = k * (alpha - 1.0f) * ema;
a2 = k * (alpha + 1.0f) * ema;
a3 = -k * ema2;
coefp = (a0 + a1) / (1.0f + b1 + b2);
coefn = (a2 + a3) / (1.0f + b1 + b2);
if (x >= w) return;
id += x * 4;
od += x;
float xp, yp, yb;
xp = *id;
yb = coefp * xp;
yp = yb;
for (int y = 0; y < h; y++)
{
float xc = *id;
float yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb;
*od = yc;
id += w * 4;
od += w;
xp = xc;
yb = yp;
yp = yc;
}
id -= w * 4;
od -= w;
float xn, xa, yn, ya;
xn = xa = *id;
yn = coefn * xn;
ya = yn;
for (int y = h - 1; y >= 0; y--)
{
float xc = *id;
float yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya;
xa = xn;
xn = xc;
ya = yn;
yn = yc;
*od = *od + yc;
id -= w * 4;
od -= w;
}
}

__device__ inline float clamp( float in, float clampMin, float clampMax) {
return fmaxf( clampMin, fminf(in, clampMax));
}

typedef struct {
float3 c0, c1, c2;
} mat3;

__device__ inline mat3 make_mat3(float3 A, float3 B, float3 C) {
mat3 D;
D.c0 = A; D.c1 = B; D.c2 = C;
return D;
}

__device__ inline mat3 mult_f_f33 (float f, mat3 A) {
float r[3][3];
float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
{A.c1.x, A.c1.y, A.c1.z},
{A.c2.x, A.c2.y, A.c2.z}};
for( int i = 0; i < 3; ++i ) {
for( int j = 0; j < 3; ++j ) {
r[i][j] = f * a[i][j];
}}
mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), 
make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
return R;
}

__device__ inline float3 mult_f3_f33 (float3 X, mat3 A) {
float r[3];
float x[3] = {X.x, X.y, X.z};
float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
{A.c1.x, A.c1.y, A.c1.z},
{A.c2.x, A.c2.y, A.c2.z}};
for( int i = 0; i < 3; ++i) {
r[i] = 0.0f;
for( int j = 0; j < 3; ++j) {
r[i] = r[i] + x[j] * a[j][i];
}}
return make_float3(r[0], r[1], r[2]);
}

__device__ inline mat3 invert_f33 (mat3 A) {
mat3 R;
float result[3][3];
float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
{A.c1.x, A.c1.y, A.c1.z},
{A.c2.x, A.c2.y, A.c2.z}};
float det =   a[0][0] * a[1][1] * a[2][2]
+ a[0][1] * a[1][2] * a[2][0]
+ a[0][2] * a[1][0] * a[2][1]
- a[2][0] * a[1][1] * a[0][2]
- a[2][1] * a[1][2] * a[0][0]
- a[2][2] * a[1][0] * a[0][1];
if( det != 0.0f ) {
result[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1];
result[0][1] = a[2][1] * a[0][2] - a[2][2] * a[0][1];
result[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1];
result[1][0] = a[2][0] * a[1][2] - a[1][0] * a[2][2];
result[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2];
result[1][2] = a[1][0] * a[0][2] - a[0][0] * a[1][2];
result[2][0] = a[1][0] * a[2][1] - a[2][0] * a[1][1];
result[2][1] = a[2][0] * a[0][1] - a[0][0] * a[2][1];
result[2][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1];
R = make_mat3(make_float3(result[0][0], result[0][1], result[0][2]), 
make_float3(result[1][0], result[1][1], result[1][2]), make_float3(result[2][0], result[2][1], result[2][2]));
return mult_f_f33( 1.0f / det, R);
}
R = make_mat3(make_float3(1.0f, 0.0f, 0.0f), 
make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
return R;
}

__device__ inline float Luma(float R, float G, float B, int L) {
float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f;
float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f;
float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f;
float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f;
float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f;
float lumaAvg = (R + G + B) / 3.0f;
float lumaMax = fmaxf(fmaxf(R, G), B);
float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : 
L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax;
return Lu;
}

__device__ inline float3 Sigmoid( float3 In, float peak, float curve, float pivot, float offset) {
float3 out;
out.x = peak / (1.0f + powf(e, (-8.9f * curve) * (In.x - pivot))) + offset;
out.y = peak / (1.0f + powf(e, (-8.9f * curve) * (In.y - pivot))) + offset;
out.z = peak / (1.0f + powf(e, (-8.9f * curve) * (In.z - pivot))) + offset;
return out;
}

__device__ inline float3 HSV_to_RGB(float3 HSV) {
float3 RGB;
if (HSV.y == 0.0f) {
RGB.x = RGB.y = RGB.z = HSV.z;
} else {
HSV.x *= 6.0f;
int i = floor(HSV.x);
float f = HSV.x - i;
i = (i >= 0) ? (i % 6) : (i % 6) + 6;
float p = HSV.z * (1.0f - HSV.y);
float q = HSV.z * (1.0f - HSV.y * f);
float t = HSV.z * (1.0f - HSV.y * (1.0f - f));
RGB.x = i == 0 ? HSV.z : i == 1 ? q : i == 2 ? p : i == 3 ? p : i == 4 ? t : HSV.z;
RGB.y = i == 0 ? t : i == 1 ? HSV.z : i == 2 ? HSV.z : i == 3 ? q : i == 4 ? p : p;
RGB.z = i == 0 ? p : i == 1 ? p : i == 2 ? t : i == 3 ? HSV.z : i == 4 ? HSV.z : q;
}
return RGB;
}

__device__ inline float3 Hue_Chart( int width, int height, int X, int Y) {
float3 HSV;
HSV.x = (float)X / ((float)width - 1.0f);
HSV.y = (float)Y / ((float)height - 1.0f);
HSV.y = powf(HSV.y, 0.4f);
HSV.z = 0.75f;
float3 RGB = HSV_to_RGB(HSV);
return RGB;
}

__device__ inline float3 saturation_f3( float3 In, float luma, float sat) {
float3 out;
out.x = (1.0f - sat) * luma + sat * In.x;
out.y = (1.0f - sat) * luma + sat * In.y;
out.z = (1.0f - sat) * luma + sat * In.z;
return out;
}

__device__ inline float Sat_Soft_Clip(float S, float softclip) {
softclip *= 0.3f;
float ss = S > softclip ? (-1.0f / ((S - softclip) / (1.0f - softclip) + 1.0f) + 1.0f) * (1.0f - softclip) + softclip : S;
return ss;
}

__device__ inline float Limiter(float val, float limiter) {
float Alpha = limiter > 1.0f ? val + (1.0f - limiter) * (1.0f - val) : limiter >= 0.0f ? (val >= limiter ? 1.0f : 
val / limiter) : limiter < -1.0f ? (1.0f - val) + (limiter + 1.0f) * val : val <= (1.0f + limiter) ? 1.0f : 
(1.0 - val) / (1.0f - (limiter + 1.0f));
Alpha = clamp(Alpha, 0.0f, 1.0f);
return Alpha;
}

__device__ inline float interpolate1D( float2 table[], float p, int Size)
{
if( p <= table[0].x ) return table[0].y;
if( p >= table[Size - 1].x ) return table[Size - 1].y;
for( int i = 0; i < Size - 1; ++i ){
if( table[i].x <= p && p < table[i + 1].x ){
float s = (p - table[i].x) / (table[i + 1].x - table[i].x);
return table[i].y * ( 1.0f - s ) + table[i+1].y * s;}}
return 0.0f;
}

__device__ inline float cubic_basis_shaper ( float x,  float w) {
float M[4][4] = { { -1.0f/6.0f,  1.0f/2.0f, -1.0f/2.0f,  1.0f/6.0f },
{  1.0f/2.0f, -1.0f,  1.0f/2.0f,  0.0f },
{ -1.0f/2.0f,  0.0f,  1.0f/2.0f,  0.0f },
{  1.0f/6.0f,  2.0f/3.0f,  1.0f/6.0f,  0.0f } };
float knots[5] = { -w/2.0f, -w/4.0f, 0.0f, w/4.0f, w/2.0f };
float y = 0.0f;
if ((x > knots[0]) && (x < knots[4])) {  
float knot_coord = (x - knots[0]) * 4.0f/w;  
int j = knot_coord;
float t = knot_coord - j;
float monomials[4] = { t*t*t, t*t, t, 1.0f };
if ( j == 3) {
y = monomials[0] * M[0][0] + monomials[1] * M[1][0] + 
monomials[2] * M[2][0] + monomials[3] * M[3][0];
} else if ( j == 2) {
y = monomials[0] * M[0][1] + monomials[1] * M[1][1] + 
monomials[2] * M[2][1] + monomials[3] * M[3][1];
} else if ( j == 1) {
y = monomials[0] * M[0][2] + monomials[1] * M[1][2] + 
monomials[2] * M[2][2] + monomials[3] * M[3][2];
} else if ( j == 0) {
y = monomials[0] * M[0][3] + monomials[1] * M[1][3] + 
monomials[2] * M[2][3] + monomials[3] * M[3][3];
} else {
y = 0.0f;
}}
return y * 3.0f/2.0f;
}

__device__ inline float center_hue( float hue, float centerH) {
float hueCentered = hue - centerH;
if (hueCentered < -180.0f) hueCentered = hueCentered + 360.0f;
else if (hueCentered > 180.0f) hueCentered = hueCentered - 360.0f;
return hueCentered;
}

__device__ inline float uncenter_hue( float hueCentered, float centerH) {
float hue = hueCentered + centerH;
if (hue < 0.0f) hue = hue + 360.0f;
else if (hue > 360.0f) hue = hue - 360.0f;
return hue;
}

__device__ inline float3 rgb_2_yab( float3 rgb) {
float3 yab = mult_f3_f33( rgb, RGB_2_YAB_MAT);
return yab;
}

__device__ inline float3 yab_2_rgb( float3 yab) {
float3 rgb = mult_f3_f33( yab, invert_f33(RGB_2_YAB_MAT));
return rgb;
}

__device__ inline float3 yab_2_ych(float3 yab) {
float3 ych = yab;
float yo = yab.y * yab.y + yab.z * yab.z;
ych.y = sqrtf(yo);
ych.z = atan2f(yab.z, yab.y) * (180.0f / M_PI);
if (ych.z < 0.0f) ych.z += 360.0f;
return ych;
}

__device__ inline float3 ych_2_yab( float3 ych )  {
float3 yab;
yab.x = ych.x;
float h = ych.z * (M_PI / 180.0f);
yab.y = ych.y * cosf(h);
yab.z = ych.y * sinf(h);
return yab;
}

__device__ inline float3 rgb_2_ych( float3 rgb) {
return yab_2_ych( rgb_2_yab( rgb));
}

__device__ inline float3 ych_2_rgb( float3 ych) {
return yab_2_rgb( ych_2_yab( ych));
}

__device__ inline float3 modify_hue( float3 ych, float Hue, float Range, 
float Shift, float Converge, float SatScale) {
float3 new_ych = ych;
float centeredHue = center_hue( ych.z, Hue);
float f_H = cubic_basis_shaper( centeredHue, Range);
float old_hue = centeredHue;
float new_hue = centeredHue + Shift;
float2 table[2] = { {0.0f, old_hue}, {1.0f, new_hue} };
float blended_hue = interpolate1D( table, f_H, 2);
if (f_H > 0.0f)  {
new_ych.z = uncenter_hue(blended_hue, Hue);
if (ych.y > 0.0f) {
new_ych.y = ych.y * (f_H * (SatScale - 1.0f) + 1.0f);
}}
float h, H, hue, range;
h = new_ych.z / 360.0f;
H = h;
hue = (Hue + Shift) / 360.0f;
hue = hue > 1.0f ? hue - 1.0f : hue < 0.0f ? hue + 1.0f : hue;
range = Range / 720.0f;
h = h - (hue - 0.5f) < 0.0f ? h - (hue - 0.5f) + 1.0f : h - (hue - 0.5f) >
1.0f ? h - (hue - 0.5f) - 1.0f : h - (hue - 0.5f);
H = h > 0.5f - range && h < 0.5f ? (1.0f - powf(1.0f - (h - (0.5f - range)) *
(1.0f/range), Converge)) * range + (0.5f - range) + (hue - 0.5f) : h > 0.5f && h < 0.5f + 
range ? powf((h - 0.5f) * (1.0f/range), Converge) * range + 0.5f + (hue - 0.5f) : H;
new_ych.z = H * 360.0f;
return new_ych;
}

__device__ inline float median(float p_Table[], int m) {
float temp;
int i, j;
for(i = 0; i < m - 1; i++) {
for(j = i + 1; j < m; j++) {
if(p_Table[j] < p_Table[i]) {
temp = p_Table[i];
p_Table[i] = p_Table[j];
p_Table[j] = temp; }}}
return p_Table[(m - 1) / 2];
}

#endif // #ifndef _HUECONVERGECUDAKERNEL_CUH_