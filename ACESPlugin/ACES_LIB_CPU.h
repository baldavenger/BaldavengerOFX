#ifndef __ACES_FUNCTIONS_CPU_H_INCLUDED__
#define __ACES_FUNCTIONS_CPU_H_INCLUDED__

#include <cmath>

typedef struct {
float x, y;
} float2;

typedef struct {
float x, y, z;
} float3;

typedef struct {
float x, y, z, w;
} float4;

typedef struct {
float x, y, z, w, m;
} float5;

typedef struct {
float2 c0, c1;
} mat2;

typedef struct {
float3 c0, c1, c2;
} mat3;

typedef struct {
float2 red; float2 green; float2 blue; float2 white;
} Chromaticities;

typedef struct {
float x; float y;
} SplineMapPoint;

typedef struct {
float coefsLow[6]; float coefsHigh[6];
SplineMapPoint minPoint; SplineMapPoint midPoint; SplineMapPoint maxPoint;
float slopeLow; float slopeHigh;
} SegmentedSplineParams_c5;

typedef struct {
float coefsLow[10]; float coefsHigh[10];
SplineMapPoint minPoint; SplineMapPoint midPoint; SplineMapPoint maxPoint;
float slopeLow; float slopeHigh;
} SegmentedSplineParams_c9;

typedef struct {
float x; float y; float slope;
} TsPoint;

typedef struct {
TsPoint Min; TsPoint Mid; TsPoint Max;
float coefsLow[6]; float coefsHigh[6];
} TsParams;

#define REF_PT				((7120.0f - 1520.0f) / 8000.0f * (100.0f / 55.0f) - log10f(0.18f)) * 1.0f
const mat3 MM = { {0.5f, -1.0f, 0.5f}, {-1.0f, 1.0f, 0.5f}, {0.5f, 0.0f, 0.0f} };
const mat3 M1 = { {0.5f, -1.0f, 0.5f}, {-1.0f, 1.0f, 0.5f}, {0.5f, 0.0f, 0.0f} };
const float TINY = 1e-10f;
const float DIM_SURROUND_GAMMA = 0.9811f;
const float ODT_SAT_FACTOR = 0.93f;
const float MIN_STOP_SDR = -6.5f;
const float MAX_STOP_SDR = 6.5f;
const float MIN_STOP_RRT = -15.0f;
const float MAX_STOP_RRT = 18.0f;
const float MIN_LUM_SDR = 0.02f;
const float MAX_LUM_SDR = 48.0f;
const float MIN_LUM_RRT = 0.0001f;
const float MAX_LUM_RRT = 10000.0f;
const float RRT_GLOW_GAIN = 0.05f;
const float RRT_GLOW_MID = 0.08f;
const float RRT_RED_SCALE = 0.82f;
const float RRT_RED_PIVOT = 0.03f;
const float RRT_RED_HUE = 0.0f;
const float RRT_RED_WIDTH = 135.0f;
const float RRT_SAT_FACTOR = 0.96f;
const float X_BRK = 0.0078125f;
const float Y_BRK = 0.155251141552511f;
const float A = 10.5402377416545f;
const float B = 0.0729055341958355f;
const float sqrt3over4 = 0.433012701892219f;
const float pq_m1 = 0.1593017578125f;
const float pq_m2 = 78.84375f;
const float pq_c1 = 0.8359375f;
const float pq_c2 = 18.8515625f;
const float pq_c3 = 18.6875f;
const float pq_C = 10000.0f;
const mat3 CDD_TO_CID = 
{ {0.75573f, 0.05901f, 0.16134f}, {0.22197f, 0.96928f, 0.07406f}, {0.02230f, -0.02829f, 0.76460f} };
const mat3 EXP_TO_ACES = 
{ {0.72286f, 0.11923f, 0.01427f}, {0.12630f, 0.76418f, 0.08213f}, {0.15084f, 0.11659f, 0.90359f} };
const Chromaticities AP0 =
{ {0.7347f, 0.2653f}, {0.0f, 1.0f}, {0.0001f, -0.077f}, {0.32168f, 0.33767f} };
const Chromaticities AP1 =
{ {0.713f, 0.293f}, {0.165f, 0.83f}, {0.128f, 0.044f}, {0.32168f, 0.33767f} };
const Chromaticities REC709_PRI =
{ {0.64f, 0.33f}, {0.3f, 0.6f}, {0.15f, 0.06f}, {0.3127f, 0.329f} };
const Chromaticities P3D60_PRI =
{ {0.68f, 0.32f}, {0.265f, 0.69f}, {0.15f, 0.06f}, {0.32168, 0.33767f} };
const Chromaticities P3D65_PRI =
{ {0.68f, 0.32f}, {0.265f, 0.69f}, {0.15f, 0.06f}, {0.3127f, 0.329f} };
const Chromaticities P3DCI_PRI =
{ {0.68f, 0.32f}, {0.265f, 0.69f}, {0.15f, 0.06f}, {0.314f, 0.351f} };
const Chromaticities ARRI_ALEXA_WG_PRI =
{ {0.684f, 0.313f}, {0.221f, 0.848f}, {0.0861f, -0.102f}, {0.3127f, 0.329f} };
const Chromaticities REC2020_PRI =
{ {0.708f, 0.292f}, {0.17f, 0.797f}, {0.131f, 0.046f}, {0.3127f, 0.329f} };
const Chromaticities RIMMROMM_PRI =
{ {0.7347f, 0.2653f}, {0.1596f, 0.8404f}, {0.0366f, 0.0001f}, {0.3457f, 0.3585f} };
const mat3 CONE_RESP_MAT_BRADFORD =
{ {0.8951f, -0.7502f, 0.0389f}, {0.2664f, 1.7135f, -0.0685f}, {-0.1614f, 0.0367f, 1.0296f} };
const mat3 CONE_RESP_MAT_CAT02 =
{ {0.7328f, -0.7036f, 0.003f}, {0.4296f, 1.6975f, 0.0136f}, {-0.1624f, 0.0061f, 0.9834f} };
const mat3 AP0_2_XYZ_MAT = 
{ {0.9525523959f, 0.3439664498f, 0.0f}, {0.0f, 0.7281660966f, 0.0f}, {0.0000936786f, -0.0721325464f, 1.0088251844f} };
const mat3 XYZ_2_AP0_MAT = 
{ {1.0498110175f, -0.4959030231f, 0.0f}, {0.0f, 1.3733130458f, 0.0f}, {-0.0000974845f, 0.0982400361f, 0.9912520182f} };
const mat3 AP1_2_XYZ_MAT = 
{ {0.6624541811f, 0.2722287168f, -0.0055746495f}, {0.1340042065f, 0.6740817658f, 0.0040607335f}, {0.156187687f, 0.0536895174f, 1.0103391003f} };
const mat3 XYZ_2_AP1_MAT = 
{ {1.6410233797f, -0.6636628587f, 0.0117218943f}, {-0.3248032942f, 1.6153315917f, -0.008284442f}, {-0.2364246952f, 0.0167563477f, 0.9883948585f} };
const mat3 AP0_2_AP1_MAT = 
{ {1.4514393161f, -0.0765537734f, 0.0083161484f}, {-0.2365107469f, 1.1762296998f, -0.0060324498f}, {-0.2149285693f, -0.0996759264f, 0.9977163014f} };
const mat3 AP1_2_AP0_MAT = 
{ {0.6954522414f, 0.0447945634f, -0.0055258826f}, {0.1406786965f, 0.8596711185f, 0.0040252103f}, {0.1638690622f, 0.0955343182f, 1.0015006723f} };
const mat3 D60_2_D65_CAT = 
{ {0.987224f, -0.00759836f, 0.00307257f}, {-0.00611327f, 1.00186f, -0.00509595f}, {0.0159533f, 0.00533002f, 1.08168f} };
const mat3 LMS_2_AP0_MAT = 
{ { 2.2034860017f, -0.5267000086f, -0.0465914122f}, {-1.4028871323f,  1.5838401289f, -0.0457828327f}, { 0.1994183978f, -0.0571107433f, 1.0924829098f} };
const mat3 ICtCp_2_LMSp_MAT = 
{ { 1.0f, 1.0f, 1.0f}, { 0.0086064753f, -0.0086064753f, 0.5600463058f}, { 0.1110335306f, -0.1110335306f, -0.3206319566f} };
const mat3 AP0_2_LMS_MAT = 
{ { 0.5729360781f, 0.1916984459f, 0.0324676922f}, { 0.5052187675f, 0.8013733145f, 0.0551294631f}, {-0.0781710859f, 0.0069006377f, 0.9123015294f} };
const mat3 LMSp_2_ICtCp_MAT = 
{ { 0.5f, 1.6137000085f, 4.378062447f}, { 0.5f, -3.3233961429f, -4.2455397991f}, { 0.0f, 1.7096961344f, -0.1325226479f} };
const mat3 SG3_2_AP0_MAT = 
{ { 0.7529825954f, 0.0217076974f, -0.0094160528f}, { 0.1433702162f, 1.0153188355f, 0.0033704179f}, { 0.1036471884f, -0.0370265329f, 1.0060456349f} };
const mat3 AP0_2_SG3_MAT = 
{ { 1.3316572111f, -0.0280131244f, 0.0125574528f}, {-0.1875611006f, 0.9887375645f, -0.0050679052f}, {-0.1440961106f, 0.0392755599f, 0.9925104526f} };
const mat3 SG3C_2_AP0_MAT = 
{ { 0.6387886672f, -0.0039159061f, -0.0299072021f}, { 0.2723514337f, 1.0880732308f, -0.0264325799f}, { 0.0888598992f, -0.0841573249f, 1.056339782f} };
const mat3 AP0_2_SG3C_MAT = 
{ { 1.5554591070f,  0.0090216145f, 0.0442640666f}, {-0.3932807985f, 0.9185569566f, 0.0118502607f}, {-0.1621783087f, 0.0724214290f, 0.9438856727f} };
const mat3 VSG3_2_AP0_MAT = 
{ { 0.7933297411f, 0.0155810585f, -0.0188647478f}, { 0.0890786256f, 1.0327123069f, 0.0127694121f}, { 0.1175916333f, -0.0482933654f, 1.0060953358f} };
const mat3 VSG3C_2_AP0_MAT = 
{ { 0.6742570921f, -0.0093136061f, -0.0382090673f}, { 0.2205717359f, 1.1059588614f, -0.0179383766f}, { 0.1051711720f, -0.0966452553f, 1.0561474439f} };
const mat3 VGAMUT_2_AP0_MAT = 
{ { 0.724382758f, 0.166748484f, 0.108497411f}, { 0.021354009f, 0.985138372f, -0.006319092f}, {-0.009234278f, -0.00104295f, 1.010272625f} };
const mat3 AWG_2_AP0_MAT = 
{ { 0.6802059161f, 0.0854150695f, 0.0020562648f}, { 0.2361367500f, 1.0174707720f, -0.0625622837f}, { 0.0836574074f, -0.1028858550f, 1.0605062481f} };
const mat3 AP0_2_AWG_MAT = 
{ { 1.5159863829f, -0.1283275799f, -0.0105107561f}, {-0.3613418588f, 1.0193145873f, 0.0608329325f}, {-0.1546444592f, 0.1090123949f, 0.9496764954f} };
const mat3 RWG_2_AP0_MAT = 
{ { 0.7850585442f, 0.0231738066f, -0.0737605663f}, { 0.0838583156f, 1.0878975877f, -0.3145898729f}, { 0.1310821505f, -0.1110709153f, 1.3883506702f} };
const mat3 AP0_2_RWG_MAT = 
{ { 1.2655392805f, -0.0205691227f, 0.0625750095f}, {-0.1352322515f,  0.9431709627f,  0.2065308369f}, {-0.1303056816f, 0.0773976700f, 0.7308939479f} };
const float3 AP1_RGB2Y = {0.2722287168f, 0.6740817658f, 0.0536895174f};

static Chromaticities make_chromaticities( float2 A, float2 B, float2 C, float2 D) {
Chromaticities E;
E.red = A; E.green = B; E.blue = C; E.white = D;
return E;
}

static float2 make_float2( float A, float B) {
float2 C;
C.x = A; C.y = B;
return C;
}

static float3 make_float3( float A, float B, float C) {
float3 D;
D.x = A; D.y = B; D.z = C;
return D;
}

static mat2 make_mat2( float2 A, float2 B) {
mat2 C;
C.c0 = A; C.c1 = B;
return C;
}

static mat3 make_mat3( float3 A, float3 B, float3 C) {
mat3 D;
D.c0 = A; D.c1 = B; D.c2 = C;
return D;
}

static float min_f3( float3 a) {
return fminf( a.x, fminf( a.y, a.z));
}

static float max_f3( float3 a) {
return fmaxf( a.x, fmaxf( a.y, a.z));
}

static float3 max_f3_f( float3 a, float b) {
float3 out;
out.x = fmaxf(a.x, b); out.y = fmaxf(a.y, b); out.z = fmaxf(a.z, b);
return out;
}

static float clip( float v) {
return fminf(v, 1.0f);
}

static float clampf( float A, float mn, float mx) {
return fmaxf( mn, fminf(A, mx));
}

static float3 clip_f3( float3 in) {
float3 out;
out.x = clip( in.x); out.y = clip( in.y); out.z = clip( in.z);
return out;
}

static float3 add_f_f3( float a, float3 b) {
float3 out;
out.x = a + b.x; out.y = a + b.y; out.z = a + b.z;
return out;
}

static float3 pow_f3( float3 a, float b) {
float3 out;
out.x = powf(a.x, b); out.y = powf(a.y, b); out.z = powf(a.z, b);
return out;
}

static float exp10f( float x) {
return powf(10.0f, x);
}

static float3 exp10_f3( float3 a) {
float3 out;
out.x = exp10f(a.x); out.y = exp10f(a.y); out.z = exp10f(a.z);
return out;
}

static float3 log10_f3( float3 a) {
float3 out;
out.x = log10f(a.x); out.y = log10f(a.y); out.z = log10f(a.z);
return out;
}

static float _sign( float x) {
float y;
if (x < 0.0f) y = -1.0f;
else if (x > 0.0f) y = 1.0f;
else y = 0.0f;
return y;
}

static float3 mult_f_f3( float f, float3 x) {
float3 r;
r.x = f * x.x; r.y = f * x.y; r.z = f * x.z;
return r;
}

static float3 add_f3_f3( float3 x, float3 y) {
float3 r;
r.x = x.x + y.x; r.y = x.y + y.y; r.z = x.z + y.z;
return r;
}

static float3 sub_f3_f3( float3 x, float3 y) {
float3 r;
r.x = x.x - y.x; r.y = x.y - y.y; r.z = x.z - y.z;
return r;
}

static float3 cross_f3_f3( float3 x, float3 y) {
float3 r;
r.z = x.x * y.y - x.y * y.x; r.x = x.y * y.z - x.z * y.y; r.y = x.z * y.x - x.x * y.z;
return r;
}

static float3 clamp_f3( float3 A, float mn, float mx) {
float3 out;
out.x = clampf( A.x, mn, mx); out.y = clampf( A.y, mn, mx); out.z = clampf( A.z, mn, mx);
return out;
}

static float dot_f3_f3( float3 x, float3 y) {
return x.x * y.x + x.y * y.y + x.z * y.z;
}

static float length_f3( float3 x) {
return sqrtf( x.x * x.x + x.y * x.y + x.z * x.z );
}

static mat2 transpose_f22( mat2 A) {
mat2 B;
B.c0 = make_float2(A.c0.x, A.c1.x); B.c1 = make_float2(A.c0.y, A.c1.y);
return B;
}

static mat3 transpose_f33( mat3 A) {
float r[3][3];
float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}};
for( int i = 0; i < 3; ++i){
for( int j = 0; j < 3; ++j){
r[i][j] = a[j][i];}}
mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
return R;
}

static mat3 mult_f_f33( float f, mat3 A) {
float r[3][3];
float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}};
for( int i = 0; i < 3; ++i ){
for( int j = 0; j < 3; ++j ){
r[i][j] = f * a[i][j];}}
mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
return R;
}

static float3 mult_f3_f33( float3 X, mat3 A) {
float r[3];
float x[3] = {X.x, X.y, X.z};
float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}};
for( int i = 0; i < 3; ++i){
r[i] = 0.0f;
for( int j = 0; j < 3; ++j){
r[i] = r[i] + x[j] * a[j][i];}}
return make_float3(r[0], r[1], r[2]);
}

static mat3 mult_f33_f33( mat3 A, mat3 B) {
float r[3][3];
float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z},
{A.c1.x, A.c1.y, A.c1.z},
{A.c2.x, A.c2.y, A.c2.z}};
float b[3][3] = {{B.c0.x, B.c0.y, B.c0.z},
{B.c1.x, B.c1.y, B.c1.z},
{B.c2.x, B.c2.y, B.c2.z}};
for( int i = 0; i < 3; ++i){
for( int j = 0; j < 3; ++j){
r[i][j] = 0.0f;
for( int k = 0; k < 3; ++k){
r[i][j] = r[i][j] + a[i][k] * b[k][j];
}}}
mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
return R;
}

static mat3 add_f33_f33( mat3 A, mat3 B) {
float r[3][3];
float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}};
float b[3][3] = {{B.c0.x, B.c0.y, B.c0.z}, {B.c1.x, B.c1.y, B.c1.z}, {B.c2.x, B.c2.y, B.c2.z}};
for( int i = 0; i < 3; ++i ){
for( int j = 0; j < 3; ++j ){
r[i][j] = a[i][j] + b[i][j];}}
mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
return R;
}

static mat3 invert_f33( mat3 A) {
mat3 R;
float result[3][3];
float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}};
float det = a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0]
+ a[0][2] * a[1][0] * a[2][1] - a[2][0] * a[1][1] * a[0][2]
- a[2][1] * a[1][2] * a[0][0] - a[2][2] * a[1][0] * a[0][1];
if( det != 0.0f ){
result[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1]; result[0][1] = a[2][1] * a[0][2] - a[2][2] * a[0][1];
result[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1]; result[1][0] = a[2][0] * a[1][2] - a[1][0] * a[2][2];
result[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2]; result[1][2] = a[1][0] * a[0][2] - a[0][0] * a[1][2];
result[2][0] = a[1][0] * a[2][1] - a[2][0] * a[1][1]; result[2][1] = a[2][0] * a[0][1] - a[0][0] * a[2][1];
result[2][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1];
R = make_mat3(make_float3(result[0][0], result[0][1], result[0][2]), make_float3(result[1][0], result[1][1],
result[1][2]), make_float3(result[2][0], result[2][1], result[2][2]));
return mult_f_f33( 1.0f / det, R);
}
R = make_mat3(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
return R;
}

static float interpolate1D( float2 table[], int Size, float p) {
if( p <= table[0].x ) return table[0].y;
if( p >= table[Size - 1].x ) return table[Size - 1].y;
for( int i = 0; i < Size - 1; ++i ){
if( table[i].x <= p && p < table[i + 1].x ){
float s = (p - table[i].x) / (table[i + 1].x - table[i].x);
return table[i].y * ( 1.0f - s ) + table[i+1].y * s;}}
return 0.0f;
}

static mat3 RGBtoXYZ( Chromaticities N) {
mat3 M = make_mat3(make_float3(N.red.x, N.red.y, 1.0f - (N.red.x + N.red.y)),
make_float3(N.green.x, N.green.y, 1.0f - (N.green.x + N.green.y)), make_float3(N.blue.x, N.blue.y, 1.0f - (N.blue.x + N.blue.y)));
float3 wh = make_float3(N.white.x / N.white.y, 1.0f, (1.0f - (N.white.x + N.white.y)) / N.white.y);
wh = mult_f3_f33(wh, invert_f33(M));
mat3 WH = make_mat3(make_float3(wh.x, 0.0f, 0.0f), make_float3(0.0f, wh.y, 0.0f), make_float3(0.0f, 0.0f, wh.z));
M = mult_f33_f33(WH, M);
return M;
}

static mat3 XYZtoRGB( Chromaticities N) {
mat3 M = invert_f33(RGBtoXYZ(N));
return M;
}

static float SLog3_to_lin( float SLog) {
float out = 0.0f;
if (SLog >= 171.2102946929f / 1023.0f){
out = exp10f((SLog * 1023.0f - 420.0f) / 261.5f) * (0.18f + 0.01f) - 0.01f;
} else {
out = (SLog * 1023.0f - 95.0f) * 0.01125000f / (171.2102946929f - 95.0f);}
return out;
}

static float lin_to_SLog3( float in) {
float out;
if (in >= 0.01125f) {
out = (420.0f + log10f((in + 0.01f) / (0.18f + 0.01f)) * 261.5f) / 1023.0f;
} else {
out = (in * (171.2102946929f - 95.0f) / 0.01125f + 95.0f) / 1023.0f;
}
return out;
}

static float VLog_to_lin( float x) {
float cutInv = 0.181f;
float b = 0.00873f;
float c = 0.241514f;
float d = 0.598206f;
if (x <= cutInv)
return (x - 0.125f) / 5.6f;
else
return exp10f((x - d) / c) - b;
}

static float lin_to_VLog( float x) {
float cut1 = 0.01f;
float b = 0.00873f;
float c = 0.241514f;
float d = 0.598206f;
if (x < cut1 )
return 5.6f * x + 0.125f;
else
return c * log10f(x + b) + d;
}

static float SLog1_to_lin( float SLog, float b, float ab, float w) {
float lin = 0.0f;
if (SLog >= ab)
lin = ( exp10f(( ( ( SLog - b) / ( w - b) - 0.616596f - 0.03f) / 0.432699f)) - 0.037584f) * 0.9f;
else if (SLog < ab)
lin = ( ( ( SLog - b) / ( w - b) - 0.030001222851889303f) / 5.0f) * 0.9f;
return lin;
}

static float SLog2_to_lin( float SLog, float b, float ab, float w) {
float lin = 0.0f;
if (SLog >= ab)
lin = ( 219.0f * ( exp10f(( ( ( SLog - b) / ( w - b) - 0.616596f - 0.03f) / 0.432699f)) - 0.037584f) / 155.0f) * 0.9f;
else if (SLog < ab)
lin = ( ( ( SLog - b) / ( w - b) - 0.030001222851889303f) / 3.53881278538813f) * 0.9f;
return lin;
}

static float CanonLog_to_lin( float clog) {
float out = 0.0f;
if(clog < 0.12512248f)
out = -( powf( 10.0f, ( 0.12512248f - clog ) / 0.45310179f ) - 1.0f ) / 10.1596f;
else
out = ( powf( 10.0f, ( clog - 0.12512248f ) / 0.45310179f ) - 1.0f ) / 10.1596f;
return out;
}

static float CanonLog2_to_lin( float clog2) {
float out = 0.0f;
if(clog2 < 0.092864125f)
out = -( powf( 10.0f, ( 0.092864125f - clog2 ) / 0.24136077f ) - 1.0f ) / 87.099375f;
else
out = ( powf( 10.0f, ( clog2 - 0.092864125f ) / 0.24136077f ) - 1.0f ) / 87.099375f;
return out;
}

static float CanonLog3_to_lin( float clog3) {
float out = 0.0f;
if(clog3 < 0.097465473f)
out = -( powf( 10.0f, ( 0.12783901f - clog3 ) / 0.36726845f ) - 1.0f ) / 14.98325f;
else if(clog3 <= 0.15277891f)
out = ( clog3 - 0.12512219f ) / 1.9754798f;
else
out = ( powf( 10.0f, ( clog3 - 0.12240537f ) / 0.36726845f ) - 1.0f ) / 14.98325f;
return out;
}

static float LogC_to_lin( float in) {
const float midGraySignal = 0.01f;
const float cut = 1.0f / 9.0f;
const float slope = 3.9086503371f;
const float offset =  -1.3885369913f;
const float encOffset = 0.3855369987f;
const float gain = 800.0f / 400.0f;
const float encGain = 0.2471896383f;
const float gray = 0.005f;
const float nz = 0.052272275f;
float out = (in - encOffset) / encGain;
float ns = (out - offset) / slope;
if (ns > cut)
ns = exp10f(out);
ns = (ns - nz) * gray;
return ns * (0.18f * gain / midGraySignal);
}

static float lin_to_LogC( float in) {
const float midGraySignal = 0.01f;
const float cut = 1.0f / 9.0f;
const float slope = 3.9086503371f;
const float offset =  -1.3885369913f;
const float encOffset = 0.3855369987f;
const float gain = 800.0f / 400.0f;
const float encGain = 0.2471896383f;
const float gray = 0.005f;
const float nz = 0.052272275f;
float out;
float ns = in / (0.18f * gain / midGraySignal);
ns = nz + (ns / gray);
if (ns > cut) {
out = log10f(ns);
} else {
out = offset + (ns * slope);
}
return encOffset + (out * encGain);
}

static float Log3G10_to_lin( float log3g10) {
const float a = 0.224282f;
const float b = 155.975327f;
const float c = 0.01f;
const float g = 15.1927f;
float linear = log3g10 < 0.0f ? (log3g10 / g) : (exp10f(log3g10 / a) - 1.0f) / b;
linear = linear - c;
return linear;
}

static float lin_to_Log3G10( float in) {
const float a = 0.224282f;
const float b = 155.975327f;
const float c = 0.01f;
const float g = 15.1927f;
float out = in + c;
if (out < 0.0f) {
out =  out * g;
} else {
out = a * log10f(out * b + 1.0f);
}
return out;
}

static float3 XYZ_2_xyY( float3 XYZ) {
float3 xyY;
float divisor = (XYZ.x + XYZ.y + XYZ.z);
if (divisor == 0.0f) divisor = 1e-10f;
xyY.x = XYZ.x / divisor;
xyY.y = XYZ.y / divisor;
xyY.z = XYZ.y;
return xyY;
}

static float3 xyY_2_XYZ( float3 xyY) {
float3 XYZ;
XYZ.x = xyY.x * xyY.z / fmaxf( xyY.y, 1e-10f);
XYZ.y = xyY.z;
XYZ.z = (1.0f - xyY.x - xyY.y) * xyY.z / fmaxf( xyY.y, 1e-10f);
return XYZ;
}

static float rgb_2_hue( float3 rgb) {
float hue = 0.0f;
if (rgb.x == rgb.y && rgb.y == rgb.z) {
hue = 0.0f;
} else {
hue = (180.0f/3.1415926535897932f) * atan2f( sqrtf(3.0f) * (rgb.y - rgb.z), 2.0f * rgb.x - rgb.y - rgb.z);
}
if (hue < 0.0f) hue = hue + 360.0f;
return hue;
}

static float rgb_2_yc( float3 rgb, float ycRadiusWeight) {
float r = rgb.x;
float g = rgb.y;
float b = rgb.z;
float chroma = sqrtf(b * (b - g) + g * (g - r) + r * (r - b));
return ( b + g + r + ycRadiusWeight * chroma) / 3.0f;
}

static mat3 calculate_cat_matrix( float2 src_xy, float2 des_xy) {
mat3 coneRespMat = CONE_RESP_MAT_BRADFORD;
const float3 src_xyY = { src_xy.x, src_xy.y, 1.0f };
const float3 des_xyY = { des_xy.x, des_xy.y, 1.0f };
float3 src_XYZ = xyY_2_XYZ( src_xyY );
float3 des_XYZ = xyY_2_XYZ( des_xyY );
float3 src_coneResp = mult_f3_f33( src_XYZ, coneRespMat);
float3 des_coneResp = mult_f3_f33( des_XYZ, coneRespMat);
mat3 vkMat = {
{ des_coneResp.x / src_coneResp.x, 0.0f, 0.0f },
{ 0.0f, des_coneResp.y / src_coneResp.y, 0.0f },
{ 0.0f, 0.0f, des_coneResp.z / src_coneResp.z }
};
mat3 cat_matrix = mult_f33_f33( coneRespMat, mult_f33_f33( vkMat, invert_f33( coneRespMat ) ) );
return cat_matrix;
}

static mat3 calc_sat_adjust_matrix( float sat, float3 rgb2Y) {
float M[3][3];
M[0][0] = (1.0f - sat) * rgb2Y.x + sat; M[1][0] = (1.0f - sat) * rgb2Y.x; M[2][0] = (1.0f - sat) * rgb2Y.x;
M[0][1] = (1.0f - sat) * rgb2Y.y; M[1][1] = (1.0f - sat) * rgb2Y.y + sat; M[2][1] = (1.0f - sat) * rgb2Y.y;
M[0][2] = (1.0f - sat) * rgb2Y.z; M[1][2] = (1.0f - sat) * rgb2Y.z; M[2][2] = (1.0f - sat) * rgb2Y.z + sat;
mat3 R = make_mat3(make_float3(M[0][0], M[0][1], M[0][2]), make_float3(M[1][0], M[1][1], M[1][2]), make_float3(M[2][0], M[2][1], M[2][2]));
R = transpose_f33(R);
return R;
}

static float moncurve_f( float x, float gamma, float offs ) {
float y;
const float fs = (( gamma - 1.0f) / offs) * powf( offs * gamma / ( ( gamma - 1.0f) * ( 1.0f + offs)), gamma);
const float xb = offs / ( gamma - 1.0f);
if ( x >= xb)
y = powf( ( x + offs) / ( 1.0f + offs), gamma);
else
y = x * fs;
return y;
}

static float moncurve_r( float y, float gamma, float offs ) {
float x;
const float yb = powf( offs * gamma / ( ( gamma - 1.0f) * ( 1.0f + offs)), gamma);
const float rs = powf( ( gamma - 1.0f) / offs, gamma - 1.0f) * powf( ( 1.0f + offs) / gamma, gamma);
if ( y >= yb)
x = ( 1.0f + offs) * powf( y, 1.0f / gamma) - offs;
else
x = y * rs;
return x;
}

static float3 moncurve_f_f3( float3 x, float gamma, float offs) {
float3 y;
y.x = moncurve_f( x.x, gamma, offs); y.y = moncurve_f( x.y, gamma, offs); y.z = moncurve_f( x.z, gamma, offs);
return y;
}

static float3 moncurve_r_f3( float3 y, float gamma, float offs) {
float3 x;
x.x = moncurve_r( y.x, gamma, offs); x.y = moncurve_r( y.y, gamma, offs); x.z = moncurve_r( y.z, gamma, offs);
return x;
}

static float bt1886_f( float V, float gamma, float Lw, float Lb) {
float a = powf( powf( Lw, 1.0f/gamma) - powf( Lb, 1.0f/gamma), gamma);
float b = powf( Lb, 1.0f/gamma) / ( powf( Lw, 1.0f/gamma) - powf( Lb, 1.0f/gamma));
float L = a * powf( fmaxf( V + b, 0.0f), gamma);
return L;
}

static float bt1886_r( float L, float gamma, float Lw, float Lb) {
float a = powf( powf( Lw, 1.0f/gamma) - powf( Lb, 1.0f/gamma), gamma);
float b = powf( Lb, 1.0f/gamma) / ( powf( Lw, 1.0f/gamma) - powf( Lb, 1.0f/gamma));
float V = powf( fmaxf( L / a, 0.0f), 1.0f/gamma) - b;
return V;
}

static float3 bt1886_f_f3( float3 V, float gamma, float Lw, float Lb) {
float3 L;
L.x = bt1886_f( V.x, gamma, Lw, Lb); L.y = bt1886_f( V.y, gamma, Lw, Lb); L.z = bt1886_f( V.z, gamma, Lw, Lb);
return L;
}

static float3 bt1886_r_f3( float3 L, float gamma, float Lw, float Lb) {
float3 V;
V.x = bt1886_r( L.x, gamma, Lw, Lb); V.y = bt1886_r( L.y, gamma, Lw, Lb); V.z = bt1886_r( L.z, gamma, Lw, Lb);
return V;
}

static float smpteRange_to_fullRange( float in) {
const float REFBLACK = ( 64.0f / 1023.0f);
const float REFWHITE = ( 940.0f / 1023.0f);
return (( in - REFBLACK) / ( REFWHITE - REFBLACK));
}

static float fullRange_to_smpteRange( float in) {
const float REFBLACK = ( 64.0f / 1023.0f);
const float REFWHITE = ( 940.0f / 1023.0f);
return ( in * ( REFWHITE - REFBLACK) + REFBLACK );
}

static float3 smpteRange_to_fullRange_f3( float3 rgbIn) {
float3 rgbOut;
rgbOut.x = smpteRange_to_fullRange( rgbIn.x); 
rgbOut.y = smpteRange_to_fullRange( rgbIn.y); 
rgbOut.z = smpteRange_to_fullRange( rgbIn.z);
return rgbOut;
}

static float3 fullRange_to_smpteRange_f3( float3 rgbIn) {
float3 rgbOut;
rgbOut.x = fullRange_to_smpteRange( rgbIn.x); 
rgbOut.y = fullRange_to_smpteRange( rgbIn.y); 
rgbOut.z = fullRange_to_smpteRange( rgbIn.z);
return rgbOut;
}

static float3 dcdm_decode( float3 XYZp) {
float3 XYZ;
XYZ.x = (52.37f/48.0f) * powf( XYZp.x, 2.6f);
XYZ.y = (52.37f/48.0f) * powf( XYZp.y, 2.6f);
XYZ.z = (52.37f/48.0f) * powf( XYZp.z, 2.6f);
return XYZ;
}

static float3 dcdm_encode( float3 XYZ) {
float3 XYZp;
XYZp.x = powf( (48.0f/52.37f) * XYZ.x, 1.0f/2.6f);
XYZp.y = powf( (48.0f/52.37f) * XYZ.y, 1.0f/2.6f);
XYZp.z = powf( (48.0f/52.37f) * XYZ.z, 1.0f/2.6f);
return XYZp;
}

static float ST2084_2_Y( float N ) {
float Np = powf( N, 1.0f / pq_m2 );
float L = Np - pq_c1;
if ( L < 0.0f )
L = 0.0f;
L = L / ( pq_c2 - pq_c3 * Np );
L = powf( L, 1.0f / pq_m1 );
return L * pq_C;
}

static float Y_2_ST2084( float C ) {
float L = C / pq_C;
float Lm = powf( L, pq_m1 );
float N = ( pq_c1 + pq_c2 * Lm ) / ( 1.0f + pq_c3 * Lm );
N = powf( N, pq_m2 );
return N;
}

static float3 Y_2_ST2084_f3( float3 in) {
float3 out;
out.x = Y_2_ST2084( in.x); out.y = Y_2_ST2084( in.y); out.z = Y_2_ST2084( in.z);
return out;
}

static float3 ST2084_2_Y_f3( float3 in) {
float3 out;
out.x = ST2084_2_Y( in.x); out.y = ST2084_2_Y( in.y); out.z = ST2084_2_Y( in.z);
return out;
}

static float3 ST2084_2_HLG_1000nits_f3( float3 PQ) {
float3 displayLinear = ST2084_2_Y_f3( PQ);
float Y_d = 0.2627f * displayLinear.x + 0.6780f * displayLinear.y + 0.0593f * displayLinear.z;
const float L_w = 1000.0f;
const float L_b = 0.0f;
const float alpha = (L_w - L_b);
const float beta = L_b;
const float gamma = 1.2f;
float3 sceneLinear;
if (Y_d == 0.0f) {
sceneLinear.x = 0.0f; sceneLinear.y = 0.0f; sceneLinear.z = 0.0f;
} else {
sceneLinear.x = powf( (Y_d - beta) / alpha, (1.0f - gamma) / gamma) * ((displayLinear.x - beta) / alpha);
sceneLinear.y = powf( (Y_d - beta) / alpha, (1.0f - gamma) / gamma) * ((displayLinear.y - beta) / alpha);
sceneLinear.z = powf( (Y_d - beta) / alpha, (1.0f - gamma) / gamma) * ((displayLinear.z - beta) / alpha);
}
const float a = 0.17883277f;
const float b = 0.28466892f;
const float c = 0.55991073f;
float3 HLG;
if (sceneLinear.x <= 1.0f / 12.0f) {
HLG.x = sqrtf(3.0f * sceneLinear.x);
} else {
HLG.x = a * logf(12.0f * sceneLinear.x-b)+c;
}
if (sceneLinear.y <= 1.0f / 12.0f) {
HLG.y = sqrtf(3.0f * sceneLinear.y);
} else {
HLG.y = a * logf(12.0f * sceneLinear.y-b)+c;
}
if (sceneLinear.z <= 1.0f / 12.0f) {
HLG.z = sqrtf(3.0f * sceneLinear.z);
} else {
HLG.z = a * logf(12.0f * sceneLinear.z - b) + c;
}
return HLG;
}

static float3 HLG_2_ST2084_1000nits_f3( float3 HLG) {
const float a = 0.17883277f;
const float b = 0.28466892f;
const float c = 0.55991073f;
const float L_w = 1000.0f;
const float L_b = 0.0f;
const float alpha = (L_w - L_b);
const float beta = L_b;
const float gamma = 1.2f;
float3 sceneLinear;
if ( HLG.x >= 0.0f && HLG.x <= 0.5f) {
sceneLinear.x = powf(HLG.x, 2.0f) / 3.0f;
} else {
sceneLinear.x = (expf((HLG.x - c) / a) + b) / 12.0f;
}
if ( HLG.y >= 0.0f && HLG.y <= 0.5f) {
sceneLinear.y = powf(HLG.y, 2.0f) / 3.0f;
} else {
sceneLinear.y = (expf((HLG.y - c) / a) + b) / 12.0f;
}
if ( HLG.z >= 0.0f && HLG.z <= 0.5f) {
sceneLinear.z = powf(HLG.z, 2.0f) / 3.0f;
} else {
sceneLinear.z = (expf((HLG.z - c) / a) + b) / 12.0f;
}
float Y_s = 0.2627f * sceneLinear.x + 0.6780f * sceneLinear.y + 0.0593f * sceneLinear.z;
float3 displayLinear;
displayLinear.x = alpha * powf( Y_s, gamma - 1.0f) * sceneLinear.x + beta;
displayLinear.y = alpha * powf( Y_s, gamma - 1.0f) * sceneLinear.y + beta;
displayLinear.z = alpha * powf( Y_s, gamma - 1.0f) * sceneLinear.z + beta;
float3 PQ = Y_2_ST2084_f3( displayLinear);
return PQ;
}

static float rgb_2_saturation( float3 rgb) {
return ( fmaxf( max_f3(rgb), TINY) - fmaxf( min_f3(rgb), TINY)) / fmaxf( max_f3(rgb), 1e-2f);
}

static SegmentedSplineParams_c5 RRT_PARAMS() {
SegmentedSplineParams_c5 A = {{ -4.0f, -4.0f, -3.1573765773f, -0.4852499958f, 1.8477324706f, 1.8477324706f},
{ -0.7185482425f, 2.0810307172f, 3.6681241237f, 4.0f, 4.0f, 4.0f}, {0.18f * exp2f(-15.0f), 0.0001f},
{0.18f, 4.8f}, {0.18f * exp2f(18.0f), 10000.0f}, 0.0f, 0.0f};
return A;
};

static float segmented_spline_c5_fwd( float x) {
SegmentedSplineParams_c5 C = RRT_PARAMS();
const int N_KNOTS_LOW = 4;
const int N_KNOTS_HIGH = 4;
float logx = log10f( fmaxf(x, 0.0f));
float logy = 0.0f;
if ( logx <= log10f(C.minPoint.x) ) {
logy = logx * C.slopeLow + ( log10f(C.minPoint.y) - C.slopeLow * log10f(C.minPoint.x) );
} else if (( logx > log10f(C.minPoint.x) ) && ( logx < log10f(C.midPoint.x) )) {
float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10f(C.minPoint.x)) / (log10f(C.midPoint.x) - log10f(C.minPoint.x));
int j = knot_coord;
float t = knot_coord - j;
float3 cf = make_float3( C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]);
float3 monomials = make_float3( t * t, t, 1.0f );
logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM));
} else if (( logx >= log10f(C.midPoint.x) ) && ( logx < log10f(C.maxPoint.x) )) {
float knot_coord = (N_KNOTS_HIGH-1) * (logx - log10(C.midPoint.x)) / (log10f(C.maxPoint.x) - log10(C.midPoint.x));
int j = knot_coord;
float t = knot_coord - j;
float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]};
float3 monomials = make_float3( t * t, t, 1.0f );
logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM));
} else {
logy = logx * C.slopeHigh + ( log10f(C.maxPoint.y) - C.slopeHigh * log10f(C.maxPoint.x) );
}
return exp10f(logy);
}

static SegmentedSplineParams_c9 ODT_48nits() {
SegmentedSplineParams_c9 A =
{{ -1.6989700043f, -1.6989700043f, -1.4779f, -1.2291f, -0.8648f, -0.448f, 0.00518f, 0.4511080334f, 0.9113744414f, 0.9113744414f},
{ 0.5154386965f, 0.8470437783f, 1.1358f, 1.3802f, 1.5197f, 1.5985f, 1.6467f, 1.6746091357f, 1.6878733390f, 1.6878733390f },
{segmented_spline_c5_fwd( 0.18f * exp2f(-6.5f) ), 0.02f}, {segmented_spline_c5_fwd( 0.18f ), 4.8f},
{segmented_spline_c5_fwd( 0.18f * exp2f(6.5f) ), 48.0f}, 0.0f, 0.04f};
return A;
};

static SegmentedSplineParams_c9 ODT_1000nits() {
SegmentedSplineParams_c9 A =
{{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f },
{ 0.8089132070f, 1.1910867930f, 1.5683f, 1.9483f, 2.3083f, 2.6384f, 2.8595f, 2.9872608805f, 3.0127391195f, 3.0127391195f },
{segmented_spline_c5_fwd( 0.18f * exp2f(-12.0f) ), 0.0001f}, {segmented_spline_c5_fwd( 0.18f ), 10.0f},
{segmented_spline_c5_fwd( 0.18 * exp2f(10.0f) ), 1000.0f}, 3.0f, 0.06f};
return A;
};

static SegmentedSplineParams_c9 ODT_2000nits() {
SegmentedSplineParams_c9 A =
{{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f },
{ 0.8019952042f, 1.1980047958f, 1.5943f, 1.9973f, 2.3783f, 2.7684f, 3.0515f, 3.2746293562f, 3.3274306351f, 3.3274306351f },
{segmented_spline_c5_fwd( 0.18f * exp2f(-12.0f) ), 0.0001f}, {segmented_spline_c5_fwd( 0.18f ), 10.0f},
{segmented_spline_c5_fwd( 0.18f * exp2f(11.0f) ), 2000.0f}, 3.0f, 0.12f};
return A;
};

static SegmentedSplineParams_c9 ODT_4000nits() {
SegmentedSplineParams_c9 A =
{{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f },
{ 0.7973186613f, 1.2026813387f, 1.6093f, 2.0108f, 2.4148f, 2.8179f, 3.1725f, 3.5344995451f, 3.6696204376f, 3.6696204376f },
{segmented_spline_c5_fwd( 0.18f * exp2f(-12.0f) ), 0.0001f}, {segmented_spline_c5_fwd( 0.18f ), 10.0f},
{segmented_spline_c5_fwd( 0.18f * exp2f(12.0f) ), 4000.0f}, 3.0f, 0.3f};
return A;
};

static float segmented_spline_c5_rev( float y) {
SegmentedSplineParams_c5 C = RRT_PARAMS();
const int N_KNOTS_LOW = 4;
const int N_KNOTS_HIGH = 4;
const float KNOT_INC_LOW = (log10f(C.midPoint.x) - log10f(C.minPoint.x)) / (N_KNOTS_LOW - 1.0f);
const float KNOT_INC_HIGH = (log10f(C.maxPoint.x) - log10f(C.midPoint.x)) / (N_KNOTS_HIGH - 1.0f);
float KNOT_Y_LOW[ N_KNOTS_LOW];
for (int i = 0; i < N_KNOTS_LOW; i = i + 1) {
KNOT_Y_LOW[i] = ( C.coefsLow[i] + C.coefsLow[i + 1]) / 2.0f;
};
float KNOT_Y_HIGH[ N_KNOTS_HIGH];
for (int i = 0; i < N_KNOTS_HIGH; i = i+1) {
KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i + 1]) / 2.0f;
};
float logy = log10f( fmaxf(y, 1e-10f));
float logx;
if (logy <= log10f(C.minPoint.y)) {
logx = log10f(C.minPoint.x);
} else if ( (logy > log10f(C.minPoint.y)) && (logy <= log10f(C.midPoint.y)) ) {
unsigned int j = 0;
float3 cf = make_float3(0.0f, 0.0f, 0.0f);
if ( logy > KNOT_Y_LOW[0] && logy <= KNOT_Y_LOW[1]) {
cf.x = C.coefsLow[0]; cf.y = C.coefsLow[1]; cf.z = C.coefsLow[2]; j = 0;
} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
cf.x = C.coefsLow[1]; cf.y = C.coefsLow[2]; cf.z = C.coefsLow[3]; j = 1;
} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
cf.x = C.coefsLow[2]; cf.y = C.coefsLow[3]; cf.z = C.coefsLow[4]; j = 2;
}
const float3 tmp = mult_f3_f33( cf, MM);
float a = tmp.x; float b = tmp.y; float c = tmp.z;
c = c - logy;
const float d = sqrtf( b * b - 4.0f * a * c);
const float t = ( 2.0f * c) / ( -d - b);
logx = log10f(C.minPoint.x) + ( t + j) * KNOT_INC_LOW;
} else if ( (logy > log10f(C.midPoint.y)) && (logy < log10f(C.maxPoint.y)) ) {
unsigned int j = 0;
float3 cf = make_float3(0.0f, 0.0f, 0.0f);
if ( logy > KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
cf.x = C.coefsHigh[0]; cf.y = C.coefsHigh[1]; cf.z = C.coefsHigh[2]; j = 0;
} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
cf.x = C.coefsHigh[1]; cf.y = C.coefsHigh[2]; cf.z = C.coefsHigh[3]; j = 1;
} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
cf.x = C.coefsHigh[2]; cf.y = C.coefsHigh[3]; cf.z = C.coefsHigh[4]; j = 2;
}
const float3 tmp = mult_f3_f33( cf, MM);
float a = tmp.x; float b = tmp.y; float c = tmp.z;
c = c - logy;
const float d = sqrtf( b * b - 4.0f * a * c);
const float t = ( 2.0f * c) / ( -d - b);
logx = log10f(C.midPoint.x) + ( t + j) * KNOT_INC_HIGH;
} else {
logx = log10f(C.maxPoint.x);
}
return exp10f( logx);
}

static float segmented_spline_c9_fwd( float x, SegmentedSplineParams_c9 C) {
const int N_KNOTS_LOW = 8;
const int N_KNOTS_HIGH = 8;
float logx = log10f( fmaxf(x, 0.0f));
float logy = 0.0f;
if ( logx <= log10f(C.minPoint.x) ) {
logy = logx * C.slopeLow + ( log10f(C.minPoint.y) - C.slopeLow * log10f(C.minPoint.x) );
} else if (( logx > log10f(C.minPoint.x) ) && ( logx < log10f(C.midPoint.x) )) {
float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10f(C.minPoint.x)) / (log10f(C.midPoint.x) - log10f(C.minPoint.x));
int j = knot_coord;
float t = knot_coord - j;
float3 cf = { C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]};
float3 monomials = make_float3( t * t, t, 1.0f );
logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM));
} else if (( logx >= log10f(C.midPoint.x) ) && ( logx < log10f(C.maxPoint.x) )) {
float knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10f(C.midPoint.x)) / (log10f(C.maxPoint.x) - log10f(C.midPoint.x));
int j = knot_coord;
float t = knot_coord - j;
float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]};
float3 monomials = make_float3( t * t, t, 1.0f );
logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM));
} else {
logy = logx * C.slopeHigh + ( log10f(C.maxPoint.y) - C.slopeHigh * log10f(C.maxPoint.x) );
}
return exp10f(logy);
}

static float segmented_spline_c9_rev( float y, SegmentedSplineParams_c9 C) {
const int N_KNOTS_LOW = 8;
const int N_KNOTS_HIGH = 8;
const float KNOT_INC_LOW = (log10f(C.midPoint.x) - log10f(C.minPoint.x)) / (N_KNOTS_LOW - 1.0f);
const float KNOT_INC_HIGH = (log10f(C.maxPoint.x) - log10f(C.midPoint.x)) / (N_KNOTS_HIGH - 1.0f);
float KNOT_Y_LOW[ N_KNOTS_LOW];
for (int i = 0; i < N_KNOTS_LOW; i = i + 1) {
KNOT_Y_LOW[ i] = ( C.coefsLow[i] + C.coefsLow[i + 1]) / 2.0f;
};
float KNOT_Y_HIGH[ N_KNOTS_HIGH];
for (int i = 0; i < N_KNOTS_HIGH; i = i + 1) {
KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i + 1]) / 2.0f;
};
float logy = log10f( fmaxf( y, 1e-10f));
float logx;
if (logy <= log10f(C.minPoint.y)) {
logx = log10f(C.minPoint.x);
} else if ( (logy > log10f(C.minPoint.y)) && (logy <= log10f(C.midPoint.y)) ) {
unsigned int j = 0;
float3 cf = make_float3(0.0f, 0.0f, 0.0f);
if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) {
cf.x = C.coefsLow[0]; cf.y = C.coefsLow[1]; cf.z = C.coefsLow[2]; j = 0;
} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
cf.x = C.coefsLow[1]; cf.y = C.coefsLow[2]; cf.z = C.coefsLow[3]; j = 1;
} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
cf.x = C.coefsLow[2]; cf.y = C.coefsLow[3]; cf.z = C.coefsLow[4]; j = 2;
} else if ( logy > KNOT_Y_LOW[ 3] && logy <= KNOT_Y_LOW[ 4]) {
cf.x = C.coefsLow[3]; cf.y = C.coefsLow[4]; cf.z = C.coefsLow[5]; j = 3;
} else if ( logy > KNOT_Y_LOW[ 4] && logy <= KNOT_Y_LOW[ 5]) {
cf.x = C.coefsLow[4]; cf.y = C.coefsLow[5]; cf.z = C.coefsLow[6]; j = 4;
} else if ( logy > KNOT_Y_LOW[ 5] && logy <= KNOT_Y_LOW[ 6]) {
cf.x = C.coefsLow[5]; cf.y = C.coefsLow[6]; cf.z = C.coefsLow[7]; j = 5;
} else if ( logy > KNOT_Y_LOW[ 6] && logy <= KNOT_Y_LOW[ 7]) {
cf.x = C.coefsLow[6]; cf.y = C.coefsLow[7]; cf.z = C.coefsLow[8]; j = 6;
}
const float3 tmp = mult_f3_f33( cf, MM);
float a = tmp.x; float b = tmp.y; float c = tmp.z;
c = c - logy;
const float d = sqrtf( b * b - 4.0f * a * c);
const float t = ( 2.0f * c) / ( -d - b);
logx = log10f(C.minPoint.x) + ( t + j) * KNOT_INC_LOW;
} else if ( (logy > log10f(C.midPoint.y)) && (logy < log10f(C.maxPoint.y)) ) {
unsigned int j = 0;
float3 cf = make_float3(0.0f, 0.0f, 0.0f);
if ( logy > KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
cf.x = C.coefsHigh[0]; cf.y = C.coefsHigh[1]; cf.z = C.coefsHigh[2]; j = 0;
} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
cf.x = C.coefsHigh[1]; cf.y = C.coefsHigh[2]; cf.z = C.coefsHigh[3]; j = 1;
} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
cf.x = C.coefsHigh[2]; cf.y = C.coefsHigh[3]; cf.z = C.coefsHigh[4]; j = 2;
} else if ( logy > KNOT_Y_HIGH[ 3] && logy <= KNOT_Y_HIGH[ 4]) {
cf.x = C.coefsHigh[3]; cf.y = C.coefsHigh[4]; cf.z = C.coefsHigh[5]; j = 3;
} else if ( logy > KNOT_Y_HIGH[ 4] && logy <= KNOT_Y_HIGH[ 5]) {
cf.x = C.coefsHigh[4]; cf.y = C.coefsHigh[5]; cf.z = C.coefsHigh[6]; j = 4;
} else if ( logy > KNOT_Y_HIGH[ 5] && logy <= KNOT_Y_HIGH[ 6]) {
cf.x = C.coefsHigh[5]; cf.y = C.coefsHigh[6]; cf.z = C.coefsHigh[7]; j = 5;
} else if ( logy > KNOT_Y_HIGH[ 6] && logy <= KNOT_Y_HIGH[ 7]) {
cf.x = C.coefsHigh[6]; cf.y = C.coefsHigh[7]; cf.z = C.coefsHigh[8]; j = 6;
}
const float3 tmp = mult_f3_f33( cf, MM);
float a = tmp.x; float b = tmp.y; float c = tmp.z;
c = c - logy;
const float d = sqrtf( b * b - 4.0f * a * c);
const float t = ( 2.0f * c) / ( -d - b);
logx = log10f(C.midPoint.x) + ( t + j) * KNOT_INC_HIGH;
} else {
logx = log10f(C.maxPoint.x);
}
return exp10f( logx);
}

static float3 segmented_spline_c9_rev_f3( float3 rgbPre) {
SegmentedSplineParams_c9 C = ODT_48nits();
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, C);
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, C);
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, C);
return rgbPost;
}

static float3 segmented_spline_c5_rev_f3( float3 rgbPre) {
float3 rgbPost;
rgbPost.x = segmented_spline_c5_rev( rgbPre.x);
rgbPost.y = segmented_spline_c5_rev( rgbPre.y);
rgbPost.z = segmented_spline_c5_rev( rgbPre.z);
return rgbPost;
}

static float lin_to_ACEScc( float in) {
if (in <= 0.0f)
return -0.3584474886f;
else if (in < exp2f(-15.0f))
return (log2f( exp2f(-16.0f) + in * 0.5f) + 9.72f) / 17.52f;
else
return (log2f(in) + 9.72f) / 17.52f;
}

static float3 ACES_to_ACEScc( float3 ACES) {
ACES = max_f3_f( ACES, 0.0f);
float3 lin_AP1 = mult_f3_f33( ACES, AP0_2_AP1_MAT);
float3 out;
out.x = lin_to_ACEScc( lin_AP1.x); out.y = lin_to_ACEScc( lin_AP1.y); out.z = lin_to_ACEScc( lin_AP1.z);
return out;
}

static float ACEScc_to_lin( float in) {
if (in < -0.3013698630f)
return (exp2f(in * 17.52f - 9.72f) - exp2f(-16.0f)) * 2.0f;
else
return exp2f(in * 17.52f - 9.72f);
}

static float3 ACEScc_to_ACES( float3 ACEScc) {
float3 lin_AP1;
lin_AP1.x = ACEScc_to_lin( ACEScc.x); lin_AP1.y = ACEScc_to_lin( ACEScc.y); lin_AP1.z = ACEScc_to_lin( ACEScc.z);
float3 ACES = mult_f3_f33( lin_AP1, AP1_2_AP0_MAT);
return ACES;
}

static float lin_to_ACEScct( float in) {
if (in <= X_BRK)
return A * in + B;
else
return (log2f(in) + 9.72f) / 17.52f;
}

static float ACEScct_to_lin( float in) {
if (in > Y_BRK)
return exp2f(in * 17.52f - 9.72f);
else
return (in - B) / A;
}

static float3 ACES_to_ACEScct( float3 in) {
float3 ap1_lin = mult_f3_f33( in, AP0_2_AP1_MAT);
float3 acescct;
acescct.x = lin_to_ACEScct( ap1_lin.x); acescct.y = lin_to_ACEScct( ap1_lin.y); acescct.z = lin_to_ACEScct( ap1_lin.z);
return acescct;
}

static float3 ACEScct_to_ACES( float3 in) {
float3 ap1_lin;
ap1_lin.x = ACEScct_to_lin( in.x); ap1_lin.y = ACEScct_to_lin( in.y); ap1_lin.z = ACEScct_to_lin( in.z);
return mult_f3_f33( ap1_lin, AP1_2_AP0_MAT);
}

static float3 ACES_to_ACEScg( float3 ACES) {
ACES = max_f3_f( ACES, 0.0f);
float3 ACEScg = mult_f3_f33( ACES, AP0_2_AP1_MAT);
return ACEScg;
}

static float3 ACEScg_to_ACES( float3 ACEScg) {
float3 ACES = mult_f3_f33( ACEScg, AP1_2_AP0_MAT);
return ACES;
}

static float3 adx_convertFromLinear( float3 aces) {
aces.x = aces.x < 0.00130127f ? (aces.x - 0.00130127f) / 0.04911331f :
aces.x < 0.001897934f ? (logf(aces.x) + 6.644415f) / 37.74261f : 
aces.x < 0.118428f ? logf(aces.x) * (0.02871031f * logf(aces.x) + 0.383914f) + 1.288383f : 
(logf(aces.x) + 4.645361f) / 4.1865183f;
aces.y =   aces.y < 0.00130127f ? (aces.y - 0.00130127f) / 0.04911331f :
aces.y < 0.001897934f ? (logf(aces.y) + 6.644415f) / 37.74261f : 
aces.y < 0.118428f ? logf(aces.y) * (0.02871031f * logf(aces.y) + 0.383914f) + 1.288383f : 
(logf(aces.y) + 4.645361f) / 4.1865183f;
aces.z =   aces.z < 0.00130127f ? (aces.z - 0.00130127f) / 0.04911331f :
aces.z < 0.001897934f ? (logf(aces.z) + 6.644415f) / 37.74261f : 
aces.z < 0.118428f ? logf(aces.z) * (0.02871031f * logf(aces.z) + 0.383914f) + 1.288383f : 
(logf(aces.z) + 4.645361f) / 4.1865183f;
return aces;
}

static float3 adx_convertToLinear( float3 aces) {
aces.x = aces.x < 0.0f ? 0.04911331f * aces.x + 0.001301270f : 
aces.x < 0.01f ? expf(37.74261f * aces.x - 6.644415f) : 
aces.x < 0.6f ? expf(-6.685996f + 2.302585f * powf(6.569476f * aces.x - 0.03258072f, 0.5f)) : 
expf(fminf(4.1865183f * aces.x - 4.645361f, 86.4f));
aces.y = aces.y < 0.0f ? 0.04911331f * aces.y + 0.001301270f : 
aces.y < 0.01f ? expf(37.74261f * aces.y - 6.644415f) : 
aces.y < 0.6f ? expf(-6.685996f + 2.302585f * powf(6.569476f * aces.y - 0.03258072f, 0.5f)) : 
expf(fminf(4.1865183f * aces.y - 4.645361f, 86.4f));
aces.z = aces.z < 0.0f ? 0.04911331f * aces.z + 0.001301270f : 
aces.z < 0.01f ? expf(37.74261f * aces.z - 6.644415f) : 
aces.z < 0.6f ? expf(-6.685996f + 2.302585f * powf(6.569476f * aces.z - 0.03258072f, 0.5f)) : 
expf(fminf(4.1865183f * aces.z - 4.645361f, 86.4f));
return aces;
}

static float3 ADX_to_ACES( float3 aces) {
aces.x = aces.x * 2.048f - 0.19f;
aces.y = aces.y * 2.048f - 0.19f;
aces.z = aces.z * 2.048f - 0.19f;
aces = mult_f3_f33(aces, CDD_TO_CID);
aces = adx_convertToLinear(aces);
aces = mult_f3_f33(aces, EXP_TO_ACES);
return aces;
}

static float3 ACES_to_ADX( float3 aces) {
aces = mult_f3_f33(aces, invert_f33(EXP_TO_ACES));
aces = adx_convertFromLinear(aces);
aces =  mult_f3_f33(aces, invert_f33(CDD_TO_CID));
aces.x = (aces.x + 0.19f) / 2.048f;
aces.y = (aces.y + 0.19f) / 2.048f;
aces.z = (aces.z + 0.19f) / 2.048f;
return aces;
}

static float3 ICpCt_to_ACES( float3 ICtCp) {
float3 LMSp = mult_f3_f33( ICtCp, ICtCp_2_LMSp_MAT);
float3 LMS;
LMS.x = ST2084_2_Y(LMSp.x);
LMS.y = ST2084_2_Y(LMSp.y);
LMS.z = ST2084_2_Y(LMSp.z);
float3 aces = mult_f3_f33(LMS, LMS_2_AP0_MAT);
float scale = 209.0f;
aces = mult_f_f3( 1.0f / scale, aces);
return aces;
}

static float3 ACES_to_ICpCt( float3 aces) {
float scale = 209.0f;
aces = mult_f_f3( scale, aces);
float3 LMS = mult_f3_f33(aces, AP0_2_LMS_MAT);
float3 LMSp;
LMSp.x = Y_2_ST2084(LMS.x);
LMSp.y = Y_2_ST2084(LMS.y);
LMSp.z = Y_2_ST2084(LMS.z);
float3 ICtCp = mult_f3_f33(LMSp, LMSp_2_ICtCp_MAT);
return ICtCp;
}

static float3 LogC_EI800_AWG_to_ACES( float3 in) {
float3 lin_AWG;
lin_AWG.x = LogC_to_lin(in.x);
lin_AWG.y = LogC_to_lin(in.y);
lin_AWG.z = LogC_to_lin(in.z);
float3 aces = mult_f3_f33( lin_AWG, AWG_2_AP0_MAT);
return aces;
}

static float3 ACES_to_LogC_EI800_AWG( float3 in) {
float3 lin_AWG = mult_f3_f33( in, AP0_2_AWG_MAT);
float3 out;
out.x = lin_to_LogC(lin_AWG.x);
out.y = lin_to_LogC(lin_AWG.y);
out.z = lin_to_LogC(lin_AWG.z);
return out;
}

static float3 Log3G10_RWG_to_ACES( float3 in) {
float3 lin_RWG;
lin_RWG.x = Log3G10_to_lin(in.x);
lin_RWG.y = Log3G10_to_lin(in.y);
lin_RWG.z = Log3G10_to_lin(in.z);
float3 aces = mult_f3_f33( lin_RWG, RWG_2_AP0_MAT);
return aces;
}

static float3 ACES_to_Log3G10_RWG( float3 in) {
float3 lin_RWG = mult_f3_f33(in, AP0_2_RWG_MAT);
float3 out;
out.x = lin_to_Log3G10(lin_RWG.x);
out.y = lin_to_Log3G10(lin_RWG.y);
out.z = lin_to_Log3G10(lin_RWG.z);
return out;
}

static float3 SLog3_SG3_to_ACES( float3 in) {
float3 lin_SG3;
lin_SG3.x = SLog3_to_lin(in.x);
lin_SG3.y = SLog3_to_lin(in.y);
lin_SG3.z = SLog3_to_lin(in.z);
float3 aces = mult_f3_f33(lin_SG3, SG3_2_AP0_MAT);
return aces;
}

static float3 ACES_to_SLog3_SG3( float3 in) {
float3 lin_SG3 = mult_f3_f33(in, AP0_2_SG3_MAT);
float3 out;
out.x = lin_to_SLog3(lin_SG3.x);
out.y = lin_to_SLog3(lin_SG3.y);
out.z = lin_to_SLog3(lin_SG3.z);
return out;
}

static float3 SLog3_SG3C_to_ACES( float3 in) {
float3 lin_SG3C;
lin_SG3C.x = SLog3_to_lin(in.x);
lin_SG3C.y = SLog3_to_lin(in.y);
lin_SG3C.z = SLog3_to_lin(in.z);
float3 aces = mult_f3_f33(lin_SG3C, SG3C_2_AP0_MAT);
return aces;
}

static float3 ACES_to_SLog3_SG3C( float3 in) {
float3 lin_SG3C = mult_f3_f33(in, AP0_2_SG3C_MAT);
float3 out;
out.x = lin_to_SLog3(lin_SG3C.x);
out.y = lin_to_SLog3(lin_SG3C.y);
out.z = lin_to_SLog3(lin_SG3C.z);
return out;
}

static float3 Venice_SLog3_SG3_to_ACES( float3 in) {
float3 lin_SG3;
lin_SG3.x = SLog3_to_lin(in.x);
lin_SG3.y = SLog3_to_lin(in.y);
lin_SG3.z = SLog3_to_lin(in.z);
float3 aces = mult_f3_f33(lin_SG3, VSG3_2_AP0_MAT);
return aces;
}

static float3 ACES_to_Venice_SLog3_SG3( float3 in) {
mat3 AP0_2_VSG3_MAT = invert_f33(VSG3_2_AP0_MAT);
float3 lin_SG3 = mult_f3_f33(in, AP0_2_VSG3_MAT);
float3 out;
out.x = lin_to_SLog3(lin_SG3.x);
out.y = lin_to_SLog3(lin_SG3.y);
out.z = lin_to_SLog3(lin_SG3.z);
return out;
}

static float3 Venice_SLog3_SG3C_to_ACES( float3 in) {
float3 lin_SG3;
lin_SG3.x = SLog3_to_lin(in.x);
lin_SG3.y = SLog3_to_lin(in.y);
lin_SG3.z = SLog3_to_lin(in.z);
float3 aces = mult_f3_f33(lin_SG3, VSG3C_2_AP0_MAT);
return aces;
}

static float3 ACES_to_Venice_SLog3_SG3C( float3 in) {
mat3 AP0_2_VSG3C_MAT = invert_f33(VSG3C_2_AP0_MAT);
float3 lin_SG3 = mult_f3_f33(in, AP0_2_VSG3C_MAT);
float3 out;
out.x = lin_to_SLog3(lin_SG3.x);
out.y = lin_to_SLog3(lin_SG3.y);
out.z = lin_to_SLog3(lin_SG3.z);
return out;
}

static float3 VLog_VGamut_to_ACES( float3 in) {
float3 lin_Vlog;
lin_Vlog.x = VLog_to_lin(in.x);
lin_Vlog.y = VLog_to_lin(in.y);
lin_Vlog.z = VLog_to_lin(in.z);
float3 aces = mult_f3_f33(lin_Vlog, VGAMUT_2_AP0_MAT);
return aces;
}

static float3 ACES_to_VLog_VGamut( float3 in) {
mat3 AP0_2_VGAMUT_MAT = invert_f33(VGAMUT_2_AP0_MAT);
float3 lin_VLog = mult_f3_f33(in, AP0_2_VGAMUT_MAT);
float3 out;
out.x = lin_to_VLog(lin_VLog.x);
out.y = lin_to_VLog(lin_VLog.y);
out.z = lin_to_VLog(lin_VLog.z);
return out;
}

static float3 IDT_Alexa_v3_raw_EI800_CCT6500( float3 In) {
float black = 256.0f / 65535.0f;
float r_lin = (In.x - black);
float g_lin = (In.y - black);
float b_lin = (In.z - black);
float3 aces;
aces.x = r_lin * 0.809931f + g_lin * 0.162741f + b_lin * 0.027328f;
aces.y = r_lin * 0.083731f + g_lin * 1.108667f + b_lin * -0.192397f;
aces.z = r_lin * 0.044166f + g_lin * -0.272038f + b_lin * 1.227872f;
return aces;
}

static float3 IDT_Panasonic_V35( float3 VLog) {
mat3 mat = VGAMUT_2_AP0_MAT;
float rLin = VLog_to_lin(VLog.x);
float gLin = VLog_to_lin(VLog.y);
float bLin = VLog_to_lin(VLog.z);
float3 out;
out.x = mat.c0.x * rLin + mat.c0.y * gLin + mat.c0.z * bLin;
out.y = mat.c1.x * rLin + mat.c1.y * gLin + mat.c1.z * bLin;
out.z = mat.c2.x * rLin + mat.c2.y * gLin + mat.c2.z * bLin;
return out;
}

static float3 IDT_Canon_C100_A_D55( float3 In) {
float iR, iG, iB;
iR = (In.x * 1023.0f - 64.0f) / 876.0f;
iG = (In.y * 1023.0f - 64.0f) / 876.0f;
iB = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 pmtx;
pmtx.x = 1.08190037262167f * iR -0.180298701368782f * iG +0.0983983287471069f * iB
+1.9458545364518f * iR*iG -0.509539936937375f * iG*iB -0.47489567735516f * iB*iR
-0.778086752197068f * iR*iR -0.7412266070049f * iG*iG +0.557894437042701f * iB*iB
-3.27787395719078f * iR*iR*iG +0.254878417638717f * iR*iR*iB +3.45581530576474f * iR*iG*iG
+0.335471713974739f * iR*iG*iB -0.43352125478476f * iR*iB*iB -1.65050137344141f * iG*iG*iB +1.46581418175682f * iG*iB*iB
+0.944646566605676f * iR*iR*iR -0.723653099155881f * iG*iG*iG -0.371076501167857f * iB*iB*iB;
pmtx.y = -0.00858997792576314f * iR +1.00673740119621f * iG +0.00185257672955608f * iB
+0.0848736138296452f * iR*iG +0.347626906448902f * iG*iB +0.0020230274463939f * iB*iR
-0.0790508414091524f * iR*iR -0.179497582958716f * iG*iG -0.175975123357072f * iB*iB
+2.30205579706951f * iR*iR*iG -0.627257613385219f * iR*iR*iB -2.90795250918851f * iR*iG*iG
+1.37002437502321f * iR*iG*iB -0.108668158565563f * iR*iB*iB -2.21150552827555f * iG*iG*iB + 1.53315057595445f * iG*iB*iB
-0.543188706699505f * iR*iR*iR +1.63793038490376f * iG*iG*iG -0.444588616836587f * iB*iB*iB;
pmtx.z = 0.12696639806511f * iR -0.011891441127869f * iG +0.884925043062759f * iB
+1.34780279822258f * iR*iG +1.03647352257365f * iG*iB +0.459113289955922f * iB*iR
-0.878157422295268f * iR*iR -1.3066278750436f * iG*iG -0.658604313413283f * iB*iB
-1.4444077996703f * iR*iR*iG +0.556676588785173f * iR*iR*iB +2.18798497054968f * iR*iG*iG
-1.43030768398665f * iR*iG*iB -0.0388323570817641f * iR*iB*iB +2.63698573112453f * iG*iG*iB -1.66598882056039f * iG*iB*iB
+0.33450249360103f * iR*iR*iR -1.65856930730901f * iG*iG*iG +0.521956184547685f * iB*iB*iB;
float3 lin;
lin.x = CanonLog_to_lin( pmtx.x);
lin.y = CanonLog_to_lin( pmtx.y);
lin.z = CanonLog_to_lin( pmtx.z);
float3 aces;
aces.x = 0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926f * lin.z;
aces.y = 0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821f * lin.z;
aces.z = 0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204f * lin.z;
return aces;
}

static float3 IDT_Canon_C100_A_Tng( float3 In) {
float iR, iG, iB;
iR = (In.x * 1023.0f - 64.0f) / 876.0f;
iG = (In.y * 1023.0f - 64.0f) / 876.0f;
iB = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 pmtx;
pmtx.x = 0.963803004454899f * iR - 0.160722202570655f * iG + 0.196919198115756f * iB
+2.03444685639819f * iR*iG - 0.442676931451021f * iG*iB - 0.407983781537509f * iB*iR
-0.640703323129254f * iR*iR - 0.860242798247848f * iG*iG + 0.317159977967446f * iB*iB
-4.80567080102966f * iR*iR*iG + 0.27118370397567f * iR*iR*iB + 5.1069005049557f * iR*iG*iG
+0.340895816920585f * iR*iG*iB - 0.486941738507862f * iR*iB*iB - 2.23737935753692f * iG*iG*iB + 1.96647555251297f * iG*iB*iB
+1.30204051766243f * iR*iR*iR - 1.06503117628554f * iG*iG*iG - 0.392473022667378f * iB*iB*iB;
pmtx.y = -0.0421935892309314f * iR +1.04845959175183f * iG - 0.00626600252090315f * iB
-0.106438896887216f * iR*iG + 0.362908621470781f * iG*iB + 0.118070700472261f * iB*iR
+0.0193542539838734f * iR*iR - 0.156083029543267f * iG*iG - 0.237811649496433f * iB*iB
+1.67916420582198f * iR*iR*iG - 0.632835327167897f * iR*iR*iB - 1.95984471387461f * iR*iG*iG
+0.953221464562814f * iR*iG*iB + 0.0599085176294623f * iR*iB*iB - 1.66452046236246f * iG*iG*iB + 1.14041188349761f * iG*iB*iB
-0.387552623550308f * iR*iR*iR + 1.14820099685512f * iG*iG*iG - 0.336153941411709f * iB*iB*iB;
pmtx.z = 0.170295033135028f * iR - 0.0682984448537245f * iG + 0.898003411718697f * iB
+1.22106821992399f * iR*iG + 1.60194865922925f * iG*iB + 0.377599191137124f * iB*iR
-0.825781428487531f * iR*iR - 1.44590868076749f * iG*iG - 0.928925961035344f * iB*iB
-0.838548997455852f * iR*iR*iG + 0.75809397217116f * iR*iR*iB + 1.32966795243196f * iR*iG*iG
-1.20021905668355f * iR*iG*iB - 0.254838995845129f * iR*iB*iB + 2.33232411639308f * iG*iG*iB - 1.86381505762773f * iG*iB*iB
+0.111576038956423f * iR*iR*iR - 1.12593315849766f * iG*iG*iG + 0.751693186157287f * iB*iB*iB;
float3 lin;
lin.x = CanonLog_to_lin( pmtx.x);
lin.y = CanonLog_to_lin( pmtx.y);
lin.z = CanonLog_to_lin( pmtx.z);
float3 aces;
aces.x = 0.566996399f * lin.x + 0.365079418f * lin.y + 0.067924183f * lin.z;
aces.y = 0.070901044f * lin.x + 0.880331008f * lin.y + 0.048767948f * lin.z;
aces.z = 0.073013542f * lin.x - 0.066540862f * lin.y + 0.99352732f * lin.z;
return aces;
}

static float3 IDT_Canon_C100mk2_A_D55( float3 In) {
float iR, iG, iB;
iR = (In.x * 1023.0f - 64.0f) / 876.0f;
iG = (In.y * 1023.0f - 64.0f) / 876.0f;
iB = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 pmtx;
pmtx.x = 1.08190037262167f * iR -0.180298701368782f * iG +0.0983983287471069f * iB
+1.9458545364518f * iR*iG -0.509539936937375f * iG*iB -0.47489567735516f * iB*iR
-0.778086752197068f * iR*iR -0.7412266070049f * iG*iG +0.557894437042701f * iB*iB
-3.27787395719078f * iR*iR*iG +0.254878417638717f * iR*iR*iB +3.45581530576474f * iR*iG*iG
+0.335471713974739f * iR*iG*iB -0.43352125478476f * iR*iB*iB -1.65050137344141f * iG*iG*iB +1.46581418175682f * iG*iB*iB
+0.944646566605676f * iR*iR*iR -0.723653099155881f * iG*iG*iG -0.371076501167857f * iB*iB*iB;
pmtx.y = -0.00858997792576314f * iR +1.00673740119621f * iG +0.00185257672955608f * iB
+0.0848736138296452f * iR*iG +0.347626906448902f * iG*iB +0.0020230274463939f * iB*iR
-0.0790508414091524f * iR*iR -0.179497582958716f * iG*iG -0.175975123357072f * iB*iB
+2.30205579706951f * iR*iR*iG -0.627257613385219f * iR*iR*iB -2.90795250918851f * iR*iG*iG
+1.37002437502321f * iR*iG*iB -0.108668158565563f * iR*iB*iB -2.21150552827555f * iG*iG*iB + 1.53315057595445f * iG*iB*iB
-0.543188706699505f * iR*iR*iR +1.63793038490376f * iG*iG*iG -0.444588616836587f * iB*iB*iB;
pmtx.z = 0.12696639806511f * iR -0.011891441127869f * iG +0.884925043062759f * iB
+1.34780279822258f * iR*iG +1.03647352257365f * iG*iB +0.459113289955922f * iB*iR
-0.878157422295268f * iR*iR -1.3066278750436f * iG*iG -0.658604313413283f * iB*iB
-1.4444077996703f * iR*iR*iG +0.556676588785173f * iR*iR*iB +2.18798497054968f * iR*iG*iG
-1.43030768398665f * iR*iG*iB -0.0388323570817641f * iR*iB*iB +2.63698573112453f * iG*iG*iB -1.66598882056039f * iG*iB*iB
+0.33450249360103f * iR*iR*iR -1.65856930730901f * iG*iG*iG +0.521956184547685f * iB*iB*iB;
float3 lin;
lin.x = CanonLog_to_lin( (pmtx.x * 876.0f + 64.0f) / 1023.0f );
lin.y = CanonLog_to_lin( (pmtx.y * 876.0f + 64.0f) / 1023.0f );
lin.z = CanonLog_to_lin( (pmtx.z * 876.0f + 64.0f) / 1023.0f );
float3 aces;
aces.x = 0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926f * lin.z;
aces.y = 0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821f * lin.z;
aces.z = 0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204f * lin.z;
return aces;
}

static float3 IDT_Canon_C100mk2_A_Tng( float3 In) {
float iR, iG, iB;
iR = (In.x * 1023.0f - 64.0f) / 876.0f;
iG = (In.y * 1023.0f - 64.0f) / 876.0f;
iB = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 pmtx;
pmtx.x = 0.963803004454899f * iR -0.160722202570655f * iG +0.196919198115756f * iB
+2.03444685639819f * iR*iG -0.442676931451021f * iG*iB -0.407983781537509f * iB*iR
-0.640703323129254f * iR*iR -0.860242798247848f * iG*iG +0.317159977967446f * iB*iB
-4.80567080102966f * iR*iR*iG +0.27118370397567f * iR*iR*iB +5.1069005049557f * iR*iG*iG
+0.340895816920585f * iR*iG*iB -0.486941738507862f * iR*iB*iB -2.23737935753692f * iG*iG*iB +1.96647555251297f * iG*iB*iB
+1.30204051766243f * iR*iR*iR -1.06503117628554f * iG*iG*iG -0.392473022667378f * iB*iB*iB;
pmtx.y = -0.0421935892309314f * iR +1.04845959175183f * iG -0.00626600252090315f * iB
-0.106438896887216f * iR*iG +0.362908621470781f * iG*iB +0.118070700472261f * iB*iR
+0.0193542539838734f * iR*iR -0.156083029543267f * iG*iG -0.237811649496433f * iB*iB
+1.67916420582198f * iR*iR*iG -0.632835327167897f * iR*iR*iB -1.95984471387461f * iR*iG*iG
+0.953221464562814f * iR*iG*iB +0.0599085176294623f * iR*iB*iB -1.66452046236246f * iG*iG*iB +1.14041188349761f * iG*iB*iB
-0.387552623550308f * iR*iR*iR +1.14820099685512f * iG*iG*iG -0.336153941411709f * iB*iB*iB;
pmtx.z = 0.170295033135028f * iR -0.0682984448537245f * iG +0.898003411718697f * iB
+1.22106821992399f * iR*iG +1.60194865922925f * iG*iB +0.377599191137124f * iB*iR
-0.825781428487531f * iR*iR -1.44590868076749f * iG*iG -0.928925961035344f * iB*iB
-0.838548997455852f * iR*iR*iG +0.75809397217116f * iR*iR*iB +1.32966795243196f * iR*iG*iG
-1.20021905668355f * iR*iG*iB -0.254838995845129f * iR*iB*iB +2.33232411639308f * iG*iG*iB -1.86381505762773f * iG*iB*iB
+0.111576038956423f * iR*iR*iR -1.12593315849766f * iG*iG*iG +0.751693186157287f * iB*iB*iB;
float3 lin;
lin.x = CanonLog_to_lin( (pmtx.x * 876.0f + 64.0f) / 1023.0f );
lin.y = CanonLog_to_lin( (pmtx.y * 876.0f + 64.0f) / 1023.0f );
lin.z = CanonLog_to_lin( (pmtx.z * 876.0f + 64.0f) / 1023.0f );
float3 aces;
aces.x = 0.566996399f * lin.x + 0.365079418f * lin.y + 0.067924183f * lin.z;
aces.y = 0.070901044f * lin.x + 0.880331008f * lin.y + 0.048767948f * lin.z;
aces.z = 0.073013542f * lin.x - 0.066540862f * lin.y + 0.99352732f * lin.z;
return aces;
}

static float3 IDT_Canon_C300_A_D55( float3 In) {
float iR, iG, iB;
iR = (In.x * 1023.0f - 64.0f) / 876.0f;
iG = (In.y * 1023.0f - 64.0f) / 876.0f;
iB = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 pmtx;
pmtx.x = 1.08190037262167f * iR -0.180298701368782f * iG +0.0983983287471069f * iB
+1.9458545364518f * iR*iG -0.509539936937375f * iG*iB -0.47489567735516f * iB*iR
-0.778086752197068f * iR*iR -0.7412266070049f * iG*iG +0.557894437042701f * iB*iB
-3.27787395719078f * iR*iR*iG +0.254878417638717f * iR*iR*iB +3.45581530576474f * iR*iG*iG
+0.335471713974739f * iR*iG*iB -0.43352125478476f * iR*iB*iB -1.65050137344141f * iG*iG*iB +1.46581418175682f * iG*iB*iB
+0.944646566605676f * iR*iR*iR -0.723653099155881f * iG*iG*iG -0.371076501167857f * iB*iB*iB;
pmtx.y = -0.00858997792576314f * iR +1.00673740119621f * iG +0.00185257672955608f * iB
+0.0848736138296452f * iR*iG +0.347626906448902f * iG*iB +0.0020230274463939f * iB*iR
-0.0790508414091524f * iR*iR -0.179497582958716f * iG*iG -0.175975123357072f * iB*iB
+2.30205579706951f * iR*iR*iG -0.627257613385219f * iR*iR*iB -2.90795250918851f * iR*iG*iG
+1.37002437502321f * iR*iG*iB -0.108668158565563f * iR*iB*iB -2.21150552827555f * iG*iG*iB + 1.53315057595445f * iG*iB*iB
-0.543188706699505f * iR*iR*iR +1.63793038490376f * iG*iG*iG -0.444588616836587f * iB*iB*iB;
pmtx.z = 0.12696639806511f * iR -0.011891441127869f * iG +0.884925043062759f * iB
+1.34780279822258f * iR*iG +1.03647352257365f * iG*iB +0.459113289955922f * iB*iR
-0.878157422295268f * iR*iR -1.3066278750436f * iG*iG -0.658604313413283f * iB*iB
-1.4444077996703f * iR*iR*iG +0.556676588785173f * iR*iR*iB +2.18798497054968f * iR*iG*iG
-1.43030768398665f * iR*iG*iB -0.0388323570817641f * iR*iB*iB +2.63698573112453f * iG*iG*iB -1.66598882056039f * iG*iB*iB
+0.33450249360103f * iR*iR*iR -1.65856930730901f * iG*iG*iG +0.521956184547685f * iB*iB*iB;
float3 lin;
lin.x = CanonLog_to_lin( pmtx.x);
lin.y = CanonLog_to_lin( pmtx.y);
lin.z = CanonLog_to_lin( pmtx.z);
float3 aces;
aces.x = 0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926 * lin.z;
aces.y = 0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821 * lin.z;
aces.z = 0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204 * lin.z;
return aces;
}

static float3 IDT_Canon_C300_A_Tng( float3 In) {
float iR, iG, iB;
iR = (In.x * 1023.0f - 64.0f) / 876.0f;
iG = (In.y * 1023.0f - 64.0f) / 876.0f;
iB = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 pmtx;
pmtx.x = 0.963803004454899f * iR -0.160722202570655f * iG +0.196919198115756f * iB
+2.03444685639819f * iR*iG -0.442676931451021f * iG*iB -0.407983781537509f * iB*iR
-0.640703323129254f * iR*iR -0.860242798247848f * iG*iG +0.317159977967446f * iB*iB
-4.80567080102966f * iR*iR*iG +0.27118370397567f * iR*iR*iB +5.1069005049557f * iR*iG*iG
+0.340895816920585f * iR*iG*iB -0.486941738507862f * iR*iB*iB -2.23737935753692f * iG*iG*iB +1.96647555251297f * iG*iB*iB
+1.30204051766243f * iR*iR*iR -1.06503117628554f * iG*iG*iG -0.392473022667378f * iB*iB*iB;
pmtx.y = -0.0421935892309314f * iR +1.04845959175183f * iG -0.00626600252090315f * iB
-0.106438896887216f * iR*iG +0.362908621470781f * iG*iB +0.118070700472261f * iB*iR
+0.0193542539838734f * iR*iR -0.156083029543267f * iG*iG -0.237811649496433f * iB*iB
+1.67916420582198f * iR*iR*iG -0.632835327167897f * iR*iR*iB -1.95984471387461f * iR*iG*iG
+0.953221464562814f * iR*iG*iB +0.0599085176294623f * iR*iB*iB -1.66452046236246f * iG*iG*iB +1.14041188349761f * iG*iB*iB
-0.387552623550308f * iR*iR*iR +1.14820099685512f * iG*iG*iG -0.336153941411709f * iB*iB*iB;
pmtx.z = 0.170295033135028f * iR -0.0682984448537245f * iG +0.898003411718697f * iB
+1.22106821992399f * iR*iG +1.60194865922925f * iG*iB +0.377599191137124f * iB*iR
-0.825781428487531f * iR*iR -1.44590868076749f * iG*iG -0.928925961035344f * iB*iB
-0.838548997455852f * iR*iR*iG +0.75809397217116f * iR*iR*iB +1.32966795243196f * iR*iG*iG
-1.20021905668355f * iR*iG*iB -0.254838995845129f * iR*iB*iB +2.33232411639308f * iG*iG*iB -1.86381505762773f * iG*iB*iB
+0.111576038956423f * iR*iR*iR -1.12593315849766f * iG*iG*iG +0.751693186157287f * iB*iB*iB;
float3 lin;
lin.x = CanonLog_to_lin( pmtx.x);
lin.y = CanonLog_to_lin( pmtx.y);
lin.z = CanonLog_to_lin( pmtx.z);
float3 aces;
aces.x = 0.566996399f * lin.x +0.365079418f * lin.y + 0.067924183f * lin.z;
aces.y = 0.070901044f * lin.x +0.880331008f * lin.y + 0.048767948f * lin.z;
aces.z = 0.073013542f * lin.x -0.066540862f * lin.y + 0.99352732f * lin.z;
return aces;
}

static float3 IDT_Canon_C500_A_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = CanonLog_to_lin( CLogIRE.x);
lin.y = CanonLog_to_lin( CLogIRE.y);
lin.z = CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.561538969f * lin.x +0.402060105f * lin.y + 0.036400926f * lin.z;
aces.y = 0.092739623f * lin.x +0.924121198f * lin.y - 0.016860821f * lin.z;
aces.z = 0.084812961f * lin.x +0.006373835f * lin.y + 0.908813204f * lin.z;
return aces;
}

static float3 IDT_Canon_C500_A_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = CanonLog_to_lin( CLogIRE.x);
lin.y = CanonLog_to_lin( CLogIRE.y);
lin.z = CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.566996399f * lin.x +0.365079418f * lin.y + 0.067924183f * lin.z;
aces.y = 0.070901044f * lin.x +0.880331008f * lin.y + 0.048767948f * lin.z;
aces.z = 0.073013542f * lin.x -0.066540862f * lin.y + 0.99352732f * lin.z;
return aces;
}

static float3 IDT_Canon_C500_B_D55( float3 In) {
float iR, iG, iB;
iR = (In.x * 1023.0f - 64.0f) / 876.0f;
iG = (In.y * 1023.0f - 64.0f) / 876.0f;
iB = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 pmtx;
pmtx.x = 1.08190037262167f * iR -0.180298701368782f * iG +0.0983983287471069f * iB
+1.9458545364518f * iR*iG -0.509539936937375f * iG*iB -0.47489567735516f * iB*iR
-0.778086752197068f * iR*iR -0.7412266070049f * iG*iG +0.557894437042701f * iB*iB
-3.27787395719078f * iR*iR*iG +0.254878417638717f * iR*iR*iB +3.45581530576474f * iR*iG*iG
+0.335471713974739f * iR*iG*iB -0.43352125478476f * iR*iB*iB -1.65050137344141f * iG*iG*iB + 1.46581418175682f * iG*iB*iB
+0.944646566605676f * iR*iR*iR -0.723653099155881f * iG*iG*iG -0.371076501167857f * iB*iB*iB;
pmtx.y = -0.00858997792576314f * iR +1.00673740119621f * iG +0.00185257672955608f * iB
+0.0848736138296452f * iR*iG +0.347626906448902f * iG*iB +0.0020230274463939f * iB*iR
-0.0790508414091524f * iR*iR -0.179497582958716f * iG*iG -0.175975123357072f * iB*iB
+2.30205579706951f * iR*iR*iG -0.627257613385219f * iR*iR*iB -2.90795250918851f * iR*iG*iG
+1.37002437502321f * iR*iG*iB -0.108668158565563f * iR*iB*iB -2.21150552827555f * iG*iG*iB + 1.53315057595445f * iG*iB*iB
-0.543188706699505f * iR*iR*iR +1.63793038490376f * iG*iG*iG -0.444588616836587f * iB*iB*iB;
pmtx.z = 0.12696639806511f * iR -0.011891441127869f * iG +0.884925043062759f * iB
+1.34780279822258f * iR*iG +1.03647352257365f * iG*iB +0.459113289955922f * iB*iR
-0.878157422295268f * iR*iR -1.3066278750436f * iG*iG -0.658604313413283f * iB*iB
-1.4444077996703f * iR*iR*iG +0.556676588785173f * iR*iR*iB +2.18798497054968f * iR*iG*iG
-1.43030768398665f * iR*iG*iB -0.0388323570817641f * iR*iB*iB +2.63698573112453f * iG*iG*iB - 1.66598882056039f * iG*iB*iB
+0.33450249360103f * iR*iR*iR -1.65856930730901f * iG*iG*iG +0.521956184547685f * iB*iB*iB;
float3 lin;
lin.x = CanonLog_to_lin( pmtx.x);
lin.y = CanonLog_to_lin( pmtx.y);
lin.z = CanonLog_to_lin( pmtx.z);
float3 aces;
aces.x = 0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926f * lin.z;
aces.y = 0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821f * lin.z;
aces.z = 0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204f * lin.z;
return aces;
}

static float3 IDT_Canon_C500_B_Tng( float3 In) {
float iR, iG, iB;
iR = (In.x * 1023.0f - 64.0f) / 876.0f;
iG = (In.y * 1023.0f - 64.0f) / 876.0f;
iB = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 pmtx;
pmtx.x = 0.963803004454899f * iR -0.160722202570655f * iG +0.196919198115756f * iB
+2.03444685639819f * iR*iG -0.442676931451021f * iG*iB -0.407983781537509f * iB*iR
-0.640703323129254f * iR*iR -0.860242798247848f * iG*iG +0.317159977967446f * iB*iB
-4.80567080102966f * iR*iR*iG +0.27118370397567f * iR*iR*iB +5.1069005049557f * iR*iG*iG
+0.340895816920585f * iR*iG*iB -0.486941738507862f * iR*iB*iB -2.23737935753692f * iG*iG*iB + 1.96647555251297f * iG*iB*iB
+1.30204051766243f * iR*iR*iR -1.06503117628554f * iG*iG*iG -0.392473022667378f * iB*iB*iB;
pmtx.y = -0.0421935892309314f * iR +1.04845959175183f * iG -0.00626600252090315f * iB
-0.106438896887216f * iR*iG +0.362908621470781f * iG*iB +0.118070700472261f * iB*iR
+0.0193542539838734f * iR*iR -0.156083029543267f * iG*iG -0.237811649496433f * iB*iB
+1.67916420582198f * iR*iR*iG -0.632835327167897f * iR*iR*iB -1.95984471387461f * iR*iG*iG
+0.953221464562814f * iR*iG*iB +0.0599085176294623f * iR*iB*iB -1.66452046236246f * iG*iG*iB + 1.14041188349761f * iG*iB*iB
-0.387552623550308f * iR*iR*iR +1.14820099685512f * iG*iG*iG -0.336153941411709f * iB*iB*iB;
pmtx.z = 0.170295033135028f * iR -0.0682984448537245f * iG +0.898003411718697f * iB
+1.22106821992399f * iR*iG +1.60194865922925f * iG*iB +0.377599191137124f * iB*iR
-0.825781428487531f * iR*iR -1.44590868076749f * iG*iG -0.928925961035344f * iB*iB
-0.838548997455852f * iR*iR*iG +0.75809397217116f * iR*iR*iB +1.32966795243196f * iR*iG*iG
-1.20021905668355f * iR*iG*iB -0.254838995845129f * iR*iB*iB +2.33232411639308f * iG*iG*iB - 1.86381505762773f * iG*iB*iB
+0.111576038956423f * iR*iR*iR -1.12593315849766f * iG*iG*iG +0.751693186157287f * iB*iB*iB;
float3 lin;
lin.x = CanonLog_to_lin( pmtx.x);
lin.y = CanonLog_to_lin( pmtx.y);
lin.z = CanonLog_to_lin( pmtx.z);
float3 aces;
aces.x = 0.566996399f * lin.x +0.365079418f * lin.y + 0.067924183f * lin.z;
aces.y = 0.070901044f * lin.x +0.880331008f * lin.y + 0.048767948f * lin.z;
aces.z = 0.073013542f * lin.x -0.066540862f * lin.y + 0.99352732f * lin.z;
return aces;
}

static float3 IDT_Canon_C500_CinemaGamut_A_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z;
aces.y = 0.003657457f * lin.x + 1.10696038f * lin.y - 0.110617837f * lin.z;
aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z;
return aces;
}

static float3 IDT_Canon_C500_CinemaGamut_A_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z;
aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z;
aces.z = 0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z;
return aces;
}

static float3 IDT_Canon_C500_DCI_P3_A_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.607160575f * lin.x + 0.299507286f * lin.y + 0.093332140f * lin.z;
aces.y = 0.004968120f * lin.x + 1.050982224f * lin.y - 0.055950343f * lin.z;
aces.z = -0.007839939f * lin.x + 0.000809127f * lin.y + 1.007030813f * lin.z;
return aces;
}

static float3 IDT_Canon_C500_DCI_P3_A_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.650279125f * lin.x + 0.253880169f * lin.y + 0.095840706f * lin.z;
aces.y = -0.026137986f * lin.x + 1.017900530f * lin.y + 0.008237456f * lin.z;
aces.z = 0.007757558f * lin.x - 0.063081669f * lin.y + 1.055324110f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog_BT2020_D_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.678891151f * lin.x + 0.158868422f * lin.y + 0.162240427f * lin.z;
aces.y = 0.045570831f * lin.x + 0.860712772f * lin.y + 0.093716397f * lin.z;
aces.z = -0.000485710f * lin.x + 0.025060196f * lin.y + 0.975425515f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.724488568f * lin.x + 0.115140904f * lin.y + 0.160370529f * lin.z;
aces.y = 0.010659276f * lin.x + 0.839605344f * lin.y + 0.149735380f * lin.z;
aces.z = 0.014560161f * lin.x - 0.028562057f * lin.y + 1.014001897f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z;
aces.y = 0.003657457f * lin.x + 1.10696038f * lin.y - 0.110617837f * lin.z;
aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z;
aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z;
aces.z = 0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog2_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog2_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog2_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.678891151f * lin.x + 0.158868422f * lin.y + 0.162240427f * lin.z;
aces.y = 0.045570831f * lin.x + 0.860712772f * lin.y + 0.093716397f * lin.z;
aces.z = -0.000485710f * lin.x + 0.025060196f * lin.y + 0.975425515f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog2_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog2_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog2_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.724488568f * lin.x + 0.115140904f * lin.y + 0.160370529f * lin.z;
aces.y = 0.010659276f * lin.x + 0.839605344f * lin.y + 0.149735380f * lin.z;
aces.z = 0.014560161f * lin.x - 0.028562057f * lin.y + 1.014001897f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog2_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog2_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog2_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z;
aces.y = 0.003657457f * lin.x + 1.10696038f * lin.y - 0.110617837f * lin.z;
aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog2_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog2_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog2_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z;
aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z;
aces.z = 0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog3_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog3_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog3_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.678891151f * lin.x + 0.158868422f * lin.y + 0.162240427f * lin.z;
aces.y = 0.045570831f * lin.x + 0.860712772f * lin.y + 0.093716397f * lin.z;
aces.z = -0.000485710f * lin.x + 0.025060196f * lin.y + 0.975425515f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog3_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog3_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog3_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.724488568f * lin.x + 0.115140904f * lin.y + 0.160370529f * lin.z;
aces.y = 0.010659276f * lin.x + 0.839605344f * lin.y + 0.149735380f * lin.z;
aces.z = 0.014560161f * lin.x - 0.028562057f * lin.y + 1.014001897f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog3_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog3_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog3_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z;
aces.y = 0.003657457f * lin.x + 1.10696038f * lin.y - 0.110617837f * lin.z;
aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z;
return aces;
}

static float3 IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng( float3 In) {
float3 CLogIRE;
CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;
float3 lin;
lin.x = 0.9f * CanonLog3_to_lin( CLogIRE.x);
lin.y = 0.9f * CanonLog3_to_lin( CLogIRE.y);
lin.z = 0.9f * CanonLog3_to_lin( CLogIRE.z);
float3 aces;
aces.x = 0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z;
aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z;
aces.z = 0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z;
return aces;
}

static float3 IDT_Sony_SLog1_SGamut( float3 In) {
mat3 SGAMUT_TO_ACES_MTX = { {0.754338638f, 0.021198141f, -0.009756991f}, {0.133697046f, 1.005410934f, 0.004508563f}, {0.111968437f, -0.026610548f, 1.005253201f} };
float B = 64.0f;
float AB = 90.0f;
float W = 940.0f;
float3 SLog;
SLog.x = In.x * 1023.0f;
SLog.y = In.y * 1023.0f;
SLog.z = In.z * 1023.0f;
float3 lin;
lin.x = SLog1_to_lin( SLog.x, B, AB, W);
lin.y = SLog1_to_lin( SLog.y, B, AB, W);
lin.z = SLog1_to_lin( SLog.z, B, AB, W);
float3 aces = mult_f3_f33( lin, SGAMUT_TO_ACES_MTX);
return aces;
}

static float3 IDT_Sony_SLog2_SGamut_Daylight( float3 In) {
mat3 SGAMUT_DAYLIGHT_TO_ACES_MTX = { {0.8764457030f, 0.0774075345f, 0.0573564351f}, {0.0145411681f, 0.9529571767f, -0.1151066335f}, {0.1090131290f, -0.0303647111f, 1.0577501984f} };
float B = 64.0f;
float AB = 90.0f;
float W = 940.0f;
float3 SLog;
SLog.x = In.x * 1023.0f;
SLog.y = In.y * 1023.0f;
SLog.z = In.z * 1023.0f;
float3 lin;
lin.x = SLog2_to_lin( SLog.x, B, AB, W);
lin.y = SLog2_to_lin( SLog.y, B, AB, W);
lin.z = SLog2_to_lin( SLog.z, B, AB, W);
float3 aces = mult_f3_f33( lin, SGAMUT_DAYLIGHT_TO_ACES_MTX);
return aces;
}

static float3 IDT_Sony_SLog2_SGamut_Tungsten( float3 In) {
mat3 SGAMUT_TUNG_TO_ACES_MTX = { { 1.0110238740f, 0.1011994504f, 0.0600766530f}, { -0.1362526051f, 0.9562196265f, -0.1010185315f}, { 0.1252287310f, -0.0574190769f, 1.0409418785f} };
float B = 64.0f;
float AB = 90.0f;
float W = 940.0f;
float3 SLog;
SLog.x = In.x * 1023.0f;
SLog.y = In.y * 1023.0f;
SLog.z = In.z * 1023.0f;
float3 lin;
lin.x = SLog2_to_lin( SLog.x, B, AB, W);
lin.y = SLog2_to_lin( SLog.y, B, AB, W);
lin.z = SLog2_to_lin( SLog.z, B, AB, W);
float3 aces = mult_f3_f33( lin, SGAMUT_TUNG_TO_ACES_MTX);
return aces;
}

static float3 IDT_Sony_SLog2_SGamut3Cine( float3 SLog2) {
mat3 matrixCoef = { {0.6387886672f, -0.0039159060f, -0.0299072021f}, {0.2723514337f, 1.0880732309f, -0.0264325799f}, {0.0888598991f, -0.0841573249f, 1.0563397820f} };
float B = 64.0f;
float AB = 90.0f;
float W = 940.0f;
float3 SLog;
SLog.x = SLog2.x * 1023.0f;
SLog.y = SLog2.y * 1023.0f;
SLog.z = SLog2.z * 1023.0f;
float3 linear;
linear.x = SLog2_to_lin( SLog.x, B, AB, W);
linear.y = SLog2_to_lin( SLog.y, B, AB, W);
linear.z = SLog2_to_lin( SLog.z, B, AB, W);
float3 aces = mult_f3_f33( linear, matrixCoef );
return aces;
}

static float3 IDT_Sony_SLog3_SGamut3( float3 SLog3) {
mat3 matrixCoef = { {0.7529825954f, 0.0217076974f, -0.0094160528f}, {0.1433702162f, 1.0153188355f, 0.0033704179f}, {0.1036471884f, -0.0370265329f, 1.0060456349f} };
float3 linear;
linear.x = SLog3_to_lin( SLog3.x );
linear.y = SLog3_to_lin( SLog3.y );
linear.z = SLog3_to_lin( SLog3.z );
float3 aces = mult_f3_f33( linear, matrixCoef );
return aces;
}

static float3 IDT_Sony_SLog3_SGamut3Cine( float3 SLog3) {
mat3 matrixCoef = { {0.6387886672f, -0.0039159060f, -0.0299072021f}, {0.2723514337f, 1.0880732309f, -0.0264325799f}, {0.0888598991f, -0.0841573249f, 1.0563397820f} };
float3 linear;
linear.x = SLog3_to_lin( SLog3.x );
linear.y = SLog3_to_lin( SLog3.y );
linear.z = SLog3_to_lin( SLog3.z );
float3 aces = mult_f3_f33( linear, matrixCoef );
return aces;
}

static float3 IDT_Sony_Venice_SGamut3( float3 linear) {
mat3 matrixCoeff = { {0.7933297411f, 0.0155810585f, -0.0188647478f}, {0.0890786256f, 1.0327123069f, 0.0127694121f}, {0.1175916333f, -0.0482933654f, 1.0060953358f} };
float3 aces = mult_f3_f33(linear, matrixCoeff);
return aces;
}

static float3 IDT_Sony_Venice_SGamut3Cine( float3 linear) {
mat3 matrixCoeff = { {0.6742570921f, -0.0093136061f, -0.0382090673f}, {0.2205717359f, 1.1059588614f, -0.0179383766f}, {0.1051711720f, -0.0966452553f, 1.0561474439f} };
float3 aces = mult_f3_f33(linear, matrixCoeff );
return aces;
}

static float Y_2_linCV( float Y, float Ymax, float Ymin) {
return (Y - Ymin) / (Ymax - Ymin);
}

static float linCV_2_Y( float linCV, float Ymax, float Ymin) {
return linCV * (Ymax - Ymin) + Ymin;
}

static float3 Y_2_linCV_f3( float3 Y, float Ymax, float Ymin) {
float3 linCV;
linCV.x = Y_2_linCV( Y.x, Ymax, Ymin); linCV.y = Y_2_linCV( Y.y, Ymax, Ymin); linCV.z = Y_2_linCV( Y.z, Ymax, Ymin);
return linCV;
}

static float3 linCV_2_Y_f3( float3 linCV, float Ymax, float Ymin) {
float3 Y;
Y.x = linCV_2_Y( linCV.x, Ymax, Ymin); Y.y = linCV_2_Y( linCV.y, Ymax, Ymin); Y.z = linCV_2_Y( linCV.z, Ymax, Ymin);
return Y;
}

static float3 darkSurround_to_dimSurround( float3 linearCV) {
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
float3 xyY = XYZ_2_xyY(XYZ);
xyY.z = fmaxf( xyY.z, 0.0f);
xyY.z = powf( xyY.z, DIM_SURROUND_GAMMA);
XYZ = xyY_2_XYZ(xyY);
return mult_f3_f33( XYZ, XYZtoRGB( AP1));
}

static float3 dimSurround_to_darkSurround( float3 linearCV) {
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
float3 xyY = XYZ_2_xyY(XYZ);
xyY.z = fmaxf( xyY.z, 0.0f);
xyY.z = powf( xyY.z, 1.0f / DIM_SURROUND_GAMMA);
XYZ = xyY_2_XYZ(xyY);
return mult_f3_f33( XYZ, XYZtoRGB( AP1));
}

static float roll_white_fwd( float in, float new_wht, float width) {
const float x0 = -1.0f;
const float x1 = x0 + width;
const float y0 = -new_wht;
const float y1 = x1;
const float m1 = (x1 - x0);
const float a = y0 - y1 + m1;
const float b = 2.0f * ( y1 - y0) - m1;
const float c = y0;
const float t = (-in - x0) / (x1 - x0);
float out = 0.0f;
if ( t < 0.0f)
out = -(t * b + c);
else if ( t > 1.0f)
out = in;
else
out = -(( t * a + b) * t + c);
return out;
}

static float roll_white_rev( float in, float new_wht, float width) {
const float x0 = -1.0f;
const float x1 = x0 + width;
const float y0 = -new_wht;
const float y1 = x1;
const float m1 = (x1 - x0);
const float a = y0 - y1 + m1;
const float b = 2.0f * ( y1 - y0) - m1;
float c = y0;
float out = 0.0f;
if ( -in < y0)
out = -x0;
else if ( -in > y1)
out = in;
else {
c = c + in;
const float discrim = sqrtf( b * b - 4.0f * a * c);
const float t = ( 2.0f * c) / ( -discrim - b);
out = -(( t * ( x1 - x0)) + x0);
}
return out;
}

static float lookup_ACESmin( float minLum ) {
float2 minTable[2] = { { log10f(MIN_LUM_RRT), MIN_STOP_RRT }, { log10f(MIN_LUM_SDR), MIN_STOP_SDR } };
return 0.18f * exp2f(interpolate1D( minTable, 2, log10f( minLum)));
}

static float lookup_ACESmax( float maxLum ) {
float2 maxTable[2] = { { log10f(MAX_LUM_SDR), MAX_STOP_SDR }, { log10f(MAX_LUM_RRT), MAX_STOP_RRT } };
return 0.18f * exp2f(interpolate1D( maxTable, 2, log10f( maxLum)));
}

static float5 init_coefsLow( TsPoint TsPointLow, TsPoint TsPointMid) {
float5 coefsLow;
float knotIncLow = (log10f(TsPointMid.x) - log10f(TsPointLow.x)) / 3.0f;
coefsLow.x = (TsPointLow.slope * (log10f(TsPointLow.x) - 0.5f * knotIncLow)) + ( log10f(TsPointLow.y) - TsPointLow.slope * log10f(TsPointLow.x));
coefsLow.y = (TsPointLow.slope * (log10f(TsPointLow.x) + 0.5f * knotIncLow)) + ( log10f(TsPointLow.y) - TsPointLow.slope * log10f(TsPointLow.x));
coefsLow.w = (TsPointMid.slope * (log10f(TsPointMid.x) - 0.5f * knotIncLow)) + ( log10f(TsPointMid.y) - TsPointMid.slope * log10f(TsPointMid.x));
coefsLow.m = (TsPointMid.slope * (log10f(TsPointMid.x) + 0.5f * knotIncLow)) + ( log10f(TsPointMid.y) - TsPointMid.slope * log10f(TsPointMid.x));
float2 bendsLow[2] = { {MIN_STOP_RRT, 0.18f}, {MIN_STOP_SDR, 0.35f} };
float pctLow = interpolate1D( bendsLow, 2, log2(TsPointLow.x / 0.18f));
coefsLow.z = log10f(TsPointLow.y) + pctLow*(log10f(TsPointMid.y) - log10f(TsPointLow.y));
return coefsLow;
}

static float5 init_coefsHigh( TsPoint TsPointMid, TsPoint TsPointMax) {
float5 coefsHigh;
float knotIncHigh = (log10f(TsPointMax.x) - log10f(TsPointMid.x)) / 3.0f;
coefsHigh.x = (TsPointMid.slope * (log10f(TsPointMid.x) - 0.5f * knotIncHigh)) + ( log10f(TsPointMid.y) - TsPointMid.slope * log10f(TsPointMid.x));
coefsHigh.y = (TsPointMid.slope * (log10f(TsPointMid.x) + 0.5f * knotIncHigh)) + ( log10f(TsPointMid.y) - TsPointMid.slope * log10f(TsPointMid.x));
coefsHigh.w = (TsPointMax.slope * (log10f(TsPointMax.x) - 0.5f * knotIncHigh)) + ( log10f(TsPointMax.y) - TsPointMax.slope * log10f(TsPointMax.x));
coefsHigh.m = (TsPointMax.slope * (log10f(TsPointMax.x) + 0.5f * knotIncHigh)) + ( log10f(TsPointMax.y) - TsPointMax.slope * log10f(TsPointMax.x));
float2 bendsHigh[2] = { {MAX_STOP_SDR, 0.89f}, {MAX_STOP_RRT, 0.90f} };
float pctHigh = interpolate1D( bendsHigh, 2, log2(TsPointMax.x / 0.18f));
coefsHigh.z = log10f(TsPointMid.y) + pctHigh*(log10f(TsPointMax.y) - log10f(TsPointMid.y));
return coefsHigh;
}

static float shift( float in, float expShift) {
return exp2f((log2f(in) - expShift));
}

static TsParams init_TsParams( float minLum, float maxLum, float expShift) {
TsPoint MIN_PT = { lookup_ACESmin(minLum), minLum, 0.0f};
TsPoint MID_PT = { 0.18f, 4.8f, 1.55f};
TsPoint MAX_PT = { lookup_ACESmax(maxLum), maxLum, 0.0f};
float5 cLow;
cLow = init_coefsLow( MIN_PT, MID_PT);
float5 cHigh;
cHigh = init_coefsHigh( MID_PT, MAX_PT);
MIN_PT.x = shift(lookup_ACESmin(minLum),expShift);
MID_PT.x = shift(0.18f, expShift);
MAX_PT.x = shift(lookup_ACESmax(maxLum),expShift);
TsParams P = { {MIN_PT.x, MIN_PT.y, MIN_PT.slope}, {MID_PT.x, MID_PT.y, MID_PT.slope},
{MAX_PT.x, MAX_PT.y, MAX_PT.slope}, {cLow.x, cLow.y, cLow.z, cLow.w, cLow.m, cLow.m},
{cHigh.x, cHigh.y, cHigh.z, cHigh.w, cHigh.m, cHigh.m} };
return P;
}

static float ssts( float x, TsParams C) {
const int N_KNOTS_LOW = 4;
const int N_KNOTS_HIGH = 4;
float logx = log10f( fmaxf(x, 1e-10f));
float logy = 0.0f;
if ( logx <= log10f(C.Min.x) ) {
logy = logx * C.Min.slope + ( log10f(C.Min.y) - C.Min.slope * log10f(C.Min.x) );
} else if (( logx > log10f(C.Min.x) ) && ( logx < log10f(C.Mid.x) )) {
float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10f(C.Min.x)) / (log10f(C.Mid.x) - log10f(C.Min.x));
int j = knot_coord;
float t = knot_coord - j;
float3 cf = { C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]};
float3 monomials = { t * t, t, 1.0f };
logy = dot_f3_f3( monomials, mult_f3_f33( cf, M1));
} else if (( logx >= log10f(C.Mid.x) ) && ( logx < log10f(C.Max.x) )) {
float knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10f(C.Mid.x)) / (log10f(C.Max.x) - log10f(C.Mid.x));
int j = knot_coord;
float t = knot_coord - j;
float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]};
float3 monomials = { t * t, t, 1.0f };
logy = dot_f3_f3( monomials, mult_f3_f33( cf, M1));
} else {
logy = logx * C.Max.slope + ( log10f(C.Max.y) - C.Max.slope * log10f(C.Max.x) );
}
return exp10f(logy);
}

static float inv_ssts( float y, TsParams C) {
const int N_KNOTS_LOW = 4;
const int N_KNOTS_HIGH = 4;
const float KNOT_INC_LOW = (log10f(C.Mid.x) - log10f(C.Min.x)) / (N_KNOTS_LOW - 1.0f);
const float KNOT_INC_HIGH = (log10f(C.Max.x) - log10f(C.Mid.x)) / (N_KNOTS_HIGH - 1.0f);
float KNOT_Y_LOW[ N_KNOTS_LOW];
for (int i = 0; i < N_KNOTS_LOW; i = i + 1) {
KNOT_Y_LOW[ i] = ( C.coefsLow[i] + C.coefsLow[i + 1]) / 2.0f;
};
float KNOT_Y_HIGH[ N_KNOTS_HIGH];
for (int i = 0; i < N_KNOTS_HIGH; i = i + 1) {
KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i + 1]) / 2.0f;
};
float logy = log10f( fmaxf(y, 1e-10f));
float logx;
if (logy <= log10f(C.Min.y)) {
logx = log10f(C.Min.x);
} else if ( (logy > log10f(C.Min.y)) && (logy <= log10f(C.Mid.y)) ) {
unsigned int j = 0;
float3 cf = make_float3(0.0f, 0.0f, 0.0f);
if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) {
cf.x = C.coefsLow[0]; cf.y = C.coefsLow[1]; cf.z = C.coefsLow[2]; j = 0;
} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
cf.x = C.coefsLow[1]; cf.y = C.coefsLow[2]; cf.z = C.coefsLow[3]; j = 1;
} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
cf.x = C.coefsLow[2]; cf.y = C.coefsLow[3]; cf.z = C.coefsLow[4]; j = 2;
}
const float3 tmp = mult_f3_f33( cf, M1);
float a = tmp.x; float b = tmp.y; float c = tmp.z;
c = c - logy;
const float d = sqrtf( b * b - 4.0f * a * c);
const float t = ( 2.0f * c) / ( -d - b);
logx = log10f(C.Min.x) + ( t + j) * KNOT_INC_LOW;
} else if ( (logy > log10f(C.Mid.y)) && (logy < log10f(C.Max.y)) ) {
unsigned int j = 0;
float3 cf = make_float3(0.0f, 0.0f, 0.0f);
if ( logy >= KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
cf.x = C.coefsHigh[0]; cf.y = C.coefsHigh[1]; cf.z = C.coefsHigh[2]; j = 0;
} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
cf.x = C.coefsHigh[1]; cf.y = C.coefsHigh[2]; cf.z = C.coefsHigh[3]; j = 1;
} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
cf.x = C.coefsHigh[2]; cf.y = C.coefsHigh[3]; cf.z = C.coefsHigh[4]; j = 2;
}
const float3 tmp = mult_f3_f33( cf, M1);
float a = tmp.x; float b = tmp.y; float c = tmp.z;
c = c - logy;
const float d = sqrtf( b * b - 4.0f * a * c);
const float t = ( 2.0f * c) / ( -d - b);
logx = log10f(C.Mid.x) + ( t + j) * KNOT_INC_HIGH;
} else {
logx = log10f(C.Max.x);
}
return exp10f( logx);
}

static float3 ssts_f3( float3 x, TsParams C) {
float3 out;
out.x = ssts( x.x, C); out.y = ssts( x.y, C); out.z = ssts( x.z, C);
return out;
}

static float3 inv_ssts_f3( float3 x, TsParams C) {
float3 out;
out.x = inv_ssts( x.x, C); out.y = inv_ssts( x.y, C); out.z = inv_ssts( x.z, C);
return out;
}

static float glow_fwd( float ycIn, float glowGainIn, float glowMid) {
float glowGainOut;
if (ycIn <= 2.0f/3.0f * glowMid) {
glowGainOut = glowGainIn;
} else if ( ycIn >= 2.0f * glowMid) {
glowGainOut = 0.0f;
} else {
glowGainOut = glowGainIn * (glowMid / ycIn - 1.0f/2.0f);
}
return glowGainOut;
}

static float glow_inv( float ycOut, float glowGainIn, float glowMid) {
float glowGainOut;
if (ycOut <= ((1.0f + glowGainIn) * 2.0f/3.0f * glowMid)) {
glowGainOut = -glowGainIn / (1.0f + glowGainIn);
} else if ( ycOut >= (2.0f * glowMid)) {
glowGainOut = 0.0f;
} else {
glowGainOut = glowGainIn * (glowMid / ycOut - 1.0f/2.0f) / (glowGainIn / 2.0f - 1.0f);
}
return glowGainOut;
}

static float sigmoid_shaper( float x) {
float t = fmaxf( 1.0f - fabs( x / 2.0f), 0.0f);
float y = 1.0f + _sign(x) * (1.0f - t * t);
return y / 2.0f;
}

static float cubic_basis_shaper ( float x, float w) {
float M[4][4] = { {-1.0f/6.0f, 3.0f/6.0f,-3.0f/6.0f, 1.0f/6.0f}, {3.0f/6.0f, -6.0f/6.0f, 3.0f/6.0f, 0.0f/6.0f},
{-3.0f/6.0f, 0.0f/6.0f, 3.0f/6.0f, 0.0f/6.0f}, {1.0f/6.0f, 4.0f/6.0f, 1.0f/6.0f, 0.0f/6.0f} };
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
y = 0.0f;}}
return y * 3.0f/2.0f;
}

static float center_hue( float hue, float centerH) {
float hueCentered = hue - centerH;
if (hueCentered < -180.0f) hueCentered = hueCentered + 360.0f;
else if (hueCentered > 180.0f) hueCentered = hueCentered - 360.0f;
return hueCentered;
}

static float uncenter_hue( float hueCentered, float centerH) {
float hue = hueCentered + centerH;
if (hue < 0.0f) hue = hue + 360.0f;
else if (hue > 360.0f) hue = hue - 360.0f;
return hue;
}

static float3 rrt_sweeteners( float3 in) {
float3 aces = in;
float saturation = rgb_2_saturation( aces);
float ycIn = rgb_2_yc(aces, 1.75f);
float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f);
float addedGlow = 1.0f + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
aces = mult_f_f3( addedGlow, aces);
float hue = rgb_2_hue( aces);
float centeredHue = center_hue( hue, RRT_RED_HUE);
float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);
aces.x = aces.x + hueWeight * saturation * (RRT_RED_PIVOT - aces.x) * (1.0f - RRT_RED_SCALE);
aces = max_f3_f( aces, 0.0f);
float3 rgbPre = mult_f3_f33( aces, AP0_2_AP1_MAT);
rgbPre = max_f3_f( rgbPre, 0.0f);
rgbPre = mult_f3_f33( rgbPre, calc_sat_adjust_matrix( RRT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
return rgbPre;
}

static float3 inv_rrt_sweeteners( float3 in) {
float3 rgbPost = in;
rgbPost = mult_f3_f33( rgbPost, invert_f33(calc_sat_adjust_matrix( RRT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
rgbPost = max_f3_f( rgbPost, 0.0f);
float3 aces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
aces = max_f3_f( aces, 0.0f);
float hue = rgb_2_hue( aces);
float centeredHue = center_hue( hue, RRT_RED_HUE);
float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);
float minChan;
if (centeredHue < 0.0f) {
minChan = aces.y;
} else {
minChan = aces.z;
}
float a = hueWeight * (1.0f - RRT_RED_SCALE) - 1.0f;
float b = aces.x - hueWeight * (RRT_RED_PIVOT + minChan) * (1.0f - RRT_RED_SCALE);
float c = hueWeight * RRT_RED_PIVOT * minChan * (1.0f - RRT_RED_SCALE);
aces.x = ( -b - sqrtf( b * b - 4.0f * a * c)) / ( 2.0f * a);
float saturation = rgb_2_saturation( aces);
float ycOut = rgb_2_yc(aces, 1.75f);
float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f);
float reducedGlow = 1.0f + glow_inv( ycOut, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
aces = mult_f_f3( ( reducedGlow), aces);
return aces;
}

static float3 limit_to_primaries( float3 XYZ, Chromaticities LIMITING_PRI) {
mat3 XYZ_2_LIMITING_PRI_MAT = XYZtoRGB( LIMITING_PRI);
mat3 LIMITING_PRI_2_XYZ_MAT = RGBtoXYZ( LIMITING_PRI);
float3 rgb = mult_f3_f33( XYZ, XYZ_2_LIMITING_PRI_MAT);
float3 limitedRgb = clamp_f3( rgb, 0.0f, 1.0f);
return mult_f3_f33( limitedRgb, LIMITING_PRI_2_XYZ_MAT);
}

static float3 dark_to_dim( float3 XYZ) {
float3 xyY = XYZ_2_xyY(XYZ);
xyY.z = fmaxf( xyY.z, 0.0f);
xyY.z = powf( xyY.z, DIM_SURROUND_GAMMA);
return xyY_2_XYZ(xyY);
}

static float3 dim_to_dark( float3 XYZ) {
float3 xyY = XYZ_2_xyY(XYZ);
xyY.z = fmaxf( xyY.z, 0.0f);
xyY.z = powf( xyY.z, 1.0f / DIM_SURROUND_GAMMA);
return xyY_2_XYZ(xyY);
}

static float3 outputTransform (
float3 in,
float Y_MIN,
float Y_MID,
float Y_MAX,
Chromaticities DISPLAY_PRI,
Chromaticities LIMITING_PRI,
int EOTF,
int SURROUND,
bool STRETCH_BLACK = true,
bool D60_SIM = false,
bool LEGAL_RANGE = false
)
{
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX, 0.0f);
float expShift = log2f(inv_ssts(Y_MID, PARAMS_DEFAULT)) - log2f(0.18f);
TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift);
float3 rgbPre = rrt_sweeteners( in);
float3 rgbPost = ssts_f3( rgbPre, PARAMS);
float3 linearCV = Y_2_linCV_f3( rgbPost, Y_MAX, Y_MIN);
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
if (SURROUND == 0) {
} else if (SURROUND == 1) {
if ((EOTF == 1) || (EOTF == 2) || (EOTF == 3)) {
XYZ = dark_to_dim( XYZ);
}} else if (SURROUND == 2) {
}
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
if (D60_SIM == false) {
if ((DISPLAY_PRI.white.x != AP0.white.x) && (DISPLAY_PRI.white.y != AP0.white.y)) {
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
}}
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
if (D60_SIM == true) {
float SCALE = 1.0f;
if ((DISPLAY_PRI.white.x == 0.3127f) && (DISPLAY_PRI.white.y == 0.329f)) {
SCALE = 0.96362f;
}
else if ((DISPLAY_PRI.white.x == 0.314f) && (DISPLAY_PRI.white.y == 0.351f)) {
linearCV.x = roll_white_fwd( linearCV.x, 0.918f, 0.5f);
linearCV.y = roll_white_fwd( linearCV.y, 0.918f, 0.5f);
linearCV.z = roll_white_fwd( linearCV.z, 0.918f, 0.5f);
SCALE = 0.96f;
}
linearCV = mult_f_f3( SCALE, linearCV);
}
linearCV = max_f3_f( linearCV, 0.0f);
float3 outputCV;
if (EOTF == 0) {
if (STRETCH_BLACK == true) {
outputCV = Y_2_ST2084_f3( max_f3_f( linCV_2_Y_f3(linearCV, Y_MAX, 0.0f), 0.0f) );
} else {
outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) );
}} else if (EOTF == 1) {
outputCV = bt1886_r_f3( linearCV, 2.4f, 1.0f, 0.0f);
} else if (EOTF == 2) {
outputCV = moncurve_r_f3( linearCV, 2.4f, 0.055f);
} else if (EOTF == 3) {
outputCV = pow_f3( linearCV, 1.0f/2.6f);
} else if (EOTF == 4) {
outputCV = linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN);
} else if (EOTF == 5) {
if (STRETCH_BLACK == true) {
outputCV = Y_2_ST2084_f3( max_f3_f( linCV_2_Y_f3(linearCV, Y_MAX, 0.0f), 0.0f) );
} else {
outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) );
}
outputCV = ST2084_2_HLG_1000nits_f3( outputCV);
}
if (LEGAL_RANGE == true) {
outputCV = fullRange_to_smpteRange_f3( outputCV);
}
return outputCV;
}

static float3 invOutputTransform (
float3 in,
float Y_MIN,
float Y_MID,
float Y_MAX,
Chromaticities DISPLAY_PRI,
Chromaticities LIMITING_PRI,
int EOTF,
int SURROUND,
bool STRETCH_BLACK = true,
bool D60_SIM = false,
bool LEGAL_RANGE = false
)
{
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX, 0.0f);
float expShift = log2f(inv_ssts(Y_MID, PARAMS_DEFAULT)) - log2f(0.18f);
TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift);
float3 outputCV = in;
if (LEGAL_RANGE == true) {
outputCV = smpteRange_to_fullRange_f3( outputCV);
}
float3 linearCV;
if (EOTF == 0) {
if (STRETCH_BLACK == true) {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0f);
} else {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN);
}} else if (EOTF == 1) {
linearCV = bt1886_f_f3( outputCV, 2.4f, 1.0f, 0.0f);
} else if (EOTF == 2) {
linearCV = moncurve_f_f3( outputCV, 2.4f, 0.055f);
} else if (EOTF == 3) {
linearCV = pow_f3( outputCV, 2.6f);
} else if (EOTF == 4) {
linearCV = Y_2_linCV_f3( outputCV, Y_MAX, Y_MIN);
} else if (EOTF == 5) {
outputCV = HLG_2_ST2084_1000nits_f3( outputCV);
if (STRETCH_BLACK == true) {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0f);
} else {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN);
}}
if (D60_SIM == true) {
float SCALE = 1.0f;
if ((DISPLAY_PRI.white.x == 0.3127f) && (DISPLAY_PRI.white.y == 0.329f)) {
SCALE = 0.96362f;
linearCV = mult_f_f3( 1.0f / SCALE, linearCV);
}
else if ((DISPLAY_PRI.white.x == 0.314f) && (DISPLAY_PRI.white.y == 0.351f)) {
SCALE = 0.96f;
linearCV.x = roll_white_rev( linearCV.x / SCALE, 0.918f, 0.5f);
linearCV.y = roll_white_rev( linearCV.y / SCALE, 0.918f, 0.5f);
linearCV.z = roll_white_rev( linearCV.z / SCALE, 0.918f, 0.5f);
}}
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
if (D60_SIM == false) {
if ((DISPLAY_PRI.white.x != AP0.white.x) && (DISPLAY_PRI.white.y != AP0.white.y)) {
XYZ = mult_f3_f33( XYZ, invert_f33(calculate_cat_matrix( AP0.white, REC709_PRI.white)) );
}}
if (SURROUND == 0) {
} else if (SURROUND == 1) {
if ((EOTF == 1) || (EOTF == 2) || (EOTF == 3)) {
XYZ = dim_to_dark( XYZ);
}} else if (SURROUND == 2) {
}
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
float3 rgbPost = linCV_2_Y_f3( linearCV, Y_MAX, Y_MIN);
float3 rgbPre = inv_ssts_f3( rgbPost, PARAMS);
float3 aces = inv_rrt_sweeteners( rgbPre);
return aces;
}

static float3 InvODT_Rec709( float3 outputCV) {
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(REC709_PRI);
float DISPGAMMA = 2.4f;
float L_W = 1.0f;
float L_B = 0.0f;
float3 linearCV = bt1886_f_f3( outputCV, DISPGAMMA, L_W, L_B);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)));
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
float3 rgbPre = linCV_2_Y_f3( linearCV, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost = segmented_spline_c9_rev_f3( rgbPre);
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_sRGB( float3 outputCV) {
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(REC709_PRI);
float DISPGAMMA = 2.4f;
float OFFSET = 0.055f;
float3 linearCV;
linearCV = moncurve_f_f3( outputCV, DISPGAMMA, OFFSET);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)));
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
float3 rgbPre = linCV_2_Y_f3( linearCV, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost = segmented_spline_c9_rev_f3( rgbPre);
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 IDT_sRGB( float3 rgb) {
float3 aces;
aces = InvODT_sRGB(rgb);
aces = segmented_spline_c5_rev_f3( aces);
aces = max_f3_f(aces, 0.0f);
return aces;
}

static float3 IDT_Rec709( float3 rgb) {
float3 aces;
aces = InvODT_Rec709(rgb);
aces = segmented_spline_c5_rev_f3( aces);
aces = max_f3_f(aces, 0.0f);
return aces;
}

static float3 ASCCDL_inACEScct ( float3 acesIn, float3 SLOPE, float3 OFFSET, float3 POWER, float SAT) {
float3 acescct = ACES_to_ACEScct( acesIn);
acescct.x = powf( clampf( (acescct.x * SLOPE.x) + OFFSET.x, 0.0f, 1.0f), 1.0f / POWER.x);
acescct.y = powf( clampf( (acescct.y * SLOPE.y) + OFFSET.y, 0.0f, 1.0f), 1.0f / POWER.y);
acescct.z = powf( clampf( (acescct.z * SLOPE.z) + OFFSET.z, 0.0f, 1.0f), 1.0f / POWER.z);
float luma = 0.2126f * acescct.x + 0.7152f * acescct.y + 0.0722f * acescct.z;
float satClamp = fmaxf(SAT, 0.0f);
acescct.x = luma + satClamp * (acescct.x - luma);
acescct.y = luma + satClamp * (acescct.y - luma);
acescct.z = luma + satClamp * (acescct.z - luma);
return ACEScct_to_ACES( acescct);
}

static float3 gamma_adjust_linear( float3 rgbIn, float GAMMA, float PIVOT) {
const float SCALAR = PIVOT / powf( PIVOT, GAMMA);
float3 rgbOut = rgbIn;
if (rgbIn.x > 0.0f) rgbOut.x = powf( rgbIn.x, GAMMA) * SCALAR;
if (rgbIn.y > 0.0f) rgbOut.y = powf( rgbIn.y, GAMMA) * SCALAR;
if (rgbIn.z > 0.0f) rgbOut.z = powf( rgbIn.z, GAMMA) * SCALAR;
return rgbOut;
}

static float3 sat_adjust( float3 rgbIn, float SAT_FACTOR) {
float3 RGB2Y = make_float3(RGBtoXYZ( REC709_PRI).c0.y, RGBtoXYZ( REC709_PRI).c1.y, RGBtoXYZ( REC709_PRI).c2.y);
const mat3 SAT_MAT = calc_sat_adjust_matrix( SAT_FACTOR, RGB2Y);
return mult_f3_f33( rgbIn, SAT_MAT);
}

static float3 rgb_2_yab( float3 rgb) {
float3 yab = mult_f3_f33( rgb, make_mat3(make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f), make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4)));
return yab;
}

static float3 yab_2_rgb( float3 yab) {
float3 rgb = mult_f3_f33( yab, invert_f33(make_mat3(make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f), make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4))));
return rgb;
}

static float3 yab_2_ych(float3 yab) {
float3 ych = yab;
float yo = yab.y * yab.y + yab.z * yab.z;
ych.y = sqrtf(yo);
ych.z = atan2f(yab.z, yab.y) * (180.0f / 3.141592653589793f);
if (ych.z < 0.0f) ych.z += 360.0f;
return ych;
}

static float3 ych_2_yab( float3 ych ) {
float3 yab;
yab.x = ych.x;
float h = ych.z * (3.141592653589793f / 180.0f);
yab.y = ych.y * cosf(h);
yab.z = ych.y * sinf(h);
return yab;
}

static float3 rgb_2_ych( float3 rgb) {
return yab_2_ych( rgb_2_yab( rgb));
}

static float3 ych_2_rgb( float3 ych) {
return yab_2_rgb( ych_2_yab( ych));
}

static float3 scale_C_at_H( float3 rgb, float centerH, float widthH, float percentC) {
float3 new_rgb = rgb;
float3 ych = rgb_2_ych( rgb);
if (ych.y > 0.0f) {
float centeredHue = center_hue( ych.z, centerH);
float f_H = cubic_basis_shaper( centeredHue, widthH);
if (f_H > 0.0f) {
float3 new_ych = ych;
new_ych.y = ych.y * (f_H * (percentC - 1.0f) + 1.0f);
new_rgb = ych_2_rgb( new_ych);
} else {
new_rgb = rgb;
}}
return new_rgb;
}

static float3 rotate_H_in_H( float3 rgb, float centerH, float widthH, float degreesShift) {
float3 ych = rgb_2_ych( rgb);
float3 new_ych = ych;
float centeredHue = center_hue( ych.z, centerH);
float f_H = cubic_basis_shaper( centeredHue, widthH);
float old_hue = centeredHue;
float new_hue = centeredHue + degreesShift;
float2 table[2] = { {0.0f, old_hue}, {1.0f, new_hue} };
float blended_hue = interpolate1D( table, 2, f_H);
if (f_H > 0.0f) new_ych.z = uncenter_hue(blended_hue, centerH);
return ych_2_rgb( new_ych);
}

static float3 scale_C( float3 rgb, float percentC) {
float3 ych = rgb_2_ych( rgb);
ych.y = ych.y * percentC;
return ych_2_rgb( ych);
}

static float3 overlay_f3( float3 a, float3 b) {
const float LUMA_CUT = lin_to_ACEScct( 0.5f);
float luma = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;
float3 out;
if (luma < LUMA_CUT) {
out.x = 2.0f * a.x * b.x;
out.y = 2.0f * a.y * b.y;
out.z = 2.0f * a.z * b.z;
} else {
out.x = 1.0f - (2.0f * (1.0f - a.x) * (1.0f - b.x));
out.y = 1.0f - (2.0f * (1.0f - a.y) * (1.0f - b.y));
out.z = 1.0f - (2.0f * (1.0f - a.z) * (1.0f - b.z));
}
return out;
}

static float3 LMT_PFE( float3 aces) {
aces = scale_C( aces, 0.7f);
float3 SLOPE = make_float3(1.0f, 1.0f, 0.94f);
float3 OFFSET = make_float3(0.0f, 0.0f, 0.02f);
float3 POWER = make_float3(1.0f, 1.0f, 1.0f);
aces = ASCCDL_inACEScct( aces, SLOPE, OFFSET, POWER, 1.0f);
aces = gamma_adjust_linear( aces, 1.5f, 0.18f);
aces = rotate_H_in_H( aces, 0.0f, 30.0f, 5.0f);
aces = rotate_H_in_H( aces, 80.0f, 60.0f, -15.0f);
aces = rotate_H_in_H( aces, 52.0f, 50.0f, -14.0f);
aces = scale_C_at_H( aces, 45.0f, 40.0f, 1.4f);
aces = rotate_H_in_H( aces, 190.0f, 40.0f, 30.0f);
aces = scale_C_at_H( aces, 240.0f, 120.0f, 1.4f);
return aces;
}

static float3 LMT_Bleach( float3 aces) {
float3 a, b, blend;
a = sat_adjust( aces, 0.9f);
a = mult_f_f3( 2.0f, a);
b = sat_adjust( aces, 0.0f);
b = gamma_adjust_linear( b, 1.2f, 0.18f);
a = ACES_to_ACEScct( a);
b = ACES_to_ACEScct( b);
blend = overlay_f3( a, b);
aces = ACEScct_to_ACES( blend);
return aces;
}

static float3 LMT_BlueLightArtifactFix( float3 aces) {
mat3 correctionMatrix =
{ {0.9404372683f, 0.0083786969f, 0.0005471261f },
{-0.0183068787f, 0.8286599939f, -0.0008833746f },
{ 0.0778696104f, 0.1629613092f, 1.0003362486f } };
float3 acesMod = mult_f3_f33( aces, correctionMatrix);
return acesMod;
}

static float compress( float dist, float lim, float thr, float pwr, bool invert) {
float comprDist;
float scl;
float nd;
float p;
if (dist < thr) {
comprDist = dist;
} else {
scl = (lim - thr) / powf(powf((1.0f - thr) / (lim - thr), -pwr) - 1.0f, 1.0f / pwr);
nd = (dist - thr) / scl;
p = powf(nd, pwr);
if (!invert) {
comprDist = thr + scl * nd / (powf(1.0f + p, 1.0f / pwr));
} else {
if (dist > (thr + scl)) {
comprDist = dist;
} else {
comprDist = thr + scl * powf(-(p / (p - 1.0f)), 1.0f / pwr);
}}}
return comprDist;
}

static float3 gamut_compress( float3 aces, float LIM_CYAN, float LIM_YELLOW, float LIM_MAGENTA, 
float THR_CYAN, float THR_YELLOW, float THR_MAGENTA, float PWR, bool invert) {
float3 linAP1 = mult_f3_f33(aces, AP0_2_AP1_MAT);
float ach = max_f3(linAP1);
float3 dist;
if (ach == 0.0f) {
dist = make_float3(0.0f, 0.0f, 0.0f);
} else {
dist.x = (ach - linAP1.x) / fabs(ach);
dist.y = (ach - linAP1.y) / fabs(ach);
dist.z = (ach - linAP1.z) / fabs(ach);
}
float3 comprDist;
comprDist.x = compress(dist.x, LIM_CYAN, THR_CYAN, PWR, invert);
comprDist.y = compress(dist.y, LIM_MAGENTA, THR_MAGENTA, PWR, invert);
comprDist.z = compress(dist.z, LIM_YELLOW, THR_YELLOW, PWR, invert);
float3 comprLinAP1;
comprLinAP1.x = ach - comprDist.x * fabs(ach);
comprLinAP1.y = ach - comprDist.y * fabs(ach);
comprLinAP1.z = ach - comprDist.z * fabs(ach);
aces = mult_f3_f33(comprLinAP1, AP1_2_AP0_MAT);
return aces;
}

static float3 LMT_GamutCompress( float3 aces) {
float LIM_CYAN =  1.147f;
float LIM_MAGENTA = 1.264f;
float LIM_YELLOW = 1.312f;
float THR_CYAN = 0.815;
float THR_MAGENTA = 0.803;
float THR_YELLOW = 0.880;
float PWR = 1.2f;
bool invert = false;
aces = gamut_compress(aces, LIM_CYAN, LIM_YELLOW, LIM_MAGENTA, 
THR_CYAN, THR_YELLOW, THR_MAGENTA, PWR, invert);
return aces;
}

static float3 h_RRT( float3 aces) {
float saturation = rgb_2_saturation( aces);
float ycIn = rgb_2_yc(aces, 1.75f);
float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f);
float addedGlow = 1.0f + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
aces = mult_f_f3( addedGlow, aces);
float hue = rgb_2_hue( aces);
float centeredHue = center_hue( hue, RRT_RED_HUE);
float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);
aces.x = aces.x + hueWeight * saturation * (RRT_RED_PIVOT - aces.x) * (1.0f - RRT_RED_SCALE);
aces = max_f3_f( aces, 0.0f);
float3 rgbPre = mult_f3_f33( aces, AP0_2_AP1_MAT);
rgbPre = max_f3_f( rgbPre, 0.0f);
rgbPre = mult_f3_f33( rgbPre, calc_sat_adjust_matrix( RRT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 rgbPost;
rgbPost.x = segmented_spline_c5_fwd( rgbPre.x);
rgbPost.y = segmented_spline_c5_fwd( rgbPre.y);
rgbPost.z = segmented_spline_c5_fwd( rgbPre.z);
float3 rgbOces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return rgbOces;
}

static float3 h_InvRRT( float3 oces) {
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c5_rev( rgbPre.x);
rgbPost.y = segmented_spline_c5_rev( rgbPre.y);
rgbPost.z = segmented_spline_c5_rev( rgbPre.z);
rgbPost = mult_f3_f33( rgbPost, invert_f33(calc_sat_adjust_matrix( RRT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
rgbPost = max_f3_f( rgbPost, 0.0f);
float3 aces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
aces = max_f3_f( aces, 0.0f);
float hue = rgb_2_hue( aces);
float centeredHue = center_hue( hue, RRT_RED_HUE);
float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);
float minChan;
if (centeredHue < 0.0f) {
minChan = aces.y;
} else {
minChan = aces.z;
}
float a = hueWeight * (1.0f - RRT_RED_SCALE) - 1.0f;
float b = aces.x - hueWeight * (RRT_RED_PIVOT + minChan) * (1.0f - RRT_RED_SCALE);
float c = hueWeight * RRT_RED_PIVOT * minChan * (1.0f - RRT_RED_SCALE);
aces.x = ( -b - sqrt( b * b - 4.0f * a * c)) / ( 2.0f * a);
float saturation = rgb_2_saturation( aces);
float ycOut = rgb_2_yc(aces, 1.75f);
float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f);
float reducedGlow = 1.0f + glow_inv( ycOut, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
aces = mult_f_f3( ( reducedGlow), aces);
return aces;
}

static float3 ODT_Rec709_100nits_dim( float3 oces) {
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float DISPGAMMA = 2.4f;
float L_W = 1.0f;
float L_B = 0.0f;
bool legalRange = false;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);
if(legalRange) outputCV = fullRange_to_smpteRange_f3( outputCV);
return outputCV;
}

static float3 ODT_Rec709_D60sim_100nits_dim( float3 oces) {
const Chromaticities DISPLAY_PRI = REC709_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.4f;
const float L_W = 1.0f;
const float L_B = 0.0f;
const float SCALE = 0.955f;
bool legalRange = false;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV.x = fminf( linearCV.x, 1.0f) * SCALE;
linearCV.y = fminf( linearCV.y, 1.0f) * SCALE;
linearCV.z = fminf( linearCV.z, 1.0f) * SCALE;
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);
if (legalRange)
outputCV = fullRange_to_smpteRange_f3( outputCV);
return outputCV;
}

static float3 ODT_Rec2020_100nits_dim( float3 oces) {
Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float DISPGAMMA = 2.4f;
float L_W = 1.0f;
float L_B = 0.0f;
bool legalRange = false;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);
if (legalRange)
outputCV = fullRange_to_smpteRange_f3( outputCV);
return outputCV;
}

static float3 ODT_Rec2020_ST2084_1000nits( float3 oces) {
Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_1000nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_1000nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_1000nits());
rgbPost = add_f_f3( -exp10f(-4.4550166483f), rgbPost);
float3 XYZ = mult_f3_f33( rgbPost, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
float3 rgb = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
rgb = max_f3_f( rgb, 0.0f);
float3 outputCV = Y_2_ST2084_f3( rgb);
return outputCV;
}

static float3 ODT_Rec2020_Rec709limited_100nits_dim( float3 oces) {
const Chromaticities DISPLAY_PRI = REC2020_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const Chromaticities LIMITING_PRI = REC709_PRI;
const float DISPGAMMA = 2.4f;
const float L_W = 1.0f;
const float L_B = 0.0f;
bool legalRange = false;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);
if (legalRange)
outputCV = fullRange_to_smpteRange_f3( outputCV);
return outputCV;
}

static float3 ODT_Rec2020_P3D65limited_100nits_dim( float3 oces) {
const Chromaticities DISPLAY_PRI = REC2020_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const Chromaticities LIMITING_PRI = P3D65_PRI;
const float DISPGAMMA = 2.4f;
const float L_W = 1.0f;
const float L_B = 0.0f;
bool legalRange = false;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);
if (legalRange)
outputCV = fullRange_to_smpteRange_f3( outputCV);
return outputCV;
}

static float3 ODT_sRGB_D60sim_100nits_dim( float3 oces) {
const Chromaticities DISPLAY_PRI = REC709_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const float DISPGAMMA = 2.4f;
const float OFFSET = 0.055f;
const float SCALE = 0.955f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV.x = fminf( linearCV.x, 1.0f) * SCALE;
linearCV.y = fminf( linearCV.y, 1.0f) * SCALE;
linearCV.z = fminf( linearCV.z, 1.0f) * SCALE;
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET);
outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET);
outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET);
return outputCV;
}

static float3 ODT_sRGB_100nits_dim( float3 oces) {
const Chromaticities DISPLAY_PRI = REC709_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const float DISPGAMMA = 2.4f;
const float OFFSET = 0.055f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET);
outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET);
outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET);
return outputCV;
}

static float3 ODT_P3DCI_48nits( float3 oces) {
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
const float NEW_WHT = 0.918f;
const float ROLL_WIDTH = 0.5f;
const float SCALE = 0.96f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH);
linearCV.x = fminf( linearCV.x, NEW_WHT) * SCALE;
linearCV.y = fminf( linearCV.y, NEW_WHT) * SCALE;
linearCV.z = fminf( linearCV.z, NEW_WHT) * SCALE;
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA);
return outputCV;
}

static float3 ODT_P3DCI_D60sim_48nits( float3 oces) {
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
const float NEW_WHT = 0.918;
const float ROLL_WIDTH = 0.5;
const float SCALE = 0.96;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH);
linearCV.x = fminf( linearCV.x, NEW_WHT) * SCALE;
linearCV.y = fminf( linearCV.y, NEW_WHT) * SCALE;
linearCV.z = fminf( linearCV.z, NEW_WHT) * SCALE;
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA);
return outputCV;
}

static float3 ODT_P3DCI_D65sim_48nits( float3 oces) {
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
const float NEW_WHT = 0.908f;
const float ROLL_WIDTH = 0.5f;
const float SCALE = 0.9575f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH);
linearCV.x = fminf( linearCV.x, NEW_WHT) * SCALE;
linearCV.y = fminf( linearCV.y, NEW_WHT) * SCALE;
linearCV.z = fminf( linearCV.z, NEW_WHT) * SCALE;
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA);
return outputCV;
}

static float3 ODT_P3D60_48nits( float3 oces) {
const Chromaticities DISPLAY_PRI = P3D60_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA);
return outputCV;
}

static float3 ODT_P3D65_48nits( float3 oces) {
const Chromaticities DISPLAY_PRI = P3D65_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA);
return outputCV;
}

static float3 ODT_P3D65_D60sim_48nits( float3 oces) {
const Chromaticities DISPLAY_PRI = P3D65_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
const float SCALE = 0.964f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV.x = fminf( linearCV.x, 1.0f) * SCALE;
linearCV.y = fminf( linearCV.y, 1.0f) * SCALE;
linearCV.z = fminf( linearCV.z, 1.0f) * SCALE;
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA);
return outputCV;
}

static float3 ODT_P3D65_Rec709limited_48nits( float3 oces) {
const Chromaticities DISPLAY_PRI = P3D65_PRI;
const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const Chromaticities LIMITING_PRI = REC709_PRI;
const float DISPGAMMA = 2.6f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA);
return outputCV;
}

static float3 ODT_DCDM( float3 oces) {
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = max_f3_f( XYZ, 0.0f);
float3 outputCV = dcdm_encode( XYZ);
return outputCV;
}

static float3 ODT_DCDM_P3D60limited( float3 oces) {
const Chromaticities LIMITING_PRI = P3D60_PRI;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
float3 outputCV = dcdm_encode( XYZ);
return outputCV;
}

static float3 ODT_DCDM_P3D65limited( float3 oces) {
const Chromaticities LIMITING_PRI = P3D65_PRI;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
float3 outputCV = dcdm_encode( XYZ);
return outputCV;
}

static float3 ODT_RGBmonitor_100nits_dim( float3 oces) {
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float DISPGAMMA = 2.4f;
float OFFSET = 0.055f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET);
outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET);
outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET);
return outputCV;
}

static float3 ODT_RGBmonitor_D60sim_100nits_dim( float3 oces) {
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float DISPGAMMA = 2.4f;
float OFFSET = 0.055f;
float SCALE = 0.955f;
float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
float3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10f(log10f(0.02f)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10f(log10f(0.02f)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10f(log10f(0.02f)));
linearCV.x = fminf( linearCV.x, 1.0f) * SCALE;
linearCV.y = fminf( linearCV.y, 1.0f) * SCALE;
linearCV.z = fminf( linearCV.z, 1.0f) * SCALE;
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)));
float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1));
linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT);
linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
float3 outputCV;
outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET);
outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET);
outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET);
return outputCV;
}

static float3 InvODT_Rec709_100nits_dim( float3 outputCV) {
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float DISPGAMMA = 2.4f;
float L_W = 1.0f;
float L_B = 0.0f;
bool legalRange = false;
if (legalRange)
outputCV = smpteRange_to_fullRange_f3( outputCV);
float3 linearCV;
linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B);
linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B);
linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)));
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
linearCV = dimSurround_to_darkSurround( linearCV);
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_Rec709_D60sim_100nits_dim( float3 outputCV) {
const Chromaticities DISPLAY_PRI = REC709_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.4f;
const float L_W = 1.0f;
const float L_B = 0.0f;
const float SCALE = 0.955f;
bool legalRange = false;
if (legalRange)
outputCV = smpteRange_to_fullRange_f3( outputCV);
float3 linearCV;
linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B);
linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B);
linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
linearCV = dimSurround_to_darkSurround( linearCV);
linearCV.x = linearCV.x / SCALE;
linearCV.y = linearCV.y / SCALE;
linearCV.z = linearCV.z / SCALE;
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_Rec2020_100nits_dim( float3 outputCV) {
Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float DISPGAMMA = 2.4f;
float L_W = 1.0f;
float L_B = 0.0f;
bool legalRange = false;
if (legalRange)
outputCV = smpteRange_to_fullRange_f3( outputCV);
float3 linearCV;
linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B);
linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B);
linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)));
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
linearCV = dimSurround_to_darkSurround( linearCV);
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_Rec2020_ST2084_1000nits( float3 outputCV) {
Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float3 rgb = ST2084_2_Y_f3( outputCV);
float3 XYZ = mult_f3_f33( rgb, DISPLAY_PRI_2_XYZ_MAT);
XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)));
float3 rgbPre = mult_f3_f33( XYZ, XYZtoRGB( AP1));
rgbPre = add_f_f3( exp10f(-4.4550166483f), rgbPre);
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_1000nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_1000nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_1000nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_sRGB_D60sim_100nits_dim( float3 outputCV) {
const Chromaticities DISPLAY_PRI = REC709_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.4f;
const float OFFSET = 0.055f;
const float SCALE = 0.955f;
float3 linearCV;
linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
linearCV = dimSurround_to_darkSurround( linearCV);
linearCV.x = linearCV.x / SCALE;
linearCV.y = linearCV.y / SCALE;
linearCV.z = linearCV.z / SCALE;
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_sRGB_100nits_dim( float3 outputCV) {
const Chromaticities DISPLAY_PRI = REC709_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.4f;
const float OFFSET = 0.055f;
float3 linearCV;
linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)));
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
linearCV = dimSurround_to_darkSurround( linearCV);
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_P3DCI_48nits( float3 outputCV) {
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
const float NEW_WHT = 0.918f;
const float ROLL_WIDTH = 0.5f;
const float SCALE = 0.96f;
float3 linearCV = pow_f3( outputCV, DISPGAMMA);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH);
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_P3DCI_D60sim_48nits( float3 outputCV) {
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
const float NEW_WHT = 0.918f;
const float ROLL_WIDTH = 0.5f;
const float SCALE = 0.96f;
float3 linearCV = pow_f3( outputCV, DISPGAMMA);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH);
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_P3DCI_D65sim_48nits( float3 outputCV) {
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
const float NEW_WHT = 0.908f;
const float ROLL_WIDTH = 0.5f;
const float SCALE = 0.9575f;
float3 linearCV = pow_f3( outputCV, DISPGAMMA);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH);
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_P3D60_48nits( float3 outputCV) {
const Chromaticities DISPLAY_PRI = P3D60_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
float3 linearCV = pow_f3( outputCV, DISPGAMMA);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_P3D65_48nits( float3 outputCV) {
const Chromaticities DISPLAY_PRI = P3D65_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
float3 linearCV = pow_f3( outputCV, DISPGAMMA);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_P3D65_D60sim_48nits( float3 outputCV) {
const Chromaticities DISPLAY_PRI = P3D65_PRI;
const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
const float DISPGAMMA = 2.6f;
const float SCALE = 0.964f;
float3 linearCV = pow_f3( outputCV, DISPGAMMA);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV.x = linearCV.x / SCALE;
linearCV.y = linearCV.y / SCALE;
linearCV.z = linearCV.z / SCALE;
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_DCDM( float3 outputCV) {
float3 XYZ = dcdm_decode( outputCV);
float3 linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_DCDM_P3D65limited( float3 outputCV) {
float3 XYZ = dcdm_decode( outputCV);
XYZ = mult_f3_f33( XYZ, invert_f33(calculate_cat_matrix( AP0.white, REC709_PRI.white)));
float3 linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_RGBmonitor_100nits_dim( float3 outputCV) {
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float DISPGAMMA = 2.4f;
float OFFSET = 0.055f;
float3 linearCV;
linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)));
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
linearCV = dimSurround_to_darkSurround( linearCV);
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 InvODT_RGBmonitor_D60sim_100nits_dim( float3 outputCV) {
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float DISPGAMMA = 2.4f;
float OFFSET = 0.055f;
float SCALE = 0.955f;
float3 linearCV;
linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);
float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT);
linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1));
linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))));
linearCV = dimSurround_to_darkSurround( linearCV);
linearCV.x = linearCV.x / SCALE;
linearCV.y = linearCV.y / SCALE;
linearCV.z = linearCV.z / SCALE;
float3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10f(log10f(0.02f)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10f(log10f(0.02f)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10f(log10f(0.02f)));
float3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());
float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT);
return oces;
}

static float3 RRTODT_P3D65_108nits_7_2nits_ST2084( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 7.2f;
float Y_MAX = 108.0f;
Chromaticities DISPLAY_PRI = P3D65_PRI;
Chromaticities LIMITING_PRI = P3D65_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_P3D65_1000nits_15nits_ST2084( float3 aces) {
const float Y_MIN = 0.0001;
const float Y_MID = 15.0;
const float Y_MAX = 1000.0;
const Chromaticities DISPLAY_PRI = P3D65_PRI;
const Chromaticities LIMITING_PRI = P3D65_PRI;
const int EOTF = 0;
const int SURROUND = 0;
const bool STRETCH_BLACK = true;
const bool D60_SIM = false;                       
const bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_P3D65_2000nits_15nits_ST2084( float3 aces) {
const float Y_MIN = 0.0001;
const float Y_MID = 15.0;
const float Y_MAX = 2000.0;
const Chromaticities DISPLAY_PRI = P3D65_PRI;
const Chromaticities LIMITING_PRI = P3D65_PRI;
const int EOTF = 0;
const int SURROUND = 0;
const bool STRETCH_BLACK = true;
const bool D60_SIM = false;                       
const bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_P3D65_4000nits_15nits_ST2084( float3 aces) {
const float Y_MIN = 0.0001;
const float Y_MID = 15.0;
const float Y_MAX = 4000.0;
const Chromaticities DISPLAY_PRI = P3D65_PRI;
const Chromaticities LIMITING_PRI = P3D65_PRI;
const int EOTF = 0;
const int SURROUND = 0;
const bool STRETCH_BLACK = true;
const bool D60_SIM = false;                       
const bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_Rec2020_1000nits_15nits_HLG( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 1000.0f;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 5;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_Rec2020_1000nits_15nits_ST2084( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 1000.0f;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_Rec2020_2000nits_15nits_ST2084( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 2000.0f;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_Rec2020_4000nits_15nits_ST2084( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 4000.0f;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_Rec709_100nits_10nits_BT1886( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 10.0f;
float Y_MAX = 100.0f;
Chromaticities DISPLAY_PRI = REC709_PRI;
Chromaticities LIMITING_PRI = REC709_PRI;
int EOTF = 1;
int SURROUND = 1;
bool STRETCH_BLACK = false;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 RRTODT_Rec709_100nits_10nits_sRGB( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 10.0f;
float Y_MAX = 100.0f;
Chromaticities DISPLAY_PRI = REC709_PRI;
Chromaticities LIMITING_PRI = REC709_PRI;
int EOTF = 2;
int SURROUND = 1;
bool STRETCH_BLACK = false;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 InvRRTODT_P3D65_108nits_7_2nits_ST2084( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 7.2f;
float Y_MAX = 108.0f;
Chromaticities DISPLAY_PRI = P3D65_PRI;
Chromaticities LIMITING_PRI = P3D65_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 InvRRTODT_P3D65_1000nits_15nits_ST2084( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 1000.0f;
Chromaticities DISPLAY_PRI = P3D65_PRI;
Chromaticities LIMITING_PRI = P3D65_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 InvRRTODT_P3D65_2000nits_15nits_ST2084( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 2000.0f;
Chromaticities DISPLAY_PRI = P3D65_PRI;
Chromaticities LIMITING_PRI = P3D65_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 InvRRTODT_P3D65_4000nits_15nits_ST2084( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 4000.0f;
Chromaticities DISPLAY_PRI = P3D65_PRI;
Chromaticities LIMITING_PRI = P3D65_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 InvRRTODT_Rec2020_1000nits_15nits_HLG( float3 cv) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 1000.0f;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 5;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return aces;
}

static float3 InvRRTODT_Rec2020_1000nits_15nits_ST2084( float3 cv) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 1000.0f;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return aces;
}

static float3 InvRRTODT_Rec2020_2000nits_15nits_ST2084( float3 cv) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 2000.0f;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return aces;
}

static float3 InvRRTODT_Rec2020_4000nits_15nits_ST2084( float3 cv) {
float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 4000.0f;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return aces;
}

static float3 InvRRTODT_Rec709_100nits_10nits_BT1886( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 10.0f;
float Y_MAX = 100.0f;
Chromaticities DISPLAY_PRI = REC709_PRI;
Chromaticities LIMITING_PRI = REC709_PRI;
int EOTF = 1;
int SURROUND = 1;
bool STRETCH_BLACK = false;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

static float3 InvRRTODT_Rec709_100nits_10nits_sRGB( float3 aces) {
float Y_MIN = 0.0001f;
float Y_MID = 10.0f;
float Y_MAX = 100.0f;
Chromaticities DISPLAY_PRI = REC709_PRI;
Chromaticities LIMITING_PRI = REC709_PRI;
int EOTF = 2;
int SURROUND = 1;
bool STRETCH_BLACK = false;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

#endif // ifndef __ACES_FUNCTIONS_CPU_H_INCLUDED__