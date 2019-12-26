#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

const char* kernelSource =  \
"#include <metal_stdlib> \n" \
"using namespace metal; \n" \
"typedef struct { \n" \
"float x, y, z, w, m; \n" \
"} Floater5; \n" \
"typedef struct { \n" \
"float2 c0, c1; \n" \
"} mat2; \n" \
"typedef struct { \n" \
"float3 c0, c1, c2; \n" \
"} mat3; \n" \
"typedef struct { \n" \
"float2 red; float2 green; float2 blue; float2 white; \n" \
"} Chromaticities; \n" \
"typedef struct { \n" \
"float x; float y; \n" \
"} SplineMapPoint; \n" \
"typedef struct { \n" \
"float coefsLow[6]; float coefsHigh[6]; \n" \
"SplineMapPoint minPoint; SplineMapPoint midPoint; SplineMapPoint maxPoint; \n" \
"float slopeLow; float slopeHigh; \n" \
"} SegmentedSplineParams_c5; \n" \
"typedef struct { \n" \
"float coefsLow[10]; float coefsHigh[10]; \n" \
"SplineMapPoint minPoint; SplineMapPoint midPoint; SplineMapPoint maxPoint; \n" \
"float slopeLow; float slopeHigh; \n" \
"} SegmentedSplineParams_c9; \n" \
"typedef struct { \n" \
"float x; float y; float slope; \n" \
"} TsPoint; \n" \
"typedef struct { \n" \
"TsPoint Min; TsPoint Mid; TsPoint Max; \n" \
"float coefsLow[6]; float coefsHigh[6]; \n" \
"} TsParams; \n" \
"#define REF_PT		((7120.0f - 1520.0f) / 8000.0f * (100.0f / 55.0f) - log10(0.18f)) * 1.0f \n" \
"__constant mat3 MM = { {0.5f, -1.0f, 0.5f}, {-1.0f, 1.0f, 0.5f}, {0.5f, 0.0f, 0.0f} }; \n" \
"__constant float TINY = 1e-10f; \n" \
"__constant float DIM_SURROUND_GAMMA = 0.9811f; \n" \
"__constant float ODT_SAT_FACTOR = 0.93f; \n" \
"__constant float MIN_STOP_SDR = -6.5f; \n" \
"__constant float MAX_STOP_SDR = 6.5f; \n" \
"__constant float MIN_STOP_RRT = -15.0f; \n" \
"__constant float MAX_STOP_RRT = 18.0f; \n" \
"__constant float MIN_LUM_SDR = 0.02f; \n" \
"__constant float MAX_LUM_SDR = 48.0f; \n" \
"__constant float MIN_LUM_RRT = 0.0001f; \n" \
"__constant float MAX_LUM_RRT = 10000.0f; \n" \
"__constant float RRT_GLOW_GAIN = 0.05f; \n" \
"__constant float RRT_GLOW_MID = 0.08f; \n" \
"__constant float RRT_RED_SCALE = 0.82f; \n" \
"__constant float RRT_RED_PIVOT = 0.03f; \n" \
"__constant float RRT_RED_HUE = 0.0f; \n" \
"__constant float RRT_RED_WIDTH = 135.0f; \n" \
"__constant float RRT_SAT_FACTOR = 0.96f; \n" \
"__constant float X_BRK = 0.0078125f; \n" \
"__constant float Y_BRK = 0.155251141552511f; \n" \
"__constant float A = 10.5402377416545f; \n" \
"__constant float B = 0.0729055341958355f; \n" \
"__constant float sqrt3over4 = 0.433012701892219f; \n" \
"__constant float pq_m1 = 0.1593017578125f; \n" \
"__constant float pq_m2 = 78.84375f; \n" \
"__constant float pq_c1 = 0.8359375f; \n" \
"__constant float pq_c2 = 18.8515625f; \n" \
"__constant float pq_c3 = 18.6875f; \n" \
"__constant float pq_C = 10000.0f; \n" \
"__constant mat3 CDD_TO_CID =  \n" \
"{ {0.75573f, 0.05901f, 0.16134f}, {0.22197f, 0.96928f, 0.07406f}, {0.02230f, -0.02829f, 0.76460f} }; \n" \
"__constant mat3 EXP_TO_ACES =  \n" \
"{ {0.72286f, 0.11923f, 0.01427f}, {0.12630f, 0.76418f, 0.08213f}, {0.15084f, 0.11659f, 0.90359f} }; \n" \
"__constant Chromaticities AP0 = \n" \
"{ {0.7347f, 0.2653f}, {0.0f, 1.0f}, {0.0001f, -0.077f}, {0.32168f, 0.33767f} }; \n" \
"__constant Chromaticities AP1 = \n" \
"{ {0.713f, 0.293f}, {0.165f, 0.83f}, {0.128f, 0.044f}, {0.32168f, 0.33767f} }; \n" \
"__constant Chromaticities REC709_PRI = \n" \
"{ {0.64f, 0.33f}, {0.3f, 0.6f}, {0.15f, 0.06f}, {0.3127f, 0.329f} }; \n" \
"__constant Chromaticities P3D60_PRI = \n" \
"{ {0.68f, 0.32f}, {0.265f, 0.69f}, {0.15f, 0.06f}, {0.32168, 0.33767f} }; \n" \
"__constant Chromaticities P3D65_PRI = \n" \
"{ {0.68f, 0.32f}, {0.265f, 0.69f}, {0.15f, 0.06f}, {0.3127f, 0.329f} }; \n" \
"__constant Chromaticities P3DCI_PRI = \n" \
"{ {0.68f, 0.32f}, {0.265f, 0.69f}, {0.15f, 0.06f}, {0.314f, 0.351f} }; \n" \
"__constant Chromaticities ARRI_ALEXA_WG_PRI = \n" \
"{ {0.684f, 0.313f}, {0.221f, 0.848f}, {0.0861f, -0.102f}, {0.3127f, 0.329f} }; \n" \
"__constant Chromaticities REC2020_PRI = \n" \
"{ {0.708f, 0.292f}, {0.17f, 0.797f}, {0.131f, 0.046f}, {0.3127f, 0.329f} }; \n" \
"__constant Chromaticities RIMMROMM_PRI = \n" \
"{ {0.7347f, 0.2653f}, {0.1596f, 0.8404f}, {0.0366f, 0.0001f}, {0.3457f, 0.3585f} }; \n" \
"__constant mat3 CONE_RESP_MATRADFORD = \n" \
"{ {0.8951f, -0.7502f, 0.0389f}, {0.2664f, 1.7135f, -0.0685f}, {-0.1614f, 0.0367f, 1.0296f} }; \n" \
"__constant mat3 CONE_RESP_MAT_CAT02 = \n" \
"{ {0.7328f, -0.7036f, 0.003f}, {0.4296f, 1.6975f, 0.0136f}, {-0.1624f, 0.0061f, 0.9834f} }; \n" \
"__constant mat3 AP0_2_XYZ_MAT =  \n" \
"{ {0.9525523959f, 0.3439664498f, 0.0f}, {0.0f, 0.7281660966f, 0.0f}, {0.0000936786f, -0.0721325464f, 1.0088251844f} }; \n" \
"__constant mat3 XYZ_2_AP0_MAT =  \n" \
"{ {1.0498110175f, -0.4959030231f, 0.0f}, {0.0f, 1.3733130458f, 0.0f}, {-0.0000974845f, 0.0982400361f, 0.9912520182f} }; \n" \
"__constant mat3 AP1_2_XYZ_MAT =  \n" \
"{ {0.6624541811f, 0.2722287168f, -0.0055746495f}, {0.1340042065f, 0.6740817658f, 0.0040607335f}, {0.156187687f, 0.0536895174f, 1.0103391003f} }; \n" \
"__constant mat3 XYZ_2_AP1_MAT =  \n" \
"{ {1.6410233797f, -0.6636628587f, 0.0117218943f}, {-0.3248032942f, 1.6153315917f, -0.008284442f}, {-0.2364246952f, 0.0167563477f, 0.9883948585f} }; \n" \
"__constant mat3 AP0_2_AP1_MAT =  \n" \
"{ {1.4514393161f, -0.0765537734f, 0.0083161484f}, {-0.2365107469f, 1.1762296998f, -0.0060324498f}, {-0.2149285693f, -0.0996759264f, 0.9977163014f} }; \n" \
"__constant mat3 AP1_2_AP0_MAT =  \n" \
"{ {0.6954522414f, 0.0447945634f, -0.0055258826f}, {0.1406786965f, 0.8596711185f, 0.0040252103f}, {0.1638690622f, 0.0955343182f, 1.0015006723f} }; \n" \
"__constant mat3 D60_2_D65_CAT =  \n" \
"{ {0.987224f, -0.00759836f, 0.00307257f}, {-0.00611327f, 1.00186f, -0.00509595f}, {0.0159533f, 0.00533002f, 1.08168f} }; \n" \
"__constant mat3 LMS_2_AP0_MAT =  \n" \
"{ { 2.2034860017f, -0.5267000086f, -0.0465914122f}, {-1.4028871323f,  1.5838401289f, -0.0457828327f}, { 0.1994183978f, -0.0571107433f, 1.0924829098f} }; \n" \
"__constant mat3 ICtCp_2_LMSp_MAT =  \n" \
"{ { 1.0f, 1.0f, 1.0f}, { 0.0086064753f, -0.0086064753f, 0.5600463058f}, { 0.1110335306f, -0.1110335306f, -0.3206319566f} }; \n" \
"__constant mat3 AP0_2_LMS_MAT =  \n" \
"{ { 0.5729360781f, 0.1916984459f, 0.0324676922f}, { 0.5052187675f, 0.8013733145f, 0.0551294631f}, {-0.0781710859f, 0.0069006377f, 0.9123015294f} }; \n" \
"__constant mat3 LMSp_2_ICtCp_MAT =  \n" \
"{ { 0.5f, 1.6137000085f, 4.378062447f}, { 0.5f, -3.3233961429f, -4.2455397991f}, { 0.0f, 1.7096961344f, -0.1325226479f} }; \n" \
"__constant mat3 SG3_2_AP0_MAT =  \n" \
"{ { 0.7529825954f, 0.0217076974f, -0.0094160528f}, { 0.1433702162f, 1.0153188355f, 0.0033704179f}, { 0.1036471884f, -0.0370265329f, 1.0060456349f} }; \n" \
"__constant mat3 AP0_2_SG3_MAT =  \n" \
"{ { 1.3316572111f, -0.0280131244f, 0.0125574528f}, {-0.1875611006f, 0.9887375645f, -0.0050679052f}, {-0.1440961106f, 0.0392755599f, 0.9925104526f} }; \n" \
"__constant mat3 SG3C_2_AP0_MAT =  \n" \
"{ { 0.6387886672f, -0.0039159061f, -0.0299072021f}, { 0.2723514337f, 1.0880732308f, -0.0264325799f}, { 0.0888598992f, -0.0841573249f, 1.056339782f} }; \n" \
"__constant mat3 AP0_2_SG3C_MAT =  \n" \
"{ { 1.5554591070f,  0.0090216145f, 0.0442640666f}, {-0.3932807985f, 0.9185569566f, 0.0118502607f}, {-0.1621783087f, 0.0724214290f, 0.9438856727f} }; \n" \
"__constant mat3 AWG_2_AP0_MAT =  \n" \
"{ { 0.6802059161f, 0.0854150695f, 0.0020562648f}, { 0.2361367500f, 1.0174707720f, -0.0625622837f}, { 0.0836574074f, -0.1028858550f, 1.0605062481f} }; \n" \
"__constant mat3 AP0_2_AWG_MAT =  \n" \
"{ { 1.5159863829f, -0.1283275799f, -0.0105107561f}, {-0.3613418588f, 1.0193145873f, 0.0608329325f}, {-0.1546444592f, 0.1090123949f, 0.9496764954f} }; \n" \
"__constant mat3 RWG_2_AP0_MAT =  \n" \
"{ { 0.7850585442f, 0.0231738066f, -0.0737605663f}, { 0.0838583156f, 1.0878975877f, -0.3145898729f}, { 0.1310821505f, -0.1110709153f, 1.3883506702f} }; \n" \
"__constant mat3 AP0_2_RWG_MAT =  \n" \
"{ { 1.2655392805f, -0.0205691227f, 0.0625750095f}, {-0.1352322515f,  0.9431709627f,  0.2065308369f}, {-0.1303056816f, 0.0773976700f, 0.7308939479f} }; \n" \
"__constant float3 AP1_RGB2Y = {0.2722287168f, 0.6740817658f, 0.0536895174f}; \n" \
"__constant float3 AP1_RGB2Y_B = {0.2722287168f, 0.6740817658f, 0.0536895174f}; \n" \
"inline Chromaticities make_chromaticities( float2 A, float2 B, float2 C, float2 D) { \n" \
"Chromaticities E; \n" \
"E.red = A; E.green = B; E.blue = C; E.white = D; \n" \
"return E; \n" \
"} \n" \
"inline float2 make_float2( float A, float B) { \n" \
"float2 out; \n" \
"out.x = A; out.y = B; \n" \
"return out; \n" \
"} \n" \
"inline float3 make_float3( float A, float B, float C) { \n" \
"float3 out; \n" \
"out.x = A; out.y = B; out.z = C; \n" \
"return out; \n" \
"} \n" \
"inline mat2 make_mat2( float2 A, float2 B) { \n" \
"mat2 C; \n" \
"C.c0 = A; C.c1 = B; \n" \
"return C; \n" \
"} \n" \
"inline mat3 make_mat3( float3 A, float3 B, float3 C) { \n" \
"mat3 D; \n" \
"D.c0 = A; D.c1 = B; D.c2 = C; \n" \
"return D; \n" \
"} \n" \
"inline float min_f3( float3 a) { \n" \
"return fmin( a.x, fmin( a.y, a.z)); \n" \
"} \n" \
"inline float max_f3( float3 a) { \n" \
"return fmax( a.x, fmax( a.y, a.z)); \n" \
"} \n" \
"inline float3 max_f3_f( float3 a, float b) { \n" \
"float3 out; \n" \
"out.x = fmax(a.x, b); out.y = fmax(a.y, b); out.z = fmax(a.z, b); \n" \
"return out; \n" \
"} \n" \
"inline float clip( float v) { \n" \
"return fmin(v, 1.0f); \n" \
"} \n" \
"inline float3 clip_f3( float3 in) { \n" \
"float3 out; \n" \
"out.x = clip( in.x); out.y = clip( in.y); out.z = clip( in.z); \n" \
"return out; \n" \
"} \n" \
"inline float3 add_f_f3( float a, float3 b) { \n" \
"float3 out; \n" \
"out.x = a + b.x; out.y = a + b.y; out.z = a + b.z; \n" \
"return out; \n" \
"} \n" \
"inline float3 pow_f3( float3 a, float b) { \n" \
"float3 out; \n" \
"out.x = pow(a.x, b); out.y = pow(a.y, b); out.z = pow(a.z, b); \n" \
"return out; \n" \
"} \n" \
"inline float3 exp10_f3( float3 a) { \n" \
"float3 out; \n" \
"out.x = exp10(a.x); out.y = exp10(a.y); out.z = exp10(a.z); \n" \
"return out; \n" \
"} \n" \
"inline float3 log10_f3( float3 a) { \n" \
"float3 out; \n" \
"out.x = log10(a.x); out.y = log10(a.y); out.z = log10(a.z); \n" \
"return out; \n" \
"} \n" \
"inline float _sign( float x) { \n" \
"float y; \n" \
"if (x < 0.0f) y = -1.0f; \n" \
"else if (x > 0.0f) y = 1.0f; \n" \
"else y = 0.0f; \n" \
"return y; \n" \
"} \n" \
"inline float3 mult_f_f3( float f, float3 x) { \n" \
"float3 r; \n" \
"r.x = f * x.x; r.y = f * x.y; r.z = f * x.z; \n" \
"return r; \n" \
"} \n" \
"inline float3 add_f3_f3( float3 x, float3 y) { \n" \
"float3 r; \n" \
"r.x = x.x + y.x; r.y = x.y + y.y; r.z = x.z + y.z; \n" \
"return r; \n" \
"} \n" \
"inline float3 sub_f3_f3( float3 x, float3 y) { \n" \
"float3 r; \n" \
"r.x = x.x - y.x; r.y = x.y - y.y; r.z = x.z - y.z; \n" \
"return r; \n" \
"} \n" \
"inline float3 cross_f3_f3( float3 x, float3 y) { \n" \
"float3 r; \n" \
"r.z = x.x * y.y - x.y * y.x; r.x = x.y * y.z - x.z * y.y; r.y = x.z * y.x - x.x * y.z; \n" \
"return r; \n" \
"} \n" \
"inline float3 clamp_f3( float3 A, float mn, float mx) { \n" \
"float3 out; \n" \
"out.x = clamp( A.x, mn, mx); out.y = clamp( A.y, mn, mx); out.z = clamp( A.z, mn, mx); \n" \
"return out; \n" \
"} \n" \
"inline float dot_f3_f3( float3 x, float3 y) { \n" \
"return x.x * y.x + x.y * y.y + x.z * y.z; \n" \
"} \n" \
"inline float length_f3( float3 x) { \n" \
"return sqrt( x.x * x.x + x.y * x.y + x.z * x.z ); \n" \
"} \n" \
"inline mat2 transpose_f22( mat2 A) { \n" \
"mat2 B; \n" \
"B.c0 = make_float2(A.c0.x, A.c1.x); B.c1 = make_float2(A.c0.y, A.c1.y); \n" \
"return B; \n" \
"} \n" \
"inline mat3 transpose_f33( mat3 A) { \n" \
"float r[3][3]; \n" \
"float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}}; \n" \
"for( int i = 0; i < 3; ++i){ \n" \
"for( int j = 0; j < 3; ++j){ \n" \
"r[i][j] = a[j][i];}} \n" \
"mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]),  \n" \
"make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2])); \n" \
"return R; \n" \
"} \n" \
"inline mat3 mult_f_f33( float f, mat3 A) { \n" \
"float r[3][3]; \n" \
"float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}}; \n" \
"for( int i = 0; i < 3; ++i ){ \n" \
"for( int j = 0; j < 3; ++j ){ \n" \
"r[i][j] = f * a[i][j];}} \n" \
"mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]),  \n" \
"make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2])); \n" \
"return R; \n" \
"} \n" \
"inline float3 mult_f3_f33( float3 X, mat3 A) { \n" \
"float r[3]; \n" \
"float x[3] = {X.x, X.y, X.z}; \n" \
"float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}}; \n" \
"for( int i = 0; i < 3; ++i){ \n" \
"r[i] = 0.0f; \n" \
"for( int j = 0; j < 3; ++j){ \n" \
"r[i] = r[i] + x[j] * a[j][i];}} \n" \
"return make_float3(r[0], r[1], r[2]); \n" \
"} \n" \
"inline mat3 mult_f33_f33( mat3 A, mat3 B) { \n" \
"float r[3][3]; \n" \
"float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, \n" \
"{A.c1.x, A.c1.y, A.c1.z}, \n" \
"{A.c2.x, A.c2.y, A.c2.z}}; \n" \
"float b[3][3] = {{B.c0.x, B.c0.y, B.c0.z}, \n" \
"{B.c1.x, B.c1.y, B.c1.z}, \n" \
"{B.c2.x, B.c2.y, B.c2.z}}; \n" \
"for( int i = 0; i < 3; ++i){ \n" \
"for( int j = 0; j < 3; ++j){ \n" \
"r[i][j] = 0.0f; \n" \
"for( int k = 0; k < 3; ++k){ \n" \
"r[i][j] = r[i][j] + a[i][k] * b[k][j]; \n" \
"}}} \n" \
"mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]),  \n" \
"make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2])); \n" \
"return R; \n" \
"} \n" \
"inline mat3 add_f33_f33( mat3 A, mat3 B) { \n" \
"float r[3][3]; \n" \
"float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}}; \n" \
"float b[3][3] = {{B.c0.x, B.c0.y, B.c0.z}, {B.c1.x, B.c1.y, B.c1.z}, {B.c2.x, B.c2.y, B.c2.z}}; \n" \
"for( int i = 0; i < 3; ++i ){ \n" \
"for( int j = 0; j < 3; ++j ){ \n" \
"r[i][j] = a[i][j] + b[i][j];}} \n" \
"mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]),  \n" \
"make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2])); \n" \
"return R; \n" \
"} \n" \
"inline mat3 invert_f33( mat3 A) { \n" \
"mat3 R; \n" \
"float result[3][3]; \n" \
"float a[3][3] = {{A.c0.x, A.c0.y, A.c0.z}, {A.c1.x, A.c1.y, A.c1.z}, {A.c2.x, A.c2.y, A.c2.z}}; \n" \
"float det = a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] \n" \
"+ a[0][2] * a[1][0] * a[2][1] - a[2][0] * a[1][1] * a[0][2] \n" \
"- a[2][1] * a[1][2] * a[0][0] - a[2][2] * a[1][0] * a[0][1]; \n" \
"if( det != 0.0f ){ \n" \
"result[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1]; result[0][1] = a[2][1] * a[0][2] - a[2][2] * a[0][1]; \n" \
"result[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1]; result[1][0] = a[2][0] * a[1][2] - a[1][0] * a[2][2]; \n" \
"result[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2]; result[1][2] = a[1][0] * a[0][2] - a[0][0] * a[1][2]; \n" \
"result[2][0] = a[1][0] * a[2][1] - a[2][0] * a[1][1]; result[2][1] = a[2][0] * a[0][1] - a[0][0] * a[2][1]; \n" \
"result[2][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1]; \n" \
"R = make_mat3(make_float3(result[0][0], result[0][1], result[0][2]), make_float3(result[1][0], result[1][1], \n" \
"result[1][2]), make_float3(result[2][0], result[2][1], result[2][2])); \n" \
"return mult_f_f33( 1.0f / det, R); \n" \
"} \n" \
"R = make_mat3(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f)); \n" \
"return R; \n" \
"} \n" \
"inline float interpolate1D( float2 table[], int Size, float p) { \n" \
"if( p <= table[0].x ) return table[0].y; \n" \
"if( p >= table[Size - 1].x ) return table[Size - 1].y; \n" \
"for( int i = 0; i < Size - 1; ++i ){ \n" \
"if( table[i].x <= p && p < table[i + 1].x ){ \n" \
"float s = (p - table[i].x) / (table[i + 1].x - table[i].x); \n" \
"return table[i].y * ( 1.0f - s ) + table[i+1].y * s;}} \n" \
"return 0.0f; \n" \
"} \n" \
"inline mat3 RGBtoXYZ( Chromaticities N) { \n" \
"mat3 M = make_mat3(make_float3(N.red.x, N.red.y, 1.0f - (N.red.x + N.red.y)), \n" \
"make_float3(N.green.x, N.green.y, 1.0f - (N.green.x + N.green.y)), make_float3(N.blue.x, N.blue.y, 1.0f - (N.blue.x + N.blue.y))); \n" \
"float3 wh = make_float3(N.white.x / N.white.y, 1.0f, (1.0f - (N.white.x + N.white.y)) / N.white.y); \n" \
"wh = mult_f3_f33(wh, invert_f33(M)); \n" \
"mat3 WH = make_mat3(make_float3(wh.x, 0.0f, 0.0f), make_float3(0.0f, wh.y, 0.0f), make_float3(0.0f, 0.0f, wh.z)); \n" \
"M = mult_f33_f33(WH, M); \n" \
"return M; \n" \
"} \n" \
"inline mat3 XYZtoRGB( Chromaticities N) { \n" \
"mat3 M = invert_f33(RGBtoXYZ(N)); \n" \
"return M; \n" \
"} \n" \
"float SLog3_to_lin( float SLog ) { \n" \
"float out = 0.0f; \n" \
"if (SLog >= 171.2102946929f / 1023.0f){ \n" \
"out = exp10((SLog * 1023.0f - 420.0f) / 261.5f) * (0.18f + 0.01f) - 0.01f; \n" \
"} else { \n" \
"out = (SLog * 1023.0f - 95.0f) * 0.01125000f / (171.2102946929f - 95.0f);} \n" \
"return out; \n" \
"} \n" \
"float lin_to_SLog3( float in) { \n" \
"float out; \n" \
"if (in >= 0.01125f) { \n" \
"out = (420.0f + log10((in + 0.01f) / (0.18f + 0.01f)) * 261.5f) / 1023.0f; \n" \
"} else { \n" \
"out = (in * (171.2102946929f - 95.0f) / 0.01125f + 95.0f) / 1023.0f; \n" \
"} \n" \
"return out; \n" \
"} \n" \
"float vLogToLinScene( float x) { \n" \
"const float cutInv = 0.181f; \n" \
"const float b = 0.00873f; \n" \
"const float c = 0.241514f; \n" \
"const float d = 0.598206f; \n" \
"if (x <= cutInv) \n" \
"return (x - 0.125f) / 5.6f; \n" \
"else \n" \
"return exp10((x - d) / c) - b; \n" \
"} \n" \
"float SLog1_to_lin( float SLog, float b, float ab, float w) { \n" \
"float lin = 0.0f; \n" \
"if (SLog >= ab) \n" \
"lin = ( exp10(( ( ( SLog - b) / ( w - b) - 0.616596f - 0.03f) / 0.432699f)) - 0.037584f) * 0.9f; \n" \
"else if (SLog < ab) \n" \
"lin = ( ( ( SLog - b) / ( w - b) - 0.030001222851889303f) / 5.0f) * 0.9f; \n" \
"return lin; \n" \
"} \n" \
"float SLog2_to_lin( float SLog, float b, float ab, float w) { \n" \
"float lin = 0.0f; \n" \
"if (SLog >= ab) \n" \
"lin = ( 219.0f * ( exp10(( ( ( SLog - b) / ( w - b) - 0.616596f - 0.03f) / 0.432699f)) - 0.037584f) / 155.0f) * 0.9f; \n" \
"else if (SLog < ab) \n" \
"lin = ( ( ( SLog - b) / ( w - b) - 0.030001222851889303f) / 3.53881278538813f) * 0.9f; \n" \
"return lin; \n" \
"} \n" \
"float CanonLog_to_lin ( float clog) { \n" \
"float out = 0.0f; \n" \
"if(clog < 0.12512248f) \n" \
"out = -( exp10(( 0.12512248f - clog ) / 0.45310179f ) - 1.0f ) / 10.1596f; \n" \
"else \n" \
"out = ( exp10(( clog - 0.12512248f ) / 0.45310179f ) - 1.0f ) / 10.1596f; \n" \
"return out; \n" \
"} \n" \
"float CanonLog2_to_lin ( float clog2) { \n" \
"float out = 0.0f; \n" \
"if(clog2 < 0.092864125f) \n" \
"out = -( exp10(( 0.092864125f - clog2 ) / 0.24136077f ) - 1.0f ) / 87.099375f; \n" \
"else \n" \
"out = ( exp10(( clog2 - 0.092864125f ) / 0.24136077f ) - 1.0f ) / 87.099375f; \n" \
"return out; \n" \
"} \n" \
"float CanonLog3_to_lin ( float clog3) { \n" \
"float out = 0.0f; \n" \
"if(clog3 < 0.097465473f) \n" \
"out = -( exp10(( 0.12783901f - clog3 ) / 0.36726845f ) - 1.0f ) / 14.98325f; \n" \
"else if(clog3 <= 0.15277891f) \n" \
"out = ( clog3 - 0.12512219f ) / 1.9754798f; \n" \
"else \n" \
"out = ( exp10(( clog3 - 0.12240537f ) / 0.36726845f ) - 1.0f ) / 14.98325f; \n" \
"return out; \n" \
"} \n" \
"float LogC_to_lin( float in) { \n" \
"const float midGraySignal = 0.01f; \n" \
"const float cut = 1.0f / 9.0f; \n" \
"const float slope = 3.9086503371f; \n" \
"const float offset =  -1.3885369913f; \n" \
"const float encOffset = 0.3855369987f; \n" \
"const float gain = 800.0f / 400.0f; \n" \
"const float encGain = 0.2471896383f; \n" \
"const float gray = 0.005f; \n" \
"const float nz = 0.052272275f; \n" \
"float out = (in - encOffset) / encGain; \n" \
"float ns = (out - offset) / slope; \n" \
"if (ns > cut) \n" \
"ns = exp10(out); \n" \
"ns = (ns - nz) * gray; \n" \
"return ns * (0.18f * gain / midGraySignal); \n" \
"} \n" \
"float lin_to_LogC( float in) { \n" \
"const float midGraySignal = 0.01f; \n" \
"const float cut = 1.0f / 9.0f; \n" \
"const float slope = 3.9086503371f; \n" \
"const float offset =  -1.3885369913f; \n" \
"const float encOffset = 0.3855369987f; \n" \
"const float gain = 800.0f / 400.0f; \n" \
"const float encGain = 0.2471896383f; \n" \
"const float gray = 0.005f; \n" \
"const float nz = 0.052272275f; \n" \
"float out; \n" \
"float ns = in / (0.18f * gain / midGraySignal); \n" \
"ns = nz + (ns / gray); \n" \
"if (ns > cut) { \n" \
"out = log10(ns); \n" \
"} else { \n" \
"out = offset + (ns * slope); \n" \
"} \n" \
"return encOffset + (out * encGain); \n" \
"} \n" \
"float Log3G10_to_lin ( float log3g10) { \n" \
"float a, b, c, g, linear; \n" \
"a = 0.224282f; b = 155.975327f; c = 0.01f; g = 15.1927f; \n" \
"linear = log3g10 < 0.0f ? (log3g10 / g) : (exp10(log3g10 / a) - 1.0f) / b; \n" \
"linear = linear - c; \n" \
"return linear; \n" \
"} \n" \
"float lin_to_Log3G10( float in) { \n" \
"const float a = 0.224282f; \n" \
"const float b = 155.975327f; \n" \
"const float c = 0.01f; \n" \
"const float g = 15.1927f; \n" \
"float out = in + c; \n" \
"if (out < 0.0f) { \n" \
"out =  out * g; \n" \
"} else { \n" \
"out = a * log10(out * b + 1.0f); \n" \
"} \n" \
"return out; \n" \
"} \n" \
"float3 inline XYZ_2_xyY( float3 XYZ) { \n" \
"float3 xyY; \n" \
"float divisor = (XYZ.x + XYZ.y + XYZ.z); \n" \
"if (divisor == 0.0f) divisor = 1e-10f; \n" \
"xyY.x = XYZ.x / divisor; \n" \
"xyY.y = XYZ.y / divisor; \n" \
"xyY.z = XYZ.y; \n" \
"return xyY; \n" \
"} \n" \
"float3 inline xyY_2_XYZ( float3 xyY) { \n" \
"float3 XYZ; \n" \
"XYZ.x = xyY.x * xyY.z / fmax( xyY.y, 1e-10f); \n" \
"XYZ.y = xyY.z; \n" \
"XYZ.z = (1.0f - xyY.x - xyY.y) * xyY.z / fmax( xyY.y, 1e-10f); \n" \
"return XYZ; \n" \
"} \n" \
"float inline rgb_2_hue( float3 rgb) { \n" \
"float hue = 0.0f; \n" \
"if (rgb.x == rgb.y && rgb.y == rgb.z) { \n" \
"hue = 0.0f; \n" \
"} else { \n" \
"hue = (180.0f/3.1415926535897932f) * atan2( sqrt(3.0f) * (rgb.y - rgb.z), 2.0f * rgb.x - rgb.y - rgb.z); \n" \
"} \n" \
"if (hue < 0.0f) hue = hue + 360.0f; \n" \
"return hue; \n" \
"} \n" \
"float inline rgb_2_yc( float3 rgb) { \n" \
"float ycRadiusWeight = 1.75f; \n" \
"float r = rgb.x; \n" \
"float g = rgb.y; \n" \
"float b = rgb.z; \n" \
"float chroma = sqrt(b * (b - g) + g * (g - r) + r * (r - b)); \n" \
"return ( b + g + r + ycRadiusWeight * chroma) / 3.0f; \n" \
"} \n" \
"mat3 calculate_cat_matrix( float2 src_xy, float2 des_xy) { \n" \
"mat3 coneRespMat = CONE_RESP_MATRADFORD; \n" \
"const float3 src_xyY = { src_xy.x, src_xy.y, 1.0f }; \n" \
"const float3 des_xyY = { des_xy.x, des_xy.y, 1.0f }; \n" \
"float3 src_XYZ = xyY_2_XYZ( src_xyY ); \n" \
"float3 des_XYZ = xyY_2_XYZ( des_xyY ); \n" \
"float3 src_coneResp = mult_f3_f33( src_XYZ, coneRespMat); \n" \
"float3 des_coneResp = mult_f3_f33( des_XYZ, coneRespMat); \n" \
"mat3 vkMat = { \n" \
"{ des_coneResp.x / src_coneResp.x, 0.0f, 0.0f }, \n" \
"{ 0.0f, des_coneResp.y / src_coneResp.y, 0.0f }, \n" \
"{ 0.0f, 0.0f, des_coneResp.z / src_coneResp.z }}; \n" \
"mat3 cat_matrix = mult_f33_f33( coneRespMat, mult_f33_f33( vkMat, invert_f33( coneRespMat ) ) ); \n" \
"return cat_matrix; \n" \
"} \n" \
"mat3 calc_sat_adjust_matrix( float sat, float3 rgb2Y) { \n" \
"float M[3][3]; \n" \
"M[0][0] = (1.0f - sat) * rgb2Y.x + sat; M[1][0] = (1.0f - sat) * rgb2Y.x; M[2][0] = (1.0f - sat) * rgb2Y.x; \n" \
"M[0][1] = (1.0f - sat) * rgb2Y.y; M[1][1] = (1.0f - sat) * rgb2Y.y + sat; M[2][1] = (1.0f - sat) * rgb2Y.y; \n" \
"M[0][2] = (1.0f - sat) * rgb2Y.z; M[1][2] = (1.0f - sat) * rgb2Y.z; M[2][2] = (1.0f - sat) * rgb2Y.z + sat; \n" \
"mat3 R = make_mat3(make_float3(M[0][0], M[0][1], M[0][2]),  \n" \
"make_float3(M[1][0], M[1][1], M[1][2]), make_float3(M[2][0], M[2][1], M[2][2])); \n" \
"R = transpose_f33(R); \n" \
"return R; \n" \
"} \n" \
"float moncurve_f( float x, float gamma, float offs ) { \n" \
"float y; \n" \
"const float fs = (( gamma - 1.0f) / offs) * pow( offs * gamma / ( ( gamma - 1.0f) * ( 1.0f + offs)), gamma); \n" \
"const float xb = offs / ( gamma - 1.0f); \n" \
"if ( x >= xb) \n" \
"y = pow( ( x + offs) / ( 1.0f + offs), gamma); \n" \
"else \n" \
"y = x * fs; \n" \
"return y; \n" \
"} \n" \
"float moncurve_r( float y, float gamma, float offs ) { \n" \
"float x; \n" \
"const float yb = pow( offs * gamma / ( ( gamma - 1.0f) * ( 1.0f + offs)), gamma); \n" \
"const float rs = pow( ( gamma - 1.0f) / offs, gamma - 1.0f) * pow( ( 1.0f + offs) / gamma, gamma); \n" \
"if ( y >= yb) \n" \
"x = ( 1.0f + offs) * pow( y, 1.0f / gamma) - offs; \n" \
"else \n" \
"x = y * rs; \n" \
"return x; \n" \
"} \n" \
"float3 moncurve_f_f3( float3 x, float gamma, float offs) { \n" \
"float3 y; \n" \
"y.x = moncurve_f( x.x, gamma, offs); y.y = moncurve_f( x.y, gamma, offs); y.z = moncurve_f( x.z, gamma, offs); \n" \
"return y; \n" \
"} \n" \
"float3 moncurve_r_f3( float3 y, float gamma, float offs) { \n" \
"float3 x; \n" \
"x.x = moncurve_r( y.x, gamma, offs); x.y = moncurve_r( y.y, gamma, offs); x.z = moncurve_r( y.z, gamma, offs); \n" \
"return x; \n" \
"} \n" \
"float bt1886_f( float V, float gamma, float Lw, float Lb) { \n" \
"float a = pow( pow( Lw, 1.0f/gamma) - pow( Lb, 1.0f/gamma), gamma); \n" \
"float b = pow( Lb, 1.0f/gamma) / ( pow( Lw, 1.0f/gamma) - pow( Lb, 1.0f/gamma)); \n" \
"float L = a * pow( fmax( V + b, 0.0f), gamma); \n" \
"return L; \n" \
"} \n" \
"float bt1886_r( float L, float gamma, float Lw, float Lb) { \n" \
"float a = pow( pow( Lw, 1.0f/gamma) - pow( Lb, 1.0f/gamma), gamma); \n" \
"float b = pow( Lb, 1.0f/gamma) / ( pow( Lw, 1.0f/gamma) - pow( Lb, 1.0f/gamma)); \n" \
"float V = pow( fmax( L / a, 0.0f), 1.0f/gamma) - b; \n" \
"return V; \n" \
"} \n" \
"float3 bt1886_f_f3( float3 V, float gamma, float Lw, float Lb) { \n" \
"float3 L; \n" \
"L.x = bt1886_f( V.x, gamma, Lw, Lb); L.y = bt1886_f( V.y, gamma, Lw, Lb); L.z = bt1886_f( V.z, gamma, Lw, Lb); \n" \
"return L; \n" \
"} \n" \
"float3 bt1886_r_f3( float3 L, float gamma, float Lw, float Lb) { \n" \
"float3 V; \n" \
"V.x = bt1886_r( L.x, gamma, Lw, Lb); V.y = bt1886_r( L.y, gamma, Lw, Lb); V.z = bt1886_r( L.z, gamma, Lw, Lb); \n" \
"return V; \n" \
"} \n" \
"float smpteRange_to_fullRange( float in) { \n" \
"const float REFBLACK = ( 64.0f / 1023.0f); \n" \
"const float REFWHITE = ( 940.0f / 1023.0f); \n" \
"return (( in - REFBLACK) / ( REFWHITE - REFBLACK)); \n" \
"} \n" \
"float fullRange_to_smpteRange( float in) { \n" \
"const float REFBLACK = ( 64.0f / 1023.0f); \n" \
"const float REFWHITE = ( 940.0f / 1023.0f); \n" \
"return ( in * ( REFWHITE - REFBLACK) + REFBLACK ); \n" \
"} \n" \
"float3 smpteRange_to_fullRange_f3( float3 rgbIn) { \n" \
"float3 rgbOut; \n" \
"rgbOut.x = smpteRange_to_fullRange( rgbIn.x); rgbOut.y = smpteRange_to_fullRange( rgbIn.y); rgbOut.z = smpteRange_to_fullRange( rgbIn.z); \n" \
"return rgbOut; \n" \
"} \n" \
"float3 fullRange_to_smpteRange_f3( float3 rgbIn) { \n" \
"float3 rgbOut; \n" \
"rgbOut.x = fullRange_to_smpteRange( rgbIn.x); rgbOut.y = fullRange_to_smpteRange( rgbIn.y); rgbOut.z = fullRange_to_smpteRange( rgbIn.z); \n" \
"return rgbOut; \n" \
"} \n" \
"float3 dcdm_decode( float3 XYZp) { \n" \
"float3 XYZ; \n" \
"XYZ.x = (52.37f/48.0f) * pow( XYZp.x, 2.6f); \n" \
"XYZ.y = (52.37f/48.0f) * pow( XYZp.y, 2.6f); \n" \
"XYZ.z = (52.37f/48.0f) * pow( XYZp.z, 2.6f); \n" \
"return XYZ; \n" \
"} \n" \
"float3 dcdm_encode( float3 XYZ) { \n" \
"float3 XYZp; \n" \
"XYZp.x = pow( (48.0f/52.37f) * XYZ.x, 1.0f/2.6f); \n" \
"XYZp.y = pow( (48.0f/52.37f) * XYZ.y, 1.0f/2.6f); \n" \
"XYZp.z = pow( (48.0f/52.37f) * XYZ.z, 1.0f/2.6f); \n" \
"return XYZp; \n" \
"} \n" \
"float ST2084_2_Y( float N ) { \n" \
"float Np = pow( N, 1.0f / pq_m2 ); \n" \
"float L = Np - pq_c1; \n" \
"if ( L < 0.0f ) \n" \
"L = 0.0f; \n" \
"L = L / ( pq_c2 - pq_c3 * Np ); \n" \
"L = pow( L, 1.0f / pq_m1 ); \n" \
"return L * pq_C; \n" \
"} \n" \
"float Y_2_ST2084( float C ) { \n" \
"float L = C / pq_C; \n" \
"float Lm = pow( L, pq_m1 ); \n" \
"float N = ( pq_c1 + pq_c2 * Lm ) / ( 1.0f + pq_c3 * Lm ); \n" \
"N = pow( N, pq_m2 ); \n" \
"return N; \n" \
"} \n" \
"float3 Y_2_ST2084_f3( float3 in) { \n" \
"float3 out; \n" \
"out.x = Y_2_ST2084( in.x); out.y = Y_2_ST2084( in.y); out.z = Y_2_ST2084( in.z); \n" \
"return out; \n" \
"} \n" \
"float3 ST2084_2_Y_f3( float3 in) { \n" \
"float3 out; \n" \
"out.x = ST2084_2_Y( in.x); out.y = ST2084_2_Y( in.y); out.z = ST2084_2_Y( in.z); \n" \
"return out; \n" \
"} \n" \
"float3 ST2084_2_HLG_1000nits_f3( float3 PQ) { \n" \
"float3 displayLinear = ST2084_2_Y_f3( PQ); \n" \
"float Y_d = 0.2627f * displayLinear.x + 0.6780f * displayLinear.y + 0.0593f * displayLinear.z; \n" \
"const float L_w = 1000.0f; \n" \
"const float L_b = 0.0f; \n" \
"const float alpha = (L_w - L_b); \n" \
"const float beta = L_b; \n" \
"const float gamma = 1.2f; \n" \
"float3 sceneLinear; \n" \
"if (Y_d == 0.0f) { \n" \
"sceneLinear.x = 0.0f; sceneLinear.y = 0.0f; sceneLinear.z = 0.0f; \n" \
"} else { \n" \
"sceneLinear.x = pow( (Y_d - beta) / alpha, (1.0f - gamma) / gamma) * ((displayLinear.x - beta) / alpha); \n" \
"sceneLinear.y = pow( (Y_d - beta) / alpha, (1.0f - gamma) / gamma) * ((displayLinear.y - beta) / alpha); \n" \
"sceneLinear.z = pow( (Y_d - beta) / alpha, (1.0f - gamma) / gamma) * ((displayLinear.z - beta) / alpha); \n" \
"} \n" \
"const float a = 0.17883277f; \n" \
"const float b = 0.28466892f; \n" \
"const float c = 0.55991073f; \n" \
"float3 HLG; \n" \
"if (sceneLinear.x <= 1.0f / 12.0f) { \n" \
"HLG.x = sqrt(3.0f * sceneLinear.x); \n" \
"} else { \n" \
"HLG.x = a * log(12.0f * sceneLinear.x-b)+c; \n" \
"} \n" \
"if (sceneLinear.y <= 1.0f / 12.0f) { \n" \
"HLG.y = sqrt(3.0f * sceneLinear.y); \n" \
"} else { \n" \
"HLG.y = a * log(12.0f * sceneLinear.y-b)+c; \n" \
"} \n" \
"if (sceneLinear.z <= 1.0f / 12.0f) { \n" \
"HLG.z = sqrt(3.0f * sceneLinear.z); \n" \
"} else { \n" \
"HLG.z = a * log(12.0f * sceneLinear.z - b) + c; \n" \
"} \n" \
"return HLG; \n" \
"} \n" \
"float3 HLG_2_ST2084_1000nits_f3( float3 HLG) { \n" \
"const float a = 0.17883277f; \n" \
"const float b = 0.28466892f; \n" \
"const float c = 0.55991073f; \n" \
"const float L_w = 1000.0f; \n" \
"const float L_b = 0.0f; \n" \
"const float alpha = (L_w - L_b); \n" \
"const float beta = L_b; \n" \
"const float gamma = 1.2f; \n" \
"float3 sceneLinear; \n" \
"if ( HLG.x >= 0.0f && HLG.x <= 0.5f) { \n" \
"sceneLinear.x = pow(HLG.x, 2.0f) / 3.0f; \n" \
"} else { \n" \
"sceneLinear.x = (exp((HLG.x - c) / a) + b) / 12.0f; \n" \
"} \n" \
"if ( HLG.y >= 0.0f && HLG.y <= 0.5f) { \n" \
"sceneLinear.y = pow(HLG.y, 2.0f) / 3.0f; \n" \
"} else { \n" \
"sceneLinear.y = (exp((HLG.y - c) / a) + b) / 12.0f; \n" \
"} \n" \
"if ( HLG.z >= 0.0f && HLG.z <= 0.5f) { \n" \
"sceneLinear.z = pow(HLG.z, 2.0f) / 3.0f; \n" \
"} else { \n" \
"sceneLinear.z = (exp((HLG.z - c) / a) + b) / 12.0f; \n" \
"} \n" \
"float Y_s = 0.2627f * sceneLinear.x + 0.6780f * sceneLinear.y + 0.0593f * sceneLinear.z; \n" \
"float3 displayLinear; \n" \
"displayLinear.x = alpha * pow( Y_s, gamma - 1.0f) * sceneLinear.x + beta; \n" \
"displayLinear.y = alpha * pow( Y_s, gamma - 1.0f) * sceneLinear.y + beta; \n" \
"displayLinear.z = alpha * pow( Y_s, gamma - 1.0f) * sceneLinear.z + beta; \n" \
"float3 PQ = Y_2_ST2084_f3( displayLinear); \n" \
"return PQ; \n" \
"} \n" \
"float rgb_2_saturation( float3 rgb) { \n" \
"return ( fmax( max_f3(rgb), TINY) - fmax( min_f3(rgb), TINY)) / fmax( max_f3(rgb), 1e-2f); \n" \
"} \n" \
"SegmentedSplineParams_c5 RRT_PARAMS() { \n" \
"SegmentedSplineParams_c5 A = {{ -4.0f, -4.0f, -3.1573765773f, -0.4852499958f, 1.8477324706f, 1.8477324706f}, \n" \
"{ -0.7185482425f, 2.0810307172f, 3.6681241237f, 4.0f, 4.0f, 4.0f}, {0.18f * pow(2.0f, -15.0f), 0.0001f}, \n" \
"{0.18f, 4.8f}, {0.18f * pow(2.0f, 18.0f), 10000.0f}, 0.0f, 0.0f}; \n" \
"return A; \n" \
"}; \n" \
"float segmented_spline_c5_fwd( float x) { \n" \
"SegmentedSplineParams_c5 C = RRT_PARAMS(); \n" \
"const int N_KNOTS_LOW = 4; \n" \
"const int N_KNOTS_HIGH = 4; \n" \
"float logx = log10( fmax(x, 0.0f)); \n" \
"float logy = 0.0f; \n" \
"if ( logx <= log10(C.minPoint.x) ) { \n" \
"logy = logx * C.slopeLow + ( log10(C.minPoint.y) - C.slopeLow * log10(C.minPoint.x) ); \n" \
"} else if (( logx > log10(C.minPoint.x) ) && ( logx < log10(C.midPoint.x) )) { \n" \
"float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10(C.minPoint.x)) / (log10(C.midPoint.x) - log10(C.minPoint.x)); \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float3 cf = make_float3( C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]); \n" \
"float3 monomials = make_float3( t * t, t, 1.0f ); \n" \
"logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM)); \n" \
"} else if (( logx >= log10(C.midPoint.x) ) && ( logx < log10(C.maxPoint.x) )) { \n" \
"float knot_coord = (N_KNOTS_HIGH-1) * (logx-log10(C.midPoint.x)) / (log10(C.maxPoint.x) - log10(C.midPoint.x)); \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]}; \n" \
"float3 monomials = make_float3( t * t, t, 1.0f ); \n" \
"logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM)); \n" \
"} else { \n" \
"logy = logx * C.slopeHigh + ( log10(C.maxPoint.y) - C.slopeHigh * log10(C.maxPoint.x) ); \n" \
"} \n" \
"return exp10(logy); \n" \
"} \n" \
"SegmentedSplineParams_c9 ODT_48nits() { \n" \
"SegmentedSplineParams_c9 A = \n" \
"{{ -1.6989700043f, -1.6989700043f, -1.4779f, -1.2291f, -0.8648f, -0.448f, 0.00518f, 0.4511080334f, 0.9113744414f, 0.9113744414f}, \n" \
"{ 0.5154386965f, 0.8470437783f, 1.1358f, 1.3802f, 1.5197f, 1.5985f, 1.6467f, 1.6746091357f, 1.6878733390f, 1.6878733390f }, \n" \
"{segmented_spline_c5_fwd( 0.18f * pow(2.0f, -6.5f) ), 0.02f}, {segmented_spline_c5_fwd( 0.18f ), 4.8f}, \n" \
"{segmented_spline_c5_fwd( 0.18f * pow(2.0f, 6.5f) ), 48.0f}, 0.0f, 0.04f}; \n" \
"return A; \n" \
"}; \n" \
"SegmentedSplineParams_c9 ODT_1000nits() { \n" \
"SegmentedSplineParams_c9 A = \n" \
"{{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f }, \n" \
"{ 0.8089132070f, 1.1910867930f, 1.5683f, 1.9483f, 2.3083f, 2.6384f, 2.8595f, 2.9872608805f, 3.0127391195f, 3.0127391195f }, \n" \
"{segmented_spline_c5_fwd( 0.18f * pow(2.0f, -12.0f) ), 0.0001f}, {segmented_spline_c5_fwd( 0.18f ), 10.0f}, \n" \
"{segmented_spline_c5_fwd( 0.18 * pow(2.0f, 10.0f) ), 1000.0f}, 3.0f, 0.06f}; \n" \
"return A; \n" \
"}; \n" \
"SegmentedSplineParams_c9 ODT_2000nits() { \n" \
"SegmentedSplineParams_c9 A = \n" \
"{{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f }, \n" \
"{ 0.8019952042f, 1.1980047958f, 1.5943f, 1.9973f, 2.3783f, 2.7684f, 3.0515f, 3.2746293562f, 3.3274306351f, 3.3274306351f }, \n" \
"{segmented_spline_c5_fwd( 0.18f * pow(2.0f, -12.0f) ), 0.0001f}, {segmented_spline_c5_fwd( 0.18f ), 10.0f}, \n" \
"{segmented_spline_c5_fwd( 0.18f * pow(2.0f, 11.0f) ), 2000.0f}, 3.0f, 0.12f}; \n" \
"return A; \n" \
"}; \n" \
"SegmentedSplineParams_c9 ODT_4000nits() { \n" \
"SegmentedSplineParams_c9 A = \n" \
"{{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f }, \n" \
"{ 0.7973186613f, 1.2026813387f, 1.6093f, 2.0108f, 2.4148f, 2.8179f, 3.1725f, 3.5344995451f, 3.6696204376f, 3.6696204376f }, \n" \
"{segmented_spline_c5_fwd( 0.18f * pow(2.0f, -12.0f) ), 0.0001f}, {segmented_spline_c5_fwd( 0.18f ), 10.0f}, \n" \
"{segmented_spline_c5_fwd( 0.18f * pow(2.0f, 12.0f) ), 4000.0f}, 3.0f, 0.3f}; \n" \
"return A; \n" \
"}; \n" \
"float segmented_spline_c5_rev( float y) { \n" \
"SegmentedSplineParams_c5 C = RRT_PARAMS(); \n" \
"const int N_KNOTS_LOW = 4; \n" \
"const int N_KNOTS_HIGH = 4; \n" \
"const float KNOT_INC_LOW = (log10(C.midPoint.x) - log10(C.minPoint.x)) / (N_KNOTS_LOW - 1.0f); \n" \
"const float KNOT_INC_HIGH = (log10(C.maxPoint.x) - log10(C.midPoint.x)) / (N_KNOTS_HIGH - 1.0f); \n" \
"float KNOT_Y_LOW[4]; \n" \
"for (int i = 0; i < N_KNOTS_LOW; i = i + 1) { \n" \
"KNOT_Y_LOW[i] = ( C.coefsLow[i] + C.coefsLow[i + 1]) / 2.0f; \n" \
"}; \n" \
"float KNOT_Y_HIGH[4]; \n" \
"for (int i = 0; i < N_KNOTS_HIGH; i = i+1) { \n" \
"KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i + 1]) / 2.0f; \n" \
"}; \n" \
"float logy = log10( fmax(y, 1e-10f)); \n" \
"float logx; \n" \
"if (logy <= log10(C.minPoint.y)) { \n" \
"logx = log10(C.minPoint.x); \n" \
"} else if ( (logy > log10(C.minPoint.y)) && (logy <= log10(C.midPoint.y)) ) { \n" \
"unsigned int j = 0; \n" \
"float3 cf = make_float3(0.0f, 0.0f, 0.0f); \n" \
"if ( logy > KNOT_Y_LOW[0] && logy <= KNOT_Y_LOW[1]) { \n" \
"cf.x = C.coefsLow[0]; cf.y = C.coefsLow[1]; cf.z = C.coefsLow[2]; j = 0; \n" \
"} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) { \n" \
"cf.x = C.coefsLow[1]; cf.y = C.coefsLow[2]; cf.z = C.coefsLow[3]; j = 1; \n" \
"} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) { \n" \
"cf.x = C.coefsLow[2]; cf.y = C.coefsLow[3]; cf.z = C.coefsLow[4]; j = 2; \n" \
"} \n" \
"const float3 tmp = mult_f3_f33( cf, MM); \n" \
"float a = tmp.x; float b = tmp.y; float c = tmp.z; \n" \
"c = c - logy; \n" \
"const float d = sqrt( b * b - 4.0f * a * c); \n" \
"const float t = ( 2.0f * c) / ( -d - b); \n" \
"logx = log10(C.minPoint.x) + ( t + j) * KNOT_INC_LOW; \n" \
"} else if ( (logy > log10(C.midPoint.y)) && (logy < log10(C.maxPoint.y)) ) { \n" \
"unsigned int j = 0; \n" \
"float3 cf = make_float3(0.0f, 0.0f, 0.0f); \n" \
"if ( logy > KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) { \n" \
"cf.x = C.coefsHigh[0]; cf.y = C.coefsHigh[1]; cf.z = C.coefsHigh[2]; j = 0; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) { \n" \
"cf.x = C.coefsHigh[1]; cf.y = C.coefsHigh[2]; cf.z = C.coefsHigh[3]; j = 1; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) { \n" \
"cf.x = C.coefsHigh[2]; cf.y = C.coefsHigh[3]; cf.z = C.coefsHigh[4]; j = 2; \n" \
"} \n" \
"const float3 tmp = mult_f3_f33( cf, MM); \n" \
"float a = tmp.x; float b = tmp.y; float c = tmp.z; \n" \
"c = c - logy; \n" \
"const float d = sqrt( b * b - 4.0f * a * c); \n" \
"const float t = ( 2.0f * c) / ( -d - b); \n" \
"logx = log10(C.midPoint.x) + ( t + j) * KNOT_INC_HIGH; \n" \
"} else { \n" \
"logx = log10(C.maxPoint.x); \n" \
"} \n" \
"return exp10( logx); \n" \
"} \n" \
"float segmented_spline_c9_fwd( float x, SegmentedSplineParams_c9 C) { \n" \
"const int N_KNOTS_LOW = 8; \n" \
"const int N_KNOTS_HIGH = 8; \n" \
"float logx = log10( fmax(x, 0.0f)); \n" \
"float logy = 0.0f; \n" \
"if ( logx <= log10(C.minPoint.x) ) { \n" \
"logy = logx * C.slopeLow + ( log10(C.minPoint.y) - C.slopeLow * log10(C.minPoint.x) ); \n" \
"} else if (( logx > log10(C.minPoint.x) ) && ( logx < log10(C.midPoint.x) )) { \n" \
"float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10(C.minPoint.x)) / (log10(C.midPoint.x) - log10(C.minPoint.x)); \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float3 cf = { C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]}; \n" \
"float3 monomials = make_float3( t * t, t, 1.0f ); \n" \
"logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM)); \n" \
"} else if (( logx >= log10(C.midPoint.x) ) && ( logx < log10(C.maxPoint.x) )) { \n" \
"float knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10(C.midPoint.x)) / (log10(C.maxPoint.x) - log10(C.midPoint.x)); \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]}; \n" \
"float3 monomials = make_float3( t * t, t, 1.0f ); \n" \
"logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM)); \n" \
"} else { \n" \
"logy = logx * C.slopeHigh + ( log10(C.maxPoint.y) - C.slopeHigh * log10(C.maxPoint.x) ); \n" \
"} \n" \
"return exp10(logy); \n" \
"} \n" \
"float segmented_spline_c9_rev( float y, SegmentedSplineParams_c9 C) { \n" \
"const int N_KNOTS_LOW = 8; \n" \
"const int N_KNOTS_HIGH = 8; \n" \
"const float KNOT_INC_LOW = (log10(C.midPoint.x) - log10(C.minPoint.x)) / (N_KNOTS_LOW - 1.0f); \n" \
"const float KNOT_INC_HIGH = (log10(C.maxPoint.x) - log10(C.midPoint.x)) / (N_KNOTS_HIGH - 1.0f); \n" \
"float KNOT_Y_LOW[8]; \n" \
"for (int i = 0; i < N_KNOTS_LOW; i = i + 1) { \n" \
"KNOT_Y_LOW[ i] = ( C.coefsLow[i] + C.coefsLow[i + 1]) / 2.0f; \n" \
"}; \n" \
"float KNOT_Y_HIGH[8]; \n" \
"for (int i = 0; i < N_KNOTS_HIGH; i = i + 1) { \n" \
"KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i + 1]) / 2.0f; \n" \
"}; \n" \
"float logy = log10( fmax( y, 1e-10f)); \n" \
"float logx; \n" \
"if (logy <= log10(C.minPoint.y)) { \n" \
"logx = log10(C.minPoint.x); \n" \
"} else if ( (logy > log10(C.minPoint.y)) && (logy <= log10(C.midPoint.y)) ) { \n" \
"unsigned int j = 0; \n" \
"float3 cf = make_float3(0.0f, 0.0f, 0.0f); \n" \
"if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) { \n" \
"cf.x = C.coefsLow[0]; cf.y = C.coefsLow[1]; cf.z = C.coefsLow[2]; j = 0; \n" \
"} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) { \n" \
"cf.x = C.coefsLow[1]; cf.y = C.coefsLow[2]; cf.z = C.coefsLow[3]; j = 1; \n" \
"} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) { \n" \
"cf.x = C.coefsLow[2]; cf.y = C.coefsLow[3]; cf.z = C.coefsLow[4]; j = 2; \n" \
"} else if ( logy > KNOT_Y_LOW[ 3] && logy <= KNOT_Y_LOW[ 4]) { \n" \
"cf.x = C.coefsLow[3]; cf.y = C.coefsLow[4]; cf.z = C.coefsLow[5]; j = 3; \n" \
"} else if ( logy > KNOT_Y_LOW[ 4] && logy <= KNOT_Y_LOW[ 5]) { \n" \
"cf.x = C.coefsLow[4]; cf.y = C.coefsLow[5]; cf.z = C.coefsLow[6]; j = 4; \n" \
"} else if ( logy > KNOT_Y_LOW[ 5] && logy <= KNOT_Y_LOW[ 6]) { \n" \
"cf.x = C.coefsLow[5]; cf.y = C.coefsLow[6]; cf.z = C.coefsLow[7]; j = 5; \n" \
"} else if ( logy > KNOT_Y_LOW[ 6] && logy <= KNOT_Y_LOW[ 7]) { \n" \
"cf.x = C.coefsLow[6]; cf.y = C.coefsLow[7]; cf.z = C.coefsLow[8]; j = 6; \n" \
"} \n" \
"const float3 tmp = mult_f3_f33( cf, MM); \n" \
"float a = tmp.x; float b = tmp.y; float c = tmp.z; \n" \
"c = c - logy; \n" \
"const float d = sqrt( b * b - 4.0f * a * c); \n" \
"const float t = ( 2.0f * c) / ( -d - b); \n" \
"logx = log10(C.minPoint.x) + ( t + j) * KNOT_INC_LOW; \n" \
"} else if ( (logy > log10(C.midPoint.y)) && (logy < log10(C.maxPoint.y)) ) { \n" \
"unsigned int j = 0; \n" \
"float3 cf = make_float3(0.0f, 0.0f, 0.0f); \n" \
"if ( logy > KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) { \n" \
"cf.x = C.coefsHigh[0]; cf.y = C.coefsHigh[1]; cf.z = C.coefsHigh[2]; j = 0; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) { \n" \
"cf.x = C.coefsHigh[1]; cf.y = C.coefsHigh[2]; cf.z = C.coefsHigh[3]; j = 1; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) { \n" \
"cf.x = C.coefsHigh[2]; cf.y = C.coefsHigh[3]; cf.z = C.coefsHigh[4]; j = 2; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 3] && logy <= KNOT_Y_HIGH[ 4]) { \n" \
"cf.x = C.coefsHigh[3]; cf.y = C.coefsHigh[4]; cf.z = C.coefsHigh[5]; j = 3; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 4] && logy <= KNOT_Y_HIGH[ 5]) { \n" \
"cf.x = C.coefsHigh[4]; cf.y = C.coefsHigh[5]; cf.z = C.coefsHigh[6]; j = 4; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 5] && logy <= KNOT_Y_HIGH[ 6]) { \n" \
"cf.x = C.coefsHigh[5]; cf.y = C.coefsHigh[6]; cf.z = C.coefsHigh[7]; j = 5; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 6] && logy <= KNOT_Y_HIGH[ 7]) { \n" \
"cf.x = C.coefsHigh[6]; cf.y = C.coefsHigh[7]; cf.z = C.coefsHigh[8]; j = 6; \n" \
"} \n" \
"const float3 tmp = mult_f3_f33( cf, MM); \n" \
"float a = tmp.x; float b = tmp.y; float c = tmp.z; \n" \
"c = c - logy; \n" \
"const float d = sqrt( b * b - 4.0f * a * c); \n" \
"const float t = ( 2.0f * c) / ( -d - b); \n" \
"logx = log10(C.midPoint.x) + ( t + j) * KNOT_INC_HIGH; \n" \
"} else { \n" \
"logx = log10(C.maxPoint.x); \n" \
"} \n" \
"return exp10( logx); \n" \
"} \n" \
"float3 segmented_spline_c9_rev_f3( float3 rgbPre) { \n" \
"SegmentedSplineParams_c9 C = ODT_48nits(); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, C); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, C); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, C); \n" \
"return rgbPost; \n" \
"} \n" \
"float3 segmented_spline_c5_rev_f3( float3 rgbPre) { \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c5_rev( rgbPre.x); \n" \
"rgbPost.y = segmented_spline_c5_rev( rgbPre.y); \n" \
"rgbPost.z = segmented_spline_c5_rev( rgbPre.z); \n" \
"return rgbPost; \n" \
"} \n" \
"float lin_to_ACEScc( float in) { \n" \
"if (in <= 0.0f) \n" \
"return -0.3584474886f; \n" \
"else if (in < pow(2.0f, -15.0f)) \n" \
"return (log2( exp2(-16.0f) + in * 0.5f) + 9.72f) / 17.52f; \n" \
"else \n" \
"return (log2(in) + 9.72f) / 17.52f; \n" \
"} \n" \
"float3 ACES_to_ACEScc( float3 ACES) { \n" \
"ACES = max_f3_f( ACES, 0.0f); \n" \
"float3 lin_AP1 = mult_f3_f33( ACES, AP0_2_AP1_MAT); \n" \
"float3 out; \n" \
"out.x = lin_to_ACEScc( lin_AP1.x); out.y = lin_to_ACEScc( lin_AP1.y); out.z = lin_to_ACEScc( lin_AP1.z); \n" \
"return out; \n" \
"} \n" \
"float ACEScc_to_lin( float in) { \n" \
"if (in < -0.3013698630f) \n" \
"return (exp2(in * 17.52f - 9.72f) - exp2(-16.0f)) * 2.0f; \n" \
"else \n" \
"return exp2(in * 17.52f - 9.72f); \n" \
"} \n" \
"float3 ACEScc_to_ACES( float3 ACEScc) { \n" \
"float3 lin_AP1; \n" \
"lin_AP1.x = ACEScc_to_lin( ACEScc.x); lin_AP1.y = ACEScc_to_lin( ACEScc.y); lin_AP1.z = ACEScc_to_lin( ACEScc.z); \n" \
"float3 ACES = mult_f3_f33( lin_AP1, AP1_2_AP0_MAT); \n" \
"return ACES; \n" \
"} \n" \
"float lin_to_ACEScct( float in) { \n" \
"if (in <= X_BRK) \n" \
"return A * in + B; \n" \
"else \n" \
"return (log2(in) + 9.72f) / 17.52f; \n" \
"} \n" \
"float ACEScct_to_lin( float in) { \n" \
"if (in > Y_BRK) \n" \
"return exp2(in * 17.52f - 9.72f); \n" \
"else \n" \
"return (in - B) / A; \n" \
"} \n" \
"float3 ACES_to_ACEScct( float3 in) { \n" \
"float3 ap1_lin = mult_f3_f33( in, AP0_2_AP1_MAT); \n" \
"float3 acescct; \n" \
"acescct.x = lin_to_ACEScct( ap1_lin.x); acescct.y = lin_to_ACEScct( ap1_lin.y); acescct.z = lin_to_ACEScct( ap1_lin.z); \n" \
"return acescct; \n" \
"} \n" \
"float3 ACEScct_to_ACES( float3 in) { \n" \
"float3 ap1_lin; \n" \
"ap1_lin.x = ACEScct_to_lin( in.x); ap1_lin.y = ACEScct_to_lin( in.y); ap1_lin.z = ACEScct_to_lin( in.z); \n" \
"return mult_f3_f33( ap1_lin, AP1_2_AP0_MAT); \n" \
"} \n" \
"float3 ACES_to_ACEScg( float3 ACES) { \n" \
"ACES = max_f3_f( ACES, 0.0f); \n" \
"float3 ACEScg = mult_f3_f33( ACES, AP0_2_AP1_MAT); \n" \
"return ACEScg; \n" \
"} \n" \
"float3 ACEScg_to_ACES( float3 ACEScg) { \n" \
"float3 ACES = mult_f3_f33( ACEScg, AP1_2_AP0_MAT); \n" \
"return ACES; \n" \
"} \n" \
"float ACESproxy_to_lin( float in) { \n" \
"float StepsPerStop = 50.0f; \n" \
"float MidCVoffset = 425.0f; \n" \
"return exp2(( in - MidCVoffset)/StepsPerStop - 2.5f); \n" \
"} \n" \
"float3 ACESproxy_to_ACES( float3 In) { \n" \
"float3 ACESproxy; \n" \
"ACESproxy.x = In.x * 1023.0f; \n" \
"ACESproxy.y = In.y * 1023.0f; \n" \
"ACESproxy.z = In.z * 1023.0f; \n" \
"float3 lin_AP1; \n" \
"lin_AP1.x = ACESproxy_to_lin( ACESproxy.x); \n" \
"lin_AP1.y = ACESproxy_to_lin( ACESproxy.y); \n" \
"lin_AP1.z = ACESproxy_to_lin( ACESproxy.z); \n" \
"float3 ACES = mult_f3_f33( lin_AP1, AP1_2_AP0_MAT); \n" \
"return ACES; \n" \
"} \n" \
"float lin_to_ACESproxy( float in) { \n" \
"float StepsPerStop = 50.0f; \n" \
"float MidCVoffset = 425.0f; \n" \
"float CVmin = 64.0f; \n" \
"float CVmax = 940.0f; \n" \
"if (in <= pow(2.0f, -9.72f)) \n" \
"return CVmin; \n" \
"else \n" \
"return fmax( CVmin, fmin( CVmax, round( (log2(in) + 2.5f) * StepsPerStop + MidCVoffset))); \n" \
"} \n" \
"float3 ACES_to_ACESproxy( float3 ACES) { \n" \
"ACES = max_f3_f( ACES, 0.0f);  \n" \
"float3 lin_AP1 = mult_f3_f33( ACES, AP0_2_AP1_MAT); \n" \
"float ACESproxy[3]; \n" \
"ACESproxy[0] = lin_to_ACESproxy( lin_AP1.x ); \n" \
"ACESproxy[1] = lin_to_ACESproxy( lin_AP1.y ); \n" \
"ACESproxy[2] = lin_to_ACESproxy( lin_AP1.z ); \n" \
"float3 out;     \n" \
"out.x = ACESproxy[0] / 1023.0f; \n" \
"out.y = ACESproxy[1] / 1023.0f; \n" \
"out.z = ACESproxy[2] / 1023.0f; \n" \
"return out; \n" \
"} \n" \
"float3 adx_convertFromLinear( float3 aces) { \n" \
"aces.x = aces.x < 0.00130127f ? (aces.x - 0.00130127f) / 0.04911331f : \n" \
"aces.x < 0.001897934f ? (log(aces.x) + 6.644415f) / 37.74261f :  \n" \
"aces.x < 0.118428f ? log(aces.x) * (0.02871031f * log(aces.x) + 0.383914f) + 1.288383f :  \n" \
"(log(aces.x) + 4.645361f) / 4.1865183f; \n" \
"aces.y =   aces.y < 0.00130127f ? (aces.y - 0.00130127f) / 0.04911331f : \n" \
"aces.y < 0.001897934f ? (log(aces.y) + 6.644415f) / 37.74261f :  \n" \
"aces.y < 0.118428f ? log(aces.y) * (0.02871031f * log(aces.y) + 0.383914f) + 1.288383f :  \n" \
"(log(aces.y) + 4.645361f) / 4.1865183f; \n" \
"aces.z =   aces.z < 0.00130127f ? (aces.z - 0.00130127f) / 0.04911331f : \n" \
"aces.z < 0.001897934f ? (log(aces.z) + 6.644415f) / 37.74261f :  \n" \
"aces.z < 0.118428f ? log(aces.z) * (0.02871031f * log(aces.z) + 0.383914f) + 1.288383f :  \n" \
"(log(aces.z) + 4.645361f) / 4.1865183f; \n" \
"return aces; \n" \
"} \n" \
"float3 adx_convertToLinear( float3 aces) { \n" \
"aces.x = aces.x < 0.0f ? 0.04911331f * aces.x + 0.001301270f :  \n" \
"aces.x < 0.01f ? exp(37.74261f * aces.x - 6.644415f) :  \n" \
"aces.x < 0.6f ? exp(-6.685996f + 2.302585f * pow(6.569476f * aces.x - 0.03258072f, 0.5f)) :  \n" \
"exp(fmin(4.1865183f * aces.x - 4.645361f, 86.4f)); \n" \
"aces.y = aces.y < 0.0f ? 0.04911331f * aces.y + 0.001301270f :  \n" \
"aces.y < 0.01f ? exp(37.74261f * aces.y - 6.644415f) :  \n" \
"aces.y < 0.6f ? exp(-6.685996f + 2.302585f * pow(6.569476f * aces.y - 0.03258072f, 0.5f)) :  \n" \
"exp(fmin(4.1865183f * aces.y - 4.645361f, 86.4f)); \n" \
"aces.z = aces.z < 0.0f ? 0.04911331f * aces.z + 0.001301270f :  \n" \
"aces.z < 0.01f ? exp(37.74261f * aces.z - 6.644415f) :  \n" \
"aces.z < 0.6f ? exp(-6.685996f + 2.302585f * pow(6.569476f * aces.z - 0.03258072f, 0.5f)) :  \n" \
"exp(fmin(4.1865183f * aces.z - 4.645361f, 86.4f)); \n" \
"return aces; \n" \
"} \n" \
"float3 ADX_to_ACES( float3 aces) { \n" \
"aces.x = aces.x * 2.048f - 0.19f; \n" \
"aces.y = aces.y * 2.048f - 0.19f; \n" \
"aces.z = aces.z * 2.048f - 0.19f; \n" \
"aces = mult_f3_f33(aces, CDD_TO_CID); \n" \
"aces = adx_convertToLinear(aces); \n" \
"aces = mult_f3_f33(aces, EXP_TO_ACES); \n" \
"return aces; \n" \
"} \n" \
"float3 ACES_to_ADX( float3 aces) { \n" \
"aces = mult_f3_f33(aces, invert_f33(EXP_TO_ACES)); \n" \
"aces = adx_convertFromLinear(aces); \n" \
"aces =  mult_f3_f33(aces, invert_f33(CDD_TO_CID)); \n" \
"aces.x = (aces.x + 0.19f) / 2.048f; \n" \
"aces.y = (aces.y + 0.19f) / 2.048f; \n" \
"aces.z = (aces.z + 0.19f) / 2.048f; \n" \
"return aces; \n" \
"} \n" \
"float3 ICpCt_to_ACES( float3 ICtCp) { \n" \
"float3 LMSp = mult_f3_f33( ICtCp, ICtCp_2_LMSp_MAT); \n" \
"float3 LMS; \n" \
"LMS.x = ST2084_2_Y(LMSp.x); \n" \
"LMS.y = ST2084_2_Y(LMSp.y); \n" \
"LMS.z = ST2084_2_Y(LMSp.z); \n" \
"float3 aces = mult_f3_f33(LMS, LMS_2_AP0_MAT); \n" \
"float scale = 209.0f; \n" \
"aces = mult_f_f3( 1.0f / scale, aces); \n" \
"return aces; \n" \
"} \n" \
"float3 ACES_to_ICpCt( float3 aces) { \n" \
"float scale = 209.0f; \n" \
"aces = mult_f_f3( scale, aces); \n" \
"float3 LMS = mult_f3_f33(aces, AP0_2_LMS_MAT); \n" \
"float3 LMSp; \n" \
"LMSp.x = Y_2_ST2084(LMS.x); \n" \
"LMSp.y = Y_2_ST2084(LMS.y); \n" \
"LMSp.z = Y_2_ST2084(LMS.z); \n" \
"float3 ICtCp = mult_f3_f33(LMSp, LMSp_2_ICtCp_MAT); \n" \
"return ICtCp; \n" \
"} \n" \
"float3 LogC_EI800_AWG_to_ACES( float3 in) { \n" \
"float3 lin_AWG; \n" \
"lin_AWG.x = LogC_to_lin(in.x); \n" \
"lin_AWG.y = LogC_to_lin(in.y); \n" \
"lin_AWG.z = LogC_to_lin(in.z); \n" \
"float3 aces = mult_f3_f33( lin_AWG, AWG_2_AP0_MAT); \n" \
"return aces; \n" \
"} \n" \
"float3 ACES_to_LogC_EI800_AWG( float3 in) { \n" \
"float3 lin_AWG = mult_f3_f33( in, AP0_2_AWG_MAT); \n" \
"float3 out; \n" \
"out.x = lin_to_LogC(lin_AWG.x); \n" \
"out.y = lin_to_LogC(lin_AWG.y); \n" \
"out.z = lin_to_LogC(lin_AWG.z); \n" \
"return out; \n" \
"} \n" \
"float3 Log3G10_RWG_to_ACES( float3 in) { \n" \
"float3 lin_RWG; \n" \
"lin_RWG.x = Log3G10_to_lin(in.x); \n" \
"lin_RWG.y = Log3G10_to_lin(in.y); \n" \
"lin_RWG.z = Log3G10_to_lin(in.z); \n" \
"float3 aces = mult_f3_f33( lin_RWG, RWG_2_AP0_MAT); \n" \
"return aces; \n" \
"} \n" \
"float3 ACES_to_Log3G10_RWG( float3 in) { \n" \
"float3 lin_RWG = mult_f3_f33(in, AP0_2_RWG_MAT); \n" \
"float3 out; \n" \
"out.x = lin_to_Log3G10(lin_RWG.x); \n" \
"out.y = lin_to_Log3G10(lin_RWG.y); \n" \
"out.z = lin_to_Log3G10(lin_RWG.z); \n" \
"return out; \n" \
"} \n" \
"float3 SLog3_SG3_to_ACES( float3 in) { \n" \
"float3 lin_SG3; \n" \
"lin_SG3.x = SLog3_to_lin(in.x); \n" \
"lin_SG3.y = SLog3_to_lin(in.y); \n" \
"lin_SG3.z = SLog3_to_lin(in.z); \n" \
"float3 aces = mult_f3_f33(lin_SG3, SG3_2_AP0_MAT); \n" \
"return aces; \n" \
"} \n" \
"float3 ACES_to_SLog3_SG3( float3 in) { \n" \
"float3 lin_SG3 = mult_f3_f33(in, AP0_2_SG3_MAT); \n" \
"float3 out; \n" \
"out.x = lin_to_SLog3(lin_SG3.x); \n" \
"out.y = lin_to_SLog3(lin_SG3.y); \n" \
"out.z = lin_to_SLog3(lin_SG3.z); \n" \
"return out; \n" \
"} \n" \
"float3 SLog3_SG3C_to_ACES( float3 in) { \n" \
"float3 lin_SG3C; \n" \
"lin_SG3C.x = SLog3_to_lin(in.x); \n" \
"lin_SG3C.y = SLog3_to_lin(in.y); \n" \
"lin_SG3C.z = SLog3_to_lin(in.z); \n" \
"float3 aces = mult_f3_f33(lin_SG3C, SG3C_2_AP0_MAT); \n" \
"return aces; \n" \
"} \n" \
"float3 ACES_to_SLog3_SG3C( float3 in) { \n" \
"float3 lin_SG3C = mult_f3_f33(in, AP0_2_SG3C_MAT); \n" \
"float3 out; \n" \
"out.x = lin_to_SLog3(lin_SG3C.x); \n" \
"out.y = lin_to_SLog3(lin_SG3C.y); \n" \
"out.z = lin_to_SLog3(lin_SG3C.z); \n" \
"return out; \n" \
"} \n" \
"float3 IDT_Alexa_v3_raw_EI800_CCT6500( float3 In){ \n" \
"float black = 256.0f / 65535.0f; \n" \
"float r_lin = (In.x - black); \n" \
"float g_lin = (In.y - black); \n" \
"float b_lin = (In.z - black); \n" \
"float3 aces; \n" \
"aces.x = r_lin * 0.809931f + g_lin * 0.162741f + b_lin * 0.027328f; \n" \
"aces.y = r_lin * 0.083731f + g_lin * 1.108667f + b_lin * -0.192397f; \n" \
"aces.z = r_lin * 0.044166f + g_lin * -0.272038f + b_lin * 1.227872f; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Panasonic_V35( float3 VLog) { \n" \
"mat3 mat = { {0.724382758f, 0.166748484f, 0.108497411f},  \n" \
"{0.021354009f, 0.985138372f, -0.006319092f},  \n" \
"{-0.009234278f, -0.00104295f, 1.010272625f} }; \n" \
"float rLin = vLogToLinScene(VLog.x); \n" \
"float gLin = vLogToLinScene(VLog.y); \n" \
"float bLin = vLogToLinScene(VLog.z); \n" \
"float3 out; \n" \
"out.x = mat.c0.x * rLin + mat.c0.y * gLin + mat.c0.z * bLin; \n" \
"out.y = mat.c1.x * rLin + mat.c1.y * gLin + mat.c1.z * bLin; \n" \
"out.z = mat.c2.x * rLin + mat.c2.y * gLin + mat.c2.z * bLin; \n" \
"return out; \n" \
"} \n" \
"float3 IDT_Canon_C100_A_D55( float3 In) { \n" \
"float iR, iG, iB; \n" \
"iR = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"iG = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"iB = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 pmtx; \n" \
"pmtx.x = 1.08190037262167 * iR -0.180298701368782 * iG +0.0983983287471069 * iB \n" \
"+1.9458545364518 * iR*iG -0.509539936937375 * iG*iB -0.47489567735516 * iB*iR \n" \
"-0.778086752197068 * iR*iR -0.7412266070049 * iG*iG +0.557894437042701 * iB*iB \n" \
"-3.27787395719078 * iR*iR*iG +0.254878417638717 * iR*iR*iB +3.45581530576474 * iR*iG*iG \n" \
"+0.335471713974739 * iR*iG*iB -0.43352125478476 * iR*iB*iB -1.65050137344141 * iG*iG*iB +1.46581418175682 * iG*iB*iB \n" \
"+0.944646566605676 * iR*iR*iR -0.723653099155881 * iG*iG*iG -0.371076501167857 * iB*iB*iB; \n" \
"pmtx.y = -0.00858997792576314 * iR +1.00673740119621 * iG +0.00185257672955608 * iB \n" \
"+0.0848736138296452 * iR*iG +0.347626906448902 * iG*iB +0.0020230274463939 * iB*iR \n" \
"-0.0790508414091524 * iR*iR -0.179497582958716 * iG*iG -0.175975123357072 * iB*iB \n" \
"+2.30205579706951 * iR*iR*iG -0.627257613385219 * iR*iR*iB -2.90795250918851 * iR*iG*iG \n" \
"+1.37002437502321 * iR*iG*iB -0.108668158565563 * iR*iB*iB -2.21150552827555 * iG*iG*iB + 1.53315057595445 * iG*iB*iB \n" \
"-0.543188706699505 * iR*iR*iR +1.63793038490376 * iG*iG*iG -0.444588616836587 * iB*iB*iB; \n" \
"pmtx.z = 0.12696639806511 * iR -0.011891441127869 * iG +0.884925043062759 * iB \n" \
"+1.34780279822258 * iR*iG +1.03647352257365 * iG*iB +0.459113289955922 * iB*iR \n" \
"-0.878157422295268 * iR*iR -1.3066278750436 * iG*iG -0.658604313413283 * iB*iB \n" \
"-1.4444077996703 * iR*iR*iG +0.556676588785173 * iR*iR*iB +2.18798497054968 * iR*iG*iG \n" \
"-1.43030768398665 * iR*iG*iB -0.0388323570817641 * iR*iB*iB +2.63698573112453 * iG*iG*iB -1.66598882056039 * iG*iB*iB \n" \
"+0.33450249360103 * iR*iR*iR -1.65856930730901 * iG*iG*iG +0.521956184547685 * iB*iB*iB; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( pmtx.x); \n" \
"lin.y = CanonLog_to_lin( pmtx.y); \n" \
"lin.z = CanonLog_to_lin( pmtx.z); \n" \
"float3 aces; \n" \
"aces.x = 0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926f * lin.z; \n" \
"aces.y = 0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821f * lin.z; \n" \
"aces.z = 0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C100_A_Tng( float3 In) { \n" \
"float iR, iG, iB; \n" \
"iR = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"iG = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"iB = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 pmtx; \n" \
"pmtx.x = 0.963803004454899 * iR - 0.160722202570655 * iG + 0.196919198115756 * iB \n" \
"+2.03444685639819 * iR*iG - 0.442676931451021 * iG*iB - 0.407983781537509 * iB*iR \n" \
"-0.640703323129254 * iR*iR - 0.860242798247848 * iG*iG + 0.317159977967446 * iB*iB \n" \
"-4.80567080102966 * iR*iR*iG + 0.27118370397567 * iR*iR*iB + 5.1069005049557 * iR*iG*iG \n" \
"+0.340895816920585 * iR*iG*iB - 0.486941738507862 * iR*iB*iB - 2.23737935753692 * iG*iG*iB + 1.96647555251297 * iG*iB*iB \n" \
"+1.30204051766243 * iR*iR*iR - 1.06503117628554 * iG*iG*iG - 0.392473022667378 * iB*iB*iB; \n" \
"pmtx.y = -0.0421935892309314 * iR +1.04845959175183 * iG - 0.00626600252090315 * iB \n" \
"-0.106438896887216 * iR*iG + 0.362908621470781 * iG*iB + 0.118070700472261 * iB*iR \n" \
"+0.0193542539838734 * iR*iR - 0.156083029543267 * iG*iG - 0.237811649496433 * iB*iB \n" \
"+1.67916420582198 * iR*iR*iG - 0.632835327167897 * iR*iR*iB - 1.95984471387461 * iR*iG*iG \n" \
"+0.953221464562814 * iR*iG*iB + 0.0599085176294623 * iR*iB*iB - 1.66452046236246 * iG*iG*iB + 1.14041188349761 * iG*iB*iB \n" \
"-0.387552623550308 * iR*iR*iR + 1.14820099685512 * iG*iG*iG - 0.336153941411709 * iB*iB*iB; \n" \
"pmtx.z = 0.170295033135028 * iR - 0.0682984448537245 * iG + 0.898003411718697 * iB \n" \
"+1.22106821992399 * iR*iG + 1.60194865922925 * iG*iB + 0.377599191137124 * iB*iR \n" \
"-0.825781428487531 * iR*iR - 1.44590868076749 * iG*iG - 0.928925961035344 * iB*iB \n" \
"-0.838548997455852 * iR*iR*iG + 0.75809397217116 * iR*iR*iB + 1.32966795243196 * iR*iG*iG \n" \
"-1.20021905668355 * iR*iG*iB - 0.254838995845129 * iR*iB*iB + 2.33232411639308 * iG*iG*iB - 1.86381505762773 * iG*iB*iB \n" \
"+0.111576038956423 * iR*iR*iR - 1.12593315849766 * iG*iG*iG + 0.751693186157287 * iB*iB*iB; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( pmtx.x); \n" \
"lin.y = CanonLog_to_lin( pmtx.y); \n" \
"lin.z = CanonLog_to_lin( pmtx.z); \n" \
"float3 aces; \n" \
"aces.x = 0.566996399f * lin.x + 0.365079418f * lin.y + 0.067924183f * lin.z; \n" \
"aces.y = 0.070901044f * lin.x + 0.880331008f * lin.y + 0.048767948f * lin.z; \n" \
"aces.z = 0.073013542f * lin.x - 0.066540862f * lin.y + 0.99352732f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C100mk2_A_D55( float3 In) { \n" \
"float iR, iG, iB; \n" \
"iR = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"iG = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"iB = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 pmtx; \n" \
"pmtx.x = 1.08190037262167 * iR -0.180298701368782 * iG +0.0983983287471069 * iB \n" \
"+1.9458545364518 * iR*iG -0.509539936937375 * iG*iB -0.47489567735516 * iB*iR \n" \
"-0.778086752197068 * iR*iR -0.7412266070049 * iG*iG +0.557894437042701 * iB*iB \n" \
"-3.27787395719078 * iR*iR*iG +0.254878417638717 * iR*iR*iB +3.45581530576474 * iR*iG*iG \n" \
"+0.335471713974739 * iR*iG*iB -0.43352125478476 * iR*iB*iB -1.65050137344141 * iG*iG*iB +1.46581418175682 * iG*iB*iB \n" \
"+0.944646566605676 * iR*iR*iR -0.723653099155881 * iG*iG*iG -0.371076501167857 * iB*iB*iB; \n" \
"pmtx.y = -0.00858997792576314 * iR +1.00673740119621 * iG +0.00185257672955608 * iB \n" \
"+0.0848736138296452 * iR*iG +0.347626906448902 * iG*iB +0.0020230274463939 * iB*iR \n" \
"-0.0790508414091524 * iR*iR -0.179497582958716 * iG*iG -0.175975123357072 * iB*iB \n" \
"+2.30205579706951 * iR*iR*iG -0.627257613385219 * iR*iR*iB -2.90795250918851 * iR*iG*iG \n" \
"+1.37002437502321 * iR*iG*iB -0.108668158565563 * iR*iB*iB -2.21150552827555 * iG*iG*iB + 1.53315057595445 * iG*iB*iB \n" \
"-0.543188706699505 * iR*iR*iR +1.63793038490376 * iG*iG*iG -0.444588616836587 * iB*iB*iB; \n" \
"pmtx.z = 0.12696639806511 * iR -0.011891441127869 * iG +0.884925043062759 * iB \n" \
"+1.34780279822258 * iR*iG +1.03647352257365 * iG*iB +0.459113289955922 * iB*iR \n" \
"-0.878157422295268 * iR*iR -1.3066278750436 * iG*iG -0.658604313413283 * iB*iB \n" \
"-1.4444077996703 * iR*iR*iG +0.556676588785173 * iR*iR*iB +2.18798497054968 * iR*iG*iG \n" \
"-1.43030768398665 * iR*iG*iB -0.0388323570817641 * iR*iB*iB +2.63698573112453 * iG*iG*iB -1.66598882056039 * iG*iB*iB \n" \
"+0.33450249360103 * iR*iR*iR -1.65856930730901 * iG*iG*iG +0.521956184547685 * iB*iB*iB; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( (pmtx.x * 876.0f + 64.0f) / 1023.0f ); \n" \
"lin.y = CanonLog_to_lin( (pmtx.y * 876.0f + 64.0f) / 1023.0f ); \n" \
"lin.z = CanonLog_to_lin( (pmtx.z * 876.0f + 64.0f) / 1023.0f ); \n" \
"float3 aces; \n" \
"aces.x = 0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926f * lin.z; \n" \
"aces.y = 0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821f * lin.z; \n" \
"aces.z = 0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C100mk2_A_Tng( float3 In) { \n" \
"float iR, iG, iB; \n" \
"iR = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"iG = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"iB = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 pmtx; \n" \
"pmtx.x = 0.963803004454899 * iR -0.160722202570655 * iG +0.196919198115756 * iB \n" \
"+2.03444685639819 * iR*iG -0.442676931451021 * iG*iB -0.407983781537509 * iB*iR \n" \
"-0.640703323129254 * iR*iR -0.860242798247848 * iG*iG +0.317159977967446 * iB*iB \n" \
"-4.80567080102966 * iR*iR*iG +0.27118370397567 * iR*iR*iB +5.1069005049557 * iR*iG*iG \n" \
"+0.340895816920585 * iR*iG*iB -0.486941738507862 * iR*iB*iB -2.23737935753692 * iG*iG*iB +1.96647555251297 * iG*iB*iB \n" \
"+1.30204051766243 * iR*iR*iR -1.06503117628554 * iG*iG*iG -0.392473022667378 * iB*iB*iB; \n" \
"pmtx.y = -0.0421935892309314 * iR +1.04845959175183 * iG -0.00626600252090315 * iB \n" \
"-0.106438896887216 * iR*iG +0.362908621470781 * iG*iB +0.118070700472261 * iB*iR \n" \
"+0.0193542539838734 * iR*iR -0.156083029543267 * iG*iG -0.237811649496433 * iB*iB \n" \
"+1.67916420582198 * iR*iR*iG -0.632835327167897 * iR*iR*iB -1.95984471387461 * iR*iG*iG \n" \
"+0.953221464562814 * iR*iG*iB +0.0599085176294623 * iR*iB*iB -1.66452046236246 * iG*iG*iB +1.14041188349761 * iG*iB*iB \n" \
"-0.387552623550308 * iR*iR*iR +1.14820099685512 * iG*iG*iG -0.336153941411709 * iB*iB*iB; \n" \
"pmtx.z = 0.170295033135028 * iR -0.0682984448537245 * iG +0.898003411718697 * iB \n" \
"+1.22106821992399 * iR*iG +1.60194865922925 * iG*iB +0.377599191137124 * iB*iR \n" \
"-0.825781428487531 * iR*iR -1.44590868076749 * iG*iG -0.928925961035344 * iB*iB \n" \
"-0.838548997455852 * iR*iR*iG +0.75809397217116 * iR*iR*iB +1.32966795243196 * iR*iG*iG \n" \
"-1.20021905668355 * iR*iG*iB -0.254838995845129 * iR*iB*iB +2.33232411639308 * iG*iG*iB -1.86381505762773 * iG*iB*iB \n" \
"+0.111576038956423 * iR*iR*iR -1.12593315849766 * iG*iG*iG +0.751693186157287 * iB*iB*iB; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( (pmtx.x * 876.0f + 64.0f) / 1023.0f ); \n" \
"lin.y = CanonLog_to_lin( (pmtx.y * 876.0f + 64.0f) / 1023.0f ); \n" \
"lin.z = CanonLog_to_lin( (pmtx.z * 876.0f + 64.0f) / 1023.0f ); \n" \
"float3 aces; \n" \
"aces.x = 0.566996399 * lin.x + 0.365079418 * lin.y + 0.067924183 * lin.z; \n" \
"aces.y = 0.070901044 * lin.x + 0.880331008 * lin.y + 0.048767948 * lin.z; \n" \
"aces.z = 0.073013542 * lin.x - 0.066540862 * lin.y + 0.99352732 * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300_A_D55( float3 In) { \n" \
"float iR, iG, iB; \n" \
"iR = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"iG = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"iB = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 pmtx; \n" \
"pmtx.x = 1.08190037262167 * iR -0.180298701368782 * iG +0.0983983287471069 * iB \n" \
"+1.9458545364518 * iR*iG -0.509539936937375 * iG*iB -0.47489567735516 * iB*iR \n" \
"-0.778086752197068 * iR*iR -0.7412266070049 * iG*iG +0.557894437042701 * iB*iB \n" \
"-3.27787395719078 * iR*iR*iG +0.254878417638717 * iR*iR*iB +3.45581530576474 * iR*iG*iG \n" \
"+0.335471713974739 * iR*iG*iB -0.43352125478476 * iR*iB*iB -1.65050137344141 * iG*iG*iB +1.46581418175682 * iG*iB*iB \n" \
"+0.944646566605676 * iR*iR*iR -0.723653099155881 * iG*iG*iG -0.371076501167857 * iB*iB*iB; \n" \
"pmtx.y = -0.00858997792576314 * iR +1.00673740119621 * iG +0.00185257672955608 * iB \n" \
"+0.0848736138296452 * iR*iG +0.347626906448902 * iG*iB +0.0020230274463939 * iB*iR \n" \
"-0.0790508414091524 * iR*iR -0.179497582958716 * iG*iG -0.175975123357072 * iB*iB \n" \
"+2.30205579706951 * iR*iR*iG -0.627257613385219 * iR*iR*iB -2.90795250918851 * iR*iG*iG \n" \
"+1.37002437502321 * iR*iG*iB -0.108668158565563 * iR*iB*iB -2.21150552827555 * iG*iG*iB + 1.53315057595445 * iG*iB*iB \n" \
"-0.543188706699505 * iR*iR*iR +1.63793038490376 * iG*iG*iG -0.444588616836587 * iB*iB*iB; \n" \
"pmtx.z = 0.12696639806511 * iR -0.011891441127869 * iG +0.884925043062759 * iB \n" \
"+1.34780279822258 * iR*iG +1.03647352257365 * iG*iB +0.459113289955922 * iB*iR \n" \
"-0.878157422295268 * iR*iR -1.3066278750436 * iG*iG -0.658604313413283 * iB*iB \n" \
"-1.4444077996703 * iR*iR*iG +0.556676588785173 * iR*iR*iB +2.18798497054968 * iR*iG*iG \n" \
"-1.43030768398665 * iR*iG*iB -0.0388323570817641 * iR*iB*iB +2.63698573112453 * iG*iG*iB -1.66598882056039 * iG*iB*iB \n" \
"+0.33450249360103 * iR*iR*iR -1.65856930730901 * iG*iG*iG +0.521956184547685 * iB*iB*iB; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( pmtx.x); \n" \
"lin.y = CanonLog_to_lin( pmtx.y); \n" \
"lin.z = CanonLog_to_lin( pmtx.z); \n" \
"float3 aces; \n" \
"aces.x = 0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926 * lin.z; \n" \
"aces.y = 0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821 * lin.z; \n" \
"aces.z = 0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204 * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300_A_Tng( float3 In) { \n" \
"float iR, iG, iB; \n" \
"iR = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"iG = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"iB = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 pmtx; \n" \
"pmtx.x = 0.963803004454899 * iR -0.160722202570655 * iG +0.196919198115756 * iB \n" \
"+2.03444685639819 * iR*iG -0.442676931451021 * iG*iB -0.407983781537509 * iB*iR \n" \
"-0.640703323129254 * iR*iR -0.860242798247848 * iG*iG +0.317159977967446 * iB*iB \n" \
"-4.80567080102966 * iR*iR*iG +0.27118370397567 * iR*iR*iB +5.1069005049557 * iR*iG*iG \n" \
"+0.340895816920585 * iR*iG*iB -0.486941738507862 * iR*iB*iB -2.23737935753692 * iG*iG*iB +1.96647555251297 * iG*iB*iB \n" \
"+1.30204051766243 * iR*iR*iR -1.06503117628554 * iG*iG*iG -0.392473022667378 * iB*iB*iB; \n" \
"pmtx.y = -0.0421935892309314 * iR +1.04845959175183 * iG -0.00626600252090315 * iB \n" \
"-0.106438896887216 * iR*iG +0.362908621470781 * iG*iB +0.118070700472261 * iB*iR \n" \
"+0.0193542539838734 * iR*iR -0.156083029543267 * iG*iG -0.237811649496433 * iB*iB \n" \
"+1.67916420582198 * iR*iR*iG -0.632835327167897 * iR*iR*iB -1.95984471387461 * iR*iG*iG \n" \
"+0.953221464562814 * iR*iG*iB +0.0599085176294623 * iR*iB*iB -1.66452046236246 * iG*iG*iB +1.14041188349761 * iG*iB*iB \n" \
"-0.387552623550308 * iR*iR*iR +1.14820099685512 * iG*iG*iG -0.336153941411709 * iB*iB*iB; \n" \
"pmtx.z = 0.170295033135028 * iR -0.0682984448537245 * iG +0.898003411718697 * iB \n" \
"+1.22106821992399 * iR*iG +1.60194865922925 * iG*iB +0.377599191137124 * iB*iR \n" \
"-0.825781428487531 * iR*iR -1.44590868076749 * iG*iG -0.928925961035344 * iB*iB \n" \
"-0.838548997455852 * iR*iR*iG +0.75809397217116 * iR*iR*iB +1.32966795243196 * iR*iG*iG \n" \
"-1.20021905668355 * iR*iG*iB -0.254838995845129 * iR*iB*iB +2.33232411639308 * iG*iG*iB -1.86381505762773 * iG*iB*iB \n" \
"+0.111576038956423 * iR*iR*iR -1.12593315849766 * iG*iG*iG +0.751693186157287 * iB*iB*iB; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( pmtx.x); \n" \
"lin.y = CanonLog_to_lin( pmtx.y); \n" \
"lin.z = CanonLog_to_lin( pmtx.z); \n" \
"float3 aces; \n" \
"aces.x = 0.566996399 * lin.x +0.365079418 * lin.y + 0.067924183 * lin.z; \n" \
"aces.y = 0.070901044 * lin.x +0.880331008 * lin.y + 0.048767948 * lin.z; \n" \
"aces.z = 0.073013542 * lin.x -0.066540862 * lin.y + 0.99352732 * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C500_A_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023 - 64) / 876; \n" \
"CLogIRE.y = (In.y * 1023 - 64) / 876; \n" \
"CLogIRE.z = (In.z * 1023 - 64) / 876; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.561538969f * lin.x +0.402060105f * lin.y + 0.036400926f * lin.z; \n" \
"aces.y = 0.092739623f * lin.x +0.924121198f * lin.y - 0.016860821f * lin.z; \n" \
"aces.z = 0.084812961f * lin.x +0.006373835f * lin.y + 0.908813204f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C500_A_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023 - 64) / 876; \n" \
"CLogIRE.y = (In.y * 1023 - 64) / 876; \n" \
"CLogIRE.z = (In.z * 1023 - 64) / 876; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.566996399 * lin.x +0.365079418 * lin.y + 0.067924183 * lin.z; \n" \
"aces.y = 0.070901044 * lin.x +0.880331008 * lin.y + 0.048767948 * lin.z; \n" \
"aces.z = 0.073013542 * lin.x -0.066540862 * lin.y + 0.99352732 * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C500_B_D55( float3 In) { \n" \
"float iR, iG, iB; \n" \
"iR = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"iG = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"iB = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 pmtx; \n" \
"pmtx.x = 1.08190037262167 * iR -0.180298701368782 * iG +0.0983983287471069 * iB \n" \
"+1.9458545364518 * iR*iG -0.509539936937375 * iG*iB -0.47489567735516 * iB*iR \n" \
"-0.778086752197068 * iR*iR -0.7412266070049 * iG*iG +0.557894437042701 * iB*iB \n" \
"-3.27787395719078 * iR*iR*iG +0.254878417638717 * iR*iR*iB +3.45581530576474 * iR*iG*iG \n" \
"+0.335471713974739 * iR*iG*iB -0.43352125478476 * iR*iB*iB -1.65050137344141 * iG*iG*iB + 1.46581418175682 * iG*iB*iB \n" \
"+0.944646566605676 * iR*iR*iR -0.723653099155881 * iG*iG*iG -0.371076501167857 * iB*iB*iB; \n" \
"pmtx.y = -0.00858997792576314 * iR +1.00673740119621 * iG +0.00185257672955608 * iB \n" \
"+0.0848736138296452 * iR*iG +0.347626906448902 * iG*iB +0.0020230274463939 * iB*iR \n" \
"-0.0790508414091524 * iR*iR -0.179497582958716 * iG*iG -0.175975123357072 * iB*iB \n" \
"+2.30205579706951 * iR*iR*iG -0.627257613385219 * iR*iR*iB -2.90795250918851 * iR*iG*iG \n" \
"+1.37002437502321 * iR*iG*iB -0.108668158565563 * iR*iB*iB -2.21150552827555 * iG*iG*iB + 1.53315057595445 * iG*iB*iB \n" \
"-0.543188706699505 * iR*iR*iR +1.63793038490376 * iG*iG*iG -0.444588616836587 * iB*iB*iB; \n" \
"pmtx.z = 0.12696639806511 * iR -0.011891441127869 * iG +0.884925043062759 * iB \n" \
"+1.34780279822258 * iR*iG +1.03647352257365 * iG*iB +0.459113289955922 * iB*iR \n" \
"-0.878157422295268 * iR*iR -1.3066278750436 * iG*iG -0.658604313413283 * iB*iB \n" \
"-1.4444077996703 * iR*iR*iG +0.556676588785173 * iR*iR*iB +2.18798497054968 * iR*iG*iG \n" \
"-1.43030768398665 * iR*iG*iB -0.0388323570817641 * iR*iB*iB +2.63698573112453 * iG*iG*iB - 1.66598882056039 * iG*iB*iB \n" \
"+0.33450249360103 * iR*iR*iR -1.65856930730901 * iG*iG*iG +0.521956184547685 * iB*iB*iB; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( pmtx.x); \n" \
"lin.y = CanonLog_to_lin( pmtx.y); \n" \
"lin.z = CanonLog_to_lin( pmtx.z); \n" \
"float3 aces; \n" \
"aces.x = 0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926f * lin.z; \n" \
"aces.y = 0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821f * lin.z; \n" \
"aces.z = 0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C500_B_Tng( float3 In) { \n" \
"float iR, iG, iB; \n" \
"iR = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"iG = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"iB = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 pmtx; \n" \
"pmtx.x = 0.963803004454899 * iR -0.160722202570655 * iG +0.196919198115756 * iB \n" \
"+2.03444685639819 * iR*iG -0.442676931451021 * iG*iB -0.407983781537509 * iB*iR \n" \
"-0.640703323129254 * iR*iR -0.860242798247848 * iG*iG +0.317159977967446 * iB*iB \n" \
"-4.80567080102966 * iR*iR*iG +0.27118370397567 * iR*iR*iB +5.1069005049557 * iR*iG*iG \n" \
"+0.340895816920585 * iR*iG*iB -0.486941738507862 * iR*iB*iB -2.23737935753692 * iG*iG*iB + 1.96647555251297 * iG*iB*iB \n" \
"+1.30204051766243 * iR*iR*iR -1.06503117628554 * iG*iG*iG -0.392473022667378 * iB*iB*iB; \n" \
"pmtx.y = -0.0421935892309314 * iR +1.04845959175183 * iG -0.00626600252090315 * iB \n" \
"-0.106438896887216 * iR*iG +0.362908621470781 * iG*iB +0.118070700472261 * iB*iR \n" \
"+0.0193542539838734 * iR*iR -0.156083029543267 * iG*iG -0.237811649496433 * iB*iB \n" \
"+1.67916420582198 * iR*iR*iG -0.632835327167897 * iR*iR*iB -1.95984471387461 * iR*iG*iG \n" \
"+0.953221464562814 * iR*iG*iB +0.0599085176294623 * iR*iB*iB -1.66452046236246 * iG*iG*iB + 1.14041188349761 * iG*iB*iB \n" \
"-0.387552623550308 * iR*iR*iR +1.14820099685512 * iG*iG*iG -0.336153941411709 * iB*iB*iB; \n" \
"pmtx.z = 0.170295033135028 * iR -0.0682984448537245 * iG +0.898003411718697 * iB \n" \
"+1.22106821992399 * iR*iG +1.60194865922925 * iG*iB +0.377599191137124 * iB*iR \n" \
"-0.825781428487531 * iR*iR -1.44590868076749 * iG*iG -0.928925961035344 * iB*iB \n" \
"-0.838548997455852 * iR*iR*iG +0.75809397217116 * iR*iR*iB +1.32966795243196 * iR*iG*iG \n" \
"-1.20021905668355 * iR*iG*iB -0.254838995845129 * iR*iB*iB +2.33232411639308 * iG*iG*iB - 1.86381505762773 * iG*iB*iB \n" \
"+0.111576038956423 * iR*iR*iR -1.12593315849766 * iG*iG*iG +0.751693186157287 * iB*iB*iB; \n" \
"float3 lin; \n" \
"lin.x = CanonLog_to_lin( pmtx.x); \n" \
"lin.y = CanonLog_to_lin( pmtx.y); \n" \
"lin.z = CanonLog_to_lin( pmtx.z); \n" \
"float3 aces; \n" \
"aces.x = 0.566996399 * lin.x +0.365079418 * lin.y + 0.067924183 * lin.z; \n" \
"aces.y = 0.070901044 * lin.x +0.880331008 * lin.y + 0.048767948 * lin.z; \n" \
"aces.z = 0.073013542 * lin.x -0.066540862 * lin.y + 0.99352732 * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C500_CinemaGamut_A_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z; \n" \
"aces.y = 0.003657457f * lin.x + 1.10696038f * lin.y - 0.110617837f * lin.z; \n" \
"aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C500_CinemaGamut_A_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z; \n" \
"aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z; \n" \
"aces.z = 0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C500_DCI_P3_A_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.607160575f * lin.x + 0.299507286f * lin.y + 0.093332140f * lin.z; \n" \
"aces.y = 0.004968120f * lin.x + 1.050982224f * lin.y - 0.055950343f * lin.z; \n" \
"aces.z = -0.007839939f * lin.x + 0.000809127f * lin.y + 1.007030813f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C500_DCI_P3_A_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.650279125f * lin.x + 0.253880169f * lin.y + 0.095840706f * lin.z; \n" \
"aces.y = -0.026137986f * lin.x + 1.017900530f * lin.y + 0.008237456f * lin.z; \n" \
"aces.z = 0.007757558f * lin.x - 0.063081669f * lin.y + 1.055324110f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog_BT2020_D_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.678891151f * lin.x + 0.158868422f * lin.y + 0.162240427f * lin.z; \n" \
"aces.y = 0.045570831f * lin.x + 0.860712772f * lin.y + 0.093716397f * lin.z; \n" \
"aces.z = -0.000485710f * lin.x + 0.025060196f * lin.y + 0.975425515f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.724488568f * lin.x + 0.115140904f * lin.y + 0.160370529f * lin.z; \n" \
"aces.y = 0.010659276f * lin.x + 0.839605344f * lin.y + 0.149735380f * lin.z; \n" \
"aces.z = 0.014560161f * lin.x - 0.028562057f * lin.y + 1.014001897f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z; \n" \
"aces.y = 0.003657457f * lin.x + 1.10696038f * lin.y - 0.110617837f * lin.z; \n" \
"aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z; \n" \
"aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z; \n" \
"aces.z = 0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog2_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog2_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog2_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.678891151f * lin.x + 0.158868422f * lin.y + 0.162240427f * lin.z; \n" \
"aces.y = 0.045570831f * lin.x + 0.860712772f * lin.y + 0.093716397f * lin.z; \n" \
"aces.z = -0.000485710f * lin.x + 0.025060196f * lin.y + 0.975425515f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog2_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog2_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog2_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.724488568f * lin.x + 0.115140904f * lin.y + 0.160370529f * lin.z; \n" \
"aces.y = 0.010659276f * lin.x + 0.839605344f * lin.y + 0.149735380f * lin.z; \n" \
"aces.z = 0.014560161f * lin.x - 0.028562057f * lin.y + 1.014001897f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog2_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog2_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog2_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z; \n" \
"aces.y = 0.003657457f * lin.x + 1.10696038f * lin.y - 0.110617837f * lin.z; \n" \
"aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog2_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog2_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog2_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z; \n" \
"aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z; \n" \
"aces.z = 0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog3_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog3_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog3_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.678891151f * lin.x + 0.158868422f * lin.y + 0.162240427f * lin.z; \n" \
"aces.y = 0.045570831f * lin.x + 0.860712772f * lin.y + 0.093716397f * lin.z; \n" \
"aces.z = -0.000485710f * lin.x + 0.025060196f * lin.y + 0.975425515f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog3_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog3_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog3_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.724488568f * lin.x + 0.115140904f * lin.y + 0.160370529f * lin.z; \n" \
"aces.y = 0.010659276f * lin.x + 0.839605344f * lin.y + 0.149735380f * lin.z; \n" \
"aces.z = 0.014560161f * lin.x - 0.028562057f * lin.y + 1.014001897f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog3_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog3_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog3_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z; \n" \
"aces.y = 0.003657457f * lin.x + 1.10696038f * lin.y - 0.110617837f * lin.z; \n" \
"aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng( float3 In) { \n" \
"float3 CLogIRE; \n" \
"CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f; \n" \
"CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f; \n" \
"float3 lin; \n" \
"lin.x = 0.9f * CanonLog3_to_lin( CLogIRE.x); \n" \
"lin.y = 0.9f * CanonLog3_to_lin( CLogIRE.y); \n" \
"lin.z = 0.9f * CanonLog3_to_lin( CLogIRE.z); \n" \
"float3 aces; \n" \
"aces.x = 0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z; \n" \
"aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z; \n" \
"aces.z = 0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z; \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Sony_SLog1_SGamut( float3 In) { \n" \
"mat3 SGAMUT_TO_ACES_MTX = { { 0.754338638f, 0.021198141f, -0.009756991f },  \n" \
"{ 0.133697046f, 1.005410934f, 0.004508563f },  \n" \
"{ 0.111968437f, -0.026610548f, 1.005253201f } }; \n" \
"float B = 64.0f; \n" \
"float AB = 90.0f; \n" \
"float W = 940.0f; \n" \
"float3 SLog; \n" \
"SLog.x = In.x * 1023.0f; \n" \
"SLog.y = In.y * 1023.0f; \n" \
"SLog.z = In.z * 1023.0f; \n" \
"float3 lin; \n" \
"lin.x = SLog1_to_lin( SLog.x, B, AB, W); \n" \
"lin.y = SLog1_to_lin( SLog.y, B, AB, W); \n" \
"lin.z = SLog1_to_lin( SLog.z, B, AB, W); \n" \
"float3 aces = mult_f3_f33( lin, SGAMUT_TO_ACES_MTX); \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Sony_SLog2_SGamut_Daylight( float3 In) { \n" \
"mat3 SGAMUT_DAYLIGHT_TO_ACES_MTX = { { 0.8764457030f, 0.0774075345f, 0.0573564351f},  \n" \
"{ 0.0145411681f, 0.9529571767f, -0.1151066335f},  \n" \
"{ 0.1090131290f, -0.0303647111f, 1.0577501984f} }; \n" \
"float B = 64.0f; \n" \
"float AB = 90.0f; \n" \
"float W = 940.0f; \n" \
"float3 SLog; \n" \
"SLog.x = In.x * 1023.0f; \n" \
"SLog.y = In.y * 1023.0f; \n" \
"SLog.z = In.z * 1023.0f; \n" \
"float3 lin; \n" \
"lin.x = SLog2_to_lin( SLog.x, B, AB, W); \n" \
"lin.y = SLog2_to_lin( SLog.y, B, AB, W); \n" \
"lin.z = SLog2_to_lin( SLog.z, B, AB, W); \n" \
"float3 aces = mult_f3_f33( lin, SGAMUT_DAYLIGHT_TO_ACES_MTX); \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Sony_SLog2_SGamut_Tungsten( float3 In) { \n" \
"mat3 SGAMUT_TUNG_TO_ACES_MTX = { { 1.0110238740f, 0.1011994504f, 0.0600766530f},  \n" \
"{ -0.1362526051f, 0.9562196265f, -0.1010185315f},  \n" \
"{ 0.1252287310f, -0.0574190769f, 1.0409418785f} }; \n" \
"float B = 64.0f; \n" \
"float AB = 90.0f; \n" \
"float W = 940.0f; \n" \
"float3 SLog; \n" \
"SLog.x = In.x * 1023.0f; \n" \
"SLog.y = In.y * 1023.0f; \n" \
"SLog.z = In.z * 1023.0f; \n" \
"float3 lin; \n" \
"lin.x = SLog2_to_lin( SLog.x, B, AB, W); \n" \
"lin.y = SLog2_to_lin( SLog.y, B, AB, W); \n" \
"lin.z = SLog2_to_lin( SLog.z, B, AB, W); \n" \
"float3 aces = mult_f3_f33( lin, SGAMUT_TUNG_TO_ACES_MTX); \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Sony_Venice_SGamut3( float3 linear) { \n" \
"mat3 matrixCoeff = { {0.7933297411f, 0.0155810585f, -0.0188647478f},  \n" \
"{0.0890786256f, 1.0327123069f, 0.0127694121f},  \n" \
"{0.1175916333f, -0.0482933654f, 1.0060953358f} }; \n" \
"float3 aces = mult_f3_f33(linear, matrixCoeff); \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Sony_Venice_SGamut3Cine( float3 linear) { \n" \
"mat3 matrixCoeff = { {0.6742570921f, -0.0093136061f, -0.0382090673f},  \n" \
"{0.2205717359f, 1.1059588614f, -0.0179383766f},  \n" \
"{0.1051711720f, -0.0966452553f, 1.0561474439f} }; \n" \
"float3 aces = mult_f3_f33(linear, matrixCoeff ); \n" \
"return aces; \n" \
"} \n" \
"float Y_2_linCV( float Y, float Ymax, float Ymin) { \n" \
"return (Y - Ymin) / (Ymax - Ymin); \n" \
"} \n" \
"float linCV_2_Y( float linCV, float Ymax, float Ymin) { \n" \
"return linCV * (Ymax - Ymin) + Ymin; \n" \
"} \n" \
"float3 Y_2_linCV_f3( float3 Y, float Ymax, float Ymin) { \n" \
"float3 linCV; \n" \
"linCV.x = Y_2_linCV( Y.x, Ymax, Ymin); linCV.y = Y_2_linCV( Y.y, Ymax, Ymin); linCV.z = Y_2_linCV( Y.z, Ymax, Ymin); \n" \
"return linCV; \n" \
"} \n" \
"float3 linCV_2_Y_f3( float3 linCV, float Ymax, float Ymin) { \n" \
"float3 Y; \n" \
"Y.x = linCV_2_Y( linCV.x, Ymax, Ymin); Y.y = linCV_2_Y( linCV.y, Ymax, Ymin); Y.z = linCV_2_Y( linCV.z, Ymax, Ymin); \n" \
"return Y; \n" \
"} \n" \
"float3 darkSurround_to_dimSurround( float3 linearCV) { \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"float3 xyY = XYZ_2_xyY(XYZ); \n" \
"xyY.z = fmax( xyY.z, 0.0f); \n" \
"xyY.z = pow( xyY.z, DIM_SURROUND_GAMMA); \n" \
"XYZ = xyY_2_XYZ(xyY); \n" \
"return mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"} \n" \
"float3 dimSurround_to_darkSurround( float3 linearCV) { \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"float3 xyY = XYZ_2_xyY(XYZ); \n" \
"xyY.z = fmax( xyY.z, 0.0f); \n" \
"xyY.z = pow( xyY.z, 1.0f / DIM_SURROUND_GAMMA); \n" \
"XYZ = xyY_2_XYZ(xyY); \n" \
"return mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"} \n" \
"float roll_white_fwd( float in, float new_wht, float width) { \n" \
"const float x0 = -1.0f; \n" \
"const float x1 = x0 + width; \n" \
"const float y0 = -new_wht; \n" \
"const float y1 = x1; \n" \
"const float m1 = (x1 - x0); \n" \
"const float a = y0 - y1 + m1; \n" \
"const float b = 2.0f * ( y1 - y0) - m1; \n" \
"const float c = y0; \n" \
"const float t = (-in - x0) / (x1 - x0); \n" \
"float out = 0.0f; \n" \
"if ( t < 0.0f) \n" \
"out = -(t * b + c); \n" \
"else if ( t > 1.0f) \n" \
"out = in; \n" \
"else \n" \
"out = -(( t * a + b) * t + c); \n" \
"return out; \n" \
"} \n" \
"float roll_white_rev( float in, float new_wht, float width) { \n" \
"const float x0 = -1.0f; \n" \
"const float x1 = x0 + width; \n" \
"const float y0 = -new_wht; \n" \
"const float y1 = x1; \n" \
"const float m1 = (x1 - x0); \n" \
"const float a = y0 - y1 + m1; \n" \
"const float b = 2.0f * ( y1 - y0) - m1; \n" \
"float c = y0; \n" \
"float out = 0.0f; \n" \
"if ( -in < y0) \n" \
"out = -x0; \n" \
"else if ( -in > y1) \n" \
"out = in; \n" \
"else { \n" \
"c = c + in; \n" \
"const float discrim = sqrt( b * b - 4.0f * a * c); \n" \
"const float t = ( 2.0f * c) / ( -discrim - b); \n" \
"out = -(( t * ( x1 - x0)) + x0); \n" \
"} \n" \
"return out; \n" \
"} \n" \
"float lookup_ACESmin( float minLum ) { \n" \
"float2 minTable[2] = { { log10(MIN_LUM_RRT), MIN_STOP_RRT }, { log10(MIN_LUM_SDR), MIN_STOP_SDR } }; \n" \
"return 0.18f * exp2(interpolate1D( minTable, 2, log10( minLum))); \n" \
"} \n" \
"float lookup_ACESmax( float maxLum ) { \n" \
"float2 maxTable[2] = { { log10(MAX_LUM_SDR), MAX_STOP_SDR }, { log10(MAX_LUM_RRT), MAX_STOP_RRT } }; \n" \
"return 0.18f * exp2(interpolate1D( maxTable, 2, log10( maxLum))); \n" \
"} \n" \
"Floater5 init_coefsLow( TsPoint TsPointLow, TsPoint TsPointMid) { \n" \
"Floater5 coefsLow; \n" \
"float knotIncLow = (log10(TsPointMid.x) - log10(TsPointLow.x)) / 3.0f; \n" \
"coefsLow.x = (TsPointLow.slope * (log10(TsPointLow.x) - 0.5f * knotIncLow)) + ( log10(TsPointLow.y) - TsPointLow.slope * log10(TsPointLow.x)); \n" \
"coefsLow.y = (TsPointLow.slope * (log10(TsPointLow.x) + 0.5f * knotIncLow)) + ( log10(TsPointLow.y) - TsPointLow.slope * log10(TsPointLow.x)); \n" \
"coefsLow.w = (TsPointMid.slope * (log10(TsPointMid.x) - 0.5f * knotIncLow)) + ( log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x)); \n" \
"coefsLow.m = (TsPointMid.slope * (log10(TsPointMid.x) + 0.5f * knotIncLow)) + ( log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x)); \n" \
"float2 bendsLow[2] = { {MIN_STOP_RRT, 0.18f}, {MIN_STOP_SDR, 0.35f} }; \n" \
"float pctLow = interpolate1D( bendsLow, 2, log2(TsPointLow.x / 0.18f)); \n" \
"coefsLow.z = log10(TsPointLow.y) + pctLow*(log10(TsPointMid.y) - log10(TsPointLow.y)); \n" \
"return coefsLow; \n" \
"} \n" \
"Floater5 init_coefsHigh( TsPoint TsPointMid, TsPoint TsPointMax) { \n" \
"Floater5 coefsHigh; \n" \
"float knotIncHigh = (log10(TsPointMax.x) - log10(TsPointMid.x)) / 3.0f; \n" \
"coefsHigh.x = (TsPointMid.slope * (log10(TsPointMid.x) - 0.5f * knotIncHigh)) + ( log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x)); \n" \
"coefsHigh.y = (TsPointMid.slope * (log10(TsPointMid.x) + 0.5f * knotIncHigh)) + ( log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x)); \n" \
"coefsHigh.w = (TsPointMax.slope * (log10(TsPointMax.x) - 0.5f * knotIncHigh)) + ( log10(TsPointMax.y) - TsPointMax.slope * log10(TsPointMax.x)); \n" \
"coefsHigh.m = (TsPointMax.slope * (log10(TsPointMax.x) + 0.5f * knotIncHigh)) + ( log10(TsPointMax.y) - TsPointMax.slope * log10(TsPointMax.x)); \n" \
"float2 bendsHigh[2] = { {MAX_STOP_SDR, 0.89f}, {MAX_STOP_RRT, 0.90f} }; \n" \
"float pctHigh = interpolate1D( bendsHigh, 2, log2(TsPointMax.x / 0.18f)); \n" \
"coefsHigh.z = log10(TsPointMid.y) + pctHigh*(log10(TsPointMax.y) - log10(TsPointMid.y)); \n" \
"return coefsHigh; \n" \
"} \n" \
"float shift( float in, float expShift) { \n" \
"return pow(2.0f, (log2(in) - expShift)); \n" \
"} \n" \
"TsParams init_TsParams( float minLum, float maxLum, float expShift) { \n" \
"TsPoint MIN_PT = { lookup_ACESmin(minLum), minLum, 0.0f}; \n" \
"TsPoint MID_PT = { 0.18f, 4.8f, 1.55f}; \n" \
"TsPoint MAX_PT = { lookup_ACESmax(maxLum), maxLum, 0.0f}; \n" \
"Floater5 cLow; \n" \
"cLow = init_coefsLow( MIN_PT, MID_PT); \n" \
"Floater5 cHigh; \n" \
"cHigh = init_coefsHigh( MID_PT, MAX_PT); \n" \
"MIN_PT.x = shift(lookup_ACESmin(minLum),expShift); \n" \
"MID_PT.x = shift(0.18f, expShift); \n" \
"MAX_PT.x = shift(lookup_ACESmax(maxLum),expShift); \n" \
"TsParams P = { {MIN_PT.x, MIN_PT.y, MIN_PT.slope}, {MID_PT.x, MID_PT.y, MID_PT.slope}, \n" \
"{MAX_PT.x, MAX_PT.y, MAX_PT.slope}, {cLow.x, cLow.y, cLow.z, cLow.w, cLow.m, cLow.m}, \n" \
"{cHigh.x, cHigh.y, cHigh.z, cHigh.w, cHigh.m, cHigh.m} }; \n" \
"return P; \n" \
"} \n" \
"float ssts( float x, TsParams C) { \n" \
"const int N_KNOTS_LOW = 4; \n" \
"const int N_KNOTS_HIGH = 4; \n" \
"float logx = log10( fmax(x, 1e-10f)); \n" \
"float logy = 0.0f; \n" \
"if ( logx <= log10(C.Min.x) ) { \n" \
"logy = logx * C.Min.slope + ( log10(C.Min.y) - C.Min.slope * log10(C.Min.x) ); \n" \
"} else if (( logx > log10(C.Min.x) ) && ( logx < log10(C.Mid.x) )) { \n" \
"float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10(C.Min.x)) / (log10(C.Mid.x) - log10(C.Min.x)); \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float3 cf = { C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]}; \n" \
"float3 monomials = { t * t, t, 1.0f }; \n" \
"logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM)); \n" \
"} else if (( logx >= log10(C.Mid.x) ) && ( logx < log10(C.Max.x) )) { \n" \
"float knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10(C.Mid.x)) / (log10(C.Max.x) - log10(C.Mid.x)); \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]}; \n" \
"float3 monomials = { t * t, t, 1.0f }; \n" \
"logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM)); \n" \
"} else { \n" \
"logy = logx * C.Max.slope + ( log10(C.Max.y) - C.Max.slope * log10(C.Max.x) ); \n" \
"} \n" \
"return exp10(logy); \n" \
"} \n" \
"float inv_ssts( float y, TsParams C) { \n" \
"const int N_KNOTS_LOW = 4; \n" \
"const int N_KNOTS_HIGH = 4; \n" \
"const float KNOT_INC_LOW = (log10(C.Mid.x) - log10(C.Min.x)) / (N_KNOTS_LOW - 1.0f); \n" \
"const float KNOT_INC_HIGH = (log10(C.Max.x) - log10(C.Mid.x)) / (N_KNOTS_HIGH - 1.0f); \n" \
"float KNOT_Y_LOW[4]; \n" \
"for (int i = 0; i < N_KNOTS_LOW; i = i + 1) { \n" \
"KNOT_Y_LOW[ i] = ( C.coefsLow[i] + C.coefsLow[i + 1]) / 2.0f; \n" \
"}; \n" \
"float KNOT_Y_HIGH[4]; \n" \
"for (int i = 0; i < N_KNOTS_HIGH; i = i + 1) { \n" \
"KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i + 1]) / 2.0f; \n" \
"}; \n" \
"float logy = log10( fmax(y, 1e-10f)); \n" \
"float logx; \n" \
"if (logy <= log10(C.Min.y)) { \n" \
"logx = log10(C.Min.x); \n" \
"} else if ( (logy > log10(C.Min.y)) && (logy <= log10(C.Mid.y)) ) { \n" \
"unsigned int j = 0; \n" \
"float3 cf = make_float3(0.0f, 0.0f, 0.0f); \n" \
"if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) { \n" \
"cf.x = C.coefsLow[0]; cf.y = C.coefsLow[1]; cf.z = C.coefsLow[2]; j = 0; \n" \
"} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) { \n" \
"cf.x = C.coefsLow[1]; cf.y = C.coefsLow[2]; cf.z = C.coefsLow[3]; j = 1; \n" \
"} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) { \n" \
"cf.x = C.coefsLow[2]; cf.y = C.coefsLow[3]; cf.z = C.coefsLow[4]; j = 2; \n" \
"} \n" \
"const float3 tmp = mult_f3_f33( cf, MM); \n" \
"float a = tmp.x; float b = tmp.y; float c = tmp.z; \n" \
"c = c - logy; \n" \
"const float d = sqrt( b * b - 4.0f * a * c); \n" \
"const float t = ( 2.0f * c) / ( -d - b); \n" \
"logx = log10(C.Min.x) + ( t + j) * KNOT_INC_LOW; \n" \
"} else if ( (logy > log10(C.Mid.y)) && (logy < log10(C.Max.y)) ) { \n" \
"unsigned int j = 0; \n" \
"float3 cf = make_float3(0.0f, 0.0f, 0.0f); \n" \
"if ( logy >= KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) { \n" \
"cf.x = C.coefsHigh[0]; cf.y = C.coefsHigh[1]; cf.z = C.coefsHigh[2]; j = 0; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) { \n" \
"cf.x = C.coefsHigh[1]; cf.y = C.coefsHigh[2]; cf.z = C.coefsHigh[3]; j = 1; \n" \
"} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) { \n" \
"cf.x = C.coefsHigh[2]; cf.y = C.coefsHigh[3]; cf.z = C.coefsHigh[4]; j = 2; \n" \
"} \n" \
"const float3 tmp = mult_f3_f33( cf, MM); \n" \
"float a = tmp.x; float b = tmp.y; float c = tmp.z; \n" \
"c = c - logy; \n" \
"const float d = sqrt( b * b - 4.0f * a * c); \n" \
"const float t = ( 2.0f * c) / ( -d - b); \n" \
"logx = log10(C.Mid.x) + ( t + j) * KNOT_INC_HIGH; \n" \
"} else { \n" \
"logx = log10(C.Max.x); \n" \
"} \n" \
"return exp10( logx); \n" \
"} \n" \
"float3 ssts_f3( float3 x, TsParams C) { \n" \
"float3 out; \n" \
"out.x = ssts( x.x, C); out.y = ssts( x.y, C); out.z = ssts( x.z, C); \n" \
"return out; \n" \
"} \n" \
"float3 inv_ssts_f3( float3 x, TsParams C) { \n" \
"float3 out; \n" \
"out.x = inv_ssts( x.x, C); out.y = inv_ssts( x.y, C); out.z = inv_ssts( x.z, C); \n" \
"return out; \n" \
"} \n" \
"float glow_fwd( float ycIn, float glowGainIn, float glowMid) { \n" \
"float glowGainOut; \n" \
"if (ycIn <= 2.0f/3.0f * glowMid) { \n" \
"glowGainOut = glowGainIn; \n" \
"} else if ( ycIn >= 2.0f * glowMid) { \n" \
"glowGainOut = 0.0f; \n" \
"} else { \n" \
"glowGainOut = glowGainIn * (glowMid / ycIn - 1.0f/2.0f); \n" \
"} \n" \
"return glowGainOut; \n" \
"} \n" \
"float glow_inv( float ycOut, float glowGainIn, float glowMid) { \n" \
"float glowGainOut; \n" \
"if (ycOut <= ((1.0f + glowGainIn) * 2.0f/3.0f * glowMid)) { \n" \
"glowGainOut = -glowGainIn / (1.0f + glowGainIn); \n" \
"} else if ( ycOut >= (2.0f * glowMid)) { \n" \
"glowGainOut = 0.0f; \n" \
"} else { \n" \
"glowGainOut = glowGainIn * (glowMid / ycOut - 1.0f/2.0f) / (glowGainIn / 2.0f - 1.0f); \n" \
"} \n" \
"return glowGainOut; \n" \
"} \n" \
"float sigmoid_shaper( float x) { \n" \
"float t = fmax( 1.0f - fabs( x / 2.0f), 0.0f); \n" \
"float y = 1.0f + _sign(x) * (1.0f - t * t); \n" \
"return y / 2.0f; \n" \
"} \n" \
"float cubic_basis_shaper ( float x, float w) { \n" \
"float M[4][4] = { {-1.0f/6.0f, 3.0f/6.0f,-3.0f/6.0f, 1.0f/6.0f}, {3.0f/6.0f, -6.0f/6.0f, 3.0f/6.0f, 0.0f/6.0f}, \n" \
"{-3.0f/6.0f, 0.0f/6.0f, 3.0f/6.0f, 0.0f/6.0f}, {1.0f/6.0f, 4.0f/6.0f, 1.0f/6.0f, 0.0f/6.0f} }; \n" \
"float knots[5] = { -w/2.0f, -w/4.0f, 0.0f, w/4.0f, w/2.0f }; \n" \
"float y = 0.0f; \n" \
"if ((x > knots[0]) && (x < knots[4])) { \n" \
"float knot_coord = (x - knots[0]) * 4.0f/w; \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float monomials[4] = { t*t*t, t*t, t, 1.0f }; \n" \
"if ( j == 3) { \n" \
"y = monomials[0] * M[0][0] + monomials[1] * M[1][0] + \n" \
"monomials[2] * M[2][0] + monomials[3] * M[3][0]; \n" \
"} else if ( j == 2) { \n" \
"y = monomials[0] * M[0][1] + monomials[1] * M[1][1] + \n" \
"monomials[2] * M[2][1] + monomials[3] * M[3][1]; \n" \
"} else if ( j == 1) { \n" \
"y = monomials[0] * M[0][2] + monomials[1] * M[1][2] + \n" \
"monomials[2] * M[2][2] + monomials[3] * M[3][2]; \n" \
"} else if ( j == 0) { \n" \
"y = monomials[0] * M[0][3] + monomials[1] * M[1][3] + \n" \
"monomials[2] * M[2][3] + monomials[3] * M[3][3]; \n" \
"} else { \n" \
"y = 0.0f;}} \n" \
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
"float3 rrt_sweeteners( float3 in) { \n" \
"float3 aces = in; \n" \
"float saturation = rgb_2_saturation( aces); \n" \
"float ycIn = rgb_2_yc( aces); \n" \
"float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f); \n" \
"float addedGlow = 1.0f + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID); \n" \
"aces = mult_f_f3( addedGlow, aces); \n" \
"float hue = rgb_2_hue( aces); \n" \
"float centeredHue = center_hue( hue, RRT_RED_HUE); \n" \
"float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH); \n" \
"aces.x = aces.x + hueWeight * saturation * (RRT_RED_PIVOT - aces.x) * (1.0f - RRT_RED_SCALE); \n" \
"aces = max_f3_f( aces, 0.0f); \n" \
"float3 rgbPre = mult_f3_f33( aces, AP0_2_AP1_MAT); \n" \
"rgbPre = max_f3_f( rgbPre, 0.0f); \n" \
"rgbPre = mult_f3_f33( rgbPre, calc_sat_adjust_matrix( RRT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"return rgbPre; \n" \
"} \n" \
"float3 inv_rrt_sweeteners( float3 in) { \n" \
"float3 rgbPost = in; \n" \
"rgbPost = mult_f3_f33( rgbPost, invert_f33(calc_sat_adjust_matrix( RRT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"rgbPost = max_f3_f( rgbPost, 0.0f); \n" \
"float3 aces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"aces = max_f3_f( aces, 0.0f); \n" \
"float hue = rgb_2_hue( aces); \n" \
"float centeredHue = center_hue( hue, RRT_RED_HUE); \n" \
"float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH); \n" \
"float minChan; \n" \
"if (centeredHue < 0.0f) { \n" \
"minChan = aces.y; \n" \
"} else { \n" \
"minChan = aces.z; \n" \
"} \n" \
"float a = hueWeight * (1.0f - RRT_RED_SCALE) - 1.0f; \n" \
"float b = aces.x - hueWeight * (RRT_RED_PIVOT + minChan) * (1.0f - RRT_RED_SCALE); \n" \
"float c = hueWeight * RRT_RED_PIVOT * minChan * (1.0f - RRT_RED_SCALE); \n" \
"aces.x = ( -b - sqrt( b * b - 4.0f * a * c)) / ( 2.0f * a); \n" \
"float saturation = rgb_2_saturation( aces); \n" \
"float ycOut = rgb_2_yc( aces); \n" \
"float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f); \n" \
"float reducedGlow = 1.0f + glow_inv( ycOut, RRT_GLOW_GAIN * s, RRT_GLOW_MID); \n" \
"aces = mult_f_f3( ( reducedGlow), aces); \n" \
"return aces; \n" \
"} \n" \
"float3 limit_to_primaries( float3 XYZ, Chromaticities LIMITING_PRI) { \n" \
"mat3 XYZ_2_LIMITING_PRI_MAT = XYZtoRGB( LIMITING_PRI); \n" \
"mat3 LIMITING_PRI_2_XYZ_MAT = RGBtoXYZ( LIMITING_PRI); \n" \
"float3 rgb = mult_f3_f33( XYZ, XYZ_2_LIMITING_PRI_MAT); \n" \
"float3 limitedRgb = clamp_f3( rgb, 0.0f, 1.0f); \n" \
"return mult_f3_f33( limitedRgb, LIMITING_PRI_2_XYZ_MAT); \n" \
"} \n" \
"float3 dark_to_dim( float3 XYZ) { \n" \
"float3 xyY = XYZ_2_xyY(XYZ); \n" \
"xyY.z = fmax( xyY.z, 0.0f); \n" \
"xyY.z = pow( xyY.z, DIM_SURROUND_GAMMA); \n" \
"return xyY_2_XYZ(xyY); \n" \
"} \n" \
"float3 dim_to_dark( float3 XYZ) { \n" \
"float3 xyY = XYZ_2_xyY(XYZ); \n" \
"xyY.z = fmax( xyY.z, 0.0f); \n" \
"xyY.z = pow( xyY.z, 1.0f / DIM_SURROUND_GAMMA); \n" \
"return xyY_2_XYZ(xyY); \n" \
"} \n" \
"float3 outputTransform \n" \
"( \n" \
"float3 in, \n" \
"float Y_MIN, \n" \
"float Y_MID, \n" \
"float Y_MAX, \n" \
"Chromaticities DISPLAY_PRI, \n" \
"Chromaticities LIMITING_PRI, \n" \
"int EOTF, \n" \
"int SURROUND, \n" \
"bool STRETCH_BLACK, \n" \
"bool D60_SIM, \n" \
"bool LEGAL_RANGE \n" \
") { \n" \
"mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI); \n" \
"TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX, 0.0f); \n" \
"float expShift = log2(inv_ssts(Y_MID, PARAMS_DEFAULT)) - log2(0.18f); \n" \
"TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift); \n" \
"float3 rgbPre = rrt_sweeteners( in); \n" \
"float3 rgbPost = ssts_f3( rgbPre, PARAMS); \n" \
"float3 linearCV = Y_2_linCV_f3( rgbPost, Y_MAX, Y_MIN); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"if (SURROUND == 0) { \n" \
"} else if (SURROUND == 1) { \n" \
"if ((EOTF == 1) || (EOTF == 2) || (EOTF == 3)) { \n" \
"XYZ = dark_to_dim( XYZ); \n" \
"}} else if (SURROUND == 2) { \n" \
"} \n" \
"XYZ = limit_to_primaries( XYZ, LIMITING_PRI); \n" \
"if (D60_SIM == false) { \n" \
"if ((DISPLAY_PRI.white.x != AP0.white.x) && (DISPLAY_PRI.white.y != AP0.white.y)) { \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"}} \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"if (D60_SIM == true) { \n" \
"float SCALE = 1.0f; \n" \
"if ((DISPLAY_PRI.white.x == 0.3127f) && (DISPLAY_PRI.white.y == 0.329f)) { \n" \
"SCALE = 0.96362f; \n" \
"} \n" \
"else if ((DISPLAY_PRI.white.x == 0.314f) && (DISPLAY_PRI.white.y == 0.351f)) { \n" \
"linearCV.x = roll_white_fwd( linearCV.x, 0.918f, 0.5f); \n" \
"linearCV.y = roll_white_fwd( linearCV.y, 0.918f, 0.5f); \n" \
"linearCV.z = roll_white_fwd( linearCV.z, 0.918f, 0.5f); \n" \
"SCALE = 0.96f; \n" \
"} \n" \
"linearCV = mult_f_f3( SCALE, linearCV); \n" \
"} \n" \
"linearCV = max_f3_f( linearCV, 0.0f); \n" \
"float3 outputCV; \n" \
"if (EOTF == 0) { \n" \
"if (STRETCH_BLACK == true) { \n" \
"outputCV = Y_2_ST2084_f3( max_f3_f( linCV_2_Y_f3(linearCV, Y_MAX, 0.0f), 0.0f) ); \n" \
"} else { \n" \
"outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) ); \n" \
"}} else if (EOTF == 1) { \n" \
"outputCV = bt1886_r_f3( linearCV, 2.4f, 1.0f, 0.0f); \n" \
"} else if (EOTF == 2) { \n" \
"outputCV = moncurve_r_f3( linearCV, 2.4f, 0.055f); \n" \
"} else if (EOTF == 3) { \n" \
"outputCV = pow_f3( linearCV, 1.0f/2.6f); \n" \
"} else if (EOTF == 4) { \n" \
"outputCV = linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN); \n" \
"} else if (EOTF == 5) { \n" \
"if (STRETCH_BLACK == true) { \n" \
"outputCV = Y_2_ST2084_f3( max_f3_f( linCV_2_Y_f3(linearCV, Y_MAX, 0.0f), 0.0f) ); \n" \
"} \n" \
"else { \n" \
"outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) ); \n" \
"} \n" \
"outputCV = ST2084_2_HLG_1000nits_f3( outputCV); \n" \
"} \n" \
"if (LEGAL_RANGE == true) { \n" \
"outputCV = fullRange_to_smpteRange_f3( outputCV); \n" \
"} \n" \
"return outputCV; \n" \
"} \n" \
"float3 invOutputTransform \n" \
"( \n" \
"float3 in, \n" \
"float Y_MIN, \n" \
"float Y_MID, \n" \
"float Y_MAX, \n" \
"Chromaticities DISPLAY_PRI, \n" \
"Chromaticities LIMITING_PRI, \n" \
"int EOTF, \n" \
"int SURROUND, \n" \
"bool STRETCH_BLACK, \n" \
"bool D60_SIM, \n" \
"bool LEGAL_RANGE \n" \
") { \n" \
"mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI); \n" \
"TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX, 0.0f); \n" \
"float expShift = log2(inv_ssts(Y_MID, PARAMS_DEFAULT)) - log2(0.18f); \n" \
"TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift); \n" \
"float3 outputCV = in; \n" \
"if (LEGAL_RANGE == true) { \n" \
"outputCV = smpteRange_to_fullRange_f3( outputCV); \n" \
"} \n" \
"float3 linearCV; \n" \
"if (EOTF == 0) { \n" \
"if (STRETCH_BLACK == true) { \n" \
"linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0f); \n" \
"} else { \n" \
"linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN); \n" \
"}} else if (EOTF == 1) { \n" \
"linearCV = bt1886_f_f3( outputCV, 2.4f, 1.0f, 0.0f); \n" \
"} else if (EOTF == 2) { \n" \
"linearCV = moncurve_f_f3( outputCV, 2.4f, 0.055f); \n" \
"} else if (EOTF == 3) { \n" \
"linearCV = pow_f3( outputCV, 2.6f); \n" \
"} else if (EOTF == 4) { \n" \
"linearCV = Y_2_linCV_f3( outputCV, Y_MAX, Y_MIN); \n" \
"} else if (EOTF == 5) { \n" \
"outputCV = HLG_2_ST2084_1000nits_f3( outputCV); \n" \
"if (STRETCH_BLACK == true) { \n" \
"linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0f); \n" \
"} else { \n" \
"linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN); \n" \
"}} \n" \
"if (D60_SIM == true) { \n" \
"float SCALE = 1.0f; \n" \
"if ((DISPLAY_PRI.white.x == 0.3127f) && (DISPLAY_PRI.white.y == 0.329f)) { \n" \
"SCALE = 0.96362f; \n" \
"linearCV = mult_f_f3( 1.0f / SCALE, linearCV); \n" \
"} \n" \
"else if ((DISPLAY_PRI.white.x == 0.314f) && (DISPLAY_PRI.white.y == 0.351f)) { \n" \
"SCALE = 0.96f; \n" \
"linearCV.x = roll_white_rev( linearCV.x / SCALE, 0.918f, 0.5f); \n" \
"linearCV.y = roll_white_rev( linearCV.y / SCALE, 0.918f, 0.5f); \n" \
"linearCV.z = roll_white_rev( linearCV.z / SCALE, 0.918f, 0.5f); \n" \
"}} \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"if (D60_SIM == false) { \n" \
"if ((DISPLAY_PRI.white.x != AP0.white.x) && (DISPLAY_PRI.white.y != AP0.white.y)) { \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33(calculate_cat_matrix( AP0.white, REC709_PRI.white)) ); \n" \
"}} \n" \
"if (SURROUND == 0) { \n" \
"} else if (SURROUND == 1) { \n" \
"if ((EOTF == 1) || (EOTF == 2) || (EOTF == 3)) { \n" \
"XYZ = dim_to_dark( XYZ); \n" \
"}} else if (SURROUND == 2) { \n" \
"} \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"float3 rgbPost = linCV_2_Y_f3( linearCV, Y_MAX, Y_MIN); \n" \
"float3 rgbPre = inv_ssts_f3( rgbPost, PARAMS); \n" \
"float3 aces = inv_rrt_sweeteners( rgbPre); \n" \
"return aces; \n" \
"} \n" \
"float3 InvODT_Rec709( float3 outputCV) { \n" \
"mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(REC709_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float L_W = 1.0f; \n" \
"float L_B = 0.0f; \n" \
"float3 linearCV = bt1886_f_f3( outputCV, DISPGAMMA, L_W, L_B); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white))); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"float3 rgbPre = linCV_2_Y_f3( linearCV, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost = segmented_spline_c9_rev_f3( rgbPre); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_sRGB( float3 outputCV) { \n" \
"mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(REC709_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float OFFSET = 0.055f; \n" \
"float3 linearCV; \n" \
"linearCV = moncurve_f_f3( outputCV, DISPGAMMA, OFFSET); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white))); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"float3 rgbPre = linCV_2_Y_f3( linearCV, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost = segmented_spline_c9_rev_f3( rgbPre); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 IDT_sRGB( float3 rgb) { \n" \
"float3 aces; \n" \
"aces = InvODT_sRGB(rgb); \n" \
"aces = segmented_spline_c5_rev_f3( aces); \n" \
"aces = max_f3_f(aces, 0.0f); \n" \
"return aces; \n" \
"} \n" \
"float3 IDT_Rec709( float3 rgb) { \n" \
"float3 aces; \n" \
"aces = InvODT_Rec709(rgb); \n" \
"aces = segmented_spline_c5_rev_f3( aces); \n" \
"aces = max_f3_f(aces, 0.0f); \n" \
"return aces; \n" \
"} \n" \
"float3 ASCCDL_inACEScct( float3 acesIn, float3 SLOPE, float3 OFFSET, float3 POWER, float SAT) { \n" \
"float3 acescct = ACES_to_ACEScct( acesIn); \n" \
"acescct.x = pow( clamp( (acescct.x * SLOPE.x) + OFFSET.x, 0.0f, 1.0f), 1.0f / POWER.x); \n" \
"acescct.y = pow( clamp( (acescct.y * SLOPE.y) + OFFSET.y, 0.0f, 1.0f), 1.0f / POWER.y); \n" \
"acescct.z = pow( clamp( (acescct.z * SLOPE.z) + OFFSET.z, 0.0f, 1.0f), 1.0f / POWER.z); \n" \
"float luma = 0.2126f * acescct.x + 0.7152f * acescct.y + 0.0722f * acescct.z; \n" \
"float satClamp = fmax(SAT, 0.0f); \n" \
"acescct.x = luma + satClamp * (acescct.x - luma); \n" \
"acescct.y = luma + satClamp * (acescct.y - luma); \n" \
"acescct.z = luma + satClamp * (acescct.z - luma); \n" \
"return ACEScct_to_ACES( acescct); \n" \
"} \n" \
"float3 gamma_adjust_linear( float3 rgbIn, float GAMMA, float PIVOT) { \n" \
"const float SCALAR = PIVOT / pow( PIVOT, GAMMA); \n" \
"float3 rgbOut = rgbIn; \n" \
"if (rgbIn.x > 0.0f) rgbOut.x = pow( rgbIn.x, GAMMA) * SCALAR; \n" \
"if (rgbIn.y > 0.0f) rgbOut.y = pow( rgbIn.y, GAMMA) * SCALAR; \n" \
"if (rgbIn.z > 0.0f) rgbOut.z = pow( rgbIn.z, GAMMA) * SCALAR; \n" \
"return rgbOut; \n" \
"} \n" \
"float3 sat_adjust( float3 rgbIn, float SAT_FACTOR) { \n" \
"float3 RGB2Y = make_float3(RGBtoXYZ( REC709_PRI).c0.y, RGBtoXYZ( REC709_PRI).c1.y, RGBtoXYZ( REC709_PRI).c2.y); \n" \
"const mat3 SAT_MAT = calc_sat_adjust_matrix( SAT_FACTOR, RGB2Y); \n" \
"return mult_f3_f33( rgbIn, SAT_MAT); \n" \
"} \n" \
"float3 rgb_2_yab( float3 rgb) { \n" \
"float3 yab = mult_f3_f33( rgb, make_mat3(make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f),  \n" \
"make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4),  \n" \
"make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4))); \n" \
"return yab; \n" \
"} \n" \
"float3 yab_2_rgb( float3 yab) { \n" \
"float3 rgb = mult_f3_f33( yab, invert_f33(make_mat3(make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f),  \n" \
"make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4)))); \n" \
"return rgb; \n" \
"} \n" \
"float3 yab_2_ych(float3 yab) { \n" \
"float3 ych = yab; \n" \
"float yo = yab.y * yab.y + yab.z * yab.z; \n" \
"ych.y = sqrt(yo); \n" \
"ych.z = atan2(yab.z, yab.y) * (180.0f / 3.1415926535897932f); \n" \
"if (ych.z < 0.0f) ych.z += 360.0f; \n" \
"return ych; \n" \
"} \n" \
"float3 ych_2_yab( float3 ych ) { \n" \
"float3 yab; \n" \
"yab.x = ych.x; \n" \
"float h = ych.z * (3.1415926535897932f / 180.0f); \n" \
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
"float3 scale_C_at_H( float3 rgb, float centerH, float widthH, float percentC) { \n" \
"float3 new_rgb = rgb; \n" \
"float3 ych = rgb_2_ych( rgb); \n" \
"if (ych.y > 0.0f) { \n" \
"float centeredHue = center_hue( ych.z, centerH); \n" \
"float f_H = cubic_basis_shaper( centeredHue, widthH); \n" \
"if (f_H > 0.0f) { \n" \
"float3 new_ych = ych; \n" \
"new_ych.y = ych.y * (f_H * (percentC - 1.0f) + 1.0f); \n" \
"new_rgb = ych_2_rgb( new_ych); \n" \
"} else { \n" \
"new_rgb = rgb; \n" \
"}} \n" \
"return new_rgb; \n" \
"} \n" \
"float3 rotate_H_in_H( float3 rgb, float centerH, float widthH, float degreesShift) { \n" \
"float3 ych = rgb_2_ych( rgb); \n" \
"float3 new_ych = ych; \n" \
"float centeredHue = center_hue( ych.z, centerH); \n" \
"float f_H = cubic_basis_shaper( centeredHue, widthH); \n" \
"float old_hue = centeredHue; \n" \
"float new_hue = centeredHue + degreesShift; \n" \
"float2 table[2] = { {0.0f, old_hue}, {1.0f, new_hue} }; \n" \
"float blended_hue = interpolate1D( table, 2, f_H); \n" \
"if (f_H > 0.0f) new_ych.z = uncenter_hue(blended_hue, centerH); \n" \
"return ych_2_rgb( new_ych); \n" \
"} \n" \
"float3 scale_C( float3 rgb, float percentC) { \n" \
"float3 ych = rgb_2_ych( rgb); \n" \
"ych.y = ych.y * percentC; \n" \
"return ych_2_rgb( ych); \n" \
"} \n" \
"float3 overlay_f3( float3 a, float3 b) { \n" \
"const float LUMA_CUT = lin_to_ACEScct( 0.5f); \n" \
"float luma = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z; \n" \
"float3 out; \n" \
"if (luma < LUMA_CUT) { \n" \
"out.x = 2.0f * a.x * b.x; \n" \
"out.y = 2.0f * a.y * b.y; \n" \
"out.z = 2.0f * a.z * b.z; \n" \
"} else { \n" \
"out.x = 1.0f - (2.0f * (1.0f - a.x) * (1.0f - b.x)); \n" \
"out.y = 1.0f - (2.0f * (1.0f - a.y) * (1.0f - b.y)); \n" \
"out.z = 1.0f - (2.0f * (1.0f - a.z) * (1.0f - b.z)); \n" \
"} \n" \
"return out; \n" \
"} \n" \
"float3 LMT_PFE( float3 aces) { \n" \
"aces = scale_C( aces, 0.7f); \n" \
"float3 SLOPE = make_float3(1.0f, 1.0f, 0.94f); \n" \
"float3 OFFSET = make_float3(0.0f, 0.0f, 0.02f); \n" \
"float3 POWER = make_float3(1.0f, 1.0f, 1.0f); \n" \
"aces = ASCCDL_inACEScct( aces, SLOPE, OFFSET, POWER, 1.0f); \n" \
"aces = gamma_adjust_linear( aces, 1.5f, 0.18f); \n" \
"aces = rotate_H_in_H( aces, 0.0f, 30.0f, 5.0f); \n" \
"aces = rotate_H_in_H( aces, 80.0f, 60.0f, -15.0f); \n" \
"aces = rotate_H_in_H( aces, 52.0f, 50.0f, -14.0f); \n" \
"aces = scale_C_at_H( aces, 45.0f, 40.0f, 1.4f); \n" \
"aces = rotate_H_in_H( aces, 190.0f, 40.0f, 30.0f); \n" \
"aces = scale_C_at_H( aces, 240.0f, 120.0f, 1.4f); \n" \
"return aces; \n" \
"} \n" \
"float3 LMT_Bleach( float3 aces) { \n" \
"float3 a, b, blend; \n" \
"a = sat_adjust( aces, 0.9f); \n" \
"a = mult_f_f3( 2.0f, a); \n" \
"b = sat_adjust( aces, 0.0f); \n" \
"b = gamma_adjust_linear( b, 1.2f, 0.18f); \n" \
"a = ACES_to_ACEScct( a); \n" \
"b = ACES_to_ACEScct( b); \n" \
"blend = overlay_f3( a, b); \n" \
"aces = ACEScct_to_ACES( blend); \n" \
"return aces; \n" \
"} \n" \
"float3 LMT_BlueLightArtifactFix( float3 aces) { \n" \
"mat3 correctionMatrix = \n" \
"{ {0.9404372683f, 0.0083786969f, 0.0005471261f }, \n" \
"{-0.0183068787f, 0.8286599939f, -0.0008833746f }, \n" \
"{ 0.0778696104f, 0.1629613092f, 1.0003362486f } }; \n" \
"float3 acesMod = mult_f3_f33( aces, correctionMatrix); \n" \
"return acesMod; \n" \
"} \n" \
"float3 RRT( float3 aces) { \n" \
"float saturation = rgb_2_saturation( aces); \n" \
"float ycIn = rgb_2_yc( aces); \n" \
"float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f); \n" \
"float addedGlow = 1.0f + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID); \n" \
"aces = mult_f_f3( addedGlow, aces); \n" \
"float hue = rgb_2_hue( aces); \n" \
"float centeredHue = center_hue( hue, RRT_RED_HUE); \n" \
"float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH); \n" \
"aces.x = aces.x + hueWeight * saturation * (RRT_RED_PIVOT - aces.x) * (1.0f - RRT_RED_SCALE); \n" \
"aces = max_f3_f( aces, 0.0f); \n" \
"float3 rgbPre = mult_f3_f33( aces, AP0_2_AP1_MAT); \n" \
"rgbPre = max_f3_f( rgbPre, 0.0f); \n" \
"rgbPre = mult_f3_f33( rgbPre, calc_sat_adjust_matrix( RRT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c5_fwd( rgbPre.x); \n" \
"rgbPost.y = segmented_spline_c5_fwd( rgbPre.y); \n" \
"rgbPost.z = segmented_spline_c5_fwd( rgbPre.z); \n" \
"float3 rgbOces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return rgbOces; \n" \
"} \n" \
"float3 InvRRT( float3 oces) { \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c5_rev( rgbPre.x); \n" \
"rgbPost.y = segmented_spline_c5_rev( rgbPre.y); \n" \
"rgbPost.z = segmented_spline_c5_rev( rgbPre.z); \n" \
"rgbPost = mult_f3_f33( rgbPost, invert_f33(calc_sat_adjust_matrix( RRT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"rgbPost = max_f3_f( rgbPost, 0.0f); \n" \
"float3 aces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"aces = max_f3_f( aces, 0.0f); \n" \
"float hue = rgb_2_hue( aces); \n" \
"float centeredHue = center_hue( hue, RRT_RED_HUE); \n" \
"float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH); \n" \
"float minChan; \n" \
"if (centeredHue < 0.0f) { \n" \
"minChan = aces.y; \n" \
"} else { \n" \
"minChan = aces.z; \n" \
"} \n" \
"float a = hueWeight * (1.0f - RRT_RED_SCALE) - 1.0f; \n" \
"float b = aces.x - hueWeight * (RRT_RED_PIVOT + minChan) * (1.0f - RRT_RED_SCALE); \n" \
"float c = hueWeight * RRT_RED_PIVOT * minChan * (1.0f - RRT_RED_SCALE); \n" \
"aces.x = ( -b - sqrt( b * b - 4.0f * a * c)) / ( 2.0f * a); \n" \
"float saturation = rgb_2_saturation( aces); \n" \
"float ycOut = rgb_2_yc( aces); \n" \
"float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f); \n" \
"float reducedGlow = 1.0f + glow_inv( ycOut, RRT_GLOW_GAIN * s, RRT_GLOW_MID); \n" \
"aces = mult_f_f3( ( reducedGlow), aces); \n" \
"return aces; \n" \
"} \n" \
"float3 ODT_Rec709_100nits_dim( float3 oces) { \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float L_W = 1.0f; \n" \
"float L_B = 0.0f; \n" \
"bool legalRange = false; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B); \n" \
"outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B); \n" \
"outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B); \n" \
"if(legalRange) outputCV = fullRange_to_smpteRange_f3( outputCV); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_Rec709_D60sim_100nits_dim( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.4f; \n" \
"const float L_W = 1.0f; \n" \
"const float L_B = 0.0f; \n" \
"const float SCALE = 0.955f; \n" \
"bool legalRange = false; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.x = fmin( linearCV.x, 1.0f) * SCALE; \n" \
"linearCV.y = fmin( linearCV.y, 1.0f) * SCALE; \n" \
"linearCV.z = fmin( linearCV.z, 1.0f) * SCALE; \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B); \n" \
"outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B); \n" \
"outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B); \n" \
"if (legalRange) { \n" \
"outputCV = fullRange_to_smpteRange_f3( outputCV); \n" \
"} \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_Rec2020_100nits_dim( float3 oces) { \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float L_W = 1.0f; \n" \
"float L_B = 0.0f; \n" \
"bool legalRange = false; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B); \n" \
"outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B); \n" \
"outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B); \n" \
"if (legalRange) { \n" \
"outputCV = fullRange_to_smpteRange_f3( outputCV); \n" \
"} \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_Rec2020_ST2084_1000nits( float3 oces) { \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_1000nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_1000nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_1000nits()); \n" \
"rgbPost = add_f_f3( -exp10(-4.4550166483f), rgbPost); \n" \
"float3 XYZ = mult_f3_f33( rgbPost, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"float3 rgb = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"rgb = max_f3_f( rgb, 0.0f); \n" \
"float3 outputCV = Y_2_ST2084_f3( rgb); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_Rec2020_Rec709limited_100nits_dim( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"const Chromaticities LIMITING_PRI = REC709_PRI; \n" \
"const float DISPGAMMA = 2.4f; \n" \
"const float L_W = 1.0f; \n" \
"const float L_B = 0.0f; \n" \
"bool legalRange = false; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"XYZ = limit_to_primaries( XYZ, LIMITING_PRI); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B); \n" \
"outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B); \n" \
"outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B); \n" \
"if (legalRange) { \n" \
"outputCV = fullRange_to_smpteRange_f3( outputCV); \n" \
"} \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_Rec2020_P3D65limited_100nits_dim( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"const Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"const float DISPGAMMA = 2.4f; \n" \
"const float L_W = 1.0f; \n" \
"const float L_B = 0.0f; \n" \
"bool legalRange = false; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"XYZ = limit_to_primaries( XYZ, LIMITING_PRI); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B); \n" \
"outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B); \n" \
"outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B); \n" \
"if (legalRange) { \n" \
"outputCV = fullRange_to_smpteRange_f3( outputCV); \n" \
"} \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_sRGB_D60sim_100nits_dim( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.4f; \n" \
"const float OFFSET = 0.055f; \n" \
"const float SCALE = 0.955f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.x = fmin( linearCV.x, 1.0f) * SCALE; \n" \
"linearCV.y = fmin( linearCV.y, 1.0f) * SCALE; \n" \
"linearCV.z = fmin( linearCV.z, 1.0f) * SCALE; \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET); \n" \
"outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET); \n" \
"outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_sRGB_100nits_dim( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.4f; \n" \
"const float OFFSET = 0.055f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET); \n" \
"outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET); \n" \
"outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_P3DCI_48nits( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = P3DCI_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"const float NEW_WHT = 0.918f; \n" \
"const float ROLL_WIDTH = 0.5f; \n" \
"const float SCALE = 0.96f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.x = fmin( linearCV.x, NEW_WHT) * SCALE; \n" \
"linearCV.y = fmin( linearCV.y, NEW_WHT) * SCALE; \n" \
"linearCV.z = fmin( linearCV.z, NEW_WHT) * SCALE; \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_P3DCI_D60sim_48nits( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = P3DCI_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"const float NEW_WHT = 0.918; \n" \
"const float ROLL_WIDTH = 0.5; \n" \
"const float SCALE = 0.96; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.x = fmin( linearCV.x, NEW_WHT) * SCALE; \n" \
"linearCV.y = fmin( linearCV.y, NEW_WHT) * SCALE; \n" \
"linearCV.z = fmin( linearCV.z, NEW_WHT) * SCALE; \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_P3DCI_D65sim_48nits( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = P3DCI_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"const float NEW_WHT = 0.908f; \n" \
"const float ROLL_WIDTH = 0.5f; \n" \
"const float SCALE = 0.9575f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.x = fmin( linearCV.x, NEW_WHT) * SCALE; \n" \
"linearCV.y = fmin( linearCV.y, NEW_WHT) * SCALE; \n" \
"linearCV.z = fmin( linearCV.z, NEW_WHT) * SCALE; \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_P3D60_48nits( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = P3D60_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_P3D65_48nits( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_P3D65_D60sim_48nits( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"const float SCALE = 0.964f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.x = fmin( linearCV.x, 1.0f) * SCALE; \n" \
"linearCV.y = fmin( linearCV.y, 1.0f) * SCALE; \n" \
"linearCV.z = fmin( linearCV.z, 1.0f) * SCALE; \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_P3D65_Rec709limited_48nits( float3 oces) { \n" \
"const Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"const mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"const Chromaticities LIMITING_PRI = REC709_PRI; \n" \
"const float DISPGAMMA = 2.6f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"XYZ = limit_to_primaries( XYZ, LIMITING_PRI); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_DCDM( float3 oces) { \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = max_f3_f( XYZ, 0.0f); \n" \
"float3 outputCV = dcdm_encode( XYZ); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_DCDM_P3D60limited( float3 oces) { \n" \
"const Chromaticities LIMITING_PRI = P3D60_PRI; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = limit_to_primaries( XYZ, LIMITING_PRI); \n" \
"float3 outputCV = dcdm_encode( XYZ); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_DCDM_P3D65limited( float3 oces) { \n" \
"const Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = limit_to_primaries( XYZ, LIMITING_PRI); \n" \
"float3 outputCV = dcdm_encode( XYZ); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_RGBmonitor_100nits_dim( float3 oces) { \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float OFFSET = 0.055f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"XYZ = mult_f3_f33( XYZ, calculate_cat_matrix( AP0.white, REC709_PRI.white)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET); \n" \
"outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET); \n" \
"outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET); \n" \
"return outputCV; \n" \
"} \n" \
"float3 ODT_RGBmonitor_D60sim_100nits_dim( float3 oces) { \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float OFFSET = 0.055f; \n" \
"float SCALE = 0.955f; \n" \
"float3 rgbPre = mult_f3_f33( oces, AP0_2_AP1_MAT); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits()); \n" \
"float3 linearCV; \n" \
"linearCV.x = Y_2_linCV( rgbPost.x, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.y = Y_2_linCV( rgbPost.y, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.z = Y_2_linCV( rgbPost.z, 48.0f, exp10(log10(0.02f))); \n" \
"linearCV.x = fmin( linearCV.x, 1.0f) * SCALE; \n" \
"linearCV.y = fmin( linearCV.y, 1.0f) * SCALE; \n" \
"linearCV.z = fmin( linearCV.z, 1.0f) * SCALE; \n" \
"linearCV = darkSurround_to_dimSurround( linearCV); \n" \
"linearCV = mult_f3_f33( linearCV, calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y))); \n" \
"float3 XYZ = mult_f3_f33( linearCV, RGBtoXYZ( AP1)); \n" \
"linearCV = mult_f3_f33( XYZ, XYZ_2_DISPLAY_PRI_MAT); \n" \
"linearCV = clamp_f3( linearCV, 0.0f, 1.0f); \n" \
"float3 outputCV; \n" \
"outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET); \n" \
"outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET); \n" \
"outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET); \n" \
"return outputCV; \n" \
"} \n" \
"float3 InvODT_Rec709_100nits_dim( float3 outputCV) { \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float L_W = 1.0f; \n" \
"float L_B = 0.0f; \n" \
"bool legalRange = false; \n" \
"if (legalRange) { \n" \
"outputCV = smpteRange_to_fullRange_f3( outputCV); \n" \
"} \n" \
"float3 linearCV; \n" \
"linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B); \n" \
"linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B); \n" \
"linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white))); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"linearCV = dimSurround_to_darkSurround( linearCV); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_Rec709_D60sim_100nits_dim( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.4f; \n" \
"const float L_W = 1.0f; \n" \
"const float L_B = 0.0f; \n" \
"const float SCALE = 0.955f; \n" \
"bool legalRange = false; \n" \
"if (legalRange) { \n" \
"outputCV = smpteRange_to_fullRange_f3( outputCV); \n" \
"} \n" \
"float3 linearCV; \n" \
"linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B); \n" \
"linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B); \n" \
"linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"linearCV = dimSurround_to_darkSurround( linearCV); \n" \
"linearCV.x = linearCV.x / SCALE; \n" \
"linearCV.y = linearCV.y / SCALE; \n" \
"linearCV.z = linearCV.z / SCALE; \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_Rec2020_100nits_dim( float3 outputCV) { \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float L_W = 1.0f; \n" \
"float L_B = 0.0f; \n" \
"bool legalRange = false; \n" \
"if (legalRange) { \n" \
"outputCV = smpteRange_to_fullRange_f3( outputCV); \n" \
"} \n" \
"float3 linearCV; \n" \
"linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B); \n" \
"linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B); \n" \
"linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white))); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"linearCV = dimSurround_to_darkSurround( linearCV); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_Rec2020_ST2084_1000nits( float3 outputCV) { \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"float3 rgb = ST2084_2_Y_f3( outputCV); \n" \
"float3 XYZ = mult_f3_f33( rgb, DISPLAY_PRI_2_XYZ_MAT); \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white))); \n" \
"float3 rgbPre = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"rgbPre = add_f_f3( exp10(-4.4550166483f), rgbPre); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_1000nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_1000nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_1000nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_sRGB_D60sim_100nits_dim( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.4f; \n" \
"const float OFFSET = 0.055f; \n" \
"const float SCALE = 0.955f; \n" \
"float3 linearCV; \n" \
"linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET); \n" \
"linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET); \n" \
"linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"linearCV = dimSurround_to_darkSurround( linearCV); \n" \
"linearCV.x = linearCV.x / SCALE; \n" \
"linearCV.y = linearCV.y / SCALE; \n" \
"linearCV.z = linearCV.z / SCALE; \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_sRGB_100nits_dim( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.4f; \n" \
"const float OFFSET = 0.055f; \n" \
"float3 linearCV; \n" \
"linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET); \n" \
"linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET); \n" \
"linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white))); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"linearCV = dimSurround_to_darkSurround( linearCV); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_P3DCI_48nits( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = P3DCI_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"const float NEW_WHT = 0.918f; \n" \
"const float ROLL_WIDTH = 0.5f; \n" \
"const float SCALE = 0.96f; \n" \
"float3 linearCV = pow_f3( outputCV, DISPGAMMA); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_P3DCI_D60sim_48nits( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = P3DCI_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"const float NEW_WHT = 0.918f; \n" \
"const float ROLL_WIDTH = 0.5f; \n" \
"const float SCALE = 0.96f; \n" \
"float3 linearCV = pow_f3( outputCV, DISPGAMMA); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_P3DCI_D65sim_48nits( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = P3DCI_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"const float NEW_WHT = 0.908f; \n" \
"const float ROLL_WIDTH = 0.5f; \n" \
"const float SCALE = 0.9575f; \n" \
"float3 linearCV = pow_f3( outputCV, DISPGAMMA); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_P3D60_48nits( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = P3D60_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"float3 linearCV = pow_f3( outputCV, DISPGAMMA); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_P3D65_48nits( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"float3 linearCV = pow_f3( outputCV, DISPGAMMA); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_P3D65_D60sim_48nits( float3 outputCV) { \n" \
"const Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"const mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI); \n" \
"const float DISPGAMMA = 2.6f; \n" \
"const float SCALE = 0.964f; \n" \
"float3 linearCV = pow_f3( outputCV, DISPGAMMA); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV.x = linearCV.x / SCALE; \n" \
"linearCV.y = linearCV.y / SCALE; \n" \
"linearCV.z = linearCV.z / SCALE; \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_DCDM( float3 outputCV) { \n" \
"float3 XYZ = dcdm_decode( outputCV); \n" \
"float3 linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_DCDM_P3D65limited( float3 outputCV) { \n" \
"float3 XYZ = dcdm_decode( outputCV); \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33(calculate_cat_matrix( AP0.white, REC709_PRI.white))); \n" \
"float3 linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_RGBmonitor_100nits_dim( float3 outputCV) { \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float OFFSET = 0.055f; \n" \
"float3 linearCV; \n" \
"linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET); \n" \
"linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET); \n" \
"linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"XYZ = mult_f3_f33( XYZ, invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white))); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"linearCV = dimSurround_to_darkSurround( linearCV); \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 InvODT_RGBmonitor_D60sim_100nits_dim( float3 outputCV) { \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI); \n" \
"float DISPGAMMA = 2.4f; \n" \
"float OFFSET = 0.055f; \n" \
"float SCALE = 0.955f; \n" \
"float3 linearCV; \n" \
"linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET); \n" \
"linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET); \n" \
"linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET); \n" \
"float3 XYZ = mult_f3_f33( linearCV, DISPLAY_PRI_2_XYZ_MAT); \n" \
"linearCV = mult_f3_f33( XYZ, XYZtoRGB( AP1)); \n" \
"linearCV = mult_f3_f33( linearCV, invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR,  \n" \
"make_float3(RGBtoXYZ( AP1).c0.y, RGBtoXYZ( AP1).c1.y, RGBtoXYZ( AP1).c2.y)))); \n" \
"linearCV = dimSurround_to_darkSurround( linearCV); \n" \
"linearCV.x = linearCV.x / SCALE; \n" \
"linearCV.y = linearCV.y / SCALE; \n" \
"linearCV.z = linearCV.z / SCALE; \n" \
"float3 rgbPre; \n" \
"rgbPre.x = linCV_2_Y( linearCV.x, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.y = linCV_2_Y( linearCV.y, 48.0f, exp10(log10(0.02f))); \n" \
"rgbPre.z = linCV_2_Y( linearCV.z, 48.0f, exp10(log10(0.02f))); \n" \
"float3 rgbPost; \n" \
"rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits()); \n" \
"rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits()); \n" \
"rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits()); \n" \
"float3 oces = mult_f3_f33( rgbPost, AP1_2_AP0_MAT); \n" \
"return oces; \n" \
"} \n" \
"float3 RRTODT_P3D65_108nits_7_2nits_ST2084( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 7.2f; \n" \
"float Y_MAX = 108.0f; \n" \
"Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_P3D65_1000nits_15nits_ST2084( float3 aces) { \n" \
"const float Y_MIN = 0.0001; \n" \
"const float Y_MID = 15.0; \n" \
"const float Y_MAX = 1000.0; \n" \
"const Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"const Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"const int EOTF = 0; \n" \
"const int SURROUND = 0; \n" \
"const bool STRETCH_BLACK = true; \n" \
"const bool D60_SIM = false;                        \n" \
"const bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_P3D65_2000nits_15nits_ST2084( float3 aces) { \n" \
"const float Y_MIN = 0.0001; \n" \
"const float Y_MID = 15.0; \n" \
"const float Y_MAX = 2000.0; \n" \
"const Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"const Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"const int EOTF = 0; \n" \
"const int SURROUND = 0; \n" \
"const bool STRETCH_BLACK = true; \n" \
"const bool D60_SIM = false;                        \n" \
"const bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_P3D65_4000nits_15nits_ST2084( float3 aces) { \n" \
"const float Y_MIN = 0.0001; \n" \
"const float Y_MID = 15.0; \n" \
"const float Y_MAX = 4000.0; \n" \
"const Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"const Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"const int EOTF = 0; \n" \
"const int SURROUND = 0; \n" \
"const bool STRETCH_BLACK = true; \n" \
"const bool D60_SIM = false;                        \n" \
"const bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_Rec2020_1000nits_15nits_HLG( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 1000.0f; \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"Chromaticities LIMITING_PRI = REC2020_PRI; \n" \
"int EOTF = 5; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_Rec2020_1000nits_15nits_ST2084( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 1000.0f; \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"Chromaticities LIMITING_PRI = REC2020_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_Rec2020_2000nits_15nits_ST2084( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 2000.0f; \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"Chromaticities LIMITING_PRI = REC2020_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_Rec2020_4000nits_15nits_ST2084( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 4000.0f; \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"Chromaticities LIMITING_PRI = REC2020_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_Rec709_100nits_10nits_BT1886( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 10.0f; \n" \
"float Y_MAX = 100.0f; \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"Chromaticities LIMITING_PRI = REC709_PRI; \n" \
"int EOTF = 1; \n" \
"int SURROUND = 1; \n" \
"bool STRETCH_BLACK = false; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 RRTODT_Rec709_100nits_10nits_sRGB( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 10.0f; \n" \
"float Y_MAX = 100.0f; \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"Chromaticities LIMITING_PRI = REC709_PRI; \n" \
"int EOTF = 2; \n" \
"int SURROUND = 1; \n" \
"bool STRETCH_BLACK = false; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 InvRRTODT_P3D65_108nits_7_2nits_ST2084( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 7.2f; \n" \
"float Y_MAX = 108.0f; \n" \
"Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 InvRRTODT_P3D65_1000nits_15nits_ST2084( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 1000.0f; \n" \
"Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 InvRRTODT_P3D65_2000nits_15nits_ST2084( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 2000.0f; \n" \
"Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 InvRRTODT_P3D65_4000nits_15nits_ST2084( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 4000.0f; \n" \
"Chromaticities DISPLAY_PRI = P3D65_PRI; \n" \
"Chromaticities LIMITING_PRI = P3D65_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 InvRRTODT_Rec2020_1000nits_15nits_HLG( float3 cv) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 1000.0f; \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"Chromaticities LIMITING_PRI = REC2020_PRI; \n" \
"int EOTF = 5; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return aces; \n" \
"} \n" \
"float3 InvRRTODT_Rec2020_1000nits_15nits_ST2084( float3 cv) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 1000.0f; \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"Chromaticities LIMITING_PRI = REC2020_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return aces; \n" \
"} \n" \
"float3 InvRRTODT_Rec2020_2000nits_15nits_ST2084( float3 cv) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 2000.0f; \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"Chromaticities LIMITING_PRI = REC2020_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return aces; \n" \
"} \n" \
"float3 InvRRTODT_Rec2020_4000nits_15nits_ST2084( float3 cv) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 15.0f; \n" \
"float Y_MAX = 4000.0f; \n" \
"Chromaticities DISPLAY_PRI = REC2020_PRI; \n" \
"Chromaticities LIMITING_PRI = REC2020_PRI; \n" \
"int EOTF = 0; \n" \
"int SURROUND = 0; \n" \
"bool STRETCH_BLACK = true; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return aces; \n" \
"} \n" \
"float3 InvRRTODT_Rec709_100nits_10nits_BT1886( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 10.0f; \n" \
"float Y_MAX = 100.0f; \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"Chromaticities LIMITING_PRI = REC709_PRI; \n" \
"int EOTF = 1; \n" \
"int SURROUND = 1; \n" \
"bool STRETCH_BLACK = false; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"float3 InvRRTODT_Rec709_100nits_10nits_sRGB( float3 aces) { \n" \
"float Y_MIN = 0.0001f; \n" \
"float Y_MID = 10.0f; \n" \
"float Y_MAX = 100.0f; \n" \
"Chromaticities DISPLAY_PRI = REC709_PRI; \n" \
"Chromaticities LIMITING_PRI = REC709_PRI; \n" \
"int EOTF = 2; \n" \
"int SURROUND = 1; \n" \
"bool STRETCH_BLACK = false; \n" \
"bool D60_SIM = false; \n" \
"bool LEGAL_RANGE = false; \n" \
"float3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, \n" \
"LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"return cv; \n" \
"} \n" \
"kernel void k_Simple( const device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Output[index] = p_Input[index];  \n" \
"p_Output[index + 1] = p_Input[index + 1];  \n" \
"p_Output[index + 2] = p_Input[index + 2];  \n" \
"p_Output[index + 3] = p_Input[index + 3];  \n" \
"}} \n" \
"kernel void k_CSCIN( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant int& p_CSCIN [[buffer (4)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces; \n" \
"aces.x = p_Input[index]; \n" \
"aces.y = p_Input[index + 1]; \n" \
"aces.z = p_Input[index + 2]; \n" \
"switch (p_CSCIN){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{aces = ACEScc_to_ACES(aces);} \n" \
"break; \n" \
"case 2: \n" \
"{aces = ACEScct_to_ACES(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = ACEScg_to_ACES(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = ACESproxy_to_ACES(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = ADX_to_ACES(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = ICpCt_to_ACES(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = LogC_EI800_AWG_to_ACES(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = Log3G10_RWG_to_ACES(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = SLog3_SG3_to_ACES(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = SLog3_SG3C_to_ACES(aces);} \n" \
"} \n" \
"p_Input[index] = aces.x;  \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"}} \n" \
"kernel void k_IDT( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant int& p_IDT [[buffer (5)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces; \n" \
"aces.x = p_Input[index]; \n" \
"aces.y = p_Input[index + 1]; \n" \
"aces.z = p_Input[index + 2]; \n" \
"switch (p_IDT){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{aces = IDT_Alexa_v3_raw_EI800_CCT6500(aces);} \n" \
"break; \n" \
"case 2: \n" \
"{aces = IDT_Panasonic_V35(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = IDT_Canon_C100_A_D55(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = IDT_Canon_C100_A_Tng(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = IDT_Canon_C100mk2_A_D55(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = IDT_Canon_C100mk2_A_Tng(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = IDT_Canon_C300_A_D55(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = IDT_Canon_C300_A_Tng(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = IDT_Canon_C500_A_D55(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = IDT_Canon_C500_A_Tng(aces);} \n" \
"break; \n" \
"case 11: \n" \
"{aces = IDT_Canon_C500_B_D55(aces);} \n" \
"break; \n" \
"case 12: \n" \
"{aces = IDT_Canon_C500_B_Tng(aces);} \n" \
"break; \n" \
"case 13: \n" \
"{aces = IDT_Canon_C500_CinemaGamut_A_D55(aces);} \n" \
"break; \n" \
"case 14: \n" \
"{aces = IDT_Canon_C500_CinemaGamut_A_Tng(aces);} \n" \
"break; \n" \
"case 15: \n" \
"{aces = IDT_Canon_C500_DCI_P3_A_D55(aces);} \n" \
"break; \n" \
"case 16: \n" \
"{aces = IDT_Canon_C500_DCI_P3_A_Tng(aces);} \n" \
"break; \n" \
"case 17: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_D55(aces);} \n" \
"break; \n" \
"case 18: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng(aces);} \n" \
"break; \n" \
"case 19: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55(aces);} \n" \
"break; \n" \
"case 20: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng(aces);} \n" \
"break; \n" \
"case 21: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55(aces);} \n" \
"break; \n" \
"case 22: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng(aces);} \n" \
"break; \n" \
"case 23: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55(aces);} \n" \
"break; \n" \
"case 24: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng(aces);} \n" \
"break; \n" \
"case 25: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55(aces);} \n" \
"break; \n" \
"case 26: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng(aces);} \n" \
"break; \n" \
"case 27: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55(aces);} \n" \
"break; \n" \
"case 28: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng(aces);} \n" \
"break; \n" \
"case 29: \n" \
"{aces = IDT_Sony_SLog1_SGamut(aces);} \n" \
"break; \n" \
"case 30: \n" \
"{aces = IDT_Sony_SLog2_SGamut_Daylight(aces);} \n" \
"break; \n" \
"case 31: \n" \
"{aces = IDT_Sony_SLog2_SGamut_Tungsten(aces);} \n" \
"break; \n" \
"case 32: \n" \
"{aces = IDT_Sony_Venice_SGamut3(aces);} \n" \
"break; \n" \
"case 33: \n" \
"{aces = IDT_Sony_Venice_SGamut3Cine(aces);} \n" \
"break; \n" \
"case 34: \n" \
"{aces = IDT_Rec709(aces);} \n" \
"break; \n" \
"case 35: \n" \
"{aces = IDT_sRGB(aces);} \n" \
"} \n" \
"p_Input[index] = aces.x;  \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"}} \n" \
"kernel void k_Exposure( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]],  \n" \
"constant int& p_Height [[buffer (3)]], constant float& p_Exposure [[buffer (6)]], uint2 id [[ thread_position_in_grid ]]) {                                 				    \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"p_Input[index] = p_Input[index] * exp2(p_Exposure); \n" \
"p_Input[index + 1] = p_Input[index + 1] * exp2(p_Exposure); \n" \
"p_Input[index + 2] = p_Input[index + 2] * exp2(p_Exposure); \n" \
"}} \n" \
"kernel void k_LMT( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant int& p_LMT [[buffer (7)]], constant float* p_LMTScale [[buffer (13)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces; \n" \
"aces.x = p_Input[index]; \n" \
"aces.y = p_Input[index + 1]; \n" \
"aces.z = p_Input[index + 2]; \n" \
"switch (p_LMT){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{ \n" \
"if(p_LMTScale[0] != 1.0f) \n" \
"aces = scale_C(aces, p_LMTScale[0]); \n" \
"if(!(p_LMTScale[1] == 1.0f && p_LMTScale[2] == 0.0f && p_LMTScale[3] == 1.0f)) { \n" \
"float3 SLOPE = {p_LMTScale[1], p_LMTScale[1], p_LMTScale[1]}; \n" \
"float3 OFFSET = {p_LMTScale[2], p_LMTScale[2], p_LMTScale[2]}; \n" \
"float3 POWER = {p_LMTScale[3], p_LMTScale[3], p_LMTScale[3]}; \n" \
"aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f); \n" \
"} \n" \
"if(p_LMTScale[4] != 1.0f) \n" \
"aces = gamma_adjust_linear(aces, p_LMTScale[4], p_LMTScale[5]); \n" \
"if(p_LMTScale[8] != 0.0f) \n" \
"aces = rotate_H_in_H(aces, p_LMTScale[6], p_LMTScale[7], p_LMTScale[8]); \n" \
"if(p_LMTScale[11] != 0.0f) \n" \
"aces = rotate_H_in_H(aces, p_LMTScale[9], p_LMTScale[10], p_LMTScale[11]); \n" \
"if(p_LMTScale[14] != 0.0f) \n" \
"aces = rotate_H_in_H(aces, p_LMTScale[12], p_LMTScale[13], p_LMTScale[14]); \n" \
"if(p_LMTScale[17] != 1.0f) \n" \
"aces = scale_C_at_H(aces, p_LMTScale[15], p_LMTScale[16], p_LMTScale[17]); \n" \
"if(p_LMTScale[20] != 0.0f) \n" \
"aces = rotate_H_in_H(aces, p_LMTScale[18], p_LMTScale[19], p_LMTScale[20]); \n" \
"if(p_LMTScale[23] != 1.0f) \n" \
"aces = scale_C_at_H(aces, p_LMTScale[21], p_LMTScale[22], p_LMTScale[23]); \n" \
"} \n" \
"break; \n" \
"case 2: \n" \
"{aces = LMT_BlueLightArtifactFix(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = LMT_PFE(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{ \n" \
"if(p_LMTScale[0] != 1.0f) \n" \
"aces = scale_C(aces, p_LMTScale[0]); \n" \
"float3 SLOPE = {p_LMTScale[1], p_LMTScale[1], p_LMTScale[1] * 0.94f}; \n" \
"float3 OFFSET = {p_LMTScale[2], p_LMTScale[2], p_LMTScale[2] + 0.02f}; \n" \
"float3 POWER = {p_LMTScale[3], p_LMTScale[3], p_LMTScale[3]}; \n" \
"aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f); \n" \
"if(p_LMTScale[4] != 1.0f) \n" \
"aces = gamma_adjust_linear(aces, p_LMTScale[4], p_LMTScale[5]); \n" \
"if(p_LMTScale[8] != 0.0f) \n" \
"aces = rotate_H_in_H(aces, p_LMTScale[6], p_LMTScale[7], p_LMTScale[8]); \n" \
"if(p_LMTScale[11] != 0.0f) \n" \
"aces = rotate_H_in_H(aces, p_LMTScale[9], p_LMTScale[10], p_LMTScale[11]); \n" \
"if(p_LMTScale[14] != 0.0f) \n" \
"aces = rotate_H_in_H(aces, p_LMTScale[12], p_LMTScale[13], p_LMTScale[14]); \n" \
"if(p_LMTScale[17] != 1.0f) \n" \
"aces = scale_C_at_H(aces, p_LMTScale[15], p_LMTScale[16], p_LMTScale[17]); \n" \
"if(p_LMTScale[20] != 0.0f) \n" \
"aces = rotate_H_in_H(aces, p_LMTScale[18], p_LMTScale[19], p_LMTScale[20]); \n" \
"if(p_LMTScale[23] != 1.0f) \n" \
"aces = scale_C_at_H(aces, p_LMTScale[21], p_LMTScale[22], p_LMTScale[23]); \n" \
"} \n" \
"break; \n" \
"case 5: \n" \
"{aces = LMT_Bleach(aces);}} \n" \
"p_Input[index] = aces.x;  \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"}} \n" \
"kernel void k_CSCOUT( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]],  \n" \
"constant int& p_CSCOUT [[buffer (8)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces; \n" \
"aces.x = p_Input[index]; \n" \
"aces.y = p_Input[index + 1]; \n" \
"aces.z = p_Input[index + 2]; \n" \
"switch (p_CSCOUT){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{aces = ACES_to_ACEScc(aces);} \n" \
"break; \n" \
"case 2: \n" \
"{aces = ACES_to_ACEScct(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = ACES_to_ACEScg(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = ACES_to_ACESproxy(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = ACES_to_ADX(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = ACES_to_ICpCt(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = ACES_to_LogC_EI800_AWG(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = ACES_to_Log3G10_RWG(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = ACES_to_SLog3_SG3(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = ACES_to_SLog3_SG3C(aces);} \n" \
"} \n" \
"p_Input[index] = aces.x;  \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"}} \n" \
"kernel void k_RRT( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], \n" \
"constant int& p_RRT [[buffer (9)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces; \n" \
"aces.x = p_Input[index]; \n" \
"aces.y = p_Input[index + 1]; \n" \
"aces.z = p_Input[index + 2]; \n" \
"aces = RRT(aces); \n" \
"p_Input[index] = aces.x;  \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"}} \n" \
"kernel void k_InvRRT( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], \n" \
"constant int& p_InvRRT [[buffer (10)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces; \n" \
"aces.x = p_Input[index]; \n" \
"aces.y = p_Input[index + 1]; \n" \
"aces.z = p_Input[index + 2]; \n" \
"aces = InvRRT(aces); \n" \
"p_Input[index] = aces.x;  \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"}} \n" \
"kernel void k_ODT( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], \n" \
"constant int& p_ODT [[buffer (11)]], constant float* p_Lum [[buffer (14)]], constant int& p_DISPLAY [[buffer (15)]],  \n" \
"constant int& p_LIMIT [[buffer (16)]], constant int& p_EOTF [[buffer (17)]], constant int& p_SURROUND [[buffer (18)]],  \n" \
"constant int* p_Switch [[buffer (19)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces; \n" \
"aces.x = p_Input[index]; \n" \
"aces.y = p_Input[index + 1]; \n" \
"aces.z = p_Input[index + 2]; \n" \
"switch (p_ODT){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{ \n" \
"float Y_MIN = p_Lum[0] * 0.0001f; \n" \
"float Y_MID = p_Lum[1]; \n" \
"float Y_MAX = p_Lum[2]; \n" \
"Chromaticities DISPLAY_PRI = p_DISPLAY == 0 ? REC2020_PRI : p_DISPLAY == 1 ? P3D60_PRI : p_DISPLAY == 2 ? P3D65_PRI : p_DISPLAY == 3 ? P3DCI_PRI : REC709_PRI; \n" \
"Chromaticities LIMITING_PRI = p_LIMIT == 0 ? REC2020_PRI : p_LIMIT == 1 ? P3D60_PRI : p_LIMIT == 2 ? P3D65_PRI : p_LIMIT == 3 ? P3DCI_PRI : REC709_PRI; \n" \
"int EOTF = p_EOTF; \n" \
"int SURROUND = p_SURROUND; \n" \
"bool STRETCH_BLACK = p_Switch[0] == 1; \n" \
"bool D60_SIM = p_Switch[1] == 1; \n" \
"bool LEGAL_RANGE = p_Switch[2] == 1; \n" \
"aces = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"} \n" \
"break; \n" \
"case 2: \n" \
"{aces = ODT_Rec709_100nits_dim(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = ODT_Rec709_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = ODT_sRGB_100nits_dim(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = ODT_sRGB_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = ODT_Rec2020_100nits_dim(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = ODT_Rec2020_Rec709limited_100nits_dim(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = ODT_Rec2020_P3D65limited_100nits_dim(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = ODT_Rec2020_ST2084_1000nits(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = ODT_P3DCI_48nits(aces);} \n" \
"break; \n" \
"case 11: \n" \
"{aces = ODT_P3DCI_D60sim_48nits(aces);} \n" \
"break; \n" \
"case 12: \n" \
"{aces = ODT_P3DCI_D65sim_48nits(aces);} \n" \
"break; \n" \
"case 13: \n" \
"{aces = ODT_P3D60_48nits(aces);} \n" \
"break; \n" \
"case 14: \n" \
"{aces = ODT_P3D65_48nits(aces);} \n" \
"break; \n" \
"case 15: \n" \
"{aces = ODT_P3D65_D60sim_48nits(aces);} \n" \
"break; \n" \
"case 16: \n" \
"{aces = ODT_P3D65_Rec709limited_48nits(aces);} \n" \
"break; \n" \
"case 17: \n" \
"{aces = ODT_DCDM(aces);} \n" \
"break; \n" \
"case 18: \n" \
"{aces = ODT_DCDM_P3D60limited(aces);} \n" \
"break; \n" \
"case 19: \n" \
"{aces = ODT_DCDM_P3D65limited(aces);} \n" \
"break; \n" \
"case 20: \n" \
"{aces = ODT_RGBmonitor_100nits_dim(aces);} \n" \
"break; \n" \
"case 21: \n" \
"{aces = ODT_RGBmonitor_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 22: \n" \
"{aces = RRTODT_Rec709_100nits_10nits_BT1886(aces);} \n" \
"break; \n" \
"case 23: \n" \
"{aces = RRTODT_Rec709_100nits_10nits_sRGB(aces);} \n" \
"break; \n" \
"case 24: \n" \
"{aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);} \n" \
"break; \n" \
"case 25: \n" \
"{aces = RRTODT_P3D65_1000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 26: \n" \
"{aces = RRTODT_P3D65_2000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 27: \n" \
"{aces = RRTODT_P3D65_4000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 28: \n" \
"{aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);} \n" \
"break; \n" \
"case 29: \n" \
"{aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 30: \n" \
"{aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 31: \n" \
"{aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);} \n" \
"} \n" \
"p_Input[index] = aces.x;  \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"}} \n" \
"kernel void k_InvODT( device float* p_Input [[buffer (1)]], constant int& p_Width [[buffer (2)]], constant int& p_Height [[buffer (3)]], \n" \
"constant int& p_InvODT [[buffer (12)]], constant float* p_Lum [[buffer (14)]], constant int& p_DISPLAY [[buffer (15)]],  \n" \
"constant int& p_LIMIT [[buffer (16)]], constant int& p_EOTF [[buffer (17)]], constant int& p_SURROUND [[buffer (18)]],  \n" \
"constant int* p_Switch [[buffer (19)]], uint2 id [[ thread_position_in_grid ]]) { \n" \
"if (id.x < p_Width && id.y < p_Height) { \n" \
"const int index = (id.y * p_Width + id.x) * 4; \n" \
"float3 aces; \n" \
"aces.x = p_Input[index]; \n" \
"aces.y = p_Input[index + 1]; \n" \
"aces.z = p_Input[index + 2]; \n" \
"switch (p_InvODT){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{ \n" \
"float Y_MIN = p_Lum[0] * 0.0001f; \n" \
"float Y_MID = p_Lum[1]; \n" \
"float Y_MAX = p_Lum[2]; \n" \
"Chromaticities DISPLAY_PRI = p_DISPLAY == 0 ? REC2020_PRI : p_DISPLAY == 1 ? P3D60_PRI : p_DISPLAY == 2 ? P3D65_PRI : p_DISPLAY == 3 ? P3DCI_PRI : REC709_PRI; \n" \
"Chromaticities LIMITING_PRI = p_LIMIT == 0 ? REC2020_PRI : p_LIMIT == 1 ? P3D60_PRI : p_LIMIT == 2 ? P3D65_PRI : p_LIMIT == 3 ? P3DCI_PRI : REC709_PRI; \n" \
"int EOTF = p_EOTF; \n" \
"int SURROUND = p_SURROUND; \n" \
"bool STRETCH_BLACK = p_Switch[0] == 1; \n" \
"bool D60_SIM = p_Switch[1] == 1; \n" \
"bool LEGAL_RANGE = p_Switch[2] == 1; \n" \
"aces = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"} \n" \
"break; \n" \
"case 2: \n" \
"{aces = InvODT_Rec709_100nits_dim(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = InvODT_Rec709_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = InvODT_sRGB_100nits_dim(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = InvODT_sRGB_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = InvODT_Rec2020_100nits_dim(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = InvODT_Rec2020_ST2084_1000nits(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = InvODT_P3DCI_48nits(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = InvODT_P3DCI_D60sim_48nits(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = InvODT_P3DCI_D65sim_48nits(aces);} \n" \
"break; \n" \
"case 11: \n" \
"{aces = InvODT_P3D60_48nits(aces);} \n" \
"break; \n" \
"case 12: \n" \
"{aces = InvODT_P3D65_48nits(aces);} \n" \
"break; \n" \
"case 13: \n" \
"{aces = InvODT_P3D65_D60sim_48nits(aces);} \n" \
"break; \n" \
"case 14: \n" \
"{aces = InvODT_DCDM(aces);} \n" \
"break; \n" \
"case 15: \n" \
"{aces = InvODT_DCDM_P3D65limited(aces);} \n" \
"break; \n" \
"case 16: \n" \
"{aces = InvODT_RGBmonitor_100nits_dim(aces);} \n" \
"break; \n" \
"case 17: \n" \
"{aces = InvODT_RGBmonitor_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 18: \n" \
"{aces = InvRRTODT_Rec709_100nits_10nits_BT1886(aces);} \n" \
"break; \n" \
"case 19: \n" \
"{aces = InvRRTODT_Rec709_100nits_10nits_sRGB(aces);} \n" \
"break; \n" \
"case 20: \n" \
"{aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);} \n" \
"break; \n" \
"case 21: \n" \
"{aces = InvRRTODT_P3D65_1000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 22: \n" \
"{aces = InvRRTODT_P3D65_2000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 23: \n" \
"{aces = InvRRTODT_P3D65_4000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 24: \n" \
"{aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);} \n" \
"break; \n" \
"case 25: \n" \
"{aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 26: \n" \
"{aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 27: \n" \
"{aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);} \n" \
"} \n" \
"p_Input[index] = aces.x;  \n" \
"p_Input[index + 1] = aces.y; \n" \
"p_Input[index + 2] = aces.z; \n" \
"}} \n" \
"\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Direction, int p_CSCIN, int p_IDT, int p_LMT, int p_CSCOUT, int p_RRT, 
int p_InvRRT, int p_ODT, int p_InvODT, float p_Exposure, float *p_LMTScale, float *p_Lum, 
int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_Switch) {
const char* Simple 		   		= "k_Simple";
const char* CSCIN 		   		= "k_CSCIN";
const char* IDT 		   		= "k_IDT";
const char* Exposure 		   	= "k_Exposure";
const char* LMT 		   		= "k_LMT";
const char* CSCOUT	 		   	= "k_CSCOUT";
const char* RRT 		   		= "k_RRT";
const char* InvRRT 		   		= "k_InvRRT";
const char* ODT 		   		= "k_ODT";
const char* InvODT 		   		= "k_InvODT";

id<MTLCommandQueue>            	queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
id<MTLDevice>                  	device = queue.device;
id<MTLLibrary>                 	metalLibrary;
id<MTLFunction>                	kernelFunction;
id<MTLComputePipelineState>    	pipelineState;
id<MTLComputePipelineState>     _Simple;
id<MTLComputePipelineState>     _CSCIN;
id<MTLComputePipelineState>     _IDT;
id<MTLComputePipelineState>     _Exposure;
id<MTLComputePipelineState>     _LMT;
id<MTLComputePipelineState>     _CSCOUT;
id<MTLComputePipelineState>     _RRT;
id<MTLComputePipelineState>     _InvRRT;
id<MTLComputePipelineState>     _ODT;
id<MTLComputePipelineState>     _InvODT;

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

if (!(metalLibrary    		= [device newLibraryWithSource:@(kernelSource) options:options error:&err])) {
fprintf(stderr, "Failed to load metal library, %s\n", err.localizedDescription.UTF8String);
return;
}
[options release];

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:Simple]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_Simple   			= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:CSCIN]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_CSCIN	   			= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:IDT]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_IDT   				= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:Exposure]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_Exposure   			= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:LMT]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_LMT   				= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:CSCOUT]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_CSCOUT   			= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:RRT]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_RRT   				= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:InvRRT]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_InvRRT   			= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:ODT]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_ODT   				= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
[metalLibrary release];
[kernelFunction release];
return;
}

if (!(kernelFunction  		= [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:InvODT]])) {
fprintf(stderr, "Failed to retrieve kernel\n");
[metalLibrary release];
return;
}

if (!(_InvODT   			= [device newComputePipelineStateWithFunction:kernelFunction error:&err])) {
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

[computeEncoder setComputePipelineState:_Simple];

int exeWidth = [_Simple threadExecutionWidth];

MTLSize threadGroupCount = MTLSizeMake(exeWidth, 1, 1);
MTLSize threadGroups     = MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

[computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
[computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 1];
[computeEncoder setBytes:&p_Width length:sizeof(int) atIndex:2];
[computeEncoder setBytes:&p_Height length:sizeof(int) atIndex:3];
[computeEncoder setBytes:&p_CSCIN length:sizeof(int) atIndex:4];
[computeEncoder setBytes:&p_IDT length:sizeof(int) atIndex:5];
[computeEncoder setBytes:&p_Exposure length:sizeof(float) atIndex:6];
[computeEncoder setBytes:&p_LMT length:sizeof(int) atIndex:7];
[computeEncoder setBytes:&p_CSCOUT length:sizeof(int) atIndex:8];
[computeEncoder setBytes:&p_RRT length:sizeof(int) atIndex:9];
[computeEncoder setBytes:&p_InvRRT length:sizeof(int) atIndex:10];
[computeEncoder setBytes:&p_ODT length:sizeof(int) atIndex:11];
[computeEncoder setBytes:&p_InvODT length:sizeof(int) atIndex:12];
[computeEncoder setBytes:p_LMTScale length:(sizeof(float) * 24) atIndex:13];
[computeEncoder setBytes:p_Lum length:(sizeof(float) * 3) atIndex:14];
[computeEncoder setBytes:&p_DISPLAY length:sizeof(int) atIndex:15];
[computeEncoder setBytes:&p_LIMIT length:sizeof(int) atIndex:16];
[computeEncoder setBytes:&p_EOTF length:sizeof(int) atIndex:17];
[computeEncoder setBytes:&p_SURROUND length:sizeof(int) atIndex:18];
[computeEncoder setBytes:p_Switch length:(sizeof(int) * 3) atIndex:19];

[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

if (p_Direction == 0) {

if (p_CSCIN != 0) {
[computeEncoder setComputePipelineState:_CSCIN];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_IDT != 0) {
[computeEncoder setComputePipelineState:_IDT];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_Exposure != 0.0f) {
[computeEncoder setComputePipelineState:_Exposure];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_LMT != 0) {
[computeEncoder setComputePipelineState:_LMT];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_CSCOUT != 0) {
[computeEncoder setComputePipelineState:_CSCOUT];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_RRT == 1 && p_ODT < 22) {
[computeEncoder setComputePipelineState:_RRT];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_ODT != 0) {
[computeEncoder setComputePipelineState:_ODT];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

} else {

if (p_InvODT != 0) {
[computeEncoder setComputePipelineState:_InvODT];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}

if (p_InvRRT == 1 && p_InvODT < 18) {
[computeEncoder setComputePipelineState:_InvRRT];
[computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];
}}

[computeEncoder endEncoding];
[commandBuffer commit];
}