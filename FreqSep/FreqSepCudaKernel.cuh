#ifndef _FREQSEPCUDAKERNEL_CU_
#define _FREQSEPCUDAKERNEL_CU_

#define BLOCK_DIM 32

int iDivUp(int a, int b) {
return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ inline float normalizedLogCToRelativeExposure( float x) {
if (x > 0.149659f)
return (powf(10.0f, (x - 0.385537f) / 0.247189f) - 0.052272f) / 5.555556f;
else
return (x - 0.092809f) / 5.367650f;
}

__device__ inline float relativeExposureToLogC( float x) {
if (x > 0.010591f)
return 0.247190f * log10f(5.555556f * x + 0.052272f) + 0.385537f;
else
return 5.367655f * x + 0.092809f;
}

__device__ inline float3 ArrilogC_to_XYZ( float3 Alexa) {
float r_lin = normalizedLogCToRelativeExposure(Alexa.x);
float g_lin = normalizedLogCToRelativeExposure(Alexa.y);
float b_lin = normalizedLogCToRelativeExposure(Alexa.z);
float3 XYZ;
XYZ.x = r_lin * 0.638008f + g_lin * 0.214704f + b_lin * 0.097744f;
XYZ.y = r_lin * 0.291954f + g_lin * 0.823841f - b_lin * 0.115795f;
XYZ.z = r_lin * 0.002798f - g_lin * 0.067034f + b_lin * 1.153294f;
return XYZ;
}

__device__ inline float3 XYZ_to_ArrilogC( float3 XYZ) {
float3 Alexa;
Alexa.x = XYZ.x * 1.789066f - XYZ.y * 0.482534f - XYZ.z * 0.200076f;
Alexa.y = XYZ.x * -0.639849f + XYZ.y * 1.3964f + XYZ.z * 0.194432f;
Alexa.z = XYZ.x * -0.041532f + XYZ.y * 0.082335f + XYZ.z * 0.878868f;
Alexa.x = relativeExposureToLogC(Alexa.x);
Alexa.y = relativeExposureToLogC(Alexa.y);
Alexa.z = relativeExposureToLogC(Alexa.z);
return Alexa;
}

__device__ inline float from_func_Rec709(float v) {
if (v < 0.08145f)
return (v < 0.0f) ? 0.0f : v * (1.0f / 4.5f);
else
return powf( (v + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f) );
}

__device__ inline float to_func_Rec709(float v) {
if (v < 0.0181f)
return (v < 0.0f) ? 0.0f : v * 4.5f;
else
return 1.0993f * powf(v, 0.45f) - (1.0993f - 1.0f);
}

__device__ inline float3 Rec709_to_XYZ( float3 rgb) {
rgb.x = from_func_Rec709(rgb.x);
rgb.y = from_func_Rec709(rgb.y);
rgb.z = from_func_Rec709(rgb.z);
float3 xyz;
xyz.x = 0.4124564f * rgb.x + 0.3575761f * rgb.y + 0.1804375f * rgb.z;
xyz.y = 0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f * rgb.z;
xyz.z = 0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z;
return xyz;
}

__device__ inline float3 XYZ_to_Rec709(float3 xyz) {
float3 rgb;
rgb.x =  3.2404542f * xyz.x + -1.5371385f * xyz.y + -0.4985314f * xyz.z;
rgb.y = -0.9692660f * xyz.x +  1.8760108f * xyz.y +  0.0415560f * xyz.z;
rgb.z =  0.0556434f * xyz.x + -0.2040259f * xyz.y +  1.0572252f * xyz.z;
rgb.x = to_func_Rec709(rgb.x);
rgb.y = to_func_Rec709(rgb.y);
rgb.z = to_func_Rec709(rgb.z);
return rgb;
}

__device__ float3 XYZ_to_LAB( float3 XYZ) {
float fx, fy, fz;
float Xn = 0.950489f;
float Zn = 1.08884f;

if(XYZ.x / Xn > 0.008856f)
fx = powf(XYZ.x / Xn, 1.0f / 3.0f);
else
fx = 7.787f * (XYZ.x / Xn) + 0.137931f;

if(XYZ.y > 0.008856f)
fy = powf(XYZ.y, 1.0f / 3.0f);
else
fy = 7.787f * XYZ.y + 0.137931f;

if(XYZ.z / Zn > 0.008856f)
fz = powf(XYZ.z / Zn, 1.0f / 3.0f);
else
fz = 7.787f * (XYZ.z / Zn) + 0.137931f;

float3 Lab;
Lab.x = 1.16f * fy - 0.16f;
Lab.y = 2.5f * (fx - fy) + 0.5f;
Lab.z = 1.0f * (fy - fz) + 0.5f;

return Lab;
}

__device__ inline float3 LAB_to_XYZ( float3 LAB) {
float3 XYZ;
float Xn = 0.950489f;
float Zn = 1.08884f;

float cy = (LAB.x + 0.16f) / 1.16f;
if(cy >= 0.206893f)
XYZ.y = cy * cy * cy;
else
XYZ.y = (cy - 0.137931f) / 7.787f;

float cx = (LAB.y - 0.5f) / 2.5f + cy;
if(cx >= 0.206893f)
XYZ.x = Xn * cx * cx * cx;
else
XYZ.x = Xn * (cx - 0.137931f) / 7.787f;

float cz = cy - (LAB.z - 0.5f);
if(cz >= 0.206893f)
XYZ.z = Zn * cz * cz * cz;
else
XYZ.z = Zn * (cz - 0.137931f) / 7.787f;

return XYZ;
}

__device__ inline float3 Rec709_to_LAB( float3 rgb) {
float3 lab;
lab = Rec709_to_XYZ(rgb);
lab = XYZ_to_LAB(lab);
return lab;
}

__device__ inline float3 LAB_to_Rec709( float3 lab) {
float3 rgb;
rgb = LAB_to_XYZ(lab);
rgb = XYZ_to_Rec709(rgb);
return rgb;
}

__device__ inline float3 ACEScct_to_XYZ( float3 in) {
const float Y_BRK = 0.155251141552511f;
const float A = 10.5402377416545f;
const float B = 0.0729055341958355f;
float3 out;
in.x = in.x > Y_BRK ? powf( 2.0f, in.x * 17.52f - 9.72f) : (in.x - B) / A;
in.y = in.y > Y_BRK ? powf( 2.0f, in.y * 17.52f - 9.72f) : (in.y - B) / A;
in.z = in.z > Y_BRK ? powf( 2.0f, in.z * 17.52f - 9.72f) : (in.z - B) / A;
out.x = 0.6624541811f * in.x + 0.1340042065f * in.y + 0.156187687f * in.z;
out.y = 0.2722287168f * in.x + 0.6740817658f * in.y + 0.0536895174f * in.z;
out.z = -0.0055746495f * in.x + 0.0040607335f * in.y + 1.0103391003f * in.z;
return out;
}

__device__ inline float3 XYZ_to_ACEScct( float3 in) {
const float X_BRK = 0.0078125f;
const float A = 10.5402377416545f;
const float B = 0.0729055341958355f;
float3 out;
out.x = 1.6410233797f * in.x + -0.3248032942f * in.y + -0.2364246952f * in.z;
out.y = -0.6636628587f * in.x + 1.6153315917f * in.y + 0.0167563477f * in.z;
out.z = 0.0117218943f * in.x + -0.008284442f * in.y + 0.9883948585f * in.z;
out.x = out.x <= X_BRK ? A * out.x + B : (log2f(out.x) + 9.72f) / 17.52f;
out.y = out.y <= X_BRK ? A * out.y + B : (log2f(out.y) + 9.72f) / 17.52f;
out.z = out.z <= X_BRK ? A * out.z + B : (log2f(out.z) + 9.72f) / 17.52f;
return out;
}

__global__ void d_arri_to_lab(float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 Alexa = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float3 lab = ArrilogC_to_XYZ(Alexa);
lab = XYZ_to_LAB(lab);
p_Output[index] = lab.x;
p_Output[index + 1] = lab.y;
p_Output[index + 2] = lab.z;
p_Input[index] = lab.x;
}
}

__global__ void d_lab_to_arri(float* p_Input, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float3 Alexa;
Alexa = LAB_to_XYZ(lab);
Alexa = XYZ_to_ArrilogC(Alexa);
p_Input[index] = Alexa.x;
p_Input[index + 1] = Alexa.y;
p_Input[index + 2] = Alexa.z;
}
}

__global__ void d_acescct_to_lab(float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 aces = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float3 lab = ACEScct_to_XYZ(aces);
lab = XYZ_to_LAB(lab);
p_Output[index] = lab.x;
p_Output[index + 1] = lab.y;
p_Output[index + 2] = lab.z;
p_Input[index] = lab.x;
}
}

__global__ void d_lab_to_acescct(float* p_Input, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float3 aces;
aces = LAB_to_XYZ(lab);
aces = XYZ_to_ACEScct(aces);
p_Input[index] = aces.x;
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}
}

__global__ void d_rec709_to_lab(float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 rgb = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float3 lab = Rec709_to_LAB(rgb);
p_Output[index] = lab.x;
p_Output[index + 1] = lab.y;
p_Output[index + 2] = lab.z;
p_Input[index] = lab.x;
}
}

__global__ void d_lab_to_rec709(float* p_Input, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float3 lab = make_float3(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float3 rgb = LAB_to_Rec709(lab);
p_Input[index] = rgb.x;
p_Input[index + 1] = rgb.y;
p_Input[index + 2] = rgb.z;
}
}

__global__ void d_transpose(float *idata, float *odata, int width, int height) {
__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];
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
}
}

__global__ void d_recursiveGaussian(float *id, float *od, int w, int h, float blur) {
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
float nsigma = blur < 0.1f ? 0.1f : blur;
float alpha = 1.695f / nsigma;
float ema = expf(-alpha);
float ema2 = expf(-2.0f * alpha),
b1 = -2.0f * ema,
b2 = ema2;
float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, coefp = 0.0f, coefn = 0.0f;
float k = (1.0f - ema) * (1.0f - ema) / (1.0f + 2.0f * alpha * ema - ema2);
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
yb = coefp*xp;
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

__global__ void FrequencySharpen(float* p_Input, float* p_Output, int p_Width, int p_Height, float sharpen, int p_Display) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
float offset = p_Display == 1 ? 0.5f : 0.0f;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
p_Input[index] = (p_Input[index] - p_Output[index]) * sharpen + offset;
if (p_Display == 1)
p_Output[index] = p_Input[index];
} 
}

__global__ void FrequencySharpenLuma(float* p_Input, float* p_Output, int p_Width, int p_Height, float sharpen, int p_Display) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
float offset = p_Display == 1 ? 0.5f : 0.0f;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;									
p_Input[index] = (p_Input[index] - p_Output[index]) * sharpen + offset;
if (p_Display == 1)
p_Output[index] = p_Output[index + 1] = p_Output[index + 2] = p_Input[index];
}
}

__global__ void LowFreqCont(float* p_Input, int p_Width, int p_Height, float contrast, float pivot, int curve, int p_Display) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float graph = 0.0f;
if(p_Display == 3){
float width = p_Width;
float height = p_Height;
float X = x;
float Y = y;
float ramp = X / (width - 1.0f);
if(curve == 1)
ramp = ramp <= pivot ? powf(ramp / pivot, contrast) * pivot : (1.0f - powf(1.0f - (ramp - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot;
else
ramp = (ramp - pivot) * contrast + pivot;
graph = ramp >= (Y - 5.0f) / height && ramp <= (Y + 5.0f) / height ? 1.0f : 0.0f;
}
if(curve == 1){
if(p_Input[index] > 0.0f && p_Input[index] < 1.0f)
p_Input[index] = p_Input[index] <= pivot ? powf(p_Input[index] / pivot, contrast) * pivot : (1.0f - powf(1.0f - (p_Input[index] - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot;
} else {
p_Input[index] = (p_Input[index] - pivot) * contrast + pivot;
}
if(p_Display == 3)
p_Input[index] = graph == 0.0f ? p_Input[index] : graph;
} 
}

__global__ void LowFreqContLuma(float* p_Input, int p_Width, int p_Height, float contrast, float pivot, int curve, int p_Display) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float graph = 0.0f;
if(curve == 1){
if(p_Input[index] > 0.0f && p_Input[index] < 1.0f)
p_Input[index] = p_Input[index] <= pivot ? powf(p_Input[index] / pivot, contrast) * pivot : (1.0f - powf(1.0f - (p_Input[index] - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot;
} else {
p_Input[index] = (p_Input[index] - pivot) * contrast + pivot;
}
if(p_Display == 2)
p_Input[index + 2] = p_Input[index + 1] = p_Input[index];
if(p_Display == 3){
float width = p_Width;
float height = p_Height;
float X = x;
float Y = y;
float ramp = X / (width - 1.0f);
if(curve == 1)
ramp = ramp <= pivot ? powf(ramp / pivot, contrast) * pivot : (1.0f - powf(1.0f - (ramp - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot;
else
ramp = (ramp - pivot) * contrast + pivot;
graph = ramp >= (Y - 5.0f) / height && ramp <= (Y + 5.0f) / height ? 1.0f : 0.0f;
p_Input[index] = graph == 0.0f ? p_Input[index] : graph;
p_Input[index + 2] = p_Input[index + 1] = p_Input[index];
}
} 
}

__global__ void FrequencyAdd(float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;											  
p_Output[index] = p_Input[index] + p_Output[index];
}
}

__global__ void SimpleKernel(float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
p_Output[index] = p_Input[index];
}
}

#endif