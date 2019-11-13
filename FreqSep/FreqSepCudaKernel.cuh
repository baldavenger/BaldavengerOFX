#ifndef _FREQSEPCUDAKERNEL_CU_
#define _FREQSEPCUDAKERNEL_CU_

#define BLOCK_DIM 32

__device__ float from_func_Rec709(float v)
{
if (v < 0.08145f) {
return (v < 0.0f) ? 0.0f : v * (1.0f / 4.5f);
} else {
return powf( (v + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f) );
}
}

__device__ float to_func_Rec709(float v)
{
if (v < 0.0181f) {
return (v < 0.0f) ? 0.0f : v * 4.5f;
} else {
return 1.0993f * powf(v, 0.45f) - (1.0993f - 1.0f);
}
}

__device__ void rgb709_to_xyz(float r, float g, float b, float *x, float *y, float *z)
{
*x = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
*y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
*z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;
}

__device__ void xyz_to_rgb709(float x, float y, float z, float *r, float *g, float *b)
{
*r =  3.2404542f * x + -1.5371385f * y + -0.4985314f * z;
*g = -0.9692660f * x +  1.8760108f * y +  0.0415560f * z;
*b =  0.0556434f * x + -0.2040259f * y +  1.0572252f * z;
}

__device__ void rgb_to_yuv709(float r, float g, float b, float *y, float *u, float *v)
{
*y =  0.2126f  * r + 0.7152f  * g + 0.0722f  * b;
*u = -0.09991f * r - 0.33609f * g + 0.436f   * b;
*v =  0.615f   * r - 0.55861f * g - 0.05639f * b;
}

__device__ void yuv_to_rgb709(float y, float u, float v, float *r, float *g, float *b)
{
*r = y + 1.28033f * v;
*g = y - 0.21482f * u - 0.38059f * v;
*b = y + 2.12798f * u;
}

__device__ float labf(float x)
{
return ( (x) >= 0.008856f ? ( powf(x, 1.0f / 3.0f) ) : (7.787f * x + 16.0f / 116.0f) );
}

__device__ void xyz_to_lab(float x, float y, float z, float *l, float *a, float *b)
{
float fx = labf( x / (0.412453f + 0.357580f + 0.180423f) );
float fy = labf( y / (0.212671f + 0.715160f + 0.072169f) );
float fz = labf( z / (0.019334f + 0.119193f + 0.950227f) );

*l = 116.0f * fy - 16.0f;
*a = 500.0f * (fx - fy);
*b = 200.0f * (fy - fz);
}

__device__ float labfi(float x)
{
return ( x >= 0.206893f ? (x * x * x) : ( (x - 16.0f / 116.0f) / 7.787f ) );
}

__device__ void lab_to_xyz(float l, float a, float b, float *x, float *y, float *z)
{
float cy = (l + 16.0f) / 116.0f;

*y = (0.212671f + 0.715160f + 0.072169f) * labfi(cy);
float cx = a / 500.0f + cy;
*x = (0.412453f + 0.357580f + 0.180423f) * labfi(cx);
float cz = cy - b / 200.0f;
*z = (0.019334f + 0.119193f + 0.950227f) * labfi(cz);
}

__device__ void rgb709_to_lab(float r, float g, float b, float *l, float *a, float *b_)
{
float x, y, z;

rgb709_to_xyz(r, g, b, &x, &y, &z);
xyz_to_lab(x, y, z, l, a, b_);
}

__device__ void lab_to_rgb709(float l, float a, float b, float *r, float *g, float *b_)
{
float x, y, z;

lab_to_xyz(l, a, b, &x, &y, &z);
xyz_to_rgb709(x, y, z, r, g, b_);
}

__global__ void d_rec709_to_yuv(float* p_Input, float* p_Output, int p_Width, int p_Height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;

float r = p_Input[index + 0];
float g = p_Input[index + 1];
float b = p_Input[index + 2];
float y, u, v;
rgb_to_yuv709(r, g, b, &y, &u, &v);
																								  
p_Output[index + 0] = y;
p_Output[index + 1] = u;
p_Output[index + 2] = v;
p_Input[index + 0] = y;
}
}

__global__ void d_yuv_to_rec709(float* p_Input, int p_Width, int p_Height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;

float y = p_Input[index + 0];
float u = p_Input[index + 1];
float v = p_Input[index + 2];
float r, g, b;
yuv_to_rgb709(y, u, v, &r, &g, &b);
																								  
p_Input[index + 0] = r;
p_Input[index + 1] = g;
p_Input[index + 2] = b;
}
}

__global__ void d_rec709_to_lab(float* p_Input, float* p_Output, int p_Width, int p_Height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;

float r = from_func_Rec709(p_Input[index + 0]);
float g = from_func_Rec709(p_Input[index + 1]);
float bb = from_func_Rec709(p_Input[index + 2]);

float l, a, b;

rgb709_to_lab(r, g, bb, &l, &a, &b);
																								  
p_Output[index + 0] = l / 100.0f;
p_Output[index + 1] = a / 200.0f + 0.5f;
p_Output[index + 2] = b / 200.0f + 0.5f;
p_Input[index + 0] = l / 100.0f;
}
}

__global__ void d_lab_to_rec709(float* p_Input, int p_Width, int p_Height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;

float l = p_Input[index + 0] * 100.0f;
float a = (p_Input[index + 1] - 0.5f) * 200.0f;
float b = (p_Input[index + 2] - 0.5f) * 200.0f;
float r, g, bb;
lab_to_rgb709(l, a, b, &r, &g, &bb);

float R = to_func_Rec709(r);
float G = to_func_Rec709(g);
float BB = to_func_Rec709(bb);
																								  
p_Input[index + 0] = R;
p_Input[index + 1] = G;
p_Input[index + 2] = BB;
}
}

__global__ void DisplayThreshold(int p_Width, int p_Height, float* p_Input)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
float thresh = p_Input[index + 0];
p_Input[index + 1] = thresh;
p_Input[index + 2] = thresh;
}
}

int iDivUp(int a, int b)
{
return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void d_transpose(float *idata, float *odata, int width, int height)
{
__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

// read the matrix tile into shared memory
unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

if ((xIndex < width) && (yIndex < height))
{
unsigned int index_in = (yIndex * width + xIndex);
block[threadIdx.y][threadIdx.x] = idata[index_in];
}

__syncthreads();

// write the transposed matrix tile to global memory
xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

if ((xIndex < height) && (yIndex < width))
{
unsigned int index_out = (yIndex * height + xIndex) * 4;
odata[index_out] = block[threadIdx.x][threadIdx.y];
}
}

__global__ void d_recursiveGaussian(float *id, float *od, int w, int h, float blur)
{
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

id += x * 4;    // advance pointers to correct column
od += x;

// forward pass
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
od += w;    // move to next row
xp = xc;
yb = yp;
yp = yc;
}

// reset pointers to point to last element in column
id -= w * 4;
od -= w;

// reverse pass
// ensures response is symmetrical
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
od -= w;  // move to previous row
}
}

__global__ void FrequencySharpen(int p_Width, int p_Height, float* p_Input, float* p_Output, float sharpen, int p_Display)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

float offset = p_Display == 1 ? 0.5f : 0.0f;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
																								  
p_Input[index] = (p_Input[index] - p_Output[index]) * sharpen + offset;
if (p_Display == 1) {
p_Output[index] = p_Input[index];
}
} 
}

__global__ void LowFreqCont(int p_Width, int p_Height, float* p_Input, float contrast, float pivot, int curve, int p_Display, int p_Gang)
{
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
if(curve == 1)
p_Input[index] = p_Input[index] <= pivot ? powf(p_Input[index] / pivot, contrast) * pivot : (1.0f - powf(1.0f - (p_Input[index] - pivot) / (1.0f - pivot), contrast)) * (1.0f - pivot) + pivot;
else
p_Input[index] = (p_Input[index] - pivot) * contrast + pivot;
if(p_Display == 3)
p_Input[index] = graph == 0.0f ? p_Input[index] : graph;
if(p_Gang == 1 && (p_Display == 2 || p_Display == 3))
p_Input[index + 2] = p_Input[index + 1] = p_Input[index];
} 
}

__global__ void FrequencyAdd(int p_Width, int p_Height, float* p_Input, float* p_Output)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
																								  
p_Output[index] = p_Input[index] + p_Output[index];
}
}

__global__ void SimpleKernel(int p_Width, int p_Height, float* p_Input, float* p_Output)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = (y * p_Width + x) * 4;
																								  
p_Output[index] = p_Input[index];
}
}

#endif