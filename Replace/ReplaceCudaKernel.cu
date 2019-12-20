#include "ReplaceCudaKernel.cuh"

cudaError_t cudaError;

__global__ void ReplaceKernelA(const float* p_Input, float* p_Output, int p_Width, int p_Height, float hueRangeA, float hueRangeB, 
float hueRangeWithRollOffA, float hueRangeWithRollOffB, float satRangeA, float satRangeB, float satRolloff, float valRangeA, 
float valRangeB, float valRolloff, int OutputAlpha, int DisplayAlpha, float p_Black, float p_White) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float hcoeff, scoeff, vcoeff;
float r, g, b, h, s, v;
r = p_Input[index];
g = p_Input[index + 1];
b = p_Input[index + 2];
float min = fmin(fmin(r, g), b);
float max = fmax(fmax(r, g), b);
v = max;
float delta = max - min;
if (max != 0.0f) {
s = delta / max;
} else {
s = 0.0f;
h = 0.0f;
}
if (delta == 0.0f) {
h = 0.0f;
} else if (r == max) {
h = (g - b) / delta;
} else if (g == max) {
h = 2 + (b - r) / delta;
} else {
h = 4 + (r - g) / delta;
}
h *= 1 / 6.0f;
if (h < 0.0f) {
h += 1.0f;
}
h *= 360.0f;
float h0 = hueRangeA;
float h1 = hueRangeB;
float h0mrolloff = hueRangeWithRollOffA;
float h1prolloff = hueRangeWithRollOffB;
if ( ( h1 < h0 && (h <= h1 || h0 <= h) ) || (h0 <= h && h <= h1) ) {
hcoeff = 1.0f;
} else {
float c0 = 0.0f;
float c1 = 0.0f;
if ( ( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0) ) {
c0 = h0 == (h0mrolloff + 360.0f) || h0 == h0mrolloff ? 1.0f : !(( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0)) ? 0.0f : 
((h < h0mrolloff ? h + 360.0f : h) - h0mrolloff) / ((h0 < h0mrolloff ? h0 + 360.0f : h0) - h0mrolloff);		
}
if ( ( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff) ) {
c1 = !(( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff)) ? 0.0f : h1prolloff == h1 ? 1.0f :
((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - (h < h1 ? h + 360.0f : h)) / ((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - h1);	
}
hcoeff = fmax(c0, c1);
}
float s0 = satRangeA;
float s1 = satRangeB;
float s0mrolloff = s0 - satRolloff;
float s1prolloff = s1 + satRolloff;
if ( s0 <= s && s <= s1 ) {
scoeff = 1.0f;
} else if ( s0mrolloff <= s && s <= s0 ) {
scoeff = (s - s0mrolloff) / satRolloff;
} else if ( s1 <= s && s <= s1prolloff ) {
scoeff = (s1prolloff - s) / satRolloff;
} else {
scoeff = 0.0f;
}
float v0 = valRangeA;
float v1 = valRangeB;
float v0mrolloff = v0 - valRolloff;
float v1prolloff = v1 + valRolloff;
if ( (v0 <= v) && (v <= v1) ) {
vcoeff = 1.0f;
} else if ( v0mrolloff <= v && v <= v0 ) {
vcoeff = (v - v0mrolloff) / valRolloff;
} else if ( v1 <= v && v <= v1prolloff ) {
vcoeff = (v1prolloff - v) / valRolloff;
} else {
vcoeff = 0.0f;
}
float coeff = fmin(fmin(hcoeff, scoeff), vcoeff);
float A = OutputAlpha == 0 ? 1.0f : OutputAlpha == 1 ? hcoeff : OutputAlpha == 2 ? scoeff :
OutputAlpha == 3 ? vcoeff : OutputAlpha == 4 ? fmin(hcoeff, scoeff) : OutputAlpha == 5 ? 
fmin(hcoeff, vcoeff) : OutputAlpha == 6 ? fmin(scoeff, vcoeff) : fmin(fmin(hcoeff, scoeff), vcoeff);
if (DisplayAlpha == 0)
A = coeff;
if (p_Black > 0.0f)
A = fmax(A - (p_Black * 4.0f) * (1.0f - A), 0.0f);
if (p_White > 0.0f)
A = fmin(A * (1.0f + p_White * 4.0f), 1.0f);
p_Output[index] = h;
p_Output[index + 1] = s;
p_Output[index + 2] = v;
p_Output[index + 3] = A;
}}

__global__ void ReplaceKernelB(const float* p_Input, float* p_Output, int p_Width, int p_Height, float hueRotation, 
float hueRotationGain, float hueMean, float satRangeA, float satRangeB, float satAdjust, float satAdjustGain, 
float valRangeA, float valRangeB, float valAdjust, float valAdjustGain, int DisplayAlpha, float mix) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float h, s, v, R, G, B, coeff;
h = p_Output[index];
s = p_Output[index + 1];
v = p_Output[index + 2];
coeff = p_Output[index + 3];
float s0 = satRangeA;
float s1 = satRangeB;
float v0 = valRangeA;
float v1 = valRangeB;
float H = (h - hueMean + 180.0f) - (int)(floor((h - hueMean + 180.0f) / 360.0f) * 360.0f) - 180.0f;
h += coeff * ( hueRotation + (hueRotationGain - 1.0f) * H );
s += coeff * ( satAdjust + (satAdjustGain - 1.0f) * (s - (s0 + s1) / 2.0f) );
if (s < 0.0f) {
s = 0.0f;
}
v += coeff * ( valAdjust + (valAdjustGain - 1.0f) * (v - (v0 + v1) / 2.0f) );
h *= 1.0f / 360.0f;
if (s == 0.0f)
R = G = B = v;
h *= 6.0f;
int i = floor(h);
float f = h - i;
i = (i >= 0) ? (i % 6) : (i % 6) + 6;
float p = v * ( 1.0f - s );
float q = v * ( 1.0f - s * f );
float t = v * ( 1.0f - s * ( 1.0f - f ));
if (i == 0){
R = v;
G = t;
B = p;}
else if (i == 1){
R = q;
G = v;
B = p;}
else if (i == 2){
R = p;
G = v;
B = t;}
else if (i == 3){
R = p;
G = q;
B = v;}
else if (i == 4){
R = t;
G = p;
B = v;}
else{
R = v;
G = p;
B = q;
}
p_Output[index] = DisplayAlpha == 1 ? coeff : Mix(R, p_Input[index], mix);
p_Output[index + 1] = DisplayAlpha == 1 ? coeff : Mix(G, p_Input[index + 1], mix);
p_Output[index + 2] = DisplayAlpha == 1 ? coeff : Mix(B, p_Input[index + 2], mix);
p_Output[index + 3] = 1.0f;
}}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Hue, 
float* p_Sat, float* p_Val, int OutputAlpha, int DisplayAlpha, float mix, float* p_Blur)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

int nthreads = 128;
float* tempBuffer;
cudaError = cudaMalloc(&tempBuffer, sizeof(float) * p_Width * p_Height);

dim3 grid(iDivUp(p_Width, BLOCK_DIM), iDivUp(p_Height, BLOCK_DIM));
dim3 gridT(iDivUp(p_Height, BLOCK_DIM), iDivUp(p_Width, BLOCK_DIM));
dim3 threadsT(BLOCK_DIM, BLOCK_DIM);

ReplaceKernelA<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Hue[0], p_Hue[1], p_Hue[2], p_Hue[3], 
p_Sat[0], p_Sat[1], p_Sat[4], p_Val[0], p_Val[1], p_Val[4], OutputAlpha, DisplayAlpha, p_Blur[0], p_Blur[1]);

if (p_Blur[2] > 0.0f) {
p_Blur[2] *= 10.0f;
d_recursiveGaussian<<< iDivUp(p_Width, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Width, p_Height, p_Blur[2]);
d_transpose<<< grid, threadsT >>>(tempBuffer, p_Output + 3, p_Width, p_Height);
d_recursiveGaussian<<< iDivUp(p_Height, nthreads), nthreads >>>(p_Output + 3, tempBuffer, p_Height, p_Width, p_Blur[2]);
d_transpose<<< gridT, threadsT >>>(tempBuffer, p_Output + 3, p_Height, p_Width);
d_garbageCore<<<blocks, threads>>>(p_Output + 3, p_Width, p_Height, p_Blur[3], p_Blur[4]);
}

int radE = ceil(p_Blur[5] * 15.0f);
int radD = ceil(p_Blur[6] * 15.0f);
int tile_w = 640;
int tile_h = 1;
dim3 blockE2(tile_w + (2 * radE), tile_h);
dim3 blockD2(tile_w + (2 * radD), tile_h);
dim3 grid2(iDivUp(p_Width, tile_w), iDivUp(p_Height, tile_h));
int tile_w2 = 8;
int tile_h2 = 64;
dim3 blockE3(tile_w2, tile_h2 + (2 * radE));
dim3 blockD3(tile_w2, tile_h2 + (2 * radD));
dim3 grid3(iDivUp(p_Width, tile_w2), iDivUp(p_Height, tile_h2));

if (p_Blur[5] > 0.0f) {
ErosionSharedStep1<<<grid2,blockE2,blockE2.y*blockE2.x*sizeof(float)>>>(p_Output + 3, tempBuffer, radE, p_Width, p_Height, tile_w, tile_h);
cudaError = cudaDeviceSynchronize();
ErosionSharedStep2<<<grid3,blockE3,blockE3.y*blockE3.x*sizeof(float)>>>(tempBuffer, p_Output + 3, radE, p_Width, p_Height, tile_w2, tile_h2);
}

if (p_Blur[6] > 0.0f) {
DilateSharedStep1<<<grid2,blockD2,blockD2.y*blockD2.x*sizeof(float)>>>(p_Output + 3, tempBuffer, radD, p_Width, p_Height, tile_w, tile_h);
cudaError = cudaDeviceSynchronize();
DilateSharedStep2<<<grid3,blockD3,blockD3.y*blockD3.x*sizeof(float)>>>(tempBuffer, p_Output + 3, radD, p_Width, p_Height, tile_w2, tile_h2);
}

ReplaceKernelB<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Hue[4], p_Hue[5], p_Hue[6], 
p_Sat[0], p_Sat[1], p_Sat[2], p_Sat[3], p_Val[0], p_Val[1], p_Val[2], p_Val[3], DisplayAlpha, mix);

cudaError = cudaFree(tempBuffer);
}