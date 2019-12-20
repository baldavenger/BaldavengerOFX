__device__ float Luma( float R, float G, float B, int L) {
float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f;
float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f;
float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f;
float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f;
float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f;
float lumaAvg = (R + G + B) / 3.0f;
float lumaMax = fmax(fmax(R, G), B);
float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : 
L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax;
return Lu;
}

__device__ void RGB_to_HSV( float r, float g, float b, float *h, float *s, float *v ) {
float min = fmin(fmin(r, g), b);
float max = fmax(fmax(r, g), b);
*v = max;
float delta = max - min;
if (max != 0.) {
*s = delta / max;
} else {
*s = 0.f;
*h = 0.f;
return;
}
if (delta == 0.) {
*h = 0.f;
} else if (r == max) {
*h = (g - b) / delta;
} else if (g == max) {
*h = 2 + (b - r) / delta;
} else {
*h = 4 + (r - g) / delta;
}
*h *= 1.0f / 6.;
if (*h < 0) {
*h += 1.0f;
}}

__device__ void HSV_to_RGB( float H, float S, float V, float *r, float *g, float *b) {
if (S == 0.0f) {
*r = *g = *b = V;
return;
}
H *= 6.0f;
int i = floor(H);
float f = H - i;
i = (i >= 0) ? (i % 6) : (i % 6) + 6;
float p = V * (1.0f - S);
float q = V * (1.0f - S * f);
float t = V * (1.0f - S * (1.0f - f));
*r = i == 0 ? V : i == 1 ? q : i == 2 ? p : i == 3 ? p : i == 4 ? t : V;
*g = i == 0 ? t : i == 1 ? V : i == 2 ? V : i == 3 ? q : i == 4 ? p : p;
*b = i == 0 ? p : i == 1 ? p : i == 2 ? t : i == 3 ? V : i == 4 ? V : q;
}

__device__ void Temp( float *R, float *G, float *B, float Temp) {
float r, g, b;
if (Temp <= 66.0f){
r = 255.0f;
} else {
r = Temp - 60.0f;
r = 329.698727446 * powf(r, -0.1332047592);
if(r < 0.0f){r = 0.0f;}
if(r > 255.0f){r = 255.0f;}
}
if (Temp <= 66.0f){
g = Temp;
g = 99.4708025861 * log(g) - 161.1195681661;
if(g < 0.0f){g = 0.0f;}
if(g > 255.0f){g = 255.0f;}
} else {
g = Temp - 60.0f;
g = 288.1221695283 * powf(g, -0.0755148492);
if(g < 0.0f){g = 0.0f;}
if(g > 255.0f){g = 255.0f;}
}
if(Temp >= 66.0f){
b = 255.0f;
} else {
if(Temp <= 19.0f){
b = 0.0f;
} else {
b = Temp - 10.0f;
b = 138.5177312231 * log(b) - 305.0447927307;
if(b < 0.0f){b = 0.0f;}
if(b > 255.0f){b = 255.0f;}
}
}
*R = r / 255.0f;
*G = g / 255.0f;
*B = b / 255.0f;
}
__device__ void VideoGradeKernelA(float R, float G, float B, float *RR, float *GG, float *BB, int p_LumaMath, int p_Gang, 
int p_GammaBias, float p_Exposure, float p_Temp, float p_Tint, float p_Hue, float p_Sat, float p_GainR, float p_GainG, float p_GainB, 
float p_GainAnchor, float p_LiftR, float p_LiftG, float p_LiftB, float p_LiftAnchor, float p_OffsetR, float p_OffsetG, float p_OffsetB, 
float p_GammaR, float p_GammaG, float p_GammaB, float p_GammaStart, float p_GammaEnd) {
if (p_Gang == 1) {
p_GainB = p_GainG = p_GainR;
p_LiftB = p_LiftG = p_LiftR;
p_OffsetB = p_OffsetG = p_OffsetR;
p_GammaB = p_GammaG = p_GammaR;
}
float Temp1 = (p_Temp / 100.0f) + 1.0f;
if (p_Exposure != 0.0f) {
R = R * powf(2.0f, p_Exposure);
G = G * powf(2.0f, p_Exposure);
B = B * powf(2.0f, p_Exposure);
}
if (Temp1 != 66.0f) {
float r, g, b, R1, G1, B1, templuma1, templuma2;
templuma1 = Luma(R, G, B, p_LumaMath);
Temp(&r, &g, &b, Temp1);
R1 = R * r;
G1 = G * g;
B1 = B * b;
templuma2 = Luma(R1, G1, B1, p_LumaMath);
R = R1 / (templuma2 / templuma1);
G = G1 / (templuma2 / templuma1);
B = B1 / (templuma2 / templuma1);
}
if (p_Tint != 0.0f) {
float tintluma1 = Luma(R, G, B, p_LumaMath);
float R1 = R * (1 + (p_Tint / 2));
float B1 = B * (1 + (p_Tint / 2));
float tintluma2 = Luma(R1, G, B1, p_LumaMath);
float tintluma3 = tintluma2 / tintluma1;
R = R1 / tintluma3;
G = G / tintluma3;
B = B1 / tintluma3;
}
if (p_Hue != 0.0f || p_Sat != 0.0f) {
p_Hue /= 360.0f;
float h, s, v;
RGB_to_HSV(R, G, B, &h, &s, &v);
float h2 = h + p_Hue;
float H2 = h2 < 0.0f ? h2 + 1.0f : h2 >= 1.0f ? h2 - 1.0f : h2;
float S = s * (1.0f + p_Sat);
HSV_to_RGB(H2, S, v, &R, &G, &B);
}
if (p_GainR != 1.0f) {
R = R >= p_GainAnchor ? (R - p_GainAnchor) * p_GainR  + p_GainAnchor: R;
}
if (p_GainG != 1.0f) {
G = G >= p_GainAnchor ? (G - p_GainAnchor) * p_GainG  + p_GainAnchor: G;
}
if (p_GainB != 1.0f) {
B = B >= p_GainAnchor ? (B - p_GainAnchor) * p_GainB  + p_GainAnchor: B;
}
if (p_LiftR != 0.0f) {
R = R <= p_LiftAnchor ? ((R / p_LiftAnchor) + p_LiftR * (1.0f - (R / p_LiftAnchor))) * p_LiftAnchor : R;
}
if (p_LiftG != 0.0f) {
G = G <= p_LiftAnchor ? ((G / p_LiftAnchor) + p_LiftG * (1.0f - (G / p_LiftAnchor))) * p_LiftAnchor : G;
}
if (p_LiftB != 0.0f) {
B = B <= p_LiftAnchor ? ((B / p_LiftAnchor) + p_LiftB * (1.0f - (B / p_LiftAnchor))) * p_LiftAnchor : B;
}
if (p_OffsetR != 0.0f) {
R += p_OffsetR;
}
if (p_OffsetG != 0.0f) {
G += p_OffsetG;
}
if (p_OffsetB != 0.0f) {
B += p_OffsetB;
}
if (p_GammaR != 0.0f) {
float Prl = R >= p_GammaStart && R <= p_GammaEnd ? powf((R - p_GammaStart) / (p_GammaEnd - p_GammaStart), 1.0f / p_GammaR) * (p_GammaEnd - p_GammaStart) + p_GammaStart : R;
float Pru = R >= p_GammaStart && R <= p_GammaEnd ? (1.0f - powf(1.0f - (R - p_GammaStart) / (p_GammaEnd - p_GammaStart), p_GammaR)) * (p_GammaEnd - p_GammaStart) + p_GammaStart : R;
R = p_GammaBias == 1 ? Pru : Prl;
}
if (p_GammaG != 0.0f) {
float Pgl = G >= p_GammaStart && G <= p_GammaEnd ? powf((G - p_GammaStart) / (p_GammaEnd - p_GammaStart), 1.0f / p_GammaG) * (p_GammaEnd - p_GammaStart) + p_GammaStart : G;
float Pgu = G >= p_GammaStart && G <= p_GammaEnd ? (1.0f - powf(1.0f - (G - p_GammaStart) / (p_GammaEnd - p_GammaStart), p_GammaG)) * (p_GammaEnd - p_GammaStart) + p_GammaStart : G;
G = p_GammaBias == 1 ? Pgu : Pgl;
}
if (p_GammaB != 0.0f) {
float Pbl = B >= p_GammaStart && B <= p_GammaEnd ? powf((B - p_GammaStart) / (p_GammaEnd - p_GammaStart), 1.0f / p_GammaB) * (p_GammaEnd - p_GammaStart) + p_GammaStart : B;
float Pbu = B >= p_GammaStart && B <= p_GammaEnd ? (1.0f - powf(1.0f - (B - p_GammaStart) / (p_GammaEnd - p_GammaStart), p_GammaB)) * (p_GammaEnd - p_GammaStart) + p_GammaStart : B;
B = p_GammaBias == 1 ? Pbu : Pbl;
}
*RR = R;
*GG = G;
*BB = B;
}

__global__ void VideoGradeKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_LumaMath, int p_Gang, 
int p_GammaBias, int p_Display, float p_Exposure, float p_Temp, float p_Tint, float p_Hue, float p_Sat, 
float p_GainR, float p_GainG, float p_GainB, float p_GainAnchor, float p_LiftR, float p_LiftG, float p_LiftB, float p_LiftAnchor, 
float p_OffsetR, float p_OffsetG, float p_OffsetB, float p_GammaR, float p_GammaG, float p_GammaB, float p_GammaStart, float p_GammaEnd) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float r1, g1, b1, R, G, B, r2, g2, b2, R1, G1, B1;
R = p_Input[index];
G = p_Input[index + 1];
B = p_Input[index + 2];
float width = p_Width;
float height = p_Height;
const float THICK = 5.0f * (width / 1920.0f);
R1 = x / (width - 1);
B1 = G1 = R1;
if (p_Display == 0) {
VideoGradeKernelA(R, G, B, &r1, &g1, &b1, p_LumaMath, p_Gang, p_GammaBias, p_Exposure, p_Temp, p_Tint, p_Hue, p_Sat, 
p_GainR, p_GainG, p_GainB, p_GainAnchor, p_LiftR, p_LiftG, p_LiftB, p_LiftAnchor, p_OffsetR, p_OffsetG, p_OffsetB, 
p_GammaR, p_GammaG, p_GammaB, p_GammaStart, p_GammaEnd);
}
if (p_Display == 1) {
VideoGradeKernelA(R1, G1, B1, &r2, &g2, &b2, p_LumaMath, p_Gang, p_GammaBias, p_Exposure, p_Temp, p_Tint, p_Hue, p_Sat, 
p_GainR, p_GainG, p_GainB, p_GainAnchor, p_LiftR, p_LiftG, p_LiftB, p_LiftAnchor, p_OffsetR, p_OffsetG, p_OffsetB, 
p_GammaR, p_GammaG, p_GammaB, p_GammaStart, p_GammaEnd);
r1 = r2 >= (y - THICK) / height && r2 <= (y + THICK) / height ? 1.0f : 0.0f;
g1 = g2 >= (y - THICK) / height && g2 <= (y + THICK) / height ? 1.0f : 0.0f;
b1 = b2 >= (y - THICK) / height && b2 <= (y + THICK) / height ? 1.0f : 0.0f;
}
if (p_Display == 2) {
VideoGradeKernelA(R, G, B, &r1, &g1, &b1, p_LumaMath, p_Gang, p_GammaBias, p_Exposure, p_Temp, p_Tint, p_Hue, p_Sat, 
p_GainR, p_GainG, p_GainB, p_GainAnchor, p_LiftR, p_LiftG, p_LiftB, p_LiftAnchor, p_OffsetR, p_OffsetG, p_OffsetB, 
p_GammaR, p_GammaG, p_GammaB, p_GammaStart, p_GammaEnd);
VideoGradeKernelA(R1, G1, B1, &r2, &g2, &b2, p_LumaMath, p_Gang, p_GammaBias, p_Exposure, p_Temp, p_Tint, p_Hue, p_Sat, 
p_GainR, p_GainG, p_GainB, p_GainAnchor, p_LiftR, p_LiftG, p_LiftB, p_LiftAnchor, p_OffsetR, p_OffsetG, p_OffsetB, 
p_GammaR, p_GammaG, p_GammaB, p_GammaStart, p_GammaEnd);
r2 = r2 >= (y - THICK) / height && r2 <= (y + THICK) / height ? 1.0f : 0.0f;
g2 = g2 >= (y - THICK) / height && g2 <= (y + THICK) / height ? 1.0f : 0.0f;
b2 = b2 >= (y - THICK) / height && b2 <= (y + THICK) / height ? 1.0f : 0.0f;
r1 = r2 == 0.0f ? r1 : r2;
g1 = g2 == 0.0f ? g1 : g2;
b1 = b2 == 0.0f ? b1 : b2;
}
p_Output[index] = r1;
p_Output[index + 1] = g1;
p_Output[index + 2] = b1;
p_Output[index + 3] = p_Input[index + 3];
}}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, 
int p_Height, int p_LumaMath, int* p_Switch, int p_Display, float* p_Scale)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

VideoGradeKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_LumaMath, p_Switch[0], p_Switch[1], 
p_Display, p_Scale[0], p_Scale[1], p_Scale[2], p_Scale[3], p_Scale[4], p_Scale[5], p_Scale[6], p_Scale[7], p_Scale[8], 
p_Scale[9], p_Scale[10], p_Scale[11], p_Scale[12], p_Scale[13], p_Scale[14], p_Scale[15], p_Scale[16], p_Scale[17], 
p_Scale[18], p_Scale[19], p_Scale[20]);
}