__constant__ float eu = 2.718281828459045f;
__constant__ float pie = 3.141592653589793f;

__global__ void Prepare(float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Display) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float ramp = (float)x / (float)(p_Width - 1);
p_Output[index] = p_Display == 1 ? ramp : p_Input[index];
p_Output[index + 1] = p_Display == 1 ? ramp : p_Input[index + 1];
p_Output[index + 2] = p_Display == 1 ? ramp : p_Input[index + 2];
p_Output[index + 3] = 1.0f;
if (p_Display == 2) {
p_Input[index] = ramp;
p_Input[index + 1] = ramp;
p_Input[index + 2] = ramp;
}}}

__global__ void FilmGradeKernelA( float* p_Input, int p_Width, int p_Height, float p_Exp) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if(x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
p_Input[index] = p_Input[index] + p_Exp * 0.01f;
}}

__global__ void FilmGradeKernelB( float* p_Input, int p_Width, int p_Height,
float p_Shad, float p_Mid, float p_High, float p_ShadP, float p_HighP, float p_ContP) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float exp = p_Input[index];
float expr1 = (p_ShadP / 2.0f) - (1.0f - p_HighP) / 4.0f;
float expr2 = (1.0f - (1.0f - p_HighP) / 2.0f) + (p_ShadP / 4.0f);
float expr3 = (exp - expr1) / (expr2 - expr1);
float expr4 =  p_ContP < 0.5f ? 0.5f - (0.5f - p_ContP) / 2.0f : 0.5f + (p_ContP - 0.5f) / 2.0f;
float expr5 = expr3 > expr4 ? (expr3 - expr4) / (2.0f - 2.0f * expr4) + 0.5f : expr3 / (2.0f * expr4);
float expr6 = (((sinf(2.0f * pie * (expr5 -1.0f / 4.0f)) + 1.0f) / 20.0f) * p_Mid * 4.0f) + expr3;
float mid = exp >= expr1 && exp <= expr2 ? expr6 * (expr2 - expr1) + expr1 : exp;
float shadup1 = mid > 0.0f ? 2.0f * (mid / p_ShadP) - logf((mid / p_ShadP) * (eu * p_Shad * 2.0f) + 1.0f) / logf(eu * p_Shad * 2.0f + 1.0f) : mid;
float shadup = mid < p_ShadP && p_Shad > 0.0f ? (shadup1 + p_Shad * (1.0f - shadup1)) * p_ShadP : mid;
float shaddown1 = shadup / p_ShadP + p_Shad * 2.0f * (1.0f - shadup / p_ShadP);
float shaddown = shadup < p_ShadP && p_Shad < 0.0f ? (shaddown1 >= 0.0f ? logf(shaddown1 * (eu * p_Shad * -2.0f) + 1.0f) / logf(eu * p_Shad * -2.0f + 1.0f) : shaddown1) * p_ShadP : shadup;
float highup1 = ((shaddown - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_High * 2.0f));
float highup = shaddown > p_HighP && p_HighP < 1.0f && p_High > 0.0f ? (2.0f * highup1 - logf(highup1 * eu * p_High + 1.0f) / logf(eu * p_High + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddown;
float highdown1 = (highup - p_HighP) / (1.0f - p_HighP);
float highdown = highup > p_HighP && p_HighP < 1.0f && p_High < 0.0f ? logf(highdown1 * (eu * p_High * -2.0f) + 1.0f) / logf(eu * p_High * -2.0f + 1.0f) * (1.0f + p_High) * (1.0f - p_HighP) + p_HighP : highup;
p_Input[index] = highdown;
}}

__global__ void FilmGradeKernelC( float* p_Input, int p_Width, int p_Height,
float p_ContR, float p_ContG, float p_ContB, float p_SatR, float p_SatG, float p_SatB, float p_ContP) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float contR = (p_Input[index] - p_ContP) * p_ContR + p_ContP;
float contG = (p_Input[index + 1] - p_ContP) * p_ContG + p_ContP;
float contB = (p_Input[index + 2] - p_ContP) * p_ContB + p_ContP;
float luma = contR * 0.2126f + contG * 0.7152f + contB * 0.0722f;
float outR = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contR * p_SatR;
float outG = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contG * p_SatG;
float outB = (1.0f - (p_SatR * 0.2126f + p_SatG * 0.7152f + p_SatB * 0.0722f)) * luma + contB * p_SatB;
p_Input[index] = outR;
p_Input[index + 1] = outG;
p_Input[index + 2] = outB;
}}

__global__ void FilmGradeKernelD( float* p_Input, float* p_Output, int p_Width, int p_Height, float p_Pivot, int p_Display) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
float height = p_Height;
float width = p_Width;
float X = x;
float Y = y;
const float RES = width / 1920.0f;
float overlay = 0.0f;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
if (p_Display == 1) {
overlay = Y / height >= p_Pivot && Y / height <= p_Pivot + 0.005f * RES ? (fmodf(X, 2.0f) != 0.0f ? 1.0f : 0.0f) : 
p_Output[index] >= (Y - 5.0f * RES) / height && p_Output[index] <= (Y + 5.0f * RES) / height ? 1.0f : 0.0f;
p_Output[index] = overlay;
}
if (p_Display == 2) {
overlay = Y / height >= p_Pivot && Y / height <= p_Pivot + 0.005f * RES ? (fmodf(X, 2.0f) != 0.0f ? 1.0f : 0.0f) : 
p_Input[index] >= (Y - 5.0f * RES) / height && p_Input[index] <= (Y + 5.0f * RES) / height ? 1.0f : 0.0f;
p_Output[index] = overlay == 0.0f ? p_Output[index] : overlay;
}}}

void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Exp, float* p_Cont, 
float* p_Sat, float* p_Shad, float* p_Mid, float* p_High, float* p_Pivot, int p_Display)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);
Prepare<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Display);
for(int ch = 0; ch < 3; ch++) {
FilmGradeKernelA<<<blocks, threads>>>(p_Output + ch, p_Width, p_Height, p_Exp[ch]);
FilmGradeKernelB<<<blocks, threads>>>(p_Output + ch, p_Width, p_Height, p_Shad[ch], p_Mid[ch], p_High[ch], p_Pivot[0], p_Pivot[1], p_Pivot[2]);
}
FilmGradeKernelC<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_Cont[0], p_Cont[1], p_Cont[2], p_Sat[0], p_Sat[1], p_Sat[2], p_Pivot[2]);
if (p_Display > 0) {
if (p_Display == 2) {
for(int ch = 0; ch < 3; ch++) {
FilmGradeKernelA<<<blocks, threads>>>(p_Input + ch, p_Width, p_Height, p_Exp[ch]);
FilmGradeKernelB<<<blocks, threads>>>(p_Input + ch, p_Width, p_Height, p_Shad[ch], p_Mid[ch], p_High[ch], p_Pivot[0], p_Pivot[1], p_Pivot[2]);
}
FilmGradeKernelC<<<blocks, threads>>>(p_Input, p_Width, p_Height, p_Cont[0], p_Cont[1], p_Cont[2], p_Sat[0], p_Sat[1], p_Sat[2], p_Pivot[2]);
}
for(int ch = 0; ch < 3; ch++) {
FilmGradeKernelD<<<blocks, threads>>>(p_Input + ch, p_Output + ch, p_Width, p_Height, p_Pivot[ch], p_Display);
}}}