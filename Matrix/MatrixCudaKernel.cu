__device__ float Luma(float R, float G, float B, int L) {
float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f;
float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f;
float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f;
float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f;
float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f;
float lumaAvg = (R + G + B) / 3.0f;
float lumaMax = fmax(fmax(R, G), B);
float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax;
return Lu;
}

__device__ float Sat(float r, float g, float b){
float min = fmin(fmin(r, g), b);
float max = fmax(fmax(r, g), b);
float delta = max - min;
float S = max != 0.0f ? delta / max : 0.0f;
return S;
}

__global__ void MatrixKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, float p_MatrixRR, 
float p_MatrixRG, float p_MatrixRB, float p_MatrixGR, float p_MatrixGG, float p_MatrixGB, float p_MatrixBR, 
float p_MatrixBG, float p_MatrixBB, int p_Luma, int p_Sat, int p_LumaMath)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height)
{
const int index = (y * p_Width + x) * 4;
float red = p_Input[index] * p_MatrixRR + p_Input[index + 1] * p_MatrixRG + p_Input[index + 2] * p_MatrixRB;
float green = p_Input[index] * p_MatrixGR + p_Input[index + 1] * p_MatrixGG + p_Input[index + 2] * p_MatrixGB;
float blue = p_Input[index] * p_MatrixBR + p_Input[index + 1] * p_MatrixBG + p_Input[index + 2] * p_MatrixBB;
if (p_Luma == 1) {
float inLuma = Luma(p_Input[index], p_Input[index + 1], p_Input[index + 2], p_LumaMath);
float outLuma = Luma(red, green, blue, p_LumaMath);
red = red * (inLuma / outLuma);
green = green * (inLuma / outLuma);
blue = blue * (inLuma / outLuma);
}
if (p_Sat == 1) {
float inSat = Sat(p_Input[index], p_Input[index + 1], p_Input[index + 2]);
float outSat = Sat(red, green, blue);
float satgap = inSat / outSat;
float sLuma = Luma(red, green, blue, p_LumaMath);
float sr = (1.0f - satgap) * sLuma + red * satgap;
float sg = (1.0f - satgap) * sLuma + green * satgap;
float sb = (1.0f - satgap) * sLuma + blue * satgap;
red = inSat == 0.0f ? sLuma : sr;
green = inSat == 0.0f ? sLuma : sg;
blue = inSat == 0.0f ? sLuma : sb;
}
p_Output[index] = red;
p_Output[index + 1] = green;
p_Output[index + 2] = blue;
p_Output[index + 3] = p_Input[index + 3]; 
}}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Matrix, int p_Luma, int p_Sat, int p_LumaMath)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

MatrixKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Matrix[0], p_Matrix[1], p_Matrix[2], 
p_Matrix[3], p_Matrix[4], p_Matrix[5], p_Matrix[6], p_Matrix[7], p_Matrix[8], p_Luma, p_Sat, p_LumaMath);
}