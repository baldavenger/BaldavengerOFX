//#include "ACES_LIB/ACES_functions.h"
//#include "ACES_LIB/ACES_Utilities_Color.h"
//#include "ACES_LIB/ACES_Transform_Common.h"
//#include "ACES_LIB/ACES_Tonescales.h"
#include "ACES_LIB/ACES_IDT.h"
#include "ACES_LIB/ACES_LMT.h"
#include "ACES_LIB/ACES_RRT.h"
#include "ACES_LIB/ACES_ODT.h"

__global__ void ACESKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_IDT, int p_LMT, int p_ODT, float p_Exposure)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < p_Width) && (y < p_Height))
{
const int index = ((y * p_Width) + x) * 4;

float3 aces;
aces.x = p_Input[index + 0];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];


switch (p_IDT)
{
case 0:
{
//aces.x *= 1.0f;
}
break;

case 1:
{
aces = IDT_Alexa_v3_logC_EI800(aces);
}
break;

case 2:
{
aces = IDT_Alexa_v3_raw_EI800_CCT65(aces);
}
break;

case 3:
{
aces = ADX10_to_ACES(aces);
}
break;

case 4:
{
aces = ADX16_to_ACES(aces);
}
}

if(p_Exposure != 0.0f)
{
aces.x *= powf(2.0f, p_Exposure);
aces.y *= powf(2.0f, p_Exposure);
aces.z *= powf(2.0f, p_Exposure);
}

switch (p_LMT)
{
case 0:
{
//aces.x *= 1.0f;
}
break;

case 1:
{
aces = LMT_Bleach_Bypass(aces);
}
break;

case 2:
{
aces = LMT_PFE(aces);
} 
}

switch (p_ODT)
{
case 0:
{
//aces.x *= 1.0f;
}
break;

case 1:
{
aces = RRT(aces);
aces = ODT_Rec709_100_dim(aces);
}
break;

case 2:
{
aces = RRT(aces);
aces = ODT_Rec2020_100nits_dim(aces);
}
break;

case 3:
{
aces = RRT(aces);
aces = ODT_Rec2020_ST2084_1000nits(aces);
}
break;

case 4:
{
aces = RRT(aces);
aces = ODT_RGBmonitor_100nits_dim(aces);
}
}


//aces = IDT_Alexa_v3_logC_EI800(aces);
//aces = ACES_to_ACEScct(aces);
//aces = LMT_Bleach_Bypass(aces);
//aces = ACEScct_to_ACES(aces);
//aces = RRT(aces);
//aces = InvRRT(aces);
//aces = ODT_Rec709_100_dim(aces);
//aces = InvODT_Rec709_100nits_dim(aces);
																										
p_Output[index + 0] = aces.x;
p_Output[index + 1] = aces.y;
p_Output[index + 2] = aces.z;
p_Output[index + 3] = p_Input[index + 3];
}
}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_IDT, int p_LMT, int p_ODT, float p_Exposure)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

ACESKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_IDT, p_LMT, p_ODT, p_Exposure);
}
