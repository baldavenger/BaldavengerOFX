#include "ACES_LIB/ACES_IDT.h"
#include "ACES_LIB/ACES_LMT.h"
#include "ACES_LIB/ACES_RRT.h"
#include "ACES_LIB/ACES_ODT.h"
#include "ACES_LIB/ACES_CSC/ACES_Conversion.h"

__global__ void ACESKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Direction, int p_IDT, int p_ACESIN, int p_LMT, int p_ACESOUT, int p_RRT, int p_InvRRT, 
int p_ODT, int p_InvODT, float p_Exposure, float p_LMTScale1, float p_LMTScale2, float p_LMTScale3, 
float p_LMTScale4, float p_LMTScale5, float p_LMTScale6, float p_LMTScale7, float p_LMTScale8, 
float p_LMTScale9, float p_LMTScale10, float p_LMTScale11, float p_LMTScale12, float p_Lum0, 
float p_Lum1, float p_Lum2, int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int p_Switch0, 
int p_Switch1, int p_Switch2)
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

if(p_Direction == 0)
{

switch (p_IDT)
{
case 0:
{

}
break;

case 1:
{
aces = ACEScc_to_ACES(aces);
}
break;

case 2:
{
aces = ACEScct_to_ACES(aces);
}
break;

case 3:
{
aces = IDT_Alexa_v3_logC_EI800(aces);
}
break;

case 4:
{
aces = IDT_Alexa_v3_raw_EI800_CCT65(aces);
}
break;

case 5:
{
aces = ADX10_to_ACES(aces);
}
break;

case 6:
{
aces = ADX16_to_ACES(aces);
}
break;

case 7:
{
aces = IDT_Sony_SLog3_SGamut3(aces);
}
break;

case 8:
{
aces = IDT_Sony_SLog3_SGamut3Cine(aces);
}
}

if(p_Exposure != 0.0f)
{
aces.x *= powf(2.0f, p_Exposure);
aces.y *= powf(2.0f, p_Exposure);
aces.z *= powf(2.0f, p_Exposure);
}

switch (p_ACESIN)
{
case 0:
{

}
break;

case 1:
{
aces = ACES_to_ACEScc(aces);
}
break;

case 2:
{
aces = ACES_to_ACEScct(aces);
}
break;

case 3:
{
aces = ACES_to_ACEScg(aces);
}
break;

case 4:
{
aces = ACES_to_ACESproxy10(aces);
}
break;

case 5:
{
aces = ACES_to_ACESproxy12(aces);
}
}

switch (p_LMT)
{
case 0:
{

}
break;

case 1:
{
aces = LMT_Analytic_4(aces);
}
break;

case 2:
{
aces = LMT_Analytic_3(aces);
}
break;

case 3:
{
aces = scale_C(aces, p_LMTScale1);

float3 SLOPE = {p_LMTScale2, p_LMTScale2, p_LMTScale2};
float3 OFFSET = {p_LMTScale3, p_LMTScale3, p_LMTScale3};
float3 POWER = {p_LMTScale4, p_LMTScale4, p_LMTScale4};

aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER);
aces = gamma_adjust_linear(aces, p_LMTScale5, p_LMTScale6);
aces = rotate_H_in_H(aces, p_LMTScale7, p_LMTScale8, p_LMTScale9);
aces = scale_C_at_H(aces, p_LMTScale10, p_LMTScale11, p_LMTScale12);
}
}

switch (p_ACESOUT)
{
case 0:
{

}
break;

case 1:
{
aces = ACEScc_to_ACES(aces);
}
break;

case 2:
{
aces = ACEScct_to_ACES(aces);
}
break;

case 3:
{
aces = ACEScg_to_ACES(aces);
}
break;

case 4:
{
aces = ACESproxy10_to_ACES(aces);
}
break;

case 5:
{
aces = ACESproxy12_to_ACES(aces);
}
}

if(p_RRT == 1)
{
aces = RRT(aces);
}

switch (p_ODT)
{
case 0:
{
}
break;

case 1:
{
float Y_MIN = p_Lum0 * 0.0001f;
float Y_MID = p_Lum1;
float Y_MAX = p_Lum2;

Chromaticities DISPLAY_PRI = p_DISPLAY == 0 ? REC2020_PRI : p_DISPLAY == 1 ? P3D60_PRI : p_DISPLAY == 2 ? P3D65_PRI : p_DISPLAY == 3 ? P3DCI_PRI : REC709_PRI;
Chromaticities LIMITING_PRI = p_LIMIT == 0 ? REC2020_PRI : p_LIMIT == 1 ? P3D60_PRI : p_LIMIT == 2 ? P3D65_PRI : p_LIMIT == 3 ? P3DCI_PRI : REC709_PRI;

int EOTF = p_EOTF;
				
int SURROUND = p_SURROUND;
					           
bool STRETCH_BLACK = p_Switch0 == 1;
bool D60_SIM = p_Switch1 == 1;
bool LEGAL_RANGE = p_Switch1 == 1;

aces = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );

}
break;

case 2:
{
aces = ACES_to_ACEScc(aces);
}
break;

case 3:
{
aces = ACES_to_ACEScct(aces);
}
break;

case 4:
{
aces = ODT_Rec709_100nits_dim(aces);
}
break;

case 5:
{
aces = ODT_Rec2020_100nits_dim(aces);
}
break;

case 6:
{
aces = ODT_Rec2020_ST2084_1000nits(aces);
}
break;

case 7:
{
aces = ODT_RGBmonitor_100nits_dim(aces);
}
break;

case 8:
{
aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);
}
break;

case 9:
{
aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);
}
break;

case 10:
{
aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);
}
break;

case 11:
{
aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);
}
break;

case 12:
{
aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);
}
}

} else {

switch (p_InvODT)
{
case 0:
{
}
break;

case 1:
{
float Y_MIN = p_Lum0 * 0.0001f;
float Y_MID = p_Lum1;
float Y_MAX = p_Lum2;

Chromaticities DISPLAY_PRI = p_DISPLAY == 0 ? REC2020_PRI : p_DISPLAY == 1 ? P3D60_PRI : p_DISPLAY == 2 ? P3D65_PRI : p_DISPLAY == 3 ? P3DCI_PRI : REC709_PRI;
Chromaticities LIMITING_PRI = p_LIMIT == 0 ? REC2020_PRI : p_LIMIT == 1 ? P3D60_PRI : p_LIMIT == 2 ? P3D65_PRI : p_LIMIT == 3 ? P3DCI_PRI : REC709_PRI;

int EOTF = p_EOTF;
				
int SURROUND = p_SURROUND;
					           
bool STRETCH_BLACK = p_Switch0 == 1;
bool D60_SIM = p_Switch1 == 1;
bool LEGAL_RANGE = p_Switch1 == 1;

aces = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );

}
break;

case 2:
{
aces = InvODT_Rec709_100nits_dim(aces);
}
break;

case 3:
{
aces = InvODT_Rec2020_100nits_dim(aces);
}
break;

case 4:
{
aces = InvODT_Rec2020_ST2084_1000nits(aces);
}
break;

case 5:
{
aces = InvODT_RGBmonitor_100nits_dim(aces);
}
break;

case 6:
{
aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);
}
break;

case 7:
{
aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);
}
break;

case 8:
{
aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);
}
break;

case 9:
{
aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);
}
break;

case 10:
{
aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);
}
}

if(p_InvRRT == 1)
{
aces = InvRRT(aces);
}

}
																										
p_Output[index + 0] = aces.x;
p_Output[index + 1] = aces.y;
p_Output[index + 2] = aces.z;
p_Output[index + 3] = p_Input[index + 3];
}
}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Direction, int p_IDT, int p_ACESIN, int p_LMT, int p_ACESOUT, int p_RRT, 
int p_InvRRT, int p_ODT, int p_InvODT, float p_Exposure, float *p_LMTScale, float *p_Lum, 
int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_Switch)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

ACESKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Direction, p_IDT, p_ACESIN, p_LMT, p_ACESOUT, 
p_RRT, p_InvRRT, p_ODT, p_InvODT, p_Exposure, p_LMTScale[0], p_LMTScale[1], p_LMTScale[2], p_LMTScale[3], p_LMTScale[4], 
p_LMTScale[5], p_LMTScale[6], p_LMTScale[7], p_LMTScale[8], p_LMTScale[9], p_LMTScale[10], p_LMTScale[11], 
p_Lum[0], p_Lum[1], p_Lum[2], p_DISPLAY, p_LIMIT, p_EOTF, p_SURROUND, p_Switch[0], p_Switch[1], p_Switch[2]);

}