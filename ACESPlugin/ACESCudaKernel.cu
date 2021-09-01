#include "ACES_LIB_CUDA.h"

__global__ void k_Simple( const float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
p_Output[index] = p_Input[index]; 
p_Output[index + 1] = p_Input[index + 1]; 
p_Output[index + 2] = p_Input[index + 2]; 
p_Output[index + 3] = p_Input[index + 3]; 
}}

__global__ void k_CSCIN( float* p_Input, int p_Width, int p_Height, int p_CSCIN) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float3 aces;
aces.x = p_Input[index];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];
switch (p_CSCIN){
case 0:
{}
break;
case 1:
{aces = ACEScc_to_ACES(aces);}
break;
case 2:
{aces = ACEScct_to_ACES(aces);}
break;
case 3:
{aces = ACEScg_to_ACES(aces);}
break;
case 4:
{aces = ADX_to_ACES(aces);}
break;
case 5:
{aces = ICpCt_to_ACES(aces);}
break;
case 6:
{aces = LogC_EI800_AWG_to_ACES(aces);}
break;
case 7:
{aces = Log3G10_RWG_to_ACES(aces);}
break;
case 8:
{aces = SLog3_SG3_to_ACES(aces);}
break;
case 9:
{aces = SLog3_SG3C_to_ACES(aces);}
break;
case 10:
{aces = Venice_SLog3_SG3_to_ACES(aces);}
break;
case 11:
{aces = Venice_SLog3_SG3C_to_ACES(aces);}
break;
case 12:
{aces = VLog_VGamut_to_ACES(aces);}
}
p_Input[index] = aces.x; 
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}}

__global__ void k_IDT( float* p_Input, int p_Width, int p_Height, int p_IDT) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float3 aces;
aces.x = p_Input[index];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];
switch (p_IDT){
case 0:
{}
break;
case 1:
{aces = IDT_Alexa_v3_raw_EI800_CCT6500(aces);}
break;
case 2:
{aces = IDT_Panasonic_V35(aces);}
break;
case 3:
{aces = IDT_Canon_C100_A_D55(aces);}
break;
case 4:
{aces = IDT_Canon_C100_A_Tng(aces);}
break;
case 5:
{aces = IDT_Canon_C100mk2_A_D55(aces);}
break;
case 6:
{aces = IDT_Canon_C100mk2_A_Tng(aces);}
break;
case 7:
{aces = IDT_Canon_C300_A_D55(aces);}
break;
case 8:
{aces = IDT_Canon_C300_A_Tng(aces);}
break;
case 9:
{aces = IDT_Canon_C500_A_D55(aces);}
break;
case 10:
{aces = IDT_Canon_C500_A_Tng(aces);}
break;
case 11:
{aces = IDT_Canon_C500_B_D55(aces);}
break;
case 12:
{aces = IDT_Canon_C500_B_Tng(aces);}
break;
case 13:
{aces = IDT_Canon_C500_CinemaGamut_A_D55(aces);}
break;
case 14:
{aces = IDT_Canon_C500_CinemaGamut_A_Tng(aces);}
break;
case 15:
{aces = IDT_Canon_C500_DCI_P3_A_D55(aces);}
break;
case 16:
{aces = IDT_Canon_C500_DCI_P3_A_Tng(aces);}
break;
case 17:
{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_D55(aces);}
break;
case 18:
{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng(aces);}
break;
case 19:
{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55(aces);}
break;
case 20:
{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng(aces);}
break;
case 21:
{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55(aces);}
break;
case 22:
{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng(aces);}
break;
case 23:
{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55(aces);}
break;
case 24:
{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng(aces);}
break;
case 25:
{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55(aces);}
break;
case 26:
{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng(aces);}
break;
case 27:
{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55(aces);}
break;
case 28:
{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng(aces);}
break;
case 29:
{aces = IDT_Sony_SLog1_SGamut(aces);}
break;
case 30:
{aces = IDT_Sony_SLog2_SGamut_Daylight(aces);}
break;
case 31:
{aces = IDT_Sony_SLog2_SGamut_Tungsten(aces);}
break;
case 32:
{aces = IDT_Sony_Venice_SGamut3(aces);}
break;
case 33:
{aces = IDT_Sony_Venice_SGamut3Cine(aces);}
break;
case 34:
{aces = IDT_Rec709(aces);}
break;
case 35:
{aces = IDT_sRGB(aces);}
}
p_Input[index] = aces.x; 
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}}


__global__ void k_Exposure( float* p_Input, int p_Width, int p_Height, float p_Exposure) {                                 				   
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
p_Input[index] = p_Input[index] * exp2(p_Exposure);
p_Input[index + 1] = p_Input[index + 1] * exp2(p_Exposure);
p_Input[index + 2] = p_Input[index + 2] * exp2(p_Exposure);
}}

__global__ void k_LMT(float* p_Input, int p_Width, int p_Height, int p_LMT,
float p_LMTScale1, float p_LMTScale2, float p_LMTScale3, float p_LMTScale4, float p_LMTScale5, 
float p_LMTScale6, float p_LMTScale7, float p_LMTScale8, float p_LMTScale9, float p_LMTScale10, 
float p_LMTScale11, float p_LMTScale12,float p_LMTScale13, float p_LMTScale14, float p_LMTScale15, 
float p_LMTScale16, float p_LMTScale17, float p_LMTScale18, float p_LMTScale19, float p_LMTScale20, 
float p_LMTScale21, float p_LMTScale22, float p_LMTScale23, float p_LMTScale24, float p_GCompress1, 
float p_GCompress2, float p_GCompress3, float p_GCompress4, float p_GCompress5, float p_GCompress6, 
float p_GCompress7, int p_GInvert) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float3 aces;
aces.x = p_Input[index];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];
switch (p_LMT){
case 0:
{}
break;
case 1:
{
if (p_LMTScale1 != 1.0f)
aces = scale_C(aces, p_LMTScale1);
if (!(p_LMTScale2 == 1.0f && p_LMTScale3 == 0.0f && p_LMTScale4 == 1.0f)) {
float3 SLOPE = {p_LMTScale2, p_LMTScale2, p_LMTScale2};
float3 OFFSET = {p_LMTScale3, p_LMTScale3, p_LMTScale3};
float3 POWER = {p_LMTScale4, p_LMTScale4, p_LMTScale4};
aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f);
}
if(p_LMTScale5 != 1.0f)
aces = gamma_adjust_linear(aces, p_LMTScale5, p_LMTScale6);
if(p_LMTScale9 != 0.0f)
aces = rotate_H_in_H(aces, p_LMTScale7, p_LMTScale8, p_LMTScale9);
if(p_LMTScale12 != 0.0f)
aces = rotate_H_in_H(aces, p_LMTScale10, p_LMTScale11, p_LMTScale12);
if(p_LMTScale15 != 0.0f)
aces = rotate_H_in_H(aces, p_LMTScale13, p_LMTScale14, p_LMTScale15);
if(p_LMTScale18 != 1.0f)
aces = scale_C_at_H(aces, p_LMTScale16, p_LMTScale17, p_LMTScale18);
if(p_LMTScale21 != 0.0f)
aces = rotate_H_in_H(aces, p_LMTScale19, p_LMTScale20, p_LMTScale21);
if(p_LMTScale24 != 1.0f)
aces = scale_C_at_H(aces, p_LMTScale22, p_LMTScale23, p_LMTScale24);
}
break;
case 2:
{aces = LMT_BlueLightArtifactFix(aces);}
break;
case 3:
{aces = LMT_PFE(aces);}
break;
case 4:
{
if (p_LMTScale1 != 1.0f)
aces = scale_C(aces, p_LMTScale1);
float3 SLOPE = {p_LMTScale2, p_LMTScale2, p_LMTScale2 * 0.94f};
float3 OFFSET = {p_LMTScale3, p_LMTScale3, p_LMTScale3 + 0.02f};
float3 POWER = {p_LMTScale4, p_LMTScale4, p_LMTScale4};
aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f);
if(p_LMTScale5 != 1.0f)
aces = gamma_adjust_linear(aces, p_LMTScale5, p_LMTScale6);
if(p_LMTScale9 != 0.0f)
aces = rotate_H_in_H(aces, p_LMTScale7, p_LMTScale8, p_LMTScale9);
if(p_LMTScale12 != 0.0f)
aces = rotate_H_in_H(aces, p_LMTScale10, p_LMTScale11, p_LMTScale12);
if(p_LMTScale15 != 0.0f)
aces = rotate_H_in_H(aces, p_LMTScale13, p_LMTScale14, p_LMTScale15);
if(p_LMTScale18 != 1.0f)
aces = scale_C_at_H(aces, p_LMTScale16, p_LMTScale17, p_LMTScale18);
if(p_LMTScale21 != 0.0f)
aces = rotate_H_in_H(aces, p_LMTScale19, p_LMTScale20, p_LMTScale21);
if(p_LMTScale24 != 1.0f)
aces = scale_C_at_H(aces, p_LMTScale22, p_LMTScale23, p_LMTScale24);
}
break;
case 5:
{aces = LMT_Bleach(aces);}
break;
case 6:
{aces = LMT_GamutCompress(aces);}
break;
case 7:
{
bool ginvert = p_GInvert == 1;
aces = gamut_compress(aces, p_GCompress1, 
p_GCompress2, p_GCompress3, p_GCompress4, p_GCompress5, p_GCompress6, 
p_GCompress7, ginvert);}
}
p_Input[index] = aces.x; 
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}}

__global__ void k_CSCOUT( float* p_Input, int p_Width, int p_Height, int p_CSCOUT) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float3 aces;
aces.x = p_Input[index];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];
switch (p_CSCOUT){
case 0:
{}
break;
case 1:
{aces = ACES_to_ACEScc(aces);}
break;
case 2:
{aces = ACES_to_ACEScct(aces);}
break;
case 3:
{aces = ACES_to_ACEScg(aces);}
break;
case 4:
{aces = ACES_to_ADX(aces);}
break;
case 5:
{aces = ACES_to_ICpCt(aces);}
break;
case 6:
{aces = ACES_to_LogC_EI800_AWG(aces);}
break;
case 7:
{aces = ACES_to_Log3G10_RWG(aces);}
break;
case 8:
{aces = ACES_to_SLog3_SG3(aces);}
break;
case 9:
{aces = ACES_to_SLog3_SG3C(aces);}
break;
case 10:
{aces = ACES_to_Venice_SLog3_SG3(aces);}
break;
case 11:
{aces = ACES_to_Venice_SLog3_SG3C(aces);}
break;
case 12:
{aces = ACES_to_VLog_VGamut(aces);}
}
p_Input[index] = aces.x; 
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}}

__global__ void k_RRT( float* p_Input, int p_Width, int p_Height, int p_RRT) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float3 aces;
aces.x = p_Input[index];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];
aces = RRT(aces);
p_Input[index] = aces.x; 
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}}

__global__ void k_InvRRT( float* p_Input, int p_Width, int p_Height, int p_InvRRT) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float3 aces;
aces.x = p_Input[index];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];
aces = InvRRT(aces);
p_Input[index] = aces.x; 
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}}

__global__ void k_ODT( float* p_Input, int p_Width, int p_Height, int p_ODT, 
float p_Lum0, float p_Lum1, float p_Lum2, int p_DISPLAY, int p_LIMIT, int p_EOTF, 
int p_SURROUND, int p_Switch0, int p_Switch1, int p_Switch2) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float3 aces;
aces.x = p_Input[index];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];
switch (p_ODT){
case 0:
{}
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
bool LEGAL_RANGE = p_Switch2 == 1;
aces = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
}
break;
case 2:
{aces = ODT_Rec709_100nits_dim(aces);}
break;
case 3:
{aces = ODT_Rec709_D60sim_100nits_dim(aces);}
break;
case 4:
{aces = ODT_sRGB_100nits_dim(aces);}
break;
case 5:
{aces = ODT_sRGB_D60sim_100nits_dim(aces);}
break;
case 6:
{aces = ODT_Rec2020_100nits_dim(aces);}
break;
case 7:
{aces = ODT_Rec2020_Rec709limited_100nits_dim(aces);}
break;
case 8:
{aces = ODT_Rec2020_P3D65limited_100nits_dim(aces);}
break;
case 9:
{aces = ODT_Rec2020_ST2084_1000nits(aces);}
break;
case 10:
{aces = ODT_P3DCI_48nits(aces);}
break;
case 11:
{aces = ODT_P3DCI_D60sim_48nits(aces);}
break;
case 12:
{aces = ODT_P3DCI_D65sim_48nits(aces);}
break;
case 13:
{aces = ODT_P3D60_48nits(aces);}
break;
case 14:
{aces = ODT_P3D65_48nits(aces);}
break;
case 15:
{aces = ODT_P3D65_D60sim_48nits(aces);}
break;
case 16:
{aces = ODT_P3D65_Rec709limited_48nits(aces);}
break;
case 17:
{aces = ODT_DCDM(aces);}
break;
case 18:
{aces = ODT_DCDM_P3D60limited(aces);}
break;
case 19:
{aces = ODT_DCDM_P3D65limited(aces);}
break;
case 20:
{aces = ODT_RGBmonitor_100nits_dim(aces);}
break;
case 21:
{aces = ODT_RGBmonitor_D60sim_100nits_dim(aces);}
break;
case 22:
{aces = RRTODT_Rec709_100nits_10nits_BT1886(aces);}
break;
case 23:
{aces = RRTODT_Rec709_100nits_10nits_sRGB(aces);}
break;
case 24:
{aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);}
break;
case 25:
{aces = RRTODT_P3D65_1000nits_15nits_ST2084(aces);}
break;
case 26:
{aces = RRTODT_P3D65_2000nits_15nits_ST2084(aces);}
break;
case 27:
{aces = RRTODT_P3D65_4000nits_15nits_ST2084(aces);}
break;
case 28:
{aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);}
break;
case 29:
{aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);}
break;
case 30:
{aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);}
break;
case 31:
{aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);}
}
p_Input[index] = aces.x; 
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}}

__global__ void k_InvODT( float* p_Input, int p_Width, int p_Height, int p_InvODT, 
float p_Lum0, float p_Lum1, float p_Lum2, int p_DISPLAY, int p_LIMIT, int p_EOTF, 
int p_SURROUND, int p_Switch0, int p_Switch1, int p_Switch2) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float3 aces;
aces.x = p_Input[index];
aces.y = p_Input[index + 1];
aces.z = p_Input[index + 2];
switch (p_InvODT){
case 0:
{}
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
bool LEGAL_RANGE = p_Switch2 == 1;
aces = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
}
break;
case 2:
{aces = InvODT_Rec709_100nits_dim(aces);}
break;
case 3:
{aces = InvODT_Rec709_D60sim_100nits_dim(aces);}
break;
case 4:
{aces = InvODT_sRGB_100nits_dim(aces);}
break;
case 5:
{aces = InvODT_sRGB_D60sim_100nits_dim(aces);}
break;
case 6:
{aces = InvODT_Rec2020_100nits_dim(aces);}
break;
case 7:
{aces = InvODT_Rec2020_ST2084_1000nits(aces);}
break;
case 8:
{aces = InvODT_P3DCI_48nits(aces);}
break;
case 9:
{aces = InvODT_P3DCI_D60sim_48nits(aces);}
break;
case 10:
{aces = InvODT_P3DCI_D65sim_48nits(aces);}
break;
case 11:
{aces = InvODT_P3D60_48nits(aces);}
break;
case 12:
{aces = InvODT_P3D65_48nits(aces);}
break;
case 13:
{aces = InvODT_P3D65_D60sim_48nits(aces);}
break;
case 14:
{aces = InvODT_DCDM(aces);}
break;
case 15:
{aces = InvODT_DCDM_P3D65limited(aces);}
break;
case 16:
{aces = InvODT_RGBmonitor_100nits_dim(aces);}
break;
case 17:
{aces = InvODT_RGBmonitor_D60sim_100nits_dim(aces);}
break;
case 18:
{aces = InvRRTODT_Rec709_100nits_10nits_BT1886(aces);}
break;
case 19:
{aces = InvRRTODT_Rec709_100nits_10nits_sRGB(aces);}
break;
case 20:
{aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);}
break;
case 21:
{aces = InvRRTODT_P3D65_1000nits_15nits_ST2084(aces);}
break;
case 22:
{aces = InvRRTODT_P3D65_2000nits_15nits_ST2084(aces);}
break;
case 23:
{aces = InvRRTODT_P3D65_4000nits_15nits_ST2084(aces);}
break;
case 24:
{aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);}
break;
case 25:
{aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);}
break;
case 26:
{aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);}
break;
case 27:
{aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);}
}
p_Input[index] = aces.x; 
p_Input[index + 1] = aces.y;
p_Input[index + 2] = aces.z;
}}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Direction, int p_CSCIN, int p_IDT, int p_LMT, int p_CSCOUT, int p_RRT, 
int p_InvRRT, int p_ODT, int p_InvODT, float p_Exposure, float *p_LMTScale, float *p_GCompress, 
int p_GInvert, float *p_Lum, int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_Switch)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

k_Simple<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height);

if (p_Direction == 0) {
if (p_CSCIN > 0)
k_CSCIN<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_CSCIN);
if (p_IDT > 0)
k_IDT<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_IDT);
if (p_Exposure != 0.0f)
k_Exposure<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_Exposure);
if (p_LMT > 0)
k_LMT<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_LMT, p_LMTScale[0], p_LMTScale[1], p_LMTScale[2], p_LMTScale[3], 
p_LMTScale[4], p_LMTScale[5], p_LMTScale[6], p_LMTScale[7], p_LMTScale[8], p_LMTScale[9], p_LMTScale[10], p_LMTScale[11], 
p_LMTScale[12], p_LMTScale[13], p_LMTScale[14], p_LMTScale[15], p_LMTScale[16], p_LMTScale[17], p_LMTScale[18], p_LMTScale[19], 
p_LMTScale[20], p_LMTScale[21], p_LMTScale[22], p_LMTScale[23], p_GCompress[0], p_GCompress[1], p_GCompress[2], p_GCompress[3], 
p_GCompress[4], p_GCompress[5], p_GCompress[6], p_GInvert);
if (p_CSCOUT > 0)
k_CSCOUT<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_CSCOUT);
if (p_RRT > 0 && p_ODT < 22)
k_RRT<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_RRT);
if (p_ODT > 0)
k_ODT<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_ODT, p_Lum[0], p_Lum[1], p_Lum[2], p_DISPLAY, p_LIMIT, 
p_EOTF, p_SURROUND, p_Switch[0], p_Switch[1], p_Switch[2]);
}

if (p_Direction == 1) {
if (p_InvODT > 0)
k_InvODT<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_InvODT, p_Lum[0], p_Lum[1], p_Lum[2], p_DISPLAY, p_LIMIT, 
p_EOTF, p_SURROUND, p_Switch[0], p_Switch[1], p_Switch[2]);
if (p_InvRRT > 0 && p_InvODT < 18)
k_InvRRT<<<blocks, threads>>>(p_Output, p_Width, p_Height, p_InvRRT);
}
}