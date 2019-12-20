__global__ void SoftClipKernel( const float* p_Input, float* p_Output, int p_Width, 
int p_Height, float p_SoftClipA, float p_SoftClipB, float p_SoftClipC, float p_SoftClipD, 
float p_SoftClipE, float p_SoftClipF, int p_SwitchA, int p_SwitchB, int p_Source) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float r = p_Input[index];
float g = p_Input[index + 1];
float b = p_Input[index + 2];

float cr = (powf(10.0f, (1023.0f * r - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);
float cg = (powf(10.0f, (1023.0f * g - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);
float cb = (powf(10.0f, (1023.0f * b - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);

float lr = r > 0.1496582f ? (powf(10.0f, (r - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (r - 0.092809f) / 5.367655f;
float lg = g > 0.1496582f ? (powf(10.0f, (g - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (g - 0.092809f) / 5.367655f;
float lb = b > 0.1496582f ? (powf(10.0f, (b - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (b - 0.092809f) / 5.367655f;

float mr = lr * 1.617523f  + lg * -0.537287f + lb * -0.080237f;
float mg = lr * -0.070573f + lg * 1.334613f  + lb * -0.26404f;
float mb = lr * -0.021102f + lg * -0.226954f + lb * 1.248056f;

float sr = p_Source == 0 ? r : p_Source == 1 ? cr : mr;
float sg = p_Source == 0 ? g : p_Source == 1 ? cg : mg;
float sb = p_Source == 0 ? b : p_Source == 1 ? cb : mb;

float Lr = sr > 1.0f ? 1.0f : sr;
float Lg = sg > 1.0f ? 1.0f : sg;
float Lb = sb > 1.0f ? 1.0f : sb;

float Hr = (sr < 1.0f ? 1.0f : sr) - 1.0f;
float Hg = (sg < 1.0f ? 1.0f : sg) - 1.0f;
float Hb = (sb < 1.0f ? 1.0f : sb) - 1.0f;

float rr = p_SoftClipA;
float gg = p_SoftClipB;
float aa = p_SoftClipC;
float bb = p_SoftClipD;
float ss = 1.0f - (p_SoftClipE / 10.0f);
float sf = 1.0f - p_SoftClipF;

float Hrr = Hr * powf(2.0f, rr);
float Hgg = Hg * powf(2.0f, rr);
float Hbb = Hb * powf(2.0f, rr);

float HR = Hrr <= 1.0f ? 1.0f - powf(1.0f - Hrr, gg) : Hrr;
float HG = Hgg <= 1.0f ? 1.0f - powf(1.0f - Hgg, gg) : Hgg;
float HB = Hbb <= 1.0f ? 1.0f - powf(1.0f - Hbb, gg) : Hbb;

float R = Lr + HR;
float G = Lg + HG;
float B = Lb + HB;

float softr = aa == 1.0f ? R : (R > aa ? (-1.0f / ((R - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : R);
float softR = bb == 1.0f ? softr : softr > 1.0f - (bb / 50.0f) ? (-1.0f / ((softr - (1.0f - (bb / 50.0f))) / 
(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softr;
float softg = (aa == 1.0f) ? G : (G > aa ? (-1.0f / ((G - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : G);
float softG = bb == 1.0f ? softg : softg > 1.0f - (bb / 50.0f) ? (-1.0f / ((softg - (1.0f - (bb / 50.0f))) / 
(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softg;
float softb = (aa == 1.0f) ? B : (B > aa ? (-1.0f / ((B - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : B);
float softB = bb == 1.0f ? softb : softb > 1.0f - (bb / 50.0f) ? (-1.0f / ((softb - (1.0f - (bb / 50.0f))) / 
(1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softb;

float Cr = (softR * -1.0f) + 1.0f;
float Cg = (softG * -1.0f) + 1.0f;
float Cb = (softB * -1.0f) + 1.0f;

float cR = ss == 1.0f ? Cr : Cr > ss ? (-1.0f / ((Cr - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cr;
float CR = sf == 1.0f ? (cR - 1.0f) * -1.0f : ((cR > 1.0f - (-p_SoftClipF / 50.0f) ? (-1.0f / ((cR - (1.0f - (-p_SoftClipF / 50.0f))) / 
(1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + (1.0f - (-p_SoftClipF / 50.0f)) : cR) - 1.0f) * -1.0f;
float cG = ss == 1.0f ? Cg : Cg > ss ? (-1.0f / ((Cg - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cg;
float CG = sf == 1.0f ? (cG - 1.0f) * -1.0f : ((cG > 1.0f - (-p_SoftClipF / 50.0f) ? (-1.0f / ((cG - (1.0f - (-p_SoftClipF / 50.0f))) / 
(1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + (1.0f - (-p_SoftClipF / 50.0f)) : cG) - 1.0f) * -1.0f;
float cB = ss == 1.0f ? Cb : Cb > ss ? (-1.0f / ((Cb - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cb;
float CB = sf == 1.0f ? (cB - 1.0f) * -1.0f : ((cB > 1.0f - (-p_SoftClipF / 50.0f) ? (-1.0f / ((cB - (1.0f - (-p_SoftClipF / 50.0f))) / 
(1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-p_SoftClipF / 50.0f))) + (1.0f - (-p_SoftClipF / 50.0f)) : cB) - 1.0f) * -1.0f;

float SR = p_Source == 0 ? CR : CR >= 0.0f && CR <= 1.0f ? (CR < 0.0181f ? (CR * 4.5f) : 1.0993f * powf(CR, 0.45f) - (1.0993f - 1.0f)) : CR;
float SG = p_Source == 0 ? CG : CG >= 0.0f && CG <= 1.0f ? (CG < 0.0181f ? (CG * 4.5f) : 1.0993f * powf(CG, 0.45f) - (1.0993f - 1.0f)) : CG;
float SB = p_Source == 0 ? CB : CB >= 0.0f && CB <= 1.0f ? (CB < 0.0181f ? (CB * 4.5f) : 1.0993f * powf(CB, 0.45f) - (1.0993f - 1.0f)) : CB;

p_Output[index] = p_SwitchA == 1 ? (SR < 1.0f ? 1.0f : SR) - 1.0f : p_SwitchB == 1 ? (SR >= 0.0f ? 0.0f : SR + 1.0f) : SR;
p_Output[index + 1] = p_SwitchA == 1 ? (SG < 1.0f ? 1.0f : SG) - 1.0f : p_SwitchB == 1 ? (SG >= 0.0f ? 0.0f : SG + 1.0f) : SG;
p_Output[index + 2] = p_SwitchA == 1 ? (SB < 1.0f ? 1.0f : SB) - 1.0f : p_SwitchB == 1 ? (SB >= 0.0f ? 0.0f : SB + 1.0f) : SB;
p_Output[index + 3] = p_Input[index + 3];
}}

void RunCudaKernel( const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Scales, int* p_Switch, int p_Source)
{
dim3 threads(128, 1, 1);
dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

SoftClipKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Scales[0], p_Scales[1], p_Scales[2], 
p_Scales[3], p_Scales[4], p_Scales[5], p_Switch[0], p_Switch[1], p_Source);
}