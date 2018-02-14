__global__ void HueConvergeKernel(int p_Width, int p_Height, float p_SwitchA, float p_SwitchB, float p_SwitchC, 
float p_SwitchD, float p_SwitchE, float p_SwitchL, float p_Alpha1, float p_Alpha2, float p_LogA, float p_LogB, float p_LogC, 
float p_LogD, float p_SatA, float p_SatB, float p_SatC, float p_HueA, float p_HueB, float p_HueC, float p_HueR1, 
float p_HueP1, float p_HueSH1, float p_HueSHP1, float p_HueR2, float p_HueP2, float p_HueSH2, float p_HueSHP2, float p_HueR3, 
float p_HueP3, float p_HueSH3, float p_HueSHP3, float p_LumaA, float p_LumaB, float p_LumaC, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
	const int index = ((y * p_Width) + x) * 4;

	// Euler's Constant e = 2.718281828459045

	float e = 2.718281828459045;

	// Default expression : 1.0f / (1.0f + _powf(e, -8.9f*(r - 0.435f)))

	// Logistic Function (Sigmoid Curve)

	float Lr = p_SwitchA == 1.0f ? p_LogA / (1.0f + powf(e, (-8.9f * p_LogB) * (p_Input[index + 0] - p_LogC))) + p_LogD : p_Input[index + 0];
	float Lg = p_SwitchA == 1.0f ? p_LogA / (1.0f + powf(e, (-8.9f * p_LogB) * (p_Input[index + 1] - p_LogC))) + p_LogD : p_Input[index + 1];
	float Lb = p_SwitchA == 1.0f ? p_LogA / (1.0f + powf(e, (-8.9f * p_LogB) * (p_Input[index + 2] - p_LogC))) + p_LogD : p_Input[index + 2];

	float mn = fmin(Lr, fmin(Lg, Lb));    
	float Mx = fmax(Lr, fmax(Lg, Lb));    
	float del_Mx = Mx - mn;

	float L = (Mx + mn) / 2.0f;
	float luma = p_Input[index + 0] * 0.2126f + p_Input[index + 1] * 0.7152f + p_Input[index + 2] * 0.0722f;
	float Sat = del_Mx == 0.0f ? 0.0f : L < 0.5f ? del_Mx / (Mx + mn) : del_Mx / (2.0f - Mx - mn);

	float del_R = ((Mx - Lr) / 6.0f + del_Mx / 2.0f) / del_Mx;
	float del_G = ((Mx - Lg) / 6.0f + del_Mx / 2.0f) / del_Mx;
	float del_B = ((Mx - Lb) / 6.0f + del_Mx / 2.0f) / del_Mx;

	float h = del_Mx == 0.0f ? 0.0f : Lr == Mx ? del_B - del_G : Lg == Mx ? 1.0f / 3.0f + 
	del_R - del_B : 2.0f / 3.0f + del_G - del_R;

	float Hh = h < 0.0f ? h + 1.0f : h > 1.0f ? h - 1.0f : h;

	// Soft Clip Saturation

	float s = Sat * p_SatA > 1.0f ? 1.0f : Sat * p_SatA;
	float ss = s > p_SatB ? (-1.0f / ((s - p_SatB) / (1.0f - p_SatB) + 1.0f) + 1.0f) * (1.0f - p_SatB) + p_SatB : s;
	float satAlphaA = p_SatC > 1.0f ? luma + (1.0f - p_SatC) * (1.0f - luma) : p_SatC >= 0.0f ? (luma >= p_SatC ? 1.0f : 
	luma / p_SatC) : p_SatC < -1.0f ? (1.0f - luma) + (p_SatC + 1.0f) * luma : luma <= (1.0f + p_SatC) ? 1.0f : 
	(1.0 - luma) / (1.0f - (p_SatC + 1.0f));
	float satAlpha = satAlphaA > 1.0f ? 1.0f : satAlphaA;

	float S = p_SwitchB == 1.0f ? ss * satAlpha + s * (1.0f - satAlpha) : Sat;

	// Hue Convergence

	float h1 = Hh - (p_HueA - 0.5f) < 0.0f ? Hh - (p_HueA - 0.5f) + 1.0f : Hh - (p_HueA - 0.5f) >
	1.0f ? Hh - (p_HueA - 0.5f) - 1.0f : Hh - (p_HueA - 0.5f);

	float Hs1 = p_HueSHP1 >= 1.0f ? 1.0f - powf(1.0f - S, p_HueSHP1) : powf(S, 1.0f/p_HueSHP1);
	float cv1 = 1.0f + (p_HueP1 - 1.0f) * Hs1;
	float curve1 = p_HueP1 + (cv1 - p_HueP1) * p_HueSH1;

	float H1 = p_SwitchC != 1.0f ? Hh : h1 > 0.5f - p_HueR1 && h1 < 0.5f ? (1.0f - powf(1.0f - (h1 - (0.5f - p_HueR1)) *
	(1.0f/p_HueR1), curve1)) * p_HueR1 + (0.5f - p_HueR1) + (p_HueA - 0.5f) : h1 > 0.5f && h1 < 0.5f + 
	p_HueR1 ? powf((h1 - 0.5f) * (1.0f/p_HueR1), curve1) * p_HueR1 + 0.5f + (p_HueA - 0.5f) : Hh;

	float h2 = H1 - (p_HueB - 0.5f) < 0.0f ? H1 - (p_HueB - 0.5f) + 1.0f : H1 - (p_HueB - 0.5f) > 
	1.0f ? H1 - (p_HueB - 0.5f) - 1.0f : H1 - (p_HueB - 0.5f);

	float Hs2 = p_HueSHP2 >= 1.0f ? 1.0f - powf(1.0f - S, p_HueSHP2) : powf(S, 1.0f/p_HueSHP2);
	float cv2 = 1.0f + (p_HueP2 - 1.0f) * Hs2;
	float curve2 = p_HueP2 + (cv2 - p_HueP2) * p_HueSH2;

	float H2 = p_SwitchD != 1.0f ? H1 : h2 > 0.5f - p_HueR2 && h2 < 0.5f ? (1.0f - powf(1.0f - (h2 - (0.5f - p_HueR2)) *
	(1.0f/p_HueR2), curve2)) * p_HueR2 + (0.5f - p_HueR2) + (p_HueB - 0.5f) : h2 > 0.5f && h2 < 0.5f + 
	p_HueR2 ? powf((h2 - 0.5f) * (1.0f/p_HueR2), curve2) * p_HueR2 + 0.5f + (p_HueB - 0.5f) : H1;

	float h3 = H2 - (p_HueC - 0.5f) < 0.0f ? H2 - (p_HueC - 0.5f) + 1.0f : H2 - (p_HueC - 0.5f) > 
	1.0f ? H2 - (p_HueC - 0.5f) - 1.0f : H2 - (p_HueC - 0.5f);

	float Hs3 = p_HueSHP3 >= 1.0f ? 1.0f - powf(1.0f - S, p_HueSHP3) : powf(S, 1.0f/p_HueSHP3);
	float cv3 = 1.0f + (p_HueP3 - 1.0f) * Hs3;
	float curve3 = p_HueP3 + (cv3 - p_HueP3) * p_HueSH3;

	float H = p_SwitchE != 1.0f ? H2 : h3 > 0.5f - p_HueR3 && h3 < 0.5f ? (1.0f - powf(1.0f - (h3 - (0.5f - p_HueR3)) *
	(1.0f/p_HueR3), curve3)) * p_HueR3 + (0.5f - p_HueR3) + (p_HueC - 0.5f) : h3 > 0.5f && h3 < 0.5f + 
	p_HueR3 ? powf((h3 - 0.5f) * (1.0f/p_HueR3), curve3) * p_HueR3 + 0.5f + (p_HueC - 0.5f) : H2;

	// HSL to RGB

	float Q = L < 0.5f ? L * (1.0f + S) : L + S - L * S;
	float P = 2.0f * L - Q;

	float RH = H + 1.0f / 3.0f < 0.0f ? H + 1.0f / 3.0f + 1.0f :
	H + 1.0f / 3.0f > 1.0f ? H + 1.0f / 3.0f - 1.0f : H + 1.0f / 3.0f;

	float RR = RH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * RH : 
	RH < 1.0f / 2.0f ? Q : RH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - RH) * 6.0f : P;

	float GH = H < 0.0f ? H + 1.0f : H > 1.0f ? H - 1.0f : H;

	float GG = GH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * GH :
	GH < 1.0f / 2.0f ? Q : GH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - GH) * 6.0f : P;

	float BH = H - 1.0f / 3.0f < 0.0f ? H - 1.0f / 3.0f + 1.0f :
	H - 1.0f / 3.0f > 1.0f ? H - 1.0f / 3.0f - 1.0f : H - 1.0f / 3.0f;

	float BB = BH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * BH :
	BH < 1.0f / 2.0f ? Q : BH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - BH) * 6.0f : P;  

	float r1 = S == 0.0f ? L : RR;
	float g1 = S == 0.0f ? L : GG;
	float b1 = S == 0.0f ? L : BB;

	// Sat vs Luma

	float luma601 = 0.299f * r1 + 0.587f * g1 + 0.114f * b1;
	float meanluma = (r1 + g1 + b1) / 3.0f;
	float del_luma = luma601 / meanluma;

	float satluma = p_LumaA >= 1.0f ? meanluma + ((1.0f -p_LumaA) * (1.0f - meanluma)) : p_LumaA >= 0.0f ? (meanluma >= p_LumaA ? 
	1.0f : meanluma / p_LumaA) : p_LumaA < -1.0f ? (1.0f - meanluma) + (p_LumaA + 1.0f) * meanluma : meanluma <= (1.0f + p_LumaA) ? 1.0f : 
	(1.0f - meanluma) / (1.0f - (p_LumaA + 1.0f));
	float satlumaA = satluma <= 0.0f ? 0.0f : satluma >= 1.0f ? 1.0f : satluma; 
	float satlumaB = (p_LumaB > 1.0f ? 1.0f - powf(1.0f - S, p_LumaB) : powf(S, 1.0f / p_LumaB)) * satlumaA * del_luma;
	float satalphaL = satlumaB > 1.0f ? 1.0f : satlumaB;

	float r = p_SwitchL == 1.0f ? (r1 * p_LumaC * satalphaL) + (r1 * (1.0f - satalphaL)) : r1;
	float g = p_SwitchL == 1.0f ? (g1 * p_LumaC * satalphaL) + (g1 * (1.0f - satalphaL)) : g1;
	float b = p_SwitchL == 1.0f ? (b1 * p_LumaC * satalphaL) + (b1 * (1.0f - satalphaL)) : b1;


	p_Output[index + 0] = p_Alpha1 == 1.0f ? satAlpha : p_Alpha2 == 1.0f ? satalphaL : r;
	p_Output[index + 1] = p_Alpha1 == 1.0f ? satAlpha : p_Alpha2 == 1.0f ? satalphaL : g;
	p_Output[index + 2] = p_Alpha1 == 1.0f ? satAlpha : p_Alpha2 == 1.0f ? satalphaL : b;
	p_Output[index + 3] = p_Input[index + 3];
   }
}

void  RunCudaKernel(int p_Width, int p_Height, float* p_Switch, float* p_Alpha, float* p_Log, float* p_Sat, float* p_Hue, 
	float* p_Luma, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    HueConvergeKernel<<<blocks, threads>>>(p_Width, p_Height, p_Switch[0], p_Switch[1], p_Switch[2], p_Switch[3], 
  p_Switch[4], p_Switch[5], p_Alpha[0], p_Alpha[1], p_Log[0], p_Log[1], p_Log[2], p_Log[3], p_Sat[0], p_Sat[1], p_Sat[2], 
  p_Hue[0], p_Hue[1], p_Hue[2], p_Hue[3], p_Hue[4], p_Hue[5], p_Hue[6], p_Hue[7], p_Hue[8], p_Hue[9], p_Hue[10], p_Hue[11], p_Hue[12],
  p_Hue[13], p_Hue[14], p_Luma[0], p_Luma[1], p_Luma[2], p_Input, p_Output);
}
