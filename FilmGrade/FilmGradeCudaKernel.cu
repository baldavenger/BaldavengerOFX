__global__ void FilmGradeKernel(int p_Width, int p_Height, float p_ExpR, float p_ExpG, float p_ExpB, 
    float p_ContR, float p_ContG, float p_ContB, float p_SatR, float p_SatG, float p_SatB, 
    float p_ShadR, float p_ShadG, float p_ShadB, float p_MidR, float p_MidG, float p_MidB, 
    float p_HighR, float p_HighG, float p_HighB, float p_ShadP, float p_HighP, float p_ContP, 
    float p_DisplayA, float p_DisplayB, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;
       
       float e = 2.718281828459045;
       float pie = 3.141592653589793;
       
	   float width = p_Width;
       float height = p_Height;    	
            	  
	   float Red = p_Input[index + 0];
	   float Green = p_Input[index + 1];
	   float Blue = p_Input[index + 2];
	   
	   float expR = Red + p_ExpR/100.0f;
	   float expG = Green + p_ExpG/100.0f;
	   float expB = Blue + p_ExpB/100.0f;
	   
	   float expr1 = (p_ShadP / 2.0f) - (1.0f - p_HighP)/4.0f;
	   float expr2 = (1.0f - (1.0f - p_HighP)/2.0f) + (p_ShadP / 4.0f);
	   float expr3R = (expR - expr1) / (expr2 - expr1);
	   float expr3G = (expG - expr1) / (expr2 - expr1);
	   float expr3B = (expB - expr1) / (expr2 - expr1);
	   float expr4 =  p_ContP < 0.5f ? 0.5f - (0.5f - p_ContP)/2.0f : 0.5f + (p_ContP - 0.5f)/2.0f;
	   float expr5R = expr3R > expr4 ? (expr3R - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3R /(2.0f*expr4);
	   float expr5G = expr3G > expr4 ? (expr3G - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3G /(2.0f*expr4);
	   float expr5B = expr3B > expr4 ? (expr3B - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3B /(2.0f*expr4);
	   float expr6R = (((sin(2.0f * pie * (expr5R -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidR*4.0f) + expr3R;
	   float expr6G = (((sin(2.0f * pie * (expr5G -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidG*4.0f) + expr3G;
	   float expr6B = (((sin(2.0f * pie * (expr5B -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidB*4.0f) + expr3B;
	   float midR = expR >= expr1 && expR <= expr2 ? expr6R * (expr2 - expr1) + expr1 : expR;
	   float midG = expG >= expr1 && expG <= expr2 ? expr6G * (expr2 - expr1) + expr1 : expG;
	   float midB = expB >= expr1 && expB <= expr2 ? expr6B * (expr2 - expr1) + expr1 : expB;
		
	   float shadupR1 = midR > 0.0f ? 2.0f * (midR/p_ShadP) - log((midR/p_ShadP) * (e * p_ShadR * 2.0f) + 1.0f)/log(e * p_ShadR * 2.0f + 1.0f) : midR;
	   float shadupR = midR < p_ShadP && p_ShadR > 0.0f ? (shadupR1 + p_ShadR * (1.0f - shadupR1)) * p_ShadP : midR;
	   float shadupG1 = midG > 0.0f ? 2.0f * (midG/p_ShadP) - log((midG/p_ShadP) * (e * p_ShadG * 2.0f) + 1.0f)/log(e * p_ShadG * 2.0f + 1.0f) : midG;
	   float shadupG = midG < p_ShadP && p_ShadG > 0.0f ? (shadupG1 + p_ShadG * (1.0f - shadupG1)) * p_ShadP : midG;
	   float shadupB1 = midB > 0.0f ? 2.0f * (midB/p_ShadP) - log((midB/p_ShadP) * (e * p_ShadB * 2.0f) + 1.0f)/log(e * p_ShadB * 2.0f + 1.0f) : midB;
	   float shadupB = midB < p_ShadP && p_ShadB > 0.0f ? (shadupB1 + p_ShadB * (1.0f - shadupB1)) * p_ShadP : midB;
	   
	   float shaddownR1 = shadupR/p_ShadP + p_ShadR*2 * (1.0f - shadupR/p_ShadP);
	   float shaddownR = shadupR < p_ShadP && p_ShadR < 0.0f ? (shaddownR1 >= 0.0f ? log(shaddownR1 * (e * p_ShadR * -2.0f) + 1.0f)/log(e * p_ShadR * -2.0f + 1.0f) : shaddownR1) * p_ShadP : shadupR;
	   float shaddownG1 = shadupG/p_ShadP + p_ShadG*2 * (1.0f - shadupG/p_ShadP);
	   float shaddownG = shadupG < p_ShadP && p_ShadG < 0.0f ? (shaddownG1 >= 0.0f ? log(shaddownG1 * (e * p_ShadG * -2.0f) + 1.0f)/log(e * p_ShadG * -2.0f + 1.0f) : shaddownG1) * p_ShadP : shadupG;
	   float shaddownB1 = shadupB/p_ShadP + p_ShadB*2 * (1.0f - shadupB/p_ShadP);
	   float shaddownB = shadupB < p_ShadP && p_ShadB < 0.0f ? (shaddownB1 >= 0.0f ? log(shaddownB1 * (e * p_ShadB * -2.0f) + 1.0f)/log(e * p_ShadB * -2.0f + 1.0f) : shaddownB1) * p_ShadP : shadupB;
	   
	   float highupR1 = ((shaddownR - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighR * 2.0f));
	   float highupR = shaddownR > p_HighP && p_HighP < 1.0f && p_HighR > 0.0f ? (2.0f * highupR1 - log(highupR1 * e * p_HighR + 1.0f)/log(e * p_HighR + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddownR;
	   float highupG1 = ((shaddownG - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighG * 2.0f));
	   float highupG = shaddownG > p_HighP && p_HighP < 1.0f && p_HighG > 0.0f ? (2.0f * highupG1 - log(highupG1 * e * p_HighG + 1.0f)/log(e * p_HighG + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddownG;
	   float highupB1 = ((shaddownB - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighB * 2.0f));
	   float highupB = shaddownB > p_HighP && p_HighP < 1.0f && p_HighB > 0.0f ? (2.0f * highupB1 - log(highupB1 * e * p_HighB + 1.0f)/log(e * p_HighB + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddownB;
	   
	   float highdownR1 = (highupR - p_HighP) / (1.0f - p_HighP);
	   float highdownR = highupR > p_HighP && p_HighP < 1.0f && p_HighR < 0.0f ? log(highdownR1 * (e * p_HighR * -2.0f) + 1.0f)/log(e * p_HighR * -2.0f + 1.0f) * (1.0f + p_HighR) * (1.0f - p_HighP) + p_HighP : highupR;
	   float highdownG1 = (highupG - p_HighP) / (1.0f - p_HighP);
	   float highdownG = highupG > p_HighP && p_HighP < 1.0f && p_HighG < 0.0f ? log(highdownG1 * (e * p_HighG * -2.0f) + 1.0f)/log(e * p_HighG * -2.0f + 1.0f) * (1.0f + p_HighG) * (1.0f - p_HighP) + p_HighP : highupG;
	   float highdownB1 = (highupB - p_HighP) / (1.0f - p_HighP);
	   float highdownB = highupB > p_HighP && p_HighP < 1.0f && p_HighB < 0.0f ? log(highdownB1 * (e * p_HighB * -2.0f) + 1.0f)/log(e * p_HighB * -2.0f + 1.0f) * (1.0f + p_HighB) * (1.0f - p_HighP) + p_HighP : highupB;
	   
	   float contR = (highdownR - p_ContP) * p_ContR + p_ContP;
	   float contG = (highdownG - p_ContP) * p_ContG + p_ContP;
	   float contB = (highdownB - p_ContP) * p_ContB + p_ContP;
	   
	   float luma = contR * 0.2126f + contG * 0.7152f + contB * 0.0722f;
	   float satR = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * luma + contR * p_SatR;
	   float satG = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * luma + contG * p_SatG;
	   float satB = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * luma + contB * p_SatB;
	   
	   float DexpR = (x / width) + p_ExpR/100.0f;
	   float DexpG = (x / width) + p_ExpG/100.0f;
	   float DexpB = (x / width) + p_ExpB/100.0f;
		
	   float Dexpr1 = (p_ShadP / 2.0f) - (1.0f - p_HighP)/4.0f;
	   float Dexpr2 = (1.0f - (1.0f - p_HighP)/2.0f) + (p_ShadP / 4.0f);
	   float Dexpr3R = (DexpR - Dexpr1) / (Dexpr2 - Dexpr1);
	   float Dexpr3G = (DexpG - Dexpr1) / (Dexpr2 - Dexpr1);
	   float Dexpr3B = (DexpB - Dexpr1) / (Dexpr2 - Dexpr1);
	   float Dexpr4 =  p_ContP < 0.5f ? 0.5f - (0.5f - p_ContP)/2.0f : 0.5f + (p_ContP - 0.5f)/2.0f;
	   float Dexpr5R = Dexpr3R > Dexpr4 ? (Dexpr3R - Dexpr4) / (2.0f - 2.0f*Dexpr4) + 0.5f : Dexpr3R /(2.0f*Dexpr4);
	   float Dexpr5G = Dexpr3G > Dexpr4 ? (Dexpr3G - Dexpr4) / (2.0f - 2.0f*Dexpr4) + 0.5f : Dexpr3G /(2.0f*Dexpr4);
	   float Dexpr5B = Dexpr3B > Dexpr4 ? (Dexpr3B - Dexpr4) / (2.0f - 2.0f*Dexpr4) + 0.5f : Dexpr3B /(2.0f*Dexpr4);
	   float Dexpr6R = (((sin(2.0f * pie * (Dexpr5R -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidR*4.0f) + Dexpr3R;
	   float Dexpr6G = (((sin(2.0f * pie * (Dexpr5G -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidG*4.0f) + Dexpr3G;
	   float Dexpr6B = (((sin(2.0f * pie * (Dexpr5B -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidB*4.0f) + Dexpr3B;
	   float DmidR = DexpR >= Dexpr1 && DexpR <= Dexpr2 ? Dexpr6R * (Dexpr2 - Dexpr1) + Dexpr1 : DexpR;
	   float DmidG = DexpG >= Dexpr1 && DexpG <= Dexpr2 ? Dexpr6G * (Dexpr2 - Dexpr1) + Dexpr1 : DexpG;
	   float DmidB = DexpB >= Dexpr1 && DexpB <= Dexpr2 ? Dexpr6B * (Dexpr2 - Dexpr1) + Dexpr1 : DexpB;
	   
	   float DshadupR1 = DmidR > 0.0f ? 2.0f * (DmidR/p_ShadP) - log((DmidR/p_ShadP) * (e * p_ShadR * 2.0f) + 1.0f)/log(e * p_ShadR * 2.0f + 1.0f) : DmidR;
	   float DshadupR = DmidR < p_ShadP && p_ShadR > 0.0f ? (DshadupR1 + p_ShadR * (1.0f - DshadupR1)) * p_ShadP : DmidR;
	   float DshadupG1 = DmidG > 0.0f ? 2.0f * (DmidG/p_ShadP) - log((DmidG/p_ShadP) * (e * p_ShadG * 2.0f) + 1.0f)/log(e * p_ShadG * 2.0f + 1.0f) : DmidG;
	   float DshadupG = DmidG < p_ShadP && p_ShadG > 0.0f ? (DshadupG1 + p_ShadG * (1.0f - DshadupG1)) * p_ShadP : DmidG;
	   float DshadupB1 = DmidB > 0.0f ? 2.0f * (DmidB/p_ShadP) - log((DmidB/p_ShadP) * (e * p_ShadB * 2.0f) + 1.0f)/log(e * p_ShadB * 2.0f + 1.0f) : DmidB;
	   float DshadupB = DmidB < p_ShadP && p_ShadB > 0.0f ? (DshadupB1 + p_ShadB * (1.0f - DshadupB1)) * p_ShadP : DmidB;
	   
	   float DshaddownR1 = (DshadupR/p_ShadP) + (p_ShadR * 2.0f * (1.0f - DshadupR/p_ShadP));
	   float DshaddownR = DshadupR < p_ShadP && p_ShadR < 0.0f ? (DshaddownR1 >= 0.0f ? log(DshaddownR1 * (e * p_ShadR * -2.0f) + 1.0f)/log(e * p_ShadR * -2.0f + 1.0f) : DshaddownR1) * p_ShadP : DshadupR;
	   float DshaddownG1 = (DshadupG/p_ShadP) + (p_ShadG * 2.0f * (1.0f - DshadupG/p_ShadP));
	   float DshaddownG = DshadupG < p_ShadP && p_ShadG < 0.0f ? (DshaddownG1 >= 0.0f ? log(DshaddownG1 * (e * p_ShadG * -2.0f) + 1.0f)/log(e * p_ShadG * -2.0f + 1.0f) : DshaddownG1) * p_ShadP : DshadupG;
	   float DshaddownB1 = (DshadupB/p_ShadP) + (p_ShadB * 2.0f * (1.0f - DshadupB/p_ShadP));
	   float DshaddownB = DshadupB < p_ShadP && p_ShadB < 0.0f ? (DshaddownB1 >= 0.0f ? log(DshaddownB1 * (e * p_ShadB * -2.0f) + 1.0f)/log(e * p_ShadB * -2.0f + 1.0f) : DshaddownB1) * p_ShadP : DshadupB;
	   
	   float DhighupR1 = ((DshaddownR - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighR * 2.0f));
	   float DhighupR = DshaddownR > p_HighP && p_HighP < 1.0f && p_HighR > 0.0f ? (2.0f * DhighupR1 - log(DhighupR1 * e * p_HighR + 1.0f)/log(e * p_HighR + 1.0f)) * (1.0f - p_HighP) + p_HighP : DshaddownR;
	   float DhighupG1 = ((DshaddownG - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighG * 2.0f));
	   float DhighupG = DshaddownG > p_HighP && p_HighP < 1.0f && p_HighG > 0.0f ? (2.0f * DhighupG1 - log(DhighupG1 * e * p_HighG + 1.0f)/log(e * p_HighG + 1.0f)) * (1.0f - p_HighP) + p_HighP : DshaddownG;
	   float DhighupB1 = ((DshaddownB - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighB * 2.0f));
	   float DhighupB = DshaddownB > p_HighP && p_HighP < 1.0f && p_HighB > 0.0f ? (2.0f * DhighupB1 - log(DhighupB1 * e * p_HighB + 1.0f)/log(e * p_HighB + 1.0f)) * (1.0f - p_HighP) + p_HighP : DshaddownB;
	   
	   float DhighdownR1 = (DhighupR - p_HighP) / (1.0f - p_HighP);
	   float DhighdownR = DhighupR > p_HighP && p_HighP < 1.0f && p_HighR < 0.0f ? log(DhighdownR1 * (e * p_HighR * -2.0f) + 1.0f)/log(e * p_HighR * -2.0f + 1.0f) * (1.0f + p_HighR) * (1.0f - p_HighP) + p_HighP  : DhighupR;
	   float DhighdownG1 = (DhighupG - p_HighP) / (1.0f - p_HighP);
	   float DhighdownG = DhighupG > p_HighP && p_HighP < 1.0f && p_HighG < 0.0f ? log(DhighdownG1 * (e * p_HighG * -2.0f) + 1.0f)/log(e * p_HighG * -2.0f + 1.0f) * (1.0f + p_HighG) * (1.0f - p_HighP) + p_HighP  : DhighupG;
	   float DhighdownB1 = (DhighupB - p_HighP) / (1.0f - p_HighP);
	   float DhighdownB = DhighupB > p_HighP && p_HighP < 1.0f && p_HighB < 0.0f ? log(DhighdownB1 * (e * p_HighB * -2.0f) + 1.0f)/log(e * p_HighB * -2.0f + 1.0f) * (1.0f + p_HighB) * (1.0f - p_HighP) + p_HighP  : DhighupB;
	   
	   float DcontR = (DhighdownR - p_ContP) * p_ContR + p_ContP;
	   float DcontG = (DhighdownG - p_ContP) * p_ContG + p_ContP;
	   float DcontB = (DhighdownB - p_ContP) * p_ContB + p_ContP;
	   
	   float Dluma = DcontR * 0.2126f + DcontG * 0.7152f + DcontB * 0.0722f;
	   float DsatR = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * Dluma + DcontR * p_SatR;
	   float DsatG = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * Dluma + DcontG * p_SatG;
	   float DsatB = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * Dluma + DcontB * p_SatB;
	   
	   float overlayR = y/(height) >= p_ShadP && y/(height) <= p_ShadP + 0.005f ? (fmodf(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : DsatR >= (y - 5)/(height) && DsatR <= (y + 5)/(height) ? 1.0f : 0.0f;
	   float overlayG = y/(height) >= p_HighP && y/(height) <= p_HighP + 0.005f ? (fmodf(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : DsatG >= (y - 5)/(height) && DsatG <= (y + 5)/(height) ? 1.0f : 0.0f;
	   float overlayB = y/(height) >= p_ContP && y/(height) <= p_ContP + 0.005f ? (fmodf(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : DsatB >= (y - 5)/(height) && DsatB <= (y + 5)/(height) ? 1.0f : 0.0f;
	   
	   float outR = p_DisplayA == 1.0f && p_DisplayB == 1.0f ? (overlayR == 0.0f ? satR : overlayR) : p_DisplayA == 1.0f ? overlayR : satR;
       float outG = p_DisplayA == 1.0f && p_DisplayB == 1.0f ? (overlayG == 0.0f ? satG : overlayG) : p_DisplayA == 1.0f ? overlayG : satG;
       float outB = p_DisplayA == 1.0f && p_DisplayB == 1.0f ? (overlayB == 0.0f ? satB : overlayB) : p_DisplayA == 1.0f ? overlayB : satB;
				 			
       p_Output[index + 0] = outR;
       p_Output[index + 1] = outG;
       p_Output[index + 2] = outB;
       p_Output[index + 3] = p_Input[index + 3];
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* p_Exp, float* p_Cont, float* p_Sat, 
float* p_Shad, float* p_Mid, float* p_High, float* p_Pivot, float* p_Display, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    FilmGradeKernel<<<blocks, threads>>>(p_Width, p_Height, p_Exp[0], p_Exp[1], p_Exp[2], p_Cont[0], p_Cont[1], p_Cont[2], 
    p_Sat[0], p_Sat[1], p_Sat[2], p_Shad[0], p_Shad[1], p_Shad[2], p_Mid[0], p_Mid[1], p_Mid[2], p_High[0], p_High[1], p_High[2], 
    p_Pivot[0], p_Pivot[1], p_Pivot[2], p_Display[0], p_Display[1], p_Input, p_Output);
}
