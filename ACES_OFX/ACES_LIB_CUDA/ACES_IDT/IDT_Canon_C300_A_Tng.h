#ifndef __IDT_CANON_C300_A_TNG_H_INCLUDED__
#define __IDT_CANON_C300_A_TNG_H_INCLUDED__

//
// ACES IDT for Canon EOS C300 (Type A)
//
// Version: 1.0 2013/04/18
// Copyright(c) 2013 Canon Inc. All Rights Reserved.
//
// Camera     : EOS C300
// Output     : MXF file and HD-SDI / HDMI.
// Illuminant : Tungsten (ISO 7589 Studio Tungsten)
//
//
// [ NOTE ]                          
//
// +This IDT is defined for images those were shot under
//  illumination sources with low color temperature like Tungsten.
//
// +This IDT can be applied for the images from MXF file, HDMI/HD-SDI.
//

__device__ inline float3 IDT_Canon_C300_A_Tng( float3 In)
{

	// CodeValue to IRE, ire = (cv-64)/(940-64)
	float iR, iG, iB;
	iR = (In.x * 1023.0f - 64.0f) / 876.0f;
	iG = (In.y * 1023.0f - 64.0f) / 876.0f;
	iB = (In.z * 1023.0f - 64.0f) / 876.0f;

	// ACES conversion matrix1 (3x19)
	float3 pmtx;

	pmtx.x =  0.963803004454899	* iR 		-0.160722202570655	* iG 		+0.196919198115756	* iB 
		  +2.03444685639819	* iR*iG 	-0.442676931451021 	* iG*iB 	-0.407983781537509	* iB*iR 
		  -0.640703323129254 	* iR*iR  	-0.860242798247848	* iG*iG 	+0.317159977967446	* iB*iB
		  -4.80567080102966	* iR*iR*iG 	+0.27118370397567	* iR*iR*iB 	+5.1069005049557	* iR*iG*iG
		  +0.340895816920585	* iR*iG*iB 	-0.486941738507862	* iR*iB*iB 	-2.23737935753692	* iG*iG*iB +1.96647555251297 * iG*iB*iB
		  +1.30204051766243	* iR*iR*iR 	-1.06503117628554	* iG*iG*iG 	-0.392473022667378	* iB*iB*iB;

	pmtx.y =  -0.0421935892309314	* iR 		+1.04845959175183	* iG 		-0.00626600252090315	* iB
		  -0.106438896887216	* iR*iG 	+0.362908621470781	* iG*iB 	+0.118070700472261	* iB*iR 
		  +0.0193542539838734	* iR*iR  	-0.156083029543267	* iG*iG 	-0.237811649496433	* iB*iB
		  +1.67916420582198	* iR*iR*iG	-0.632835327167897	* iR*iR*iB 	-1.95984471387461	* iR*iG*iG
		  +0.953221464562814	* iR*iG*iB 	+0.0599085176294623	* iR*iB*iB 	-1.66452046236246	* iG*iG*iB +1.14041188349761 * iG*iB*iB
		  -0.387552623550308	* iR*iR*iR 	+1.14820099685512	* iG*iG*iG 	-0.336153941411709	* iB*iB*iB;

	pmtx.z = 0.170295033135028	* iR 		-0.0682984448537245	* iG 		+0.898003411718697	* iB
		  +1.22106821992399	* iR*iG		+1.60194865922925	* iG*iB		+0.377599191137124	* iB*iR
		  -0.825781428487531	* iR*iR  	-1.44590868076749	* iG*iG 	-0.928925961035344	* iB*iB
		  -0.838548997455852	* iR*iR*iG 	+0.75809397217116	* iR*iR*iB 	+1.32966795243196	* iR*iG*iG
		  -1.20021905668355	* iR*iG*iB 	-0.254838995845129	* iR*iB*iB 	+2.33232411639308	* iG*iG*iB -1.86381505762773 * iG*iB*iB
		  +0.111576038956423	* iR*iR*iR 	-1.12593315849766	* iG*iG*iG 	+0.751693186157287	* iB*iB*iB;


	// Canon-Log to linear 
	float3 lin;

	lin.x = Clip(CanonLog_to_linear( pmtx.x));
	lin.y = Clip(CanonLog_to_linear( pmtx.y));
	lin.z = Clip(CanonLog_to_linear( pmtx.z));
	
	// ACES conversion matrix2
	float3 aces;

	aces.x =  Clip(0.566996399   * lin.x  +0.365079418 * lin.y + 0.067924183 * lin.z);
	aces.y =  Clip(0.070901044   * lin.x  +0.880331008 * lin.y + 0.048767948 * lin.z);
	aces.z =  Clip(0.073013542   * lin.x  -0.066540862 * lin.y + 0.99352732  * lin.z);

	return aces;
}

#endif