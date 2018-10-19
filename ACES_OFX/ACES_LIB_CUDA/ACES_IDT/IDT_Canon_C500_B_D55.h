#ifndef __IDT_CANON_C500_B_D55_H_INCLUDED__
#define __IDT_CANON_C500_B_D55_H_INCLUDED__

//
// ACES IDT for Canon EOS C500 (Type B)
//
// Version: 1.0 2013/04/18
// Copyright(c) 2013 Canon Inc. All Rights Reserved.
//
// Camera     : EOS C500
// Output     : MXF file and HD-SDI / HDMI.
// Illuminant : Daylight (CIE Illuminant D55)
//
//
// [ NOTE ]                          
//
// +This IDT is defined for images those were shot under daylight
//  and general illumination sources except Tungsten.
//
// +This IDT can be applied for the images from MXF file, HDMI/HD-SDI.
//
// +Different IDT should be refered for 3G-SDI 1 and 2, Cinema RAW(.rmf) file, Mon.1 and 2
//  
// 

__device__ inline float3 IDT_Canon_C500_B_D55( float3 In)
{

	// CodeValue to IRE, ire = (cv-64)/(940-64)
	float iR, iG, iB;
	iR = (In.x * 1023.0f - 64.0f) / 876.0f;
	iG = (In.y * 1023.0f - 64.0f) / 876.0f;
	iB = (In.z * 1023.0f - 64.0f) / 876.0f;

	// ACES conversion matrix1 (3x19)
	float3 pmtx;

	pmtx.x =  1.08190037262167	* iR 		-0.180298701368782	* iG 		+0.0983983287471069	* iB 
		  +1.9458545364518	* iR*iG 	-0.509539936937375	* iG*iB 	-0.47489567735516	* iB*iR 
		  -0.778086752197068 	* iR*iR  	-0.7412266070049	* iG*iG 	+0.557894437042701	* iB*iB
		  -3.27787395719078	* iR*iR*iG 	+0.254878417638717	* iR*iR*iB 	+3.45581530576474	* iR*iG*iG
		  +0.335471713974739	* iR*iG*iB 	-0.43352125478476	* iR*iB*iB 	-1.65050137344141	* iG*iG*iB +1.46581418175682 * iG*iB*iB
		  +0.944646566605676	* iR*iR*iR 	-0.723653099155881	* iG*iG*iG 	-0.371076501167857	* iB*iB*iB;

	pmtx.y = -0.00858997792576314	* iR 		+1.00673740119621	* iG 		+0.00185257672955608	* iB
		  +0.0848736138296452	* iR*iG 	+0.347626906448902	* iG*iB 	+0.0020230274463939	* iB*iR 
		  -0.0790508414091524	* iR*iR  	-0.179497582958716	* iG*iG 	-0.175975123357072	* iB*iB
		  +2.30205579706951	* iR*iR*iG	-0.627257613385219	* iR*iR*iB 	-2.90795250918851	* iR*iG*iG
		  +1.37002437502321	* iR*iG*iB 	-0.108668158565563	* iR*iB*iB 	-2.21150552827555	* iG*iG*iB + 1.53315057595445 * iG*iB*iB
		  -0.543188706699505	* iR*iR*iR 	+1.63793038490376	* iG*iG*iG 	-0.444588616836587	* iB*iB*iB;

	pmtx.z = 0.12696639806511	* iR 		-0.011891441127869	* iG 		+0.884925043062759	* iB
		  +1.34780279822258	* iR*iG		+1.03647352257365	* iG*iB		+0.459113289955922	* iB*iR
		  -0.878157422295268	* iR*iR  	-1.3066278750436	* iG*iG 	-0.658604313413283	* iB*iB
		  -1.4444077996703	* iR*iR*iG 	+0.556676588785173	* iR*iR*iB 	+2.18798497054968	* iR*iG*iG
		  -1.43030768398665	* iR*iG*iB 	-0.0388323570817641	* iR*iB*iB 	+2.63698573112453	* iG*iG*iB -1.66598882056039 * iG*iB*iB
		  +0.33450249360103	* iR*iR*iR 	-1.65856930730901	* iG*iG*iG 	+0.521956184547685	* iB*iB*iB;


	// Canon-Log to linear 
	float3 lin;

	lin.x = Clip(CanonLog_to_linear( pmtx.x));
	lin.y = Clip(CanonLog_to_linear( pmtx.y));
	lin.z = Clip(CanonLog_to_linear( pmtx.z));

	// ACES conversion matrix2
	float3 aces;

	aces.x =  Clip(0.561538969f * lin.x + 0.402060105f * lin.y + 0.036400926f * lin.z);
	aces.y =  Clip(0.092739623f * lin.x + 0.924121198f * lin.y - 0.016860821f * lin.z);
	aces.z =  Clip(0.084812961f * lin.x + 0.006373835f * lin.y + 0.908813204f * lin.z);

	return aces;
}

#endif