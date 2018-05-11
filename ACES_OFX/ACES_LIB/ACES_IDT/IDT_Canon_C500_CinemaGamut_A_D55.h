#ifndef __IDT_CANON_C500_CINEMAGAMUT_A_D55_H_INCLUDED__
#define __IDT_CANON_C500_CINEMAGAMUT_A_D55_H_INCLUDED__

//
// ACES IDT for Canon EOS C500 (Type D)
//
// IDT for C500 Cinema Gamut
// Version: 1.1  2013/11/11
// Copyright(c) 2013 Canon Inc. All Rights Reserved.
//
// Camera     : EOS C500
// Output     : 3G-SDI 1 and 2, Cinema RAW(.rmf) file, Mon.1 and 2
// Illuminant : Daylight (CIE Illuminant D55)
// Color Gamut: Cinema Gamut
//
// [ NOTE ]                          
//
// +This IDT is defined for images those were shot under daylight
//  and general illumination sources except Tungsten.
//
// +This IDT can be applied for the images from 3G-SDI 1 / 2,
//  Cinema RAW(.rmf) file or Mon.1 / 2.
//
// +Different IDT should be refered for MXF file and other outputs
//  such as HD-SDI / HDMI.
// 

__device__ inline float3 IDT_Canon_C500_CinemaGamut_A_D55( float3 In)
{

	// CodeValue to IRE, ire = (cv-64)/(940-64)
	float3 CLogIRE;
	CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
	CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
	CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;

	// Canon-Log to linear 
	float3 lin;
	lin.x = 0.9f * CanonLog_to_linear( CLogIRE.x);
	lin.y = 0.9f * CanonLog_to_linear( CLogIRE.y);
	lin.z = 0.9f * CanonLog_to_linear( CLogIRE.z);
	
	// ACES conversion matrix
	float3 aces;

	aces.x =  0.763064455f * lin.x + 0.149021161f * lin.y + 0.087914384f * lin.z;
	aces.y =  0.003657457f * lin.x + 1.10696038f  * lin.y - 0.110617837f * lin.z;
	aces.z = -0.009407794f * lin.x - 0.218383305f * lin.y + 1.227791099f * lin.z;

	return aces;
}

#endif