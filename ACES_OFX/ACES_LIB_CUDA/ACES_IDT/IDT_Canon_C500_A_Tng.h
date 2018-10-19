#ifndef __IDT_CANON_C500_A_TNG_H_INCLUDED__
#define __IDT_CANON_C500_A_TNG_H_INCLUDED__

/* ********************************************* */
//
// ACES IDT for Canon EOS C500 (Type A)
//
// Version: 1.0 2013/01/07
// Copyright(c) 2013 Canon Inc. All Rights Reserved.

//
// Camera     : EOS C500
// Output     : 3G-SDI 1 and 2, Cinema RAW(.rmf) file, Mon.1 and 2
// Illuminant : Tungsten (ISO 7589 Studio Tungsten)
//
//
// [ NOTE ]                          
//
// +This IDT is defined for images those were shot under
//  illumination sources with low color temperature like Tungsten. 
//
// +This IDT can be applied for the images from 3G-SDI 1 / 2,
//  Cinema RAW(.rmf) file or Mon.1 / 2.
//
// +Different IDT should be refered for MXF file and other outputs
//  such as HD-SDI / HDMI.
// 

__device__ inline float3 IDT_Canon_C500_A_Tng( float3 In) 
{
	// CodeValue to IRE, ire = (cv-64)/(940-64)
	float3 CLogIRE;
	CLogIRE.x = (In.x * 1023 - 64) / 876;
	CLogIRE.y = (In.y * 1023 - 64) / 876;
	CLogIRE.z = (In.z * 1023 - 64) / 876;

	// Canon-Log to linear 
	float3 lin;
	lin.x = CanonLog_to_linear( CLogIRE.x);
	lin.y = CanonLog_to_linear( CLogIRE.y);
	lin.z = CanonLog_to_linear( CLogIRE.z);
	
	// ACES conversion matrix
	float3 aces;

	aces.x = 0.566996399 * lin.x +0.365079418 * lin.y + 0.067924183 * lin.z;
	aces.y = 0.070901044 * lin.x +0.880331008 * lin.y + 0.048767948 * lin.z;
	aces.z = 0.073013542 * lin.x -0.066540862 * lin.y + 0.99352732 * lin.z;

	return aces;
}

#endif