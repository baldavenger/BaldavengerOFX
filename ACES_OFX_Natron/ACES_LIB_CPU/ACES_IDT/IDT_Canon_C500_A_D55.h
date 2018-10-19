#ifndef __IDT_CANON_C500_A_D55_H_INCLUDED__
#define __IDT_CANON_C500_A_D55_H_INCLUDED__

/* ********************************************* */
//
// ACES IDT for Canon EOS C500 (Type A)
//
// Version: 1.0 2013/01/07
// Copyright(c) 2013 Canon Inc. All Rights Reserved.

/* ********************************************* */
//
// Camera     : EOS C500
// Output     : 3G-SDI 1 and 2, Cinema RAW(.rmf) file, Mon.1 and 2
// Illuminant : Daylight (CIE Illuminant D55)
//
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
/* ********************************************* */


inline float3 IDT_Canon_C500_A_D55( float3 In) 
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

	aces.x = 0.561538969f * lin.x +0.402060105f * lin.y + 0.036400926f * lin.z;
	aces.y = 0.092739623f * lin.x +0.924121198f * lin.y - 0.016860821f * lin.z;
	aces.z = 0.084812961f * lin.x +0.006373835f * lin.y + 0.908813204f * lin.z;

	return aces;
}

#endif