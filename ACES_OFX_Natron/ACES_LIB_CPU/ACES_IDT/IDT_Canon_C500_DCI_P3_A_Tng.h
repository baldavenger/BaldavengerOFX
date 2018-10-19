#ifndef __IDT_CANON_C500_DCI_P3_A_TNG_H_INCLUDED__
#define __IDT_CANON_C500_DCI_P3_A_TNG_H_INCLUDED__
//
// ACES IDT for Canon EOS C500 (Type C)
//
// IDT for C500 DCI-P3+ Gamut
// Version: 1.1  2013/11/11
// Copyright(c) 2013 Canon Inc. All Rights Reserved.
//
// Camera     : EOS C500
// Output     : 3G-SDI 1 and 2, Cinema RAW(.rmf) file, Mon.1 and 2
// Illuminant : Tungsten (ISO 7589 Studio Tungsten)
// Color Gamut: DCI-P3+
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

inline float3 IDT_Canon_C500_DCI_P3_A_Tng( float3 In)
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

	aces.x =  0.650279125f * lin.x + 0.253880169f * lin.y + 0.095840706f * lin.z;
	aces.y = -0.026137986f * lin.x + 1.017900530f * lin.y + 0.008237456f * lin.z;
	aces.z =  0.007757558f * lin.x - 0.063081669f * lin.y + 1.055324110f * lin.z;

	return aces;
}

#endif