#ifndef __IDT_CANON_C300MK2_CANONLOG2_BT2020_B_D55_H_INCLUDED__
#define __IDT_CANON_C300MK2_CANONLOG2_BT2020_B_D55_H_INCLUDED__

//
// ACES Input Transform for EOS C300 Mark II (Type B)
//
// Input Transform for EOS C300 Mark II BT.2020 / Canon Log 2
// ACES Version: 1.0
// Version: 1.1  2016/9/8
// Copyright(c) 2016 Canon Inc. All Rights Reserved.
//
// Camera      : EOS C300 Mark II
// Illuminant  : Daylight (CIE Illuminant D55)
// Color Gamut : BT.2020
// Gamma       : Canon Log 2
// Color Matrix: Neutral
//
// [ NOTE ]
//
// +This Input Transform is defined for images those were shot under daylight
//  and general illumination sources except Tungsten.
//

__device__ inline float3 IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55( float3 In)
{

	// CodeValue to IRE, ire = (cv-64)/(940-64)
	float3 CLogIRE;
	CLogIRE.x = (In.x * 1023.0f - 64.0f) / 876.0f;
	CLogIRE.y = (In.y * 1023.0f - 64.0f) / 876.0f;
	CLogIRE.z = (In.z * 1023.0f - 64.0f) / 876.0f;

	// Canon-Log to linear 
	float3 lin;
	lin.x = 0.9f * CanonLog2_to_linear( CLogIRE.x);
	lin.y = 0.9f * CanonLog2_to_linear( CLogIRE.y);
	lin.z = 0.9f * CanonLog2_to_linear( CLogIRE.z);
	
	// ACES conversion matrix
	float3 aces;

	aces.x =  0.678891151f * lin.x + 0.158868422f * lin.y + 0.162240427f * lin.z;
	aces.y =  0.045570831f * lin.x + 0.860712772f * lin.y + 0.093716397f * lin.z;
	aces.z = -0.000485710f * lin.x + 0.025060196f * lin.y + 0.975425515f * lin.z;

	return aces;
}

#endif