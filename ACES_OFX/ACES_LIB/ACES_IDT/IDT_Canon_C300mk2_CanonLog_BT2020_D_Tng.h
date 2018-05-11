#ifndef __IDT_CANON_C300MK2_CANONLOG_BT2020_D_TNG_H_INCLUDED__
#define __IDT_CANON_C300MK2_CANONLOG_BT2020_D_TNG_H_INCLUDED__

//
// ACES Input Transform for EOS C300 Mark II (Type D)
//
// Input Transform for EOS C300 Mark II BT.2020 / Canon Log
// ACES Version: 1.0
// Version: 1.1  2016/9/8
// Copyright(c) 2016 Canon Inc. All Rights Reserved.
//
// Camera      : EOS C300 Mark II
// Illuminant  : Tungsten (ISO 7589 Studio Tungsten)
// Color Gamut : BT.2020
// Gamma       : Canon Log
// Color Matrix: Neutral
//
// [ NOTE ]
//
// +This Input Transform is defined for images those were shot under
//  illumination sources with low color temperature like Tungsten. 
//

__device__ inline float3 IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng( float3 In)
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

	aces.x = 0.724488568f * lin.x + 0.115140904f * lin.y + 0.160370529f * lin.z;
	aces.y = 0.010659276f * lin.x + 0.839605344f * lin.y + 0.149735380f * lin.z;
	aces.z = 0.014560161f * lin.x - 0.028562057f * lin.y + 1.014001897f * lin.z;

	return aces;
}

#endif