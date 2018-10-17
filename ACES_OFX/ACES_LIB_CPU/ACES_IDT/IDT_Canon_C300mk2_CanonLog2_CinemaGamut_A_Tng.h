#ifndef __IDT_CANON_C300MK2_CANONLOG2_CINEMAGAMUT_A_TNG_H_INCLUDED__
#define __IDT_CANON_C300MK2_CANONLOG2_CINEMAGAMUT_A_TNG_H_INCLUDED__

//
// ACES Input Transform for EOS C300 Mark II (Type A)
//
// Input Transform for EOS C300 Mark II Cinema Gamut / Canon Log 2
// ACES Version: 1.0
// Version: 1.1  2016/9/8
// Copyright(c) 2016 Canon Inc. All Rights Reserved.
//
// Camera      : EOS C300 Mark II
// Illuminant  : Tungsten (ISO 7589 Studio Tungsten)
// Color Gamut : Cinema Gamut
// Gamma       : Canon Log 2
// Color Matrix: Neutral
//
// [ NOTE ]
//
// +This Input Transform is defined for images those were shot under
//  illumination sources with low color temperature like Tungsten. 
//

inline float3 IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng( float3 In)
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

	aces.x =  0.817416293f * lin.x + 0.090755698f * lin.y + 0.091828009f * lin.z;
	aces.y = -0.035361374f * lin.x + 1.065690585f * lin.y - 0.030329211f * lin.z;
	aces.z =  0.010390366f * lin.x - 0.299271107f * lin.y + 1.288880741f * lin.z;

	return aces;
}

#endif