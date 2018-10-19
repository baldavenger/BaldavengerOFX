#ifndef __IDT_SONY_SLOG2_SGAMUT_TUNGSTEN_10_H_INCLUDED__
#define __IDT_SONY_SLOG2_SGAMUT_TUNGSTEN_10_H_INCLUDED__

//
// IDT for Sony Cameras - 10 bits - Tungsten (3200K or 4300K)
// Provided by Sony Electronics Corp.
//


inline float3 IDT_Sony_SLog2_SGamut_Tungsten_10( float3 In)
{

mat3 SGAMUT_TUNG_TO_ACES_MTX = { { 1.0110238740f,  0.1011994504f,  0.0600766530f},
  								{ -0.1362526051f,  0.9562196265f, -0.1010185315f},
								{ 0.1252287310f, -0.0574190769f,  1.0409418785f} };
							
const float B = 64.0f;
const float AB = 90.0f;
const float W = 940.0f;

// Prepare input values based on application bit depth handling
float3 SLog;
SLog.x = In.x * 1023.0f;
SLog.y = In.y * 1023.0f;
SLog.z = In.z * 1023.0f;

// 10-bit Sony S-log to linear S-gamut
float3 lin;
lin.x = SLog2_to_lin( SLog.x, B, AB, W);
lin.y = SLog2_to_lin( SLog.y, B, AB, W);
lin.z = SLog2_to_lin( SLog.z, B, AB, W);

// S-Gamut to ACES matrix
float3 aces = mult_f3_f33( lin, SGAMUT_TUNG_TO_ACES_MTX);

return aces;

}

#endif