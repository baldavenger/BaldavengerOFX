#ifndef __ACES_IDT_H_INCLUDED__
#define __ACES_IDT_H_INCLUDED__

// Combined IDT support files

#include "ACES_functions.h"
#include "ACES_Utilities_Color.h"
#include "ACES_Transform_Common.h"
#include "ACES_Tonescales.h"
#include "ADX_to_ACES.h"

__device__ inline float3 DolbyPQ_to_Lin( float3 In)
{
  float3 out;
  out.x = ST2084_2_Y( In.x );
  out.y = ST2084_2_Y( In.y );
  out.z = ST2084_2_Y( In.z );
  return out;
}

__device__ inline float3 IDT_Alexa_v3_raw_EI800_CCT65( float3 In)
{
float EI = 800.0f;
float black = 256.0f / 65535.0f;
float exp_factor = 0.18f / (0.01f * (400.0f / EI));

// convert to white-balanced, black-subtracted linear values
float r_lin = (In.x - black) * exp_factor;
float g_lin = (In.y - black) * exp_factor;
float b_lin = (In.z - black) * exp_factor;

// convert to ACES primaries using CCT-dependent matrix
float3 aces;
aces.x = r_lin * 0.809931f + g_lin * 0.162741f + b_lin * 0.027328f;
aces.y = r_lin * 0.083731f + g_lin * 1.108667f + b_lin * -0.192397f;
aces.z = r_lin * 0.044166f + g_lin * -0.272038f + b_lin * 1.227872f;
return aces;
}

__device__ inline float normalizedLogCToRelativeExposure(float x)
{
	if (x > 0.149659f)
		return (powf(10.0f, (x - 0.385537f) / 0.247189f) - 0.052272f) / 5.555556f;
	else
		return (x - 0.092809f) / 5.367650f;
}

__device__ inline float3 IDT_Alexa_v3_logC_EI800( float3 Alexa)
{
float r_lin = normalizedLogCToRelativeExposure(Alexa.x);
float g_lin = normalizedLogCToRelativeExposure(Alexa.y);
float b_lin = normalizedLogCToRelativeExposure(Alexa.z);

float3 aces;
aces.x = r_lin * 0.680206f + g_lin * 0.236137f + b_lin * 0.083658f;
aces.y = r_lin * 0.085415f + g_lin * 1.017471f + b_lin * -0.102886f;
aces.z = r_lin * 0.002057f + g_lin * -0.062563f + b_lin * 1.060506f;
return aces;
}



#endif