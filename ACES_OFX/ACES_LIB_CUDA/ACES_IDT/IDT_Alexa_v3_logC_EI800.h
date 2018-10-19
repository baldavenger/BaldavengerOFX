#ifndef __IDT_ALEXA_V3_LOGC_EI800_H_INCLUDED__
#define __IDT_ALEXA_V3_LOGC_EI800_H_INCLUDED__

// ARRI ALEXA IDT for ALEXA logC files
//  with camera EI set to 800

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