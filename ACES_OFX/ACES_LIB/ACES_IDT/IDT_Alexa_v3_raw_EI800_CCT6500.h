#ifndef __IDT_ALEXA_V3_RAW_EI800__CCT6500_H_INCLUDED__
#define __IDT_ALEXA_V3_RAW_EI800__CCT6500_H_INCLUDED__

// ARRI ALEXA IDT for ALEXA linear files
//  with camera EI set to 800
//  and CCT of adopted white set to 6500K


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

#endif