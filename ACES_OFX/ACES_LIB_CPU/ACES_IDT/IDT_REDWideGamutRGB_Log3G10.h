#ifndef __IDT_REDWIDEGAMUTRGB_LOG3G10_H_INCLUDED__
#define __IDT_REDWIDEGAMUTRGB_LOG3G10_H_INCLUDED__

// REDWideGamutRGB Log3G10 to ACES


inline float3 IDT_REDWideGamutRGB_Log3G10( float3 log3G10)
{
float r_lin = Log3G10_to_linear(log3G10.x);
float g_lin = Log3G10_to_linear(log3G10.y);
float b_lin = Log3G10_to_linear(log3G10.z);

float3 aces;
aces.x = r_lin * 0.785043f + g_lin * 0.083844f + b_lin * 0.131118f;
aces.y = r_lin * 0.023172f + g_lin * 1.087892f + b_lin * -0.111055f;
aces.z = r_lin * -0.073769f + g_lin * -0.314639f + b_lin * 1.388537f;

return aces;
}

#endif