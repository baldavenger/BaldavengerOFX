#ifndef __ACES_LMT_ANALYTIC_1C_H_INCLUDED__
#define __ACES_LMT_ANALYTIC_1C_H_INCLUDED__

// LMT Analytic 1c

#include "../ACES_LMT.h"


inline float3 LMT_Analytic_1c( float3 aces)
{

// Adjust contrast
const float GAMMA = 0.6f;
aces = gamma_adjust_linear( aces, GAMMA);

// Adjust saturation
const float SAT_FACTOR = 0.85f;
aces = sat_adjust( aces, SAT_FACTOR);

return aces;
}

#endif