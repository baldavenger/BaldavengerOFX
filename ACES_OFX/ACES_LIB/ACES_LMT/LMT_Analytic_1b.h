#ifndef __ACES_LMT_ANALYTIC_1B_H_INCLUDED__
#define __ACES_LMT_ANALYTIC_1B_H_INCLUDED__

// LMT Analytic 1b

#include "../ACES_LMT.h"


__device__ inline float3 LMT_Analytic_1b( float3 aces)
{

float3 SLOPE = {0.6f, 0.6f, 0.6f};
float3 OFFSET = {0.1f, 0.1f, 0.1f};
float3 POWER = {0.83f, 0.83f, 0.83f};
float SAT = 0.92f;

aces = ASCCDL_inACEScct( aces, SLOPE, OFFSET, POWER, SAT);

return aces;
}

#endif