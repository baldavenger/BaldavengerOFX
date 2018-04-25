#ifndef __ACES_LMT_ANALYTIC_1A_H_INCLUDED__
#define __ACES_LMT_ANALYTIC_1A_H_INCLUDED__

// LMT Analytic 1a

#include "../ACES_LMT.h"


__device__ inline float3 LMT_Analytic_1a( float3 aces)
{

float3 SLOPE = {0.85f, 0.85f, 0.85f};
float3 OFFSET = {0.024f, 0.024f, 0.024f};
float3 POWER = {0.9f, 0.9f, 0.9f};
float SAT = 0.94f;

aces = ASCCDL_inACEScct( aces, SLOPE, OFFSET, POWER, SAT);

return aces;
}

#endif