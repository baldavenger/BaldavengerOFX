#ifndef __ACES_LMT_ANALYTIC_4_H_INCLUDED__
#define __ACES_LMT_ANALYTIC_4_H_INCLUDED__

// "Bleach bypass" look

#include "../ACES_LMT.h"

__device__ inline float3 LMT_Analytic_4( float3 aces)
{

float3 a, b, blend;
a = sat_adjust( aces, 0.9f);
a = mult_f_f3( 2.0f, a);
					  
b = sat_adjust( aces, 0.0f);
b = gamma_adjust_linear( b, 1.2f);

a = ACES_to_ACEScct( a);
b = ACES_to_ACEScct( b);

blend = overlay_f3( a, b);
aces = ACEScct_to_ACES( blend);

return aces;

}

#endif