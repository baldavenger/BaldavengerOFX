#ifndef __ACES_LMT_ANALYTIC_3_H_INCLUDED__
#define __ACES_LMT_ANALYTIC_3_H_INCLUDED__

// 
// Look Modification Transform
//


#include "../ACES_LMT.h"


inline float3 LMT_Analytic_3( float3 aces)
{

aces = scale_C( aces, 0.7f);

float3 SLOPE = make_float3(1.0f, 1.0f, 0.94f);
float3 OFFSET = make_float3(0.0f, 0.0f, 0.02f);
float3 POWER = make_float3(1.0f, 1.0f, 1.0f);
aces = ASCCDL_inACEScct( aces, SLOPE, OFFSET, POWER);

aces = gamma_adjust_linear( aces, 1.5f, 0.18f);

aces = rotate_H_in_H( aces, 0.0f, 30.0f, 5.0f);

aces = rotate_H_in_H( aces, 80.0f, 60.0f, -15.0f);

aces = rotate_H_in_H( aces, 52.0f, 50.0f, -14.0f);

aces = scale_C_at_H( aces, 45.0f, 40.0f, 1.4f);

aces = rotate_H_in_H( aces, 190.0f, 40.0f, 30.0f);

aces = scale_C_at_H( aces, 240.0f, 120.0f, 1.4f);

return aces;
}

#endif