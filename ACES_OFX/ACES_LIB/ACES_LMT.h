#ifndef __ACES_LMT_H_INCLUDED__
#define __ACES_LMT_H_INCLUDED__

// Combined LMT support files

#include "ACES_functions.h"
#include "ACES_Utilities_Color.h"
#include "ACES_Transform_Common.h"
#include "ACES_Tonescales.h"
#include "ACES_RRT_Common.h"
#include "ACES_LMT_Common.h"


__device__ inline float3 overlay_f3( float3 a, float3 b)
{
const float LUMA_CUT = lin_to_ACEScct( 0.5f); 

float luma = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;

float3 out;
if (luma < LUMA_CUT) {
out.x = 2.0f * a.x * b.x;
out.y = 2.0f * a.y * b.y;
out.z = 2.0f * a.z * b.z;
} else {
out.x = 1.0f - (2.0f * (1.0f - a.x) * (1.0f - b.x));
out.y = 1.0f - (2.0f * (1.0f - a.y) * (1.0f - b.y));
out.z = 1.0f - (2.0f * (1.0f - a.z) * (1.0f - b.z));
}

return out;
}

__device__ inline float3 LMT_Bleach_Bypass( float3 aces)
{

float3 a, b, blend;
a = sat_adjust( aces, 0.9f);
a = mult_f_f3( 2.0f, a);
					  
b = sat_adjust( aces, 0.0f);
b = gamma_adjust_linear( b, 1.2f);

a = ACES_to_ACEScct( a);
b = ACES_to_ACEScct( b);

blend = overlay_f3( a, b);

aces = ACEScct_to_ACES(blend);

return aces;
}

__device__ inline float3 LMT_PFE(float3 aces)
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