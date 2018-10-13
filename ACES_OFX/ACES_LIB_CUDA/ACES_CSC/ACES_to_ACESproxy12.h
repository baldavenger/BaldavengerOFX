#ifndef __ACES_TO_ACES_PROXY12_H_INCLUDED__
#define __ACES_TO_ACES_PROXY12_H_INCLUDED__

//
// ACES Color Space Conversion - ACES to ACESproxy (12-bit)
//
// converts ACES2065-1 (AP0 w/ linear encoding) to 
//          ACESproxy (AP1 w/ ACESproxy encoding)
//

//import "ACESlib.Transform_Common";

__device__ inline int lin_to_ACESproxy( float in)
{
float StepsPerStop = 200.0f;
float MidCVoffset = 1700.0f;
int CVmin = 256;
int CVmax = 3760;
if (in <= powf(2.0f, -9.72f))
return CVmin;
else
return fmaxf( CVmin, fmin( CVmax, round( (log2f(in) + 2.5f) * StepsPerStop + MidCVoffset)));
}

__device__ inline float3 ACES_to_ACESproxy10( float3 ACES)
{

ACES = clamp_f3( ACES, 0.0f, HALF_POS_INF);
float3 lin_AP1 = mult_f3_f44( ACES, AP0_2_AP1_MAT);

int ACESproxy[3];
ACESproxy[0] = lin_to_ACESproxy( lin_AP1.x );
ACESproxy[1] = lin_to_ACESproxy( lin_AP1.y );
ACESproxy[2] = lin_to_ACESproxy( lin_AP1.z );

float3 out;
out.x = ACESproxy[0] / 4095.0f;
out.y = ACESproxy[1] / 4095.0f;
out.z = ACESproxy[2] / 4095.0f;
return out;
}

#endif