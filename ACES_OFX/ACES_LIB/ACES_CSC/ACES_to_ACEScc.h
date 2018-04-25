#ifndef __ACES_TO_ACESCC_H_INCLUDED__
#define __ACES_TO_ACESCC_H_INCLUDED__

//
// ACES Color Space Conversion - ACES to ACEScc
//
// converts ACES2065-1 (AP0 w/ linear encoding) to 
//          ACEScc (AP1 w/ logarithmic encoding)
//

//import "ACESlib.Transform_Common";

__device__ inline float lin_to_ACEScc( float in)
{
  if (in <= 0)
    return -0.3584474886f; // =(log2( pow(2.,-16.))+9.72)/17.52
  else if (in < powf(2.0f, -15.0f))
    return (log2f( pow(2.0f, -16.0f) + in * 0.5f) + 9.72f) / 17.52f;
  else // (in >= pow(2.,-15))
    return (log2f(in) + 9.72f) / 17.52f;
}

__device__ inline float3 ACES_to_ACEScc( float3 ACES)
{
    
    ACES = clamp_f3( ACES, 0.0f, HALF_POS_INF);
    float3 lin_AP1 = mult_f3_f44( ACES, AP0_2_AP1_MAT);

	float3 out;
    out.x = lin_to_ACEScc( lin_AP1.x);
    out.y = lin_to_ACEScc( lin_AP1.y);
    out.z = lin_to_ACEScc( lin_AP1.z);
    return out;
}

#endif