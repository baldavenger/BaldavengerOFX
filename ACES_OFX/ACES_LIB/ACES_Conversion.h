#ifndef __ACES_CONVERSION_H_INCLUDED__
#define __ACES_CONVERSION_H_INCLUDED__

//
// ACES Color Space Conversions
//

#include "ACES_functions.h"
#include "ACES_Transform_Common.h"

__device__ inline float lin_to_ACEScc( float in)
{
  if (in <= 0)
    return -0.3584474886f; // =(log2( pow(2.,-16.))+9.72)/17.52
  else if (in < powf(2.0f, -15.0f))
    return (log2f( powf(2.0f, -16.0f) + in * 0.5f) + 9.72f) / 17.52f;
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

__device__ inline float ACEScc_to_lin( float in)
{
  if (in < -0.3013698630f) // (9.72-15)/17.52
    return (powf( 2.0f, in * 17.52f - 9.72f) - powf( 2.0f, -16.0f)) * 2.0f;
  else if ( in < (log2f(HALF_MAX) + 9.72f) / 17.52f )
    return powf( 2.0f, in * 17.52f - 9.72f);
  else // (in >= (log2(HALF_MAX)+9.72)/17.52)
    return HALF_MAX;
}

__device__ inline float3 ACEScc_to_ACES( float3 ACEScc)
{
    float3 lin_AP1;
    lin_AP1.x = ACEScc_to_lin( ACEScc.x);
    lin_AP1.y = ACEScc_to_lin( ACEScc.y);
    lin_AP1.z = ACEScc_to_lin( ACEScc.z);

    float3 ACES = mult_f3_f44( lin_AP1, AP1_2_AP0_MAT);
	return ACES;  
}

__device__ inline float3 ACES_to_ACEScg( float3 ACES)
{
    
    ACES = clamp_f3( ACES, 0.0f, HALF_POS_INF);

    float3 ACEScg = mult_f3_f44( ACES, AP0_2_AP1_MAT);
	
	return ACEScg;
}

__device__ inline float3 ACEScg_to_ACES( float3 ACEScg)
{

float3 ACES = mult_f3_f44( ACEScg, AP1_2_AP0_MAT);
return ACES;

}

__device__ inline float lin_to_ACESproxy10( float in)
{
float StepsPerStop = 50.0f;
float MidCVoffset = 425.0f;
float CVmin = 64.0f;
float CVmax = 940.0f;	
if (in <= powf(2.0f, -9.72f))
return CVmin;
else
return max( CVmin, min( CVmax, round( (log2f(in) + 2.5f) * StepsPerStop + MidCVoffset)));
}

__device__ inline float3 ACES_to_ACESproxy10( float3 ACES)
{

ACES = clamp_f3( ACES, 0.0f, HALF_POS_INF); 
float3 lin_AP1 = mult_f3_f44( ACES, AP0_2_AP1_MAT);

int ACESproxy[3];
ACESproxy[0] = lin_to_ACESproxy10( lin_AP1.x );
ACESproxy[1] = lin_to_ACESproxy10( lin_AP1.y );
ACESproxy[2] = lin_to_ACESproxy10( lin_AP1.z );

float3 out;    
out.x = ACESproxy[0] / 1023.0f;
out.y = ACESproxy[1] / 1023.0f;
out.z = ACESproxy[2] / 1023.0f;
return out;
}

__device__ inline float ACESproxy10_to_lin( float in)
{
float StepsPerStop = 50.0f;
float MidCVoffset = 425.0f;
//int CVmin = 64;
//int CVmax = 940;  
return powf( 2.0f, ( in - MidCVoffset)/StepsPerStop - 2.5f);
}


__device__ inline float3 ACESproxy10_to_ACES( float3 In)
{
    
    float ACESproxy[3];
    ACESproxy[0] = In.x * 1023.0f;
    ACESproxy[1] = In.y * 1023.0f;
    ACESproxy[2] = In.z * 1023.0f;

    float3 lin_AP1;
    lin_AP1.x = ACESproxy10_to_lin( ACESproxy[0]);
    lin_AP1.y = ACESproxy10_to_lin( ACESproxy[1]);
    lin_AP1.z = ACESproxy10_to_lin( ACESproxy[2]);

    float3 ACES = mult_f3_f44( lin_AP1, AP1_2_AP0_MAT);
	return ACES;
}

__device__ inline float lin_to_ACESproxy12( float in)
{
float StepsPerStop = 200.0f;
float MidCVoffset = 1700.0f;
float CVmin = 256.0f;
float CVmax = 3760.0f;
if (in <= powf(2.0f, -9.72f))
return CVmin;
else
return max( CVmin, min( CVmax, round( (log2f(in) + 2.5f) * StepsPerStop + MidCVoffset)));
}

__device__ inline float3 ACES_to_ACESproxy12( float3 ACES)
{

ACES = clamp_f3( ACES, 0.0f, HALF_POS_INF);
float3 lin_AP1 = mult_f3_f44( ACES, AP0_2_AP1_MAT);

int ACESproxy[3];
ACESproxy[0] = lin_to_ACESproxy12( lin_AP1.x );
ACESproxy[1] = lin_to_ACESproxy12( lin_AP1.y );
ACESproxy[2] = lin_to_ACESproxy12( lin_AP1.z );

float3 out;
out.x = ACESproxy[0] / 4095.0f;
out.y = ACESproxy[1] / 4095.0f;
out.z = ACESproxy[2] / 4095.0f;
return out;
}

__device__ inline float ACESproxy12_to_lin( float in)
{
float StepsPerStop = 200.0f;
float MidCVoffset = 1700.0f;
//int CVmin = 256;
//int CVmax = 3760;

return powf( 2.0f, ( in - MidCVoffset)/StepsPerStop - 2.5f);
}


__device__ inline float3 ACESproxy12_to_ACES( float3 In)
{
    
float ACESproxy[3];
ACESproxy[0] = In.x * 4095.0f;
ACESproxy[1] = In.y * 4095.0f;
ACESproxy[2] = In.z * 4095.0f;

float3 lin_AP1;
lin_AP1.x = ACESproxy12_to_lin( ACESproxy[0]);
lin_AP1.y = ACESproxy12_to_lin( ACESproxy[1]);
lin_AP1.z = ACESproxy12_to_lin( ACESproxy[2]);

float3 ACES = mult_f3_f44( lin_AP1, AP1_2_AP0_MAT);
return ACES;

}

#endif