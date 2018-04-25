#ifndef __ACESCG_TO_ACES_H_INCLUDED__
#define __ACESCG_TO_ACES_H_INCLUDED__


//
// ACES Color Space Conversion - ACEScg to ACES
//
// converts ACEScg (AP1 w/ linear encoding) to
//          ACES2065-1 (AP0 w/ linear encoding)
//

//import "ACESlib.Transform_Common";


__device__ inline float3 ACEScg_to_ACES( float3 ACEScg)
{

float3 ACES = mult_f3_f44( ACEScg, AP1_2_AP0_MAT);
return ACES;

}

#endif