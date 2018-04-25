#ifndef __ACES_TO_ACESCCT_H_INCLUDED__
#define __ACES_TO_ACESCCT_H_INCLUDED__

//
// ACES Color Space Conversion - ACES to ACEScct
//
// converts ACES2065-1 (AP0 w/ linear encoding) to 
//          ACEScct (AP1 w/ logarithmic encoding)
//

//import "ACESlib.Transform_Common";

/*
__CONSTANT__ float X_BRK = 0.0078125f;
__CONSTANT__ float Y_BRK = 0.155251141552511f;
__CONSTANT__ float A = 10.5402377416545f;
__CONSTANT__ float B = 0.0729055341958355f;
*/

__device__ inline float lin_to_ACEScct( float in)
{
    if (in <= X_BRK)
        return A * in + B;
    else // (in > X_BRK)
        return (_log2f(in) + 9.72f) / 17.52f;
}

__device__ inline float3 ACES_to_ACEScct( float3 ACES)
{

    float3 lin_AP1 = mult_f3_f44( ACES, AP0_2_AP1_MAT);

    float3 out;
    out.x = lin_to_ACEScct( lin_AP1.x);
    out.y = lin_to_ACEScct( lin_AP1.y);
    out.z = lin_to_ACEScct( lin_AP1.z);
    return out;
}

#endif