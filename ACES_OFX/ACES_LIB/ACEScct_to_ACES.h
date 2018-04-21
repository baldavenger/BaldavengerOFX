//
// ACES Color Space Conversion - ACEScct to ACES
//
// converts ACEScct (AP1 w/ ACESlog encoding) to 
//          ACES2065-1 (AP0 w/ linear encoding)
//

//import "ACESlib.Transform_Common";

/*
const float X_BRK = 0.0078125;
const float Y_BRK = 0.155251141552511;
const float A = 10.5402377416545;
const float B = 0.0729055341958355;
*/


__device__ inline float ACEScct_to_lin( float in)
{
if (in > Y_BRK)
return powf( 2.0f, in * 17.52f - 9.72f);
else
return (in - B) / A;
}

__device__ inline float3 ACEScct_to_ACES( float3 ACEScct)
{
float3 lin_AP1;
lin_AP1.x = ACEScct_to_lin( ACEScct.x);
lin_AP1.y = ACEScct_to_lin( ACEScct.y);
lin_AP1.z = ACEScct_to_lin( ACEScct.z);

float3 ACES = mult_f3_f44( lin_AP1, AP1_2_AP0_MAT);
return ACES;
}