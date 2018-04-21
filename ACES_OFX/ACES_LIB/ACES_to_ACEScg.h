//
// ACES Color Space Conversion - ACES to ACEScg
//
// converts ACES2065-1 (AP0 w/ linear encoding) to 
//          ACEScg (AP1 w/ linear encoding)
//

//import "ACESlib.Transform_Common";


__device__ inline float3 ACES_to_ACEScg( float3 ACES)
{
    
    ACES = clamp_f3( ACES, 0.0f, HALF_POS_INF);

    float3 ACEScg = mult_f3_f44( ACES, AP0_2_AP1_MAT);
	
	return ACEScg;
}