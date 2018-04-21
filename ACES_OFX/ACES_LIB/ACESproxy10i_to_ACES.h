//
// ACES Color Space Conversion - ACESproxy (10-bit) to ACES
//
// converts ACESproxy (AP1 w/ ACESproxy encoding) to 
//          ACES2065-1 (AP0 w/ linear encoding)
//

//import "ACESlib.Transform_Common";

/*
const float StepsPerStop = 50.;
const float MidCVoffset = 425.;
const int CVmin = 64;
const int CVmax = 940;
*/

__device__ inline float ACESproxy10_to_lin( int in)
{
float StepsPerStop = 50.0f;
float MidCVoffset = 425.0f;
int CVmin = 64;
int CVmax = 940;  
return powf( 2.0f, ( in - MidCVoffset)/StepsPerStop - 2.5f);
}


__device__ inline float3 ACESproxy10_to_ACES( int In[3])
{
    
    int ACESproxy[3];
    ACESproxy[0] = In[0] * 1023.0f;
    ACESproxy[1] = In[1] * 1023.0f;
    ACESproxy[2] = In[2] * 1023.0f;

    float3 lin_AP1;
    lin_AP1.x = ACESproxy10_to_lin( ACESproxy[0]);
    lin_AP1.y = ACESproxy10_to_lin( ACESproxy[1]);
    lin_AP1.z = ACESproxy10_to_lin( ACESproxy[2]);

    float3 ACES = mult_f3_f44( lin_AP1, AP1_2_AP0_MAT);
	return ACES;
}