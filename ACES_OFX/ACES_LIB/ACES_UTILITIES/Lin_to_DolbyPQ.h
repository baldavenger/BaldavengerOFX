#ifndef __LIN_TO_DOLBYPQ_H_INCLUDED__
#define __LIN_TO_DOLBYPQ_H_INCLUDED__

// 
// Generic transform from linear to encoding specified in SMPTE ST2084
// 

//import "ACESlib.Utilities_Color";


__device__ inline float3 Lin_to_DolbyPQ( float3 In)
{  
float3 out;
out.x = Y_2_ST2084( In.x );
out.y = Y_2_ST2084( In.x );
out.z = Y_2_ST2084( In.x );
return out;  
}

#endif