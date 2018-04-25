#ifndef __LIN_TO_LOG2_PARAM_H_INCLUDED__
#define __LIN_TO_LOG2_PARAM_H_INCLUDED__

// 
// Generic transform from linear to a log base-2 encoding
// 

//import "ACESlib.Utilities";

__device__ inline float lin_to_log2_32f( float lin, float middleGrey, float minExposure, float maxExposure) 
{
if (lin <= 0.0f) return 0.0f;
float lg2 = log2f(lin / middleGrey);
float logNorm = (lg2 - minExposure)/(maxExposure - minExposure);
if( logNorm < 0.0f) logNorm = 0.0f;
return logNorm;
}

__device__ inline float Lin_to_Log2_param( float3 In, float middleGrey, float minExposure, float maxExposure)
{
float3 out;	
out.x = lin_to_log2_32f( In.x, middleGrey, minExposure, maxExposure);
out.y = lin_to_log2_32f( In.y, middleGrey, minExposure, maxExposure);
out.z = lin_to_log2_32f( In.z, middleGrey, minExposure, maxExposure);
return out;
}

#endif