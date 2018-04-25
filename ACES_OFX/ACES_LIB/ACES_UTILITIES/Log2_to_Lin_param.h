#ifndef __LOG2_TO_LIN_PARAM_H_INCLUDED__
#define __LOG2_TO_LIN_PARAM_H_INCLUDED__

// 
// Generic transform from log base-2 encoding to linear
// 

__device__ inline float log2_to_lin_32f( float logNorm, float middleGrey, float minExposure, float maxExposure)
{
if (logNorm < 0.0f) return 0.0f;
float lg2 = logNorm * (maxExposure - minExposure) + minExposure;
float lin = powf(2.0f, lg2) * middleGrey;
return lin;
}

__device__ inline float3 Log2_to_Lin_param( float3 In, float middleGrey, float minExposure, float maxExposure)
{
float3 out;
out.x = log2_to_lin_32f( In.x, middleGrey, minExposure, maxExposure);
out.y = log2_to_lin_32f( In.y, middleGrey, minExposure, maxExposure);
out.z = log2_to_lin_32f( In.z, middleGrey, minExposure, maxExposure);
return out;
}

#endif