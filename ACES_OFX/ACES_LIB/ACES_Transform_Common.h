#ifndef __ACES_TRANSFORM_COMMON_H_INCLUDED__
#define __ACES_TRANSFORM_COMMON_H_INCLUDED__

//
// Contains functions and constants shared by multiple forward and inverse transforms 
//

#define AP0_2_XYZ_MAT 				RGBtoXYZ( AP0, 1.0f)

#define XYZ_2_AP0_MAT 				XYZtoRGB( AP0, 1.0f)

#define AP1_2_XYZ_MAT  				RGBtoXYZ( AP1, 1.0f)

#define XYZ_2_AP1_MAT 				XYZtoRGB( AP1, 1.0f)

#define AP0_2_AP1_MAT 				mult_f44_f44( AP0_2_XYZ_MAT, XYZ_2_AP1_MAT)

#define AP1_2_AP0_MAT 				mult_f44_f44( AP1_2_XYZ_MAT, XYZ_2_AP0_MAT)

#define AP1_RGB2Y  					make_float3(AP1_2_XYZ_MAT.c0.y, AP1_2_XYZ_MAT.c1.y, AP1_2_XYZ_MAT.c2.y)

__constant__ float TINY = 1e-10;

__device__ inline float rgb_2_saturation( float3 rgb)
{
  return ( fmaxf( max_f3(rgb), TINY) - fmaxf( min_f3(rgb), TINY)) / fmaxf( max_f3(rgb), 1e-2);
}

#endif