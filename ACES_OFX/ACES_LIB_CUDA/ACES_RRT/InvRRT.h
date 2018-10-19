#ifndef __INV_RRT_H_INCLUDED__
#define __INV_RRT_H_INCLUDED__

// 
// Inverse Reference Rendering Transform (RRT)
//
//   Input is OCES
//   Output is ACES
//

__device__ float3 inline InvRRT( float3 oces)
{
  
  // --- OCES to RGB rendering space --- //
    float3 rgbPre = mult_f3_f44( oces, AP0_2_AP1_MAT);

  // --- Apply the tonescale independently in rendering-space RGB --- //
    float3 rgbPost;
    rgbPost.x = segmented_spline_c5_rev( rgbPre.x);
    rgbPost.y = segmented_spline_c5_rev( rgbPre.y);
    rgbPost.z = segmented_spline_c5_rev( rgbPre.z);

  // --- Global desaturation --- //
    rgbPost = mult_f3_f33( rgbPost, invert_f33(RRT_SAT_MAT));

    rgbPost = clamp_f3( rgbPost, 0.0f, HALF_MAX);

  // --- RGB rendering space to ACES --- //
    float3 aces = mult_f3_f44( rgbPost, AP1_2_AP0_MAT);

    aces = clamp_f3( aces, 0.0f, HALF_MAX);

  // --- Red modifier --- //
    float hue = rgb_2_hue( aces);
    float centeredHue = center_hue( hue, RRT_RED_HUE);
    float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);

    float minChan;
    if (centeredHue < 0) { // min_f3(aces) = aces.y (i.e. magenta-red)
      minChan = aces.y;
    } else { // min_f3(aces) = aces.z (i.e. yellow-red)
      minChan = aces.z;
    }

    float a = hueWeight * (1.0f - RRT_RED_SCALE) - 1.0f;
    float b = aces.x - hueWeight * (RRT_RED_PIVOT + minChan) * (1.0f - RRT_RED_SCALE);
    float c = hueWeight * RRT_RED_PIVOT * minChan * (1.0f - RRT_RED_SCALE);

    aces.x = ( -b - sqrtf( b * b - 4.0f * a * c)) / ( 2.0f * a);

  // --- Glow module --- //
    float saturation = rgb_2_saturation( aces);
    float ycOut = rgb_2_yc( aces);
    float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f);
    float reducedGlow = 1.0f + glow_inv( ycOut, RRT_GLOW_GAIN * s, RRT_GLOW_MID);

    aces = mult_f_f3( ( reducedGlow), aces);

	return aces;
}

#endif