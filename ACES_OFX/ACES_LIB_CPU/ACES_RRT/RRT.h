#ifndef __RRT_H_INCLUDED__
#define __RRT_H_INCLUDED__
// 
// Reference Rendering Transform (RRT)
//
//   Input is ACES
//   Output is OCES
//

float3 inline _RRT( float3 aces)
{
  
  // --- Glow module --- //
    float saturation = rgb_2_saturation( aces);
    float ycIn = rgb_2_yc( aces);
    float s = sigmoid_shaper( (saturation - 0.4f) / 0.2f);
    float addedGlow = 1.0f + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);

    aces = mult_f_f3( addedGlow, aces);

  // --- Red modifier --- //
    float hue = rgb_2_hue( aces);
    float centeredHue = center_hue( hue, RRT_RED_HUE);
    float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);

    aces.x = aces.x + hueWeight * saturation * (RRT_RED_PIVOT - aces.x) * (1.0f - RRT_RED_SCALE);

  // --- ACES to RGB rendering space --- //
    aces = clamp_f3( aces, 0.0f, HALF_POS_INF);  // avoids saturated negative colors from becoming positive in the matrix

    float3 rgbPre = mult_f3_f44( aces, AP0_2_AP1_MAT);

    rgbPre = clamp_f3( rgbPre, 0.0f, HALF_MAX);

  // --- Global desaturation --- //
    rgbPre = mult_f3_f33( rgbPre, RRT_SAT_MAT);

  // --- Apply the tonescale independently in rendering-space RGB --- //
    float3 rgbPost;
    rgbPost.x = segmented_spline_c5_fwd( rgbPre.x);
    rgbPost.y = segmented_spline_c5_fwd( rgbPre.y);
    rgbPost.z = segmented_spline_c5_fwd( rgbPre.z);

  // --- RGB rendering space to OCES --- //
    float3 rgbOces = mult_f3_f44( rgbPost, AP1_2_AP0_MAT);

	return rgbOces;
}

#endif