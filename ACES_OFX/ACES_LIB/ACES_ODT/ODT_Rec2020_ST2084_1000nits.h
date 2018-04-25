#ifndef __ODT_REC2020_ST2084_1000NITS_H_INCLUDED__
#define __ODT_REC2020_ST2084_1000NITS_H_INCLUDED__

// 
// Output Device Transform - Rec.2020 (1000 cd/m^2)
//


__device__ float3 inline ODT_Rec2020_ST2084_1000nits( float3 oces)
{

  Chromaticities DISPLAY_PRI = REC2020_PRI;
  mat4 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI, 1.0f);
  
  // OCES to RGB rendering space
    float3 rgbPre = mult_f3_f44( oces, AP0_2_AP1_MAT);

  // Apply the tonescale independently in rendering-space RGB
    float3 rgbPost;
    rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_1000nits());
    rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_1000nits());
    rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_1000nits());

  // Subtract small offset to allow for a code value of 0
    rgbPost = add_f_f3( -pow10(-4.4550166483), rgbPost);

  // Convert to display primary encoding
    // Rendering space RGB to XYZ
    float3 XYZ = mult_f3_f44( rgbPost, AP1_2_XYZ_MAT);

      // Apply CAT from ACES white point to assumed observer adapted white point
    XYZ = mult_f3_f33( XYZ, D60_2_D65_CAT);

    // CIE XYZ to display primaries
    float3 rgb = mult_f3_f44( XYZ, XYZ_2_DISPLAY_PRI_MAT);

  // Handle out-of-gamut values
    // Clip values < 0 (i.e. projecting outside the display primaries)
    rgb = clamp_f3( rgb, 0.0f, HALF_POS_INF);

  // Encode with ST2084 transfer function
    float3 outputCV = Y_2_ST2084_f3( rgb);

	return outputCV;
}

#endif