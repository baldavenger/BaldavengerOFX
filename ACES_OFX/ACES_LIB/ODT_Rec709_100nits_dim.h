// 
// Output Device Transform - Rec709
//

/*
#define DISPLAY_PRI						REC709_PRI
#define XYZ_2_DISPLAY_PRI_MAT			XYZtoRGB(DISPLAY_PRI, 1.0f)

__constant__  float DISPGAMMA = 2.4f; 
__constant__  float L_W = 1.0f;
__constant__  float L_B = 0.0f;
__constant__  bool legalRange = false;
*/


__device__ float3 inline ODT_Rec709_100nits_dim( float3 oces)
{

  Chromaticities DISPLAY_PRI = REC709_PRI;
  mat4 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI, 1.0f);
  float DISPGAMMA = 2.4f;
  float L_W = 1.0f;
  float L_B = 0.0f;
  bool legalRange = false;
  
  // OCES to RGB rendering space
    float3 rgbPre = mult_f3_f44( oces, AP0_2_AP1_MAT);

  // Apply the tonescale independently in rendering-space RGB
    float3 rgbPost;
    rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
    rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
    rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

  // Scale luminance to linear code value
    float3 linearCV;
    linearCV.x = Y_2_linCV( rgbPost.x, CINEMA_WHITE, CINEMA_BLACK);
    linearCV.y = Y_2_linCV( rgbPost.y, CINEMA_WHITE, CINEMA_BLACK);
    linearCV.z = Y_2_linCV( rgbPost.z, CINEMA_WHITE, CINEMA_BLACK);

  // Apply gamma adjustment to compensate for dim surround
    linearCV = darkSurround_to_dimSurround( linearCV);

  // Apply desaturation to compensate for luminance difference
    linearCV = mult_f3_f33( linearCV, ODT_SAT_MAT);
    
  // Convert to display primary encoding
    // Rendering space RGB to XYZ
    float3 XYZ = mult_f3_f44( linearCV, AP1_2_XYZ_MAT);

      // Apply CAT from ACES white point to assumed observer adapted white point
      XYZ = mult_f3_f33( XYZ, D60_2_D65_CAT);

    // CIE XYZ to display primaries
    linearCV = mult_f3_f44( XYZ, XYZ_2_DISPLAY_PRI_MAT);

  // Handle out-of-gamut values
    // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
  
  // Encode linear code values with transfer function
    float3 outputCV;
    outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
    outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
    outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);

  // Default output is full range, check if legalRange param was set to true
    if (legalRange) {
    outputCV = fullRange_to_smpteRange_f3( outputCV);
    }


    return outputCV;
}
