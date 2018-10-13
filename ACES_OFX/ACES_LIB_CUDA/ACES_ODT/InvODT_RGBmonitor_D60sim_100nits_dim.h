#ifndef __INVODT_RGBMONITOR_D60SIM_100NITS_DIM_H_INCLUDED__
#define __INVODT_RGBMONITOR_D60SIM_100NITS_DIM_H_INCLUDED__

// 
// Inverse Output Device Transform - RGB computer monitor (D60 simulation)
//

__device__ float3 inline InvODT_RGBmonitor_D60sim_100nits_dim( float3 outputCV)
{
 
  Chromaticities DISPLAY_PRI = REC709_PRI;
  mat4 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI, 1.0f);
  float DISPGAMMA = 2.4f;
  float OFFSET = 0.055f;
  float SCALE = 0.955f;
  
  // Decode to linear code values with inverse transfer function
    float3 linearCV;
    // moncurve_f with gamma of 2.4 and offset of 0.055 matches the EOTF found in IEC 61966-2-1:1999 (sRGB)
    linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
    linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
    linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);

  // Convert from display primary encoding
    // Display primaries to CIE XYZ
    float3 XYZ = mult_f3_f44( linearCV, DISPLAY_PRI_2_XYZ_MAT);
  
    // CIE XYZ to rendering space RGB
    linearCV = mult_f3_f44( XYZ, XYZ_2_AP1_MAT);

  // Undo desaturation to compensate for luminance difference
    linearCV = mult_f3_f33( linearCV, invert_f33( ODT_SAT_MAT));

  // Undo gamma adjustment to compensate for dim surround
    linearCV = dimSurround_to_darkSurround( linearCV);

  // Undo scaling done for D60 simulation
    linearCV.x = linearCV.x / SCALE;
    linearCV.y = linearCV.y / SCALE;
    linearCV.z = linearCV.z / SCALE;
  
  // Scale linear code value to luminance
    float3 rgbPre;
    rgbPre.x = linCV_2_Y( linearCV.x, CINEMA_WHITE, CINEMA_BLACK);
    rgbPre.y = linCV_2_Y( linearCV.y, CINEMA_WHITE, CINEMA_BLACK);
    rgbPre.z = linCV_2_Y( linearCV.z, CINEMA_WHITE, CINEMA_BLACK);

  // Apply the tonescale independently in rendering-space RGB
    float3 rgbPost;
    rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
    rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
    rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

  // Rendering space RGB to OCES
    float3 oces = mult_f3_f44( rgbPost, AP1_2_AP0_MAT);

    return oces;
}

#endif