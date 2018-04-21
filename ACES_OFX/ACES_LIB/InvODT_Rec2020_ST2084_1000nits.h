// 
// Inverse Output Device Transform - Rec.2020 (1000 cd/m^2)
//

/*
import "ACESlib.Utilities";
import "ACESlib.Transform_Common";
import "ACESlib.ODT_Common";
import "ACESlib.Tonescales";

#define DISPLAY_PRI							REC2020_PRI
#define DISPLAY_PRI_2_XYZ_MAT				RGBtoXYZ(DISPLAY_PRI, 1.0f)
*/

__device__ inline float3 InvODT_Rec2020_ST2084_1000nits( float3 outputCV)
{  

  Chromaticities DISPLAY_PRI = REC2020_PRI;
  mat4 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI, 1.0f);
  
  // Decode with inverse ST2084 transfer function
    float3 rgb = ST2084_2_Y_f3( outputCV);

  // Convert from display primary encoding
    // Display primaries to CIE XYZ
    float3 XYZ = mult_f3_f44( rgb, DISPLAY_PRI_2_XYZ_MAT);

      // Apply CAT from assumed observer adapted white to ACES white point
    XYZ = mult_f3_f33( XYZ, invert_f33( D60_2_D65_CAT));

    // CIE XYZ to rendering space RGB
    float3 rgbPre = mult_f3_f44( XYZ, XYZ_2_AP1_MAT);

  // Add small offset that was used to allow for a code value of 0
    rgbPre = add_f_f3( pow10(-4.4550166483f), rgbPre);

  // Apply the tonescale independently in rendering-space RGB
    float3 rgbPost;
    rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_1000nits());
    rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_1000nits());
    rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_1000nits());

  // Rendering space RGB to OCES
    float3 oces = mult_f3_f44( rgbPost, AP1_2_AP0_MAT);
    
    return oces;
}