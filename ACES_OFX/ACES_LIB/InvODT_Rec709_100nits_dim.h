// 
// Inverse Output Device Transform - Rec709
//

/*
import "ACESlib.Utilities";
import "ACESlib.Transform_Common";
import "ACESlib.ODT_Common";
import "ACESlib.Tonescales";

#define DISPLAY_PRI							REC709_PRI
#define DISPLAY_PRI_2_XYZ_MAT				RGBtoXYZ(DISPLAY_PRI, 1.0f)

__constant__ float DISPGAMMA = 2.4f; 
__constant__ float L_W = 1.0f;
__constant__ float L_B = 0.0f;
__constant__ bool legalRange = false;
*/

__device__ inline float3 InvODT_Rec709_100nits_dim( float3 outputCV)
{  

  Chromaticities DISPLAY_PRI = REC709_PRI;
  mat4 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI, 1.0f);
  float DISPGAMMA = 2.4f;
  float L_W = 1.0f;
  float L_B = 0.0f;
  bool legalRange = false;
  
  // Default output is full range, check if legalRange param was set to true
    if (legalRange) {
      outputCV = smpteRange_to_fullRange_f3( outputCV);
    }

  // Decode to linear code values with inverse transfer function
    float3 linearCV;
    linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B);
    linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B);
    linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B);

  // Convert from display primary encoding
    // Display primaries to CIE XYZ
    float3 XYZ = mult_f3_f44( linearCV, DISPLAY_PRI_2_XYZ_MAT);
  
      // Apply CAT from assumed observer adapted white to ACES white point
    XYZ = mult_f3_f33( XYZ, invert_f33( D60_2_D65_CAT));

    // CIE XYZ to rendering space RGB
    linearCV = mult_f3_f44( XYZ, XYZ_2_AP1_MAT);

  // Undo desaturation to compensate for luminance difference
    linearCV = mult_f3_f33( linearCV, invert_f33( ODT_SAT_MAT));

  // Undo gamma adjustment to compensate for dim surround
    linearCV = dimSurround_to_darkSurround( linearCV);

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