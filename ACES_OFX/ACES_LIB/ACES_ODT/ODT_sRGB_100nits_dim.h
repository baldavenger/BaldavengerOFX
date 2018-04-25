#ifndef __ODT_SRGB_100NITS_DIM_H_INCLUDED__
#define __ODT_SRGB_100NITS_DIM_H_INCLUDED__
// 
// Output Device Transform - RGB computer monitor
//

//
// Summary :
//  This transform is intended for mapping OCES onto a desktop computer monitor 
//  typical of those used in motion picture visual effects production. These 
//  monitors may occasionally be referred to as "sRGB" displays, however, the 
//  monitor for which this transform is designed does not exactly match the 
//  specifications in IEC 61966-2-1:1999.
// 
//  The assumed observer adapted white is D65, and the viewing environment is 
//  that of a dim surround. 
//
//  The monitor specified is intended to be more typical of those found in 
//  visual effects production.
//
// Device Primaries : 
//  Primaries are those specified in Rec. ITU-R BT.709
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.64      0.33
//              Green:        0.3       0.6
//              Blue:         0.15      0.06
//              White:        0.3127    0.329     100 cd/m^2
//
// Display EOTF :
//  The reference electro-optical transfer function specified in 
//  IEC 61966-2-1:1999.
//  Note: This EOTF is *NOT* gamma 2.2
//
// Signal Range:
//    This transform outputs full range code values.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.3127       0.329
//
// Viewing Environment:
//   This ODT has a compensation for viewing environment variables more typical 
//   of those associated with video mastering.
//


__device__ inline float3 ODT_sRGB_100nits_dim( float3 oces)
{

	const Chromaticities DISPLAY_PRI = REC709_PRI;
	const mat4 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI, 1.0f);

	// NOTE: The EOTF is *NOT* gamma 2.4, it follows IEC 61966-2-1:1999
	const float DISPGAMMA = 2.4f; 
	const float OFFSET = 0.055f;

    
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
    // moncurve_r with gamma of 2.4 and offset of 0.055 matches the EOTF found in IEC 61966-2-1:1999 (sRGB)
    outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET);
    outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET);
    outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET);

    return outputCV;
    
}

#endif