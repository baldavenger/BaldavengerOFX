#ifndef __ODT_DCDM_H_INCLUDED__
#define __ODT_DCDM_H_INCLUDED__

// 
// Output Device Transform - DCDM (X'Y'Z')
//

// 
// Summary :
//  The output of this transform follows the encoding specified in SMPTE 
//  S428-1-2006. The gamut is a device-independent colorimetric encoding based  
//  on CIE XYZ. Therefore, output values are not limited to any physical 
//  device's actual color gamut that is determined by its color primaries.
// 
//  The assumed observer adapted white is D60, and the viewing environment is 
//  that of a dark theater. 
//
//  This transform shall be used for a device calibrated to match the Digital 
//  Cinema Reference Projector Specification outlined in SMPTE RP 431-2-2007.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.32168      0.33767
//
// Viewing Environment:
//  Environment specified in SMPTE RP 431-2-2007
//


//const float DISPGAMMA = 2.6; 



__device__ inline float3 ODT_DCDM( float3 oces)
{

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

    // Rendering space RGB to XYZ
    float3 XYZ = mult_f3_f44( linearCV, AP1_2_XYZ_MAT);

    // Handle out-of-gamut values
    // There should not be any negative values but will clip just to ensure no 
    // math errors occur with the gamma function in the EOTF
    XYZ = clamp_f3( XYZ, 0.0f, HALF_POS_INF);

    // Encode linear code values with transfer function
    float3 outputCV = dcdm_encode( XYZ);

    return outputCV;
}

#endif