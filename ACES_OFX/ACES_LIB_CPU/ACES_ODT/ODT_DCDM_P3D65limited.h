#ifndef __ODT_DCDM_P3D65LIMITED_H_INCLUDED__
#define __ODT_DCDM_P3D65LIMITED_H_INCLUDED__

// 
// Output Device Transform - DCDM (X'Y'Z') (P3D65 Limited)
//

//
// Summary:
//  This ODT encodes XYZ colorimetry that is gamut-limited to P3 primaries with 
//  a D65 whitepoint. This has two advantages:
// 
//   1) "Gamut mapping" is explicitly controlled by the ODT by clipping any XYZ 
//      values that map outside of the P3 gamut. Without this step, it would be 
//      left to the projector to handle any XYZ values outside of the P3 gamut. 
//      In most devices, this is performed using a simple clip, but not always.   
//      If out-of-gamut values are left to be handled by the device, different 
//      image appearance could potentially result on different devices even 
//      though they have the same gamut.
//
//   2) Assuming the content was graded (and approved) on a projector with a 
//      P3D65 gamut, limiting the colors to that gamut assures there will be
//      no unexpected color appearance if the DCP is later viewed on a device 
//      with a wider gamut.
// 
//  The assumed observer adapted white is D65, and the viewing environment is 
//  that of a dark theater. 
//
//  This transform shall be used for a device calibrated to match the Digital 
//  Cinema Reference Projector Specification outlined in SMPTE RP 431-2-2007.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.3127       0.329
//
// Viewing Environment:
//  Environment specified in SMPTE RP 431-2-2007
//

//const float DISPGAMMA = 2.6; 


inline float3 ODT_DCDM_P3D65limited( float3 oces)
{

	const Chromaticities LIMITING_PRI = P3D65_PRI;
	
    // ACES to RGB rendering space
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

    // Constrain to limiting primaries
    XYZ = limit_to_primaries( XYZ, LIMITING_PRI);

    // Encode linear code values with transfer function
    float3 outputCV = dcdm_encode( XYZ);

    return outputCV;
}

#endif