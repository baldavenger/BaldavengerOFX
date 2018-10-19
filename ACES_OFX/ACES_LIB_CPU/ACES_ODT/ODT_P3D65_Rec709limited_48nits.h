#ifndef __ODT_P3D65_REC709LIMITED_48NITS_H_INCLUDED__
#define __ODT_P3D65_REC709LIMITED_48NITS_H_INCLUDED__

// 
// Output Device Transform - P3D65 (Rec.709 Limited)
//

//
// Summary :
//  This transform is intended for mapping OCES onto a projector with P3
//  primaries that has been set up such that equal neutral RGB code values
//  sent to the projector produce a chromaticity equal to that of D65
//  and that the peak neutral luminance is 48 nits. The assumed observer 
//  adapted white is D65, and the viewing environment is a dark surround.
// 
//  Color values are limited to the Rec.709 gamut for applications where a
//  match is expected between a digital cinema projected image and a Rec.709
//  reference monitor.
//
//  A possible use case for this transform would be mastering for a theatrical
//  release with a creative white point of D65 and ultimate delivery in Rec.709.
//
// Device Primaries : 
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.68      0.32
//              Green:        0.265     0.69
//              Blue:         0.15      0.06
//              White:        0.3127    0.329     48 cd/m^2
//
// Display EOTF :
//  Gamma: 2.6
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.3127       0.329
//
// Viewing Environment:
//  Environment specified in SMPTE RP 431-2-2007
//



inline float3 ODT_P3D65_Rec709limited_48nits( float3 oces)
{

	const Chromaticities DISPLAY_PRI = P3D65_PRI;
	const mat4 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI, 1.0f);

	const Chromaticities LIMITING_PRI = REC709_PRI;

	const float DISPGAMMA = 2.6f;

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

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    float3 XYZ = mult_f3_f44( linearCV, AP1_2_XYZ_MAT);

    // Apply CAT from ACES white point to assumed observer adapted white point
    XYZ = mult_f3_f33( XYZ, D60_2_D65_CAT);

    // Constrain to limiting primaries
    XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
    
    // CIE XYZ to display primaries
    linearCV = mult_f3_f44( XYZ, XYZ_2_DISPLAY_PRI_MAT);

    // Handle out-of-gamut values
    // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = clamp_f3( linearCV, 0.0f, 1.0f);
  
    // Encode linear code values with transfer function
    float3 outputCV = pow_f3( linearCV, 1.0f / DISPGAMMA);
  
    return outputCV;
    
}

#endif