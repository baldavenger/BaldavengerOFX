#ifndef __ODT_REC2020_REC709LIMITED_100NITS_DIM_H_INCLUDED__
#define __ODT_REC2020_REC709LIMITED_100NITS_DIM_H_INCLUDED__

// 
// Output Device Transform - Rec.2020 (Rec.709 Limited)
//

//
// Summary :
//  This transform is intended for mapping OCES onto a Rec.2020 broadcast 
//  monitor that is calibrated to a D65 white point at 100 cd/m^2. The assumed 
//  observer adapted white is D65, and the viewing environment is that of a dim 
//  surround. 
// 
//  Color values are limited to the Rec.709 gamut for applications where a 
//  match is expected between a Rec.2020 image and a Rec.709 reference 
//  monitor.
//
//  Assuming the content was graded (and approved) on a Rec.709 display,
//  limiting the colors to P3D65 assures there will be no unexpected colors 
//  when viewed on a Rec.2020 device with a (potentially) wider gamut.
//
//  A possible use case for this transform would be UHDTV/video mastering.
//
// Device Primaries : 
//  Primaries are those specified in Rec. ITU-R BT.2020
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.708     0.292
//              Green:        0.17      0.797
//              Blue:         0.131     0.046
//              White:        0.3127    0.329     100 cd/m^2
//
// Display EOTF :
//  The reference electro-optical transfer function specified in 
//  Rec. ITU-R BT.1886.
//
// Signal Range:
//    By default, this transform outputs full range code values. If instead a 
//    SMPTE "legal" signal is desired, there is a runtime flag to output 
//    SMPTE legal signal. In ctlrender, this can be achieved by appending 
//    '-param1 legalRange 1' after the '-ctl odt.ctl' string.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.3127       0.329
//
// Viewing Environment:
//   This ODT has a compensation for viewing environment variables more typical 
//   of those associated with video mastering.
//


inline float3 ODT_Rec2020_Rec709limited_100nits_dim( float3 oces)
{
    
    const Chromaticities DISPLAY_PRI = REC2020_PRI;
	const mat4 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI, 1.0f);

	const Chromaticities LIMITING_PRI = REC709_PRI;

	const float DISPGAMMA = 2.4f; 
	const float L_W = 1.0f;
	const float L_B = 0.0f;

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

    // Constrain to limiting primaries
    XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
    
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

#endif