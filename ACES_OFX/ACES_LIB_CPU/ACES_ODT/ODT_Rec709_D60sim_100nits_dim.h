#ifndef __ODT_REC709_D60SIM_100NITS_DIM_H_INCLUDED__
#define __ODT_REC709_D60SIM_100NITS_DIM_H_INCLUDED__

// 
// Output Device Transform - Rec709 (D60 simulation)
//

//
// Summary :
//  This transform is intended for mapping OCES onto a Rec.709 broadcast monitor
//  that is calibrated to a D65 white point at 100 cd/m^2. The assumed observer 
//  adapted white is D60, and the viewing environment is a dim surround. 
//
//  A possible use case for this transform would be cinema "soft-proofing".
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
//                                     0.32168      0.33767
//
// Viewing Environment:
//   This ODT has a compensation for viewing environment variables more typical 
//   of those associated with video mastering.
//


inline float3 ODT_Rec709_D60sim_100nits_dim( float3 oces)
{

	const Chromaticities DISPLAY_PRI = REC709_PRI;
	const mat4 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI, 1.0f);

	const float DISPGAMMA = 2.4f; 
	const float L_W = 1.0f;
	const float L_B = 0.0f;

	const float SCALE = 0.955f;
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

    // --- Compensate for different white point being darker  --- //
    // This adjustment corrects for an issue that exists in ODTs where the 
    // device is calibrated to a white chromaticity other than that of the 
    // adapted white.
    // In order to produce D60 on a device calibrated to D65 white (i.e. 
    // equal display code values yield CIE x,y chromaticities of 0.3217, 0.329) 
    // the red channel is higher than blue and green to compensate for the 
    // "bluer" D60 white. This is the intended behavior but it means that 
    // without compensation, as highlights increase, the red channel will hit 
    // the device maximum first and clip, resulting in a chromaticity shift as 
    // the green and blue channels continue to increase.
    // To avoid this clipping behavior, a slight scale factor is applied to 
    // allow the ODT to simulate D60 within the D65 calibration white point. 

    // Scale and clamp white to avoid casted highlights due to D60 simulation
    linearCV.x = fminf( linearCV.x, 1.0f) * SCALE;
    linearCV.y = fminf( linearCV.y, 1.0f) * SCALE;
    linearCV.z = fminf( linearCV.z, 1.0f) * SCALE;

    // Apply gamma adjustment to compensate for dim surround
    linearCV = darkSurround_to_dimSurround( linearCV);

    // Apply desaturation to compensate for luminance difference
    linearCV = mult_f3_f33( linearCV, ODT_SAT_MAT);

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    float3 XYZ = mult_f3_f44( linearCV, AP1_2_XYZ_MAT);

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
