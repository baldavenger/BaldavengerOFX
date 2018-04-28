#ifndef __ODT_P3DCI_48NITS_H_INCLUDED__
#define __ODT_P3DCI_48NITS_H_INCLUDED__


// 
// Output Device Transform - P3DCI (D60 simulation)
//

//
// Summary :
//  This transform is intended for mapping OCES onto a P3 digital cinema 
//  projector that is calibrated to a DCI white point at 48 cd/m^2. The assumed 
//  observer adapted white is D60, and the viewing environment is that of a dark
//  theater. 
//
// Device Primaries : 
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.68      0.32
//              Green:        0.265     0.69
//              Blue:         0.15      0.06
//              White:        0.314     0.351     48 cd/m^2
//
// Display EOTF :
//  Gamma: 2.6
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.32168      0.33767
//
// Viewing Environment:
//  Environment specified in SMPTE RP 431-2-2007
//


__device__ inline float3 ODT_P3DCI_48nits( float3 oces)
{

	const Chromaticities DISPLAY_PRI = P3DCI_PRI;
	const mat4 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI, 1.0f);

	const float DISPGAMMA = 2.6f; 

	// Rolloff white settings for P3DCI (D60 simulation)
	const float NEW_WHT = 0.918f;
	const float ROLL_WIDTH = 0.5f;    
	const float SCALE = 0.96f;

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
    // In order to produce D60 on a device calibrated to DCI white (i.e. 
    // equal display code values yield CIE x,y chromaticities of 0.314, 0.351) 
    // the red channel is higher than blue and green to compensate for the 
    // "greener" DCI white. This is the intended behavior but it means that 
    // without compensation, as highlights increase, the red channel will hit 
    // the device maximum first and clip, resulting in a chromaticity shift as 
    // the blue and green channels continue to increase.
    // To avoid this clipping behavior, a slight scale factor is applied to 
    // allow the ODT to simulate D60 within the DCI calibration white point. 
    // However, the magnitude of the scale factor required was considered too 
    // large; therefore, the scale factor was reduced and the additional 
    // required compression was achieved via a reshaping of the highlight 
    // rolloff in conjunction with the scale. The shape of this rolloff was 
    // determined through subjective experiments and deemed to best reproduce 
    // the "character" of the highlights in the P3D60 ODT.

    // Roll off highlights to avoid need for as much scaling
    linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH);
    linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH);
    linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH);

    // Scale and clamp white to avoid casted highlights due to D60 simulation
    linearCV.x = fmin( linearCV.x, NEW_WHT) * SCALE;
    linearCV.y = fmin( linearCV.y, NEW_WHT) * SCALE;
    linearCV.z = fmin( linearCV.z, NEW_WHT) * SCALE;

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    float3 XYZ = mult_f3_f44( linearCV, AP1_2_XYZ_MAT);

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