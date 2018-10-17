#ifndef __INVODT_P3DCI_D65SIM_48NITS_H_INCLUDED__
#define __INVODT_P3DCI_D65SIM_48NITS_H_INCLUDED__

// 
// Inverse Output Device Transform - P3DCI (D65 Simulation)
//

inline float3 InvODT_P3DCI_D65sim_48nits( float3 outputCV)
{

	const Chromaticities DISPLAY_PRI = P3DCI_PRI;
	const mat4 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI, 1.0f);

	const float DISPGAMMA = 2.6f; 

	// Rolloff white settings for P3DCI (D65 simulation)
	const float NEW_WHT = 0.908f;
	const float ROLL_WIDTH = 0.5f;    
	const float SCALE = 0.9575f;

    // Decode to linear code values with inverse transfer function
    float3 linearCV = pow_f3( outputCV, DISPGAMMA);

    // Convert from display primary encoding
    // Display primaries to CIE XYZ
    float3 XYZ = mult_f3_f44( linearCV, DISPLAY_PRI_2_XYZ_MAT);

    // CIE XYZ to rendering space RGB
    linearCV = mult_f3_f44( XYZ, XYZ_2_AP1_MAT);

    // Undo highlight roll-off and scaling
    linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH);
    linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH);
    linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH);

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