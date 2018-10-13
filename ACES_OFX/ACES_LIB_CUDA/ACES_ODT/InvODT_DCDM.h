#ifndef __INVODT_DCDM_H_INCLUDED__
#define __INVODT_DCDM_H_INCLUDED__

// 
// Inverse Output Device Transform - DCDM (X'Y'Z')
//


//const float DISPGAMMA = 2.6; 



__device__ inline float3 InvODT_DCDM( float3 outputCV)
{
    
    // Decode with inverse transfer function
    float3 XYZ = dcdm_decode( outputCV);

    // CIE XYZ to rendering space RGB
    float3 linearCV = mult_f3_f44( XYZ, XYZ_2_AP1_MAT);

    // Scale code value to luminance
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