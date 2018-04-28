#ifndef __ACES_OUTPUTTRANSFORMS_H_INCLUDED__
#define __ACES_OUTPUTTRANSFORMS_H_INCLUDED__

#include "ACES_functions.h"
#include "ACES_Transform_Common.h"
#include "ACES_RRT_Common.h"
#include "ACES_ODT_Common.h"
#include "ACES_SSTS.h"


__device__ inline float3 limit_to_primaries
( 
    float3 XYZ, 
    Chromaticities LIMITING_PRI
)
{
    mat4 XYZ_2_LIMITING_PRI_MAT = XYZtoRGB( LIMITING_PRI, 1.0f);
    mat4 LIMITING_PRI_2_XYZ_MAT = RGBtoXYZ( LIMITING_PRI, 1.0f);

    // XYZ to limiting primaries
    float3 rgb = mult_f3_f44( XYZ, XYZ_2_LIMITING_PRI_MAT);

    // Clip any values outside the limiting primaries
    float3 limitedRgb = clamp_f3( rgb, 0.0f, 1.0f);
    
    // Convert limited RGB to XYZ
    return mult_f3_f44( limitedRgb, LIMITING_PRI_2_XYZ_MAT);
}

__device__ inline float3 dark_to_dim( float3 XYZ)
{
  float3 xyY = XYZ_2_xyY(XYZ);
  xyY.z = clamp( xyY.z, 0.0f, HALF_POS_INF);
  xyY.z = powf( xyY.z, DIM_SURROUND_GAMMA);
  return xyY_2_XYZ(xyY);
}

__device__ inline float3 dim_to_dark( float3 XYZ)
{
  float3 xyY = XYZ_2_xyY(XYZ);
  xyY.z = clamp( xyY.z, 0.0f, HALF_POS_INF);
  xyY.z = powf( xyY.z, 1.0f/DIM_SURROUND_GAMMA);
  return xyY_2_XYZ(xyY);
}

__device__ inline float3 outputTransform
(
    float3 in,
    float Y_MIN,
    float Y_MID,
    float Y_MAX,    
    Chromaticities DISPLAY_PRI,
    Chromaticities LIMITING_PRI,
    int EOTF,  
    int SURROUND,
    bool STRETCH_BLACK = true,
    bool D60_SIM = false,
    bool LEGAL_RANGE = false
)
{
    mat4 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI, 1.0f);

    // NOTE: This is a bit of a hack - probably a more direct way to do this.
    // Fix in future version
    TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX);
    float expShift = log2f(inv_ssts(Y_MID, PARAMS_DEFAULT))- log2f(0.18f);
    TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift);

    // RRT sweeteners
    float3 rgbPre = rrt_sweeteners( in);

    // Apply the tonescale independently in rendering-space RGB
    float3 rgbPost = ssts_f3( rgbPre, PARAMS);

    // At this point data encoded AP1, scaled absolute luminance (cd/m^2)

    // Scale to linear code value
//     if (EOTF != 0) {  // ST-2084 (PQ)
//         rgbPost = Y_2_linCV_f3( rgbPost, Y_MAX, Y_MIN);
//     }
    float3 linearCV = Y_2_linCV_f3( rgbPost, Y_MAX, Y_MIN);
    
    // Rendering primaries to XYZ
    float3 XYZ = mult_f3_f44( linearCV, AP1_2_XYZ_MAT);

    // Apply gamma adjustment to compensate for dim surround
    // NOTE: This section would only apply for SDR. This is a placeholder block.
    // TOD0: Come up with new surround compensation algorithm, applicable across
    // all dynamic ranges and supporting dark/dim/normal surround.
//     if (SURROUND == 1) {
//         print( "\nDim surround\n");
//         XYZ = dark_to_dim( XYZ);
//     }

    // Gamut limit to limiting primaries
    // NOTE: Would be nice to just say
    //    if (LIMITING_PRI != DISPLAY_PRI)
    // but you can't because Chromaticities do not work with bool comparison operator
    // For now, limit no matter what.
    XYZ = limit_to_primaries( XYZ, LIMITING_PRI); 
    
    // Apply CAT from ACES white point to assumed observer adapted white point
    // TODO: Needs to expand from just supporting D60 sim to allow for any
    // observer adapted white point.
    if (D60_SIM == false) {
        if ((DISPLAY_PRI.white.x != AP0.white.x) &
            (DISPLAY_PRI.white.y != AP0.white.y)) {
            mat3 CAT = calculate_cat_matrix( AP0.white, DISPLAY_PRI.white);
            XYZ = mult_f3_f33( XYZ, D60_2_D65_CAT);
        }
    }

    // CIE XYZ to display encoding primaries
    linearCV = mult_f3_f44( XYZ, XYZ_2_DISPLAY_PRI_MAT);

    // Scale to avoid clipping when device calibration is different from D60. 
    // To simulate D60, unequal code values are sent to the display.
    // TODO: Needs to expand from just supporting D60 sim to allow for any
    // observer adapted white point.
    if (D60_SIM == true) {
        /* TODO: The scale requires calling itself. Scale is same no matter the luminance.
           Currently precalculated for D65, DCI. If DCI, roll_white_fwd is used also.
           This needs a more complex algorithm to handle all cases.
        */
        float SCALE = 1.0f;
        if ((DISPLAY_PRI.white.x == 0.3127f) & 
            (DISPLAY_PRI.white.y == 0.329f)) { // D65
                SCALE = 0.96362f;
        } 
        else if ((DISPLAY_PRI.white.x == 0.314f) & 
                 (DISPLAY_PRI.white.y == 0.351f)) { // DCI
                linearCV.x = roll_white_fwd( linearCV.x, 0.918f, 0.5f);
                linearCV.y = roll_white_fwd( linearCV.y, 0.918f, 0.5f);
                linearCV.z = roll_white_fwd( linearCV.z, 0.918f, 0.5f);
                SCALE = 0.96f;                
        } 
        linearCV = mult_f_f3( SCALE, linearCV);
    }


    // Clip values < 0 (i.e. projecting outside the display primaries)
    // NOTE: P3 red and values close to it fall outside of Rec.2020 green-red 
    // boundary
    linearCV = clamp_f3( linearCV, 0.0f, HALF_POS_INF);

    // EOTF
    // 0: ST-2084 (PQ)
    // 1: BT.1886 (Rec.709/2020 settings)
    // 2: sRGB (mon_curve w/ presets)
    //    moncurve_r with gamma of 2.4 and offset of 0.055 matches the EOTF found in IEC 61966-2-1:1999 (sRGB)
    // 3: gamma 2.6
    // 4: linear (no EOTF)
    // 5: HLG
    float3 outputCV;
    if (EOTF == 0) {  // ST-2084 (PQ)
        // NOTE: This is a kludgy way of ensuring a PQ code value of 0. Ideally,
        // luminance would map directly to code value, but colorists don't like
        // that. Might just need the tonescale to go darker so that darkest
        // values through the tone scale quantize to code value of 0.
        if (STRETCH_BLACK == true) {
            outputCV = Y_2_ST2084_f3( clamp_f3( linCV_2_Y_f3(linearCV, Y_MAX, 0.0f), 0.0f, HALF_POS_INF) );
        } else {
            outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) );        
        }
    } else if (EOTF == 1) { // BT.1886 (Rec.709/2020 settings)
        outputCV = bt1886_r_f3( linearCV, 2.4f, 1.0f, 0.0f);
    } else if (EOTF == 2) { // sRGB (mon_curve w/ presets)
        outputCV = moncurve_r_f3( linearCV, 2.4f, 0.055f);
    } else if (EOTF == 3) { // gamma 2.6
        outputCV = pow_f3( linearCV, 1.0f/2.6f);
    } else if (EOTF == 4) { // linear
        outputCV = linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN);
    } else if (EOTF == 5) { // HLG
        // NOTE: HLG just maps ST.2084 output to HLG encoding. 
        // TODO: Restructure if/else tree to minimize this redundancy.
        if (STRETCH_BLACK == true) {
            outputCV = Y_2_ST2084_f3( clamp_f3( linCV_2_Y_f3(linearCV, Y_MAX, 0.0f), 0.0f, HALF_POS_INF) );
        }
        else {
            outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) );        
        }
        outputCV = ST2084_2_HLG_1000nits( outputCV);
    }

    return outputCV;    
}

__device__ inline float3 invOutputTransform
(
    float3 in,
    float Y_MIN,
    float Y_MID,
    float Y_MAX,    
    Chromaticities DISPLAY_PRI,
    Chromaticities LIMITING_PRI,
    int EOTF,  
    int SURROUND,
    bool STRETCH_BLACK = true,
    bool D60_SIM = false,
    bool LEGAL_RANGE = false
)
{
    mat4 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI, 1.0f);

    // NOTE: This is a bit of a hack - probably a more direct way to do this.
    // Update in accordance with forward algorithm.
    TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX);
    float expShift = log2f(inv_ssts(Y_MID, PARAMS_DEFAULT))- log2f(0.18f);
    TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift);

    float3 outputCV = in;
    // Inverse EOTF
    // 0: ST-2084 (PQ)
    // 1: BT.1886 (Rec.709/2020 settings)
    // 2: sRGB (mon_curve w/ presets)
    //    moncurve_r with gamma of 2.4 and offset of 0.055 matches the EOTF found in IEC 61966-2-1:1999 (sRGB)
    // 3: gamma 2.6
    // 4: linear (no EOTF)
    // 5: HLG
    float3 linearCV;
    if (EOTF == 0) {  // ST-2084 (PQ)
        if (STRETCH_BLACK == true) {
            linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0f);
        } else {
            linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN);
        }
    } else if (EOTF == 1) { // BT.1886 (Rec.709/2020 settings)
        linearCV = bt1886_f_f3( outputCV, 2.4f, 1.0f, 0.0f);
    } else if (EOTF == 2) { // sRGB (mon_curve w/ presets)
        linearCV = moncurve_f_f3( outputCV, 2.4f, 0.055f);
    } else if (EOTF == 3) { // gamma 2.6
        linearCV = pow_f3( outputCV, 2.6f);
    } else if (EOTF == 4) { // linear
        linearCV = Y_2_linCV_f3( outputCV, Y_MAX, Y_MIN);
    } else if (EOTF == 5) { // HLG
        outputCV = HLG_2_ST2084_1000nits( outputCV);
        if (STRETCH_BLACK == true) {
            linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0f);
        } else {
            linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN);
        }
    }

    // Un-scale
    if (D60_SIM == true) {
        /* TODO: The scale requires calling itself. Need an algorithm for this.
            Scale is same no matter the luminance.
            Currently using precalculated values for D65, DCI.
            If DCI, roll_white_fwd is used also.
        */
        float SCALE = 1.0f;
        if ((DISPLAY_PRI.white.x == 0.3127f) & 
            (DISPLAY_PRI.white.y == 0.329f)) { // D65
                SCALE = 0.96362f;
                linearCV = mult_f_f3( 1.0f/SCALE, linearCV);
        } 
        else if ((DISPLAY_PRI.white.x == 0.314f) & 
                 (DISPLAY_PRI.white.y == 0.351f)) { // DCI
                SCALE = 0.96f;                
                linearCV.x = roll_white_rev( linearCV.x/SCALE, 0.918f, 0.5f);
                linearCV.y = roll_white_rev( linearCV.y/SCALE, 0.918f, 0.5f);
                linearCV.z = roll_white_rev( linearCV.z/SCALE, 0.918f, 0.5f);
        } 

    }    

    // Encoding primaries to CIE XYZ
    float3 XYZ = mult_f3_f44( linearCV, DISPLAY_PRI_2_XYZ_MAT);

    // Undo CAT from assumed observer adapted white point to ACES white point
    if (D60_SIM == false) {
        if ((DISPLAY_PRI.white.x != AP0.white.x) &
            (DISPLAY_PRI.white.y != AP0.white.y)) {
            mat3 CAT = calculate_cat_matrix( AP0.white, DISPLAY_PRI.white);
            XYZ = mult_f3_f33( XYZ, invert_f33(D60_2_D65_CAT) );
        }
    }

    // Apply gamma adjustment to compensate for dim surround
//     if (SURROUND == 1) {
//         print( "\nDim surround\n");
//         XYZ = dim_to_dark( XYZ);
//     }

    // XYZ to rendering primaries
    linearCV = mult_f3_f44( XYZ, XYZ_2_AP1_MAT);

    float3 rgbPost = linCV_2_Y_f3( linearCV, Y_MAX, Y_MIN);

    // Apply the inverse tonescale independently in rendering-space RGB
    float3 rgbPre = inv_ssts_f3( rgbPost, PARAMS);

    // RRT sweeteners
    float3 aces = inv_rrt_sweeteners( rgbPre);

    return aces;
}

#endif