#ifndef __ACES_ODT_COMMON_H_INCLUDED__
#define __ACES_ODT_COMMON_H_INCLUDED__

//
// Contains functions and constants shared by forward and inverse ODT transforms 
//


// Target white and black points for cinema system tonescale
#define CINEMA_WHITE		48.0f
#define CINEMA_BLACK		powf(10.0f, log10f(0.02f)) // CINEMA_WHITE / 2400. 
    // CINEMA_BLACK is defined in this roundabout manner in order to be exactly equal to 
    // the result returned by the cinema 48-nit ODT tonescale.
    // Though the min point of the tonescale is designed to return 0.02, the tonescale is 
    // applied in log-log space, which loses precision on the antilog. The tonescale 
    // return value is passed into Y_2_linCV, where CINEMA_BLACK is subtracted. If 
    // CINEMA_BLACK is defined as simply 0.02, then the return value of this subfunction
    // is very, very small but not equal to 0, and attaining a CV of 0 is then impossible.
    // For all intents and purposes, CINEMA_BLACK=0.02.


// Gamma compensation factor
__constant__ float DIM_SURROUND_GAMMA = 0.9811f;

// Saturation compensation factor
__constant__ float ODT_SAT_FACTOR = 0.93f;
#define ODT_SAT_MAT					calc_sat_adjust_matrix( ODT_SAT_FACTOR, AP1_RGB2Y)


#define D60_2_D65_CAT				calculate_cat_matrix( AP0.white, REC709_PRI.white)


__device__ inline float Y_2_linCV( float Y, float Ymax, float Ymin) 
{
  return (Y - Ymin) / (Ymax - Ymin);
}

__device__ inline float linCV_2_Y( float linCV, float Ymax, float Ymin) 
{
  return linCV * (Ymax - Ymin) + Ymin;
}

__device__ inline float3 Y_2_linCV_f3( float3 Y, float Ymax, float Ymin)
{
    float3 linCV;
    linCV.x = Y_2_linCV( Y.x, Ymax, Ymin);
    linCV.y = Y_2_linCV( Y.y, Ymax, Ymin);
    linCV.z = Y_2_linCV( Y.z, Ymax, Ymin);
    return linCV;
}

__device__ inline float3 linCV_2_Y_f3( float3 linCV, float Ymax, float Ymin)
{
    float3 Y;
    Y.x = linCV_2_Y( linCV.x, Ymax, Ymin);
    Y.y = linCV_2_Y( linCV.y, Ymax, Ymin);
    Y.z = linCV_2_Y( linCV.z, Ymax, Ymin);
    return Y;
}

__device__ inline float3 darkSurround_to_dimSurround( float3 linearCV)
{
  float3 XYZ = mult_f3_f44( linearCV, AP1_2_XYZ_MAT); 

  float3 xyY = XYZ_2_xyY(XYZ);
  xyY.z = clamp( xyY.z, 0.0f, HALF_POS_INF);
  xyY.z = powf( xyY.z, DIM_SURROUND_GAMMA);
  XYZ = xyY_2_XYZ(xyY);

  return mult_f3_f44( XYZ, XYZ_2_AP1_MAT);
}

__device__ inline float3 dimSurround_to_darkSurround( float3 linearCV)
{
  float3 XYZ = mult_f3_f44( linearCV, AP1_2_XYZ_MAT); 

  float3 xyY = XYZ_2_xyY(XYZ);
  xyY.z = clamp( xyY.z, 0.0f, HALF_POS_INF);
  xyY.z = powf( xyY.z, 1.0f/DIM_SURROUND_GAMMA);
  XYZ = xyY_2_XYZ(xyY);

  return mult_f3_f44( XYZ, XYZ_2_AP1_MAT);
}

/* ---- Functions to compress highlights ---- */
// allow for simulated white points without clipping

__device__ inline float roll_white_fwd( 
    float in,      // color value to adjust (white scaled to around 1.0)
    float new_wht, // white adjustment (e.g. 0.9 for 10% darkening)
    float width    // adjusted width (e.g. 0.25 for top quarter of the tone scale)
  )
{
    const float x0 = -1.0f;
    const float x1 = x0 + width;
    const float y0 = -new_wht;
    const float y1 = x1;
    const float m1 = (x1 - x0);
    const float a = y0 - y1 + m1;
    const float b = 2.0f * ( y1 - y0) - m1;
    const float c = y0;
    const float t = (-in - x0) / (x1 - x0);
    float out = 0.0f;
    if ( t < 0.0f)
        out = -(t * b + c);
    else if ( t > 1.0f)
        out = in;
    else
        out = -(( t * a + b) * t + c);
    return out;
}

__device__ inline float roll_white_rev( 
    float in,      // color value to adjust (white scaled to around 1.0)
    float new_wht, // white adjustment (e.g. 0.9 for 10% darkening)
    float width    // adjusted width (e.g. 0.25 for top quarter of the tone scale)
  )
{
    const float x0 = -1.0f;
    const float x1 = x0 + width;
    const float y0 = -new_wht;
    const float y1 = x1;
    const float m1 = (x1 - x0);
    const float a = y0 - y1 + m1;
    const float b = 2.0f * ( y1 - y0) - m1;
    float c = y0;
    float out = 0.0f;
    if ( -in < y0)
        out = -x0;
    else if ( -in > y1)
        out = in;
    else {
        c = c + in;
        const float discrim = sqrtf( b * b - 4.0f * a * c);
        const float t = ( 2.0f * c) / ( -discrim - b);
        out = -(( t * ( x1 - x0)) + x0);
    }
    return out;
}

#endif