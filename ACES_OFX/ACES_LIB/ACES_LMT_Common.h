#ifndef __ACES_LMT_COMMON_H_INCLUDED__
#define __ACES_LMT_COMMON_H_INCLUDED__

//__constant__ float PIE = 3.14159265358979323846264338327950288f;
__constant__ float X_BRK = 0.0078125f;
__constant__ float Y_BRK = 0.155251141552511f;
__constant__ float A = 10.5402377416545f;
__constant__ float B = 0.0729055341958355f;

__device__ inline float lin_to_ACEScct( float in)
{
    if (in <= X_BRK)
        return A * in + B;
    else // (in > X_BRK)
        return (log2f(in) + 9.72f) / 17.52f;
}

__device__ inline float ACEScct_to_lin( float in)
{
    if (in > Y_BRK)
        return powf( 2.0f, in * 17.52f - 9.72f);
    else
        return (in - B) / A;
}

__device__ inline float3 ACES_to_ACEScct( float3 in)
{
    // AP0 to AP1
    float3 ap1_lin = mult_f3_f44( in, AP0_2_AP1_MAT);

    // Linear to ACEScct
    float3 acescct;
    acescct.x = lin_to_ACEScct( ap1_lin.x);
    acescct.y = lin_to_ACEScct( ap1_lin.y);
    acescct.z = lin_to_ACEScct( ap1_lin.z);
    
    return acescct;
}

__device__ inline float3 ACEScct_to_ACES( float3 in)
{
    // ACEScct to linear
    float3 ap1_lin;
    ap1_lin.x = ACEScct_to_lin( in.x);
    ap1_lin.y = ACEScct_to_lin( in.y);
    ap1_lin.z = ACEScct_to_lin( in.z);

    // AP1 to AP0
    return mult_f3_f44( ap1_lin, AP1_2_AP0_MAT);
}

__device__ inline float3 ASCCDL_inACEScct
(
    float3 acesIn, 
    float3 SLOPE = make_float3(1.0f, 1.0f, 1.0f),
    float3 OFFSET = make_float3(0.0f, 0.0f, 0.0f),
    float3 POWER = make_float3(1.0f, 1.0f, 1.0f),
    float SAT = 1.0f
)
{
    // Convert ACES to ACEScct
    float3 acescct = ACES_to_ACEScct( acesIn);

    // ASC CDL
    // Slope, Offset, Power
    acescct.x = powf( clamp( (acescct.x * SLOPE.x) + OFFSET.x, 0.0f, 1.0f), 1.0f / POWER.x);
    acescct.y = powf( clamp( (acescct.y * SLOPE.y) + OFFSET.y, 0.0f, 1.0f), 1.0f / POWER.y);
    acescct.z = powf( clamp( (acescct.z * SLOPE.z) + OFFSET.z, 0.0f, 1.0f), 1.0f / POWER.z);
    
    // Saturation
    float luma = 0.2126f * acescct.x + 0.7152f * acescct.y + 0.0722f * acescct.z;

    float satClamp = clamp( SAT, 0.0f, 31744.0f); //HALF_POS_INF);    
    acescct.x = luma + satClamp * (acescct.x - luma);
    acescct.y = luma + satClamp * (acescct.y - luma);
    acescct.z = luma + satClamp * (acescct.z - luma);

    // Convert ACEScct to ACES
    return ACEScct_to_ACES( acescct);
}

__device__ inline float3 gamma_adjust_linear( 
    float3 rgbIn, 
    float GAMMA, 
    float PIVOT = 0.18f
)
{
    const float SCALAR = PIVOT / powf( PIVOT, GAMMA);

    float3 rgbOut = rgbIn;
    if (rgbIn.x > 0) rgbOut.x = powf( rgbIn.x, GAMMA) * SCALAR;
    if (rgbIn.y > 0) rgbOut.y = powf( rgbIn.y, GAMMA) * SCALAR;
    if (rgbIn.z > 0) rgbOut.z = powf( rgbIn.z, GAMMA) * SCALAR;

    return rgbOut;
}
                                
#define REC709_2_XYZ_MAT 			RGBtoXYZ( REC709_PRI, 1.0f)   
#define REC709_RGB2Y 				make_float3(REC709_2_XYZ_MAT.c0.y, REC709_2_XYZ_MAT.c1.y, REC709_2_XYZ_MAT.c2.y)

__device__ inline float3 sat_adjust(
    float3 rgbIn,
    float SAT_FACTOR,
    float3 RGB2Y = REC709_RGB2Y
)
{
    const mat3 SAT_MAT = calc_sat_adjust_matrix( SAT_FACTOR, RGB2Y);    

    return mult_f3_f33( rgbIn, SAT_MAT);
}

// RGB / YAB / YCH conversion functions
// "YAB" is a geometric space of a unit cube rotated so its neutral axis run vertically
// "YCH" is a cylindrical representation of this, where C is the "chroma" and H is "hue"
// These are geometrically defined via ratios of RGB and are not intended to have any
// correlation to perceptual luminance, chrominance, or hue.

__constant__ float sqrt3over4 = 0.433012701892219f;  // sqrt(3.)/4.

#define RGB_2_YAB_MAT  make_mat3(make_float3(1.0f/3.0f, 1.0f/2.0f, 0.0f), make_float3(1.0f/3.0f, -1.0f/4.0f, sqrt3over4), make_float3(1.0f/3.0f, -1.0f/4.0f, -sqrt3over4))

__device__ inline float3 rgb_2_yab( float3 rgb)
{
  float3 yab = mult_f3_f33( rgb, RGB_2_YAB_MAT);

  return yab;
}

__device__ inline float3 yab_2_rgb( float3 yab)
{
  float3 rgb = mult_f3_f33( yab, invert_f33(RGB_2_YAB_MAT));

  return rgb;
}

__device__ float3 yab_2_ych(float3 yab)
{
float3 ych = yab;
float yo = yab.y * yab.y + yab.z * yab.z;
ych.y = sqrtf(yo);
ych.z = atan2f(yab.z, yab.y) * (180.0f / M_PI);
if (ych.z < 0.0f) ych.z += 360.0f;

return ych;
}

__device__ inline float3 ych_2_yab( float3 ych ) 
{
  float3 yab;
  yab.x = ych.x;

  float h = ych.z * (M_PI / 180.0f);
  yab.y = ych.y * cosf(h);
  yab.z = ych.y * sinf(h);

  return yab;
}

__device__ inline float3 rgb_2_ych( float3 rgb) 
{
  return yab_2_ych( rgb_2_yab( rgb));
}

__device__ inline float3 ych_2_rgb( float3 ych) 
{
  return yab_2_rgb( ych_2_yab( ych));
}

// Regions of hue are targeted using a cubic basis shaper function. The controls for 
// the shape of this function are the center/peak (in degrees), and the full width 
// (in degrees) at the base. Values in the center of the function get 1.0 of an 
// adjustment while values at the tails of the function get 0.0 adjustment.
//
// For the purposes of tuning, the hues are located at the following hue angles:
//   Y = 60
//   G = 120
//   C = 180
//   B = 240
//   M = 300
//   R = 360 / 0
__device__ inline float3 scale_C_at_H
( 
    float3 rgb, 
    float centerH,   // center of targeted hue region (in degrees)
    float widthH,    // full width at base of targeted hue region (in degrees)
    float percentC   // percentage of scale: 1.0 is no adjustment (i.e. 100%)
)
{
    float3 new_rgb = rgb;
    
    float3 ych = rgb_2_ych( rgb);

    if (ych.y > 0.0f) {  // Only do the chroma adjustment if pixel is non-neutral

        float centeredHue = center_hue( ych.z, centerH);
        float f_H = cubic_basis_shaper( centeredHue, widthH);

        if (f_H > 0.0f) {
            // Scale chroma in affected hue region
            float3 new_ych = ych;
            new_ych.y = ych.y * (f_H * (percentC - 1.0f) + 1.0f);
            new_rgb = ych_2_rgb( new_ych);
        } else { 
            // If not in affected hue region, just return original values
            // This helps to avoid precision errors that can occur in the RGB->YCH->RGB 
            // conversion
            new_rgb = rgb; 
        }
    }

    return new_rgb;
}


// Regions of hue are targeted using a cubic basis shaper function. The controls for 
// the shape of this function are the center/peak (in degrees), and the full width 
// (in degrees) at the base. Values in the center of the function get 1.0 of an 
// adjustment while values at the tails of the function get 0.0 adjustment.
//
// For the purposes of tuning, the hues are located at the following hue angles:
//   Y = 60
//   G = 120
//   C = 180
//   B = 240
//   M = 300
//   R = 360 / 0
__device__ inline float3 rotate_H_in_H
( 
    float3 rgb,
    float centerH,        // center of targeted hue region (in degrees)
    float widthH,         // full width at base of targeted hue region (in degrees)
    float degreesShift    // how many degrees (w/ sign) to rotate hue
)
{
    float3 ych = rgb_2_ych( rgb);
    float3 new_ych = ych;

    float centeredHue = center_hue( ych.z, centerH);
    float f_H = cubic_basis_shaper( centeredHue, widthH);

    float old_hue = centeredHue;
    float new_hue = centeredHue + degreesShift;
    float2 table[2] = { {0.0f, old_hue}, 
                          {1.0f, new_hue} };
    float blended_hue = interpolate1D( table, f_H);
        
    if (f_H > 0.0f) new_ych.z = uncenter_hue(blended_hue, centerH);
    
    return ych_2_rgb( new_ych);
}

__device__ inline float3 scale_C( 
    float3 rgb, 
    float percentC      // < 1 is a decrease, 1.0 is unchanged, > 1 is an increase
)
{
    float3 ych = rgb_2_ych( rgb);
    ych.y = ych.y * percentC;
    
    return ych_2_rgb( ych);
}

__device__ inline float3 overlay_f3( float3 a, float3 b)
{
const float LUMA_CUT = lin_to_ACEScct( 0.5f); 

float luma = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;

float3 out;
if (luma < LUMA_CUT) {
out.x = 2.0f * a.x * b.x;
out.y = 2.0f * a.y * b.y;
out.z = 2.0f * a.z * b.z;
} else {
out.x = 1.0f - (2.0f * (1.0f - a.x) * (1.0f - b.x));
out.y = 1.0f - (2.0f * (1.0f - a.y) * (1.0f - b.y));
out.z = 1.0f - (2.0f * (1.0f - a.z) * (1.0f - b.z));
}

return out;
}


#endif