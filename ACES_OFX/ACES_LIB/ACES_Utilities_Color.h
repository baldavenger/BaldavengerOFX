#ifndef __ACES_UTILITIES_COLOR_H_INCLUDED__
#define __ACES_UTILITIES_COLOR_H_INCLUDED__

//
// Color related constants and functions
//

__constant__ Chromaticities AP0 = // ACES Primaries from SMPTE ST2065-1
{
  { 0.73470,  0.26530},
  { 0.00000,  1.00000},
  { 0.00010, -0.07700},
  { 0.32168,  0.33767}
};

__constant__ Chromaticities AP1 = // Working space and rendering primaries for ACES 1.0
{
  {0.713,   0.293},
  {0.165,   0.830},
  {0.128,   0.044},
  {0.32168, 0.33767}
};

__constant__ Chromaticities REC709_PRI =
{
  { 0.64000,  0.33000},
  { 0.30000,  0.60000},
  { 0.15000,  0.06000},
  { 0.31270,  0.32900}
};

__constant__ Chromaticities P3D60_PRI =
{
  { 0.68000,  0.32000},
  { 0.26500,  0.69000},
  { 0.15000,  0.06000},
  { 0.32168,  0.33767}
};

__constant__ Chromaticities P3D65_PRI =
{
  { 0.68000,  0.32000},
  { 0.26500,  0.69000},
  { 0.15000,  0.06000},
  { 0.31270,  0.32900}
};

__constant__ Chromaticities P3DCI_PRI =
{
  { 0.68000,  0.32000},
  { 0.26500,  0.69000},
  { 0.15000,  0.06000},
  { 0.31400,  0.35100}
};

__constant__ Chromaticities ARRI_ALEXA_WG_PRI =
{
  {0.68400, 0.31300},
  {0.22100, 0.84800},
  {0.08610, -0.10200},
  {0.31270, 0.32900}
};

__constant__ Chromaticities REC2020_PRI = 
{
  {0.70800, 0.29200},
  {0.17000, 0.79700},
  {0.13100, 0.04600},
  {0.31270, 0.32900}
};

__constant__ Chromaticities RIMMROMM_PRI = 
{
  {0.7347, 0.2653},
  {0.1596, 0.8404},
  {0.0366, 0.0001},
  {0.3457, 0.3585}
};


/* ---- Conversion Functions ---- */
// Various transformations between color encodings and data representations
//

// Transformations between CIE XYZ tristimulus values and CIE x,y 
// chromaticity coordinates
__device__ inline float3 XYZ_2_xyY( float3 XYZ)
{  
  float3 xyY;
  float divisor = (XYZ.x + XYZ.y + XYZ.z);
  if (divisor == 0.0f) divisor = 1e-10;
  xyY.x = XYZ.x / divisor;
  xyY.y = XYZ.y / divisor;  
  xyY.z = XYZ.y;
  
  return xyY;
}

__device__ inline float3 xyY_2_XYZ( float3 xyY)
{
  float3 XYZ;
  XYZ.x = xyY.x * xyY.z / fmax( xyY.y, 1e-10);
  XYZ.y = xyY.z;  
  XYZ.z = (1.0f - xyY.x - xyY.y) * xyY.z / fmax( xyY.y, 1e-10);

  return XYZ;
}


// Transformations from RGB to other color representations
__device__ inline float rgb_2_hue( float3 rgb) 
{
  // Returns a geometric hue angle in degrees (0-360) based on RGB values.
  // For neutral colors, hue is undefined and the function will return a quiet NaN value.
  float hue;
  if (rgb.x == rgb.y && rgb.y == rgb.z) {
    hue = 0.0f;//FLT_NAN; // RGB triplets where RGB are equal have an undefined hue
  } else {
    hue = (180.0f/M_PI) * atan2f( sqrtf(3.0f)*(rgb.y-rgb.z), 2*rgb.x-rgb.y-rgb.z);
  }
    
  if (hue < 0.0f) hue = hue + 360.0f;

  return hue;
}

__device__ inline float rgb_2_yc( float3 rgb, float ycRadiusWeight = 1.75f)
{
  // Converts RGB to a luminance proxy, here called YC
  // YC is ~ Y + K * Chroma
  // Constant YC is a cone-shaped surface in RGB space, with the tip on the 
  // neutral axis, towards white.
  // YC is normalized: RGB 1 1 1 maps to YC = 1
  //
  // ycRadiusWeight defaults to 1.75, although can be overridden in function 
  // call to rgb_2_yc
  // ycRadiusWeight = 1 -> YC for pure cyan, magenta, yellow == YC for neutral 
  // of same value
  // ycRadiusWeight = 2 -> YC for pure red, green, blue  == YC for  neutral of 
  // same value.

  float r = rgb.x; 
  float g = rgb.y; 
  float b = rgb.z;
  
  float chroma = sqrtf(b*(b-g)+g*(g-r)+r*(r-b));

  return ( b + g + r + ycRadiusWeight * chroma) / 3.0f;
}


/* ---- Chromatic Adaptation ---- */

__constant__ mat3 CONE_RESP_MAT_BRADFORD = {
  { 0.89510f, -0.75020f,  0.03890f},
  { 0.26640f,  1.71350f, -0.06850f},
  {-0.16140f,  0.03670f,  1.02960f}
};

__constant__ mat3 CONE_RESP_MAT_CAT02 = {
  { 0.73280f, -0.70360f,  0.00300f},
  { 0.42960f,  1.69750f,  0.01360f},
  {-0.16240f,  0.00610f,  0.98340f}
};

__device__ inline mat3 calculate_cat_matrix
  ( float2 src_xy,         // x,y chromaticity of source white
    float2 des_xy,         // x,y chromaticity of destination white
    mat3 coneRespMat = CONE_RESP_MAT_BRADFORD
  )
{
  //
  // Calculates and returns a 3x3 Von Kries chromatic adaptation transform 
  // from src_xy to des_xy using the cone response primaries defined 
  // by coneRespMat. By default, coneRespMat is set to CONE_RESP_MAT_BRADFORD. 
  // The default coneRespMat can be overridden at runtime. 
  //

  const float3 src_xyY = { src_xy.x, src_xy.y, 1.0f };
  const float3 des_xyY = { des_xy.x, des_xy.y, 1.0f };

  float3 src_XYZ = xyY_2_XYZ( src_xyY );
  float3 des_XYZ = xyY_2_XYZ( des_xyY );

  float3 src_coneResp = mult_f3_f33( src_XYZ, coneRespMat);
  float3 des_coneResp = mult_f3_f33( des_XYZ, coneRespMat);

  mat3 vkMat = {
      { des_coneResp.x / src_coneResp.x, 0.0f, 0.0f },
      { 0.0f, des_coneResp.y / src_coneResp.y, 0.0f },
      { 0.0f, 0.0f, des_coneResp.z / src_coneResp.z }
  };

  mat3 cat_matrix = mult_f33_f33( coneRespMat, mult_f33_f33( vkMat, invert_f33( coneRespMat ) ) );

  return cat_matrix;
}

__device__ inline mat3 calc_sat_adjust_matrix 
  ( float sat,
    float3 rgb2Y
  )
{
  //
  // This function determines the terms for a 3x3 saturation matrix that is
  // based on the luminance of the input.
  //
  float M[3][3];
  M[0][0] = (1.0f - sat) * rgb2Y.x + sat;
  M[1][0] = (1.0f - sat) * rgb2Y.x;
  M[2][0] = (1.0f - sat) * rgb2Y.x;
  
  M[0][1] = (1.0f - sat) * rgb2Y.y;
  M[1][1] = (1.0f - sat) * rgb2Y.y + sat;
  M[2][1] = (1.0f - sat) * rgb2Y.y;
  
  M[0][2] = (1.0f - sat) * rgb2Y.z;
  M[1][2] = (1.0f - sat) * rgb2Y.z;
  M[2][2] = (1.0f - sat) * rgb2Y.z + sat;
  
  mat3 R = make_mat3(make_float3(M[0][0], M[0][1], M[0][2]), 
  make_float3(M[1][0], M[1][1], M[1][2]), make_float3(M[2][0], M[2][1], M[2][2]));

  R = transpose_f33(R);    
  return R;
} 

/* ---- Signal encode/decode functions ---- */

__device__ inline float moncurve_f( float x, float gamma, float offs )
{
  // Forward monitor curve
  float y;
  const float fs = (( gamma - 1.0f) / offs) * powf( offs * gamma / ( ( gamma - 1.0f) * ( 1.0f + offs)), gamma);
  const float xb = offs / ( gamma - 1.0f);
  if ( x >= xb) 
    y = powf( ( x + offs) / ( 1.0f + offs), gamma);
  else
    y = x * fs;
  return y;
}

__device__ inline float moncurve_r( float y, float gamma, float offs )
{
  // Reverse monitor curve
  float x;
  const float yb = powf( offs * gamma / ( ( gamma - 1.0f) * ( 1.0f + offs)), gamma);
  const float rs = powf( ( gamma - 1.0f) / offs, gamma - 1.0f) * powf( ( 1.0f + offs) / gamma, gamma);
  if ( y >= yb) 
    x = ( 1.0f + offs) * powf( y, 1.0f / gamma) - offs;
  else
    x = y * rs;
  return x;
}

__device__ inline float3 moncurve_f_f3( float3 x, float gamma, float offs)
{
    float3 y;
    y.x = moncurve_f( x.x, gamma, offs);
    y.y = moncurve_f( x.y, gamma, offs);
    y.z = moncurve_f( x.z, gamma, offs);
    return y;
}

__device__ inline float3 moncurve_r_f3( float3 y, float gamma, float offs)
{
    float3 x;
    x.x = moncurve_r( y.x, gamma, offs);
    x.y = moncurve_r( y.y, gamma, offs);
    x.z = moncurve_r( y.z, gamma, offs);
    return x;
}

__device__ inline float bt1886_f( float V, float gamma, float Lw, float Lb)
{
  // The reference EOTF specified in Rec. ITU-R BT.1886
  // L = a(max[(V+b),0])^g
  float a = powf( powf( Lw, 1.0f/gamma) - powf( Lb, 1.0f/gamma), gamma);
  float b = powf( Lb, 1.0f/gamma) / ( powf( Lw, 1.0f/gamma) - powf( Lb, 1.0f/gamma));
  float L = a * powf( fmax( V + b, 0.0f), gamma);
  return L;
}

__device__ inline float bt1886_r( float L, float gamma, float Lw, float Lb)
{
  // The reference EOTF specified in Rec. ITU-R BT.1886
  // L = a(max[(V+b),0])^g
  float a = powf( powf( Lw, 1.0f/gamma) - powf( Lb, 1.0f/gamma), gamma);
  float b = powf( Lb, 1.0f/gamma) / ( powf( Lw, 1.0f/gamma) - powf( Lb, 1.0f/gamma));
  float V = powf( fmax( L / a, 0.0f), 1.0f/gamma) - b;
  return V;
}

__device__ inline float3 bt1886_f_f3( float3 V, float gamma, float Lw, float Lb)
{
    float3 L;
    L.x = bt1886_f( V.x, gamma, Lw, Lb);
    L.y = bt1886_f( V.y, gamma, Lw, Lb);
    L.z = bt1886_f( V.z, gamma, Lw, Lb);
    return L;
}

__device__ inline float3 bt1886_r_f3( float3 L, float gamma, float Lw, float Lb)
{
    float3 V;
    V.x = bt1886_r( L.x, gamma, Lw, Lb);
    V.y = bt1886_r( L.y, gamma, Lw, Lb);
    V.z = bt1886_r( L.z, gamma, Lw, Lb);
    return V;
}


// SMPTE Range vs Full Range scaling formulas
__device__ inline float smpteRange_to_fullRange( float in)
{
    const float REFBLACK = (  64.0f / 1023.0f);
    const float REFWHITE = ( 940.0f / 1023.0f);

    return (( in - REFBLACK) / ( REFWHITE - REFBLACK));
}

__device__ inline float fullRange_to_smpteRange( float in)
{
    const float REFBLACK = (  64.0f / 1023.0f);
    const float REFWHITE = ( 940.0f / 1023.0f);

    return ( in * ( REFWHITE - REFBLACK) + REFBLACK );
}

__device__ inline float3 smpteRange_to_fullRange_f3( float3 rgbIn)
{
    float3 rgbOut;
    rgbOut.x = smpteRange_to_fullRange( rgbIn.x);
    rgbOut.y = smpteRange_to_fullRange( rgbIn.y);
    rgbOut.z = smpteRange_to_fullRange( rgbIn.z);

    return rgbOut;
}

__device__ inline float3 fullRange_to_smpteRange_f3( float3 rgbIn)
{
    float3 rgbOut;
    rgbOut.x = fullRange_to_smpteRange( rgbIn.x);
    rgbOut.y = fullRange_to_smpteRange( rgbIn.y);
    rgbOut.z = fullRange_to_smpteRange( rgbIn.z);

    return rgbOut;
}


// SMPTE 431-2 defines the DCDM color encoding equations. 
// The equations for the decoding of the encoded color information are the 
// inverse of the encoding equations
// Note: Here the 4095 12-bit scalar is not used since the output of CTL is 0-1.
__device__ inline float3 dcdm_decode( float3 XYZp)
{
    float3 XYZ;
    XYZ.x = (52.37f/48.0f) * powf( XYZp.x, 2.6f);  
    XYZ.y = (52.37f/48.0f) * powf( XYZp.y, 2.6f);  
    XYZ.z = (52.37f/48.0f) * powf( XYZp.z, 2.6f);  

    return XYZ;
}

__device__ inline float3 dcdm_encode( float3 XYZ)
{
    float3 XYZp;
    XYZp.x = powf( (48.0f/52.37f) * XYZ.x, 1.0f/2.6f);
    XYZp.y = powf( (48.0f/52.37f) * XYZ.y, 1.0f/2.6f);
    XYZp.z = powf( (48.0f/52.37f) * XYZ.z, 1.0f/2.6f);

    return XYZp;
}


// Base functions from SMPTE ST 2084-2014

// Constants from SMPTE ST 2084-2014
__constant__ float pq_m1 = 0.1593017578125f; // ( 2610.0 / 4096.0 ) / 4.0;
__constant__ float pq_m2 = 78.84375f; // ( 2523.0 / 4096.0 ) * 128.0;
__constant__ float pq_c1 = 0.8359375f; // 3424.0 / 4096.0 or pq_c3 - pq_c2 + 1.0;
__constant__ float pq_c2 = 18.8515625f; // ( 2413.0 / 4096.0 ) * 32.0;
__constant__ float pq_c3 = 18.6875f; // ( 2392.0 / 4096.0 ) * 32.0;

__constant__ float pq_C = 10000.0f;

// Converts from the non-linear perceptually quantized space to linear cd/m^2
// Note that this is in float, and assumes normalization from 0 - 1
// (0 - pq_C for linear) and does not handle the integer coding in the Annex 
// sections of SMPTE ST 2084-2014
__device__ inline float ST2084_2_Y( float N )
{
  // Note that this does NOT handle any of the signal range
  // considerations from 2084 - this assumes full range (0 - 1)
  float Np = powf( N, 1.0f / pq_m2 );
  float L = Np - pq_c1;
  if ( L < 0.0f )
    L = 0.0f;
  L = L / ( pq_c2 - pq_c3 * Np );
  L = powf( L, 1.0f / pq_m1 );
  return L * pq_C; // returns cd/m^2
}

// Converts from linear cd/m^2 to the non-linear perceptually quantized space
// Note that this is in float, and assumes normalization from 0 - 1
// (0 - pq_C for linear) and does not handle the integer coding in the Annex 
// sections of SMPTE ST 2084-2014
__device__ inline float Y_2_ST2084( float C )
//pq_r
{
  // Note that this does NOT handle any of the signal range
  // considerations from 2084 - this returns full range (0 - 1)
  float L = C / pq_C;
  float Lm = powf( L, pq_m1 );
  float N = ( pq_c1 + pq_c2 * Lm ) / ( 1.0f + pq_c3 * Lm );
  N = powf( N, pq_m2 );
  return N;
}

__device__ inline float3 Y_2_ST2084_f3( float3 in)
{
  // converts from linear cd/m^2 to PQ code values
  
  float3 out;
  out.x = Y_2_ST2084( in.x);
  out.y = Y_2_ST2084( in.y);
  out.z = Y_2_ST2084( in.z);

  return out;
}

__device__ inline float3 ST2084_2_Y_f3( float3 in)
{
  // converts from PQ code values to linear cd/m^2
  
  float3 out;
  out.x = ST2084_2_Y( in.x);
  out.y = ST2084_2_Y( in.y);
  out.z = ST2084_2_Y( in.z);

  return out;
}

// Conversion of PQ signal to HLG, as detailed in Section 7 of ITU-R BT.2390-0
__device__ inline float3 ST2084_2_HLG_1000nits( float3 PQ) 
{
    // ST.2084 EOTF (non-linear PQ to display light)
    float3 displayLinear = ST2084_2_Y_f3( PQ);

    // HLG Inverse EOTF (i.e. HLG inverse OOTF followed by the HLG OETF)
    // HLG Inverse OOTF (display linear to scene linear)
    float Y_d = 0.2627f * displayLinear.x + 0.6780f * displayLinear.y + 0.0593f * displayLinear.z;
    const float L_w = 1000.0f;
    const float L_b = 0.0f;
    const float alpha = (L_w-L_b);
    const float beta = L_b;
    const float gamma = 1.2f;
    
    float3 sceneLinear;
    sceneLinear.x = powf( (Y_d-beta)/alpha, (1.0f - gamma)/gamma) * ((displayLinear.x-beta)/alpha);
    sceneLinear.y = powf( (Y_d-beta)/alpha, (1.0f - gamma)/gamma) * ((displayLinear.y-beta)/alpha);
    sceneLinear.z = powf( (Y_d-beta)/alpha, (1.0f - gamma)/gamma) * ((displayLinear.z-beta)/alpha);

    // HLG OETF (scene linear to non-linear signal value)
    const float a = 0.17883277f;
    const float b = 0.28466892f; // 1.-4.*a;
    const float c = 0.55991073f; // 0.5-a*logf(4.*a);

    float3 HLG;
    if (sceneLinear.x <= 1.0f/12) {
        HLG.x = sqrtf(3.0f * sceneLinear.x);
    } else {
        HLG.x = a*logf(12.0f * sceneLinear.x-b)+c;
    }
    if (sceneLinear.y <= 1.0f/12) {
        HLG.y = sqrtf(3.0f * sceneLinear.y);
    } else {
        HLG.y = a*logf(12.0f * sceneLinear.y-b)+c;
    }
    if (sceneLinear.z <= 1.0f/12) {
        HLG.z = sqrtf(3.0f * sceneLinear.z);
    } else {
        HLG.z = a*logf(12.0f * sceneLinear.z-b)+c;
    }

    return HLG;
}


// Conversion of HLG to PQ signal, as detailed in Section 7 of ITU-R BT.2390-0
__device__ inline float3 HLG_2_ST2084_1000nits( float3 HLG) 
{
    const float a = 0.17883277f;
    const float b = 0.28466892f; // 1.-4.*a;
    const float c = 0.55991073f; // 0.5-a*logf(4.*a);

    const float L_w = 1000.0f;
    const float L_b = 0.0f;
    const float alpha = (L_w-L_b);
    const float beta = L_b;
    const float gamma = 1.2f;

    // HLG EOTF (non-linear signal value to display linear)
    // HLG to scene-linear
    float3 sceneLinear;
    if ( HLG.x >= 0.0f && HLG.x <= 0.5f) {
        sceneLinear.x = powf(HLG.x,2.0f)/3.0f;
    } else {
        sceneLinear.x = (expf((HLG.x-c)/a)+b)/12.0f;
    }        
    if ( HLG.y >= 0.0f && HLG.y <= 0.5f) {
        sceneLinear.y = powf(HLG.y,2.0f)/3.0f;
    } else {
        sceneLinear.y = (expf((HLG.y-c)/a)+b)/12.0f;
    }        
    if ( HLG.z >= 0.0f && HLG.z <= 0.5f) {
        sceneLinear.z = powf(HLG.z,2.0f)/3.0f;
    } else {
        sceneLinear.z = (expf((HLG.z-c)/a)+b)/12.0f;
    }        
    
    float Y_s = 0.2627f * sceneLinear.x + 0.6780f * sceneLinear.y + 0.0593f * sceneLinear.z;

    // Scene-linear to display-linear
    float3 displayLinear;
    displayLinear.x = alpha * powf( Y_s, gamma - 1.0f) * sceneLinear.x + beta;
    displayLinear.y = alpha * powf( Y_s, gamma - 1.0f) * sceneLinear.y + beta;
    displayLinear.z = alpha * powf( Y_s, gamma - 1.0f) * sceneLinear.z + beta;
        
    // ST.2084 Inverse EOTF
    float3 PQ = Y_2_ST2084_f3( displayLinear);

    return PQ;
}

#endif