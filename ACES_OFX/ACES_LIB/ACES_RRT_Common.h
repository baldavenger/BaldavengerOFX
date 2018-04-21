#ifndef __ACES_RRT_COMMON_H_INCLUDED__
#define __ACES_RRT_COMMON_H_INCLUDED__

//
// Contains functions and __constant__ants shared by forward and inverse RRT transforms
//


// "Glow" module __constant__ants
__constant__ float RRT_GLOW_GAIN = 0.05f;
__constant__ float RRT_GLOW_MID = 0.08f;

// Red modifier __constant__ants
__constant__ float RRT_RED_SCALE = 0.82f;
__constant__ float RRT_RED_PIVOT = 0.03f;
__constant__ float RRT_RED_HUE = 0.0f;
__constant__ float RRT_RED_WIDTH = 135.0f;

// Desaturation contants
__constant__ float RRT_SAT_FACTOR = 0.96f;

#define RRT_SAT_MAT 			calc_sat_adjust_matrix( RRT_SAT_FACTOR, AP1_RGB2Y)

// ------- Glow module functions
__device__ inline float glow_fwd( float ycIn, float glowGainIn, float glowMid)
{
   float glowGainOut;

   if (ycIn <= 2.0f/3.0f * glowMid) {
     glowGainOut = glowGainIn;
   } else if ( ycIn >= 2.0f * glowMid) {
     glowGainOut = 0.0f;
   } else {
     glowGainOut = glowGainIn * (glowMid / ycIn - 1.0f/2.0f);
   }

   return glowGainOut;
}

__device__ inline float glow_inv( float ycOut, float glowGainIn, float glowMid)
{
    float glowGainOut;

    if (ycOut <= ((1 + glowGainIn) * 2.0f/3.0f * glowMid)) {
    glowGainOut = -glowGainIn / (1 + glowGainIn);
    } else if ( ycOut >= (2.0f * glowMid)) {
    glowGainOut = 0.0f;
    } else {
    glowGainOut = glowGainIn * (glowMid / ycOut - 1.0f/2.0f) / (glowGainIn / 2.0f - 1.0f);
    }

    return glowGainOut;
}

__device__ inline float sigmoid_shaper( float x)
{
    // Sigmoid function in the range 0 to 1 spanning -2 to +2.

    float t = max( 1.0f - fabs( x / 2.0f), 0.0f);
    float y = 1.0f + sign(x) * (1.0f - t * t);

    return y / 2.0f;
}


// ------- Red modifier functions
__device__ inline float cubic_basis_shaper
( 
  float x, 
  float w   // full base width of the shaper function (in degrees)
)
{
  float M[4][4] = { { -1.0f/6,  3.0f/6, -3.0f/6,  1.0f/6 },
                    {  3.0f/6, -6.0f/6,  3.0f/6,  0.0f/6 },
                    { -3.0f/6,  0.0f/6,  3.0f/6,  0.0f/6 },
                    {  1.0f/6,  4.0f/6,  1.0f/6,  0.0f/6 } };
  
  float knots[5] = { -w/2.0f,
                     -w/4.0f,
                     0.0f,
                     w/4.0f,
                     w/2.0f };
  
  float y = 0.0f;
  if ((x > knots[0]) && (x < knots[4])) {  
    float knot_coord = (x - knots[0]) * 4.0f/w;  
    int j = knot_coord;
    float t = knot_coord - j;
      
    float monomials[4] = { t*t*t, t*t, t, 1.0f };

    // (if/else structure required for compatibility with CTL < v1.5.)
    if ( j == 3) {
      y = monomials[0] * M[0][0] + monomials[1] * M[1][0] + 
          monomials[2] * M[2][0] + monomials[3] * M[3][0];
    } else if ( j == 2) {
      y = monomials[0] * M[0][1] + monomials[1] * M[1][1] + 
          monomials[2] * M[2][1] + monomials[3] * M[3][1];
    } else if ( j == 1) {
      y = monomials[0] * M[0][2] + monomials[1] * M[1][2] + 
          monomials[2] * M[2][2] + monomials[3] * M[3][2];
    } else if ( j == 0) {
      y = monomials[0] * M[0][3] + monomials[1] * M[1][3] + 
          monomials[2] * M[2][3] + monomials[3] * M[3][3];
    } else {
      y = 0.0f;
    }
  }
  
  return y * 3/2.0f;
}

__device__ inline float center_hue( float hue, float centerH)
{
  float hueCentered = hue - centerH;
  if (hueCentered < -180.0f) hueCentered = hueCentered + 360.0f;
  else if (hueCentered > 180.0f) hueCentered = hueCentered - 360.0f;
  return hueCentered;
}

__device__ inline float uncenter_hue( float hueCentered, float centerH)
{
  float hue = hueCentered + centerH;
  if (hue < 0.0f) hue = hue + 360.0f;
  else if (hue > 360.0f) hue = hue - 360.0f;
  return hue;
}

#endif