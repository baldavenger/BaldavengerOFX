#ifndef __ACES_TONESCALES_H_INCLUDED__
#define __ACES_TONESCALES_H_INCLUDED__

// Textbook monomial to basis-function conversion matrix.
__constant__ mat3 MM = { {  0.5f, -1.0f, 0.5f },
  						{ -1.0f,  1.0f, 0.5f },
  						{  0.5f,  0.0f, 0.0f }};
typedef struct
{
float x;
float y;
} SplineMapPoint;

typedef struct
{
  float coefsLow[6];    // coefs for B-spline between minPoint and midPoint (units of log luminance)
  float coefsHigh[6];   // coefs for B-spline between midPoint and maxPoint (units of log luminance)
  SplineMapPoint minPoint; // {luminance, luminance} linear extension below this
  SplineMapPoint midPoint; // {luminance, luminance} 
  SplineMapPoint maxPoint; // {luminance, luminance} linear extension above this
  float slopeLow;       // log-log slope of low linear extension
  float slopeHigh;      // log-log slope of high linear extension
} SegmentedSplineParams_c5;

typedef struct
{
  float coefsLow[10];    // coefs for B-spline between minPoint and midPoint (units of log luminance)
  float coefsHigh[10];   // coefs for B-spline between midPoint and maxPoint (units of log luminance)
  SplineMapPoint minPoint; // {luminance, luminance} linear extension below this
  SplineMapPoint midPoint; // {luminance, luminance} 
  SplineMapPoint maxPoint; // {luminance, luminance} linear extension above this
  float slopeLow;       // log-log slope of low linear extension
  float slopeHigh;      // log-log slope of high linear extension
} SegmentedSplineParams_c9;

__device__ inline SegmentedSplineParams_c5 RRT_PARAMS()
{
SegmentedSplineParams_c5 A = {{ -4.0f, -4.0f, -3.1573765773f, -0.4852499958f, 1.8477324706f, 1.8477324706f}, 
{ -0.7185482425f, 2.0810307172f, 3.6681241237f, 4.0f, 4.0f, 4.0f}, {0.18f * powf(2.0f, -15.0f), 0.0001f}, 
{0.18f, 4.8f}, {0.18f * powf(2.0f, 18.0f), 10000.0f}, 0.0f, 0.0f};
return A;
};

__device__ float inline segmented_spline_c5_fwd( float x)
{
  SegmentedSplineParams_c5 C = RRT_PARAMS();
  const int N_KNOTS_LOW = 4;
  const int N_KNOTS_HIGH = 4;

  // Check for negatives or zero before taking the log. If negative or zero,
  // set to HALF_MIN.
  float logx = log10f( fmax(x, HALF_MIN )); 

  float logy;

  if ( logx <= log10f(C.minPoint.x) ) { 

    logy = logx * C.slopeLow + ( log10f(C.minPoint.y) - C.slopeLow * log10f(C.minPoint.x) );

  } else if (( logx > log10f(C.minPoint.x) ) && ( logx < log10f(C.midPoint.x) )) {

    float knot_coord = (N_KNOTS_LOW-1) * (logx - log10f(C.minPoint.x))/(log10f(C.midPoint.x) - log10f(C.minPoint.x));
    int j = knot_coord;
    float t = knot_coord - j;

    float3 cf = make_float3( C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]);
    
    float3 monomials = make_float3( t * t, t, 1.0f );
    logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM));

  } else if (( logx >= log10f(C.midPoint.x) ) && ( logx < log10f(C.maxPoint.x) )) {

    float knot_coord = (N_KNOTS_HIGH-1) * (logx-log10(C.midPoint.x))/(log10f(C.maxPoint.x)-log10(C.midPoint.x));
    int j = knot_coord;
    float t = knot_coord - j;

    float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]}; 
     
    float3 monomials = make_float3( t * t, t, 1.0f );
    logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM));

  } else { //if ( logIn >= log10f(C.maxPoint.x) ) { 

    logy = logx * C.slopeHigh + ( log10f(C.maxPoint.y) - C.slopeHigh * log10f(C.maxPoint.x) );

  }

  return pow10(logy);
  
}

__device__ float inline segmented_spline_c5_rev
  ( 
    float y
  )
{  
  SegmentedSplineParams_c5 C = RRT_PARAMS();
  const int N_KNOTS_LOW = 4;
  const int N_KNOTS_HIGH = 4;

  const float KNOT_INC_LOW = (log10f(C.midPoint.x) - log10f(C.minPoint.x)) / (N_KNOTS_LOW - 1.);
  const float KNOT_INC_HIGH = (log10f(C.maxPoint.x) - log10f(C.midPoint.x)) / (N_KNOTS_HIGH - 1.);
  
  // KNOT_Y is luminance of the spline at each knot
  float KNOT_Y_LOW[ N_KNOTS_LOW];
  for (int i = 0; i < N_KNOTS_LOW; i = i+1) {
    KNOT_Y_LOW[ i] = ( C.coefsLow[i] + C.coefsLow[i+1]) / 2.;
  };

  float KNOT_Y_HIGH[ N_KNOTS_HIGH];
  for (int i = 0; i < N_KNOTS_HIGH; i = i+1) {
    KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i+1]) / 2.;
  };

  float logy = log10f( max(y,1e-10));

  float logx;
  if (logy <= log10f(C.minPoint.y)) {

    logx = log10f(C.minPoint.x);

  } else if ( (logy > log10f(C.minPoint.y)) && (logy <= log10f(C.midPoint.y)) ) {

    unsigned int j;
    float3 cf;
    if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) {
        cf.x = C.coefsLow[0];  cf.y = C.coefsLow[1];  cf.z = C.coefsLow[2];  j = 0;
    } else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
        cf.x = C.coefsLow[1];  cf.y = C.coefsLow[2];  cf.z = C.coefsLow[3];  j = 1;
    } else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
        cf.x = C.coefsLow[2];  cf.y = C.coefsLow[3];  cf.z = C.coefsLow[4];  j = 2;
    } 
    
    const float3 tmp = mult_f3_f33( cf, MM);

    float a = tmp.x;
    float b = tmp.y;
    float c = tmp.z;
    c = c - logy;

    const float d = sqrtf( b * b - 4.0f * a * c);

    const float t = ( 2.0f * c) / ( -d - b);

    logx = log10f(C.minPoint.x) + ( t + j) * KNOT_INC_LOW;

  } else if ( (logy > log10f(C.midPoint.y)) && (logy < log10f(C.maxPoint.y)) ) {

    unsigned int j;
    float3 cf;
    if ( logy > KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
        cf.x = C.coefsHigh[0];  cf.y = C.coefsHigh[1];  cf.z = C.coefsHigh[2];  j = 0;
    } else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
        cf.x = C.coefsHigh[1];  cf.y = C.coefsHigh[2];  cf.z = C.coefsHigh[3];  j = 1;
    } else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
        cf.x = C.coefsHigh[2];  cf.y = C.coefsHigh[3];  cf.z = C.coefsHigh[4];  j = 2;
    } 
    
    const float3 tmp = mult_f3_f33( cf, MM);

    float a = tmp.x;
    float b = tmp.y;
    float c = tmp.z;
    c = c - logy;

    const float d = sqrt( b * b - 4. * a * c);

    const float t = ( 2.0f * c) / ( -d - b);

    logx = log10f(C.midPoint.x) + ( t + j) * KNOT_INC_HIGH;

  } else { //if ( logy >= log10f(C.maxPoint.y) ) {

    logx = log10f(C.maxPoint.x);

  }
  
  return pow10( logx);

}

__device__ inline SegmentedSplineParams_c9 ODT_48nits()
{
  SegmentedSplineParams_c9 A =
  {{ -1.6989700043f, -1.6989700043f, -1.4779f, -1.2291f, -0.8648f, -0.448f, 0.00518f, 0.4511080334f, 0.9113744414f, 0.9113744414f},
  { 0.5154386965f, 0.8470437783f, 1.1358f, 1.3802f, 1.5197f, 1.5985f, 1.6467f, 1.6746091357f, 1.6878733390f, 1.6878733390f },
  {segmented_spline_c5_fwd( 0.18f * powf(2.0f, -6.5f) ),  0.02f},
  {segmented_spline_c5_fwd( 0.18f ), 4.8f},
  {segmented_spline_c5_fwd( 0.18f * powf(2.0f, 6.5f) ), 48.0f},
  0.0f, 0.04f};
  return A;
};

__device__ inline SegmentedSplineParams_c9 ODT_1000nits()
{
  SegmentedSplineParams_c9 A =
  {{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f },
  { 0.8089132070f, 1.1910867930f, 1.5683f, 1.9483f, 2.3083f, 2.6384f, 2.8595f, 2.9872608805f, 3.0127391195f, 3.0127391195f },
  {segmented_spline_c5_fwd( 0.18f * powf(2.0f, -12.0f) ), 0.0001f},
  {segmented_spline_c5_fwd( 0.18f ), 10.0f},
  {segmented_spline_c5_fwd( 0.18 * powf(2.0f, 10.0f) ), 1000.0f},
  3.0f, 0.06f};
  return A;
};

__device__ inline SegmentedSplineParams_c9 ODT_2000nits()
{
  SegmentedSplineParams_c9 A =
  {{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f },
  { 0.8019952042f, 1.1980047958f, 1.5943f, 1.9973f, 2.3783f, 2.7684f, 3.0515f, 3.2746293562f, 3.3274306351f, 3.3274306351f },
  {segmented_spline_c5_fwd( 0.18f * powf(2.0f, -12.0f) ), 0.0001f},
  {segmented_spline_c5_fwd( 0.18f ), 10.0f},
  {segmented_spline_c5_fwd( 0.18f * powf(2.0f, 11.0f) ),  2000.0f},
  3.0f, 0.12f};
  return A;
};

__device__ inline SegmentedSplineParams_c9 ODT_4000nits()
{
  SegmentedSplineParams_c9 A =
  {{ -4.9706219331f, -3.0293780669f, -2.1262f, -1.5105f, -1.0578f, -0.4668f, 0.11938f, 0.7088134201f, 1.2911865799f, 1.2911865799f },
  { 0.7973186613f, 1.2026813387f, 1.6093f, 2.0108f, 2.4148f, 2.8179f, 3.1725f, 3.5344995451f, 3.6696204376f, 3.6696204376f },
  {segmented_spline_c5_fwd( 0.18f * powf(2.0f, -12.0f) ), 0.0001f},
  {segmented_spline_c5_fwd( 0.18f ), 10.0f},
  {segmented_spline_c5_fwd( 0.18f * powf(2.0f, 12.0f) ), 4000.0f},
  3.0f, 0.3f};
  return A;
};

__device__ inline float segmented_spline_c9_fwd
  ( 
    float x,
    SegmentedSplineParams_c9 C
  )
{    
  //SegmentedSplineParams_c9 C = ODT_48nits();
  const int N_KNOTS_LOW = 8;
  const int N_KNOTS_HIGH = 8;

  // Check for negatives or zero before taking the log. If negative or zero,
  // set to HALF_MIN.
  float logx = log10f( max(x, HALF_MIN )); 

  float logy;

  if ( logx <= log10f(C.minPoint.x) ) { 

    logy = logx * C.slopeLow + ( log10f(C.minPoint.y) - C.slopeLow * log10f(C.minPoint.x) );

  } else if (( logx > log10f(C.minPoint.x) ) && ( logx < log10f(C.midPoint.x) )) {

    float knot_coord = (N_KNOTS_LOW-1) * (logx-log10(C.minPoint.x))/(log10f(C.midPoint.x)-log10(C.minPoint.x));
    int j = knot_coord;
    float t = knot_coord - j;

    float3 cf = { C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]};
    
    float3 monomials = make_float3( t * t, t, 1.0f );
    logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM));

  } else if (( logx >= log10f(C.midPoint.x) ) && ( logx < log10f(C.maxPoint.x) )) {

    float knot_coord = (N_KNOTS_HIGH-1) * (logx-log10(C.midPoint.x))/(log10f(C.maxPoint.x)-log10(C.midPoint.x));
    int j = knot_coord;
    float t = knot_coord - j;

    float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]}; 
    
    float3 monomials = make_float3( t * t, t, 1.0f );
    logy = dot_f3_f3( monomials, mult_f3_f33( cf, MM));

  } else { //if ( logIn >= log10f(C.maxPoint.x) ) { 

    logy = logx * C.slopeHigh + ( log10f(C.maxPoint.y) - C.slopeHigh * log10f(C.maxPoint.x) );

  }

  return pow10(logy);
  
}

__device__ float inline segmented_spline_c9_rev
  ( 
    float y,
    SegmentedSplineParams_c9 C
  )
{  
  //SegmentedSplineParams_c9 C = ODT_48nits();
  const int N_KNOTS_LOW = 8;
  const int N_KNOTS_HIGH = 8;

  const float KNOT_INC_LOW = (log10f(C.midPoint.x) - log10f(C.minPoint.x)) / (N_KNOTS_LOW - 1.);
  const float KNOT_INC_HIGH = (log10f(C.maxPoint.x) - log10f(C.midPoint.x)) / (N_KNOTS_HIGH - 1.);
  
  // KNOT_Y is luminance of the spline at each knot
  float KNOT_Y_LOW[ N_KNOTS_LOW];
  for (int i = 0; i < N_KNOTS_LOW; i = i+1) {
    KNOT_Y_LOW[ i] = ( C.coefsLow[i] + C.coefsLow[i+1]) / 2.;
  };

  float KNOT_Y_HIGH[ N_KNOTS_HIGH];
  for (int i = 0; i < N_KNOTS_HIGH; i = i+1) {
    KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i+1]) / 2.;
  };

  float logy = log10f( max( y, 1e-10));

  float logx;
  if (logy <= log10f(C.minPoint.y)) {

    logx = log10f(C.minPoint.x);

  } else if ( (logy > log10f(C.minPoint.y)) && (logy <= log10f(C.midPoint.y)) ) {

    unsigned int j;
    float3 cf;
    if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) {
        cf.x = C.coefsLow[0];  cf.y = C.coefsLow[1];  cf.z = C.coefsLow[2];  j = 0;
    } else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
        cf.x = C.coefsLow[1];  cf.y = C.coefsLow[2];  cf.z = C.coefsLow[3];  j = 1;
    } else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
        cf.x = C.coefsLow[2];  cf.y = C.coefsLow[3];  cf.z = C.coefsLow[4];  j = 2;
    } else if ( logy > KNOT_Y_LOW[ 3] && logy <= KNOT_Y_LOW[ 4]) {
        cf.x = C.coefsLow[3];  cf.y = C.coefsLow[4];  cf.z = C.coefsLow[5];  j = 3;
    } else if ( logy > KNOT_Y_LOW[ 4] && logy <= KNOT_Y_LOW[ 5]) {
        cf.x = C.coefsLow[4];  cf.y = C.coefsLow[5];  cf.z = C.coefsLow[6];  j = 4;
    } else if ( logy > KNOT_Y_LOW[ 5] && logy <= KNOT_Y_LOW[ 6]) {
        cf.x = C.coefsLow[5];  cf.y = C.coefsLow[6];  cf.z = C.coefsLow[7];  j = 5;
    } else if ( logy > KNOT_Y_LOW[ 6] && logy <= KNOT_Y_LOW[ 7]) {
        cf.x = C.coefsLow[6];  cf.y = C.coefsLow[7];  cf.z = C.coefsLow[8];  j = 6;
    }
    
    const float3 tmp = mult_f3_f33( cf, MM);

    float a = tmp.x;
    float b = tmp.y;
    float c = tmp.z;
    c = c - logy;

    const float d = sqrt( b * b - 4. * a * c);

    const float t = ( 2. * c) / ( -d - b);

    logx = log10f(C.minPoint.x) + ( t + j) * KNOT_INC_LOW;

  } else if ( (logy > log10f(C.midPoint.y)) && (logy < log10f(C.maxPoint.y)) ) {

    unsigned int j;
    float3 cf;
    if ( logy > KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
        cf.x = C.coefsHigh[0];  cf.y = C.coefsHigh[1];  cf.z = C.coefsHigh[2];  j = 0;
    } else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
        cf.x = C.coefsHigh[1];  cf.y = C.coefsHigh[2];  cf.z = C.coefsHigh[3];  j = 1;
    } else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
        cf.x = C.coefsHigh[2];  cf.y = C.coefsHigh[3];  cf.z = C.coefsHigh[4];  j = 2;
    } else if ( logy > KNOT_Y_HIGH[ 3] && logy <= KNOT_Y_HIGH[ 4]) {
        cf.x = C.coefsHigh[3];  cf.y = C.coefsHigh[4];  cf.z = C.coefsHigh[5];  j = 3;
    } else if ( logy > KNOT_Y_HIGH[ 4] && logy <= KNOT_Y_HIGH[ 5]) {
        cf.x = C.coefsHigh[4];  cf.y = C.coefsHigh[5];  cf.z = C.coefsHigh[6];  j = 4;
    } else if ( logy > KNOT_Y_HIGH[ 5] && logy <= KNOT_Y_HIGH[ 6]) {
        cf.x = C.coefsHigh[5];  cf.y = C.coefsHigh[6];  cf.z = C.coefsHigh[7];  j = 5;
    } else if ( logy > KNOT_Y_HIGH[ 6] && logy <= KNOT_Y_HIGH[ 7]) {
        cf.x = C.coefsHigh[6];  cf.y = C.coefsHigh[7];  cf.z = C.coefsHigh[8];  j = 6;
    }
    
    const float3 tmp = mult_f3_f33( cf, MM);

    float a = tmp.x;
    float b = tmp.y;
    float c = tmp.z;
    c = c - logy;

    const float d = sqrt( b * b - 4. * a * c);

    const float t = ( 2. * c) / ( -d - b);

    logx = log10f(C.midPoint.x) + ( t + j) * KNOT_INC_HIGH;

  } else { //if ( logy >= log10f(C.maxPoint.y) ) {

    logx = log10f(C.maxPoint.x);

  }
  
  return pow10( logx);
}

#endif