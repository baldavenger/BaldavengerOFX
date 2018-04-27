#ifndef __ACES_FUNCTIONS_H_INCLUDED__
#define __ACES_FUNCTIONS_H_INCLUDED__

#define HALF_DIG		3
#define HALF_MANT_DIG	11
#define HALF_MAX_10_EXP	+4
#define HALF_MAX_EXP	+16
#define HALF_MIN_10_EXP	-4
#define HALF_MIN_EXP	-13
#define HALF_RADIX		2
#define HALF_POS_INF	31744.0f
#define HALF_MIN		5.96046448e-08 // Smallest positive half
#define HALF_MIN_NORM	6.10351562e-05 // Smallest positive normalized half
#define HALF_MAX		65504.0f        // Largest positive half
#define HALF_EPSILON	0.00097656f // Smallest positive e for which half (1.0 + e) != half (1.0)
#define HALF_NAN		65535.0f
#define HALF_POS_INF	31744.0f
#define HALF_NEG_INF	64512.0f

#ifndef M_PI
#define M_PI			3.14159265358979323846264338327950288
#endif

__device__ inline int size(float array[])
{
int Size = sizeof(array)/sizeof(array[0]);
return Size + 1;
}

__device__ inline int size(float2 array[])
{
int Size = sizeof(array)/sizeof(array[0]);
return Size + 1;
}

__device__ inline int size(float3 array[])
{
int Size = sizeof(array)/sizeof(array[0]);
return Size + 1;
}

typedef struct
{
float2 red;
float2 green;
float2 blue;
float2 white;
} Chromaticities;

__device__ inline Chromaticities make_chromaticities(float2 A, float2 B, float2 C, float2 D)
{
Chromaticities E;
E.red = A;
E.green = B;
E.blue = C;
E.white = D;
return E;
}

typedef struct
{
float x, y, z, w, m;
} float5;

typedef struct
{
float2 c0, c1;
} mat2;

typedef struct
{
float3 c0, c1, c2;
} mat3;

typedef struct
{
float4 c0, c1, c2, c3;
} mat4;

__device__ inline float4 make_float4(float3 A, float B)
{
float4 C = make_float4(A.x, A.y, A.z, B);
return C;
}

__device__ inline mat2 make_mat2(float2 A, float2 B)
{
mat2 C;
C.c0 = A;
C.c1 = B;
return C;
}

__device__ inline mat2 transpose(mat2 A)
{
mat2 B;
B.c0 = make_float2(A.c0.x, A.c1.x);
B.c1 = make_float2(A.c0.y, A.c1.y);
return B;
}

__device__ inline mat3 make_mat3(float3 A, float3 B, float3 C)
{
mat3 D;
D.c0 = A;
D.c1 = B;
D.c2 = C;
return D;
}

__device__ inline mat3 transpose(mat3 A)
{
mat3 B;
B.c0 = make_float3(A.c0.x, A.c1.x, A.c2.x);
B.c1 = make_float3(A.c0.y, A.c1.y, A.c2.y);
B.c2 = make_float3(A.c0.z, A.c1.z, A.c2.z);
return B;
}

__device__ inline mat4 make_mat4(float4 A, float4 B, float4 C, float4 D)
{
mat4 E;
E.c0 = A;
E.c1 = B;
E.c2 = C;
E.c3 = D;
return E;
}

__device__ inline mat4 make_mat4(mat3 A, float B)
{
mat4 C;
C.c0 = make_float4(A.c0, 0.0f);
C.c1 = make_float4(A.c1, 0.0f);
C.c2 = make_float4(A.c2, 0.0f);
C.c3 = make_float4(0.0f, 0.0f, 0.0f, B);
return C;
}

__device__ inline mat4 transpose(mat4 A)
{
mat4 B;
B.c0 = make_float4(A.c0.x, A.c1.x, A.c2.x, A.c3.x);
B.c1 = make_float4(A.c0.y, A.c1.y, A.c2.y, A.c3.y);
B.c2 = make_float4(A.c0.z, A.c1.z, A.c2.z, A.c3.z);
B.c3 = make_float4(A.c0.w, A.c1.w, A.c2.w, A.c3.w);
return B;
}
/*
__device__ inline bool isfinite_f (float x)
{
  return x != FLT_POS_INF && x != FLT_NEG_INF && x != FLT_NAN;
}

__device__ inline bool isfinite_h (half x)
{
  return x != HALF_POS_INF && x != HALF_NEG_INF && x != HALF_NAN;
}

__device__ inline bool isnormal_f (float x)
{
  return isfinite_f(x) && x != 0.0f;
}
__device__ inline bool isnormal_h (half x)
{
  return isfinite_h(x) && x != 0.0f;
}

__device__ inline bool isinf_f (float x)
{
  return x == FLT_POS_INF || x == FLT_NEG_INF;
}
__device__ inline bool isinf_h (half x)
{
  return x == HALF_POS_INF || x != HALF_NEG_INF;
}

__device__ inline float min( float a, float b)
{
  if (a < b)
    return a;
  else
    return b;
}

__device__ inline float max( float a, float b)
{
  if (a > b)
    return a;
  else
    return b;
}
*/

__device__ inline float min_f3( float3 a)
{
  return min( a.x, min( a.y, a.z));
}

__device__ inline float max_f3( float3 a)
{
  return max( a.x, max( a.y, a.z));
}

__device__ inline float clip( float v)
{
  return min(v, 1.0f);
}

__device__ inline float3 clip_f3( float3 in)
{
  float3 out;
  out.x = clip( in.x);
  out.y = clip( in.y);
  out.z = clip( in.z);

  return out;
}

__device__ inline float clamp( float in, float clampMin, float clampMax)
{
  // Note: Numeric constants can be used in place of a min or max value (i.e. 
  // use HALF_NEG_INF in place of clampMin or HALF_POS_INF in place of clampMax)
  
  return max( clampMin, min(in, clampMax));
}

__device__ inline float3 add_f_f3( float a, float3 b)
{
  float3 out;
  out.x = a + b.x;
  out.y = a + b.y;
  out.z = a + b.z;
  return out;
}

__device__ inline float3 pow_f3( float3 a, float b)
{
  float3 out;
  out.x = powf(a.x, b);
  out.y = powf(a.y, b);
  out.z = powf(a.z, b);
  return out;
}

__device__ inline float pow10(float x)
{
  return powf(10.0f, x);
}

__device__ inline float3 pow10_f3( float3 a)
{
  float3 out;
  out.x = pow10(a.x);
  out.y = pow10(a.y);
  out.z = pow10(a.z);
  return out;
}

__device__ inline float3 log10_f3( float3 a)
{
  float3 out;
  out.x = log10(a.x);
  out.y = log10(a.y);
  out.z = log10(a.z);
  return out;
}

/*
__device__ inline float round(float x)
{
  int x1;
 
  if (x < 0.0f)
    x1 = x - 0.5f;
  else
    x1 = x + 0.5f;
 
  return x1;
}

__device__ inline float log2(float x)
{
  return logf(x) / logf(2.0f);
}
*/

__device__ inline int sign( float x)
{
    // Signum function:
    //  sign(X) returns 1 if the element is greater than zero, 0 if it equals zero 
    //  and -1 if it is less than zero

    int y;
    if (x < 0) { 
        y = -1;
    } else if (x > 0) {
        y = 1;
    } else {
        y = 0;
    }

    return y;
}
/*
__device__ inline half pow10_h(float x)
{
  return powf(10.0f, x);
}

__device__ inline float hypot(float x, float y)
{
  return sqrtf(x*x + y*y);
}
*/
// lookup

__device__ inline float lookup1D(float table[], float pMin, float pMax, float p)
{
  int Size = size(table);
  if( p < pMin ) return table[ 0 ];
  if( p > pMax ) return table[ Size - 1 ];
  float t = (p - pMin) / (pMax - pMin ) * (Size - 1 );
  int i = floor( t );
  float s = t - i;
  return table[i] * ( 1 - s ) + table[i+1] * s; 
}

__device__ inline float lookupCubic1D (float table[], float pMin, float pMax, float p)
{
  int Size = size(table);
  if( p < pMin ) return table[ 0 ];
  if( p > pMax ) return table[ Size - 1 ];
  
  float t = (p - pMin) / (pMax - pMin) * (Size - 1 );
  int i = floor (t);
  float s = t - i;
  float m0;
  float m1;
  if( i > 0 )
  {
    m0 = (table[i+1] - table[i-1]) / 2;
  }
  if( i < Size-2 )
  {
    m1 = (table[i+2] - table[i]) / 2;
  }
  if( i == 0) {
    m0 = (3 * table[i+1] - table[i] - m1);
  }
  if( i == Size-2 )
  {
    m1 = (3 * table[i+1] - table[i] - m0);
  }
  return table[i] * (2 * s*s*s - 3 * s*s + 1) + m0 * (s*s*s - 2 * s*s + s) + table[i+1] * (-2 * s*s*s + 3 * s*s) + m1 * (s*s*s - s*s);
}

__device__ inline float interpolate1D (float2 table[], float p)
{
  int Size = size(table);
  if( p <= table[0].x ) return table[0].y;
  if( p >= table[Size-1].x ) return table[Size-1].y;
  
  for( int i = 0; i < Size - 1; ++i )
  {
    if( table[i].x <= p && p < table[i+1].x )
    {
      float s = (p - table[i].x) / (table[i+1].x - table[i].x);
      return table[i].y * ( 1 - s ) + table[i+1].y * s;
    }
  }
  return 0.0f;
}

__device__ inline float interpolateCubic1D (float2 table[], float p)
{
  int Size = size(table);
  if( p <= table[0].x ) return table[0].y;
  if( p >= table[Size-1].x ) return table[Size-1].y;
  
  if( Size == 2 ) {
    float s = (p - table[0].x) / (table[1].x - table[0].x);
    return (1.0f - s) * table[0].y + s * table[1].y;
  }
  
  for( int i = 0; i < Size - 1; ++i )
  {
    if( table[i].x <= p && p < table[i+1].x )
    {
      float s = (p - table[i].x) / (table[i+1].x - table[i].x);
      float dx1 = (table[i+1].x - table[i].x);
      float dy1 = (table[i+1].y - table[i].y);
      
      float m0;
      float m1;
      if( i > 0 )
      {
        float dy0 = (table[i].y - table[i-1].y);
        float dx0 = (table[i].x - table[i-1].x);
        m0 = (dy1 + dx1 * dy0 / dx0) / 2;
      }
      if( i < Size-2 )
      {
        float dx2 = (table[i+2].x - table[i+1].x);
        float dy2 = (table[i+2].y - table[i+1].y);
        m1 = (dy1 + dx1 * dy2 / dx2) / 2;
      }
      if( i == 0) {
        m0 = (3 * dy1 - m1) / 2;
      }
      if( i == Size-2 )
      {
        m1 = (3 * dy1 - m0) / 2;
      }
      return table[i].y * (2 * s*s*s - 3 * s*s + 1) +
          m0 * (s*s*s - 2 * s*s + s) +
          table[i+1].y * (-2 * s*s*s + 3 * s*s) +
          m1 * (s*s*s - s*s);
    }
  }
  return 0.0f;
}
/*
__device__ inline float3 lookup3D_f3
         (float table[][][][],
          float3 pMin,
          float3 pMax,
          float3 p)
{
  int Size = size(table);
  float3 result;
  int iMax = Size - 1;
  int jMax = table[0].size - 1;
  int kMax = table[0][0].size - 1;
  float3 q;
  q.x = max (pMin.x, min (pMax.x, p.x));
  q.y = max (pMin.y, min (pMax.y, p.y));
  q.z = max (pMin.z, min (pMax.z, p.z));
  
  float ti = (p.x - pMin.x) / (pMax.x - pMin.x) * iMax;
  int i = floor (ti);
  float si = ti - i;
  float tj = (p.y - pMin.y) / (pMax.y - pMin.y) * jMax;
  int j = floor (tj);
  float sj = tj - j;
  float tk = (p.z - pMin.z) / (pMax.z - pMin.z) * kMax;
  int k = floor (tk);
  float sk = tk - k;
  
  result.x 	= ((table[i][j][k][0] * (1-si) + table[i+1][j][k][0] * si) * (1-sj)
			  + (table[i][j+1][k][0] * (1-si) + table[i+1][j+1][k][0] * si) * sj ) * (1-sk)
			  + ((table[i][j][k+1][0] * (1-si) + table[i+1][j][k+1][0] * si) * (1-sj)
			  + (table[i][j+1][k+1][0] * (1-si) + table[i+1][j+1][k+1][0] * si) * sj ) * sk;

  result.y 	= ((table[i][j][k][1] * (1-si) + table[i+1][j][k][1] * si) * (1-sj)
			  + (table[i][j+1][k][1] * (1-si) + table[i+1][j+1][k][1] * si) * sj ) * (1-sk)
			  + ((table[i][j][k+1][1] * (1-si) + table[i+1][j][k+1][1] * si) * (1-sj)
			  + (table[i][j+1][k+1][1] * (1-si) + table[i+1][j+1][k+1][1] * si) * sj ) * sk;

  result.z 	= ((table[i][j][k][2] * (1-si) + table[i+1][j][k][2] * si) * (1-sj)
			  + (table[i][j+1][k][2] * (1-si) + table[i+1][j+1][k][2] * si) * sj ) * (1-sk)
			  + ((table[i][j][k+1][2] * (1-si) + table[i+1][j][k+1][2] * si) * (1-sj)
			  + (table[i][j+1][k+1][2] * (1-si) + table[i+1][j+1][k+1][2] * si) * sj ) * sk;

  return result;
}

__device__ inline void lookup3D_f
       (float table[][][][3],
        float3 pMin,
        float3 pMax,
        float p0, float p1, float p2,
        float *q0, float *q1, float *q2)
{
  float3 p;
  p.x = p0;
  p.y = p1;
  p.z = p2;
  float3 result = lookup3D_f3( table, pMin, pMax, p );
  *q0 = result.x;
  *q1 = result.y;
  *q2 = result.z;
}

__device__ inline void lookup3D_h
       (float table[][][][3],
        float3 pMin,
        float3 pMax,
        half p0, half p1, half p2,
        half *q0, half *q1, half *q2)
{
  lookup3D_f( table, pMin, pMax, p0, p1, p2, *q0, *q1, *q2 );
}
*/
// Color conversion functions
/*
struct Chromaticities
{
    float red[2];
    float green[2];
    float blue[2];
    float white[2];
};
*/
// float[4][4] RGBtoXYZ(Chromaticities c, float Y)
// {
// }
// 
// float[4][4] XYZtoRGB(Chromaticities c, float Y)
// {
// }
// 
// float[3] XYZtoLuv(float XYZ[3], float XYZn[3]);
// float[3] LuvtoXYZ(float Luv[3], float XYZn[3]);
//float[3] XYZtoLab(float XYZ[3], float XYZn[3]);
// float[3] LabtoXYZ(float Lab[3], float XYZn[3]);
// 

// Vectors/Matrix operations

__device__ inline float3 mult_f_f3(float f, float3 x)
{
  float3 r;
  r.x = f * x.x;
  r.y = f * x.y;
  r.z = f * x.z;
  return r;
}

__device__ inline float3 add_f3_f3(float3 x, float3 y)
{
  float3 r;
  r.x = x.x + y.x;
  r.y = x.y + y.y;
  r.z = x.z + y.z;
  return r;
}

__device__ inline float3 sub_f3_f3(float3 x, float3 y)
{
  float3 r;
  r.x = x.x - y.x;
  r.y = x.y - y.y;
  r.z = x.z - y.z;
  return r;
}

__device__ inline float3 cross_f3_f3(float3 x, float3 y)
{
  float3 r;
  r.z = x.x * y.y - x.y * y.x;
  r.x = x.y * y.z - x.z * y.y;
  r.y = x.z * y.x - x.x * y.z;
  return r;
}

__device__ inline float3 clamp_f3( float3 in, float clampMin, float clampMax)
{
  // Note: Numeric constants can be used in place of a min or max value (i.e. 
  // use HALF_NEG_INF in place of clampMin or HALF_POS_INF in place of clampMax)

  float3 out;
  out.x = clamp( in.x, clampMin, clampMax);
  out.y = clamp( in.y, clampMin, clampMax);
  out.z = clamp( in.z, clampMin, clampMax);
      
  return out;
}

__device__ inline float dot_f3_f3(float3 x, float3 y)
{
  return x.x * y.x + x.y * y.y + x.z * y.z;
}

__device__ inline float length_f3 (float3 x)
{
  return sqrtf( x.x * x.x + x.y * x.y + x.z * x.z );
}

__device__ inline mat3 mult_f33_f33 (mat3 A, mat3 B)
{
  float r[3][3];
  float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
  		 {A.c1.x, A.c1.y, A.c1.z},
  		 {A.c2.x, A.c2.y, A.c2.z}};
  float b[3][3] =	{{B.c0.x, B.c0.y, B.c0.z},
  		 {B.c1.x, B.c1.y, B.c1.z},
  		 {B.c2.x, B.c2.y, B.c2.z}};
  
  for( int i = 0; i < 3; ++i)
  {
    for( int j = 0; j < 3; ++j)
    {
      r[i][j] = 0.0f;
      for( int k = 0; k < 3; ++k)
      {
        r[i][j] = r[i][j] + a[i][k] * b[k][j];
      }
    }
  }
  mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), 
  make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
  return R;
}

__device__ inline mat4 mult_f44_f44 (mat4 A, mat4 B)
{
 float r[4][4];
 float a[4][4] =	{{A.c0.x, A.c0.y, A.c0.z, A.c0.w},
  		 			{A.c1.x, A.c1.y, A.c1.z, A.c1.w},
  		 			{A.c2.x, A.c2.y, A.c2.z, A.c2.w},
  		 			{A.c3.x, A.c3.y, A.c3.z, A.c3.w}};
 float b[4][4] =	{{B.c0.x, B.c0.y, B.c0.z, B.c0.w},
  		 			{B.c1.x, B.c1.y, B.c1.z, B.c1.w},
  		 			{B.c2.x, B.c2.y, B.c2.z, B.c2.w},
  		 			{B.c3.x, B.c3.y, B.c3.z, B.c3.w}};
  for( int i = 0; i < 4; ++i)
  {
    for( int j = 0; j < 4; ++j)
    {
      r[i][j] = 0.0f;
      for( int k = 0; k < 4; ++k)
      {
        r[i][j] = r[i][j] + a[i][k] * b[k][j];
      }
    }
  }
  mat4 R = make_mat4(make_float4(r[0][0], r[0][1], r[0][2], r[0][3]), 
  make_float4(r[1][0], r[1][1], r[1][2], r[1][3]), make_float4(r[2][0], r[2][1], r[2][2], r[2][3]),
  make_float4(r[3][0], r[3][1], r[3][2], r[3][3]));
  return R;
}

__device__ inline mat3 mult_f_f33 (float f, mat3 A)
{
  float r[3][3];
  float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
  		 {A.c1.x, A.c1.y, A.c1.z},
  		 {A.c2.x, A.c2.y, A.c2.z}};
  for( int i = 0; i < 3; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      r[i][j] = f * a[i][j];
    }
  }
  mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), 
  make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
  return R;
}

__device__ inline mat4 mult_f_f44 (float f, mat4 A)
{
  float r[4][4];
  float a[4][4] =	{{A.c0.x, A.c0.y, A.c0.z, A.c0.w},
  		 {A.c1.x, A.c1.y, A.c1.z, A.c1.w},
  		 {A.c2.x, A.c2.y, A.c2.z, A.c2.w},
  		 {A.c3.x, A.c3.y, A.c3.z, A.c3.w}};
  for( int i = 0; i < 4; ++i )
  {
    for( int j = 0; j < 4; ++j )
    {
      r[i][j] = f * a[i][j];
    }
  }
  mat4 R = make_mat4(make_float4(r[0][0], r[0][1], r[0][2], r[0][3]), 
  make_float4(r[1][0], r[1][1], r[1][2], r[1][3]), make_float4(r[2][0], r[2][1], r[2][2], r[2][3]),
  make_float4(r[3][0], r[3][1], r[3][2], r[3][3]));
  return R;
}

__device__ inline mat3 add_f33_f33(mat3 A, mat3 B)
{
  float r[3][3];
  float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
  		 {A.c1.x, A.c1.y, A.c1.z},
  		 {A.c2.x, A.c2.y, A.c2.z}};
  float b[3][3] =	{{B.c0.x, B.c0.y, B.c0.z},
  		 {B.c1.x, B.c1.y, B.c1.z},
  		 {B.c2.x, B.c2.y, B.c2.z}};
  for( int i = 0; i < 3; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      r[i][j] = a[i][j] + b[i][j];
    }
  }
  mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), 
  make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
  return R;
}

__device__ inline mat4 add_f44_f44(mat4 A, mat4 B)
{
  float r[4][4];
  float a[4][4] =	{{A.c0.x, A.c0.y, A.c0.z, A.c0.w},
  		 			{A.c1.x, A.c1.y, A.c1.z, A.c1.w},
  		 			{A.c2.x, A.c2.y, A.c2.z, A.c2.w},
  		 			{A.c3.x, A.c3.y, A.c3.z, A.c3.w}};
  float b[4][4] =	{{B.c0.x, B.c0.y, B.c0.z, B.c0.w},
  		 			{B.c1.x, B.c1.y, B.c1.z, B.c1.w},
  		 			{B.c2.x, B.c2.y, B.c2.z, B.c2.w},
  		 			{B.c3.x, B.c3.y, B.c3.z, B.c3.w}};
  for( int i = 0; i < 4; ++i )
  {
    for( int j = 0; j < 4; ++j )
    {
      r[i][j] = a[i][j] + b[i][j];
    }
  }
  mat4 R = make_mat4(make_float4(r[0][0], r[0][1], r[0][2], r[0][3]), 
  make_float4(r[1][0], r[1][1], r[1][2], r[1][3]), make_float4(r[2][0], r[2][1], r[2][2], r[2][3]),
  make_float4(r[3][0], r[3][1], r[3][2], r[3][3]));
  return R;
}

__device__ inline mat3 invert_f33 (mat3 A)
{
  mat3 R;
  float result[3][3];
  float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
  		 {A.c1.x, A.c1.y, A.c1.z},
  		 {A.c2.x, A.c2.y, A.c2.z}};
  		 
  float det =   a[0][0] * a[1][1] * a[2][2]
              + a[0][1] * a[1][2] * a[2][0]
              + a[0][2] * a[1][0] * a[2][1]
              - a[2][0] * a[1][1] * a[0][2]
              - a[2][1] * a[1][2] * a[0][0]
              - a[2][2] * a[1][0] * a[0][1];
  if( det != 0.0 )
  {
    result[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    result[0][1] = a[2][1] * a[0][2] - a[2][2] * a[0][1];
    result[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    result[1][0] = a[2][0] * a[1][2] - a[1][0] * a[2][2];
    result[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2];
    result[1][2] = a[1][0] * a[0][2] - a[0][0] * a[1][2];
    result[2][0] = a[1][0] * a[2][1] - a[2][0] * a[1][1];
    result[2][1] = a[2][0] * a[0][1] - a[0][0] * a[2][1];
    result[2][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1];
    
    R = make_mat3(make_float3(result[0][0], result[0][1], result[0][2]), 
    make_float3(result[1][0], result[1][1], result[1][2]), make_float3(result[2][0], result[2][1], result[2][2]));
    return mult_f_f33( 1.0f / det, R);
  }
  //result[3][3] = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };
  R = make_mat3(make_float3(1.0f, 0.0f, 0.0f), 
  make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
  return R;
}

__device__ inline mat4 invert_f44 (mat4 A)
{
  mat4 R;
  mat4 unity = make_mat4(make_float4(1.0f, 0.0f, 0.0f, 0.0f), 
  		make_float4(0.0f, 1.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 1.0f, 0.0f),
  		make_float4(0.0f, 0.0f, 0.0f, 1.0f));
  float a[4][4] =	{{A.c0.x, A.c0.y, A.c0.z, A.c0.w},
  		 			{A.c1.x, A.c1.y, A.c1.z, A.c1.w},
  		 			{A.c2.x, A.c2.y, A.c2.z, A.c2.w},
  		 			{A.c3.x, A.c3.y, A.c3.z, A.c3.w}};
  if (a[0][3] != 0 || a[1][3] != 0 || a[2][3] != 0 || a[3][3] != 1)
  {
    int i;
    int j;
    int k;
    float s[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    float t[4][4] = {{a[0][0], a[0][1], a[0][2], a[0][3]},
    				 {a[1][0], a[1][1], a[1][2], a[1][3]},
    				 {a[2][0], a[2][1], a[2][2], a[2][3]},
    				 {a[3][0], a[3][1], a[3][2], a[3][3]}};

    // Forward elimination
  
    for (i = 0; i < 3 ; ++i)
    {
      int pivot = i;

      float pivotsize = t[i][i];

      if (pivotsize < 0)
      {
        pivotsize = -pivotsize;
      }

      for (j = i + 1; j < 4; ++j)
      {
        float tmp = t[j][i];

        if (tmp < 0)
          tmp = -tmp;

        if (tmp > pivotsize)
        {
          pivot = j;
          pivotsize = tmp;
        }
      }
      if (pivotsize == 0)
      {
        //s = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
        
  		return unity;
      }

      if (pivot != i)
      {
        for (j = 0; j < 4; ++j)
        {
          float tmp = t[i][j];
          t[i][j] = t[pivot][j];
          t[pivot][j] = tmp;

          tmp = s[i][j];
          s[i][j] = s[pivot][j];
          s[pivot][j] = tmp;
        }
      }

      for (j = i + 1; j < 4; ++j)
      {
        float f = t[j][i] / t[i][i];
        for (k = 0; k < 4; ++k)
        {
          t[j][k] = t[j][k] - f * t[i][k];
          s[j][k] = s[j][k] - f * s[i][k];
        }
      }
    }
    // Backward substitution
    for (i = 3; i >= 0; --i)
    {
      float f = t[i][i];
      if ( f == 0)
      {
        //s = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
        
  		return unity;
      }
      for (j = 0; j < 4; ++j)
      {
        t[i][j] = t[i][j] / f;
        s[i][j] = s[i][j] / f;
      }

      for (j = 0; j < i; ++j)
      {
        f = t[j][i];
  
        for (k = 0; k < 4; ++k)
        {
          t[j][k] = t[j][k] - f * t[i][k];
          s[j][k] = s[j][k] - f * s[i][k];
        }
      }
    }
    R = make_mat4(make_float4(s[0][0], s[0][1], s[0][2], s[0][3]), 
  	make_float4(s[1][0], s[1][1], s[1][2], s[1][3]), make_float4(s[2][0], s[2][1], s[2][2], s[2][3]),
  	make_float4(s[3][0], s[3][1], s[3][2], s[3][3]));
  	return R;
  	
  } else {

    float result[4][4];
    result[0][0] = a[1][1] * a[2][2] - a[2][1] * a[1][2];
    result[0][1] = a[2][1] * a[0][2] - a[0][1] * a[2][2];
    result[0][2] = a[0][1] * a[1][2] - a[1][1] * a[0][2];
    result[0][3] = 0;

    result[1][0] = a[2][0] * a[1][2] - a[1][0] * a[2][2];
    result[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2];
    result[1][2] = a[1][0] * a[0][2] - a[0][0] * a[1][2];
    result[1][3] = 0;

    result[0][0] = a[1][0] * a[2][1] - a[2][0] * a[1][1];
    result[0][1] = a[2][0] * a[0][1] - a[0][0] * a[2][1];
    result[0][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1];
    result[0][3] = 0;

    result[3][0] = 0.0f;
    result[3][1] = 0.0f;
    result[3][2] = 0.0f;
    result[3][3] = 1.0f;

    float r = a[0][0] * result[0][0] + a[0][1] * result[1][0] + a[0][2] * result[2][0];

    if(fabs(r) >= 1)
    {
      for(int i = 0; i < 3; ++i)
      {
        for(int j = 0; j < 3; ++j)
        {
          result[i][j] = result[i][j] / r;
        }
      }
    }
    else
    {
      float mr = fabs(r);// / FLT_MIN;

      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          if (mr > fabs(result[i][j]))
          {
            result[i][j] = result[i][j] / r;
          }
          else
          {
            
  			return unity;
          }
        }
      }
    }
    result[3][0] = -a[3][0] * result[0][0] - a[3][1] * result[1][0] - a[3][2] * result[2][0];
    result[3][1] = -a[3][0] * result[0][1] - a[3][1] * result[1][1] - a[3][2] * result[2][1];
    result[3][2] = -a[3][0] * result[0][2] - a[3][1] * result[1][2] - a[3][2] * result[2][2];
    R = make_mat4(make_float4(result[0][0], result[0][1], result[0][2], result[0][3]), 
  	make_float4(result[1][0], result[1][1], result[1][2], result[1][3]), make_float4(result[2][0], result[2][1], result[2][2], result[2][3]),
  	make_float4(result[3][0], result[3][1], result[3][2], result[3][3]));
  	return R;
  }
  //float result[4][4];
  //R = make_mat4(make_float4(result[0][0], result[0][1], result[0][2], result[0][3]), 
  //make_float4(result[1][0], result[1][1], result[1][2], result[1][3]), make_float4(result[2][0], result[2][1], result[2][2], result[2][3]),
  //make_float4(result[3][0], result[3][1], result[3][2], result[3][3]));
  //return R;
  
  //return unity;
}

// End of special copyright notice

__device__ inline mat3 transpose_f33 (mat3 A)
{
  float r[3][3];
  float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
  		 {A.c1.x, A.c1.y, A.c1.z},
  		 {A.c2.x, A.c2.y, A.c2.z}};
  
  for( int i = 0; i < 3; ++i)
  {
    for( int j = 0; j < 3; ++j)
    {
      r[i][j] = a[j][i];
    }
  }
  mat3 R = make_mat3(make_float3(r[0][0], r[0][1], r[0][2]), 
  make_float3(r[1][0], r[1][1], r[1][2]), make_float3(r[2][0], r[2][1], r[2][2]));
  return R;
}

__device__ inline mat4 transpose_f44 (mat4 A)
{
  float r[4][4];
  float a[4][4] =	{{A.c0.x, A.c0.y, A.c0.z, A.c0.w},
  		 {A.c1.x, A.c1.y, A.c1.z, A.c1.w},
  		 {A.c2.x, A.c2.y, A.c2.z, A.c2.w},
  		 {A.c3.x, A.c3.y, A.c3.z, A.c3.w}};
  for( int i = 0; i < 4; ++i)
  {
    for( int j = 0; j < 4; ++j)
    {
      r[i][j] = a[j][i];
    }
  }
  mat4 R = make_mat4(make_float4(r[0][0], r[0][1], r[0][2], r[0][3]), 
  make_float4(r[1][0], r[1][1], r[1][2], r[1][3]), make_float4(r[2][0], r[2][1], r[2][2], r[2][3]),
  make_float4(r[3][0], r[3][1], r[3][2], r[3][3]));
  return R;
}

__device__ inline float3 mult_f3_f33 (float3 X, mat3 A)
{
  float r[3];
  float x[3] = {X.x, X.y, X.z};
  float a[3][3] =	{{A.c0.x, A.c0.y, A.c0.z},
  		 			{A.c1.x, A.c1.y, A.c1.z},
  		 			{A.c2.x, A.c2.y, A.c2.z}};
  for( int i = 0; i < 3; ++i)
  {
    r[i] = 0.0f;
    for( int j = 0; j < 3; ++j)
    {
      r[i] = r[i] + x[j] * a[j][i];
    }
  }
  return make_float3(r[0], r[1], r[2]);
}

__device__ inline float3 mult_f3_f44 (float3 X, mat4 A)
{
  float r[3];
  float x[3] = {X.x, X.y, X.z};
  float a[4][4] =	{{A.c0.x, A.c0.y, A.c0.z, A.c0.w},
  		 {A.c1.x, A.c1.y, A.c1.z, A.c1.w},
  		 {A.c2.x, A.c2.y, A.c2.z, A.c2.w},
  		 {A.c3.x, A.c3.y, A.c3.z, A.c3.w}};
  for( int i = 0; i < 3; ++i)
  {
    r[i] = 0.0;
    for( int j = 0; j < 3; ++j)
    {
      r[i] = r[i] + x[j] * a[j][i];
    }
    r[i] = r[i] + a[3][i];
  }
  float s = 1.0f / (x[0] * a[0][3] + x[1] * a[1][3] + x[2] * a[2][3] + a[3][3]);
  for( int k = 0; k < 3; ++k)
  {
    r[k] = r[k] * s;
  }
  return make_float3(r[0], r[1], r[2]);
}

__device__ inline mat3 RGBtoXYZ( Chromaticities N)
{
mat3 M = make_mat3(make_float3(N.red.x, N.red.y, 1.0f - (N.red.x + N.red.y)),
make_float3(N.green.x, N.green.y, 1.0f - (N.green.x + N.green.y)),
make_float3(N.blue.x, N.blue.y, 1.0f - (N.blue.x + N.blue.y)));

float3 wh = make_float3(N.white.x / N.white.y, 1.0f, (1.0f - (N.white.x + N.white.y)) / N.white.y);
wh = mult_f3_f33(wh, invert_f33(M));
mat3 WH =  make_mat3(make_float3(wh.x, 0.0f, 0.0f), 
make_float3(0.0f, wh.y, 0.0f), make_float3(0.0f, 0.0f, wh.z));
//M = mult_f33_f33(M, WH);
M = mult_f33_f33(WH, M);
return M;
}

__device__ inline mat4 RGBtoXYZ( Chromaticities N, float W)
{
mat3 A = RGBtoXYZ(N);
mat4 M = make_mat4(A, W);
return M;
}
/*
__device__ inline mat4 RGBtoXYZ( Chromaticities N, float W)
{
mat3 A = make_mat3(make_float3(N.red.x, N.red.y, 1.0f - (N.red.x + N.red.y)),
				make_float3(N.green.x, N.green.y, 1.0f - (N.green.x + N.green.y)),
				make_float3(N.blue.x, N.blue.y, 1.0f - (N.blue.x + N.blue.y)));

float3 wh = make_float3(N.white.x / N.white.y, 1.0f, (1.0f - (N.white.x + N.white.y)) / N.white.y);
wh = mult_f3_f33(wh, invert_f33(A));
mat3 WH =  make_mat3(make_float3(wh.x, 0.0f, 0.0f), make_float3(0.0f, wh.y, 0.0f), make_float3(0.0f, 0.0f, wh.z));
A = mult_f33_f33(A, WH);
mat4 M = make_mat4(A, W);
return M;
}
*/
__device__ inline mat3 XYZtoRGB( Chromaticities N)
{
mat3 M = invert_f33(RGBtoXYZ(N));
return M;
}

__device__ inline mat4 XYZtoRGB( Chromaticities N, float W)
{
mat3 A = XYZtoRGB(N);
mat4 M = make_mat4(A, W);
return M;
}
/*
__device__ inline mat4 XYZtoRGB( Chromaticities N, float W)
{
mat4 M = invert_f44(RGBtoXYZ(N, W));
return M;
}
*/

__device__ inline float SLog3_to_linear( float SLog )
{
	float out;

	if (SLog >= 171.2102946929f / 1023.0f)
	{
		out = powf(10.0f, (SLog * 1023.0f - 420.0f) / 261.5f) * (0.18f + 0.01f) - 0.01f;
	}
	else
	{
		out = (SLog * 1023.0f - 95.0f) * 0.01125000f / (171.2102946929f - 95.0f);
	}

	return out;
}

__device__ inline float vLogToLinScene( float x)
{
	const float cutInv = 0.181f;
	const float b = 0.00873f;
	const float c = 0.241514f;
	const float d = 0.598206f;
	
	if (x <= cutInv)
		return (x - 0.125f) / 5.6f;
	else
		return powf(10.0f, (x - d) / c) - b;
}

__device__ inline float CanonLog_to_linear ( float clog_ire)
{
return (powf(10.0f, (clog_ire - 0.0730597f) / 0.529136f) - 1) / 10.1596f;
}

#endif