#ifndef __ACES_SSTS_H_INCLUDED__
#define __ACES_SSTS_H_INCLUDED__

// Textbook monomial to basis-function conversion matrix.
__constant__ mat3 M1 = {
  {  0.5f, -1.0f, 0.5f },
  { -1.0f,  1.0f, 0.5f },
  {  0.5f,  0.0f, 0.0f }
};

typedef struct
{
    float x;        // ACES
    float y;        // luminance
    float slope;    // 
} TsPoint;

typedef struct
{
    TsPoint Min;
    TsPoint Mid;
    TsPoint Max;
    float coefsLow[6];
    float coefsHigh[6];    
} TsParams;



// TODO: Move all "magic numbers" (i.e. values in interpolation tables, etc.) to top 
// and define as constants

__constant__ float MIN_STOP_SDR = -6.5f;
__constant__ float MAX_STOP_SDR = 6.5f;

__constant__ float MIN_STOP_RRT = -15.0f;
__constant__ float MAX_STOP_RRT = 18.0f;

__constant__ float MIN_LUM_SDR = 0.02f;
__constant__ float MAX_LUM_SDR = 48.0f;

__constant__ float MIN_LUM_RRT = 0.0001f;
__constant__ float MAX_LUM_RRT = 10000.0f;


__device__ inline float lookup_ACESmin( float minLum )
{
    float2 minTable[2] = { { log10f(MIN_LUM_RRT), MIN_STOP_RRT }, 
                                   { log10f(MIN_LUM_SDR), MIN_STOP_SDR } };

    return 0.18f * powf( 2.0f, interpolate1D( minTable, log10f( minLum)));
}

__device__ inline float lookup_ACESmax( float maxLum )
{
    float2 maxTable[2] = { { log10f(MAX_LUM_SDR), MAX_STOP_SDR }, 
                                   { log10f(MAX_LUM_RRT), MAX_STOP_RRT } };

    return 0.18f * powf( 2.0f, interpolate1D( maxTable, log10f( maxLum)));
}

__device__ inline float5 init_coefsLow(
    TsPoint TsPointLow,
    TsPoint TsPointMid
)
{
    float5 coefsLow;

    float knotIncLow = (log10f(TsPointMid.x) - log10f(TsPointLow.x)) / 3.0f;
    // float halfKnotInc = (log10f(TsPointMid.x) - log10f(TsPointLow.x)) / 6.;

    // Determine two lowest coefficients (straddling minPt)
    coefsLow.x = (TsPointLow.slope * (log10f(TsPointLow.x) - 0.5f * knotIncLow)) + ( log10f(TsPointLow.y) - TsPointLow.slope * log10f(TsPointLow.x));
    coefsLow.y = (TsPointLow.slope * (log10f(TsPointLow.x) + 0.5f * knotIncLow)) + ( log10f(TsPointLow.y) - TsPointLow.slope * log10f(TsPointLow.x));
    // NOTE: if slope=0, then the above becomes just 
        // coefsLow[0] = log10f(TsPointLow.y);
        // coefsLow[1] = log10f(TsPointLow.y);
    // leaving it as a variable for now in case we decide we need non-zero slope extensions

    // Determine two highest coefficients (straddling midPt)
    coefsLow.w = (TsPointMid.slope * (log10f(TsPointMid.x) - 0.5f * knotIncLow)) + ( log10f(TsPointMid.y) - TsPointMid.slope * log10f(TsPointMid.x));
    coefsLow.m = (TsPointMid.slope * (log10f(TsPointMid.x) + 0.5f * knotIncLow)) + ( log10f(TsPointMid.y) - TsPointMid.slope * log10f(TsPointMid.x));
    
    // Middle coefficient (which defines the "sharpness of the bend") is linearly interpolated
    float2 bendsLow[2] = { {MIN_STOP_RRT, 0.18f}, 
                    {MIN_STOP_SDR, 0.35f} };
    float pctLow = interpolate1D( bendsLow, log2f(TsPointLow.x / 0.18f));
    coefsLow.z = log10f(TsPointLow.y) + pctLow*(log10f(TsPointMid.y)-log10f(TsPointLow.y));

    return coefsLow;
} 

__device__ inline float5 init_coefsHigh( 
    TsPoint TsPointMid, 
    TsPoint TsPointMax
)
{
    float5 coefsHigh;

    float knotIncHigh = (log10f(TsPointMax.x) - log10f(TsPointMid.x)) / 3.0f;
    // float halfKnotInc = (log10f(TsPointMax.x) - log10f(TsPointMid.x)) / 6.;

    // Determine two lowest coefficients (straddling midPt)
    coefsHigh.x = (TsPointMid.slope * (log10f(TsPointMid.x) - 0.5f * knotIncHigh)) + ( log10f(TsPointMid.y) - TsPointMid.slope * log10f(TsPointMid.x));
    coefsHigh.y = (TsPointMid.slope * (log10f(TsPointMid.x) + 0.5f * knotIncHigh)) + ( log10f(TsPointMid.y) - TsPointMid.slope * log10f(TsPointMid.x));

    // Determine two highest coefficients (straddling maxPt)
    coefsHigh.w = (TsPointMax.slope * (log10f(TsPointMax.x) - 0.5f * knotIncHigh)) + ( log10f(TsPointMax.y) - TsPointMax.slope * log10f(TsPointMax.x));
    coefsHigh.m = (TsPointMax.slope * (log10f(TsPointMax.x) + 0.5f * knotIncHigh)) + ( log10f(TsPointMax.y) - TsPointMax.slope * log10f(TsPointMax.x));
    // NOTE: if slope=0, then the above becomes just
        // coefsHigh[0] = log10f(TsPointHigh.y);
        // coefsHigh[1] = log10f(TsPointHigh.y);
    // leaving it as a variable for now in case we decide we need non-zero slope extensions
    
    // Middle coefficient (which defines the "sharpness of the bend") is linearly interpolated
    float2 bendsHigh[2] = { {MAX_STOP_SDR, 0.89f}, 
                    	{MAX_STOP_RRT, 0.90f} };
    float pctHigh = interpolate1D( bendsHigh, log2f(TsPointMax.x / 0.18f));
    coefsHigh.z = log10f(TsPointMid.y) + pctHigh*(log10f(TsPointMax.y)-log10f(TsPointMid.y));
    
    return coefsHigh;
}


__device__ inline float shift( float in, float expShift)
{
    return powf(2.0f, (log2f(in)-expShift));
}


__device__ inline TsParams init_TsParams(
    float minLum,
    float maxLum,
    float expShift = 0
)
{
    TsPoint MIN_PT = { lookup_ACESmin(minLum), minLum, 0.0f};
    TsPoint MID_PT = { 0.18f, 4.8f, 1.55f};
    TsPoint MAX_PT = { lookup_ACESmax(maxLum), maxLum, 0.0f};
    float5 cLow;
    cLow = init_coefsLow( MIN_PT, MID_PT);
    float5 cHigh;
    cHigh = init_coefsHigh( MID_PT, MAX_PT);
    MIN_PT.x = shift(lookup_ACESmin(minLum),expShift);
    MID_PT.x = shift(0.18f, expShift);
    MAX_PT.x = shift(lookup_ACESmax(maxLum),expShift);

    TsParams P = {
        {MIN_PT.x, MIN_PT.y, MIN_PT.slope},
        {MID_PT.x, MID_PT.y, MID_PT.slope},
        {MAX_PT.x, MAX_PT.y, MAX_PT.slope},
        {cLow.x, cLow.y, cLow.z, cLow.w, cLow.m, cLow.m},
        {cHigh.x, cHigh.y, cHigh.z, cHigh.w, cHigh.m, cHigh.m}
    };
         
    return P;
}


__device__ inline float ssts
( 
    float x,
    TsParams C
)
{
    const int N_KNOTS_LOW = 4;
    const int N_KNOTS_HIGH = 4;

    // Check for negatives or zero before taking the log. If negative or zero,
    // set to HALF_MIN.
    float logx = log10f( fmax(x, HALF_MIN )); 

    float logy;

    if ( logx <= log10f(C.Min.x) ) { 

        logy = logx * C.Min.slope + ( log10f(C.Min.y) - C.Min.slope * log10f(C.Min.x) );

    } else if (( logx > log10f(C.Min.x) ) && ( logx < log10f(C.Mid.x) )) {

        float knot_coord = (N_KNOTS_LOW-1) * (logx-log10f(C.Min.x))/(log10f(C.Mid.x)-log10f(C.Min.x));
        int j = knot_coord;
        float t = knot_coord - j;

        float3 cf = { C.coefsLow[ j], C.coefsLow[ j + 1], C.coefsLow[ j + 2]};

        float3 monomials = { t * t, t, 1.0f };
        logy = dot_f3_f3( monomials, mult_f3_f33( cf, M1));

    } else if (( logx >= log10f(C.Mid.x) ) && ( logx < log10f(C.Max.x) )) {

        float knot_coord = (N_KNOTS_HIGH-1) * (logx-log10f(C.Mid.x))/(log10f(C.Max.x)-log10f(C.Mid.x));
        int j = knot_coord;
        float t = knot_coord - j;

        float3 cf = { C.coefsHigh[ j], C.coefsHigh[ j + 1], C.coefsHigh[ j + 2]}; 

        float3 monomials = { t * t, t, 1.0f };
        logy = dot_f3_f3( monomials, mult_f3_f33( cf, M1));

    } else { //if ( logIn >= log10f(C.Max.x) ) { 

        logy = logx * C.Max.slope + ( log10f(C.Max.y) - C.Max.slope * log10f(C.Max.x) );

    }

    return pow10f(logy);

}


__device__ inline float inv_ssts
( 
    float y,
    TsParams C
)
{  
    const int N_KNOTS_LOW = 4;
    const int N_KNOTS_HIGH = 4;

    const float KNOT_INC_LOW = (log10f(C.Mid.x) - log10f(C.Min.x)) / (N_KNOTS_LOW - 1.0f);
    const float KNOT_INC_HIGH = (log10f(C.Max.x) - log10f(C.Mid.x)) / (N_KNOTS_HIGH - 1.0f);

    // KNOT_Y is luminance of the spline at each knot
    float KNOT_Y_LOW[ N_KNOTS_LOW];
    for (int i = 0; i < N_KNOTS_LOW; i = i+1) {
    KNOT_Y_LOW[ i] = ( C.coefsLow[i] + C.coefsLow[i+1]) / 2.;
    };

    float KNOT_Y_HIGH[ N_KNOTS_HIGH];
    for (int i = 0; i < N_KNOTS_HIGH; i = i+1) {
    KNOT_Y_HIGH[ i] = ( C.coefsHigh[i] + C.coefsHigh[i+1]) / 2.0f;
    };

    float logy = log10f( max(y,1e-10));

    float logx;
    if (logy <= log10f(C.Min.y)) {

        logx = log10f(C.Min.x);

    } else if ( (logy > log10f(C.Min.y)) && (logy <= log10f(C.Mid.y)) ) {

        unsigned int j;
        float3 cf;
        if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) {
            cf.x = C.coefsLow[0];  cf.y = C.coefsLow[1];  cf.z = C.coefsLow[2];  j = 0;
        } else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
            cf.x = C.coefsLow[1];  cf.y = C.coefsLow[2];  cf.z = C.coefsLow[3];  j = 1;
        } else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
            cf.x = C.coefsLow[2];  cf.y = C.coefsLow[3];  cf.z = C.coefsLow[4];  j = 2;
        } 

        const float3 tmp = mult_f3_f33( cf, M1);

        float a = tmp.x;
        float b = tmp.y;
        float c = tmp.z;
        c = c - logy;

        const float d = sqrtf( b * b - 4.0f * a * c);

        const float t = ( 2.0f * c) / ( -d - b);

        logx = log10f(C.Min.x) + ( t + j) * KNOT_INC_LOW;

    } else if ( (logy > log10f(C.Mid.y)) && (logy < log10f(C.Max.y)) ) {

        unsigned int j;
        float3 cf;
        if ( logy >= KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
            cf.x = C.coefsHigh[0];  cf.y = C.coefsHigh[1];  cf.z = C.coefsHigh[2];  j = 0;
        } else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
            cf.x = C.coefsHigh[1];  cf.y = C.coefsHigh[2];  cf.z = C.coefsHigh[3];  j = 1;
        } else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
            cf.x = C.coefsHigh[2];  cf.y = C.coefsHigh[3];  cf.z = C.coefsHigh[4];  j = 2;
        } 

        const float3 tmp = mult_f3_f33( cf, M1);

        float a = tmp.x;
        float b = tmp.y;
        float c = tmp.z;
        c = c - logy;

        const float d = sqrtf( b * b - 4.0f * a * c);

        const float t = ( 2.0f * c) / ( -d - b);

        logx = log10f(C.Mid.x) + ( t + j) * KNOT_INC_HIGH;

    } else { //if ( logy >= log10f(C.Max.y) ) {

        logx = log10f(C.Max.x);

    }

    return pow10f( logx);

}


__device__ inline float3 ssts_f3
( 
    float3 x,
    TsParams C
)
{
    float3 out;
    out.x = ssts( x.x, C);
    out.y = ssts( x.y, C);
    out.z = ssts( x.z, C);

    return out;
}


__device__ inline float3 inv_ssts_f3
( 
    float3 x,
    TsParams C
)
{
    float3 out;
    out.x = inv_ssts( x.x, C);
    out.y = inv_ssts( x.y, C);
    out.z = inv_ssts( x.z, C);

    return out;
}

#endif