#ifndef __ADX16_TO_ACES_H_INCLUDED__
#define __ADX16_TO_ACES_H_INCLUDED__

// 
// Universal ADX16 to ACES Transform
//


__constant__ mat3 CDD_TO_CID = { {0.75573f, 0.05901f, 0.16134f},
    							{0.22197f, 0.96928f, 0.07406f},
    							{0.02230f, -0.02829f, 0.76460f} };

__constant__ mat3 EXP_TO_ACES = { {0.72286f, 0.11923f, 0.01427f},
    							{0.12630f, 0.76418f, 0.08213f},
    							{0.15084f, 0.11659f, 0.90359f} };

__constant__ float2 LUT_1D[11] = { {-0.190000000000000, -6.000000000000000},
    							{ 0.010000000000000, -2.721718645000000},
    							{ 0.028000000000000, -2.521718645000000},
    							{ 0.054000000000000, -2.321718645000000},
    							{ 0.095000000000000, -2.121718645000000},
   								{ 0.145000000000000, -1.921718645000000},
   		 						{ 0.220000000000000, -1.721718645000000},
    							{ 0.300000000000000, -1.521718645000000},
    							{ 0.400000000000000, -1.321718645000000},
    							{ 0.500000000000000, -1.121718645000000},
    							{ 0.600000000000000, -0.926545676714876} };

#define REF_PT 					(7120.0f - 1520.0f) / 8000.0f * (100.0f / 55.0f) - log10f(0.18f)


__device__ inline float3 ADX16_to_ACES( float3 ADX16)
{
    // Prepare input values based on application bit depth handling
    float3 adx;
    adx.x = ADX16.x * 65535.0f;
    adx.y = ADX16.y * 65535.0f;
    adx.z = ADX16.z * 65535.0f;

    // Convert ADX16 values to Channel Dependent Density values
    float3 cdd = ( adx - 1520.0f) / 8000.0f;
    
    // Convert Channel Dependent Density values into Channel Independent Density values
    float3 cid = mult_f3_f33( cdd, CDD_TO_CID);

    // Convert Channel Independent Density values to Relative Log Exposure values
    float3 logE;
    if ( cid.x <= 0.6f) logE.x = interpolate1D( LUT_1D, cid.x);
    if ( cid.y <= 0.6f) logE.y = interpolate1D( LUT_1D, cid.y);
    if ( cid.z <= 0.6f) logE.z = interpolate1D( LUT_1D, cid.z);

    if ( cid.x > 0.6f) logE.x = ( 100.0f / 55.0f) * cid.x - REF_PT;
    if ( cid.y > 0.6f) logE.y = ( 100.0f / 55.0f) * cid.y - REF_PT;
    if ( cid.z > 0.6f) logE.z = ( 100.0f / 55.0f) * cid.z - REF_PT;

    // Convert Relative Log Exposure values to Relative Exposure values
    float3 exp;
    exp.x = powf( 10.0f, logE.x);
    exp.y = powf( 10.0f, logE.y);
    exp.z = powf( 10.0f, logE.z);

    // Convert Relative Exposure values to ACES values
    float3 aces = mult_f3_f33( exp, EXP_TO_ACES);
	return aces;
}

#endif