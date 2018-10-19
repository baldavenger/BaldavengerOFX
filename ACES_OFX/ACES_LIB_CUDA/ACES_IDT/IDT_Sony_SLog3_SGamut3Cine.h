#ifndef __IDT_SONY_SLOG3_SGAMUT3CINE_H_INCLUDED__
#define __IDT_SONY_SLOG3_SGAMUT3CINE_H_INCLUDED__

//------------------------------------------------------------------------------------
//  S-Gamut 3.Cine To ACES(Primaries0) matrix
//------------------------------------------------------------------------------------
/*
__CONSTANT__ mat3 matrixCoef =
{
	{  0.6387886672f, -0.0039159060f, -0.0299072021f },
	{  0.2723514337f,  1.0880732309f, -0.0264325799f },
	{  0.0888598991f, -0.0841573249f,  1.0563397820f }
};

//------------------------------------------------------------------------------------
//  S-Log 3 to linear
//------------------------------------------------------------------------------------
__device__ inline float SLog3_to_linear( float SLog )
{
	float out;

	if (SLog >= 171.2102946929f / 1023.0f)
	{
		out = _powf(10.0f, (SLog * 1023.0f - 420.0f) / 261.5f) * (0.18f + 0.01f) - 0.01f;
	}
	else
	{
		out = (SLog * 1023.0f - 95.0f) * 0.01125000f / (171.2102946929f - 95.0f);
	}

	return out;
}
*/
//------------------------------------------------------------------------------------
//  main
//------------------------------------------------------------------------------------
__device__ inline float3 IDT_Sony_SLog3_SGamut3Cine( float3 SLog3)
{

const mat3 matrixCoef = { {  0.6387886672f, -0.0039159060f, -0.0299072021f },
						{  0.2723514337f,  1.0880732309f, -0.0264325799f },
						{  0.0888598991f, -0.0841573249f,  1.0563397820f } };
	
	float3 linear;
	linear.x = SLog3_to_linear( SLog3.x );
	linear.y = SLog3_to_linear( SLog3.y );
	linear.z = SLog3_to_linear( SLog3.z );

	float3 aces = mult_f3_f33( linear, matrixCoef );

	return aces;
}

#endif