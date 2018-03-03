__device__ float3 ADD3(float3 A, float3 B)
{
	float3 C;
	C.x = A.x + B.x;
	C.y = A.y + B.y;
	C.z = A.z + B.z;

	return C;
}

__device__ float snoise(float3 uv, float res)
{
	//const float3 s = make_float3(1e0, 1e2, 1e4);
	
	uv.x *= res;
	uv.y *= res;
	uv.z *= res;
	
	float3 uv0;
	uv0.x = floor(fmod(uv.x, res)) * 1e0;
	uv0.y = floor(fmod(uv.y, res)) * 1e2;
	uv0.z = floor(fmod(uv.z, res)) * 1e4;
	
	float3 uv1;
	uv1.x = floor(fmod(uv.x + 1.0f, res)) * 1e0;
	uv1.y = floor(fmod(uv.y + 1.0f, res)) * 1e2;
	uv1.z = floor(fmod(uv.z + 1.0f, res)) * 1e4;
	
	float3 FR;
	FR.x = uv.x - floor(uv.x);
	FR.x = FR.x * FR.x * (3.0f - 2.0f * FR.x);
	FR.y = uv.y - floor(uv.y);
	FR.y = FR.y * FR.y * (3.0f - 2.0f * FR.y);
	FR.z = uv.z - floor(uv.z);
	FR.z = FR.z * FR.z * (3.0f - 2.0f * FR.z);
	
	float4 v = make_float4(uv0.x + uv0.y + uv0.z, uv1.x + uv0.y + uv0.z,
		      	  uv0.x + uv1.y + uv0.z, uv1.x + uv1.y + uv0.z);
    
	float4 r;
	float RX = sinf(v.x * 1e-3) * 1e5;
	r.x = RX - floor(RX);
	float RY = sinf(v.y * 1e-3) * 1e5;
	r.y = RY - floor(RY);
	float RZ = sinf(v.z * 1e-3) * 1e5;
	r.z = RZ - floor(RZ);
	float RW = sinf(v.w * 1e-3) * 1e5;
	r.w = RW - floor(RW);
	
	float AA = r.x * (1.0f - FR.x) + r.y * FR.x;
	float BB = r.z * (1.0f - FR.x) + r.w * FR.x;
	float r0 = AA * (1.0f - FR.y) + BB * FR.y;
	
	RX = sinf((v.x + uv1.z - uv0.z) * 1e-3) * 1e5;
	r.x = RX - floor(RX);
	RY = sinf((v.y + uv1.z - uv0.z) * 1e-3) * 1e5;
	r.y = RY - floor(RY);
	RZ = sinf((v.z + uv1.z - uv0.z) * 1e-3) * 1e5;
	r.z = RZ - floor(RZ);
	RW = sinf((v.w + uv1.z - uv0.z) * 1e-3) * 1e5;
	r.w = RW - floor(RW);
	
	AA = r.x * (1.0f - FR.x) + r.y * FR.x;
	BB = r.z * (1.0f - FR.x) + r.w * FR.x;
	float r1 = AA * (1.0f - FR.y) + BB * FR.y;
	
	float NOSE = (r0 * (1.0f - FR.z) + r1 * FR.z) * 2.0f - 1.0f;
	
	return NOSE;
}

__global__ void SauronKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
float p_Speed, float p_LengthA, float p_LengthB, float p_Horizontal, float p_Vertical, float p_Opacity, float p_PluginTime)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
    const int index = (y * p_Width + x) * 4;
    
    float adsk_time = p_PluginTime;
	float paramFlameSpeed = p_Speed;
	
	float2 p;
	
	p.x = float(x) / p_Width;
	p.x -= 0.5f;
	p.x *= (float)(p_Width / p_Height);
	p.x -= p_Horizontal;
	
	p.y = float(y) / p_Height;
	p.y -= 0.5f;
	p.y -= p_Vertical;
	
	float LengthA = sqrtf(p.x * p.x * 2.0f + p.y * p.y * 2.0f);
	LengthA *= 1.0f / p_LengthA;
	float LengthB = sqrtf(p.x * p.x + p.y * p.y);
	LengthB *= powf(2.0f, p_LengthB);
	
	float color = 3.0f - (3.0f * LengthA);
	
	float3 coord;
	coord = make_float3(atan2f(p.x, p.y) / 6.2832f + 0.5f, LengthB * 0.4f, 0.5f);
	
	for(int i = 1; i <= 7; i++)
	{
		float power = powf(2.0f, float(i));
		color += (1.5f / power) * snoise(ADD3(coord, make_float3(0.0f, -adsk_time * paramFlameSpeed / 100.0f * 0.05f, adsk_time * paramFlameSpeed / 100.0f * 0.01f)), power * 16.0f);
	}
	
	float3 RGB;
	RGB = make_float3(color, powf(fmaxf(color, 0.0f), 2.0f) * 0.4f, powf(fmaxf(color, 0.0f), 3.0f) * 0.15f);
	      																										   
	p_Output[index + 0] = 0.7f * RGB.x * p_Opacity;
	p_Output[index + 1] = 0.7f * RGB.y * p_Opacity;
	p_Output[index + 2] = 0.7f * RGB.z * p_Opacity;
	p_Output[index + 3] = p_Input[index + 3];
  }
}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
float p_Speed, float p_LengthA, float p_LengthB, float p_Horizontal, float p_Vertical, float p_Opacity, float p_PluginTime)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    SauronKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Speed, p_LengthA, p_LengthB, 
    p_Horizontal, p_Vertical, p_Opacity, p_PluginTime);
}
