#ifndef __IDT_PANASONIC_V35_H_INCLUDED__
#define __IDT_PANASONIC_V35_H_INCLUDED__

// IDT for Panasonic Varicam 35
// by Panasonic



__device__ inline float3 IDT_Panasonic_V35( float3 VLog)
{

const mat3 mat = { {0.724382758f, 0.166748484f, 0.108497411f},
                 {0.021354009f, 0.985138372f, -0.006319092f},
                 {-0.009234278f, -0.00104295f, 1.010272625f} };
                        
	  // Convert V-Log to linear scene reflectance:

	  float rLin = vLogToLinScene(VLog.x);
	  float gLin = vLogToLinScene(VLog.y);
	  float bLin = vLogToLinScene(VLog.z);

	  // Apply IDT matrix:
	  float3 out;
	  out.x = mat.c0.x * rLin + mat.c0.y * gLin + mat.c0.z * bLin;
	  out.y = mat.c1.x * rLin + mat.c1.y * gLin + mat.c1.z * bLin;
	  out.z = mat.c2.x * rLin + mat.c2.y * gLin + mat.c2.z * bLin;
	  
	  return out;
}

#endif