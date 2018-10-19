#ifndef __ACES_LMT_BLUELIGHTARTIFACTFIX_H_INCLUDED__
#define __ACES_LMT_BLUELIGHTARTIFACTFIX_H_INCLUDED__

//
// LMT for desaturating blue hues in order to reduce the artifact where bright 
// blue colors (e.g. sirens, headlights, LED lighting, etc.) may become 
// oversaturated or exhibit hue shifts as a result of clipping.
// 

#include "../ACES_LMT.h"


inline float3 LMT_BlueLightArtifactFix( float3 aces)
{

mat3 correctionMatrix = { { 0.9404372683f,  0.0083786969f,  0.0005471261f},
  						{-0.0183068787f,  0.8286599939f, -0.0008833746f},
  						{ 0.0778696104f,  0.1629613092f,  1.0003362486f} };

float3 acesMod = mult_f3_f33( aces, correctionMatrix);

return acesMod;

}

#endif