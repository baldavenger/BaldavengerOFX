#ifndef __ACES_LMT_FIXHIGHLIGHTIMAGEARTIFACTS_H_INCLUDED__
#define __ACES_LMT_FIXHIGHLIGHTIMAGEARTIFACTS_H_INCLUDED__

// LMT Fix Highlights
//
// LMT for fixing occasional image artifacts seen in bright highlights (e.g. sirens,headlights,etc.)
// Note that this will change scene colorimetry! (but tends to do so in a pleasing way)
//

#include "../ACES_LMT.h"


__device__ inline float3 LMT_FixHighlightImageArtifacts( float3 aces)
{

mat3 correctionMatrix = { { 0.9404372683f,  0.0083786969f,  0.0005471261f},
  						{-0.0183068787f,  0.8286599939f, -0.0008833746f},
  						{ 0.0778696104f,  0.1629613092f,  1.0003362486f} };

float3 acesMod = mult_f3_f33( aces, correctionMatrix);

return acesMod;

}

#endif