#ifndef __INVRRTODT_P3D65_108NITS_7_2NITS_ST2084_H_INCLUDED__
#define __INVRRTODT_P3D65_108NITS_7_2NITS_ST2084_H_INCLUDED__

//import "ACESlib.Utilities";
//import "ACESlib.OutputTransforms";


/*
const float Y_MIN = 0.0001;                     // black luminance (cd/m^2)
const float Y_MID = 7.2;                        // mid-point luminance (cd/m^2)
const float Y_MAX = 108.0;                      // peak white luminance (cd/m^2)

const Chromaticities DISPLAY_PRI = P3D65_PRI;   // encoding primaries (device setup)
const Chromaticities LIMITING_PRI = P3D65_PRI;  // limiting primaries

const int EOTF = 0;                             // 0: ST-2084 (PQ)
                                                // 1: BT.1886 (Rec.709/2020 settings) 
                                                // 2: sRGB (mon_curve w/ presets)
                                                // 3: gamma 2.6
                                                // 4: linear (no EOTF)
                                                // 5: HLG

const int SURROUND = 0;                         // 0: dark
                                                // 1: dim
                                                // 2: normal

const bool STRETCH_BLACK = true;                // stretch black luminance to a PQ code value of 0
const bool D60_SIM = false;                       
const bool LEGAL_RANGE = false;
*/

inline float3 InvRRTODT_P3D65_108nits_7_2nits_ST2084( float3 cv)
{

float Y_MIN = 0.0001f;
float Y_MID = 7.2f;
float Y_MAX = 108.0f;

Chromaticities DISPLAY_PRI = P3D65_PRI;
Chromaticities LIMITING_PRI = P3D65_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;                       
bool LEGAL_RANGE = false;

float3 aces = invOutputTransform( cv, Y_MIN,
									  Y_MID,
									  Y_MAX,
									  DISPLAY_PRI,
									  LIMITING_PRI,
									  EOTF,
									  SURROUND,
									  STRETCH_BLACK,
									  D60_SIM,
									  LEGAL_RANGE );

return aces;
}

#endif