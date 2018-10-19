#ifndef __INVRRTODT_REC2020_1000NITS_15NITS_HLG_H_INCLUDED__
#define __INVRRTODT_REC2020_1000NITS_15NITS_HLG_H_INCLUDED__

//import "ACESlib.Utilities";
//import "ACESlib.OutputTransforms";


// Support for HLG in ACES is tricky, since HLG is "scene-referred" and ACES 
// Output transforms produce display-referred output. ACES supports HLG at
// 1000 nits by converting the ST.2084 output to HLG using the method specified
// in Section 7 of ITU-R BT.2390-0. 

/*
const float Y_MIN = 0.0001;                     // black luminance (cd/m^2)
const float Y_MID = 15.0;                       // mid-point luminance (cd/m^2)
const float Y_MAX = 1000.0;                     // peak white luminance (cd/m^2)

const Chromaticities DISPLAY_PRI = REC2020_PRI; // encoding primaries (device setup)
const Chromaticities LIMITING_PRI = REC2020_PRI;// limiting primaries

const int EOTF = 5;                             // 0: ST-2084 (PQ)
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


inline float3 InvRRTODT_Rec2020_1000nits_15nits_HLG( float3 cv)
{

float Y_MIN = 0.0001f;
float Y_MID = 15.0f;
float Y_MAX = 1000.0f;

Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 5;
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