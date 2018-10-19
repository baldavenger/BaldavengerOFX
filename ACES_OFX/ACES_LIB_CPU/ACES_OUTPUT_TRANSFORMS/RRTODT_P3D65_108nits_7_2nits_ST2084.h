#ifndef __RRTODT_P3D65_108NITS_7_2NITS_ST2084_H_INCLUDED__
#define __RRTODT_P3D65_108NITS_7_2NITS_ST2084_H_INCLUDED__

// 
// Output Transform - P3D65 (108 cd/m^2)
//

//
// Summary :
//  This transform maps ACES onto a P3D65 ST.2084 HDR display calibrated 
//  to a D65 white point at 108 cd/m^2. The assumed observer adapted white is 
//  D65, and the viewing environment is that of a dark surround. Mid-gray
//  luminance is targeted at 7.2 cd/m^2.
//
//  A use case for this transform would be mastering for a theatrical release
//  in Dolby Cinema.
//
// Device Primaries : 
//  Primaries are those specified in Rec. ITU-R BT.2020
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.68      0.32
//              Green:        0.265     0.69
//              Blue:         0.15      0.06
//              White:        0.3127    0.329     108 cd/m^2
//              18% Gray:     0.3127    0.329     7.2 cd/m^2
//
// Display EOTF :
//  The reference electro-optical transfer function specified in SMPTE ST 
//  2084-2014. This transform makes no attempt to address the Annex functions 
//  which address integer quantization.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.3127       0.329
//
// Viewing Environment:
//  Environment specified in SMPTE RP 431-2-2007
//



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


inline float3 RRTODT_P3D65_108nits_7_2nits_ST2084( float3 aces) 
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
    
float3 cv = outputTransform( aces, Y_MIN,
								   Y_MID,
								   Y_MAX,
								   DISPLAY_PRI,
								   LIMITING_PRI,
								   EOTF,
								   SURROUND,
								   STRETCH_BLACK,
								   D60_SIM,
								   LEGAL_RANGE );

return cv;
}

#endif