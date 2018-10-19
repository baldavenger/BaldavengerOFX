#include "ACESPlugin.h"

#include "ACES_LIB_CPU/ACES_IDT.h"
#include "ACES_LIB_CPU/ACES_LMT.h"
#include "ACES_LIB_CPU/ACES_RRT.h"
#include "ACES_LIB_CPU/ACES_ODT.h"

#include <cstring>
#include <cmath>
#include <stdio.h>
using std::string;
#include <string> 
#include <fstream>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#ifdef __APPLE__
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/ACES_lut"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT\\ACES_lut"
#else
#define kPluginScript "/home/resolve/LUT/ACES_lut"
#endif

#define kPluginName "ACES 1.1"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"ACES 1.1"

#define kPluginIdentifier "OpenFX.Yo.ACESPlugin"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 2

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kParamInput "Input"
#define kParamInputLabel "Process"
#define kParamInputHint "Standard or Inverse"
#define kParamInputOptionStandard "Standard"
#define kParamInputOptionStandardHint "Standard"
#define kParamInputOptionInverse "Inverse"
#define kParamInputOptionInverseHint "Inverse"

enum InputEnum
{
    eInputStandard,
    eInputInverse,
};

#define kParamIDT "IDT"
#define kParamIDTLabel "IDT"
#define kParamIDTHint "IDT"
#define kParamIDTOptionBypass "Bypass"
#define kParamIDTOptionBypassHint "Bypass"
#define kParamIDTOptionACEScc "ACEScc"
#define kParamIDTOptionACESccHint "ACEScc to ACES"
#define kParamIDTOptionACEScct "ACEScct"
#define kParamIDTOptionACEScctHint "ACEScct to ACES"
#define kParamIDTOptionAlexaLogC800 "Alexa LogC EI800"
#define kParamIDTOptionAlexaLogC800Hint "Alexa LogC EI800 to ACES"
#define kParamIDTOptionAlexaRaw800 "Alexa Raw EI800"
#define kParamIDTOptionAlexaRaw800Hint "Alexa Raw EI800 to ACES"
#define kParamIDTOptionADX10 "ADX10"
#define kParamIDTOptionADX10Hint "ADX10 to ACES"
#define kParamIDTOptionADX16 "ADX16"
#define kParamIDTOptionADX16Hint "ADX16 to ACES"
#define kParamIDTOptionPanasonicV35 "Panasonic V35 VLog"
#define kParamIDTOptionPanasonicV35Hint "Panasonic V35 VLog"
#define kParamIDTOptionREDWideGamutRGBLog3G10 "Log3G10 REDWideGamutRGB"
#define kParamIDTOptionREDWideGamutRGBLog3G10Hint "Log3G10 REDWideGamutRGB"

#define kParamIDTOptionCanonC100AD55 "Canon C100 A D55"
#define kParamIDTOptionCanonC100AD55Hint "Canon CLog C100 A D55"
#define kParamIDTOptionCanonC100ATNG "Canon C100 A Tungsten"
#define kParamIDTOptionCanonC100ATNGHint "Canon CLog C100 A Tungsten"
#define kParamIDTOptionCanonC100mk2AD55 "Canon C100mk2 A D55"
#define kParamIDTOptionCanonC100mk2AD55Hint "Canon CLog C100mk2 A D55"
#define kParamIDTOptionCanonC100mk2ATNG "Canon C100mk2 A Tungsten"
#define kParamIDTOptionCanonC100mk2ATNGHint "Canon CLog C100mk2 A Tungsten"

#define kParamIDTOptionCanonC300AD55 "Canon C300 A D55"
#define kParamIDTOptionCanonC300AD55Hint "Canon CLog C300 A D55"
#define kParamIDTOptionCanonC300ATNG "Canon C300 A Tungsten"
#define kParamIDTOptionCanonC300ATNGHint "Canon CLog C300 A Tungsten"

#define kParamIDTOptionCanonC500AD55 "Canon C500 A D55"
#define kParamIDTOptionCanonC500AD55Hint "Canon CLog C500 A D55"
#define kParamIDTOptionCanonC500ATNG "Canon C500 A Tungsten"
#define kParamIDTOptionCanonC500ATNGHint "Canon CLog C500 A Tungsten"
#define kParamIDTOptionCanonC500BD55 "Canon C500 B D55"
#define kParamIDTOptionCanonC500BD55Hint "Canon CLog C500 B D55"
#define kParamIDTOptionCanonC500BTNG "Canon C500 B Tungsten"
#define kParamIDTOptionCanonC500BTNGHint "Canon CLog C500 B Tungsten"
#define kParamIDTOptionCanonC500CinemaGamutAD55 "Canon C500 CinemaGamut A D55"
#define kParamIDTOptionCanonC500CinemaGamutAD55Hint "Canon CLog C500 CinemaGamut A D55"
#define kParamIDTOptionCanonC500CinemaGamutATNG "Canon C500 CinemaGamut A Tungsten"
#define kParamIDTOptionCanonC500CinemaGamutATNGHint "Canon CLog C500 CinemaGamut A Tungsten"
#define kParamIDTOptionCanonC500DCIP3AD55 "Canon C500 DCI P3 A D55"
#define kParamIDTOptionCanonC500DCIP3AD55Hint "Canon CLog C500 DCI P3 A D55"
#define kParamIDTOptionCanonC500DCIP3ATNG "Canon C500 DCI P3 A Tungsten"
#define kParamIDTOptionCanonC500DCIP3ATNGHint "Canon CLog C500 DCI P3 A Tungsten"

#define kParamIDTOptionCanonC300mk2CanonLogBT2020DD55 "Canon C300mk2 BT2020 D D55"
#define kParamIDTOptionCanonC300mk2CanonLogBT2020DD55Hint "Canon CLog C300mk2 BT2020 D D55"
#define kParamIDTOptionCanonC300mk2CanonLogBT2020DTNG "Canon C300mk2 BT2020 D Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLogBT2020DTNGHint "Canon CLog C300mk2 BT2020 D Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCD55 "Canon C300mk2 CinemaGamut C D55"
#define kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCD55Hint "Canon CLog C300mk2 CinemaGamut C D55"
#define kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCTNG "Canon C300mk2 CinemaGamut C Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCTNGHint "Canon CLog C300mk2 CinemaGamut C Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLog2BT2020BD55 "Canon CLog2 C300mk2 BT2020 B D55"
#define kParamIDTOptionCanonC300mk2CanonLog2BT2020BD55Hint "Canon CLog2 C300mk2 BT2020 B D55"
#define kParamIDTOptionCanonC300mk2CanonLog2BT2020BTNG "Canon CLog2 C300mk2 BT2020 B Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLog2BT2020BTNGHint "Canon CLog2 C300mk2 BT2020 B Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutAD55 "Canon CLog2 C300mk2 CinemaGamut A D55"
#define kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutAD55Hint "Canon CLog2 C300mk2 CinemaGamut A D55"
#define kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutATNG "Canon CLog2 C300mk2 CinemaGamut A Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutATNGHint "Canon CLog2 C300mk2 CinemaGamut A Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLog3BT2020FD55 "Canon CLog3 C300mk2 BT2020 F D55"
#define kParamIDTOptionCanonC300mk2CanonLog3BT2020FD55Hint "Canon CLog3 C300mk2 BT2020 F D55"
#define kParamIDTOptionCanonC300mk2CanonLog3BT2020FTNG "Canon CLog3 C300mk2 BT2020 F Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLog3BT2020FTNGHint "Canon CLog3 C300mk2 BT2020 F Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutED55 "Canon CLog3 C300mk2 CinemaGamut E D55"
#define kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutED55Hint "Canon CLog3 C300mk2 CinemaGamut E D55"
#define kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutETNG "Canon CLog3 C300mk2 CinemaGamut E Tungsten"
#define kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutETNGHint "Canon CLog3 C300mk2 CinemaGamut E Tungsten"

#define kParamIDTOptionSonySLog1SGamut10 "Sony SLog1 SGamut 10"
#define kParamIDTOptionSonySLog1SGamut10Hint "Sony SLog1 SGamut 10"
#define kParamIDTOptionSonySLog1SGamut12 "Sony SLog1 SGamut 12"
#define kParamIDTOptionSonySLog1SGamut12Hint "Sony SLog1 SGamut 12"
#define kParamIDTOptionSonySLog2SGamutDaylight10 "Sony SLog2 SGamut Daylight 10"
#define kParamIDTOptionSonySLog2SGamutDaylight10Hint "Sony SLog2 SGamut Daylight 10"
#define kParamIDTOptionSonySLog2SGamutDaylight12 "Sony SLog2 SGamut Daylight 12"
#define kParamIDTOptionSonySLog2SGamutDaylight12Hint "Sony SLog2 SGamut Daylight 12"
#define kParamIDTOptionSonySLog2SGamutTungsten10 "Sony SLog2 SGamut Tungsten 10"
#define kParamIDTOptionSonySLog2SGamutTungsten10Hint "Sony SLog2 SGamut Tungsten 10"
#define kParamIDTOptionSonySLog2SGamutTungsten12 "Sony SLog2 SGamut Tungsten 12"
#define kParamIDTOptionSonySLog2SGamutTungsten12Hint "Sony SLog2 SGamut Tungsten 12"
#define kParamIDTOptionSonySLog3SGamut3 "Sony SLog3 SGamut3"
#define kParamIDTOptionSonySLog3SGamut3Hint "Sony SLog3 SGamut3"
#define kParamIDTOptionSonySLog3SGamut3Cine "Sony SLog3 SGamut3Cine"
#define kParamIDTOptionSonySLog3SGamut3CineHint "Sony SLog3 SGamut3Cine"

enum IDTEnum
{
    eIDTBypass,
    eIDTACEScc,
    eIDTACEScct,
    eIDTAlexaLogC800,
    eIDTAlexaRaw800,
    eIDTADX10,
    eIDTADX16,
    eIDTPanasonicV35,
    eIDTREDWideGamutRGBLog3G10,
    eIDTCanonC100AD55,
    eIDTCanonC100ATNG,
    eIDTCanonC100mk2AD55,
    eIDTCanonC100mk2ATNG,
    eIDTCanonC300AD55,
    eIDTCanonC300ATNG,
    eIDTCanonC500AD55,
    eIDTCanonC500ATNG,
    eIDTCanonC500BD55,
    eIDTCanonC500BTNG,
    eIDTCanonC500CinemaGamutAD55,
    eIDTCanonC500CinemaGamutATNG,
    eIDTCanonC500DCIP3AD55,
    eIDTCanonC500DCIP3ATNG,
    eIDTCanonC300mk2CanonLogBT2020DD55,
    eIDTCanonC300mk2CanonLogBT2020DTNG,
    eIDTCanonC300mk2CanonLogCinemaGamutCD55,
    eIDTCanonC300mk2CanonLogCinemaGamutCTNG,
    eIDTCanonC300mk2CanonLog2BT2020BD55,
    eIDTCanonC300mk2CanonLog2BT2020BTNG,
    eIDTCanonC300mk2CanonLog2CinemaGamutAD55,
    eIDTCanonC300mk2CanonLog2CinemaGamutATNG,
    eIDTCanonC300mk2CanonLog3BT2020FD55,
    eIDTCanonC300mk2CanonLog3BT2020FTNG,
    eIDTCanonC300mk2CanonLog3CinemaGamutED55,
    eIDTCanonC300mk2CanonLog3CinemaGamutETNG,
    eIDTSonySLog1SGamut10,
    eIDTSonySLog1SGamut12,
    eIDTSonySLog2SGamutDaylight10,
    eIDTSonySLog2SGamutDaylight12,
    eIDTSonySLog2SGamutTungsten10,
    eIDTSonySLog2SGamutTungsten12,
    eIDTSonySLog3SGamut3,
    eIDTSonySLog3SGamut3Cine,
};

#define kParamACESIN "ACESIN"
#define kParamACESINLabel "ACES to"
#define kParamACESINHint "Convert from ACES to"
#define kParamACESINOptionBypass "Bypass"
#define kParamACESINOptionBypassHint "Bypass"
#define kParamACESINOptionACEScc "ACEScc"
#define kParamACESINOptionACESccHint "ACEScc"
#define kParamACESINOptionACEScct "ACEScct"
#define kParamACESINOptionACEScctHint "ACEScct"
#define kParamACESINOptionACEScg "ACEScg"
#define kParamACESINOptionACEScgHint "ACEScg"
#define kParamACESINOptionACESproxy10 "ACESproxy10"
#define kParamACESINOptionACESproxy10Hint "ACESproxy10"
#define kParamACESINOptionACESproxy12 "ACESproxy12"
#define kParamACESINOptionACESproxy12Hint "ACESproxy12"


enum ACESINEnum
{
    eACESINBypass,
    eACESINACEScc,
    eACESINACEScct,
    eACESINACEScg,
    eACESINACESproxy10,
    eACESINACESproxy12,
};

#define kParamLMT "LMT"
#define kParamLMTLabel "LMT"
#define kParamLMTHint "LMT"
#define kParamLMTOptionBypass "Bypass"
#define kParamLMTOptionBypassHint "Bypass"
#define kParamLMTOptionCustom "Custom"
#define kParamLMTOptionCustomHint "Custom LMT"
#define kParamLMTOptionBleach "Bleach Bypass"
#define kParamLMTOptionBleachHint "Bleach Bypass"
#define kParamLMTOptionPFE "PFE"
#define kParamLMTOptionPFEHint "Print Film Emulation"
#define kParamLMTOptionFix "Blue Light Fix"
#define kParamLMTOptionFixHint "Blue Light Artifact Fix"

enum LMTEnum
{
    eLMTBypass,
    eLMTCustom,
    eLMTBleach,
    eLMTPFE,
    eLMTFix,
};

#define kParamACESOUT "ACESOUT"
#define kParamACESOUTLabel "ACES from"
#define kParamACESOUTHint "Convert to ACES from "
#define kParamACESOUTOptionBypass "Bypass"
#define kParamACESOUTOptionBypassHint "Bypass"
#define kParamACESOUTOptionACEScc "ACEScc"
#define kParamACESOUTOptionACESccHint "ACEScc"
#define kParamACESOUTOptionACEScct "ACEScct"
#define kParamACESOUTOptionACEScctHint "ACEScct"
#define kParamACESOUTOptionACEScg "ACEScg"
#define kParamACESOUTOptionACEScgHint "ACEScg"
#define kParamACESOUTOptionACESproxy10 "ACESproxy10"
#define kParamACESOUTOptionACESproxy10Hint "ACESproxy10"
#define kParamACESOUTOptionACESproxy12 "ACESproxy12"
#define kParamACESOUTOptionACESproxy12Hint "ACESproxy12"


enum ACESOUTEnum
{
    eACESOUTBypass,
    eACESOUTACEScc,
    eACESOUTACEScct,
    eACESOUTACEScg,
    eACESOUTACESproxy10,
    eACESOUTACESproxy12,
};

#define kParamRRT "RRT"
#define kParamRRTLabel "RRT"
#define kParamRRTHint "RRT"
#define kParamRRTOptionBypass "Bypass"
#define kParamRRTOptionBypassHint "Bypass"
#define kParamRRTOptionEnabled "Enabled"
#define kParamRRTOptionEnabledHint "Enabled"

enum RRTEnum
{
    eRRTBypass,
    eRRTEnabled,
};

#define kParamInvRRT "InvRRT"
#define kParamInvRRTLabel "Inverse RRT"
#define kParamInvRRTHint "Inverse RRT"
#define kParamInvRRTOptionBypass "Bypass"
#define kParamInvRRTOptionBypassHint "Bypass"
#define kParamInvRRTOptionEnabled "Enabled"
#define kParamInvRRTOptionEnabledHint "Enabled"

enum InvRRTEnum
{
    eInvRRTBypass,
    eInvRRTEnabled,
};

#define kParamODT "ODT"
#define kParamODTLabel "ODT"
#define kParamODTHint "ODT"
#define kParamODTOptionBypass "Bypass"
#define kParamODTOptionBypassHint "Bypass"
#define kParamODTOptionCustom "Custom"
#define kParamODTOptionCustomHint "Custom ODT"
#define kParamODTOptionACEScc "ACEScc"
#define kParamODTOptionACESccHint "ACES to ACEScc"
#define kParamODTOptionACEScct "ACEScct"
#define kParamODTOptionACEScctHint "ACES to ACEScct"
#define kParamODTOptionRec709_100dim "Rec709 100nits Dim"
#define kParamODTOptionRec709_100dimHint "Rec.709 100nits Dim"
#define kParamODTOptionRec709_D60sim_100dim "Rec709 D60sim 100nits Dim"
#define kParamODTOptionRec709_D60sim_100dimHint "Rec.709 D60sim 100nits Dim"
#define kParamODTOptionSRGB_100dim "sRGB 100nits Dim"
#define kParamODTOptionSRGB_100dimHint "sRGB 100nits Dim"
#define kParamODTOptionSRGB_D60sim_100dim "sRGB D60sim 100nits Dim"
#define kParamODTOptionSRGB_D60sim_100dimHint "sRGB D60sim 100nits Dim"
#define kParamODTOptionRec2020_100dim "Rec2020 100nits Dim"
#define kParamODTOptionRec2020_100dimHint "Rec.2020 100nits Dim"
#define kParamODTOptionRec2020_Rec709limited_100dim "Rec2020 Rec709limited 100nits Dim"
#define kParamODTOptionRec2020_Rec709limited_100dimHint "Rec.2020 Rec709limited 100nits Dim"
#define kParamODTOptionRec2020_P3D65limited_100dim "Rec2020 P3D65limited 100nits Dim"
#define kParamODTOptionRec2020_P3D65limited_100dimHint "Rec.2020 P3D65limited 100nits Dim"
#define kParamODTOptionRec2020_ST2084_1000 "Rec2020 ST2084 1000nits"
#define kParamODTOptionRec2020_ST2084_1000Hint "Rec.2020 ST2084 1000nits"
#define kParamODTOptionP3DCI_48 "P3DCI 48nits"
#define kParamODTOptionP3DCI_48Hint "P3DCI 48nits"
#define kParamODTOptionP3DCI_D60sim_48 "P3DCI D60sim 48nits"
#define kParamODTOptionP3DCI_D60sim_48Hint "P3DCI D60sim 48nits"
#define kParamODTOptionP3DCI_D65sim_48 "P3DCI D65sim 48nits"
#define kParamODTOptionP3DCI_D65sim_48Hint "P3DCI D65sim 48nits"
#define kParamODTOptionP3D60_48 "P3D60 48nits"
#define kParamODTOptionP3D60_48Hint "P3D60 48nits"
#define kParamODTOptionP3D65_48 "P3D65 48nits"
#define kParamODTOptionP3D65_48Hint "P3D65 48nits"
#define kParamODTOptionP3D65_D60sim_48 "P3D65 D60sim 48nits"
#define kParamODTOptionP3D65_D60sim_48Hint "P3D65 D60sim 48nits"
#define kParamODTOptionP3D65_Rec709limited_48 "P3D65 Rec709limited 48nits"
#define kParamODTOptionP3D65_Rec709limited_48Hint "P3D65 Rec709limited 48nits"
#define kParamODTOptionDCDM "DCDM"
#define kParamODTOptionDCDMHint "DCDM"
#define kParamODTOptionDCDM_P3D60limited "DCDM P3D60limited"
#define kParamODTOptionDCDM_P3D60limitedHint "DCDM P3D60limited"
#define kParamODTOptionDCDM_P3D65limited "DCDM P3D65limited"
#define kParamODTOptionDCDM_P3D65limitedHint "DCDM P3D65limited"
#define kParamODTOptionRGBmonitor_100dim "RGB monitor 100nits Dim"
#define kParamODTOptionRGBmonitor_100dimHint "RGB monitor 100nits Dim"
#define kParamODTOptionRGBmonitor_D60sim_100dim "RGB monitor D60sim 100nits Dim"
#define kParamODTOptionRGBmonitor_D60sim_100dimHint "RGB monitor D60sim 100nits Dim"
#define kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084 "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLG "RRTODT Rec2020 1000nits 15nits HLG"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint "RRTODT Rec.2020 1000nits 15nits HLG"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084 "RRTODT Rec2020 1000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint "RRTODT Rec.2020 1000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084 "RRTODT Rec2020 2000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint "RRTODT Rec.2020 2000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084 "RRTODT Rec2020 4000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint "RRTODT Rec.2020 4000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec709_100nits_10nits_BT1886 "RRTODT Rec709 100nits 10nits BT1886"
#define kParamODTOptionRRTODT_Rec709_100nits_10nits_BT1886Hint "RRTODT Rec.709 100nits 10nits BT1886"
#define kParamODTOptionRRTODT_Rec709_100nits_10nits_sRGB "RRTODT Rec709 100nits 10nits sRGB"
#define kParamODTOptionRRTODT_Rec709_100nits_10nits_sRGBHint "RRTODT Rec.709 100nits 10nits sRGB"


enum ODTEnum
{
    eODTBypass,
    eODTCustom,
    eODTACEScc,
    eODTACEScct,
    eODTRec709_100dim,
    eODTRec709_D60sim_100dim,
    eODTSRGB_100dim,
    eODTSRGB_D60sim_100dim,
    eODTRec2020_100dim,
    eODTRec2020_Rec709limited_100dim,
	eODTRec2020_P3D65limited_100dim,
    eODTRec2020_ST2084_1000,
    eODTP3DCI_48,
    eODTP3DCI_D60sim_48,
    eODTP3DCI_D65sim_48,
    eODTP3D60_48,
    eODTP3D65_48,
    eODTP3D65_D60sim_48,
    eODTP3D65_Rec709limited_48,
	eODTDCDM,
	eODTDCDM_P3D60limited,
	eODTDCDM_P3D65limited,
    eODTRGBmonitor_100dim,
    eODTRGBmonitor_D60sim_100dim,
    eODTRRTODT_P3D65_108nits_7_2nits_ST2084,
    eODTRRTODT_Rec2020_1000nits_15nits_HLG,
    eODTRRTODT_Rec2020_1000nits_15nits_ST2084,
    eODTRRTODT_Rec2020_2000nits_15nits_ST2084,
    eODTRRTODT_Rec2020_4000nits_15nits_ST2084,
    eODTRRTODT_Rec709_100nits_10nits_BT1886,
    eODTRRTODT_Rec709_100nits_10nits_sRGB,
};


#define kParamInvODT "InvODT"
#define kParamInvODTLabel "Inverse ODT"
#define kParamInvODTHint "Inverse ODT"
#define kParamInvODTOptionBypass "Bypass"
#define kParamInvODTOptionBypassHint "Bypass"
#define kParamInvODTOptionCustom "Custom"
#define kParamInvODTOptionCustomHint "Custom Inverse ODT"
#define kParamInvODTOptionRec709_100dim "Rec709 100nits Dim"
#define kParamInvODTOptionRec709_100dimHint "Rec.709 100nits Dim"
#define kParamInvODTOptionRec709_D60sim_100dim "Rec709 D60sim 100nits Dim"
#define kParamInvODTOptionRec709_D60sim_100dimHint "Rec.709 D60sim 100nits Dim"
#define kParamInvODTOptionSRGB_100dim "sRGB 100nits Dim"
#define kParamInvODTOptionSRGB_100dimHint "sRGB 100nits Dim"
#define kParamInvODTOptionSRGB_D60sim_100dim "sRGB D60sim 100nits Dim"
#define kParamInvODTOptionSRGB_D60sim_100dimHint "sRGB D60sim 100nits Dim"
#define kParamInvODTOptionRec2020_100dim "Rec2020 100nits Dim"
#define kParamInvODTOptionRec2020_100dimHint "Rec.2020 100nits Dim"
#define kParamInvODTOptionRec2020_ST2084_1000 "Rec2020 ST2084 1000nits"
#define kParamInvODTOptionRec2020_ST2084_1000Hint "Rec.2020 ST2084 1000nits"
#define kParamInvODTOptionP3DCI_48 "P3DCI 48nits"
#define kParamInvODTOptionP3DCI_48Hint "P3DCI 48nits"
#define kParamInvODTOptionP3DCI_D60sim_48 "P3DCI D60sim 48nits"
#define kParamInvODTOptionP3DCI_D60sim_48Hint "P3DCI D60sim 48nits"
#define kParamInvODTOptionP3DCI_D65sim_48 "P3DCI D65sim 48nits"
#define kParamInvODTOptionP3DCI_D65sim_48Hint "P3DCI D65sim 48nits"
#define kParamInvODTOptionP3D60_48 "P3D60 48nits"
#define kParamInvODTOptionP3D60_48Hint "P3D60 48nits"
#define kParamInvODTOptionP3D65_48 "P3D65 48nits"
#define kParamInvODTOptionP3D65_48Hint "P3D65 48nits"
#define kParamInvODTOptionP3D65_D60sim_48 "P3D65 D60sim 48nits"
#define kParamInvODTOptionP3D65_D60sim_48Hint "P3D65 D60sim 48nits"
#define kParamInvODTOptionDCDM "DCDM"
#define kParamInvODTOptionDCDMHint "DCDM"
#define kParamInvODTOptionDCDM_P3D65limited "DCDM P3D65limited"
#define kParamInvODTOptionDCDM_P3D65limitedHint "DCDM P3D65limited"
#define kParamInvODTOptionRGBmonitor_100dim "RGB monitor 100nits Dim"
#define kParamInvODTOptionRGBmonitor_100dimHint "RGB monitor 100nits Dim"
#define kParamInvODTOptionRGBmonitor_D60sim_100dim "RGB monitor D60sim 100nits Dim"
#define kParamInvODTOptionRGBmonitor_D60sim_100dimHint "RGB monitor D60sim 100nits Dim"
#define kParamInvODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084 "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamInvODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_HLG "RRTODT Rec2020 1000nits 15nits HLG"
#define kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint "RRTODT Rec2020 1000nits 15nits HLG"
#define kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084 "RRTODT Rec2020 1000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint "RRTODT Rec2020 1000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084 "RRTODT Rec2020 2000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint "RRTODT Rec2020 2000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084 "RRTODT Rec2020 4000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint "RRTODT Rec2020 4000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec709_100nits_10nits_BT1886 "RRTODT Rec709 100nits 10nits BT1886"
#define kParamInvODTOptionRRTODT_Rec709_100nits_10nits_BT1886Hint "RRTODT Rec.709 100nits 10nits BT1886"
#define kParamInvODTOptionRRTODT_Rec709_100nits_10nits_sRGB "RRTODT Rec709 100nits 10nits sRGB"
#define kParamInvODTOptionRRTODT_Rec709_100nits_10nits_sRGBHint "RRTODT Rec.709 100nits 10nits sRGB"

enum InvODTEnum
{
    eInvODTBypass,
    eInvODTCustom,
    eInvODTRec709_100dim,
    eInvODTRec709_D60sim_100dim,
    eInvODTSRGB_100dim,
    eInvODTSRGB_D60sim_100dim,
    eInvODTRec2020_100dim,
    eInvODTRec2020_ST2084_1000,
    eInvODTP3DCI_48,
    eInvODTP3DCI_D60sim_48,
    eInvODTP3DCI_D65sim_48,
    eInvODTP3D60_48,
    eInvODTP3D65_48,
	eInvODTP3D65_D60sim_48,
	eInvODTDCDM,
	eInvODTDCDM_P3D65limited,
    eInvODTRGBmonitor_100dim,
    eInvODTRGBmonitor_D60sim_100dim,
    eInvODTRRTODT_P3D65_108nits_7_2nits_ST2084,
    eInvODTRRTODT_Rec2020_1000nits_15nits_HLG,
    eInvODTRRTODT_Rec2020_1000nits_15nits_ST2084,
    eInvODTRRTODT_Rec2020_2000nits_15nits_ST2084,
    eInvODTRRTODT_Rec2020_4000nits_15nits_ST2084,
    eInvODTRRTODT_Rec709_100nits_10nits_BT1886,
    eInvODTRRTODT_Rec709_100nits_10nits_sRGB,
};

#define kParamDISPLAY "DISPLAY"
#define kParamDISPLAYLabel "Display Primaries"
#define kParamDISPLAYHint "display primaries"
#define kParamDISPLAYOptionRec2020 "Rec2020"
#define kParamDISPLAYOptionRec2020Hint "Rec.2020 primaries"
#define kParamDISPLAYOptionP3D60 "P3D60"
#define kParamDISPLAYOptionP3D60Hint "P3D60 primaries"
#define kParamDISPLAYOptionP3D65 "P3D65"
#define kParamDISPLAYOptionP3D65Hint "P3D65 primaries"
#define kParamDISPLAYOptionP3DCI "P3DCI"
#define kParamDISPLAYOptionP3DCIHint "P3DCI primaries"
#define kParamDISPLAYOptionRec709 "Rec709"
#define kParamDISPLAYOptionRec709Hint "Rec.709 primaries"

enum DISPLAYEnum
{
    eDISPLAYRec2020,
    eDISPLAYP3D60,
    eDISPLAYP3D65,
    eDISPLAYP3DCI,
    eDISPLAYRec709,
};

#define kParamLIMIT "LIMIT"
#define kParamLIMITLabel "Limiting Primaries"
#define kParamLIMITHint "limiting primaries"
#define kParamLIMITOptionRec2020 "Rec2020"
#define kParamLIMITOptionRec2020Hint "Rec.2020 primaries"
#define kParamLIMITOptionP3D60 "P3D60"
#define kParamLIMITOptionP3D60Hint "P3D60 primaries"
#define kParamLIMITOptionP3D65 "P3D65"
#define kParamLIMITOptionP3D65Hint "P3D65 primaries"
#define kParamLIMITOptionP3DCI "P3DCI"
#define kParamLIMITOptionP3DCIHint "P3DCI primaries"
#define kParamLIMITOptionRec709 "Rec709"
#define kParamLIMITOptionRec709Hint "Rec.709 primaries"

enum LIMITEnum
{
    eLIMITRec2020,
    eLIMITP3D60,
    eLIMITP3D65,
    eLIMITP3DCI,
    eLIMITRec709,
};

#define kParamEOTF "EOTF"
#define kParamEOTFLabel "EOTF"
#define kParamEOTFHint "EOTF"
#define kParamEOTFOptionST2084 "ST2084"
#define kParamEOTFOptionST2084Hint "ST-2084 PQ"
#define kParamEOTFOptionBT1886 "BT1886"
#define kParamEOTFOptionBT1886Hint "BT1886 primaries"
#define kParamEOTFOptionsRGB "sRGB"
#define kParamEOTFOptionsRGBHint "sRGB"
#define kParamEOTFOptionGAMMA26 "Gamma 2.6"
#define kParamEOTFOptionGAMMA26Hint "gamma 2.6"
#define kParamEOTFOptionLINEAR "Linear"
#define kParamEOTFOptionLINEARHint "linear (no EOTF)"
#define kParamEOTFOptionHLG "HLG"
#define kParamEOTFOptionHLGHint "hybrid log gamma"

enum EOTFEnum
{
    eEOTFST2084,
    eEOTFBT1886,
    eEOTFsRGB,
    eEOTFGAMMA26,
    eEOTFLINEAR,
    eEOTFHLG,
};

#define kParamSURROUND "SURROUND"
#define kParamSURROUNDLabel "Surround"
#define kParamSURROUNDHint "Surround"
#define kParamSURROUNDOptionDark "Dark"
#define kParamSURROUNDOptionDarkHint "Dark"
#define kParamSURROUNDOptionDim "Dim"
#define kParamSURROUNDOptionDimHint "Dim"
#define kParamSURROUNDOptionNormal "Normal"
#define kParamSURROUNDOptionNormalHint "Normal"

enum SURROUNDEnum
{
    eSURROUNDDark,
    eSURROUNDDim,
    eSURROUNDNormal,
    
};

////////////////////////////////////////////////////////////////////////////////

namespace {
    struct RGBValues {
        double r,g,b;
        RGBValues(double v) : r(v), g(v), b(v) {}
        RGBValues() : r(0), g(0), b(0) {}
    };
}

class ACES : public OFX::ImageProcessor
{
public:
    explicit ACES(OFX::ImageEffect& p_Instance);

    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);
    
    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(int p_Direction, int p_IDT, int p_ACESIN, int p_LMT, int p_ACESOUT, 
    int p_RRT, int p_InvRRT, int p_ODT, int p_InvODT, float p_Exposure, float *p_LMTScale, 
    float *p_Lum, int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_SWitch);

private:
    OFX::Image* _srcImg;
    int _direction;
    int _idt;
    int _acesin;
    int _lmt;
    int _acesout;
    int _rrt;
    int _invrrt;
    int _odt;
    int _invodt;
    float _exposure;
    float _lmtscale[24];
    float _lum[3];
    int _display;
    int _limit;
    int _eotf;
    int _surround;
    int _switch[3];
};

ACES::ACES(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

void ACES::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
{
if (_effect.abort()) break;
float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
{
float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
if (srcPix)
{
float3 aces = make_float3(srcPix[0], srcPix[1], srcPix[2]);
if(_direction == 0)
{
switch (_idt)
{
case 0:
{
}
break;
case 1:
{
aces = ACEScc_to_ACES(aces);
}
break;
case 2:
{
aces = ACEScct_to_ACES(aces);
}
break;
case 3:
{
aces = IDT_Alexa_v3_logC_EI800(aces);
}
break;
case 4:
{
aces = IDT_Alexa_v3_raw_EI800_CCT65(aces);
}
break;
case 5:
{
aces = ADX10_to_ACES(aces);
}
break;
case 6:
{
aces = ADX16_to_ACES(aces);
}
break;
case 7:
{
aces = IDT_Panasonic_V35(aces);
}
break;
case 8:
{
aces = IDT_REDWideGamutRGB_Log3G10(aces);
}
break;
case 9:
{
aces = IDT_Canon_C100_A_D55(aces);
}
break;
case 10:
{
aces = IDT_Canon_C100_A_Tng(aces);
}
break;
case 11:
{
aces = IDT_Canon_C100mk2_A_D55(aces);
}
break;
case 12:
{
aces = IDT_Canon_C100mk2_A_Tng(aces);
}
break;
case 13:
{
aces = IDT_Canon_C300_A_D55(aces);
}
break;
case 14:
{
aces = IDT_Canon_C300_A_Tng(aces);
}
break;
case 15:
{
aces = IDT_Canon_C500_A_D55(aces);
}
break;
case 16:
{
aces = IDT_Canon_C500_A_Tng(aces);
}
break;
case 17:
{
aces = IDT_Canon_C500_B_D55(aces);
}
break;
case 18:
{
aces = IDT_Canon_C500_B_Tng(aces);
}
break;
case 19:
{
aces = IDT_Canon_C500_CinemaGamut_A_D55(aces);
}
break;
case 20:
{
aces = IDT_Canon_C500_CinemaGamut_A_Tng(aces);
}
break;
case 21:
{
aces = IDT_Canon_C500_DCI_P3_A_D55(aces);
}
break;
case 22:
{
aces = IDT_Canon_C500_DCI_P3_A_Tng(aces);
}
break;
case 23:
{
aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_D55(aces);
}
break;
case 24:
{
aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng(aces);
}
break;
case 25:
{
aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55(aces);
}
break;
case 26:
{
aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng(aces);
}
break;
case 27:
{
aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55(aces);
}
break;
case 28:
{
aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng(aces);
}
break;
case 29:
{
aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55(aces);
}
break;
case 30:
{
aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng(aces);
}
break;
case 31:
{
aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55(aces);
}
break;
case 32:
{
aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng(aces);
}
break;
case 33:
{
aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55(aces);
}
break;
case 34:
{
aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng(aces);
}
break;
case 35:
{
aces = IDT_Sony_SLog1_SGamut_10(aces);
}
break;
case 36:
{
aces = IDT_Sony_SLog1_SGamut_12(aces);
}
break;
case 37:
{
aces = IDT_Sony_SLog2_SGamut_Daylight_10(aces);
}
break;
case 38:
{
aces = IDT_Sony_SLog2_SGamut_Daylight_12(aces);
}
break;
case 39:
{
aces = IDT_Sony_SLog2_SGamut_Tungsten_10(aces);
}
break;
case 40:
{
aces = IDT_Sony_SLog2_SGamut_Tungsten_12(aces);
}
break;
case 41:
{
aces = IDT_Sony_SLog3_SGamut3(aces);
}
break;
case 42:
{
aces = IDT_Sony_SLog3_SGamut3Cine(aces);
}
}
if(_exposure != 0.0f)
{
aces.x *= powf(2.0f, _exposure);
aces.y *= powf(2.0f, _exposure);
aces.z *= powf(2.0f, _exposure);
}
switch (_acesin)
{
case 0:
{
}
break;
case 1:
{
aces = ACES_to_ACEScc(aces);
}
break;
case 2:
{
aces = ACES_to_ACEScct(aces);
}
break;
case 3:
{
aces = ACES_to_ACEScg(aces);
}
break;
case 4:
{
aces = ACES_to_ACESproxy10(aces);
}
break;
case 5:
{
aces = ACES_to_ACESproxy12(aces);
}
}
switch (_lmt)
{
case 0:
{
}
break;
case 1:
{
if(_lmtscale[0] != 1.0f)
aces = scale_C(aces, _lmtscale[0]);
if(!(_lmtscale[1] == 1.0f && _lmtscale[2] == 0.0f && _lmtscale[3] == 1.0f))
{
float3 SLOPE = {_lmtscale[1], _lmtscale[1], _lmtscale[1]};
float3 OFFSET = {_lmtscale[2], _lmtscale[2], _lmtscale[2]};
float3 POWER = {_lmtscale[3], _lmtscale[3], _lmtscale[3]};
aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER);
}
if(_lmtscale[4] != 1.0f)
aces = gamma_adjust_linear(aces, _lmtscale[4], _lmtscale[5]);
if(_lmtscale[8] != 0.0f)
aces = rotate_H_in_H(aces, _lmtscale[6], _lmtscale[7], _lmtscale[8]);
if(_lmtscale[11] != 0.0f)
aces = rotate_H_in_H(aces, _lmtscale[9], _lmtscale[10], _lmtscale[11]);
if(_lmtscale[14] != 0.0f)
aces = rotate_H_in_H(aces, _lmtscale[12], _lmtscale[13], _lmtscale[14]);
if(_lmtscale[17] != 1.0f)
aces = scale_C_at_H(aces, _lmtscale[15], _lmtscale[16], _lmtscale[17]);
if(_lmtscale[20] != 0.0f)
aces = rotate_H_in_H(aces, _lmtscale[18], _lmtscale[19], _lmtscale[20]);
if(_lmtscale[23] != 1.0f)
aces = scale_C_at_H(aces, _lmtscale[21], _lmtscale[22], _lmtscale[23]);
}
break;
case 2:
{
aces = LMT_Analytic_4(aces);
}
break;
case 3:
{
aces = LMT_Analytic_3(aces);
}
break;
case 4:
{
aces = LMT_BlueLightArtifactFix(aces);
}
}
switch (_acesout)
{
case 0:
{
}
break;
case 1:
{
aces = ACEScc_to_ACES(aces);
}
break;
case 2:
{
aces = ACEScct_to_ACES(aces);
}
break;
case 3:
{
aces = ACEScg_to_ACES(aces);
}
break;
case 4:
{
aces = ACESproxy10_to_ACES(aces);
}
break;
case 5:
{
aces = ACESproxy12_to_ACES(aces);
}
}
if(_rrt == 1)
{
aces = _RRT(aces);
}
switch (_odt)
{
case 0:
{
}
break;
case 1:
{
float Y_MIN = _lum[0] * 0.0001f;
float Y_MID = _lum[1];
float Y_MAX = _lum[2];
Chromaticities DISPLAY_PRI = _display == 0 ? REC2020_PRI : _display == 1 ? P3D60_PRI : _display == 2 ? P3D65_PRI : _display == 3 ? P3DCI_PRI : REC709_PRI;
Chromaticities LIMITING_PRI = _limit == 0 ? REC2020_PRI : _limit == 1 ? P3D60_PRI : _limit == 2 ? P3D65_PRI : _limit == 3 ? P3DCI_PRI : REC709_PRI;
int EOTF = _eotf;
int SURROUND = _surround;		   
bool STRETCH_BLACK = _switch[0] == 1;
bool D60_SIM = _switch[1] == 1;
bool LEGAL_RANGE = _switch[2] == 1;
aces = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
}
break;
case 2:
{
aces = ACES_to_ACEScc(aces);
}
break;
case 3:
{
aces = ACES_to_ACEScct(aces);
}
break;
case 4:
{
aces = ODT_Rec709_100nits_dim(aces);
}
break;
case 5:
{
aces = ODT_Rec709_D60sim_100nits_dim(aces);
}
break;
case 6:
{
aces = ODT_sRGB_100nits_dim(aces);
}
break;
case 7:
{
aces = ODT_sRGB_D60sim_100nits_dim(aces);
}
break;
case 8:
{
aces = ODT_Rec2020_100nits_dim(aces);
}
break;
case 9:
{
aces = ODT_Rec2020_Rec709limited_100nits_dim(aces);
}
break;
case 10:
{
aces = ODT_Rec2020_P3D65limited_100nits_dim(aces);
}
break;
case 11:
{
aces = ODT_Rec2020_ST2084_1000nits(aces);
}
break;
case 12:
{
aces = ODT_P3DCI_48nits(aces);
}
break;
case 13:
{
aces = ODT_P3DCI_D60sim_48nits(aces);
}
break;
case 14:
{
aces = ODT_P3DCI_D65sim_48nits(aces);
}
break;
case 15:
{
aces = ODT_P3D60_48nits(aces);
}
break;
case 16:
{
aces = ODT_P3D65_48nits(aces);
}
break;
case 17:
{
aces = ODT_P3D65_D60sim_48nits(aces);
}
break;
case 18:
{
aces = ODT_P3D65_Rec709limited_48nits(aces);
}
break;
case 19:
{
aces = ODT_DCDM(aces);
}
break;
case 20:
{
aces = ODT_DCDM_P3D60limited(aces);
}
break;
case 21:
{
aces = ODT_DCDM_P3D65limited(aces);
}
break;
case 22:
{
aces = ODT_RGBmonitor_100nits_dim(aces);
}
break;
case 23:
{
aces = ODT_RGBmonitor_D60sim_100nits_dim(aces);
}
break;
case 24:
{
aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);
}
break;
case 25:
{
aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);
}
break;
case 26:
{
aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);
}
break;
case 27:
{
aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);
}
break;
case 28:
{
aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);
}
break;
case 29:
{
aces = RRTODT_Rec709_100nits_10nits_BT1886(aces);
}
break;
case 30:
{
aces = RRTODT_Rec709_100nits_10nits_sRGB(aces);
}
}
} else {
switch (_invodt)
{
case 0:
{
}
break;
case 1:
{
float Y_MIN = _lum[0] * 0.0001f;
float Y_MID = _lum[1];
float Y_MAX = _lum[2];
Chromaticities DISPLAY_PRI = _display == 0 ? REC2020_PRI : _display == 1 ? P3D60_PRI : _display == 2 ? P3D65_PRI : _display == 3 ? P3DCI_PRI : REC709_PRI;
Chromaticities LIMITING_PRI = _limit == 0 ? REC2020_PRI : _limit == 1 ? P3D60_PRI : _limit == 2 ? P3D65_PRI : _limit == 3 ? P3DCI_PRI : REC709_PRI;
int EOTF = _eotf;		
int SURROUND = _surround;		   
bool STRETCH_BLACK = _switch[0] == 1;
bool D60_SIM = _switch[1] == 1;
bool LEGAL_RANGE = _switch[2] == 1;
aces = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
}
break;
case 2:
{
aces = InvODT_Rec709_100nits_dim(aces);
}
break;
case 3:
{
aces = InvODT_Rec709_D60sim_100nits_dim(aces);
}
break;
case 4:
{
aces = InvODT_sRGB_100nits_dim(aces);
}
break;
case 5:
{
aces = InvODT_sRGB_D60sim_100nits_dim(aces);
}
break;
case 6:
{
aces = InvODT_Rec2020_100nits_dim(aces);
}
break;
case 7:
{
aces = InvODT_Rec2020_ST2084_1000nits(aces);
}
break;
case 8:
{
aces = InvODT_P3DCI_48nits(aces);
}
break;
case 9:
{
aces = InvODT_P3DCI_D60sim_48nits(aces);
}
break;
case 10:
{
aces = InvODT_P3DCI_D65sim_48nits(aces);
}
break;
case 11:
{
aces = InvODT_P3D60_48nits(aces);
}
break;
case 12:
{
aces = InvODT_P3D65_48nits(aces);
}
break;
case 13:
{
aces = InvODT_P3D65_D60sim_48nits(aces);
}
break;
case 14:
{
aces = InvODT_DCDM(aces);
}
break;
case 15:
{
aces = InvODT_DCDM_P3D65limited(aces);
}
break;
case 16:
{
aces = InvODT_RGBmonitor_100nits_dim(aces);
}
break;
case 17:
{
aces = InvODT_RGBmonitor_D60sim_100nits_dim(aces);
}
break;
case 18:
{
aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);
}
break;
case 19:
{
aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);
}
break;
case 20:
{
aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);
}
break;
case 21:
{
aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);
}
break;
case 22:
{
aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);
}
break;
case 23:
{
aces = InvRRTODT_Rec709_100nits_10nits_BT1886(aces);
}
break;
case 24:
{
aces = InvRRTODT_Rec709_100nits_10nits_sRGB(aces);
}
}
if(_invrrt == 1)
{
aces = _InvRRT(aces);
}
}
dstPix[0] = aces.x;
dstPix[1] = aces.y;
dstPix[2] = aces.z;
dstPix[3] = srcPix[3];
}
else
{
for (int c = 0; c < 4; ++c)
{
dstPix[c] = 0;
}
}

dstPix += 4;
}
}
}

void ACES::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ACES::setScales(int p_Direction, int p_IDT, int p_ACESIN, int p_LMT, int p_ACESOUT, int p_RRT, 
int p_InvRRT, int p_ODT, int p_InvODT, float p_Exposure, float *p_LMTScale, float *p_Lum, 
int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_Switch)
{
_direction = p_Direction;
_idt = p_IDT;
_acesin = p_ACESIN;
_lmt = p_LMT;
_acesout = p_ACESOUT;
_rrt = p_RRT;
_invrrt = p_InvRRT;
_odt = p_ODT;
_invodt = p_InvODT;
_exposure = p_Exposure;
_lmtscale[0] = p_LMTScale[0];
_lmtscale[1] = p_LMTScale[1];
_lmtscale[2] = p_LMTScale[2];
_lmtscale[3] = p_LMTScale[3];
_lmtscale[4] = p_LMTScale[4];
_lmtscale[5] = p_LMTScale[5];
_lmtscale[6] = p_LMTScale[6];
_lmtscale[7] = p_LMTScale[7];
_lmtscale[8] = p_LMTScale[8];
_lmtscale[9] = p_LMTScale[9];
_lmtscale[10] = p_LMTScale[10];
_lmtscale[11] = p_LMTScale[11];
_lmtscale[12] = p_LMTScale[12];
_lmtscale[13] = p_LMTScale[13];
_lmtscale[14] = p_LMTScale[14];
_lmtscale[15] = p_LMTScale[15];
_lmtscale[16] = p_LMTScale[16];
_lmtscale[17] = p_LMTScale[17];
_lmtscale[18] = p_LMTScale[18];
_lmtscale[19] = p_LMTScale[19];
_lmtscale[20] = p_LMTScale[20];
_lmtscale[21] = p_LMTScale[21];
_lmtscale[22] = p_LMTScale[22];
_lmtscale[23] = p_LMTScale[23];
_lum[0] = p_Lum[0];
_lum[1] = p_Lum[1];
_lum[2] = p_Lum[2];
_display = p_DISPLAY;
_limit = p_LIMIT;
_eotf = p_EOTF;
_surround = p_SURROUND;
_switch[0] = p_Switch[0];
_switch[1] = p_Switch[1];
_switch[2] = p_Switch[2];
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class ACESPlugin : public OFX::ImageEffect
{
public:
    explicit ACESPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
     /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(ACES &p_ACES, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    
    OFX::ChoiceParam* m_Direction;
    OFX::ChoiceParam* m_IDT;
    OFX::ChoiceParam* m_ACESIN;
    OFX::ChoiceParam* m_LMT;
    OFX::ChoiceParam* m_ACESOUT;
    OFX::ChoiceParam* m_RRT;
    OFX::ChoiceParam* m_InvRRT;
	OFX::ChoiceParam* m_ODT;
	OFX::ChoiceParam* m_InvODT;
	
	OFX::DoubleParam* m_Exposure;
	OFX::DoubleParam* m_ScaleC;
	OFX::DoubleParam* m_Slope;
	OFX::DoubleParam* m_Offset;
	OFX::DoubleParam* m_Power;
	OFX::DoubleParam* m_Gamma;
	OFX::DoubleParam* m_Pivot;
	OFX::DoubleParam* m_RotateH1;
	OFX::DoubleParam* m_Range1;
	OFX::DoubleParam* m_Shift1;
	OFX::DoubleParam* m_RotateH2;
	OFX::DoubleParam* m_Range2;
	OFX::DoubleParam* m_Shift2;
	OFX::DoubleParam* m_RotateH3;
	OFX::DoubleParam* m_Range3;
	OFX::DoubleParam* m_Shift3;
	OFX::DoubleParam* m_HueCH1;
	OFX::DoubleParam* m_RangeCH1;
	OFX::DoubleParam* m_ScaleCH1;
	OFX::DoubleParam* m_RotateH4;
	OFX::DoubleParam* m_Range4;
	OFX::DoubleParam* m_Shift4;
	OFX::DoubleParam* m_HueCH2;
	OFX::DoubleParam* m_RangeCH2;
	OFX::DoubleParam* m_ScaleCH2;
	
	OFX::DoubleParam* m_YMIN;
	OFX::DoubleParam* m_YMID;
	OFX::DoubleParam* m_YMAX;
	OFX::ChoiceParam* m_DISPLAY;
    OFX::ChoiceParam* m_LIMIT;
    OFX::ChoiceParam* m_EOTF;
    OFX::ChoiceParam* m_SURROUND;
    OFX::BooleanParam* m_STRETCH;
    OFX::BooleanParam* m_D60SIM;
    OFX::BooleanParam* m_LEGALRANGE;
	
	OFX::Double2DParam* m_Input;
	OFX::IntParam* m_Cube;
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
};

ACESPlugin::ACESPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    
    m_Direction = fetchChoiceParam(kParamInput);
    m_IDT = fetchChoiceParam(kParamIDT);
    m_ACESIN = fetchChoiceParam(kParamACESIN);
	m_LMT = fetchChoiceParam(kParamLMT);
	m_ACESOUT = fetchChoiceParam(kParamACESOUT);
	m_RRT = fetchChoiceParam(kParamRRT);
	m_InvRRT = fetchChoiceParam(kParamInvRRT);
	m_ODT = fetchChoiceParam(kParamODT);
	m_InvODT = fetchChoiceParam(kParamInvODT);		
	
	m_Exposure = fetchDoubleParam("Exposure");
	m_ScaleC = fetchDoubleParam("ScaleC");
	m_Slope = fetchDoubleParam("Slope");
	m_Offset = fetchDoubleParam("Offset");
	m_Power = fetchDoubleParam("Power");
	m_Gamma = fetchDoubleParam("Gamma");
	m_Pivot = fetchDoubleParam("Pivot");
	m_RotateH1 = fetchDoubleParam("RotateH1");
	m_Range1 = fetchDoubleParam("Range1");
	m_Shift1 = fetchDoubleParam("Shift1");
	m_RotateH2 = fetchDoubleParam("RotateH2");
	m_Range2 = fetchDoubleParam("Range2");
	m_Shift2 = fetchDoubleParam("Shift2");
	m_RotateH3 = fetchDoubleParam("RotateH3");
	m_Range3 = fetchDoubleParam("Range3");
	m_Shift3 = fetchDoubleParam("Shift3");
	m_HueCH1 = fetchDoubleParam("HueCH1");
	m_RangeCH1 = fetchDoubleParam("RangeCH1");
	m_ScaleCH1 = fetchDoubleParam("ScaleCH1");
	m_RotateH4 = fetchDoubleParam("RotateH4");
	m_Range4 = fetchDoubleParam("Range4");
	m_Shift4 = fetchDoubleParam("Shift4");
	m_HueCH2 = fetchDoubleParam("HueCH2");
	m_RangeCH2 = fetchDoubleParam("RangeCH2");
	m_ScaleCH2 = fetchDoubleParam("ScaleCH2");
	
	m_YMIN = fetchDoubleParam("YMIN");
	m_YMID = fetchDoubleParam("YMID");
	m_YMAX = fetchDoubleParam("YMAX");
	m_DISPLAY = fetchChoiceParam("DISPLAY");
	m_LIMIT = fetchChoiceParam("LIMIT");
	m_EOTF = fetchChoiceParam("EOTF");
	m_SURROUND = fetchChoiceParam("SURROUND");
	m_STRETCH = fetchBooleanParam("STRETCH");
	m_D60SIM = fetchBooleanParam("D60SIM");
	m_LEGALRANGE = fetchBooleanParam("LEGALRANGE");

    m_Input = fetchDouble2DParam("range");
    m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Cube = fetchIntParam("cube");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");

}

void ACESPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ACES ACES(*this);
        setupAndProcess(ACES, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool ACESPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{ 
    int IDT_i;
    m_IDT->getValueAtTime(p_Args.time, IDT_i);
    IDTEnum IDT = (IDTEnum)IDT_i;
    int _idt = IDT_i;
    bool idt = _idt == 0;
    
    int ACESIN_i;
    m_ACESIN->getValueAtTime(p_Args.time, ACESIN_i);
    ACESINEnum ACESIN = (ACESINEnum)ACESIN_i;
    int _acesin = ACESIN_i;
    bool acesin = _acesin == 0;
    
    int LMT_i;
    m_LMT->getValueAtTime(p_Args.time, LMT_i);
    LMTEnum LMT = (LMTEnum)LMT_i;
    int _lmt = LMT_i;
    bool lmt = _lmt == 0;
    
    int ACESOUT_i;
    m_ACESOUT->getValueAtTime(p_Args.time, ACESOUT_i);
    ACESOUTEnum ACESOUT = (ACESOUTEnum)ACESOUT_i;
    int _acesout = ACESOUT_i;
    bool acesout = _acesout == 0;
    
    int RRT_i;
    m_RRT->getValueAtTime(p_Args.time, RRT_i);
    RRTEnum RRT = (RRTEnum)RRT_i;
    int _rrt = RRT_i;
    bool rrt = _rrt == 0;
    
    int InvRRT_i;
    m_InvRRT->getValueAtTime(p_Args.time, InvRRT_i);
    InvRRTEnum InvRRT = (InvRRTEnum)InvRRT_i;
    int _invrrt = InvRRT_i;
    bool invrrt = _invrrt == 0;
    
    int ODT_i;
    m_ODT->getValueAtTime(p_Args.time, ODT_i);
    ODTEnum ODT = (ODTEnum)ODT_i;
    int _odt = ODT_i;
    bool odt = _odt == 0;
    
    int InvODT_i;
    m_InvODT->getValueAtTime(p_Args.time, InvODT_i);
    InvODTEnum InvODT = (InvODTEnum)InvODT_i;
    int _invodt = InvODT_i;
    bool invodt = _invodt == 0;
    
    float _exposure = m_Exposure->getValueAtTime(p_Args.time);
    bool exposure = _exposure = 0.0f;
    
    if (idt && acesin && lmt && acesout && rrt && invrrt && odt && invodt && exposure)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void ACESPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
 	
 	if(p_ParamName == kParamInput)
    {
 	int Direction_i;
    m_Direction->getValueAtTime(p_Args.time, Direction_i);
    InputEnum Direction = (InputEnum)Direction_i;
    int direction = Direction_i;
    
    bool forward = Direction_i == 0;
    bool inverse = Direction_i == 1;
    
    int LMT_i;
    m_LMT->getValueAtTime(p_Args.time, LMT_i);
    LMTEnum LMT = (LMTEnum)LMT_i;
    
	bool custom = LMT_i == 1;
    
	m_IDT->setIsSecretAndDisabled(!forward);
	m_ACESIN->setIsSecretAndDisabled(!forward);
	m_LMT->setIsSecretAndDisabled(!forward);
	m_ACESOUT->setIsSecretAndDisabled(!forward);
	m_RRT->setIsSecretAndDisabled(!forward);
	m_InvRRT->setIsSecretAndDisabled(forward);
	m_ODT->setIsSecretAndDisabled(!forward);
	m_InvODT->setIsSecretAndDisabled(forward);
	
	m_Exposure->setIsSecretAndDisabled(!forward);
	m_ScaleC->setIsSecretAndDisabled(!forward || !custom);
	m_Slope->setIsSecretAndDisabled(!forward || !custom);
	m_Offset->setIsSecretAndDisabled(!forward || !custom);
	m_Power->setIsSecretAndDisabled(!forward || !custom);
	m_Gamma->setIsSecretAndDisabled(!forward || !custom);
	m_Pivot->setIsSecretAndDisabled(!forward || !custom);
	m_RotateH1->setIsSecretAndDisabled(!forward || !custom);
	m_Range1->setIsSecretAndDisabled(!forward || !custom);
	m_Shift1->setIsSecretAndDisabled(!forward || !custom);
	m_RotateH2->setIsSecretAndDisabled(!forward || !custom);
	m_Range2->setIsSecretAndDisabled(!forward || !custom);
	m_Shift2->setIsSecretAndDisabled(!forward || !custom);
	m_RotateH3->setIsSecretAndDisabled(!forward || !custom);
	m_Range3->setIsSecretAndDisabled(!forward || !custom);
	m_Shift3->setIsSecretAndDisabled(!forward || !custom);
	m_HueCH1->setIsSecretAndDisabled(!forward || !custom);
	m_RangeCH1->setIsSecretAndDisabled(!forward || !custom);
	m_ScaleCH1->setIsSecretAndDisabled(!forward || !custom);
	m_RotateH4->setIsSecretAndDisabled(!forward || !custom);
	m_Range4->setIsSecretAndDisabled(!forward || !custom);
	m_Shift4->setIsSecretAndDisabled(!forward || !custom);
	m_HueCH2->setIsSecretAndDisabled(!forward || !custom);
	m_RangeCH2->setIsSecretAndDisabled(!forward || !custom);
	m_ScaleCH2->setIsSecretAndDisabled(!forward || !custom);
	
	int ODT_i;
    m_ODT->getValueAtTime(p_Args.time, ODT_i);
    ODTEnum ODT = (ODTEnum)ODT_i;
	bool odt = ODT_i == 1;
	
	int InvODT_i;
    m_InvODT->getValueAtTime(p_Args.time, InvODT_i);
    InvODTEnum InvODT = (InvODTEnum)InvODT_i;
	bool invodt = InvODT_i == 1;
	
	m_YMIN->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_YMID->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_YMAX->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_DISPLAY->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_LIMIT->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_EOTF->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_SURROUND->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_STRETCH->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_D60SIM->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	m_LEGALRANGE->setIsSecretAndDisabled(!(odt && forward) && !(invodt && inverse));
	
	}
	
	if(p_ParamName == kParamLMT)
    {
    
    int LMT_i;
    m_LMT->getValueAtTime(p_Args.time, LMT_i);
    LMTEnum LMT = (LMTEnum)LMT_i;
    
	bool custom = LMT_i == 1;
	
	m_ScaleC->setIsSecretAndDisabled(!custom);
	m_Slope->setIsSecretAndDisabled(!custom);
	m_Offset->setIsSecretAndDisabled(!custom);
	m_Power->setIsSecretAndDisabled(!custom);
	m_Gamma->setIsSecretAndDisabled(!custom);
	m_Pivot->setIsSecretAndDisabled(!custom);
	m_RotateH1->setIsSecretAndDisabled(!custom);
	m_Range1->setIsSecretAndDisabled(!custom);
	m_Shift1->setIsSecretAndDisabled(!custom);
	m_RotateH2->setIsSecretAndDisabled(!custom);
	m_Range2->setIsSecretAndDisabled(!custom);
	m_Shift2->setIsSecretAndDisabled(!custom);
	m_RotateH3->setIsSecretAndDisabled(!custom);
	m_Range3->setIsSecretAndDisabled(!custom);
	m_Shift3->setIsSecretAndDisabled(!custom);
	m_HueCH1->setIsSecretAndDisabled(!custom);
	m_RangeCH1->setIsSecretAndDisabled(!custom);
	m_ScaleCH1->setIsSecretAndDisabled(!custom);
	m_RotateH4->setIsSecretAndDisabled(!custom);
	m_Range4->setIsSecretAndDisabled(!custom);
	m_Shift4->setIsSecretAndDisabled(!custom);
	m_HueCH2->setIsSecretAndDisabled(!custom);
	m_RangeCH2->setIsSecretAndDisabled(!custom);
	m_ScaleCH2->setIsSecretAndDisabled(!custom);
	}
	
	if(p_ParamName == kParamODT)
    {
    
    int ODT_i;
    m_ODT->getValueAtTime(p_Args.time, ODT_i);
    ODTEnum ODT = (ODTEnum)ODT_i;
    
	bool custom = ODT_i == 1;
	
	m_YMIN->setIsSecretAndDisabled(!custom);
	m_YMID->setIsSecretAndDisabled(!custom);
	m_YMAX->setIsSecretAndDisabled(!custom);
	m_DISPLAY->setIsSecretAndDisabled(!custom);
	m_LIMIT->setIsSecretAndDisabled(!custom);
	m_EOTF->setIsSecretAndDisabled(!custom);
	m_SURROUND->setIsSecretAndDisabled(!custom);
	m_STRETCH->setIsSecretAndDisabled(!custom);
	m_D60SIM->setIsSecretAndDisabled(!custom);
	m_LEGALRANGE->setIsSecretAndDisabled(!custom);
	}
	
	if(p_ParamName == kParamInvODT)
    {
    
    int InvODT_i;
    m_InvODT->getValueAtTime(p_Args.time, InvODT_i);
    InvODTEnum InvODT = (InvODTEnum)InvODT_i;
    
	bool custom = InvODT_i == 1;
	
	m_YMIN->setIsSecretAndDisabled(!custom);
	m_YMID->setIsSecretAndDisabled(!custom);
	m_YMAX->setIsSecretAndDisabled(!custom);
	m_DISPLAY->setIsSecretAndDisabled(!custom);
	m_LIMIT->setIsSecretAndDisabled(!custom);
	m_EOTF->setIsSecretAndDisabled(!custom);
	m_SURROUND->setIsSecretAndDisabled(!custom);
	m_STRETCH->setIsSecretAndDisabled(!custom);
	m_D60SIM->setIsSecretAndDisabled(!custom);
	m_LEGALRANGE->setIsSecretAndDisabled(!custom);
	}
 	
 	   
    if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	
    int Direction_i;
    m_Direction->getValueAtTime(p_Args.time, Direction_i);
    InputEnum Direction = (InputEnum)Direction_i;
    int _direction = Direction_i;
	
	int IDT_i;
    m_IDT->getValueAtTime(p_Args.time, IDT_i);
    IDTEnum IDT = (IDTEnum)IDT_i;
    int _idt = IDT_i;
    
    int ACESIN_i;
    m_ACESIN->getValueAtTime(p_Args.time, ACESIN_i);
    ACESINEnum ACESIN = (ACESINEnum)ACESIN_i;
    int _acesin = ACESIN_i;
    
    int LMT_i;
    m_LMT->getValueAtTime(p_Args.time, LMT_i);
    LMTEnum LMT = (LMTEnum)LMT_i;
    int _lmt = LMT_i;
    
    int ACESOUT_i;
    m_ACESOUT->getValueAtTime(p_Args.time, ACESOUT_i);
    ACESOUTEnum ACESOUT = (ACESOUTEnum)ACESOUT_i;
    int _acesout = ACESOUT_i;
    
    int RRT_i;
    m_RRT->getValueAtTime(p_Args.time, RRT_i);
    RRTEnum RRT = (RRTEnum)RRT_i;
    int _rrt = RRT_i;
    
    int InvRRT_i;
    m_InvRRT->getValueAtTime(p_Args.time, InvRRT_i);
    InvRRTEnum InvRRT = (InvRRTEnum)InvRRT_i;
    int _invrrt = InvRRT_i;
    
    int ODT_i;
    m_ODT->getValueAtTime(p_Args.time, ODT_i);
    ODTEnum ODT = (ODTEnum)ODT_i;
    int _odt = ODT_i;
    
    int InvODT_i;
    m_InvODT->getValueAtTime(p_Args.time, InvODT_i);
    InvODTEnum InvODT = (InvODTEnum)InvODT_i;
    int _invodt = InvODT_i;
    
    float _exposure = m_Exposure->getValueAtTime(p_Args.time);
    float _lmtscale[24];
    _lmtscale[0] = m_ScaleC->getValueAtTime(p_Args.time);
    _lmtscale[1] = m_Slope->getValueAtTime(p_Args.time);
    _lmtscale[2] = m_Offset->getValueAtTime(p_Args.time);
    _lmtscale[3] = m_Power->getValueAtTime(p_Args.time);
    _lmtscale[4] = m_Gamma->getValueAtTime(p_Args.time);
    _lmtscale[5] = m_Pivot->getValueAtTime(p_Args.time);
    
    _lmtscale[6] = m_RotateH1->getValueAtTime(p_Args.time);
    _lmtscale[7] = m_Range1->getValueAtTime(p_Args.time);
    _lmtscale[8] = m_Shift1->getValueAtTime(p_Args.time);
    
    _lmtscale[9] = m_RotateH2->getValueAtTime(p_Args.time);
    _lmtscale[10] = m_Range2->getValueAtTime(p_Args.time);
    _lmtscale[11] = m_Shift2->getValueAtTime(p_Args.time);
    
    _lmtscale[12] = m_RotateH3->getValueAtTime(p_Args.time);
    _lmtscale[13] = m_Range3->getValueAtTime(p_Args.time);
    _lmtscale[14] = m_Shift3->getValueAtTime(p_Args.time);
    
    _lmtscale[15] = m_HueCH1->getValueAtTime(p_Args.time);
    _lmtscale[16] = m_RangeCH1->getValueAtTime(p_Args.time);
    _lmtscale[17] = m_ScaleCH1->getValueAtTime(p_Args.time);
    
    _lmtscale[18] = m_RotateH4->getValueAtTime(p_Args.time);
    _lmtscale[19] = m_Range4->getValueAtTime(p_Args.time);
    _lmtscale[20] = m_Shift4->getValueAtTime(p_Args.time);
    
    _lmtscale[21] = m_HueCH2->getValueAtTime(p_Args.time);
    _lmtscale[22] = m_RangeCH2->getValueAtTime(p_Args.time);
    _lmtscale[23] = m_ScaleCH2->getValueAtTime(p_Args.time);
    
    float _lum[3];
    _lum[0] = m_YMIN->getValueAtTime(p_Args.time);
    _lum[0] *= 0.0001f;
    _lum[1] = m_YMID->getValueAtTime(p_Args.time);
    _lum[2] = m_YMAX->getValueAtTime(p_Args.time);
    
    int DISPLAY_i;
    m_DISPLAY->getValueAtTime(p_Args.time, DISPLAY_i);
    DISPLAYEnum DISPLAY = (DISPLAYEnum)DISPLAY_i;
    int _display = DISPLAY_i;
    
    int LIMIT_i;
    m_LIMIT->getValueAtTime(p_Args.time, LIMIT_i);
    LIMITEnum LIMIT = (LIMITEnum)LIMIT_i;
    int _limit = LIMIT_i;
    
    int EOTF_i;
    m_EOTF->getValueAtTime(p_Args.time, EOTF_i);
    EOTFEnum EOTF = (EOTFEnum)EOTF_i;
    int _eotf = EOTF_i;
    
    int SURROUND_i;
    m_SURROUND->getValueAtTime(p_Args.time, SURROUND_i);
    SURROUNDEnum SURROUND = (SURROUNDEnum)SURROUND_i;
    int _surround = SURROUND_i;
    
    int _switch[3];
    bool stretch = m_STRETCH->getValueAtTime(p_Args.time);
	_switch[0] = stretch ? 1 : 0;
	bool d60sim = m_D60SIM->getValueAtTime(p_Args.time);
	_switch[1] = d60sim ? 1 : 0;
	bool legalrange = m_LEGALRANGE->getValueAtTime(p_Args.time);
	_switch[2] = legalrange ? 1 : 0;
    
	if(p_ParamName == "button1")
    {
    
	double inputA, inputB;
    m_Input->getValueAtTime(p_Args.time, inputA, inputB);
	
	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	int cube = (int)m_Cube->getValueAtTime(p_Args.time);
	int total = cube * cube * cube;
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".cube to " + PATH + " ?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + NAME + ".cube").c_str(), "w");
	if (pFile != NULL) {
	fprintf (pFile, "# Iridas 3D LUT export\n" \
	"\n");
	fprintf (pFile, "LUT_3D_SIZE %d\n" \
	"DOMAIN_MIN %.10f %.10f %.10f\n" \
	"DOMAIN_MAX %.10f %.10f %.10f\n" \
	"\n", cube, inputA, inputA, inputA, inputB, inputB, inputB);
	for( int i = 0; i < total; ++i ){
    float R = fmod(i, cube) / (cube - 1) * (inputB - inputA) + inputA;
    float G = fmod(floor(i / cube), cube) / (cube - 1) * (inputB - inputA) + inputA;
    float B = fmod(floor(i / (cube * cube)), cube) / (cube - 1) * (inputB - inputA) + inputA;
    float3 aces = make_float3(R, G, B);
    if(_direction == 0)
	{
	switch (_idt)
	{
	case 0:
	{
	}
	break;
	case 1:
	{
	aces = ACEScc_to_ACES(aces);
	}
	break;
	case 2:
	{
	aces = ACEScct_to_ACES(aces);
	}
	break;
	case 3:
	{
	aces = IDT_Alexa_v3_logC_EI800(aces);
	}
	break;
	case 4:
	{
	aces = IDT_Alexa_v3_raw_EI800_CCT65(aces);
	}
	break;
	case 5:
	{
	aces = ADX10_to_ACES(aces);
	}
	break;
	case 6:
	{
	aces = ADX16_to_ACES(aces);
	}
	break;
	case 7:
	{
	aces = IDT_Panasonic_V35(aces);
	}
	break;
	case 8:
	{
	aces = IDT_REDWideGamutRGB_Log3G10(aces);
	}
	break;
	case 9:
	{
	aces = IDT_Canon_C100_A_D55(aces);
	}
	break;
	case 10:
	{
	aces = IDT_Canon_C100_A_Tng(aces);
	}
	break;
	case 11:
	{
	aces = IDT_Canon_C100mk2_A_D55(aces);
	}
	break;
	case 12:
	{
	aces = IDT_Canon_C100mk2_A_Tng(aces);
	}
	break;
	case 13:
	{
	aces = IDT_Canon_C300_A_D55(aces);
	}
	break;
	case 14:
	{
	aces = IDT_Canon_C300_A_Tng(aces);
	}
	break;
	case 15:
	{
	aces = IDT_Canon_C500_A_D55(aces);
	}
	break;
	case 16:
	{
	aces = IDT_Canon_C500_A_Tng(aces);
	}
	break;
	case 17:
	{
	aces = IDT_Canon_C500_B_D55(aces);
	}
	break;
	case 18:
	{
	aces = IDT_Canon_C500_B_Tng(aces);
	}
	break;
	case 19:
	{
	aces = IDT_Canon_C500_CinemaGamut_A_D55(aces);
	}
	break;
	case 20:
	{
	aces = IDT_Canon_C500_CinemaGamut_A_Tng(aces);
	}
	break;
	case 21:
	{
	aces = IDT_Canon_C500_DCI_P3_A_D55(aces);
	}
	break;
	case 22:
	{
	aces = IDT_Canon_C500_DCI_P3_A_Tng(aces);
	}
	break;
	case 23:
	{
	aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_D55(aces);
	}
	break;
	case 24:
	{
	aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng(aces);
	}
	break;
	case 25:
	{
	aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55(aces);
	}
	break;
	case 26:
	{
	aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng(aces);
	}
	break;
	case 27:
	{
	aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55(aces);
	}
	break;
	case 28:
	{
	aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng(aces);
	}
	break;
	case 29:
	{
	aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55(aces);
	}
	break;
	case 30:
	{
	aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng(aces);
	}
	break;
	case 31:
	{
	aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55(aces);
	}
	break;
	case 32:
	{
	aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng(aces);
	}
	break;
	case 33:
	{
	aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55(aces);
	}
	break;
	case 34:
	{
	aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng(aces);
	}
	break;
	case 35:
	{
	aces = IDT_Sony_SLog1_SGamut_10(aces);
	}
	break;
	case 36:
	{
	aces = IDT_Sony_SLog1_SGamut_12(aces);
	}
	break;
	case 37:
	{
	aces = IDT_Sony_SLog2_SGamut_Daylight_10(aces);
	}
	break;
	case 38:
	{
	aces = IDT_Sony_SLog2_SGamut_Daylight_12(aces);
	}
	break;
	case 39:
	{
	aces = IDT_Sony_SLog2_SGamut_Tungsten_10(aces);
	}
	break;
	case 40:
	{
	aces = IDT_Sony_SLog2_SGamut_Tungsten_12(aces);
	}
	break;
	case 41:
	{
	aces = IDT_Sony_SLog3_SGamut3(aces);
	}
	break;
	case 42:
	{
	aces = IDT_Sony_SLog3_SGamut3Cine(aces);
	}
	}
	if(_exposure != 0.0f)
	{
	aces.x *= powf(2.0f, _exposure);
	aces.y *= powf(2.0f, _exposure);
	aces.z *= powf(2.0f, _exposure);
	}
	switch (_acesin)
	{
	case 0:
	{
	}
	break;
	case 1:
	{
	aces = ACES_to_ACEScc(aces);
	}
	break;
	case 2:
	{
	aces = ACES_to_ACEScct(aces);
	}
	break;
	case 3:
	{
	aces = ACES_to_ACEScg(aces);
	}
	break;
	case 4:
	{
	aces = ACES_to_ACESproxy10(aces);
	}
	break;
	case 5:
	{
	aces = ACES_to_ACESproxy12(aces);
	}
	}
	switch (_lmt)
	{
	case 0:
	{
	}
	break;
	case 1:
	{
	if(_lmtscale[0] != 1.0f)
	aces = scale_C(aces, _lmtscale[0]);
	if(!(_lmtscale[1] == 1.0f && _lmtscale[2] == 0.0f && _lmtscale[3] == 1.0f))
	{
	float3 SLOPE = {_lmtscale[1], _lmtscale[1], _lmtscale[1]};
	float3 OFFSET = {_lmtscale[2], _lmtscale[2], _lmtscale[2]};
	float3 POWER = {_lmtscale[3], _lmtscale[3], _lmtscale[3]};
	aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER);
	}
	if(_lmtscale[4] != 1.0f)
	aces = gamma_adjust_linear(aces, _lmtscale[4], _lmtscale[5]);
	if(_lmtscale[8] != 0.0f)
	aces = rotate_H_in_H(aces, _lmtscale[6], _lmtscale[7], _lmtscale[8]);
	if(_lmtscale[11] != 0.0f)
	aces = rotate_H_in_H(aces, _lmtscale[9], _lmtscale[10], _lmtscale[11]);
	if(_lmtscale[14] != 0.0f)
	aces = rotate_H_in_H(aces, _lmtscale[12], _lmtscale[13], _lmtscale[14]);
	if(_lmtscale[17] != 1.0f)
	aces = scale_C_at_H(aces, _lmtscale[15], _lmtscale[16], _lmtscale[17]);
	if(_lmtscale[20] != 0.0f)
	aces = rotate_H_in_H(aces, _lmtscale[18], _lmtscale[19], _lmtscale[20]);
	if(_lmtscale[23] != 1.0f)
	aces = scale_C_at_H(aces, _lmtscale[21], _lmtscale[22], _lmtscale[23]);
	}
	break;
	case 2:
	{
	aces = LMT_Analytic_4(aces);
	}
	break;
	case 3:
	{
	aces = LMT_Analytic_3(aces);
	}
	break;
	case 4:
	{
	aces = LMT_BlueLightArtifactFix(aces);
	}
	}
	switch (_acesout)
	{
	case 0:
	{
	}
	break;
	case 1:
	{
	aces = ACEScc_to_ACES(aces);
	}
	break;
	case 2:
	{
	aces = ACEScct_to_ACES(aces);
	}
	break;
	case 3:
	{
	aces = ACEScg_to_ACES(aces);
	}
	break;
	case 4:
	{
	aces = ACESproxy10_to_ACES(aces);
	}
	break;
	case 5:
	{
	aces = ACESproxy12_to_ACES(aces);
	}
	}
	if(_rrt == 1)
	{
	aces = _RRT(aces);
	}
	switch (_odt)
	{
	case 0:
	{
	}
	break;
	case 1:
	{
	float Y_MIN = _lum[0] * 0.0001f;
	float Y_MID = _lum[1];
	float Y_MAX = _lum[2];
	Chromaticities DISPLAY_PRI = _display == 0 ? REC2020_PRI : _display == 1 ? P3D60_PRI : _display == 2 ? P3D65_PRI : _display == 3 ? P3DCI_PRI : REC709_PRI;
	Chromaticities LIMITING_PRI = _limit == 0 ? REC2020_PRI : _limit == 1 ? P3D60_PRI : _limit == 2 ? P3D65_PRI : _limit == 3 ? P3DCI_PRI : REC709_PRI;
	int EOTF = _eotf;
	int SURROUND = _surround;		   
	bool STRETCH_BLACK = _switch[0] == 1;
	bool D60_SIM = _switch[1] == 1;
	bool LEGAL_RANGE = _switch[2] == 1;
	aces = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
	}
	break;
	case 2:
	{
	aces = ACES_to_ACEScc(aces);
	}
	break;
	case 3:
	{
	aces = ACES_to_ACEScct(aces);
	}
	break;
	case 4:
	{
	aces = ODT_Rec709_100nits_dim(aces);
	}
	break;
	case 5:
	{
	aces = ODT_Rec709_D60sim_100nits_dim(aces);
	}
	break;
	case 6:
	{
	aces = ODT_sRGB_100nits_dim(aces);
	}
	break;
	case 7:
	{
	aces = ODT_sRGB_D60sim_100nits_dim(aces);
	}
	break;
	case 8:
	{
	aces = ODT_Rec2020_100nits_dim(aces);
	}
	break;
	case 9:
	{
	aces = ODT_Rec2020_Rec709limited_100nits_dim(aces);
	}
	break;
	case 10:
	{
	aces = ODT_Rec2020_P3D65limited_100nits_dim(aces);
	}
	break;
	case 11:
	{
	aces = ODT_Rec2020_ST2084_1000nits(aces);
	}
	break;
	case 12:
	{
	aces = ODT_P3DCI_48nits(aces);
	}
	break;
	case 13:
	{
	aces = ODT_P3DCI_D60sim_48nits(aces);
	}
	break;
	case 14:
	{
	aces = ODT_P3DCI_D65sim_48nits(aces);
	}
	break;
	case 15:
	{
	aces = ODT_P3D60_48nits(aces);
	}
	break;
	case 16:
	{
	aces = ODT_P3D65_48nits(aces);
	}
	break;
	case 17:
	{
	aces = ODT_P3D65_D60sim_48nits(aces);
	}
	break;
	case 18:
	{
	aces = ODT_P3D65_Rec709limited_48nits(aces);
	}
	break;
	case 19:
	{
	aces = ODT_DCDM(aces);
	}
	break;
	case 20:
	{
	aces = ODT_DCDM_P3D60limited(aces);
	}
	break;
	case 21:
	{
	aces = ODT_DCDM_P3D65limited(aces);
	}
	break;
	case 22:
	{
	aces = ODT_RGBmonitor_100nits_dim(aces);
	}
	break;
	case 23:
	{
	aces = ODT_RGBmonitor_D60sim_100nits_dim(aces);
	}
	break;
	case 24:
	{
	aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);
	}
	break;
	case 25:
	{
	aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);
	}
	break;
	case 26:
	{
	aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);
	}
	break;
	case 27:
	{
	aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);
	}
	break;
	case 28:
	{
	aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);
	}
	break;
	case 29:
	{
	aces = RRTODT_Rec709_100nits_10nits_BT1886(aces);
	}
	break;
	case 30:
	{
	aces = RRTODT_Rec709_100nits_10nits_sRGB(aces);
	}
	}
	} else {
	switch (_invodt)
	{
	case 0:
	{
	}
	break;
	case 1:
	{
	float Y_MIN = _lum[0] * 0.0001f;
	float Y_MID = _lum[1];
	float Y_MAX = _lum[2];
	Chromaticities DISPLAY_PRI = _display == 0 ? REC2020_PRI : _display == 1 ? P3D60_PRI : _display == 2 ? P3D65_PRI : _display == 3 ? P3DCI_PRI : REC709_PRI;
	Chromaticities LIMITING_PRI = _limit == 0 ? REC2020_PRI : _limit == 1 ? P3D60_PRI : _limit == 2 ? P3D65_PRI : _limit == 3 ? P3DCI_PRI : REC709_PRI;
	int EOTF = _eotf;		
	int SURROUND = _surround;		   
	bool STRETCH_BLACK = _switch[0] == 1;
	bool D60_SIM = _switch[1] == 1;
	bool LEGAL_RANGE = _switch[2] == 1;
	aces = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
	}
	break;
	case 2:
	{
	aces = InvODT_Rec709_100nits_dim(aces);
	}
	break;
	case 3:
	{
	aces = InvODT_Rec709_D60sim_100nits_dim(aces);
	}
	break;
	case 4:
	{
	aces = InvODT_sRGB_100nits_dim(aces);
	}
	break;
	case 5:
	{
	aces = InvODT_sRGB_D60sim_100nits_dim(aces);
	}
	break;
	case 6:
	{
	aces = InvODT_Rec2020_100nits_dim(aces);
	}
	break;
	case 7:
	{
	aces = InvODT_Rec2020_ST2084_1000nits(aces);
	}
	break;
	case 8:
	{
	aces = InvODT_P3DCI_48nits(aces);
	}
	break;
	case 9:
	{
	aces = InvODT_P3DCI_D60sim_48nits(aces);
	}
	break;
	case 10:
	{
	aces = InvODT_P3DCI_D65sim_48nits(aces);
	}
	break;
	case 11:
	{
	aces = InvODT_P3D60_48nits(aces);
	}
	break;
	case 12:
	{
	aces = InvODT_P3D65_48nits(aces);
	}
	break;
	case 13:
	{
	aces = InvODT_P3D65_D60sim_48nits(aces);
	}
	break;
	case 14:
	{
	aces = InvODT_DCDM(aces);
	}
	break;
	case 15:
	{
	aces = InvODT_DCDM_P3D65limited(aces);
	}
	break;
	case 16:
	{
	aces = InvODT_RGBmonitor_100nits_dim(aces);
	}
	break;
	case 17:
	{
	aces = InvODT_RGBmonitor_D60sim_100nits_dim(aces);
	}
	break;
	case 18:
	{
	aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);
	}
	break;
	case 19:
	{
	aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);
	}
	break;
	case 20:
	{
	aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);
	}
	break;
	case 21:
	{
	aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);
	}
	break;
	case 22:
	{
	aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);
	}
	break;
	case 23:
	{
	aces = InvRRTODT_Rec709_100nits_10nits_BT1886(aces);
	}
	break;
	case 24:
	{
	aces = InvRRTODT_Rec709_100nits_10nits_sRGB(aces);
	}
	}
	if(_invrrt == 1)
	{
	aces = _InvRRT(aces);
	}
	}
    fprintf (pFile, "%.10f %.10f %.10f\n", aces.x, aces.y, aces.z);
    }
    fclose (pFile);
	} else {
    sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".cube to " + PATH + ". Check Permissions."));
	}	
	}
	}
        
}

void ACESPlugin::setupAndProcess(ACES& p_ACES, const OFX::RenderArguments& p_Args)
{
    // Get the dst image
    std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }
	
	int Direction_i;
    m_Direction->getValueAtTime(p_Args.time, Direction_i);
    InputEnum Direction = (InputEnum)Direction_i;
    int _direction = Direction_i;
	
	int IDT_i;
    m_IDT->getValueAtTime(p_Args.time, IDT_i);
    IDTEnum IDT = (IDTEnum)IDT_i;
    int _idt = IDT_i;
    
    int ACESIN_i;
    m_ACESIN->getValueAtTime(p_Args.time, ACESIN_i);
    ACESINEnum ACESIN = (ACESINEnum)ACESIN_i;
    int _acesin = ACESIN_i;
    
    int LMT_i;
    m_LMT->getValueAtTime(p_Args.time, LMT_i);
    LMTEnum LMT = (LMTEnum)LMT_i;
    int _lmt = LMT_i;
    
    int ACESOUT_i;
    m_ACESOUT->getValueAtTime(p_Args.time, ACESOUT_i);
    ACESOUTEnum ACESOUT = (ACESOUTEnum)ACESOUT_i;
    int _acesout = ACESOUT_i;
    
    int RRT_i;
    m_RRT->getValueAtTime(p_Args.time, RRT_i);
    RRTEnum RRT = (RRTEnum)RRT_i;
    int _rrt = RRT_i;
    
    int InvRRT_i;
    m_InvRRT->getValueAtTime(p_Args.time, InvRRT_i);
    InvRRTEnum InvRRT = (InvRRTEnum)InvRRT_i;
    int _invrrt = InvRRT_i;
    
    int ODT_i;
    m_ODT->getValueAtTime(p_Args.time, ODT_i);
    ODTEnum ODT = (ODTEnum)ODT_i;
    int _odt = ODT_i;
    
    int InvODT_i;
    m_InvODT->getValueAtTime(p_Args.time, InvODT_i);
    InvODTEnum InvODT = (InvODTEnum)InvODT_i;
    int _invodt = InvODT_i;
    
    float _exposure = m_Exposure->getValueAtTime(p_Args.time);
    float _lmtscale[24];
    _lmtscale[0] = m_ScaleC->getValueAtTime(p_Args.time);
    _lmtscale[1] = m_Slope->getValueAtTime(p_Args.time);
    _lmtscale[2] = m_Offset->getValueAtTime(p_Args.time);
    _lmtscale[3] = m_Power->getValueAtTime(p_Args.time);
    _lmtscale[4] = m_Gamma->getValueAtTime(p_Args.time);
    _lmtscale[5] = m_Pivot->getValueAtTime(p_Args.time);
    
    _lmtscale[6] = m_RotateH1->getValueAtTime(p_Args.time);
    _lmtscale[7] = m_Range1->getValueAtTime(p_Args.time);
    _lmtscale[8] = m_Shift1->getValueAtTime(p_Args.time);
    
    _lmtscale[9] = m_RotateH2->getValueAtTime(p_Args.time);
    _lmtscale[10] = m_Range2->getValueAtTime(p_Args.time);
    _lmtscale[11] = m_Shift2->getValueAtTime(p_Args.time);
    
    _lmtscale[12] = m_RotateH3->getValueAtTime(p_Args.time);
    _lmtscale[13] = m_Range3->getValueAtTime(p_Args.time);
    _lmtscale[14] = m_Shift3->getValueAtTime(p_Args.time);
    
    _lmtscale[15] = m_HueCH1->getValueAtTime(p_Args.time);
    _lmtscale[16] = m_RangeCH1->getValueAtTime(p_Args.time);
    _lmtscale[17] = m_ScaleCH1->getValueAtTime(p_Args.time);
    
    _lmtscale[18] = m_RotateH4->getValueAtTime(p_Args.time);
    _lmtscale[19] = m_Range4->getValueAtTime(p_Args.time);
    _lmtscale[20] = m_Shift4->getValueAtTime(p_Args.time);
    
    _lmtscale[21] = m_HueCH2->getValueAtTime(p_Args.time);
    _lmtscale[22] = m_RangeCH2->getValueAtTime(p_Args.time);
    _lmtscale[23] = m_ScaleCH2->getValueAtTime(p_Args.time);
    
    float _lum[3];
    _lum[0] = m_YMIN->getValueAtTime(p_Args.time);
    _lum[1] = m_YMID->getValueAtTime(p_Args.time);
    _lum[2] = m_YMAX->getValueAtTime(p_Args.time);
    
    int DISPLAY_i;
    m_DISPLAY->getValueAtTime(p_Args.time, DISPLAY_i);
    DISPLAYEnum DISPLAY = (DISPLAYEnum)DISPLAY_i;
    int _display = DISPLAY_i;
    
    int LIMIT_i;
    m_LIMIT->getValueAtTime(p_Args.time, LIMIT_i);
    LIMITEnum LIMIT = (LIMITEnum)LIMIT_i;
    int _limit = LIMIT_i;
    
    int EOTF_i;
    m_EOTF->getValueAtTime(p_Args.time, EOTF_i);
    EOTFEnum EOTF = (EOTFEnum)EOTF_i;
    int _eotf = EOTF_i;
    
    int SURROUND_i;
    m_SURROUND->getValueAtTime(p_Args.time, SURROUND_i);
    SURROUNDEnum SURROUND = (SURROUNDEnum)SURROUND_i;
    int _surround = SURROUND_i;
    
    int _switch[3];
    bool stretch = m_STRETCH->getValueAtTime(p_Args.time);
	_switch[0] = stretch ? 1 : 0;
	bool d60sim = m_D60SIM->getValueAtTime(p_Args.time);
	_switch[1] = d60sim ? 1 : 0;
	bool legalrange = m_LEGALRANGE->getValueAtTime(p_Args.time);
	_switch[2] = legalrange ? 1 : 0;
	
    // Set the images
    p_ACES.setDstImg(dst.get());
    p_ACES.setSrcImg(src.get());

    // Set the render window
    p_ACES.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_ACES.setScales(_direction, _idt, _acesin, _lmt, _acesout, _rrt, _invrrt, _odt, 
    _invodt, _exposure, _lmtscale, _lum, _display, _limit, _eotf, _surround, _switch);

    // Call the base class process member, this will call the derived templated process code
    p_ACES.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ACESPluginFactory::ACESPluginFactory()
    : OFX::PluginFactoryHelper<ACESPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ACESPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);
    
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setDoubleType(eDoubleTypeScale);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void ACESPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);
    
    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");
    
    {
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamInput);
	param->setLabel(kParamInputLabel);
	param->setHint(kParamInputHint);
	assert(param->getNOptions() == (int)eInputStandard);
	param->appendOption(kParamInputOptionStandard, kParamInputOptionStandardHint);
	assert(param->getNOptions() == (int)eInputInverse);
	param->appendOption(kParamInputOptionInverse, kParamInputOptionInverseHint);
	param->setDefault( (int)eInputStandard );
	param->setAnimates(false);
    page->addChild(*param);
	}
    
    {
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamIDT);
	param->setLabel(kParamIDTLabel);
	param->setHint(kParamIDTHint);
	assert(param->getNOptions() == (int)eIDTBypass);
	param->appendOption(kParamIDTOptionBypass, kParamIDTOptionBypassHint);
	assert(param->getNOptions() == (int)eIDTACEScc);
	param->appendOption(kParamIDTOptionACEScc, kParamIDTOptionACESccHint);
	assert(param->getNOptions() == (int)eIDTACEScct);
	param->appendOption(kParamIDTOptionACEScct, kParamIDTOptionACEScctHint);
	assert(param->getNOptions() == (int)eIDTAlexaLogC800);
	param->appendOption(kParamIDTOptionAlexaLogC800, kParamIDTOptionAlexaLogC800Hint);
	assert(param->getNOptions() == (int)eIDTAlexaRaw800);
	param->appendOption(kParamIDTOptionAlexaRaw800, kParamIDTOptionAlexaRaw800Hint);
	assert(param->getNOptions() == (int)eIDTADX10);
	param->appendOption(kParamIDTOptionADX10, kParamIDTOptionADX10Hint);
	assert(param->getNOptions() == (int)eIDTADX16);
	param->appendOption(kParamIDTOptionADX16, kParamIDTOptionADX16Hint);
	assert(param->getNOptions() == (int)eIDTPanasonicV35);
	param->appendOption(kParamIDTOptionPanasonicV35, kParamIDTOptionPanasonicV35Hint);
	assert(param->getNOptions() == (int)eIDTREDWideGamutRGBLog3G10);
	param->appendOption(kParamIDTOptionREDWideGamutRGBLog3G10, kParamIDTOptionREDWideGamutRGBLog3G10Hint);
	assert(param->getNOptions() == (int)eIDTCanonC100AD55);
	param->appendOption(kParamIDTOptionCanonC100AD55, kParamIDTOptionCanonC100AD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC100ATNG);
	param->appendOption(kParamIDTOptionCanonC100ATNG, kParamIDTOptionCanonC100ATNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC100mk2AD55);
	param->appendOption(kParamIDTOptionCanonC100mk2AD55, kParamIDTOptionCanonC100mk2AD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC100mk2ATNG);
	param->appendOption(kParamIDTOptionCanonC100mk2ATNG, kParamIDTOptionCanonC100mk2ATNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC300AD55);
	param->appendOption(kParamIDTOptionCanonC300AD55, kParamIDTOptionCanonC300AD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC300ATNG);
	param->appendOption(kParamIDTOptionCanonC300ATNG, kParamIDTOptionCanonC300ATNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC500AD55);
	param->appendOption(kParamIDTOptionCanonC500AD55, kParamIDTOptionCanonC500AD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC500ATNG);
	param->appendOption(kParamIDTOptionCanonC500ATNG, kParamIDTOptionCanonC500ATNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC500BD55);
	param->appendOption(kParamIDTOptionCanonC500BD55, kParamIDTOptionCanonC500BD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC500BTNG);
	param->appendOption(kParamIDTOptionCanonC500BTNG, kParamIDTOptionCanonC500BTNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC500CinemaGamutAD55);
	param->appendOption(kParamIDTOptionCanonC500CinemaGamutAD55, kParamIDTOptionCanonC500CinemaGamutAD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC500CinemaGamutATNG);
	param->appendOption(kParamIDTOptionCanonC500CinemaGamutATNG, kParamIDTOptionCanonC500CinemaGamutATNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC500DCIP3AD55);
	param->appendOption(kParamIDTOptionCanonC500DCIP3AD55, kParamIDTOptionCanonC500DCIP3AD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC500DCIP3ATNG);
	param->appendOption(kParamIDTOptionCanonC500DCIP3ATNG, kParamIDTOptionCanonC500DCIP3ATNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLogBT2020DD55);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLogBT2020DD55, kParamIDTOptionCanonC300mk2CanonLogBT2020DD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLogBT2020DTNG);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLogBT2020DTNG, kParamIDTOptionCanonC300mk2CanonLogBT2020DTNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLogCinemaGamutCD55);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCD55, kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLogCinemaGamutCTNG);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCTNG, kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCTNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLog2BT2020BD55);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLog2BT2020BD55, kParamIDTOptionCanonC300mk2CanonLog2BT2020BD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLog2BT2020BTNG);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLog2BT2020BTNG, kParamIDTOptionCanonC300mk2CanonLog2BT2020BTNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLog2CinemaGamutAD55);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutAD55, kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutAD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLog2CinemaGamutATNG);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutATNG, kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutATNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLog3BT2020FD55);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLog3BT2020FD55, kParamIDTOptionCanonC300mk2CanonLog3BT2020FD55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLog3BT2020FTNG);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLog3BT2020FTNG, kParamIDTOptionCanonC300mk2CanonLog3BT2020FTNGHint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLog3CinemaGamutED55);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutED55, kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutED55Hint);
	assert(param->getNOptions() == (int)eIDTCanonC300mk2CanonLog3CinemaGamutETNG);
	param->appendOption(kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutETNG, kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutETNGHint);
	assert(param->getNOptions() == (int)eIDTSonySLog1SGamut10);
	param->appendOption(kParamIDTOptionSonySLog1SGamut10, kParamIDTOptionSonySLog1SGamut10Hint);
	assert(param->getNOptions() == (int)eIDTSonySLog1SGamut12);
	param->appendOption(kParamIDTOptionSonySLog1SGamut12, kParamIDTOptionSonySLog1SGamut12Hint);
	assert(param->getNOptions() == (int)eIDTSonySLog2SGamutDaylight10);
	param->appendOption(kParamIDTOptionSonySLog2SGamutDaylight10, kParamIDTOptionSonySLog2SGamutDaylight10Hint);
	assert(param->getNOptions() == (int)eIDTSonySLog2SGamutDaylight12);
	param->appendOption(kParamIDTOptionSonySLog2SGamutDaylight12, kParamIDTOptionSonySLog2SGamutDaylight12Hint);
	assert(param->getNOptions() == (int)eIDTSonySLog2SGamutTungsten10);
	param->appendOption(kParamIDTOptionSonySLog2SGamutTungsten10, kParamIDTOptionSonySLog2SGamutTungsten10Hint);
	assert(param->getNOptions() == (int)eIDTSonySLog2SGamutTungsten12);
	param->appendOption(kParamIDTOptionSonySLog2SGamutTungsten12, kParamIDTOptionSonySLog2SGamutTungsten12Hint);
	assert(param->getNOptions() == (int)eIDTSonySLog3SGamut3);
	param->appendOption(kParamIDTOptionSonySLog3SGamut3, kParamIDTOptionSonySLog3SGamut3Hint);
	assert(param->getNOptions() == (int)eIDTSonySLog3SGamut3Cine);
	param->appendOption(kParamIDTOptionSonySLog3SGamut3Cine, kParamIDTOptionSonySLog3SGamut3CineHint);
	param->setDefault( (int)eIDTBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(false);
    page->addChild(*param);
	}
	
	DoubleParamDescriptor* param = defineScaleParam(p_Desc, "Exposure", "exposure", "scale in stops", 0);
	param->setDefault(0.0f);
	param->setRange(-10.0f, 10.0f);
	param->setIncrement(0.001f);
	param->setDisplayRange(-10.0f, 10.0f);
	param->setIsSecretAndDisabled(false);
	page->addChild(*param);
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamACESIN);
	param->setLabel(kParamACESINLabel);
	param->setHint(kParamACESINHint);
	assert(param->getNOptions() == (int)eACESINBypass);
	param->appendOption(kParamACESINOptionBypass, kParamACESINOptionBypassHint);
	assert(param->getNOptions() == (int)eACESINACEScc);
	param->appendOption(kParamACESINOptionACEScc, kParamACESINOptionACESccHint);
	assert(param->getNOptions() == (int)eACESINACEScct);
	param->appendOption(kParamACESINOptionACEScct, kParamACESINOptionACEScctHint);
	assert(param->getNOptions() == (int)eACESINACEScg);
	param->appendOption(kParamACESINOptionACEScg, kParamACESINOptionACEScgHint);
	assert(param->getNOptions() == (int)eACESINACESproxy10);
	param->appendOption(kParamACESINOptionACESproxy10, kParamACESINOptionACESproxy10Hint);
	assert(param->getNOptions() == (int)eACESINACESproxy12);
	param->appendOption(kParamACESINOptionACESproxy12, kParamACESINOptionACESproxy12Hint);
	param->setDefault( (int)eACESINBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(false);
    page->addChild(*param);
	}
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamLMT);
	param->setLabel(kParamLMTLabel);
	param->setHint(kParamLMTHint);
	assert(param->getNOptions() == (int)eLMTBypass);
	param->appendOption(kParamLMTOptionBypass, kParamLMTOptionBypassHint);
	assert(param->getNOptions() == (int)eLMTCustom);
	param->appendOption(kParamLMTOptionCustom, kParamLMTOptionCustomHint);
	assert(param->getNOptions() == (int)eLMTBleach);
	param->appendOption(kParamLMTOptionBleach, kParamLMTOptionBleachHint);
	assert(param->getNOptions() == (int)eLMTPFE);
	param->appendOption(kParamLMTOptionPFE, kParamLMTOptionPFEHint);
	assert(param->getNOptions() == (int)eLMTFix);
	param->appendOption(kParamLMTOptionFix, kParamLMTOptionFixHint);
	param->setDefault( (int)eLMTBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(false);
    page->addChild(*param);
	}
	
	param = defineScaleParam(p_Desc, "ScaleC", "color boost", "scale color", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
	
	param = defineScaleParam(p_Desc, "Slope", "slope", "slope", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Offset", "offset", "offset", 0);
    param->setDefault(0.0);
    param->setRange(-10.0, 10.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-2.0, 2.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
        
    param = defineScaleParam(p_Desc, "Power", "power", "power", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Gamma", "contrast", "gamma adjust linear", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Pivot", "contrast pivot", "pivot point", 0);
    param->setDefault(0.18);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "RotateH1", "hue rotation hue1", "rotate hue at hue in degrees", 0);
    param->setDefault(30.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Range1", "hue rotation range1", "hue range in degrees for rotating hue", 0);
    param->setDefault(60.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Shift1", "hue rotation1", "shift hue range in degrees", 0);
    param->setDefault(0.0);
    param->setRange(-360.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-360.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "RotateH2", "hue rotation hue2", "rotate hue at hue in degrees", 0);
    param->setDefault(210.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Range2", "hue rotation range2", "hue range in degrees for rotating hue", 0);
    param->setDefault(60.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Shift2", "hue rotation2", "shift hue range in degrees", 0);
    param->setDefault(0.0);
    param->setRange(-360.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-360.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "RotateH3", "hue rotation hue3", "rotate hue at hue in degrees", 0);
    param->setDefault(120.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Range3", "hue rotation range3", "hue range in degrees for rotating hue", 0);
    param->setDefault(60.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Shift3", "hue rotation3", "shift hue range in degrees", 0);
    param->setDefault(0.0);
    param->setRange(-360.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-360.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "HueCH1", "color scale hue1", "scale color at hue in degrees", 0);
    param->setDefault(30.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "RangeCH1", "color scale range1", "hue range in degrees for scaling color", 0);
    param->setDefault(60.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "ScaleCH1", "color scale1", "scale color at hue range", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "RotateH4", "hue rotation hue4", "rotate hue at hue in degrees", 0);
    param->setDefault(0.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Range4", "hue rotation range4", "hue range in degrees for rotating hue", 0);
    param->setDefault(60.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Shift4", "hue rotation4", "shift hue range in degrees", 0);
    param->setDefault(0.0);
    param->setRange(-360.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-360.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "HueCH2", "color scale hue2", "scale color at hue in degrees", 0);
    param->setDefault(210.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "RangeCH2", "color scale range2", "hue range in degrees for scaling color", 0);
    param->setDefault(60.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "ScaleCH2", "color scale2", "scale color at hue range", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    {
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamACESOUT);
	param->setLabel(kParamACESOUTLabel);
	param->setHint(kParamACESOUTHint);
	assert(param->getNOptions() == (int)eACESOUTBypass);
	param->appendOption(kParamACESOUTOptionBypass, kParamACESOUTOptionBypassHint);
	assert(param->getNOptions() == (int)eACESOUTACEScc);
	param->appendOption(kParamACESOUTOptionACEScc, kParamACESOUTOptionACESccHint);
	assert(param->getNOptions() == (int)eACESOUTACEScct);
	param->appendOption(kParamACESOUTOptionACEScct, kParamACESOUTOptionACEScctHint);
	assert(param->getNOptions() == (int)eACESOUTACEScg);
	param->appendOption(kParamACESOUTOptionACEScg, kParamACESOUTOptionACEScgHint);
	assert(param->getNOptions() == (int)eACESOUTACESproxy10);
	param->appendOption(kParamACESOUTOptionACESproxy10, kParamACESOUTOptionACESproxy10Hint);
	assert(param->getNOptions() == (int)eACESOUTACESproxy12);
	param->appendOption(kParamACESOUTOptionACESproxy12, kParamACESOUTOptionACESproxy12Hint);
	param->setDefault( (int)eACESOUTBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(false);
    page->addChild(*param);
	}
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamRRT);
	param->setLabel(kParamRRTLabel);
	param->setHint(kParamRRTHint);
	assert(param->getNOptions() == (int)eRRTBypass);
	param->appendOption(kParamRRTOptionBypass, kParamRRTOptionBypassHint);
	assert(param->getNOptions() == (int)eRRTEnabled);
	param->appendOption(kParamRRTOptionEnabled, kParamRRTOptionEnabledHint);
	param->setDefault( (int)eRRTBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(false);
    page->addChild(*param);
	}
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamODT);
	param->setLabel(kParamODTLabel);
	param->setHint(kParamODTHint);
	assert(param->getNOptions() == (int)eODTBypass);
	param->appendOption(kParamODTOptionBypass, kParamODTOptionBypassHint);
	assert(param->getNOptions() == (int)eODTCustom);
	param->appendOption(kParamODTOptionCustom, kParamODTOptionCustomHint);
	assert(param->getNOptions() == (int)eODTACEScc);
	param->appendOption(kParamODTOptionACEScc, kParamODTOptionACESccHint);
	assert(param->getNOptions() == (int)eODTACEScct);
	param->appendOption(kParamODTOptionACEScct, kParamODTOptionACEScctHint);
	assert(param->getNOptions() == (int)eODTRec709_100dim);
	param->appendOption(kParamODTOptionRec709_100dim, kParamODTOptionRec709_100dimHint);
	assert(param->getNOptions() == (int)eODTRec709_D60sim_100dim);
	param->appendOption(kParamODTOptionRec709_D60sim_100dim, kParamODTOptionRec709_D60sim_100dimHint);
	assert(param->getNOptions() == (int)eODTSRGB_100dim);
	param->appendOption(kParamODTOptionSRGB_100dim, kParamODTOptionSRGB_100dimHint);
	assert(param->getNOptions() == (int)eODTSRGB_D60sim_100dim);
	param->appendOption(kParamODTOptionSRGB_D60sim_100dim, kParamODTOptionSRGB_D60sim_100dimHint);
	assert(param->getNOptions() == (int)eODTRec2020_100dim);
	param->appendOption(kParamODTOptionRec2020_100dim, kParamODTOptionRec2020_100dimHint);
	assert(param->getNOptions() == (int)eODTRec2020_Rec709limited_100dim);
	param->appendOption(kParamODTOptionRec2020_Rec709limited_100dim, kParamODTOptionRec2020_Rec709limited_100dimHint);
	assert(param->getNOptions() == (int)eODTRec2020_P3D65limited_100dim);
	param->appendOption(kParamODTOptionRec2020_P3D65limited_100dim, kParamODTOptionRec2020_P3D65limited_100dimHint);
	assert(param->getNOptions() == (int)eODTRec2020_ST2084_1000);
	param->appendOption(kParamODTOptionRec2020_ST2084_1000, kParamODTOptionRec2020_ST2084_1000Hint);
	assert(param->getNOptions() == (int)eODTP3DCI_48);
	param->appendOption(kParamODTOptionP3DCI_48, kParamODTOptionP3DCI_48Hint);
	assert(param->getNOptions() == (int)eODTP3DCI_D60sim_48);
	param->appendOption(kParamODTOptionP3DCI_D60sim_48, kParamODTOptionP3DCI_D60sim_48Hint);
	assert(param->getNOptions() == (int)eODTP3DCI_D65sim_48);
	param->appendOption(kParamODTOptionP3DCI_D65sim_48, kParamODTOptionP3DCI_D65sim_48Hint);
	assert(param->getNOptions() == (int)eODTP3D60_48);
	param->appendOption(kParamODTOptionP3D60_48, kParamODTOptionP3D60_48Hint);
	assert(param->getNOptions() == (int)eODTP3D65_48);
	param->appendOption(kParamODTOptionP3D65_48, kParamODTOptionP3D65_48Hint);
	assert(param->getNOptions() == (int)eODTP3D65_D60sim_48);
	param->appendOption(kParamODTOptionP3D65_D60sim_48, kParamODTOptionP3D65_D60sim_48Hint);
	assert(param->getNOptions() == (int)eODTP3D65_Rec709limited_48);
	param->appendOption(kParamODTOptionP3D65_Rec709limited_48, kParamODTOptionP3D65_Rec709limited_48Hint);
	assert(param->getNOptions() == (int)eODTDCDM);
	param->appendOption(kParamODTOptionDCDM, kParamODTOptionDCDMHint);
	assert(param->getNOptions() == (int)eODTDCDM_P3D60limited);
	param->appendOption(kParamODTOptionDCDM_P3D60limited, kParamODTOptionDCDM_P3D60limitedHint);
	assert(param->getNOptions() == (int)eODTDCDM_P3D65limited);
	param->appendOption(kParamODTOptionDCDM_P3D65limited, kParamODTOptionDCDM_P3D65limitedHint);
	assert(param->getNOptions() == (int)eODTRGBmonitor_100dim);
	param->appendOption(kParamODTOptionRGBmonitor_100dim, kParamODTOptionRGBmonitor_100dimHint);
	assert(param->getNOptions() == (int)eODTRGBmonitor_D60sim_100dim);
	param->appendOption(kParamODTOptionRGBmonitor_D60sim_100dim, kParamODTOptionRGBmonitor_D60sim_100dimHint);
	assert(param->getNOptions() == (int)eODTRRTODT_P3D65_108nits_7_2nits_ST2084);
	param->appendOption(kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084, kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint);
	assert(param->getNOptions() == (int)eODTRRTODT_Rec2020_1000nits_15nits_HLG);
	param->appendOption(kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLG, kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint);
	assert(param->getNOptions() == (int)eODTRRTODT_Rec2020_1000nits_15nits_ST2084);
	param->appendOption(kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084, kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint);
	assert(param->getNOptions() == (int)eODTRRTODT_Rec2020_2000nits_15nits_ST2084);
	param->appendOption(kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084, kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint);
	assert(param->getNOptions() == (int)eODTRRTODT_Rec2020_4000nits_15nits_ST2084);
	param->appendOption(kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084, kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint);
	assert(param->getNOptions() == (int)eODTRRTODT_Rec709_100nits_10nits_BT1886);
	param->appendOption(kParamODTOptionRRTODT_Rec709_100nits_10nits_BT1886, kParamODTOptionRRTODT_Rec709_100nits_10nits_BT1886Hint);
	assert(param->getNOptions() == (int)eODTRRTODT_Rec709_100nits_10nits_sRGB);
	param->appendOption(kParamODTOptionRRTODT_Rec709_100nits_10nits_sRGB, kParamODTOptionRRTODT_Rec709_100nits_10nits_sRGBHint);
	param->setDefault( (int)eODTBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(false);
    page->addChild(*param);
	}
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamInvODT);
	param->setLabel(kParamInvODTLabel);
	param->setHint(kParamInvODTHint);
	assert(param->getNOptions() == (int)eInvODTBypass);
	param->appendOption(kParamInvODTOptionBypass, kParamInvODTOptionBypassHint);
	assert(param->getNOptions() == (int)eInvODTCustom);
	param->appendOption(kParamInvODTOptionCustom, kParamInvODTOptionCustomHint);
	assert(param->getNOptions() == (int)eInvODTRec709_100dim);
	param->appendOption(kParamInvODTOptionRec709_100dim, kParamInvODTOptionRec709_100dimHint);
	assert(param->getNOptions() == (int)eInvODTRec709_D60sim_100dim);
	param->appendOption(kParamInvODTOptionRec709_D60sim_100dim, kParamInvODTOptionRec709_D60sim_100dimHint);
	assert(param->getNOptions() == (int)eInvODTSRGB_100dim);
	param->appendOption(kParamInvODTOptionSRGB_100dim, kParamInvODTOptionSRGB_100dimHint);
	assert(param->getNOptions() == (int)eInvODTSRGB_D60sim_100dim);
	param->appendOption(kParamInvODTOptionSRGB_D60sim_100dim, kParamInvODTOptionSRGB_D60sim_100dimHint);
	assert(param->getNOptions() == (int)eInvODTRec2020_100dim);
	param->appendOption(kParamInvODTOptionRec2020_100dim, kParamInvODTOptionRec2020_100dimHint);
	assert(param->getNOptions() == (int)eInvODTRec2020_ST2084_1000);
	param->appendOption(kParamInvODTOptionRec2020_ST2084_1000, kParamInvODTOptionRec2020_ST2084_1000Hint);
	assert(param->getNOptions() == (int)eInvODTP3DCI_48);
	param->appendOption(kParamInvODTOptionP3DCI_48, kParamInvODTOptionP3DCI_48Hint);
	assert(param->getNOptions() == (int)eInvODTP3DCI_D60sim_48);
	param->appendOption(kParamInvODTOptionP3DCI_D60sim_48, kParamInvODTOptionP3DCI_D60sim_48Hint);
	assert(param->getNOptions() == (int)eInvODTP3DCI_D65sim_48);
	param->appendOption(kParamInvODTOptionP3DCI_D65sim_48, kParamInvODTOptionP3DCI_D65sim_48Hint);
	assert(param->getNOptions() == (int)eInvODTP3D60_48);
	param->appendOption(kParamInvODTOptionP3D60_48, kParamInvODTOptionP3D60_48Hint);
	assert(param->getNOptions() == (int)eInvODTP3D65_48);
	param->appendOption(kParamInvODTOptionP3D65_48, kParamInvODTOptionP3D65_48Hint);
	assert(param->getNOptions() == (int)eInvODTP3D65_D60sim_48);
	param->appendOption(kParamInvODTOptionP3D65_D60sim_48, kParamInvODTOptionP3D65_D60sim_48Hint);
	assert(param->getNOptions() == (int)eInvODTDCDM);
	param->appendOption(kParamInvODTOptionDCDM, kParamInvODTOptionDCDMHint);
	assert(param->getNOptions() == (int)eInvODTDCDM_P3D65limited);
	param->appendOption(kParamInvODTOptionDCDM_P3D65limited, kParamInvODTOptionDCDM_P3D65limitedHint);
	assert(param->getNOptions() == (int)eInvODTRGBmonitor_100dim);
	param->appendOption(kParamInvODTOptionRGBmonitor_100dim, kParamInvODTOptionRGBmonitor_100dimHint);
	assert(param->getNOptions() == (int)eInvODTRGBmonitor_D60sim_100dim);
	param->appendOption(kParamInvODTOptionRGBmonitor_D60sim_100dim, kParamInvODTOptionRGBmonitor_D60sim_100dimHint);
	assert(param->getNOptions() == (int)eInvODTRRTODT_P3D65_108nits_7_2nits_ST2084);
	param->appendOption(kParamInvODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084, kParamInvODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint);
	assert(param->getNOptions() == (int)eInvODTRRTODT_Rec2020_1000nits_15nits_HLG);
	param->appendOption(kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_HLG, kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint);
	assert(param->getNOptions() == (int)eInvODTRRTODT_Rec2020_1000nits_15nits_ST2084);
	param->appendOption(kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084, kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint);
	assert(param->getNOptions() == (int)eInvODTRRTODT_Rec2020_2000nits_15nits_ST2084);
	param->appendOption(kParamInvODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084, kParamInvODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint);
	assert(param->getNOptions() == (int)eInvODTRRTODT_Rec2020_4000nits_15nits_ST2084);
	param->appendOption(kParamInvODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084, kParamInvODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint);
	assert(param->getNOptions() == (int)eInvODTRRTODT_Rec709_100nits_10nits_BT1886);
	param->appendOption(kParamInvODTOptionRRTODT_Rec709_100nits_10nits_BT1886, kParamInvODTOptionRRTODT_Rec709_100nits_10nits_BT1886Hint);
	assert(param->getNOptions() == (int)eInvODTRRTODT_Rec709_100nits_10nits_sRGB);
	param->appendOption(kParamInvODTOptionRRTODT_Rec709_100nits_10nits_sRGB, kParamInvODTOptionRRTODT_Rec709_100nits_10nits_sRGBHint);
	param->setDefault( (int)eInvODTBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(true);
    page->addChild(*param);
	}
	
	param = p_Desc.defineDoubleParam("YMIN");
    param->setLabel("Black Luminance e-4");
    param->setHint("black luminance (cd/m^2) 1 = 0.0001");
    param->setDefault(1.0);
    param->setRange(1.0, 1000.0);
    param->setIncrement(1);
    param->setDisplayRange(1.0, 1000.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("YMID");
    param->setLabel("Mid-point Luminance");
    param->setHint("mid-point luminance (cd/m^2)");
    param->setDefault(15.0);
    param->setRange(0.0, 100.0);
    param->setIncrement(0.1);
    param->setDisplayRange(0.0, 100.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("YMAX");
    param->setLabel("Peak White Luminance");
    param->setHint("peak white luminance (cd/m^2)");
    param->setDefault(1000.0);
    param->setRange(48.0, 10000.0);
    param->setIncrement(1);
    param->setDisplayRange(48.0, 10000.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    {
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamDISPLAY);
	param->setLabel(kParamDISPLAYLabel);
	param->setHint(kParamDISPLAYHint);
	assert(param->getNOptions() == (int)eDISPLAYRec2020);
	param->appendOption(kParamDISPLAYOptionRec2020, kParamDISPLAYOptionRec2020Hint);
	assert(param->getNOptions() == (int)eDISPLAYP3D60);
	param->appendOption(kParamDISPLAYOptionP3D60, kParamDISPLAYOptionP3D60Hint);
	assert(param->getNOptions() == (int)eDISPLAYP3D65);
	param->appendOption(kParamDISPLAYOptionP3D65, kParamDISPLAYOptionP3D65Hint);
	assert(param->getNOptions() == (int)eDISPLAYP3DCI);
	param->appendOption(kParamDISPLAYOptionP3DCI, kParamDISPLAYOptionP3DCIHint);
	assert(param->getNOptions() == (int)eDISPLAYRec709);
	param->appendOption(kParamDISPLAYOptionRec709, kParamDISPLAYOptionRec709Hint);
	param->setDefault( (int)eDISPLAYRec2020 );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(true);
    page->addChild(*param);
	}
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamLIMIT);
	param->setLabel(kParamLIMITLabel);
	param->setHint(kParamLIMITHint);
	assert(param->getNOptions() == (int)eLIMITRec2020);
	param->appendOption(kParamLIMITOptionRec2020, kParamLIMITOptionRec2020Hint);
	assert(param->getNOptions() == (int)eLIMITP3D60);
	param->appendOption(kParamLIMITOptionP3D60, kParamLIMITOptionP3D60Hint);
	assert(param->getNOptions() == (int)eLIMITP3D65);
	param->appendOption(kParamLIMITOptionP3D65, kParamLIMITOptionP3D65Hint);
	assert(param->getNOptions() == (int)eLIMITP3DCI);
	param->appendOption(kParamLIMITOptionP3DCI, kParamLIMITOptionP3DCIHint);
	assert(param->getNOptions() == (int)eLIMITRec709);
	param->appendOption(kParamLIMITOptionRec709, kParamLIMITOptionRec709Hint);
	param->setDefault( (int)eLIMITRec2020 );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(true);
    page->addChild(*param);
	}
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamEOTF);
	param->setLabel(kParamEOTFLabel);
	param->setHint(kParamEOTFHint);
	assert(param->getNOptions() == (int)eEOTFST2084);
	param->appendOption(kParamEOTFOptionST2084, kParamEOTFOptionST2084Hint);
	assert(param->getNOptions() == (int)eEOTFBT1886);
	param->appendOption(kParamEOTFOptionBT1886, kParamEOTFOptionBT1886Hint);
	assert(param->getNOptions() == (int)eEOTFsRGB);
	param->appendOption(kParamEOTFOptionsRGB, kParamEOTFOptionsRGBHint);
	assert(param->getNOptions() == (int)eEOTFGAMMA26);
	param->appendOption(kParamEOTFOptionGAMMA26, kParamEOTFOptionGAMMA26Hint);
	assert(param->getNOptions() == (int)eEOTFLINEAR);
	param->appendOption(kParamEOTFOptionLINEAR, kParamEOTFOptionLINEARHint);
	assert(param->getNOptions() == (int)eEOTFHLG);
	param->appendOption(kParamEOTFOptionHLG, kParamEOTFOptionHLGHint);
	param->setDefault( (int)eEOTFST2084 );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(true);
    page->addChild(*param);
	}
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamSURROUND);
	param->setLabel(kParamSURROUNDLabel);
	param->setHint(kParamSURROUNDHint);
	assert(param->getNOptions() == (int)eSURROUNDDark);
	param->appendOption(kParamSURROUNDOptionDark, kParamSURROUNDOptionDarkHint);
	assert(param->getNOptions() == (int)eSURROUNDDim);
	param->appendOption(kParamSURROUNDOptionDim, kParamSURROUNDOptionDimHint);
	assert(param->getNOptions() == (int)eSURROUNDNormal);
	param->appendOption(kParamSURROUNDOptionNormal, kParamSURROUNDOptionNormalHint);
	param->setDefault( (int)eSURROUNDDim );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(true);
    page->addChild(*param);
	}
	
	BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("STRETCH");
    boolParam->setLabel("Stretch Black");
    boolParam->setHint("stretch black luminance to a PQ code value of 0");
    boolParam->setDefault(true);
    boolParam->setIsSecretAndDisabled(true);
    page->addChild(*boolParam);
    
    boolParam = p_Desc.defineBooleanParam("D60SIM");
    boolParam->setLabel("D60 SIM");
    boolParam->setHint("D60 Sim");
    boolParam->setDefault(false);
    boolParam->setIsSecretAndDisabled(true);
    page->addChild(*boolParam);
    
    boolParam = p_Desc.defineBooleanParam("LEGALRANGE");
    boolParam->setLabel("Legal Range");
    boolParam->setHint("Legal Range");
    boolParam->setDefault(false);
    boolParam->setIsSecretAndDisabled(true);
    page->addChild(*boolParam);
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamInvRRT);
	param->setLabel(kParamInvRRTLabel);
	param->setHint(kParamInvRRTHint);
	assert(param->getNOptions() == (int)eInvRRTBypass);
	param->appendOption(kParamInvRRTOptionBypass, kParamInvRRTOptionBypassHint);
	assert(param->getNOptions() == (int)eInvRRTEnabled);
	param->appendOption(kParamInvRRTOptionEnabled, kParamInvRRTOptionEnabledHint);
	param->setDefault( (int)eInvRRTBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(true);
    page->addChild(*param);
	}
	
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("info");
    param->setLabel("Info");
    page->addChild(*param);
    }
    {    
    GroupParamDescriptor* lutexport = p_Desc.defineGroupParam("LUT Export");
    lutexport->setOpen(false);
    lutexport->setHint("export LUT");
      if (page) {
            page->addChild(*lutexport);
            }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("name");
	param->setLabel("Label");
	param->setHint("overwrites if the same");
	param->setDefault("01");
	param->setParent(*lutexport);
	page->addChild(*param);
	}
	{
	StringParamDescriptor* param = p_Desc.defineStringParam("path");
	param->setLabel("Directory");
	param->setHint("make sure it's the absolute path");
	param->setStringType(eStringTypeFilePath);
	param->setDefault(kPluginScript);
	param->setFilePathExists(false);
	param->setParent(*lutexport);
	page->addChild(*param);
	}
    {
	Double2DParamDescriptor* param = p_Desc.defineDouble2DParam("range");
	param->setLabel("3D Input Range");
	param->setHint("set input range for 3D LUT");
	param->setDefault(0.0, 1.0);
	param->setParent(*lutexport);
	page->addChild(*param);
    }
    {
    IntParamDescriptor* Param = p_Desc.defineIntParam("cube");
	Param->setLabel("cube size");
    Param->setHint("3d lut cube size");
    Param->setDefault(33);
    Param->setRange(3, 129);
    Param->setDisplayRange(3, 129);
    Param->setParent(*lutexport);
    page->addChild(*Param);
    }
	{
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button1");
    param->setLabel("Export LUT");
    param->setHint("create 3D LUT");
    param->setParent(*lutexport);
    page->addChild(*param);
    }
	}        
    
}

ImageEffect* ACESPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum)
{
    return new ACESPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static ACESPluginFactory ACESPlugin;
    p_FactoryArray.push_back(&ACESPlugin);
}