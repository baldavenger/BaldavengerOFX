#include "ACESPlugin.h"

#include "ACES_LIB_CPU.h"

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
#include "exprtk.hpp"

#ifdef __APPLE__
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/ACES_DCTL/"
#define kPluginScript2 "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT\\ACES_DCTL\\"
#define kPluginScript2 "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT\\"
#else
#define kPluginScript "/home/resolve/LUT/ACES_DCTL/"
#define kPluginScript2 "/home/resolve/LUT/"
#endif

#define kPluginName "ACES 1.2"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"ACES 1.2"

#define kPluginIdentifier "BaldavengerOFX.ACES"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kParamInput "h_Input"
#define kParamInputLabel "Process"
#define kParamInputHint "ACES Standard or Inverse"
#define kParamInputOptionStandard "Standard"
#define kParamInputOptionStandardHint "Standard"
#define kParamInputOptionInverse "Inverse"
#define kParamInputOptionInverseHint "Inverse"

enum InputEnum
{
eInputStandard,
eInputInverse,
};

#define kParamCSCIN "h_CSCIN"
#define kParamCSCINLabel "ACES CSC in"
#define kParamCSCINHint "Convert to ACES from "
#define kParamCSCINOptionBypass "Bypass"
#define kParamCSCINOptionBypassHint "Bypass"
#define kParamCSCINOptionACEScc "ACEScc"
#define kParamCSCINOptionACESccHint "ACEScc to ACES"
#define kParamCSCINOptionACEScct "ACEScct"
#define kParamCSCINOptionACEScctHint "ACEScct to ACES"
#define kParamCSCINOptionACEScg "ACEScg"
#define kParamCSCINOptionACEScgHint "ACEScg to ACES"
#define kParamCSCINOptionACESproxy "ACESproxy"
#define kParamCSCINOptionACESproxyHint "ACESproxy to ACES"
#define kParamCSCINOptionADX "ADX"
#define kParamCSCINOptionADXHint "ADX to ACES"
#define kParamCSCINOptionICPCT "ICpCt"
#define kParamCSCINOptionICPCTHint "ICpCt to ACES"
#define kParamCSCINOptionLOGCAWG "LogC AlexaWG"
#define kParamCSCINOptionLOGCAWGHint "LogC AlexaWideGamut to ACES"
#define kParamCSCINOptionLOG3G10RWG "Log3G10 RedWG"
#define kParamCSCINOptionLOG3G10RWGHint "Log3G10 RedWideGamut to ACES"
#define kParamCSCINOptionSLOG3SG3 "Sony SLog3 SGamut3"
#define kParamCSCINOptionSLOG3SG3Hint "Slog3 SGamut3 to ACES"
#define kParamCSCINOptionSLOG3SG3C "Sony SLog3 SGamut3Cine"
#define kParamCSCINOptionSLOG3SG3CHint "Slog3 SGamut3Cine to ACES"

enum CSCINEnum
{
eCSCINBypass,
eCSCINACEScc,
eCSCINACEScct,
eCSCINACEScg,
eCSCINACESproxy,
eCSCINADX,
eCSCINICPCT,
eCSCINLOGCAWG,
eCSCINLOG3G10RWG,
eCSCINSLOG3SG3,
eCSCINSLOG3SG3C,
};

#define kParamIDT "k_IDT"
#define kParamIDTLabel "IDT"
#define kParamIDTHint "ACES IDT"
#define kParamIDTOptionBypass "Bypass"
#define kParamIDTOptionBypassHint "Bypass"
#define kParamIDTOptionAlexaRaw800 "Alexa Raw EI800"
#define kParamIDTOptionAlexaRaw800Hint "Alexa Raw EI800 to ACES"
#define kParamIDTOptionPanasonicV35 "Panasonic V35 VLog"
#define kParamIDTOptionPanasonicV35Hint "Panasonic V35 VLog"

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

#define kParamIDTOptionSonySLog1SGamut "Sony SLog1 SGamut"
#define kParamIDTOptionSonySLog1SGamutHint "Sony SLog1 SGamut"
#define kParamIDTOptionSonySLog2SGamutDaylight "Sony SLog2 SGamut Daylight"
#define kParamIDTOptionSonySLog2SGamutDaylightHint "Sony SLog2 SGamut Daylight"
#define kParamIDTOptionSonySLog2SGamutTungsten "Sony SLog2 SGamut Tungsten"
#define kParamIDTOptionSonySLog2SGamutTungstenHint "Sony SLog2 SGamut Tungsten"

#define kParamIDTOptionSonyVeniceSGamut3 "Sony Venice SGamut3"
#define kParamIDTOptionSonyVeniceSGamut3Hint "Sony Venice SGamut3"
#define kParamIDTOptionSonyVeniceSGamut3Cine "Sony Venice SGamut3Cine"
#define kParamIDTOptionSonyVeniceSGamut3CineHint "Sony Venice SGamut3Cine"

#define kParamIDTOptionRec709 "Rec709"
#define kParamIDTOptionRec709Hint "Rec709"
#define kParamIDTOptionSRGB "sRGB"
#define kParamIDTOptionSRGBHint "sRGB"

enum IDTEnum
{
eIDTBypass,
eIDTAlexaRaw800,
eIDTPanasonicV35,
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
eIDTSonySLog1SGamut,
eIDTSonySLog2SGamutDaylight,
eIDTSonySLog2SGamutTungsten,
eIDTSonyVeniceSGamut3,
eIDTSonyVeniceSGamut3Cine,
eIDTRec709,
eIDTSRGB,
};

#define kParamLMT "h_LMT"
#define kParamLMTLabel "LMT"
#define kParamLMTHint "ACES LMT"
#define kParamLMTOptionBypass "Bypass"
#define kParamLMTOptionBypassHint "Bypass"
#define kParamLMTOptionCustom "Custom"
#define kParamLMTOptionCustomHint "Custom LMT"
#define kParamLMTOptionFix "Blue Light Fix"
#define kParamLMTOptionFixHint "Blue Light Artifact Fix"
#define kParamLMTOptionPFE "PFE"
#define kParamLMTOptionPFEHint "Print Film Emulation"
#define kParamLMTOptionPFECustom "Custom PFE"
#define kParamLMTOptionPFECustomHint "Custom Print Film Emulation"
#define kParamLMTOptionBleach "Bleach Bypass"
#define kParamLMTOptionBleachHint "Bleach Bypass"

enum LMTEnum
{
eLMTBypass,
eLMTCustom,
eLMTFix,
eLMTPFE,
eLMTPFECustom,
eLMTBleach,
};

#define kParamCSCOUT "h_CSCOUT"
#define kParamCSCOUTLabel "ACES CSC out"
#define kParamCSCOUTHint "Convert from ACES to"
#define kParamCSCOUTOptionBypass "Bypass"
#define kParamCSCOUTOptionBypassHint "Bypass"
#define kParamCSCOUTOptionACEScc "ACEScc"
#define kParamCSCOUTOptionACESccHint "ACES to ACEScc"
#define kParamCSCOUTOptionACEScct "ACEScct"
#define kParamCSCOUTOptionACEScctHint "ACES to ACEScct"
#define kParamCSCOUTOptionACEScg "ACEScg"
#define kParamCSCOUTOptionACEScgHint "ACES to ACEScg"
#define kParamCSCOUTOptionACESproxy "ACESproxy"
#define kParamCSCOUTOptionACESproxyHint "ACES to ACESproxy"
#define kParamCSCOUTOptionADX "ADX"
#define kParamCSCOUTOptionADXHint "ACES to ADX"
#define kParamCSCOUTOptionICPCT "ICpCt"
#define kParamCSCOUTOptionICPCTHint "ACES to ICpCt"
#define kParamCSCOUTOptionLOGCAWG "LogC AlexaWG"
#define kParamCSCOUTOptionLOGCAWGHint "ACES to LogC AlexaWideGamut"
#define kParamCSCOUTOptionLOG3G10RWG "Log3G10 RedWG"
#define kParamCSCOUTOptionLOG3G10RWGHint "ACES to Log3G10 RedWideGamut"
#define kParamCSCOUTOptionSLOG3SG3 "Sony SLog3 SGamut3"
#define kParamCSCOUTOptionSLOG3SG3Hint "ACES to Slog3 SGamut3"
#define kParamCSCOUTOptionSLOG3SG3C "Sony SLog3 SGamut3Cine"
#define kParamCSCOUTOptionSLOG3SG3CHint "ACES to Slog3 SGamut3Cine"

enum CSCOUTEnum
{
eCSCOUTBypass,
eCSCOUTACEScc,
eCSCOUTACEScct,
eCSCOUTACEScg,
eCSCOUTACESproxy,
eCSCOUTADX,
eCSCOUTICPCT,
eCSCOUTLOGCAWG,
eCSCOUTLOG3G10RWG,
eCSCOUTSLOG3SG3,
eCSCOUTSLOG3SG3C,
};

#define kParamRRT "h_RRT"
#define kParamRRTLabel "RRT"
#define kParamRRTHint "ACES RRT"
#define kParamRRTOptionBypass "Bypass"
#define kParamRRTOptionBypassHint "Bypass"
#define kParamRRTOptionEnabled "Enabled"
#define kParamRRTOptionEnabledHint "Enabled"

enum RRTEnum
{
eRRTBypass,
eRRTEnabled,
};

#define kParamInvRRT "h_InvRRT"
#define kParamInvRRTLabel "Inverse RRT"
#define kParamInvRRTHint "ACES Inverse RRT"
#define kParamInvRRTOptionBypass "Bypass"
#define kParamInvRRTOptionBypassHint "Bypass"
#define kParamInvRRTOptionEnabled "Enabled"
#define kParamInvRRTOptionEnabledHint "Enabled"

enum InvRRTEnum
{
eInvRRTBypass,
eInvRRTEnabled,
};

#define kParamODT "h_ODT"
#define kParamODTLabel "ODT"
#define kParamODTHint "ACES ODT"
#define kParamODTOptionBypass "Bypass"
#define kParamODTOptionBypassHint "Bypass"
#define kParamODTOptionCustom "Custom"
#define kParamODTOptionCustomHint "Custom ODT"
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

#define kParamODTOptionRRTODT_Rec709_100nits_10nits_BT1886 "RRTODT Rec709 100nits 10nits BT1886"
#define kParamODTOptionRRTODT_Rec709_100nits_10nits_BT1886Hint "RRTODT Rec.709 100nits 10nits BT1886"
#define kParamODTOptionRRTODT_Rec709_100nits_10nits_sRGB "RRTODT Rec709 100nits 10nits sRGB"
#define kParamODTOptionRRTODT_Rec709_100nits_10nits_sRGBHint "RRTODT Rec.709 100nits 10nits sRGB"
#define kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084 "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamODTOptionRRTODT_P3D65_1000nits_15nits_ST2084 "RRTODT P3D65 1000nits 15nits ST2084"
#define kParamODTOptionRRTODT_P3D65_1000nits_15nits_ST2084Hint "RRTODT P3D65 1000nits 15nits ST2084"
#define kParamODTOptionRRTODT_P3D65_2000nits_15nits_ST2084 "RRTODT P3D65 2000nits 15nits ST2084"
#define kParamODTOptionRRTODT_P3D65_2000nits_15nits_ST2084Hint "RRTODT P3D65 2000nits 15nits ST2084"
#define kParamODTOptionRRTODT_P3D65_4000nits_15nits_ST2084 "RRTODT P3D65 4000nits 15nits ST2084"
#define kParamODTOptionRRTODT_P3D65_4000nits_15nits_ST2084Hint "RRTODT P3D65 4000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLG "RRTODT Rec2020 1000nits 15nits HLG"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint "RRTODT Rec.2020 1000nits 15nits HLG"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084 "RRTODT Rec2020 1000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint "RRTODT Rec.2020 1000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084 "RRTODT Rec2020 2000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint "RRTODT Rec.2020 2000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084 "RRTODT Rec2020 4000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint "RRTODT Rec.2020 4000nits 15nits ST2084"


enum ODTEnum
{
eODTBypass,
eODTCustom,
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
eODTRRTODT_Rec709_100nits_10nits_BT1886,
eODTRRTODT_Rec709_100nits_10nits_sRGB,
eODTRRTODT_P3D65_108nits_7_2nits_ST2084,
eODTRRTODT_P3D65_1000nits_15nits_ST2084,
eODTRRTODT_P3D65_2000nits_15nits_ST2084,
eODTRRTODT_P3D65_4000nits_15nits_ST2084,
eODTRRTODT_Rec2020_1000nits_15nits_HLG,
eODTRRTODT_Rec2020_1000nits_15nits_ST2084,
eODTRRTODT_Rec2020_2000nits_15nits_ST2084,
eODTRRTODT_Rec2020_4000nits_15nits_ST2084,
};

#define kParamInvODT "h_InvODT"
#define kParamInvODTLabel "Inverse ODT"
#define kParamInvODTHint "ACES Inverse ODT"
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

#define kParamInvODTOptionRRTODT_Rec709_100nits_10nits_BT1886 "RRTODT Rec709 100nits 10nits BT1886"
#define kParamInvODTOptionRRTODT_Rec709_100nits_10nits_BT1886Hint "RRTODT Rec.709 100nits 10nits BT1886"
#define kParamInvODTOptionRRTODT_Rec709_100nits_10nits_sRGB "RRTODT Rec709 100nits 10nits sRGB"
#define kParamInvODTOptionRRTODT_Rec709_100nits_10nits_sRGBHint "RRTODT Rec.709 100nits 10nits sRGB"
#define kParamInvODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084 "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamInvODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamInvODTOptionRRTODT_P3D65_1000nits_15nits_ST2084 "RRTODT P3D65 1000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_P3D65_1000nits_15nits_ST2084Hint "RRTODT P3D65 1000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_P3D65_2000nits_15nits_ST2084 "RRTODT P3D65 2000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_P3D65_2000nits_15nits_ST2084Hint "RRTODT P3D65 2000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_P3D65_4000nits_15nits_ST2084 "RRTODT P3D65 4000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_P3D65_4000nits_15nits_ST2084Hint "RRTODT P3D65 4000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_HLG "RRTODT Rec2020 1000nits 15nits HLG"
#define kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint "RRTODT Rec2020 1000nits 15nits HLG"
#define kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084 "RRTODT Rec2020 1000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint "RRTODT Rec2020 1000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084 "RRTODT Rec2020 2000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint "RRTODT Rec2020 2000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084 "RRTODT Rec2020 4000nits 15nits ST2084"
#define kParamInvODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint "RRTODT Rec2020 4000nits 15nits ST2084"

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
eInvODTRRTODT_Rec709_100nits_10nits_BT1886,
eInvODTRRTODT_Rec709_100nits_10nits_sRGB,
eInvODTRRTODT_P3D65_108nits_7_2nits_ST2084,
eInvODTRRTODT_P3D65_1000nits_15nits_ST2084,
eInvODTRRTODT_P3D65_2000nits_15nits_ST2084,
eInvODTRRTODT_P3D65_4000nits_15nits_ST2084,
eInvODTRRTODT_Rec2020_1000nits_15nits_HLG,
eInvODTRRTODT_Rec2020_1000nits_15nits_ST2084,
eInvODTRRTODT_Rec2020_2000nits_15nits_ST2084,
eInvODTRRTODT_Rec2020_4000nits_15nits_ST2084,
};

#define kParamDISPLAY "h_DISPLAY"
#define kParamDISPLAYLabel "Display Primaries"
#define kParamDISPLAYHint "ACES display primaries"
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

#define kParamLIMIT "h_LIMIT"
#define kParamLIMITLabel "Limiting Primaries"
#define kParamLIMITHint "ACES limiting primaries"
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

#define kParamEOTF "h_EOTF"
#define kParamEOTFLabel "EOTF"
#define kParamEOTFHint "ACES EOTF"
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

#define kParamSURROUND "h_SURROUND"
#define kParamSURROUNDLabel "Surround"
#define kParamSURROUNDHint "ACES Surround"
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

static const string kParamMathLUT = "lut";
static const string kParamMathLUTLabel = "shaper expr";
static const string kParamMathLUTHint = "expression for 1D LUT shaper";

#define kParamSHAPER "Shaper"
#define kParamSHAPERLabel "1D Shaper"
#define kParamSHAPERHint "Include 1D Shaper LUT"
#define kParamSHAPEROptionBypass "Bypass"
#define kParamSHAPEROptionBypassHint "No 1D Shaper"
#define kParamSHAPEROptionACEScc "Linear to ACEScc"
#define kParamSHAPEROptionACESccHint "ACES Linear to ACEScc 1D Shaper"
#define kParamSHAPEROptionACEScct "Linear to ACEScct"
#define kParamSHAPEROptionACEScctHint "ACES Linear to ACEScct 1D Shaper"
#define kParamSHAPEROptionACEScustom "Custom"
#define kParamSHAPEROptionACEScustomHint "Custom expression for 1D Shaper"

enum SHAPEREnum
{
eSHAPERBypass,
eSHAPERACEScc,
eSHAPERACEScct,
eSHAPERACEScustom,
};

////////////////////////////////////////////////////////////////////////////////

namespace {
struct RGBValues {
double r,g,b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};
}

struct MathProperties {
const string name;
string content;
bool processFlag;
};

class ACES : public OFX::ImageProcessor
{
public:
explicit ACES(OFX::ImageEffect& p_Instance);

virtual void processImagesCUDA();
//virtual void processImagesMetal();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(int p_Direction, int p_CSCIN, int p_IDT, int p_LMT, int p_CSCOUT, 
int p_RRT, int p_InvRRT, int p_ODT, int p_InvODT, float p_Exposure, float *p_LMTScale, 
float *p_Lum, int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_SWitch);

private:
OFX::Image* _srcImg;
int _direction;
int _cscin;
int _idt;
int _lmt;
int _cscout;
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
: OFX::ImageProcessor(p_Instance) {}

extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Direction, int p_CSCIN, int p_IDT, int p_LMT, int p_CSCOUT, int p_RRT, int p_InvRRT, 
int p_ODT, int p_InvODT, float p_Exposure, float* p_LMTScale, float *p_Lum, 
int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_Switch);

void ACES::processImagesCUDA()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunCudaKernel(input, output, width, height, _direction, _cscin, _idt, _lmt, _cscout, _rrt, 
_invrrt, _odt, _invodt, _exposure, _lmtscale, _lum, _display, _limit, _eotf, _surround, _switch);
}
/*
#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Direction, int p_CSCIN, int p_IDT, int p_LMT, int p_CSCOUT, int p_RRT, int p_InvRRT, 
int p_ODT, int p_InvODT, float p_Exposure, float* p_LMTScale, float *p_Lum, 
int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_Switch);
#endif

void ACES::processImagesMetal()
{
#ifdef __APPLE__
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunMetalKernel(_pMetalCmdQ, input, output, width, height, _direction, _cscin, _idt, _lmt, _cscout, _rrt, 
_invrrt, _odt, _invodt, _exposure, _lmtscale, _lum, _display, _limit, _eotf, _surround, _switch);
#endif
}
*/
void ACES::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
if (_effect.abort()) break;
float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
if (srcPix) {
float3 aces = make_float3(srcPix[0], srcPix[1], srcPix[2]);
if (_direction == 0) {
switch (_cscin){
case 0:
{}
break;
case 1:
{aces = ACEScc_to_ACES(aces);}
break;
case 2:
{aces = ACEScct_to_ACES(aces);}
break;
case 3:
{aces = ACEScg_to_ACES(aces);}
break;
case 4:
{aces = ACESproxy_to_ACES(aces);}
break;
case 5:
{aces = ADX_to_ACES(aces);}
break;
case 6:
{aces = ICpCt_to_ACES(aces);}
break;
case 7:
{aces = LogC_EI800_AWG_to_ACES(aces);}
break;
case 8:
{aces = Log3G10_RWG_to_ACES(aces);}
break;
case 9:
{aces = SLog3_SG3_to_ACES(aces);}
break;
case 10:
{aces = SLog3_SG3C_to_ACES(aces);}}

switch (_idt){
case 0:
{}
break;
case 1:
{aces = IDT_Alexa_v3_raw_EI800_CCT6500(aces);}
break;
case 2:
{aces = IDT_Panasonic_V35(aces);}
break;
case 3:
{aces = IDT_Canon_C100_A_D55(aces);}
break;
case 4:
{aces = IDT_Canon_C100_A_Tng(aces);}
break;
case 5:
{aces = IDT_Canon_C100mk2_A_D55(aces);}
break;
case 6:
{aces = IDT_Canon_C100mk2_A_Tng(aces);}
break;
case 7:
{aces = IDT_Canon_C300_A_D55(aces);}
break;
case 8:
{aces = IDT_Canon_C300_A_Tng(aces);}
break;
case 9:
{aces = IDT_Canon_C500_A_D55(aces);}
break;
case 10:
{aces = IDT_Canon_C500_A_Tng(aces);}
break;
case 11:
{aces = IDT_Canon_C500_B_D55(aces);}
break;
case 12:
{aces = IDT_Canon_C500_B_Tng(aces);}
break;
case 13:
{aces = IDT_Canon_C500_CinemaGamut_A_D55(aces);}
break;
case 14:
{aces = IDT_Canon_C500_CinemaGamut_A_Tng(aces);}
break;
case 15:
{aces = IDT_Canon_C500_DCI_P3_A_D55(aces);}
break;
case 16:
{aces = IDT_Canon_C500_DCI_P3_A_Tng(aces);}
break;
case 17:
{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_D55(aces);}
break;
case 18:
{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng(aces);}
break;
case 19:
{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55(aces);}
break;
case 20:
{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng(aces);}
break;
case 21:
{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55(aces);}
break;
case 22:
{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng(aces);}
break;
case 23:
{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55(aces);}
break;
case 24:
{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng(aces);}
break;
case 25:
{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55(aces);}
break;
case 26:
{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng(aces);}
break;
case 27:
{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55(aces);}
break;
case 28:
{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng(aces);}
break;
case 29:
{aces = IDT_Sony_SLog1_SGamut(aces);}
break;
case 30:
{aces = IDT_Sony_SLog2_SGamut_Daylight(aces);}
break;
case 31:
{aces = IDT_Sony_SLog2_SGamut_Tungsten(aces);}
break;
case 32:
{aces = IDT_Sony_Venice_SGamut3(aces);}
break;
case 33:
{aces = IDT_Sony_Venice_SGamut3Cine(aces);}
break;
case 34:
{aces = IDT_Rec709(aces);}
break;
case 35:
{aces = IDT_sRGB(aces);}
}

if (_exposure != 0.0f) {
aces.x *= exp2f(_exposure);
aces.y *= exp2f(_exposure);
aces.z *= exp2f(_exposure);
}

switch (_lmt)
{
case 0:
{}
break;
case 1:
{
if(_lmtscale[0] != 1.0f)
aces = scale_C(aces, _lmtscale[0]);
if(!(_lmtscale[1] == 1.0f && _lmtscale[2] == 0.0f && _lmtscale[3] == 1.0f)) {
float3 SLOPE = {_lmtscale[1], _lmtscale[1], _lmtscale[1]};
float3 OFFSET = {_lmtscale[2], _lmtscale[2], _lmtscale[2]};
float3 POWER = {_lmtscale[3], _lmtscale[3], _lmtscale[3]};
aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f);
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
{aces = LMT_BlueLightArtifactFix(aces);}
break;
case 3:
{aces = LMT_PFE(aces);}
break;
case 4:
{
if(_lmtscale[0] != 1.0f)
aces = scale_C(aces, _lmtscale[0]);
float3 SLOPE = {_lmtscale[1], _lmtscale[1], _lmtscale[1] * 0.94f};
float3 OFFSET = {_lmtscale[2], _lmtscale[2], _lmtscale[2] + 0.02f};
float3 POWER = {_lmtscale[3], _lmtscale[3], _lmtscale[3]};
aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f);
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
case 5:
{aces = LMT_Bleach(aces);}
}

switch (_cscout){
case 0:
{}
break;
case 1:
{aces = ACES_to_ACEScc(aces);}
break;
case 2:
{aces = ACES_to_ACEScct(aces);}
break;
case 3:
{aces = ACES_to_ACEScg(aces);}
break;
case 4:
{aces = ACES_to_ACESproxy(aces);}
break;
case 5:
{aces = ACES_to_ADX(aces);}
break;
case 6:
{aces = ACES_to_ICpCt(aces);}
break;
case 7:
{aces = ACES_to_LogC_EI800_AWG(aces);}
break;
case 8:
{aces = ACES_to_Log3G10_RWG(aces);}
break;
case 9:
{aces = ACES_to_SLog3_SG3(aces);}
break;
case 10:
{aces = ACES_to_SLog3_SG3C(aces);}
}

if (_rrt == 1 && _odt < 22)
aces = h_RRT(aces);

switch (_odt)
{
case 0:
{}
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
{aces = ODT_Rec709_100nits_dim(aces);}
break;
case 3:
{aces = ODT_Rec709_D60sim_100nits_dim(aces);}
break;
case 4:
{aces = ODT_sRGB_100nits_dim(aces);}
break;
case 5:
{aces = ODT_sRGB_D60sim_100nits_dim(aces);}
break;
case 6:
{aces = ODT_Rec2020_100nits_dim(aces);}
break;
case 7:
{aces = ODT_Rec2020_Rec709limited_100nits_dim(aces);}
break;
case 8:
{aces = ODT_Rec2020_P3D65limited_100nits_dim(aces);}
break;
case 9:
{aces = ODT_Rec2020_ST2084_1000nits(aces);}
break;
case 10:
{aces = ODT_P3DCI_48nits(aces);}
break;
case 11:
{aces = ODT_P3DCI_D60sim_48nits(aces);}
break;
case 12:
{aces = ODT_P3DCI_D65sim_48nits(aces);}
break;
case 13:
{aces = ODT_P3D60_48nits(aces);}
break;
case 14:
{aces = ODT_P3D65_48nits(aces);}
break;
case 15:
{aces = ODT_P3D65_D60sim_48nits(aces);}
break;
case 16:
{aces = ODT_P3D65_Rec709limited_48nits(aces);}
break;
case 17:
{aces = ODT_DCDM(aces);}
break;
case 18:
{aces = ODT_DCDM_P3D60limited(aces);}
break;
case 19:
{aces = ODT_DCDM_P3D65limited(aces);}
break;
case 20:
{aces = ODT_RGBmonitor_100nits_dim(aces);}
break;
case 21:
{aces = ODT_RGBmonitor_D60sim_100nits_dim(aces);}
break;
case 22:
{aces = RRTODT_Rec709_100nits_10nits_BT1886(aces);}
break;
case 23:
{aces = RRTODT_Rec709_100nits_10nits_sRGB(aces);}
break;
case 24:
{aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);}
break;
case 25:
{aces = RRTODT_P3D65_1000nits_15nits_ST2084(aces);}
break;
case 26:
{aces = RRTODT_P3D65_2000nits_15nits_ST2084(aces);}
break;
case 27:
{aces = RRTODT_P3D65_4000nits_15nits_ST2084(aces);}
break;
case 28:
{aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);}
break;
case 29:
{aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);}
break;
case 30:
{aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);}
break;
case 31:
{aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);}
}
} else {

switch (_invodt)
{
case 0:
{}
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
{aces = InvODT_Rec709_100nits_dim(aces);}
break;
case 3:
{aces = InvODT_Rec709_D60sim_100nits_dim(aces);}
break;
case 4:
{aces = InvODT_sRGB_100nits_dim(aces);}
break;
case 5:
{aces = InvODT_sRGB_D60sim_100nits_dim(aces);}
break;
case 6:
{aces = InvODT_Rec2020_100nits_dim(aces);}
break;
case 7:
{aces = InvODT_Rec2020_ST2084_1000nits(aces);}
break;
case 8:
{aces = InvODT_P3DCI_48nits(aces);}
break;
case 9:
{aces = InvODT_P3DCI_D60sim_48nits(aces);}
break;
case 10:
{aces = InvODT_P3DCI_D65sim_48nits(aces);}
break;
case 11:
{aces = InvODT_P3D60_48nits(aces);}
break;
case 12:
{aces = InvODT_P3D65_48nits(aces);}
break;
case 13:
{aces = InvODT_P3D65_D60sim_48nits(aces);}
break;
case 14:
{aces = InvODT_DCDM(aces);}
break;
case 15:
{aces = InvODT_DCDM_P3D65limited(aces);}
break;
case 16:
{aces = InvODT_RGBmonitor_100nits_dim(aces);}
break;
case 17:
{aces = InvODT_RGBmonitor_D60sim_100nits_dim(aces);}
break;
case 18:
{aces = InvRRTODT_Rec709_100nits_10nits_BT1886(aces);}
break;
case 19:
{aces = InvRRTODT_Rec709_100nits_10nits_sRGB(aces);}
break;
case 20:
{aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);}
break;
case 21:
{aces = InvRRTODT_P3D65_1000nits_15nits_ST2084(aces);}
break;
case 22:
{aces = InvRRTODT_P3D65_2000nits_15nits_ST2084(aces);}
break;
case 23:
{aces = InvRRTODT_P3D65_4000nits_15nits_ST2084(aces);}
break;
case 24:
{aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);}
break;
case 25:
{aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);}
break;
case 26:
{aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);}
break;
case 27:
{aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);}
}

if(_invrrt == 1 && _invodt < 18)
aces = h_InvRRT(aces);
}

dstPix[0] = aces.x;
dstPix[1] = aces.y;
dstPix[2] = aces.z;
dstPix[3] = srcPix[3];
} else {
for (int c = 0; c < 4; ++c) {
dstPix[c] = 0;
}}
dstPix += 4;
}}}

void ACES::setSrcImg(OFX::Image* p_SrcImg) {
_srcImg = p_SrcImg;
}

void ACES::setScales(int p_Direction, int p_CSCIN, int p_IDT, int p_LMT, int p_CSCOUT, int p_RRT, 
int p_InvRRT, int p_ODT, int p_InvODT, float p_Exposure, float *p_LMTScale, float *p_Lum, 
int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_Switch)
{
_direction = p_Direction;
_cscin = p_CSCIN;
_idt = p_IDT;
_lmt = p_LMT;
_cscout = p_CSCOUT;
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

class ACESPlugin : public OFX::ImageEffect
{
public:
explicit ACESPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
void setupAndProcess(ACES &p_ACES, const OFX::RenderArguments& p_Args);

private:
OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;
OFX::ChoiceParam* m_Direction;
OFX::ChoiceParam* m_CSCIN;
OFX::ChoiceParam* m_IDT;
OFX::ChoiceParam* m_LMT;
OFX::ChoiceParam* m_CSCOUT;
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
OFX::ChoiceParam* m_Shaper;
OFX::StringParam* m_ExprLUT;
OFX::Double2DParam* m_Input1;
OFX::Double2DParam* m_Input2;
OFX::IntParam* m_Lutsize;
OFX::IntParam* m_Cube;
OFX::IntParam* m_Precision;
OFX::StringParam* m_Path;
OFX::StringParam* m_Name;
OFX::StringParam* m_Path2;
OFX::StringParam* m_Name2;
OFX::PushButtonParam* m_Info;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
};

ACESPlugin::ACESPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
m_Direction = fetchChoiceParam(kParamInput);
m_CSCIN = fetchChoiceParam(kParamCSCIN);
m_IDT = fetchChoiceParam(kParamIDT);
m_LMT = fetchChoiceParam(kParamLMT);
m_CSCOUT = fetchChoiceParam(kParamCSCOUT);
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
m_DISPLAY = fetchChoiceParam(kParamDISPLAY);
m_LIMIT = fetchChoiceParam(kParamLIMIT);
m_EOTF = fetchChoiceParam(kParamEOTF);
m_SURROUND = fetchChoiceParam(kParamSURROUND);
m_STRETCH = fetchBooleanParam("STRETCH");
m_D60SIM = fetchBooleanParam("D60SIM");
m_LEGALRANGE = fetchBooleanParam("LEGALRANGE");
m_Shaper = fetchChoiceParam(kParamSHAPER);
m_ExprLUT = fetchStringParam(kParamMathLUT);
assert(m_ExprLUT);
m_Input1 = fetchDouble2DParam("range1");
m_Input2 = fetchDouble2DParam("range2");
m_Path = fetchStringParam("path");
m_Name = fetchStringParam("name");
m_Path2 = fetchStringParam("path2");
m_Name2 = fetchStringParam("name2");
m_Cube = fetchIntParam("cube");
m_Precision = fetchIntParam("precision");
m_Lutsize = fetchIntParam("lutsize");
m_Info = fetchPushButtonParam("info");
m_Button1 = fetchPushButtonParam("button1");
m_Button2 = fetchPushButtonParam("button2");
}

void ACESPlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA)) {
ACES ACES(*this);
setupAndProcess(ACES, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool ACESPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{ 
int _idt;
m_IDT->getValueAtTime(p_Args.time, _idt);
bool idt = _idt == 0;
int _cscin;
m_CSCIN->getValueAtTime(p_Args.time, _cscin);
bool cscin = _cscin == 0;
int _lmt;
m_LMT->getValueAtTime(p_Args.time, _lmt);
bool lmt = _lmt == 0;
int _cscout;
m_CSCOUT->getValueAtTime(p_Args.time, _cscout);
bool cscout = _cscout == 0;
int _rrt;
m_RRT->getValueAtTime(p_Args.time, _rrt);
bool rrt = _rrt == 0;
int _invrrt;
m_InvRRT->getValueAtTime(p_Args.time, _invrrt);
bool invrrt = _invrrt == 0;
int _odt;
m_ODT->getValueAtTime(p_Args.time, _odt);
bool odt = _odt == 0;
int _invodt;
m_InvODT->getValueAtTime(p_Args.time, _invodt);
bool invodt = _invodt == 0;
float _exposure = m_Exposure->getValueAtTime(p_Args.time);
bool exposure = _exposure == 0.0f;

if (idt && cscin && lmt && cscout && rrt && invrrt && odt && invodt && exposure) {
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void ACESPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if (p_ParamName == kParamInput) {
int direction;
m_Direction->getValueAtTime(p_Args.time, direction);
bool forward = direction == 0;
bool inverse = direction == 1;
int _lmt;
m_LMT->getValueAtTime(p_Args.time, _lmt);
bool custom = _lmt == 1;
m_CSCIN->setIsSecretAndDisabled(!forward);
m_IDT->setIsSecretAndDisabled(!forward);
m_LMT->setIsSecretAndDisabled(!forward);
m_CSCOUT->setIsSecretAndDisabled(!forward);
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
int _odt;
m_ODT->getValueAtTime(p_Args.time, _odt);
bool odt = _odt == 1;
int _invodt;
m_InvODT->getValueAtTime(p_Args.time, _invodt);
bool invodt = _invodt == 1;
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

if (p_ParamName == kParamLMT) {
int _lmt;
m_LMT->getValueAtTime(p_Args.time, _lmt);
bool custom = _lmt == 1 || _lmt == 4;
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

if (_lmt == 4) {
m_ScaleC->setValue(0.7);
m_Slope->setValue(1.0);
m_Offset->setValue(0.0);
m_Power->setValue(1.0);
m_Gamma->setValue(1.5);
m_Pivot->setValue(0.18);
m_RotateH1->setValue(0.0);
m_Range1->setValue(30.0);
m_Shift1->setValue(5.0);
m_RotateH2->setValue(80.0);
m_Range2->setValue(60.0);
m_Shift2->setValue(-15.0);
m_RotateH3->setValue(52.0);
m_Range3->setValue(50.0);
m_Shift3->setValue(-14.0);
m_HueCH1->setValue(45.0);
m_RangeCH1->setValue(40.0);
m_ScaleCH1->setValue(1.4);
m_RotateH4->setValue(190.0);
m_Range4->setValue(40.0);
m_Shift4->setValue(30.0);
m_HueCH2->setValue(240.0);
m_RangeCH2->setValue(120.0);
m_ScaleCH2->setValue(1.4);
}

if (_lmt == 1) {
m_ScaleC->setValue(1.0);
m_Slope->setValue(1.0);
m_Offset->setValue(0.0);
m_Power->setValue(1.0);
m_Gamma->setValue(1.0);
m_Pivot->setValue(0.18);
m_RotateH1->setValue(0.0);
m_Range1->setValue(30.0);
m_Shift1->setValue(0.0);
m_RotateH2->setValue(80.0);
m_Range2->setValue(60.0);
m_Shift2->setValue(0.0);
m_RotateH3->setValue(52.0);
m_Range3->setValue(50.0);
m_Shift3->setValue(0.0);
m_HueCH1->setValue(45.0);
m_RangeCH1->setValue(40.0);
m_ScaleCH1->setValue(1.0);
m_RotateH4->setValue(190.0);
m_Range4->setValue(40.0);
m_Shift4->setValue(0.0);
m_HueCH2->setValue(240.0);
m_RangeCH2->setValue(120.0);
m_ScaleCH2->setValue(1.0);
}}

if (p_ParamName == kParamODT) {
int _odt;
m_ODT->getValueAtTime(p_Args.time, _odt);
bool custom = _odt == 1;
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

if (p_ParamName == kParamInvODT) {
int _invodt;
m_InvODT->getValueAtTime(p_Args.time, _invodt);
bool custom = _invodt == 1;
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

if (p_ParamName == "info") {
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

int _direction;
m_Direction->getValueAtTime(p_Args.time, _direction);
int _cscin;
m_CSCIN->getValueAtTime(p_Args.time, _cscin);
int _idt;
m_IDT->getValueAtTime(p_Args.time, _idt);
int _lmt;
m_LMT->getValueAtTime(p_Args.time, _lmt);
int _cscout;
m_CSCOUT->getValueAtTime(p_Args.time, _cscout);
int _rrt;
m_RRT->getValueAtTime(p_Args.time, _rrt);
int _invrrt;
m_InvRRT->getValueAtTime(p_Args.time, _invrrt);
int _odt;
m_ODT->getValueAtTime(p_Args.time, _odt);
int _invodt;
m_InvODT->getValueAtTime(p_Args.time, _invodt);
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
int _display;
m_DISPLAY->getValueAtTime(p_Args.time, _display);
int _limit;
m_LIMIT->getValueAtTime(p_Args.time, _limit);
int _eotf;
m_EOTF->getValueAtTime(p_Args.time, _eotf);
int _surround;
m_SURROUND->getValueAtTime(p_Args.time, _surround);
int _switch[3];
bool stretch = m_STRETCH->getValueAtTime(p_Args.time);
_switch[0] = stretch ? 1 : 0;
bool d60sim = m_D60SIM->getValueAtTime(p_Args.time);
_switch[1] = d60sim ? 1 : 0;
bool legalrange = m_LEGALRANGE->getValueAtTime(p_Args.time);
_switch[2] = legalrange ? 1 : 0;

if (p_ParamName == "button1") {
string PATH;
m_Path->getValue(PATH);
string NAME;
m_Name->getValue(NAME);
OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + " ?");
if (reply == OFX::Message::eMessageReplyYes) {
FILE * pFile;
pFile = fopen ((PATH + NAME + ".dctl").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "// ACES 1.2 DCTL export \n" \
"// Should be placed in directory as ACES_LIB.h \n" \
" \n" \
"#include \"ACES_LIB.h\" \n" \
" \n" \
"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B) \n" \
"{ \n" \
"int Direction = %d; \n" \
"int CscIn = %d; \n" \
"int Idt = %d; \n" \
"int Lmt = %d; \n" \
"int CscOut = %d; \n" \
"int Rrt = %d; \n" \
"int InvRrt = %d; \n" \
"int Odt = %d; \n" \
"int InvOdt = %d; \n" \
"float Exposure = %ff; \n" \
"float Color_Boost = %ff; \n" \
"float Slope = %ff; \n" \
"float Offset = %ff; \n" \
"float Power = %ff; \n" \
"float Contrast = %ff; \n" \
"float Pivot = %ff; \n" \
"float RotateH1 = %ff; \n" \
"float Range1 = %ff; \n" \
"float Shift1 = %ff; \n" \
"float RotateH2 = %ff; \n" \
"float Range2 = %ff; \n" \
"float Shift2 = %ff; \n" \
"float RotateH3 = %ff; \n" \
"float Range3 = %ff; \n" \
"float Shift3 = %ff; \n" \
"float HueCH1 = %ff; \n" \
"float RangeCH1 = %ff; \n" \
"float ScaleCH1 = %ff; \n" \
"float RotateH4 = %ff; \n" \
"float Range4 = %ff; \n" \
"float Shift4 = %ff; \n" \
"float HueCH2 = %ff; \n" \
"float RangeCH2 = %ff; \n" \
"float ScaleCH2 = %ff; \n" \
"float Black_Luminance = %ff; \n" \
"float Midpoint_Luminance = %ff; \n" \
"float Peak_white_Luminance = %ff; \n" \
"int Display = %d; \n" \
"int Limit = %d; \n" \
"int Eotf = %d; \n" \
"int Surround = %d; \n" \
"int Stretch_Black_Luminance = %d; \n" \
"int D60sim = %d; \n" \
"int Legal_Range = %d; \n" \
"float3 aces = make_float3(p_R, p_G, p_B); \n" \
"if (Direction == 0) { \n" \
"switch (CscIn){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{aces = ACEScc_to_ACES(aces);} \n" \
"break; \n" \
"case 2: \n" \
"{aces = ACEScct_to_ACES(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = ACEScg_to_ACES(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = ACESproxy_to_ACES(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = ADX_to_ACES(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = ICpCt_to_ACES(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = LogC_EI800_AWG_to_ACES(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = Log3G10_RWG_to_ACES(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = SLog3_SG3_to_ACES(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = SLog3_SG3C_to_ACES(aces);}} \n" \
"switch (Idt){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{aces = IDT_Alexa_v3_raw_EI800_CCT6500(aces);} \n" \
"break; \n" \
"case 2: \n" \
"{aces = IDT_Panasonic_V35(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = IDT_Canon_C100_A_D55(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = IDT_Canon_C100_A_Tng(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = IDT_Canon_C100mk2_A_D55(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = IDT_Canon_C100mk2_A_Tng(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = IDT_Canon_C300_A_D55(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = IDT_Canon_C300_A_Tng(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = IDT_Canon_C500_A_D55(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = IDT_Canon_C500_A_Tng(aces);} \n" \
"break; \n" \
"case 11: \n" \
"{aces = IDT_Canon_C500_B_D55(aces);} \n" \
"break; \n" \
"case 12: \n" \
"{aces = IDT_Canon_C500_B_Tng(aces);} \n" \
"break; \n" \
"case 13: \n" \
"{aces = IDT_Canon_C500_CinemaGamut_A_D55(aces);} \n" \
"break; \n" \
"case 14: \n" \
"{aces = IDT_Canon_C500_CinemaGamut_A_Tng(aces);} \n" \
"break; \n" \
"case 15: \n" \
"{aces = IDT_Canon_C500_DCI_P3_A_D55(aces);} \n" \
"break; \n" \
"case 16: \n" \
"{aces = IDT_Canon_C500_DCI_P3_A_Tng(aces);} \n" \
"break; \n" \
"case 17: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_D55(aces);} \n" \
"break; \n" \
"case 18: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng(aces);} \n" \
"break; \n" \
"case 19: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55(aces);} \n" \
"break; \n" \
"case 20: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng(aces);} \n" \
"break; \n" \
"case 21: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55(aces);} \n" \
"break; \n" \
"case 22: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng(aces);} \n" \
"break; \n" \
"case 23: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55(aces);} \n" \
"break; \n" \
"case 24: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng(aces);} \n" \
"break; \n" \
"case 25: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55(aces);} \n" \
"break; \n" \
"case 26: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng(aces);} \n" \
"break; \n" \
"case 27: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55(aces);} \n" \
"break; \n" \
"case 28: \n" \
"{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng(aces);} \n" \
"break; \n" \
"case 29: \n" \
"{aces = IDT_Sony_SLog1_SGamut(aces);} \n" \
"break; \n" \
"case 30: \n" \
"{aces = IDT_Sony_SLog2_SGamut_Daylight(aces);} \n" \
"break; \n" \
"case 31: \n" \
"{aces = IDT_Sony_SLog2_SGamut_Tungsten(aces);} \n" \
"break; \n" \
"case 32: \n" \
"{aces = IDT_Sony_Venice_SGamut3(aces);} \n" \
"break; \n" \
"case 33: \n" \
"{aces = IDT_Sony_Venice_SGamut3Cine(aces);} \n" \
"break; \n" \
"case 34: \n" \
"{aces = IDT_Rec709(aces);} \n" \
"break; \n" \
"case 35: \n" \
"{aces = IDT_sRGB(aces);}} \n" \
"if (Exposure != 0.0f) { \n" \
"aces.x *= _exp2f(Exposure); \n" \
"aces.y *= _exp2f(Exposure); \n" \
"aces.z *= _exp2f(Exposure); \n" \
"} \n" \
"switch (Lmt){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{ \n" \
"if (Color_Boost != 1.0f) \n" \
"aces = scale_C(aces, Color_Boost); \n" \
"if (!(Slope == 1.0f && Offset == 0.0f && Power == 1.0f)) { \n" \
"float3 SLOPE = {Slope, Slope, Slope}; \n" \
"float3 OFFSET = {Offset, Offset, Offset}; \n" \
"float3 POWER = {Power, Power, Power}; \n" \
"aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f); \n" \
"} \n" \
"if (Contrast != 1.0f) \n" \
"aces = gamma_adjust_linear(aces, Contrast, Pivot); \n" \
"if (Shift1 != 0.0f) \n" \
"aces = rotate_H_in_H(aces, RotateH1, Range1, Shift1); \n" \
"if (Shift2 != 0.0f) \n" \
"aces = rotate_H_in_H(aces, RotateH2, Range2, Shift2); \n" \
"if (Shift3 != 0.0f) \n" \
"aces = rotate_H_in_H(aces, RotateH3, Range3, Shift3); \n" \
"if (ScaleCH1 != 1.0f) \n" \
"aces = scale_C_at_H(aces, HueCH1, RangeCH1, ScaleCH1); \n" \
"if (Shift4 != 0.0f) \n" \
"aces = rotate_H_in_H(aces, RotateH4, Range4, Shift4); \n" \
"if (ScaleCH2 != 1.0f) \n" \
"aces = scale_C_at_H(aces, HueCH2, RangeCH2, ScaleCH2); \n" \
"} \n" \
"break; \n" \
"case 2: \n" \
"{ \n" \
"aces = LMT_BlueLightArtifactFix(aces); \n" \
"} \n" \
"break; \n" \
"case 3: \n" \
"{ \n" \
"aces = LMT_PFE(aces); \n" \
"} \n" \
"break; \n" \
"case 4: \n" \
"{ \n" \
"if (Color_Boost != 1.0f) \n" \
"aces = scale_C(aces, Color_Boost); \n" \
"float3 SLOPE = {Slope, Slope, Slope * 0.94f}; \n" \
"float3 OFFSET = {Offset, Offset, Offset + 0.02f}; \n" \
"float3 POWER = {Power, Power, Power}; \n" \
"aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f); \n" \
"if (Contrast != 1.0f) \n" \
"aces = gamma_adjust_linear(aces, Contrast, Pivot); \n" \
"if (Shift1 != 0.0f) \n" \
"aces = rotate_H_in_H(aces, RotateH1, Range1, Shift1); \n" \
"if (Shift2 != 0.0f) \n" \
"aces = rotate_H_in_H(aces, RotateH2, Range2, Shift2); \n" \
"if (Shift3 != 0.0f) \n" \
"aces = rotate_H_in_H(aces, RotateH3, Range3, Shift3); \n" \
"if (ScaleCH1 != 1.0f) \n" \
"aces = scale_C_at_H(aces, HueCH1, RangeCH1, ScaleCH1); \n" \
"if (Shift4 != 0.0f) \n" \
"aces = rotate_H_in_H(aces, RotateH4, Range4, Shift4); \n" \
"if (ScaleCH2 != 1.0f) \n" \
"aces = scale_C_at_H(aces, HueCH2, RangeCH2, ScaleCH2); \n" \
"} \n" \
"break; \n" \
"case 5: \n" \
"{ \n" \
"aces = LMT_Bleach(aces); \n" \
"}} \n" \
"switch (CscOut){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{aces = ACES_to_ACEScc(aces);} \n" \
"break; \n" \
"case 2: \n" \
"{aces = ACES_to_ACEScct(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = ACES_to_ACEScg(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = ACES_to_ACESproxy(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = ACES_to_ADX(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = ACES_to_ICpCt(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = ACES_to_LogC_EI800_AWG(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = ACES_to_Log3G10_RWG(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = ACES_to_SLog3_SG3(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = ACES_to_SLog3_SG3C(aces);} \n" \
"} \n" \
"if (Rrt == 1 && Odt < 22) \n" \
"aces = RRT(aces); \n" \
"switch (Odt){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{ \n" \
"float Y_MIN = Black_Luminance; \n" \
"float Y_MID = Midpoint_Luminance; \n" \
"float Y_MAX = Peak_white_Luminance; \n" \
"Chromaticities DISPLAY_PRI = Display == 0 ? REC2020_PRI : Display == 1 ? P3D60_PRI : Display == 2 ? P3D65_PRI : Display == 3 ? P3DCI_PRI : REC709_PRI; \n" \
"Chromaticities LIMITING_PRI = Limit == 0 ? REC2020_PRI : Limit == 1 ? P3D60_PRI : Limit == 2 ? P3D65_PRI : Limit == 3 ? P3DCI_PRI : REC709_PRI; \n" \
"int EOTF = Eotf; \n" \
"int SURROUND = Surround; \n" \
"bool STRETCH_BLACK = Stretch_Black_Luminance == 1; \n" \
"bool D60_SIM = D60sim == 1; \n" \
"bool LEGAL_RANGE = Legal_Range == 1; \n" \
"aces = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"} \n" \
"break; \n" \
"case 2: \n" \
"{aces = ODT_Rec709_100nits_dim(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = ODT_Rec709_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = ODT_sRGB_100nits_dim(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = ODT_sRGB_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = ODT_Rec2020_100nits_dim(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = ODT_Rec2020_Rec709limited_100nits_dim(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = ODT_Rec2020_P3D65limited_100nits_dim(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = ODT_Rec2020_ST2084_1000nits(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = ODT_P3DCI_48nits(aces);} \n" \
"break; \n" \
"case 11: \n" \
"{aces = ODT_P3DCI_D60sim_48nits(aces);} \n" \
"break; \n" \
"case 12: \n" \
"{aces = ODT_P3DCI_D65sim_48nits(aces);} \n" \
"break; \n" \
"case 13: \n" \
"{aces = ODT_P3D60_48nits(aces);} \n" \
"break; \n" \
"case 14: \n" \
"{aces = ODT_P3D65_48nits(aces);} \n" \
"break; \n" \
"case 15: \n" \
"{aces = ODT_P3D65_D60sim_48nits(aces);} \n" \
"break; \n" \
"case 16: \n" \
"{aces = ODT_P3D65_Rec709limited_48nits(aces);} \n" \
"break; \n" \
"case 17: \n" \
"{aces = ODT_DCDM(aces);} \n" \
"break; \n" \
"case 18: \n" \
"{aces = ODT_DCDM_P3D60limited(aces);} \n" \
"break; \n" \
"case 19: \n" \
"{aces = ODT_DCDM_P3D65limited(aces);} \n" \
"break; \n" \
"case 20: \n" \
"{aces = ODT_RGBmonitor_100nits_dim(aces);} \n" \
"break; \n" \
"case 21: \n" \
"{aces = ODT_RGBmonitor_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 22: \n" \
"{aces = RRTODT_Rec709_100nits_10nits_BT1886(aces);} \n" \
"break; \n" \
"case 23: \n" \
"{aces = RRTODT_Rec709_100nits_10nits_sRGB(aces);} \n" \
"break; \n" \
"case 24: \n" \
"{aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);} \n" \
"break; \n" \
"case 25: \n" \
"{aces = RRTODT_P3D65_1000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 26: \n" \
"{aces = RRTODT_P3D65_2000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 27: \n" \
"{aces = RRTODT_P3D65_4000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 28: \n" \
"{aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);} \n" \
"break; \n" \
"case 29: \n" \
"{aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 30: \n" \
"{aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 31: \n" \
"{aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);} \n" \
"} \n" \
"} else { \n" \
"switch (InvOdt){ \n" \
"case 0: \n" \
"{} \n" \
"break; \n" \
"case 1: \n" \
"{ \n" \
"float Y_MIN = Black_Luminance; \n" \
"float Y_MID = Midpoint_Luminance; \n" \
"float Y_MAX = Peak_white_Luminance; \n" \
"Chromaticities DISPLAY_PRI = Display == 0 ? REC2020_PRI : Display == 1 ? P3D60_PRI : Display == 2 ? P3D65_PRI : Display == 3 ? P3DCI_PRI : REC709_PRI; \n" \
"Chromaticities LIMITING_PRI = Limit == 0 ? REC2020_PRI : Limit == 1 ? P3D60_PRI : Limit == 2 ? P3D65_PRI : Limit == 3 ? P3DCI_PRI : REC709_PRI; \n" \
"int EOTF = Eotf; \n" \
"int SURROUND = Surround; \n" \
"bool STRETCH_BLACK = Stretch_Black_Luminance == 1; \n" \
"bool D60_SIM = D60sim == 1; \n" \
"bool LEGAL_RANGE = Legal_Range == 1; \n" \
"aces = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI, LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE ); \n" \
"} \n" \
"break; \n" \
"case 2: \n" \
"{aces = InvODT_Rec709_100nits_dim(aces);} \n" \
"break; \n" \
"case 3: \n" \
"{aces = InvODT_Rec709_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 4: \n" \
"{aces = InvODT_sRGB_100nits_dim(aces);} \n" \
"break; \n" \
"case 5: \n" \
"{aces = InvODT_sRGB_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 6: \n" \
"{aces = InvODT_Rec2020_100nits_dim(aces);} \n" \
"break; \n" \
"case 7: \n" \
"{aces = InvODT_Rec2020_ST2084_1000nits(aces);} \n" \
"break; \n" \
"case 8: \n" \
"{aces = InvODT_P3DCI_48nits(aces);} \n" \
"break; \n" \
"case 9: \n" \
"{aces = InvODT_P3DCI_D60sim_48nits(aces);} \n" \
"break; \n" \
"case 10: \n" \
"{aces = InvODT_P3DCI_D65sim_48nits(aces);} \n" \
"break; \n" \
"case 11: \n" \
"{aces = InvODT_P3D60_48nits(aces);} \n" \
"break; \n" \
"case 12: \n" \
"{aces = InvODT_P3D65_48nits(aces);} \n" \
"break; \n" \
"case 13: \n" \
"{aces = InvODT_P3D65_D60sim_48nits(aces);} \n" \
"break; \n" \
"case 14: \n" \
"{aces = InvODT_DCDM(aces);} \n" \
"break; \n" \
"case 15: \n" \
"{aces = InvODT_DCDM_P3D65limited(aces);} \n" \
"break; \n" \
"case 16: \n" \
"{aces = InvODT_RGBmonitor_100nits_dim(aces);} \n" \
"break; \n" \
"case 17: \n" \
"{aces = InvODT_RGBmonitor_D60sim_100nits_dim(aces);} \n" \
"break; \n" \
"case 18: \n" \
"{aces = InvRRTODT_Rec709_100nits_10nits_BT1886(aces);} \n" \
"break; \n" \
"case 19: \n" \
"{aces = InvRRTODT_Rec709_100nits_10nits_sRGB(aces);} \n" \
"break; \n" \
"case 20: \n" \
"{aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);} \n" \
"break; \n" \
"case 21: \n" \
"{aces = InvRRTODT_P3D65_1000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 22: \n" \
"{aces = InvRRTODT_P3D65_2000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 23: \n" \
"{aces = InvRRTODT_P3D65_4000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 24: \n" \
"{aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);} \n" \
"break; \n" \
"case 25: \n" \
"{aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 26: \n" \
"{aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);} \n" \
"break; \n" \
"case 27: \n" \
"{aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);} \n" \
"} \n" \
"if (InvRrt == 1 && InvOdt < 18) \n" \
"aces = InvRRT(aces); \n" \
"} \n" \
"return make_float3(aces.x, aces.y, aces.z); \n" \
"}\n", _direction, _cscin, _idt, _lmt, _cscout, _rrt, _invrrt, _odt, _invodt, _exposure, 
_lmtscale[0], _lmtscale[1], _lmtscale[2], _lmtscale[3], _lmtscale[4], _lmtscale[5], _lmtscale[6], _lmtscale[7], 
_lmtscale[8], _lmtscale[9], _lmtscale[10], _lmtscale[11], _lmtscale[12], _lmtscale[13], _lmtscale[14], _lmtscale[15], 
_lmtscale[16], _lmtscale[17], _lmtscale[18], _lmtscale[19], _lmtscale[20], _lmtscale[21], _lmtscale[22], _lmtscale[23], 
_lum[0], _lum[1], _lum[2], _display, _limit, _eotf, _surround, _switch[0], _switch[1], _switch[2]);
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
}}}

if (p_ParamName == kParamSHAPER) {
int Shaper;
m_Shaper->getValueAtTime(p_Args.time, Shaper);
double min_cc = ACEScc_to_lin(0.0);
double max_cc = ACEScc_to_lin(1.0);
double min_cct = ACEScct_to_lin(0.0);
double max_cct = ACEScct_to_lin(1.0);
if (Shaper == 0) {
m_Input1->setIsSecretAndDisabled(true);
m_Lutsize->setIsSecretAndDisabled(true);
m_ExprLUT->setIsSecretAndDisabled(true);
m_Input2->setValue(0.0, 1.0);
}else if (Shaper == 1){
m_Input1->setIsSecretAndDisabled(false);
m_Lutsize->setIsSecretAndDisabled(false);
m_ExprLUT->setIsSecretAndDisabled(true);
m_Input1->setValue(min_cc, max_cc);
}
else if (Shaper == 2){
m_Input1->setIsSecretAndDisabled(false);
m_Lutsize->setIsSecretAndDisabled(false);
m_ExprLUT->setIsSecretAndDisabled(true);
m_Input1->setValue(min_cct, max_cct);
} else {
m_Input1->setIsSecretAndDisabled(false);
m_Lutsize->setIsSecretAndDisabled(false);
m_ExprLUT->setIsSecretAndDisabled(false);
}}

if (p_ParamName == "range1" || p_ParamName == kParamMathLUT || p_ParamName == kParamSHAPER) {
int Shaper;
m_Shaper->getValueAtTime(p_Args.time, Shaper);
double shaperA = 0.0;
double shaperB = 0.0;
double inputA = 0.0;
double inputB = 0.0;
m_Input1->getValueAtTime(p_Args.time, shaperA, shaperB);
double ccA, ccB, cctA, cctB;
ccA = lin_to_ACEScc(shaperA);
ccB = lin_to_ACEScc(shaperB);
cctA = lin_to_ACEScct(shaperA);
cctB = lin_to_ACEScct(shaperB);
string exprLUT;
m_ExprLUT->getValue(exprLUT);
bool processLUT = !exprLUT.empty();
exprLUT = exprLUT.empty() ? "x" : exprLUT;
double temPix = 0.0;
bool doLUT = processLUT;

exprtk::symbol_table<double> symbol_table;
symbol_table.add_variable("x",temPix);
MathProperties exprLUT_props = {kParamMathLUT, exprLUT, true};
exprtk::function_compositor<double> compositor(symbol_table);
compositor.add("lerp", " a*(c-b)+b;", "a","b","c");

MathProperties E = exprLUT_props;
exprtk::expression<double> expressionLUT;
expressionLUT.register_symbol_table(symbol_table);
exprtk::parser<double> parserLUT;
doLUT = parserLUT.compile(E.content,expressionLUT);

if (Shaper == 0){}
else if(Shaper == 1){
m_Input2->setValue(ccA, ccB);
} else if(Shaper == 2){
m_Input2->setValue(cctA, cctB);
} else {
temPix = shaperA;
inputA = doLUT ? expressionLUT.value() : shaperA;
temPix = shaperB;
inputB = doLUT ? expressionLUT.value() : shaperB;
m_Input2->setValue(inputA, inputB);
}}

if (p_ParamName == "button2") {
int Shaper;
m_Shaper->getValueAtTime(p_Args.time, Shaper);
double shaper = 0.0;
double shaperA = 0.0;
double shaperB = 0.0;
double inputA = 0.0;
double inputB = 0.0;
m_Input1->getValueAtTime(p_Args.time, shaperA, shaperB);
m_Input2->getValueAtTime(p_Args.time, inputA, inputB);
string exprLUT;
m_ExprLUT->getValue(exprLUT);
bool processLUT = !exprLUT.empty();
exprLUT = exprLUT.empty() ? "x" : exprLUT;
double temPix;
bool doLUT = processLUT;

exprtk::symbol_table<double> symbol_table;
symbol_table.add_variable("x",temPix);
MathProperties exprLUT_props = {kParamMathLUT, exprLUT, true};
exprtk::function_compositor<double> compositor(symbol_table);
compositor.add("lerp", " a*(c-b)+b;", "a","b","c");

MathProperties E = exprLUT_props;
exprtk::expression<double> expressionLUT;
expressionLUT.register_symbol_table(symbol_table);
exprtk::parser<double> parserLUT;
doLUT = parserLUT.compile(E.content,expressionLUT);

string PATH;
m_Path2->getValue(PATH);
string NAME;
m_Name2->getValue(NAME);

int cube = (int)m_Cube->getValueAtTime(p_Args.time);
int decimal = (int)m_Precision->getValueAtTime(p_Args.time);
int total = cube * cube * cube;
int lutsize = pow(2, m_Lutsize->getValueAtTime(p_Args.time));

OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".cube to " + PATH + " ?");
if (reply == OFX::Message::eMessageReplyYes) {
FILE * pFile;
pFile = fopen ((PATH + NAME + ".cube").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "# Resolve 3D LUT export\n" \
"\n");
if (Shaper > 0)
fprintf (pFile, "LUT_1D_SIZE %d\n" \
"LUT_1D_INPUT_RANGE %.*f %.*f\n" \
"\n", lutsize, decimal, shaperA, decimal, shaperB);
fprintf (pFile, "LUT_3D_SIZE %d\n" \
"LUT_3D_INPUT_RANGE %.*f %.*f\n" \
"\n", cube, decimal, inputA, decimal, inputB);
if (Shaper > 0){
for( int i = 0; i < lutsize; ++i ){
shaper = ((double)i / (lutsize - 1)) * (shaperB - shaperA) + shaperA;
if(Shaper == 1){
shaper = lin_to_ACEScc(shaper);
} else if(Shaper == 2){
shaper = lin_to_ACEScct(shaper);
} else {
temPix = shaper;
shaper = doLUT ? expressionLUT.value() : shaper;
}
fprintf (pFile, "%.*f %.*f %.*f\n", decimal, shaper, decimal, shaper, decimal, shaper);
}
fprintf (pFile, "\n");
}
for( int i = 0; i < total; ++i ) {
float R = fmod(i, cube) / (cube - 1) * (inputB - inputA) + inputA;
float G = fmod(floor(i / cube), cube) / (cube - 1) * (inputB - inputA) + inputA;
float B = fmod(floor(i / (cube * cube)), cube) / (cube - 1) * (inputB - inputA) + inputA;
float3 aces = make_float3(R, G, B);
if (_direction == 0) {
switch (_cscin){
case 0:
{}
break;
case 1:
{aces = ACEScc_to_ACES(aces);}
break;
case 2:
{aces = ACEScct_to_ACES(aces);}
break;
case 3:
{aces = ACEScg_to_ACES(aces);}
break;
case 4:
{aces = ACESproxy_to_ACES(aces);}
break;
case 5:
{aces = ADX_to_ACES(aces);}
break;
case 6:
{aces = ICpCt_to_ACES(aces);}
break;
case 7:
{aces = LogC_EI800_AWG_to_ACES(aces);}
break;
case 8:
{aces = Log3G10_RWG_to_ACES(aces);}
break;
case 9:
{aces = SLog3_SG3_to_ACES(aces);}
break;
case 10:
{aces = SLog3_SG3C_to_ACES(aces);}}
switch (_idt){
case 0:
{}
break;
case 1:
{aces = IDT_Alexa_v3_raw_EI800_CCT6500(aces);}
break;
case 2:
{aces = IDT_Panasonic_V35(aces);}
break;
case 3:
{aces = IDT_Canon_C100_A_D55(aces);}
break;
case 4:
{aces = IDT_Canon_C100_A_Tng(aces);}
break;
case 5:
{aces = IDT_Canon_C100mk2_A_D55(aces);}
break;
case 6:
{aces = IDT_Canon_C100mk2_A_Tng(aces);}
break;
case 7:
{aces = IDT_Canon_C300_A_D55(aces);}
break;
case 8:
{aces = IDT_Canon_C300_A_Tng(aces);}
break;
case 9:
{aces = IDT_Canon_C500_A_D55(aces);}
break;
case 10:
{aces = IDT_Canon_C500_A_Tng(aces);}
break;
case 11:
{aces = IDT_Canon_C500_B_D55(aces);}
break;
case 12:
{aces = IDT_Canon_C500_B_Tng(aces);}
break;
case 13:
{aces = IDT_Canon_C500_CinemaGamut_A_D55(aces);}
break;
case 14:
{aces = IDT_Canon_C500_CinemaGamut_A_Tng(aces);}
break;
case 15:
{aces = IDT_Canon_C500_DCI_P3_A_D55(aces);}
break;
case 16:
{aces = IDT_Canon_C500_DCI_P3_A_Tng(aces);}
break;
case 17:
{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_D55(aces);}
break;
case 18:
{aces = IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng(aces);}
break;
case 19:
{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55(aces);}
break;
case 20:
{aces = IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng(aces);}
break;
case 21:
{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55(aces);}
break;
case 22:
{aces = IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng(aces);}
break;
case 23:
{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55(aces);}
break;
case 24:
{aces = IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng(aces);}
break;
case 25:
{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55(aces);}
break;
case 26:
{aces = IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng(aces);}
break;
case 27:
{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55(aces);}
break;
case 28:
{aces = IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng(aces);}
break;
case 29:
{aces = IDT_Sony_SLog1_SGamut(aces);}
break;
case 30:
{aces = IDT_Sony_SLog2_SGamut_Daylight(aces);}
break;
case 31:
{aces = IDT_Sony_SLog2_SGamut_Tungsten(aces);}
break;
case 32:
{aces = IDT_Sony_Venice_SGamut3(aces);}
break;
case 33:
{aces = IDT_Sony_Venice_SGamut3Cine(aces);}
break;
case 34:
{aces = IDT_Rec709(aces);}
break;
case 35:
{aces = IDT_sRGB(aces);}
}
if (_exposure != 0.0f) {
aces.x *= exp2f(_exposure);
aces.y *= exp2f(_exposure);
aces.z *= exp2f(_exposure);
}
switch (_lmt)
{
case 0:
{}
break;
case 1:
{
if(_lmtscale[0] != 1.0f)
aces = scale_C(aces, _lmtscale[0]);
if(!(_lmtscale[1] == 1.0f && _lmtscale[2] == 0.0f && _lmtscale[3] == 1.0f)) {
float3 SLOPE = {_lmtscale[1], _lmtscale[1], _lmtscale[1]};
float3 OFFSET = {_lmtscale[2], _lmtscale[2], _lmtscale[2]};
float3 POWER = {_lmtscale[3], _lmtscale[3], _lmtscale[3]};
aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f);
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
{aces = LMT_BlueLightArtifactFix(aces);}
break;
case 3:
{aces = LMT_PFE(aces);}
break;
case 4:
{
if(_lmtscale[0] != 1.0f)
aces = scale_C(aces, _lmtscale[0]);
float3 SLOPE = {_lmtscale[1], _lmtscale[1], _lmtscale[1] * 0.94f};
float3 OFFSET = {_lmtscale[2], _lmtscale[2], _lmtscale[2] + 0.02f};
float3 POWER = {_lmtscale[3], _lmtscale[3], _lmtscale[3]};
aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, 1.0f);
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
case 5:
{aces = LMT_Bleach(aces);}
}
switch (_cscout){
case 0:
{}
break;
case 1:
{aces = ACES_to_ACEScc(aces);}
break;
case 2:
{aces = ACES_to_ACEScct(aces);}
break;
case 3:
{aces = ACES_to_ACEScg(aces);}
break;
case 4:
{aces = ACES_to_ACESproxy(aces);}
break;
case 5:
{aces = ACES_to_ADX(aces);}
break;
case 6:
{aces = ACES_to_ICpCt(aces);}
break;
case 7:
{aces = ACES_to_LogC_EI800_AWG(aces);}
break;
case 8:
{aces = ACES_to_Log3G10_RWG(aces);}
break;
case 9:
{aces = ACES_to_SLog3_SG3(aces);}
break;
case 10:
{aces = ACES_to_SLog3_SG3C(aces);}
}
if (_rrt == 1 && _odt < 22)
aces = h_RRT(aces);
switch (_odt)
{
case 0:
{}
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
{aces = ODT_Rec709_100nits_dim(aces);}
break;
case 3:
{aces = ODT_Rec709_D60sim_100nits_dim(aces);}
break;
case 4:
{aces = ODT_sRGB_100nits_dim(aces);}
break;
case 5:
{aces = ODT_sRGB_D60sim_100nits_dim(aces);}
break;
case 6:
{aces = ODT_Rec2020_100nits_dim(aces);}
break;
case 7:
{aces = ODT_Rec2020_Rec709limited_100nits_dim(aces);}
break;
case 8:
{aces = ODT_Rec2020_P3D65limited_100nits_dim(aces);}
break;
case 9:
{aces = ODT_Rec2020_ST2084_1000nits(aces);}
break;
case 10:
{aces = ODT_P3DCI_48nits(aces);}
break;
case 11:
{aces = ODT_P3DCI_D60sim_48nits(aces);}
break;
case 12:
{aces = ODT_P3DCI_D65sim_48nits(aces);}
break;
case 13:
{aces = ODT_P3D60_48nits(aces);}
break;
case 14:
{aces = ODT_P3D65_48nits(aces);}
break;
case 15:
{aces = ODT_P3D65_D60sim_48nits(aces);}
break;
case 16:
{aces = ODT_P3D65_Rec709limited_48nits(aces);}
break;
case 17:
{aces = ODT_DCDM(aces);}
break;
case 18:
{aces = ODT_DCDM_P3D60limited(aces);}
break;
case 19:
{aces = ODT_DCDM_P3D65limited(aces);}
break;
case 20:
{aces = ODT_RGBmonitor_100nits_dim(aces);}
break;
case 21:
{aces = ODT_RGBmonitor_D60sim_100nits_dim(aces);}
break;
case 22:
{aces = RRTODT_Rec709_100nits_10nits_BT1886(aces);}
break;
case 23:
{aces = RRTODT_Rec709_100nits_10nits_sRGB(aces);}
break;
case 24:
{aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);}
break;
case 25:
{aces = RRTODT_P3D65_1000nits_15nits_ST2084(aces);}
break;
case 26:
{aces = RRTODT_P3D65_2000nits_15nits_ST2084(aces);}
break;
case 27:
{aces = RRTODT_P3D65_4000nits_15nits_ST2084(aces);}
break;
case 28:
{aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);}
break;
case 29:
{aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);}
break;
case 30:
{aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);}
break;
case 31:
{aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);}
}} else {
switch (_invodt)
{
case 0:
{}
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
{aces = InvODT_Rec709_100nits_dim(aces);}
break;
case 3:
{aces = InvODT_Rec709_D60sim_100nits_dim(aces);}
break;
case 4:
{aces = InvODT_sRGB_100nits_dim(aces);}
break;
case 5:
{aces = InvODT_sRGB_D60sim_100nits_dim(aces);}
break;
case 6:
{aces = InvODT_Rec2020_100nits_dim(aces);}
break;
case 7:
{aces = InvODT_Rec2020_ST2084_1000nits(aces);}
break;
case 8:
{aces = InvODT_P3DCI_48nits(aces);}
break;
case 9:
{aces = InvODT_P3DCI_D60sim_48nits(aces);}
break;
case 10:
{aces = InvODT_P3DCI_D65sim_48nits(aces);}
break;
case 11:
{aces = InvODT_P3D60_48nits(aces);}
break;
case 12:
{aces = InvODT_P3D65_48nits(aces);}
break;
case 13:
{aces = InvODT_P3D65_D60sim_48nits(aces);}
break;
case 14:
{aces = InvODT_DCDM(aces);}
break;
case 15:
{aces = InvODT_DCDM_P3D65limited(aces);}
break;
case 16:
{aces = InvODT_RGBmonitor_100nits_dim(aces);}
break;
case 17:
{aces = InvODT_RGBmonitor_D60sim_100nits_dim(aces);}
break;
case 18:
{aces = InvRRTODT_Rec709_100nits_10nits_BT1886(aces);}
break;
case 19:
{aces = InvRRTODT_Rec709_100nits_10nits_sRGB(aces);}
break;
case 20:
{aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);}
break;
case 21:
{aces = InvRRTODT_P3D65_1000nits_15nits_ST2084(aces);}
break;
case 22:
{aces = InvRRTODT_P3D65_2000nits_15nits_ST2084(aces);}
break;
case 23:
{aces = InvRRTODT_P3D65_4000nits_15nits_ST2084(aces);}
break;
case 24:
{aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);}
break;
case 25:
{aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);}
break;
case 26:
{aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);}
break;
case 27:
{aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);}
}
if(_invrrt == 1 && _invodt < 18)
aces = h_InvRRT(aces);
}
fprintf (pFile, "%.*f %.*f %.*f\n", decimal, aces.x, decimal, aces.y, decimal, aces.z);
}
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".cube to " + PATH + ". Check Permissions."));
}}}}

void ACESPlugin::setupAndProcess(ACES& p_ACES, const OFX::RenderArguments& p_Args)
{
std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

if (srcBitDepth != dstBitDepth || srcComponents != dstComponents) {
OFX::throwSuiteStatusException(kOfxStatErrValue);
}

int _direction;
m_Direction->getValueAtTime(p_Args.time, _direction);
int _cscin;
m_CSCIN->getValueAtTime(p_Args.time, _cscin);
int _idt;
m_IDT->getValueAtTime(p_Args.time, _idt);
int _lmt;
m_LMT->getValueAtTime(p_Args.time, _lmt);
int _cscout;
m_CSCOUT->getValueAtTime(p_Args.time, _cscout);
int _rrt;
m_RRT->getValueAtTime(p_Args.time, _rrt);
int _invrrt;
m_InvRRT->getValueAtTime(p_Args.time, _invrrt);
int _odt;
m_ODT->getValueAtTime(p_Args.time, _odt);
int _invodt;
m_InvODT->getValueAtTime(p_Args.time, _invodt);
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

int _display;
m_DISPLAY->getValueAtTime(p_Args.time, _display);
int _limit;
m_LIMIT->getValueAtTime(p_Args.time, _limit);
int _eotf;
m_EOTF->getValueAtTime(p_Args.time, _eotf);
int _surround;
m_SURROUND->getValueAtTime(p_Args.time, _surround);

int _switch[3];
bool stretch = m_STRETCH->getValueAtTime(p_Args.time);
_switch[0] = stretch ? 1 : 0;
bool d60sim = m_D60SIM->getValueAtTime(p_Args.time);
_switch[1] = d60sim ? 1 : 0;
bool legalrange = m_LEGALRANGE->getValueAtTime(p_Args.time);
_switch[2] = legalrange ? 1 : 0;

p_ACES.setDstImg(dst.get());
p_ACES.setSrcImg(src.get());

// Setup GPU Render arguments
p_ACES.setGPURenderArgs(p_Args);

p_ACES.setRenderWindow(p_Args.renderWindow);

p_ACES.setScales(_direction, _cscin, _idt, _lmt, _cscout, _rrt, _invrrt, _odt, 
_invodt, _exposure, _lmtscale, _lum, _display, _limit, _eotf, _surround, _switch);

p_ACES.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ACESPluginFactory::ACESPluginFactory()
: PluginFactoryHelper<ACESPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}

void ACESPluginFactory::describe(ImageEffectDescriptor& p_Desc)
{
p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
p_Desc.setPluginGrouping(kPluginGrouping);
p_Desc.setPluginDescription(kPluginDescription);

p_Desc.addSupportedContext(eContextFilter);
p_Desc.addSupportedContext(eContextGeneral);
p_Desc.addSupportedBitDepth(eBitDepthFloat);

p_Desc.setSingleInstance(false);
p_Desc.setHostFrameThreading(false);
p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
p_Desc.setSupportsTiles(kSupportsTiles);
p_Desc.setTemporalClipAccess(false);
p_Desc.setRenderTwiceAlways(false);
p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

//p_Desc.setSupportsOpenCLRender(true);
p_Desc.setSupportsCudaRender(true);
//#ifdef __APPLE__
//p_Desc.setSupportsMetalRender(true);
//#endif
}

static DoubleParamDescriptor* defineScaleParam(ImageEffectDescriptor& p_Desc, const std::string& p_Name, 
const std::string& p_Label, const std::string& p_Hint, GroupParamDescriptor* p_Parent) {
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
param->setParent(*p_Parent);
return param;
}

void ACESPluginFactory::describeInContext(ImageEffectDescriptor& p_Desc, ContextEnum)
{
ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
srcClip->addSupportedComponent(ePixelComponentRGBA);
srcClip->setTemporalClipAccess(false);
srcClip->setSupportsTiles(kSupportsTiles);
srcClip->setIsMask(false);

ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
dstClip->addSupportedComponent(ePixelComponentRGBA);
dstClip->addSupportedComponent(ePixelComponentAlpha);
dstClip->setSupportsTiles(kSupportsTiles);

PageParamDescriptor* page = p_Desc.definePageParam("Controls");

ChoiceParamDescriptor *choiceparam = p_Desc.defineChoiceParam(kParamInput);
choiceparam->setLabel(kParamInputLabel);
choiceparam->setHint(kParamInputHint);
assert(choiceparam->getNOptions() == (int)eInputStandard);
choiceparam->appendOption(kParamInputOptionStandard, kParamInputOptionStandardHint);
assert(choiceparam->getNOptions() == (int)eInputInverse);
choiceparam->appendOption(kParamInputOptionInverse, kParamInputOptionInverseHint);
choiceparam->setDefault( (int)eInputStandard );
choiceparam->setAnimates(false);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamCSCIN);
choiceparam->setLabel(kParamCSCINLabel);
choiceparam->setHint(kParamCSCINHint);
assert(choiceparam->getNOptions() == (int)eCSCINBypass);
choiceparam->appendOption(kParamCSCINOptionBypass, kParamCSCINOptionBypassHint);
assert(choiceparam->getNOptions() == (int)eCSCINACEScc);
choiceparam->appendOption(kParamCSCINOptionACEScc, kParamCSCINOptionACESccHint);
assert(choiceparam->getNOptions() == (int)eCSCINACEScct);
choiceparam->appendOption(kParamCSCINOptionACEScct, kParamCSCINOptionACEScctHint);
assert(choiceparam->getNOptions() == (int)eCSCINACEScg);
choiceparam->appendOption(kParamCSCINOptionACEScg, kParamCSCINOptionACEScgHint);
assert(choiceparam->getNOptions() == (int)eCSCINACESproxy);
choiceparam->appendOption(kParamCSCINOptionACESproxy, kParamCSCINOptionACESproxyHint);
assert(choiceparam->getNOptions() == (int)eCSCINADX);
choiceparam->appendOption(kParamCSCINOptionADX, kParamCSCINOptionADXHint);
assert(choiceparam->getNOptions() == (int)eCSCINICPCT);
choiceparam->appendOption(kParamCSCINOptionICPCT, kParamCSCINOptionICPCTHint);
assert(choiceparam->getNOptions() == (int)eCSCINLOGCAWG);
choiceparam->appendOption(kParamCSCINOptionLOGCAWG, kParamCSCINOptionLOGCAWGHint);
assert(choiceparam->getNOptions() == (int)eCSCINLOG3G10RWG);
choiceparam->appendOption(kParamCSCINOptionLOG3G10RWG, kParamCSCINOptionLOG3G10RWGHint);
assert(choiceparam->getNOptions() == (int)eCSCINSLOG3SG3);
choiceparam->appendOption(kParamCSCINOptionSLOG3SG3, kParamCSCINOptionSLOG3SG3Hint);
assert(choiceparam->getNOptions() == (int)eCSCINSLOG3SG3C);
choiceparam->appendOption(kParamCSCINOptionSLOG3SG3C, kParamCSCINOptionSLOG3SG3CHint);
choiceparam->setDefault( (int)eCSCINBypass );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(false);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamIDT);
choiceparam->setLabel(kParamIDTLabel);
choiceparam->setHint(kParamIDTHint);
assert(choiceparam->getNOptions() == (int)eIDTBypass);
choiceparam->appendOption(kParamIDTOptionBypass, kParamIDTOptionBypassHint);
assert(choiceparam->getNOptions() == (int)eIDTAlexaRaw800);
choiceparam->appendOption(kParamIDTOptionAlexaRaw800, kParamIDTOptionAlexaRaw800Hint);
assert(choiceparam->getNOptions() == (int)eIDTPanasonicV35);
choiceparam->appendOption(kParamIDTOptionPanasonicV35, kParamIDTOptionPanasonicV35Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC100AD55);
choiceparam->appendOption(kParamIDTOptionCanonC100AD55, kParamIDTOptionCanonC100AD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC100ATNG);
choiceparam->appendOption(kParamIDTOptionCanonC100ATNG, kParamIDTOptionCanonC100ATNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC100mk2AD55);
choiceparam->appendOption(kParamIDTOptionCanonC100mk2AD55, kParamIDTOptionCanonC100mk2AD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC100mk2ATNG);
choiceparam->appendOption(kParamIDTOptionCanonC100mk2ATNG, kParamIDTOptionCanonC100mk2ATNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300AD55);
choiceparam->appendOption(kParamIDTOptionCanonC300AD55, kParamIDTOptionCanonC300AD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300ATNG);
choiceparam->appendOption(kParamIDTOptionCanonC300ATNG, kParamIDTOptionCanonC300ATNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC500AD55);
choiceparam->appendOption(kParamIDTOptionCanonC500AD55, kParamIDTOptionCanonC500AD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC500ATNG);
choiceparam->appendOption(kParamIDTOptionCanonC500ATNG, kParamIDTOptionCanonC500ATNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC500BD55);
choiceparam->appendOption(kParamIDTOptionCanonC500BD55, kParamIDTOptionCanonC500BD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC500BTNG);
choiceparam->appendOption(kParamIDTOptionCanonC500BTNG, kParamIDTOptionCanonC500BTNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC500CinemaGamutAD55);
choiceparam->appendOption(kParamIDTOptionCanonC500CinemaGamutAD55, kParamIDTOptionCanonC500CinemaGamutAD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC500CinemaGamutATNG);
choiceparam->appendOption(kParamIDTOptionCanonC500CinemaGamutATNG, kParamIDTOptionCanonC500CinemaGamutATNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC500DCIP3AD55);
choiceparam->appendOption(kParamIDTOptionCanonC500DCIP3AD55, kParamIDTOptionCanonC500DCIP3AD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC500DCIP3ATNG);
choiceparam->appendOption(kParamIDTOptionCanonC500DCIP3ATNG, kParamIDTOptionCanonC500DCIP3ATNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLogBT2020DD55);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLogBT2020DD55, kParamIDTOptionCanonC300mk2CanonLogBT2020DD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLogBT2020DTNG);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLogBT2020DTNG, kParamIDTOptionCanonC300mk2CanonLogBT2020DTNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLogCinemaGamutCD55);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCD55, kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLogCinemaGamutCTNG);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCTNG, kParamIDTOptionCanonC300mk2CanonLogCinemaGamutCTNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLog2BT2020BD55);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLog2BT2020BD55, kParamIDTOptionCanonC300mk2CanonLog2BT2020BD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLog2BT2020BTNG);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLog2BT2020BTNG, kParamIDTOptionCanonC300mk2CanonLog2BT2020BTNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLog2CinemaGamutAD55);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutAD55, kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutAD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLog2CinemaGamutATNG);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutATNG, kParamIDTOptionCanonC300mk2CanonLog2CinemaGamutATNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLog3BT2020FD55);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLog3BT2020FD55, kParamIDTOptionCanonC300mk2CanonLog3BT2020FD55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLog3BT2020FTNG);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLog3BT2020FTNG, kParamIDTOptionCanonC300mk2CanonLog3BT2020FTNGHint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLog3CinemaGamutED55);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutED55, kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutED55Hint);
assert(choiceparam->getNOptions() == (int)eIDTCanonC300mk2CanonLog3CinemaGamutETNG);
choiceparam->appendOption(kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutETNG, kParamIDTOptionCanonC300mk2CanonLog3CinemaGamutETNGHint);
assert(choiceparam->getNOptions() == (int)eIDTSonySLog1SGamut);
choiceparam->appendOption(kParamIDTOptionSonySLog1SGamut, kParamIDTOptionSonySLog1SGamutHint);
assert(choiceparam->getNOptions() == (int)eIDTSonySLog2SGamutDaylight);
choiceparam->appendOption(kParamIDTOptionSonySLog2SGamutDaylight, kParamIDTOptionSonySLog2SGamutDaylightHint);
assert(choiceparam->getNOptions() == (int)eIDTSonySLog2SGamutTungsten);
choiceparam->appendOption(kParamIDTOptionSonySLog2SGamutTungsten, kParamIDTOptionSonySLog2SGamutTungstenHint);
assert(choiceparam->getNOptions() == (int)eIDTSonyVeniceSGamut3);
choiceparam->appendOption(kParamIDTOptionSonyVeniceSGamut3, kParamIDTOptionSonyVeniceSGamut3Hint);
assert(choiceparam->getNOptions() == (int)eIDTSonyVeniceSGamut3Cine);
choiceparam->appendOption(kParamIDTOptionSonyVeniceSGamut3Cine, kParamIDTOptionSonyVeniceSGamut3CineHint);
assert(choiceparam->getNOptions() == (int)eIDTRec709);
choiceparam->appendOption(kParamIDTOptionRec709, kParamIDTOptionRec709Hint);
assert(choiceparam->getNOptions() == (int)eIDTSRGB);
choiceparam->appendOption(kParamIDTOptionSRGB, kParamIDTOptionSRGBHint);
choiceparam->setDefault( (int)eIDTBypass );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(false);
page->addChild(*choiceparam);

DoubleParamDescriptor* param = defineScaleParam(p_Desc, "Exposure", "exposure", "scale in stops", 0);
param->setDefault(0.0f);
param->setRange(-10.0f, 10.0f);
param->setIncrement(0.001f);
param->setDisplayRange(-10.0f, 10.0f);
param->setIsSecretAndDisabled(false);
page->addChild(*param);

choiceparam = p_Desc.defineChoiceParam(kParamLMT);
choiceparam->setLabel(kParamLMTLabel);
choiceparam->setHint(kParamLMTHint);
assert(choiceparam->getNOptions() == (int)eLMTBypass);
choiceparam->appendOption(kParamLMTOptionBypass, kParamLMTOptionBypassHint);
assert(choiceparam->getNOptions() == (int)eLMTCustom);
choiceparam->appendOption(kParamLMTOptionCustom, kParamLMTOptionCustomHint);
assert(choiceparam->getNOptions() == (int)eLMTFix);
choiceparam->appendOption(kParamLMTOptionFix, kParamLMTOptionFixHint);
assert(choiceparam->getNOptions() == (int)eLMTPFE);
choiceparam->appendOption(kParamLMTOptionPFE, kParamLMTOptionPFEHint);
assert(choiceparam->getNOptions() == (int)eLMTPFECustom);
choiceparam->appendOption(kParamLMTOptionPFECustom, kParamLMTOptionPFECustomHint);
assert(choiceparam->getNOptions() == (int)eLMTBleach);
choiceparam->appendOption(kParamLMTOptionBleach, kParamLMTOptionBleachHint);
choiceparam->setDefault( (int)eLMTBypass );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(false);
page->addChild(*choiceparam);

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
param->setDefault(0.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Range1", "hue rotation range1", "hue range in degrees for rotating hue", 0);
param->setDefault(60.0);
param->setRange(0.0, 180.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Shift1", "hue rotation1", "shift hue range in degrees", 0);
param->setDefault(0.0);
param->setRange(-90.0, 90.0);
param->setIncrement(0.01);
param->setDisplayRange(-90.0, 90.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "RotateH2", "hue rotation hue2", "rotate hue at hue in degrees", 0);
param->setDefault(80.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Range2", "hue rotation range2", "hue range in degrees for rotating hue", 0);
param->setDefault(60.0);
param->setRange(0.0, 180.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Shift2", "hue rotation2", "shift hue range in degrees", 0);
param->setDefault(0.0);
param->setRange(-90.0, 90.0);
param->setIncrement(0.01);
param->setDisplayRange(-90.0, 90.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "RotateH3", "hue rotation hue3", "rotate hue at hue in degrees", 0);
param->setDefault(52.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Range3", "hue rotation range3", "hue range in degrees for rotating hue", 0);
param->setDefault(60.0);
param->setRange(0.0, 180.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Shift3", "hue rotation3", "shift hue range in degrees", 0);
param->setDefault(0.0);
param->setRange(-90.0, 90.0);
param->setIncrement(0.01);
param->setDisplayRange(-90.0, 90.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "HueCH1", "color scale hue1", "scale color at hue in degrees", 0);
param->setDefault(45.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "RangeCH1", "color scale range1", "hue range in degrees for scaling color", 0);
param->setDefault(60.0);
param->setRange(0.0, 180.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleCH1", "color scale1", "scale color at hue range", 0);
param->setDefault(1.0);
param->setRange(0.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "RotateH4", "hue rotation hue4", "rotate hue at hue in degrees", 0);
param->setDefault(190.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Range4", "hue rotation range4", "hue range in degrees for rotating hue", 0);
param->setDefault(60.0);
param->setRange(0.0, 180.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Shift4", "hue rotation4", "shift hue range in degrees", 0);
param->setDefault(0.0);
param->setRange(-90.0, 90.0);
param->setIncrement(0.01);
param->setDisplayRange(-90.0, 90.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "HueCH2", "color scale hue2", "scale color at hue in degrees", 0);
param->setDefault(240.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "RangeCH2", "color scale range2", "hue range in degrees for scaling color", 0);
param->setDefault(60.0);
param->setRange(0.0, 180.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleCH2", "color scale2", "scale color at hue range", 0);
param->setDefault(1.0);
param->setRange(0.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

choiceparam = p_Desc.defineChoiceParam(kParamCSCOUT);
choiceparam->setLabel(kParamCSCOUTLabel);
choiceparam->setHint(kParamCSCOUTHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTBypass);
choiceparam->appendOption(kParamCSCOUTOptionBypass, kParamCSCOUTOptionBypassHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTACEScc);
choiceparam->appendOption(kParamCSCOUTOptionACEScc, kParamCSCOUTOptionACESccHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTACEScct);
choiceparam->appendOption(kParamCSCOUTOptionACEScct, kParamCSCOUTOptionACEScctHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTACEScg);
choiceparam->appendOption(kParamCSCOUTOptionACEScg, kParamCSCOUTOptionACEScgHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTACESproxy);
choiceparam->appendOption(kParamCSCOUTOptionACESproxy, kParamCSCOUTOptionACESproxyHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTADX);
choiceparam->appendOption(kParamCSCOUTOptionADX, kParamCSCOUTOptionADXHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTICPCT);
choiceparam->appendOption(kParamCSCOUTOptionICPCT, kParamCSCOUTOptionICPCTHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTLOGCAWG);
choiceparam->appendOption(kParamCSCOUTOptionLOGCAWG, kParamCSCOUTOptionLOGCAWGHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTLOG3G10RWG);
choiceparam->appendOption(kParamCSCOUTOptionLOG3G10RWG, kParamCSCOUTOptionLOG3G10RWGHint);
assert(choiceparam->getNOptions() == (int)eCSCOUTSLOG3SG3);
choiceparam->appendOption(kParamCSCOUTOptionSLOG3SG3, kParamCSCOUTOptionSLOG3SG3Hint);
assert(choiceparam->getNOptions() == (int)eCSCOUTSLOG3SG3C);
choiceparam->appendOption(kParamCSCOUTOptionSLOG3SG3C, kParamCSCOUTOptionSLOG3SG3CHint);
choiceparam->setDefault( (int)eCSCOUTBypass );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(false);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamRRT);
choiceparam->setLabel(kParamRRTLabel);
choiceparam->setHint(kParamRRTHint);
assert(choiceparam->getNOptions() == (int)eRRTBypass);
choiceparam->appendOption(kParamRRTOptionBypass, kParamRRTOptionBypassHint);
assert(choiceparam->getNOptions() == (int)eRRTEnabled);
choiceparam->appendOption(kParamRRTOptionEnabled, kParamRRTOptionEnabledHint);
choiceparam->setDefault( (int)eRRTBypass );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(false);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamODT);
choiceparam->setLabel(kParamODTLabel);
choiceparam->setHint(kParamODTHint);
assert(choiceparam->getNOptions() == (int)eODTBypass);
choiceparam->appendOption(kParamODTOptionBypass, kParamODTOptionBypassHint);
assert(choiceparam->getNOptions() == (int)eODTCustom);
choiceparam->appendOption(kParamODTOptionCustom, kParamODTOptionCustomHint);
assert(choiceparam->getNOptions() == (int)eODTRec709_100dim);
choiceparam->appendOption(kParamODTOptionRec709_100dim, kParamODTOptionRec709_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTRec709_D60sim_100dim);
choiceparam->appendOption(kParamODTOptionRec709_D60sim_100dim, kParamODTOptionRec709_D60sim_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTSRGB_100dim);
choiceparam->appendOption(kParamODTOptionSRGB_100dim, kParamODTOptionSRGB_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTSRGB_D60sim_100dim);
choiceparam->appendOption(kParamODTOptionSRGB_D60sim_100dim, kParamODTOptionSRGB_D60sim_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTRec2020_100dim);
choiceparam->appendOption(kParamODTOptionRec2020_100dim, kParamODTOptionRec2020_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTRec2020_Rec709limited_100dim);
choiceparam->appendOption(kParamODTOptionRec2020_Rec709limited_100dim, kParamODTOptionRec2020_Rec709limited_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTRec2020_P3D65limited_100dim);
choiceparam->appendOption(kParamODTOptionRec2020_P3D65limited_100dim, kParamODTOptionRec2020_P3D65limited_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTRec2020_ST2084_1000);
choiceparam->appendOption(kParamODTOptionRec2020_ST2084_1000, kParamODTOptionRec2020_ST2084_1000Hint);
assert(choiceparam->getNOptions() == (int)eODTP3DCI_48);
choiceparam->appendOption(kParamODTOptionP3DCI_48, kParamODTOptionP3DCI_48Hint);
assert(choiceparam->getNOptions() == (int)eODTP3DCI_D60sim_48);
choiceparam->appendOption(kParamODTOptionP3DCI_D60sim_48, kParamODTOptionP3DCI_D60sim_48Hint);
assert(choiceparam->getNOptions() == (int)eODTP3DCI_D65sim_48);
choiceparam->appendOption(kParamODTOptionP3DCI_D65sim_48, kParamODTOptionP3DCI_D65sim_48Hint);
assert(choiceparam->getNOptions() == (int)eODTP3D60_48);
choiceparam->appendOption(kParamODTOptionP3D60_48, kParamODTOptionP3D60_48Hint);
assert(choiceparam->getNOptions() == (int)eODTP3D65_48);
choiceparam->appendOption(kParamODTOptionP3D65_48, kParamODTOptionP3D65_48Hint);
assert(choiceparam->getNOptions() == (int)eODTP3D65_D60sim_48);
choiceparam->appendOption(kParamODTOptionP3D65_D60sim_48, kParamODTOptionP3D65_D60sim_48Hint);
assert(choiceparam->getNOptions() == (int)eODTP3D65_Rec709limited_48);
choiceparam->appendOption(kParamODTOptionP3D65_Rec709limited_48, kParamODTOptionP3D65_Rec709limited_48Hint);
assert(choiceparam->getNOptions() == (int)eODTDCDM);
choiceparam->appendOption(kParamODTOptionDCDM, kParamODTOptionDCDMHint);
assert(choiceparam->getNOptions() == (int)eODTDCDM_P3D60limited);
choiceparam->appendOption(kParamODTOptionDCDM_P3D60limited, kParamODTOptionDCDM_P3D60limitedHint);
assert(choiceparam->getNOptions() == (int)eODTDCDM_P3D65limited);
choiceparam->appendOption(kParamODTOptionDCDM_P3D65limited, kParamODTOptionDCDM_P3D65limitedHint);
assert(choiceparam->getNOptions() == (int)eODTRGBmonitor_100dim);
choiceparam->appendOption(kParamODTOptionRGBmonitor_100dim, kParamODTOptionRGBmonitor_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTRGBmonitor_D60sim_100dim);
choiceparam->appendOption(kParamODTOptionRGBmonitor_D60sim_100dim, kParamODTOptionRGBmonitor_D60sim_100dimHint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_Rec709_100nits_10nits_BT1886);
choiceparam->appendOption(kParamODTOptionRRTODT_Rec709_100nits_10nits_BT1886, kParamODTOptionRRTODT_Rec709_100nits_10nits_BT1886Hint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_Rec709_100nits_10nits_sRGB);
choiceparam->appendOption(kParamODTOptionRRTODT_Rec709_100nits_10nits_sRGB, kParamODTOptionRRTODT_Rec709_100nits_10nits_sRGBHint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_P3D65_108nits_7_2nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084, kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_P3D65_1000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_P3D65_1000nits_15nits_ST2084, kParamODTOptionRRTODT_P3D65_1000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_P3D65_2000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_P3D65_2000nits_15nits_ST2084, kParamODTOptionRRTODT_P3D65_2000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_P3D65_4000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_P3D65_4000nits_15nits_ST2084, kParamODTOptionRRTODT_P3D65_4000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_Rec2020_1000nits_15nits_HLG);
choiceparam->appendOption(kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLG, kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_Rec2020_1000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084, kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_Rec2020_2000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084, kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eODTRRTODT_Rec2020_4000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084, kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint);
choiceparam->setDefault( (int)eODTBypass );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(false);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamInvODT);
choiceparam->setLabel(kParamInvODTLabel);
choiceparam->setHint(kParamInvODTHint);
assert(choiceparam->getNOptions() == (int)eInvODTBypass);
choiceparam->appendOption(kParamInvODTOptionBypass, kParamInvODTOptionBypassHint);
assert(choiceparam->getNOptions() == (int)eInvODTCustom);
choiceparam->appendOption(kParamInvODTOptionCustom, kParamInvODTOptionCustomHint);
assert(choiceparam->getNOptions() == (int)eInvODTRec709_100dim);
choiceparam->appendOption(kParamInvODTOptionRec709_100dim, kParamInvODTOptionRec709_100dimHint);
assert(choiceparam->getNOptions() == (int)eInvODTRec709_D60sim_100dim);
choiceparam->appendOption(kParamInvODTOptionRec709_D60sim_100dim, kParamInvODTOptionRec709_D60sim_100dimHint);
assert(choiceparam->getNOptions() == (int)eInvODTSRGB_100dim);
choiceparam->appendOption(kParamInvODTOptionSRGB_100dim, kParamInvODTOptionSRGB_100dimHint);
assert(choiceparam->getNOptions() == (int)eInvODTSRGB_D60sim_100dim);
choiceparam->appendOption(kParamInvODTOptionSRGB_D60sim_100dim, kParamInvODTOptionSRGB_D60sim_100dimHint);
assert(choiceparam->getNOptions() == (int)eInvODTRec2020_100dim);
choiceparam->appendOption(kParamInvODTOptionRec2020_100dim, kParamInvODTOptionRec2020_100dimHint);
assert(choiceparam->getNOptions() == (int)eInvODTRec2020_ST2084_1000);
choiceparam->appendOption(kParamInvODTOptionRec2020_ST2084_1000, kParamInvODTOptionRec2020_ST2084_1000Hint);
assert(choiceparam->getNOptions() == (int)eInvODTP3DCI_48);
choiceparam->appendOption(kParamInvODTOptionP3DCI_48, kParamInvODTOptionP3DCI_48Hint);
assert(choiceparam->getNOptions() == (int)eInvODTP3DCI_D60sim_48);
choiceparam->appendOption(kParamInvODTOptionP3DCI_D60sim_48, kParamInvODTOptionP3DCI_D60sim_48Hint);
assert(choiceparam->getNOptions() == (int)eInvODTP3DCI_D65sim_48);
choiceparam->appendOption(kParamInvODTOptionP3DCI_D65sim_48, kParamInvODTOptionP3DCI_D65sim_48Hint);
assert(choiceparam->getNOptions() == (int)eInvODTP3D60_48);
choiceparam->appendOption(kParamInvODTOptionP3D60_48, kParamInvODTOptionP3D60_48Hint);
assert(choiceparam->getNOptions() == (int)eInvODTP3D65_48);
choiceparam->appendOption(kParamInvODTOptionP3D65_48, kParamInvODTOptionP3D65_48Hint);
assert(choiceparam->getNOptions() == (int)eInvODTP3D65_D60sim_48);
choiceparam->appendOption(kParamInvODTOptionP3D65_D60sim_48, kParamInvODTOptionP3D65_D60sim_48Hint);
assert(choiceparam->getNOptions() == (int)eInvODTDCDM);
choiceparam->appendOption(kParamInvODTOptionDCDM, kParamInvODTOptionDCDMHint);
assert(choiceparam->getNOptions() == (int)eInvODTDCDM_P3D65limited);
choiceparam->appendOption(kParamInvODTOptionDCDM_P3D65limited, kParamInvODTOptionDCDM_P3D65limitedHint);
assert(choiceparam->getNOptions() == (int)eInvODTRGBmonitor_100dim);
choiceparam->appendOption(kParamInvODTOptionRGBmonitor_100dim, kParamInvODTOptionRGBmonitor_100dimHint);
assert(choiceparam->getNOptions() == (int)eInvODTRGBmonitor_D60sim_100dim);
choiceparam->appendOption(kParamInvODTOptionRGBmonitor_D60sim_100dim, kParamInvODTOptionRGBmonitor_D60sim_100dimHint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_Rec709_100nits_10nits_BT1886);
choiceparam->appendOption(kParamInvODTOptionRRTODT_Rec709_100nits_10nits_BT1886, kParamInvODTOptionRRTODT_Rec709_100nits_10nits_BT1886Hint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_Rec709_100nits_10nits_sRGB);
choiceparam->appendOption(kParamInvODTOptionRRTODT_Rec709_100nits_10nits_sRGB, kParamInvODTOptionRRTODT_Rec709_100nits_10nits_sRGBHint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_P3D65_108nits_7_2nits_ST2084);
choiceparam->appendOption(kParamInvODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084, kParamInvODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_P3D65_1000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_P3D65_1000nits_15nits_ST2084, kParamODTOptionRRTODT_P3D65_1000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_P3D65_2000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_P3D65_2000nits_15nits_ST2084, kParamODTOptionRRTODT_P3D65_2000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_P3D65_4000nits_15nits_ST2084);
choiceparam->appendOption(kParamODTOptionRRTODT_P3D65_4000nits_15nits_ST2084, kParamODTOptionRRTODT_P3D65_4000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_Rec2020_1000nits_15nits_HLG);
choiceparam->appendOption(kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_HLG, kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_Rec2020_1000nits_15nits_ST2084);
choiceparam->appendOption(kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084, kParamInvODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_Rec2020_2000nits_15nits_ST2084);
choiceparam->appendOption(kParamInvODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084, kParamInvODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint);
assert(choiceparam->getNOptions() == (int)eInvODTRRTODT_Rec2020_4000nits_15nits_ST2084);
choiceparam->appendOption(kParamInvODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084, kParamInvODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint);
choiceparam->setDefault( (int)eInvODTBypass );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(true);
page->addChild(*choiceparam);

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

choiceparam = p_Desc.defineChoiceParam(kParamDISPLAY);
choiceparam->setLabel(kParamDISPLAYLabel);
choiceparam->setHint(kParamDISPLAYHint);
assert(choiceparam->getNOptions() == (int)eDISPLAYRec2020);
choiceparam->appendOption(kParamDISPLAYOptionRec2020, kParamDISPLAYOptionRec2020Hint);
assert(choiceparam->getNOptions() == (int)eDISPLAYP3D60);
choiceparam->appendOption(kParamDISPLAYOptionP3D60, kParamDISPLAYOptionP3D60Hint);
assert(choiceparam->getNOptions() == (int)eDISPLAYP3D65);
choiceparam->appendOption(kParamDISPLAYOptionP3D65, kParamDISPLAYOptionP3D65Hint);
assert(choiceparam->getNOptions() == (int)eDISPLAYP3DCI);
choiceparam->appendOption(kParamDISPLAYOptionP3DCI, kParamDISPLAYOptionP3DCIHint);
assert(choiceparam->getNOptions() == (int)eDISPLAYRec709);
choiceparam->appendOption(kParamDISPLAYOptionRec709, kParamDISPLAYOptionRec709Hint);
choiceparam->setDefault( (int)eDISPLAYRec2020 );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(true);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamLIMIT);
choiceparam->setLabel(kParamLIMITLabel);
choiceparam->setHint(kParamLIMITHint);
assert(choiceparam->getNOptions() == (int)eLIMITRec2020);
choiceparam->appendOption(kParamLIMITOptionRec2020, kParamLIMITOptionRec2020Hint);
assert(choiceparam->getNOptions() == (int)eLIMITP3D60);
choiceparam->appendOption(kParamLIMITOptionP3D60, kParamLIMITOptionP3D60Hint);
assert(choiceparam->getNOptions() == (int)eLIMITP3D65);
choiceparam->appendOption(kParamLIMITOptionP3D65, kParamLIMITOptionP3D65Hint);
assert(choiceparam->getNOptions() == (int)eLIMITP3DCI);
choiceparam->appendOption(kParamLIMITOptionP3DCI, kParamLIMITOptionP3DCIHint);
assert(choiceparam->getNOptions() == (int)eLIMITRec709);
choiceparam->appendOption(kParamLIMITOptionRec709, kParamLIMITOptionRec709Hint);
choiceparam->setDefault( (int)eLIMITRec2020 );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(true);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamEOTF);
choiceparam->setLabel(kParamEOTFLabel);
choiceparam->setHint(kParamEOTFHint);
assert(choiceparam->getNOptions() == (int)eEOTFST2084);
choiceparam->appendOption(kParamEOTFOptionST2084, kParamEOTFOptionST2084Hint);
assert(choiceparam->getNOptions() == (int)eEOTFBT1886);
choiceparam->appendOption(kParamEOTFOptionBT1886, kParamEOTFOptionBT1886Hint);
assert(choiceparam->getNOptions() == (int)eEOTFsRGB);
choiceparam->appendOption(kParamEOTFOptionsRGB, kParamEOTFOptionsRGBHint);
assert(choiceparam->getNOptions() == (int)eEOTFGAMMA26);
choiceparam->appendOption(kParamEOTFOptionGAMMA26, kParamEOTFOptionGAMMA26Hint);
assert(choiceparam->getNOptions() == (int)eEOTFLINEAR);
choiceparam->appendOption(kParamEOTFOptionLINEAR, kParamEOTFOptionLINEARHint);
assert(choiceparam->getNOptions() == (int)eEOTFHLG);
choiceparam->appendOption(kParamEOTFOptionHLG, kParamEOTFOptionHLGHint);
choiceparam->setDefault( (int)eEOTFST2084 );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(true);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamSURROUND);
choiceparam->setLabel(kParamSURROUNDLabel);
choiceparam->setHint(kParamSURROUNDHint);
assert(choiceparam->getNOptions() == (int)eSURROUNDDark);
choiceparam->appendOption(kParamSURROUNDOptionDark, kParamSURROUNDOptionDarkHint);
assert(choiceparam->getNOptions() == (int)eSURROUNDDim);
choiceparam->appendOption(kParamSURROUNDOptionDim, kParamSURROUNDOptionDimHint);
assert(choiceparam->getNOptions() == (int)eSURROUNDNormal);
choiceparam->appendOption(kParamSURROUNDOptionNormal, kParamSURROUNDOptionNormalHint);
choiceparam->setDefault( (int)eSURROUNDDim );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(true);
page->addChild(*choiceparam);

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

choiceparam = p_Desc.defineChoiceParam(kParamInvRRT);
choiceparam->setLabel(kParamInvRRTLabel);
choiceparam->setHint(kParamInvRRTHint);
assert(choiceparam->getNOptions() == (int)eInvRRTBypass);
choiceparam->appendOption(kParamInvRRTOptionBypass, kParamInvRRTOptionBypassHint);
assert(choiceparam->getNOptions() == (int)eInvRRTEnabled);
choiceparam->appendOption(kParamInvRRTOptionEnabled, kParamInvRRTOptionEnabledHint);
choiceparam->setDefault( (int)eInvRRTBypass );
choiceparam->setAnimates(false);
choiceparam->setIsSecretAndDisabled(true);
page->addChild(*choiceparam);

PushButtonParamDescriptor* pushparam = p_Desc.definePushButtonParam("info");
pushparam->setLabel("Info");
page->addChild(*pushparam);

GroupParamDescriptor* script = p_Desc.defineGroupParam("DCTL Export");
script->setOpen(false);
script->setHint("export DCTL");
if (page)
page->addChild(*script);

StringParamDescriptor* stringparam = p_Desc.defineStringParam("name");
stringparam->setLabel("Name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("ACESexport");
stringparam->setParent(*script);
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam("path");
stringparam->setLabel("Directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript);
stringparam->setFilePathExists(false);
stringparam->setParent(*script);
page->addChild(*stringparam);

pushparam = p_Desc.definePushButtonParam("button1");
pushparam->setLabel("Export DCTL");
pushparam->setHint("create DCTL version");
pushparam->setParent(*script);
page->addChild(*pushparam);
    
GroupParamDescriptor* lutexport = p_Desc.defineGroupParam("LUT Export");
lutexport->setOpen(false);
lutexport->setHint("export LUT");
if (page)
page->addChild(*lutexport);

stringparam = p_Desc.defineStringParam("name2");
stringparam->setLabel("Name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("ACESexport");
stringparam->setParent(*lutexport);
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam("path2");
stringparam->setLabel("Directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript2);
stringparam->setFilePathExists(false);
stringparam->setParent(*lutexport);
page->addChild(*stringparam);

choiceparam = p_Desc.defineChoiceParam(kParamSHAPER);
choiceparam->setLabel(kParamSHAPERLabel);
choiceparam->setHint(kParamSHAPERHint);
assert(choiceparam->getNOptions() == (int)eSHAPERBypass);
choiceparam->appendOption(kParamSHAPEROptionBypass, kParamSHAPEROptionBypassHint);
assert(choiceparam->getNOptions() == (int)eSHAPERACEScc);
choiceparam->appendOption(kParamSHAPEROptionACEScc, kParamSHAPEROptionACESccHint);
assert(choiceparam->getNOptions() == (int)eSHAPERACEScct);
choiceparam->appendOption(kParamSHAPEROptionACEScct, kParamSHAPEROptionACEScctHint);
assert(choiceparam->getNOptions() == (int)eSHAPERACEScustom);
choiceparam->appendOption(kParamSHAPEROptionACEScustom, kParamSHAPEROptionACEScustomHint);
choiceparam->setDefault( (int)eSHAPERBypass );
choiceparam->setAnimates(false);
choiceparam->setParent(*lutexport);
page->addChild(*choiceparam);

stringparam = p_Desc.defineStringParam(kParamMathLUT);
stringparam->setLabel(kParamMathLUTLabel);
stringparam->setHint(kParamMathLUTHint);
stringparam->setDefault("x");
stringparam->setAnimates(false);
stringparam->setIsSecretAndDisabled(true);
stringparam->setParent(*lutexport);
if (page)
page->addChild(*stringparam);

Double2DParamDescriptor* param2D = p_Desc.defineDouble2DParam("range1");
param2D->setLabel("1D Input Range");
param2D->setHint("set input range for LUT");
param2D->setDefault(0.0, 1.0);
param2D->setIsSecretAndDisabled(true);
param2D->setParent(*lutexport);
page->addChild(*param2D);

IntParamDescriptor* intparam = p_Desc.defineIntParam("lutsize");
intparam->setLabel("1D size");
intparam->setHint("1D lut size in bytes");
intparam->setDefault(12);
intparam->setRange(8, 14);
intparam->setDisplayRange(8, 14);
intparam->setIsSecretAndDisabled(true);
intparam->setParent(*lutexport);
page->addChild(*intparam);

param2D = p_Desc.defineDouble2DParam("range2");
param2D->setLabel("3D Input Range");
param2D->setHint("set input range for 3D LUT");
param2D->setDefault(0.0, 1.0);
param2D->setParent(*lutexport);
page->addChild(*param2D);

intparam = p_Desc.defineIntParam("cube");
intparam->setLabel("cube size");
intparam->setHint("3d lut cube size");
intparam->setDefault(33);
intparam->setRange(3, 129);
intparam->setDisplayRange(3, 129);
intparam->setParent(*lutexport);
page->addChild(*intparam);

intparam = p_Desc.defineIntParam("precision");
intparam->setLabel("precision");
intparam->setHint("number of decimal points");
intparam->setDefault(6);
intparam->setRange(1, 12);
intparam->setDisplayRange(3, 10);
intparam->setParent(*lutexport);
page->addChild(*intparam);

pushparam = p_Desc.definePushButtonParam("button2");
pushparam->setLabel("Export LUT");
pushparam->setHint("create 3D LUT");
pushparam->setParent(*lutexport);
page->addChild(*pushparam);
}

ImageEffect* ACESPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum)
{
return new ACESPlugin(p_Handle);
}

void Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static ACESPluginFactory ACESPlugin;
p_FactoryArray.push_back(&ACESPlugin);
}