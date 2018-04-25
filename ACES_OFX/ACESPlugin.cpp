#include "ACESPlugin.h"

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
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#else
#define kPluginScript "/home/resolve/LUT"
#endif

#define kPluginName "ACES"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"ACES"

#define kPluginIdentifier "OpenFX.Yo.ACES"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

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
#define kParamLMTOptionBleach "Bleach Bypass"
#define kParamLMTOptionBleachHint "Bleach Bypass"
#define kParamLMTOptionPFE "PFE"
#define kParamLMTOptionPFEHint "Print Film Emulation"
#define kParamLMTOptionCustom "Custom"
#define kParamLMTOptionCustomHint "Custom LMT"


enum LMTEnum
{
    eLMTBypass,
    eLMTBleach,
    eLMTPFE,
    eLMTCustom,
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
#define kParamODTOptionRec2020_100dim "Rec2020 100nits Dim"
#define kParamODTOptionRec2020_100dimHint "Rec.2020 100nits Dim"
#define kParamODTOptionRec2020_ST2084_1000 "Rec2020 ST2084 1000nits"
#define kParamODTOptionRec2020_ST2084_1000Hint "Rec.2020 ST2084 1000nits"
#define kParamODTOptionRGBmonitor_100dim "RGB monitor 100nits Dim"
#define kParamODTOptionRGBmonitor_100dimHint "RGB monitor 100nits Dim"
#define kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084 "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamODTOptionRRTODT_P3D65_108nits_7_2nits_ST2084Hint "RRTODT P3D65 108nits 7.2nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLG "RRTODT Rec2020 1000nits 15nits HLG"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_HLGHint "RRTODT Rec2020 1000nits 15nits HLG"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084 "RRTODT Rec2020 1000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_1000nits_15nits_ST2084Hint "RRTODT Rec2020 1000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084 "RRTODT Rec2020 2000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_2000nits_15nits_ST2084Hint "RRTODT Rec2020 2000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084 "RRTODT Rec2020 4000nits 15nits ST2084"
#define kParamODTOptionRRTODT_Rec2020_4000nits_15nits_ST2084Hint "RRTODT Rec2020 4000nits 15nits ST2084"


enum ODTEnum
{
    eODTBypass,
    eODTCustom,
    eODTACEScc,
    eODTACEScct,
    eODTRec709_100dim,
    eODTRec2020_100dim,
    eODTRec2020_ST2084_1000,
    eODTRGBmonitor_100dim,
    eODTRRTODT_P3D65_108nits_7_2nits_ST2084,
    eODTRRTODT_Rec2020_1000nits_15nits_HLG,
    eODTRRTODT_Rec2020_1000nits_15nits_ST2084,
    eODTRRTODT_Rec2020_2000nits_15nits_ST2084,
    eODTRRTODT_Rec2020_4000nits_15nits_ST2084,
};

//aces = RRTODT_P3D65_108nits_7_2nits_ST2084(aces);
//aces = RRTODT_Rec2020_1000nits_15nits_HLG(aces);
//aces = RRTODT_Rec2020_1000nits_15nits_ST2084(aces);
//aces = RRTODT_Rec2020_2000nits_15nits_ST2084(aces);
//aces = RRTODT_Rec2020_4000nits_15nits_ST2084(aces);

//aces = InvRRTODT_P3D65_108nits_7_2nits_ST2084(aces);
//aces = InvRRTODT_Rec2020_1000nits_15nits_HLG(aces);
//aces = InvRRTODT_Rec2020_1000nits_15nits_ST2084(aces);
//aces = InvRRTODT_Rec2020_2000nits_15nits_ST2084(aces);
//aces = InvRRTODT_Rec2020_4000nits_15nits_ST2084(aces);


#define kParamInvODT "InvODT"
#define kParamInvODTLabel "Inverse ODT"
#define kParamInvODTHint "Inverse ODT"
#define kParamInvODTOptionBypass "Bypass"
#define kParamInvODTOptionBypassHint "Bypass"
#define kParamInvODTOptionCustom "Custom"
#define kParamInvODTOptionCustomHint "Custom Inverse ODT"
#define kParamInvODTOptionRec709_100dim "Rec709 100nits Dim"
#define kParamInvODTOptionRec709_100dimHint "Rec.709 100nits Dim"
#define kParamInvODTOptionRec2020_100dim "Rec2020 100nits Dim"
#define kParamInvODTOptionRec2020_100dimHint "Rec.2020 100nits Dim"
#define kParamInvODTOptionRec2020_ST2084_1000 "Rec2020 ST2084 1000nits"
#define kParamInvODTOptionRec2020_ST2084_1000Hint "Rec.2020 ST2084 1000nits"
#define kParamInvODTOptionRGBmonitor_100dim "RGB monitor 100nits Dim"
#define kParamInvODTOptionRGBmonitor_100dimHint "RGB monitor 100nits Dim"
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

enum InvODTEnum
{
    eInvODTBypass,
    eInvODTCustom,
    eInvODTRec709_100dim,
    eInvODTRec2020_100dim,
    eInvODTRec2020_ST2084_1000,
    eInvODTRGBmonitor_100dim,
    eInvODTRRTODT_P3D65_108nits_7_2nits_ST2084,
    eInvODTRRTODT_Rec2020_1000nits_15nits_HLG,
    eInvODTRRTODT_Rec2020_1000nits_15nits_ST2084,
    eInvODTRRTODT_Rec2020_2000nits_15nits_ST2084,
    eInvODTRRTODT_Rec2020_4000nits_15nits_ST2084,
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

    virtual void processImagesCUDA();
    //virtual void processImagesOpenCL();
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
    float _lmtscale[13];
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

extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Direction, int p_IDT, int p_ACESIN, int p_LMT, int p_ACESOUT, int p_RRT, int p_InvRRT, 
int p_ODT, int p_InvODT, float p_Exposure, float* p_LMTScale, float *p_Lum, 
int p_DISPLAY, int p_LIMIT, int p_EOTF, int p_SURROUND, int *p_Switch);

void ACES::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(input, output, width, height, _direction, _idt, _acesin, _lmt, _acesout, _rrt, 
    _invrrt, _odt, _invodt, _exposure, _lmtscale, _lum, _display, _limit, _eotf, _surround, _switch);
}
/*
extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, const float* p_Input, float* p_Output);

void ACES::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, input, output);
}
*/
void ACES::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            // do we have a source image to scale up
            if (srcPix)
            {
   			    dstPix[0] = srcPix[0];
			    dstPix[1] = srcPix[1];
			    dstPix[2] = srcPix[2];
			    dstPix[3] = srcPix[3];
            }
            
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
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
	OFX::DoubleParam* m_Sat;
	OFX::DoubleParam* m_Gamma;
	OFX::DoubleParam* m_Pivot;
	OFX::DoubleParam* m_RotateH;
	OFX::DoubleParam* m_Range;
	OFX::DoubleParam* m_Shift;
	OFX::DoubleParam* m_HueCH;
	OFX::DoubleParam* m_RangeCH;
	OFX::DoubleParam* m_ScaleCH;
	
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

    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
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
	m_Sat = fetchDoubleParam("Sat");
	m_Gamma = fetchDoubleParam("Gamma");
	m_Pivot = fetchDoubleParam("Pivot");
	m_RotateH = fetchDoubleParam("RotateH");
	m_Range = fetchDoubleParam("Range");
	m_Shift = fetchDoubleParam("Shift");
	m_HueCH = fetchDoubleParam("HueCH");
	m_RangeCH = fetchDoubleParam("RangeCH");
	m_ScaleCH = fetchDoubleParam("ScaleCH");
	
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

    m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");

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
   
    
    
    if (m_SrcClip)
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
    
	bool custom = LMT_i == 3;
    
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
	m_Sat->setIsSecretAndDisabled(!forward || !custom);
	m_Gamma->setIsSecretAndDisabled(!forward || !custom);
	m_Pivot->setIsSecretAndDisabled(!forward || !custom);
	m_RotateH->setIsSecretAndDisabled(!forward || !custom);
	m_Range->setIsSecretAndDisabled(!forward || !custom);
	m_Shift->setIsSecretAndDisabled(!forward || !custom);
	m_HueCH->setIsSecretAndDisabled(!forward || !custom);
	m_RangeCH->setIsSecretAndDisabled(!forward || !custom);
	m_ScaleCH->setIsSecretAndDisabled(!forward || !custom);
	
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
    
	bool custom = LMT_i == 3;
	
	m_ScaleC->setIsSecretAndDisabled(!custom);
	m_Slope->setIsSecretAndDisabled(!custom);
	m_Offset->setIsSecretAndDisabled(!custom);
	m_Power->setIsSecretAndDisabled(!custom);
	m_Sat->setIsSecretAndDisabled(!custom);
	m_Gamma->setIsSecretAndDisabled(!custom);
	m_Pivot->setIsSecretAndDisabled(!custom);
	m_RotateH->setIsSecretAndDisabled(!custom);
	m_Range->setIsSecretAndDisabled(!custom);
	m_Shift->setIsSecretAndDisabled(!custom);
	m_HueCH->setIsSecretAndDisabled(!custom);
	m_RangeCH->setIsSecretAndDisabled(!custom);
	m_ScaleCH->setIsSecretAndDisabled(!custom);
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
	
	if(p_ParamName == "button1")
    {
       
	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// ACESPlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"	float r = p_R;\n" \
	"	float g = p_G;\n" \
	"	float b = p_B;\n" \
	"\n" \
	"	return make_float3(r, g, b);\n" \
	"}\n");
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}

	if(p_ParamName == "button2")
    {
    
	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".nk to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".nk").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, " Group {\n" \
	" name ACES\n" \
	" selected true\n" \
	"}\n" \
	" Input {\n" \
  	" name Input1\n" \
	" }\n" \
	" Output {\n" \
  	" name Output1\n" \
	" }\n" \
	"end_group\n");
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
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
    float _lmtscale[13];
    _lmtscale[0] = m_ScaleC->getValueAtTime(p_Args.time);
    _lmtscale[1] = m_Slope->getValueAtTime(p_Args.time);
    _lmtscale[2] = m_Offset->getValueAtTime(p_Args.time);
    _lmtscale[3] = m_Power->getValueAtTime(p_Args.time);
    _lmtscale[4] = m_Sat->getValueAtTime(p_Args.time);
    _lmtscale[5] = m_Gamma->getValueAtTime(p_Args.time);
    _lmtscale[6] = m_Pivot->getValueAtTime(p_Args.time);
    _lmtscale[7] = m_RotateH->getValueAtTime(p_Args.time);
    _lmtscale[8] = m_Range->getValueAtTime(p_Args.time);
    _lmtscale[9] = m_Shift->getValueAtTime(p_Args.time);
    _lmtscale[10] = m_HueCH->getValueAtTime(p_Args.time);
    _lmtscale[11] = m_RangeCH->getValueAtTime(p_Args.time);
    _lmtscale[12] = m_ScaleCH->getValueAtTime(p_Args.time);
    
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

    // Setup OpenCL and CUDA Render arguments
    p_ACES.setGPURenderArgs(p_Args);

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

    // Setup OpenCL and CUDA render capability flags
    p_Desc.setSupportsOpenCLRender(true);
    p_Desc.setSupportsCudaRender(true);
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
	assert(param->getNOptions() == (int)eLMTBleach);
	param->appendOption(kParamLMTOptionBleach, kParamLMTOptionBleachHint);
	assert(param->getNOptions() == (int)eLMTPFE);
	param->appendOption(kParamLMTOptionPFE, kParamLMTOptionPFEHint);
	assert(param->getNOptions() == (int)eLMTCustom);
	param->appendOption(kParamLMTOptionCustom, kParamLMTOptionCustomHint);
	param->setDefault( (int)eLMTBypass );
	param->setAnimates(false);
	param->setIsSecretAndDisabled(false);
    page->addChild(*param);
	}
	
	param = defineScaleParam(p_Desc, "ScaleC", "scale_C", "scaleC", 0);
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
    
    param = defineScaleParam(p_Desc, "Sat", "saturation", "saturation", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Gamma", "gamma_adjust_linear", "contrast", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Pivot", "pivot", "pivot point", 0);
    param->setDefault(0.18);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "RotateH", "rotate_H_in_Hue", "rotate_H_in_Hue", 0);
    param->setDefault(180.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Range", "range", "range", 0);
    param->setDefault(0.0);
    param->setRange(0.0, 180.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 180.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Shift", "shift", "shift", 0);
    param->setDefault(0.0);
    param->setRange(-90.0, 90.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-90.0, 90.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "HueCH", "scale_C_at_Hue", "scale_C_at_Hue", 0);
    param->setDefault(180.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "RangeCH", "rangeCH", "rangeCH", 0);
    param->setDefault(0.0);
    param->setRange(0.0, 180.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 180.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "ScaleCH", "scaleCH", "scaleCH", 0);
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
	assert(param->getNOptions() == (int)eODTRec2020_100dim);
	param->appendOption(kParamODTOptionRec2020_100dim, kParamODTOptionRec2020_100dimHint);
	assert(param->getNOptions() == (int)eODTRec2020_ST2084_1000);
	param->appendOption(kParamODTOptionRec2020_ST2084_1000, kParamODTOptionRec2020_ST2084_1000Hint);
	assert(param->getNOptions() == (int)eODTRGBmonitor_100dim);
	param->appendOption(kParamODTOptionRGBmonitor_100dim, kParamODTOptionRGBmonitor_100dimHint);
	
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
	assert(param->getNOptions() == (int)eInvODTRec2020_100dim);
	param->appendOption(kParamInvODTOptionRec2020_100dim, kParamInvODTOptionRec2020_100dimHint);
	assert(param->getNOptions() == (int)eInvODTRec2020_ST2084_1000);
	param->appendOption(kParamInvODTOptionRec2020_ST2084_1000, kParamInvODTOptionRec2020_ST2084_1000Hint);
	assert(param->getNOptions() == (int)eInvODTRGBmonitor_100dim);
	param->appendOption(kParamInvODTOptionRGBmonitor_100dim, kParamInvODTOptionRGBmonitor_100dimHint);
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
    GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
    script->setOpen(false);
    script->setHint("export DCTL and Nuke script");
      if (page) {
            page->addChild(*script);
            }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button1");
    param->setLabel("Export DCTL");
    param->setHint("create DCTL version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button2");
    param->setLabel("Export Nuke script");
    param->setHint("create NUKE version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("name");
	param->setLabel("Name");
	param->setHint("overwrites if the same");
	param->setDefault("ACES");
	param->setParent(*script);
	page->addChild(*param);
	}
	{
	StringParamDescriptor* param = p_Desc.defineStringParam("path");
	param->setLabel("Directory");
	param->setHint("make sure it's the absolute path");
	param->setStringType(eStringTypeFilePath);
	param->setDefault(kPluginScript);
	param->setFilePathExists(false);
	param->setParent(*script);
	page->addChild(*param);
	}
	}        
    
}

ImageEffect* ACESPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new ACESPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static ACESPluginFactory ACESPlugin;
    p_FactoryArray.push_back(&ACESPlugin);
}
