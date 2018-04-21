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

#define kParamIDT "IDT"
#define kParamIDTLabel "IDT"
#define kParamIDTHint "IDT"
#define kParamIDTOptionBypass "Bypass"
#define kParamIDTOptionBypassHint "Bypass"
#define kParamIDTOptionAlexaLogC800 "Alexa LogC EI800"
#define kParamIDTOptionAlexaLogC800Hint "Alexa LogC EI800"
#define kParamIDTOptionAlexaRaw800 "Alexa Raw EI800"
#define kParamIDTOptionAlexaRaw800Hint "Alexa Raw EI800"
#define kParamIDTOptionADX10 "ADX10"
#define kParamIDTOptionADX10Hint "ADX10"
#define kParamIDTOptionADX16 "ADX16"
#define kParamIDTOptionADX16Hint "ADX16"


enum IDTEnum
{
    eIDTBypass,
    eIDTAlexaLogC800,
    eIDTAlexaRaw800,
    eIDTADX10,
    eIDTADX16,
    
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


enum LMTEnum
{
    eLMTBypass,
    eLMTBleach,
    eLMTPFE,
    
};

#define kParamODT "ODT"
#define kParamODTLabel "ODT"
#define kParamODTHint "RRT + selected ODT"
#define kParamODTOptionBypass "Bypass"
#define kParamODTOptionBypassHint "Bypass"
#define kParamODTOptionRec709_100dim "Rec709 100nits Dim"
#define kParamODTOptionRec709_100dimHint "Rec709 100nits Dim"
#define kParamODTOptionRec2020_100dim "Rec2020 100nits Dim"
#define kParamODTOptionRec2020_100dimHint "Rec2020 100nits Dim"
#define kParamODTOptionRec2020_ST2084_1000 "Rec2020 ST2084 1000nits"
#define kParamODTOptionRec2020_ST2084_1000Hint "Rec2020 ST2084 1000nits"
#define kParamODTOptionRGBmonitor_100dim "RGB monitor 100nits Dim"
#define kParamODTOptionRGBmonitor_100dimHint "RGB monitor 100nits Dim"


enum ODTEnum
{
    eODTBypass,
    eODTRec709_100dim,
    eODTRec2020_100dim,
    eODTRec2020_ST2084_1000,
    eODTRGBmonitor_100dim,
    
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
    void setScales(int p_IDT, int p_LMT, int p_ODT, float p_Exposure);

private:
    OFX::Image* _srcImg;
    int _idt;
    int _lmt;
    int _odt;
    float _exposure;
};

ACES::ACES(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_IDT, int p_LMT, int p_ODT, float p_Exposure);

void ACES::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(input, output, width, height, _idt, _lmt, _odt, _exposure);
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

void ACES::setScales(int p_IDT, int p_LMT, int p_ODT, float p_Exposure)
{
_idt = p_IDT;
_lmt = p_LMT;
_odt = p_ODT;
_exposure = p_Exposure;
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
    
    OFX::ChoiceParam* m_IDT;
    OFX::ChoiceParam* m_LMT;
	OFX::ChoiceParam* m_ODT;
	
	OFX::DoubleParam* m_Exposure;

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
    
    m_IDT = fetchChoiceParam(kParamIDT);
	m_LMT = fetchChoiceParam(kParamLMT);
	m_ODT = fetchChoiceParam(kParamODT);		
	m_Exposure = fetchDoubleParam("Exposure");

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
	
	int IDT_i;
    m_IDT->getValueAtTime(p_Args.time, IDT_i);
    IDTEnum IDT = (IDTEnum)IDT_i;
    
    int idt = IDT_i;
    
    int LMT_i;
    m_LMT->getValueAtTime(p_Args.time, LMT_i);
    LMTEnum LMT = (LMTEnum)LMT_i;
    
    int lmt = LMT_i;
    
    int ODT_i;
    m_ODT->getValueAtTime(p_Args.time, ODT_i);
    ODTEnum ODT = (ODTEnum)ODT_i;
    
    int odt = ODT_i;
    
    float exposure = m_Exposure->getValueAtTime(p_Args.time);
    	
    // Set the images
    p_ACES.setDstImg(dst.get());
    p_ACES.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_ACES.setGPURenderArgs(p_Args);

    // Set the render window
    p_ACES.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_ACES.setScales(idt, lmt, odt, exposure);

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
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamIDT);
	param->setLabel(kParamIDTLabel);
	param->setHint(kParamIDTHint);
	assert(param->getNOptions() == (int)eIDTBypass);
	param->appendOption(kParamIDTOptionBypass, kParamIDTOptionBypassHint);
	assert(param->getNOptions() == (int)eIDTAlexaLogC800);
	param->appendOption(kParamIDTOptionAlexaLogC800, kParamIDTOptionAlexaLogC800Hint);
	assert(param->getNOptions() == (int)eIDTAlexaRaw800);
	param->appendOption(kParamIDTOptionAlexaRaw800, kParamIDTOptionAlexaRaw800Hint);
	assert(param->getNOptions() == (int)eIDTADX10);
	param->appendOption(kParamIDTOptionADX10, kParamIDTOptionADX10Hint);
	assert(param->getNOptions() == (int)eIDTADX16);
	param->appendOption(kParamIDTOptionADX16, kParamIDTOptionADX16Hint);
	param->setDefault( (int)eIDTBypass );
	param->setAnimates(false);
    page->addChild(*param);
	}
	
	DoubleParamDescriptor* param = defineScaleParam(p_Desc, "Exposure", "exposure", "scale in stops", 0);
	param->setDefault(0.0f);
	param->setRange(-10.0f, 10.0f);
	param->setIncrement(0.001f);
	param->setDisplayRange(-10.0f, 10.0f);
	page->addChild(*param);
	
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
	param->setDefault( (int)eLMTBypass );
	param->setAnimates(false);
    page->addChild(*param);
	}
	
	{
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamODT);
	param->setLabel(kParamODTLabel);
	param->setHint(kParamODTHint);
	assert(param->getNOptions() == (int)eODTBypass);
	param->appendOption(kParamODTOptionBypass, kParamODTOptionBypassHint);
	assert(param->getNOptions() == (int)eODTRec709_100dim);
	param->appendOption(kParamODTOptionRec709_100dim, kParamODTOptionRec709_100dimHint);
	assert(param->getNOptions() == (int)eODTRec2020_100dim);
	param->appendOption(kParamODTOptionRec2020_100dim, kParamODTOptionRec2020_100dimHint);
	assert(param->getNOptions() == (int)eODTRec2020_ST2084_1000);
	param->appendOption(kParamODTOptionRec2020_ST2084_1000, kParamODTOptionRec2020_ST2084_1000Hint);
	assert(param->getNOptions() == (int)eODTRGBmonitor_100dim);
	param->appendOption(kParamODTOptionRGBmonitor_100dim, kParamODTOptionRGBmonitor_100dimHint);
	param->setDefault( (int)eODTBypass );
	param->setAnimates(false);
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
