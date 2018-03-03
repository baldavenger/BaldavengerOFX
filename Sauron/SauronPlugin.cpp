#include "SauronPlugin.h"

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

#define kPluginName "Sauron"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Sauron"

#define kPluginIdentifier "OpenFX.Yo.Sauron"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

namespace {
    struct RGBValues {
        double r,g,b;
        RGBValues(double v) : r(v), g(v), b(v) {}
        RGBValues() : r(0), g(0), b(0) {}
    };
}

class Sauron : public OFX::ImageProcessor
{
public:
    explicit Sauron(OFX::ImageEffect& p_Instance);

    virtual void processImagesCUDA();
    //virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);
    
    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float p_Speed, float p_LengthA, float p_LengthB, float p_Horizontal, float p_Vertical, float p_Opacity, float p_PluginTime);

private:
    OFX::Image* _srcImg;
    float _speed;
    float _lengthA;
    float _lengthB;
    float _horizontal;
    float _vertical;
    float _opacity;
    float _pluginTime;
};

Sauron::Sauron(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
float p_Speed, float p_LengthA, float p_LengthB, float p_Horizontal, float p_Vertical, float p_Opacity, float p_PluginTime);

void Sauron::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(input, output, width, height, _speed, _lengthA, _lengthB, _horizontal, _vertical, _opacity, _pluginTime);
}
/*
extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, const float* p_Input, float* p_Output);

void Sauron::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, input, output);
}
*/
void Sauron::multiThreadProcessImages(OfxRectI p_ProcWindow)
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

void Sauron::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void Sauron::setScales(float p_Speed, float p_LengthA, float p_LengthB, float p_Horizontal, float p_Vertical, float p_Opacity, float p_PluginTime)
{
	_speed = p_Speed;
	_lengthA = p_LengthA;
	_lengthB = p_LengthB;
	_horizontal = p_Horizontal;
	_vertical = p_Vertical;
	_opacity = p_Opacity;
	_pluginTime = p_PluginTime;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class SauronPlugin : public OFX::ImageEffect
{
public:
    explicit SauronPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
     /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(Sauron &p_Sauron, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    
    OFX::DoubleParam* m_Speed;
    OFX::DoubleParam* m_LengthA;
    OFX::DoubleParam* m_LengthB;
    OFX::DoubleParam* m_Horizontal;
    OFX::DoubleParam* m_Vertical;
    OFX::DoubleParam* m_Opacity;

    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
};

SauronPlugin::SauronPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    
    m_Speed = fetchDoubleParam("Speed");
    m_LengthA = fetchDoubleParam("LengthA");
    m_LengthB = fetchDoubleParam("LengthB");
    m_Horizontal = fetchDoubleParam("Horizontal");
    m_Vertical = fetchDoubleParam("Vertical");
    m_Opacity = fetchDoubleParam("Opacity");

    m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");

}

void SauronPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        Sauron Sauron(*this);
        setupAndProcess(Sauron, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool SauronPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
   
    
    
    if (m_SrcClip)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void SauronPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
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
    	
	fprintf (pFile, "// SauronPlugin DCTL export\n" \
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
	" name Sauron\n" \
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


void SauronPlugin::setupAndProcess(Sauron& p_Sauron, const OFX::RenderArguments& p_Args)
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

	float _speed = m_Speed->getValueAtTime(p_Args.time);
	float _lengthA = m_LengthA->getValueAtTime(p_Args.time);
	float _lengthB = m_LengthB->getValueAtTime(p_Args.time);
	float _horizontal = m_Horizontal->getValueAtTime(p_Args.time);
	float _vertical = m_Vertical->getValueAtTime(p_Args.time);
	float _opacity = m_Opacity->getValueAtTime(p_Args.time);
    
    float _plugintime = 1.0f + (1.0f * p_Args.time);
    
    // Set the images
    p_Sauron.setDstImg(dst.get());
    p_Sauron.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_Sauron.setGPURenderArgs(p_Args);

    // Set the render window
    p_Sauron.setRenderWindow(p_Args.renderWindow);
    
    // Set the scales
    p_Sauron.setScales(_speed, _lengthA, _lengthB, _horizontal, _vertical, _opacity, _plugintime);

    // Call the base class process member, this will call the derived templated process code
    p_Sauron.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

SauronPluginFactory::SauronPluginFactory()
    : OFX::PluginFactoryHelper<SauronPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void SauronPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

void SauronPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
    
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam("Speed");
    param->setLabel("flame speed");
    param->setHint("flame speed");
    param->setDefault(10.0);
    param->setRange(-100.0, 100.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-100.0, 100.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("LengthA");
    param->setLabel("width");
    param->setHint("width");
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("LengthB");
    param->setLabel("depth");
    param->setHint("depth");
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Horizontal");
    param->setLabel("horizontal");
    param->setHint("horizontal");
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Vertical");
    param->setLabel("vertical");
    param->setHint("vertical");
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Opacity");
    param->setLabel("strength");
    param->setHint("strength");
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
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
	param->setDefault("Sauron");
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

ImageEffect* SauronPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new SauronPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static SauronPluginFactory SauronPlugin;
    p_FactoryArray.push_back(&SauronPlugin);
}
