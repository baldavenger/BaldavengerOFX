#include "FrequencyPlugin.h"

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
#define kPluginScript2 "/Library/Application Support/FilmLight/shaders"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#define kPluginScript2 "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#else
#define kPluginScript "/opt/resolve/LUT"
#define kPluginScript2 "/opt/resolve/LUT"
#endif

#define kPluginName "Frequency Separation"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Frequency Separation"

#define kPluginIdentifier "OpenFX.Yo.Frequency"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kParamFrequency "ColourSpace"
#define kParamFrequencyLabel "colour space"
#define kParamFrequencyHint "colour space"
#define kParamFrequencyOptionRGB "RGB"
#define kParamFrequencyOptionRGBHint "no conversion"
#define kParamFrequencyOptionYUV "YUV"
#define kParamFrequencyOptionYUVHint "Rec.709 to Y'UV"
#define kParamFrequencyOptionLAB "LAB"
#define kParamFrequencyOptionLABHint "Rec.709 to LAB"

enum FrequencyEnum
{
    eFrequencyRGB,
    eFrequencyYUV,
    eFrequencyLAB,
};

////////////////////////////////////////////////////////////////////////////////

namespace {
    struct RGBValues {
        double r,g,b;
        RGBValues(double v) : r(v), g(v), b(v) {}
        RGBValues() : r(0), g(0), b(0) {}
    };
}

class Frequency : public OFX::ImageProcessor
{
public:
    explicit Frequency(OFX::ImageEffect& p_Instance);

    //virtual void processImagesCUDA();
    //virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);
    
    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(int p_Space, float* p_Blur, float* p_Sharpen, int* p_Switch);

private:
    OFX::Image* _srcImg;
    int _space;
    float _blur[6];
    float _sharpen[3];
    int _switch[3];
};

Frequency::Frequency(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}
/*
extern void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Space, float* p_Blur, float* p_Sharpen, int* p_Switch);

void Frequency::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(input, output, width, height, _space, _blur, _sharpen, _switch);
}

extern void RunOpenCLKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Space, float* p_Blur, float* p_Sharpen, int* p_Switch);

void Frequency::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, input, output, width, height, _space, _blur, _sharpen, _switch);
}
*/
void Frequency::multiThreadProcessImages(OfxRectI p_ProcWindow)
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

void Frequency::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void Frequency::setScales(int p_Space, float* p_Blur, float* p_Sharpen, int* p_Switch)
{
   _space = p_Space;
   _blur[0] = p_Blur[0];
   _blur[1] = p_Blur[1];
   _blur[2] = p_Blur[2];
   _blur[3] = p_Blur[3];
   _blur[4] = p_Blur[4];
   _blur[5] = p_Blur[5];
   _sharpen[0] = p_Sharpen[0];
   _sharpen[1] = p_Sharpen[1];
   _sharpen[2] = p_Sharpen[2];
   _switch[0] = p_Switch[0];
   _switch[1] = p_Switch[1];
   _switch[2] = p_Switch[2];
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class FrequencyPlugin : public OFX::ImageEffect
{
public:
    explicit FrequencyPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
     /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(Frequency &p_Frequency, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    
    OFX::ChoiceParam* m_Space;
    OFX::DoubleParam* m_Blur1;
    OFX::DoubleParam* m_Blur2;
    OFX::DoubleParam* m_Blur3;
    OFX::DoubleParam* m_Sharpen1;
    OFX::DoubleParam* m_Sharpen2;
    OFX::DoubleParam* m_Sharpen3;
    OFX::DoubleParam* m_Blur4;
    OFX::DoubleParam* m_Blur5;
    OFX::DoubleParam* m_Blur6;
    OFX::BooleanParam* m_Gang1;
    OFX::BooleanParam* m_Gang2;
    OFX::BooleanParam* m_Display;
    
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Path2;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
	OFX::PushButtonParam* m_Button3;
};

FrequencyPlugin::FrequencyPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    
    m_Space = fetchChoiceParam(kParamFrequency);
    m_Blur1 = fetchDoubleParam("Blur1");
    m_Blur2 = fetchDoubleParam("Blur2");
    m_Blur3 = fetchDoubleParam("Blur3");
    m_Sharpen1 = fetchDoubleParam("Sharpen1");
    m_Sharpen2 = fetchDoubleParam("Sharpen2");
    m_Sharpen3 = fetchDoubleParam("Sharpen3");
    m_Blur4 = fetchDoubleParam("Blur4");
    m_Blur5 = fetchDoubleParam("Blur5");
    m_Blur6 = fetchDoubleParam("Blur6");
    
    m_Gang1 = fetchBooleanParam("Gang1");
    m_Gang2 = fetchBooleanParam("Gang2");
    m_Display = fetchBooleanParam("Display");

    m_Path = fetchStringParam("Path");
    m_Path2 = fetchStringParam("Path2");
	m_Name = fetchStringParam("Name");
	m_Info = fetchPushButtonParam("Info");
	m_Button1 = fetchPushButtonParam("Button1");
	m_Button2 = fetchPushButtonParam("Button2");
	m_Button3 = fetchPushButtonParam("Button3");

}

void FrequencyPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        Frequency Frequency(*this);
        setupAndProcess(Frequency, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool FrequencyPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
   
    
    
    if (m_SrcClip)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void FrequencyPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    if(p_ParamName == kParamFrequency)
    {
    int space_i;
    m_Space->getValueAtTime(p_Args.time, space_i);
    FrequencyEnum FrequencyFilter = (FrequencyEnum)space_i;
    
    bool RGB = space_i == 0;
    bool YUV = space_i == 1;
    bool LAB = space_i == 2;
    
    bool gang1 = m_Gang1->getValueAtTime(p_Args.time);
	bool gang2 = m_Gang2->getValueAtTime(p_Args.time);
	bool display = m_Display->getValueAtTime(p_Args.time);
	
	
	if(RGB){
	m_Gang1->setIsSecretAndDisabled(!RGB);
	m_Gang2->setIsSecretAndDisabled(RGB);
	m_Gang1->setValue(true);
	m_Blur1->setLabel("threshold RGB");
	m_Blur2->setLabel("threshold G");
	m_Blur3->setLabel("threshold B");
	m_Sharpen1->setLabel("sharpen RGB");
	m_Sharpen2->setLabel("sharpen G");
	m_Sharpen3->setLabel("sharpen B");
	m_Blur4->setLabel("blur RGB");
	m_Blur5->setLabel("blur G");
	m_Blur6->setLabel("blur B");
	}
	
	if(YUV){
	m_Gang1->setIsSecretAndDisabled(YUV);
	m_Gang2->setIsSecretAndDisabled(!YUV);
	m_Gang2->setValue(true);
	m_Blur2->setIsSecretAndDisabled(YUV);
	m_Blur3->setIsSecretAndDisabled(YUV);
	m_Sharpen2->setIsSecretAndDisabled(YUV);
	m_Sharpen3->setIsSecretAndDisabled(YUV);
	m_Blur5->setIsSecretAndDisabled(!YUV);
	m_Blur1->setLabel("threshold Y");
	m_Sharpen1->setLabel("sharpen Y");
	m_Blur4->setLabel("blur Y");
	m_Blur5->setLabel("blur UV");
	m_Blur6->setLabel("blur V");
	}
	
	if(LAB){
	m_Gang1->setIsSecretAndDisabled(LAB);
	m_Gang2->setIsSecretAndDisabled(!LAB);
	m_Gang2->setValue(true);
	m_Blur2->setIsSecretAndDisabled(LAB);
	m_Blur3->setIsSecretAndDisabled(LAB);
	m_Sharpen2->setIsSecretAndDisabled(LAB);
	m_Sharpen3->setIsSecretAndDisabled(LAB);
	m_Blur5->setIsSecretAndDisabled(!LAB);
	m_Blur1->setLabel("threshold Luma");
	m_Sharpen1->setLabel("sharpen Luma");
	m_Blur4->setLabel("blur Luma");
	m_Blur5->setLabel("blur AB");
	m_Blur6->setLabel("blur B");
	}
    }
    
    if(p_ParamName == "Gang1")
    {
    
    bool gang1 = m_Gang1->getValueAtTime(p_Args.time);
    
    if(!gang1) {
    m_Blur1->setLabel("threshold R");
    m_Sharpen1->setLabel("sharpen R");
    m_Blur4->setLabel("blur R");
    } else {
    m_Blur1->setLabel("threshold RGB");
    m_Sharpen1->setLabel("sharpen RGB");
    m_Blur4->setLabel("blur RGB");
    }
    
    m_Blur2->setIsSecretAndDisabled(gang1);
	m_Blur3->setIsSecretAndDisabled(gang1);
	m_Sharpen2->setIsSecretAndDisabled(gang1);
	m_Sharpen3->setIsSecretAndDisabled(gang1);
	m_Blur5->setIsSecretAndDisabled(gang1);
	m_Blur6->setIsSecretAndDisabled(gang1);
	
	}
	
	if(p_ParamName == "Gang2")
    {
    
    int space_i;
    m_Space->getValueAtTime(p_Args.time, space_i);
    FrequencyEnum FrequencyFilter = (FrequencyEnum)space_i;
    
    bool YUV = space_i == 1;
    bool LAB = space_i == 2;
    
    bool gang2 = m_Gang2->getValueAtTime(p_Args.time);
    
    if(LAB) {
    if(!gang2) {
    m_Blur5->setLabel("blur A");
    } else {
    m_Blur5->setLabel("blur AB");
    }}
    
    if(YUV) {
    if(!gang2) {
    m_Blur5->setLabel("blur U");
    } else {
    m_Blur5->setLabel("blur UV");
    }}
    
	m_Blur6->setIsSecretAndDisabled(gang2);
	}
    
    
    if(p_ParamName == "Info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if(p_ParamName == "Button1")
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
    	
	fprintf (pFile, "// Frequency Separation plugin DCTL export\n" \
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

	if(p_ParamName == "Button2")
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
	" name Frequency Separation\n" \
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
    
    if(p_ParamName == "Button3")
    {
    
    string PATH2;
	m_Path2->getValue(PATH2);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".glsl and " + NAME + ".xml to " + PATH2 + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	FILE * pFile2;
	
	pFile = fopen ((PATH2 + "/" + NAME + ".glsl").c_str(), "w");
	pFile2 = fopen ((PATH2 + "/" + NAME + ".xml").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile,
	"// Frequency Separation Shader  \n" \
	"  \n" \
	"#version 120  \n" \
	"uniform sampler2D front;  \n" \
	"uniform float adsk_result_w, adsk_result_h;  \n" \
	"\n");
	fclose (pFile);
	fprintf (pFile2,
	"<ShaderNodePreset SupportsAdaptiveDegradation=\"0\" Description=\"Frequency\" Name=\"Frequency\"> \n" \
	"<Shader OutputBitDepth=\"Output\" Index=\"1\"> \n" \
	"</Shader> \n" \
	"<Page Name=\"Frequency\" Page=\"0\"> \n" \
	"<Col Name=\"Frequency\" Col=\"0\" Page=\"0\"> \n" \
	"</Col> \n" \
	"</Page> \n" \
	"</ShaderNodePreset> \n");
	fclose (pFile2);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".glsl and " + NAME + ".xml to " + PATH2  + ". Check Permissions."));
	}	
	}
	}
    
}


void FrequencyPlugin::setupAndProcess(Frequency& p_Frequency, const OFX::RenderArguments& p_Args)
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

	int space_i;
    m_Space->getValueAtTime(p_Args.time, space_i);
    FrequencyEnum FrequencyFilter = (FrequencyEnum)space_i;
    
    int _space = space_i;
    
    float _blur[6], _sharpen[3];
    int _switch[3];

	_blur[0] = m_Blur1->getValueAtTime(p_Args.time);
	_blur[1] = m_Blur2->getValueAtTime(p_Args.time);
	_blur[2] = m_Blur3->getValueAtTime(p_Args.time);
	_blur[3] = m_Blur4->getValueAtTime(p_Args.time);
	_blur[4] = m_Blur5->getValueAtTime(p_Args.time);
	_blur[5] = m_Blur6->getValueAtTime(p_Args.time);
	
	_blur[0] *= 2.0f;
	_blur[1] *= 2.0f;
	_blur[2] *= 2.0f;
	_blur[3] *= 5.0f;
	_blur[4] *= 5.0f;
	_blur[5] *= 5.0f;
	
	_sharpen[0] = m_Sharpen1->getValueAtTime(p_Args.time);
	_sharpen[1] = m_Sharpen2->getValueAtTime(p_Args.time);
	_sharpen[2] = m_Sharpen3->getValueAtTime(p_Args.time);
	
	_sharpen[0] = (_sharpen[0] * 2.0f) + 1.0f;
	_sharpen[1] = (_sharpen[1] * 2.0f) + 1.0f;
	_sharpen[2] = (_sharpen[2] * 2.0f) + 1.0f;
	
	bool gang1 = m_Gang1->getValueAtTime(p_Args.time);
	_switch[0] = gang1 ? 1 : 0;
	bool gang2 = m_Gang2->getValueAtTime(p_Args.time);
	_switch[1] = gang2 ? 1 : 0;
	bool display = m_Display->getValueAtTime(p_Args.time);
	_switch[2] = display ? 1 : 0;
	
    // Set the images
    p_Frequency.setDstImg(dst.get());
    p_Frequency.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    //p_Frequency.setGPURenderArgs(p_Args);

    // Set the render window
    p_Frequency.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_Frequency.setScales(_space, _blur, _sharpen, _switch);

    // Call the base class process member, this will call the derived templated process code
    p_Frequency.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

FrequencyPluginFactory::FrequencyPluginFactory()
    : OFX::PluginFactoryHelper<FrequencyPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void FrequencyPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
    //p_Desc.setSupportsOpenCLRender(true);
    //p_Desc.setSupportsCudaRender(true);
}

void FrequencyPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamFrequency);
	param->setLabel(kParamFrequencyLabel);
	param->setHint(kParamFrequencyHint);
	assert(param->getNOptions() == (int)eFrequencyRGB);
	param->appendOption(kParamFrequencyOptionRGB, kParamFrequencyOptionRGBHint);
	assert(param->getNOptions() == (int)eFrequencyYUV);
	param->appendOption(kParamFrequencyOptionYUV, kParamFrequencyOptionYUVHint);
	assert(param->getNOptions() == (int)eFrequencyLAB);
	param->appendOption(kParamFrequencyOptionLAB, kParamFrequencyOptionLABHint);
	param->setDefault( (int)eFrequencyRGB );
	param->setAnimates(false);
    page->addChild(*param);
	}
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("Display");
    boolParam->setLabel("display high frequency");
    boolParam->setHint("Display high frequency");
    boolParam->setDefault(false);
    page->addChild(*boolParam);
    
    boolParam = p_Desc.defineBooleanParam("Gang1");
    boolParam->setLabel("gang all");
    boolParam->setHint("Adjust 3 channels with single parameter");
    boolParam->setDefault(true);
    page->addChild(*boolParam);
    
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam("Blur1");
    param->setLabel("threshold RGB");
    param->setHint("Adjust threshold");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Blur2");
    param->setLabel("threshold G");
    param->setHint("Adjust threshold");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Blur3");
    param->setLabel("threshold B");
    param->setHint("Adjust threshold");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Sharpen1");
    param->setLabel("sharpen RGB");
    param->setHint("sharpen");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Sharpen2");
    param->setLabel("sharpen G");
    param->setHint("sharpen");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Sharpen3");
    param->setLabel("sharpen B");
    param->setHint("sharpen");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Blur4");
    param->setLabel("blur RGB");
    param->setHint("Adjust blur");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Blur5");
    param->setLabel("blur G");
    param->setHint("Adjust blur");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Blur6");
    param->setLabel("blur B");
    param->setHint("Adjust blur");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    boolParam = p_Desc.defineBooleanParam("Gang2");
    boolParam->setLabel("gang chroma channels");
    boolParam->setHint("Adjust both chroma channels with single parameter");
    boolParam->setDefault(true);
    boolParam->setIsSecretAndDisabled(true);
    page->addChild(*boolParam);
    
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("Info");
    param->setLabel("Info");
    page->addChild(*param);
    }
    
    {    
    GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
    script->setOpen(false);
    script->setHint("Export DCTL and Nuke script");
      if (page) {
            page->addChild(*script);
            }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("Button1");
    param->setLabel("Export DCTL");
    param->setHint("Create DCTL version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("Button2");
    param->setLabel("Export Nuke script");
    param->setHint("Create NUKE version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("Name");
	param->setLabel("Name");
	param->setHint("Overwrites if the same");
	param->setDefault("FrequencySeparation");
	param->setParent(*script);
	page->addChild(*param);
	}
	{
	StringParamDescriptor* param = p_Desc.defineStringParam("Path");
	param->setLabel("Directory");
	param->setHint("Make sure it's the absolute path");
	param->setStringType(eStringTypeFilePath);
	param->setDefault(kPluginScript);
	param->setFilePathExists(false);
	param->setParent(*script);
	page->addChild(*param);
	}
	{
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("Button3");
    param->setLabel("Export Shader");
    param->setHint("create Shader version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("Path2");
	param->setLabel("Shader Directory");
	param->setHint("Make sure it's the absolute path");
	param->setStringType(eStringTypeFilePath);
	param->setDefault(kPluginScript2);
	param->setFilePathExists(false);
	param->setParent(*script);
	page->addChild(*param);
	}
	}        
}

ImageEffect* FrequencyPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new FrequencyPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static FrequencyPluginFactory FrequencyPlugin;
    p_FactoryArray.push_back(&FrequencyPlugin);
}
