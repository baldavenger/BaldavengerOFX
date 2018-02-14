#include "ConvolutionPlugin.h"

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

#define kPluginName "Convolution"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Convolution Filters"

#define kPluginIdentifier "OpenFX.Yo.Convolution"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kParamConvolution "ConvolutionFilter"
#define kParamConvolutionLabel "filter type"
#define kParamConvolutionHint "types of convolution filter"
#define kParamConvolutionOptionGaus "Gaussian Blur"
#define kParamConvolutionOptionGausHint "recursive gaussian blur"
#define kParamConvolutionOptionSimple "Simple Blur"
#define kParamConvolutionOptionSimpleHint "simple recursive blur"
#define kParamConvolutionOptionBox "Box Blur"
#define kParamConvolutionOptionBoxHint "box blur"
#define kParamConvolutionOptionFrequency "Frequency Separation"
#define kParamConvolutionOptionFrequencyHint "blur low frequency, sharpen high frequency"
#define kParamConvolutionOptionEdgeDetect "Edge Detect"
#define kParamConvolutionOptionEdgeDetectHint "edge detect"
#define kParamConvolutionOptionEdgeEnhance "Edge Enhance"
#define kParamConvolutionOptionEdgeEnhanceHint "edge enhance"
#define kParamConvolutionOptionErode "Erode"
#define kParamConvolutionOptionErodeHint "erodes dark area of matte"
#define kParamConvolutionOptionDilate "Dilate"
#define kParamConvolutionOptionDilateHint "expands dark area of matte"
#define kParamConvolutionOptionCustom "Custom"
#define kParamConvolutionOptionCustomHint "custom 3x3 spatial matrix"

enum ConvolutionEnum
{
    eConvolutionGaus,
    eConvolutionSimple,
    eConvolutionBox,
    eConvolutionFrequency,
    eConvolutionEdgeDetect,
    eConvolutionEdgeEnhance,
    eConvolutionErode,
    eConvolutionDilate,
    eConvolutionCustom,
};

////////////////////////////////////////////////////////////////////////////////

namespace {
    struct RGBValues {
        double r,g,b;
        RGBValues(double v) : r(v), g(v), b(v) {}
        RGBValues() : r(0), g(0), b(0) {}
    };
}

class Convolution : public OFX::ImageProcessor
{
public:
    explicit Convolution(OFX::ImageEffect& p_Instance);

    //virtual void processImagesCUDA();
    //virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);
    
    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(int p_Convolve, float p_Adjust1, float p_Adjust2, float p_Threshold, int p_Display, float p_Matrix11, float p_Matrix12, 
		float p_Matrix13, float p_Matrix21, float p_Matrix22, float p_Matrix23, float p_Matrix31, float p_Matrix32, float p_Matrix33);

private:
    OFX::Image* _srcImg;
    int convolve;
    float adjust1;
    float adjust2;
    float threshold;
    int display;
    float matrix[9];
};

Convolution::Convolution(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}
/*
extern void RunCudaKernel(int p_Width, int p_Height, int p_Convolve, float p_Adjust1, 
float p_Adjust2, float p_Threshold, int p_Display, float* p_Matrix, float* p_Input, float* p_Output);

void Convolution::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, convolve, adjust1, adjust2, threshold, display, matrix, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, int p_Convolve, 
float p_Adjust1, float p_Adjust2, float p_Threshold, int p_Display, float* p_Matrix, float* p_Input, float* p_Output);

void Convolution::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, convolve, adjust1, adjust2, threshold, display, matrix, input, output);
}
*/
void Convolution::multiThreadProcessImages(OfxRectI p_ProcWindow)
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

void Convolution::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void Convolution::setScales(int p_Convolve, float p_Adjust1, float p_Adjust2, float p_Threshold, int p_Display, float p_Matrix11, float p_Matrix12, 
			float p_Matrix13, float p_Matrix21, float p_Matrix22, float p_Matrix23, float p_Matrix31, float p_Matrix32, float p_Matrix33)
{
   convolve = p_Convolve;
   adjust1 = p_Adjust1;
   adjust2 = p_Adjust2;
   threshold = p_Threshold;
   display = p_Display;
   matrix[0] = p_Matrix11;
   matrix[1] = p_Matrix12;
   matrix[2] = p_Matrix13;
   matrix[3] = p_Matrix21;
   matrix[4] = p_Matrix22;
   matrix[5] = p_Matrix23;
   matrix[6] = p_Matrix31;
   matrix[7] = p_Matrix32;
   matrix[8] = p_Matrix33;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class ConvolutionPlugin : public OFX::ImageEffect
{
public:
    explicit ConvolutionPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
     /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(Convolution &p_Convolution, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    
    OFX::ChoiceParam* m_Convolve;
    OFX::DoubleParam* m_Adjust1;
    OFX::DoubleParam* m_Adjust2;
    OFX::DoubleParam* m_Threshold;
    OFX::BooleanParam* m_Display;
    OFX::Double3DParam* m_Row1;
    OFX::Double3DParam* m_Row2;
    OFX::Double3DParam* m_Row3;
    
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Path2;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
	OFX::PushButtonParam* m_Button3;
};

ConvolutionPlugin::ConvolutionPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    
    m_Convolve = fetchChoiceParam(kParamConvolution);
    m_Adjust1 = fetchDoubleParam("Adjust1");
    m_Adjust2 = fetchDoubleParam("Adjust2");
    m_Threshold = fetchDoubleParam("Threshold");
    m_Display = fetchBooleanParam("display");
    m_Row1 = fetchDouble3DParam("row1");
    m_Row2 = fetchDouble3DParam("row2");
    m_Row3 = fetchDouble3DParam("row3");

    m_Path = fetchStringParam("path");
    m_Path2 = fetchStringParam("path2");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");
	m_Button3 = fetchPushButtonParam("button3");

}

void ConvolutionPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        Convolution Convolution(*this);
        setupAndProcess(Convolution, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool ConvolutionPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
   
    
    
    if (m_SrcClip)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void ConvolutionPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    if(p_ParamName == kParamConvolution)
    {
    int convolve_i;
    m_Convolve->getValueAtTime(p_Args.time, convolve_i);
    ConvolutionEnum ConvolutionFilter = (ConvolutionEnum)convolve_i;
    
    bool freq = convolve_i == 3;
    bool edge = convolve_i == 4;
    bool enhance = convolve_i == 5;
    bool erode = convolve_i == 6;
    bool dilate = convolve_i == 7;
    bool cust = convolve_i == 8;
    
    if(erode || dilate) {
    m_Adjust1->setLabel("amount");
    } else
    if(cust || enhance) {
    m_Adjust1->setLabel("scale");
    m_Display->setLabel("normalise");
    } else {
    m_Adjust1->setLabel("blur");
    m_Adjust2->setLabel("sharpen");
    m_Threshold->setLabel("threshold");
    }
    
    if(edge) {
    m_Adjust1->setIsSecretAndDisabled(true);
    } else {
    m_Adjust1->setIsSecretAndDisabled(false);
    }
    
    m_Adjust2->setIsSecretAndDisabled(!freq);
    m_Threshold->setIsSecretAndDisabled(!freq && !edge);
    m_Display->setIsSecretAndDisabled(!freq && !cust);
    m_Row1->setIsSecretAndDisabled(!cust);
    m_Row2->setIsSecretAndDisabled(!cust);
    m_Row3->setIsSecretAndDisabled(!cust);
    
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
    	
	fprintf (pFile, "// ConvolutionPlugin DCTL export\n" \
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
	" name Convolution\n" \
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
    
    if(p_ParamName == "button3")
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
	"// Convolution Shader  \n" \
	"  \n" \
	"#version 120  \n" \
	"uniform sampler2D front;  \n" \
	"uniform float adsk_result_w, adsk_result_h;  \n" \
	"\n");
	fclose (pFile);
	fprintf (pFile2,
	"<ShaderNodePreset SupportsAdaptiveDegradation=\"0\" Description=\"Convolution\" Name=\"Convolution\"> \n" \
	"<Shader OutputBitDepth=\"Output\" Index=\"1\"> \n" \
	"</Shader> \n" \
	"<Page Name=\"Convolution\" Page=\"0\"> \n" \
	"<Col Name=\"Convolution\" Col=\"0\" Page=\"0\"> \n" \
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


void ConvolutionPlugin::setupAndProcess(Convolution& p_Convolution, const OFX::RenderArguments& p_Args)
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

	int convolve_i;
    m_Convolve->getValueAtTime(p_Args.time, convolve_i);
    ConvolutionEnum ConvolutionFilter = (ConvolutionEnum)convolve_i;
    
    int _convolve = convolve_i;

	double _adjust1 = m_Adjust1->getValueAtTime(p_Args.time);
	double _adjust2 = m_Adjust2->getValueAtTime(p_Args.time);
	double _threshold = m_Threshold->getValueAtTime(p_Args.time);
	bool display = m_Display->getValueAtTime(p_Args.time);
	int _display = display ? 1 : 0;
	
	RGBValues rMatrix;
    m_Row1->getValueAtTime(p_Args.time, rMatrix.r, rMatrix.g, rMatrix.b);
    double r1Matrix = rMatrix.r;
    double r2Matrix = rMatrix.g;
    double r3Matrix = rMatrix.b;
    
    RGBValues gMatrix;
    m_Row2->getValueAtTime(p_Args.time, gMatrix.r, gMatrix.g, gMatrix.b);
    double g1Matrix = gMatrix.r;
    double g2Matrix = gMatrix.g;
    double g3Matrix = gMatrix.b;
    
    RGBValues bMatrix;
    m_Row3->getValueAtTime(p_Args.time, bMatrix.r, bMatrix.g, bMatrix.b);
    double b1Matrix = bMatrix.r;
    double b2Matrix = bMatrix.g;
    double b3Matrix = bMatrix.b;

    // Set the images
    p_Convolution.setDstImg(dst.get());
    p_Convolution.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    //p_Convolution.setGPURenderArgs(p_Args);

    // Set the render window
    p_Convolution.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_Convolution.setScales(_convolve, _adjust1, _adjust2, _threshold, _display, 
    r1Matrix, r2Matrix, r3Matrix, g1Matrix, g2Matrix, g3Matrix, b1Matrix, b2Matrix, b3Matrix);

    // Call the base class process member, this will call the derived templated process code
    p_Convolution.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ConvolutionPluginFactory::ConvolutionPluginFactory()
    : OFX::PluginFactoryHelper<ConvolutionPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ConvolutionPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

void ConvolutionPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
	ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamConvolution);
	param->setLabel(kParamConvolutionLabel);
	param->setHint(kParamConvolutionHint);
	assert(param->getNOptions() == (int)eConvolutionGaus);
	param->appendOption(kParamConvolutionOptionGaus, kParamConvolutionOptionGausHint);
	assert(param->getNOptions() == (int)eConvolutionSimple);
	param->appendOption(kParamConvolutionOptionSimple, kParamConvolutionOptionSimpleHint);
	assert(param->getNOptions() == (int)eConvolutionBox);
	param->appendOption(kParamConvolutionOptionBox, kParamConvolutionOptionBoxHint);
	assert(param->getNOptions() == (int)eConvolutionFrequency);
	param->appendOption(kParamConvolutionOptionFrequency, kParamConvolutionOptionFrequencyHint);
	assert(param->getNOptions() == (int)eConvolutionEdgeDetect);
	param->appendOption(kParamConvolutionOptionEdgeDetect, kParamConvolutionOptionEdgeDetectHint);
	assert(param->getNOptions() == (int)eConvolutionEdgeEnhance);
	param->appendOption(kParamConvolutionOptionEdgeEnhance, kParamConvolutionOptionEdgeEnhanceHint);
	assert(param->getNOptions() == (int)eConvolutionErode);
	param->appendOption(kParamConvolutionOptionErode, kParamConvolutionOptionErodeHint);
	assert(param->getNOptions() == (int)eConvolutionDilate);
	param->appendOption(kParamConvolutionOptionDilate, kParamConvolutionOptionDilateHint);
	assert(param->getNOptions() == (int)eConvolutionCustom);
	param->appendOption(kParamConvolutionOptionCustom, kParamConvolutionOptionCustomHint);
	param->setDefault( (int)eConvolutionGaus );
	param->setAnimates(false);
    page->addChild(*param);
	}
    
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam("Adjust1");
    param->setLabel("blur");
    param->setHint("adjust convolution");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Adjust2");
    param->setLabel("sharpen");
    param->setHint("adjust convolution");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    param = p_Desc.defineDoubleParam("Threshold");
    param->setLabel("threshold");
    param->setHint("adjust threshold");
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setIsSecretAndDisabled(true);
    page->addChild(*param);
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("display");
    boolParam->setLabel("display high frequency");
    //boolParam->setHint("display high frequency");
    boolParam->setDefault(false);
    boolParam->setIsSecretAndDisabled(true);
    page->addChild(*boolParam);
    
    Double3DParamDescriptor* paramR = p_Desc.defineDouble3DParam("row1");
    paramR->setLabel("Row 1");
    paramR->setDefault(0, 0, 0);
    paramR->setIncrement(1);
    paramR->setIsSecretAndDisabled(true);
    page->addChild(*paramR);
    
    Double3DParamDescriptor* paramG = p_Desc.defineDouble3DParam("row2");
    paramG->setLabel("Row 2");
    paramG->setDefault(0, 1, 0);
    paramG->setIncrement(1);
    paramG->setIsSecretAndDisabled(true);
    page->addChild(*paramG);
    
    Double3DParamDescriptor* paramB = p_Desc.defineDouble3DParam("row3");
    paramB->setLabel("Row 3");
    paramB->setDefault(0, 0, 0);
    paramB->setIncrement(1);
    paramB->setIsSecretAndDisabled(true);
    page->addChild(*paramB);
	
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
	param->setDefault("Convolution");
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
	{
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button3");
    param->setLabel("Export Shader");
    param->setHint("create Shader version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("path2");
	param->setLabel("Shader Directory");
	param->setHint("make sure it's the absolute path");
	param->setStringType(eStringTypeFilePath);
	param->setDefault(kPluginScript2);
	param->setFilePathExists(false);
	param->setParent(*script);
	page->addChild(*param);
	}
	}        
}

ImageEffect* ConvolutionPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new ConvolutionPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static ConvolutionPluginFactory ConvolutionPlugin;
    p_FactoryArray.push_back(&ConvolutionPlugin);
}
