#include "VideoGradePlugin.h"

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

#define kPluginName "VideoGrade"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"VideoGrade: Video style (Rec.709) grading, with Lift, Gamma, Gain, and Offset controls, adjustable \n" \
"anchors for Lift and Gain, adjustable region for Gamma, and Upper Gamma Bias option."

#define kPluginIdentifier "OpenFX.Yo.VideoGrade"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

class ImageScaler : public OFX::ImageProcessor
{
public:
    explicit ImageScaler(OFX::ImageEffect& p_Instance);

	virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float p_SwitchA, float p_ScaleL, float p_ScaleLA, float p_ScaleG, float p_ScaleGa, float p_ScaleGb, float p_ScaleGG, float p_ScaleGA, float p_ScaleO);

private:
    OFX::Image* _srcImg;
    float _switch[1];
    float _scales[8];
    
};

ImageScaler::ImageScaler(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(int p_Width, int p_Height, float* p_Switch, float* p_Gain, const float* p_Input, float* p_Output);

void ImageScaler::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, _switch, _scales, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Switch, float* p_Gain, const float* p_Input, float* p_Output);

void ImageScaler::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _switch, _scales, input, output);
}

void ImageScaler::multiThreadProcessImages(OfxRectI p_ProcWindow)
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
                  float GR = srcPix[0] >= _scales[6] ? (srcPix[0] - _scales[6]) * _scales[5]  + _scales[6]: srcPix[0];
                  float LR = GR <= _scales[1] ? (((GR / _scales[1]) + (_scales[0] * (1 - (GR / _scales[1])))) * _scales[1]) + _scales[7]: GR + _scales[7];
                  float Prl = LR >= _scales[3] && LR <= _scales[4] ? pow((LR - _scales[3]) / (_scales[4] - _scales[3]), 1.0/_scales[2]) * (_scales[4] - _scales[3]) + _scales[3] : LR;
                  float Pru = LR >= _scales[3] && LR <= _scales[4] ? (1.0 - pow(1.0 - (LR - _scales[3]) / (_scales[4] - _scales[3]), _scales[2])) * (_scales[4] - _scales[3]) + _scales[3] : LR;
                  float R = _switch[0] == 1 ? Pru : Prl;
                  
                  float GG = srcPix[1] >= _scales[6] ? (srcPix[1] - _scales[6]) * _scales[5]  + _scales[6]: srcPix[1];
                  float LG = GG <= _scales[1] ? (((GG / _scales[1]) + (_scales[0] * (1 - (GG / _scales[1])))) * _scales[1]) + _scales[7]: GG + _scales[7];
                  float Pgl = LG >= _scales[3] && LG <= _scales[4] ? pow((LG - _scales[3]) / (_scales[4] - _scales[3]), 1.0/_scales[2]) * (_scales[4] - _scales[3]) + _scales[3] : LG;
                  float Pgu = LG >= _scales[3] && LG <= _scales[4] ? (1.0 - pow(1.0 - (LG - _scales[3]) / (_scales[4] - _scales[3]), _scales[2])) * (_scales[4] - _scales[3]) + _scales[3] : LG;
                  float G = _switch[0] == 1 ? Pgu : Pgl;
                  
                  float GB = srcPix[2] >= _scales[6] ? (srcPix[2] - _scales[6]) * _scales[5]  + _scales[6]: srcPix[2];
                  float LB = GB <= _scales[1] ? (((GB / _scales[1]) + (_scales[0] * (1 - (GB / _scales[1])))) * _scales[1]) + _scales[7]: GB + _scales[7];
                  float Pbl = LB >= _scales[3] && LB <= _scales[4] ? pow((LB - _scales[3]) / (_scales[4] - _scales[3]), 1.0/_scales[2]) * (_scales[4] - _scales[3]) + _scales[3] : LB;
                  float Pbu = LB >= _scales[3] && LB <= _scales[4] ? (1.0 - pow(1.0 - (LB - _scales[3]) / (_scales[4] - _scales[3]), _scales[2])) * (_scales[4] - _scales[3]) + _scales[3] : LB;
                  float B = _switch[0] == 1 ? Pbu : Pbl;
                  
                  
                  dstPix[0] = R;
                  dstPix[1] = G;
                  dstPix[2] = B;
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

void ImageScaler::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ImageScaler::setScales(float p_SwitchA, float p_ScaleL, float p_ScaleLA, float p_ScaleG, float p_ScaleGa, float p_ScaleGb, float p_ScaleGG, float p_ScaleGA, float p_ScaleO)
{
    _switch[0] = p_SwitchA;
    _scales[0] = p_ScaleL;
    _scales[1] = p_ScaleLA;
    _scales[2] = p_ScaleG;
    _scales[3] = p_ScaleGa;
    _scales[4] = p_ScaleGb;
    _scales[5] = p_ScaleGG;
    _scales[6] = p_ScaleGA;
    _scales[7] = p_ScaleO;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class VideoGradePlugin : public OFX::ImageEffect
{
public:
    explicit VideoGradePlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    
    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(ImageScaler &p_ImageScaler, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

	OFX::BooleanParam* m_SwitchA;
    OFX::DoubleParam* m_ScaleL;
    OFX::DoubleParam* m_ScaleLA;
    OFX::DoubleParam* m_ScaleG;
    OFX::DoubleParam* m_ScaleGa;
    OFX::DoubleParam* m_ScaleGb;
    OFX::DoubleParam* m_ScaleGG;
    OFX::DoubleParam* m_ScaleGA;
    OFX::DoubleParam* m_ScaleO;
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
};

VideoGradePlugin::VideoGradePlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

	m_SwitchA = fetchBooleanParam("switch1");
    m_ScaleL = fetchDoubleParam("scaleL");
    m_ScaleLA = fetchDoubleParam("scaleLA");
    m_ScaleG = fetchDoubleParam("scaleG");
    m_ScaleGa = fetchDoubleParam("scaleGa");
    m_ScaleGb = fetchDoubleParam("scaleGb");
    m_ScaleGG = fetchDoubleParam("scaleGG");
    m_ScaleGA = fetchDoubleParam("scaleGA");
    m_ScaleO = fetchDoubleParam("scaleO");
    m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");
    
}

void VideoGradePlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ImageScaler imageScaler(*this);
        setupAndProcess(imageScaler, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool VideoGradePlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    double lScale = m_ScaleL->getValueAtTime(p_Args.time);
    double gScale = m_ScaleG->getValueAtTime(p_Args.time);
    double ggScale = m_ScaleGG->getValueAtTime(p_Args.time);
    double oScale = m_ScaleO->getValueAtTime(p_Args.time);
    

    if ((lScale == 1.0) && (gScale == 1.0) && (ggScale == 1.0) && (oScale == 0.0))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void VideoGradePlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
 
 	if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if(p_ParamName == "button1")
    {
    
    bool Switch = m_SwitchA->getValueAtTime(p_Args.time);
	int upper = Switch ? 1 : 0;
	
    float lift = m_ScaleL->getValueAtTime(p_Args.time);
    float liftanchor = m_ScaleLA->getValueAtTime(p_Args.time);
    float gamma = m_ScaleG->getValueAtTime(p_Args.time);
    float anchorA = m_ScaleGa->getValueAtTime(p_Args.time);
    float anchorB = m_ScaleGb->getValueAtTime(p_Args.time);
    float gain = m_ScaleGG->getValueAtTime(p_Args.time);
    float gainanchor = m_ScaleGA->getValueAtTime(p_Args.time);
    float offset = m_ScaleO->getValueAtTime(p_Args.time);
    
    string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// VideoGradePlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"    \n" \
	"    int switchA = %d;\n" \
	"    bool p_SwitchA = switchA == 1;\n" \
	"    \n" \
	"    float p_Lift = %ff;\n" \
	"    float p_LiftAnchor = %ff;\n" \
	"    float p_Gamma = %ff;\n" \
	"    float p_GammaAreaA = %ff;\n" \
	"    float p_GammaAreaB = %ff;\n" \
	"    float p_Gain = %ff;\n" \
	"    float p_GainAnchor = %ff;\n" \
	"    float p_Offset = %ff;\n" \
	"    \n" \
	"	float GR = p_R >= p_GainAnchor ? (p_R - p_GainAnchor) * p_Gain  + p_GainAnchor: p_R;\n" \
	"	float LR = GR <= p_LiftAnchor ? (((GR / p_LiftAnchor) + (p_Lift * (1.0f - (GR / p_LiftAnchor)))) * p_LiftAnchor) + p_Offset: GR + p_Offset;\n" \
	"	float Prl = LR >= p_GammaAreaA && LR <= p_GammaAreaB ? pow((LR - p_GammaAreaA) / (p_GammaAreaB - p_GammaAreaA), 1.0f/p_Gamma) * (p_GammaAreaB - p_GammaAreaA) + p_GammaAreaA : LR;\n" \
	"	float Pru = LR >= p_GammaAreaA && LR <= p_GammaAreaB ? (1.0f - pow(1.0f - (LR - p_GammaAreaA) / (p_GammaAreaB - p_GammaAreaA), p_Gamma)) * (p_GammaAreaB - p_GammaAreaA) + p_GammaAreaA : LR;\n" \
	"	const float r = p_SwitchA ? Pru : Prl;\n" \
	"\n" \
	"	float GG = p_G >= p_GainAnchor ? (p_G - p_GainAnchor) * p_Gain  + p_GainAnchor: p_G;\n" \
	"	float LG = GG <= p_LiftAnchor ? (((GG / p_LiftAnchor) + (p_Lift * (1.0f - (GG / p_LiftAnchor)))) * p_LiftAnchor) + p_Offset: GG + p_Offset;\n" \
	"	float Pgl = LG >= p_GammaAreaA && LG <= p_GammaAreaB ? pow((LG - p_GammaAreaA) / (p_GammaAreaB - p_GammaAreaA), 1.0f/p_Gamma) * (p_GammaAreaB - p_GammaAreaA) + p_GammaAreaA : LG;\n" \
	"	float Pgu = LG >= p_GammaAreaA && LG <= p_GammaAreaB ? (1.0f - pow(1.0f - (LG - p_GammaAreaA) / (p_GammaAreaB - p_GammaAreaA), p_Gamma)) * (p_GammaAreaB - p_GammaAreaA) + p_GammaAreaA : LG;\n" \
	"	const float g = p_SwitchA ? Pgu : Pgl;\n" \
	"\n" \
	"	float GB = p_B >= p_GainAnchor ? (p_B - p_GainAnchor) * p_Gain  + p_GainAnchor: p_B;\n" \
	"	float LB = GB <= p_LiftAnchor ? (((GB / p_LiftAnchor) + (p_Lift * (1.0f - (GB / p_LiftAnchor)))) * p_LiftAnchor) + p_Offset: GB + p_Offset;\n" \
	"	float Pbl = LB >= p_GammaAreaA && LB <= p_GammaAreaB ? pow((LB - p_GammaAreaA) / (p_GammaAreaB - p_GammaAreaA), 1.0f/p_Gamma) * (p_GammaAreaB - p_GammaAreaA) + p_GammaAreaA : LB;\n" \
	"	float Pbu = LB >= p_GammaAreaA && LB <= p_GammaAreaB ? (1.0f - pow(1.0f - (LB - p_GammaAreaA) / (p_GammaAreaB - p_GammaAreaA), p_Gamma)) * (p_GammaAreaB - p_GammaAreaA) + p_GammaAreaA : LB;\n" \
	"	const float b = p_SwitchA ? Pbu : Pbl;\n" \
	" \n" \
	"    return make_float3(r, g, b);\n" \
	"}\n", upper, lift, liftanchor, gamma, anchorA, anchorB, gain, gainanchor, offset);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}
	
	if(p_ParamName == "button2")
    {
	
	bool Switch = m_SwitchA->getValueAtTime(p_Args.time);
	int upper = Switch ? 1 : 0;
	
    float lift = m_ScaleL->getValueAtTime(p_Args.time);
    float liftanchor = m_ScaleLA->getValueAtTime(p_Args.time);
    float gamma = m_ScaleG->getValueAtTime(p_Args.time);
    float anchorA = m_ScaleGa->getValueAtTime(p_Args.time);
    float anchorB = m_ScaleGb->getValueAtTime(p_Args.time);
    float gain = m_ScaleGG->getValueAtTime(p_Args.time);
    float gainanchor = m_ScaleGA->getValueAtTime(p_Args.time);
    float offset = m_ScaleO->getValueAtTime(p_Args.time);
    
    string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".nk to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".nk").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "Group {\n" \
	" inputs 0\n" \
	" name VideGrade\n" \
	" xpos -79\n" \
	" ypos -26\n" \
	"}\n" \
	" Input {\n" \
	"  inputs 0\n" \
	"  name Input1\n" \
	"  xpos -186\n" \
	"  ypos -123\n" \
	" }\n" \
	" Expression {\n" \
	"  temp_name0 gain\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 gainanchor\n" \
	"  temp_expr1 %f\n" \
	"  expr0 \"r >= gainanchor ? (r - gainanchor) * gain  + gainanchor : r\"\n" \
	"  expr1 \"g >= gainanchor ? (g - gainanchor) * gain  + gainanchor : g\"\n" \
	"  expr2 \"b >= gainanchor ? (b - gainanchor) * gain  + gainanchor : b\"\n" \
	"  name gain_anchor\n" \
	"  xpos -186\n" \
	"  ypos -74\n" \
	" }\n" \
	" Expression {\n" \
	"  temp_name0 lift\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 liftanchor\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 offset\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"r <= liftanchor ? (((r / liftanchor) + (lift * (1.0 - (r / liftanchor)))) * liftanchor) + offset: r + offset\"\n" \
	"  expr1 \"g <= liftanchor ? (((g / liftanchor) + (lift * (1.0 - (g / liftanchor)))) * liftanchor) + offset: g + offset\"\n" \
	"  expr2 \"b <= liftanchor ? (((b / liftanchor) + (lift * (1.0 - (b / liftanchor)))) * liftanchor) + offset: b + offset\"\n" \
	"  name liftanchor\n" \
	"  xpos -186\n" \
	"  ypos -38\n" \
	" }\n" \
	"set N2741f900 [stack 0]\n" \
	" Expression {\n" \
	"  temp_name0 gamma\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 anchorA\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 anchorB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"r >= anchorA && r <= anchorB ? (1.0 - pow(1.0 - (r - anchorA) / (anchorB - anchorA), gamma)) * (anchorB - anchorA) + anchorA : r\"\n" \
	"  expr1 \"g >= anchorA && g <= anchorB ? (1.0 - pow(1.0 - (g - anchorA) / (anchorB - anchorA), gamma)) * (anchorB - anchorA) + anchorA : g\"\n" \
	"  expr2 \"b >= anchorA && b <= anchorB ? (1.0 - pow(1.0 - (b - anchorA) / (anchorB - anchorA), gamma)) * (anchorB - anchorA) + anchorA : b\"\n" \
	"  name gammaupper\n" \
	"  xpos -265\n" \
	"  ypos 19\n" \
	" }\n" \
	"push $N2741f900\n" \
	" Expression {\n" \
	"  temp_name0 gamma\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 anchorA\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 anchorB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"r >= anchorA && r <= anchorB ? pow((r - anchorA) / (anchorB - anchorA), 1.0 / gamma) * (anchorB - anchorA) + anchorA : r\"\n" \
	"  expr1 \"g >= anchorA && g <= anchorB ? pow((g - anchorA) / (anchorB - anchorA), 1.0 / gamma) * (anchorB - anchorA) + anchorA : g\"\n" \
	"  expr2 \"b >= anchorA && b <= anchorB ? pow((b - anchorA) / (anchorB - anchorA), 1.0 / gamma) * (anchorB - anchorA) + anchorA : b\"\n" \
	"  name gammalower\n" \
	"  xpos -132\n" \
	"  ypos 20\n" \
	" }\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name upperswitch\n" \
	"  xpos -194\n" \
	"  ypos 74\n" \
	" }\n" \
	" Output {\n" \
	"  name Output1\n" \
	"  xpos -194\n" \
	"  ypos 174\n" \
	" }\n" \
	"end_group\n", gain, gainanchor, lift, liftanchor, offset, gamma, anchorA, anchorB, gamma, anchorA, anchorB, upper);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
	}	
	}
	}
}

void VideoGradePlugin::setupAndProcess(ImageScaler& p_ImageScaler, const OFX::RenderArguments& p_Args)
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

	bool Switch = m_SwitchA->getValueAtTime(p_Args.time);
	float aSwitch = Switch ? 1.0f : 0.0f;
	
    float lScale = m_ScaleL->getValueAtTime(p_Args.time);
    float lAScale = m_ScaleLA->getValueAtTime(p_Args.time);
    float gScale = m_ScaleG->getValueAtTime(p_Args.time);
    float gaScale = m_ScaleGa->getValueAtTime(p_Args.time);
    float gbScale = m_ScaleGb->getValueAtTime(p_Args.time);
    float ggScale = m_ScaleGG->getValueAtTime(p_Args.time);
    float gAScale = m_ScaleGA->getValueAtTime(p_Args.time);
    float oScale = m_ScaleO->getValueAtTime(p_Args.time);

    // Set the images
    p_ImageScaler.setDstImg(dst.get());
    p_ImageScaler.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_ImageScaler.setGPURenderArgs(p_Args);

    // Set the render window
    p_ImageScaler.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_ImageScaler.setScales(aSwitch, lScale, lAScale, gScale, gaScale, gbScale, ggScale, gAScale, oScale);

    // Call the base class process member, this will call the derived templated process code
    p_ImageScaler.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

VideoGradePluginFactory::VideoGradePluginFactory()
    : OFX::PluginFactoryHelper<VideoGradePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void VideoGradePluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
	param->setLabel(p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    //param->setDefault(1.0);
    //param->setRange(-10.0, 10.0);
    //param->setIncrement(0.001);
    //param->setDisplayRange(-5.0, 5.0);
    //param->setDoubleType(eDoubleTypeScale);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void VideoGradePluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

    // Make the four component params
    
    
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scaleL", "Lift", "L from the LGG", 0);
    param->setDefault(0.0);
    param->setRange(-1.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-2.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleLA", "Lift anchor", "adjust anchor point", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleG", "Gamma", "G from the LGG", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("switch1");
    boolParam->setDefault(false);
    boolParam->setHint("higer instead of lower curve bias");
    boolParam->setLabel("Upper Gamma Bias");
    page->addChild(*boolParam);
    
    param = defineScaleParam(p_Desc, "scaleGa", "Gamma boundary start", "adjust", 0);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleGb", "Gamma boundary end", "adjust", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleGG", "Gain", "Double G from the LGG", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleGA", "Gain anchor", "adjust anchor point", 0);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleO", "Offset", "Bonus O", 0);
    param->setDefault(0.0);
    param->setRange(-1.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-1.0, 1.0);
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
	param->setDefault("VideoGrade");
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

ImageEffect* VideoGradePluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new VideoGradePlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static VideoGradePluginFactory VideoGradePlugin;
    p_FactoryArray.push_back(&VideoGradePlugin);
}
