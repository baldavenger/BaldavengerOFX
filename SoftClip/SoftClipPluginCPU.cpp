#include "SoftClipPlugin.h"

#include <cstring>
#include <cmath>
#include <stdio.h>
using std::string;
#include <string> 
#include <fstream>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsParam.h"
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

#define kPluginName "SoftClip"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"SoftClip: Highlight and Shadow roll-off control, with the option of viewing the signal between 1.0 and 2.0, or \n" \
"0.0 and -1.0. Cineon option converts from cineon log to linear, then Rec.709 gamma at the final transform, \n" \
"and LogC Alexa Wide Gamut converts from logC to linear, then Arri Wide Gamut to Rec.709 primaries, and finally \n" \
"Rec.709 gamma at the final transform. All other controls take effect before the final transform." 

#define kPluginIdentifier "OpenFX.Yo.SoftClip"
#define kOfxParamPropChoiceOption "OfxParamPropChoiceOption"
#define kOfxParamPropDefault "OfxParamPropDefault"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 3

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kParamSourceLog "sourceLog"
#define kParamSourceLogLabel "source log"
#define kParamSourceLogHint "Source log signal to be converted"
#define kParamSourceLogOptionBypass "Bypass"
#define kParamSourceLogOptionBypassHint "Soft-Clip Only"
#define kParamSourceLogOptionCineon "Cineon"
#define kParamSourceLogOptionCineonHint "Cineon Log only"
#define kParamSourceLogOptionLogc "LogC Alexa Wide Gamut"
#define kParamSourceLogOptionLogcHint "LogC and Alexa Wide Gamut Primaries"

enum SourceLogEnum {
    eSourceBypass,
    eSourceLogCineon,
    eSourceLogLogc,
};

////////////////////////////////////////////////////////////////////////////////

class SoftClip : public OFX::ImageProcessor
{
public:
    explicit SoftClip(OFX::ImageEffect& p_Instance);

    //virtual void processImagesCUDA();
    //virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
	void setScales(float p_ScaleA, float p_ScaleB, float p_ScaleC, float p_ScaleD, float p_Shadow, float p_Floor,
	float p_SwitchA, float p_SwitchB, float p_SourceA, float p_SourceB, float p_SourceC);

private:
    OFX::Image* _srcImg;
    float _scales[6];
	float _switch[2];
	float _source[3];
};

SoftClip::SoftClip(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}
/*
extern void RunCudaKernel(int p_Width, int p_Height, float* p_SoftClip, float* p_Switch, float* p_Source, const float* p_Input, float* p_Output);

void SoftClip::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, _scales, _switch, _source, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_SoftClip, float* p_Switch, float* p_Source, const float* p_Input, float* p_Output);

void SoftClip::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _scales, _switch, _source, input, output);
}
*/
void SoftClip::multiThreadProcessImages(OfxRectI p_ProcWindow)
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
                float r = srcPix[0];  								
				float g = srcPix[1];  								
				float b = srcPix[2];  								
																	 
				float cr = (pow(10.0f, (1023.0f * r - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);
				float cg = (pow(10.0f, (1023.0f * g - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);
				float cb = (pow(10.0f, (1023.0f * b - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);

				float lr = r > 0.1496582f ? (pow(10.0f, (r - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (r - 0.092809f) / 5.367655f;
				float lg = g > 0.1496582f ? (pow(10.0f, (g - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (g - 0.092809f) / 5.367655f;
				float lb = b > 0.1496582f ? (pow(10.0f, (b - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (b - 0.092809f) / 5.367655f;

				float mr = (lr * 1.617523f)  + (lg * -0.537287f) + (lb * -0.080237f);
				float mg = (lr * -0.070573f) + (lg * 1.334613f)  + (lb * -0.26404f);
				float mb = (lr * -0.021102f) + (lg * -0.226954f) + (lb * 1.248056f);

				float sr = (_source[0] == 1.0f) ? r : ((_source[1] == 1.0f) ? cr : mr);
				float sg = (_source[0] == 1.0f) ? g : ((_source[1] == 1.0f) ? cg : mg);
				float sb = (_source[0] == 1.0f) ? b : ((_source[1] == 1.0f) ? cb : mb);

				float Lr = sr > 1.0f ? 1.0f : sr;
				float Lg = sg > 1.0f ? 1.0f : sg;
				float Lb = sb > 1.0f ? 1.0f : sb;

				float Hr = (sr < 1.0f ? 1.0f : sr) - 1.0f;
				float Hg = (sg < 1.0f ? 1.0f : sg) - 1.0f;
				float Hb = (sb < 1.0f ? 1.0f : sb) - 1.0f;

				float rr = _scales[0];
				float gg = _scales[1];
				float aa = _scales[2];
				float bb = _scales[3];
				float ss = 1.0f - (_scales[4] / 10.0f);
				float sf = 1.0f - _scales[5];

				float Hrr = Hr * pow(2.0f, rr);
				float Hgg = Hg * pow(2.0f, rr);
				float Hbb = Hb * pow(2.0f, rr);

				float HR = Hrr <= 1.0f ? 1.0f - pow(1.0f - Hrr, gg) : Hrr;							
				float HG = Hgg <= 1.0f ? 1.0f - pow(1.0f - Hgg, gg) : Hgg;							
				float HB = Hbb <= 1.0f ? 1.0f - pow(1.0f - Hbb, gg) : Hbb;					
				
				float R = Lr + HR;
				float G = Lg + HG;
				float B = Lb + HB;

				float softr = aa == 1.0f ? R : (R > aa ? (-1.0f / ((R - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : R);
				float softR = bb == 1.0f ? softr : softr > 1.0f - (bb / 50.0f) ? (-1.0f / ((softr - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 
				1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softr;
				float softg = (aa == 1.0f) ? G : (G > aa ? (-1.0f / ((G - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : G);
				float softG = bb == 1.0f ? softg : softg > 1.0f - (bb / 50.0f) ? (-1.0f / ((softg - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 
				1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softg;
				float softb = (aa == 1.0f) ? B : (B > aa ? (-1.0f / ((B - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : B);
				float softB = bb == 1.0f ? softb : softb > 1.0f - (bb / 50.0f) ? (-1.0f / ((softb - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) + 
				1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softb;
				
				float Cr = (softR * -1.0f) + 1.0f;
				float Cg = (softG * -1.0f) + 1.0f;
				float Cb = (softB * -1.0f) + 1.0f;
				
				float cR = ss == 1.0f ? Cr : Cr > ss ? (-1.0f / ((Cr - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cr;
				float CR = sf == 1.0f ? (cR - 1.0f) * -1.0f : ((cR > 1.0f - (-_scales[5] / 50.0f) ? (-1.0f / ((cR - (1.0f - (-_scales[5] / 50.0f))) / 
				(1.0f - (1.0f - (-_scales[5] / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-_scales[5] / 50.0f))) + (1.0f - (-_scales[5] / 50.0f)) : cR) - 1.0f) * -1.0f;
				float cG = ss == 1.0f ? Cg : Cg > ss ? (-1.0f / ((Cg - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cg;
				float CG = sf == 1.0f ? (cG - 1.0f) * -1.0f : ((cG > 1.0f - (-_scales[5] / 50.0f) ? (-1.0f / ((cG - (1.0f - (-_scales[5] / 50.0f))) / 
				(1.0f - (1.0f - (-_scales[5] / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-_scales[5] / 50.0f))) + (1.0f - (-_scales[5] / 50.0f)) : cG) - 1.0f) * -1.0f;
				float cB = ss == 1.0f ? Cb : Cb > ss ? (-1.0f / ((Cb - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cb;
				float CB = sf == 1.0f ? (cB - 1.0f) * -1.0f : ((cB > 1.0f - (-_scales[5] / 50.0f) ? (-1.0f / ((cB - (1.0f - (-_scales[5] / 50.0f))) / 
				(1.0f - (1.0f - (-_scales[5] / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-_scales[5] / 50.0f))) + (1.0f - (-_scales[5] / 50.0f)) : cB) - 1.0f) * -1.0f;
				
				//x > threshold ? (-1 / ((x - threshold) / (1 - threshold) + 1) + 1) * (1 - threshold) + threshold : x

				float SR = _source[0] == 1.0f ? CR : CR >= 0.0f && CR <= 1.0f ? (CR < 0.0181f ? (CR * 4.5f) : 1.0993f * pow(CR, 0.45f) - (1.0993f - 1.0f)) : CR;
				float SG = _source[0] == 1.0f ? CG : CG >= 0.0f && CG <= 1.0f ? (CG < 0.0181f ? (CG * 4.5f) : 1.0993f * pow(CG, 0.45f) - (1.0993f - 1.0f)) : CG;
				float SB = _source[0] == 1.0f ? CB : CB >= 0.0f && CB <= 1.0f ? (CB < 0.0181f ? (CB * 4.5f) : 1.0993f * pow(CB, 0.45f) - (1.0993f - 1.0f)) : CB;
				
				dstPix[0] = _switch[0] == 1.0f ? (SR < 1.0f ? 1.0f : SR) - 1.0f : _switch[1] == 1.0f ? (SR >= 0.0f ? 0.0f : SR + 1.0f) : SR;
				dstPix[1] = _switch[0] == 1.0f ? (SG < 1.0f ? 1.0f : SG) - 1.0f : _switch[1] == 1.0f ? (SG >= 0.0f ? 0.0f : SG + 1.0f) : SG;
				dstPix[2] = _switch[0] == 1.0f ? (SB < 1.0f ? 1.0f : SB) - 1.0f : _switch[1] == 1.0f ? (SB >= 0.0f ? 0.0f : SB + 1.0f) : SB;
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

void SoftClip::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void SoftClip::setScales(float p_ScaleA, float p_ScaleB, float p_ScaleC, float p_ScaleD, float p_Shadow, float p_Floor,
float p_SwitchA, float p_SwitchB, float p_SourceA, float p_SourceB, float p_SourceC)
{
    _scales[0] = p_ScaleA;
    _scales[1] = p_ScaleB;
    _scales[2] = p_ScaleC;
    _scales[3] = p_ScaleD;
    _scales[4] = p_Shadow;
    _scales[5] = p_Floor;
    _switch[0] = p_SwitchA;
    _switch[1] = p_SwitchB;
    _source[0] = p_SourceA;
    _source[1] = p_SourceB;
    _source[2] = p_SourceC;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class SoftClipPlugin : public OFX::ImageEffect
{
public:
    explicit SoftClipPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    
    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(SoftClip &p_SoftClip, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::DoubleParam* m_ScaleA;
    OFX::DoubleParam* m_ScaleB;
    OFX::DoubleParam* m_ScaleC;
    OFX::DoubleParam* m_ScaleD;
    OFX::DoubleParam* m_Shadow;
    OFX::DoubleParam* m_Floor;
    OFX::BooleanParam* m_SwitchA;
    OFX::BooleanParam* m_SwitchB;
    OFX::ChoiceParam* m_SourceLog;
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Path2;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
	OFX::PushButtonParam* m_Button3;
};

SoftClipPlugin::SoftClipPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_ScaleA = fetchDoubleParam("scaleA");
    m_ScaleB = fetchDoubleParam("scaleB");
    m_ScaleC = fetchDoubleParam("scaleC");
    m_ScaleD = fetchDoubleParam("scaleD");
    m_Shadow = fetchDoubleParam("shadow");
    m_Floor = fetchDoubleParam("floor");
    m_SwitchA = fetchBooleanParam("highlight");
    m_SwitchB = fetchBooleanParam("displayShadow");
    m_SourceLog = fetchChoiceParam("sourceLog");
    m_Path = fetchStringParam("path");
    m_Path2 = fetchStringParam("path2");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");
	m_Button3 = fetchPushButtonParam("button3");
   
}

void SoftClipPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        SoftClip SoftClip(*this);
        setupAndProcess(SoftClip, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool SoftClipPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
   
    double aScale = m_ScaleA->getValueAtTime(p_Args.time);
    double bScale = m_ScaleB->getValueAtTime(p_Args.time);
    double cScale = m_ScaleC->getValueAtTime(p_Args.time);
    double dScale = m_ScaleD->getValueAtTime(p_Args.time);
    
    
    int sourceLog_i;
    m_SourceLog->getValueAtTime(p_Args.time, sourceLog_i);
    SourceLogEnum sourceLog = (SourceLogEnum)sourceLog_i;
    
    bool aSource = sourceLog_i == 0;
    
     
    
    if ((aScale == 1.0) && (bScale == 1.0) && (cScale == 1.0) && (dScale == 1.0) && (aSource))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void SoftClipPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
 
 	if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if (p_ParamName == "highlight") {
	
        bool hideShadow = m_SwitchA->getValueAtTime(p_Args.time);
        m_SwitchB->setIsSecretAndDisabled(hideShadow);
    }
    
    if (p_ParamName == "displayShadow") {
	
        bool hideHighlight = m_SwitchB->getValueAtTime(p_Args.time);
        m_SwitchA->setIsSecretAndDisabled(hideHighlight);
    }
	
	if(p_ParamName == "button1")
    {
       
    float compress = m_ScaleA->getValueAtTime(p_Args.time);
    float redistribute = m_ScaleB->getValueAtTime(p_Args.time);
    float highlight = m_ScaleC->getValueAtTime(p_Args.time);
    float ceiling = m_ScaleD->getValueAtTime(p_Args.time);
    float shadow = m_Shadow->getValueAtTime(p_Args.time);
    float floor = m_Floor->getValueAtTime(p_Args.time);
    
    int sourceLog_i;
    m_SourceLog->getValueAtTime(p_Args.time, sourceLog_i);
    SourceLogEnum sourceLog = (SourceLogEnum)sourceLog_i;
    
    bool aSource = sourceLog_i == 0;
    bool bSource = sourceLog_i == 1;
    bool cSource = sourceLog_i == 2;

	int bypass = aSource ? 1 : 0;
	int cineon = bSource ? 1 : 0;
	int logC = cSource ? 1 : 0;
	
	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// SoftClipPlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"\n" \
	"  	// Choose bypass, cineon, or logC (when cineon bool = false)\n" \
	"	int bypass = %d;\n" \
	"	int cineon = %d;\n" \
	"	bool Bypass = bypass == 1;\n" \
	"	bool Cineon = cineon == 1;\n" \
	"\n" \
	"	// Highlight, Soft-Clip, and Shadow parameters\n" \
	"	float HighlightCompress = %ff;\n" \
	"	float HighlightRedistribute = %ff;\n" \
	"	float HighSoftClip = %ff;\n" \
	"	float HighSoftClipCeiling = %ff;\n" \
	"	float ShadowSoftClip = %ff;\n" \
	"	float ShadowSoftClipFloor = %ff;\n" \
	"\n" \
	"	float cr = (_powf(10.0f, (1023.0f * p_R - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);\n" \
	"	float cg = (_powf(10.0f, (1023.0f * p_G - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);\n" \
	"	float cb = (_powf(10.0f, (1023.0f * p_B - 685.0f) / 300.0f) - 0.0108f) / (1.0f - 0.0108f);\n" \
	"\n" \
	"	float lr = p_R > 0.1496582f ? (_powf(10.0f, (p_R - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (p_R - 0.092809f) / 5.367655f;\n" \
	"	float lg = p_G > 0.1496582f ? (_powf(10.0f, (p_G - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (p_G - 0.092809f) / 5.367655f;\n" \
	"	float lb = p_B > 0.1496582f ? (_powf(10.0f, (p_B - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f : (p_B - 0.092809f) / 5.367655f;\n" \
	"\n" \
	"	float mr = (lr * 1.617523f) + (lg * -0.537287f) + (lb * -0.080237f);\n" \
	"	float mg = (lr * -0.070573f) + (lg * 1.334613f) + (lb * -0.26404f);\n" \
	"	float mb = (lr * -0.021102f) + (lg * -0.226954f) + (lb * 1.248056f);\n" \
	"\n" \
	"	float sr = Bypass ? p_R : (Cineon ? cr : mr);\n" \
	"	float sg = Bypass ? p_G : (Cineon ? cg : mg);\n" \
	"	float sb = Bypass ? p_B : (Cineon ? cb : mb);\n" \
	"\n" \
	"	float Lr = sr > 1.0f ? 1.0f : sr;\n" \
	"	float Lg = sg > 1.0f ? 1.0f : sg;\n" \
	"	float Lb = sb > 1.0f ? 1.0f : sb;\n" \
	"\n" \
	"	float Hr = (sr < 1.0f ? 1.0f : sr) - 1.0f;\n" \
	"	float Hg = (sg < 1.0f ? 1.0f : sg) - 1.0f;\n" \
	"	float Hb = (sb < 1.0f ? 1.0f : sb) - 1.0f;\n" \
	"\n" \
	"	float rr = HighlightCompress;\n" \
	"	float gg = HighlightRedistribute;\n" \
	"	float aa = HighSoftClip;\n" \
	"	float bb = HighSoftClipCeiling;\n" \
	"	float ss = 1.0f - (ShadowSoftClip / 10.0f);\n" \
	"	float sf = 1.0f - ShadowSoftClipFloor;\n" \
	"\n" \
	"	float Hrr = Hr * _powf(2.0f, rr);\n" \
	"	float Hgg = Hg * _powf(2.0f, rr);\n" \
	"	float Hbb = Hb * _powf(2.0f, rr);\n" \
	"	float HR = Hrr <= 1.0f ? 1.0f - _powf(1.0f - Hrr, gg) : Hrr;\n" \
	"	float HG = Hgg <= 1.0f ? 1.0f - _powf(1.0f - Hgg, gg) : Hgg;\n" \
	"	float HB = Hbb <= 1.0f ? 1.0f - _powf(1.0f - Hbb, gg) : Hbb;\n" \
	"\n" \
	"	float R = Lr + HR;\n" \
	"	float G = Lg + HG;\n" \
	"	float B = Lb + HB;\n" \
	"\n" \
	"	float softr = aa == 1.0f ? R : (R > aa ? (-1.0f / ((R - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : R); \n" \
	"	float softR = bb == 1.0f ? softr : softr > 1.0f - (bb / 50.0f) ? (-1.0f / ((softr - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) +  \n" \
	"	1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softr; \n" \
	"	float softg = (aa == 1.0f) ? G : (G > aa ? (-1.0f / ((G - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : G); \n" \
	"	float softG = bb == 1.0f ? softg : softg > 1.0f - (bb / 50.0f) ? (-1.0f / ((softg - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) +  \n" \
	"	1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softg; \n" \
	"	float softb = (aa == 1.0f) ? B : (B > aa ? (-1.0f / ((B - aa) / (bb - aa) + 1.0f) + 1.0f) * (bb - aa) + aa : B); \n" \
	"	float softB = bb == 1.0f ? softb : softb > 1.0f - (bb / 50.0f) ? (-1.0f / ((softb - (1.0f - (bb / 50.0f))) / (1.0f - (1.0f - (bb / 50.0f))) + 1.0f) +  \n" \
	"	1.0f) * (1.0f - (1.0f - (bb / 50.0f))) + (1.0f - (bb / 50.0f)) : softb; \n" \
	"\n" \
	"	float Cr = (softR * -1.0f) + 1.0f; \n" \
	"	float Cg = (softG * -1.0f) + 1.0f; \n" \
	"	float Cb = (softB * -1.0f) + 1.0f; \n" \
	"\n" \
	"	float cR = ss == 1.0f ? Cr : Cr > ss ? (-1.0f / ((Cr - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cr; \n" \
	"	float CR = sf == 1.0f ? (cR - 1.0f) * -1.0f : ((cR > 1.0f - (-ShadowSoftClipFloor / 50.0f) ? (-1.0f / ((cR - (1.0f - (-ShadowSoftClipFloor / 50.0f))) /  \n" \
	"	(1.0f - (1.0f - (-ShadowSoftClipFloor / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-ShadowSoftClipFloor / 50.0f))) + (1.0f - (-ShadowSoftClipFloor / 50.0f)) : cR) - 1.0f) * -1.0f; \n" \
	"	float cG = ss == 1.0f ? Cg : Cg > ss ? (-1.0f / ((Cg - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cg; \n" \
	"	float CG = sf == 1.0f ? (cG - 1.0f) * -1.0f : ((cG > 1.0f - (-ShadowSoftClipFloor / 50.0f) ? (-1.0f / ((cG - (1.0f - (-ShadowSoftClipFloor / 50.0f))) /  \n" \
	"	(1.0f - (1.0f - (-ShadowSoftClipFloor / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-ShadowSoftClipFloor / 50.0f))) + (1.0f - (-ShadowSoftClipFloor / 50.0f)) : cG) - 1.0f) * -1.0f; \n" \
	"	float cB = ss == 1.0f ? Cb : Cb > ss ? (-1.0f / ((Cb - ss) / (sf - ss) + 1.0f) + 1.0f) * (sf - ss) + ss : Cb; \n" \
	"	float CB = sf == 1.0f ? (cB - 1.0f) * -1.0f : ((cB > 1.0f - (-ShadowSoftClipFloor / 50.0f) ? (-1.0f / ((cB - (1.0f - (-ShadowSoftClipFloor / 50.0f))) /  \n" \
	"	(1.0f - (1.0f - (-ShadowSoftClipFloor / 50.0f))) + 1.0f) + 1.0f) * (1.0f - (1.0f - (-ShadowSoftClipFloor / 50.0f))) + (1.0f - (-ShadowSoftClipFloor / 50.0f)) : cB) - 1.0f) * -1.0f; \n" \
	"\n" \
	"	float SR = Bypass ? CR : CR >= 0.0f && CR <= 1.0f ? (CR < 0.0181f ? (CR * 4.5f) : 1.0993f * _powf(CR, 0.45f) - (1.0993f - 1.0f)) : CR; \n" \
	"	float SG = Bypass ? CG : CG >= 0.0f && CG <= 1.0f ? (CG < 0.0181f ? (CG * 4.5f) : 1.0993f * _powf(CG, 0.45f) - (1.0993f - 1.0f)) : CG; \n" \
	"	float SB = Bypass ? CB : CB >= 0.0f && CB <= 1.0f ? (CB < 0.0181f ? (CB * 4.5f) : 1.0993f * _powf(CB, 0.45f) - (1.0993f - 1.0f)) : CB; \n" \
	"\n" \
	"	const float r = SR;\n" \
	"	const float g = SG;\n" \
	"	const float b = SB;\n" \
	"\n" \
	"    return make_float3(r, g, b);\n" \
	"}\n", bypass, cineon, compress, redistribute, highlight, ceiling, shadow, floor);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}

	if(p_ParamName == "button2")
    {
	
	float compress = m_ScaleA->getValueAtTime(p_Args.time);
    float redistribute = m_ScaleB->getValueAtTime(p_Args.time);
    float highlight = m_ScaleC->getValueAtTime(p_Args.time);
    float ceiling = m_ScaleD->getValueAtTime(p_Args.time);
    float shadow = m_Shadow->getValueAtTime(p_Args.time);
    float floor = m_Floor->getValueAtTime(p_Args.time);
        
    int sourceLog_i;
    m_SourceLog->getValueAtTime(p_Args.time, sourceLog_i);
    SourceLogEnum sourceLog = (SourceLogEnum)sourceLog_i;
    
    int source = sourceLog_i;
    
    string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".nk to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".nk").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "Group { \n" \
	" inputs 0 \n" \
	" name SoftClip \n" \
	" xpos -102 \n" \
	" ypos 63 \n" \
	"} \n" \
	" Input { \n" \
	"  inputs 0 \n" \
	"  name Input1 \n" \
	"  xpos -230 \n" \
	"  ypos -158 \n" \
	" } \n" \
	"set N92f92230 [stack 0] \n" \
	" Expression { \n" \
	"  expr0 \"r > 0.1496582 ? (pow(10, (r - 0.385537) / 0.2471896) - 0.052272) / 5.555556 : (r - 0.092809) / 5.367655\" \n" \
	"  expr1 \"g > 0.1496582 ? (pow(10, (g - 0.385537) / 0.2471896) - 0.052272) / 5.555556 : (g - 0.092809) / 5.367655\" \n" \
	"  expr2 \"b > 0.1496582 ? (pow(10, (b - 0.385537) / 0.2471896) - 0.052272) / 5.555556 : (b - 0.092809) / 5.367655\" \n" \
	"  name LogC_to_lin \n" \
	"  xpos -140 \n" \
	"  ypos -125 \n" \
	" } \n" \
	" Expression { \n" \
	"  expr0 \"r * 1.617523  + g * -0.537287f + b * -0.080237\" \n" \
	"  expr1 \"r * -0.070573 + g * 1.334613  + b * -0.26404\" \n" \
	"  expr2 \"r * -0.021102 + g * -0.226954 + b * 1.248056\" \n" \
	"  name AWG_to_Rec709 \n" \
	"  xpos -140 \n" \
	"  ypos -93 \n" \
	" } \n" \
	"push $N92f92230 \n" \
	" Expression { \n" \
	"  expr0 \"(pow(10.0, (1023 * r - 685) / 300) - 0.0108) / (1.0 - 0.0108)\" \n" \
	"  expr1 \"(pow(10.0, (1023 * g - 685) / 300) - 0.0108) / (1.0 - 0.0108)\" \n" \
	"  expr2 \"(pow(10.0, (1023 * b - 685) / 300) - 0.0108) / (1.0 - 0.0108)\" \n" \
	"  name Cineon_to_lin \n" \
	"  xpos -311 \n" \
	"  ypos -114 \n" \
	" } \n" \
	"push $N92f92230 \n" \
	" Switch { \n" \
	"  inputs 3 \n" \
	"  which %d \n" \
	"  name Source \n" \
	"  xpos -230 \n" \
	"  ypos -56 \n" \
	" } \n" \
	"set N92fae270 [stack 0] \n" \
	" Expression { \n" \
	"  expr0 \"r > 1.0 ? 1.0 : r\" \n" \
	"  expr1 \"g > 1.0 ? 1.0 : g\" \n" \
	"  expr2 \"b > 1.0 ? 1.0 : b\" \n" \
	"  name highlightsL \n" \
	"  xpos -303 \n" \
	"  ypos -19 \n" \
	" } \n" \
	"push $N92fae270 \n" \
	" Expression { \n" \
	"  expr0 \"(r < 1.0 ? 1.0 : r) - 1.0\" \n" \
	"  expr1 \"(g < 1.0 ? 1.0 : g) - 1.0\" \n" \
	"  expr2 \"(b < 1.0 ? 1.0 : b) - 1.0\" \n" \
	"  name highlightsH \n" \
	"  xpos -168 \n" \
	"  ypos -22 \n" \
	" } \n" \
	" Expression { \n" \
	"  temp_name0 compress \n" \
	"  temp_expr0 \"pow(2, %f)\" \n" \
	"  temp_name1 power \n" \
	"  temp_expr1 %f \n" \
	"  expr0 \"r * compress <= 1.0 ? 1.0 - pow(1.0 - (r * compress), power) : r * compress\" \n" \
	"  expr1 \"g * compress <= 1.0 ? 1.0 - pow(1.0 - (g * compress), power) : g * compress\" \n" \
	"  expr2 \"b * compress <= 1.0 ? 1.0 - pow(1.0 - (b * compress), power) : b * compress\" \n" \
	"  name highlight_compress \n" \
	"  xpos -168 \n" \
	"  ypos 15 \n" \
	" } \n" \
	" MergeExpression { \n" \
	"  inputs 2 \n" \
	"  expr0 \"Ar + Br\" \n" \
	"  expr1 \"Ag + Bg\" \n" \
	"  expr2 \"Ab + Bb\" \n" \
	"  name Signal_recombined \n" \
	"  xpos -232 \n" \
	"  ypos 58 \n" \
	" } \n" \
	" Expression { \n" \
	"  temp_name0 min \n" \
	"  temp_expr0 %f \n" \
	"  temp_name1 max \n" \
	"  temp_expr1 %f \n" \
	"  expr0 \"(min == 1.0) ? r : (r > min ? (-1.0 / ((r - min) / (max - min) + 1.0) + 1.0) * (max - min) + min : r)\" \n" \
	"  expr1 \"(min == 1.0) ? g : (g > min ? (-1.0 / ((g - min) / (max - min) + 1.0) + 1.0) * (max - min) + min : g)\" \n" \
	"  expr2 \"(min == 1.0) ? b : (b > min ? (-1.0 / ((b - min) / (max - min) + 1.0) + 1.0) * (max - min) + min : b)\" \n" \
	"  name highlight_softclipA \n" \
	"  xpos -232 \n" \
	"  ypos 89 \n" \
	" } \n" \
	" Expression { \n" \
	"  temp_name0 max \n" \
	"  temp_expr0 %f \n" \
	"  expr0 \"(max == 1.0) ? r : r > 1.0 - (max / 50.0) ? (-1.0 / ((r - (1.0 - (max / 50.0))) / (1.0 - (1.0 - (max / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (max / 50.0))) + (1.0 - (max / 50.0)) : r\" \n" \
	"  expr1 \"(max == 1.0) ? g : g > 1.0 - (max / 50.0) ? (-1.0 / ((g - (1.0 - (max / 50.0))) / (1.0 - (1.0 - (max / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (max / 50.0))) + (1.0 - (max / 50.0)) : g\" \n" \
	"  expr2 \"(max == 1.0) ? b : b > 1.0 - (max / 50.0) ? (-1.0 / ((b - (1.0 - (max / 50.0))) / (1.0 - (1.0 - (max / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (max / 50.0))) + (1.0 - (max / 50.0)) : b\" \n" \
	"  name highlight_softclipB \n" \
	"  xpos -232 \n" \
	"  ypos 128 \n" \
	" } \n" \
	" Expression { \n" \
	"  expr0 \"(r * -1.0) + 1.0\" \n" \
	"  expr1 \"(g * -1.0) + 1.0\" \n" \
	"  expr2 \"(b * -1.0) + 1.0\" \n" \
	"  name shadow_softclipA \n" \
	"  xpos -233 \n" \
	"  ypos 162 \n" \
	" } \n" \
	" Expression { \n" \
	"  temp_name0 shadow \n" \
	"  temp_expr0 %f \n" \
	"  temp_name1 floor \n" \
	"  temp_expr1 %f \n" \
	"  temp_name2 ss \n" \
	"  temp_expr2 \"1.0 - (shadow / 10.0)\" \n" \
	"  temp_name3 sf \n" \
	"  temp_expr3 \"1.0 - (floor)\" \n" \
	"  expr0 \"(ss == 1.0) ? r : r > ss ? (-1.0 / ((r - ss) / (sf - ss) + 1.0) + 1.0) * (sf - ss) + ss : r\" \n" \
	"  expr1 \"(ss == 1.0) ? g : g > ss ? (-1.0 / ((g - ss) / (sf - ss) + 1.0) + 1.0) * (sf - ss) + ss : g\" \n" \
	"  expr2 \"(ss == 1.0) ? b : b > ss ? (-1.0 / ((b - ss) / (sf - ss) + 1.0) + 1.0) * (sf - ss) + ss : b\" \n" \
	"  name shadow_softclipB \n" \
	"  xpos -233 \n" \
	"  ypos 192 \n" \
	" } \n" \
	" Expression { \n" \
	"  temp_name0 floor \n" \
	"  temp_expr0 %f \n" \
	"  temp_name1 sf \n" \
	"  temp_expr1 \"1.0 - floor\" \n" \
	"  expr0 \"(sf == 1.0) ? (r - 1.0) * -1.0 : ((r > 1.0 - (-floor / 50.0) ? (-1.0 / ((r - (1.0 - (-floor / 50.0))) /\n(1.0 - (1.0 - (-floor / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (-floor / 50.0))) + (1.0 - (-floor / 50.0)) : r) - 1.0) * -1.0\" \n" \
	"  expr1 \"(sf == 1.0) ? (g - 1.0) * -1.0 : ((g > 1.0 - (-floor / 50.0) ? (-1.0 / ((g - (1.0 - (-floor / 50.0))) /\n(1.0 - (1.0 - (-floor / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (-floor / 50.0))) + (1.0 - (-floor / 50.0)) : g) - 1.0) * -1.0\" \n" \
	"  expr2 \"(sf == 1.0) ? (b - 1.0) * -1.0 : ((b > 1.0 - (-floor / 50.0) ? (-1.0 / ((b - (1.0 - (-floor / 50.0))) /\n(1.0 - (1.0 - (-floor / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (-floor / 50.0))) + (1.0 - (-floor / 50.0)) : b) - 1.0) * -1.0\" \n" \
	"  name shadow_softclipC \n" \
	"  xpos -233 \n" \
	"  ypos 225 \n" \
	" } \n" \
	"set N956dacf0 [stack 0] \n" \
	" Expression { \n" \
	"  expr0 \"r >= 0.0 && r <= 1.0 ? (r < 0.018 ? r * 4.5 : 1.099 * pow(r, 0.45) - 0.099) : r\" \n" \
	"  expr1 \"g >= 0.0 && g <= 1.0 ? (g < 0.018 ? g * 4.5 : 1.099 * pow(g, 0.45) - 0.099) : g\" \n" \
	"  expr2 \"b >= 0.0 && b <= 1.0 ? (b < 0.018 ? b * 4.5 : 1.099 * pow(b, 0.45) - 0.099) : b\" \n" \
	"  name Rec709 \n" \
	"  xpos -287 \n" \
	"  ypos 252 \n" \
	" } \n" \
	"set N92fd6840 [stack 0] \n" \
	" Dot { \n" \
	"  name Dot2 \n" \
	"  xpos -294 \n" \
	"  ypos 291 \n" \
	" } \n" \
	"push $N92fd6840 \n" \
	" Dot { \n" \
	"  name Dot1 \n" \
	"  xpos -253 \n" \
	"  ypos 285 \n" \
	" } \n" \
	"push $N956dacf0 \n" \
	" Switch { \n" \
	"  inputs 3 \n" \
	"  which %d \n" \
	"  name Log_Bypass_Switch \n" \
	"  xpos -233 \n" \
	"  ypos 308 \n" \
	" } \n" \
	" Output { \n" \
	"  name Output1 \n" \
	"  xpos -232 \n" \
	"  ypos 344 \n" \
	" } \n" \
	" end_group\n", source, compress, redistribute, highlight, ceiling, ceiling, shadow, floor, floor, source);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
	}	
	}
	}
	
	if(p_ParamName == "button3")
    {
	
	float compress = m_ScaleA->getValueAtTime(p_Args.time);
    float redistribute = m_ScaleB->getValueAtTime(p_Args.time);
    float highlight = m_ScaleC->getValueAtTime(p_Args.time);
    float ceiling = m_ScaleD->getValueAtTime(p_Args.time);
    float shadow = m_Shadow->getValueAtTime(p_Args.time);
    float floor = m_Floor->getValueAtTime(p_Args.time);
        
    int sourceLog_i;
    m_SourceLog->getValueAtTime(p_Args.time, sourceLog_i);
    SourceLogEnum sourceLog = (SourceLogEnum)sourceLog_i;
    
    int source = sourceLog_i;
    
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
	"// SoftClip Shader \n" \
	" \n" \
	"#version 120 \n" \
	"uniform sampler2D front; \n" \
	"uniform float adsk_result_w, adsk_result_h; \n" \
	"uniform float p_SoftClipA; \n" \
	"uniform float p_SoftClipB; \n" \
	"uniform float p_SoftClipC; \n" \
	"uniform float p_SoftClipD; \n" \
	"uniform float p_SoftClipE; \n" \
	"uniform float p_SoftClipF; \n" \
	"uniform int p_Source; \n" \
	"uniform bool p_SwitchA; \n" \
	"uniform bool p_SwitchB; \n" \
	" \n" \
	"vec3 from_Cineon(vec3 col) \n" \
	"{ \n" \
	"	col.r = (pow(10.0, (1023.0 * col.r - 685.0) / 300.0) - 0.0108) / (1.0 - 0.0108); \n" \
	"	col.g = (pow(10.0, (1023.0 * col.g - 685.0) / 300.0) - 0.0108) / (1.0 - 0.0108); \n" \
	"	col.b = (pow(10.0, (1023.0 * col.b - 685.0) / 300.0) - 0.0108) / (1.0 - 0.0108); \n" \
	"	 \n" \
	"	return col; \n" \
	"} \n" \
	" \n" \
	"vec3 from_logc(vec3 col) \n" \
	"{ \n" \
	"    float r = col.r > 0.1496582 ? (pow(10.0, (col.r - 0.385537) / 0.2471896) - 0.052272) / 5.555556 : (col.r - 0.092809) / 5.367655; \n" \
	"    float g = col.g > 0.1496582 ? (pow(10.0, (col.g - 0.385537) / 0.2471896) - 0.052272) / 5.555556 : (col.g - 0.092809) / 5.367655; \n" \
	"    float b = col.b > 0.1496582 ? (pow(10.0, (col.b - 0.385537) / 0.2471896) - 0.052272) / 5.555556 : (col.b - 0.092809) / 5.367655; \n" \
	"     \n" \
	"    col.r = r * 1.617523 + g * -0.537287 + b * -0.080237; \n" \
	"	col.g = r * -0.070573 + g * 1.334613 + b * -0.26404; \n" \
	"	col.b = r * -0.021102 + g * -0.226954 + b * 1.248056; \n" \
	" \n" \
	"    return col; \n" \
	"} \n" \
	" \n" \
	"vec3 to_sRGB(vec3 col) \n" \
	"{ \n" \
	"    if (col.r >= 0.0 && col.r <= 1.0) { \n" \
	"         col.r = (1.055 * pow(col.r, 1.0 / 2.4)) - .055; \n" \
	"    } \n" \
	" \n" \
	"    if (col.g >= 0.0 && col.g <= 1.0) { \n" \
	"         col.g = (1.055 * pow(col.g, 1.0 / 2.4)) - .055; \n" \
	"    } \n" \
	" \n" \
	"    if (col.b >= 0.0 && col.b <= 1.0) { \n" \
	"         col.b = (1.055 * pow(col.b, 1.0 / 2.4)) - .055; \n" \
	"    } \n" \
	" \n" \
	"    return col; \n" \
	"} \n" \
	" \n" \
	" \n" \
	"void main(void) \n" \
	"{ \n" \
	"vec2 uv = gl_FragCoord.xy / vec2( adsk_result_w, adsk_result_h); \n" \
	"vec3 COL = texture2D(front, uv).rgb; \n" \
	"	 \n" \
	"if (p_Source == 1) { \n" \
	"            COL = from_Cineon(COL); \n" \
	"        } else if (p_Source == 2) { \n" \
	"            COL = from_logc(COL); \n" \
	"        } \n" \
	"         \n" \
	"float Lr = COL.r > 1.0 ? 1.0 : COL.r; \n" \
	"float Lg = COL.g > 1.0 ? 1.0 : COL.g; \n" \
	"float Lb = COL.b > 1.0 ? 1.0 : COL.b; \n" \
	" \n" \
	"float Hr = (COL.r < 1.0 ? 1.0 : COL.r) - 1.0; \n" \
	"float Hg = (COL.g < 1.0 ? 1.0 : COL.g) - 1.0; \n" \
	"float Hb = (COL.b < 1.0 ? 1.0 : COL.b) - 1.0; \n" \
	" \n" \
	"float rr = p_SoftClipA; \n" \
	"float gg = p_SoftClipB; \n" \
	"float aa = p_SoftClipC; \n" \
	"float bb = p_SoftClipD; \n" \
	"float ss = 1.0 - (p_SoftClipE / 10.0); \n" \
	"float sf = 1.0 - p_SoftClipF; \n" \
	" \n" \
	"float Hrr = Hr * pow(2.0, rr); \n" \
	"float Hgg = Hg * pow(2.0, rr); \n" \
	"float Hbb = Hb * pow(2.0, rr); \n" \
	" \n" \
	"float HR = Hrr <= 1.0 ? 1.0 - pow(1.0 - Hrr, gg) : Hrr; \n" \
	"float HG = Hgg <= 1.0 ? 1.0 - pow(1.0 - Hgg, gg) : Hgg; \n" \
	"float HB = Hbb <= 1.0 ? 1.0 - pow(1.0 - Hbb, gg) : Hbb; \n" \
	" \n" \
	"float R = Lr + HR; \n" \
	"float G = Lg + HG; \n" \
	"float B = Lb + HB; \n" \
	" \n" \
	"float softr = aa == 1.0 ? R : (R > aa ? (-1.0 / ((R - aa) / (bb - aa) + 1.0) + 1.0) * (bb - aa) + aa : R); \n" \
	"float softR = bb == 1.0 ? softr : softr > 1.0 - (bb / 50.0) ? (-1.0 / ((softr - (1.0 - (bb / 50.0))) / (1.0 - (1.0 - (bb / 50.0))) + 1.0) +  \n" \
	"1.0) * (1.0 - (1.0 - (bb / 50.0))) + (1.0 - (bb / 50.0)) : softr; \n" \
	"float softg = (aa == 1.0) ? G : (G > aa ? (-1.0 / ((G - aa) / (bb - aa) + 1.0) + 1.0) * (bb - aa) + aa : G); \n" \
	"float softG = bb == 1.0 ? softg : softg > 1.0 - (bb / 50.0) ? (-1.0 / ((softg - (1.0 - (bb / 50.0))) / (1.0 - (1.0 - (bb / 50.0))) + 1.0) +  \n" \
	"1.0) * (1.0 - (1.0 - (bb / 50.0))) + (1.0 - (bb / 50.0)) : softg; \n" \
	"float softb = (aa == 1.0) ? B : (B > aa ? (-1.0 / ((B - aa) / (bb - aa) + 1.0) + 1.0) * (bb - aa) + aa : B); \n" \
	"float softB = bb == 1.0 ? softb : softb > 1.0 - (bb / 50.0) ? (-1.0 / ((softb - (1.0 - (bb / 50.0))) / (1.0 - (1.0 - (bb / 50.0))) + 1.0) +  \n" \
	"1.0) * (1.0 - (1.0 - (bb / 50.0))) + (1.0 - (bb / 50.0)) : softb; \n" \
	" \n" \
	"float Cr = (softR * -1.0) + 1.0; \n" \
	"float Cg = (softG * -1.0) + 1.0; \n" \
	"float Cb = (softB * -1.0) + 1.0; \n" \
	" \n" \
	"float cR = ss == 1.0 ? Cr : Cr > ss ? (-1.0 / ((Cr - ss) / (sf - ss) + 1.0) + 1.0) * (sf - ss) + ss : Cr; \n" \
	"COL.r = sf == 1.0 ? (cR - 1.0) * -1.0 : ((cR > 1.0 - (-p_SoftClipF / 50.0) ? (-1.0 / ((cR - (1.0 - (-p_SoftClipF / 50.0))) /  \n" \
	"(1.0 - (1.0 - (-p_SoftClipF / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (-p_SoftClipF / 50.0))) + (1.0 - (-p_SoftClipF / 50.0)) : cR) - 1.0) * -1.0; \n" \
	"float cG = ss == 1.0 ? Cg : Cg > ss ? (-1.0 / ((Cg - ss) / (sf - ss) + 1.0) + 1.0) * (sf - ss) + ss : Cg; \n" \
	"COL.g = sf == 1.0 ? (cG - 1.0) * -1.0 : ((cG > 1.0 - (-p_SoftClipF / 50.0) ? (-1.0 / ((cG - (1.0 - (-p_SoftClipF / 50.0))) /  \n" \
	"(1.0 - (1.0 - (-p_SoftClipF / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (-p_SoftClipF / 50.0))) + (1.0 - (-p_SoftClipF / 50.0)) : cG) - 1.0) * -1.0; \n" \
	"float cB = ss == 1.0 ? Cb : Cb > ss ? (-1.0 / ((Cb - ss) / (sf - ss) + 1.0) + 1.0) * (sf - ss) + ss : Cb; \n" \
	"COL.b = sf == 1.0 ? (cB - 1.0) * -1.0 : ((cB > 1.0 - (-p_SoftClipF / 50.0) ? (-1.0 / ((cB - (1.0 - (-p_SoftClipF / 50.0))) /  \n" \
	"(1.0 - (1.0 - (-p_SoftClipF / 50.0))) + 1.0) + 1.0) * (1.0 - (1.0 - (-p_SoftClipF / 50.0))) + (1.0 - (-p_SoftClipF / 50.0)) : cB) - 1.0) * -1.0; \n" \
	" \n" \
	"COL = p_Source == 0 ? COL : to_sRGB(COL); \n" \
	" \n" \
	"if (p_SwitchA) { \n" \
	"COL.r = (COL.r < 1.0 ? 1.0 : COL.r) - 1.0; \n" \
	"COL.g = (COL.g < 1.0 ? 1.0 : COL.g) - 1.0; \n" \
	"COL.b = (COL.b < 1.0 ? 1.0 : COL.b) - 1.0; \n" \
	"}  \n" \
	"else if (p_SwitchB) { \n" \
	"COL.r = COL.r >= 0.0 ? 0.0 : COL.r + 1.0; \n" \
	"COL.g = COL.g >= 0.0 ? 0.0 : COL.g + 1.0; \n" \
	"COL.b = COL.b >= 0.0 ? 0.0 : COL.b + 1.0; \n" \
	"} \n" \
	" \n" \
	"gl_FragColor = vec4(COL, 1.0); \n" \
	"} \n");
	fclose (pFile);
	fprintf (pFile2,
	"<ShaderNodePreset SupportsAdaptiveDegradation=\"0\" Description=\"SoftClip\" Name=\"SoftClip\">\n" \
	"<Shader OutputBitDepth=\"Output\" Index=\"1\">\n" \
	"<Uniform Index=\"0\" NoInput=\"Error\" Tooltip=\"\" DisplayName=\"input\" Mipmaps=\"False\" GL_TEXTURE_WRAP_T=\"GL_REPEAT\" GL_TEXTURE_WRAP_S=\"GL_REPEAT\" GL_TEXTURE_MAG_FILTER=\"GL_NEAREST\" GL_TEXTURE_MIN_FILTER=\"GL_NEAREST\" Type=\"sampler2D\" Name=\"front\">\n" \
	"</Uniform>\n" \
	"<Uniform Max=\"3\" Min=\"0\" Default=\"%d\" Inc=\"1\" Tooltip=\"Working colorspace\" Row=\"0\" Col=\"0\" Page=\"0\" DisplayName=\"Source\" Type=\"int\" Name=\"p_Source\" ValueType=\"Popup\">\n" \
	"<PopupEntry Title=\"Bypass\" Value=\"0\">\n" \
	"</PopupEntry>\n" \
	"<PopupEntry Title=\"Cineon\" Value=\"1\">\n" \
	"</PopupEntry>\n" \
	"<PopupEntry Title=\"LogC V3 WG\" Value=\"2\">\n" \
	"</PopupEntry>\n" \
	"</Uniform>\n" \
	"<Uniform Row=\"1\" Col=\"0\" Page=\"0\" Default=\"False\" Tooltip=\"\" Type=\"bool\" DisplayName=\"Display Highlights\" Name=\"p_SwitchA\">\n" \
	"</Uniform>\n" \
	"<Uniform ResDependent=\"None\" Max=\"0.0\" Min=\"-8.0\" Default=\"%f\" Inc=\"0.001\" Tooltip=\"\" Row=\"4\" Col=\"0\" Page=\"0\" DisplayName=\"Highlight Compress\" Type=\"float\" Name=\"p_SoftClipA\">\n" \
	"</Uniform>\n" \
	"<Uniform ResDependent=\"None\" Max=\"10.0\" Min=\"0.0\" Default=\"%f\" Inc=\"0.001\" Tooltip=\"\" Row=\"5\" Col=\"0\" Page=\"0\" DisplayName=\"Redistribute\" Type=\"float\" Name=\"p_SoftClipB\">\n" \
	"</Uniform>\n" \
	"<Uniform ResDependent=\"None\" Max=\"1.0\" Min=\"0.0\" Default=\"%f\" Inc=\"0.001\" Tooltip=\"\" Row=\"2\" Col=\"0\" Page=\"0\" DisplayName=\"Highlight SoftClip Min\" Type=\"float\" Name=\"p_SoftClipC\">\n" \
	"</Uniform>\n" \
	"<Uniform ResDependent=\"None\" Max=\"5.0\" Min=\"1.0\" Default=\"%f\" Inc=\"0.001\" Tooltip=\"\" Row=\"3\" Col=\"0\" Page=\"0\" DisplayName=\"Highlight SoftClip Max\" Type=\"float\" Name=\"p_SoftClipD\">\n" \
	"</Uniform>\n" \
	"<Uniform Row=\"1\" Col=\"1\" Page=\"0\" Default=\"False\" Tooltip=\"\" Type=\"bool\" DisplayName=\"Display Shadows\" Name=\"p_SwitchB\">\n" \
	"</Uniform>\n" \
	"<Uniform ResDependent=\"None\" Max=\"1.0\" Min=\"0.0\" Default=\"%f\" Inc=\"0.001\" Tooltip=\"\" Row=\"2\" Col=\"1\" Page=\"0\" DisplayName=\"Shadow SoftClip Min\" Type=\"float\" Name=\"p_SoftClipE\">\n" \
	"</Uniform>\n" \
	"<Uniform ResDependent=\"None\" Max=\"0.0\" Min=\"-1.0\" Default=\"%f\" Inc=\"0.001\" Tooltip=\"\" Row=\"3\" Col=\"1\" Page=\"0\" DisplayName=\"Shadow SoftClip Max\" Type=\"float\" Name=\"p_SoftClipF\">\n" \
	"</Uniform>\n" \
	"</Shader>\n" \
	"<Page Name=\"SoftClip\" Page=\"0\">\n" \
	"<Col Name=\"SoftClip\" Col=\"0\" Page=\"0\">\n" \
	"</Col>\n" \
	"</Page>\n" \
	"</ShaderNodePreset>\n", source, compress, redistribute, highlight, ceiling, shadow, floor);
	fclose (pFile2);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".glsl and " + NAME + ".xml to " + PATH2  + ". Check Permissions."));
	}	
	}
	}
}

void SoftClipPlugin::setupAndProcess(SoftClip& p_SoftClip, const OFX::RenderArguments& p_Args)
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


    double aScale = m_ScaleA->getValueAtTime(p_Args.time);
    double bScale = m_ScaleB->getValueAtTime(p_Args.time);
    double cScale = m_ScaleC->getValueAtTime(p_Args.time);
    double dScale = m_ScaleD->getValueAtTime(p_Args.time);
    double shadow = m_Shadow->getValueAtTime(p_Args.time);
    double floor = m_Floor->getValueAtTime(p_Args.time);
        
    bool aSwitch = m_SwitchA->getValueAtTime(p_Args.time);
	float aSwitchF = aSwitch ? 1.0f : 0.0f;
	
	bool bSwitch = m_SwitchB->getValueAtTime(p_Args.time);
	float bSwitchF = bSwitch ? 1.0f : 0.0f;

    int sourceLog_i;
    m_SourceLog->getValueAtTime(p_Args.time, sourceLog_i);
    SourceLogEnum sourceLog = (SourceLogEnum)sourceLog_i;
    
    bool aSource = sourceLog_i == 0;
    bool bSource = sourceLog_i == 1;
    bool cSource = sourceLog_i == 2;

	float aSourceF = aSource ? 1.0f : 0.0f;
	float bSourceF = bSource ? 1.0f : 0.0f;
	float cSourceF = cSource ? 1.0f : 0.0f;

    
    // Set the images
    p_SoftClip.setDstImg(dst.get());
    p_SoftClip.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    //p_SoftClip.setGPURenderArgs(p_Args);

    // Set the render window
    p_SoftClip.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_SoftClip.setScales(aScale, bScale, cScale, dScale, shadow, floor, aSwitchF, bSwitchF, aSourceF, bSourceF, cSourceF);

    // Call the base class process member, this will call the derived templated process code
    p_SoftClip.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

SoftClipPluginFactory::SoftClipPluginFactory()
    : OFX::PluginFactoryHelper<SoftClipPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void SoftClipPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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


void SoftClipPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
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

    //Drop down menu
    ChoiceParamDescriptor* choiceparam = p_Desc.defineChoiceParam("sourceLog");
    choiceparam->setLabel(kParamSourceLogLabel);
    choiceparam->setHint(kParamSourceLogHint);
    assert(choiceparam->getNOptions() == eSourceBypass);
    choiceparam->appendOption(kParamSourceLogOptionBypass, kParamSourceLogOptionBypassHint);
    assert(choiceparam->getNOptions() == eSourceLogCineon);
    choiceparam->appendOption(kParamSourceLogOptionCineon, kParamSourceLogOptionCineonHint);
    assert(choiceparam->getNOptions() == eSourceLogLogc);
    choiceparam->appendOption(kParamSourceLogOptionLogc, kParamSourceLogOptionLogcHint);
    page->addChild(*choiceparam);
     
     }
        
    // Group param to group the scales
    GroupParamDescriptor* componentScalesGroup = p_Desc.defineGroupParam("alphaChannel");
    componentScalesGroup->setOpen(false);
    componentScalesGroup->setHint("Soft-clip adjustment");
    componentScalesGroup->setLabel("Highlights");

    {
    // Add a boolean to enable the component scale
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("highlight");
    boolParam->setDefault(false);
    boolParam->setHint("Displays signal values over 1.0");
    boolParam->setLabel("display highlights");
    boolParam->setParent(*componentScalesGroup);
    page->addChild(*boolParam);
    }
            
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scaleA", "highlight compress", "Highlights compressed by stops", componentScalesGroup);
    param->setDefault(0.0);
    param->setRange(-10.0, 0.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-6.0, 0.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleB", "redistribute", "Adjust highlight compression", componentScalesGroup);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.2, 5.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleC", "highlight soft-clip min", "Adjust soft-clip min threshhold", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleD", "highlight soft-clip max", "Adjust soft-clip max threshhold", 0);
    param->setDefault(1.0);
    param->setRange(1.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(1.0, 5.0);
    page->addChild(*param);
    
	{
	GroupParamDescriptor* shadowSoft = p_Desc.defineGroupParam("shadowSoft");
	shadowSoft->setOpen(false);
	shadowSoft->setLabel("Shadows");
	
	{
	BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("displayShadow");
	boolParam->setDefault(false);
	boolParam->setHint("Display signal below 0.0");
	boolParam->setLabel("display shadows");
	boolParam->setParent(*shadowSoft);
	page->addChild(*boolParam);
	}
	
	param = defineScaleParam(p_Desc, "shadow", "shadow soft-clip", "Adjust shadow soft-clip", shadowSoft);
	param->setDefault(0.0);
	param->setRange(0.0, 1.0);
	param->setDisplayRange(0.0, 1.0);
	param->setIncrement(0.001);
	param->setDoubleType(eDoubleTypeScale);
	page->addChild(*param);
	
	param = defineScaleParam(p_Desc, "floor", "shadow soft-clip floor", "Adjust shadow soft-clip min threshhold", shadowSoft);
    param->setDefault(0.0);
    param->setRange(0.0, -0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, -0.5);
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
	param->setDefault("SoftClip");
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

ImageEffect* SoftClipPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new SoftClipPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static SoftClipPluginFactory SoftClipPlugin;
    p_FactoryArray.push_back(&SoftClipPlugin);
}
