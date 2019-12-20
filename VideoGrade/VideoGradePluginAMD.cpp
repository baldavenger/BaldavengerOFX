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
#define kPluginScript2 "/Library/Application Support/FilmLight/shaders"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#define kPluginScript2 "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#else
#define kPluginScript "/opt/resolve/LUT"
#define kPluginScript2 "/opt/resolve/LUT"
#endif

#define kPluginName "VideoGrade"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"VideoGrade: Video style grading, with Lift, Gamma, Gain, and Offset controls, adjustable \n" \
"anchors for Lift and Gain, adjustable region for Gamma, and Upper Gamma Bias option."

#define kPluginIdentifier "BaldavengerOFX.VideoGrade"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kResolutionScale	(float)width / 1920.0f

#define kParamLuminanceMath "LuminanceMath"
#define kParamLuminanceMathLabel "luma math"
#define kParamLuminanceMathHint "Formula used to compute luma from RGB values."
#define kParamLuminanceMathOptionRec709 "Rec.709"
#define kParamLuminanceMathOptionRec709Hint "Use Rec.709 (0.2126r + 0.7152g + 0.0722b)."
#define kParamLuminanceMathOptionRec2020 "Rec.2020"
#define kParamLuminanceMathOptionRec2020Hint "Use Rec.2020 (0.2627r + 0.6780g + 0.0593b)."
#define kParamLuminanceMathOptionDCIP3 "DCI P3"
#define kParamLuminanceMathOptionDCIP3Hint "Use DCI P3 (0.209492r + 0.721595g + 0.0689131b)."
#define kParamLuminanceMathOptionACESAP0 "ACES AP0"
#define kParamLuminanceMathOptionACESAP0Hint "Use ACES AP0 (0.3439664498r + 0.7281660966g + -0.0721325464b)."
#define kParamLuminanceMathOptionACESAP1 "ACES AP1"
#define kParamLuminanceMathOptionACESAP1Hint "Use ACES AP1 (0.2722287168r +  0.6740817658g +  0.0536895174b)."
#define kParamLuminanceMathOptionAverage "Average"
#define kParamLuminanceMathOptionAverageHint "Use average of r, g, b."
#define kParamLuminanceMathOptionMaximum "Max"
#define kParamLuminanceMathOptionMaximumHint "Use MAX of r, g, b."

enum LuminanceMathEnum
{
eLuminanceMathRec709,
eLuminanceMathRec2020,
eLuminanceMathDCIP3,
eLuminanceMathACESAP0,
eLuminanceMathACESAP1,
eLuminanceMathAverage,
eLuminanceMathMaximum,
};

#define kParamDisplayGraph "displayGraph"
#define kParamDisplayGraphLabel "display graph"
#define kParamDisplayGraphHint "display curve graph"
#define kParamDisplayGraphOptionOff "Off"
#define kParamDisplayGraphOptionOffHint "no graph displayed"
#define kParamDisplayGraphOptionON "On"
#define kParamDisplayGraphOptionONHint "display curve graph"
#define kParamDisplayGraphOptionOVER "Overlay"
#define kParamDisplayGraphOptionOVERHint "overlay curve graph"

enum DisplayGraphEnum {
eDisplayGraphOff,
eDisplayGraphON,
eDisplayGraphOVER,
};

////////////////////////////////////////////////////////////////////////////////

class VideoGrade : public OFX::ImageProcessor
{
public:
explicit VideoGrade(OFX::ImageEffect& p_Instance);

//virtual void processImagesCUDA();
virtual void processImagesOpenCL();
virtual void processImagesMetal();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(int p_LumaMath, int* p_Switch, int p_Display, float* p_Scale);

private:
OFX::Image* _srcImg;
int _lumaMath;
int _switch[2];
int _display;
float _scales[21];
};

VideoGrade::VideoGrade(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}
/*
extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_LumaMath, int* p_Switch, int p_Display, float* p_Scales);

void VideoGrade::processImagesCUDA()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunCudaKernel(input, output, width, height, _lumaMath, _switch, _display, _scales);
}
*/
extern void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, 
int p_Width, int p_Height, int p_LumaMath, int* p_Switch, int p_Display, float* p_Scales);

void VideoGrade::processImagesOpenCL()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunOpenCLKernel(_pOpenCLCmdQ, input, output, width, height, _lumaMath, _switch, _display, _scales);
}

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, 
int p_Height, int p_LumaMath, int* p_Switch, int p_Display, float* p_Scales);
#endif

void VideoGrade::processImagesMetal()
{
#ifdef __APPLE__
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunMetalKernel(_pMetalCmdQ, input, output, width, height, _lumaMath, _switch, _display, _scales);
#endif
}

void VideoGrade::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
if (_effect.abort()) break;
float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
if (srcPix) {
dstPix[0] = srcPix[0];
dstPix[1] = srcPix[1];
dstPix[2] = srcPix[2];
dstPix[3] = srcPix[3];
} else {
for (int c = 0; c < 4; ++c) {
dstPix[c] = 0;
}}
dstPix += 4;
}}}

void VideoGrade::setSrcImg(OFX::Image* p_SrcImg) {
_srcImg = p_SrcImg;
}

void VideoGrade::setScales(int p_LumaMath, int* p_Switch, int p_Display, float* p_Scales)
{
_lumaMath = p_LumaMath;
_switch[0] = p_Switch[0];
_switch[1] = p_Switch[1];
_display = p_Display;
_scales[0] = p_Scales[0];
_scales[1] = p_Scales[1];
_scales[2] = p_Scales[2];
_scales[3] = p_Scales[3];
_scales[4] = p_Scales[4];
_scales[5] = p_Scales[5];
_scales[6] = p_Scales[6];
_scales[7] = p_Scales[7];
_scales[8] = p_Scales[8];
_scales[9] = p_Scales[9];
_scales[10] = p_Scales[10];
_scales[11] = p_Scales[11];
_scales[12] = p_Scales[12];
_scales[13] = p_Scales[13];
_scales[14] = p_Scales[14];
_scales[15] = p_Scales[15];
_scales[16] = p_Scales[16];
_scales[17] = p_Scales[17];
_scales[18] = p_Scales[18];
_scales[19] = p_Scales[19];
_scales[20] = p_Scales[20];   
}

////////////////////////////////////////////////////////////////////////////////

class VideoGradePlugin : public OFX::ImageEffect
{
public:
explicit VideoGradePlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

void setupAndProcess(VideoGrade &p_VideoGrade, const OFX::RenderArguments& p_Args);

private:

OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;

OFX::ChoiceParam* m_LuminanceMath;
OFX::BooleanParam* m_Gang;
OFX::DoubleParam* m_Exposure;
OFX::DoubleParam* m_Temp;
OFX::DoubleParam* m_Tint;
OFX::DoubleParam* m_Hue;
OFX::DoubleParam* m_Sat;
OFX::DoubleParam* m_GainR;
OFX::DoubleParam* m_GainG;
OFX::DoubleParam* m_GainB;
OFX::DoubleParam* m_GainAnchor;
OFX::DoubleParam* m_LiftR;
OFX::DoubleParam* m_LiftG;
OFX::DoubleParam* m_LiftB;
OFX::DoubleParam* m_LiftAnchor;
OFX::DoubleParam* m_OffsetR;
OFX::DoubleParam* m_OffsetG;
OFX::DoubleParam* m_OffsetB;
OFX::DoubleParam* m_GammaR;
OFX::DoubleParam* m_GammaG;
OFX::DoubleParam* m_GammaB;
OFX::BooleanParam* m_GammaBias;
OFX::DoubleParam* m_GammaStart;
OFX::DoubleParam* m_GammaEnd;
OFX::ChoiceParam* m_DisplayGraph;
OFX::PushButtonParam* m_Info;
OFX::StringParam* m_Name;
OFX::StringParam* m_Path;
OFX::StringParam* m_Path2;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
OFX::PushButtonParam* m_Button3;
};

VideoGradePlugin::VideoGradePlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

m_LuminanceMath = fetchChoiceParam(kParamLuminanceMath);
m_Gang = fetchBooleanParam("Gang");
m_Exposure = fetchDoubleParam("Exposure");
m_Temp = fetchDoubleParam("Temp");
m_Tint = fetchDoubleParam("Tint");
m_Hue = fetchDoubleParam("Hue");
m_Sat = fetchDoubleParam("Sat");
m_GainR = fetchDoubleParam("GainR");
m_GainG = fetchDoubleParam("GainG");
m_GainB = fetchDoubleParam("GainB");
m_GainAnchor = fetchDoubleParam("GainAnchor");
m_LiftR = fetchDoubleParam("LiftR");
m_LiftG = fetchDoubleParam("LiftG");
m_LiftB = fetchDoubleParam("LiftB");
m_LiftAnchor = fetchDoubleParam("LiftAnchor");
m_OffsetR = fetchDoubleParam("OffsetR");
m_OffsetG = fetchDoubleParam("OffsetG");
m_OffsetB = fetchDoubleParam("OffsetB");
m_GammaR = fetchDoubleParam("GammaR");
m_GammaG = fetchDoubleParam("GammaG");
m_GammaB = fetchDoubleParam("GammaB");
m_GammaBias = fetchBooleanParam("GammaBias");
m_GammaStart = fetchDoubleParam("GammaStart");
m_GammaEnd = fetchDoubleParam("GammaEnd");
m_DisplayGraph = fetchChoiceParam(kParamDisplayGraph);
m_Info = fetchPushButtonParam("Info");
m_Name = fetchStringParam("Name");
m_Path = fetchStringParam("Path");
m_Path2 = fetchStringParam("Path2");
m_Button1 = fetchPushButtonParam("Button1");
m_Button2 = fetchPushButtonParam("Button2");
m_Button3 = fetchPushButtonParam("Button3");
}

void VideoGradePlugin::render(const OFX::RenderArguments& p_Args)
{
if (m_DstClip->getPixelDepth() == OFX::eBitDepthFloat && m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA) {
VideoGrade imageScaler(*this);
setupAndProcess(imageScaler, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool VideoGradePlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
float _scale = m_Exposure->getValueAtTime(p_Args.time);
if (_scale == 0.0f) {
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void VideoGradePlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if (p_ParamName == "info") {
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

if (p_ParamName == "Gang") {
bool _gang = m_Gang->getValueAtTime(p_Args.time);
if(!_gang){
m_GainG->setIsSecretAndDisabled(false);
m_GainB->setIsSecretAndDisabled(false);
m_GainR->setLabel("gain R");
m_LiftG->setIsSecretAndDisabled(false);
m_LiftB->setIsSecretAndDisabled(false);
m_LiftR->setLabel("lift R");
m_OffsetG->setIsSecretAndDisabled(false);
m_OffsetB->setIsSecretAndDisabled(false);
m_OffsetR->setLabel("offset R");
m_GammaG->setIsSecretAndDisabled(false);
m_GammaB->setIsSecretAndDisabled(false);
m_GammaR->setLabel("gamma R");
}
if (_gang) {
m_GainG->setIsSecretAndDisabled(true);
m_GainB->setIsSecretAndDisabled(true);
m_GainR->setLabel("gain");
m_LiftG->setIsSecretAndDisabled(true);
m_LiftB->setIsSecretAndDisabled(true);
m_LiftR->setLabel("lift");
m_OffsetG->setIsSecretAndDisabled(true);
m_OffsetB->setIsSecretAndDisabled(true);
m_OffsetR->setLabel("offset");
m_GammaG->setIsSecretAndDisabled(true);
m_GammaB->setIsSecretAndDisabled(true);
m_GammaR->setLabel("gamma");
}}}

void VideoGradePlugin::setupAndProcess(VideoGrade& p_VideoGrade, const OFX::RenderArguments& p_Args)
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

int _lumaMath;
m_LuminanceMath->getValueAtTime(p_Args.time, _lumaMath);

int _display;
m_DisplayGraph->getValueAtTime(p_Args.time, _display);

float _scales[21];
int _switch[2];

bool Switch0 = m_Gang->getValueAtTime(p_Args.time);
bool Switch1 = m_GammaBias->getValueAtTime(p_Args.time);
_switch[0] = Switch0 ? 1 : 0;
_switch[1] = Switch1 ? 1 : 0;

_scales[0] = m_Exposure->getValueAtTime(p_Args.time);
_scales[1] = m_Temp->getValueAtTime(p_Args.time);
_scales[2] = m_Tint->getValueAtTime(p_Args.time);
_scales[3] = m_Hue->getValueAtTime(p_Args.time);
_scales[4] = m_Sat->getValueAtTime(p_Args.time);
_scales[5] = m_GainR->getValueAtTime(p_Args.time);
_scales[6] = m_GainG->getValueAtTime(p_Args.time);
_scales[7] = m_GainB->getValueAtTime(p_Args.time);
_scales[8] = m_GainAnchor->getValueAtTime(p_Args.time);
_scales[9] = m_LiftR->getValueAtTime(p_Args.time);
_scales[10] = m_LiftG->getValueAtTime(p_Args.time);
_scales[11] = m_LiftB->getValueAtTime(p_Args.time);
_scales[12] = m_LiftAnchor->getValueAtTime(p_Args.time);
_scales[13] = m_OffsetR->getValueAtTime(p_Args.time);
_scales[14] = m_OffsetG->getValueAtTime(p_Args.time);
_scales[15] = m_OffsetB->getValueAtTime(p_Args.time);
_scales[16] = m_GammaR->getValueAtTime(p_Args.time);
_scales[17] = m_GammaG->getValueAtTime(p_Args.time);
_scales[18] = m_GammaB->getValueAtTime(p_Args.time);
_scales[19] = m_GammaStart->getValueAtTime(p_Args.time);
_scales[20] = m_GammaEnd->getValueAtTime(p_Args.time);

p_VideoGrade.setDstImg(dst.get());
p_VideoGrade.setSrcImg(src.get());

// Setup GPU Render arguments
p_VideoGrade.setGPURenderArgs(p_Args);

p_VideoGrade.setRenderWindow(p_Args.renderWindow);

p_VideoGrade.setScales(_lumaMath, _switch, _display, _scales);

p_VideoGrade.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

VideoGradePluginFactory::VideoGradePluginFactory()
: OFX::PluginFactoryHelper<VideoGradePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void VideoGradePluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

p_Desc.setSupportsOpenCLRender(true);
//p_Desc.setSupportsCudaRender(true);
#ifdef __APPLE__
p_Desc.setSupportsMetalRender(true);
#endif
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, 
const std::string& p_Label, const std::string& p_Hint, GroupParamDescriptor* p_Parent) {
DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
param->setLabel(p_Label);
param->setScriptName(p_Name);
param->setHint(p_Hint);
param->setDoubleType(eDoubleTypeScale);
if (p_Parent)
param->setParent(*p_Parent);
return param;
}

void VideoGradePluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum)
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

BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("Gang");
boolParam->setDefault(true);
boolParam->setHint("Gang RGB channels together as one parameter");
boolParam->setLabel("gang rgb");
page->addChild(*boolParam);

DoubleParamDescriptor* param = defineScaleParam(p_Desc, "Exposure", "exposure", "Exposure by stops in linear light", 0);
param->setDefault(0.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-6.0, 6.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Temp", "temp", "Temperature", 0);
param->setDefault(6500);
param->setRange(1000, 40000);
param->setIncrement(1);
param->setDisplayRange(2000, 15000);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Tint", "tint", "Tint", 0);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Hue", "hue", "Rotate hue", 0);
param->setDefault(0.0);
param->setRange(-360.0, 360.0);
param->setIncrement(0.1);
param->setDisplayRange(-360.0, 360.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Sat", "sat", "Saturation", 0);
param->setDefault(0.0);
param->setRange(-1.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "GainR", "gain", "Gain signal", 0);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 5.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "GainG", "gain G", "Gain green channel", 0);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 5.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "GainB", "gain B", "Gain blue channel", 0);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 5.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "GainAnchor", "gain anchor", "adjust anchor point", 0);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "LiftR", "lift", "Lift signal", 0);
param->setDefault(0.0);
param->setRange(-4.0, 4.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "LiftG", "lift G", "Lift green channel", 0);
param->setDefault(0.0);
param->setRange(-4.0, 4.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "LiftB", "lift B", "Lift blue channel", 0);
param->setDefault(0.0);
param->setRange(-4.0, 4.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "LiftAnchor", "lift anchor", "adjust anchor point", 0);
param->setDefault(1.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "OffsetR", "offset", "Offset signal", 0);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "OffsetG", "offset G", "Offset green channel", 0);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "OffsetB", "offset B", "Offset blue channel", 0);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "GammaR", "gamma", "Gamma signal", 0);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 5.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "GammaG", "gamma G", "Gamma green channel", 0);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 5.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = defineScaleParam(p_Desc, "GammaB", "gamma B", "Gamma blue channel", 0);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 5.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

boolParam = p_Desc.defineBooleanParam("GammaBias");
boolParam->setDefault(false);
boolParam->setHint("higer instead of lower curve bias");
boolParam->setLabel("upper gamma bias");
page->addChild(*boolParam);

param = defineScaleParam(p_Desc, "GammaStart", "gamma region start", "adjust", 0);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "GammaEnd", "gamma region end", "adjust", 0);
param->setDefault(1.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

ChoiceParamDescriptor* choiceparam = p_Desc.defineChoiceParam(kParamLuminanceMath);
choiceparam->setLabel(kParamLuminanceMathLabel);
choiceparam->setHint(kParamLuminanceMathHint);
assert(choiceparam->getNOptions() == eLuminanceMathRec709);
choiceparam->appendOption(kParamLuminanceMathOptionRec709, kParamLuminanceMathOptionRec709Hint);
assert(choiceparam->getNOptions() == eLuminanceMathRec2020);
choiceparam->appendOption(kParamLuminanceMathOptionRec2020, kParamLuminanceMathOptionRec2020Hint);
assert(choiceparam->getNOptions() == eLuminanceMathDCIP3);
choiceparam->appendOption(kParamLuminanceMathOptionDCIP3, kParamLuminanceMathOptionDCIP3Hint);
assert(choiceparam->getNOptions() == eLuminanceMathACESAP0);
choiceparam->appendOption(kParamLuminanceMathOptionACESAP0, kParamLuminanceMathOptionACESAP0Hint);
assert(choiceparam->getNOptions() == eLuminanceMathACESAP1);
choiceparam->appendOption(kParamLuminanceMathOptionACESAP1, kParamLuminanceMathOptionACESAP1Hint);
assert(choiceparam->getNOptions() == eLuminanceMathAverage);
choiceparam->appendOption(kParamLuminanceMathOptionAverage, kParamLuminanceMathOptionAverageHint);
assert(choiceparam->getNOptions() == eLuminanceMathMaximum);
choiceparam->appendOption(kParamLuminanceMathOptionMaximum, kParamLuminanceMathOptionMaximumHint);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamDisplayGraph);
choiceparam->setLabel(kParamDisplayGraphLabel);
choiceparam->setHint(kParamDisplayGraphHint);
assert(choiceparam->getNOptions() == eDisplayGraphOff);
choiceparam->appendOption(kParamDisplayGraphOptionOff, kParamDisplayGraphOptionOffHint);
assert(choiceparam->getNOptions() == eDisplayGraphON);
choiceparam->appendOption(kParamDisplayGraphOptionON, kParamDisplayGraphOptionONHint);
assert(choiceparam->getNOptions() == eDisplayGraphOVER);
choiceparam->appendOption(kParamDisplayGraphOptionOVER, kParamDisplayGraphOptionOVERHint);
choiceparam->setDefault(eDisplayGraphOff);
choiceparam->setAnimates(false);
page->addChild(*choiceparam);

PushButtonParamDescriptor* pushparam = p_Desc.definePushButtonParam("Info");
pushparam->setLabel("info");
page->addChild(*pushparam);

GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
script->setOpen(false);
script->setHint("export DCTL and Nuke script");
if (page)
page->addChild(*script);

pushparam = p_Desc.definePushButtonParam("Button1");
pushparam->setLabel("export dctl");
pushparam->setHint("create DCTL version");
pushparam->setParent(*script);
page->addChild(*pushparam);

pushparam = p_Desc.definePushButtonParam("Button2");
pushparam->setLabel("export nuke script");
pushparam->setHint("create NUKE version");
pushparam->setParent(*script);
page->addChild(*pushparam);

StringParamDescriptor* stringparam = p_Desc.defineStringParam("Name");
stringparam->setLabel("name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("VideoGrade");
stringparam->setParent(*script);
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam("Path");
stringparam->setLabel("directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript);
stringparam->setFilePathExists(false);
stringparam->setParent(*script);
page->addChild(*stringparam);

pushparam = p_Desc.definePushButtonParam("Button3");
pushparam->setLabel("export shader");
pushparam->setHint("create Shader version");
pushparam->setParent(*script);
page->addChild(*pushparam);

stringparam = p_Desc.defineStringParam("Path2");
stringparam->setLabel("shader directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript2);
stringparam->setFilePathExists(false);
stringparam->setParent(*script);
page->addChild(*stringparam);
}

ImageEffect* VideoGradePluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum)
{
return new VideoGradePlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static VideoGradePluginFactory VideoGradePlugin;
p_FactoryArray.push_back(&VideoGradePlugin);
}