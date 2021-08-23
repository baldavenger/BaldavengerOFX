#include "FreqEQPlugin.h"

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

#define kPluginName "Frequency EQ"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"Frequency Equaliser"

#define kPluginIdentifier "BaldavengerOFX.FreqEQ"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kParamDisplay "Texture"
#define kParamDisplayLabel "Display Frequencies"
#define kParamDisplayHint "displays isolated frequency ranges"
#define kParamDisplayOptionBypass "None"
#define kParamDisplayOptionBypassHint "no isolated display"
#define kParamDisplayOptionEQ1 "EQ1"
#define kParamDisplayOptionEQ1Hint "eq band 1"
#define kParamDisplayOptionEQ2 "EQ2"
#define kParamDisplayOptionEQ2Hint "eq band 2"
#define kParamDisplayOptionEQ3 "EQ3"
#define kParamDisplayOptionEQ3Hint "eq band 3"
#define kParamDisplayOptionEQ4 "EQ4"
#define kParamDisplayOptionEQ4Hint "eq band 4"
#define kParamDisplayOptionEQ5 "EQ5"
#define kParamDisplayOptionEQ5Hint "eq band 5"
#define kParamDisplayOptionEQ6 "EQ6"
#define kParamDisplayOptionEQ6Hint "eq band 6"
#define kParamDisplayOptionEQ7 "Combined"
#define kParamDisplayOptionEQ7Hint "all bands combined"

enum DisplayEnum
{
eDisplayBypass,
eDisplayEQ1,
eDisplayEQ2,
eDisplayEQ3,
eDisplayEQ4,
eDisplayEQ5,
eDisplayEQ6,
eDisplayEQ7,
};

////////////////////////////////////////////////////////////////////////////////

namespace {
struct RGBValues {
double r,g,b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};
}

class FrequencyEQ : public OFX::ImageProcessor
{
public:
explicit FrequencyEQ(OFX::ImageEffect& p_Instance);

virtual void processImagesMetal();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(float* p_EQ, int p_Switch, int p_Grey);

private:
OFX::Image* _srcImg;
float _eq[8];
int _switch;
int _grey;
};

FrequencyEQ::FrequencyEQ(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_EQ, int p_Switch, int p_Grey);
#endif

void FrequencyEQ::processImagesMetal()
{
#ifdef __APPLE__
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunMetalKernel(_pMetalCmdQ, input, output, width, height, _eq, _switch, _grey);
#endif
}

void FrequencyEQ::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
{
if (_effect.abort()) break;

float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
{
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

void FrequencyEQ::setSrcImg(OFX::Image* p_SrcImg)
{
_srcImg = p_SrcImg;
}

void FrequencyEQ::setScales(float* p_EQ, int p_Switch, int p_Grey)
{
_eq[0] = p_EQ[0];
_eq[1] = p_EQ[1];
_eq[2] = p_EQ[2];
_eq[3] = p_EQ[3];
_eq[4] = p_EQ[4];
_eq[5] = p_EQ[5];
_eq[6] = p_EQ[6];
_eq[7] = p_EQ[7];
_switch = p_Switch;
_grey = p_Grey;
}

////////////////////////////////////////////////////////////////////////////////

class FrequencyEQPlugin : public OFX::ImageEffect
{
public:
explicit FrequencyEQPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);

virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

void setupAndProcess(FrequencyEQ &p_FrequencyEQ, const OFX::RenderArguments& p_Args);

private:
OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;

OFX::DoubleParam* m_EQ1;
OFX::DoubleParam* m_EQ2;
OFX::DoubleParam* m_EQ3;
OFX::DoubleParam* m_EQ4;
OFX::DoubleParam* m_EQ5;
OFX::DoubleParam* m_EQ6;
OFX::DoubleParam* m_EQ7;
OFX::DoubleParam* m_EQ8;
OFX::ChoiceParam* m_Display;
OFX::BooleanParam* m_Grey;
OFX::PushButtonParam* m_Info;
};

FrequencyEQPlugin::FrequencyEQPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

m_EQ1 = fetchDoubleParam("EQ1");
m_EQ2 = fetchDoubleParam("EQ2");
m_EQ3 = fetchDoubleParam("EQ3");
m_EQ4 = fetchDoubleParam("EQ4");
m_EQ5 = fetchDoubleParam("EQ5");
m_EQ6 = fetchDoubleParam("EQ6");
m_EQ6 = fetchDoubleParam("EQ6");
m_EQ7 = fetchDoubleParam("Thresh");
m_EQ8 = fetchDoubleParam("Blend");

m_Display = fetchChoiceParam(kParamDisplay);
m_Grey = fetchBooleanParam("Grey");
m_Info = fetchPushButtonParam("Info");
}

void FrequencyEQPlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA)) {
FrequencyEQ FrequencyEQ(*this);
setupAndProcess(FrequencyEQ, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool FrequencyEQPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
if (m_SrcClip)
{
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void FrequencyEQPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if(p_ParamName == "Info") {
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}
}

void FrequencyEQPlugin::setupAndProcess(FrequencyEQ& p_FrequencyEQ, const OFX::RenderArguments& p_Args)
{
std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents)) {
OFX::throwSuiteStatusException(kOfxStatErrValue);
}

float _eq[8];

int display_i;
m_Display->getValueAtTime(p_Args.time, display_i);
DisplayEnum DisplayFilter = (DisplayEnum)display_i;

int _switch = display_i;

bool b_grey = m_Grey->getValueAtTime(p_Args.time);
int _grey = b_grey ? 1 : 0;

_eq[0] = m_EQ1->getValueAtTime(p_Args.time);
_eq[1] = m_EQ2->getValueAtTime(p_Args.time);
_eq[2] = m_EQ3->getValueAtTime(p_Args.time);
_eq[3] = m_EQ4->getValueAtTime(p_Args.time);
_eq[4] = m_EQ5->getValueAtTime(p_Args.time);
_eq[5] = m_EQ6->getValueAtTime(p_Args.time);
_eq[6] = m_EQ7->getValueAtTime(p_Args.time);
_eq[7] = m_EQ8->getValueAtTime(p_Args.time);

p_FrequencyEQ.setDstImg(dst.get());
p_FrequencyEQ.setSrcImg(src.get());

// Setup GPU Render arguments
p_FrequencyEQ.setGPURenderArgs(p_Args);

p_FrequencyEQ.setRenderWindow(p_Args.renderWindow);

p_FrequencyEQ.setScales(_eq, _switch, _grey);

p_FrequencyEQ.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

FrequencyEQPluginFactory::FrequencyEQPluginFactory()
: OFX::PluginFactoryHelper<FrequencyEQPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void FrequencyEQPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

// Setup GPU render capability flags
#ifdef __APPLE__
p_Desc.setSupportsMetalRender(true);
#endif
}

void FrequencyEQPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

DoubleParamDescriptor* param = p_Desc.defineDoubleParam("EQ1");
param->setLabel("EQ1");
param->setHint("frequency gain");
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("EQ2");
param->setLabel("EQ2");
param->setHint("frequency gain");
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("EQ3");
param->setLabel("EQ3");
param->setHint("frequency gain");
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("EQ4");
param->setLabel("EQ4");
param->setHint("frequency gain");
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("EQ5");
param->setLabel("EQ5");
param->setHint("frequency gain");
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("EQ6");
param->setLabel("EQ6");
param->setHint("frequency gain");
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Thresh");
param->setLabel("Base Frequency");
param->setHint("adjust initial frequency band");
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

ChoiceParamDescriptor *choiceParam = p_Desc.defineChoiceParam(kParamDisplay);
choiceParam->setLabel(kParamDisplayLabel);
choiceParam->setHint(kParamDisplayHint);
assert(choiceParam->getNOptions() == (int)eDisplayBypass);
choiceParam->appendOption(kParamDisplayOptionBypass, kParamDisplayOptionBypassHint);
assert(choiceParam->getNOptions() == (int)eDisplayEQ1);
choiceParam->appendOption(kParamDisplayOptionEQ1, kParamDisplayOptionEQ1Hint);
assert(choiceParam->getNOptions() == (int)eDisplayEQ2);
choiceParam->appendOption(kParamDisplayOptionEQ2, kParamDisplayOptionEQ2Hint);
assert(choiceParam->getNOptions() == (int)eDisplayEQ3);
choiceParam->appendOption(kParamDisplayOptionEQ3, kParamDisplayOptionEQ3Hint);
assert(choiceParam->getNOptions() == (int)eDisplayEQ4);
choiceParam->appendOption(kParamDisplayOptionEQ4, kParamDisplayOptionEQ4Hint);
assert(choiceParam->getNOptions() == (int)eDisplayEQ5);
choiceParam->appendOption(kParamDisplayOptionEQ5, kParamDisplayOptionEQ5Hint);
assert(choiceParam->getNOptions() == (int)eDisplayEQ6);
choiceParam->appendOption(kParamDisplayOptionEQ6, kParamDisplayOptionEQ6Hint);
assert(choiceParam->getNOptions() == (int)eDisplayEQ7);
choiceParam->appendOption(kParamDisplayOptionEQ7, kParamDisplayOptionEQ7Hint);
choiceParam->setDefault( (int)eDisplayBypass );
choiceParam->setAnimates(false);
page->addChild(*choiceParam);

BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("Grey");
boolParam->setLabel("On Grey");
boolParam->setHint("show texture on grey background");
boolParam->setDefault(true);
page->addChild(*boolParam);

param = p_Desc.defineDoubleParam("Blend");
param->setLabel("Global Blend");
param->setHint("blend between filtered and unfiltered image");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

PushButtonParamDescriptor* pushParam = p_Desc.definePushButtonParam("Info");
pushParam->setLabel("Info");
page->addChild(*pushParam);
}

ImageEffect* FrequencyEQPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
return new FrequencyEQPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static FrequencyEQPluginFactory FrequencyEQPlugin;
p_FactoryArray.push_back(&FrequencyEQPlugin);
}
