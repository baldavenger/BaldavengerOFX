#include "FreqSepPlugin.h"

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
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"Frequency Separation"

#define kPluginIdentifier "BaldavengerOFX.FreqSep"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kResolutionScale	(float)width / 1920.0f

#define kParamFreqSep "ColourSpace"
#define kParamFreqSepLabel "colour space"
#define kParamFreqSepHint "colour space"
#define kParamFreqSepOptionRGB "RGB"
#define kParamFreqSepOptionRGBHint "no conversion"
#define kParamFreqSepOptionLABREC709 "LAB from Rec709"
#define kParamFreqSepOptionLABREC709Hint "Rec.709 to LAB"
#define kParamFreqSepOptionLABLOGC "LAB from LogC"
#define kParamFreqSepOptionLABLOGCHint "LogC to LAB"
#define kParamFreqSepOptionLABACES "LAB from ACEScct"
#define kParamFreqSepOptionLABACESHint "ACEScct to LAB"

#define kParamDisplay "Display"
#define kParamDisplayLabel "display"
#define kParamDisplayHint "display options"
#define kParamDisplayOptionBypass "Output"
#define kParamDisplayOptionBypassHint "final output"
#define kParamDisplayOptionHigh "High Frequency"
#define kParamDisplayOptionHighHint "high frequency on middle grey background"
#define kParamDisplayOptionLow "Low Frequency"
#define kParamDisplayOptionLowHint "low frequency"
#define kParamDisplayOptionLowOver "Low Frequency and Overlay"
#define kParamDisplayOptionLowOverHint "low frequency with contrast overlay"

enum FreqSepEnum
{
eFreqSepRGB,
eFreqSepLABREC709,
eFreqSepLABLOGC,
eFreqSepLABACES,
};

enum DisplayEnum
{
eDisplayBypass,
eDisplayHigh,
eDisplayLow,
eDisplayLowOver,
};

////////////////////////////////////////////////////////////////////////////////

class FreqSep : public OFX::ImageProcessor
{
public:
explicit FreqSep(OFX::ImageEffect& p_Instance);

virtual void processImagesCUDA();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(int p_Space, int p_Display, float* p_Blur, float* p_Sharpen, float* p_Cont, int* p_Switch);

private:
OFX::Image* _srcImg;
int _space;
int _display;
float _blur[6];
float _sharpen[3];
float _cont[2];
int _switch[3];
};

FreqSep::FreqSep(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Space, int p_Display, float* p_Blur, float* p_Sharpen, float* p_Cont, int* p_Switch);

void FreqSep::processImagesCUDA()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

for(int c = 0; c < 6; c++){
_blur[c] = _blur[c] * kResolutionScale; 
}

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunCudaKernel(input, output, width, height, _space, _display, _blur, _sharpen, _cont, _switch);
}

void FreqSep::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
{
if (_effect.abort()) break;

float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
{
float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);


if (srcPix)
{
dstPix[0] = srcPix[0];
dstPix[1] = srcPix[1];
dstPix[2] = srcPix[2];
dstPix[3] = srcPix[3];
}

else
{
for (int c = 0; c < 4; ++c)
{
   dstPix[c] = 0;
}
}

dstPix += 4;
}
}
}

void FreqSep::setSrcImg(OFX::Image* p_SrcImg)
{
_srcImg = p_SrcImg;
}

void FreqSep::setScales(int p_Space, int p_Display, float* p_Blur, float* p_Sharpen, float* p_Cont, int* p_Switch)
{
_space = p_Space;
_display = p_Display;
_blur[0] = p_Blur[0];
_blur[1] = p_Blur[1];
_blur[2] = p_Blur[2];
_blur[3] = p_Blur[3];
_blur[4] = p_Blur[4];
_blur[5] = p_Blur[5];
_sharpen[0] = p_Sharpen[0];
_sharpen[1] = p_Sharpen[1];
_sharpen[2] = p_Sharpen[2];
_cont[0] = p_Cont[0];
_cont[1] = p_Cont[1];
_switch[0] = p_Switch[0];
_switch[1] = p_Switch[1];
_switch[2] = p_Switch[2];
}

////////////////////////////////////////////////////////////////////////////////

class FreqSepPlugin : public OFX::ImageEffect
{
public:
explicit FreqSepPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);

virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

void setupAndProcess(FreqSep &p_FreqSep, const OFX::RenderArguments& p_Args);

private:
OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;

OFX::ChoiceParam* m_Space;
OFX::ChoiceParam* m_Display;
OFX::DoubleParam* m_Blur1;
OFX::DoubleParam* m_Blur2;
OFX::DoubleParam* m_Blur3;
OFX::DoubleParam* m_Sharpen1;
OFX::DoubleParam* m_Sharpen2;
OFX::DoubleParam* m_Sharpen3;
OFX::DoubleParam* m_Blur4;
OFX::DoubleParam* m_Blur5;
OFX::DoubleParam* m_Blur6;
OFX::DoubleParam* m_Contrast;
OFX::DoubleParam* m_Pivot;
OFX::BooleanParam* m_Gang1;
OFX::BooleanParam* m_Gang2;
OFX::BooleanParam* m_Curve;

OFX::StringParam* m_Path;
OFX::StringParam* m_Path2;
OFX::StringParam* m_Name;
OFX::PushButtonParam* m_Info;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
OFX::PushButtonParam* m_Button3;
};

FreqSepPlugin::FreqSepPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

m_Space = fetchChoiceParam(kParamFreqSep);
m_Display = fetchChoiceParam(kParamDisplay);
m_Blur1 = fetchDoubleParam("Blur1");
m_Blur2 = fetchDoubleParam("Blur2");
m_Blur3 = fetchDoubleParam("Blur3");
m_Sharpen1 = fetchDoubleParam("Sharpen1");
m_Sharpen2 = fetchDoubleParam("Sharpen2");
m_Sharpen3 = fetchDoubleParam("Sharpen3");
m_Blur4 = fetchDoubleParam("Blur4");
m_Blur5 = fetchDoubleParam("Blur5");
m_Blur6 = fetchDoubleParam("Blur6");
m_Contrast = fetchDoubleParam("Contrast");
m_Pivot = fetchDoubleParam("Pivot");

m_Gang1 = fetchBooleanParam("Gang1");
m_Gang2 = fetchBooleanParam("Gang2");
m_Curve = fetchBooleanParam("Curve");

m_Path = fetchStringParam("Path");
m_Path2 = fetchStringParam("Path2");
m_Name = fetchStringParam("Name");
m_Info = fetchPushButtonParam("Info");
m_Button1 = fetchPushButtonParam("Button1");
m_Button2 = fetchPushButtonParam("Button2");
m_Button3 = fetchPushButtonParam("Button3");
}

void FreqSepPlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
{
FreqSep FreqSep(*this);
setupAndProcess(FreqSep, p_Args);
}
else
{
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}
}

bool FreqSepPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{

if (m_SrcClip)
{
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}

return false;
}

void FreqSepPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if(p_ParamName == kParamFreqSep)
{
int space_i;
m_Space->getValueAtTime(p_Args.time, space_i);
FreqSepEnum FreqSepFilter = (FreqSepEnum)space_i;

bool RGB = space_i == 0;
bool LAB = space_i != 0;

bool gang1 = m_Gang1->getValueAtTime(p_Args.time);
bool gang2 = m_Gang2->getValueAtTime(p_Args.time);

if(RGB){
m_Gang1->setIsSecretAndDisabled(!RGB);
m_Gang2->setIsSecretAndDisabled(RGB);
m_Gang1->setValue(true);
m_Curve->setValue(true);
m_Blur1->setLabel("highfreq threshold");
m_Blur2->setLabel("highfreq threshold G");
m_Blur3->setLabel("highfreq threshold B");
m_Sharpen1->setLabel("highfreq contrast");
m_Sharpen2->setLabel("highfreq contrast G");
m_Sharpen3->setLabel("highfreq contrast B");
m_Blur4->setLabel("lowfreq blur");
m_Blur5->setLabel("lowfreq blur G");
m_Blur6->setLabel("lowfreq blur B");
}

if(LAB){
m_Gang1->setIsSecretAndDisabled(LAB);
m_Gang2->setIsSecretAndDisabled(!LAB);
m_Gang2->setValue(true);
m_Curve->setValue(false);
m_Blur2->setIsSecretAndDisabled(LAB);
m_Blur3->setIsSecretAndDisabled(LAB);
m_Sharpen2->setIsSecretAndDisabled(LAB);
m_Sharpen3->setIsSecretAndDisabled(LAB);
m_Blur5->setIsSecretAndDisabled(!LAB);
m_Blur1->setLabel("highfreq threshold L");
m_Sharpen1->setLabel("highfreq contrast L");
m_Blur4->setLabel("lowfreq blur L");
m_Blur5->setLabel("blur AB");
m_Blur6->setLabel("blur B");
}
}

if(p_ParamName == "Gang1")
{

bool gang1 = m_Gang1->getValueAtTime(p_Args.time);

if(!gang1) {
m_Blur1->setLabel("highfreq threshold R");
m_Sharpen1->setLabel("highfreq contrast R");
m_Blur4->setLabel("lowfreq blur R");
} else {
m_Blur1->setLabel("highfreq threshold");
m_Sharpen1->setLabel("highfreq contrast");
m_Blur4->setLabel("lowfreq blur");
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
FreqSepEnum FreqSepFilter = (FreqSepEnum)space_i;
bool LAB = space_i != 0;
bool gang2 = m_Gang2->getValueAtTime(p_Args.time);
if(LAB) {
if(!gang2) {
m_Blur5->setLabel("blur A");
} else {
m_Blur5->setLabel("blur AB");
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

fprintf (pFile, "// FreqSep Separation plugin DCTL export\n" \
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
" name FreqSep Separation\n" \
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
"<ShaderNodePreset SupportsAdaptiveDegradation=\"0\" Description=\"FreqSep\" Name=\"FreqSep\"> \n" \
"<Shader OutputBitDepth=\"Output\" Index=\"1\"> \n" \
"</Shader> \n" \
"<Page Name=\"FreqSep\" Page=\"0\"> \n" \
"<Col Name=\"FreqSep\" Col=\"0\" Page=\"0\"> \n" \
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

void FreqSepPlugin::setupAndProcess(FreqSep& p_FreqSep, const OFX::RenderArguments& p_Args)
{
std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
{
OFX::throwSuiteStatusException(kOfxStatErrValue);
}

int space_i;
m_Space->getValueAtTime(p_Args.time, space_i);
FreqSepEnum FreqSepFilter = (FreqSepEnum)space_i;
int _space = space_i;

int display_i;
m_Display->getValueAtTime(p_Args.time, display_i);
DisplayEnum DisplayFilter = (DisplayEnum)display_i;
int _display = display_i;

float _blur[6], _sharpen[3], _cont[2];
int _switch[3];

_blur[0] = m_Blur1->getValueAtTime(p_Args.time);
_blur[1] = m_Blur2->getValueAtTime(p_Args.time);
_blur[2] = m_Blur3->getValueAtTime(p_Args.time);
_blur[3] = m_Blur4->getValueAtTime(p_Args.time);
_blur[4] = m_Blur5->getValueAtTime(p_Args.time);
_blur[5] = m_Blur6->getValueAtTime(p_Args.time);

_blur[0] *= 4.0f;
_blur[1] *= 4.0f;
_blur[2] *= 4.0f;
_blur[3] *= 10.0f;
_blur[4] *= 10.0f;
_blur[5] *= 10.0f;

_sharpen[0] = m_Sharpen1->getValueAtTime(p_Args.time);
_sharpen[1] = m_Sharpen2->getValueAtTime(p_Args.time);
_sharpen[2] = m_Sharpen3->getValueAtTime(p_Args.time);

_sharpen[0] = _sharpen[0] > 0.0f ? (_sharpen[0] * 2.0f) + 1.0f : _sharpen[0] + 1.0f;
_sharpen[1] = _sharpen[1] > 0.0f ? (_sharpen[1] * 2.0f) + 1.0f : _sharpen[1] + 1.0f;
_sharpen[2] = _sharpen[2] > 0.0f ? (_sharpen[2] * 2.0f) + 1.0f : _sharpen[2] + 1.0f;

_cont[0] = m_Contrast->getValueAtTime(p_Args.time);
_cont[0] = _cont[0] > 0.0f ? _cont[0] * 2.0f + 1.0f : _cont[0] + 1.0f;

_cont[1] = m_Pivot->getValueAtTime(p_Args.time);

bool gang1 = m_Gang1->getValueAtTime(p_Args.time);
_switch[0] = gang1 ? 1 : 0;
bool gang2 = m_Gang2->getValueAtTime(p_Args.time);
_switch[1] = gang2 ? 1 : 0;
bool curve = m_Curve->getValueAtTime(p_Args.time);
_switch[2] = curve ? 1 : 0;

p_FreqSep.setDstImg(dst.get());
p_FreqSep.setSrcImg(src.get());

p_FreqSep.setGPURenderArgs(p_Args);

p_FreqSep.setRenderWindow(p_Args.renderWindow);

p_FreqSep.setScales(_space, _display, _blur, _sharpen, _cont, _switch);

p_FreqSep.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

FreqSepPluginFactory::FreqSepPluginFactory()
: OFX::PluginFactoryHelper<FreqSepPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void FreqSepPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

p_Desc.setSupportsCudaRender(true);
}

void FreqSepPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

{
ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamFreqSep);
param->setLabel(kParamFreqSepLabel);
param->setHint(kParamFreqSepHint);
assert(param->getNOptions() == (int)eFreqSepRGB);
param->appendOption(kParamFreqSepOptionRGB, kParamFreqSepOptionRGBHint);
assert(param->getNOptions() == (int)eFreqSepLABREC709);
param->appendOption(kParamFreqSepOptionLABREC709, kParamFreqSepOptionLABREC709Hint);
assert(param->getNOptions() == (int)eFreqSepLABLOGC);
param->appendOption(kParamFreqSepOptionLABLOGC, kParamFreqSepOptionLABLOGCHint);
assert(param->getNOptions() == (int)eFreqSepLABACES);
param->appendOption(kParamFreqSepOptionLABACES, kParamFreqSepOptionLABACESHint);
param->setDefault( (int)eFreqSepRGB );
param->setAnimates(false);
page->addChild(*param);
}

{
ChoiceParamDescriptor *param = p_Desc.defineChoiceParam(kParamDisplay);
param->setLabel(kParamDisplayLabel);
param->setHint(kParamDisplayHint);
assert(param->getNOptions() == (int)eDisplayBypass);
param->appendOption(kParamDisplayOptionBypass, kParamDisplayOptionBypassHint);
assert(param->getNOptions() == (int)eDisplayHigh);
param->appendOption(kParamDisplayOptionHigh, kParamDisplayOptionHighHint);
assert(param->getNOptions() == (int)eDisplayLow);
param->appendOption(kParamDisplayOptionLow, kParamDisplayOptionLowHint);
assert(param->getNOptions() == (int)eDisplayLowOver);
param->appendOption(kParamDisplayOptionLowOver, kParamDisplayOptionLowOverHint);
param->setDefault( (int)eDisplayBypass );
param->setAnimates(false);
page->addChild(*param);
}

BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("Gang1");
boolParam->setLabel("gang all");
boolParam->setHint("Adjust 3 channels with single parameter");
boolParam->setDefault(true);
page->addChild(*boolParam);

DoubleParamDescriptor* param = p_Desc.defineDoubleParam("Blur1");
param->setLabel("highfreq threshold");
param->setHint("Adjust high frequency threshold");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Blur2");
param->setLabel("highfreq threshold G");
param->setHint("Adjust high frequency threshold");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Blur3");
param->setLabel("highfreq threshold B");
param->setHint("Adjust high frequency threshold");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Sharpen1");
param->setLabel("highfreq contrast");
param->setHint("high frequency contrast");
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Sharpen2");
param->setLabel("highfreq contrast G");
param->setHint("high frequency contrast");
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Sharpen3");
param->setLabel("highfreq contrast B");
param->setHint("high frequency contrast");
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Blur4");
param->setLabel("lowfreq blur");
param->setHint("Adjust low frequency blur");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Blur5");
param->setLabel("lowfreq blur G");
param->setHint("Adjust low frequency blur");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Blur6");
param->setLabel("lowfreq blur B");
param->setHint("Adjust low frequency blur");
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

param = p_Desc.defineDoubleParam("Contrast");
param->setLabel("lowfreq contrast");
param->setHint("low frequency contrast");
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
param->setIsSecretAndDisabled(false);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Pivot");
param->setLabel("lowfreq contrast pivot");
param->setHint("low frequency contrast pivot");
param->setDefault(0.5);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setIsSecretAndDisabled(false);
page->addChild(*param);

boolParam = p_Desc.defineBooleanParam("Curve");
boolParam->setLabel("S-Curve contrast");
boolParam->setHint("use S-Curve instead of linear contrast");
boolParam->setDefault(true);
boolParam->setIsSecretAndDisabled(false);
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
param->setDefault("FreqSepSeparation");
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

ImageEffect* FreqSepPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
return new FreqSepPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static FreqSepPluginFactory FreqSepPlugin;
p_FactoryArray.push_back(&FreqSepPlugin);
}
