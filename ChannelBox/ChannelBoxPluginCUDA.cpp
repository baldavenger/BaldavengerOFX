#include "ChannelBoxPlugin.h"

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

#define kPluginName "ChannelBox"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"Channel Limit: Limit specific colour channels so that they never exceed the value of one or both of the other two channels.\n" \
"Channel Swap: Rebuild channels using information from the other two. \n" \
"Combine with the built-in Luma Mask to limit the effect to a specific range."

#define kPluginIdentifier "BaldavengerOFX.ChannelBox"
#define kOfxParamPropChoiceOption "OfxParamPropChoiceOption"
#define kOfxParamPropDefault "OfxParamPropDefault"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 2

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kResolutionScale	(float)width / 1920.0f

#define kParamChoice "choice"
#define kParamChoiceLabel "process"
#define kParamChoiceHint "limit or swap"
#define kParamChoiceOptionLimit "Channel Limit"
#define kParamChoiceLimitHint "channel limit"
#define kParamChoiceOptionSwap "Channel Swap"
#define kParamChoiceSwapHint "channel swap"

enum ChoiceEnum {
eChoiceLimit,
eChoiceSwap,
};

#define kParamChannelBox "channelBox"
#define kParamChannelBoxLabel "options"
#define kParamChannelBoxHint "Channels and Limits"
#define kParamChannelBoxOptionBR "Blue < Red"
#define kParamChannelBoxOptionBRHint "Limits Blue to Red"
#define kParamChannelBoxOptionBG "Blue < Green"
#define kParamChannelBoxOptionBGHint "Limits Blue to Green"
#define kParamChannelBoxOptionBGR "Blue < Green & Red"
#define kParamChannelBoxOptionBGRHint "Limits Blue to Green & Red"
#define kParamChannelBoxOptionBGRX "Blue < Max(Green, Red)"
#define kParamChannelBoxOptionBGRXHint "Limits Blue to Max of Green & Red"
#define kParamChannelBoxOptionGR "Green < Red"
#define kParamChannelBoxOptionGRHint "Limits Green to Red"
#define kParamChannelBoxOptionGB "Green < Blue"
#define kParamChannelBoxOptionGBHint "Limits Green to Blue"
#define kParamChannelBoxOptionGBR "Green < Blue & Red"
#define kParamChannelBoxOptionGBRHint "Limits Green to Blue & Red"
#define kParamChannelBoxOptionGBRX "Green < Max(Blue, Red)"
#define kParamChannelBoxOptionGBRXHint "Limits Green to Max of Blue & Red"
#define kParamChannelBoxOptionRG "Red < Green"
#define kParamChannelBoxOptionRGHint "Limits Red to Green"
#define kParamChannelBoxOptionRB "Red < Blue"
#define kParamChannelBoxOptionRBHint "Limits Red to Blue"
#define kParamChannelBoxOptionRBG "Red < Blue & Green"
#define kParamChannelBoxOptionRBGHint "Limits Red to Blue & Green"
#define kParamChannelBoxOptionRBGX "Red < Max(Blue, Green)"
#define kParamChannelBoxOptionRBGXHint "Limits Red to Max of Blue & Green"

enum ChannelBoxEnum {
eChannelBoxBR,
eChannelBoxBG,
eChannelBoxBGR,
eChannelBoxBGRX,
eChannelBoxGR,
eChannelBoxGB,
eChannelBoxGBR,
eChannelBoxGBRX,
eChannelBoxRG,
eChannelBoxRB,
eChannelBoxRBG,
eChannelBoxRBGX,
};

#define kParamLuminanceMath "luminanceMath"
#define kParamLuminanceMathLabel "luma math"
#define kParamLuminanceMathHint "Formula used to compute luminance from RGB values."
#define kParamLuminanceMathOptionRec709 "Rec. 709"
#define kParamLuminanceMathOptionRec709Hint "Use Rec. 709 (0.2126r + 0.7152g + 0.0722b)."
#define kParamLuminanceMathOptionRec2020 "Rec. 2020"
#define kParamLuminanceMathOptionRec2020Hint "Use Rec. 2020 (0.2627r + 0.6780g + 0.0593b)."
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

enum LuminanceMathEnum {
eLuminanceMathRec709,
eLuminanceMathRec2020,
eLuminanceMathDCIP3,
eLuminanceMathACESAP0,
eLuminanceMathACESAP1,
eLuminanceMathAverage,
eLuminanceMathMaximum,
};

////////////////////////////////////////////////////////////////////////////////

namespace {
struct RGBValues {
double r,g,b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};
}

inline float Luma(float R, float G, float B, int L) {
float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f;
float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f;
float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f;
float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f;
float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f;
float lumaAvg = (R + G + B) / 3.0f;
float lumaMax = fmax(fmax(R, G), B);
float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax;
return Lu;
}

class ChannelBox : public OFX::ImageProcessor
{
public:
explicit ChannelBox(OFX::ImageEffect& p_Instance);

virtual void processImagesCUDA();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(int p_Choice, int p_ChannelBox, float* p_ChannelSwap, int p_LumaMath, int* p_Switch, float* p_Mask);

private:
OFX::Image* _srcImg;
int _choice;
int _channelBox;
float _channelSwap[9];
int _lumaMath;
int _switch[2];
float _mask[4];
};

ChannelBox::ChannelBox(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Choice, int p_ChannelBox, float* p_ChannelSwap, int p_LumaMath, int* p_Switch, float* p_Mask);

void ChannelBox::processImagesCUDA()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

_mask[1] = _mask[1] * kResolutionScale;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunCudaKernel(input, output, width, height, _choice, _channelBox, _channelSwap, _lumaMath, _switch, _mask);
}

void ChannelBox::multiThreadProcessImages(OfxRectI p_ProcWindow)
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

void ChannelBox::setSrcImg(OFX::Image* p_SrcImg) {
_srcImg = p_SrcImg;
}

void ChannelBox::setScales(int p_Choice, int p_ChannelBox, float* p_ChannelSwap, int p_LumaMath, int* p_Switch, float* p_Mask)
{
_choice = p_Choice;
_channelBox = p_ChannelBox;
_channelSwap[0] = p_ChannelSwap[0];
_channelSwap[1] = p_ChannelSwap[1];
_channelSwap[2] = p_ChannelSwap[2];
_channelSwap[3] = p_ChannelSwap[3];
_channelSwap[4] = p_ChannelSwap[4];
_channelSwap[5] = p_ChannelSwap[5];
_channelSwap[6] = p_ChannelSwap[6];
_channelSwap[7] = p_ChannelSwap[7];
_channelSwap[8] = p_ChannelSwap[8];
_lumaMath = p_LumaMath;
_switch[0] = p_Switch[0];
_switch[1] = p_Switch[1];
_mask[0] = p_Mask[0];
_mask[1] = p_Mask[1];
_mask[2] = p_Mask[2];
_mask[3] = p_Mask[3];
}

////////////////////////////////////////////////////////////////////////////////

class ChannelBoxPlugin : public OFX::ImageEffect
{
public:
explicit ChannelBoxPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
void setupAndProcess(ChannelBox &p_ChannelBox, const OFX::RenderArguments& p_Args);

private:

OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;
OFX::ChoiceParam* m_Choice;
OFX::ChoiceParam* m_ChannelBox;
OFX::ChoiceParam* m_luminanceMath;
OFX::GroupParam* m_Versus;
OFX::DoubleParam* m_RedvGreen;
OFX::DoubleParam* m_RedvBlue;
OFX::DoubleParam* m_RedvGreenBlue;
OFX::DoubleParam* m_GreenvRed;
OFX::DoubleParam* m_GreenvBlue;
OFX::DoubleParam* m_GreenvRedBlue;
OFX::DoubleParam* m_BluevRed;
OFX::DoubleParam* m_BluevGreen;
OFX::DoubleParam* m_BluevRedGreen;
OFX::BooleanParam* m_SwitchA;
OFX::BooleanParam* m_SwitchB;
OFX::DoubleParam* m_Mask;
OFX::DoubleParam* m_Blur;
OFX::DoubleParam* m_Garbage;
OFX::DoubleParam* m_Core;
OFX::StringParam* m_Path;
OFX::StringParam* m_Path2;
OFX::StringParam* m_Name;
OFX::PushButtonParam* m_Info;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
OFX::PushButtonParam* m_Button3;
};

ChannelBoxPlugin::ChannelBoxPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
m_Choice = fetchChoiceParam(kParamChoice);
m_ChannelBox = fetchChoiceParam(kParamChannelBox);
m_luminanceMath = fetchChoiceParam(kParamLuminanceMath);
m_Versus = fetchGroupParam("Channel Swap");
m_RedvGreen = fetchDoubleParam("ScaleRvG");
m_RedvBlue = fetchDoubleParam("ScaleRvB");
m_RedvGreenBlue = fetchDoubleParam("ScaleRvGB");
m_GreenvRed = fetchDoubleParam("ScaleGvR");
m_GreenvBlue = fetchDoubleParam("ScaleGvB");
m_GreenvRedBlue = fetchDoubleParam("ScaleGvRB");
m_BluevRed = fetchDoubleParam("ScaleBvR");
m_BluevGreen = fetchDoubleParam("ScaleBvG");
m_BluevRedGreen = fetchDoubleParam("ScaleBvRG");
m_SwitchA = fetchBooleanParam("Preserve");
m_SwitchB = fetchBooleanParam("Display");
m_Mask = fetchDoubleParam("Mask");
m_Blur = fetchDoubleParam("Blur");
m_Garbage = fetchDoubleParam("Garbage");
m_Core = fetchDoubleParam("Core");
m_Path = fetchStringParam("Path");
m_Path2 = fetchStringParam("Path2");
m_Name = fetchStringParam("Name");
m_Info = fetchPushButtonParam("Info");
m_Button1 = fetchPushButtonParam("Button1");
m_Button2 = fetchPushButtonParam("Button2");
m_Button3 = fetchPushButtonParam("Button3");
}

void ChannelBoxPlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA)) {
ChannelBox channelBox(*this);
setupAndProcess(channelBox, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool ChannelBoxPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
int choice_i;
m_Choice->getValueAtTime(p_Args.time, choice_i);
ChoiceEnum choice = (ChoiceEnum)choice_i;
int _choice = choice_i;
if (_choice != 0 && _choice != 1) {
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void ChannelBoxPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if (p_ParamName == "Info") {
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

if (p_ParamName == kParamChoice) {
int choice_i;
m_Choice->getValueAtTime(p_Args.time, choice_i);
ChoiceEnum choice = (ChoiceEnum)choice_i;
bool limit = choice_i == 0;
bool swap = choice_i == 1;
if (limit) {
m_Versus->setIsSecretAndDisabled(!swap);
m_ChannelBox->setIsSecretAndDisabled(false);
}

if (swap) {
m_Versus->setIsSecretAndDisabled(false);
m_ChannelBox->setIsSecretAndDisabled(!limit);
}}

if (p_ParamName == "Blur") {
float blur = m_Blur->getValueAtTime(p_Args.time);
bool gc = blur != 0.0;
m_Garbage->setEnabled(gc);
m_Core->setEnabled(gc);
}

if (p_ParamName == "Button1") {
string PATH;
m_Path->getValue(PATH);
string NAME;
m_Name->getValue(NAME);
OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
if (reply == OFX::Message::eMessageReplyYes) {
FILE * pFile;
pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "// ChannelBox plugin DCTL export\n" \
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
}}}

if (p_ParamName == "Button2") {
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
" name ChannelBox\n" \
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
}}}

if (p_ParamName == "Button3") {

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
"// ChannelBox Shader  \n" \
"  \n" \
"#version 120  \n" \
"uniform sampler2D front;  \n" \
"uniform float adsk_result_w, adsk_result_h;  \n" \
"\n");
fclose (pFile);
fprintf (pFile2,
"<ShaderNodePreset SupportsAdaptiveDegradation=\"0\" Description=\"ChannelBox\" Name=\"ChannelBox\"> \n" \
"<Shader OutputBitDepth=\"Output\" Index=\"1\"> \n" \
"</Shader> \n" \
"<Page Name=\"ChannelBox\" Page=\"0\"> \n" \
"<Col Name=\"ChannelBox\" Col=\"0\" Page=\"0\"> \n" \
"</Col> \n" \
"</Page> \n" \
"</ShaderNodePreset> \n");
fclose (pFile2);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".glsl and " + NAME + ".xml to " + PATH2  + ". Check Permissions."));
}}}}

void ChannelBoxPlugin::setupAndProcess(ChannelBox& p_ChannelBox, const OFX::RenderArguments& p_Args)
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

int choice_i;
m_Choice->getValueAtTime(p_Args.time, choice_i);
ChoiceEnum choice = (ChoiceEnum)choice_i;
int channelBox_i;
m_ChannelBox->getValueAtTime(p_Args.time, channelBox_i);
ChannelBoxEnum channelBox = (ChannelBoxEnum)channelBox_i;
int luminanceMath_i;
m_luminanceMath->getValueAtTime(p_Args.time, luminanceMath_i);
LuminanceMathEnum luminanceMath = (LuminanceMathEnum)luminanceMath_i;
int _choice = choice_i;
int _channelBox = channelBox_i;
int _lumaMath = luminanceMath_i;
float _channelSwap[9];
int _switch[2];
float _mask[4];
_channelSwap[0] = m_RedvGreen->getValueAtTime(p_Args.time);
_channelSwap[1] = m_RedvBlue->getValueAtTime(p_Args.time);
_channelSwap[2] = m_RedvGreenBlue->getValueAtTime(p_Args.time);
_channelSwap[3] = m_GreenvRed->getValueAtTime(p_Args.time);
_channelSwap[4] = m_GreenvBlue->getValueAtTime(p_Args.time);
_channelSwap[5] = m_GreenvRedBlue->getValueAtTime(p_Args.time);
_channelSwap[6] = m_BluevRed->getValueAtTime(p_Args.time);
_channelSwap[7] = m_BluevGreen->getValueAtTime(p_Args.time);
_channelSwap[8] = m_BluevRedGreen->getValueAtTime(p_Args.time);

bool aSwitch = m_SwitchA->getValueAtTime(p_Args.time);
bool bSwitch = m_SwitchB->getValueAtTime(p_Args.time);

_switch[0] = aSwitch ? 1 : 0;
_switch[1] = bSwitch ? 1 : 0;

_mask[0] = m_Mask->getValueAtTime(p_Args.time);
_mask[1] = m_Blur->getValueAtTime(p_Args.time);
_mask[2] = m_Garbage->getValueAtTime(p_Args.time);
_mask[3] = m_Core->getValueAtTime(p_Args.time);

p_ChannelBox.setDstImg(dst.get());
p_ChannelBox.setSrcImg(src.get());

// Setup GPU Render arguments
p_ChannelBox.setGPURenderArgs(p_Args);

p_ChannelBox.setRenderWindow(p_Args.renderWindow);

p_ChannelBox.setScales(_choice, _channelBox, _channelSwap, _lumaMath, _switch, _mask);

p_ChannelBox.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ChannelBoxPluginFactory::ChannelBoxPluginFactory()
: OFX::PluginFactoryHelper<ChannelBoxPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ChannelBoxPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
p_Desc.setSupportsCudaRender(true);
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, 
const std::string& p_Label, const std::string& p_Hint, GroupParamDescriptor* p_Parent) {
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
param->setParent(*p_Parent);
return param;
}

void ChannelBoxPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

ChoiceParamDescriptor* choiceparam = p_Desc.defineChoiceParam(kParamChoice);
choiceparam->setLabel(kParamChoiceLabel);
choiceparam->setHint(kParamChoiceHint);
assert(choiceparam->getNOptions() == eChoiceLimit);
choiceparam->appendOption(kParamChoiceOptionLimit, kParamChoiceLimitHint);
assert(choiceparam->getNOptions() == eChoiceSwap);
choiceparam->appendOption(kParamChoiceOptionSwap, kParamChoiceSwapHint);
page->addChild(*choiceparam);

choiceparam = p_Desc.defineChoiceParam(kParamChannelBox);
choiceparam->setLabel(kParamChannelBoxLabel);
choiceparam->setHint(kParamChannelBoxHint);
assert(choiceparam->getNOptions() == eChannelBoxBR);
choiceparam->appendOption(kParamChannelBoxOptionBR, kParamChannelBoxOptionBRHint);
assert(choiceparam->getNOptions() == eChannelBoxBG);
choiceparam->appendOption(kParamChannelBoxOptionBG, kParamChannelBoxOptionBGHint);
assert(choiceparam->getNOptions() == eChannelBoxBGR);
choiceparam->appendOption(kParamChannelBoxOptionBGR, kParamChannelBoxOptionBGRHint);
assert(choiceparam->getNOptions() == eChannelBoxBGRX);
choiceparam->appendOption(kParamChannelBoxOptionBGRX, kParamChannelBoxOptionBGRXHint);
assert(choiceparam->getNOptions() == eChannelBoxGR);
choiceparam->appendOption(kParamChannelBoxOptionGR, kParamChannelBoxOptionGRHint);
assert(choiceparam->getNOptions() == eChannelBoxGB);
choiceparam->appendOption(kParamChannelBoxOptionGB, kParamChannelBoxOptionGBHint);
assert(choiceparam->getNOptions() == eChannelBoxGBR);
choiceparam->appendOption(kParamChannelBoxOptionGBR, kParamChannelBoxOptionGBRHint);
assert(choiceparam->getNOptions() == eChannelBoxGBRX);
choiceparam->appendOption(kParamChannelBoxOptionGBRX, kParamChannelBoxOptionGBRXHint);
assert(choiceparam->getNOptions() == eChannelBoxRG);
choiceparam->appendOption(kParamChannelBoxOptionRG, kParamChannelBoxOptionRGHint);
assert(choiceparam->getNOptions() == eChannelBoxRB);
choiceparam->appendOption(kParamChannelBoxOptionRB, kParamChannelBoxOptionRBHint);
assert(choiceparam->getNOptions() == eChannelBoxRBG);
choiceparam->appendOption(kParamChannelBoxOptionRBG, kParamChannelBoxOptionRBGHint);
assert(choiceparam->getNOptions() == eChannelBoxRBGX);
choiceparam->appendOption(kParamChannelBoxOptionRBGX, kParamChannelBoxOptionRBGXHint);
choiceparam->setIsSecretAndDisabled(false);
page->addChild(*choiceparam);

PushButtonParamDescriptor* pushparam = p_Desc.definePushButtonParam("Info");
pushparam->setLabel("info");
page->addChild(*pushparam);

GroupParamDescriptor* versus = p_Desc.defineGroupParam("Channel Swap");
versus->setOpen(true);
versus->setHint("channel swap");
versus->setIsSecretAndDisabled(true);

DoubleParamDescriptor* param = defineScaleParam(p_Desc, "ScaleRvG", "RED vs Green", "Swap Red with Green", versus);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);   

param = defineScaleParam(p_Desc, "ScaleRvB", "RED vs Blue", "Swap Red with Blue", versus);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleRvGB", "RED vs GreenBlue", "Swap Red with Green & Blue", versus);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param); 

param = defineScaleParam(p_Desc, "ScaleGvR", "GREEN vs Red", "Swap Green with Red", versus);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param); 

param = defineScaleParam(p_Desc, "ScaleGvB", "GREEN vs Blue", "Swap Green with Blue", versus);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param); 

param = defineScaleParam(p_Desc, "ScaleGvRB", "GREEN vs RedBlue", "Swap Green with Red & Blue", versus);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param); 

param = defineScaleParam(p_Desc, "ScaleBvR", "BLUE vs Red", "Swap Blue with Red", versus);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param); 

param = defineScaleParam(p_Desc, "ScaleBvG", "BLUE vs Green", "Swap Blue with Green", versus);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param); 

param = defineScaleParam(p_Desc, "ScaleBvRG", "BLUE vs RedGreen", "Swap Blue with Red & Green", versus);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("Preserve");
boolParam->setDefault(false);
boolParam->setHint("scales channels to preserve original luma");
boolParam->setLabel("preserve luma");
page->addChild(*boolParam);

choiceparam = p_Desc.defineChoiceParam(kParamLuminanceMath);
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

boolParam = p_Desc.defineBooleanParam("Display");
boolParam->setDefault(false);
boolParam->setHint("Displays alpha on RGB Channels");
boolParam->setLabel("display mask");
page->addChild(*boolParam);

param = defineScaleParam(p_Desc, "Mask", "luma mask", "shadows < 0 > highlights", 0);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Blur", "blur", "Gaussian blur the alpha", 0);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Garbage", "garbage", "Alpha garbage", 0);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
param->setEnabled(false);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Core", "core", "Alpha core", 0);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
param->setEnabled(false);
page->addChild(*param);

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
stringparam->setDefault("ChannelBox");
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

ImageEffect* ChannelBoxPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
return new ChannelBoxPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static ChannelBoxPluginFactory ChannelBoxPlugin;
p_FactoryArray.push_back(&ChannelBoxPlugin);
}