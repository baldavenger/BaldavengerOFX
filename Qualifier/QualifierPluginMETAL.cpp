#include "QualifierPlugin.h"

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

#define kPluginName "Qualifier"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"Qualifier: Combined Luma, Saturation, and Hue based keyer. Use eyedropper to isolate specific \n" \
"range and finetune with the controls."

#define kPluginIdentifier "BaldavengerOFX.Qualifier"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 3

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kResolutionScale	(float)width / 1920.0f

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
#define kParamLuminanceMathOptionMaximumHint "Use MAX or r, g, b."

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

#define kParamOutputAlpha "OutputAlpha"
#define kParamOutputAlphaLabel "output alpha"
#define kParamOutputAlphaHint "Output alpha channel. This can either be one of the coefficients for hue, saturation, luma, or a combination of those."
#define kParamOutputAlphaOptionAll "Hue Saturation Luma"
#define kParamOutputAlphaOptionAllHint "Alpha is set to min(Hue mask,Saturation mask,Luma mask)"
#define kParamOutputAlphaOptionHue "Hue"
#define kParamOutputAlphaOptionHueHint "Set Alpha to the Hue modification mask"
#define kParamOutputAlphaOptionSaturation "Saturation"
#define kParamOutputAlphaOptionSaturationHint "Set Alpha to the Saturation modification mask"
#define kParamOutputAlphaOptionLuma "Luma"
#define kParamOutputAlphaOptionLumaHint "Alpha is set to the Luma mask"
#define kParamOutputAlphaOptionHueSaturation "Hue Saturation"
#define kParamOutputAlphaOptionHueSaturationHint "Alpha is set to min(Hue mask,Saturation mask)"
#define kParamOutputAlphaOptionHueLuma "Hue Luma"
#define kParamOutputAlphaOptionHueLumaHint "Alpha is set to min(Hue mask,Luma mask)"
#define kParamOutputAlphaOptionSaturationLuma "Saturation Luma"
#define kParamOutputAlphaOptionSaturationLumaHint "Alpha is set to min(Saturation mask,Luma mask)"
#define kParamOutputAlphaOptionOff "Off"
#define kParamOutputAlphaOptionOffHint "Alpha channel is kept unmodified"

enum OutputAlphaEnum
{
eOutputAlphaAll,
eOutputAlphaHue,
eOutputAlphaSaturation,
eOutputAlphaLuma,
eOutputAlphaHueSaturation,
eOutputAlphaHueLuma,
eOutputAlphaSaturationLuma,
eOutputAlphaOff,
};

#define MIN_SATURATION 0.1f
#define MIN_VALUE 0.1f
#define OFXS_HUE_CIRCLE 1.0f

////////////////////////////////////////////////////////////////////////////////

namespace {
struct RGBValues {
double r,g,b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};
}

inline float RGB_to_Luma(float R, float G, float B, int L) {
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

void rgb_to_hsv( float r, float g, float b, float *h, float *s, float *v ) {
float min = fmin(fmin(r, g), b);
float max = fmax(fmax(r, g), b);
*v = max;
float delta = max - min;
if (max != 0.) {
*s = delta / max;
} else {
*s = 0.f;
*h = 0.f;
return;
}
if (delta == 0.) {
*h = 0.f;
} else if (r == max) {
*h = (g - b) / delta;
} else if (g == max) {
*h = 2 + (b - r) / delta;
} else {
*h = 4 + (r - g) / delta;
}
*h *= OFXS_HUE_CIRCLE / 6.;
if (*h < 0) {
*h += OFXS_HUE_CIRCLE;
}}

class Qualifier : public OFX::ImageProcessor
{
public:
explicit Qualifier(OFX::ImageEffect& p_Instance);

virtual void processImagesMetal();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(int* p_Switch, float* p_AlphaH, float* p_AlphaS, float* p_AlphaL, float p_Mix, int p_Math, int p_OutputAlpha,
float p_Black, float p_White, float p_Blur, float p_Garbage, float p_Core, float p_Erode, float p_Dilate, float* p_HSV);

private:
OFX::Image* _srcImg;
int _switch[6];
float _alphaH[7];
float _alphaS[7];
float _alphaL[7];
float _mix;
int _math;
int _outputAlpha;
float _black;
float _white;
float _blur;
float _garbage;
float _core;
float _erode;
float _dilate;
float _hsv[3];
};

Qualifier::Qualifier(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, 
int* p_Switch, float* p_AlphaH, float* p_AlphaS, float* p_AlphaL, float p_Mix, int p_Math, int p_OutputAlpha, 
float p_Black, float p_White, float p_Blur, float p_Garbage, float p_Core, float p_Erode, float p_Dilate, float* p_HSV);
#endif

void Qualifier::processImagesMetal()
{
#ifdef __APPLE__
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

_blur *= kResolutionScale;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunMetalKernel(_pMetalCmdQ, input, output, width, height, _switch, _alphaH, _alphaS, _alphaL, _mix, _math,
_outputAlpha, _black, _white, _blur, _garbage, _core, _erode, _dilate, _hsv);
#endif
}

void Qualifier::multiThreadProcessImages(OfxRectI p_ProcWindow)
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
dstPix[c] = 0.0f;
}}
dstPix += 4;
}}}

void Qualifier::setSrcImg(OFX::Image* p_SrcImg)
{
_srcImg = p_SrcImg;
}

void Qualifier::setScales(int* p_Switch, float* p_AlphaH, float* p_AlphaS, float* p_AlphaL, float p_Mix, int p_Math, int p_OutputAlpha, 
float p_Black, float p_White, float p_Blur, float p_Garbage, float p_Core, float p_Erode, float p_Dilate, float* p_HSV)
{
_alphaH[0] = p_AlphaH[0];
_alphaH[1] = p_AlphaH[1];
_alphaH[2] = p_AlphaH[2];
_alphaH[3] = p_AlphaH[3];
_alphaH[4] = p_AlphaH[4];
_alphaH[5] = p_AlphaH[5];
_alphaH[6] = p_AlphaH[6];

_alphaS[0] = p_AlphaS[0];
_alphaS[1] = p_AlphaS[1];
_alphaS[2] = p_AlphaS[2];
_alphaS[3] = p_AlphaS[3];
_alphaS[4] = p_AlphaS[4];
_alphaS[5] = p_AlphaS[5];
_alphaS[6] = p_AlphaS[6];

_alphaL[0] = p_AlphaL[0];
_alphaL[1] = p_AlphaL[1];
_alphaL[2] = p_AlphaL[2];
_alphaL[3] = p_AlphaL[3];
_alphaL[4] = p_AlphaL[4];
_alphaL[5] = p_AlphaL[5];
_alphaL[6] = p_AlphaL[6];

_switch[0] = p_Switch[0];
_switch[1] = p_Switch[1];
_switch[2] = p_Switch[2];
_switch[3] = p_Switch[3];
_switch[4] = p_Switch[4];
_switch[5] = p_Switch[5];

_mix = p_Mix;
_black = p_Black;
_white = p_White;
_blur = p_Blur;
_garbage = p_Garbage;
_core = p_Core;
_erode = p_Erode;
_dilate = p_Dilate;
_math = p_Math;
_outputAlpha = p_OutputAlpha;
_hsv[0] = p_HSV[0];
_hsv[1] = p_HSV[1];
_hsv[2] = p_HSV[2];
}

////////////////////////////////////////////////////////////////////////////////

class QualifierPlugin : public OFX::ImageEffect
{
public:
explicit QualifierPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);

virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

void setupAndProcess(Qualifier &p_Qualifier, const OFX::RenderArguments& p_Args);

private:
OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;

OFX::GroupParam* m_GroupH;
OFX::GroupParam* m_GroupS;
OFX::GroupParam* m_GroupL;

OFX::ChoiceParam* m_luminanceMath;
OFX::ChoiceParam* m_outputAlpha;
OFX::RGBParam *m_Sample;

OFX::DoubleParam* m_AlphaHA;
OFX::DoubleParam* m_AlphaHB;
OFX::DoubleParam* m_AlphaHC;
OFX::DoubleParam* m_AlphaHD;
OFX::DoubleParam* m_AlphaHE;
OFX::DoubleParam* m_AlphaHF;
OFX::DoubleParam* m_AlphaHO;

OFX::DoubleParam* m_AlphaSA;
OFX::DoubleParam* m_AlphaSB;
OFX::DoubleParam* m_AlphaSC;
OFX::DoubleParam* m_AlphaSD;
OFX::DoubleParam* m_AlphaSE;
OFX::DoubleParam* m_AlphaSF;
OFX::DoubleParam* m_AlphaSO;

OFX::DoubleParam* m_AlphaLA;
OFX::DoubleParam* m_AlphaLB;
OFX::DoubleParam* m_AlphaLC;
OFX::DoubleParam* m_AlphaLD;
OFX::DoubleParam* m_AlphaLE;
OFX::DoubleParam* m_AlphaLF;
OFX::DoubleParam* m_AlphaLO;

OFX::DoubleParam* m_Mix;
OFX::DoubleParam* m_Black;
OFX::DoubleParam* m_White;
OFX::DoubleParam* m_Blur;
OFX::DoubleParam* m_Garbage;
OFX::DoubleParam* m_Core;
OFX::DoubleParam* m_Erode;
OFX::DoubleParam* m_Dilate;

OFX::DoubleParam* m_HsvA;
OFX::DoubleParam* m_HsvB;
OFX::DoubleParam* m_HsvC;

OFX::BooleanParam* m_SwitchA;
OFX::BooleanParam* m_SwitchB;
OFX::BooleanParam* m_SwitchC;
OFX::BooleanParam* m_SwitchD;
OFX::BooleanParam* m_SwitchE;
OFX::BooleanParam* m_SwitchF;

OFX::StringParam* m_Path;
OFX::StringParam* m_Path2;
OFX::StringParam* m_Name;
OFX::PushButtonParam* m_Info;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
OFX::PushButtonParam* m_Button3;
};

QualifierPlugin::QualifierPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
m_GroupH = fetchGroupParam("Hue Range");
m_GroupS = fetchGroupParam("Saturation Range");
m_GroupL = fetchGroupParam("Luma Range");

m_luminanceMath = fetchChoiceParam(kParamLuminanceMath);
m_outputAlpha = fetchChoiceParam(kParamOutputAlpha);
m_Sample = fetchRGBParam("Sample");

m_AlphaHA = fetchDoubleParam("AlphaHA");
m_AlphaHB = fetchDoubleParam("AlphaHB");
m_AlphaHC = fetchDoubleParam("AlphaHC");
m_AlphaHD = fetchDoubleParam("AlphaHD");
m_AlphaHE = fetchDoubleParam("AlphaHE");
m_AlphaHF = fetchDoubleParam("AlphaHF");
m_AlphaHO = fetchDoubleParam("AlphaHO");

m_AlphaSA = fetchDoubleParam("AlphaSA");
m_AlphaSB = fetchDoubleParam("AlphaSB");
m_AlphaSC = fetchDoubleParam("AlphaSC");
m_AlphaSD = fetchDoubleParam("AlphaSD");
m_AlphaSE = fetchDoubleParam("AlphaSE");
m_AlphaSF = fetchDoubleParam("AlphaSF");
m_AlphaSO = fetchDoubleParam("AlphaSO");

m_AlphaLA = fetchDoubleParam("AlphaLA");
m_AlphaLB = fetchDoubleParam("AlphaLB");
m_AlphaLC = fetchDoubleParam("AlphaLC");
m_AlphaLD = fetchDoubleParam("AlphaLD");
m_AlphaLE = fetchDoubleParam("AlphaLE");
m_AlphaLF = fetchDoubleParam("AlphaLF");
m_AlphaLO = fetchDoubleParam("AlphaLO");

m_Mix = fetchDoubleParam("Mix");
m_Black = fetchDoubleParam("Black");
m_White = fetchDoubleParam("White");
m_Blur = fetchDoubleParam("Blur");
m_Garbage = fetchDoubleParam("Garbage");
m_Core = fetchDoubleParam("Core");
m_Erode = fetchDoubleParam("Erode");
m_Dilate = fetchDoubleParam("Dilate");

m_HsvA = fetchDoubleParam("Hue1");
m_HsvB = fetchDoubleParam("Sat1");
m_HsvC = fetchDoubleParam("Luma1");

m_SwitchA = fetchBooleanParam("Display");
m_SwitchB = fetchBooleanParam("InvertAlpha");
m_SwitchC = fetchBooleanParam("InvertAlphaH");
m_SwitchD = fetchBooleanParam("InvertAlphaS");
m_SwitchE = fetchBooleanParam("InvertAlphaL");
m_SwitchF = fetchBooleanParam("Warning");

m_Path = fetchStringParam("Path");
m_Path2 = fetchStringParam("Path2");
m_Name = fetchStringParam("Name");
m_Info = fetchPushButtonParam("Info");
m_Button1 = fetchPushButtonParam("Button1");
m_Button2 = fetchPushButtonParam("Button2");
m_Button3 = fetchPushButtonParam("Button3");
}

void QualifierPlugin::render(const OFX::RenderArguments& p_Args)
{
if (m_DstClip->getPixelDepth() == OFX::eBitDepthFloat && m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA) {
Qualifier qualifier(*this);
setupAndProcess(qualifier, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool QualifierPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
float AlphaL[4];
AlphaL[0] = m_AlphaLA->getValueAtTime(p_Args.time);
AlphaL[1] = m_AlphaLB->getValueAtTime(p_Args.time);
AlphaL[2] = m_AlphaLC->getValueAtTime(p_Args.time);
AlphaL[3] = m_AlphaLD->getValueAtTime(p_Args.time);
if (AlphaL[0] == 1.0f && AlphaL[1] == 1.0f && AlphaL[2] == 0.0f && AlphaL[3] == 0.0f) {
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void QualifierPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{

if(p_ParamName == "Info"){
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

if(p_ParamName == "Button1") {
string PATH;
m_Path->getValue(PATH);
string NAME;
m_Name->getValue(NAME);
OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
if (reply == OFX::Message::eMessageReplyYes) {
FILE * pFile;
pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "// Qualifier DCTL export \n" \
"\n" \
"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
"{\n" \
"float r = p_R;\n" \
"float g = p_G;\n" \
"float b = p_B;\n" \
"\n" \
"return make_float3(r, g, b);\n" \
"}\n"); 
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
}}}

if(p_ParamName == "Button2") {
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
" name Qualifier\n" \
"}\n" \
" Input {\n" \
"  inputs 0\n" \
"  name Input1\n" \
" }\n" \
" Output {\n" \
"  name Output1\n" \
" }\n" \
"end_group\n");
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
}}}

if(p_ParamName == "Button3") {
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
"// Qualifier Shader  \n" \
"  \n" \
"#version 120  \n" \
"uniform sampler2D front;  \n" \
"uniform float adsk_result_w, adsk_result_h;  \n" \
"\n");
fclose (pFile);
fprintf (pFile2,
"<ShaderNodePreset SupportsAdaptiveDegradation=\"0\" Description=\"Qualifier\" Name=\"Qualifier\"> \n" \
"<Shader OutputBitDepth=\"Output\" Index=\"1\"> \n" \
"</Shader> \n" \
"<Page Name=\"Qualifier\" Page=\"0\"> \n" \
"<Col Name=\"Qualifier\" Col=\"0\" Page=\"0\"> \n" \
"</Col> \n" \
"</Page> \n" \
"</ShaderNodePreset> \n");
fclose (pFile2);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".glsl and " + NAME + ".xml to " + PATH2  + ". Check Permissions."));
}}}

if (p_ParamName == "Sample") {
RGBValues sample;
m_Sample->getValueAtTime(p_Args.time, sample.r, sample.g, sample.b);
bool reset = sample.r == 1.0f && sample.g == 1.0f && sample.b == 1.0f;
int Math;
m_luminanceMath->getValueAtTime(p_Args.time, Math);
float hue, sat, v, lum;
lum = RGB_to_Luma(sample.r, sample.g, sample.b, Math);
rgb_to_hsv(sample.r, sample.g, sample.b, &hue, &sat, &v);
if (sat < MIN_SATURATION && v < MIN_VALUE)
hue = 0.0f;
float h1 = reset ? 1.0f : hue == 0.0f ? 0.0f : (hue + 0.2f > 1.0f ? 1.0f : hue + 0.2f);
float h2 = reset ? 0.0f : hue - 0.2f < 0.0001f ? 0.0001f : hue - 0.2f;
float h3 = reset ? 1.0f : hue > 0.9f ? 1.0f : 0.95f;
float h4 = reset ? 0.0f : hue < 0.1f ? 0.0f : 0.05f;
m_AlphaHA->setValue(h1);
m_AlphaHB->setValue(h3);
m_AlphaHC->setValue(h4);
m_AlphaHD->setValue(h2);
float S1 = reset ? 1.0f : sat + 0.2f > 1.0f ? 1.0f : sat + 0.2f;
float S2 = reset ? 0.0f : sat - 0.2f < 0.0f ? 0.0f : sat - 0.2f;
float S3 = reset ? 1.0f : sat > 0.9f ? 1.0f : 0.95f;
float S4 = reset ? 0.0f : sat < 0.1f ? 0.0f : 0.05f;
m_AlphaSA->setValue(S1);
m_AlphaSB->setValue(S3);
m_AlphaSC->setValue(S4);
m_AlphaSD->setValue(S2);
float L1 = reset ? 1.0f : lum + 0.2f > 1.0f ? 1.0f : lum + 0.2f;
float L2 = reset ? 0.0f : lum - 0.2f < 0.0f ? 0.0f : lum - 0.2f;
float L3 = reset ? 1.0f : lum > 0.9f ? 1.0f : 0.95f;
float L4 = reset ? 0.0f : lum < 0.1f ? 0.0f : 0.05f;
m_AlphaLA->setValue(L1);
m_AlphaLB->setValue(L3);
m_AlphaLC->setValue(L4);
m_AlphaLD->setValue(L2);
}

if (p_ParamName == "AlphaHA") {
float hl = m_AlphaHD->getValueAtTime(p_Args.time);
float HL = hl == 0.0f ? 0.0001f : hl;
m_AlphaHD->setValue(HL);
}

if (p_ParamName == "AlphaHB") {
float hl = m_AlphaHD->getValueAtTime(p_Args.time);
float HL = hl == 0.0f ? 0.0001f : hl;
m_AlphaHD->setValue(HL);
}

if (p_ParamName == "Blur") {
float blur = m_Blur->getValueAtTime(p_Args.time);
bool gc = blur != 0.0f;
m_Garbage->setEnabled(gc);
m_Core->setEnabled(gc);
}

if (p_ParamName == "OutputAlpha") {
int Output;
m_outputAlpha->getValueAtTime(p_Args.time, Output);
bool gh = Output == 2 || Output == 3 || Output == 6;
bool gs = Output == 1 || Output == 3 || Output == 5;
bool gl = Output == 1 || Output == 2 || Output == 4;
m_GroupH->setIsSecretAndDisabled(gh);
m_GroupS->setIsSecretAndDisabled(gs);
m_GroupL->setIsSecretAndDisabled(gl);
}}

void QualifierPlugin::setupAndProcess(Qualifier& p_Qualifier, const OFX::RenderArguments& p_Args)
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

int Math;
m_luminanceMath->getValueAtTime(p_Args.time, Math);
int Output;
m_outputAlpha->getValueAtTime(p_Args.time, Output);

float AlphaH[7], AlphaS[7], AlphaL[7], HSV[3];
int Switch[6];

AlphaH[0] = m_AlphaHA->getValueAtTime(p_Args.time);
AlphaH[1] = m_AlphaHB->getValueAtTime(p_Args.time);
AlphaH[2] = m_AlphaHC->getValueAtTime(p_Args.time);
AlphaH[3] = m_AlphaHD->getValueAtTime(p_Args.time);
AlphaH[4] = m_AlphaHE->getValueAtTime(p_Args.time);
AlphaH[5] = m_AlphaHF->getValueAtTime(p_Args.time);
AlphaH[6] = m_AlphaHO->getValueAtTime(p_Args.time);

AlphaS[0] = m_AlphaSA->getValueAtTime(p_Args.time);
AlphaS[1] = m_AlphaSB->getValueAtTime(p_Args.time);
AlphaS[2] = m_AlphaSC->getValueAtTime(p_Args.time);
AlphaS[3] = m_AlphaSD->getValueAtTime(p_Args.time);
AlphaS[4] = m_AlphaSE->getValueAtTime(p_Args.time);
AlphaS[5] = m_AlphaSF->getValueAtTime(p_Args.time);
AlphaS[6] = m_AlphaSO->getValueAtTime(p_Args.time);

AlphaL[0] = m_AlphaLA->getValueAtTime(p_Args.time);
AlphaL[1] = m_AlphaLB->getValueAtTime(p_Args.time);
AlphaL[2] = m_AlphaLC->getValueAtTime(p_Args.time);
AlphaL[3] = m_AlphaLD->getValueAtTime(p_Args.time);
AlphaL[4] = m_AlphaLE->getValueAtTime(p_Args.time);
AlphaL[5] = m_AlphaLF->getValueAtTime(p_Args.time);
AlphaL[6] = m_AlphaLO->getValueAtTime(p_Args.time);

float mix = m_Mix->getValueAtTime(p_Args.time);
float black = m_Black->getValueAtTime(p_Args.time);
float white = m_White->getValueAtTime(p_Args.time);
float blur = m_Blur->getValueAtTime(p_Args.time);
float garbage = m_Garbage->getValueAtTime(p_Args.time);
float core = m_Core->getValueAtTime(p_Args.time);
float erode = m_Erode->getValueAtTime(p_Args.time);
float dilate = m_Dilate->getValueAtTime(p_Args.time);

HSV[0] = m_HsvA->getValueAtTime(p_Args.time);
HSV[1] = m_HsvB->getValueAtTime(p_Args.time);
HSV[2] = m_HsvC->getValueAtTime(p_Args.time);

bool aSwitch = m_SwitchA->getValueAtTime(p_Args.time);
bool bSwitch = m_SwitchB->getValueAtTime(p_Args.time);
bool cSwitch = m_SwitchC->getValueAtTime(p_Args.time);
bool dSwitch = m_SwitchD->getValueAtTime(p_Args.time);
bool eSwitch = m_SwitchE->getValueAtTime(p_Args.time);
bool fSwitch = m_SwitchF->getValueAtTime(p_Args.time);

Switch[0] = aSwitch ? 1 : 0;
Switch[1] = bSwitch ? 1 : 0;
Switch[2] = cSwitch ? 1 : 0;
Switch[3] = dSwitch ? 1 : 0;
Switch[4] = eSwitch ? 1 : 0;
Switch[5] = fSwitch ? 1 : 0;

p_Qualifier.setDstImg(dst.get());
p_Qualifier.setSrcImg(src.get());

// Setup GPU Render arguments
p_Qualifier.setGPURenderArgs(p_Args);

p_Qualifier.setRenderWindow(p_Args.renderWindow);

p_Qualifier.setScales(Switch, AlphaH, AlphaS, AlphaL, mix, Math, Output, 
black, white, blur, garbage, core, erode, dilate, HSV);

p_Qualifier.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

QualifierPluginFactory::QualifierPluginFactory()
: OFX::PluginFactoryHelper<QualifierPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void QualifierPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

#ifdef __APPLE__
p_Desc.setSupportsMetalRender(true);
#endif
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, 
const std::string& p_Label, const std::string& p_Hint, GroupParamDescriptor* p_Parent) {
DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
param->setLabels(p_Label, p_Label, p_Label);
param->setScriptName(p_Name);
param->setHint(p_Hint);
param->setDefault(1.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.0f, 1.0f);
param->setDoubleType(eDoubleTypeScale);
if (p_Parent)
param->setParent(*p_Parent);
return param;
}

void QualifierPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

RGBParamDescriptor *rgbparam = p_Desc.defineRGBParam("Sample");
rgbparam->setLabel("hsl sample");
rgbparam->setHint("click on pixel");
rgbparam->setDefault(1.0f, 1.0f, 1.0f);
rgbparam->setDisplayRange(0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f);
rgbparam->setAnimates(true);
page->addChild(*rgbparam);

BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("Display");
boolParam->setDefault(false);
boolParam->setHint("Displays alpha");
boolParam->setLabel("display alpha");
page->addChild(*boolParam);

boolParam = p_Desc.defineBooleanParam("InvertAlpha");
boolParam->setDefault(false);
boolParam->setHint("Inverts the Alpha Channel");
boolParam->setLabel("invert");
page->addChild(*boolParam);

boolParam = p_Desc.defineBooleanParam("Warning");
boolParam->setDefault(false);
boolParam->setHint("Non-pure white = yellow, non-pure black = blue");
boolParam->setLabel("alpha warning");
page->addChild(*boolParam);

ChoiceParamDescriptor *chparam = p_Desc.defineChoiceParam(kParamOutputAlpha);
chparam->setLabel(kParamOutputAlphaLabel);
chparam->setHint(kParamOutputAlphaHint);
assert(chparam->getNOptions() == (int)eOutputAlphaAll);
chparam->appendOption(kParamOutputAlphaOptionAll, kParamOutputAlphaOptionAllHint);
assert(chparam->getNOptions() == (int)eOutputAlphaHue);
chparam->appendOption(kParamOutputAlphaOptionHue, kParamOutputAlphaOptionHueHint);
assert(chparam->getNOptions() == (int)eOutputAlphaSaturation);
chparam->appendOption(kParamOutputAlphaOptionSaturation, kParamOutputAlphaOptionSaturationHint);
assert(chparam->getNOptions() == (int)eOutputAlphaLuma);
chparam->appendOption(kParamOutputAlphaOptionLuma, kParamOutputAlphaOptionLumaHint);
assert(chparam->getNOptions() == (int)eOutputAlphaHueSaturation);
chparam->appendOption(kParamOutputAlphaOptionHueSaturation, kParamOutputAlphaOptionHueSaturationHint);
assert(chparam->getNOptions() == (int)eOutputAlphaHueLuma);
chparam->appendOption(kParamOutputAlphaOptionHueLuma, kParamOutputAlphaOptionHueLumaHint);
assert(chparam->getNOptions() == (int)eOutputAlphaSaturationLuma);
chparam->appendOption(kParamOutputAlphaOptionSaturationLuma, kParamOutputAlphaOptionSaturationLumaHint);
assert(chparam->getNOptions() == (int)eOutputAlphaOff);
chparam->appendOption(kParamOutputAlphaOptionOff, kParamOutputAlphaOptionOffHint);
chparam->setDefault( (int)eOutputAlphaAll );
chparam->setAnimates(false);
page->addChild(*chparam);

PushButtonParamDescriptor* pushparam = p_Desc.definePushButtonParam("Info");
pushparam->setLabel("Info");
page->addChild(*pushparam);

GroupParamDescriptor* hue = p_Desc.defineGroupParam("Hue Range");
hue->setOpen(false);
hue->setHint("Adjust hue range");
if (page)
page->addChild(*hue);

boolParam = p_Desc.defineBooleanParam("InvertAlphaH");
boolParam->setDefault(false);
boolParam->setHint("Inverts the Alpha Channel");
boolParam->setLabel("invert");
boolParam->setParent(*hue);

DoubleParamDescriptor* param = defineScaleParam(p_Desc, "AlphaHA", "high", "High limit of alpha channel", hue);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaHB", "high fade", "Roll-off between high limit", hue);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaHC", "low fade", "Roll-off between low limit", hue);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaHD", "low", "Low limit of alpha channel", hue);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaHE", "high fade curve", "Easy out / Easy in", hue);
param->setDefault(1.0f);
param->setRange(0.2f, 5.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.2f, 5.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaHF", "low fade curve", "Easy out / Easy in", hue);
param->setDefault(1.0f);
param->setRange(0.2f, 5.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.2f, 5.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaHO", "rotate", "Rotate Hue Wheel", hue);
param->setDefault(0.0f);
param->setRange(-1.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(-1.0f, 1.0f);
page->addChild(*param);

GroupParamDescriptor* sat = p_Desc.defineGroupParam("Saturation Range");
sat->setOpen(false);
sat->setHint("Adjust saturation range");
if (page)
page->addChild(*sat);

boolParam = p_Desc.defineBooleanParam("InvertAlphaS");
boolParam->setDefault(false);
boolParam->setHint("Inverts the Alpha Channel");
boolParam->setLabel("Invert");
boolParam->setParent(*sat);

param = defineScaleParam(p_Desc, "AlphaSA", "high", "High limit of alpha channel", sat);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaSB", "high fade", "Roll-off between high limit", sat);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaSC", "low fade", "Roll-off between low limit", sat);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaSD", "low", "Low limit of alpha channel", sat);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaSE", "high fade curve", "Easy out / Easy in", sat);
param->setDefault(1.0f);
param->setRange(0.2f, 5.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.2f, 5.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaSF", "low fade curve", "Easy out / Easy in", sat);
param->setDefault(1.0f);
param->setRange(0.2f, 5.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.2f, 5.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaSO", "offset", "Offset Saturation", sat);
param->setDefault(0.0f);
param->setRange(-1.0f, 1.0f);
param->setIncrement(0.001);
param->setDisplayRange(-1.0f, 1.0f);
page->addChild(*param);

GroupParamDescriptor* luma = p_Desc.defineGroupParam("Luma Range");
luma->setOpen(false);
luma->setHint("Adjust luma range");
if (page)
page->addChild(*luma);

boolParam = p_Desc.defineBooleanParam("InvertAlphaL");
boolParam->setDefault(false);
boolParam->setHint("Inverts the Alpha Channel");
boolParam->setLabel("invert");
boolParam->setParent(*luma);

chparam = p_Desc.defineChoiceParam(kParamLuminanceMath);
chparam->setLabel(kParamLuminanceMathLabel);
chparam->setHint(kParamLuminanceMathHint);
assert(chparam->getNOptions() == eLuminanceMathRec709);
chparam->appendOption(kParamLuminanceMathOptionRec709, kParamLuminanceMathOptionRec709Hint);
assert(chparam->getNOptions() == eLuminanceMathRec2020);
chparam->appendOption(kParamLuminanceMathOptionRec2020, kParamLuminanceMathOptionRec2020Hint);
assert(chparam->getNOptions() == eLuminanceMathDCIP3);
chparam->appendOption(kParamLuminanceMathOptionDCIP3, kParamLuminanceMathOptionDCIP3Hint);
assert(chparam->getNOptions() == eLuminanceMathACESAP0);
chparam->appendOption(kParamLuminanceMathOptionACESAP0, kParamLuminanceMathOptionACESAP0Hint);
assert(chparam->getNOptions() == eLuminanceMathACESAP1);
chparam->appendOption(kParamLuminanceMathOptionACESAP1, kParamLuminanceMathOptionACESAP1Hint);
assert(chparam->getNOptions() == eLuminanceMathAverage);
chparam->appendOption(kParamLuminanceMathOptionAverage, kParamLuminanceMathOptionAverageHint);
assert(chparam->getNOptions() == eLuminanceMathMaximum);
chparam->appendOption(kParamLuminanceMathOptionMaximum, kParamLuminanceMathOptionMaximumHint);
chparam->setParent(*luma);
page->addChild(*chparam);

param = defineScaleParam(p_Desc, "AlphaLA", "high", "High limit of alpha channel", luma);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaLB", "high fade", "Roll-off between high limit", luma);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaLC", "low fade", "Roll-off between low limit", luma);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaLD", "low", "Low limit of alpha channel", luma);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaLE", "high fade curve", "Easy out / Easy in", luma);
param->setDefault(1.0f);
param->setRange(0.2f, 5.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.2f, 5.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaLF", "low fade curve", "Easy out / Easy in", luma);
param->setDefault(1.0f);
param->setRange(0.2f, 5.0f);
param->setIncrement(0.001);
param->setDisplayRange(0.2f, 5.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "AlphaLO", "offset", "Offset Luma", luma);
param->setDefault(0.0f);
param->setRange(-1.0f, 1.0f);
param->setIncrement(0.001);
param->setDisplayRange(-1.0f, 1.0f);
page->addChild(*param);

GroupParamDescriptor* clean = p_Desc.defineGroupParam("Clean Alpha");
clean->setOpen(false);
clean->setHint("Clean alpha tools");
if (page)
page->addChild(*clean);

param = defineScaleParam(p_Desc, "Black", "clip black", "Clip alpha shadows", clean);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "White", "clip white", "Clip alpha highlights", clean);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Blur", "blur", "Gaussian blur the alpha", clean);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Garbage", "garbage", "Alpha garbage", clean);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
param->setEnabled(false);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Core", "core", "Alpha core", clean);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
param->setEnabled(false);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Erode", "erode", "Shrink black", clean);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Dilate", "dilate", "Shrink white", clean);
param->setDefault(0.0f);
param->setRange(0.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(0.0f, 1.0f);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Mix", "mix", "Blends new alpha with original alpha", 0);
param->setDefault(0.0f);
param->setRange(-1.0f, 1.0f);
param->setIncrement(0.001f);
param->setDisplayRange(-1.0f, 1.0f);
page->addChild(*param);

GroupParamDescriptor* huesatv = p_Desc.defineGroupParam("HSV");
huesatv->setOpen(false);
huesatv->setHint("Adjust hue, sat, luma");
if (page)
page->addChild(*huesatv);

param = defineScaleParam(p_Desc, "Hue1", "hue", "hue rotate", huesatv);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Sat1", "sat", "saturation adjust", huesatv);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Luma1", "luma", "luma adjust", huesatv);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
script->setOpen(false);
script->setHint("export DCTL and Nuke script");
if (page)
page->addChild(*script);

pushparam = p_Desc.definePushButtonParam("Button1");
pushparam->setLabel("Export DCTL");
pushparam->setHint("create DCTL version");
pushparam->setParent(*script);
page->addChild(*pushparam);

pushparam = p_Desc.definePushButtonParam("Button2");
pushparam->setLabel("Export Nuke script");
pushparam->setHint("create NUKE version");
pushparam->setParent(*script);
page->addChild(*pushparam);

StringParamDescriptor* stringparam = p_Desc.defineStringParam("Name");
stringparam->setLabel("Name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("Qualifier");
stringparam->setParent(*script);
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam("Path");
stringparam->setLabel("Directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript);
stringparam->setFilePathExists(false);
stringparam->setParent(*script);
page->addChild(*stringparam);

pushparam = p_Desc.definePushButtonParam("Button3");
pushparam->setLabel("Export Shader");
pushparam->setHint("create Shader version");
pushparam->setParent(*script);
page->addChild(*pushparam);

stringparam = p_Desc.defineStringParam("Path2");
stringparam->setLabel("Shader Directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript2);
stringparam->setFilePathExists(false);
stringparam->setParent(*script);
page->addChild(*stringparam);
}

ImageEffect* QualifierPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum) {
return new QualifierPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static QualifierPluginFactory QualifierPlugin;
p_FactoryArray.push_back(&QualifierPlugin);
}