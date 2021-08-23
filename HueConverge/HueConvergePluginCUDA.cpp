#include "HueConvergePlugin.h"

#include <stdio.h>
#include <cmath>
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
#define kPluginScript "/opt/resolve/LUT"
#endif

#define kPluginName "HueConverge"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"HueConverge: Allows for the isolation and rotation + convergence of specific ranges of hue. Also \n" \
"included are log controls, saturation soft-clip, and individual alphas."

#define kPluginIdentifier "BaldavengerOFX.HueConverge"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kResolutionScale	(float)width / 1920.0f

#define kParamHueMedian "hueMedian"
#define kParamHueMedianLabel "hue median"
#define kParamHueMedianHint "apply median filter to hue channel"
#define kParamHueMedianOptionOff "Off"
#define kParamHueMedianOptionOffHint "no median filter"
#define kParamHueMedianOption3x3 "3x3 Median Filter"
#define kParamHueMedianOption3x3Hint "apply 3x3 median filter to hue channel"
#define kParamHueMedianOption5x5 "5x5 Median Filter"
#define kParamHueMedianOption5x5Hint "apply 5x5 median filter to hue channel"

enum HueMedianEnum {
eHueMedianOff,
eHueMedian3x3,
eHueMedian5x5,
};

#define kParamLuminanceMath "luminanceMath"
#define kParamLuminanceMathLabel "luma math"
#define kParamLuminanceMathHint "Formula used to compute luminance from RGB values"
#define kParamLuminanceMathOptionRec709 "Rec.709"
#define kParamLuminanceMathOptionRec709Hint "Use Rec.709 (0.2126r + 0.7152g + 0.0722b)"
#define kParamLuminanceMathOptionRec2020 "Rec.2020"
#define kParamLuminanceMathOptionRec2020Hint "Use Rec.2020 (0.2627r + 0.6780g + 0.0593b)"
#define kParamLuminanceMathOptionDCIP3 "DCI P3"
#define kParamLuminanceMathOptionDCIP3Hint "Use DCI P3 (0.209492r + 0.721595g + 0.0689131b)"
#define kParamLuminanceMathOptionACESAP0 "ACES AP0"
#define kParamLuminanceMathOptionACESAP0Hint "Use ACES AP0 (0.3439664498r + 0.7281660966g + -0.0721325464b)"
#define kParamLuminanceMathOptionACESAP1 "ACES AP1"
#define kParamLuminanceMathOptionACESAP1Hint "Use ACES AP1 (0.2722287168r +  0.6740817658g +  0.0536895174b)"
#define kParamLuminanceMathOptionAverage "Average"
#define kParamLuminanceMathOptionAverageHint "Use average of r, g, b"
#define kParamLuminanceMathOptionMaximum "Max"
#define kParamLuminanceMathOptionMaximumHint "Use MAX or r, g, b"


enum LuminanceMathEnum {
eLuminanceMathRec709,
eLuminanceMathRec2020,
eLuminanceMathDCIP3,
eLuminanceMathACESAP0,
eLuminanceMathACESAP1,
eLuminanceMathAverage,
eLuminanceMathMaximum,
};

#define kParamIsolate "isolate"
#define kParamIsolateLabel "isolate hue range"
#define kParamIsolateHint "isolate hue range"
#define kParamIsolateOptionOff "Off"
#define kParamIsolateOptionOffHint "off"
#define kParamIsolateOptionHue1 "Isolate Hue1 Range"
#define kParamIsolateOptionHue1Hint "isolate hue1 range"
#define kParamIsolateOptionHue2 "Isolate Hue2 Range"
#define kParamIsolateOptionHue2Hint "isolate hue2 range"
#define kParamIsolateOptionHue3 "Isolate Hue3 Range"
#define kParamIsolateOptionHue3Hint "isolate hue1 range"

enum IsolateEnum {
eIsolateOff,
eIsolateHue1,
eIsolateHue2,
eIsolateHue3,
};

#define kParamDisplayAlpha "displayAlpha"
#define kParamDisplayAlphaLabel "display alpha"
#define kParamDisplayAlphaHint "display process specific alpha channel "
#define kParamDisplayAlphaOptionOff "Off"
#define kParamDisplayAlphaOptionOffHint "off"
#define kParamDisplayAlphaOptionHue1 "Hue One Luma and Saturation limiter"
#define kParamDisplayAlphaOptionHue1Hint "hue one luma and saturation limiter"
#define kParamDisplayAlphaOptionHue2 "Hue Two Luma and Saturation limiter"
#define kParamDisplayAlphaOptionHue2Hint "hue two luma and saturation limiter"
#define kParamDisplayAlphaOptionHue3 "Hue Three Luma and Saturation limiter"
#define kParamDisplayAlphaOptionHue3Hint "hue three luma and saturation limiter"
#define kParamDisplayAlphaOptionSatSoft "Saturation Soft-Clip Luma limiter"
#define kParamDisplayAlphaOptionSatSoftHint "saturation soft-clip luma limiter"
#define kParamDisplayAlphaOptionHue "Hue Channel"
#define kParamDisplayAlphaOptionHueHint "hue channel"
#define kParamDisplayAlphaOptionSat "Saturation Channel"
#define kParamDisplayAlphaOptionSatHint "saturation channel"
#define kParamDisplayAlphaOptionLuma "Luma Channel"
#define kParamDisplayAlphaOptionLumaHint "luma channel based on co-efficients"

enum DisplayAlphaEnum {
eDisplayAlphaOff,
eDisplayAlphaHue1,
eDisplayAlphaHue2,
eDisplayAlphaHue3,
eDisplayAlphaSatSoft,
eDisplayAlphaHue,
eDisplayAlphaSat,
eDisplayAlphaLuma,
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

float RGBtoHUE(float R, float G, float B) {
float mn = fmin(R, fmin(G, B));    
float Mx = fmax(R, fmax(G, B));    
float del_Mx = Mx - mn;
float del_R = ((Mx - R) / 6.0f + del_Mx / 2.0f) / del_Mx;
float del_G = ((Mx - G) / 6.0f + del_Mx / 2.0f) / del_Mx;
float del_B = ((Mx - B) / 6.0f + del_Mx / 2.0f) / del_Mx;
float h = del_Mx == 0.0f ? 0.0f : R == Mx ? del_B - del_G : G == Mx ? 1.0f / 3.0f + 
del_R - del_B : 2.0f / 3.0f + del_G - del_R;
float Hh = h < 0.0f ? h + 1.0f : h > 1.0f ? h - 1.0f : h;
return Hh;
}

float RGBtoHUE2( float r, float g, float b) {
float hue;
float min = fminf(fminf(r, g), b);
float max = fmaxf(fmaxf(r, g), b);
float delta = max - min;
if (max == 0.0f)
return 0.0f;
if (delta == 0.0f) {
hue = 0.0f;
} else if (r == max) {
hue = (g - b) / delta;
} else if (g == max) {
hue = 2 + (b - r) / delta;
} else {
hue = 4 + (r - g) / delta;
}
hue *= 1.0f / 6.0f;
hue = hue < 0.0f ? hue + 1.0f : hue > 1.0f ? hue - 1.0f : hue;
return hue;
}

void HSV_to_RGB(float H, float S, float V, float *r, float *g, float *b) {
if (S == 0.0f) {
*r = *g = *b = V;
} else {
H *= 6.0f;
int i = floor(H);
float f = H - i;
i = (i >= 0) ? (i % 6) : (i % 6) + 6;
float p = V * (1.0f - S);
float q = V * (1.0f - S * f);
float t = V * (1.0f - S * (1.0f - f));
*r = i == 0 ? V : i == 1 ? q : i == 2 ? p : i == 3 ? p : i == 4 ? t : V;
*g = i == 0 ? t : i == 1 ? V : i == 2 ? V : i == 3 ? q : i == 4 ? p : p;
*b = i == 0 ? p : i == 1 ? p : i == 2 ? t : i == 3 ? V : i == 4 ? V : q;
}
}

class HueConverge : public OFX::ImageProcessor
{
public:
explicit HueConverge(OFX::ImageEffect& p_Instance);

virtual void processImagesCUDA();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(int *p_Switch, float *p_Log, float *p_Sat, float *p_Hue1, float *p_Hue2, 
float *p_Hue3, int p_Display, float* p_Blur, int p_Math, int p_HueMedian, int p_Isolate);

private:
OFX::Image* _srcImg;
int _switch[5];
float _log[4];
float _sat[4];
float _hue1[7];
float _hue2[7];
float _hue3[7];
int _display;
float _blur[4];
int _math;
int _hueMedian;
int _isolate;
};

HueConverge::HueConverge(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, int* p_Switch, float* p_Log, float* p_Sat, 
float *p_Hue1, float *p_Hue2, float *p_Hue3, int p_Display, float* p_Blur, int p_Math, int p_HueMedian, int p_Isolate);

void HueConverge::processImagesCUDA()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

for(int c = 0; c < 4; c++){
_blur[c] = _blur[c] * kResolutionScale; 
}

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunCudaKernel(input, output, width, height, _switch, _log, _sat, _hue1, _hue2, _hue3, _display, _blur, _math, _hueMedian, _isolate);
}

void HueConverge::multiThreadProcessImages(OfxRectI p_ProcWindow)
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
}
else {
for (int c = 0; c < 4; ++c) {
dstPix[c] = 0.0f;
}}
dstPix += 4;
}}}

void HueConverge::setSrcImg(OFX::Image* p_SrcImg)
{
_srcImg = p_SrcImg;
}

void HueConverge::setScales(int *p_Switch, float *p_Log, float *p_Sat, float *p_Hue1, 
float *p_Hue2, float *p_Hue3, int p_Display, float* p_Blur, int p_Math, int p_HueMedian, int p_Isolate)
{
_switch[0] = p_Switch[0];
_switch[1] = p_Switch[1];
_switch[2] = p_Switch[2];
_switch[3] = p_Switch[3];
_switch[4] = p_Switch[4];

_log[0] = p_Log[0];
_log[1] = p_Log[1];
_log[2] = p_Log[2];
_log[3] = p_Log[3];

_sat[0] = p_Sat[0];
_sat[1] = p_Sat[1];
_sat[2] = p_Sat[2];
_sat[3] = p_Sat[3];

_hue1[0] = p_Hue1[0];
_hue1[1] = p_Hue1[1];
_hue1[2] = p_Hue1[2];
_hue1[3] = p_Hue1[3];
_hue1[4] = p_Hue1[4];
_hue1[5] = p_Hue1[5];
_hue1[6] = p_Hue1[6];

_hue2[0] = p_Hue2[0];
_hue2[1] = p_Hue2[1];
_hue2[2] = p_Hue2[2];
_hue2[3] = p_Hue2[3];
_hue2[4] = p_Hue2[4];
_hue2[5] = p_Hue2[5];
_hue2[6] = p_Hue2[6];

_hue3[0] = p_Hue3[0];
_hue3[1] = p_Hue3[1];
_hue3[2] = p_Hue3[2];
_hue3[3] = p_Hue3[3];
_hue3[4] = p_Hue3[4];
_hue3[5] = p_Hue3[5];
_hue3[6] = p_Hue3[6];

_blur[0] = p_Blur[0];
_blur[1] = p_Blur[1];
_blur[2] = p_Blur[2];
_blur[3] = p_Blur[3];

_math = p_Math;
_display = p_Display;
_hueMedian = p_HueMedian;
_isolate = p_Isolate;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class HueConvergePlugin : public OFX::ImageEffect
{
public:
explicit HueConvergePlugin(OfxImageEffectHandle p_Handle);

/* Override the render */
virtual void render(const OFX::RenderArguments& p_Args);

/* Override is identity */
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

/* Override changedParam */
virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

/* Set up and run a processor */
void setupAndProcess(HueConverge &p_HueConverge, const OFX::RenderArguments& p_Args);

private:
// Does not own the following pointers
OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;

OFX::DoubleParam* m_Log1;
OFX::DoubleParam* m_Log2;
OFX::DoubleParam* m_Log3;
OFX::DoubleParam* m_Log4;
OFX::BooleanParam* m_SwitchLog;
OFX::DoubleParam* m_SatAdd;
OFX::DoubleParam* m_SatMinus;

OFX::ChoiceParam* m_HueMedian;

OFX::BooleanParam* m_SwitchHue1;
OFX::RGBParam *m_HueA;
OFX::DoubleParam* m_Hue1;
OFX::DoubleParam* m_Range1;
OFX::DoubleParam* m_Shift1;
OFX::DoubleParam* m_Converge1;
OFX::DoubleParam* m_ScaleCH1;
OFX::DoubleParam* m_LumaAlpha1;
OFX::DoubleParam* m_SatAlpha1;
OFX::DoubleParam* m_Blur1;

OFX::BooleanParam* m_SwitchHue2;
OFX::RGBParam *m_HueB;
OFX::DoubleParam* m_Hue2;
OFX::DoubleParam* m_Range2;
OFX::DoubleParam* m_Shift2;
OFX::DoubleParam* m_Converge2;
OFX::DoubleParam* m_ScaleCH2;
OFX::DoubleParam* m_LumaAlpha2;
OFX::DoubleParam* m_SatAlpha2;
OFX::DoubleParam* m_Blur2;

OFX::BooleanParam* m_SwitchHue3;
OFX::RGBParam *m_HueC;
OFX::DoubleParam* m_Hue3;
OFX::DoubleParam* m_Range3;
OFX::DoubleParam* m_Shift3;
OFX::DoubleParam* m_Converge3;
OFX::DoubleParam* m_ScaleCH3;
OFX::DoubleParam* m_LumaAlpha3;
OFX::DoubleParam* m_SatAlpha3;
OFX::DoubleParam* m_Blur3;

OFX::DoubleParam* m_SatSoft;
OFX::DoubleParam* m_SatSoftLumaAlpha;
OFX::DoubleParam* m_Blur4;

OFX::ChoiceParam* m_LuminanceMath;
OFX::ChoiceParam* m_Isolate;
OFX::ChoiceParam* m_DisplayAlpha;
OFX::BooleanParam* m_HueChart;

OFX::StringParam* m_Path;
OFX::StringParam* m_Name;
OFX::PushButtonParam* m_Info;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
};

HueConvergePlugin::HueConvergePlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

m_Log1 = fetchDoubleParam("Peak");
m_Log2 = fetchDoubleParam("Contrast");
m_Log3 = fetchDoubleParam("Pivot");
m_Log4 = fetchDoubleParam("Offset");
m_SwitchLog = fetchBooleanParam("LogSwitch");
m_SatAdd = fetchDoubleParam("SatAdd");
m_SatMinus = fetchDoubleParam("SatMinus");

m_HueMedian = fetchChoiceParam(kParamHueMedian);

m_SwitchHue1 = fetchBooleanParam("HueSwitch1");
m_HueA = fetchRGBParam("HueA");
m_Hue1 = fetchDoubleParam("Hue1");
m_Range1 = fetchDoubleParam("Range1");
m_Shift1 = fetchDoubleParam("Shift1");
m_Converge1 = fetchDoubleParam("Converge1");
m_ScaleCH1 = fetchDoubleParam("ScaleCH1");
m_LumaAlpha1 = fetchDoubleParam("LumaAlpha1");
m_SatAlpha1 = fetchDoubleParam("SatAlpha1");
m_Blur1 = fetchDoubleParam("Blur1");

m_SwitchHue2 = fetchBooleanParam("HueSwitch2");
m_HueB = fetchRGBParam("HueB");
m_Hue2 = fetchDoubleParam("Hue2");
m_Range2 = fetchDoubleParam("Range2");
m_Shift2 = fetchDoubleParam("Shift2");
m_Converge2 = fetchDoubleParam("Converge2");
m_ScaleCH2 = fetchDoubleParam("ScaleCH2");
m_LumaAlpha2 = fetchDoubleParam("LumaAlpha2");
m_SatAlpha2 = fetchDoubleParam("SatAlpha2");
m_Blur2 = fetchDoubleParam("Blur2");

m_SwitchHue3 = fetchBooleanParam("HueSwitch3");
m_HueC = fetchRGBParam("HueC");
m_Hue3 = fetchDoubleParam("Hue3");
m_Range3 = fetchDoubleParam("Range3");
m_Shift3 = fetchDoubleParam("Shift3");
m_Converge3 = fetchDoubleParam("Converge3");
m_ScaleCH3 = fetchDoubleParam("ScaleCH3");
m_LumaAlpha3 = fetchDoubleParam("LumaAlpha3");
m_SatAlpha3 = fetchDoubleParam("SatAlpha3");
m_Blur3 = fetchDoubleParam("Blur3");

m_SatSoft = fetchDoubleParam("SatSoft");
m_SatSoftLumaAlpha = fetchDoubleParam("SatSoftLumaAlpha");
m_Blur4 = fetchDoubleParam("Blur4");

m_LuminanceMath = fetchChoiceParam(kParamLuminanceMath);
m_Isolate = fetchChoiceParam(kParamIsolate);
m_DisplayAlpha = fetchChoiceParam(kParamDisplayAlpha);
m_HueChart = fetchBooleanParam("HueChart");

m_Path = fetchStringParam("Path");
m_Name = fetchStringParam("Name");
m_Info = fetchPushButtonParam("Info");
m_Button1 = fetchPushButtonParam("Button1");
m_Button2 = fetchPushButtonParam("Button2");
}

void HueConvergePlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
{
HueConverge HueConverge(*this);
setupAndProcess(HueConverge, p_Args);
}
else
{
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool HueConvergePlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{ 
bool aSwitch = m_SwitchLog->getValueAtTime(p_Args.time);
bool bSwitch = m_SwitchHue1->getValueAtTime(p_Args.time);
bool cSwitch = m_SwitchHue2->getValueAtTime(p_Args.time);
bool dSwitch = m_SwitchHue3->getValueAtTime(p_Args.time);

if ( !aSwitch && !bSwitch && !cSwitch && !dSwitch )
{
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void HueConvergePlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{

if(p_ParamName == "Info")
{
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

if(p_ParamName == "HueA" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues hueA;
m_HueA->getValueAtTime(p_Args.time, hueA.r, hueA.g, hueA.b);
float rHueA = hueA.r;
float gHueA = hueA.g;
float bHueA = hueA.b;
float hue1 = RGBtoHUE2(rHueA, gHueA, bHueA) * 360.0f;
m_Hue1->setValue(hue1);
}

if(p_ParamName == "Hue1" && p_Args.reason == OFX::eChangeUserEdit)
{
float hueA = m_Hue1->getValueAtTime(p_Args.time) / 360.0f;
float r, g, b;
HSV_to_RGB(hueA, 0.75f, 0.75f, &r, &g, &b);
m_HueA->setValue(r, g, b);
}

if(p_ParamName == "HueB" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues hueB;
m_HueB->getValueAtTime(p_Args.time, hueB.r, hueB.g, hueB.b);
float rHueB = hueB.r;
float gHueB = hueB.g;
float bHueB = hueB.b;
float hue2 = RGBtoHUE2(rHueB, gHueB, bHueB) * 360.0f;
m_Hue2->setValue(hue2);
}

if(p_ParamName == "Hue2" && p_Args.reason == OFX::eChangeUserEdit)
{
float hueB = m_Hue2->getValueAtTime(p_Args.time) / 360.0f;
float r, g, b;
HSV_to_RGB(hueB, 0.75f, 0.75f, &r, &g, &b);
m_HueB->setValue(r, g, b);
}

if(p_ParamName == "HueC" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues hueC;
m_HueC->getValueAtTime(p_Args.time, hueC.r, hueC.g, hueC.b);
float rHueC = hueC.r;
float gHueC = hueC.g;
float bHueC = hueC.b;
float hue3 = RGBtoHUE2(rHueC, gHueC, bHueC) * 360.0f;
m_Hue3->setValue(hue3);
}

if(p_ParamName == "Hue3" && p_Args.reason == OFX::eChangeUserEdit)
{
float hueC = m_Hue3->getValueAtTime(p_Args.time) / 360.0f;
float r, g, b;
HSV_to_RGB(hueC, 0.75f, 0.75f, &r, &g, &b);
m_HueC->setValue(r, g, b);
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

fprintf (pFile, "// HueConvergePlugin DCTL export\n" \
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
" name HueConverge\n" \
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
}	

void HueConvergePlugin::setupAndProcess(HueConverge& p_HueConverge, const OFX::RenderArguments& p_Args)
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

bool logSwitch = m_SwitchLog->getValueAtTime(p_Args.time);
bool hue1Switch = m_SwitchHue1->getValueAtTime(p_Args.time);
bool hue2Switch = m_SwitchHue2->getValueAtTime(p_Args.time);
bool hue3Switch = m_SwitchHue3->getValueAtTime(p_Args.time);
bool chartSwitch = m_HueChart->getValueAtTime(p_Args.time);

int _switch[5];
_switch[0] = (logSwitch) ? 1 : 0;
_switch[1] = (hue1Switch) ? 1 : 0;
_switch[2] = (hue2Switch) ? 1 : 0;
_switch[3] = (hue3Switch) ? 1 : 0;
_switch[4] = (chartSwitch) ? 1 : 0;

float _log[4];
_log[0] = m_Log1->getValueAtTime(p_Args.time);
_log[1] = m_Log2->getValueAtTime(p_Args.time);
_log[2] = m_Log3->getValueAtTime(p_Args.time);
_log[3] = m_Log4->getValueAtTime(p_Args.time);

float _sat[4];
_sat[0] = m_SatAdd->getValueAtTime(p_Args.time);
_sat[1] = m_SatMinus->getValueAtTime(p_Args.time);
_sat[2] = m_SatSoft->getValueAtTime(p_Args.time);
_sat[3] = m_SatSoftLumaAlpha->getValueAtTime(p_Args.time);
_sat[2] = 1.0f - _sat[2];

int _hueMedian;
m_HueMedian->getValueAtTime(p_Args.time, _hueMedian);

float _hue1[7];
_hue1[0] = m_Hue1->getValueAtTime(p_Args.time);
_hue1[1] = m_Range1->getValueAtTime(p_Args.time);
_hue1[2] = m_Shift1->getValueAtTime(p_Args.time);
_hue1[3] = m_Converge1->getValueAtTime(p_Args.time) + 1.0f;
_hue1[3] = _hue1[3] < 1.0f ? -1.0f/(_hue1[3] - 2.0f) : _hue1[3];
_hue1[4] = m_ScaleCH1->getValueAtTime(p_Args.time);
_hue1[5] = m_LumaAlpha1->getValueAtTime(p_Args.time);
_hue1[6] = m_SatAlpha1->getValueAtTime(p_Args.time);

float _hue2[7];
_hue2[0] = m_Hue2->getValueAtTime(p_Args.time);
_hue2[1] = m_Range2->getValueAtTime(p_Args.time);
_hue2[2] = m_Shift2->getValueAtTime(p_Args.time);
_hue2[3] = m_Converge2->getValueAtTime(p_Args.time) + 1.0f;
_hue2[3] = _hue2[3] < 1.0f ? -1.0f/(_hue2[3] - 2.0f) : _hue2[3];
_hue2[4] = m_ScaleCH2->getValueAtTime(p_Args.time);
_hue2[5] = m_LumaAlpha2->getValueAtTime(p_Args.time);
_hue2[6] = m_SatAlpha2->getValueAtTime(p_Args.time);

float _hue3[7];
_hue3[0] = m_Hue3->getValueAtTime(p_Args.time);
_hue3[1] = m_Range3->getValueAtTime(p_Args.time);
_hue3[2] = m_Shift3->getValueAtTime(p_Args.time);
_hue3[3] = m_Converge3->getValueAtTime(p_Args.time) + 1.0f;
_hue3[3] = _hue3[3] < 1.0f ? -1.0f/(_hue3[3] - 2.0f) : _hue3[3];
_hue3[4] = m_ScaleCH3->getValueAtTime(p_Args.time);
_hue3[5] = m_LumaAlpha3->getValueAtTime(p_Args.time);
_hue3[6] = m_SatAlpha3->getValueAtTime(p_Args.time);

float _blur[4];
_blur[0] = m_Blur1->getValueAtTime(p_Args.time) * 20;
_blur[1] = m_Blur2->getValueAtTime(p_Args.time) * 20;
_blur[2] = m_Blur3->getValueAtTime(p_Args.time) * 20;
_blur[3] = m_Blur4->getValueAtTime(p_Args.time) * 20;

int _math;
m_LuminanceMath->getValueAtTime(p_Args.time, _math);
int _isolate;
m_Isolate->getValueAtTime(p_Args.time, _isolate);
int _display;
m_DisplayAlpha->getValueAtTime(p_Args.time, _display);

p_HueConverge.setDstImg(dst.get());
p_HueConverge.setSrcImg(src.get());

// Setup GPU Render arguments
p_HueConverge.setGPURenderArgs(p_Args);

p_HueConverge.setRenderWindow(p_Args.renderWindow);

p_HueConverge.setScales(_switch, _log, _sat, _hue1, _hue2, _hue3, _display, _blur, _math, _hueMedian, _isolate);

p_HueConverge.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

HueConvergePluginFactory::HueConvergePluginFactory()
: OFX::PluginFactoryHelper<HueConvergePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void HueConvergePluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

p_Desc.setSupportsCudaRender(true);
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, 
const std::string& p_Label, const std::string& p_Hint, GroupParamDescriptor* p_Parent) {
DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
param->setLabels(p_Label, p_Label, p_Label);
param->setScriptName(p_Name);
param->setHint(p_Hint);
param->setDefault(1.0);
param->setRange(0.0, 10.0);
param->setIncrement(0.1);
param->setDisplayRange(0.0, 10.0);
param->setDoubleType(eDoubleTypeScale);
if (p_Parent)
param->setParent(*p_Parent);
return param;
}

void HueConvergePluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

GroupParamDescriptor* log1 = p_Desc.defineGroupParam("Log");
log1->setOpen(false);
log1->setHint("log adjustments");
if (page)
page->addChild(*log1);

DoubleParamDescriptor* param = defineScaleParam(p_Desc, "Peak", "peak", "Curve Peak", log1);
param->setDefault(1.0);
param->setRange(0.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Contrast", "contrast", "contrast", log1);
param->setDefault(1.0);
param->setRange(0.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Pivot", "pivot", "pivot Point", log1);
param->setDefault(0.435);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Offset", "offset", "offset", log1);
param->setDefault(0.0);
param->setRange(-1.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(-1.0, 1.0);
page->addChild(*param);

BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("LogSwitch");
boolParam->setDefault(false);
boolParam->setHint("apply logistic function");
boolParam->setLabel("apply function");
boolParam->setParent(*log1);
page->addChild(*boolParam);

param = defineScaleParam(p_Desc, "SatAdd", "additive saturation", "saturation via additive process", log1);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "SatMinus", "subtractive saturation", "saturation via subtractive process", log1);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

ChoiceParamDescriptor* choiceParam = p_Desc.defineChoiceParam(kParamHueMedian);
choiceParam->setLabel(kParamHueMedianLabel);
choiceParam->setHint(kParamHueMedianHint);
assert(choiceParam->getNOptions() == eHueMedianOff);
choiceParam->appendOption(kParamHueMedianOptionOff, kParamHueMedianOptionOffHint);
assert(choiceParam->getNOptions() == eHueMedian3x3);
choiceParam->appendOption(kParamHueMedianOption3x3, kParamHueMedianOption3x3Hint);
assert(choiceParam->getNOptions() == eHueMedian5x5);
choiceParam->appendOption(kParamHueMedianOption5x5, kParamHueMedianOption5x5Hint);
choiceParam->setDefault(eHueMedianOff);
choiceParam->setAnimates(false);
page->addChild(*choiceParam);

GroupParamDescriptor* hue1 = p_Desc.defineGroupParam("Hue One");
hue1->setOpen(true);
hue1->setHint("hue based modifiers");
if(page)
page->addChild(*hue1);

boolParam = p_Desc.defineBooleanParam("HueSwitch1");
boolParam->setDefault(true);
boolParam->setHint("enable controls");
boolParam->setLabel("enable");
boolParam->setParent(*hue1);
page->addChild(*boolParam);

RGBParamDescriptor *RGBparam = p_Desc.defineRGBParam("HueA");
RGBparam->setLabel("hue");
RGBparam->setHint("hue");
RGBparam->setDefault(158.0/255.0, 79.0/255.0, 0.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true); // can animate
RGBparam->setParent(*hue1);
page->addChild(*RGBparam);

param = defineScaleParam(p_Desc, "Hue1", "hue", "hue", hue1);
param->setDefault(30.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Range1", "hue range", "hue range in degrees", hue1);
param->setDefault(60.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Shift1", "rotate hue", "shift hue range in degrees", hue1);
param->setDefault(0.0);
param->setRange(-360.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(-180.0, 180.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Converge1", "converge", "converge hue", hue1);
param->setDefault(0.0);
param->setRange(-5.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(-5.0, 5.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleCH1", "color scale", "scale color at hue range", hue1);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "LumaAlpha1", "luma limiter", "use luma alpha channel to limit", hue1);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "SatAlpha1", "sat limiter", "use sat alpha channel to limit", hue1);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Blur1", "blur alpha", "blur alpha channel", hue1);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

GroupParamDescriptor* hue2 = p_Desc.defineGroupParam("Hue Two");
hue2->setOpen(false);
hue2->setHint("hue based modifiers");
if (page)
page->addChild(*hue2);

boolParam = p_Desc.defineBooleanParam("HueSwitch2");
boolParam->setDefault(false);
boolParam->setHint("enable controls");
boolParam->setLabel("enable");
boolParam->setParent(*hue2);
page->addChild(*boolParam);

RGBparam = p_Desc.defineRGBParam("HueB");
RGBparam->setLabel("hue");
RGBparam->setHint("hue");
RGBparam->setDefault(0.0, 79.0/255.0, 158.0/255.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
RGBparam->setParent(*hue2);
page->addChild(*RGBparam);

param = defineScaleParam(p_Desc, "Hue2", "hue", "hue", hue2);
param->setDefault(210.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Range2", "hue range", "hue range in degrees", hue2);
param->setDefault(60.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Shift2", "rotate hue", "shift hue range in degrees", hue2);
param->setDefault(0.0);
param->setRange(-360.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(-180.0, 180.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Converge2", "converge", "converge hue", hue2);
param->setDefault(0.0);
param->setRange(-5.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(-5.0, 5.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleCH2", "color scale", "scale color at hue range", hue2);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "LumaAlpha2", "luma limiter", "use luma alpha channel to limit", hue2);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "SatAlpha2", "sat limiter", "use sat alpha channel to limit", hue2);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Blur2", "blur alpha", "blur alpha channel", hue2);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

GroupParamDescriptor* hue3 = p_Desc.defineGroupParam("Hue Three");
hue3->setOpen(false);
hue3->setHint("hue based modifiers");
if (page)
page->addChild(*hue3);

boolParam = p_Desc.defineBooleanParam("HueSwitch3");
boolParam->setDefault(false);
boolParam->setHint("enable controls");
boolParam->setLabel("enable");
boolParam->setParent(*hue3);
page->addChild(*boolParam);

RGBparam = p_Desc.defineRGBParam("HueC");
RGBparam->setLabel("hue");
RGBparam->setHint("hue");
RGBparam->setDefault(1.0, 1.0, 1.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
RGBparam->setParent(*hue3);
page->addChild(*RGBparam);

param = defineScaleParam(p_Desc, "Hue3", "hue", "hue", hue3);
param->setDefault(0.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 360.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Range3", "hue range", "hue range in degrees", hue3);
param->setDefault(60.0);
param->setRange(0.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(0.0, 180.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Shift3", "rotate hue", "shift hue range in degrees", hue3);
param->setDefault(0.0);
param->setRange(-360.0, 360.0);
param->setIncrement(0.01);
param->setDisplayRange(-180.0, 180.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Converge3", "converge", "converge hue", hue3);
param->setDefault(0.0);
param->setRange(-5.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(-5.0, 5.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleCH3", "color scale", "scale color at hue range", hue3);
param->setDefault(1.0);
param->setRange(0.0, 5.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "LumaAlpha3", "luma limiter", "use luma alpha channel to limit", hue3);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "SatAlpha3", "sat limiter", "use sat alpha channel to limit", hue3);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Blur3", "blur alpha", "blur alpha channel", hue3);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "SatSoft", "saturation soft-clip", "saturation soft-clip", 0);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "SatSoftLumaAlpha", "luma limiter", "use luma alpha channel to limit saturation soft-clip", 0);
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "Blur4", "blur alpha", "blur alpha channel", 0);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

choiceParam = p_Desc.defineChoiceParam(kParamLuminanceMath);
choiceParam->setLabel(kParamLuminanceMathLabel);
choiceParam->setHint(kParamLuminanceMathHint);
assert(choiceParam->getNOptions() == eLuminanceMathRec709);
choiceParam->appendOption(kParamLuminanceMathOptionRec709, kParamLuminanceMathOptionRec709Hint);
assert(choiceParam->getNOptions() == eLuminanceMathRec2020);
choiceParam->appendOption(kParamLuminanceMathOptionRec2020, kParamLuminanceMathOptionRec2020Hint);
assert(choiceParam->getNOptions() == eLuminanceMathDCIP3);
choiceParam->appendOption(kParamLuminanceMathOptionDCIP3, kParamLuminanceMathOptionDCIP3Hint);
assert(choiceParam->getNOptions() == eLuminanceMathACESAP0);
choiceParam->appendOption(kParamLuminanceMathOptionACESAP0, kParamLuminanceMathOptionACESAP0Hint);
assert(choiceParam->getNOptions() == eLuminanceMathACESAP1);
choiceParam->appendOption(kParamLuminanceMathOptionACESAP1, kParamLuminanceMathOptionACESAP1Hint);
assert(choiceParam->getNOptions() == eLuminanceMathAverage);
choiceParam->appendOption(kParamLuminanceMathOptionAverage, kParamLuminanceMathOptionAverageHint);
assert(choiceParam->getNOptions() == eLuminanceMathMaximum);
choiceParam->appendOption(kParamLuminanceMathOptionMaximum, kParamLuminanceMathOptionMaximumHint);
choiceParam->setDefault(eDisplayAlphaOff);
choiceParam->setAnimates(false);
page->addChild(*choiceParam);

choiceParam = p_Desc.defineChoiceParam(kParamIsolate);
choiceParam->setLabel(kParamIsolateLabel);
choiceParam->setHint(kParamIsolateHint);
assert(choiceParam->getNOptions() == eIsolateOff);
choiceParam->appendOption(kParamIsolateOptionOff, kParamIsolateOptionOffHint);
assert(choiceParam->getNOptions() == eIsolateHue1);
choiceParam->appendOption(kParamIsolateOptionHue1, kParamIsolateOptionHue1Hint);
assert(choiceParam->getNOptions() == eIsolateHue2);
choiceParam->appendOption(kParamIsolateOptionHue2, kParamIsolateOptionHue2Hint);
assert(choiceParam->getNOptions() == eIsolateHue3);
choiceParam->appendOption(kParamIsolateOptionHue3, kParamIsolateOptionHue3Hint);
choiceParam->setDefault(eIsolateOff);
choiceParam->setAnimates(false);
page->addChild(*choiceParam);

choiceParam = p_Desc.defineChoiceParam(kParamDisplayAlpha);
choiceParam->setLabel(kParamDisplayAlphaLabel);
choiceParam->setHint(kParamDisplayAlphaHint);
assert(choiceParam->getNOptions() == eDisplayAlphaOff);
choiceParam->appendOption(kParamDisplayAlphaOptionOff, kParamDisplayAlphaOptionOffHint);
assert(choiceParam->getNOptions() == eDisplayAlphaHue1);
choiceParam->appendOption(kParamDisplayAlphaOptionHue1, kParamDisplayAlphaOptionHue1Hint);
assert(choiceParam->getNOptions() == eDisplayAlphaHue2);
choiceParam->appendOption(kParamDisplayAlphaOptionHue2, kParamDisplayAlphaOptionHue2Hint);
assert(choiceParam->getNOptions() == eDisplayAlphaHue3);
choiceParam->appendOption(kParamDisplayAlphaOptionHue3, kParamDisplayAlphaOptionHue3Hint);
assert(choiceParam->getNOptions() == eDisplayAlphaSatSoft);
choiceParam->appendOption(kParamDisplayAlphaOptionSatSoft, kParamDisplayAlphaOptionSatSoftHint);
assert(choiceParam->getNOptions() == eDisplayAlphaHue);
choiceParam->appendOption(kParamDisplayAlphaOptionHue, kParamDisplayAlphaOptionHueHint);
assert(choiceParam->getNOptions() == eDisplayAlphaSat);
choiceParam->appendOption(kParamDisplayAlphaOptionSat, kParamDisplayAlphaOptionSatHint);
assert(choiceParam->getNOptions() == eDisplayAlphaLuma);
choiceParam->appendOption(kParamDisplayAlphaOptionLuma, kParamDisplayAlphaOptionLumaHint);
choiceParam->setDefault(eDisplayAlphaOff);
choiceParam->setAnimates(false);
page->addChild(*choiceParam);

boolParam = p_Desc.defineBooleanParam("HueChart");
boolParam->setDefault(false);
boolParam->setHint("display hue chart");
boolParam->setLabel("display hue chart");
page->addChild(*boolParam);

PushButtonParamDescriptor* pushparam = p_Desc.definePushButtonParam("Info");
pushparam->setLabel("Info");
page->addChild(*pushparam);

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
stringparam->setDefault("HueConverge");
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
}

ImageEffect* HueConvergePluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum) {
return new HueConvergePlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static HueConvergePluginFactory HueConvergePlugin;
p_FactoryArray.push_back(&HueConvergePlugin);
}