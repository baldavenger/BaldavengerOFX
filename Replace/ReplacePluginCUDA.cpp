#include "ReplacePlugin.h"

#include <cstring>
using std::string;
#include <string> 
#include <fstream>

#include <cmath>
#include <cfloat>
#include <algorithm>

#include "ofxsProcessing.h"
#include "ofxsCoords.h"
#include "ofxsLut.h"
#include "ofxsMacros.h"
#include "ofxsRectangleInteract.h"
#include "ofxsThreadSuite.h"
#include "ofxsMultiThread.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#ifdef __APPLE__
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#else
#define kPluginScript "/opt/resolve/LUT"
#endif

using namespace OFX;

#define kPluginName "Replace"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"Adjust hue, saturation and luma, or perform colour replacement. \n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Colour replacement: Set the srcColour and dstColour parameters. The range of the replacement is determined by the  \n" \
"three groups of parameters: Hue, Saturation and Luma \n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Colour adjust: Use the Rotation of the Hue parameter and the Adjustment of the Saturation and Luma. \n" \
"The ranges and falloff parameters allow for more complex adjustments \n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Hue keyer: Set the outputAlpha parameter to All, and select Display Alpha. \n" \
"First, set the Range parameter of the Hue parameter set and then work down the other Ranges parameters, \n" \
"tuning with the range Falloff and Adjustment parameters. \n" \

#define kPluginIdentifier "BaldavengerOFX.Replace"

#define kPluginVersionMajor 2
#define kPluginVersionMinor 0

#define kSupportsTiles 1
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 0
#define kSupportsMultipleClipPARs false
#define kSupportsMultipleClipDepths false
#define kRenderThreadSafety eRenderFullySafe

#define kResolutionScale	(float)width / 1920.0f

#define kGroupColourReplacement "colourReplacement"
#define kGroupColourReplacementLabel "Colour Replacement"
#define kGroupColourReplacementHint "replace a given colour by another colour by setting srcColour and dstColour. Set Src Colour first, then Dst Colour"
#define kParamSrcColour "srcColour"
#define kParamSrcColourLabel "Source Colour"
#define kParamSrcColourHint "source colour for replacement. Changing this parameter sets the hue, saturation and luma ranges for this colour, and sets the fallofs to default values"
#define kParamDstColour "dstColour"
#define kParamDstColourLabel "Destination Colour"
#define kParamDstColourHint "destination colour for replacement. Changing this parameter sets the hue rotation, and saturation and luma adjustments. Should be set after Src Colour"

#define kParamEnableRectangle "enableRectangle"
#define kParamEnableRectangleLabel "Src Analysis Rectangle"
#define kParamEnableRectangleHint "enable the rectangle interact for analysis of Src and Dst colours and ranges"

#define kParamSetSrcFromRectangle "setSrcFromRectangle"
#define kParamSetSrcFromRectangleLabel "Set Source from Rectangle"
#define kParamSetSrcFromRectangleHint "set the Src colour and ranges and the adjustments from the colours of the source image within the selection rectangle and the Dst Colour"

#define kGroupHue "hue"
#define kGroupHueLabel "Hue"
#define kGroupHueHint "hue modification settings."
#define kParamHueRange "hueRange"
#define kParamHueRangeLabel "Hue Range"
#define kParamHueRangeHint "range of colour hues that are modified (in degrees). Red is 0, green is 120, blue is 240. The affected hue range is the smallest interval. For example, if the range is (12, 348), then the selected range is red plus or minus 12 degrees. Exception: if the range width is exactly 360, then all hues are modified"
#define kParamHueRotation "hueRotation"
#define kParamHueRotationLabel "Hue Rotation"
#define kParamHueRotationHint "rotation of colour hues (in degrees) within the range"
#define kParamHueRotationGain "hueRotationGain"
#define kParamHueRotationGainLabel "Hue Rotation Gain"
#define kParamHueRotationGainHint "factor to be applied to the rotation of colour hues (in degrees) within the range. A value of 0 will set all values within range to a constant (computed at the center of the range), and a value of 1 will add hueRotation to all values within range"
#define kParamHueRangeRolloff "hueRangeRolloff"
#define kParamHueRangeRolloffLabel "Hue Range Rolloff"
#define kParamHueRangeRolloffHint "interval (in degrees) around Hue Range, where hue rotation decreases progressively to zero"

#define kGroupSaturation "saturation"
#define kGroupSaturationLabel "Saturation"
#define kGroupSaturationHint "saturation modification settings"
#define kParamSaturationRange "saturationRange"
#define kParamSaturationRangeLabel "Saturation Range"
#define kParamSaturationRangeHint "range of colour saturations that are modified"
#define kParamSaturationAdjustment "saturationAdjustment"
#define kParamSaturationAdjustmentLabel "Saturation Adjustment"
#define kParamSaturationAdjustmentHint "adjustment of colour saturations within the range. Saturation is clamped to zero to avoid colour inversions"
#define kParamSaturationAdjustmentGain "saturationAdjustmentGain"
#define kParamSaturationAdjustmentGainLabel "Saturation Adjustment Gain"
#define kParamSaturationAdjustmentGainHint "factor to be applied to the saturation adjustment within the range. A value of 0 will set all values within range to a constant (computed at the center of the range), and a value of 1 will add saturationAdjustment to all values within range"
#define kParamSaturationRangeRolloff "saturationRangeRolloff"
#define kParamSaturationRangeRolloffLabel "Saturation Range Rolloff"
#define kParamSaturationRangeRolloffHint "interval (in degrees) around Saturation Range, where saturation rotation decreases progressively to zero"

#define kGroupBrightness "luma"
#define kGroupBrightnessLabel "Luma"
#define kGroupBrightnessHint "luma modification settings"
#define kParamBrightnessRange "LumaRange"
#define kParamBrightnessRangeLabel "Luma Range"
#define kParamBrightnessRangeHint "range of luma that is modified"
#define kParamBrightnessAdjustment "lumaAdjustment"
#define kParamBrightnessAdjustmentLabel "Luma Adjustment"
#define kParamBrightnessAdjustmentHint "adjustment of luma within the range"
#define kParamBrightnessAdjustmentGain "lumaAdjustmentGain"
#define kParamBrightnessAdjustmentGainLabel "Luma Adjustment Gain"
#define kParamBrightnessAdjustmentGainHint "factor to be applied to the luma adjustment within the range. A value of 0 will set all values within range to a constant (computed at the center of the range), and a value of 1 will add lumaAdjustment to all values within range"
#define kParamBrightnessRangeRolloff "lumaRangeRolloff"
#define kParamBrightnessRangeRolloffLabel "Luma Range Rolloff"
#define kParamBrightnessRangeRolloffHint "interval (in degrees) around Brightness Range, where luma rotation decreases progressively to zero"

#define kParamOutputAlpha "outputAlpha"
#define kParamOutputAlphaLabel "Output Alpha"
#define kParamOutputAlphaHint "output alpha channel. This can either be the source alpha, one of the coefficients for hue, saturation, brightness, or a combination of those. If it is not source alpha, the image on output are unpremultiplied, even if input is premultiplied"
#define kParamOutputAlphaOptionOff "Off"
#define kParamOutputAlphaOptionOffHint "alpha channel is kept unmodified"
#define kParamOutputAlphaOptionHue "Hue"
#define kParamOutputAlphaOptionHueHint "set alpha to the Hue modification mask"
#define kParamOutputAlphaOptionSaturation "Saturation"
#define kParamOutputAlphaOptionSaturationHint "set alpha to the Saturation modification mask"
#define kParamOutputAlphaOptionBrightness "Luma"
#define kParamOutputAlphaOptionBrightnessHint "alpha is set to the Luma mask"
#define kParamOutputAlphaOptionHueSaturation "Min(Hue, Saturation)"
#define kParamOutputAlphaOptionHueSaturationHint "alpha is set to min(Hue mask,Saturation mask)"
#define kParamOutputAlphaOptionHueBrightness "Min(Hue, Luma)"
#define kParamOutputAlphaOptionHueBrightnessHint "alpha is set to min(Hue mask, Luma mask)"
#define kParamOutputAlphaOptionSaturationBrightness "Min(Saturation, Luma)"
#define kParamOutputAlphaOptionSaturationBrightnessHint "alpha is set to min(Saturation mask, Luma mask)"
#define kParamOutputAlphaOptionAll "Min(Hue, Saturation, Luma)"
#define kParamOutputAlphaOptionAllHint "alpha is set to min(Hue mask, Saturation mask, Luma mask)"

enum OutputAlphaEnum
{
eOutputAlphaOff,
eOutputAlphaHue,
eOutputAlphaSaturation,
eOutputAlphaBrightness,
eOutputAlphaHueSaturation,
eOutputAlphaHueBrightness,
eOutputAlphaSaturationBrightness,
eOutputAlphaAll,
};

#define kParamDisplayAlpha "displayAlpha"
#define kParamDisplayAlphaLabel "Display Alpha"
#define kParamDisplayAlphaHint "displays derived alpha channel"

#define kParamMix "mix"
#define kParamMixLabel "Mix"
#define kParamMixHint "blend between input and ouput image"

#define kGroupClean "clean"
#define kGroupCleanLabel "Clean Alpha"
#define kGroupCleanHint "clean alpha channel"
#define kParamCleanWhite "white"
#define kParamCleanWhiteLabel "Clip White"
#define kParamCleanWhiteHint "clip white"
#define kParamCleanBlack "black"
#define kParamCleanBlackLabel "Clip Black"
#define kParamCleanBlackHint "clip black"
#define kParamCleanBlur "blur"
#define kParamCleanBlurLabel "Blur"
#define kParamCleanBlurHint "blur alpha"
#define kParamCleanGarbage "garbage"
#define kParamCleanGarbageLabel "Garbage"
#define kParamCleanGarbageHint "garbage matte"
#define kParamCleanCore "core"
#define kParamCleanCoreLabel "Core"
#define kParamCleanCoreHint "core matte"
#define kParamCleanErode "erode"
#define kParamCleanErodeLabel "Erode"
#define kParamCleanErodeHint "erode matte"
#define kParamCleanDilate "dilate"
#define kParamCleanDilateLabel "Dilate"
#define kParamCleanDilateHint "dilate matte"

#define kParamDefaultsNormalised "defaultsNormalised"
#define MIN_SATURATION	0.1
#define MIN_VALUE	0.1

#ifndef M_PI
#define M_PI	3.14159265358979323846264338327950288
#endif

#define DEFAULT_RECTANGLE_ROLLOFF	0.5

static bool gHostSupportsDefaultCoordinateSystem = true;
static inline double normalizeAngle(double a) {
int c = (int)std::floor(a / 360.0);
a -= c * 360.0;
assert(a >= 0.0 && a <= 360.0);
return a;
}
static inline double normalizeAngleSigned(double a) {
return normalizeAngle(a + 180.0) - 180.0;
}
static inline bool angleWithinRange(double h, double h0, double h1){
assert(0.0 <= h && h <= 360.0 && 0.0 <= h0 && h0 <= 360.0 && 0.0 <= h1 && h1 <= 360.0);
return ( ( h1 < h0 && (h <= h1 || h0 <= h) ) || (h0 <= h && h <= h1 ) );
}
template<typename T>
T
ipow(T base, int exp) {
T result = T(1);
if (exp >= 0) {
while (exp) {
if (exp & 1) {
result *= base;
}
exp >>= 1;
base *= base;
}} else {
exp = -exp;
while (exp) {
if (exp & 1) {
result /= base;
}
exp >>= 1;
base *= base;
}}
return result;
}
static double ffloor(double val, int decimals) {
int p = ipow(10, decimals);
return std::floor(val * p) / p;
}
static double fround(double val, int decimals) {
int p = ipow(10, decimals);
return std::floor(val * p + 0.5) / p;
}
static double fceil(double val, int decimals) {
int p = ipow(10, decimals);
return std::ceil(val * p) / p;
}
static inline double angleCoeff01(double h, double h0, double h1) {
assert(0.0 <= h && h <= 360.0 && 0.0 <= h0 && h0 <= 360.0 && 0.0 <= h1 && h1 <= 360.0);
if ( h1 == (h0 + 360.0) ) {
return 1.0;
}
if ( !angleWithinRange(h, h0, h1) ) {
return 0.0;
}
if (h1 == h0) {
return 1.0;
}
if (h1 < h0) {
h1 += 360.0;
if (h < h0) {
h += 360.0;
}}
assert(h0 <= h && h <= h1);
return (h - h0) / (h1 - h0);
}
static inline double angleCoeff10(double h, double h0, double h1) {
assert(0.0 <= h && h <= 360.0 && 0.0 <= h0 && h0 <= 360.0 && 0.0 <= h1 && h1 <= 360.0);
if ( !angleWithinRange(h, h0, h1) ) {
return 0.0;
}
if (h1 == h0) {
return 1.0;
}
if (h1 < h0) {
h1 += 360.0;
if (h < h0) {
h += 360.0;
}}
assert(h0 <= h && h <= h1);
return (h1 - h) / (h1 - h0);
}

class Replace : public ImageProcessor
{
public:
explicit Replace(ImageEffect &instance);

virtual void processImagesCUDA();
virtual void processImagesOpenCL();
//virtual void processImagesMetal();
virtual void multiThreadProcessImages(OfxRectI procWindow);

void setSrcImg(Image* p_SrcImg);
void setScales(float* p_Hue, float* p_Sat, float* p_Val, int OutputAlpha, int DisplayAlpha, float mix, float* p_Blur);

private:
Image* _srcImg;
float _hue[8];
float _sat[5];
float _val[5];
int _outputAlpha;
int _displayAlpha;
float _mix;
float _blur[7];
};

Replace::Replace(ImageEffect& instance)
: ImageProcessor(instance)
{
}

extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, 
float* p_Hue, float* p_Sat, float* p_Val, int OutputAlpha, int DisplayAlpha, float mix, float* p_Blur);

void Replace::processImagesCUDA()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

_blur[2] *= kResolutionScale;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunCudaKernel(input, output, width, height, _hue, _sat, _val, _outputAlpha, _displayAlpha, _mix, _blur);
}

extern void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, 
float* p_Hue, float* p_Sat, float* p_Val, int OutputAlpha, int DisplayAlpha, float mix, float* p_Blur);

void Replace::processImagesOpenCL()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

_blur[2] *= kResolutionScale;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunOpenCLKernel(_pOpenCLCmdQ, input, output, width, height, _hue, _sat, _val, _outputAlpha, _displayAlpha, _mix, _blur);
}
/*
#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, 
float* p_Hue, float* p_Sat, float* p_Val, int OutputAlpha, int DisplayAlpha, float mix, float* p_Blur);
#endif

void Replace::processImagesMetal()
{
#ifdef __APPLE__
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

_blur[2] *= kResolutionScale;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunMetalKernel(_pMetalCmdQ, input, output, width, height, _hue, _sat, _val, _outputAlpha, _displayAlpha, _mix, _blur);
#endif
}
*/
void Replace::multiThreadProcessImages(OfxRectI procWindow)
{
for (int y = procWindow.y1; y < procWindow.y2; ++y) {
if (_effect.abort()) break;
float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(procWindow.x1, y));
for (int x = procWindow.x1; x < procWindow.x2; ++x) {
float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
float hcoeff, scoeff, vcoeff;
float h, s, v;
OFX::Color::rgb_to_hsv(srcPix[0], srcPix[1], srcPix[2], &h, &s, &v);
h *= 360.0f / OFXS_HUE_CIRCLE;
const float h0 = _hue[0];
const float h1 = _hue[1];
const float h0mrolloff = _hue[2];
const float h1prolloff = _hue[3];
if ( angleWithinRange(h, h0, h1) ) {
hcoeff = 1.0f;
} else {
float c0 = 0.0f;
float c1 = 0.0f;
if ( angleWithinRange(h, h0mrolloff, h0) ) {
c0 = angleCoeff01(h, h0mrolloff, h0);
}
if ( angleWithinRange(h, h1, h1prolloff) ) {
c1 = angleCoeff10(h, h1, h1prolloff);
}
hcoeff = fmax(c0, c1);
}
assert(0 <= hcoeff && hcoeff <= 1.0f);
const float s0 = _sat[0];
const float s1 = _sat[1];
const float s0mrolloff = s0 - _sat[4];
const float s1prolloff = s1 + _sat[4];
if ( s0 <= s && s <= s1 ) {
scoeff = 1.0f;
} else if ( s0mrolloff <= s && s <= s0 ) {
scoeff = (s - s0mrolloff) / _sat[4];
} else if ( s1 <= s && s <= s1prolloff ) {
scoeff = (s1prolloff - s) / _sat[4];
} else {
scoeff = 0.0f;
}
assert(0 <= scoeff && scoeff <= 1.0f);
const float v0 = _val[0];
const float v1 = _val[1];
const float v0mrolloff = v0 - _val[4];
const float v1prolloff = v1 + _val[4];
if ( v0 <= v && v <= v1 ) {
vcoeff = 1.0f;
} else if ( v0mrolloff <= v && v <= v0 ) {
vcoeff = (v - v0mrolloff) / _val[4];
} else if ( v1 <= v && v <= v1prolloff ) {
vcoeff = (v1prolloff - v) / _val[4];
} else {
vcoeff = 0.0f;
}
assert(0.0f <= vcoeff && vcoeff <= 1.0f);
float coeff = fmin(fmin(hcoeff, scoeff), vcoeff);
assert(0.0f <= coeff && coeff <= 1.0f);
if (coeff <= 0.0f) {
dstPix[0] = srcPix[0];
dstPix[1] = srcPix[1];
dstPix[2] = srcPix[2];
} else {
h += coeff * ( _hue[4] + (_hue[5] - 1.0f) * normalizeAngleSigned(h - _hue[6]) );
s += coeff * ( _sat[2] + (_sat[3] - 1.0f) * (s - (s0 + s1) / 2.0f) );
if (s < 0.0f) {
s = 0.0f;
}
v += coeff * ( _val[2] + (_val[3] - 1.0f) * (v - (v0 + v1) / 2.0f) );
h *= OFXS_HUE_CIRCLE / 360.0f;
OFX::Color::hsv_to_rgb(h, s, v, &dstPix[0], &dstPix[1], &dstPix[2]);
}
if (srcPix) {
float a = _outputAlpha == 0 ? 1.0f : _outputAlpha == 1 ? hcoeff : _outputAlpha == 2 ? scoeff :
_outputAlpha == 3 ? vcoeff : _outputAlpha == 4 ? fmin(hcoeff, scoeff) : _outputAlpha == 5 ? 
fmin(hcoeff, vcoeff) : _outputAlpha == 6 ? fmin(scoeff, vcoeff) : fmin(fmin(hcoeff, scoeff), vcoeff);
dstPix[0] = _displayAlpha == 1 ? a : dstPix[0] * (1.0f - _mix) + srcPix[0] * _mix;
dstPix[1] = _displayAlpha == 1 ? a : dstPix[1] * (1.0f - _mix) + srcPix[1] * _mix;
dstPix[2] = _displayAlpha == 1 ? a : dstPix[2] * (1.0f - _mix) + srcPix[2] * _mix;
dstPix[3] = _outputAlpha != 0 ? a : srcPix[3];
} else {
for (int c = 0; c < 4; ++c) {
dstPix[c] = 0;
}}
dstPix += 4;
}}}

void Replace::setSrcImg(Image* p_SrcImg)
{
_srcImg = p_SrcImg;
}

void Replace::setScales(float* p_Hue, float* p_Sat, float* p_Val, int OutputAlpha, int DisplayAlpha, float mix, float* p_Blur)
{
_hue[0] = p_Hue[0];
_hue[1] = p_Hue[1];
_hue[2] = p_Hue[2];
_hue[3] = p_Hue[3];
_hue[4] = p_Hue[4];
_hue[5] = p_Hue[5];
_hue[6] = p_Hue[6];
_hue[7] = p_Hue[7];
_sat[0] = p_Sat[0];
_sat[1] = p_Sat[1];
_sat[2] = p_Sat[2];
_sat[3] = p_Sat[3];
_sat[4] = p_Sat[4];
_val[0] = p_Val[0];
_val[1] = p_Val[1];
_val[2] = p_Val[2];
_val[3] = p_Val[3];
_val[4] = p_Val[4];
_outputAlpha = OutputAlpha;
_displayAlpha = DisplayAlpha;
_mix = mix;
_blur[0] = p_Blur[0];
_blur[1] = p_Blur[1];
_blur[2] = p_Blur[2];
_blur[3] = p_Blur[3];
_blur[4] = p_Blur[4];
_blur[5] = p_Blur[5];
_blur[6] = p_Blur[6];

float h0 = _hue[0];
float h1 = _hue[1];
if ( h1 == (h0 + 360.0f) ) {
_hue[0] = 0.0f;
_hue[1] = 360.0f;
_hue[7] = 0.0f;
_hue[2] = 0.0f;
_hue[3] = 360.0f;
_hue[6] = 0.0f;
} else {
h0 = normalizeAngle(h0);
h1 = normalizeAngle(h1);
if (h1 < h0) {
std::swap(h0, h1);
}
if ( (h1 - h0) > 180.0f ) {
std::swap(h0, h1);
}
assert (0.0f <= h0 && h0 <= 360.0f && 0.0f <= h1 && h1 <= 360.0f);
_hue[0] = h0;
_hue[1] = h1;
if (_hue[7] < 0.0f) {
_hue[7] = 0.0f;
} else if (_hue[7] >= 180.0f) {
_hue[7] = 180.0f;
}
_hue[2] = normalizeAngle(h0 - _hue[7]);
_hue[3] = normalizeAngle(h1 + _hue[7]);
_hue[6] = normalizeAngle(h0 + normalizeAngleSigned(h1 - h0) / 2.0f);
}
if (_sat[1] < _sat[0]) {
std::swap(_sat[0], _sat[1]);
}
if (_sat[4] < 0.0f) {
_sat[4] = 0.0f;
}
if (_val[1] < _val[0]) {
std::swap(_val[0], _val[1]);
}
if (_val[4] < 0.0f) {
_val[4] = 0.0f;
}}

typedef struct HSVColour
{
HSVColour() : h(0), s(0), v(0) {}
double h, s, v;
} HSVColour;
typedef struct HSVColourF
{
HSVColourF() : h(0), s(0), v(0) {}
float h, s, v;
} HSVColourF;

class HueMeanProcessorBase
: public OFX::ImageProcessor
{
protected:
unsigned long _count;
double _sumsinh, _sumcosh;

public:
HueMeanProcessorBase(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
, _count(0)
, _sumsinh(0)
, _sumcosh(0)
{
}

~HueMeanProcessorBase()
{
}

double getResult() {
if (_count <= 0) {
return 0;
} else {
double meansinh = _sumsinh / _count;
double meancosh = _sumcosh / _count;
return normalizeAngle(std::atan2(meansinh, meancosh) * 180 / M_PI);
}}

protected:
void addResults(double sumsinh, double sumcosh, unsigned long count)
{
_sumsinh += sumsinh;
_sumcosh += sumcosh;
_count += count;
}};

template <class PIX, int nComponents, int maxValue>
class HueMeanProcessor
: public HueMeanProcessorBase
{
public:
HueMeanProcessor(ImageEffect &instance)
: HueMeanProcessorBase(instance)
{
}

~HueMeanProcessor()
{
}

private:

void pixToHSV(const PIX *p, HSVColourF* hsv)
{
if ( (nComponents == 4) || (nComponents == 3) ) {
float r, g, b;
r = p[0] / (float)maxValue;
g = p[1] / (float)maxValue;
b = p[2] / (float)maxValue;
Color::rgb_to_hsv(r, g, b, &hsv->h, &hsv->s, &hsv->v);
hsv->h *= 360 / OFXS_HUE_CIRCLE;
} else {
*hsv = HSVColourF();
}}

void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
{
double sumsinh = 0.0;
double sumcosh = 0.0;
unsigned long count = 0;
assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
_dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
for (int y = procWindow.y1; y < procWindow.y2; ++y) {
if ( _effect.abort() ) {
break;
}
PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);
double sumsinhLine = 0.0;
double sumcoshLine = 0.0;
for (int x = procWindow.x1; x < procWindow.x2; ++x) {
HSVColourF hsv;
pixToHSV(dstPix, &hsv);
if ( (hsv.s > MIN_SATURATION) && (hsv.v > MIN_VALUE) ) {
sumsinhLine += std::sin(hsv.h * M_PI / 180);
sumcoshLine += std::cos(hsv.h * M_PI / 180);
++count;
}
dstPix += nComponents;
}
sumsinh += sumsinhLine;
sumcosh += sumcoshLine;
}
addResults(sumsinh, sumcosh, count);
}};

class HSVRangeProcessorBase
: public ImageProcessor
{
protected:
float _hmean;

private:
float _dhmin;
float _dhmax;
float _smin;
float _smax;
float _vmin;
float _vmax;

public:
HSVRangeProcessorBase(ImageEffect &instance)
: ImageProcessor(instance)
, _hmean(0)
, _dhmin(FLT_MAX)
, _dhmax(-FLT_MAX)
, _smin(FLT_MAX)
, _smax(-FLT_MAX)
, _vmin(FLT_MAX)
, _vmax(-FLT_MAX)
{
}

~HSVRangeProcessorBase()
{
}

void setHueMean(float hmean)
{
_hmean = hmean;
}

void getResults(HSVColour *hsvmin, HSVColour *hsvmax)
{
if (_dhmax - _dhmin > 179.9) {
hsvmin->h = 0.0;
hsvmax->h = 360.0;
} else {
hsvmin->h = normalizeAngle(_hmean + _dhmin);
hsvmax->h = normalizeAngle(_hmean + _dhmax);
}
hsvmin->s = _smin;
hsvmax->s = _smax;
hsvmin->v = _vmin;
hsvmax->v = _vmax;
}

protected:
void addResults(const float dhmin,
const float dhmax,
const float smin,
const float smax,
const float vmin,
const float vmax)
{
if (dhmin < _dhmin) { _dhmin = dhmin; }
if (dhmax > _dhmax) { _dhmax = dhmax; }
if (smin < _smin) { _smin = smin; }
if (smax > _smax) { _smax = smax; }
if (vmin < _vmin) { _vmin = vmin; }
if (vmax > _vmax) { _vmax = vmax; }
}};

template <class PIX, int nComponents, int maxValue>
class HSVRangeProcessor
: public HSVRangeProcessorBase
{
public:
HSVRangeProcessor(ImageEffect &instance)
: HSVRangeProcessorBase(instance)
{
}

~HSVRangeProcessor()
{
}

private:

void pixToHSV(const PIX *p, HSVColourF* hsv)
{
if ( (nComponents == 4) || (nComponents == 3) ) {
float r, g, b;
r = p[0] / (float)maxValue;
g = p[1] / (float)maxValue;
b = p[2] / (float)maxValue;
Color::rgb_to_hsv(r, g, b, &hsv->h, &hsv->s, &hsv->v);
hsv->h *= 360 / OFXS_HUE_CIRCLE;
} else {
*hsv = HSVColourF();
}}

void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
{
assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
_dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
float dhmin = 0.0;
float dhmax = 0.0;
float smin = FLT_MAX;
float smax = -FLT_MAX;
float vmin = FLT_MAX;
float vmax = -FLT_MAX;
for (int y = procWindow.y1; y < procWindow.y2; ++y) {
if ( _effect.abort() ) {
break;
}
PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);
for (int x = procWindow.x1; x < procWindow.x2; ++x) {
HSVColourF hsv;
pixToHSV(dstPix, &hsv);
if ( (hsv.s > MIN_SATURATION) && (hsv.v > MIN_VALUE) ) {
float dh = normalizeAngleSigned(hsv.h - _hmean);
if (dh < dhmin) { dhmin = dh; }
if (dh > dhmax) { dhmax = dh; }
}
if (hsv.s < smin) { smin = hsv.s; }
if (hsv.s > smax) { smax = hsv.s; }
if (hsv.v < vmin) { vmin = hsv.v; }
if (hsv.v > vmax) { vmax = hsv.v; }
dstPix += nComponents;
}}
addResults(dhmin, dhmax, smin, smax, vmin, vmax);
}};

////////////////////////////////////////////////////////////////////////////////

class ReplacePlugin
: public ImageEffect
{
public:
ReplacePlugin(OfxImageEffectHandle handle)
: ImageEffect(handle)
, _dstClip(0)
, _srcClip(0)
, _srcColour(0)
, _dstColour(0)
, _hueRange(0)
, _hueRotation(0)
, _hueRotationGain(0)
, _hueRangeRolloff(0)
, _saturationRange(0)
, _saturationAdjustment(0)
, _saturationAdjustmentGain(0)
, _saturationRangeRolloff(0)
, _brightnessRange(0)
, _brightnessAdjustment(0)
, _brightnessAdjustmentGain(0)
, _brightnessRangeRolloff(0)
, _outputAlpha(0)
, _displayAlpha(0)
, _mix(0)

{
_dstClip = fetchClip(kOfxImageEffectOutputClipName);
assert( _dstClip && (!_dstClip->isConnected() || _dstClip->getPixelComponents() == ePixelComponentRGB ||
_dstClip->getPixelComponents() == ePixelComponentRGBA) );
_srcClip = getContext() == eContextGenerator ? NULL : fetchClip(kOfxImageEffectSimpleSourceClipName);
assert( (!_srcClip && getContext() == eContextGenerator) ||
( _srcClip && (!_srcClip->isConnected() || _srcClip->getPixelComponents() ==  ePixelComponentRGB ||
_srcClip->getPixelComponents() == ePixelComponentRGBA) ) );

_btmLeft = fetchDouble2DParam(kParamRectangleInteractBtmLeft);
_size = fetchDouble2DParam(kParamRectangleInteractSize);
_enableRectangle = fetchBooleanParam(kParamEnableRectangle);
assert(_btmLeft && _size && _enableRectangle);
_setSrcFromRectangle = fetchPushButtonParam(kParamSetSrcFromRectangle);
assert(_setSrcFromRectangle);
_srcColour = fetchRGBParam(kParamSrcColour);
_dstColour = fetchRGBParam(kParamDstColour);
_hueRange = fetchDouble2DParam(kParamHueRange);
_hueRotation = fetchDoubleParam(kParamHueRotation);
_hueRotationGain = fetchDoubleParam(kParamHueRotationGain);
_hueRangeRolloff = fetchDoubleParam(kParamHueRangeRolloff);
_saturationRange = fetchDouble2DParam(kParamSaturationRange);
_saturationAdjustment = fetchDoubleParam(kParamSaturationAdjustment);
_saturationAdjustmentGain = fetchDoubleParam(kParamSaturationAdjustmentGain);
_saturationRangeRolloff = fetchDoubleParam(kParamSaturationRangeRolloff);
_brightnessRange = fetchDouble2DParam(kParamBrightnessRange);
_brightnessAdjustment = fetchDoubleParam(kParamBrightnessAdjustment);
_brightnessAdjustmentGain = fetchDoubleParam(kParamBrightnessAdjustmentGain);
_brightnessRangeRolloff = fetchDoubleParam(kParamBrightnessRangeRolloff);
m_Black = fetchDoubleParam(kParamCleanBlack);
m_White = fetchDoubleParam(kParamCleanWhite);
m_Blur = fetchDoubleParam(kParamCleanBlur);
m_Garbage = fetchDoubleParam(kParamCleanGarbage);
m_Core = fetchDoubleParam(kParamCleanCore);
m_Erode = fetchDoubleParam(kParamCleanErode);
m_Dilate = fetchDoubleParam(kParamCleanDilate);

assert(_srcColour && _dstColour &&
_hueRange && _hueRotation && _hueRotationGain && _hueRangeRolloff &&
_saturationRange && _saturationAdjustment && _saturationAdjustmentGain && _saturationRangeRolloff &&
_brightnessRange && _brightnessAdjustment && _brightnessAdjustmentGain && _brightnessRangeRolloff);
_outputAlpha = fetchChoiceParam(kParamOutputAlpha);
assert(_outputAlpha);
_displayAlpha = fetchBooleanParam(kParamDisplayAlpha);
assert(_displayAlpha);
_mix = fetchDoubleParam(kParamMix);
assert(_mix);

bool enableRectangle = _enableRectangle->getValue();
_btmLeft->setIsSecretAndDisabled(!enableRectangle);
_size->setIsSecretAndDisabled(!enableRectangle);
_setSrcFromRectangle->setIsSecretAndDisabled(!enableRectangle);
_srcColour->setEnabled(!enableRectangle);

if ( paramExists(kParamDefaultsNormalised) ) {
BooleanParam* param = fetchBooleanParam(kParamDefaultsNormalised);
assert(param);
bool normalised = param->getValue();
if (normalised) {
OfxPointD size = getProjectExtent();
OfxPointD origin = getProjectOffset();
OfxPointD p;
beginEditBlock(kParamDefaultsNormalised);
_btmLeft->getValue(p.x, p.y);
_btmLeft->setValue(p.x * size.x + origin.x, p.y * size.y + origin.y);
_size->getValue(p.x, p.y);
_size->setValue(p.x * size.x, p.y * size.y);
param->setValue(false);
endEditBlock();
}}}

private:
virtual void render(const RenderArguments &args) OVERRIDE FINAL;
void setupAndProcess(Replace &, const RenderArguments &args);
virtual bool isIdentity(const IsIdentityArguments &args, Clip * &identityClip, double &identityTime, int& view, std::string& plane); //OVERRIDE FINAL;
virtual void changedParam(const InstanceChangedArgs &args, const std::string &paramName) OVERRIDE FINAL;
virtual void getClipPreferences(ClipPreferencesSetter &clipPreferences) OVERRIDE FINAL;
bool computeWindow(const Image* srcImg, double time, OfxRectI *analysisWindow);
void setSrcFromRectangle(const Image* srcImg, double time, const OfxRectI& analysisWindow);
void setSrcFromRectangleProcess(HueMeanProcessorBase &huemeanprocessor, HSVRangeProcessorBase &rangeprocessor, const Image* srcImg, double /*time*/, const OfxRectI &analysisWindow, double *hmean, HSVColour *hsvmin, HSVColour *hsvmax);

template <class PIX, int nComponents, int maxValue>
void setSrcFromRectangleComponentsDepth(const Image* srcImg,double time,
const OfxRectI &analysisWindow, double *hmean, HSVColour *hsvmin, HSVColour *hsvmax) {
HueMeanProcessor<PIX, nComponents, maxValue> fred1(*this);
HSVRangeProcessor<PIX, nComponents, maxValue> fred2(*this);
setSrcFromRectangleProcess(fred1, fred2, srcImg, time, analysisWindow, hmean, hsvmin, hsvmax);
}

template <int nComponents>
void setSrcFromRectangleComponents(const Image* srcImg, double time,
const OfxRectI &analysisWindow, double *hmean, HSVColour *hsvmin, HSVColour *hsvmax) {
BitDepthEnum srcBitDepth = srcImg->getPixelDepth();
switch (srcBitDepth) {
case eBitDepthUByte: {
setSrcFromRectangleComponentsDepth<unsigned char, nComponents, 255>(srcImg, time, analysisWindow, hmean, hsvmin, hsvmax);
break;
}
case eBitDepthUShort: {
setSrcFromRectangleComponentsDepth<unsigned short, nComponents, 65535>(srcImg, time, analysisWindow, hmean, hsvmin, hsvmax);
break;
}
case eBitDepthFloat: {
setSrcFromRectangleComponentsDepth<float, nComponents, 1>(srcImg, time, analysisWindow, hmean, hsvmin, hsvmax);
break;
}
default:
throwSuiteStatusException(kOfxStatErrUnsupported);
}}

private:
Clip *_dstClip;
Clip *_srcClip;
Double2DParam* _btmLeft;
Double2DParam* _size;
BooleanParam* _enableRectangle;
PushButtonParam* _setSrcFromRectangle;
RGBParam *_srcColour;
RGBParam *_dstColour;
Double2DParam *_hueRange;
DoubleParam *_hueRotation;
DoubleParam *_hueRotationGain;
DoubleParam *_hueRangeRolloff;
Double2DParam *_saturationRange;
DoubleParam *_saturationAdjustment;
DoubleParam *_saturationAdjustmentGain;
DoubleParam *_saturationRangeRolloff;
Double2DParam *_brightnessRange;
DoubleParam *_brightnessAdjustment;
DoubleParam *_brightnessAdjustmentGain;
DoubleParam *_brightnessRangeRolloff;
ChoiceParam *_outputAlpha;
BooleanParam* _displayAlpha;
DoubleParam *_mix;
DoubleParam* m_Black;
DoubleParam* m_White;
DoubleParam* m_Blur;
DoubleParam* m_Garbage;
DoubleParam* m_Core;
DoubleParam* m_Erode;
DoubleParam* m_Dilate;

};

////////////////////////////////////////////////////////////////////////////////

void ReplacePlugin::setupAndProcess(Replace& p_Replace, const RenderArguments &args)
{
const double time = args.time;
std::auto_ptr<OFX::Image> dst( _dstClip->fetchImage(time) );
if ( !dst.get() ) {
OFX::throwSuiteStatusException(kOfxStatFailed);
}
OFX::BitDepthEnum dstBitDepth    = dst->getPixelDepth();
OFX::PixelComponentEnum dstComponents  = dst->getPixelComponents();
if ( ( dstBitDepth != _dstClip->getPixelDepth() ) ||
( dstComponents != _dstClip->getPixelComponents() ) ) {
setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong depth or components");
OFX::throwSuiteStatusException(kOfxStatFailed);
}
if ( (dst->getRenderScale().x != args.renderScale.x) ||
( dst->getRenderScale().y != args.renderScale.y) ||
( ( dst->getField() != OFX::eFieldNone) && ( dst->getField() != args.fieldToRender) ) ) {
setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
OFX::throwSuiteStatusException(kOfxStatFailed);
}
int outputalpha_i;
_outputAlpha->getValueAtTime(time, outputalpha_i);
OutputAlphaEnum outputAlpha = (OutputAlphaEnum)outputalpha_i;
if (outputAlpha != eOutputAlphaOff) {
if (dstComponents != OFX::ePixelComponentRGBA) {
setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host dit not take into account output components");
OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
return;
}}

std::auto_ptr<OFX::Image> src( ( _srcClip && _srcClip->isConnected() ) ? _srcClip->fetchImage(time) : 0 );
if ( src.get() ) {
if ( (src->getRenderScale().x != args.renderScale.x) ||
( src->getRenderScale().y != args.renderScale.y) ||
( ( src->getField() != OFX::eFieldNone) && ( src->getField() != args.fieldToRender) ) ) {
setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
OFX::throwSuiteStatusException(kOfxStatFailed);
}
OFX::BitDepthEnum srcBitDepth      = src->getPixelDepth();
OFX::PixelComponentEnum srcComponents = src->getPixelComponents();
if ( (srcBitDepth != dstBitDepth) || ( (outputAlpha == eOutputAlphaOff) && (srcComponents != dstComponents) ) ) {
OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
}}

int OutputAlpha = outputalpha_i;
bool displayAlpha = _displayAlpha->getValue();
int DisplayAlpha = displayAlpha ? 1 : 0;
double hueRangeA, hueRangeB;
_hueRange->getValueAtTime(time, hueRangeA, hueRangeB);
double hueRangeWithRollOffA, hueRangeWithRollOffB;
hueRangeWithRollOffA = hueRangeWithRollOffB = 0.0;
double hueRotation = _hueRotation->getValueAtTime(time);
double hueRotationGain = _hueRotationGain->getValueAtTime(time);
double hueMean = 0.0;
double hueRolloff = _hueRangeRolloff->getValueAtTime(time);
double satRangeA, satRangeB;
_saturationRange->getValueAtTime(time, satRangeA, satRangeB);
double satAdjust = _saturationAdjustment->getValueAtTime(time);
double satAdjustGain = _saturationAdjustmentGain->getValueAtTime(time);
double satRolloff = _saturationRangeRolloff->getValueAtTime(time);
double valRangeA, valRangeB;
_brightnessRange->getValueAtTime(time, valRangeA, valRangeB);
double valAdjust = _brightnessAdjustment->getValueAtTime(time);
double valAdjustGain = _brightnessAdjustmentGain->getValueAtTime(time);
double valRolloff = _brightnessRangeRolloff->getValueAtTime(time);
double mix = _mix->getValueAtTime(time);

float _Hue[8];
float _Sat[5];
float _Val[5];
float _Blur[7];
_Hue[0] = hueRangeA;
_Hue[1] = hueRangeB;
_Hue[2] = hueRangeWithRollOffA;
_Hue[3] = hueRangeWithRollOffB;
_Hue[4] = hueRotation;
_Hue[5] = hueRotationGain;
_Hue[6] = hueMean;
_Hue[7] = hueRolloff;
_Sat[0] = satRangeA;
_Sat[1] = satRangeB;
_Sat[2] = satAdjust;
_Sat[3] = satAdjustGain;
_Sat[4] = satRolloff;
_Val[0] = valRangeA;
_Val[1] = valRangeB;
_Val[2] = valAdjust;
_Val[3] = valAdjustGain;
_Val[4] = valRolloff;

_Blur[0] = m_Black->getValueAtTime(time);
_Blur[1] = m_White->getValueAtTime(time);
_Blur[2] = m_Blur->getValueAtTime(time);
_Blur[3] = m_Garbage->getValueAtTime(time);
_Blur[4] = m_Core->getValueAtTime(time);
_Blur[5] = m_Erode->getValueAtTime(time);
_Blur[6] = m_Dilate->getValueAtTime(time);

p_Replace.setScales(_Hue, _Sat, _Val, OutputAlpha, DisplayAlpha, mix, _Blur);

p_Replace.setDstImg(dst.get());
p_Replace.setSrcImg(src.get());

// Setup GPU Render arguments
p_Replace.setGPURenderArgs(args);

p_Replace.setRenderWindow(args.renderWindow);

p_Replace.process();
} 

void ReplacePlugin::render(const RenderArguments &args)
{
BitDepthEnum dstBitDepth    = _dstClip->getPixelDepth();
PixelComponentEnum dstComponents  = _dstClip->getPixelComponents();
assert( kSupportsMultipleClipPARs   || !_srcClip || _srcClip->getPixelAspectRatio() == _dstClip->getPixelAspectRatio() );
assert( kSupportsMultipleClipDepths || !_srcClip || _srcClip->getPixelDepth()       == _dstClip->getPixelDepth() );
assert(dstComponents == ePixelComponentRGB || dstComponents == ePixelComponentRGBA);
if (dstComponents == ePixelComponentRGBA || ePixelComponentRGB) {
Replace fred(*this);
setupAndProcess(fred, args);
} else {
throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool ReplacePlugin::isIdentity(const IsIdentityArguments &args, Clip * &identityClip, double &, int&, std::string&) {
if (!_srcClip || !_srcClip->isConnected()) {
return false;
}
const double time = args.time;
if (_srcClip->getPixelComponents() == ePixelComponentRGBA) {
int outputalpha_i;
_outputAlpha->getValueAtTime(time, outputalpha_i);
OutputAlphaEnum outputAlpha = (OutputAlphaEnum)outputalpha_i;
if (outputAlpha != eOutputAlphaOff) {
double hueMin, hueMax;
_hueRange->getValueAtTime(time, hueMin, hueMax);
bool alphaHue = (hueMin != 0.0 || hueMax != 360.0);
double satMin, satMax;
_saturationRange->getValueAtTime(time, satMin, satMax);
bool alphaSat = (satMin != 0.0 || satMax != 1.0);
double valMin, valMax;
_brightnessRange->getValueAtTime(time, valMin, valMax);
bool alphaVal = (valMin != 0.0 || valMax != 1.0);
switch (outputAlpha) {
case eOutputAlphaOff:
break;
case eOutputAlphaHue:
if (alphaHue) {
return false;
}
break;
case eOutputAlphaSaturation:
if (alphaSat) {
return false;
}
break;
case eOutputAlphaBrightness:
if (alphaVal) {
return false;
}
break;
case eOutputAlphaHueSaturation:
if (alphaHue || alphaSat) {
return false;
}
break;
case eOutputAlphaHueBrightness:
if (alphaHue || alphaVal) {
return false;
}
break;
case eOutputAlphaSaturationBrightness:
if (alphaSat || alphaVal) {
return false;
}
break;
case eOutputAlphaAll:
if (alphaHue || alphaSat || alphaVal) {
return false;
}
break;
}}}

double hueRotation;
_hueRotation->getValueAtTime(time, hueRotation);
double saturationAdjustment;
_saturationAdjustment->getValueAtTime(time, saturationAdjustment);
double brightnessAdjustment;
_brightnessAdjustment->getValueAtTime(time, brightnessAdjustment);
if ( (hueRotation == 0.0) && (saturationAdjustment == 0.0) && (brightnessAdjustment == 0.0) ) {
identityClip = _srcClip;
return true;
}
return false;
}

bool ReplacePlugin::computeWindow(const OFX::Image* srcImg, double time, OfxRectI *analysisWindow) {
OfxRectD regionOfInterest;
bool enableRectangle = _enableRectangle->getValueAtTime(time);
if (!enableRectangle && _srcClip) {
return false;
} else {
_btmLeft->getValueAtTime(time, regionOfInterest.x1, regionOfInterest.y1);
_size->getValueAtTime(time, regionOfInterest.x2, regionOfInterest.y2);
regionOfInterest.x2 += regionOfInterest.x1;
regionOfInterest.y2 += regionOfInterest.y1;
}
OFX::Coords::toPixelEnclosing(regionOfInterest, srcImg->getRenderScale(), srcImg->getPixelAspectRatio(), analysisWindow);
return OFX::Coords::rectIntersection(*analysisWindow, srcImg->getBounds(), analysisWindow);
}

void ReplacePlugin::setSrcFromRectangle(const OFX::Image* srcImg, double time, const OfxRectI &analysisWindow)
{
double hmean = 0.0;
HSVColour hsvmin, hsvmax;
OFX::PixelComponentEnum srcComponents = srcImg->getPixelComponents();
assert(srcComponents == OFX::ePixelComponentAlpha || srcComponents == OFX::ePixelComponentRGB || srcComponents == OFX::ePixelComponentRGBA);
if (srcComponents == OFX::ePixelComponentAlpha) {
setSrcFromRectangleComponents<1>(srcImg, time, analysisWindow, &hmean, &hsvmin, &hsvmax);
} else if (srcComponents == OFX::ePixelComponentRGBA) {
setSrcFromRectangleComponents<4>(srcImg, time, analysisWindow, &hmean, &hsvmin, &hsvmax);
} else if (srcComponents == OFX::ePixelComponentRGB) {
setSrcFromRectangleComponents<3>(srcImg, time, analysisWindow, &hmean, &hsvmin, &hsvmax);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
return;
}
if ( abort() ) {
return;
}
float h = normalizeAngle(hmean);
float s = (hsvmin.s + hsvmax.s) / 2;
float v = (hsvmin.v + hsvmax.v) / 2;
float r = 0.0f;
float g = 0.0f;
float b = 0.0f;
OFX::Color::hsv_to_rgb(h * OFXS_HUE_CIRCLE / 360.0, s, v, &r, &g, &b);
double tor, tog, tob;
_dstColour->getValueAtTime(time, tor, tog, tob);
float toh, tos, tov;
OFX::Color::rgb_to_hsv( (float)tor, (float)tog, (float)tob, &toh, &tos, &tov );
double dh = normalizeAngleSigned(toh * 360.0 / OFXS_HUE_CIRCLE - h);
beginEditBlock("setSrcFromRectangle");
_srcColour->setValue( fround(r, 4), fround(g, 4), fround(b, 4) );
_hueRange->setValue( ffloor(hsvmin.h, 2), fceil(hsvmax.h, 2) );
double hrange = hsvmax.h - hsvmin.h;
if (hrange < 0) {
hrange += 360.0;
}
double hrolloff = fmin(hrange * DEFAULT_RECTANGLE_ROLLOFF, (360 - hrange) / 2);
_hueRangeRolloff->setValue( ffloor(hrolloff, 2) );
if (tov != 0.0) {
_hueRotation->setValue( fround(dh, 2) );
}
_saturationRange->setValue( ffloor(hsvmin.s, 4), fceil(hsvmax.s, 4) );
_saturationRangeRolloff->setValue( ffloor( (hsvmax.s - hsvmin.s) * DEFAULT_RECTANGLE_ROLLOFF, 4 ) );
if (tov != 0.0) {
_saturationAdjustment->setValue( fround(tos - s, 4) );
}
_brightnessRange->setValue( ffloor(hsvmin.v, 4), fceil(hsvmax.v, 4) );
_brightnessRangeRolloff->setValue( ffloor( (hsvmax.v - hsvmin.v) * DEFAULT_RECTANGLE_ROLLOFF, 4 ) );
_brightnessAdjustment->setValue( fround(tov - v, 4) );
endEditBlock();
}

void ReplacePlugin::setSrcFromRectangleProcess(HueMeanProcessorBase& p_Huemeanprocessor,
HSVRangeProcessorBase& p_Hsvrangeprocessor, const OFX::Image* srcImg, double, 
const OfxRectI &analysisWindow, double *hmean, HSVColour *hsvmin, HSVColour *hsvmax) {
p_Huemeanprocessor.setDstImg( const_cast<OFX::Image*>(srcImg) );
p_Huemeanprocessor.setRenderWindow(analysisWindow);
p_Huemeanprocessor.process();
if ( abort() ) {
return;
}
*hmean = p_Huemeanprocessor.getResult();
p_Hsvrangeprocessor.setDstImg( const_cast<OFX::Image*>(srcImg) );
p_Hsvrangeprocessor.setRenderWindow(analysisWindow);
p_Hsvrangeprocessor.setHueMean(*hmean);
p_Hsvrangeprocessor.process();
if ( abort() ) {
return;
}
p_Hsvrangeprocessor.getResults(hsvmin, hsvmax);
}

void ReplacePlugin::changedParam(const OFX::InstanceChangedArgs& args, const std::string& p_ParamName) {
const double time = args.time;
if(p_ParamName == "info")
{
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}
if ( (p_ParamName == kParamSrcColour) && (args.reason == OFX::eChangeUserEdit) ) {
double r, g, b;
_srcColour->getValueAtTime(time, r, g, b);
float h, s, v;
OFX::Color::rgb_to_hsv( (float)r, (float)g, (float)b, &h, &s, &v );
h *= 360.0 / OFXS_HUE_CIRCLE;
double tor, tog, tob;
_dstColour->getValueAtTime(time, tor, tog, tob);
float toh, tos, tov;
OFX::Color::rgb_to_hsv( (float)tor, (float)tog, (float)tob, &toh, &tos, &tov );
toh *= 360.0 / OFXS_HUE_CIRCLE;
double dh = normalizeAngleSigned(toh - h);
beginEditBlock("setSrc");
_hueRange->setValue(h, h);
_hueRangeRolloff->setValue(50.0);
if (tov != 0.0) {
_hueRotation->setValue(dh);
}
_saturationRange->setValue(s, s);
_saturationRangeRolloff->setValue(0.3);
if (tov != 0.0) {
_saturationAdjustment->setValue(tos - s);
}
_brightnessRange->setValue(v, v);
_brightnessRangeRolloff->setValue(0.3);
_brightnessAdjustment->setValue(tov - v);
endEditBlock();
} else if (p_ParamName == kParamEnableRectangle) {
bool enableRectangle = _enableRectangle->getValueAtTime(time);
_btmLeft->setIsSecretAndDisabled(!enableRectangle);
_size->setIsSecretAndDisabled(!enableRectangle);
_setSrcFromRectangle->setIsSecretAndDisabled(!enableRectangle);
_srcColour->setEnabled(!enableRectangle);
} else if ( (p_ParamName == kParamSetSrcFromRectangle) && (args.reason == OFX::eChangeUserEdit) ) {
std::auto_ptr<OFX::Image> src( ( _srcClip && _srcClip->isConnected() ) ? _srcClip->fetchImage(args.time) : 0 );
if ( src.get() ) {
if ( (src->getRenderScale().x != args.renderScale.x) ||
( src->getRenderScale().y != args.renderScale.y) ) {
setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
OFX::throwSuiteStatusException(kOfxStatFailed);
}
OfxRectI analysisWindow;
bool intersect = computeWindow(src.get(), args.time, &analysisWindow);
if (intersect) {
#  ifdef kOfxImageEffectPropInAnalysis
getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 1, false);
#  endif
setSrcFromRectangle(src.get(), args.time, analysisWindow);
#  ifdef kOfxImageEffectPropInAnalysis
getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 0, false);
#  endif
}}} else if ( (p_ParamName == kParamDstColour) && (args.reason == OFX::eChangeUserEdit) ) {
double r, g, b;
_srcColour->getValueAtTime(time, r, g, b);
float h, s, v;
OFX::Color::rgb_to_hsv( (float)r, (float)g, (float)b, &h, &s, &v );
h *= 360.0 / OFXS_HUE_CIRCLE;
double tor, tog, tob;
_dstColour->getValueAtTime(time, tor, tog, tob);
float toh, tos, tov;
OFX::Color::rgb_to_hsv( (float)tor, (float)tog, (float)tob, &toh, &tos, &tov );
toh *= 360.0 / OFXS_HUE_CIRCLE;
double dh = normalizeAngleSigned(toh - h);
beginEditBlock("setDst");
if (tov != 0.0) {
_hueRotation->setValue(dh);
_saturationAdjustment->setValue(tos - s);
}
_brightnessAdjustment->setValue(tov - v);
endEditBlock();
}

if (p_ParamName == kParamCleanBlur) {
float blur = m_Blur->getValueAtTime(time);
bool gc = blur != 0.0f;
m_Garbage->setEnabled(gc);
m_Core->setEnabled(gc);
}}

void ReplacePlugin::getClipPreferences(ClipPreferencesSetter& p_ClipPreferences)
{
int outputalpha_i;
_outputAlpha->getValue(outputalpha_i);
OutputAlphaEnum outputAlpha = (OutputAlphaEnum)outputalpha_i;
if (outputAlpha != eOutputAlphaOff) {
p_ClipPreferences.setClipComponents(*_dstClip, OFX::ePixelComponentRGBA);
p_ClipPreferences.setClipComponents(*_srcClip, OFX::ePixelComponentRGBA);
}}

class ReplaceInteract
: public OFX::RectangleInteract
{
public:

ReplaceInteract(OfxInteractHandle p_Handle, ImageEffect* effect)
: RectangleInteract(p_Handle, effect)
, _enableRectangle(0)
{
_enableRectangle = effect->fetchBooleanParam(kParamEnableRectangle);
addParamToSlaveTo(_enableRectangle);
}

private:

virtual bool draw(const OFX::DrawArgs& args) OVERRIDE FINAL
{
bool enableRectangle = _enableRectangle->getValueAtTime(args.time);
if (enableRectangle) {
return RectangleInteract::draw(args);
}
return false;
}

virtual bool penMotion(const OFX::PenArgs& args) OVERRIDE FINAL
{
bool enableRectangle = _enableRectangle->getValueAtTime(args.time);
if (enableRectangle) {
return RectangleInteract::penMotion(args);
}
return false;
}

virtual bool penDown(const OFX::PenArgs& args) OVERRIDE FINAL
{
bool enableRectangle = _enableRectangle->getValueAtTime(args.time);
if (enableRectangle) {
return RectangleInteract::penDown(args);
}
return false;
}

virtual bool penUp(const OFX::PenArgs& args) OVERRIDE FINAL
{
bool enableRectangle = _enableRectangle->getValueAtTime(args.time);
if (enableRectangle) {
return RectangleInteract::penUp(args);
}
return false;
}
OFX::BooleanParam* _enableRectangle;
};

class ReplaceOverlayDescriptor
: public OFX::DefaultEffectOverlayDescriptor<ReplaceOverlayDescriptor, ReplaceInteract>
{
};

ReplacePluginFactory::ReplacePluginFactory()
: OFX::PluginFactoryHelper<ReplacePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ReplacePluginFactory::describe(OFX::ImageEffectDescriptor &desc)
{
desc.setLabel(kPluginName);
desc.setPluginGrouping(kPluginGrouping);
desc.setPluginDescription(kPluginDescription);

desc.addSupportedContext(eContextFilter);
desc.addSupportedContext(eContextGeneral);
desc.addSupportedContext(eContextPaint);
desc.addSupportedBitDepth(eBitDepthFloat);

desc.setSingleInstance(false);
desc.setHostFrameThreading(false);
desc.setSupportsMultiResolution(kSupportsMultiResolution);
desc.setSupportsTiles(kSupportsTiles);
desc.setTemporalClipAccess(false);
desc.setRenderTwiceAlways(false);
desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);
desc.setSupportsMultipleClipDepths(kSupportsMultipleClipDepths);

// Setup GPU render capability flags
desc.setSupportsOpenCLRender(true);
desc.setSupportsCudaRender(true);
//#ifdef __APPLE__
//desc.setSupportsMetalRender(true);
//#endif
desc.setOverlayInteractDescriptor(new ReplaceOverlayDescriptor);
}

void ReplacePluginFactory::describeInContext(ImageEffectDescriptor& desc, ContextEnum)
{
ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
srcClip->addSupportedComponent(ePixelComponentRGBA);
srcClip->addSupportedComponent(ePixelComponentRGB);
srcClip->setTemporalClipAccess(false);
srcClip->setSupportsTiles(kSupportsTiles);
ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
dstClip->addSupportedComponent(ePixelComponentRGBA);
dstClip->addSupportedComponent(ePixelComponentRGB);
dstClip->setSupportsTiles(kSupportsTiles);

PageParamDescriptor *page = desc.definePageParam("Controls");

GroupParamDescriptor *group = desc.defineGroupParam(kGroupColourReplacement);
if (group) {
group->setLabel(kGroupColourReplacementLabel);
group->setHint(kGroupColourReplacementHint);
group->setEnabled(true);
if (page)
page->addChild(*group);
}

BooleanParamDescriptor *boolparam = desc.defineBooleanParam(kParamEnableRectangle);
boolparam->setLabel(kParamEnableRectangleLabel);
boolparam->setHint(kParamEnableRectangleHint);
boolparam->setDefault(false);
boolparam->setAnimates(false);
boolparam->setEvaluateOnChange(false);
if (group)
boolparam->setParent(*group);
if (page)
page->addChild(*boolparam);

Double2DParamDescriptor* param2D = desc.defineDouble2DParam(kParamRectangleInteractBtmLeft);
param2D->setLabel(kParamRectangleInteractBtmLeftLabel);
param2D->setDoubleType(eDoubleTypeXYAbsolute);
if ( param2D->supportsDefaultCoordinateSystem() ) {
param2D->setDefaultCoordinateSystem(eCoordinatesNormalised);
} else {
gHostSupportsDefaultCoordinateSystem = false;
}
param2D->setDefault(0.4, 0.4);
param2D->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(0, 0, 10000, 10000);
param2D->setIncrement(1.0);
param2D->setHint(kParamRectangleInteractBtmLeftHint);
param2D->setDigits(0);
param2D->setEvaluateOnChange(false);
param2D->setAnimates(true);
if (group)
param2D->setParent(*group);
if (page)
page->addChild(*param2D);

param2D = desc.defineDouble2DParam(kParamRectangleInteractSize);
param2D->setLabel(kParamRectangleInteractSizeLabel);
param2D->setDoubleType(eDoubleTypeXY);
if ( param2D->supportsDefaultCoordinateSystem() ) {
param2D->setDefaultCoordinateSystem(eCoordinatesNormalised);
} else {
gHostSupportsDefaultCoordinateSystem = false;
}
param2D->setDefault(0.2, 0.2);
param2D->setRange(0.0, 0.0, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(0, 0, 10000, 10000);
param2D->setIncrement(1.0);
param2D->setDimensionLabels(kParamRectangleInteractSizeDim1, kParamRectangleInteractSizeDim2);
param2D->setHint(kParamRectangleInteractSizeHint);
param2D->setDigits(0);
param2D->setEvaluateOnChange(false);
param2D->setAnimates(true);
if (group)
param2D->setParent(*group);
if (page)
page->addChild(*param2D);

PushButtonParamDescriptor *pushparam = desc.definePushButtonParam(kParamSetSrcFromRectangle);
pushparam->setLabel(kParamSetSrcFromRectangleLabel);
pushparam->setHint(kParamSetSrcFromRectangleHint);
if (group)
pushparam->setParent(*group);
if (page)
page->addChild(*pushparam);

RGBParamDescriptor *paramRGB = desc.defineRGBParam(kParamSrcColour);
paramRGB->setLabel(kParamSrcColourLabel);
paramRGB->setHint(kParamSrcColourHint);
paramRGB->setEvaluateOnChange(false);
if (group)
paramRGB->setParent(*group);
if (page)
page->addChild(*paramRGB);

paramRGB = desc.defineRGBParam(kParamDstColour);
paramRGB->setLabel(kParamDstColourLabel);
paramRGB->setHint(kParamDstColourHint);
paramRGB->setEvaluateOnChange(false);
if (group)
paramRGB->setParent(*group);
if (page)
page->addChild(*paramRGB);

DoubleParamDescriptor *param = desc.defineDoubleParam(kParamMix);
param->setLabel(kParamMixLabel);
param->setHint(kParamMixHint);
param->setRange(0.0, 1.0);
param->setDisplayRange(0.0, 1.0);
param->setDefault(0.0);
if (group)
param->setParent(*group);
if (page)
page->addChild(*param);

pushparam = desc.definePushButtonParam("info");
pushparam->setLabel("Info");
if (group)
pushparam->setParent(*group);
if (page)
page->addChild(*pushparam);

GroupParamDescriptor *groupHue = desc.defineGroupParam(kGroupHue);
if (groupHue) {
groupHue->setLabel(kGroupHueLabel);
groupHue->setHint(kGroupHueHint);
groupHue->setEnabled(true);
if (page)
page->addChild(*groupHue);
}

param2D = desc.defineDouble2DParam(kParamHueRange);
param2D->setLabel(kParamHueRangeLabel);
param2D->setHint(kParamHueRangeHint);
param2D->setDimensionLabels("", "");
param2D->setDefault(0.0, 360.0);
param2D->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(0.0, 0.0, 360.0, 360.0);
param2D->setDoubleType(eDoubleTypeAngle);
param2D->setUseHostNativeOverlayHandle(false);
if (groupHue)
param2D->setParent(*groupHue);
if (page)
page->addChild(*param2D);

param = desc.defineDoubleParam(kParamHueRotation);
param->setLabel(kParamHueRotationLabel);
param->setHint(kParamHueRotationHint);
param->setRange(-DBL_MAX, DBL_MAX);
param->setDisplayRange(-180.0, 180.0);
param->setDoubleType(eDoubleTypeAngle);
if (groupHue)
param->setParent(*groupHue);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamHueRotationGain);
param->setLabel(kParamHueRotationGainLabel);
param->setHint(kParamHueRotationGainHint);
param->setRange(-DBL_MAX, DBL_MAX);
param->setDisplayRange(0.0, 2.0);
param->setDefault(1.0);
if (groupHue)
param->setParent(*groupHue);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamHueRangeRolloff);
param->setLabel(kParamHueRangeRolloffLabel);
param->setHint(kParamHueRangeRolloffHint);
param->setRange(0.0, 180.0);
param->setDisplayRange(0.0, 180.0);
param->setDoubleType(eDoubleTypeAngle);
if (groupHue)
param->setParent(*groupHue);
if (page)
page->addChild(*param);

GroupParamDescriptor *groupSat = desc.defineGroupParam(kGroupSaturation);
if (groupSat) {
groupSat->setLabel(kGroupSaturationLabel);
groupSat->setHint(kGroupSaturationHint);
groupSat->setEnabled(true);
if (page)
page->addChild(*groupSat);
}

param2D = desc.defineDouble2DParam(kParamSaturationRange);
param2D->setLabel(kParamSaturationRangeLabel);
param2D->setHint(kParamSaturationRangeHint);
param2D->setDimensionLabels("", "");
param2D->setDefault(0.0, 1.0);
param2D->setRange(0.0, 0.0, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(0.0, 0.0, 1.0, 1.0);
param2D->setUseHostNativeOverlayHandle(false);
if (groupSat)
param2D->setParent(*groupSat);
if (page)
page->addChild(*param2D);

param = desc.defineDoubleParam(kParamSaturationAdjustment);
param->setLabel(kParamSaturationAdjustmentLabel);
param->setHint(kParamSaturationAdjustmentHint);
param->setRange(-1.0, 1.0);
param->setDisplayRange(-1.0, 1.0);
if (groupSat)
param->setParent(*groupSat);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamSaturationAdjustmentGain);
param->setLabel(kParamSaturationAdjustmentGainLabel);
param->setHint(kParamSaturationAdjustmentGainHint);
param->setRange(-DBL_MAX, DBL_MAX);
param->setDisplayRange(0.0, 2.0);
param->setDefault(1.0);
if (groupSat)
param->setParent(*groupSat);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamSaturationRangeRolloff);
param->setLabel(kParamSaturationRangeRolloffLabel);
param->setHint(kParamSaturationRangeRolloffHint);
param->setRange(0.0, 1.0);
param->setDisplayRange(0.0, 1.0);
if (groupSat)
param->setParent(*groupSat);
if (page)
page->addChild(*param);

GroupParamDescriptor *groupBright = desc.defineGroupParam(kGroupBrightness);
if (groupBright) {
groupBright->setLabel(kGroupBrightnessLabel);
groupBright->setHint(kGroupBrightnessHint);
groupBright->setEnabled(true);
if (page)
page->addChild(*groupBright);
}

param2D = desc.defineDouble2DParam(kParamBrightnessRange);
param2D->setLabel(kParamBrightnessRangeLabel);
param2D->setHint(kParamBrightnessRangeHint);
param2D->setDimensionLabels("", "");
param2D->setDefault(0.0, 1.0);
param2D->setRange(0.0, 0.0, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(0.0, 0.0, 1.0, 1.0);
param2D->setUseHostNativeOverlayHandle(false);
if (groupBright)
param2D->setParent(*groupBright);
if (page)
page->addChild(*param2D);

param = desc.defineDoubleParam(kParamBrightnessAdjustment);
param->setLabel(kParamBrightnessAdjustmentLabel);
param->setHint(kParamBrightnessAdjustmentHint);
param->setRange(-DBL_MAX, DBL_MAX);
param->setDisplayRange(-1.0, 1.0);
if (groupBright)
param->setParent(*groupBright);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamBrightnessAdjustmentGain);
param->setLabel(kParamBrightnessAdjustmentGainLabel);
param->setHint(kParamBrightnessAdjustmentGainHint);
param->setRange(-DBL_MAX, DBL_MAX);
param->setDisplayRange(0.0, 2.0);
param->setDefault(1.0);
if (groupBright)
param->setParent(*groupBright);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamBrightnessRangeRolloff);
param->setLabel(kParamBrightnessRangeRolloffLabel);
param->setHint(kParamBrightnessRangeRolloffHint);
param->setRange(0.0, DBL_MAX);
param->setDisplayRange(0.0, 1.0);
if (groupBright)
param->setParent(*groupBright);
if (page)
page->addChild(*param);

ChoiceParamDescriptor *choiceparam = desc.defineChoiceParam(kParamOutputAlpha);
choiceparam->setLabel(kParamOutputAlphaLabel);
choiceparam->setHint(kParamOutputAlphaHint);
assert(choiceparam->getNOptions() == (int)eOutputAlphaOff);
choiceparam->appendOption(kParamOutputAlphaOptionOff, kParamOutputAlphaOptionOffHint);
assert(choiceparam->getNOptions() == (int)eOutputAlphaHue);
choiceparam->appendOption(kParamOutputAlphaOptionHue, kParamOutputAlphaOptionHueHint);
assert(choiceparam->getNOptions() == (int)eOutputAlphaSaturation);
choiceparam->appendOption(kParamOutputAlphaOptionSaturation, kParamOutputAlphaOptionSaturationHint);
assert(choiceparam->getNOptions() == (int)eOutputAlphaBrightness);
choiceparam->appendOption(kParamOutputAlphaOptionBrightness, kParamOutputAlphaOptionBrightnessHint);
assert(choiceparam->getNOptions() == (int)eOutputAlphaHueSaturation);
choiceparam->appendOption(kParamOutputAlphaOptionHueSaturation, kParamOutputAlphaOptionHueSaturationHint);
assert(choiceparam->getNOptions() == (int)eOutputAlphaHueBrightness);
choiceparam->appendOption(kParamOutputAlphaOptionHueBrightness, kParamOutputAlphaOptionHueBrightnessHint);
assert(choiceparam->getNOptions() == (int)eOutputAlphaSaturationBrightness);
choiceparam->appendOption(kParamOutputAlphaOptionSaturationBrightness, kParamOutputAlphaOptionSaturationBrightnessHint);
assert(choiceparam->getNOptions() == (int)eOutputAlphaAll);
choiceparam->appendOption(kParamOutputAlphaOptionAll, kParamOutputAlphaOptionAllHint);
choiceparam->setDefault( (int)eOutputAlphaAll );
choiceparam->setAnimates(false);
desc.addClipPreferencesSlaveParam(*choiceparam);
if (page)
page->addChild(*choiceparam);

boolparam = desc.defineBooleanParam(kParamDisplayAlpha);
boolparam->setLabel(kParamDisplayAlphaLabel);
boolparam->setHint(kParamDisplayAlphaHint);
boolparam->setDefault(false);
boolparam->setAnimates(false);
if (page)
page->addChild(*boolparam);

GroupParamDescriptor *groupclean = desc.defineGroupParam(kGroupClean);
if (groupclean) {
groupclean->setOpen(false);
groupclean->setLabel(kGroupCleanLabel);
groupclean->setHint(kGroupCleanHint);
groupclean->setEnabled(true);
if (page)
page->addChild(*groupclean);
}

param = desc.defineDoubleParam(kParamCleanBlack);
param->setLabel(kParamCleanBlackLabel);
param->setHint(kParamCleanBlackHint);
param->setDefault(0.0);
param->setRange(0.0f, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
if (groupclean)
param->setParent(*groupclean);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamCleanWhite);
param->setLabel(kParamCleanWhiteLabel);
param->setHint(kParamCleanWhiteHint);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
if (groupclean)
param->setParent(*groupclean);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamCleanBlur);
param->setLabel(kParamCleanBlurLabel);
param->setHint(kParamCleanBlurHint);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
if (groupclean)
param->setParent(*groupclean);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamCleanGarbage);
param->setLabel(kParamCleanGarbageLabel);
param->setHint(kParamCleanGarbageHint);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setEnabled(false);
if (groupclean)
param->setParent(*groupclean);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamCleanCore);
param->setLabel(kParamCleanCoreLabel);
param->setHint(kParamCleanCoreHint);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setEnabled(false);
if (groupclean)
param->setParent(*groupclean);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamCleanErode);
param->setLabel(kParamCleanErodeLabel);
param->setHint(kParamCleanErodeHint);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
if (groupclean)
param->setParent(*groupclean);
if (page)
page->addChild(*param);

param = desc.defineDoubleParam(kParamCleanDilate);
param->setLabel(kParamCleanDilateLabel);
param->setHint(kParamCleanDilateHint);
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
if (groupclean)
param->setParent(*groupclean);
if (page)
page->addChild(*param);

if (!gHostSupportsDefaultCoordinateSystem) {
boolparam = desc.defineBooleanParam(kParamDefaultsNormalised);
boolparam->setDefault(true);
boolparam->setEvaluateOnChange(false);
boolparam->setIsSecretAndDisabled(true);
boolparam->setIsPersistant(true);
boolparam->setAnimates(false);
if (page)
page->addChild(*boolparam);
}}

ImageEffect* ReplacePluginFactory::createInstance(OfxImageEffectHandle handle, ContextEnum)
{
return new ReplacePlugin(handle);
}

void Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static ReplacePluginFactory replacePlugin;
p_FactoryArray.push_back(&replacePlugin);
}