#include "ScanPlugin.h"

#include <cstring>
using std::string;
#include <string> 
#include <fstream>

#include <stdio.h>
#include <cmath>
#include <cfloat>
#include <climits>
#include <algorithm>
#include <limits>


#include "ofxsProcessing.h"
#include "ofxsRectangleInteract.h"
#include "ofxsMacros.h"
#include "ofxsCopier.h"
#include "ofxsCoords.h"
#include "ofxsLut.h"
#include "ofxsMultiThread.h"
#include "ofxsLog.h"
#include "ofxsThreadSuite.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#include <windows.h>
#define isnan _isnan
#else
using std::isnan;
#endif

#ifdef __APPLE__
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#else
#define kPluginScript "/home/resolve/LUT"
#endif

using namespace OFX;

#define kPluginName "Scan"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"White Balance and Image Analysis. Use ColorPicker or Rectangle to sample pixel values. \n" \
"Derive Min, Max, Mean RGB and HSV values, Median RGB values, and Min, Max Luma values \n" \
"and their pixel coordinates. Apply White Balance with ColorPicker, Mean, or Median source \n" \
"via Gain, Offset, or Lift. Option to Preserve Luma and/or apply luminance mask (Luma Limiter) \n" \
"based on coefficients selected in Min/Max Luma section. Median sample region is atomatically \n" \
"scaled down if it exceeds the safety limit.\n" \

#define kPluginIdentifier "BaldavengerOFX.Scan"
#define kPluginVersionMajor 2 
#define kPluginVersionMinor 2 

#define kSupportsTiles 1
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 0
#define kSupportsMultipleClipPARs false
#define kSupportsMultipleClipDepths false
#define kRenderThreadSafety eRenderFullySafe

#define kResolutionScale	(float)width / 1920.0f

#define kParamRestrictToRectangle "restrictToRectangle"
#define kParamRestrictToRectangleLabel "Restrict to Rectangle"
#define kParamRestrictToRectangleHint "Restrict statistics computation to a rectangle."

#define kParamSampleType "Sample Type"
#define kParamSampleTypeLabel "Sample Type"
#define kParamSampleTypeHint "ColorPicker, Mean Average, or Median"
#define kParamSampleTypeOptionColorPicker "ColorPicker"
#define kParamSampleTypeOptionColorPickerHint "Use ColorPicker Sample Tool"
#define kParamSampleTypeOptionMean "Mean Average"
#define kParamSampleTypeOptionMeanHint "Use Mean Average"
#define kParamSampleTypeOptionMedian "Median Average"
#define kParamSampleTypeOptionMedianHint "Use Median Average"

enum SampleTypeEnum
{
eSampleTypeColorPicker,
eSampleTypeMean,
eSampleTypeMedian,
};

#define kParamBalanceType "Balance Type"
#define kParamBalanceTypeLabel "Balance Type"
#define kParamBalanceTypeHint "White Balance Formula"
#define kParamBalanceTypeOptionGain "Gain"
#define kParamBalanceTypeOptionGainHint "Use Gain Formula"
#define kParamBalanceTypeOptionOffset "Offset"
#define kParamBalanceTypeOptionOffsetHint "Use Offset Formula"
#define kParamBalanceTypeOptionLift "Lift"
#define kParamBalanceTypeOptionLiftHint "Use Lift Formula"

enum BalanceTypeEnum
{
eBalanceTypeGain,
eBalanceTypeOffset,
eBalanceTypeLift,
};

#define kParamAnalyzeFrame "analyzeFrame"
#define kParamAnalyzeFrameLabel "Analyze Frame RGB"
#define kParamAnalyzeFrameHint "Analyze current frame and set values."

#define kParamAnalyzeFrameMed "analyzeFrameMed"
#define kParamAnalyzeFrameMedLabel "Get Median"
#define kParamAnalyzeFrameMedHint "Analyze current frame and get median values."

#define kParamClearFrame "clearFrame"
#define kParamClearFrameLabel "Reset"
#define kParamClearFrameHint "Clear analysis for current frame."

#define kParamGroupRGB "RGB"

#define kParamStatMin "statMin"
#define kParamStatMinLabel "Min."
#define kParamStatMinHint "Minimum value."

#define kParamStatMax "statMax"
#define kParamStatMaxLabel "Max."
#define kParamStatMaxHint "Maximum value."

#define kParamStatMean "statMean"
#define kParamStatMeanLabel "Mean"
#define kParamStatMeanHint "The mean is the average. Add up the values, and divide by the number of values."

#define kParamStatMedian "statMedian"
#define kParamStatMedianLabel "Median"
#define kParamStatMedianHint "The median average. The middle value. Very processor intensive. Sample area restricted to limited size."

#define kParamGroupHSV "HSV"

#define kParamAnalyzeFrameHSV "analyzeFrameHSV"
#define kParamAnalyzeFrameHSVLabel "Analyze Frame HSV"
#define kParamAnalyzeFrameHSVHint "Analyze current frame as HSV and set values."

#define kParamClearFrameHSV "clearFrameHSV"
#define kParamClearFrameHSVLabel "Reset"
#define kParamClearFrameHSVHint "Clear HSV analysis for current frame."

#define kParamStatHSVMin "statHSVMin"
#define kParamStatHSVMinLabel "HSV Min."
#define kParamStatHSVMinHint "Minimum value."

#define kParamStatHSVMax "statHSVMax"
#define kParamStatHSVMaxLabel "HSV Max."
#define kParamStatHSVMaxHint "Maximum value."

#define kParamStatHSVMean "statHSVMean"
#define kParamStatHSVMeanLabel "HSV Mean"
#define kParamStatHSVMeanHint "The mean is the average. Add up the values, and divide by the number of values."

#define kParamGroupLuma "Min/Max Luma"

#define kParamAnalyzeFrameLuma "analyzeFrameLuma"
#define kParamAnalyzeFrameLumaLabel "Analyze Frame Luma"
#define kParamAnalyzeFrameLumaHint "Analyze current frame and set MIN/MAX luma values."

#define kParamClearFrameLuma "clearFrameLuma"
#define kParamClearFrameLumaLabel "Reset"
#define kParamClearFrameLumaHint "Clear luma analysis for current frame."

#define kParamLuminanceMath "luminanceMath"
#define kParamLuminanceMathLabel "Luminance Math"
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

#define kParamMaxLumaPix "maxLumaPix"
#define kParamMaxLumaPixLabel "Max Luma Pixel"
#define kParamMaxLumaPixHint "Position of the pixel with the maximum luma value."
#define kParamMaxLumaPixVal "maxLumaPixVal"
#define kParamMaxLumaPixValLabel "Max Luma Pixel Value"
#define kParamMaxLumaPixValHint "RGB value for the pixel with the maximum luma value."

#define kParamMinLumaPix "minLumaPix"
#define kParamMinLumaPixLabel "Min Luma Pixel"
#define kParamMinLumaPixHint "Position of the pixel with the minimum luma value."
#define kParamMinLumaPixVal "minLumaPixVal"
#define kParamMinLumaPixValLabel "Min Luma Pixel Value"
#define kParamMinLumaPixValHint "RGB value for the pixel with the minimum luma value."

#define kParamDefaultsNormalised "defaultsNormalised"

static bool gHostSupportsDefaultCoordinateSystem = true;

#define MEDIAN_LIMIT 100
#define kOfxFlagInfiniteMax INT_MAX
#define kOfxFlagInfiniteMin INT_MIN

struct RGBValues {
double r, g, b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};

static double Luma(double R, double G, double B, int L) {
double lumaRec709 = R * 0.2126 + G * 0.7152 + B * 0.0722;
double lumaRec2020 = R * 0.2627 + G * 0.6780 + B * 0.0593;
double lumaDCIP3 = R * 0.209492 + G * 0.721595 + B * 0.0689131;
double lumaACESAP0 = R * 0.3439664498 + G * 0.7281660966 + B * -0.0721325464;
double lumaACESAP1 = R * 0.2722287168 + G * 0.6740817658 + B * 0.0536895174;
double lumaAvg = (R + G + B) / 3.0;
double lumaMax = fmax(fmax(R, G), B);
double Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : L == 3 ? 
lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax;
return Lu;
} 

static void RGB_to_HSV( double r, double g, double b, double *h, double *s, double *v ) {
double min = fmin(fmin(r, g), b);
double max = fmax(fmax(r, g), b);
*v = max;
double delta = max - min;
if (max != 0.0) {
*s = delta / max;
} else {
*s = 0.0;
*h = 0.0;
return;
}
if (delta == 0.0) {
*h = 0.0;
} else if (r == max) {
*h = (g - b) / delta;
} else if (g == max) {
*h = 2.0 + (b - r) / delta;
} else {
*h = 4.0 + (r - g) / delta;
}
*h *= 1.0 / 6.0;
if (*h < 0.0) {
*h += 1.0;
}}

struct Results {
Results()
: MIN( std::numeric_limits<double>::infinity() )
, MAX( -std::numeric_limits<double>::infinity() )
, mean(0.0)
, median(0.0)
, maxVal( -std::numeric_limits<double>::infinity() )
, minVal( std::numeric_limits<double>::infinity() ) {
maxPos.x = maxPos.y = minPos.x = minPos.y = 0.0;
}
RGBValues MIN;
RGBValues MAX;
RGBValues mean;
RGBValues median;
OfxPointD maxPos;
RGBValues maxVal;
OfxPointD minPos;
RGBValues minVal;
};

class ScanProcessorBase : public ImageProcessor
{
protected:
unsigned long _count;

public:
ScanProcessorBase(ImageEffect &instance) : ImageProcessor(instance), _count(0) {}
virtual ~ScanProcessorBase() {}
virtual void setPrevResults(double time, const Results &results) = 0;
virtual void getResults(Results *results) = 0;

protected:
template<class PIX, int nComponents, int maxValue>
void toRGB(const PIX *p, RGBValues* rgb) {
if ( (nComponents == 4) || (nComponents == 3) ) {
double r, g, b;
rgb->r = p[0] / (double)maxValue;
rgb->g = p[1] / (double)maxValue;
rgb->b = p[2] / (double)maxValue;
} else {
rgb->r = 0.0;
rgb->g = 0.0;
rgb->b = 0.0;
}}

template<class PIX, int nComponents, int maxValue>
void pixToHSV(const PIX *p, double hsv[3]) {
if ( (nComponents == 4) || (nComponents == 3) ) {
double r, g, b;
r = p[0] / (double)maxValue;
g = p[1] / (double)maxValue;
b = p[2] / (double)maxValue;
RGB_to_HSV(r, g, b, &hsv[0], &hsv[1], &hsv[2]);
hsv[0] *= 360 / OFXS_HUE_CIRCLE;
double MIN = fmin(fmin(r, g), b);
double MAX = fmax(fmax(r, g), b);
} else {
hsv[0] = hsv[1] = hsv[2] = 0.0f;
}}

template<class PIX, int nComponents, int maxValue>
void toComponents(const RGBValues& rgb, PIX *p) {
if (nComponents == 4) {
p[0] = rgb.r;
p[1] = rgb.g;
p[2] = rgb.b;
} else if (nComponents == 3) {
p[0] = rgb.r;
p[1] = rgb.g;
p[2] = rgb.b;
}}};

class Scan : public ImageProcessor
{
public:
explicit Scan(ImageEffect &instance);

virtual void processImagesMetal();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(Image* p_SrcImg);
void setScales(double* p_Gain, double* p_Offset, double* p_Lift, double p_LumaBalance, 
double p_LumaLimit, double p_Blur, int* p_Switch, int p_LumaMath, int* p_LumaMaxXY, int* p_LumaMinXY, int* p_DisplayXY);

private:
Image* _srcImg;
float _gain[2];
float _offset[2];
float _lift[2];
float _lumaBalance;
float _lumaLimit;
float _blur;
int _switch[5];
int _lumaMath;
int _lumaMaxXY[2];
int _lumaMinXY[2];
int _displayXY[2];
};

Scan::Scan(ImageEffect& instance) : ImageProcessor(instance) {}

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Gain, 
float* p_Offset, float* p_Lift, float p_LumaBalance, float p_LumaLimit, float p_Blur, int* p_Switch, int p_LumaMath, 
int* p_LumaMaxXY, int* p_LumaMinXY, int* p_DisplayXY, int Radius);
#endif

void Scan::processImagesMetal()
{
#ifdef __APPLE__
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

int _radius = (int)(15.0f * kResolutionScale);
_blur *= kResolutionScale;

RunMetalKernel(_pMetalCmdQ, input, output, width, height, _gain, _offset, _lift, _lumaBalance, 
_lumaLimit, _blur, _switch, _lumaMath, _lumaMaxXY, _lumaMinXY, _displayXY, _radius);
#endif
}

void Scan::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
if (_effect.abort()) break;
double* dstPix = static_cast<double*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
double* srcPix = static_cast<double*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
if (srcPix){
double luma = Luma(srcPix[0], srcPix[1], srcPix[2], _lumaMath);
double alpha = _lumaLimit > 1.0f ? luma + (1.0f - _lumaLimit) * (1.0f - luma) : _lumaLimit >= 0.0f ? (luma >= _lumaLimit ? 
1.0f : luma / _lumaLimit) : _lumaLimit < -1.0f ? (1.0f - luma) + (_lumaLimit + 1.0f) * luma : luma <= (1.0f + _lumaLimit) ? 1.0f : 
(1.0f - luma) / (1.0f - (_lumaLimit + 1.0f));
double Alpha = alpha > 1.0f ? 1.0f : alpha < 0.0f ? 0.0f : alpha;

double BalR = _switch[0] == 1 ? srcPix[0] * _gain[0] : _switch[1] == 1 ? srcPix[0] + _offset[0] : srcPix[0] + (_lift[0] * (1.0 - srcPix[0]));
double BalB = _switch[0] == 1 ? srcPix[2] * _gain[1] : _switch[1] == 1 ? srcPix[2] + _offset[1] : srcPix[2] + (_lift[1] * (1.0 - srcPix[2]));
double Red = _switch[2]  = 1 ? ( _switch[3] == 1 ? BalR * _lumaBalance : BalR) : srcPix[0];
double Green = _switch[3] == 1 ? srcPix[1] * _lumaBalance : srcPix[1]; 
double Blue = _switch[2] == 1 ? ( _switch[3] == 1 ? BalB * _lumaBalance : BalB) : srcPix[2];

dstPix[0] = _switch[4] == 1 ? Alpha : Red * Alpha + srcPix[0] * (1.0 - Alpha);
dstPix[1] = _switch[4] == 1 ? Alpha : Green * Alpha + srcPix[1] * (1.0 - Alpha);
dstPix[2] = _switch[4] == 1 ? Alpha : Blue * Alpha + srcPix[2] * (1.0 - Alpha);
dstPix[3] = _switch[4] == 1 ? srcPix[3] : Alpha;
} else {
for (int c = 0; c < 4; ++c) {
dstPix[c] = 0;
}}
dstPix += 4;
}}}

void Scan::setSrcImg(Image* p_SrcImg) {
_srcImg = p_SrcImg;
}

void Scan::setScales( double* p_Gain, double* p_Offset, double* p_Lift, double p_LumaBalance, 
double p_LumaLimit, double p_Blur, int* p_Switch, int p_LumaMath, int* p_LumaMaxXY, int* p_LumaMinXY, int* p_DisplayXY) {
_gain[0] = p_Gain[0];
_gain[1] = p_Gain[1];
_offset[0] = p_Offset[0];
_offset[1] = p_Offset[1];
_lift[0] = p_Lift[0];
_lift[1] = p_Lift[1];
_lumaBalance = p_LumaBalance;
_lumaLimit = p_LumaLimit;
_switch[0] = p_Switch[0];
_switch[1] = p_Switch[1];
_switch[2] = p_Switch[2];
_switch[3] = p_Switch[3];
_switch[4] = p_Switch[4];
_lumaMath = p_LumaMath;
_blur = p_Blur;
_lumaMaxXY[0] = p_LumaMaxXY[0];
_lumaMaxXY[1] = p_LumaMaxXY[1];
_lumaMinXY[0] = p_LumaMinXY[0];
_lumaMinXY[1] = p_LumaMinXY[1];
_displayXY[0] = p_DisplayXY[0];
_displayXY[1] = p_DisplayXY[1];
}

template <class PIX, int nComponents, int maxValue>
class ImageMinMaxMeanProcessor : public ScanProcessorBase
{
private:
double _MIN[nComponents];
double _MAX[nComponents];
double _sum[nComponents];

public:
ImageMinMaxMeanProcessor(ImageEffect &instance) : ScanProcessorBase(instance)
{
std::fill( _MIN, _MIN + nComponents, +std::numeric_limits<double>::infinity() );
std::fill( _MAX, _MAX + nComponents, -std::numeric_limits<double>::infinity() );
std::fill(_sum, _sum + nComponents, 0.0);
}

~ImageMinMaxMeanProcessor() {}

void setPrevResults(double , const Results & ) OVERRIDE FINAL {}

void getResults(Results *results) OVERRIDE FINAL
{
if (_count > 0) {
toRGB<double, nComponents, 1>(_MIN, &results->MIN);
toRGB<double, nComponents, 1>(_MAX, &results->MAX);
double mean[nComponents];
for (int c = 0; c < nComponents; ++c) {
mean[c] = _sum[c] / _count;
}
toRGB<double, nComponents, 1>(mean, &results->mean);
}}

private:

void addResults(double MIN[nComponents], double MAX[nComponents], double sum[nComponents], unsigned long count)
{
for (int c = 0; c < nComponents; ++c) {
_MIN[c] = (fmin(_MIN[c], MIN[c])) / maxValue;
_MAX[c] = (fmax(_MAX[c], MAX[c])) / maxValue;
_sum[c] += sum[c] / maxValue;
}
_count += count;
}

void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
{
double MIN[nComponents], MAX[nComponents], sum[nComponents];
std::fill( MIN, MIN + nComponents, +std::numeric_limits<double>::infinity() );
std::fill( MAX, MAX + nComponents, -std::numeric_limits<double>::infinity() );
std::fill(sum, sum + nComponents, 0.0);
unsigned long count = 0;
assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
_dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
for (int y = procWindow.y1; y < procWindow.y2; ++y) {
if ( _effect.abort() ) {
break;
}

PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);
double sumLine[nComponents];
std::fill(sumLine, sumLine + nComponents, 0.0);
for (int x = procWindow.x1; x < procWindow.x2; ++x) {
for (int c = 0; c < nComponents; ++c) {
double v = *dstPix;
MIN[c] = fmin(MIN[c], v);
MAX[c] = fmax(MAX[c], v);
sumLine[c] += v;
++dstPix;
}}
for (int c = 0; c < nComponents; ++c) {
sum[c] += sumLine[c];
}
count += procWindow.x2 - procWindow.x1;
}
addResults(MIN, MAX, sum, count);
}};

#define nComponentsM 3

template <class PIX, int nComponents, int maxValue>
class ImageMedianProcessor : public ScanProcessorBase
{
private:
double _median[nComponentsM];

public:
ImageMedianProcessor(ImageEffect &instance)
: ScanProcessorBase(instance)
{
std::fill(_median, _median + nComponentsM, 0.0);
}

~ImageMedianProcessor()
{
}

void setPrevResults(double , const Results &results) OVERRIDE FINAL {}
	
void getResults(Results *results) OVERRIDE FINAL
{
if (_count > 0) {
toRGB<double, nComponentsM, 1>(_median, &results->median);  
}}

private:

double MEDIAN(std::vector <double> p_Table, int m) {
double temp, val;
int i, j;
for(i = 0; i < m - 1; i++) {
for(j = i + 1; j < m; j++) {
if(p_Table[j] < p_Table[i]) {
temp = p_Table[i];
p_Table[i] = p_Table[j];
p_Table[j] = temp;
}}}
val = m % 2 != 0 ? (p_Table[(m - 1) / 2] + p_Table[(m + 1) / 2]) / 2 : p_Table[m / 2];
return val;
}

void addResults(std::vector <double> red, std::vector <double> green, std::vector <double> blue, unsigned long count)
{
_count += count;
_median[0] = MEDIAN(red, _count) / (double)maxValue;
_median[1] = MEDIAN(green, _count) / (double)maxValue;
_median[2] = MEDIAN(blue, _count) / (double)maxValue;
}

void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
{
std::vector <double> red, green, blue;
unsigned long count = 0;

assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
_dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
for (int y = procWindow.y1; y < procWindow.y2; ++y) {
if ( _effect.abort() ) {
break;
}

PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);
for (int x = procWindow.x1; x < procWindow.x2; ++x) {

red.push_back(dstPix[0]);
green.push_back(dstPix[1]);
blue.push_back(dstPix[2]);
dstPix += nComponents;
}
count += procWindow.x2 - procWindow.x1;
}
addResults(red, green, blue, count);
}};

#define nComponentsHSV 3

template <class PIX, int nComponents, int maxValue>
class ImageHSVMinMaxMeanProcessor : public ScanProcessorBase
{
private:
double _MIN[nComponentsHSV];
double _MAX[nComponentsHSV];
double _sum[nComponentsHSV];

public:
ImageHSVMinMaxMeanProcessor(ImageEffect &instance) : ScanProcessorBase(instance)
{
std::fill( _MIN, _MIN + nComponentsHSV, +std::numeric_limits<double>::infinity() );
std::fill( _MAX, _MAX + nComponentsHSV, -std::numeric_limits<double>::infinity() );
std::fill(_sum, _sum + nComponentsHSV, 0.0);
}
~ImageHSVMinMaxMeanProcessor()
{
}

void setPrevResults(double , const Results & ) OVERRIDE FINAL {}

void getResults(Results *results) OVERRIDE FINAL
{
if (_count > 0) {
toRGB<double, nComponentsHSV, 1>(_MIN, &results->MIN);
toRGB<double, nComponentsHSV, 1>(_MAX, &results->MAX);
double mean[nComponentsHSV];
for (int c = 0; c < nComponentsHSV; ++c) {
mean[c] = _sum[c] / _count;
}
toRGB<double, nComponentsHSV, 1>(mean, &results->mean);
}}

private:

void addResults(double MIN[nComponentsHSV],
double MAX[nComponentsHSV],
double sum[nComponentsHSV],
unsigned long count)
{
for (int c = 0; c < nComponentsHSV - 1; ++c) {
_MIN[c] = (fmin(_MIN[c], MIN[c]));
_MAX[c] = (fmax(_MAX[c], MAX[c]));
_sum[c] += sum[c];
}
_MIN[2] = (fmin(_MIN[2], MIN[2])) / maxValue;
_MAX[2] = (fmax(_MAX[2], MAX[2])) / maxValue;
_sum[2] += sum[2] / maxValue;
_count += count;
}

void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
{
double MIN[nComponentsHSV], MAX[nComponentsHSV], sum[nComponentsHSV];
std::fill( MIN, MIN + nComponentsHSV, +std::numeric_limits<double>::infinity() );
std::fill( MAX, MAX + nComponentsHSV, -std::numeric_limits<double>::infinity() );
std::fill(sum, sum + nComponentsHSV, 0.0);
unsigned long count = 0;

assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
_dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
for (int y = procWindow.y1; y < procWindow.y2; ++y) {
if ( _effect.abort() ) {
break;
}

PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);
double sumLine[nComponentsHSV];
std::fill(sumLine, sumLine + nComponentsHSV, 0.0);

for (int x = procWindow.x1; x < procWindow.x2; ++x) {
double hsv[nComponentsHSV];
pixToHSV<PIX, nComponents, maxValue>(dstPix, hsv);
for (int c = 0; c < nComponentsHSV; ++c) {
double v = hsv[c];
MIN[c] = fmin(MIN[c], v);
MAX[c] = fmax(MAX[c], v);
sumLine[c] += v;
}
dstPix += nComponents;
}
for (int c = 0; c < nComponentsHSV; ++c) {
sum[c] += sumLine[c];
}
count += procWindow.x2 - procWindow.x1;
}
addResults(MIN, MAX, sum, count);
}};

template <class PIX, int nComponents, int maxValue>
class ImageLumaProcessor : public ScanProcessorBase
{
private:
OfxPointD _maxPos;
double _maxVal[nComponents];
double _maxLuma;
OfxPointD _minPos;
double _minVal[nComponents];
double _minLuma;
LuminanceMathEnum _luminanceMath;

public:
ImageLumaProcessor(ImageEffect &instance)
: ScanProcessorBase(instance)
, _luminanceMath(eLuminanceMathRec709)
{
_maxPos.x = _maxPos.y = 0.0;
std::fill( _maxVal, _maxVal + nComponents, -std::numeric_limits<double>::infinity() );
_maxLuma = -std::numeric_limits<double>::infinity();
_minPos.x = _minPos.y = 0.0;
std::fill( _minVal, _minVal + nComponents, +std::numeric_limits<double>::infinity() );
_minLuma = +std::numeric_limits<double>::infinity();
}
ImageLumaProcessor()
{
}

void setPrevResults(double time, const Results & ) OVERRIDE FINAL
{
ChoiceParam* luminanceMath = _effect.fetchChoiceParam(kParamLuminanceMath);
assert(luminanceMath);
int luma;
luminanceMath->getValueAtTime(time, luma);
_luminanceMath = (LuminanceMathEnum)luma;
}

void getResults(Results *results) OVERRIDE FINAL
{
results->maxPos = _maxPos;
toRGB<double, nComponents, 1>(_maxVal, &results->maxVal);
results->minPos = _minPos;
toRGB<double, nComponents, 1>(_minVal, &results->minVal);
}

private:

double luminance (const PIX *p) {
if (nComponents == 4 || nComponents == 3) {
double R, G, B;
R = p[0] / (double)maxValue;
G = p[1] / (double)maxValue;
B = p[2] / (double)maxValue;
int luma = _luminanceMath;
return Luma(R, G, B, luma);
}
return 0.0;
}

void addResults(const OfxPointD& maxPos, double maxVal[nComponents], double maxLuma, const OfxPointD& minPos, double minVal[nComponents], double minLuma)
{
if (maxLuma > _maxLuma) {
_maxPos = maxPos;
for (int c = 0; c < nComponents; ++c) {
_maxVal[c] = maxVal[c];
}
_maxLuma = maxLuma;
}
if (minLuma < _minLuma) {
_minPos = minPos;
for (int c = 0; c < nComponents; ++c) {
_minVal[c] = minVal[c];
}
_minLuma = minLuma;
}}

void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
{
OfxPointD maxPos = {0.0, 0.0};
double maxVal[nComponents] = {0.0};
double maxLuma = -std::numeric_limits<double>::infinity();
OfxPointD minPos = {0.0, 0.0};
double minVal[nComponents] = {0.0};
double minLuma = +std::numeric_limits<double>::infinity();

assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
_dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
for (int y = procWindow.y1; y < procWindow.y2; ++y) {
if ( _effect.abort() ) {
break;
}

PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);

for (int x = procWindow.x1; x < procWindow.x2; ++x) {
double luma = luminance(dstPix);
if (luma > maxLuma) {
maxPos.x = x;
maxPos.y = y;
for (int c = 0; c < nComponents; ++c) {
maxVal[c] = dstPix[c] / (double)maxValue;
}
maxLuma = luma;
}
if (luma < minLuma) {
minPos.x = x;
minPos.y = y;
for (int c = 0; c < nComponents; ++c) {
minVal[c] = dstPix[c] / (double)maxValue;
}
minLuma = luma;
}
dstPix += nComponents;
}}
addResults(maxPos, maxVal, maxLuma, minPos, minVal, minLuma);
}};

////////////////////////////////////////////////////////////////////////////////

class ScanPlugin : public ImageEffect
{
public:

ScanPlugin(OfxImageEffectHandle handle)
: ImageEffect(handle)
, _dstClip(0)
, _srcClip(0)
, _btmLeft(0)
, _size(0)
, _restrictToRectangle(0)
{
_dstClip = fetchClip(kOfxImageEffectOutputClipName);
assert( _dstClip && (!_dstClip->isConnected() || _dstClip->getPixelComponents() == ePixelComponentAlpha ||
_dstClip->getPixelComponents() == ePixelComponentRGB || _dstClip->getPixelComponents() == ePixelComponentRGBA) );
_srcClip = getContext() == eContextGenerator ? NULL : fetchClip(kOfxImageEffectSimpleSourceClipName);
assert( (!_srcClip && getContext() == eContextGenerator) || ( _srcClip && (!_srcClip->isConnected() || _srcClip->getPixelComponents() ==  ePixelComponentAlpha ||
_srcClip->getPixelComponents() == ePixelComponentRGB || _srcClip->getPixelComponents() == ePixelComponentRGBA) ) );
		   
m_Balance = fetchRGBParam("balance");
m_Rgb = fetchDouble3DParam("rgbVal");
m_Hsl = fetchDouble3DParam("hslVal");
m_White = fetchBooleanParam("whiteBalance");
_sampleType = fetchChoiceParam(kParamSampleType);
_balanceType = fetchChoiceParam(kParamBalanceType);
_preserveLuma = fetchBooleanParam("preserveLuma");
_lumaLimiter = fetchDoubleParam("lumaLimiter");
_blurAlpha = fetchDoubleParam("blurAlpha");
_displayAlpha = fetchBooleanParam("displayAlpha");
m_Path = fetchStringParam("path");
m_Name = fetchStringParam("name");
m_Button1 = fetchPushButtonParam("button1");
m_displayMax = fetchBooleanParam("displayMax");
m_displayMin = fetchBooleanParam("displayMin");

_btmLeft = fetchDouble2DParam(kParamRectangleInteractBtmLeft);
_size = fetchDouble2DParam(kParamRectangleInteractSize);
_restrictToRectangle = fetchBooleanParam(kParamRestrictToRectangle);
assert(_btmLeft && _size && _restrictToRectangle);
_statMin = fetchDouble3DParam(kParamStatMin);
_statMax = fetchDouble3DParam(kParamStatMax);
_statMean = fetchDouble3DParam(kParamStatMean);
_statMedian = fetchDouble3DParam(kParamStatMedian);
assert(_statMin && _statMax && _statMean && _statMedian);
_analyzeFrame = fetchPushButtonParam(kParamAnalyzeFrame);
assert(_analyzeFrame);
_analyzeFrameMed = fetchPushButtonParam(kParamAnalyzeFrameMed);
assert(_analyzeFrameMed);
_statHSVMin = fetchDouble3DParam(kParamStatHSVMin);
_statHSVMax = fetchDouble3DParam(kParamStatHSVMax);
_statHSVMean = fetchDouble3DParam(kParamStatHSVMean);
assert(_statHSVMin && _statHSVMax && _statHSVMean);
_analyzeFrameHSV = fetchPushButtonParam(kParamAnalyzeFrameHSV);
assert(_analyzeFrameHSV);
_luminanceMath = fetchChoiceParam(kParamLuminanceMath);
_maxLumaPix = fetchDouble2DParam(kParamMaxLumaPix);
_maxLumaPixVal = fetchDouble3DParam(kParamMaxLumaPixVal);
_minLumaPix = fetchDouble2DParam(kParamMinLumaPix);
_minLumaPixVal = fetchDouble3DParam(kParamMinLumaPixVal);
assert(_luminanceMath && _maxLumaPix && _maxLumaPixVal && _minLumaPix && _minLumaPixVal);

bool restrictToRectangle = _restrictToRectangle->getValue();
_btmLeft->setIsSecretAndDisabled(!restrictToRectangle);
_size->setIsSecretAndDisabled(!restrictToRectangle);
bool WhiteBalance = m_White->getValue();
_sampleType->setIsSecretAndDisabled(!WhiteBalance);
_balanceType->setIsSecretAndDisabled(!WhiteBalance);
_preserveLuma->setIsSecretAndDisabled(!WhiteBalance);

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

virtual bool isIdentity(const IsIdentityArguments &args, Clip * &identityClip, double &identityTime, int& view, std::string& plane);
virtual void render(const RenderArguments &args) OVERRIDE FINAL;
virtual void getRegionsOfInterest(const RegionsOfInterestArguments &args, RegionOfInterestSetter &rois) OVERRIDE FINAL;
virtual bool getRegionOfDefinition(const RegionOfDefinitionArguments &args, OfxRectD & rod) OVERRIDE FINAL;
virtual void changedParam(const InstanceChangedArgs &args, const std::string &paramName) OVERRIDE FINAL;

void setupAndProcess(Scan &, const RenderArguments &args);
void YoProcess(ScanProcessorBase &processor, const Image* srcImg, double time, const OfxRectI &analysisWindow, 
const Results &prevResults, Results *results);

bool computeWindow(const Image* srcImg, double time, OfxRectI *analysisWindow);

void update(const Image* srcImg, double time, const OfxRectI& analysisWindow);
void updateMed(const Image* srcImg, double time, const OfxRectI& analysisWindow);
void updateHSV(const Image* srcImg, double time, const OfxRectI& analysisWindow);
void updateLuma(const Image* srcImg, double time, const OfxRectI& analysisWindow);

template <template<class PIX, int nComponents, int maxValue> class Processor, class PIX, int nComponents, int maxValue>
void updateSubComponentsDepth(const Image* srcImg, double time, const OfxRectI &analysisWindow, const Results& prevResults, Results* results)
{
Processor<PIX, nComponents, maxValue> fred(*this);
YoProcess(fred, srcImg, time, analysisWindow, prevResults, results);
}

template <template<class PIX, int nComponents, int maxValue> class Processor, int nComponents>
void updateSubComponents(const Image* srcImg, double time, const OfxRectI &analysisWindow, const Results& prevResults, Results* results)
{
BitDepthEnum srcBitDepth = srcImg->getPixelDepth();

switch (srcBitDepth) {
case eBitDepthUByte: {
updateSubComponentsDepth<Processor, unsigned char, nComponents, 255>(srcImg, time, analysisWindow, prevResults, results);
break;
}
case eBitDepthUShort: {
updateSubComponentsDepth<Processor, unsigned short, nComponents, 65535>(srcImg, time, analysisWindow, prevResults, results);
break;
}
case eBitDepthFloat: {
updateSubComponentsDepth<Processor, float, nComponents, 1>(srcImg, time, analysisWindow, prevResults, results);
break;
}
default:
throwSuiteStatusException(kOfxStatErrUnsupported);
}}

template <template<class PIX, int nComponents, int maxValue> class Processor>
void updateSub(const Image* srcImg, double time, const OfxRectI &analysisWindow, const Results& prevResults, Results* results)
{
PixelComponentEnum srcComponents  = srcImg->getPixelComponents();
assert(srcComponents == ePixelComponentAlpha || srcComponents == ePixelComponentRGB || srcComponents == ePixelComponentRGBA);
if (srcComponents == ePixelComponentAlpha) {
updateSubComponents<Processor, 1>(srcImg, time, analysisWindow, prevResults, results);
} else if (srcComponents == ePixelComponentRGBA) {
updateSubComponents<Processor, 4>(srcImg, time, analysisWindow, prevResults, results);
} else if (srcComponents == ePixelComponentRGB) {
updateSubComponents<Processor, 3>(srcImg, time, analysisWindow, prevResults, results);
} else {
throwSuiteStatusException(kOfxStatErrUnsupported);
}}

private:

Clip *_dstClip;
Clip *_srcClip;
Double2DParam* _btmLeft;
Double2DParam* _size;
ChoiceParam* _sampleType;
ChoiceParam* _balanceType;
BooleanParam* _preserveLuma;
BooleanParam* _displayAlpha;
DoubleParam* _lumaLimiter;
DoubleParam* _blurAlpha;
BooleanParam* _restrictToRectangle;   
Double3DParam* _statMin;
Double3DParam* _statMax;
Double3DParam* _statMean;
Double3DParam* _statMedian;
PushButtonParam* _analyzeFrame;
PushButtonParam* _analyzeFrameMed;
Double3DParam* _statHSVMin;
Double3DParam* _statHSVMax;
Double3DParam* _statHSVMean;
PushButtonParam* _analyzeFrameHSV;
ChoiceParam* _luminanceMath;
Double2DParam* _maxLumaPix;
Double3DParam* _maxLumaPixVal;
Double2DParam* _minLumaPix;
Double3DParam* _minLumaPixVal;
BooleanParam* m_displayMax;
BooleanParam* m_displayMin;

StringParam* m_Path;
StringParam* m_Name;
PushButtonParam* m_Button1;

RGBParam *m_Balance;
Double3DParam* m_Rgb;
Double3DParam* m_Hsl;
BooleanParam* m_White;
};

////////////////////////////////////////////////////////////////////////////////

void ScanPlugin::setupAndProcess(Scan& p_Scan, const RenderArguments &args)
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
if ( (srcBitDepth != dstBitDepth) ||  (srcComponents != dstComponents) )  {
OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
}}

int sampleType;
_sampleType->getValueAtTime(args.time, sampleType);
int balanceType;
_balanceType->getValueAtTime(args.time, balanceType);
int _lumaMath;
_luminanceMath->getValueAtTime(args.time, _lumaMath);

RGBValues colorSample;
m_Balance->getValueAtTime(args.time, colorSample.r, colorSample.g, colorSample.b);

RGBValues meanSample;
_statMean->getValueAtTime(args.time, meanSample.r, meanSample.g, meanSample.b);

RGBValues medianSample;
_statMedian->getValueAtTime(args.time, medianSample.r, medianSample.g, medianSample.b);

double BalanceR = sampleType == 0 ? colorSample.r : sampleType == 1 ? meanSample.r : medianSample.r;
double BalanceG = sampleType == 0 ? colorSample.g : sampleType == 1 ? meanSample.g : medianSample.g;
double BalanceB = sampleType == 0 ? colorSample.b : sampleType == 1 ? meanSample.b : medianSample.b;

double rGain = BalanceG / BalanceR;
double bGain = BalanceG / BalanceB;

double rOffset = BalanceG - BalanceR;
double bOffset = BalanceG - BalanceB;

double lumaMathChoice = Luma(BalanceR, BalanceG, BalanceB, _lumaMath);
double _lumaBalance = lumaMathChoice / BalanceG;

double rLift = (BalanceG - BalanceR) / (1.0 - BalanceR);
double bLift = (BalanceG - BalanceB) / (1.0 - BalanceB);

bool preserveLuma = _preserveLuma->getValueAtTime(args.time);
bool displayAlpha = _displayAlpha->getValueAtTime(args.time);

bool whiteBalance = m_White->getValueAtTime(args.time);
int WhiteBalance = whiteBalance ? 1 : 0;
int PreserveLuma = preserveLuma ? 1 : 0;
int DisplayAlpha = displayAlpha ? 1 : 0;

int GainBalance = balanceType == 0 ? 1 : 0;
int OffsetBalance = balanceType == 1 ? 1 : 0;
int LiftBalance = balanceType == 2 ? 1 : 0;

double _lumaLimit = _lumaLimiter->getValueAtTime(args.time);
double _blur = _blurAlpha->getValueAtTime(args.time);

int _displayXY[2];
_displayXY[0] = m_displayMax->getValueAtTime(args.time);
_displayXY[1] = m_displayMin->getValueAtTime(args.time);


OfxPointD maxlumaXY;
OfxPointD minlumaXY;
_maxLumaPix->getValueAtTime(args.time, maxlumaXY.x, maxlumaXY.y);
_minLumaPix->getValueAtTime(args.time, minlumaXY.x, minlumaXY.y);

int _maxlumaXY[2];
int _minlumaXY[2];
_maxlumaXY[0] = (int)maxlumaXY.x;
_maxlumaXY[1] = (int)maxlumaXY.y;
_minlumaXY[0] = (int)minlumaXY.x;
_minlumaXY[1] = (int)minlumaXY.y;

double _gain[2];
double _offset[2];
double _lift[2];
int _switch[5];

_gain[0] = rGain;
_gain[1] = bGain;
_offset[0] = rOffset;
_offset[1] = bOffset;
_lift[0] = rLift;
_lift[1] = bLift;
_switch[0] = GainBalance;
_switch[1] = OffsetBalance;
_switch[2] = WhiteBalance;
_switch[3] = PreserveLuma;
_switch[4] = DisplayAlpha;

p_Scan.setDstImg(dst.get());
p_Scan.setSrcImg(src.get());

// Setup GPU Render arguments
p_Scan.setGPURenderArgs(args);

p_Scan.setRenderWindow(args.renderWindow);

p_Scan.setScales(_gain, _offset, _lift, _lumaBalance, _lumaLimit, 
_blur, _switch, _lumaMath, _maxlumaXY, _minlumaXY, _displayXY);

p_Scan.process();
} 

void ScanPlugin::render(const RenderArguments &args)
{
if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.0) || (args.renderScale.y != 1.0) ) ) {
throwSuiteStatusException(kOfxStatFailed);
}

assert( kSupportsMultipleClipPARs   || !_srcClip || _srcClip->getPixelAspectRatio() == _dstClip->getPixelAspectRatio() );
assert( kSupportsMultipleClipDepths || !_srcClip || _srcClip->getPixelDepth()       == _dstClip->getPixelDepth() );

std::auto_ptr<Image> dst( _dstClip->fetchImage(args.time) );
if ( !dst.get() ) {
throwSuiteStatusException(kOfxStatFailed);
}
if ( (dst->getRenderScale().x != args.renderScale.x) || ( dst->getRenderScale().y != args.renderScale.y) ||
( ( dst->getField() != eFieldNone)  && ( dst->getField() != args.fieldToRender) ) ) {
setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
throwSuiteStatusException(kOfxStatFailed);
}
BitDepthEnum dstBitDepth = dst->getPixelDepth();
PixelComponentEnum dstComponents = dst->getPixelComponents();
std::auto_ptr<const Image> src( ( _srcClip && _srcClip->isConnected() ) ? _srcClip->fetchImage(args.time) : 0 );
if ( src.get() ) {
if ( (src->getRenderScale().x != args.renderScale.x) || ( src->getRenderScale().y != args.renderScale.y) ||
( ( src->getField() != eFieldNone)  && ( src->getField() != args.fieldToRender) ) ) {
setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
throwSuiteStatusException(kOfxStatFailed);
}
BitDepthEnum srcBitDepth = src->getPixelDepth();
PixelComponentEnum srcComponents = src->getPixelComponents();
if ( (srcBitDepth != dstBitDepth) || (srcComponents != dstComponents) ) {
throwSuiteStatusException(kOfxStatErrImageFormat);
}}

if (dstComponents == ePixelComponentRGBA || ePixelComponentRGB) {
Scan fred(*this);
setupAndProcess(fred, args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

void ScanPlugin::getRegionsOfInterest(const RegionsOfInterestArguments &args, RegionOfInterestSetter &rois)
{
bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);
if (restrictToRectangle) {
OfxRectD regionOfInterest;
_btmLeft->getValueAtTime(args.time, regionOfInterest.x1, regionOfInterest.y1);
_size->getValueAtTime(args.time, regionOfInterest.x2, regionOfInterest.y2);
regionOfInterest.x2 += regionOfInterest.x1;
regionOfInterest.y2 += regionOfInterest.y1;
Coords::rectBoundingBox(args.regionOfInterest, regionOfInterest, &regionOfInterest);
rois.setRegionOfInterest(*_srcClip, regionOfInterest);
}}

bool ScanPlugin::getRegionOfDefinition(const RegionOfDefinitionArguments &args, OfxRectD & )
{
if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.0) || (args.renderScale.y != 1.0) ) ) {
throwSuiteStatusException(kOfxStatFailed);
}
return false;
}

bool ScanPlugin::isIdentity(const IsIdentityArguments &args, Clip * &identityClip, double & identityTime, int& , std::string& )
{
RGBValues colorSample;
m_Balance->getValueAtTime(args.time, colorSample.r, colorSample.g, colorSample.b);

double rBalance = colorSample.r;
double gBalance = colorSample.g;
double bBalance = colorSample.b;

if (rBalance == 1.0 && gBalance == 1.0 && bBalance == 1.0) {
identityClip = _srcClip;
identityTime = args.time;
return true;
}
return false;
}

void ScanPlugin::changedParam(const InstanceChangedArgs &args, const std::string &paramName)
{
if(paramName == "info") {
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.0) || (args.renderScale.y != 1.0) ) ) {
throwSuiteStatusException(kOfxStatFailed);
}

bool doAnalyzeRGB = false;
bool doAnalyzeMed = false;
bool doAnalyzeHSV = false;
bool doAnalyzeLuma = false;
OfxRectI analysisWindow;
const double time = args.time;

if (paramName == "whiteBalance") {
bool WhiteBalance = m_White->getValueAtTime(time);
_sampleType->setIsSecretAndDisabled(!WhiteBalance);
_balanceType->setIsSecretAndDisabled(!WhiteBalance);
_preserveLuma->setIsSecretAndDisabled(!WhiteBalance);
}

if (paramName == kParamRestrictToRectangle) {
bool restrictToRectangle = _restrictToRectangle->getValueAtTime(time);
_btmLeft->setIsSecretAndDisabled(!restrictToRectangle);
_size->setIsSecretAndDisabled(!restrictToRectangle);
}
if (paramName == kParamAnalyzeFrame) {
doAnalyzeRGB = true;
}
if (paramName == kParamAnalyzeFrameMed) {
double medx, medy, medLx, medLy;
int medX, medY, medLX, medLY;
int MedSq = MEDIAN_LIMIT * MEDIAN_LIMIT;
int MedPlus = MEDIAN_LIMIT * 2;
beginEditBlock("analyzeFrameMed");
_restrictToRectangle->setValue(true);
_size->getValueAtTime(time, medx, medy);
_btmLeft->getValueAtTime(time, medLx, medLy);
if (medx * medy > MedSq){
medX = medx / (medx + medy) * MedPlus;
medY = medy / (medx + medy) * MedPlus;
medLX = medLx + (medx - medX) / 2;
medLY = medLy + (medy - medY) / 2;
_size->setValue(medX, medY);
_btmLeft->setValue(medLX, medLY);
}
endEditBlock();
doAnalyzeMed = true;
}
if (paramName == kParamAnalyzeFrameHSV) {
doAnalyzeHSV = true;
}
if (paramName == kParamAnalyzeFrameLuma) {
doAnalyzeLuma = true;
}
if (paramName == kParamClearFrame) {
_statMin->setValue(0.0, 0.0, 0.0);
_statMax->setValue(1.0, 1.0, 1.0);
_statMean->setValue(1.0, 1.0, 1.0);
_statMedian->setValue(1.0, 1.0, 1.0);
}
if (paramName == kParamClearFrameHSV) {
_statHSVMin->setValue(0.0, 0.0, 0.0);
_statHSVMax->setValue(1.0, 1.0, 1.0);
_statHSVMean->setValue(1.0, 1.0, 1.0);
}
if (paramName == kParamClearFrameLuma) {
_maxLumaPix->setValue(0, 0);
_maxLumaPixVal->setValue(0.0, 0.0, 0.0);
_minLumaPix->setValue(0, 0);
_minLumaPixVal->setValue(1.0, 1.0, 1.0);
}

if ( (doAnalyzeRGB || doAnalyzeMed || doAnalyzeHSV || doAnalyzeLuma) && _srcClip && _srcClip->isConnected() ) {
std::auto_ptr<Image> src( ( _srcClip && _srcClip->isConnected() ) ? _srcClip->fetchImage(args.time) : 0 );
if ( src.get() ) {
if ( (src->getRenderScale().x != args.renderScale.x) ||
( src->getRenderScale().y != args.renderScale.y) ) {
setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
throwSuiteStatusException(kOfxStatFailed);
}
bool intersect = computeWindow(src.get(), args.time, &analysisWindow);
if (intersect) {
#ifdef kOfxImageEffectPropInAnalysis
getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 1, false);
#endif
beginEditBlock("analyzeFrame");
if (doAnalyzeRGB) {
update(src.get(), args.time, analysisWindow);
}
if (doAnalyzeMed) {
updateMed(src.get(), args.time, analysisWindow);
}
if (doAnalyzeHSV) {
updateHSV(src.get(), args.time, analysisWindow);
}
if (doAnalyzeLuma) {
updateLuma(src.get(), args.time, analysisWindow);
}
endEditBlock();
#ifdef kOfxImageEffectPropInAnalysis
getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 0, false);
#endif
}}}

#ifdef kOfxImageEffectPropInAnalysis
getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 0, false);
#endif

if (paramName == "balance") {
RGBValues colorSample;
m_Balance->getValueAtTime(args.time, colorSample.r, colorSample.g, colorSample.b);
m_Rgb->setValue(colorSample.r, colorSample.g, colorSample.b);

double Min = fmin(colorSample.r, fmin(colorSample.g, colorSample.b));    
double Max = fmax(colorSample.r, fmax(colorSample.g, colorSample.b));    
double del_Max = Max - Min;

double L = (Max + Min) / 2.0;
double S = del_Max == 0.0 ? 0.0 : (L < 0.5f ? del_Max / (Max + Min) : del_Max / (2.0 - Max - Min));

double del_R = (((Max - colorSample.r) / 6.0) + (del_Max / 2.0)) / del_Max;
double del_G = (((Max - colorSample.g) / 6.0) + (del_Max / 2.0)) / del_Max;
double del_B = (((Max - colorSample.b) / 6.0) + (del_Max / 2.0)) / del_Max;

double h = del_Max == 0.0 ? 0.0 : (colorSample.r == Max ? del_B - del_G : (colorSample.g == Max ? (1.0 / 3.0) + del_R - del_B : (2.0 / 3.0) + del_G - del_R));
double H = h < 0.0 ? h + 1.0 : (h > 1.0 ? h - 1.0 : h);

m_Hsl->setValue(H, S, L);
}

if(paramName == "button1") {

int sampleType_i;
_sampleType->getValueAtTime(args.time, sampleType_i);
SampleTypeEnum sampleType = (SampleTypeEnum)sampleType_i;

bool ColorSample = sampleType_i == 0;
bool MeanSample = sampleType_i == 1;
bool MedianSample = sampleType_i == 2;

int balanceType_i;
_balanceType->getValueAtTime(args.time, balanceType_i);
BalanceTypeEnum balanceType = (BalanceTypeEnum)balanceType_i;

int luminanceMath_i;
_luminanceMath->getValueAtTime(args.time, luminanceMath_i);
LuminanceMathEnum luminanceMath = (LuminanceMathEnum)luminanceMath_i;

bool Rec709LuminanceMath = luminanceMath_i == 0;
bool Rec2020LuminanceMath = luminanceMath_i == 1;
bool DCIP3LuminanceMath = luminanceMath_i == 2;
bool ACESAP0LuminanceMath = luminanceMath_i == 3;
bool ACESAP1LuminanceMath = luminanceMath_i == 4;
bool AvgLuminanceMath = luminanceMath_i == 5;
bool MaxLuminanceMath = luminanceMath_i == 6;

RGBValues colorSample;
m_Balance->getValueAtTime(args.time, colorSample.r, colorSample.g, colorSample.b);

RGBValues meanSample;
_statMean->getValueAtTime(args.time, meanSample.r, meanSample.g, meanSample.b);

RGBValues medianSample;
_statMedian->getValueAtTime(args.time, medianSample.r, medianSample.g, medianSample.b);

double BalanceR = ColorSample ? colorSample.r : MeanSample ? meanSample.r : medianSample.r;
double BalanceG = ColorSample ? colorSample.g : MeanSample ? meanSample.g : medianSample.g;
double BalanceB = ColorSample ? colorSample.b : MeanSample ? meanSample.b : medianSample.b;

double rGain = BalanceG/BalanceR;
double bGain = BalanceG/BalanceB;

double rOffset = BalanceG - BalanceR;
double bOffset = BalanceG - BalanceB;

double lumaRec709 = BalanceR * 0.2126 + BalanceG * 0.7152 + BalanceB * 0.0722;
double lumaRec2020 = BalanceR * 0.2627 + BalanceG * 0.6780 + BalanceB * 0.0593;
double lumaDCIP3 = BalanceR * 0.209492 + BalanceG * 0.721595 + BalanceB * 0.0689131;
double lumaACESAP0 = BalanceR * 0.3439664498 + BalanceG * 0.7281660966 + BalanceB * -0.0721325464;
double lumaACESAP1 = BalanceR * 0.2722287168 + BalanceG * 0.6740817658 + BalanceB * 0.0536895174;
double lumaAvg = (BalanceR + BalanceG + BalanceB) / 3.0;
double lumaMax = fmax(fmax(BalanceR, BalanceG), BalanceB);
double lumaMathChoice = Rec709LuminanceMath ? lumaRec709 : Rec2020LuminanceMath ? lumaRec2020 : DCIP3LuminanceMath ? lumaDCIP3 : 
ACESAP0LuminanceMath ? lumaACESAP0 : ACESAP1LuminanceMath ? lumaACESAP1 : AvgLuminanceMath ? lumaAvg : lumaMax;
double lumaMath = lumaMathChoice / BalanceG;

double rLift = (BalanceG - BalanceR) / (1.0 - BalanceR);
double bLift = (BalanceG - BalanceB) / (1.0 - BalanceB);

bool preserveLuma = _preserveLuma->getValueAtTime(args.time);
bool displayAlpha = _displayAlpha->getValueAtTime(args.time);

float lumaLimit = _lumaLimiter->getValueAtTime(args.time);

bool whiteBalance = m_White->getValueAtTime(args.time);
int WhiteBalance = whiteBalance ? 1 : 0;
int PreserveLuma = preserveLuma ? 1 : 0;
int DisplayAlpha = displayAlpha ? 1 : 0;

bool gainBalance = balanceType_i == 0;
bool offsetBalance = balanceType_i == 1;
bool liftBalance = balanceType_i == 2;

int GainBalance = gainBalance ? 1 : 0;
int OffsetBalance = offsetBalance ? 1 : 0;
int LiftBalance = liftBalance ? 1 : 0;

int LumaRec709 = Rec709LuminanceMath ? 1 : 0;
int LumaRec2020 = Rec2020LuminanceMath ? 1 : 0;
int LumaDCIP3 = DCIP3LuminanceMath ? 1 : 0;
int LumaACESAP0 = ACESAP0LuminanceMath ? 1 : 0;
int LumaACESAP1 = ACESAP1LuminanceMath ? 1 : 0;
int LumaAvg = AvgLuminanceMath ? 1 : 0;

string PATH;
m_Path->getValue(PATH);

string NAME;
m_Name->getValue(NAME);

OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
if (reply == OFX::Message::eMessageReplyYes) {

FILE * pFile;

pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
if (pFile != NULL) {

fprintf (pFile, "// ScanPlugin DCTL export\n" \
"\n" \
"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
"{\n" \
"float balGainR = %ff;\n" \
"float balGainB = %ff;\n" \
"float balOffsetR = %ff;\n" \
"float balOffsetB = %ff;\n" \
"float balLiftR = %ff;\n" \
"float balLiftB = %ff;\n" \
"float lumaMath = %ff;\n" \
"float lumaLimit = %ff;\n" \
"int GainBalance = %d;\n" \
"int OffsetBalance = %d;\n" \
"int WhiteBalance = %d;\n" \
"int PreserveLuma = %d;\n" \
"int LumaRec709 = %d;\n" \
"int LumaRec2020 = %d;\n" \
"int LumaDCIP3 = %d;\n" \
"int LumaACESAP0 = %d;\n" \
"int LumaACESAP1 = %d;\n" \
"int LumaAvg = %d;\n" \
"\n" \
"float lumaRec709 = p_R * 0.2126f + p_G * 0.7152f + p_B * 0.0722f;\n" \
"float lumaRec2020 = p_R * 0.2627f + p_G * 0.6780f + p_B * 0.0593f;\n" \
"float lumaDCIP3 = p_R * 0.209492f + p_G * 0.721595f + p_B * 0.0689131f;\n" \
"float lumaACESAP0 = p_R * 0.3439664498f + p_G * 0.7281660966f + p_B * -0.0721325464f;\n" \
"float lumaACESAP1 = p_R * 0.2722287168f + p_G * 0.6740817658f + p_B * 0.0536895174f;\n" \
"float lumaAvg = (p_R + p_G + p_B) / 3.0f;\n" \
"float lumaMax = _fmaxf(_fmaxf(p_R, p_G), p_B);\n" \
"float luma = LumaRec709 == 1 ? lumaRec709 : LumaRec2020 == 1 ? lumaRec2020 : LumaDCIP3 == 1 ? lumaDCIP3 : \n" \
"LumaACESAP0 == 1 ? lumaACESAP0 : LumaACESAP1 == 1 ? lumaACESAP1 : LumaAvg == 1 ? lumaAvg : lumaMax;\n" \
"\n" \
"float alpha = lumaLimit > 1.0f ? luma + (1.0f - lumaLimit) * (1.0f - luma) : lumaLimit >= 0.0f ? (luma >= lumaLimit ? \n" \
"1.0f : luma / lumaLimit) : lumaLimit < -1.0f ? (1.0f - luma) + (lumaLimit + 1.0f) * luma : luma <= (1.0f + lumaLimit) ? 1.0f : \n" \
"(1.0f - luma) / (1.0f - (lumaLimit + 1.0f));\n" \
"float Alpha = alpha > 1.0f ? 1.0f : alpha < 0.0f ? 0.0f : alpha;\n" \
"\n" \
"float BalR = GainBalance == 1 ? p_R * balGainR : OffsetBalance == 1 ? p_R + balOffsetR : p_R + (balLiftR * (1.0f - p_R));\n" \
"float BalB = GainBalance == 1 ? p_B * balGainB : OffsetBalance == 1 ? p_B + balOffsetB : p_B + (balLiftB * (1.0f - p_B));\n" \
"float Red = WhiteBalance == 1 ? ( PreserveLuma == 1 ? BalR * lumaMath : BalR) : p_R;\n" \
"float Green = WhiteBalance == 1 && PreserveLuma == 1 ? p_G * lumaMath : p_G;\n" \
"float Blue = WhiteBalance == 1 ? ( PreserveLuma == 1 ? BalB * lumaMath : BalB) : p_B;\n" \
"\n" \
"float r = Red * Alpha + p_R * (1.0f - Alpha);\n" \
"float g = Green * Alpha + p_G * (1.0f - Alpha);\n" \
"float b = Blue * Alpha + p_B * (1.0f - Alpha);\n" \
"return make_float3(r, g, b);\n" \
"}\n", rGain, bGain, rOffset, bOffset, rLift, bLift, lumaMath, lumaLimit, GainBalance, OffsetBalance, WhiteBalance, PreserveLuma, 
LumaRec709, LumaRec2020, LumaDCIP3, LumaACESAP0, LumaACESAP1, LumaAvg);
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
}}}}

void ScanPlugin::YoProcess(ScanProcessorBase &processor, const Image* srcImg, double time,
const OfxRectI &analysisWindow, const Results &prevResults, Results *results)
{
processor.setDstImg( const_cast<Image*>(srcImg) );

processor.setRenderWindow(analysisWindow);

processor.setPrevResults(time, prevResults);

processor.process();

if ( !abort() ) {
processor.getResults(results);
}}

bool ScanPlugin::computeWindow(const Image* srcImg, double time, OfxRectI *analysisWindow)
{
OfxRectD regionOfInterest;
bool restrictToRectangle = _restrictToRectangle->getValueAtTime(time);

if (!restrictToRectangle && _srcClip) {
regionOfInterest = _srcClip->getRegionOfDefinition(time);
OfxPointD size = getProjectSize();
OfxPointD offset = getProjectOffset();
if (regionOfInterest.x1 <= kOfxFlagInfiniteMin) {
regionOfInterest.x1 = offset.x;
}
if (regionOfInterest.x2 >= kOfxFlagInfiniteMax) {
regionOfInterest.x2 = offset.x + size.x;
}
if (regionOfInterest.y1 <= kOfxFlagInfiniteMin) {
regionOfInterest.y1 = offset.y;
}
if (regionOfInterest.y2 >= kOfxFlagInfiniteMax) {
regionOfInterest.y2 = offset.y + size.y;
}
} else {
_btmLeft->getValueAtTime(time, regionOfInterest.x1, regionOfInterest.y1);
_size->getValueAtTime(time, regionOfInterest.x2, regionOfInterest.y2);
regionOfInterest.x2 += regionOfInterest.x1;
regionOfInterest.y2 += regionOfInterest.y1;
}
Coords::toPixelEnclosing(regionOfInterest, srcImg->getRenderScale(), srcImg->getPixelAspectRatio(), analysisWindow);
return Coords::rectIntersection(*analysisWindow, srcImg->getBounds(), analysisWindow);
}

void ScanPlugin::update(const Image* srcImg, double time, const OfxRectI &analysisWindow)
{
Results results;
if ( !abort() ) {
updateSub<ImageMinMaxMeanProcessor>(srcImg, time, analysisWindow, results, &results);
}
if ( abort() ) {
return;
}
_statMin->setValueAtTime(time, results.MIN.r, results.MIN.g, results.MIN.b);
_statMax->setValueAtTime(time, results.MAX.r, results.MAX.g, results.MAX.b);
_statMean->setValueAtTime(time, results.mean.r, results.mean.g, results.mean.b);
}

void ScanPlugin::updateMed(const Image* srcImg, double time, const OfxRectI &analysisWindow)
{
Results results;
if ( !abort() ) {
updateSub<ImageMedianProcessor>(srcImg, time, analysisWindow, results, &results);
}
if ( abort() ) {
return;
}
_statMedian->setValueAtTime(time, results.median.r, results.median.g, results.median.b);
}

void ScanPlugin::updateHSV(const Image* srcImg, double time, const OfxRectI &analysisWindow)
{
Results results;
if ( !abort() ) {
updateSub<ImageHSVMinMaxMeanProcessor>(srcImg, time, analysisWindow, results, &results);
}
if ( abort() ) {
return;
}
_statHSVMin->setValueAtTime(time, results.MIN.r, results.MIN.g, results.MIN.b);
_statHSVMax->setValueAtTime(time, results.MAX.r, results.MAX.g, results.MAX.b);
_statHSVMean->setValueAtTime(time, results.mean.r, results.mean.g, results.mean.b);
}

void ScanPlugin::updateLuma(const Image* srcImg, double time, const OfxRectI &analysisWindow)
{
Results results;
if ( !abort() ) {
updateSub<ImageLumaProcessor>(srcImg, time, analysisWindow, results, &results);
}
if ( abort() ) {
return;
}
_maxLumaPix->setValueAtTime(time, results.maxPos.x, results.maxPos.y);
_maxLumaPixVal->setValueAtTime(time, results.maxVal.r, results.maxVal.g, results.maxVal.b);
_minLumaPix->setValueAtTime(time, results.minPos.x, results.minPos.y);
_minLumaPixVal->setValueAtTime(time, results.minVal.r, results.minVal.g, results.minVal.b);
}

class ScanInteract : public RectangleInteract
{
public:

ScanInteract(OfxInteractHandle handle, ImageEffect* effect) : RectangleInteract(handle, effect), _restrictToRectangle(0)
{
_restrictToRectangle = effect->fetchBooleanParam(kParamRestrictToRectangle);
addParamToSlaveTo(_restrictToRectangle);
}

private:

virtual bool draw(const DrawArgs &args) OVERRIDE FINAL
{
bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);
if (restrictToRectangle) {
return RectangleInteract::draw(args);
}
return false;
}

virtual bool penMotion(const PenArgs &args) OVERRIDE FINAL
{
bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);
if (restrictToRectangle) {
return RectangleInteract::penMotion(args);
}
return false;
}

virtual bool penDown(const PenArgs &args) OVERRIDE FINAL
{
bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);
if (restrictToRectangle) {
return RectangleInteract::penDown(args);
}
return false;
}

virtual bool penUp(const PenArgs &args) OVERRIDE FINAL
{
bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);
if (restrictToRectangle) {
return RectangleInteract::penUp(args);
}
return false;
}

BooleanParam* _restrictToRectangle;
};

class ScanOverlayDescriptor : public DefaultEffectOverlayDescriptor<ScanOverlayDescriptor, ScanInteract>
{
};

ScanPluginFactory::ScanPluginFactory()
: OFX::PluginFactoryHelper<ScanPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ScanPluginFactory::describe(ImageEffectDescriptor &desc)
{
desc.setLabel(kPluginName);
desc.setPluginGrouping(kPluginGrouping);
desc.setPluginDescription(kPluginDescription);

desc.addSupportedContext(eContextGeneral);
desc.addSupportedContext(eContextFilter);

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
#ifdef __APPLE__
desc.setSupportsMetalRender(true);
#endif
desc.setOverlayInteractDescriptor(new ScanOverlayDescriptor);
}

void ScanPluginFactory::describeInContext(ImageEffectDescriptor &desc, ContextEnum)
{
ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
srcClip->addSupportedComponent(ePixelComponentRGBA);
srcClip->addSupportedComponent(ePixelComponentRGB);
srcClip->addSupportedComponent(ePixelComponentAlpha);
srcClip->setTemporalClipAccess(false);
srcClip->setSupportsTiles(kSupportsTiles);
srcClip->setIsMask(false);
srcClip->setOptional(false);

ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
dstClip->addSupportedComponent(ePixelComponentRGBA);
dstClip->addSupportedComponent(ePixelComponentRGB);
dstClip->addSupportedComponent(ePixelComponentAlpha);
dstClip->setSupportsTiles(kSupportsTiles);

PageParamDescriptor *page = desc.definePageParam("Controls");

RGBParamDescriptor *paramRGB = desc.defineRGBParam("balance");
paramRGB->setLabel("Sample Pixel");
paramRGB->setHint("sample pixel RGB value");
paramRGB->setDefault(1.0, 1.0, 1.0);
paramRGB->setRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
paramRGB->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
paramRGB->setAnimates(true);
page->addChild(*paramRGB);

Double3DParamDescriptor* param3D = desc.defineDouble3DParam("rgbVal");
param3D->setLabel("RGB Values");
param3D->setDimensionLabels("r", "g", "b");
param3D->setDefault(1.0, 1.0, 1.0);
param3D->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
page->addChild(*param3D);

param3D = desc.defineDouble3DParam("hslVal");
param3D->setLabel("HSL Values");
param3D->setDimensionLabels("r", "g", "b");
param3D->setDefault(0.0, 0.0, 1.0);
param3D->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
page->addChild(*param3D);

BooleanParamDescriptor* boolParam = desc.defineBooleanParam("whiteBalance");
boolParam->setDefault(false);
boolParam->setHint("white balance image");
boolParam->setLabel("White Balance");
page->addChild(*boolParam);

ChoiceParamDescriptor* choiceparam = desc.defineChoiceParam(kParamSampleType);
choiceparam->setLabel(kParamSampleTypeLabel);
choiceparam->setHint(kParamSampleTypeHint);
assert(choiceparam->getNOptions() == eSampleTypeColorPicker);
choiceparam->appendOption(kParamSampleTypeOptionColorPicker, kParamSampleTypeOptionColorPickerHint);
assert(choiceparam->getNOptions() == eSampleTypeMean);
choiceparam->appendOption(kParamSampleTypeOptionMean, kParamSampleTypeOptionMeanHint);
assert(choiceparam->getNOptions() == eSampleTypeMedian);
choiceparam->appendOption(kParamSampleTypeOptionMedian, kParamSampleTypeOptionMedianHint);
page->addChild(*choiceparam);

choiceparam = desc.defineChoiceParam(kParamBalanceType);
choiceparam->setLabel(kParamBalanceTypeLabel);
choiceparam->setHint(kParamBalanceTypeHint);
assert(choiceparam->getNOptions() == eBalanceTypeGain);
choiceparam->appendOption(kParamBalanceTypeOptionGain, kParamBalanceTypeOptionGainHint);
assert(choiceparam->getNOptions() == eBalanceTypeOffset);
choiceparam->appendOption(kParamBalanceTypeOptionOffset, kParamBalanceTypeOptionOffsetHint);
assert(choiceparam->getNOptions() == eBalanceTypeLift);
choiceparam->appendOption(kParamBalanceTypeOptionLift, kParamBalanceTypeOptionLiftHint);
page->addChild(*choiceparam);

boolParam = desc.defineBooleanParam("preserveLuma");
boolParam->setDefault(false);
boolParam->setHint("preserve luma value");
boolParam->setLabel("Preserve Luma");
page->addChild(*boolParam);

PushButtonParamDescriptor* pushparam = desc.definePushButtonParam("info");
pushparam->setLabel("Info");
page->addChild(*pushparam);

GroupParamDescriptor* limitgroup = desc.defineGroupParam("lumaLimit");
limitgroup->setOpen(false);
limitgroup->setLabel("Luma Limiter");
page->addChild(*limitgroup);

DoubleParamDescriptor *param = desc.defineDoubleParam("lumaLimiter");
param->setLabel("Luma Limiter");
param->setHint("limit to luma range");
param->setDefault(0.0);
param->setRange(-2.0, 2.0);
param->setDisplayRange(-2.0, 2.0);
param->setIncrement(0.001);
param->setDoubleType(eDoubleTypeScale);
param->setParent(*limitgroup);
page->addChild(*param);

param = desc.defineDoubleParam("blurAlpha");
param->setLabel("Blur Alpha");
param->setHint("blur alpha channel");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setParent(*limitgroup);
page->addChild(*param);

boolParam = desc.defineBooleanParam("displayAlpha");
boolParam->setDefault(false);
boolParam->setHint("display alpha channel");
boolParam->setLabel("Display Alpha");
boolParam->setParent(*limitgroup);
page->addChild(*boolParam);

GroupParamDescriptor* samplegroup = desc.defineGroupParam("sampleRegion");
samplegroup->setLabel("Sample Region");
if (page)
page->addChild(*samplegroup);

boolParam = desc.defineBooleanParam(kParamRestrictToRectangle);
boolParam->setLabel(kParamRestrictToRectangleLabel);
boolParam->setHint(kParamRestrictToRectangleHint);
boolParam->setDefault(true);
boolParam->setAnimates(false);
boolParam->setParent(*samplegroup);
page->addChild(*boolParam);

Double2DParamDescriptor* param2D = desc.defineDouble2DParam(kParamRectangleInteractBtmLeft);
param2D->setLabel(kParamRectangleInteractBtmLeftLabel);
param2D->setDoubleType(eDoubleTypeXYAbsolute);
param2D->setDefault(860, 440);
param2D->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(-10000, -10000, 10000, 10000);
param2D->setIncrement(1);
param2D->setHint(kParamRectangleInteractBtmLeftHint);
param2D->setDigits(0);
param2D->setEvaluateOnChange(false);
param2D->setAnimates(true);
param2D->setParent(*samplegroup);
page->addChild(*param2D);

param2D = desc.defineDouble2DParam(kParamRectangleInteractSize);
param2D->setLabel(kParamRectangleInteractSizeLabel);
param2D->setDoubleType(eDoubleTypeXYAbsolute);
param2D->setDefault(200, 200);
param2D->setRange(0, 0, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(0, 0, 10000, 10000);
param2D->setIncrement(1.);
param2D->setDimensionLabels(kParamRectangleInteractSizeDim1, kParamRectangleInteractSizeDim2);
param2D->setHint(kParamRectangleInteractSizeHint);
param2D->setDigits(0);
param2D->setEvaluateOnChange(false);
param2D->setAnimates(true);
param2D->setParent(*samplegroup);
page->addChild(*param2D);

GroupParamDescriptor* groupRGB = desc.defineGroupParam(kParamGroupRGB);
groupRGB->setLabel(kParamGroupRGB);
if (page)
page->addChild(*groupRGB);

param3D = desc.defineDouble3DParam(kParamStatMin);
param3D->setLabel(kParamStatMinLabel);
param3D->setDimensionLabels("r", "g", "b");
param3D->setHint(kParamStatMinHint);
param3D->setDefault(0.0, 0.0, 0.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000.0, -10000.0, -10000.0, 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupRGB);
page->addChild(*param3D);

param3D = desc.defineDouble3DParam(kParamStatMax);
param3D->setLabel(kParamStatMaxLabel);
param3D->setDimensionLabels("r", "g", "b");
param3D->setHint(kParamStatMaxHint);
param3D->setDefault(1.0, 1.0, 1.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000.0, -10000.0, -10000.0, 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupRGB);
page->addChild(*param3D);

param3D = desc.defineDouble3DParam(kParamStatMean);
param3D->setLabel(kParamStatMeanLabel);
param3D->setDimensionLabels("r", "g", "b");
param3D->setHint(kParamStatMeanHint);
param3D->setDefault(1.0, 1.0, 1.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupRGB);
page->addChild(*param3D);

pushparam = desc.definePushButtonParam(kParamAnalyzeFrame);
pushparam->setLabel(kParamAnalyzeFrameLabel);
pushparam->setHint(kParamAnalyzeFrameHint);
pushparam->setParent(*groupRGB);
page->addChild(*pushparam);

param3D = desc.defineDouble3DParam(kParamStatMedian);
param3D->setLabel(kParamStatMedianLabel);
param3D->setDimensionLabels("r", "g", "b");
param3D->setHint(kParamStatMedianHint);
param3D->setDefault(1.0, 1.0, 1.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupRGB);
page->addChild(*param3D);

pushparam = desc.definePushButtonParam(kParamAnalyzeFrameMed);
pushparam->setLabel(kParamAnalyzeFrameMedLabel);
pushparam->setHint(kParamAnalyzeFrameMedHint);
pushparam->setParent(*groupRGB);
page->addChild(*pushparam);

pushparam = desc.definePushButtonParam(kParamClearFrame);
pushparam->setLabel(kParamClearFrameLabel);
pushparam->setHint(kParamClearFrameHint);
pushparam->setParent(*groupRGB);
page->addChild(*pushparam);

GroupParamDescriptor* groupHSV = desc.defineGroupParam(kParamGroupHSV);
groupHSV->setOpen(false);
groupHSV->setLabel(kParamGroupHSV);
if (page)
page->addChild(*groupHSV);

param3D = desc.defineDouble3DParam(kParamStatHSVMin);
param3D->setLabel(kParamStatHSVMinLabel);
param3D->setHint(kParamStatHSVMinHint);
param3D->setDefault(0.0, 0.0, 0.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000.0, -10000.0, -10000.0, 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setDimensionLabels("h", "s", "v");
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupHSV);
page->addChild(*param3D);

param3D = desc.defineDouble3DParam(kParamStatHSVMax);
param3D->setLabel(kParamStatHSVMaxLabel);
param3D->setHint(kParamStatHSVMaxHint);
param3D->setDefault(1.0, 1.0, 1.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setDimensionLabels("h", "s", "v");
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupHSV);
page->addChild(*param3D);

param3D = desc.defineDouble3DParam(kParamStatHSVMean);
param3D->setLabel(kParamStatHSVMeanLabel);
param3D->setHint(kParamStatHSVMeanHint);
param3D->setDefault(1.0, 1.0, 1.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setDimensionLabels("h", "s", "v");
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupHSV);
page->addChild(*param3D);

pushparam = desc.definePushButtonParam(kParamAnalyzeFrameHSV);
pushparam->setLabel(kParamAnalyzeFrameHSVLabel);
pushparam->setHint(kParamAnalyzeFrameHSVHint);
pushparam->setParent(*groupHSV);
page->addChild(*pushparam);

pushparam = desc.definePushButtonParam(kParamClearFrameHSV);
pushparam->setLabel(kParamClearFrameHSVLabel);
pushparam->setHint(kParamClearFrameHSVHint);
pushparam->setParent(*groupHSV);
page->addChild(*pushparam);

GroupParamDescriptor* groupLuma = desc.defineGroupParam(kParamGroupLuma);
groupLuma->setOpen(false);
groupLuma->setLabel(kParamGroupLuma);
if (page)
page->addChild(*groupLuma);

choiceparam = desc.defineChoiceParam(kParamLuminanceMath);
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
choiceparam->setParent(*groupLuma);
page->addChild(*choiceparam);

param2D = desc.defineDouble2DParam(kParamMaxLumaPix);
param2D->setDoubleType(eDoubleTypeXYAbsolute);
param2D->setUseHostNativeOverlayHandle(true);
param2D->setLabel(kParamMaxLumaPixLabel);
param2D->setHint(kParamMaxLumaPixHint);
param2D->setDimensionLabels("x", "y");
param2D->setEvaluateOnChange(false);
param2D->setAnimates(true);
param2D->setParent(*groupLuma);
page->addChild(*param2D);

param3D = desc.defineDouble3DParam(kParamMaxLumaPixVal);
param3D->setLabel(kParamMaxLumaPixValLabel);
param3D->setDimensionLabels("r", "g", "b");
param3D->setHint(kParamMaxLumaPixValHint);
param3D->setDefault(0.0, 0.0, 0.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupLuma);
page->addChild(*param3D);

param2D = desc.defineDouble2DParam(kParamMinLumaPix);
param2D->setDoubleType(eDoubleTypeXYAbsolute);
param2D->setUseHostNativeOverlayHandle(true);
param2D->setLabel(kParamMinLumaPixLabel);
param2D->setHint(kParamMinLumaPixHint);
param2D->setDimensionLabels("x", "y");
param2D->setEvaluateOnChange(false);
param2D->setAnimates(true);
param2D->setParent(*groupLuma);
page->addChild(*param2D);

param3D = desc.defineDouble3DParam(kParamMinLumaPixVal);
param3D->setLabel(kParamMinLumaPixValLabel);
param3D->setDimensionLabels("r", "g", "b");
param3D->setHint(kParamMinLumaPixValHint);
param3D->setDefault(0.0, 0.0, 0.0);
param3D->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
param3D->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
param3D->setDoubleType(eDoubleTypeScale);
param3D->setIncrement(0.0001);
param3D->setDigits(6);
param3D->setEvaluateOnChange(false);
param3D->setAnimates(true);
param3D->setParent(*groupLuma);
page->addChild(*param3D);

pushparam = desc.definePushButtonParam(kParamAnalyzeFrameLuma);
pushparam->setLabel(kParamAnalyzeFrameLumaLabel);
pushparam->setHint(kParamAnalyzeFrameLumaHint);
pushparam->setParent(*groupLuma);
page->addChild(*pushparam);

pushparam = desc.definePushButtonParam(kParamClearFrameLuma);
pushparam->setLabel(kParamClearFrameLumaLabel);
pushparam->setHint(kParamClearFrameLumaHint);
pushparam->setParent(*groupLuma);
page->addChild(*pushparam);

boolParam = desc.defineBooleanParam("displayMax");
boolParam->setDefault(false);
boolParam->setHint("display max pixel");
boolParam->setLabel("Display Max Pixel");
boolParam->setParent(*groupLuma);
page->addChild(*boolParam);

boolParam = desc.defineBooleanParam("displayMin");
boolParam->setDefault(false);
boolParam->setHint("display min pixel");
boolParam->setLabel("Display Min Pixel");
boolParam->setParent(*groupLuma);
page->addChild(*boolParam);
    
GroupParamDescriptor* script = desc.defineGroupParam("Script Export");
script->setOpen(false);
script->setHint("export DCTL script");
page->addChild(*script);

pushparam = desc.definePushButtonParam("button1");
pushparam->setLabel("Export DCTL");
pushparam->setHint("create DCTL version");
pushparam->setParent(*script);
page->addChild(*pushparam);

StringParamDescriptor* stringparam = desc.defineStringParam("name");
stringparam->setLabel("Name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("Scan");
stringparam->setParent(*script);
page->addChild(*stringparam);

stringparam = desc.defineStringParam("path");
stringparam->setLabel("Directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript);
stringparam->setFilePathExists(false);
stringparam->setParent(*script);
page->addChild(*stringparam);        
}

ImageEffect* ScanPluginFactory::createInstance(OfxImageEffectHandle handle, ContextEnum) {
return new ScanPlugin(handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static ScanPluginFactory scanPlugin;
p_FactoryArray.push_back(&scanPlugin);
}