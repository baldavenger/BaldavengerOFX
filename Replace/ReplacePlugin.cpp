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
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"Adjust hue, saturation and brightness, or perform colour replacement. \n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Colour replacement: Set the srcColour and dstColour parameters. The range of the replacement is determined by the  \n" \
"three groups of parameters: Hue, Saturation and Brightness. \n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Colour adjust: Use the Rotation of the Hue parameter and the Adjustment of the Saturation and Lightness. \n" \
"The ranges and falloff parameters allow for more complex adjustments. \n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Hue keyer: Set the outputAlpha parameter to All, and select Display Alpha. \n" \
"First, set the Range parameter of the Hue parameter set and then work down the other Ranges parameters, \n" \
"tuning with the range Falloff and Adjustment parameters. \n" \

#define kPluginIdentifier "OpenFX.Yo.Replace"

#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles 1
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 0
#define kSupportsMultipleClipPARs false
#define kSupportsMultipleClipDepths false
#define kRenderThreadSafety eRenderFullySafe


#define kGroupColourReplacement "colourReplacement"
#define kGroupColourReplacementLabel "Colour Replacement"
#define kGroupColourReplacementHint "Easily replace a given colour by another colour by setting srcColour and dstColour. Set Src Colour first, then Dst Colour."
#define kParamSrcColour "srcColour"
#define kParamSrcColourLabel "Src Colour"
#define kParamSrcColourHint "Source colour for replacement. Changing this parameter sets the hue, saturation and brightness ranges for this colour, and sets the fallofs to default values."
#define kParamDstColour "dstColour"
#define kParamDstColourLabel "Dst Colour"
#define kParamDstColourHint "Destination colour for replacement. Changing this parameter sets the hue rotation, and saturation and brightness adjustments. Should be set after Src Colour."

#define kParamEnableRectangle "enableRectangle"
#define kParamEnableRectangleLabel "Src Analysis Rectangle"
#define kParamEnableRectangleHint "Enable the rectangle interact for analysis of Src and Dst colours and ranges."

#define kParamSetSrcFromRectangle "setSrcFromRectangle"
#define kParamSetSrcFromRectangleLabel "Set Src from Rectangle"
#define kParamSetSrcFromRectangleHint "Set the Src colour and ranges and the adjustments from the colours of the source image within the selection rectangle and the Dst Colour."

#define kGroupHue "hue"
#define kGroupHueLabel "Hue"
#define kGroupHueHint "Hue modification settings."
#define kParamHueRange "hueRange"
#define kParamHueRangeLabel "Hue Range"
#define kParamHueRangeHint "Range of colour hues that are modified (in degrees). Red is 0, green is 120, blue is 240. The affected hue range is the smallest interval. For example, if the range is (12, 348), then the selected range is red plus or minus 12 degrees. Exception: if the range width is exactly 360, then all hues are modified."
#define kParamHueRotation "hueRotation"
#define kParamHueRotationLabel "Hue Rotation"
#define kParamHueRotationHint "Rotation of colour hues (in degrees) within the range."
#define kParamHueRotationGain "hueRotationGain"
#define kParamHueRotationGainLabel "Hue Rotation Gain"
#define kParamHueRotationGainHint "Factor to be applied to the rotation of colour hues (in degrees) within the range. A value of 0 will set all values within range to a constant (computed at the center of the range), and a value of 1 will add hueRotation to all values within range."
#define kParamHueRangeRolloff "hueRangeRolloff"
#define kParamHueRangeRolloffLabel "Hue Range Rolloff"
#define kParamHueRangeRolloffHint "Interval (in degrees) around Hue Range, where hue rotation decreases progressively to zero."

#define kGroupSaturation "saturation"
#define kGroupSaturationLabel "Saturation"
#define kGroupSaturationHint "Saturation modification settings."
#define kParamSaturationRange "saturationRange"
#define kParamSaturationRangeLabel "Saturation Range"
#define kParamSaturationRangeHint "Range of colour saturations that are modified."
#define kParamSaturationAdjustment "saturationAdjustment"
#define kParamSaturationAdjustmentLabel "Saturation Adjustment"
#define kParamSaturationAdjustmentHint "Adjustment of colour saturations within the range. Saturation is clamped to zero to avoid colour inversions."
#define kParamSaturationAdjustmentGain "saturationAdjustmentGain"
#define kParamSaturationAdjustmentGainLabel "Saturation Adjustment Gain"
#define kParamSaturationAdjustmentGainHint "Factor to be applied to the saturation adjustment within the range. A value of 0 will set all values within range to a constant (computed at the center of the range), and a value of 1 will add saturationAdjustment to all values within range."
#define kParamSaturationRangeRolloff "saturationRangeRolloff"
#define kParamSaturationRangeRolloffLabel "Saturation Range Rolloff"
#define kParamSaturationRangeRolloffHint "Interval (in degrees) around Saturation Range, where saturation rotation decreases progressively to zero."

#define kGroupBrightness "brightness"
#define kGroupBrightnessLabel "Brightness"
#define kGroupBrightnessHint "Brightness modification settings."
#define kParamBrightnessRange "brightnessRange"
#define kParamBrightnessRangeLabel "Brightness Range"
#define kParamBrightnessRangeHint "Range of colour brightnesss that are modified."
#define kParamBrightnessAdjustment "brightnessAdjustment"
#define kParamBrightnessAdjustmentLabel "Brightness Adjustment"
#define kParamBrightnessAdjustmentHint "Adjustment of colour brightnesss within the range."
#define kParamBrightnessAdjustmentGain "brightnessAdjustmentGain"
#define kParamBrightnessAdjustmentGainLabel "Brightness Adjustment Gain"
#define kParamBrightnessAdjustmentGainHint "Factor to be applied to the brightness adjustment within the range. A value of 0 will set all values within range to a constant (computed at the center of the range), and a value of 1 will add brightnessAdjustment to all values within range."
#define kParamBrightnessRangeRolloff "brightnessRangeRolloff"
#define kParamBrightnessRangeRolloffLabel "Brightness Range Rolloff"
#define kParamBrightnessRangeRolloffHint "Interval (in degrees) around Brightness Range, where brightness rotation decreases progressively to zero."

#define kParamOutputAlpha "outputAlpha"
#define kParamOutputAlphaLabel "Output Alpha"
#define kParamOutputAlphaHint "Output alpha channel. This can either be the source alpha, one of the coefficients for hue, saturation, brightness, or a combination of those. If it is not source alpha, the image on output are unpremultiplied, even if input is premultiplied."
#define kParamOutputAlphaOptionOff "Off"
#define kParamOutputAlphaOptionOffHint "Alpha channel is kept unmodified"
#define kParamOutputAlphaOptionHue "Hue"
#define kParamOutputAlphaOptionHueHint "Set Alpha to the Hue modification mask"
#define kParamOutputAlphaOptionSaturation "Saturation"
#define kParamOutputAlphaOptionSaturationHint "Set Alpha to the Saturation modification mask"
#define kParamOutputAlphaOptionBrightness "Brightness"
#define kParamOutputAlphaOptionBrightnessHint "Alpha is set to the Brighness mask"
#define kParamOutputAlphaOptionHueSaturation "min(Hue,Saturation)"
#define kParamOutputAlphaOptionHueSaturationHint "Alpha is set to min(Hue mask,Saturation mask)"
#define kParamOutputAlphaOptionHueBrightness "min(Hue,Brightness)"
#define kParamOutputAlphaOptionHueBrightnessHint "Alpha is set to min(Hue mask,Brightness mask)"
#define kParamOutputAlphaOptionSaturationBrightness "min(Saturation,Brightness)"
#define kParamOutputAlphaOptionSaturationBrightnessHint "Alpha is set to min(Saturation mask,Brightness mask)"
#define kParamOutputAlphaOptionAll "min(all)"
#define kParamOutputAlphaOptionAllHint "Alpha is set to min(Hue mask,Saturation mask,Brightness mask)"

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
#define kParamDisplayAlphaHint "Displays derived alpha channel."

#define kParamMix "mix"
#define kParamMixLabel "Mix"
#define kParamMixHint "Blend between input and ouput image."

#define kParamDefaultsNormalised "defaultsNormalised"

// to compute the rolloff for a default distribution, we approximate the gaussian with a piecewise linear function
// f(0) = 1, f'(0) = 0
// f(sigma*0.5*sqrt(12)) = 1/2, f'(sigma*0.5*sqrt(12)) = g'(sigma) (g is exp(-x^2/(2*sigma^2)))
// f(inf) = 0, f'(inf) = 0
//#define GAUSSIAN_ROLLOFF 0.8243606354 // exp(1/2)/2
//#define GAUSSIAN_RANGE 1.7320508075 // 0.5*sqrt(12)

// minimum S and V components to take hue into account (hue is too noisy below these values)
#define MIN_SATURATION 0.1
#define MIN_VALUE 0.1

#ifndef M_PI
#define M_PI			3.14159265358979323846264338327950288
#endif

// default fraction of the min-max interval to use as rolloff after rectangle analysis
#define DEFAULT_RECTANGLE_ROLLOFF 0.5

static bool gHostSupportsDefaultCoordinateSystem = true; // for kParamDefaultsNormalised

/* algorithm:
   - convert to HSV
   - compute H, S, and V coefficients: 1 within range, dropping to 0 at range+-rolloff
   - compute min of the three coeffs. coeff = min(hcoeff,scoeff,vcoeff)
   - if global coeff is 0, don't change anything.
   - else, adjust hue by hueRotation*coeff, etc.
   - convert back to RGB

   - when setting srcColour: compute hueRange, satRange, valRange (as empty ranges), set rolloffs to (50,0.3,0.3)
   - when setting dstColour: compute hueRotation, satAdjust and valAdjust
 */

//
static inline
double
normalizeAngle(double a)
{
    int c = (int)std::floor(a / 360);

    a -= c * 360;
    assert(a >= 0 && a <= 360);

    return a;
}

static inline
double
normalizeAngleSigned(double a)
{
    return normalizeAngle(a + 180.) - 180.;
}

static inline
bool
angleWithinRange(double h,
                 double h0,
                 double h1)
{
    assert(0 <= h && h <= 360 && 0 <= h0 && h0 <= 360 && 0 <= h1 && h1 <= 360);

    return ( ( h1 < h0 && (h <= h1 || h0 <= h) ) || (h0 <= h && h <= h1) );
}

// Exponentiation by squaring
// works with positive or negative integer exponents
template<typename T>
T
ipow(T base,
     int exp)
{
    T result = T(1);

    if (exp >= 0) {
        while (exp) {
            if (exp & 1) {
                result *= base;
            }
            exp >>= 1;
            base *= base;
        }
    } else {
        exp = -exp;
        while (exp) {
            if (exp & 1) {
                result /= base;
            }
            exp >>= 1;
            base *= base;
        }
    }

    return result;
}

static double
ffloor(double val,
       int decimals)
{
    int p = ipow(10, decimals);

    return std::floor(val * p) / p;
}

static double
fround(double val,
       int decimals)
{
    int p = ipow(10, decimals);

    return std::floor(val * p + 0.5) / p;
}

static double
fceil(double val,
      int decimals)
{
    int p = ipow(10, decimals);

    return std::ceil(val * p) / p;
}

// returns:
// - 0 if outside of [h0, h1]
// - 0 at h0
// - 1 at h1
// - linear from h0 to h1
static inline
double
angleCoeff01(double h,
             double h0,
             double h1)
{
    assert(0 <= h && h <= 360 && 0 <= h0 && h0 <= 360 && 0 <= h1 && h1 <= 360);
    if ( h1 == (h0 + 360.) ) {
        // interval is the whole hue circle
        return 1.;
    }
    if ( !angleWithinRange(h, h0, h1) ) {
        return 0.;
    }
    if (h1 == h0) {
        return 1.;
    }
    if (h1 < h0) {
        h1 += 360;
        if (h < h0) {
            h += 360;
        }
    }
    assert(h0 <= h && h <= h1);

    return (h - h0) / (h1 - h0);
}

// returns:
// - 0 if outside of [h0, h1]
// - 1 at h0
// - 0 at h1
// - linear from h0 to h1
static inline
double
angleCoeff10(double h,
             double h0,
             double h1)
{
    assert(0 <= h && h <= 360 && 0 <= h0 && h0 <= 360 && 0 <= h1 && h1 <= 360);
    if ( !angleWithinRange(h, h0, h1) ) {
        return 0.;
    }
    if (h1 == h0) {
        return 1.;
    }
    if (h1 < h0) {
        h1 += 360;
        if (h < h0) {
            h += 360;
        }
    }
    assert(h0 <= h && h <= h1);

    return (h1 - h) / (h1 - h0);
}


class ImageScaler : public ImageProcessor
{
public:
    explicit ImageScaler(ImageEffect &instance);

    virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI procWindow);

    void setSrcImg(Image* p_SrcImg);
    void setScales(float hueRangeA, float hueRangeB, float hueRangeWithRollOffA, float hueRangeWithRollOffB, 
    float hueRotation, float hueMean, float hueRotationGain, float hueRolloff, float satRangeA,
    float satRangeB, float satAdjust, float satAdjustGain, float satRolloff, float valRangeA, float valRangeB, 
    float valAdjust, float valAdjustGain, float valRolloff, int OutputAlpha, int DisplayAlpha, float mix);
    
    
private:
    Image* _srcImg;
    float _hueRange[2];
    float _hueRangeWithRolloff[2];
    float _hueRotation;
    float _hueMean;
    float _hueRotationGain;
    float _hueRolloff;
    float _satRange[2];
    float _satAdjust;
    float _satAdjustGain;
    float _satRolloff;
    float _valRange[2];
    float _valAdjust;
    float _valAdjustGain;
    float _valRolloff;
    int _outputAlpha;
    int _displayAlpha;
    float _mix;
};

ImageScaler::ImageScaler(ImageEffect& instance)
	: ImageProcessor(instance)
{
}

extern void RunCudaKernel(int p_Width, int p_Height, float* hueRange, float* hueRangeWithRollOff, 
	float hueRotation, float hueMean, float hueRotationGain, float hueRolloff, float* satRange, 
	float satAdjust, float satAdjustGain, float satRolloff, float* valRange, float valAdjust, 
	float valAdjustGain, float valRolloff, int OutputAlpha, int DisplayAlpha, float mix, 
    const float* p_Input, float* p_Output);

void ImageScaler::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

	RunCudaKernel(width, height, _hueRange, _hueRangeWithRolloff, 
	_hueRotation, _hueMean, _hueRotationGain, _hueRolloff, _satRange,
    _satAdjust, _satAdjustGain, _satRolloff, _valRange, _valAdjust, 
    _valAdjustGain, _valRolloff, _outputAlpha, _displayAlpha, _mix, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* hueRange, float* hueRangeWithRollOff, 
	float hueRotation, float hueMean, float hueRotationGain, float hueRolloff, float* satRange, 
	float satAdjust, float satAdjustGain, float satRolloff, float* valRange, float valAdjust, 
	float valAdjustGain, float valRolloff, int OutputAlpha, int DisplayAlpha, float mix, 
    const float* p_Input, float* p_Output);

void ImageScaler::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());


    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _hueRange, _hueRangeWithRolloff, 
	_hueRotation,_hueMean, _hueRotationGain, _hueRolloff, _satRange,
    _satAdjust, _satAdjustGain, _satRolloff, _valRange, _valAdjust, 
    _valAdjustGain, _valRolloff, _outputAlpha, _displayAlpha, _mix, input, output);
}

void ImageScaler::multiThreadProcessImages(OfxRectI procWindow)
    {
        for (int y = procWindow.y1; y < procWindow.y2; ++y)
    {
        if (_effect.abort()) break;

         float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(procWindow.x1, y));

            for (int x = procWindow.x1; x < procWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
            float hcoeff, scoeff, vcoeff;
            float h, s, v;

			OFX::Color::rgb_to_hsv(srcPix[0], srcPix[1], srcPix[2], &h, &s, &v);

			h *= 360. / OFXS_HUE_CIRCLE;
			const double h0 = _hueRange[0];
			const double h1 = _hueRange[1];
			const double h0mrolloff = _hueRangeWithRolloff[0];
			const double h1prolloff = _hueRangeWithRolloff[1];
			// the affected
			if ( angleWithinRange(h, h0, h1) ) {
				hcoeff = 1.f;
			} else {
				double c0 = 0.;
				double c1 = 0.;
				// check if we are in the rolloff area
				if ( angleWithinRange(h, h0mrolloff, h0) ) {
					c0 = angleCoeff01(h, h0mrolloff, h0);
				}
				if ( angleWithinRange(h, h1, h1prolloff) ) {
					c1 = angleCoeff10(h, h1, h1prolloff);
				}
				hcoeff = (float)fmax(c0, c1);
			}
			assert(0 <= hcoeff && hcoeff <= 1.);
			const double s0 = _satRange[0];
			const double s1 = _satRange[1];
			const double s0mrolloff = s0 - _satRolloff;
			const double s1prolloff = s1 + _satRolloff;
			if ( (s0 <= s) && (s <= s1) ) {
				scoeff = 1.f;
			} else if ( (s0mrolloff <= s) && (s <= s0) ) {
				scoeff = (float)(s - s0mrolloff) / (float)_satRolloff;
			} else if ( (s1 <= s) && (s <= s1prolloff) ) {
				scoeff = (float)(s1prolloff - s) / (float)_satRolloff;
			} else {
				scoeff = 0.f;
			}
			assert(0 <= scoeff && scoeff <= 1.);
			const double v0 = _valRange[0];
			const double v1 = _valRange[1];
			const double v0mrolloff = v0 - _valRolloff;
			const double v1prolloff = v1 + _valRolloff;
			if ( (v0 <= v) && (v <= v1) ) {
				vcoeff = 1.f;
			} else if ( (v0mrolloff <= v) && (v <= v0) ) {
				vcoeff = (float)(v - v0mrolloff) / (float)_valRolloff;
			} else if ( (v1 <= v) && (v <= v1prolloff) ) {
				vcoeff = (float)(v1prolloff - v) / (float)_valRolloff;
			} else {
				vcoeff = 0.f;
			}
			assert(0 <= vcoeff && vcoeff <= 1.);
			float coeff = fmin(fmin(hcoeff, scoeff), vcoeff);
			assert(0 <= coeff && coeff <= 1.);
			if (coeff <= 0.) {
				dstPix[0] = srcPix[0];
				dstPix[1] = srcPix[1];
				dstPix[2] = srcPix[2];
			} else {
				//h += coeff * (float)_hueRotation;
				h += coeff * ( (float)_hueRotation + (_hueRotationGain - 1.) * normalizeAngleSigned(h - _hueMean) );
				s += coeff * ( (float)_satAdjust + (_satAdjustGain - 1.) * (s - (s0 + s1) / 2) );
				if (s < 0) {
					s = 0;
				}
				v += coeff * ( (float)_valAdjust + (_valAdjustGain - 1.) * (v - (v0 + v1) / 2) );
				h *= OFXS_HUE_CIRCLE / 360.;
			
				OFX::Color::hsv_to_rgb(h, s, v, &dstPix[0], &dstPix[1], &dstPix[2]);
			}
               
                if (srcPix)
            {
				float a = _outputAlpha == 0 ? 1.0f : _outputAlpha == 1 ? hcoeff : _outputAlpha == 2 ? scoeff :
				_outputAlpha == 3 ? vcoeff : _outputAlpha == 4 ? fmin(hcoeff, scoeff) : _outputAlpha == 5 ? 
				fmin(hcoeff, vcoeff) : _outputAlpha == 6 ? fmin(scoeff, vcoeff) : fmin(fmin(hcoeff, scoeff), vcoeff);
				dstPix[0] = _displayAlpha == 1 ? a : dstPix[0] * (1. - _mix) + srcPix[0] * _mix;
				dstPix[1] = _displayAlpha == 1 ? a : dstPix[1] * (1. - _mix) + srcPix[1] * _mix;
				dstPix[2] = _displayAlpha == 1 ? a : dstPix[2] * (1. - _mix) + srcPix[2] * _mix;
				dstPix[3] = _outputAlpha != 0 ? a : srcPix[3];
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

void ImageScaler::setSrcImg(Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ImageScaler::setScales(float hueRangeA, float hueRangeB, float hueRangeWithRollOffA, float hueRangeWithRollOffB, 
    float hueRotation, float hueMean, float hueRotationGain, float hueRolloff, float satRangeA,
    float satRangeB, float satAdjust, float satAdjustGain, float satRolloff, float valRangeA, float valRangeB, 
    float valAdjust, float valAdjustGain, float valRolloff, int OutputAlpha, int DisplayAlpha, float mix)
{
    _hueRange[0] = hueRangeA;
    _hueRange[1] = hueRangeB;
    _hueRangeWithRolloff[0] = hueRangeWithRollOffA;
    _hueRangeWithRolloff[1] = hueRangeWithRollOffB;
    _hueRotation = hueRotation;
    _hueMean = hueMean;
    _hueRotationGain = hueRotationGain;
    _hueRolloff = hueRolloff;
    _satRange[0] = satRangeA;
    _satRange[1] = satRangeB;
    _satAdjust = satAdjust;
    _satAdjustGain = satAdjustGain;
    _satRolloff = satRolloff;
    _valRange[0] = valRangeA;
    _valRange[1] = valRangeB;
    _valAdjust = valAdjust;
    _valAdjustGain = valAdjustGain;
    _valRolloff = valRolloff;
    _outputAlpha = OutputAlpha;
    _displayAlpha = DisplayAlpha;
    _mix = mix;
    
    float h0 = _hueRange[0];
	float h1 = _hueRange[1];
	if ( h1 == (h0 + 360.) ) {
		// special case: select any hue (useful to rotate all colours)
		_hueRange[0] = 0.;
		_hueRange[1] = 360.;
		_hueRolloff = 0.;
		_hueRangeWithRolloff[0] = 0.;
		_hueRangeWithRolloff[1] = 360.;
		_hueMean = 0.;
	} else {
		h0 = normalizeAngle(h0);
		h1 = normalizeAngle(h1);
		if (h1 < h0) {
			std::swap(h0, h1);
		}
		// take the smallest of both angles
		if ( (h1 - h0) > 180. ) {
			std::swap(h0, h1);
		}
		assert (0 <= h0 && h0 <= 360 && 0 <= h1 && h1 <= 360);
		_hueRange[0] = h0;
		_hueRange[1] = h1;
		// set strict bounds on rolloff
		if (_hueRolloff < 0.) {
			_hueRolloff = 0.;
		} else if (_hueRolloff >= 180.) {
			_hueRolloff = 180.;
		}
		_hueRangeWithRolloff[0] = normalizeAngle(h0 - _hueRolloff);
		_hueRangeWithRolloff[1] = normalizeAngle(h1 + _hueRolloff);
		_hueMean = normalizeAngle(h0 + normalizeAngleSigned(h1 - h0) / 2);
	}
	if (_satRange[1] < _satRange[0]) {
		std::swap(_satRange[0], _satRange[1]);
	}
	if (_satRolloff < 0.) {
		_satRolloff = 0.;
	}
	if (_valRange[1] < _valRange[0]) {
		std::swap(_valRange[0], _valRange[1]);
	}
	if (_valRolloff < 0.) {
		_valRolloff = 0.;
	}
    
}


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

    double getResult()
    {
        if (_count <= 0) {
            return 0;
        } else {
            double meansinh = _sumsinh / _count;
            double meancosh = _sumcosh / _count;

            // angle mean and sdev from https://en.wikipedia.org/wiki/Directional_statistics#Measures_of_location_and_spread
            return normalizeAngle(std::atan2(meansinh, meancosh) * 180 / M_PI);
            //*huesdev = std::sqrt(fmax(0., -std::log(meansinh*meansinh+meancosh*meancosh)))*180/M_PI;
        }
    }

protected:
    void addResults(double sumsinh,
                    double sumcosh,
                    unsigned long count)
    {
        _sumsinh += sumsinh;
        _sumcosh += sumcosh;
        _count += count;
    }
};

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

    void pixToHSV(const PIX *p,
                  HSVColourF* hsv)
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
        }
    }

    void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
    {
        double sumsinh = 0.;
        double sumcosh = 0.;
        unsigned long count = 0;

        assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
               _dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
        for (int y = procWindow.y1; y < procWindow.y2; ++y) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);

            // partial sums to avoid underflows
            double sumsinhLine = 0.;
            double sumcoshLine = 0.;

            for (int x = procWindow.x1; x < procWindow.x2; ++x) {
                HSVColourF hsv;
                pixToHSV(dstPix, &hsv);
                if ( (hsv.s > MIN_SATURATION) && (hsv.v > MIN_VALUE) ) {
                    // only take into account pixels that really have a hue
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
    }
};

class HSVRangeProcessorBase
    : public ImageProcessor
{
protected:
    float _hmean;

private:
    float _dhmin; // -180..180
    float _dhmax; // -180..180
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

    void getResults(HSVColour *hsvmin,
                    HSVColour *hsvmax)
    {
        if (_dhmax - _dhmin > 179.9) {
            // more than half circle, take the full circle
            hsvmin->h = 0.;
            hsvmax->h = 360.;
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
    }
};

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

    void pixToHSV(const PIX *p,
                  HSVColourF* hsv)
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
        }
    }

    void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
    {
        assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
               _dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
        float dhmin = 0.;
        float dhmax = 0.;
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
                    float dh = normalizeAngleSigned(hsv.h - _hmean); // relative angle with hmean
                    if (dh < dhmin) { dhmin = dh; }
                    if (dh > dhmax) { dhmax = dh; }
                }
                if (hsv.s < smin) { smin = hsv.s; }
                if (hsv.s > smax) { smax = hsv.s; }
                if (hsv.v < vmin) { vmin = hsv.v; }
                if (hsv.v > vmax) { vmax = hsv.v; }

                dstPix += nComponents;
            }
        }

        addResults(dhmin, dhmax, smin, smax, vmin, vmax);
    }
};

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class ReplacePlugin
    : public ImageEffect
{
public:
    /** @brief ctor */
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
        
        // update visibility
        bool enableRectangle = _enableRectangle->getValue();
        _btmLeft->setIsSecretAndDisabled(!enableRectangle);
        _size->setIsSecretAndDisabled(!enableRectangle);
        _setSrcFromRectangle->setIsSecretAndDisabled(!enableRectangle);
        _srcColour->setEnabled(!enableRectangle);

        // honor kParamDefaultsNormalised
        if ( paramExists(kParamDefaultsNormalised) ) {
            // Some hosts (e.g. Resolve) may not support normalized defaults (setDefaultCoordinateSystem(eCoordinatesNormalised))
            // handle these ourselves!
            BooleanParam* param = fetchBooleanParam(kParamDefaultsNormalised);
            assert(param);
            bool normalised = param->getValue();
            if (normalised) {
                OfxPointD size = getProjectExtent();
                OfxPointD origin = getProjectOffset();
                OfxPointD p;
                // we must denormalise all parameters for which setDefaultCoordinateSystem(eCoordinatesNormalised) couldn't be done
                beginEditBlock(kParamDefaultsNormalised);
                _btmLeft->getValue(p.x, p.y);
                _btmLeft->setValue(p.x * size.x + origin.x, p.y * size.y + origin.y);
                _size->getValue(p.x, p.y);
                _size->setValue(p.x * size.x, p.y * size.y);
                param->setValue(false);
                endEditBlock();
            }
        }
    }

private:
    /* Override the render */
    virtual void render(const RenderArguments &args) OVERRIDE FINAL;

    /* set up and run a processor */
    void setupAndProcess(ImageScaler &, const RenderArguments &args);

    virtual bool isIdentity(const IsIdentityArguments &args, Clip * &identityClip, double &identityTime, int& view, std::string& plane); //OVERRIDE FINAL;
    virtual void changedParam(const InstanceChangedArgs &args, const std::string &paramName) OVERRIDE FINAL;

    /** @brief called when a clip has just been changed in some way (a rewire maybe) */
    //virtual void changedClip(const InstanceChangedArgs &args, const std::string &clipName) OVERRIDE FINAL;
    virtual void getClipPreferences(ClipPreferencesSetter &clipPreferences) OVERRIDE FINAL;

    // compute computation window in srcImg
    bool computeWindow(const Image* srcImg, double time, OfxRectI *analysisWindow);

    // update image statistics
    void setSrcFromRectangle(const Image* srcImg, double time, const OfxRectI& analysisWindow);

    void setSrcFromRectangleProcess(HueMeanProcessorBase &huemeanprocessor, HSVRangeProcessorBase &rangeprocessor, const Image* srcImg, double /*time*/, const OfxRectI &analysisWindow, double *hmean, HSVColour *hsvmin, HSVColour *hsvmax);

    template <class PIX, int nComponents, int maxValue>
    void setSrcFromRectangleComponentsDepth(const Image* srcImg,
                                            double time,
                                            const OfxRectI &analysisWindow,
                                            double *hmean,
                                            HSVColour *hsvmin,
                                            HSVColour *hsvmax)
    {
        HueMeanProcessor<PIX, nComponents, maxValue> fred1(*this);
        HSVRangeProcessor<PIX, nComponents, maxValue> fred2(*this);
        setSrcFromRectangleProcess(fred1, fred2, srcImg, time, analysisWindow, hmean, hsvmin, hsvmax);
    }

    template <int nComponents>
    void setSrcFromRectangleComponents(const Image* srcImg,
                                       double time,
                                       const OfxRectI &analysisWindow,
                                       double *hmean,
                                       HSVColour *hsvmin,
                                       HSVColour *hsvmax)
    {
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
        }
    }

private:
    // do not need to delete these, the ImageEffect is managing them for us
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
    
};

////////////////////////////////////////////////////////////////////////////////
/** @brief render for the filter */

////////////////////////////////////////////////////////////////////////////////
// basic plugin render function, just a skelington to instantiate templates from

/* set up and run a processor */
void
ReplacePlugin::setupAndProcess(ImageScaler& p_ImageScaler, const RenderArguments &args)
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
         ( ( dst->getField() != OFX::eFieldNone) /* for DaVinci Resolve */ && ( dst->getField() != args.fieldToRender) ) ) {
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
        }
    }

    std::auto_ptr<OFX::Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                    _srcClip->fetchImage(time) : 0 );
    if ( src.get() ) {
        if ( (src->getRenderScale().x != args.renderScale.x) ||
             ( src->getRenderScale().y != args.renderScale.y) ||
             ( ( src->getField() != OFX::eFieldNone) /* for DaVinci Resolve */ && ( src->getField() != args.fieldToRender) ) ) {
            setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
            OFX::throwSuiteStatusException(kOfxStatFailed);
        }
        OFX::BitDepthEnum srcBitDepth      = src->getPixelDepth();
        OFX::PixelComponentEnum srcComponents = src->getPixelComponents();
        // set the components of _dstClip
        if ( (srcBitDepth != dstBitDepth) || ( (outputAlpha == eOutputAlphaOff) && (srcComponents != dstComponents) ) ) {
            OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
        }
    }
    
	int OutputAlpha = outputalpha_i;
	
	bool displayAlpha = _displayAlpha->getValue();
	int DisplayAlpha = displayAlpha ? 1 : 0;
    
	double hueRangeA, hueRangeB;
    _hueRange->getValueAtTime(time, hueRangeA, hueRangeB);
    double hueRangeWithRollOffA, hueRangeWithRollOffB;
    hueRangeWithRollOffA = hueRangeWithRollOffB = 0.; // set in setValues()
    double hueRotation = _hueRotation->getValueAtTime(time);
    double hueRotationGain = _hueRotationGain->getValueAtTime(time);
    double hueMean = 0.; // set in setValues()
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

	p_ImageScaler.setScales(hueRangeA, hueRangeB, hueRangeWithRollOffA, hueRangeWithRollOffB, 
	hueRotation, hueMean, hueRotationGain, hueRolloff, satRangeA, satRangeB, satAdjust, satAdjustGain, 
	satRolloff, valRangeA, valRangeB, valAdjust, valAdjustGain, valRolloff, OutputAlpha, DisplayAlpha, mix);
	
	p_ImageScaler.setDstImg(dst.get());
	p_ImageScaler.setSrcImg(src.get());
   
	// Setup OpenCL and CUDA Render arguments
	p_ImageScaler.setGPURenderArgs(args);
   
	p_ImageScaler.setRenderWindow(args.renderWindow);
	
	p_ImageScaler.process();
} 

void
ReplacePlugin::render(const RenderArguments &args)
{
    // instantiate the render code based on the pixel depth of the dst clip
    BitDepthEnum dstBitDepth    = _dstClip->getPixelDepth();
    PixelComponentEnum dstComponents  = _dstClip->getPixelComponents();

    assert( kSupportsMultipleClipPARs   || !_srcClip || _srcClip->getPixelAspectRatio() == _dstClip->getPixelAspectRatio() );
    assert( kSupportsMultipleClipDepths || !_srcClip || _srcClip->getPixelDepth()       == _dstClip->getPixelDepth() );
    assert(dstComponents == ePixelComponentRGB || dstComponents == ePixelComponentRGBA);
    
    if (dstComponents == ePixelComponentRGBA || ePixelComponentRGB)
        {
        	ImageScaler fred(*this);
            setupAndProcess(fred, args);
        }
        
        else
        {
            throwSuiteStatusException(kOfxStatErrUnsupported);
        }
   
} // ReplacePlugin::render

bool
ReplacePlugin::isIdentity(const IsIdentityArguments &args,
                          Clip * &identityClip,
                          double & /*identityTime*/
                          , int& /*view*/, std::string& /*plane*/)
{
    if (!_srcClip || !_srcClip->isConnected()) {
        return false;
    }
    const double time = args.time;

    if (_srcClip->getPixelComponents() == ePixelComponentRGBA) {
        // check cases where alpha is affected, even if colours don't change
        int outputalpha_i;
		_outputAlpha->getValueAtTime(time, outputalpha_i);
    	OutputAlphaEnum outputAlpha = (OutputAlphaEnum)outputalpha_i;
        if (outputAlpha != eOutputAlphaOff) {
            double hueMin, hueMax;
            _hueRange->getValueAtTime(time, hueMin, hueMax);
            bool alphaHue = (hueMin != 0. || hueMax != 360.);
            double satMin, satMax;
            _saturationRange->getValueAtTime(time, satMin, satMax);
            bool alphaSat = (satMin != 0. || satMax != 1.);
            double valMin, valMax;
            _brightnessRange->getValueAtTime(time, valMin, valMax);
            bool alphaVal = (valMin != 0. || valMax != 1.);
            switch (outputAlpha) {
            // coverity[dead_error_begin]
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
            }
        }
    }

    // isIdentity=true if hueRotation, satAdjust and valAdjust = 0.
    double hueRotation;
    _hueRotation->getValueAtTime(time, hueRotation);
    double saturationAdjustment;
    _saturationAdjustment->getValueAtTime(time, saturationAdjustment);
    double brightnessAdjustment;
    _brightnessAdjustment->getValueAtTime(time, brightnessAdjustment);
    if ( (hueRotation == 0.) && (saturationAdjustment == 0.) && (brightnessAdjustment == 0.) ) {
        identityClip = _srcClip;

        return true;
    }


    return false;
} // ReplacePlugin::isIdentity

bool
ReplacePlugin::computeWindow(const OFX::Image* srcImg,
                             double time,
                             OfxRectI *analysisWindow)
{
    OfxRectD regionOfInterest;
    bool enableRectangle = _enableRectangle->getValueAtTime(time);

    if (!enableRectangle && _srcClip) {
        return false; // no analysis in this case
        /*
           // use the src region of definition as rectangle, but avoid infinite rectangle
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
         */
    } else {
        _btmLeft->getValueAtTime(time, regionOfInterest.x1, regionOfInterest.y1);
        _size->getValueAtTime(time, regionOfInterest.x2, regionOfInterest.y2);
        regionOfInterest.x2 += regionOfInterest.x1;
        regionOfInterest.y2 += regionOfInterest.y1;
    }
    OFX::Coords::toPixelEnclosing(regionOfInterest,
                             srcImg->getRenderScale(),
                             srcImg->getPixelAspectRatio(),
                             analysisWindow);

    return OFX::Coords::rectIntersection(*analysisWindow, srcImg->getBounds(), analysisWindow);
}

void
ReplacePlugin::setSrcFromRectangle(const OFX::Image* srcImg,
                                   double time,
                                   const OfxRectI &analysisWindow)
{
    double hmean = 0.;
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
        // coverity[dead_error_line]
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);

        return;
    }

    if ( abort() ) {
        return;
    }

    float h = normalizeAngle(hmean);
    float s = (hsvmin.s + hsvmax.s) / 2;
    float v = (hsvmin.v + hsvmax.v) / 2;
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
    OFX::Color::hsv_to_rgb(h * OFXS_HUE_CIRCLE / 360., s, v, &r, &g, &b);
    double tor, tog, tob;
    _dstColour->getValueAtTime(time, tor, tog, tob);
    float toh, tos, tov;
    OFX::Color::rgb_to_hsv( (float)tor, (float)tog, (float)tob, &toh, &tos, &tov );
    double dh = normalizeAngleSigned(toh * 360. / OFXS_HUE_CIRCLE - h);
    // range is from mean+sdev*(GAUSSIAN_RANGE-GAUSSIAN_ROLLOFF) to mean+sdev*(GAUSSIAN_RANGE+GAUSSIAN_ROLLOFF)
    beginEditBlock("setSrcFromRectangle");
    _srcColour->setValue( fround(r, 4), fround(g, 4), fround(b, 4) );
    _hueRange->setValue( ffloor(hsvmin.h, 2), fceil(hsvmax.h, 2) );
    double hrange = hsvmax.h - hsvmin.h;
    if (hrange < 0) {
        hrange += 360.;
    }
    double hrolloff = fmin(hrange * DEFAULT_RECTANGLE_ROLLOFF, (360 - hrange) / 2);
    _hueRangeRolloff->setValue( ffloor(hrolloff, 2) );
    if (tov != 0.) { // no need to rotate if target colour is black
        _hueRotation->setValue( fround(dh, 2) );
    }
    _saturationRange->setValue( ffloor(hsvmin.s, 4), fceil(hsvmax.s, 4) );
    _saturationRangeRolloff->setValue( ffloor( (hsvmax.s - hsvmin.s) * DEFAULT_RECTANGLE_ROLLOFF, 4 ) );
    if (tov != 0.) { // no need to adjust saturation if target colour is black
        _saturationAdjustment->setValue( fround(tos - s, 4) );
    }
    _brightnessRange->setValue( ffloor(hsvmin.v, 4), fceil(hsvmax.v, 4) );
    _brightnessRangeRolloff->setValue( ffloor( (hsvmax.v - hsvmin.v) * DEFAULT_RECTANGLE_ROLLOFF, 4 ) );
    _brightnessAdjustment->setValue( fround(tov - v, 4) );
    endEditBlock();
} // ReplacePlugin::setSrcFromRectangle

/* set up and run a processor */
void
ReplacePlugin::setSrcFromRectangleProcess(HueMeanProcessorBase& p_Huemeanprocessor,
                                          HSVRangeProcessorBase& p_Hsvrangeprocessor,
                                          const OFX::Image* srcImg,
                                          double /*time*/,
                                          const OfxRectI &analysisWindow,
                                          double *hmean,
                                          HSVColour *hsvmin,
                                          HSVColour *hsvmax)
{
    // set the images
    p_Huemeanprocessor.setDstImg( const_cast<OFX::Image*>(srcImg) ); // not a bug: we only set dst

    // set the render window
    p_Huemeanprocessor.setRenderWindow(analysisWindow);

    // Call the base class process member, this will call the derived templated process code
    p_Huemeanprocessor.process();

    if ( abort() ) {
        return;
    }

    *hmean = p_Huemeanprocessor.getResult();

    // set the images
    p_Hsvrangeprocessor.setDstImg( const_cast<OFX::Image*>(srcImg) ); // not a bug: we only set dst

    // set the render window
    p_Hsvrangeprocessor.setRenderWindow(analysisWindow);
    p_Hsvrangeprocessor.setHueMean(*hmean);


    // Call the base class process member, this will call the derived templated process code
    p_Hsvrangeprocessor.process();

    if ( abort() ) {
        return;
    }
    p_Hsvrangeprocessor.getResults(hsvmin, hsvmax);
}

void
ReplacePlugin::changedParam(const OFX::InstanceChangedArgs& args,
                            const std::string& p_ParamName)
{
    const double time = args.time;
    
    if(p_ParamName == "info")
    {
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	}

    if ( (p_ParamName == kParamSrcColour) && (args.reason == OFX::eChangeUserEdit) ) {
        // - when setting srcColour: compute hueRange, satRange, valRange (as empty ranges), set rolloffs to (50,0.3,0.3)
        double r, g, b;
        _srcColour->getValueAtTime(time, r, g, b);
        float h, s, v;
        OFX::Color::rgb_to_hsv( (float)r, (float)g, (float)b, &h, &s, &v );
        h *= 360. / OFXS_HUE_CIRCLE;
        double tor, tog, tob;
        _dstColour->getValueAtTime(time, tor, tog, tob);
        float toh, tos, tov;
        OFX::Color::rgb_to_hsv( (float)tor, (float)tog, (float)tob, &toh, &tos, &tov );
        toh *= 360. / OFXS_HUE_CIRCLE;
        double dh = normalizeAngleSigned(toh - h);
        beginEditBlock("setSrc");
        _hueRange->setValue(h, h);
        _hueRangeRolloff->setValue(50.);
        if (tov != 0.) { // no need to rotate if target colour is black
            _hueRotation->setValue(dh);
        }
        _saturationRange->setValue(s, s);
        _saturationRangeRolloff->setValue(0.3);
        if (tov != 0.) { // no need to adjust saturation if target colour is black
            _saturationAdjustment->setValue(tos - s);
        }
        _brightnessRange->setValue(v, v);
        _brightnessRangeRolloff->setValue(0.3);
        _brightnessAdjustment->setValue(tov - v);
        endEditBlock();
    } else if (p_ParamName == kParamEnableRectangle) {
        // update visibility
        bool enableRectangle = _enableRectangle->getValueAtTime(time);
        _btmLeft->setIsSecretAndDisabled(!enableRectangle);
        _size->setIsSecretAndDisabled(!enableRectangle);
        _setSrcFromRectangle->setIsSecretAndDisabled(!enableRectangle);
        _srcColour->setEnabled(!enableRectangle);
    } else if ( (p_ParamName == kParamSetSrcFromRectangle) && (args.reason == OFX::eChangeUserEdit) ) {
        std::auto_ptr<OFX::Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                  _srcClip->fetchImage(args.time) : 0 );
        if ( src.get() ) {
            if ( (src->getRenderScale().x != args.renderScale.x) ||
                 ( src->getRenderScale().y != args.renderScale.y) ) {
                setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
                OFX::throwSuiteStatusException(kOfxStatFailed);
            }
            OfxRectI analysisWindow;
            bool intersect = computeWindow(src.get(), args.time, &analysisWindow);
            if (intersect) {
#             ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
                getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 1, false);
#             endif
                setSrcFromRectangle(src.get(), args.time, analysisWindow);
#             ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
                getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 0, false);
#             endif
            }
        }
    } else if ( (p_ParamName == kParamDstColour) && (args.reason == OFX::eChangeUserEdit) ) {
        // - when setting dstColour: compute hueRotation, satAdjust and valAdjust
        double r, g, b;
        _srcColour->getValueAtTime(time, r, g, b);
        float h, s, v;
        OFX::Color::rgb_to_hsv( (float)r, (float)g, (float)b, &h, &s, &v );
        h *= 360. / OFXS_HUE_CIRCLE;
        double tor, tog, tob;
        _dstColour->getValueAtTime(time, tor, tog, tob);
        float toh, tos, tov;
        OFX::Color::rgb_to_hsv( (float)tor, (float)tog, (float)tob, &toh, &tos, &tov );
        toh *= 360. / OFXS_HUE_CIRCLE;
        double dh = normalizeAngleSigned(toh - h);
        beginEditBlock("setDst");
        if (tov != 0.) { // no need to adjust hue or saturation if target colour is black
            _hueRotation->setValue(dh);
            _saturationAdjustment->setValue(tos - s);
        }
        _brightnessAdjustment->setValue(tov - v);
        endEditBlock();
    }
} // ReplacePlugin::changedParam



/* Override the clip preferences */
void
ReplacePlugin::getClipPreferences(ClipPreferencesSetter& p_ClipPreferences)
{

    // set the components of _dstClip
   int outputalpha_i;
	_outputAlpha->getValue(outputalpha_i);
    OutputAlphaEnum outputAlpha = (OutputAlphaEnum)outputalpha_i;

    if (outputAlpha != eOutputAlphaOff) {
        // output must be RGBA, output image is unpremult
        p_ClipPreferences.setClipComponents(*_dstClip, OFX::ePixelComponentRGBA);
        p_ClipPreferences.setClipComponents(*_srcClip, OFX::ePixelComponentRGBA);
    }
}

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

    // overridden functions from Interact to do things
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

    //virtual bool keyDown(const KeyArgs &args) OVERRIDE;
    //virtual bool keyUp(const KeyArgs & args) OVERRIDE;
    //virtual void loseFocus(const FocusArgs &args) OVERRIDE FINAL;


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

void
ReplacePluginFactory::describe(OFX::ImageEffectDescriptor &desc)
{
    // basic labels
    desc.setLabel(kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

    desc.addSupportedContext(eContextFilter);
    desc.addSupportedContext(eContextGeneral);
    desc.addSupportedContext(eContextPaint);
    //desc.addSupportedBitDepth(eBitDepthUByte);
    //desc.addSupportedBitDepth(eBitDepthUShort);
    desc.addSupportedBitDepth(eBitDepthFloat);

    // set a few flags
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(kSupportsMultiResolution);
    desc.setSupportsTiles(kSupportsTiles);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);
    desc.setSupportsMultipleClipDepths(kSupportsMultipleClipDepths);
    //desc.setRenderThreadSafety(kRenderThreadSafety);
    
     // Setup OpenCL and CUDA render capability flags
    desc.setSupportsOpenCLRender(true);
    desc.setSupportsCudaRender(true);
    
    desc.setOverlayInteractDescriptor(new ReplaceOverlayDescriptor);
}

void
ReplacePluginFactory::describeInContext(ImageEffectDescriptor& desc,
                                        OFX::ContextEnum /*_context*/)
{
    // Source clip only in the filter context
    // create the mandated source clip
    ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);

    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->addSupportedComponent(ePixelComponentRGB);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);

    // create the mandated output clip
    ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentRGB);
    dstClip->setSupportsTiles(kSupportsTiles);


    // make some pages and to things in
    PageParamDescriptor *page = desc.definePageParam("Controls");

    {
        GroupParamDescriptor *group = desc.defineGroupParam(kGroupColourReplacement);
        if (group) {
            group->setLabel(kGroupColourReplacementLabel);
            group->setHint(kGroupColourReplacementHint);
            group->setEnabled(true);
            if (page) {
                page->addChild(*group);
            }
        }

        // enableRectangle
        {
            BooleanParamDescriptor *param = desc.defineBooleanParam(kParamEnableRectangle);
            param->setLabel(kParamEnableRectangleLabel);
            param->setHint(kParamEnableRectangleHint);
            param->setDefault(false);
            param->setAnimates(false);
            param->setEvaluateOnChange(false);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // btmLeft
        {
            Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamRectangleInteractBtmLeft);
            param->setLabel(kParamRectangleInteractBtmLeftLabel);
            param->setDoubleType(OFX::eDoubleTypeXYAbsolute);
            if ( param->supportsDefaultCoordinateSystem() ) {
                param->setDefaultCoordinateSystem(eCoordinatesNormalised); // no need of kParamDefaultsNormalised
            } else {
                gHostSupportsDefaultCoordinateSystem = false; // no multithread here, see kParamDefaultsNormalised
            }
            param->setDefault(0.4, 0.4);
            param->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(0, 0, 10000, 10000); // Resolve requires display range or values are clamped to (-1,1)
            param->setIncrement(1.);
            param->setHint(kParamRectangleInteractBtmLeftHint);
            param->setDigits(0);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // size
        {
            Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamRectangleInteractSize);
            param->setLabel(kParamRectangleInteractSizeLabel);
            param->setDoubleType(eDoubleTypeXY);
            if ( param->supportsDefaultCoordinateSystem() ) {
                param->setDefaultCoordinateSystem(eCoordinatesNormalised); // no need of kParamDefaultsNormalised
            } else {
                gHostSupportsDefaultCoordinateSystem = false; // no multithread here, see kParamDefaultsNormalised
            }
            param->setDefault(0.2, 0.2);
            param->setRange(0., 0., DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(0, 0, 10000, 10000); // Resolve requires display range or values are clamped to (-1,1)
            param->setIncrement(1.);
            param->setDimensionLabels(kParamRectangleInteractSizeDim1, kParamRectangleInteractSizeDim2);
            param->setHint(kParamRectangleInteractSizeHint);
            param->setDigits(0);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamSetSrcFromRectangle);
            param->setLabel(kParamSetSrcFromRectangleLabel);
            param->setHint(kParamSetSrcFromRectangleHint);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            RGBParamDescriptor *param = desc.defineRGBParam(kParamSrcColour);
            param->setLabel(kParamSrcColourLabel);
            param->setHint(kParamSrcColourHint);
            param->setEvaluateOnChange(false);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            RGBParamDescriptor *param = desc.defineRGBParam(kParamDstColour);
            param->setLabel(kParamDstColourLabel);
            param->setHint(kParamDstColourHint);
            param->setEvaluateOnChange(false);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        
        {
			DoubleParamDescriptor *param = desc.defineDoubleParam(kParamMix);
			param->setLabel(kParamMixLabel);
			param->setHint(kParamMixHint);
			param->setRange(0., 1.);
			param->setDisplayRange(0., 1.);
			param->setDefault(0.);
			if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
		}
		
		{
			PushButtonParamDescriptor* param = desc.definePushButtonParam("info");
			param->setLabel("Info");
			if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
    	}
    }

    {
        GroupParamDescriptor *group = desc.defineGroupParam(kGroupHue);
        if (group) {
            group->setLabel(kGroupHueLabel);
            group->setHint(kGroupHueHint);
            group->setEnabled(true);
            if (page) {
                page->addChild(*group);
            }
        }

        {
            Double2DParamDescriptor *param = desc.defineDouble2DParam(kParamHueRange);
            param->setLabel(kParamHueRangeLabel);
            param->setHint(kParamHueRangeHint);
            param->setDimensionLabels("", ""); // the two values have the same meaning (they just define a range)
            param->setDefault(0., 360.);
            param->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(0., 0., 360., 360.);
            param->setDoubleType(eDoubleTypeAngle);
            param->setUseHostNativeOverlayHandle(false);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamHueRotation);
            param->setLabel(kParamHueRotationLabel);
            param->setHint(kParamHueRotationHint);
            param->setRange(-DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(-180., 180.);
            param->setDoubleType(eDoubleTypeAngle);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamHueRotationGain);
            param->setLabel(kParamHueRotationGainLabel);
            param->setHint(kParamHueRotationGainHint);
            param->setRange(-DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(0., 2.);
            param->setDefault(1.);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamHueRangeRolloff);
            param->setLabel(kParamHueRangeRolloffLabel);
            param->setHint(kParamHueRangeRolloffHint);
            param->setRange(0., 180.);
            param->setDisplayRange(0., 180.);
            param->setDoubleType(eDoubleTypeAngle);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
    }

    {
        GroupParamDescriptor *group = desc.defineGroupParam(kGroupSaturation);
        if (group) {
            group->setLabel(kGroupSaturationLabel);
            group->setHint(kGroupSaturationHint);
            group->setEnabled(true);
            if (page) {
                page->addChild(*group);
            }
        }
        {
            Double2DParamDescriptor *param = desc.defineDouble2DParam(kParamSaturationRange);
            param->setLabel(kParamSaturationRangeLabel);
            param->setHint(kParamSaturationRangeHint);
            param->setDimensionLabels("", ""); // the two values have the same meaning (they just define a range)
            param->setDefault(0., 1.);
            param->setRange(0., 0., DBL_MAX, DBL_MAX);
            param->setDisplayRange(0., 0., 1, 1);
            param->setUseHostNativeOverlayHandle(false);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamSaturationAdjustment);
            param->setLabel(kParamSaturationAdjustmentLabel);
            param->setHint(kParamSaturationAdjustmentHint);
            param->setRange(-1., 1.);
            param->setDisplayRange(-1., 1.);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamSaturationAdjustmentGain);
            param->setLabel(kParamSaturationAdjustmentGainLabel);
            param->setHint(kParamSaturationAdjustmentGainHint);
            param->setRange(-DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(0., 2.);
            param->setDefault(1.);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamSaturationRangeRolloff);
            param->setLabel(kParamSaturationRangeRolloffLabel);
            param->setHint(kParamSaturationRangeRolloffHint);
            param->setRange(0., 1.);
            param->setDisplayRange(0., 1.);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
    }

    {
        GroupParamDescriptor *group = desc.defineGroupParam(kGroupBrightness);
        if (group) {
            group->setLabel(kGroupBrightnessLabel);
            group->setHint(kGroupBrightnessHint);
            group->setEnabled(true);
            if (page) {
                page->addChild(*group);
            }
        }

        {
            Double2DParamDescriptor *param = desc.defineDouble2DParam(kParamBrightnessRange);
            param->setLabel(kParamBrightnessRangeLabel);
            param->setHint(kParamBrightnessRangeHint);
            param->setDimensionLabels("", ""); // the two values have the same meaning (they just define a range)
            param->setDefault(0., 1.);
            param->setRange(0., 0., DBL_MAX, DBL_MAX);
            param->setDisplayRange(0., 0., 1, 1);
            param->setUseHostNativeOverlayHandle(false);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamBrightnessAdjustment);
            param->setLabel(kParamBrightnessAdjustmentLabel);
            param->setHint(kParamBrightnessAdjustmentHint);
            param->setRange(-DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(-1., 1.);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamBrightnessAdjustmentGain);
            param->setLabel(kParamBrightnessAdjustmentGainLabel);
            param->setHint(kParamBrightnessAdjustmentGainHint);
            param->setRange(-DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(0., 2.);
            param->setDefault(1.);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamBrightnessRangeRolloff);
            param->setLabel(kParamBrightnessRangeRolloffLabel);
            param->setHint(kParamBrightnessRangeRolloffHint);
            param->setRange(0., DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(0., 1.);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
    }

    {
        ChoiceParamDescriptor *param = desc.defineChoiceParam(kParamOutputAlpha);
        param->setLabel(kParamOutputAlphaLabel);
        param->setHint(kParamOutputAlphaHint);
        assert(param->getNOptions() == (int)eOutputAlphaOff);
        param->appendOption(kParamOutputAlphaOptionOff, kParamOutputAlphaOptionOffHint);
        assert(param->getNOptions() == (int)eOutputAlphaHue);
        param->appendOption(kParamOutputAlphaOptionHue, kParamOutputAlphaOptionHueHint);
        assert(param->getNOptions() == (int)eOutputAlphaSaturation);
        param->appendOption(kParamOutputAlphaOptionSaturation, kParamOutputAlphaOptionSaturationHint);
        assert(param->getNOptions() == (int)eOutputAlphaBrightness);
        param->appendOption(kParamOutputAlphaOptionBrightness, kParamOutputAlphaOptionBrightnessHint);
        assert(param->getNOptions() == (int)eOutputAlphaHueSaturation);
        param->appendOption(kParamOutputAlphaOptionHueSaturation, kParamOutputAlphaOptionHueSaturationHint);
        assert(param->getNOptions() == (int)eOutputAlphaHueBrightness);
        param->appendOption(kParamOutputAlphaOptionHueBrightness, kParamOutputAlphaOptionHueBrightnessHint);
        assert(param->getNOptions() == (int)eOutputAlphaSaturationBrightness);
        param->appendOption(kParamOutputAlphaOptionSaturationBrightness, kParamOutputAlphaOptionSaturationBrightnessHint);
        assert(param->getNOptions() == (int)eOutputAlphaAll);
        param->appendOption(kParamOutputAlphaOptionAll, kParamOutputAlphaOptionAllHint);
        param->setDefault( (int)eOutputAlphaOff );
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

	{
        BooleanParamDescriptor *param = desc.defineBooleanParam(kParamDisplayAlpha);
        param->setLabel(kParamDisplayAlphaLabel);
        param->setHint(kParamDisplayAlphaHint);
        param->setDefault(false);
        param->setAnimates(false);
		if (page) {
			page->addChild(*param);
		}
    }
    
    // Some hosts (e.g. Resolve) may not support normalized defaults (setDefaultCoordinateSystem(eCoordinatesNormalised))
    if (!gHostSupportsDefaultCoordinateSystem) {
        BooleanParamDescriptor* param = desc.defineBooleanParam(kParamDefaultsNormalised);
        param->setDefault(true);
        param->setEvaluateOnChange(false);
        param->setIsSecretAndDisabled(true);
        param->setIsPersistant(true);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
} 

ImageEffect* ReplacePluginFactory::createInstance(OfxImageEffectHandle handle, ContextEnum /*p_Context*/)
{
    return new ReplacePlugin(handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static ReplacePluginFactory replacePlugin;
    p_FactoryArray.push_back(&replacePlugin);
}
