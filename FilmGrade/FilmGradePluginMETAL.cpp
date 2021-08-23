#include "FilmGradePlugin.h"

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
#define kPluginScript "/home/resolve/LUT"
#endif

#define kPluginName "FilmGrade"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"FilmGrade: Film-style (log scan) grading. Order of operations are Exposure, Midtone, Shadows, \n" \
"Highlights, Contrast, Saturation."

#define kPluginIdentifier "BaldavengerOFX.FilmGrade"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

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

namespace {
struct RGBValues {
double r,g,b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};
}

class FilmGrade : public OFX::ImageProcessor
{
public:
explicit FilmGrade(OFX::ImageEffect& p_Instance);

virtual void processImagesMetal();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(float p_ExpR, float p_ExpG, float p_ExpB, float p_ContR, float p_ContG, 
float p_ContB, float p_SatR, float p_SatG, float p_SatB, float p_ShadR, float p_ShadG, 
float p_ShadB, float p_MidR, float p_MidG, float p_MidB, float p_HighR, float p_HighG, 
float p_HighB, float p_ShadP, float p_HighP, float p_ContP, int p_Display);

private:
OFX::Image* _srcImg;
float _exp[3];
float _cont[3];
float _sat[3];
float _shad[3];
float _mid[3];
float _high[3];
float _pivot[3];
int _display;
};

FilmGrade::FilmGrade(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Exp, 
float* p_Cont, float* p_Sat, float* p_Shad, float* p_Mid, float* p_High, float* p_Pivot, int p_Display);
#endif

void FilmGrade::processImagesMetal()
{
#ifdef __APPLE__
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunMetalKernel(_pMetalCmdQ, input, output, width, height, _exp, _cont, _sat, _shad, _mid, _high, _pivot, _display);
#endif
}

void FilmGrade::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
{
if (_effect.abort()) break;

float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
{
float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

if (srcPix) {
float width = p_ProcWindow.x2;
float height = p_ProcWindow.y2;

float e = 2.718281828459045f;
float pie = 3.141592653589793f;

float Red = srcPix[0];
float Green = srcPix[1];
float Blue = srcPix[2];

float expR = Red + _exp[0]/100.0f;
float expG = Green + _exp[1]/100.0f;
float expB = Blue + _exp[2]/100.0f;

float expr1 = (_pivot[0] / 2.0f) - (1.0f - _pivot[1])/4.0f;
float expr2 = (1.0f - (1.0f - _pivot[1])/2.0f) + (_pivot[0] / 4.0f);
float expr3R = (expR - expr1) / (expr2 - expr1);
float expr3G = (expG - expr1) / (expr2 - expr1);
float expr3B = (expB - expr1) / (expr2 - expr1);
float expr4 =  _pivot[2] < 0.5f ? 0.5f - (0.5f - _pivot[2])/2.0f : 0.5f + (_pivot[2] - 0.5f)/2.0f;
float expr5R = expr3R > expr4 ? (expr3R - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3R /(2.0f*expr4);
float expr5G = expr3G > expr4 ? (expr3G - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3G /(2.0f*expr4);
float expr5B = expr3B > expr4 ? (expr3B - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3B /(2.0f*expr4);
float expr6R = (((sin(2.0f * pie * (expr5R -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[0]*4.0f) + expr3R;
float expr6G = (((sin(2.0f * pie * (expr5G -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[1]*4.0f) + expr3G;
float expr6B = (((sin(2.0f * pie * (expr5B -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[2]*4.0f) + expr3B;
float midR = expR >= expr1 && expR <= expr2 ? expr6R * (expr2 - expr1) + expr1 : expR;
float midG = expG >= expr1 && expG <= expr2 ? expr6G * (expr2 - expr1) + expr1 : expG;
float midB = expB >= expr1 && expB <= expr2 ? expr6B * (expr2 - expr1) + expr1 : expB;

float shadupR1 = midR > 0.0f ? 2.0f * (midR/_pivot[0]) - log((midR/_pivot[0]) * (e * _shad[0] * 2.0f) + 1.0f)/log(e * _shad[0] * 2.0f + 1.0f) : midR;
float shadupR = midR < _pivot[0] && _shad[0] > 0.0f ? (shadupR1 + _shad[0] * (1.0f - shadupR1)) * _pivot[0] : midR;
float shadupG1 = midG > 0.0f ? 2.0f * (midG/_pivot[0]) - log((midG/_pivot[0]) * (e * _shad[1] * 2.0f) + 1.0f)/log(e * _shad[1] * 2.0f + 1.0f) : midG;
float shadupG = midG < _pivot[0] && _shad[1] > 0.0f ? (shadupG1 + _shad[1] * (1.0f - shadupG1)) * _pivot[0] : midG;
float shadupB1 = midB > 0.0f ? 2.0f * (midB/_pivot[0]) - log((midB/_pivot[0]) * (e * _shad[2] * 2.0f) + 1.0f)/log(e * _shad[2] * 2.0f + 1.0f) : midB;
float shadupB = midB < _pivot[0] && _shad[2] > 0.0f ? (shadupB1 + _shad[2] * (1.0f - shadupB1)) * _pivot[0] : midB;

float shaddownR1 = (shadupR/_pivot[0]) + (_shad[0] * 2.0f * (1.0f - shadupR/_pivot[0]));
float shaddownR = shadupR < _pivot[0] && _shad[0] < 0.0f ? (shaddownR1 >= 0.0f ? log(shaddownR1 * (e * _shad[0] * -2.0f) + 1.0f)/log(e * _shad[0] * -2.0f + 1.0f) : shaddownR1) * _pivot[0] : shadupR;
float shaddownG1 = (shadupG/_pivot[0]) + (_shad[1] * 2.0f * (1.0f - shadupG/_pivot[0]));
float shaddownG = shadupG < _pivot[0] && _shad[1] < 0.0f ? (shaddownG1 >= 0.0f ? log(shaddownG1 * (e * _shad[1] * -2.0f) + 1.0f)/log(e * _shad[1] * -2.0f + 1.0f) : shaddownG1) * _pivot[0] : shadupG;
float shaddownB1 = (shadupB/_pivot[0]) + (_shad[2] * 2.0f * (1.0f - shadupB/_pivot[0]));
float shaddownB = shadupB < _pivot[0] && _shad[2] < 0.0f ? (shaddownB1 >= 0.0f ? log(shaddownB1 * (e * _shad[2] * -2.0f) + 1.0f)/log(e * _shad[2] * -2.0f + 1.0f) : shaddownB1) * _pivot[0] : shadupB;

float highupR1 = ((shaddownR - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[0] * 2.0f));
float highupR = shaddownR > _pivot[1] && _pivot[1] < 1.0f && _high[0] > 0.0f ? (2.0f * highupR1 - log(highupR1 * e * _high[0] + 1.0f)/log(e * _high[0] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : shaddownR;
float highupG1 = ((shaddownG - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[1] * 2.0f));
float highupG = shaddownG > _pivot[1] && _pivot[1] < 1.0f && _high[1] > 0.0f ? (2.0f * highupG1 - log(highupG1 * e * _high[1] + 1.0f)/log(e * _high[1] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : shaddownG;
float highupB1 = ((shaddownB - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[2] * 2.0f));
float highupB = shaddownB > _pivot[1] && _pivot[1] < 1.0f && _high[2] > 0.0f ? (2.0f * highupB1 - log(highupB1 * e * _high[2] + 1.0f)/log(e * _high[2] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : shaddownB;

float highdownR1 = (highupR - _pivot[1]) / (1.0f - _pivot[1]);
float highdownR = highupR > _pivot[1] && _pivot[1] < 1.0f && _high[0] < 0.0f ? log(highdownR1 * (e * _high[0] * -2.0f) + 1.0f)/log(e * _high[0] * -2.0f + 1.0f) * (1.0f + _high[0]) * (1.0f - _pivot[1]) + _pivot[1]  : highupR;
float highdownG1 = (highupG - _pivot[1]) / (1.0f - _pivot[1]);
float highdownG = highupG > _pivot[1] && _pivot[1] < 1.0f && _high[1] < 0.0f ? log(highdownG1 * (e * _high[1] * -2.0f) + 1.0f)/log(e * _high[1] * -2.0f + 1.0f) * (1.0f + _high[1]) * (1.0f - _pivot[1]) + _pivot[1]  : highupG;
float highdownB1 = (highupB - _pivot[1]) / (1.0f - _pivot[1]);
float highdownB = highupB > _pivot[1] && _pivot[1] < 1.0f && _high[2] < 0.0f ? log(highdownB1 * (e * _high[2] * -2.0f) + 1.0f)/log(e * _high[2] * -2.0f + 1.0f) * (1.0f + _high[2]) * (1.0f - _pivot[1]) + _pivot[1]  : highupB;

float contR = (highdownR - _pivot[2]) * _cont[0] + _pivot[2];
float contG = (highdownG - _pivot[2]) * _cont[1] + _pivot[2];
float contB = (highdownB - _pivot[2]) * _cont[2] + _pivot[2];

float luma = contR * 0.2126f + contG * 0.7152f + contB * 0.0722f;
float satR = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * luma + contR * _sat[0];
float satG = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * luma + contG * _sat[1];
float satB = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * luma + contB * _sat[2];

float DexpR = (x / width) + _exp[0]/100.0f;
float DexpG = (x / width) + _exp[1]/100.0f;
float DexpB = (x / width) + _exp[2]/100.0f;

float Dexpr1 = (_pivot[0] / 2.0f) - (1.0f - _pivot[1])/4.0f;
float Dexpr2 = (1.0f - (1.0f - _pivot[1])/2.0f) + (_pivot[0] / 4.0f);
float Dexpr3R = (DexpR - Dexpr1) / (Dexpr2 - Dexpr1);
float Dexpr3G = (DexpG - Dexpr1) / (Dexpr2 - Dexpr1);
float Dexpr3B = (DexpB - Dexpr1) / (Dexpr2 - Dexpr1);
float Dexpr4 =  _pivot[2] < 0.5f ? 0.5f - (0.5f - _pivot[2])/2.0f : 0.5f + (_pivot[2] - 0.5f)/2.0f;
float Dexpr5R = Dexpr3R > Dexpr4 ? (Dexpr3R - Dexpr4) / (2.0f - 2.0f*Dexpr4) + 0.5f : Dexpr3R /(2.0f*Dexpr4);
float Dexpr5G = Dexpr3G > Dexpr4 ? (Dexpr3G - Dexpr4) / (2.0f - 2.0f*Dexpr4) + 0.5f : Dexpr3G /(2.0f*Dexpr4);
float Dexpr5B = Dexpr3B > Dexpr4 ? (Dexpr3B - Dexpr4) / (2.0f - 2.0f*Dexpr4) + 0.5f : Dexpr3B /(2.0f*Dexpr4);
float Dexpr6R = (((sin(2.0f * pie * (Dexpr5R -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[0]*4.0f) + Dexpr3R;
float Dexpr6G = (((sin(2.0f * pie * (Dexpr5G -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[1]*4.0f) + Dexpr3G;
float Dexpr6B = (((sin(2.0f * pie * (Dexpr5B -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[2]*4.0f) + Dexpr3B;
float DmidR = DexpR >= Dexpr1 && DexpR <= Dexpr2 ? Dexpr6R * (Dexpr2 - Dexpr1) + Dexpr1 : DexpR;
float DmidG = DexpG >= Dexpr1 && DexpG <= Dexpr2 ? Dexpr6G * (Dexpr2 - Dexpr1) + Dexpr1 : DexpG;
float DmidB = DexpB >= Dexpr1 && DexpB <= Dexpr2 ? Dexpr6B * (Dexpr2 - Dexpr1) + Dexpr1 : DexpB;

float DshadupR1 = DmidR > 0.0f ? 2.0f * (DmidR/_pivot[0]) - log((DmidR/_pivot[0]) * (e * _shad[0] * 2.0f) + 1.0f)/log(e * _shad[0] * 2.0f + 1.0f) : DmidR;
float DshadupR = DmidR < _pivot[0] && _shad[0] > 0.0f ? (DshadupR1 + _shad[0] * (1.0f - DshadupR1)) * _pivot[0] : DmidR;
float DshadupG1 = DmidG > 0.0f ? 2.0f * (DmidG/_pivot[0]) - log((DmidG/_pivot[0]) * (e * _shad[1] * 2.0f) + 1.0f)/log(e * _shad[1] * 2.0f + 1.0f) : DmidG;
float DshadupG = DmidG < _pivot[0] && _shad[1] > 0.0f ? (DshadupG1 + _shad[1] * (1.0f - DshadupG1)) * _pivot[0] : DmidG;
float DshadupB1 = DmidB > 0.0f ? 2.0f * (DmidB/_pivot[0]) - log((DmidB/_pivot[0]) * (e * _shad[2] * 2.0f) + 1.0f)/log(e * _shad[2] * 2.0f + 1.0f) : DmidB;
float DshadupB = DmidB < _pivot[0] && _shad[2] > 0.0f ? (DshadupB1 + _shad[2] * (1.0f - DshadupB1)) * _pivot[0] : DmidB;

float DshaddownR1 = (DshadupR/_pivot[0]) + (_shad[0] * 2.0f * (1.0f - DshadupR/_pivot[0]));
float DshaddownR = DshadupR < _pivot[0] && _shad[0] < 0.0f ? (DshaddownR1 >= 0.0f ? log(DshaddownR1 * (e * _shad[0] * -2.0f) + 1.0f)/log(e * _shad[0] * -2.0f + 1.0f) : DshaddownR1) * _pivot[0] : DshadupR;
float DshaddownG1 = (DshadupG/_pivot[0]) + (_shad[1] * 2.0f * (1.0f - DshadupG/_pivot[0]));
float DshaddownG = DshadupG < _pivot[0] && _shad[1] < 0.0f ? (DshaddownG1 >= 0.0f ? log(DshaddownG1 * (e * _shad[1] * -2.0f) + 1.0f)/log(e * _shad[1] * -2.0f + 1.0f) : DshaddownG1) * _pivot[0] : DshadupG;
float DshaddownB1 = (DshadupB/_pivot[0]) + (_shad[2] * 2.0f * (1.0f - DshadupB/_pivot[0]));
float DshaddownB = DshadupB < _pivot[0] && _shad[2] < 0.0f ? (DshaddownB1 >= 0.0f ? log(DshaddownB1 * (e * _shad[2] * -2.0f) + 1.0f)/log(e * _shad[2] * -2.0f + 1.0f) : DshaddownB1) * _pivot[0] : DshadupB;

float DhighupR1 = ((DshaddownR - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[0] * 2.0f));
float DhighupR = DshaddownR > _pivot[1] && _pivot[1] < 1.0f && _high[0] > 0.0f ? (2.0f * DhighupR1 - log(DhighupR1 * e * _high[0] + 1.0f)/log(e * _high[0] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : DshaddownR;
float DhighupG1 = ((DshaddownG - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[1] * 2.0f));
float DhighupG = DshaddownG > _pivot[1] && _pivot[1] < 1.0f && _high[1] > 0.0f ? (2.0f * DhighupG1 - log(DhighupG1 * e * _high[1] + 1.0f)/log(e * _high[1] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : DshaddownG;
float DhighupB1 = ((DshaddownB - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[2] * 2.0f));
float DhighupB = DshaddownB > _pivot[1] && _pivot[1] < 1.0f && _high[2] > 0.0f ? (2.0f * DhighupB1 - log(DhighupB1 * e * _high[2] + 1.0f)/log(e * _high[2] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : DshaddownB;

float DhighdownR1 = (DhighupR - _pivot[1]) / (1.0f - _pivot[1]);
float DhighdownR = DhighupR > _pivot[1] && _pivot[1] < 1.0f && _high[0] < 0.0f ? log(DhighdownR1 * (e * _high[0] * -2.0f) + 1.0f)/log(e * _high[0] * -2.0f + 1.0f) * (1.0f + _high[0]) * (1.0f - _pivot[1]) + _pivot[1]  : DhighupR;
float DhighdownG1 = (DhighupG - _pivot[1]) / (1.0f - _pivot[1]);
float DhighdownG = DhighupG > _pivot[1] && _pivot[1] < 1.0f && _high[1] < 0.0f ? log(DhighdownG1 * (e * _high[1] * -2.0f) + 1.0f)/log(e * _high[1] * -2.0f + 1.0f) * (1.0f + _high[1]) * (1.0f - _pivot[1]) + _pivot[1]  : DhighupG;
float DhighdownB1 = (DhighupB - _pivot[1]) / (1.0f - _pivot[1]);
float DhighdownB = DhighupB > _pivot[1] && _pivot[1] < 1.0f && _high[2] < 0.0f ? log(DhighdownB1 * (e * _high[2] * -2.0f) + 1.0f)/log(e * _high[2] * -2.0f + 1.0f) * (1.0f + _high[2]) * (1.0f - _pivot[1]) + _pivot[1]  : DhighupB;

float DcontR = (DhighdownR - _pivot[2]) * _cont[0] + _pivot[2];
float DcontG = (DhighdownG - _pivot[2]) * _cont[1] + _pivot[2];
float DcontB = (DhighdownB - _pivot[2]) * _cont[2] + _pivot[2];

float Dluma = DcontR * 0.2126f + DcontG * 0.7152f + DcontB * 0.0722f;
float DsatR = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * Dluma + DcontR * _sat[0];
float DsatG = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * Dluma + DcontG * _sat[1];
float DsatB = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * Dluma + DcontB * _sat[2];

float overlayR = y/(height) >= _pivot[0] && y/(height) <= _pivot[0] + 0.005f ? (fmod(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : DsatR >= (y - 5)/(height) && DsatR <= (y + 5)/(height) ? 1.0f : 0.0f;
float overlayG = y/(height) >= _pivot[1] && y/(height) <= _pivot[1] + 0.005f ? (fmod(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : DsatG >= (y - 5)/(height) && DsatG <= (y + 5)/(height) ? 1.0f : 0.0f;
float overlayB = y/(height) >= _pivot[2] && y/(height) <= _pivot[2] + 0.005f ? (fmod(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : DsatB >= (y - 5)/(height) && DsatB <= (y + 5)/(height) ? 1.0f : 0.0f;

float outR = _display == 2 ? (overlayR == 0.0f ? satR : overlayR) : _display == 1 ? overlayR : satR;
float outG = _display == 2 ? (overlayG == 0.0f ? satG : overlayG) : _display == 1 ? overlayG : satG;
float outB = _display == 2 ? (overlayB == 0.0f ? satB : overlayB) : _display == 1 ? overlayB : satB;

dstPix[0] = outR;
dstPix[1] = outG;
dstPix[2] = outB;
dstPix[3] = srcPix[3];
} else {
for (int c = 0; c < 4; ++c)
{
dstPix[c] = 0;
}}
dstPix += 4;
}}}

void FilmGrade::setSrcImg(OFX::Image* p_SrcImg) {
_srcImg = p_SrcImg;
}

void FilmGrade::setScales(float p_ExpR, float p_ExpG, float p_ExpB, float p_ContR, 
float p_ContG, float p_ContB, float p_SatR, float p_SatG, float p_SatB, float p_ShadR, 
float p_ShadG, float p_ShadB, float p_MidR, float p_MidG, float p_MidB, float p_HighR, 
float p_HighG, float p_HighB, float p_ShadP, float p_HighP, float p_ContP, int p_Display) {
_exp[0] = p_ExpR;
_exp[1] = p_ExpG;
_exp[2] = p_ExpB;
_cont[0] = p_ContR;
_cont[1] = p_ContG;
_cont[2] = p_ContB;
_sat[0] = p_SatR;
_sat[1] = p_SatG;
_sat[2] = p_SatB;
_shad[0] = p_ShadR;
_shad[1] = p_ShadG;
_shad[2] = p_ShadB;
_mid[0] = p_MidR;
_mid[1] = p_MidG;
_mid[2] = p_MidB;
_high[0] = p_HighR;
_high[1] = p_HighG;
_high[2] = p_HighB;
_pivot[0] = p_ShadP;
_pivot[1] = p_HighP;
_pivot[2] = p_ContP;
_display = p_Display;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class FilmGradePlugin : public OFX::ImageEffect
{
public:
explicit FilmGradePlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

void setupAndProcess(FilmGrade &p_FilmGrade, const OFX::RenderArguments& p_Args);

private:

OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;

OFX::RGBParam *m_ExpSwatch;
OFX::RGBParam *m_ContSwatch;
OFX::RGBParam *m_SatSwatch;
OFX::RGBParam *m_ShadSwatch;
OFX::RGBParam *m_MidSwatch;
OFX::RGBParam *m_HighSwatch;

OFX::DoubleParam* m_Exp;
OFX::DoubleParam* m_ExpR;
OFX::DoubleParam* m_ExpG;
OFX::DoubleParam* m_ExpB;
OFX::DoubleParam* m_Cont;
OFX::DoubleParam* m_ContR;
OFX::DoubleParam* m_ContG;
OFX::DoubleParam* m_ContB;
OFX::DoubleParam* m_Sat;
OFX::DoubleParam* m_SatR;
OFX::DoubleParam* m_SatG;
OFX::DoubleParam* m_SatB;
OFX::DoubleParam* m_Shad;
OFX::DoubleParam* m_ShadR;
OFX::DoubleParam* m_ShadG;
OFX::DoubleParam* m_ShadB;
OFX::DoubleParam* m_Mid;
OFX::DoubleParam* m_MidR;
OFX::DoubleParam* m_MidG;
OFX::DoubleParam* m_MidB;
OFX::DoubleParam* m_High;
OFX::DoubleParam* m_HighR;
OFX::DoubleParam* m_HighG;
OFX::DoubleParam* m_HighB;
OFX::DoubleParam* m_ShadP;
OFX::DoubleParam* m_ShadPP;
OFX::DoubleParam* m_HighP;
OFX::DoubleParam* m_HighPP;
OFX::DoubleParam* m_ContP;
OFX::DoubleParam* m_ContPP;
OFX::ChoiceParam* m_DisplayGraph;
OFX::StringParam* m_Path;
OFX::StringParam* m_Name;
OFX::PushButtonParam* m_Info;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
};

FilmGradePlugin::FilmGradePlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

m_ExpSwatch = fetchRGBParam("expSwatch");
m_ContSwatch = fetchRGBParam("contSwatch");
m_SatSwatch = fetchRGBParam("satSwatch");
m_ShadSwatch = fetchRGBParam("shadSwatch");
m_MidSwatch = fetchRGBParam("midSwatch");
m_HighSwatch = fetchRGBParam("highSwatch");

m_Exp = fetchDoubleParam("exp");
m_ExpR = fetchDoubleParam("expR");
m_ExpG = fetchDoubleParam("expG");
m_ExpB = fetchDoubleParam("expB");
m_Cont = fetchDoubleParam("cont");
m_ContR = fetchDoubleParam("contR");
m_ContG = fetchDoubleParam("contG");
m_ContB = fetchDoubleParam("contB");
m_Sat = fetchDoubleParam("sat");
m_SatR = fetchDoubleParam("satR");
m_SatG = fetchDoubleParam("satG");
m_SatB = fetchDoubleParam("satB");
m_Shad = fetchDoubleParam("shad");
m_ShadR = fetchDoubleParam("shadR");
m_ShadG = fetchDoubleParam("shadG");
m_ShadB = fetchDoubleParam("shadB");
m_Mid = fetchDoubleParam("mid");
m_MidR = fetchDoubleParam("midR");
m_MidG = fetchDoubleParam("midG");
m_MidB = fetchDoubleParam("midB");
m_High = fetchDoubleParam("high");
m_HighR = fetchDoubleParam("highR");
m_HighG = fetchDoubleParam("highG");
m_HighB = fetchDoubleParam("highB");
m_ShadP = fetchDoubleParam("shadP");
m_ShadPP = fetchDoubleParam("shadPP");
m_HighP = fetchDoubleParam("highP");
m_HighPP = fetchDoubleParam("highPP");
m_ContP = fetchDoubleParam("contP");
m_ContPP = fetchDoubleParam("contPP");
m_DisplayGraph = fetchChoiceParam(kParamDisplayGraph);
m_Path = fetchStringParam("path");
m_Name = fetchStringParam("name");
m_Info = fetchPushButtonParam("info");
m_Button1 = fetchPushButtonParam("button1");
m_Button2 = fetchPushButtonParam("button2");
}

void FilmGradePlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
{
FilmGrade imageScaler(*this);
setupAndProcess(imageScaler, p_Args);
}
else
{
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool FilmGradePlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
float exp = m_Exp->getValueAtTime(p_Args.time);
float expR = m_ExpR->getValueAtTime(p_Args.time);
float expG = m_ExpG->getValueAtTime(p_Args.time);
float expB = m_ExpB->getValueAtTime(p_Args.time);
float cont = m_Cont->getValueAtTime(p_Args.time);
float contR = m_ContR->getValueAtTime(p_Args.time);
float contG = m_ContG->getValueAtTime(p_Args.time);
float contB = m_ContB->getValueAtTime(p_Args.time);
float sat = m_Sat->getValueAtTime(p_Args.time);
float satR = m_SatR->getValueAtTime(p_Args.time);
float satG = m_SatG->getValueAtTime(p_Args.time);
float satB = m_SatB->getValueAtTime(p_Args.time);
float shad = m_Shad->getValueAtTime(p_Args.time);
float shadR = m_ShadR->getValueAtTime(p_Args.time);
float shadG = m_ShadG->getValueAtTime(p_Args.time);
float shadB = m_ShadB->getValueAtTime(p_Args.time);
float mid = m_Mid->getValueAtTime(p_Args.time);
float midR = m_MidR->getValueAtTime(p_Args.time);
float midG = m_MidG->getValueAtTime(p_Args.time);
float midB = m_MidB->getValueAtTime(p_Args.time);
float high = m_High->getValueAtTime(p_Args.time);
float highR = m_HighR->getValueAtTime(p_Args.time);
float highG = m_HighG->getValueAtTime(p_Args.time);
float highB = m_HighB->getValueAtTime(p_Args.time);
int aDisplay;
m_DisplayGraph->getValueAtTime(p_Args.time, aDisplay);

if ((exp == 0.0f) && (expR == 0.0f) && (expG == 0.0f) && (expB == 0.0f) && (cont == 1.0f) && (contR == 1.0f) && (contG == 1.0f) && (contB == 1.0f) && 
(sat == 1.0f) && (satR == 1.0f) && (satG == 1.0f) && (satB == 1.0f) && (shad == 0.0f) && (shadR == 0.0f) && (shadG == 0.0f) && (shadB == 0.0f) && 
(mid == 0.0f) && (midR == 0.0f) && (midG == 0.0f) && (midB == 0.0f) && (high == 0.0f) && (highR == 0.0f) && (highG == 0.0f) && (highB == 0.0f) && (aDisplay == 0))
{
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void FilmGradePlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if(p_ParamName == "info")
{
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

if(p_ParamName == "button1")
{
float expR = m_ExpR->getValueAtTime(p_Args.time);
float expG = m_ExpG->getValueAtTime(p_Args.time);
float expB = m_ExpB->getValueAtTime(p_Args.time);
float contR = m_ContR->getValueAtTime(p_Args.time);
float contG = m_ContG->getValueAtTime(p_Args.time);
float contB = m_ContB->getValueAtTime(p_Args.time);
float satR = m_SatR->getValueAtTime(p_Args.time);
float satG = m_SatG->getValueAtTime(p_Args.time);
float satB = m_SatB->getValueAtTime(p_Args.time);
float shadR = m_ShadR->getValueAtTime(p_Args.time);
float shadG = m_ShadG->getValueAtTime(p_Args.time);
float shadB = m_ShadB->getValueAtTime(p_Args.time);
float midR = m_MidR->getValueAtTime(p_Args.time);
float midG = m_MidG->getValueAtTime(p_Args.time);
float midB = m_MidB->getValueAtTime(p_Args.time);
float highR = m_HighR->getValueAtTime(p_Args.time);
float highG = m_HighG->getValueAtTime(p_Args.time);
float highB = m_HighB->getValueAtTime(p_Args.time);
float shadP = m_ShadP->getValueAtTime(p_Args.time) / 1023.0;
float highP = m_HighP->getValueAtTime(p_Args.time) / 1023.0;
float contP = m_ContP->getValueAtTime(p_Args.time) / 1023.0;
int display;
m_DisplayGraph->getValueAtTime(p_Args.time, display);

string PATH;
m_Path->getValue(PATH);

string NAME;
m_Name->getValue(NAME);

OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
if (reply == OFX::Message::eMessageReplyYes) {

FILE * pFile;

pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
if (pFile != NULL) {

fprintf (pFile, "// FilmGrade DCTL export\n" \
"\n" \
"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
"{\n" \
"float p_ExpR = %ff;\n" \
"float p_ExpG = %ff;\n" \
"float p_ExpB = %ff;\n" \
"float p_ContR = %ff;\n" \
"float p_ContG = %ff;\n" \
"float p_ContB = %ff;\n" \
"float p_SatR = %ff;\n" \
"float p_SatG = %ff;\n" \
"float p_SatB = %ff;\n" \
"float p_ShadR = %ff;\n" \
"float p_ShadG = %ff;\n" \
"float p_ShadB = %ff;\n" \
"float p_MidR = %ff;\n" \
"float p_MidG = %ff;\n" \
"float p_MidB = %ff;\n" \
"float p_HighR = %ff;\n" \
"float p_HighG = %ff;\n" \
"float p_HighB = %ff;\n" \
"float p_ShadP = %ff;\n" \
"float p_HighP = %ff;\n" \
"float p_ContP = %ff;\n" \
"int p_Display = %d;\n" \
"\n" \
"float e = 2.718281828459045;\n" \
"float pie = 3.141592653589793;\n" \
"\n" \
"float width = p_Width;\n" \
"float height = p_Height;    	\n" \
"\n" \
"float Red = p_Display != 1.0f ? p_R : p_X / width;\n" \
"float Green = p_Display != 1.0f ? p_G : p_X / width;\n" \
"float Blue = p_Display != 1.0f ? p_B : p_X / width;\n" \
"\n" \
"float expR = Red + p_ExpR/100.0f;\n" \
"float expG = Green + p_ExpG/100.0f;\n" \
"float expB = Blue + p_ExpB/100.0f;\n" \
"\n" \
"float expr1 = (p_ShadP / 2.0f) - (1.0f - p_HighP)/4.0f;\n" \
"float expr2 = (1.0f - (1.0f - p_HighP)/2.0f) + (p_ShadP / 4.0f);\n" \
"float expr3R = (expR - expr1) / (expr2 - expr1);\n" \
"float expr3G = (expG - expr1) / (expr2 - expr1);\n" \
"float expr3B = (expB - expr1) / (expr2 - expr1);\n" \
"float expr4 =  p_ContP < 0.5f ? 0.5f - (0.5f - p_ContP)/2.0f : 0.5f + (p_ContP - 0.5f)/2.0f;\n" \
"float expr5R = expr3R > expr4 ? (expr3R - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3R /(2.0f*expr4);\n" \
"float expr5G = expr3G > expr4 ? (expr3G - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3G /(2.0f*expr4);\n" \
"float expr5B = expr3B > expr4 ? (expr3B - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3B /(2.0f*expr4);\n" \
"float expr6R = (((_sinf(2.0f * pie * (expr5R -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidR*4.0f) + expr3R;\n" \
"float expr6G = (((_sinf(2.0f * pie * (expr5G -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidG*4.0f) + expr3G;\n" \
"float expr6B = (((_sinf(2.0f * pie * (expr5B -1.0f/4.0f)) + 1.0f) / 20.0f) * p_MidB*4.0f) + expr3B;\n" \
"float midR = expR >= expr1 && expR <= expr2 ? expr6R * (expr2 - expr1) + expr1 : expR;\n" \
"float midG = expG >= expr1 && expG <= expr2 ? expr6G * (expr2 - expr1) + expr1 : expG;\n" \
"float midB = expB >= expr1 && expB <= expr2 ? expr6B * (expr2 - expr1) + expr1 : expB;\n" \
"\n" \
"float shadupR1 = midR > 0.0f ? 2.0f * (midR/p_ShadP) - _logf((midR/p_ShadP) * (e * p_ShadR * 2.0f) + 1.0f)/_logf(e * p_ShadR * 2.0f + 1.0f) : midR;\n" \
"float shadupR = midR < p_ShadP && p_ShadR > 0.0f ? (shadupR1 + p_ShadR * (1.0f - shadupR1)) * p_ShadP : midR;\n" \
"float shadupG1 = midG > 0.0f ? 2.0f * (midG/p_ShadP) - _logf((midG/p_ShadP) * (e * p_ShadG * 2.0f) + 1.0f)/_logf(e * p_ShadG * 2.0f + 1.0f) : midG;\n" \
"float shadupG = midG < p_ShadP && p_ShadG > 0.0f ? (shadupG1 + p_ShadG * (1.0f - shadupG1)) * p_ShadP : midG;\n" \
"float shadupB1 = midB > 0.0f ? 2.0f * (midB/p_ShadP) - _logf((midB/p_ShadP) * (e * p_ShadB * 2.0f) + 1.0f)/_logf(e * p_ShadB * 2.0f + 1.0f) : midB;\n" \
"float shadupB = midB < p_ShadP && p_ShadB > 0.0f ? (shadupB1 + p_ShadB * (1.0f - shadupB1)) * p_ShadP : midB;\n" \
"\n" \
"float shaddownR1 = shadupR/p_ShadP + p_ShadR*2 * (1.0f - shadupR/p_ShadP);\n" \
"float shaddownR = shadupR < p_ShadP && p_ShadR < 0.0f ? (shaddownR1 >= 0.0f ? _logf(shaddownR1 * (e * p_ShadR * -2.0f) + 1.0f)/_logf(e * p_ShadR * -2.0f + 1.0f) : shaddownR1) * p_ShadP : shadupR;\n" \
"float shaddownG1 = shadupG/p_ShadP + p_ShadG*2 * (1.0f - shadupG/p_ShadP);\n" \
"float shaddownG = shadupG < p_ShadP && p_ShadG < 0.0f ? (shaddownG1 >= 0.0f ? _logf(shaddownG1 * (e * p_ShadG * -2.0f) + 1.0f)/_logf(e * p_ShadG * -2.0f + 1.0f) : shaddownG1) * p_ShadP : shadupG;\n" \
"float shaddownB1 = shadupB/p_ShadP + p_ShadB*2 * (1.0f - shadupB/p_ShadP);\n" \
"float shaddownB = shadupB < p_ShadP && p_ShadB < 0.0f ? (shaddownB1 >= 0.0f ? _logf(shaddownB1 * (e * p_ShadB * -2.0f) + 1.0f)/_logf(e * p_ShadB * -2.0f + 1.0f) : shaddownB1) * p_ShadP : shadupB;\n" \
"\n" \
"float highupR1 = ((shaddownR - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighR * 2.0f));\n" \
"float highupR = shaddownR > p_HighP && p_HighP < 1.0f && p_HighR > 0.0f ? (2.0f * highupR1 - _logf(highupR1 * e * p_HighR + 1.0f)/_logf(e * p_HighR + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddownR;\n" \
"float highupG1 = ((shaddownG - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighG * 2.0f));\n" \
"float highupG = shaddownG > p_HighP && p_HighP < 1.0f && p_HighG > 0.0f ? (2.0f * highupG1 - _logf(highupG1 * e * p_HighG + 1.0f)/_logf(e * p_HighG + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddownG;\n" \
"float highupB1 = ((shaddownB - p_HighP) / (1.0f - p_HighP)) * (1.0f + (p_HighB * 2.0f));\n" \
"float highupB = shaddownB > p_HighP && p_HighP < 1.0f && p_HighB > 0.0f ? (2.0f * highupB1 - _logf(highupB1 * e * p_HighB + 1.0f)/_logf(e * p_HighB + 1.0f)) * (1.0f - p_HighP) + p_HighP : shaddownB;\n" \
"\n" \
"float highdownR1 = (highupR - p_HighP) / (1.0f - p_HighP);\n" \
"float highdownR = highupR > p_HighP && p_HighP < 1.0f && p_HighR < 0.0f ? _logf(highdownR1 * (e * p_HighR * -2.0f) + 1.0f)/_logf(e * p_HighR * -2.0f + 1.0f) * (1.0f + p_HighR) * (1.0f - p_HighP) + p_HighP : highupR;\n" \
"float highdownG1 = (highupG - p_HighP) / (1.0f - p_HighP);\n" \
"float highdownG = highupG > p_HighP && p_HighP < 1.0f && p_HighG < 0.0f ? _logf(highdownG1 * (e * p_HighG * -2.0f) + 1.0f)/_logf(e * p_HighG * -2.0f + 1.0f) * (1.0f + p_HighG) * (1.0f - p_HighP) + p_HighP : highupG;\n" \
"float highdownB1 = (highupB - p_HighP) / (1.0f - p_HighP);\n" \
"float highdownB = highupB > p_HighP && p_HighP < 1.0f && p_HighB < 0.0f ? _logf(highdownB1 * (e * p_HighB * -2.0f) + 1.0f)/_logf(e * p_HighB * -2.0f + 1.0f) * (1.0f + p_HighB) * (1.0f - p_HighP) + p_HighP : highupB;\n" \
"\n" \
"float contR = (highdownR - p_ContP) * p_ContR + p_ContP;\n" \
"float contG = (highdownG - p_ContP) * p_ContG + p_ContP;\n" \
"float contB = (highdownB - p_ContP) * p_ContB + p_ContP;\n" \
"\n" \
"float luma = contR * 0.2126f + contG * 0.7152f + contB * 0.0722f;\n" \
"float satR = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * luma + contR * p_SatR;\n" \
"float satG = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * luma + contG * p_SatG;\n" \
"float satB = (1.0f - (p_SatR*0.2126f + p_SatG* 0.7152f + p_SatB*0.0722f)) * luma + contB * p_SatB;\n" \
"\n" \
"float r = p_Display != 1 ? satR : p_Y / height >= p_ShadP && p_Y / height <= p_ShadP + 0.005f ? (_fmod(p_X, 2.0f) != 0.0f ? 1.0f : 0.0f) : satR >= (p_Y - 5) / height && satR <= (p_Y + 5) / height ? 1.0f : 0.0f;\n" \
"float g = p_Display != 1 ? satG : p_Y / height >= p_HighP && p_Y / height <= p_HighP + 0.005f ? (_fmod(p_X, 2.0f) != 0.0f ? 1.0f : 0.0f) : satG >= (p_Y - 5) / height && satG <= (p_Y + 5) / height ? 1.0f : 0.0f;\n" \
"float b = p_Display != 1 ? satB : p_Y / height >= p_ContP && p_Y / height <= p_ContP + 0.005f ? (_fmod(p_X, 2.0f) != 0.0f ? 1.0f : 0.0f) : satB >= (p_Y - 5) / height && satB <= (p_Y + 5) / height ? 1.0f : 0.0f;\n" \
"\n" \
"return make_float3(r, g, b);	\n" \
"}\n", expR, expG, expB, contR, contG, contB, satR, satG, satB, shadR, shadG, 
shadB, midR, midG, midB, highR, highG, highB, shadP, highP, contP, display);
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
}}}

if(p_ParamName == "button2")
{
float expR = m_ExpR->getValueAtTime(p_Args.time);
float expG = m_ExpG->getValueAtTime(p_Args.time);
float expB = m_ExpB->getValueAtTime(p_Args.time);
float contR = m_ContR->getValueAtTime(p_Args.time);
float contG = m_ContG->getValueAtTime(p_Args.time);
float contB = m_ContB->getValueAtTime(p_Args.time);
float satR = m_SatR->getValueAtTime(p_Args.time);
float satG = m_SatG->getValueAtTime(p_Args.time);
float satB = m_SatB->getValueAtTime(p_Args.time);
float shadR = m_ShadR->getValueAtTime(p_Args.time);
float shadG = m_ShadG->getValueAtTime(p_Args.time);
float shadB = m_ShadB->getValueAtTime(p_Args.time);
float midR = m_MidR->getValueAtTime(p_Args.time);
float midG = m_MidG->getValueAtTime(p_Args.time);
float midB = m_MidB->getValueAtTime(p_Args.time);
float highR = m_HighR->getValueAtTime(p_Args.time);
float highG = m_HighG->getValueAtTime(p_Args.time);
float highB = m_HighB->getValueAtTime(p_Args.time);
float shadP = m_ShadP->getValueAtTime(p_Args.time) / 1023.0;
float highP = m_HighP->getValueAtTime(p_Args.time) / 1023.0;
float contP = m_ContP->getValueAtTime(p_Args.time) / 1023.0;
int display;
m_DisplayGraph->getValueAtTime(p_Args.time, display);


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
" name FilmGrade\n" \
" xpos -165\n" \
" ypos 247\n" \
"}\n" \
" Input {\n" \
"  inputs 0\n" \
"  name Input1\n" \
"  xpos 256\n" \
"  ypos 310\n" \
" }\n" \
"set N6005e3e0 [stack 0]\n" \
" Expression {\n" \
"  expr0 \"x / width\"\n" \
"  expr1 \"x / width\"\n" \
"  expr2 \"x / width\"\n" \
"  name Display1\n" \
"  xpos 326\n" \
"  ypos 346\n" \
" }\n" \
"push $N6005e3e0\n" \
" Switch {\n" \
"  inputs 2\n" \
"  which %d\n" \
"  name Display_switch\n" \
"  xpos 256\n" \
"  ypos 388\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ExpR\n" \
"  temp_expr0 \"%f / 100\"\n" \
"  temp_name1 ExpG\n" \
"  temp_expr1 \"%f / 100\"\n" \
"  temp_name2 ExpB\n" \
"  temp_expr2 \"%f / 100\"\n" \
"  expr0 \"r + ExpR\"\n" \
"  expr1 \"g + ExpG\"\n" \
"  expr2 \"b + ExpB\"\n" \
"  name Exposure\n" \
"  xpos 258\n" \
"  ypos 423\n" \
" }\n" \
"set N600e5860 [stack 0]\n" \
"push $N600e5860\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 \"%f\"\n" \
"  temp_name1 HighP\n" \
"  temp_expr1 \"%f\"\n" \
"  temp_name2 expr1\n" \
"  temp_expr2 \"(ShadP / 2) - (1 - HighP) / 4\"\n" \
"  temp_name3 expr2\n" \
"  temp_expr3 \"(1 - (1 - HighP) / 2) + (ShadP / 4)\"\n" \
"  expr0 \"(r - expr1) / (expr2 - expr1)\"\n" \
"  expr1 \"(g - expr1) / (expr2 - expr1)\"\n" \
"  expr2 \"(b - expr1) / (expr2 - expr1)\"\n" \
"  name Pivots\n" \
"  xpos 140\n" \
"  ypos 455\n" \
" }\n" \
"set N600a6650 [stack 0]\n" \
" Expression {\n" \
"  temp_name0 ContP\n" \
"  temp_expr0 \"%f\"\n" \
"  temp_name1 expr4\n" \
"  temp_expr1 \"ContP < 0.5 ? 0.5 - (0.5 - ContP) / 2 : 0.5 + (ContP - 0.5) / 2\"\n" \
"  expr0 \"r > expr4 ? (r - expr4) / (2 - 2 * expr4) + 0.5 : r / (2 * expr4)\"\n" \
"  expr1 \"g > expr4 ? (g - expr4) / (2 - 2 * expr4) + 0.5 : g / (2 * expr4)\"\n" \
"  expr2 \"b > expr4 ? (b - expr4) / (2 - 2 * expr4) + 0.5 : b / (2 * expr4)\"\n" \
"  name Mids_1\n" \
"  xpos 45\n" \
"  ypos 488\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 MidR\n" \
"  temp_expr0 %f\n" \
"  temp_name1 MidG\n" \
"  temp_expr1 %f\n" \
"  temp_name2 MidB\n" \
"  temp_expr2 %f\n" \
"  temp_name3 pie\n" \
"  temp_expr3 3.141592653589793\n" \
"  expr0 \"((sin(2 * pie * (r - 1/4)) + 1) / 20) * MidR * 4\"\n" \
"  expr1 \"((sin(2 * pie * (g - 1/4)) + 1) / 20) * MidG * 4\"\n" \
"  expr2 \"((sin(2 * pie * (b - 1/4)) + 1) / 20) * MidB * 4\"\n" \
"  name Mids_2\n" \
"  xpos 45\n" \
"  ypos 512\n" \
" }\n" \
"push $N600a6650\n" \
" Merge2 {\n" \
"  inputs 2\n" \
"  operation plus\n" \
"  Achannels rgb\n" \
"  Bchannels rgb\n" \
"  output rgb\n" \
"  name Mids_3\n" \
"  xpos 86\n" \
"  ypos 552\n" \
" }\n" \
"set N60030410 [stack 0]\n" \
" ShuffleCopy {\n" \
"  inputs 2\n" \
"  in rgb\n" \
"  in2 rgb\n" \
"  green red\n" \
"  blue red\n" \
"  out rgb\n" \
"  name Mids_R1\n" \
"  xpos 158\n" \
"  ypos 614\n" \
" }\n", display, expR, expG, expB, shadP, highP, contP, midR, midG, midB);
fprintf (pFile, " Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 \"%f\"\n" \
"  temp_name1 HighP\n" \
"  temp_expr1 \"%f\"\n" \
"  temp_name2 expr1\n" \
"  temp_expr2 \"(ShadP / 2) - (1 - HighP) / 4\"\n" \
"  temp_name3 expr2\n" \
"  temp_expr3 \"(1 - (1 - HighP) / 2) + (ShadP / 4)\"\n" \
"  expr0 \"g >= expr1 && r <= expr2 ? r * (expr2 - expr1) + expr1 : g\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name Mids_R2\n" \
"  xpos 158\n" \
"  ypos 638\n" \
" }\n" \
"push $N600e5860\n" \
"push $N60030410\n" \
" ShuffleCopy {\n" \
"  inputs 2\n" \
"  in rgb\n" \
"  in2 rgb\n" \
"  red green2\n" \
"  green green\n" \
"  blue green\n" \
"  out rgb\n" \
"  name Mids_G1\n" \
"  xpos 258\n" \
"  ypos 615\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 \"%f\"\n" \
"  temp_name1 HighP\n" \
"  temp_expr1 \"%f\"\n" \
"  temp_name2 expr1\n" \
"  temp_expr2 \"(ShadP / 2) - (1 - HighP) / 4\"\n" \
"  temp_name3 expr2\n" \
"  temp_expr3 \"(1 - (1 - HighP) / 2) + (ShadP / 4)\"\n" \
"  expr0 0\n" \
"  expr1 \"g >= expr1 && r <= expr2 ? r * (expr2 - expr1) + expr1 : g\"\n" \
"  expr2 0\n" \
"  name Mids_G2\n" \
"  xpos 258\n" \
"  ypos 639\n" \
" }\n" \
" ShuffleCopy {\n" \
"  inputs 2\n" \
"  in rgb\n" \
"  in2 rgb\n" \
"  red red\n" \
"  blue black\n" \
"  out rgb\n" \
"  name Mids_4\n" \
"  xpos 204\n" \
"  ypos 673\n" \
" }\n" \
"push $N600e5860\n" \
"push $N60030410\n" \
" ShuffleCopy {\n" \
"  inputs 2\n" \
"  in rgb\n" \
"  in2 rgb\n" \
"  red blue2\n" \
"  green blue\n" \
"  blue blue\n" \
"  out rgb\n" \
"  name Mids_B1\n" \
"  xpos 358\n" \
"  ypos 616\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 \"%f\"\n" \
"  temp_name1 HighP\n" \
"  temp_expr1 \"%f\"\n" \
"  temp_name2 expr1\n" \
"  temp_expr2 \"(ShadP / 2) - (1 - HighP) / 4\"\n" \
"  temp_name3 expr2\n" \
"  temp_expr3 \"(1 - (1 - HighP) / 2) + (ShadP / 4)\"\n" \
"  expr0 0\n" \
"  expr1 0\n" \
"  expr2 \"g >= expr1 && r <= expr2 ? r * (expr2 - expr1) + expr1 : g\"\n" \
"  name Mids_B2\n" \
"  xpos 358\n" \
"  ypos 640\n" \
" }\n" \
" ShuffleCopy {\n" \
"  inputs 2\n" \
"  in rgb\n" \
"  in2 rgb\n" \
"  red red\n" \
"  green green\n" \
"  out rgb\n" \
"  name Mids_5\n" \
"  xpos 250\n" \
"  ypos 710\n" \
" }\n" \
"set N600373e0 [stack 0]\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 \"%f\"\n" \
"  temp_name1 ShadR\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 r\n" \
"  expr1 \"r > 0 ? 2 * (r / ShadP) - log((r / ShadP) * (e * ShadR * 2) + 1) / log(e * ShadR * 2 + 1) : r\"\n" \
"  expr2 0\n" \
"  name ShadupR1\n" \
"  xpos 152\n" \
"  ypos 746\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 \"%f\"\n" \
"  temp_name1 ShadR\n" \
"  temp_expr1 %f\n" \
"  expr0 \"r < ShadP && ShadR > 0 ? (g + ShadR * (1 - g)) * ShadP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name ShadupR2\n" \
"  xpos 152\n" \
"  ypos 770\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 \"%f\"\n" \
"  temp_name1 ShadR\n" \
"  temp_expr1 %f\n" \
"  expr0 r\n" \
"  expr1 \"r / ShadP + ShadR * 2 * (1 - r / ShadP)\"\n" \
"  expr2 0\n" \
"  name ShaddownR1\n" \
"  xpos 152\n" \
"  ypos 794\n" \
" }\n", shadP, highP, shadP, highP, shadP, highP, shadP, shadR, shadP, shadR, shadP, shadR);
fprintf (pFile, " Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadR\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r < ShadP && ShadR < 0 ? (g >= 0 ? log(g * (e * ShadR * -2) + 1) / log(e * ShadR * -2 + 1) : g) * ShadP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name ShaddownR2\n" \
"  xpos 152\n" \
"  ypos 818\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighR\n" \
"  temp_expr1 %f\n" \
"  expr0 r\n" \
"  expr1 \"((r - HighP) / (1 - HighP)) * (1 + (HighR * 2));\"\n" \
"  expr2 0\n" \
"  name HighupR1\n" \
"  xpos 152\n" \
"  ypos 842\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighR\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r > HighP && HighP < 1 && HighR > 0 ? (2 * g - log(g * e * HighR + 1) / log(e * HighR + 1)) * (1 - HighP) + HighP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name HighupR2\n" \
"  xpos 152\n" \
"  ypos 866\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  expr0 r\n" \
"  expr1 \"(r - HighP) / (1 - HighP)\"\n" \
"  expr2 0\n" \
"  name HighdownR1\n" \
"  xpos 152\n" \
"  ypos 890\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighR\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r > HighP && HighP < 1 && HighR < 0 ? log(g * (e * HighR * -2) + 1) / log(e * HighR * -2 + 1) * (1 + HighR) * (1 - HighP) + HighP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name HighdownR2\n" \
"  xpos 152\n" \
"  ypos 914\n" \
" }\n" \
"push $N600373e0\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadG\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 g\n" \
"  expr1 \"g > 0 ? 2 * (g / ShadP) - log((g / ShadP) * (e * ShadG * 2) + 1) / log(e * ShadG * 2 + 1) : g\"\n" \
"  expr2 0\n" \
"  name ShadupG1\n" \
"  xpos 250\n" \
"  ypos 747\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadG\n" \
"  temp_expr1 %f\n" \
"  expr0 \"r < ShadP && ShadG > 0 ? (g + ShadG * (1 - g)) * ShadP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name ShadupG2\n" \
"  xpos 250\n" \
"  ypos 771\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadG\n" \
"  temp_expr1 %f\n" \
"  expr0 r\n" \
"  expr1 \"r / ShadP + ShadG * 2 * (1 - r / ShadP)\"\n" \
"  expr2 0\n" \
"  name ShaddownG1\n" \
"  xpos 250\n" \
"  ypos 795\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadG\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r < ShadP && ShadG < 0 ? (g >= 0 ? log(g * (e * ShadG * -2) + 1) / log(e * ShadG * -2 + 1) : g) * ShadP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name ShaddownG2\n" \
"  xpos 250\n" \
"  ypos 819\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighG\n" \
"  temp_expr1 %f\n" \
"  expr0 r\n" \
"  expr1 \"((r - HighP) / (1 - HighP)) * (1 + (HighG * 2));\"\n" \
"  expr2 0\n" \
"  name HighupG1\n" \
"  xpos 250\n" \
"  ypos 843\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighG\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r > HighP && HighP < 1 && HighG > 0 ? (2 * g - log(g * e * HighG + 1) / log(e * HighG + 1)) * (1 - HighP) + HighP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name HighupG2\n" \
"  xpos 250\n" \
"  ypos 867\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  expr0 r\n" \
"  expr1 \"(r - HighP) / (1 - HighP)\"\n" \
"  expr2 0\n" \
"  name HighdownG1\n" \
"  xpos 250\n" \
"  ypos 891\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighG\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r > HighP && HighP < 1 && HighG < 0 ? log(g * (e * HighG * -2) + 1) / log(e * HighG * -2 + 1) * (1 + HighG) * (1 - HighP) + HighP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name HighdownG2\n" \
"  xpos 250\n" \
"  ypos 915\n" \
" }\n" \
" ShuffleCopy {\n" \
"  inputs 2\n" \
"  in rgb\n" \
"  in2 rgb\n" \
"  red red\n" \
"  green red2\n" \
"  blue black\n" \
"  out rgb\n" \
"  name RG\n" \
"  xpos 217\n" \
"  ypos 957\n" \
" }\n" \
"push $N600373e0\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadB\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 b\n" \
"  expr1 \"b > 0 ? 2 * (b / ShadP) - log((b / ShadP) * (e * ShadB * 2) + 1) / log(e * ShadB * 2 + 1) : b\"\n" \
"  expr2 0\n" \
"  name ShadupB1\n" \
"  xpos 349\n" \
"  ypos 747\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadB\n" \
"  temp_expr1 %f\n" \
"  expr0 \"r < ShadP && ShadB > 0 ? (g + ShadB * (1 - g)) * ShadP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name ShadupB2\n" \
"  xpos 349\n" \
"  ypos 771\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadB\n" \
"  temp_expr1 %f\n" \
"  expr0 r\n" \
"  expr1 \"r / ShadP + ShadB * 2 * (1 - r / ShadP)\"\n" \
"  expr2 0\n" \
"  name ShaddownB1\n" \
"  xpos 349\n" \
"  ypos 795\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ShadB\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r < ShadP && ShadB < 0 ? (g >= 0 ? log(g * (e * ShadB * -2) + 1) / log(e * ShadB * -2 + 1) : g) * ShadP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name ShaddownB2\n" \
"  xpos 349\n" \
"  ypos 819\n" \
" }\n", shadP, shadR, highP, highR, highP, highR,
highP, highP, highR, shadP, shadG, shadP, shadG, shadP, shadG,
shadP, shadG, highP, highG, highP, highG, highP, highP, highG,
shadP, shadB, shadP, shadB, shadP, shadB, shadP, shadB);
fprintf (pFile, " Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighB\n" \
"  temp_expr1 %f\n" \
"  expr0 r\n" \
"  expr1 \"((r - HighP) / (1 - HighP)) * (1 + (HighB * 2));\"\n" \
"  expr2 0\n" \
"  name HighupB1\n" \
"  xpos 349\n" \
"  ypos 843\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighB\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r > HighP && HighP < 1 && HighB > 0 ? (2 * g - log(g * e * HighB + 1) / log(e * HighB + 1)) * (1 - HighP) + HighP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name HighupB2\n" \
"  xpos 349\n" \
"  ypos 867\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  expr0 r\n" \
"  expr1 \"(r - HighP) / (1 - HighP)\"\n" \
"  expr2 0\n" \
"  name HighdownB1\n" \
"  xpos 349\n" \
"  ypos 891\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 HighP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighB\n" \
"  temp_expr1 %f\n" \
"  temp_name2 e\n" \
"  temp_expr2 2.718281828459045\n" \
"  expr0 \"r > HighP && HighP < 1 && HighB < 0 ? log(g * (e * HighB * -2) + 1) / log(e * HighB * -2 + 1) * (1 + HighB) * (1 - HighP) + HighP : r\"\n" \
"  expr1 0\n" \
"  expr2 0\n" \
"  name HighdownB2\n" \
"  xpos 349\n" \
"  ypos 915\n" \
" }\n" \
" ShuffleCopy {\n" \
"  inputs 2\n" \
"  in rgb\n" \
"  in2 rgb\n" \
"  red red\n" \
"  green green\n" \
"  blue red2\n" \
"  out rgb\n" \
"  name RGB\n" \
"  xpos 262\n" \
"  ypos 998\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 ContP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 ContR\n" \
"  temp_expr1 %f\n" \
"  temp_name2 ContG\n" \
"  temp_expr2 %f\n" \
"  temp_name3 ContB\n" \
"  temp_expr3 %f\n" \
"  expr0 \"(r - ContP) * ContR + ContP\"\n" \
"  expr1 \"(g - ContP) * ContG + ContP\"\n" \
"  expr2 \"(b - ContP) * ContB + ContP\"\n" \
"  name Contrast\n" \
"  selected true\n" \
"  xpos 262\n" \
"  ypos 1022\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 SatR\n" \
"  temp_expr0 %f\n" \
"  temp_name1 SatG\n" \
"  temp_expr1 %f\n" \
"  temp_name2 SatB\n" \
"  temp_expr2 %f\n" \
"  temp_name3 luma\n" \
"  temp_expr3 \"r * 0.2126 + g * 0.7152 + b * 0.0722\"\n" \
"  expr0 \"(1 - (SatR * 0.2126 + SatG * 0.7152 + SatB * 0.0722)) * luma + r * SatR\"\n" \
"  expr1 \"(1 - (SatR * 0.2126 + SatG * 0.7152 + SatB * 0.0722)) * luma + g * SatG\"\n" \
"  expr2 \"(1 - (SatR * 0.2126 + SatG * 0.7152 + SatB * 0.0722)) * luma + b * SatB\"\n" \
"  name Saturation\n" \
"  xpos 262\n" \
"  ypos 1046\n" \
" }\n" \
"set N50a1c140 [stack 0]\n" \
" Expression {\n" \
"  temp_name0 ShadP\n" \
"  temp_expr0 %f\n" \
"  temp_name1 HighP\n" \
"  temp_expr1 %f\n" \
"  temp_name2 ContP\n" \
"  temp_expr2 %f\n" \
"  expr0 \"y / height >= ShadP && y / height <= ShadP + 0.005 ? (fmod(x, 5) != 0 ? 1 : 0) : r >= (y - 5) / height && r <= (y + 5) / height ? 1 : 0\"\n" \
"  expr1 \"y / height >= HighP && y / height <= HighP + 0.005 ? (fmod(x, 5) != 0 ? 1 : 0) : g >= (y - 5) / height && g <= (y + 5) / height ? 1 : 0\"\n" \
"  expr2 \"y / height >= ContP && y / height <= ContP + 0.005 ? (fmod(x, 5) != 0 ? 1 : 0) : b >= (y - 5) / height && b <= (y + 5) / height ? 1 : 0\"\n" \
"  name Display2\n" \
"  xpos 333\n" \
"  ypos 1079\n" \
" }\n" \
"push $N50a1c140\n" \
" Switch {\n" \
"  inputs 2\n" \
"  which %d\n" \
"  name Display_switch2\n" \
"  xpos 262\n" \
"  ypos 1117\n" \
" }\n" \
" Output {\n" \
"  name Output1\n" \
"  xpos 262\n" \
"  ypos 1217\n" \
" }\n" \
"end_group\n", highP, highB, highP, highB, highP, highP, highB, contP, contR,
contG, contB, satR, satG, satB, shadP, highP, contP, display);
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
}}}

if (p_ParamName == "exp" && p_Args.reason == OFX::eChangeUserEdit)
{
float exp = m_Exp->getValueAtTime(p_Args.time);
float expR = m_ExpR->getValueAtTime(p_Args.time);
float expG = m_ExpG->getValueAtTime(p_Args.time);
float expB = m_ExpB->getValueAtTime(p_Args.time);

float exp1 = (expR + expG + expB)/3.0f;
float expr = expR + (exp - exp1);
float expg = expG + (exp - exp1);
float expb = expB + (exp - exp1);

float expR1 = expr > 20.0f ? expr - 20.0f : expr < -20.0f ? expr + 20.0f : 0.0f;
float expG1 = expg > 20.0f ? expg - 20.0f : expg < -20.0f ? expg + 20.0f : 0.0f;
float expB1 = expb > 20.0f ? expb - 20.0f : expb < -20.0f ? expb + 20.0f : 0.0f;

float expR2 = expR + (exp - exp1 - expR1 + expG1/2.0f + expB1/2.0f);
float expG2 = expG + (exp - exp1 - expG1 + expR1/2.0f + expB1/2.0f);
float expB2 = expB + (exp - exp1 - expB1 + expR1/2.0f + expG1/2.0f);

float ExpSwatchR = expR2 >= expG2 && expR2 >= expB2 ? 1.0f : 1.0f - (fmax(expG2, expB2) - expR2)/40.0f;
float ExpSwatchG = expG2 >= expR2 && expG2 >= expB2 ? 1.0f : 1.0f - (fmax(expR2, expB2) - expG2)/40.0f;
float ExpSwatchB = expB2 >= expR2 && expB2 >= expG2 ? 1.0f : 1.0f - (fmax(expR2, expG2) - expB2)/40.0f;

beginEditBlock("expR");
beginEditBlock("expG");
beginEditBlock("expB");
beginEditBlock("expSwatch");

m_ExpR->setValue(expR2);
m_ExpG->setValue(expG2);
m_ExpB->setValue(expB2);
m_ExpSwatch->setValue(ExpSwatchR, ExpSwatchG, ExpSwatchB);

endEditBlock();
}

if (p_ParamName == "expR" && p_Args.reason == OFX::eChangeUserEdit)
{
float expR = m_ExpR->getValueAtTime(p_Args.time);
float expG = m_ExpG->getValueAtTime(p_Args.time);
float expB = m_ExpB->getValueAtTime(p_Args.time);
float Exp = (expR + expG + expB)/3.0f;

float ExpSwatchR = expR >= expG && expR >= expB ? 1.0f : 1.0f - (fmax(expG, expB) - expR)/40.0f;
float ExpSwatchG = expG >= expR && expG >= expB ? 1.0f : 1.0f - (fmax(expR, expB) - expG)/40.0f;
float ExpSwatchB = expB >= expR && expB >= expG ? 1.0f : 1.0f - (fmax(expR, expG) - expB)/40.0f;

beginEditBlock("exp");
beginEditBlock("expSwatch");

m_Exp->setValue(Exp);
m_ExpSwatch->setValue(ExpSwatchR, ExpSwatchG, ExpSwatchB);

endEditBlock();
}

if (p_ParamName == "expG" && p_Args.reason == OFX::eChangeUserEdit)
{
float expR = m_ExpR->getValueAtTime(p_Args.time);
float expG = m_ExpG->getValueAtTime(p_Args.time);
float expB = m_ExpB->getValueAtTime(p_Args.time);
float Exp = (expR + expG + expB)/3.0f;

float ExpSwatchR = expR >= expG && expR >= expB ? 1.0f : 1.0f - (fmax(expG, expB) - expR)/40.0f;
float ExpSwatchG = expG >= expR && expG >= expB ? 1.0f : 1.0f - (fmax(expR, expB) - expG)/40.0f;
float ExpSwatchB = expB >= expR && expB >= expG ? 1.0f : 1.0f - (fmax(expR, expG) - expB)/40.0f;

beginEditBlock("exp");
beginEditBlock("expSwatch");

m_Exp->setValue(Exp);
m_ExpSwatch->setValue(ExpSwatchR, ExpSwatchG, ExpSwatchB);

endEditBlock();
}

if (p_ParamName == "expB" && p_Args.reason == OFX::eChangeUserEdit)
{
float expR = m_ExpR->getValueAtTime(p_Args.time);
float expG = m_ExpG->getValueAtTime(p_Args.time);
float expB = m_ExpB->getValueAtTime(p_Args.time);
float Exp = (expR + expG + expB)/3.0f;

float ExpSwatchR = expR >= expG && expR >= expB ? 1.0f : 1.0f - (fmax(expG, expB) - expR)/40.0f;
float ExpSwatchG = expG >= expR && expG >= expB ? 1.0f : 1.0f - (fmax(expR, expB) - expG)/40.0f;
float ExpSwatchB = expB >= expR && expB >= expG ? 1.0f : 1.0f - (fmax(expR, expG) - expB)/40.0f;

beginEditBlock("exp");
beginEditBlock("expSwatch");

m_Exp->setValue(Exp);
m_ExpSwatch->setValue(ExpSwatchR, ExpSwatchG, ExpSwatchB);

endEditBlock();
}


if (p_ParamName == "expSwatch" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues expSwatch;
m_ExpSwatch->getValueAtTime(p_Args.time, expSwatch.r, expSwatch.g, expSwatch.b);
float exp = m_Exp->getValueAtTime(p_Args.time);

float expr = exp + (expSwatch.r - (expSwatch.g + expSwatch.b)/2.0f) * (20.0f - sqrt(exp*exp));
float expg = exp + (expSwatch.g - (expSwatch.r + expSwatch.b)/2.0f) * (20.0f - sqrt(exp*exp));
float expb = exp + (expSwatch.b - (expSwatch.r + expSwatch.g)/2.0f) * (20.0f - sqrt(exp*exp));

beginEditBlock("expR");
beginEditBlock("expG");
beginEditBlock("expB");

m_ExpR->setValue(expr);
m_ExpG->setValue(expg);
m_ExpB->setValue(expb);

endEditBlock();
}

if (p_ParamName == "cont" && p_Args.reason == OFX::eChangeUserEdit)
{
float cont = m_Cont->getValueAtTime(p_Args.time);
float contR = m_ContR->getValueAtTime(p_Args.time);
float contG = m_ContG->getValueAtTime(p_Args.time);
float contB = m_ContB->getValueAtTime(p_Args.time);

float cont1 = (contR + contG + contB)/3.0f;
float contr = contR + (cont - cont1);
float contg = contG + (cont - cont1);
float contb = contB + (cont - cont1);

float contR1 = contr > 3.0f ? contr - 3.0f : contr < 0.0f ? contr : 0.0f;
float contG1 = contg > 3.0f ? contg - 3.0f : contg < 0.0f ? contg : 0.0f;
float contB1 = contb > 3.0f ? contb - 3.0f : contb < 0.0f ? contb : 0.0f;

float contR2 = contR + (cont - cont1 - contR1 + contG1/2.0f + contB1/2.0f);
float contG2 = contG + (cont - cont1 - contG1 + contR1/2.0f + contB1/2.0f);
float contB2 = contB + (cont - cont1 - contB1 + contR1/2.0f + contG1/2.0f);

float ContSwatchR = contR2 >= contG2 && contR2 >= contB2 ? 1.0f : 1.0f - (fmax(contG2, contB2) - contR2)/3.0f;
float ContSwatchG = contG2 >= contR2 && contG2 >= contB2 ? 1.0f : 1.0f - (fmax(contR2, contB2) - contG2)/3.0f;
float ContSwatchB = contB2 >= contR2 && contB2 >= contG2 ? 1.0f : 1.0f - (fmax(contR2, contG2) - contB2)/3.0f;

beginEditBlock("contR");
beginEditBlock("contG");
beginEditBlock("contB");
beginEditBlock("contSwatch");

m_ContR->setValue(contR2);
m_ContG->setValue(contG2);
m_ContB->setValue(contB2);
m_ContSwatch->setValue(ContSwatchR, ContSwatchG, ContSwatchB);

endEditBlock();
}

if (p_ParamName == "contR" && p_Args.reason == OFX::eChangeUserEdit)
{
float contR = m_ContR->getValueAtTime(p_Args.time);
float contG = m_ContG->getValueAtTime(p_Args.time);
float contB = m_ContB->getValueAtTime(p_Args.time);

float Cont = (contR + contG + contB)/3.0f;

float ContSwatchR = contR >= contG && contR >= contB ? 1.0f : 1.0f - (fmax(contG, contB) - contR)/3.0f;
float ContSwatchG = contG >= contR && contG >= contB ? 1.0f : 1.0f - (fmax(contR, contB) - contG)/3.0f;
float ContSwatchB = contB >= contR && contB >= contG ? 1.0f : 1.0f - (fmax(contR, contG) - contB)/3.0f;

beginEditBlock("cont");
beginEditBlock("contSwatch");

m_Cont->setValue(Cont);
m_ContSwatch->setValue(ContSwatchR, ContSwatchG, ContSwatchB);

endEditBlock();
}

if (p_ParamName == "contG" && p_Args.reason == OFX::eChangeUserEdit)
{
float contR = m_ContR->getValueAtTime(p_Args.time);
float contG = m_ContG->getValueAtTime(p_Args.time);
float contB = m_ContB->getValueAtTime(p_Args.time);

float Cont = (contR + contG + contB)/3.0f;

float ContSwatchR = contR >= contG && contR >= contB ? 1.0f : 1.0f - (fmax(contG, contB) - contR)/3.0f;
float ContSwatchG = contG >= contR && contG >= contB ? 1.0f : 1.0f - (fmax(contR, contB) - contG)/3.0f;
float ContSwatchB = contB >= contR && contB >= contG ? 1.0f : 1.0f - (fmax(contR, contG) - contB)/3.0f;

beginEditBlock("cont");
beginEditBlock("contSwatch");

m_Cont->setValue(Cont);
m_ContSwatch->setValue(ContSwatchR, ContSwatchG, ContSwatchB);

endEditBlock();
}

if (p_ParamName == "contB" && p_Args.reason == OFX::eChangeUserEdit)
{
float contR = m_ContR->getValueAtTime(p_Args.time);
float contG = m_ContG->getValueAtTime(p_Args.time);
float contB = m_ContB->getValueAtTime(p_Args.time);

float Cont = (contR + contG + contB)/3.0f;

float ContSwatchR = contR >= contG && contR >= contB ? 1.0f : 1.0f - (fmax(contG, contB) - contR)/3.0f;
float ContSwatchG = contG >= contR && contG >= contB ? 1.0f : 1.0f - (fmax(contR, contB) - contG)/3.0f;
float ContSwatchB = contB >= contR && contB >= contG ? 1.0f : 1.0f - (fmax(contR, contG) - contB)/3.0f;

beginEditBlock("cont");
beginEditBlock("contSwatch");

m_Cont->setValue(Cont);
m_ContSwatch->setValue(ContSwatchR, ContSwatchG, ContSwatchB);

endEditBlock();
}

if (p_ParamName == "contSwatch" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues contSwatch;
m_ContSwatch->getValueAtTime(p_Args.time, contSwatch.r, contSwatch.g, contSwatch.b);
float cont = m_Cont->getValueAtTime(p_Args.time);

float cont1 = cont >= 1.0f ? (3.0f - cont) : cont;
float contr = cont + (contSwatch.r - (contSwatch.g + contSwatch.b)/2.0f) * cont1;
float contg = cont + (contSwatch.g - (contSwatch.r + contSwatch.b)/2.0f) * cont1;
float contb = cont + (contSwatch.b - (contSwatch.r + contSwatch.g)/2.0f) * cont1;

beginEditBlock("contR");
beginEditBlock("contG");
beginEditBlock("contB");

m_ContR->setValue(contr);
m_ContG->setValue(contg);
m_ContB->setValue(contb);

endEditBlock();
}


if (p_ParamName == "sat" && p_Args.reason == OFX::eChangeUserEdit)
{
float sat = m_Sat->getValueAtTime(p_Args.time);
float satR = m_SatR->getValueAtTime(p_Args.time);
float satG = m_SatG->getValueAtTime(p_Args.time);
float satB = m_SatB->getValueAtTime(p_Args.time);

float sat1 = (satR + satG + satB)/3.0f;
float satr = satR + (sat - sat1);
float satg = satG + (sat - sat1);
float satb = satB + (sat - sat1);

float satR1 = satr > 3.0f ? satr - 3.0f : satr < 0.0f ? satr : 0.0f;
float satG1 = satg > 3.0f ? satg - 3.0f : satg < 0.0f ? satg : 0.0f;
float satB1 = satb > 3.0f ? satb - 3.0f : satb < 0.0f ? satb : 0.0f;

float satR2 = satR + (sat - sat1 - satR1 + satG1/2.0f + satB1/2.0f);
float satG2 = satG + (sat - sat1 - satG1 + satR1/2.0f + satB1/2.0f);
float satB2 = satB + (sat - sat1 - satB1 + satR1/2.0f + satG1/2.0f);

float SatSwatchR = satR2 >= satG2 && satR2 >= satB2 ? 1.0f : 1.0f - (fmax(satG2, satB2) - satR2)/3.0f;
float SatSwatchG = satG2 >= satR2 && satG2 >= satB2 ? 1.0f : 1.0f - (fmax(satR2, satB2) - satG2)/3.0f;
float SatSwatchB = satB2 >= satR2 && satB2 >= satG2 ? 1.0f : 1.0f - (fmax(satR2, satG2) - satB2)/3.0f;

beginEditBlock("satR");
beginEditBlock("satG");
beginEditBlock("satB");
beginEditBlock("satSwatch");

m_SatR->setValue(satR2);
m_SatG->setValue(satG2);
m_SatB->setValue(satB2);
m_SatSwatch->setValue(SatSwatchR, SatSwatchG, SatSwatchB);

endEditBlock();
}

if (p_ParamName == "satR" && p_Args.reason == OFX::eChangeUserEdit)
{
float satR = m_SatR->getValueAtTime(p_Args.time);
float satG = m_SatG->getValueAtTime(p_Args.time);
float satB = m_SatB->getValueAtTime(p_Args.time);

float Sat = (satR + satG + satB)/3.0f;

float SatSwatchR = satR >= satG && satR >= satB ? 1.0f : 1.0f - (fmax(satG, satB) - satR)/3.0f;
float SatSwatchG = satG >= satR && satG >= satB ? 1.0f : 1.0f - (fmax(satR, satB) - satG)/3.0f;
float SatSwatchB = satB >= satR && satB >= satG ? 1.0f : 1.0f - (fmax(satR, satG) - satB)/3.0f;

beginEditBlock("sat");
beginEditBlock("satSwatch");

m_Sat->setValue(Sat);
m_SatSwatch->setValue(SatSwatchR, SatSwatchG, SatSwatchB);

endEditBlock();
}

if (p_ParamName == "satG" && p_Args.reason == OFX::eChangeUserEdit)
{
float satR = m_SatR->getValueAtTime(p_Args.time);
float satG = m_SatG->getValueAtTime(p_Args.time);
float satB = m_SatB->getValueAtTime(p_Args.time);

float Sat = (satR + satG + satB)/3.0f;

float SatSwatchR = satR >= satG && satR >= satB ? 1.0f : 1.0f - (fmax(satG, satB) - satR)/3.0f;
float SatSwatchG = satG >= satR && satG >= satB ? 1.0f : 1.0f - (fmax(satR, satB) - satG)/3.0f;
float SatSwatchB = satB >= satR && satB >= satG ? 1.0f : 1.0f - (fmax(satR, satG) - satB)/3.0f;

beginEditBlock("sat");
beginEditBlock("satSwatch");

m_Sat->setValue(Sat);
m_SatSwatch->setValue(SatSwatchR, SatSwatchG, SatSwatchB);

endEditBlock();
}

if (p_ParamName == "satB" && p_Args.reason == OFX::eChangeUserEdit)
{
float satR = m_SatR->getValueAtTime(p_Args.time);
float satG = m_SatG->getValueAtTime(p_Args.time);
float satB = m_SatB->getValueAtTime(p_Args.time);

float Sat = (satR + satG + satB)/3.0f;

float SatSwatchR = satR >= satG && satR >= satB ? 1.0f : 1.0f - (fmax(satG, satB) - satR)/3.0f;
float SatSwatchG = satG >= satR && satG >= satB ? 1.0f : 1.0f - (fmax(satR, satB) - satG)/3.0f;
float SatSwatchB = satB >= satR && satB >= satG ? 1.0f : 1.0f - (fmax(satR, satG) - satB)/3.0f;

beginEditBlock("sat");
beginEditBlock("satSwatch");

m_Sat->setValue(Sat);
m_SatSwatch->setValue(SatSwatchR, SatSwatchG, SatSwatchB);

endEditBlock();
}

if (p_ParamName == "satSwatch" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues satSwatch;
m_SatSwatch->getValueAtTime(p_Args.time, satSwatch.r, satSwatch.g, satSwatch.b);
float sat = m_Sat->getValueAtTime(p_Args.time);

float sat1 = sat >= 1.0f ? (3.0f - sat) : sat;
float satr = sat + (satSwatch.r - (satSwatch.g + satSwatch.b)/2.0f) * sat1;
float satg = sat + (satSwatch.g - (satSwatch.r + satSwatch.b)/2.0f) * sat1;
float satb = sat + (satSwatch.b - (satSwatch.r + satSwatch.g)/2.0f) * sat1;

beginEditBlock("satR");
beginEditBlock("satG");
beginEditBlock("satB");

m_SatR->setValue(satr);
m_SatG->setValue(satg);
m_SatB->setValue(satb);

endEditBlock();
}

if (p_ParamName == "mid" && p_Args.reason == OFX::eChangeUserEdit)
{
float mid = m_Mid->getValueAtTime(p_Args.time);
float midR = m_MidR->getValueAtTime(p_Args.time);
float midG = m_MidG->getValueAtTime(p_Args.time);
float midB = m_MidB->getValueAtTime(p_Args.time);

float mid1 = (midR + midG + midB)/3.0f;
float midr = midR + (mid - mid1);
float midg = midG + (mid - mid1);
float midb = midB + (mid - mid1);

float midR1 = midr > 0.5f ? midr - 0.5f : midr < -0.5f ? midr + 0.5f : 0.0f;
float midG1 = midg > 0.5f ? midg - 0.5f : midg < -0.5f ? midg + 0.5f : 0.0f;
float midB1 = midb > 0.5f ? midb - 0.5f : midb < -0.5f ? midb + 0.5f : 0.0f;

float midR2 = midR + (mid - mid1 - midR1 + midG1/2.0f + midB1/2.0f);
float midG2 = midG + (mid - mid1 - midG1 + midR1/2.0f + midB1/2.0f);
float midB2 = midB + (mid - mid1 - midB1 + midR1/2.0f + midG1/2.0f);

float MidSwatchR = midR2 >= midG2 && midR2 >= midB2 ? 1.0f : 1.0f - (fmax(midG2, midB2) - midR2);
float MidSwatchG = midG2 >= midR2 && midG2 >= midB2 ? 1.0f : 1.0f - (fmax(midR2, midB2) - midG2);
float MidSwatchB = midB2 >= midR2 && midB2 >= midG2 ? 1.0f : 1.0f - (fmax(midR2, midG2) - midB2);

beginEditBlock("midR");
beginEditBlock("midG");
beginEditBlock("midB");
beginEditBlock("midSwatch");

m_MidR->setValue(midR2);
m_MidG->setValue(midG2);
m_MidB->setValue(midB2);
m_MidSwatch->setValue(MidSwatchR, MidSwatchG, MidSwatchB);

endEditBlock();
}

if (p_ParamName == "midR" && p_Args.reason == OFX::eChangeUserEdit)
{
float midR = m_MidR->getValueAtTime(p_Args.time);
float midG = m_MidG->getValueAtTime(p_Args.time);
float midB = m_MidB->getValueAtTime(p_Args.time);

float Mid = (midR + midG + midB)/3.0f;

float MidSwatchR = midR >= midG && midR >= midB ? 1.0f : 1.0f - (fmax(midG, midB) - midR);
float MidSwatchG = midG >= midR && midG >= midB ? 1.0f : 1.0f - (fmax(midR, midB) - midG);
float MidSwatchB = midB >= midR && midB >= midG ? 1.0f : 1.0f - (fmax(midR, midG) - midB);

beginEditBlock("mid");
beginEditBlock("midSwatch");

m_Mid->setValue(Mid);
m_MidSwatch->setValue(MidSwatchR, MidSwatchG, MidSwatchB);

endEditBlock();
}

if (p_ParamName == "midG" && p_Args.reason == OFX::eChangeUserEdit)
{
float midR = m_MidR->getValueAtTime(p_Args.time);
float midG = m_MidG->getValueAtTime(p_Args.time);
float midB = m_MidB->getValueAtTime(p_Args.time);

float Mid = (midR + midG + midB)/3.0f;

float MidSwatchR = midR >= midG && midR >= midB ? 1.0f : 1.0f - (fmax(midG, midB) - midR);
float MidSwatchG = midG >= midR && midG >= midB ? 1.0f : 1.0f - (fmax(midR, midB) - midG);
float MidSwatchB = midB >= midR && midB >= midG ? 1.0f : 1.0f - (fmax(midR, midG) - midB);

beginEditBlock("mid");
beginEditBlock("midSwatch");

m_Mid->setValue(Mid);
m_MidSwatch->setValue(MidSwatchR, MidSwatchG, MidSwatchB);

endEditBlock();
}

if (p_ParamName == "midB" && p_Args.reason == OFX::eChangeUserEdit)
{
float midR = m_MidR->getValueAtTime(p_Args.time);
float midG = m_MidG->getValueAtTime(p_Args.time);
float midB = m_MidB->getValueAtTime(p_Args.time);

float Mid = (midR + midG + midB)/3.0f;

float MidSwatchR = midR >= midG && midR >= midB ? 1.0f : 1.0f - (fmax(midG, midB) - midR);
float MidSwatchG = midG >= midR && midG >= midB ? 1.0f : 1.0f - (fmax(midR, midB) - midG);
float MidSwatchB = midB >= midR && midB >= midG ? 1.0f : 1.0f - (fmax(midR, midG) - midB);

beginEditBlock("mid");
beginEditBlock("midSwatch");

m_Mid->setValue(Mid);
m_MidSwatch->setValue(MidSwatchR, MidSwatchG, MidSwatchB);

endEditBlock();
}

if (p_ParamName == "midSwatch" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues midSwatch;
m_MidSwatch->getValueAtTime(p_Args.time, midSwatch.r, midSwatch.g, midSwatch.b);
float mid = m_Mid->getValueAtTime(p_Args.time);

float midr = mid + (midSwatch.r - (midSwatch.g + midSwatch.b)/2.0f) * (0.5f - sqrt(mid*mid));
float midg = mid + (midSwatch.g - (midSwatch.r + midSwatch.b)/2.0f) * (0.5f - sqrt(mid*mid));
float midb = mid + (midSwatch.b - (midSwatch.r + midSwatch.g)/2.0f) * (0.5f - sqrt(mid*mid));

beginEditBlock("midR");
beginEditBlock("midG");
beginEditBlock("midB");

m_MidR->setValue(midr);
m_MidG->setValue(midg);
m_MidB->setValue(midb);

endEditBlock();
}

if (p_ParamName == "shad" && p_Args.reason == OFX::eChangeUserEdit)
{
float shad = m_Shad->getValueAtTime(p_Args.time);
float shadR = m_ShadR->getValueAtTime(p_Args.time);
float shadG = m_ShadG->getValueAtTime(p_Args.time);
float shadB = m_ShadB->getValueAtTime(p_Args.time);

float shad1 = (shadR + shadG + shadB)/3.0f;
float shadr = shadR + (shad - shad1);
float shadg = shadG + (shad - shad1);
float shadb = shadB + (shad - shad1);

float shadR1 = shadr > 0.5f ? shadr - 0.5f : shadr < -0.5f ? shadr + 0.5f : 0.0f;
float shadG1 = shadg > 0.5f ? shadg - 0.5f : shadg < -0.5f ? shadg + 0.5f : 0.0f;
float shadB1 = shadb > 0.5f ? shadb - 0.5f : shadb < -0.5f ? shadb + 0.5f : 0.0f;

float shadR2 = shadR + (shad - shad1 - shadR1 + shadG1/2.0f + shadB1/2.0f);
float shadG2 = shadG + (shad - shad1 - shadG1 + shadR1/2.0f + shadB1/2.0f);
float shadB2 = shadB + (shad - shad1 - shadB1 + shadR1/2.0f + shadG1/2.0f);

float ShadSwatchR = shadR2 >= shadG2 && shadR2 >= shadB2 ? 1.0f : 1.0f - (fmax(shadG2, shadB2) - shadR2);
float ShadSwatchG = shadG2 >= shadR2 && shadG2 >= shadB2 ? 1.0f : 1.0f - (fmax(shadR2, shadB2) - shadG2);
float ShadSwatchB = shadB2 >= shadR2 && shadB2 >= shadG2 ? 1.0f : 1.0f - (fmax(shadR2, shadG2) - shadB2);

beginEditBlock("shadR");
beginEditBlock("shadG");
beginEditBlock("shadB");
beginEditBlock("shadSwatch");

m_ShadR->setValue(shadR2);
m_ShadG->setValue(shadG2);
m_ShadB->setValue(shadB2);
m_ShadSwatch->setValue(ShadSwatchR, ShadSwatchG, ShadSwatchB);

endEditBlock();
}

if (p_ParamName == "shadR" && p_Args.reason == OFX::eChangeUserEdit)
{
float shadR = m_ShadR->getValueAtTime(p_Args.time);
float shadG = m_ShadG->getValueAtTime(p_Args.time);
float shadB = m_ShadB->getValueAtTime(p_Args.time);

float Shad = (shadR + shadG + shadB)/3.0f;

float ShadSwatchR = shadR >= shadG && shadR >= shadB ? 1.0f : 1.0f - (fmax(shadG, shadB) - shadR);
float ShadSwatchG = shadG >= shadR && shadG >= shadB ? 1.0f : 1.0f - (fmax(shadR, shadB) - shadG);
float ShadSwatchB = shadB >= shadR && shadB >= shadG ? 1.0f : 1.0f - (fmax(shadR, shadG) - shadB);

beginEditBlock("shad");
beginEditBlock("shadSwatch");

m_Shad->setValue(Shad);
m_ShadSwatch->setValue(ShadSwatchR, ShadSwatchG, ShadSwatchB);

endEditBlock();
}

if (p_ParamName == "shadG" && p_Args.reason == OFX::eChangeUserEdit)
{
float shadR = m_ShadR->getValueAtTime(p_Args.time);
float shadG = m_ShadG->getValueAtTime(p_Args.time);
float shadB = m_ShadB->getValueAtTime(p_Args.time);

float Shad = (shadR + shadG + shadB)/3.0f;

float ShadSwatchR = shadR >= shadG && shadR >= shadB ? 1.0f : 1.0f - (fmax(shadG, shadB) - shadR);
float ShadSwatchG = shadG >= shadR && shadG >= shadB ? 1.0f : 1.0f - (fmax(shadR, shadB) - shadG);
float ShadSwatchB = shadB >= shadR && shadB >= shadG ? 1.0f : 1.0f - (fmax(shadR, shadG) - shadB);

beginEditBlock("shad");
beginEditBlock("shadSwatch");

m_Shad->setValue(Shad);
m_ShadSwatch->setValue(ShadSwatchR, ShadSwatchG, ShadSwatchB);

endEditBlock();
}

if (p_ParamName == "shadB" && p_Args.reason == OFX::eChangeUserEdit)
{
float shadR = m_ShadR->getValueAtTime(p_Args.time);
float shadG = m_ShadG->getValueAtTime(p_Args.time);
float shadB = m_ShadB->getValueAtTime(p_Args.time);

float Shad = (shadR + shadG + shadB)/3.0f;

float ShadSwatchR = shadR >= shadG && shadR >= shadB ? 1.0f : 1.0f - (fmax(shadG, shadB) - shadR);
float ShadSwatchG = shadG >= shadR && shadG >= shadB ? 1.0f : 1.0f - (fmax(shadR, shadB) - shadG);
float ShadSwatchB = shadB >= shadR && shadB >= shadG ? 1.0f : 1.0f - (fmax(shadR, shadG) - shadB);

beginEditBlock("shad");
beginEditBlock("shadSwatch");

m_Shad->setValue(Shad);
m_ShadSwatch->setValue(ShadSwatchR, ShadSwatchG, ShadSwatchB);

endEditBlock();
}

if (p_ParamName == "shadSwatch" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues shadSwatch;
m_ShadSwatch->getValueAtTime(p_Args.time, shadSwatch.r, shadSwatch.g, shadSwatch.b);
float shad = m_Shad->getValueAtTime(p_Args.time);

float shadr = shad + (shadSwatch.r - (shadSwatch.g + shadSwatch.b)/2.0f) * (0.5f - sqrt(shad*shad));
float shadg = shad + (shadSwatch.g - (shadSwatch.r + shadSwatch.b)/2.0f) * (0.5f - sqrt(shad*shad));
float shadb = shad + (shadSwatch.b - (shadSwatch.r + shadSwatch.g)/2.0f) * (0.5f - sqrt(shad*shad));

beginEditBlock("shadR");
beginEditBlock("shadG");
beginEditBlock("shadB");

m_ShadR->setValue(shadr);
m_ShadG->setValue(shadg);
m_ShadB->setValue(shadb);

endEditBlock();
}

if (p_ParamName == "high" && p_Args.reason == OFX::eChangeUserEdit)
{
float high = m_High->getValueAtTime(p_Args.time);
float highR = m_HighR->getValueAtTime(p_Args.time);
float highG = m_HighG->getValueAtTime(p_Args.time);
float highB = m_HighB->getValueAtTime(p_Args.time);

float high1 = (highR + highG + highB)/3.0f;
float highr = highR + (high - high1);
float highg = highG + (high - high1);
float highb = highB + (high - high1);

float highR1 = highr > 0.5f ? highr - 0.5f : highr < -0.5f ? highr + 0.5f : 0.0f;
float highG1 = highg > 0.5f ? highg - 0.5f : highg < -0.5f ? highg + 0.5f : 0.0f;
float highB1 = highb > 0.5f ? highb - 0.5f : highb < -0.5f ? highb + 0.5f : 0.0f;

float highR2 = highR + (high - high1 - highR1 + highG1/2.0f + highB1/2.0f);
float highG2 = highG + (high - high1 - highG1 + highR1/2.0f + highB1/2.0f);
float highB2 = highB + (high - high1 - highB1 + highR1/2.0f + highG1/2.0f);

float HighSwatchR = highR2 >= highG2 && highR2 >= highB2 ? 1.0f : 1.0f - (fmax(highG2, highB2) - highR2);
float HighSwatchG = highG2 >= highR2 && highG2 >= highB2 ? 1.0f : 1.0f - (fmax(highR2, highB2) - highG2);
float HighSwatchB = highB2 >= highR2 && highB2 >= highG2 ? 1.0f : 1.0f - (fmax(highR2, highG2) - highB2);

beginEditBlock("highR");
beginEditBlock("highG");
beginEditBlock("highB");
beginEditBlock("highSwatch");

m_HighR->setValue(highR2);
m_HighG->setValue(highG2);
m_HighB->setValue(highB2);
m_HighSwatch->setValue(HighSwatchR, HighSwatchG, HighSwatchB);

endEditBlock();
}

if (p_ParamName == "highR" && p_Args.reason == OFX::eChangeUserEdit)
{
float highR = m_HighR->getValueAtTime(p_Args.time);
float highG = m_HighG->getValueAtTime(p_Args.time);
float highB = m_HighB->getValueAtTime(p_Args.time);

float High = (highR + highG + highB)/3.0f;

float HighSwatchR = highR >= highG && highR >= highB ? 1.0f : 1.0f - (fmax(highG, highB) - highR);
float HighSwatchG = highG >= highR && highG >= highB ? 1.0f : 1.0f - (fmax(highR, highB) - highG);
float HighSwatchB = highB >= highR && highB >= highG ? 1.0f : 1.0f - (fmax(highR, highG) - highB);

beginEditBlock("high");
beginEditBlock("highSwatch");

m_High->setValue(High);
m_HighSwatch->setValue(HighSwatchR, HighSwatchG, HighSwatchB);

endEditBlock();
}

if (p_ParamName == "highG" && p_Args.reason == OFX::eChangeUserEdit)
{
float highR = m_HighR->getValueAtTime(p_Args.time);
float highG = m_HighG->getValueAtTime(p_Args.time);
float highB = m_HighB->getValueAtTime(p_Args.time);

float High = (highR + highG + highB)/3.0f;

float HighSwatchR = highR >= highG && highR >= highB ? 1.0f : 1.0f - (fmax(highG, highB) - highR);
float HighSwatchG = highG >= highR && highG >= highB ? 1.0f : 1.0f - (fmax(highR, highB) - highG);
float HighSwatchB = highB >= highR && highB >= highG ? 1.0f : 1.0f - (fmax(highR, highG) - highB);

beginEditBlock("high");
beginEditBlock("highSwatch");

m_High->setValue(High);
m_HighSwatch->setValue(HighSwatchR, HighSwatchG, HighSwatchB);

endEditBlock();
}

if (p_ParamName == "highB" && p_Args.reason == OFX::eChangeUserEdit)
{
float highR = m_HighR->getValueAtTime(p_Args.time);
float highG = m_HighG->getValueAtTime(p_Args.time);
float highB = m_HighB->getValueAtTime(p_Args.time);

float High = (highR + highG + highB)/3.0f;

float HighSwatchR = highR >= highG && highR >= highB ? 1.0f : 1.0f - (fmax(highG, highB) - highR);
float HighSwatchG = highG >= highR && highG >= highB ? 1.0f : 1.0f - (fmax(highR, highB) - highG);
float HighSwatchB = highB >= highR && highB >= highG ? 1.0f : 1.0f - (fmax(highR, highG) - highB);

beginEditBlock("high");
beginEditBlock("highSwatch");

m_High->setValue(High);
m_HighSwatch->setValue(HighSwatchR, HighSwatchG, HighSwatchB);

endEditBlock();
}

if (p_ParamName == "highSwatch" && p_Args.reason == OFX::eChangeUserEdit)
{
RGBValues highSwatch;
m_HighSwatch->getValueAtTime(p_Args.time, highSwatch.r, highSwatch.g, highSwatch.b);
float high = m_High->getValueAtTime(p_Args.time);

float highr = high + (highSwatch.r - (highSwatch.g + highSwatch.b)/2.0f) * (0.5f - sqrt(high*high));
float highg = high + (highSwatch.g - (highSwatch.r + highSwatch.b)/2.0f) * (0.5f - sqrt(high*high));
float highb = high + (highSwatch.b - (highSwatch.r + highSwatch.g)/2.0f) * (0.5f - sqrt(high*high));

beginEditBlock("highR");
beginEditBlock("highG");
beginEditBlock("highB");

m_HighR->setValue(highr);
m_HighG->setValue(highg);
m_HighB->setValue(highb);

endEditBlock();
}

if (p_ParamName == "shadP" && p_Args.reason == OFX::eChangeUserEdit)
{
float shadP = m_ShadP->getValueAtTime(p_Args.time);
m_ShadPP->setValue(shadP);
}

if (p_ParamName == "shadPP" && p_Args.reason == OFX::eChangeUserEdit)
{
float shadPP = m_ShadPP->getValueAtTime(p_Args.time);
m_ShadP->setValue(shadPP);
}

if (p_ParamName == "highP" && p_Args.reason == OFX::eChangeUserEdit)
{
float highP = m_HighP->getValueAtTime(p_Args.time);
m_HighPP->setValue(highP);
}

if (p_ParamName == "highPP" && p_Args.reason == OFX::eChangeUserEdit)
{
float highPP = m_HighPP->getValueAtTime(p_Args.time);
m_HighP->setValue(highPP);
}

if (p_ParamName == "contP" && p_Args.reason == OFX::eChangeUserEdit)
{
float contP = m_ContP->getValueAtTime(p_Args.time);
m_ContPP->setValue(contP);
}

if (p_ParamName == "contPP" && p_Args.reason == OFX::eChangeUserEdit)
{
float contPP = m_ContPP->getValueAtTime(p_Args.time);
m_ContP->setValue(contPP);
}}

void FilmGradePlugin::setupAndProcess(FilmGrade& p_FilmGrade, const OFX::RenderArguments& p_Args)
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

float expR = m_ExpR->getValueAtTime(p_Args.time);
float expG = m_ExpG->getValueAtTime(p_Args.time);
float expB = m_ExpB->getValueAtTime(p_Args.time);
float contR = m_ContR->getValueAtTime(p_Args.time);
float contG = m_ContG->getValueAtTime(p_Args.time);
float contB = m_ContB->getValueAtTime(p_Args.time);
float satR = m_SatR->getValueAtTime(p_Args.time);
float satG = m_SatG->getValueAtTime(p_Args.time);
float satB = m_SatB->getValueAtTime(p_Args.time);
float shadR = m_ShadR->getValueAtTime(p_Args.time);
float shadG = m_ShadG->getValueAtTime(p_Args.time);
float shadB = m_ShadB->getValueAtTime(p_Args.time);
float midR = m_MidR->getValueAtTime(p_Args.time);
float midG = m_MidG->getValueAtTime(p_Args.time);
float midB = m_MidB->getValueAtTime(p_Args.time);
float highR = m_HighR->getValueAtTime(p_Args.time);
float highG = m_HighG->getValueAtTime(p_Args.time);
float highB = m_HighB->getValueAtTime(p_Args.time);
float shadP = m_ShadP->getValueAtTime(p_Args.time) / 1023.0;
float highP = m_HighP->getValueAtTime(p_Args.time) / 1023.0;
float contP = m_ContP->getValueAtTime(p_Args.time) / 1023.0;

int displayGraph;
m_DisplayGraph->getValueAtTime(p_Args.time, displayGraph);

p_FilmGrade.setDstImg(dst.get());
p_FilmGrade.setSrcImg(src.get());

// Setup GPU Render arguments
p_FilmGrade.setGPURenderArgs(p_Args);

p_FilmGrade.setRenderWindow(p_Args.renderWindow);

p_FilmGrade.setScales(expR, expG, expB, contR, contG, contB, satR, satG, satB, 
shadR, shadG, shadB, midR, midG, midB, highR, highG, highB, shadP, highP, contP, displayGraph);

p_FilmGrade.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

FilmGradePluginFactory::FilmGradePluginFactory()
: OFX::PluginFactoryHelper<FilmGradePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void FilmGradePluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
if (p_Parent)
param->setParent(*p_Parent);
return param;
}

void FilmGradePluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

GroupParamDescriptor* ecs = p_Desc.defineGroupParam("ExpContSat");
ecs->setOpen(true);
ecs->setHint("Exposure Contrast Saturation");
if (page)
page->addChild(*ecs);

GroupParamDescriptor* exp = p_Desc.defineGroupParam("Exposure RGB");
exp->setOpen(false);
exp->setHint("Exposure Channels");
exp->setParent(*ecs);
if (page)
page->addChild(*exp);

RGBParamDescriptor *RGBparam = p_Desc.defineRGBParam("expSwatch");
RGBparam->setLabel("Exposure");
RGBparam->setHint("exposure colour wheel");
RGBparam->setDefault(1.0, 1.0, 1.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
RGBparam->setParent(*exp);
page->addChild(*RGBparam);

DoubleParamDescriptor* param = defineScaleParam(p_Desc, "expR", "Exposure Red", "red offset", exp);
param->setDefault(0.0);
param->setRange(-20.0, 20.0);
param->setIncrement(0.01);
param->setDisplayRange(-20.0, 20.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "expG", "Exposure Green", "green offset", exp);
param->setDefault(0.0);
param->setRange(-20.0, 20.0);
param->setIncrement(0.01);
param->setDisplayRange(-20.0, 20.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "expB", "Exposure Blue", "blue offset", exp);
param->setDefault(0.0);
param->setRange(-20.0, 20.0);
param->setIncrement(0.01);
param->setDisplayRange(-20.0, 20.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "exp", "Exposure", "offset", ecs);
param->setDefault(0.0);
param->setRange(-20.0, 20.0);
param->setIncrement(0.01);
param->setDisplayRange(-20.0, 20.0);
page->addChild(*param);

GroupParamDescriptor* con = p_Desc.defineGroupParam("Contrast RGB");
con->setOpen(false);
con->setHint("Contrast Channels");
con->setParent(*ecs);
if (page)
page->addChild(*con);

RGBparam = p_Desc.defineRGBParam("contSwatch");
RGBparam->setLabel("Contrast");
RGBparam->setHint("contrast colour wheel");
RGBparam->setDefault(1.0, 1.0, 1.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
RGBparam->setParent(*con);
page->addChild(*RGBparam);

param = defineScaleParam(p_Desc, "contR", "Contrast Red", "red contrast", con);
param->setDefault(1.0);
param->setRange(0.0, 3.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "contG", "Contrast Green", "green contrast", con);
param->setDefault(1.0);
param->setRange(0.0, 3.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "contB", "Contrast Blue", "blue contrast", con);
param->setDefault(1.0);
param->setRange(0.0, 3.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "cont", "Contrast", "contrast", ecs);
param->setDefault(1.0);
param->setRange(0.0, 3.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "contP", "Contrast Pivot", "contrast pivot point", ecs);
param->setDefault(445.0);
param->setRange(0.0, 1023.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1023.0);
page->addChild(*param);

GroupParamDescriptor* sat = p_Desc.defineGroupParam("Saturation RGB");
sat->setOpen(false);
sat->setHint("Contrast Channels");
sat->setParent(*ecs);
if (page)
page->addChild(*sat);

RGBparam = p_Desc.defineRGBParam("satSwatch");
RGBparam->setLabel("Saturation");
RGBparam->setHint("saturation colour wheel");
RGBparam->setDefault(1.0, 1.0, 1.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
RGBparam->setParent(*sat);
page->addChild(*RGBparam);

param = defineScaleParam(p_Desc, "satR", "Saturation Red", "red saturation", sat);
param->setDefault(1.0);
param->setRange(0.0, 3.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "satG", "Saturation Green", "green saturation", sat);
param->setDefault(1.0);
param->setRange(0.0, 3.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "satB", "Saturation Blue", "blue saturation", sat);
param->setDefault(1.0);
param->setRange(0.0, 3.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "sat", "Saturation", "saturation", ecs);
param->setDefault(1.0);
param->setRange(0.0, 3.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 3.0);
page->addChild(*param);

GroupParamDescriptor* smh = p_Desc.defineGroupParam("ShadMidHigh");
smh->setOpen(false);
smh->setHint("Shadows Midtones Highlights");
if (page)
page->addChild(*smh);

GroupParamDescriptor* shad = p_Desc.defineGroupParam("Shadows RGB");
shad->setOpen(false);
shad->setHint("Shadows Channels");
shad->setParent(*smh);
if (page)
page->addChild(*shad);

RGBparam = p_Desc.defineRGBParam("shadSwatch");
RGBparam->setLabel("Shadows");
RGBparam->setHint("shadows colour wheel");
RGBparam->setDefault(1.0, 1.0, 1.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
RGBparam->setParent(*shad);
page->addChild(*RGBparam);

param = defineScaleParam(p_Desc, "shadR", "Shadows Red", "red shadows", shad);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "shadG", "Shadows Green", "green shadows", shad);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "shadB", "Shadows Blue", "blue shadows", shad);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "shad", "Shadows", "shadow region", smh);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "shadP", "Shadows Pivot", "shadows pivot point", smh);
param->setDefault(400.0);
param->setRange(0.0, 1023.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1023.0);
page->addChild(*param);

GroupParamDescriptor* mid = p_Desc.defineGroupParam("Midtones RGB");
mid->setOpen(false);
mid->setHint("Midtones Channels");
mid->setParent(*smh);
if (page)
page->addChild(*mid);

RGBparam = p_Desc.defineRGBParam("midSwatch");
RGBparam->setLabel("Midtones");
RGBparam->setHint("midtones colour wheel");
RGBparam->setDefault(1.0, 1.0, 1.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
RGBparam->setParent(*mid);
page->addChild(*RGBparam);

param = defineScaleParam(p_Desc, "midR", "Midtones Red", "red midtones", mid);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "midG", "Midtones Green", "green midtones", mid);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "midB", "Midtones Blue", "blue midtones", mid);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "mid", "Midtones", "midtones region", smh);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

GroupParamDescriptor* high = p_Desc.defineGroupParam("Highlights RGB");
high->setOpen(false);
high->setHint("Highlights Channels");
high->setParent(*smh);
if (page)
page->addChild(*high);

RGBparam = p_Desc.defineRGBParam("highSwatch");
RGBparam->setLabel("Highlights");
RGBparam->setHint("highlights colour wheel");
RGBparam->setDefault(1.0, 1.0, 1.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
RGBparam->setParent(*high);
page->addChild(*RGBparam);

param = defineScaleParam(p_Desc, "highR", "Highlights Red", "red highlights", high);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "highG", "Highlights Green", "green highlights", high);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "highB", "Highlights Blue", "blue highlights", high);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "high", "Highlights", "highlight region", smh);
param->setDefault(0.0);
param->setRange(-0.5, 0.5);
param->setIncrement(0.001);
param->setDisplayRange(-0.5, 0.5);
page->addChild(*param);

param = defineScaleParam(p_Desc, "highP", "Highlights Pivot", "highlights pivot point", smh);
param->setDefault(500.0);
param->setRange(0.0, 1023.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1023.0);
page->addChild(*param);  

GroupParamDescriptor* adv = p_Desc.defineGroupParam("Advanced");
adv->setOpen(false);
adv->setHint("Advanced Controls");
if (page)
page->addChild(*adv);

ChoiceParamDescriptor* choiceParam = p_Desc.defineChoiceParam(kParamDisplayGraph);
choiceParam->setLabel(kParamDisplayGraphLabel);
choiceParam->setHint(kParamDisplayGraphHint);
assert(choiceParam->getNOptions() == eDisplayGraphOff);
choiceParam->appendOption(kParamDisplayGraphOptionOff, kParamDisplayGraphOptionOffHint);
assert(choiceParam->getNOptions() == eDisplayGraphON);
choiceParam->appendOption(kParamDisplayGraphOptionON, kParamDisplayGraphOptionONHint);
assert(choiceParam->getNOptions() == eDisplayGraphOVER);
choiceParam->appendOption(kParamDisplayGraphOptionOVER, kParamDisplayGraphOptionOVERHint);
choiceParam->setDefault(eDisplayGraphOff);
choiceParam->setAnimates(false);
page->addChild(*choiceParam);

param = defineScaleParam(p_Desc, "shadPP", "Shadows Pivot", "shadows pivot point", adv);
param->setDefault(400.0);
param->setRange(0.0, 1023.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1023.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "highPP", "Highlights Pivot", "highlights pivot point", adv);
param->setDefault(500.0);
param->setRange(0.0, 1023.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1023.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "contPP", "Contrast Pivot", "contrast pivot point", adv);
param->setDefault(445.0);
param->setRange(0.0, 1023.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1023.0);
page->addChild(*param);

PushButtonParamDescriptor* Pushparam = p_Desc.definePushButtonParam("info");
Pushparam->setLabel("Info");
page->addChild(*Pushparam);

GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
script->setOpen(false);
script->setHint("export DCTL and Nuke script");
if (page)
page->addChild(*script);

Pushparam = p_Desc.definePushButtonParam("button1");
Pushparam->setLabel("Export DCTL");
Pushparam->setHint("create DCTL version");
Pushparam->setParent(*script);
page->addChild(*Pushparam);

Pushparam = p_Desc.definePushButtonParam("button2");
Pushparam->setLabel("Export Nuke script");
Pushparam->setHint("create NUKE version");
Pushparam->setParent(*script);
page->addChild(*Pushparam);

StringParamDescriptor* Stringparam = p_Desc.defineStringParam("name");
Stringparam->setLabel("Name");
Stringparam->setHint("overwrites if the same");
Stringparam->setDefault("FilmGrade");
Stringparam->setParent(*script);
page->addChild(*Stringparam);

Stringparam = p_Desc.defineStringParam("path");
Stringparam->setLabel("Directory");
Stringparam->setHint("make sure it's the absolute path");
Stringparam->setStringType(eStringTypeFilePath);
Stringparam->setDefault(kPluginScript);
Stringparam->setFilePathExists(false);
Stringparam->setParent(*script);
page->addChild(*Stringparam);
}

ImageEffect* FilmGradePluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
return new FilmGradePlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static FilmGradePluginFactory FilmGradePlugin;
p_FactoryArray.push_back(&FilmGradePlugin);
}