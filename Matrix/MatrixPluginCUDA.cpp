#include "MatrixPlugin.h"

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

#define kPluginName "Matrix"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"Matrix: 3x3 Matrix with additional RGB Mixer controls. Includes Normalise, Inverse, and Luma and Saturation preserve options."

#define kPluginIdentifier "BaldavengerOFX.Matrix"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 2

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kParamLuminanceMath "luminanceMath"
#define kParamLuminanceMathLabel "luma math"
#define kParamLuminanceMathHint "Formula used to compute luma from RGB values."
#define kParamLuminanceMathOptionRec709 "Rec. 709"
#define kParamLuminanceMathOptionRec709Hint "Use Rec.709 (0.2126r + 0.7152g + 0.0722b)."
#define kParamLuminanceMathOptionRec2020 "Rec. 2020"
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

////////////////////////////////////////////////////////////////////////////////

namespace {
struct RGBValues {
double r,g,b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};
}

inline float _Luma(float R, float G, float B, int L) {
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

inline float _Sat(float r, float g, float b){
float min = fmin(fmin(r, g), b);
float max = fmax(fmax(r, g), b);
float delta = max - min;
float S = max != 0.0f ? delta / max : 0.0f;
return S;
}

class Matrix : public OFX::ImageProcessor
{
public:
explicit Matrix(OFX::ImageEffect& p_Instance);

virtual void processImagesCUDA();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(float* p_Matrix, int p_Luma, int p_Sat, int p_LumaMath);

private:
OFX::Image* _srcImg;
float _matrix[9];
int _luma;
int _sat;
int _lumaMath;
};

Matrix::Matrix(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, 
int p_Height, float* p_Matrix, int p_Luma, int p_Sat, int p_LumaMath);

void Matrix::processImagesCUDA()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunCudaKernel(input, output, width, height, _matrix, _luma, _sat, _lumaMath);
}

void Matrix::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
{
if (_effect.abort()) break;
float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
{
float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
if (srcPix) {
float red = srcPix[0] * _matrix[0] + srcPix[1] * _matrix[1] + srcPix[2] * _matrix[2];
float green = srcPix[0] * _matrix[3] + srcPix[1] * _matrix[4] + srcPix[2] * _matrix[5];
float blue = srcPix[0] * _matrix[6] + srcPix[1] * _matrix[7] + srcPix[2] * _matrix[8];

if (_luma == 1) {
float inLuma = _Luma(srcPix[0], srcPix[1], srcPix[2], _lumaMath);
float outLuma = _Luma(red, green, blue, _lumaMath);
red = red * (inLuma / outLuma);
green = green * (inLuma / outLuma);
blue = blue * (inLuma / outLuma);
}

if (_sat == 1) {
float inSat = _Sat(srcPix[0], srcPix[1], srcPix[2]);
float outSat = _Sat(red, green, blue);
float satgap = inSat / outSat;
float sLuma = _Luma(red, green, blue, _lumaMath);
float sr = (1.0f - satgap) * sLuma + red * satgap;
float sg = (1.0f - satgap) * sLuma + green * satgap;
float sb = (1.0f - satgap) * sLuma + blue * satgap;
red = inSat == 0.0f ? sLuma : sr;
green = inSat == 0.0f ? sLuma : sg;
blue = inSat == 0.0f ? sLuma : sb;
}
dstPix[0] = red;
dstPix[1] = green;
dstPix[2] = blue;
dstPix[3] = srcPix[3];
} else {
for (int c = 0; c < 4; ++c) {
dstPix[c] = 0;
}}
dstPix += 4;
}}}

void Matrix::setSrcImg(OFX::Image* p_SrcImg) {
_srcImg = p_SrcImg;
}

void Matrix::setScales(float* p_Matrix, int p_Luma, int p_Sat, int p_LumaMath) {
_matrix[0] = p_Matrix[0];
_matrix[1] = p_Matrix[1];
_matrix[2] = p_Matrix[2];
_matrix[3] = p_Matrix[3];
_matrix[4] = p_Matrix[4];
_matrix[5] = p_Matrix[5];
_matrix[6] = p_Matrix[6];
_matrix[7] = p_Matrix[7];
_matrix[8] = p_Matrix[8];
_luma = p_Luma;
_sat = p_Sat;
_lumaMath = p_LumaMath;
}

////////////////////////////////////////////////////////////////////////////////

class MatrixPlugin : public OFX::ImageEffect
{
public:
explicit MatrixPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

void setupAndProcess(Matrix &p_Matrix, const OFX::RenderArguments& p_Args);

float _Luma(float R, float G, float B, int L) {
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

float _Sat(float r, float g, float b) {
float min = fmin(fmin(r, g), b);
float max = fmax(fmax(r, g), b);
float delta = max - min;
float S = max != 0.0f ? delta / max : 0.0f;
return S;
}

private:

OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;

OFX::Double3DParam* m_MatrixR;
OFX::Double3DParam* m_MatrixG;
OFX::Double3DParam* m_MatrixB;
OFX::BooleanParam* m_Luma;
OFX::BooleanParam* m_Sat;
OFX::ChoiceParam* m_LuminanceMath;
OFX::DoubleParam* m_RedRed;
OFX::DoubleParam* m_RedGreen;
OFX::DoubleParam* m_RedBlue;
OFX::DoubleParam* m_GreenRed;
OFX::DoubleParam* m_GreenGreen;
OFX::DoubleParam* m_GreenBlue;
OFX::DoubleParam* m_BlueRed;
OFX::DoubleParam* m_BlueGreen;
OFX::DoubleParam* m_BlueBlue;
OFX::IntParam* m_Cube;
OFX::StringParam* m_Name;
OFX::PushButtonParam* m_Invert;
OFX::PushButtonParam* m_Info;
OFX::StringParam* m_Path;
OFX::StringParam* m_Path2;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
OFX::PushButtonParam* m_Button3;
OFX::PushButtonParam* m_Button4;
};

MatrixPlugin::MatrixPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

m_MatrixR = fetchDouble3DParam("MatrixR");
m_MatrixG = fetchDouble3DParam("MatrixG");
m_MatrixB = fetchDouble3DParam("MatrixB");
m_Luma = fetchBooleanParam("Luma");
m_Sat = fetchBooleanParam("Sat");
m_LuminanceMath = fetchChoiceParam(kParamLuminanceMath);
m_RedRed = fetchDoubleParam("ScaleRR");
m_RedGreen = fetchDoubleParam("ScaleRG");
m_RedBlue = fetchDoubleParam("ScaleRB");
m_GreenRed = fetchDoubleParam("ScaleGR");
m_GreenGreen = fetchDoubleParam("ScaleGG");
m_GreenBlue = fetchDoubleParam("ScaleGB");
m_BlueRed = fetchDoubleParam("ScaleBR");
m_BlueGreen = fetchDoubleParam("ScaleBG");
m_BlueBlue = fetchDoubleParam("ScaleBB");
m_Cube = fetchIntParam("Cube");
m_Name = fetchStringParam("Name");
m_Invert = fetchPushButtonParam("Invert");
m_Info = fetchPushButtonParam("Info");
m_Path = fetchStringParam("Path");
m_Path2 = fetchStringParam("Path2");
m_Button1 = fetchPushButtonParam("Button1");
m_Button2 = fetchPushButtonParam("Button2");
m_Button3 = fetchPushButtonParam("Button3");
m_Button4 = fetchPushButtonParam("Button4");
}

void MatrixPlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
{
Matrix imageScaler(*this);
setupAndProcess(imageScaler, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool MatrixPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
RGBValues rScale, gScale, bScale;
m_MatrixR->getValueAtTime(p_Args.time, rScale.r, rScale.g, rScale.b);
m_MatrixG->getValueAtTime(p_Args.time, gScale.r, gScale.g, gScale.b);
m_MatrixB->getValueAtTime(p_Args.time, bScale.r, bScale.g, bScale.b);

if ((rScale.r == 1.0) && (rScale.g == 0.0) && (rScale.b == 0.0) && (gScale.r == 0.0) && (gScale.g == 1.0) && 
(gScale.b == 0.0) && (bScale.r == 0.0) && (bScale.g == 0.0) && (bScale.b == 1.0)) {
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void MatrixPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
RGBValues rScale, gScale, bScale;

if ((p_ParamName == "MatrixR" || p_ParamName == "MatrixG" || p_ParamName == "MatrixB") && p_Args.reason == OFX::eChangeUserEdit)
{
m_MatrixR->getValueAtTime(p_Args.time, rScale.r, rScale.g, rScale.b);
m_MatrixG->getValueAtTime(p_Args.time, gScale.r, gScale.g, gScale.b);
m_MatrixB->getValueAtTime(p_Args.time, bScale.r, bScale.g, bScale.b);

beginEditBlock("3x3 Matrix");
m_RedRed->setValue(rScale.r);
m_RedGreen->setValue(rScale.g);
m_RedBlue->setValue(rScale.b);
m_GreenRed->setValue(gScale.r);
m_GreenGreen->setValue(gScale.g);
m_GreenBlue->setValue(gScale.b);
m_BlueRed->setValue(bScale.r);
m_BlueGreen->setValue(bScale.g);
m_BlueBlue->setValue(bScale.b);
endEditBlock();
}

if ((p_ParamName == "ScaleRR" || p_ParamName == "ScaleRG" || p_ParamName == "ScaleRB") && p_Args.reason == OFX::eChangeUserEdit)
{
float redred = m_RedRed->getValueAtTime(p_Args.time);
float redgreen = m_RedGreen->getValueAtTime(p_Args.time);
float redblue = m_RedBlue->getValueAtTime(p_Args.time);
m_MatrixR->setValue(redred, redgreen, redblue);
}

if ((p_ParamName == "ScaleGR" || p_ParamName == "ScaleGG" || p_ParamName == "ScaleGB") && p_Args.reason == OFX::eChangeUserEdit)
{
float greenred = m_GreenRed->getValueAtTime(p_Args.time);
float greengreen = m_GreenGreen->getValueAtTime(p_Args.time);
float greenblue = m_GreenBlue->getValueAtTime(p_Args.time);
m_MatrixG->setValue(greenred, greengreen, greenblue);
}

if ((p_ParamName == "ScaleBR" || p_ParamName == "ScaleBG" || p_ParamName == "ScaleBB") && p_Args.reason == OFX::eChangeUserEdit)
{
float bluered = m_BlueRed->getValueAtTime(p_Args.time);
float bluegreen = m_BlueGreen->getValueAtTime(p_Args.time);
float blueblue = m_BlueBlue->getValueAtTime(p_Args.time);
m_MatrixB->setValue(bluered, bluegreen, blueblue);
}

if (p_ParamName == "Normalise")
{
OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Get Normalised version of 3x3 Matrix");
if (reply == OFX::Message::eMessageReplyYes) {
m_MatrixR->getValueAtTime(p_Args.time, rScale.r, rScale.g, rScale.b);
m_MatrixG->getValueAtTime(p_Args.time, gScale.r, gScale.g, gScale.b);
m_MatrixB->getValueAtTime(p_Args.time, bScale.r, bScale.g, bScale.b);
float MatrixA[9] = {rScale.r, rScale.g, rScale.b, gScale.r, gScale.g, gScale.b, bScale.r, bScale.g, bScale.b};
float midi;
float Matmin = MatrixA[0];
for(int m = 1; m < 9; m++) {
midi = fmin(Matmin, MatrixA[m]);
Matmin = midi;
}

float rrm = rScale.r - Matmin;
float rgm = rScale.g - Matmin;
float rbm = rScale.b - Matmin;
float grm = gScale.r - Matmin;
float ggm = gScale.g - Matmin;
float gbm = gScale.b - Matmin;
float brm = bScale.r - Matmin;
float bgm = bScale.g - Matmin;
float bbm = bScale.b - Matmin;

float MatrixB[9] = {rrm, rgm, rbm, grm, ggm, gbm, brm, bgm, bbm};
float Matmax = MatrixB[0];
for(int m = 1; m < 9; m++) {
midi = fmax(Matmax, MatrixB[m]);
Matmax = midi;
}

float rr = rrm / Matmax;
float rg = rgm / Matmax;
float rb = rbm / Matmax;
float gr = grm / Matmax;
float gg = ggm / Matmax;
float gb = gbm / Matmax;
float br = brm / Matmax;
float bg = bgm / Matmax;
float bb = bbm / Matmax;

beginEditBlock("Matrix");	
m_MatrixR->setValue(rr, rg, rb);
m_MatrixG->setValue(gr, gg, gb);
m_MatrixB->setValue(br, bg, bb);
m_RedRed->setValue(rr);
m_RedGreen->setValue(rg);
m_RedBlue->setValue(rb);
m_GreenRed->setValue(gr);
m_GreenGreen->setValue(gg);
m_GreenBlue->setValue(gb);
m_BlueRed->setValue(br);
m_BlueGreen->setValue(bg);
m_BlueBlue->setValue(bb);
endEditBlock();
}}

if (p_ParamName == "Invert")
{
OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Get Inverse of 3x3 Matrix");
if (reply == OFX::Message::eMessageReplyYes) {

m_MatrixR->getValueAtTime(p_Args.time, rScale.r, rScale.g, rScale.b);
m_MatrixG->getValueAtTime(p_Args.time, gScale.r, gScale.g, gScale.b);
m_MatrixB->getValueAtTime(p_Args.time, bScale.r, bScale.g, bScale.b);

float A = gScale.g * bScale.b - gScale.b * bScale.g;
float B = -(gScale.r * bScale.b - gScale.b * bScale.r);
float C = gScale.r * bScale.g - gScale.g * bScale.r;
float D = -(rScale.g * bScale.b - rScale.b * bScale.g);
float E = rScale.r * bScale.b - rScale.b * bScale.r;
float F = -(rScale.r * bScale.g - rScale.g * bScale.r);
float G = rScale.g * gScale.b - rScale.b * gScale.g;
float H = -(rScale.r * gScale.b - rScale.b * gScale.r);
float I = rScale.r * gScale.g - rScale.g * gScale.r;

float det = rScale.r * A + rScale.g * B + rScale.b * C;

float rr = A / det;
float rg = D / det;
float rb = G / det;
float gr = B / det;
float gg = E / det;
float gb = H / det;
float br = C / det;
float bg = F / det;
float bb = I / det;

beginEditBlock("RGB Mixer");	
m_MatrixR->setValue(rr, rg, rb);
m_MatrixG->setValue(gr, gg, gb);
m_MatrixB->setValue(br, bg, bb);
m_RedRed->setValue(rr);
m_RedGreen->setValue(rg);
m_RedBlue->setValue(rb);
m_GreenRed->setValue(gr);
m_GreenGreen->setValue(gg);
m_GreenBlue->setValue(gb);
m_BlueRed->setValue(br);
m_BlueGreen->setValue(bg);
m_BlueBlue->setValue(bb);
endEditBlock();
}}

if(p_ParamName == "Info")
{
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}	

if(p_ParamName == "Button1")
{
m_MatrixR->getValueAtTime(p_Args.time, rScale.r, rScale.g, rScale.b);
m_MatrixG->getValueAtTime(p_Args.time, gScale.r, gScale.g, gScale.b);
m_MatrixB->getValueAtTime(p_Args.time, bScale.r, bScale.g, bScale.b);
float Matrix[9] = {rScale.r, rScale.g, rScale.b, gScale.r, gScale.g, gScale.b, bScale.r, bScale.g, bScale.b};

int lumaMath;
m_LuminanceMath->getValueAtTime(p_Args.time, lumaMath);
bool luma = m_Luma->getValueAtTime(p_Args.time);
int lumaSwitch = luma ? 1 : 0;
bool sat = m_Sat->getValueAtTime(p_Args.time);
int satSwitch = sat ? 1 : 0;

string PATH;
m_Path->getValue(PATH);
string NAME;
m_Name->getValue(NAME);

OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
if (reply == OFX::Message::eMessageReplyYes) {

FILE * pFile;
pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "// Matrix DCTL export\n" \
"\n" \
"__DEVICE__ float Luma(float R, float G, float B, int L) { \n" \
"float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f; \n" \
"float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f; \n" \
"float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f; \n" \
"float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f; \n" \
"float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f; \n" \
"float lumaAvg = (R + G + B) / 3.0f; \n" \
"float lumaMax = _fmaxf(_fmaxf(R, G), B); \n" \
"float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax; \n" \
"return Lu; \n" \
"} \n" \
" \n" \
"__DEVICE__ float Sat(float r, float g, float b){ \n" \
"float min = _fminf(_fminf(r, g), b); \n" \
"float max = _fmaxf(_fmaxf(r, g), b); \n" \
"float delta = max - min; \n" \
"float S = max != 0.0f ? delta / max : 0.0f; \n" \
"return S; \n" \
"} \n" \
" \n" \
"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B) \n" \
"{ \n" \
"const float Matrix[9] =  {%ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff}; \n" \
"int lumaSwitch = %d; \n" \
"int satSwitch = %d; \n" \
"int lumaMath = %d; \n" \
" \n" \
"float red = p_R * Matrix[0] + p_G * Matrix[1] + p_B * Matrix[2]; \n" \
"float green = p_R * Matrix[3] + p_G * Matrix[4] + p_B * Matrix[5]; \n" \
"float blue = p_R * Matrix[6] + p_G * Matrix[7] + p_B * Matrix[8]; \n" \
"if (lumaSwitch == 1) { \n" \
"float inLuma = Luma(p_R, p_G, p_B, lumaMath); \n" \
"float outLuma = Luma(red, green, blue, lumaMath); \n" \
"red = red * (inLuma / outLuma); \n" \
"green = green * (inLuma / outLuma); \n" \
"blue = blue * (inLuma / outLuma); \n" \
"} \n" \
" \n" \
"if (satSwitch == 1) { \n" \
"float inSat = Sat(p_R, p_G, p_B); \n" \
"float outSat = Sat(red, green, blue); \n" \
"float satgap = inSat / outSat; \n" \
"float sLuma = Luma(red, green, blue, lumaMath); \n" \
"float sr = (1.0f - satgap) * sLuma + red * satgap; \n" \
"float sg = (1.0f - satgap) * sLuma + green * satgap; \n" \
"float sb = (1.0f - satgap) * sLuma + blue * satgap; \n" \
"red = inSat == 0.0f ? sLuma : sr; \n" \
"green = inSat == 0.0f ? sLuma : sg; \n" \
"blue = inSat == 0.0f ? sLuma : sb; \n" \
"} \n" \
"return make_float3(red, green, blue); \n" \
"}\n", Matrix[0], Matrix[1], Matrix[2], Matrix[3], Matrix[4], 
Matrix[5], Matrix[6], Matrix[7], Matrix[8], lumaSwitch, satSwitch, lumaMath);
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
}}}

if(p_ParamName == "Button2")
{
m_MatrixR->getValueAtTime(p_Args.time, rScale.r, rScale.g, rScale.b);
m_MatrixG->getValueAtTime(p_Args.time, gScale.r, gScale.g, gScale.b);
m_MatrixB->getValueAtTime(p_Args.time, bScale.r, bScale.g, bScale.b);
float Matrix[9] = {rScale.r, rScale.g, rScale.b, gScale.r, gScale.g, gScale.b, bScale.r, bScale.g, bScale.b};

bool luma = m_Luma->getValueAtTime(p_Args.time);
int lumaswitch = luma ? 1 : 0;
bool sat = m_Sat->getValueAtTime(p_Args.time);
int satswitch = sat ? 1 : 0;

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
"inputs 0\n" \
"name Matrix\n" \
"selected true\n" \
"xpos -231\n" \
"ypos -318\n" \
"}\n" \
"Input {\n" \
"inputs 0\n" \
"name Input1\n" \
"selected true\n" \
"xpos -29\n" \
"ypos -318\n" \
"}\n" \
"NoOp {\n" \
"name NoOp1\n" \
"xpos -29\n" \
"ypos -277\n" \
"}\n" \
"set N3aa0a920 [stack 0]\n" \
"Colorspace {\n" \
"colorspace_in sRGB\n" \
"colorspace_out HSL\n" \
"name HSL_orig\n" \
"xpos -135\n" \
"ypos -225\n" \
"}\n" \
"push $N3aa0a920\n" \
"Expression {\n" \
"expr0 \"r*%f + g*%f + b*%f\"\n" \
"expr1 \"r*%f + g*%f + b*%f\"\n" \
"expr2 \"r*%f + g*%f + b*%f\"\n" \
"name Matrix_3x3\n" \
"xpos 78\n" \
"ypos -225\n" \
"}\n" \
"Colorspace {\n" \
"colorspace_in sRGB\n" \
"colorspace_out YCbCr\n" \
"name YUV_after\n" \
"xpos 78\n" \
"ypos -162\n" \
"}\n" \
"set N3b5f1f10 [stack 0]\n" \
"push $N3aa0a920\n" \
"Colorspace {\n" \
"colorspace_in sRGB\n" \
"colorspace_out YCbCr\n" \
"name YUV_orig\n" \
"xpos -29\n" \
"ypos -224\n" \
"}\n" \
"ShuffleCopy {\n" \
"inputs 2\n" \
"in rgb\n" \
"in2 rgb\n" \
"green green\n" \
"blue blue\n" \
"out rgb\n" \
"name Yorig_UVafter\n" \
"xpos -29\n" \
"ypos -189\n" \
"}\n" \
"push $N3b5f1f10\n" \
"Switch {\n" \
"inputs 2\n" \
"which %d\n" \
"name Luma_Switch\n" \
"xpos 40\n" \
"ypos -105\n" \
"}\n" \
"Colorspace {\n" \
"colorspace_in YCbCr\n" \
"colorspace_out sRGB\n" \
"name RGB_after_YUV\n" \
"xpos 31\n" \
"ypos -68\n" \
"}\n" \
"Colorspace {\n" \
"colorspace_in sRGB\n" \
"colorspace_out HSL\n" \
"name HSL_after\n" \
"xpos 31\n" \
"ypos -30\n" \
"}\n" \
"set N3c72b600 [stack 0]\n" \
"ShuffleCopy {\n" \
"inputs 2\n" \
"in rgb\n" \
"in2 rgb\n" \
"green green\n" \
"out rgb\n" \
"name H_after_S_orig_L_after\n" \
"xpos -25\n" \
"ypos 12\n" \
"}\n" \
"push $N3c72b600\n" \
"Switch {\n" \
"inputs 2\n" \
"which %d\n" \
"name Sat_Switch\n" \
"xpos 12\n" \
"ypos 48\n" \
"}\n" \
"Colorspace {\n" \
"colorspace_in HSL\n" \
"colorspace_out sRGB\n" \
"name RGB_after_HSL\n" \
"xpos 12\n" \
"ypos 85\n" \
"}\n" \
"Output {\n" \
"name Output1\n" \
"xpos 12\n" \
"ypos 133\n" \
"}\n" \
"end_group\n", Matrix[0], Matrix[1], Matrix[2], Matrix[3], Matrix[4], 
Matrix[5], Matrix[6], Matrix[7], Matrix[8], lumaswitch, satswitch);
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
}}}

if(p_ParamName == "Button3")
{
m_MatrixR->getValueAtTime(p_Args.time, rScale.r, rScale.g, rScale.b);
m_MatrixG->getValueAtTime(p_Args.time, gScale.r, gScale.g, gScale.b);
m_MatrixB->getValueAtTime(p_Args.time, bScale.r, bScale.g, bScale.b);
float Matrix[9] = {rScale.r, rScale.g, rScale.b, gScale.r, gScale.g, gScale.b, bScale.r, bScale.g, bScale.b};

int lumaMath;
m_LuminanceMath->getValueAtTime(p_Args.time, lumaMath);
bool luma = m_Luma->getValueAtTime(p_Args.time);
int lumaSwitch = luma ? 1 : 0;
bool sat = m_Sat->getValueAtTime(p_Args.time);
int satSwitch = sat ? 1 : 0;
int cube = (int)m_Cube->getValueAtTime(p_Args.time);
int total = cube * cube * cube;

string PATH;
m_Path->getValue(PATH);
string NAME;
m_Name->getValue(NAME);

OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".cube to " + PATH + "?");
if (reply == OFX::Message::eMessageReplyYes) {

FILE * pFile;

pFile = fopen ((PATH + "/" + NAME + ".cube").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "# Resolve 3D LUT export\n" \
"LUT_3D_SIZE %d\n" \
"LUT_3D_INPUT_RANGE 0.0 1.0\n" \
"\n", cube);
for( int i = 0; i < total; ++i ){
int r = fmod(i, cube);
int g = fmod(floor(i / cube), cube);
int b = fmod(floor(i / (cube * cube)), cube);
float R = (float)r / (cube - 1);
float G = (float)g / (cube - 1);
float B = (float)b / (cube - 1);
float red = R * Matrix[0] + G * Matrix[1] + B * Matrix[2];
float green = R * Matrix[3] + G * Matrix[4] + B * Matrix[5];
float blue = R * Matrix[6] + G * Matrix[7] + B * Matrix[8];
if (luma == 1) {
float inLuma = _Luma(R, G, B, lumaMath);
float outLuma = _Luma(red, green, blue, lumaMath);
red = red * (inLuma / outLuma);
green = green * (inLuma / outLuma);
blue = blue * (inLuma / outLuma);
}
if (sat == 1) {
float inSat = _Sat(R, G, B);
float outSat = _Sat(red, green, blue);
float satgap = inSat / outSat;
float sLuma = _Luma(red, green, blue, lumaMath);
float sr = (1.0f - satgap) * sLuma + red * satgap;
float sg = (1.0f - satgap) * sLuma + green * satgap;
float sb = (1.0f - satgap) * sLuma + blue * satgap;
red = inSat == 0.0f ? sLuma : sr;
green = inSat == 0.0f ? sLuma : sg;
blue = inSat == 0.0f ? sLuma : sb;
}
fprintf (pFile, "%f %f %f\n", red, green, blue);
}
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".cube to " + PATH  + ". Check Permissions."));
}}}

if(p_ParamName == "Button4")
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
"// Matrix Shader  \n" \
"  \n" \
"#version 120  \n" \
"uniform sampler2D front;  \n" \
"uniform float adsk_result_w, adsk_result_h;  \n" \
"\n");
fclose (pFile);
fprintf (pFile2,
"<ShaderNodePreset SupportsAdaptiveDegradation=\"0\" Description=\"Matrix\" Name=\"Matrix\"> \n" \
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
}}}}

void MatrixPlugin::setupAndProcess(Matrix& p_Matrix, const OFX::RenderArguments& p_Args)
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

int _lumaMath;
m_LuminanceMath->getValueAtTime(p_Args.time, _lumaMath);

float _matrix[9];

RGBValues rScale, gScale, bScale;
m_MatrixR->getValueAtTime(p_Args.time, rScale.r, rScale.g, rScale.b);
_matrix[0] = rScale.r;
_matrix[1] = rScale.g;
_matrix[2] = rScale.b;

m_MatrixG->getValueAtTime(p_Args.time, gScale.r, gScale.g, gScale.b);
_matrix[3] = gScale.r;
_matrix[4] = gScale.g;
_matrix[5] = gScale.b;

m_MatrixB->getValueAtTime(p_Args.time, bScale.r, bScale.g, bScale.b);
_matrix[6] = bScale.r;
_matrix[7] = bScale.g;
_matrix[8] = bScale.b;

bool lumaB = m_Luma->getValueAtTime(p_Args.time);
int _luma = lumaB ? 1 : 0;

bool satB = m_Sat->getValueAtTime(p_Args.time);
int _sat = satB ? 1 : 0;

p_Matrix.setDstImg(dst.get());
p_Matrix.setSrcImg(src.get());

// Setup GPU Render arguments
p_Matrix.setGPURenderArgs(p_Args);

p_Matrix.setRenderWindow(p_Args.renderWindow);

p_Matrix.setScales(_matrix, _luma, _sat, _lumaMath);

p_Matrix.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

MatrixPluginFactory::MatrixPluginFactory()
: OFX::PluginFactoryHelper<MatrixPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void MatrixPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
param->setLabel(p_Label);
param->setScriptName(p_Name);
param->setHint(p_Hint);
if (p_Parent) {
param->setParent(*p_Parent);
}
return param;
}

void MatrixPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum)
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

GroupParamDescriptor* matrix = p_Desc.defineGroupParam("3x3 Matrix");
matrix->setOpen(true);
matrix->setHint("3x3 Matrix");
if (page) {
page->addChild(*matrix);
}

Double3DParamDescriptor* paramR = p_Desc.defineDouble3DParam("MatrixR");
paramR->setLabel("Red channel");
paramR->setDefault(1.0, 0.0, 0.0);
paramR->setParent(*matrix);
page->addChild(*paramR);

Double3DParamDescriptor* paramG = p_Desc.defineDouble3DParam("MatrixG");
paramG->setLabel("Green channel");
paramG->setDefault(0.0, 1.0, 0.0);
paramG->setParent(*matrix);
page->addChild(*paramG);

Double3DParamDescriptor* paramB = p_Desc.defineDouble3DParam("MatrixB");
paramB->setLabel("Blue channel");
paramB->setDefault(0.0, 0.0, 1.0);
paramB->setParent(*matrix);
page->addChild(*paramB);

PushButtonParamDescriptor* pushparam = p_Desc.definePushButtonParam("Invert");
pushparam->setLabel("inverse matrix");
pushparam->setParent(*matrix);
page->addChild(*pushparam);

pushparam = p_Desc.definePushButtonParam("Normalise");
pushparam->setLabel("normalise matrix");
pushparam->setParent(*matrix);
page->addChild(*pushparam);

BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("Luma");
boolParam->setDefault(false);
boolParam->setHint("Preserves original luma");
boolParam->setLabel("preserve luma");
boolParam->setParent(*matrix);
page->addChild(*boolParam);

boolParam = p_Desc.defineBooleanParam("Sat");
boolParam->setDefault(false);
boolParam->setHint("Preserves original saturation");
boolParam->setLabel("preserve saturation");
boolParam->setParent(*matrix);
page->addChild(*boolParam);

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
choiceparam->setParent(*matrix);
page->addChild(*choiceparam);

pushparam = p_Desc.definePushButtonParam("Info");
pushparam->setLabel("info");
pushparam->setParent(*matrix);
page->addChild(*pushparam);

GroupParamDescriptor* mixer = p_Desc.defineGroupParam("RGB Mixer");
mixer->setOpen(false);
mixer->setHint("RGB Mixer");
if (page) {
page->addChild(*mixer);
}

DoubleParamDescriptor* param = defineScaleParam(p_Desc, "ScaleRR", "Red red", "Red component of Red channel", mixer);
param->setDefault(1.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleRG", "Red green", "Green component of Red channel", mixer);
param->setDefault(0.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleRB", "Red blue", "Blue component of Red channel", mixer);
param->setDefault(0.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleGR", "Green red", "Red component of Green channel", mixer);
param->setDefault(0.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleGG", "Green green", "Green component of Green channel", mixer);
param->setDefault(1.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleGB", "Green blue", "Blue component of Green channel", mixer);
param->setDefault(0.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleBR", "Blue red", "Red component of Blue channel", mixer);
param->setDefault(0.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleBG", "Blue green", "Green component of Blue channel", mixer);
param->setDefault(0.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

param = defineScaleParam(p_Desc, "ScaleBB", "Blue blue", "Blue component of Blue channel", mixer);
param->setDefault(1.0);
param->setRange(-10.0, 10.0);
param->setIncrement(0.001);
param->setDisplayRange(-2.0, 2.0);
page->addChild(*param);

GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
script->setOpen(false);
script->setHint("export dctl and nuke script");
if (page) {
page->addChild(*script);
}

pushparam = p_Desc.definePushButtonParam("Button1");
pushparam->setLabel("export dctl");
pushparam->setHint("create dctl version");
pushparam->setParent(*script);
page->addChild(*pushparam);

pushparam = p_Desc.definePushButtonParam("Button2");
pushparam->setLabel("export nuke script");
pushparam->setHint("create nuke version");
pushparam->setParent(*script);
page->addChild(*pushparam);

pushparam = p_Desc.definePushButtonParam("Button3");
pushparam->setLabel("export 3d lut");
pushparam->setHint("create 3d look-up table");
pushparam->setParent(*script);
page->addChild(*pushparam);

IntParamDescriptor* Param = p_Desc.defineIntParam("Cube");
Param->setLabel("cube size");
Param->setHint("3d lut cube size");
Param->setDefault(33);
Param->setRange(3, 129);
Param->setDisplayRange(3, 129);
Param->setParent(*script);
page->addChild(*Param);

StringParamDescriptor* stringparam = p_Desc.defineStringParam("Name");
stringparam->setLabel("name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("Matrix");
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

pushparam = p_Desc.definePushButtonParam("Button4");
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

ImageEffect* MatrixPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
return new MatrixPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static MatrixPluginFactory MatrixPlugin;
p_FactoryArray.push_back(&MatrixPlugin);
}