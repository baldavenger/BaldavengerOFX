#include "ConvolutionPlugin.h"

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

#define kPluginName "Convolution"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Convolution Filters"

#define kPluginIdentifier "BaldavengerOFX.Convolution"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 3

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kResolutionScale	(float)width / 1920.0f

#define kParamConvolution "ConvolutionFilter"
#define kParamConvolutionLabel "filter type"
#define kParamConvolutionHint "types of convolution filter"
#define kParamConvolutionOptionGaus "Gaussian Blur"
#define kParamConvolutionOptionGausHint "recursive gaussian blur"
#define kParamConvolutionOptionSimple "Simple Blur"
#define kParamConvolutionOptionSimpleHint "simple recursive blur"
#define kParamConvolutionOptionBox "Box Blur"
#define kParamConvolutionOptionBoxHint "box blur"
#define kParamConvolutionOptionFrequency "Frequency Separation"
#define kParamConvolutionOptionFrequencyHint "blur low frequency, sharpen high frequency"
#define kParamConvolutionOptionEdgeDetect "Edge Detect"
#define kParamConvolutionOptionEdgeDetectHint "edge detect"
#define kParamConvolutionOptionEdgeEnhance "Edge Enhance"
#define kParamConvolutionOptionEdgeEnhanceHint "edge enhance"
#define kParamConvolutionOptionErode "Erode"
#define kParamConvolutionOptionErodeHint "erodes dark area of matte"
#define kParamConvolutionOptionDilate "Dilate"
#define kParamConvolutionOptionDilateHint "expands dark area of matte"
#define kParamConvolutionOptionScatter "Scatter"
#define kParamConvolutionOptionScatterHint "disperse pixels within controlled range"
#define kParamConvolutionOptionCustom "Custom"
#define kParamConvolutionOptionCustomHint "custom 3x3 spatial matrix"

enum ConvolutionEnum
{
eConvolutionGaus,
eConvolutionSimple,
eConvolutionBox,
eConvolutionFrequency,
eConvolutionEdgeDetect,
eConvolutionEdgeEnhance,
eConvolutionErode,
eConvolutionDilate,
eConvolutionScatter,
eConvolutionCustom,
};

////////////////////////////////////////////////////////////////////////////////

namespace {
struct RGBValues {
double r,g,b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};
}

class Convolution : public OFX::ImageProcessor
{
public:
explicit Convolution(OFX::ImageEffect& p_Instance);

//virtual void processImagesCUDA();
//virtual void processImagesOpenCL();
//virtual void processImagesMetal();
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

void setSrcImg(OFX::Image* p_SrcImg);
void setScales(int p_Convolve, int p_Display, float* p_Adjust, float* p_Matrix);

private:
OFX::Image* _srcImg;
int _convolve;
int _display;
float _adjust[3];
float _matrix[9];
};

Convolution::Convolution(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}
/*
extern void RunCudaKernel(float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Convolve, int p_Display, float* p_Adjust, float* p_Matrix);

void Convolution::processImagesCUDA()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

_adjust[0] = _adjust[0] * kResolutionScale;
_adjust[2] = _adjust[2] * kResolutionScale;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunCudaKernel(input, output, width, height, _convolve, _display, _adjust, _matrix);
}

extern void RunOpenCLKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Convolve, int p_Display, float* p_Adjust, float* p_Matrix);

void Convolution::processImagesOpenCL()
{
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

_adjust[0] = _adjust[0] * kResolutionScale;
_adjust[2] = _adjust[2] * kResolutionScale;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunOpenCLKernel(_pOpenCLCmdQ, input, output, width, height, _convolve, _display, _adjust, _matrix);
}

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, float* p_Input, float* p_Output, int p_Width, int p_Height, 
int p_Convolve, int p_Display, float* p_Adjust, float* p_Matrix);
#endif

void Convolution::processImagesMetal()
{
#ifdef __APPLE__
const OfxRectI& bounds = _srcImg->getBounds();
const int width = bounds.x2 - bounds.x1;
const int height = bounds.y2 - bounds.y1;

_adjust[0] = _adjust[0] * kResolutionScale;
_adjust[2] = _adjust[2] * kResolutionScale;

float* input = static_cast<float*>(_srcImg->getPixelData());
float* output = static_cast<float*>(_dstImg->getPixelData());

RunMetalKernel(_pMetalCmdQ, input, output, width, height, _convolve, _display, _adjust, _matrix);
#endif
}
*/

void Convolution::multiThreadProcessImages(OfxRectI p_ProcWindow)
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

void Convolution::setSrcImg(OFX::Image* p_SrcImg)
{
_srcImg = p_SrcImg;
}

void Convolution::setScales(int p_Convolve, int p_Display, float* p_Adjust, float* p_Matrix)
{
_convolve = p_Convolve;
_display = p_Display;
_adjust[0] = p_Adjust[0];
_adjust[1] = p_Adjust[1];
_adjust[2] = p_Adjust[2];
_matrix[0] = p_Matrix[0];
_matrix[1] = p_Matrix[1];
_matrix[2] = p_Matrix[2];
_matrix[3] = p_Matrix[3];
_matrix[4] = p_Matrix[4];
_matrix[5] = p_Matrix[5];
_matrix[6] = p_Matrix[6];
_matrix[7] = p_Matrix[7];
_matrix[8] = p_Matrix[8];
}

////////////////////////////////////////////////////////////////////////////////

class ConvolutionPlugin : public OFX::ImageEffect
{
public:
explicit ConvolutionPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);

virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

void setupAndProcess(Convolution &p_Convolution, const OFX::RenderArguments& p_Args);

private:

OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;

OFX::ChoiceParam* m_Convolve;
OFX::DoubleParam* m_Adjust1;
OFX::DoubleParam* m_Adjust2;
OFX::DoubleParam* m_Adjust3;
OFX::BooleanParam* m_Display;
OFX::Double3DParam* m_Row1;
OFX::Double3DParam* m_Row2;
OFX::Double3DParam* m_Row3;

OFX::StringParam* m_Path;
OFX::StringParam* m_Path2;
OFX::StringParam* m_Name;
OFX::PushButtonParam* m_Info;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
OFX::PushButtonParam* m_Button3;
};

ConvolutionPlugin::ConvolutionPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

m_Convolve = fetchChoiceParam(kParamConvolution);
m_Adjust1 = fetchDoubleParam("Adjust1");
m_Adjust2 = fetchDoubleParam("Adjust2");
m_Adjust3 = fetchDoubleParam("Adjust3");
m_Display = fetchBooleanParam("Display");
m_Row1 = fetchDouble3DParam("Row1");
m_Row2 = fetchDouble3DParam("Row2");
m_Row3 = fetchDouble3DParam("Row3");

m_Path = fetchStringParam("Path");
m_Path2 = fetchStringParam("Path2");
m_Name = fetchStringParam("Name");
m_Info = fetchPushButtonParam("Info");
m_Button1 = fetchPushButtonParam("Button1");
m_Button2 = fetchPushButtonParam("Button2");
m_Button3 = fetchPushButtonParam("Button3");
}

void ConvolutionPlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
{
Convolution Convolution(*this);
setupAndProcess(Convolution, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool ConvolutionPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
if (m_SrcClip) {
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void ConvolutionPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if(p_ParamName == kParamConvolution)
{
int convolve_i;
m_Convolve->getValueAtTime(p_Args.time, convolve_i);
ConvolutionEnum ConvolutionFilter = (ConvolutionEnum)convolve_i;

bool freq = convolve_i == 3;
bool edge = convolve_i == 4;
bool enhance = convolve_i == 5;
bool erode = convolve_i == 6;
bool dilate = convolve_i == 7;
bool scatter = convolve_i == 8;
bool cust = convolve_i == 9;

if(erode || dilate) {
m_Adjust1->setLabel("amount");
} else
if(scatter) {
m_Adjust1->setLabel("amount");
m_Adjust2->setLabel("mix");
} else
if(cust || enhance) {
m_Adjust1->setLabel("scale");
m_Display->setLabel("normalise");
} else {
m_Adjust1->setLabel("blur");
m_Adjust2->setLabel("sharpen");
m_Adjust3->setLabel("threshold");
}

if(edge) {
m_Adjust1->setIsSecretAndDisabled(true);
} else {
m_Adjust1->setIsSecretAndDisabled(false);
}

m_Adjust2->setIsSecretAndDisabled(!freq && !scatter);
m_Adjust3->setIsSecretAndDisabled(!freq && !edge);
m_Display->setIsSecretAndDisabled(!freq && !cust);
m_Row1->setIsSecretAndDisabled(!cust);
m_Row2->setIsSecretAndDisabled(!cust);
m_Row3->setIsSecretAndDisabled(!cust);
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
fprintf (pFile, "// ConvolutionPlugin DCTL export\n" \
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
" name Convolution\n" \
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
"// Convolution Shader  \n" \
"  \n" \
"#version 120  \n" \
"uniform sampler2D front;  \n" \
"uniform float adsk_result_w, adsk_result_h;  \n" \
"\n");
fclose (pFile);
fprintf (pFile2,
"<ShaderNodePreset SupportsAdaptiveDegradation=\"0\" Description=\"Convolution\" Name=\"Convolution\"> \n" \
"<Shader OutputBitDepth=\"Output\" Index=\"1\"> \n" \
"</Shader> \n" \
"<Page Name=\"Convolution\" Page=\"0\"> \n" \
"<Col Name=\"Convolution\" Col=\"0\" Page=\"0\"> \n" \
"</Col> \n" \
"</Page> \n" \
"</ShaderNodePreset> \n");
fclose (pFile2);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".glsl and " + NAME + ".xml to " + PATH2  + ". Check Permissions."));
}}}
}

void ConvolutionPlugin::setupAndProcess(Convolution& p_Convolution, const OFX::RenderArguments& p_Args)
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

int convolve_i;
m_Convolve->getValueAtTime(p_Args.time, convolve_i);
ConvolutionEnum ConvolutionFilter = (ConvolutionEnum)convolve_i;
int _convolve = convolve_i;

bool display = m_Display->getValueAtTime(p_Args.time);
int _display = display ? 1 : 0;

float _adjust[3];
float _matrix[9];

_adjust[0] = m_Adjust1->getValueAtTime(p_Args.time);
_adjust[1] = m_Adjust2->getValueAtTime(p_Args.time);
_adjust[2] = m_Adjust3->getValueAtTime(p_Args.time);

RGBValues rMatrix, gMatrix, bMatrix;
m_Row1->getValueAtTime(p_Args.time, rMatrix.r, rMatrix.g, rMatrix.b);
m_Row2->getValueAtTime(p_Args.time, gMatrix.r, gMatrix.g, gMatrix.b);
m_Row3->getValueAtTime(p_Args.time, bMatrix.r, bMatrix.g, bMatrix.b);
_matrix[0] = rMatrix.r;
_matrix[1] = rMatrix.g;
_matrix[2] = rMatrix.b;
_matrix[3] = gMatrix.r;
_matrix[4] = gMatrix.g;
_matrix[5] = gMatrix.b;
_matrix[6] = bMatrix.r;
_matrix[7] = bMatrix.g;
_matrix[8] = bMatrix.b;

p_Convolution.setDstImg(dst.get());
p_Convolution.setSrcImg(src.get());

// Setup GPU Render arguments
//p_Convolution.setGPURenderArgs(p_Args);

// Set the render window
p_Convolution.setRenderWindow(p_Args.renderWindow);

// Set the scales
p_Convolution.setScales(_convolve, _display, _adjust, _matrix);

// Call the base class process member, this will call the derived templated process code
p_Convolution.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ConvolutionPluginFactory::ConvolutionPluginFactory()
: OFX::PluginFactoryHelper<ConvolutionPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ConvolutionPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
//p_Desc.setSupportsOpenCLRender(true);
//p_Desc.setSupportsCudaRender(true);
//#ifdef __APPLE__
//p_Desc.setSupportsMetalRender(true);
//#endif
}

void ConvolutionPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

ChoiceParamDescriptor *choiceparam = p_Desc.defineChoiceParam(kParamConvolution);
choiceparam->setLabel(kParamConvolutionLabel);
choiceparam->setHint(kParamConvolutionHint);
assert(choiceparam->getNOptions() == (int)eConvolutionGaus);
choiceparam->appendOption(kParamConvolutionOptionGaus, kParamConvolutionOptionGausHint);
assert(choiceparam->getNOptions() == (int)eConvolutionSimple);
choiceparam->appendOption(kParamConvolutionOptionSimple, kParamConvolutionOptionSimpleHint);
assert(choiceparam->getNOptions() == (int)eConvolutionBox);
choiceparam->appendOption(kParamConvolutionOptionBox, kParamConvolutionOptionBoxHint);
assert(choiceparam->getNOptions() == (int)eConvolutionFrequency);
choiceparam->appendOption(kParamConvolutionOptionFrequency, kParamConvolutionOptionFrequencyHint);
assert(choiceparam->getNOptions() == (int)eConvolutionEdgeDetect);
choiceparam->appendOption(kParamConvolutionOptionEdgeDetect, kParamConvolutionOptionEdgeDetectHint);
assert(choiceparam->getNOptions() == (int)eConvolutionEdgeEnhance);
choiceparam->appendOption(kParamConvolutionOptionEdgeEnhance, kParamConvolutionOptionEdgeEnhanceHint);
assert(choiceparam->getNOptions() == (int)eConvolutionErode);
choiceparam->appendOption(kParamConvolutionOptionErode, kParamConvolutionOptionErodeHint);
assert(choiceparam->getNOptions() == (int)eConvolutionDilate);
choiceparam->appendOption(kParamConvolutionOptionDilate, kParamConvolutionOptionDilateHint);
assert(choiceparam->getNOptions() == (int)eConvolutionScatter);
choiceparam->appendOption(kParamConvolutionOptionScatter, kParamConvolutionOptionScatterHint);
assert(choiceparam->getNOptions() == (int)eConvolutionCustom);
choiceparam->appendOption(kParamConvolutionOptionCustom, kParamConvolutionOptionCustomHint);
choiceparam->setDefault( (int)eConvolutionGaus );
choiceparam->setAnimates(false);
page->addChild(*choiceparam);

DoubleParamDescriptor* param = p_Desc.defineDoubleParam("Adjust1");
param->setLabel("blur");
param->setHint("adjust convolution");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Adjust2");
param->setLabel("sharpen");
param->setHint("adjust convolution");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

param = p_Desc.defineDoubleParam("Adjust3");
param->setLabel("threshold");
param->setHint("adjust threshold");
param->setDefault(0.0);
param->setRange(0.0, 1.0);
param->setIncrement(0.001);
param->setDisplayRange(0.0, 1.0);
param->setIsSecretAndDisabled(true);
page->addChild(*param);

BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("Display");
boolParam->setLabel("display high frequency");
//boolParam->setHint("display high frequency");
boolParam->setDefault(false);
boolParam->setIsSecretAndDisabled(true);
page->addChild(*boolParam);

Double3DParamDescriptor* paramR = p_Desc.defineDouble3DParam("Row1");
paramR->setLabel("row 1");
paramR->setDefault(0, 0, 0);
paramR->setIncrement(1);
paramR->setIsSecretAndDisabled(true);
page->addChild(*paramR);

Double3DParamDescriptor* paramG = p_Desc.defineDouble3DParam("Row2");
paramG->setLabel("row 2");
paramG->setDefault(0, 1, 0);
paramG->setIncrement(1);
paramG->setIsSecretAndDisabled(true);
page->addChild(*paramG);

Double3DParamDescriptor* paramB = p_Desc.defineDouble3DParam("Row3");
paramB->setLabel("row 3");
paramB->setDefault(0, 0, 0);
paramB->setIncrement(1);
paramB->setIsSecretAndDisabled(true);
page->addChild(*paramB);

PushButtonParamDescriptor* pushparam = p_Desc.definePushButtonParam("Info");
pushparam->setLabel("info");
page->addChild(*pushparam);

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
pushparam->setHint("create nUKE version");
pushparam->setParent(*script);
page->addChild(*pushparam);

StringParamDescriptor* stringparam = p_Desc.defineStringParam("Name");
stringparam->setLabel("name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("Convolution");
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
pushparam->setHint("create shader version");
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

ImageEffect* ConvolutionPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
return new ConvolutionPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static ConvolutionPluginFactory ConvolutionPlugin;
p_FactoryArray.push_back(&ConvolutionPlugin);
}
