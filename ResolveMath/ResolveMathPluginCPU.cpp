#include "ResolveMathPlugin.h"
#include <cstring>
#include <cmath>
#include <cfloat>
#include <stdio.h>
using std::string;
#include <string> 
#include <fstream>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsLog.h"
#include "ofxsProcessing.h"
#include "ofxsMacros.h"
#include "exprtk.hpp"

#ifdef __APPLE__
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT\\"
#else
#define kPluginScript "/home/resolve/LUT/"
#endif

using namespace OFX;
using namespace std;

#define kPluginName "ResolveMath"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"Useful information : The following references can be applied to the expressions \n" \
"Red, Green, Blue channels: r, g, b // Coordinates: x, y // Operators: +, -, *, /, ^, =\n" \
"Functions: min, max, avg, sum, abs, fmod, ceil, floor, round, pow, exp, log, root\n" \
"sqrt, lerp, sin, cos, tan, asin, acos, atan, hypot // Conditionals: ==, !=, >=, && \n" \
"if(a == b, c, d) : If a equals b then c, else d // a == b ? c : d  If a equals b then c, else d\n" \
"clamp(a,b,c) : a clamped to between b and c // pi : 3.1415926536 // width, height"

#define kPluginIdentifier "BaldavengerOFX.ResolveMath"
#define kPluginVersionMajor 1 
#define kPluginVersionMinor 4 

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsRenderScale 1
#define kSupportsMultipleClipPARs false
#define kSupportsMultipleClipDepths true
#define kRenderThreadSafety eRenderFullySafe

static const string  kParamExpr1Name 	= "expr1";
static const string  kParamExpr1Label	= "expr1";
static const string  kParamExpr1Hint 	= "You can define an expression here and reference it in ResolveMath fields as 'expr1'";
static const string  kParamExpr2Name 	= "expr2";
static const string  kParamExpr2Label	= "expr2";
static const string  kParamExpr2Hint 	= "Reference in ResolveMath fields as'expr2'";
static const string  kParamExpr3Name 	= "expr3";
static const string  kParamExpr3Label	= "expr3";
static const string  kParamExpr3Hint 	= "Reference in ResolveMath fields as'expr3'";

static const string  kParamResolveMathR 		= "red";
static const string  kParamResolveMathRLabel	= "Red Output";
static const string  kParamResolveMathRHint 	= "Red Channel output";
static const string  kParamResolveMathG     	= "green";
static const string  kParamResolveMathGLabel	= "Green Output";
static const string  kParamResolveMathGHint 	= "Green Channel output";
static const string  kParamResolveMathB     	= "blue";
static const string  kParamResolveMathBLabel 	= "Blue Output";
static const string  kParamResolveMathBHint 	= "Blue Channel output";
static const string  kParamResolveMathLUT   	= "lut";
static const string  kParamResolveMathLUTLabel 	= "1D expr";
static const string  kParamResolveMathLUTHint 	= "expression for 1D LUT";

static const string  kParamParam1Name 	= "param1";
static const string  kParamParam1Label	= "param1";
static const string  kParamParam1Hint 	= "Reference in ResolveMath fields as 'param1'";

static const string  kParamParam2Name 	= "param2";
static const string  kParamParam2Label	= "param2";
static const string  kParamParam2Hint 	= "Reference in ResolveMath fields as 'param2'";

static const string  kParamParam3Name 	= "param3";
static const string  kParamParam3Label	= "param3";
static const string  kParamParam3Hint 	= "Reference in ResolveMath fields as 'param3'";

static const string  kParamParam4Name 	= "param4";
static const string  kParamParam4Label	= "param4";
static const string  kParamParam4Hint 	= "Reference in ResolveMath fields as 'param4.r||g||b'";

static const string  kParamParam5Name 	= "param5";
static const string  kParamParam5Label	= "mix";
static const string  kParamParam5Hint 	= "Mix factor between original and transformed image";

#define kParamLUT "LUT"
#define kParamLUTLabel "Export"
#define kParamLUTHint "LUT version"
#define kParamLUTOption3D "3D LUT"
#define kParamLUTOption3DHint "3D LUT"
#define kParamLUTOption1D "1D LUT"
#define kParamLUTOption1DHint "1D LUT"
#define kParamLUTOptionCombo "1D + 3D Combo"
#define kParamLUTOptionComboHint "1D Shaper + 3D LUT"

enum LUTEnum
{
eLUT3D,
eLUT1D,
eLUTCombo,
};

namespace {
struct RGBValues {
double r,g,b;
RGBValues(double v) : r(v), g(v), b(v) {}
RGBValues() : r(0), g(0), b(0) {}
};
}

struct ResolveMathProperties {
const string name;
string content;
bool processFlag;
};

namespace {
string replace_pattern(string text, const string& pattern, const string& replacement){
size_t pos = 0;
while ((pos = text.find(pattern,pos)) != string::npos){
text.replace(pos,pattern.length(),replacement);
pos += replacement.length();
}
return text;
}}

class ResolveMath : public OFX::ImageProcessor
{
public:

explicit ResolveMath(OFX::ImageEffect& p_Instance);
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);
void setSrcImg(OFX::Image* p_SrcImg);

void setScales(const string& expr1, const string& expr2, const string& expr3, 
const string& exprR, const string& exprG, const string& exprB, double param1, 
double param2, double param3, const RGBValues& param4, double param5,
bool processR, bool processG, bool processB);

private:
OFX::Image *_srcImg;
string _expr1;
string _expr2;
string _expr3;
string _exprR;
string _exprG;
string _exprB;
double _param1;
double _param2;
double _param3;
RGBValues _param4;
double _param5;
bool _processR;
bool _processG;
bool _processB;
};

ResolveMath::ResolveMath(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

void ResolveMath::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
float temPix[3];
bool doR = _processR;
bool doG = _processG;
bool doB = _processB;
float param1;
float param2;
float param3;
float param4_red;
float param4_green;
float param4_blue;
float param5;
float x_coord;
float y_coord;
float width;
float height;
exprtk::symbol_table<float> symbol_table;
symbol_table.add_constants();
symbol_table.add_variable("r",temPix[0]);
symbol_table.add_variable("g",temPix[1]);
symbol_table.add_variable("b",temPix[2]);
symbol_table.add_variable("param1", param1);
symbol_table.add_variable("param2", param2);
symbol_table.add_variable("param3", param3);
symbol_table.add_variable("param4_r",param4_red);
symbol_table.add_variable("param4_g",param4_green);
symbol_table.add_variable("param4_b",param4_blue);
symbol_table.add_variable("x",x_coord);
symbol_table.add_variable("y",y_coord);
symbol_table.add_variable("width",width);
symbol_table.add_variable("height",height);
ResolveMathProperties expr1_props = {kParamExpr1Name, _expr1, true};
ResolveMathProperties expr2_props = {kParamExpr2Name, _expr2, true};
ResolveMathProperties expr3_props = {kParamExpr3Name, _expr3, true};
ResolveMathProperties exprR_props = {kParamResolveMathR, _exprR, true};
ResolveMathProperties exprG_props = {kParamResolveMathG, _exprG, true};
ResolveMathProperties exprB_props = {kParamResolveMathB, _exprB, true};
const int Esize = 6;
ResolveMathProperties E[Esize] = {expr1_props, expr2_props, expr3_props, 
exprR_props, exprG_props, exprB_props};
for (int i = 0; i != Esize; ++i) {
for (int k = 0; k != Esize; ++ k) {
if (E[i].content.find(E[i].name) != string::npos){
E[i].content.clear();
E[i].processFlag = false;
} else if (i != k && !E[i].content.empty() && !E[k].content.empty() ) { 
E[i].content  = replace_pattern(E[i].content,E[k].name,"("+E[k].content+")");
}}
E[i].content = replace_pattern(E[i].content,"param1.","param1_");
E[i].content = replace_pattern(E[i].content,"param2.","param2_");
E[i].content = replace_pattern(E[i].content,"param3.","param3_");
E[i].content = replace_pattern(E[i].content,"param4.","param4_");
//E[i].content = replace_pattern(E[i].content,"=",":=");
//E[i].content = replace_pattern(E[i].content,":=:=","==");
}
exprtk::function_compositor<float> compositor(symbol_table);
// define function lerp(a,b,c) {a*(c-b)+b}
compositor.add("lerp", " a*(c-b)+b;", "a","b","c");
exprtk::expression<float> expressionR;
expressionR.register_symbol_table(symbol_table);
exprtk::parser<float> parserR;
doR = parserR.compile(E[3].content,expressionR);
exprtk::expression<float> expressionG;
expressionG.register_symbol_table(symbol_table);
exprtk::parser<float> parserG;
doG = parserG.compile(E[4].content,expressionG);
exprtk::expression<float> expressionB;
expressionB.register_symbol_table(symbol_table);
exprtk::parser<float> parserB;
doB = parserB.compile(E[5].content,expressionB);

for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
if (_effect.abort()) break;
float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
temPix[0] = srcPix[0];
temPix[1] = srcPix[1];
temPix[2] = srcPix[2];
param1 = (float)_param1;
param2 = (float)_param2;
param3 = (float)_param3;
param4_red = (float)_param4.r;
param4_green = (float)_param4.g;
param4_blue = (float)_param4.b;
param5 = (float)_param5;
x_coord = x;
y_coord = y;
width = p_ProcWindow.x2;
height = p_ProcWindow.y2;
for (int c = 0; c < 4; ++c) {
if (doR && c == 0) {
dstPix[0] = expressionR.value() * (1.0 - param5) + srcPix[0] * param5;
} else if (doG && c == 1) {
dstPix[1] = expressionG.value() * (1.0 - param5) + srcPix[1] * param5;
} else if (doB && c == 2) {
dstPix[2] = expressionB.value() * (1.0 - param5) + srcPix[2] * param5;
} else {
dstPix[c] = srcPix[c];
}}
dstPix += 4;
}}}

void ResolveMath::setSrcImg(OFX::Image* p_SrcImg) {
_srcImg = p_SrcImg;
}

void ResolveMath::setScales(const string& expr1, const string& expr2,
const string& expr3, const string& exprR, const string& exprG, const string& exprB,
double param1, double param2, double param3, const RGBValues& param4, double param5,
bool processR, bool processG, bool processB)
{
_expr1 = expr1;
_expr2 = expr2;
_expr3 = expr3;
_exprR = exprR;
_exprG = exprG;
_exprB = exprB;
_param1 = param1;
_param2 = param2;
_param3 = param3;
_param4 = param4;
_param5 = param5;
_processR = processR;
_processG = processG;
_processB = processB;
}

////////////////////////////////////////////////////////////////////////////////

class ResolveMathPlugin : public OFX::ImageEffect
{
public:
explicit ResolveMathPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
void setupAndProcess(ResolveMath &p_ResolveMath, const OFX::RenderArguments& p_Args);

private:

OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;
OFX::StringParam* _expr1;
OFX::StringParam* _expr2;
OFX::StringParam* _expr3;
OFX::StringParam* _exprR;
OFX::StringParam* _exprG;
OFX::StringParam* _exprB;
OFX::StringParam* _exprLUT;
OFX::DoubleParam* _param1;
OFX::DoubleParam* _param2;
OFX::DoubleParam* _param3;
OFX::RGBParam *_param4;
OFX::DoubleParam* _param5;
OFX::ChoiceParam* m_LUT;
OFX::Double2DParam* m_Input1;
OFX::Double2DParam* m_Input2;
OFX::IntParam* m_Lutsize;
OFX::IntParam* m_Cube;
OFX::IntParam* m_Precision;
OFX::StringParam* m_Path;
OFX::StringParam* m_Name;
OFX::StringParam* m_Path2;
OFX::StringParam* m_Name2;
OFX::PushButtonParam* m_Info;
OFX::PushButtonParam* m_Button1;
OFX::PushButtonParam* m_Button2;
OFX::PushButtonParam* m_Button3;
};

ResolveMathPlugin::ResolveMathPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
_expr1 = fetchStringParam(kParamExpr1Name);
_expr2 = fetchStringParam(kParamExpr2Name);
_expr3 = fetchStringParam(kParamExpr3Name);
_exprR = fetchStringParam(kParamResolveMathR);
_exprG = fetchStringParam(kParamResolveMathG);
_exprB = fetchStringParam(kParamResolveMathB);
_exprLUT = fetchStringParam(kParamResolveMathLUT);
assert(_expr1 && _expr2 && _expr3 && _exprR && _exprG && _exprB && _exprLUT);
_param1 = fetchDoubleParam(kParamParam1Name);
assert(_param1);
_param2 = fetchDoubleParam(kParamParam2Name);
assert(_param2);
_param3 = fetchDoubleParam(kParamParam3Name);
assert(_param3);
_param4 = fetchRGBParam(kParamParam4Name);
assert(_param4);
_param5 = fetchDoubleParam(kParamParam5Name);
assert(_param5);
m_Path = fetchStringParam("path");
m_Name = fetchStringParam("name");
m_Info = fetchPushButtonParam("info");
m_Button1 = fetchPushButtonParam("button1");
m_Button2 = fetchPushButtonParam("button2");
m_LUT = fetchChoiceParam(kParamLUT);
assert(m_LUT);
m_Input1 = fetchDouble2DParam("range1");
assert(m_Input1);
m_Input2 = fetchDouble2DParam("range2");
assert(m_Input2);
m_Path2 = fetchStringParam("path2");
m_Name2 = fetchStringParam("name2");
m_Cube = fetchIntParam("cube");
m_Precision = fetchIntParam("precision");
m_Lutsize = fetchIntParam("lutsize");
m_Button3 = fetchPushButtonParam("button3");    
}

void ResolveMathPlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA)) {
ResolveMath ResolveMath(*this);
setupAndProcess(ResolveMath, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool ResolveMathPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
string exprR, exprG, exprB;
_exprR->getValue(exprR);
_exprG->getValue(exprG);
_exprB->getValue(exprB);
if (exprR.empty() && exprG.empty() && exprB.empty()) {
p_IdentityClip = m_SrcClip;
p_IdentityTime = p_Args.time;
return true;
}
return false;
}

void ResolveMathPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if (p_ParamName == "info") {
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

if (p_ParamName == "button1") {
string expr1, expr2, expr3, exprR, exprG, exprB;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_exprR->getValue(exprR);
_exprG->getValue(exprG);
_exprB->getValue(exprB);
expr1 = expr1.empty() ? "0.000000f" : expr1;
expr2 = expr2.empty() ? "0.000000f" : expr2;
expr3 = expr3.empty() ? "0.000000f" : expr3;
exprR = exprR.empty() ? "r " : exprR;
exprG = exprG.empty() ? "g " : exprG;
exprB = exprB.empty() ? "b " : exprB;
float param1 = _param1->getValueAtTime(p_Args.time);
float param2 = _param2->getValueAtTime(p_Args.time);
float param3 = _param3->getValueAtTime(p_Args.time);
float param5 = _param5->getValueAtTime(p_Args.time);
RGBValues param4;
_param4->getValueAtTime(p_Args.time, param4.r, param4.g, param4.b);
expr1 = replace_pattern(expr1, "param4.r", "param4r");
expr2 = replace_pattern(expr2, "param4.r", "param4r");
expr3 = replace_pattern(expr3, "param4.r", "param4r");
exprR = replace_pattern(exprR, "param4.r", "param4r");
exprG = replace_pattern(exprG, "param4.r", "param4r");
exprB = replace_pattern(exprB, "param4.r", "param4r");
expr1 = replace_pattern(expr1, "param4.g", "param4g");
expr2 = replace_pattern(expr2, "param4.g", "param4g");
expr3 = replace_pattern(expr3, "param4.g", "param4g");
exprR = replace_pattern(exprR, "param4.g", "param4g");
exprG = replace_pattern(exprG, "param4.g", "param4g");
exprB = replace_pattern(exprB, "param4.g", "param4g");
expr1 = replace_pattern(expr1, "param4.b", "param4b");
expr2 = replace_pattern(expr2, "param4.b", "param4b");
expr3 = replace_pattern(expr3, "param4.b", "param4b");
exprR = replace_pattern(exprR, "param4.b", "param4b");
exprG = replace_pattern(exprG, "param4.b", "param4b");
exprB = replace_pattern(exprB, "param4.b", "param4b");
expr1 = replace_pattern(expr1, "pow", "_powf");
expr2 = replace_pattern(expr2, "pow", "_powf");
expr3 = replace_pattern(expr3, "pow", "_powf");
exprR = replace_pattern(exprR, "pow", "_powf");
exprG = replace_pattern(exprG, "pow", "_powf");
exprB = replace_pattern(exprB, "pow", "_powf");

string PATH;
m_Path->getValue(PATH);
string NAME;
m_Name->getValue(NAME);
Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
if (reply == Message::eMessageReplyYes) {
FILE * pFile;
pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "// ResolveMath DCTL export\n" \
"\n" \
"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
"{\n" \
"\n");
fprintf (pFile, "float r = p_R; \n" \
"float g = p_G; \n" \
"float b = p_B; \n" \
"float x = p_X; \n" \
"float y = p_Y; \n" \
"int width = p_Width; \n" \
"int height = p_Height; \n" \
"\n" \
"float mix = %ff; \n" \
"float param1 = %ff; \n" \
"float param2 = %ff; \n" \
"float param3 = %ff; \n" \
"float param4r = %ff; \n" \
"float param4g = %ff; \n" \
"float param4b = %ff;\n", param5, param1, param2, param3, param4.r, param4.g, param4.b);
fprintf (pFile, "\n" \
"float expr1 = %s; \n" \
"float expr2 = %s; \n" \
"float expr3 = %s; \n" \
"\n" \
"float R1 = %s; \n" \
"float G1 = %s; \n" \
"float B1 = %s; \n" \
"\n" \
"float R = R1 * (1.0f - mix) + r * mix; \n" \
"float G = G1 * (1.0f - mix) + g * mix; \n" \
"float B = B1 * (1.0f - mix) + b * mix; \n" \
"\n" \
"return make_float3(R, G, B); \n" \
"}\n", expr1.c_str(), expr2.c_str(), expr3.c_str(), exprR.c_str(), exprG.c_str(), exprB.c_str());
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to" + PATH  + ". Check Permissions."));
}}}

if (p_ParamName == "button2") {
string expr1, expr2, expr3, exprR,  exprG,  exprB;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_exprR->getValue(exprR);
_exprG->getValue(exprG);
_exprB->getValue(exprB);
expr1 = expr1.empty() ? "0" : expr1;
expr2 = expr2.empty() ? "0" : expr2;
expr3 = expr3.empty() ? "0" : expr3;
exprR = exprR.empty() ? "r " : exprR;
exprG = exprG.empty() ? "g " : exprG;
exprB = exprB.empty() ? "b " : exprB;
float param1 = _param1->getValueAtTime(p_Args.time);
float param2 = _param2->getValueAtTime(p_Args.time);
float param3 = _param3->getValueAtTime(p_Args.time);
float param5 = _param5->getValueAtTime(p_Args.time);
param5 = 1.0 - param5;
RGBValues param4;
_param4->getValueAtTime(p_Args.time, param4.r, param4.g, param4.b);
expr1 = replace_pattern(expr1, "param4.r", "Ba");
expr2 = replace_pattern(expr2, "param4.r", "Ba");
expr3 = replace_pattern(expr3, "param4.r", "Ba");
exprR = replace_pattern(exprR, "param4.r", "Ba");
exprG = replace_pattern(exprG, "param4.r", "Ba");
exprB = replace_pattern(exprB, "param4.r", "Ba");
expr1 = replace_pattern(expr1, "param4.g", "Aa");
expr2 = replace_pattern(expr2, "param4.g", "Aa");
expr3 = replace_pattern(expr3, "param4.g", "Aa");
exprR = replace_pattern(exprR, "param4.g", "Aa");
exprG = replace_pattern(exprG, "param4.g", "Aa");
exprB = replace_pattern(exprB, "param4.g", "Aa");
expr1 = replace_pattern(expr1, "r ", "Br ");
expr2 = replace_pattern(expr2, "r ", "Br ");
expr3 = replace_pattern(expr3, "r ", "Br ");
exprR = replace_pattern(exprR, "r ", "Br ");
exprG = replace_pattern(exprG, "r ", "Br ");
exprB = replace_pattern(exprB, "r ", "Br ");
expr1 = replace_pattern(expr1, " r", " Br");
expr2 = replace_pattern(expr2, " r", " Br");
expr3 = replace_pattern(expr3, " r", " Br");
exprR = replace_pattern(exprR, " r", " Br");
exprG = replace_pattern(exprG, " r", " Br");
exprB = replace_pattern(exprB, " r", " Br");
expr1 = replace_pattern(expr1, "g ", "Bg ");
expr2 = replace_pattern(expr2, "g ", "Bg ");
expr3 = replace_pattern(expr3, "g ", "Bg ");
exprR = replace_pattern(exprR, "g ", "Bg ");
exprG = replace_pattern(exprG, "g ", "Bg ");
exprB = replace_pattern(exprB, "g ", "Bg ");
expr1 = replace_pattern(expr1, " g", " Bg");
expr2 = replace_pattern(expr2, " g", " Bg");
expr3 = replace_pattern(expr3, " g", " Bg");
exprR = replace_pattern(exprR, " g", " Bg");
exprG = replace_pattern(exprG, " g", " Bg");
exprB = replace_pattern(exprB, " g", " Bg");
expr1 = replace_pattern(expr1, "b ", "Bb ");
expr2 = replace_pattern(expr2, "b ", "Bb ");
expr3 = replace_pattern(expr3, "b ", "Bb ");
exprR = replace_pattern(exprR, "b ", "Bb ");
exprG = replace_pattern(exprG, "b ", "Bb ");
exprB = replace_pattern(exprB, "b ", "Bb ");
expr1 = replace_pattern(expr1, " b", " Bb");
expr2 = replace_pattern(expr2, " b", " Bb");
expr3 = replace_pattern(expr3, " b", " Bb");
exprR = replace_pattern(exprR, " b", " Bb");
exprG = replace_pattern(exprG, " b", " Bb");
exprB = replace_pattern(exprB, " b", " Bb");
expr1 = replace_pattern(expr1, "param1", "Ar");
expr2 = replace_pattern(expr2, "param1", "Ar");
expr3 = replace_pattern(expr3, "param1", "Ar");
exprR = replace_pattern(exprR, "param1", "Ar");
exprG = replace_pattern(exprG, "param1", "Ar");
exprB = replace_pattern(exprB, "param1", "Ar");
expr1 = replace_pattern(expr1, "param2", "Ag");
expr2 = replace_pattern(expr2, "param2", "Ag");
expr3 = replace_pattern(expr3, "param2", "Ag");
exprR = replace_pattern(exprR, "param2", "Ag");
exprG = replace_pattern(exprG, "param2", "Ag");
exprB = replace_pattern(exprB, "param2", "Ag");
expr1 = replace_pattern(expr1, "param3", "Ab");
expr2 = replace_pattern(expr2, "param3", "Ab");
expr3 = replace_pattern(expr3, "param3", "Ab");
exprR = replace_pattern(exprR, "param3", "Ab");
exprG = replace_pattern(exprG, "param3", "Ab");
exprB = replace_pattern(exprB, "param3", "Ab");

string PATH;
m_Path->getValue(PATH);
string NAME;
m_Name->getValue(NAME);
Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".nk to " + PATH + "?");
if (reply == Message::eMessageReplyYes) {
FILE * pFile;
pFile = fopen ((PATH + "/" + NAME + ".nk").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "Group {\n" \
" inputs 0\n" \
" name ResolveMath\n" \
" selected true\n" \
" xpos -119\n" \
" ypos -76\n" \
"}\n" \
" Expression {\n" \
"  inputs 0\n" \
"  temp_name0 param1\n" \
"  temp_expr0 %f\n" \
"  temp_name1 param2\n" \
"  temp_expr1 %f\n" \
"  temp_name2 param3\n" \
"  temp_expr2 %f\n" \
"  temp_name3 param4.g\n" \
"  temp_expr3 %f\n" \
"  expr0 param1\n" \
"  expr1 param2\n" \
"  expr2 param3\n" \
"  channel3 {-rgba.red -rgba.green -rgba.blue rgba.alpha}\n" \
"  expr3 param4.g\n" \
"  name param1_param2_param3_param4g\n" \
"  xpos -275\n" \
"  ypos -46\n" \
"  hide_input true\n" \
" }\n" \
" Input {\n" \
"  inputs 0\n" \
"  name Input1\n" \
"  xpos -200\n" \
"  ypos -123\n" \
" }\n" \
" Expression {\n" \
"  temp_name0 param4.r\n" \
"  temp_expr0 %f\n" \
"  channel3 {-rgba.red -rgba.green -rgba.blue rgba.alpha}\n" \
"  expr3 param4.r\n" \
"  name RGB_param4r\n" \
"  selected true\n" \
"  xpos -130\n" \
"  ypos -47\n" \
" }\n", param1, param2, param3, param4.g, param4.r);
fprintf (pFile, " MergeExpression {\n" \
"  inputs 2\n" \
"  temp_name0 expr1\n" \
"  temp_expr0 %s\n" \
"  temp_name1 expr2\n" \
"  temp_expr1 %s\n" \
"  temp_name2 expr3\n" \
"  temp_expr2 %s\n" \
"  temp_name3 param4.b\n" \
"  temp_expr3 %f\n" \
"  expr0 %s\n" \
"  expr1 %s\n" \
"  expr2 %s\n" \
"  channel3 none\n" \
"  mix %f\n" \
"  name RGB_expressions_combined\n" \
"  xpos -196\n" \
"  ypos 32\n" \
" }\n" \
" Output {\n" \
"  name Output1\n" \
"  xpos -196\n" \
"  ypos 132\n" \
" }\n" \
"end_group\n", expr1.c_str(), expr2.c_str(), expr3.c_str(), param4.b, exprR.c_str(), exprG.c_str(), exprB.c_str(), param5);
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to" + PATH  + ". Check Permissions."));
}}}

if (p_ParamName == kParamLUT) {
int LUT_i;
m_LUT->getValueAtTime(p_Args.time, LUT_i);
LUTEnum LUT = (LUTEnum)LUT_i;
int Lut = LUT_i;
if (Lut == 0) {
m_Input1->setIsSecretAndDisabled(true);
m_Lutsize->setIsSecretAndDisabled(true);
_exprLUT->setIsSecretAndDisabled(true);
m_Input2->setIsSecretAndDisabled(false);
m_Cube->setIsSecretAndDisabled(false);
} else if (Lut == 1) {
m_Input1->setIsSecretAndDisabled(false);
m_Lutsize->setIsSecretAndDisabled(false);
_exprLUT->setIsSecretAndDisabled(false);
m_Input2->setIsSecretAndDisabled(true);
m_Cube->setIsSecretAndDisabled(true);
} else {
m_Input1->setIsSecretAndDisabled(false);
m_Lutsize->setIsSecretAndDisabled(false);
_exprLUT->setIsSecretAndDisabled(false);
m_Input2->setIsSecretAndDisabled(false);
m_Cube->setIsSecretAndDisabled(false);
}}

if (p_ParamName == "range1" || p_ParamName == kParamResolveMathLUT || p_ParamName == kParamLUT) {
string expr1;
string expr2;
string expr3;
string exprR;
string exprG;
string exprB;
string exprLUT;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_exprR->getValue(exprR);
_exprG->getValue(exprG);
_exprB->getValue(exprB);
_exprLUT->getValue(exprLUT);
double Param1;
_param1->getValueAtTime(p_Args.time,Param1);
double Param2;
_param2->getValueAtTime(p_Args.time,Param2);
double Param3;
_param3->getValueAtTime(p_Args.time,Param3);
RGBValues Param4;
_param4->getValueAtTime(p_Args.time, Param4.r, Param4.g, Param4.b);
double Param5;
_param5->getValueAtTime(p_Args.time,Param5);
OfxPointD _position;
_position.x = 1;
_position.y = 1;
bool processR = !exprR.empty();
bool processG = !exprG.empty();
bool processB = !exprB.empty();
bool processLUT = !exprLUT.empty();
expr1 = expr1.empty() ? "0" : expr1;
expr2 = expr2.empty() ? "0" : expr2;
expr3 = expr3.empty() ? "0" : expr3;
exprR = exprR.empty() ? "r " : exprR;
exprG = exprG.empty() ? "g " : exprG;
exprB = exprB.empty() ? "b " : exprB;
exprLUT = exprLUT.empty() ? "input" : exprLUT;
double temPix[4];
bool doR = processR;
bool doG = processG;
bool doB = processB;
bool doLUT = processLUT;
double param1;
double param2;
double param3;
double param4_red;
double param4_green;
double param4_blue;
double param5;
double x_coord;
double y_coord;
double width;
double height;
exprtk::symbol_table<double> symbol_table;
symbol_table.add_constants();
symbol_table.add_variable("r", temPix[0]);
symbol_table.add_variable("g", temPix[1]);
symbol_table.add_variable("b", temPix[2]);
symbol_table.add_variable("input", temPix[3]);
symbol_table.add_variable("param1", param1);
symbol_table.add_variable("param2", param2);
symbol_table.add_variable("param3", param3);
symbol_table.add_variable("param4_r", param4_red);
symbol_table.add_variable("param4_g", param4_green);
symbol_table.add_variable("param4_b", param4_blue);
symbol_table.add_variable("x", x_coord);
symbol_table.add_variable("y", y_coord);
symbol_table.add_variable("width", width);
symbol_table.add_variable("height", height);
ResolveMathProperties expr1_props = {kParamExpr1Name, expr1, true};
ResolveMathProperties expr2_props = {kParamExpr2Name, expr2, true};
ResolveMathProperties expr3_props = {kParamExpr3Name, expr3, true};
ResolveMathProperties exprR_props = {kParamResolveMathR, exprR, true};
ResolveMathProperties exprG_props = {kParamResolveMathG, exprG, true};
ResolveMathProperties exprB_props = {kParamResolveMathB, exprB, true};
ResolveMathProperties exprLUT_props = {kParamResolveMathLUT, exprLUT, true};
const int Esize = 7;
ResolveMathProperties E[Esize] = {expr1_props, expr2_props, expr3_props, 
exprR_props, exprG_props, exprB_props, exprLUT_props};
for (int i = 0; i != Esize; ++i) {
for (int k = 0; k != Esize; ++ k) {
if (E[i].content.find(E[i].name) != string::npos){
E[i].content.clear();
E[i].processFlag = false;
} else if (i != k && !E[i].content.empty() && !E[k].content.empty() ) { 
E[i].content  = replace_pattern(E[i].content,E[k].name,"("+E[k].content+")");
}}
E[i].content = replace_pattern(E[i].content,"param1.","param1_");
E[i].content = replace_pattern(E[i].content,"param2.","param2_");
E[i].content = replace_pattern(E[i].content,"param3.","param3_");
E[i].content = replace_pattern(E[i].content,"param4.","param4_");
}
exprtk::function_compositor<double> compositor(symbol_table);
compositor.add("lerp", " a*(c-b)+b;", "a","b","c");
exprtk::expression<double> expressionR;
expressionR.register_symbol_table(symbol_table);
exprtk::parser<double> parserR;
doR = parserR.compile(E[3].content,expressionR);
exprtk::expression<double> expressionG;
expressionG.register_symbol_table(symbol_table);
exprtk::parser<double> parserG;
doG = parserG.compile(E[4].content,expressionG);
exprtk::expression<double> expressionB;
expressionB.register_symbol_table(symbol_table);
exprtk::parser<double> parserB;
doB = parserB.compile(E[5].content,expressionB);
exprtk::expression<double> expressionLUT;
expressionLUT.register_symbol_table(symbol_table);
exprtk::parser<double> parserLUT;
doLUT = parserLUT.compile(E[6].content,expressionLUT);
param1 = Param1;
param2 = Param2;
param3 = Param3;
param4_red = Param4.r;
param4_green = Param4.g;
param4_blue = Param4.b;
param5 = Param5;
x_coord = _position.x;
y_coord = _position.y;
std::auto_ptr<OFX::Image> src( m_SrcClip->fetchImage(p_Args.time) );
const OfxRectI& bounds = src->getBounds();
width = bounds.x2 - bounds.x1;
height = bounds.y2 - bounds.y1;
int LUT_i;
m_LUT->getValueAtTime(p_Args.time, LUT_i);
LUTEnum LUT = (LUTEnum)LUT_i;
int Lut = LUT_i;
double shaperA = 0.0;
double shaperB = 0.0;
double inputA = 0.0;
double inputB = 0.0;
m_Input1->getValueAtTime(p_Args.time, shaperA, shaperB);
temPix[3] = shaperA;
inputA = doLUT ? expressionLUT.value() : shaperA;
temPix[3] = shaperB;
inputB = doLUT ? expressionLUT.value() : shaperB;
if (Lut == 2)
m_Input2->setValue(inputA, inputB);
}

if (p_ParamName == "button3") {
string expr1;
string expr2;
string expr3;
string exprR;
string exprG;
string exprB;
string exprLUT;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_exprR->getValue(exprR);
_exprG->getValue(exprG);
_exprB->getValue(exprB);
_exprLUT->getValue(exprLUT);
double Param1;
_param1->getValueAtTime(p_Args.time,Param1);
double Param2;
_param2->getValueAtTime(p_Args.time,Param2);
double Param3;
_param3->getValueAtTime(p_Args.time,Param3);
RGBValues Param4;
_param4->getValueAtTime(p_Args.time, Param4.r, Param4.g, Param4.b);
double Param5;
_param5->getValueAtTime(p_Args.time,Param5);
OfxPointD _position;
_position.x = 1;
_position.y = 1;
bool processR = !exprR.empty();
bool processG = !exprG.empty();
bool processB = !exprB.empty();
bool processLUT = !exprLUT.empty();
expr1 = expr1.empty() ? "0" : expr1;
expr2 = expr2.empty() ? "0" : expr2;
expr3 = expr3.empty() ? "0" : expr3;
exprR = exprR.empty() ? "r " : exprR;
exprG = exprG.empty() ? "g " : exprG;
exprB = exprB.empty() ? "b " : exprB;
exprLUT = exprLUT.empty() ? "input" : exprLUT;
double temPix[4];
bool doR = processR;
bool doG = processG;
bool doB = processB;
bool doLUT = processLUT;
double param1;
double param2;
double param3;
double param4_red;
double param4_green;
double param4_blue;
double param5;
double x_coord;
double y_coord;
double width;
double height;
exprtk::symbol_table<double> symbol_table;
symbol_table.add_constants();
symbol_table.add_variable("r",temPix[0]);
symbol_table.add_variable("g",temPix[1]);
symbol_table.add_variable("b",temPix[2]);
symbol_table.add_variable("input",temPix[3]);
symbol_table.add_variable("param1", param1);
symbol_table.add_variable("param2", param2);
symbol_table.add_variable("param3", param3);
symbol_table.add_variable("param4_r",param4_red);
symbol_table.add_variable("param4_g",param4_green);
symbol_table.add_variable("param4_b",param4_blue);
symbol_table.add_variable("x",x_coord);
symbol_table.add_variable("y",y_coord);
symbol_table.add_variable("width",width);
symbol_table.add_variable("height",height);
ResolveMathProperties expr1_props = {kParamExpr1Name, expr1, true};
ResolveMathProperties expr2_props = {kParamExpr2Name, expr2, true};
ResolveMathProperties expr3_props = {kParamExpr3Name, expr3, true};
ResolveMathProperties exprR_props = {kParamResolveMathR, exprR, true};
ResolveMathProperties exprG_props = {kParamResolveMathG, exprG, true};
ResolveMathProperties exprB_props = {kParamResolveMathB, exprB, true};
ResolveMathProperties exprLUT_props = {kParamResolveMathLUT, exprLUT, true};
const int Esize = 7;
ResolveMathProperties E[Esize] = {expr1_props, expr2_props, expr3_props, 
exprR_props, exprG_props, exprB_props, exprLUT_props};
for (int i = 0; i != Esize; ++i) {
for (int k = 0; k != Esize; ++ k) {
if (E[i].content.find(E[i].name) != string::npos){
E[i].content.clear();
E[i].processFlag = false;
} else if ((i != k) && !E[i].content.empty() && !E[k].content.empty() ) { 
E[i].content  = replace_pattern(E[i].content,E[k].name,"("+E[k].content+")");
}}
E[i].content = replace_pattern(E[i].content,"param1.","param1_");
E[i].content = replace_pattern(E[i].content,"param2.","param2_");
E[i].content = replace_pattern(E[i].content,"param3.","param3_");
E[i].content = replace_pattern(E[i].content,"param4.","param4_");
}

exprtk::function_compositor<double> compositor(symbol_table);
compositor.add("lerp", " a*(c-b)+b;", "a","b","c");
exprtk::expression<double> expressionR;
expressionR.register_symbol_table(symbol_table);
exprtk::parser<double> parserR;
doR = parserR.compile(E[3].content,expressionR);
exprtk::expression<double> expressionG;
expressionG.register_symbol_table(symbol_table);
exprtk::parser<double> parserG;
doG = parserG.compile(E[4].content,expressionG);
exprtk::expression<double> expressionB;
expressionB.register_symbol_table(symbol_table);
exprtk::parser<double> parserB;
doB = parserB.compile(E[5].content,expressionB);
exprtk::expression<double> expressionLUT;
expressionLUT.register_symbol_table(symbol_table);
exprtk::parser<double> parserLUT;
doLUT = parserLUT.compile(E[6].content,expressionLUT);
param1 = Param1;
param2 = Param2;
param3 = Param3;
param4_red = Param4.r;
param4_green = Param4.g;
param4_blue = Param4.b;
param5 = Param5;
x_coord = _position.x;
y_coord = _position.y;
std::auto_ptr<OFX::Image> src( m_SrcClip->fetchImage(p_Args.time) );
const OfxRectI& bounds = src->getBounds();
width = bounds.x2 - bounds.x1;
height = bounds.y2 - bounds.y1;

string PATH;
m_Path2->getValue(PATH);
string NAME;
m_Name2->getValue(NAME);

int LUT_i;
m_LUT->getValueAtTime(p_Args.time, LUT_i);
LUTEnum LUT = (LUTEnum)LUT_i;
int Lut = LUT_i;
double shaper = 0.0;
double shaper1 = 0.0;
double shaperA = 0.0;
double shaperB = 0.0;
m_Input1->getValueAtTime(p_Args.time, shaperA, shaperB);
double inputA = 0.0;
double inputB = 0.0;
m_Input2->getValueAtTime(p_Args.time, inputA, inputB);
int cube = (int)m_Cube->getValueAtTime(p_Args.time);
int decimal = (int)m_Precision->getValueAtTime(p_Args.time);
int total = cube * cube * cube;
int lutsize = pow(2, m_Lutsize->getValueAtTime(p_Args.time));

OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".cube to " + PATH + " ?");
if (reply == OFX::Message::eMessageReplyYes) {
FILE * pFile;
pFile = fopen ((PATH + NAME + ".cube").c_str(), "w");
if (pFile != NULL) {
fprintf (pFile, "# Resolve LUT export\n" \
"\n");
if (Lut > 0)
fprintf (pFile, "LUT_1D_SIZE %d\n" \
"LUT_1D_INPUT_RANGE %.*f %.*f\n" \
"\n", lutsize, decimal, shaperA, decimal, shaperB);
if (Lut != 1) {
fprintf (pFile, "LUT_3D_SIZE %d\n" \
"LUT_3D_INPUT_RANGE %.*f %.*f\n" \
"\n", cube, decimal, inputA, decimal, inputB);
}
if (Lut > 0) {
for( int i = 0; i < lutsize; ++i ){
shaper = ((double)i / (lutsize - 1)) * (shaperB - shaperA) + shaperA;
temPix[3] = shaper;
shaper1 = doLUT ? expressionLUT.value() : shaper;
fprintf (pFile, "%.*f %.*f %.*f\n", decimal, shaper1, decimal, shaper1, decimal, shaper1);
}
fprintf (pFile, "\n");
}
if (Lut != 1) {
for( int i = 0; i < total; ++i ){
double R = fmod(i, cube) / (cube - 1) * (inputB - inputA) + inputA;
double G = fmod(floor(i / cube), cube) / (cube - 1) * (inputB - inputA) + inputA;
double B = fmod(floor(i / (cube * cube)), cube) / (cube - 1) * (inputB - inputA) + inputA;
temPix[0] = R;
temPix[1] = G;
temPix[2] = B;
x_coord = fmod(i, cube);
y_coord = fmod(floor(i / cube), cube); 
R = doR ? expressionR.value() * (1.0 - param5) + R * param5 : R;
G = doG ? expressionG.value() * (1.0 - param5) + G * param5 : G;
B = doB ? expressionB.value() * (1.0 - param5) + B * param5 : B;
fprintf (pFile, "%.*f %.*f %.*f\n", decimal, R, decimal, G, decimal, B);
}}
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".cube to " + PATH + ". Check Permissions."));
}}}}

void ResolveMathPlugin::setupAndProcess(ResolveMath& p_ResolveMath, const OFX::RenderArguments& p_Args)
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
string expr1;
string expr2;
string expr3;
string exprR;
string exprG;
string exprB;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_exprR->getValue(exprR);
_exprG->getValue(exprG);
_exprB->getValue(exprB);
double param1;
_param1->getValueAtTime(p_Args.time,param1);
double param2;
_param2->getValueAtTime(p_Args.time,param2);
double param3;
_param3->getValueAtTime(p_Args.time,param3);
RGBValues param4;
_param4->getValueAtTime(p_Args.time, param4.r, param4.g, param4.b);
double param5;
_param5->getValueAtTime(p_Args.time,param5);
bool processR = !exprR.empty();
bool processG = !exprG.empty();
bool processB = !exprB.empty();

p_ResolveMath.setDstImg(dst.get());
p_ResolveMath.setSrcImg(src.get());

p_ResolveMath.setRenderWindow(p_Args.renderWindow);

p_ResolveMath.setScales(expr1, expr2, expr3, exprR, exprG, exprB,
param1, param2, param3, param4, param5, processR, processG, processB);

p_ResolveMath.process();   
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ResolveMathPluginFactory::ResolveMathPluginFactory()
: PluginFactoryHelper<ResolveMathPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ResolveMathPluginFactory::describe(ImageEffectDescriptor& p_Desc)
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
}

void ResolveMathPluginFactory::describeInContext(ImageEffectDescriptor& p_Desc, ContextEnum)
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

PageParamDescriptor *page = p_Desc.definePageParam("Controls");

StringParamDescriptor* stringparam = p_Desc.defineStringParam(kParamExpr1Name);
stringparam->setLabel(kParamExpr1Label);
stringparam->setHint(kParamExpr1Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamExpr2Name);
stringparam->setLabel(kParamExpr2Label);
stringparam->setHint(kParamExpr2Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamExpr3Name);
stringparam->setLabel(kParamExpr3Label);
stringparam->setHint(kParamExpr3Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamResolveMathR);
stringparam->setLabel(kParamResolveMathRLabel);
stringparam->setHint(kParamResolveMathRHint);
stringparam->setDefault("r");
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamResolveMathG);
stringparam->setLabel(kParamResolveMathGLabel);
stringparam->setHint(kParamResolveMathGHint);
stringparam->setDefault("g");
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamResolveMathB);
stringparam->setLabel(kParamResolveMathBLabel);
stringparam->setHint(kParamResolveMathBHint);
stringparam->setDefault("b");
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam1Name);
param->setLabel(kParamParam1Label);
param->setHint(kParamParam1Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

param = p_Desc.defineDoubleParam(kParamParam2Name);
param->setLabel(kParamParam2Label);
param->setHint(kParamParam2Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

param = p_Desc.defineDoubleParam(kParamParam3Name);
param->setLabel(kParamParam3Label);
param->setHint(kParamParam3Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

RGBParamDescriptor *RGBparam = p_Desc.defineRGBParam(kParamParam4Name);
RGBparam->setLabel(kParamParam4Label);
RGBparam->setHint(kParamParam4Hint);
RGBparam->setDefault(1.0, 1.0, 1.0);
RGBparam->setRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
RGBparam->setAnimates(true);
if (page)
page->addChild(*RGBparam);

param = p_Desc.defineDoubleParam(kParamParam5Name);
param->setLabel(kParamParam5Label);
param->setHint(kParamParam5Hint);
param->setDefault(0.0);
param->setIncrement(0.001);
param->setRange(0.0, 1.0);
param->setDisplayRange(0.0, 1.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

PushButtonParamDescriptor* pushparam = p_Desc.definePushButtonParam("info");
pushparam->setLabel("Info");
pushparam->setHint("useful info");
page->addChild(*pushparam);

GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
script->setOpen(false);
script->setHint("export DCTL and Nuke script");
if (page)
page->addChild(*script);

pushparam = p_Desc.definePushButtonParam("button1");
pushparam->setLabel("Export DCTL");
pushparam->setHint("create DCTL version");
pushparam->setParent(*script);
page->addChild(*pushparam);

pushparam = p_Desc.definePushButtonParam("button2");
pushparam->setLabel("Export Nuke script");
pushparam->setHint("create NUKE version");
pushparam->setParent(*script);
page->addChild(*pushparam);

stringparam = p_Desc.defineStringParam("name");
stringparam->setLabel("Name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("ResolveMath");
stringparam->setParent(*script);
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam("path");
stringparam->setLabel("Directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript);
stringparam->setFilePathExists(true);
stringparam->setParent(*script);
page->addChild(*stringparam);
    
GroupParamDescriptor* lutexport = p_Desc.defineGroupParam("LUT Export");
lutexport->setOpen(false);
lutexport->setHint("export LUT");
if (page)
page->addChild(*lutexport);

stringparam = p_Desc.defineStringParam("name2");
stringparam->setLabel("Name");
stringparam->setHint("overwrites if the same");
stringparam->setDefault("ResolveMath");
stringparam->setParent(*lutexport);
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam("path2");
stringparam->setLabel("Directory");
stringparam->setHint("make sure it's the absolute path");
stringparam->setStringType(eStringTypeFilePath);
stringparam->setDefault(kPluginScript);
stringparam->setFilePathExists(true);
stringparam->setParent(*lutexport);
page->addChild(*stringparam);

ChoiceParamDescriptor *choiceparam = p_Desc.defineChoiceParam(kParamLUT);
choiceparam->setLabel(kParamLUTLabel);
choiceparam->setHint(kParamLUTHint);
assert(choiceparam->getNOptions() == (int)eLUT3D);
choiceparam->appendOption(kParamLUTOption3D, kParamLUTOption3DHint);
assert(choiceparam->getNOptions() == (int)eLUT1D);
choiceparam->appendOption(kParamLUTOption1D, kParamLUTOption1DHint);
assert(choiceparam->getNOptions() == (int)eLUTCombo);
choiceparam->appendOption(kParamLUTOptionCombo, kParamLUTOptionComboHint);
choiceparam->setDefault( (int)eLUT3D );
choiceparam->setAnimates(false);
choiceparam->setParent(*lutexport);
page->addChild(*choiceparam);

stringparam = p_Desc.defineStringParam(kParamResolveMathLUT);
stringparam->setLabel(kParamResolveMathLUTLabel);
stringparam->setHint(kParamResolveMathLUTHint);
stringparam->setDefault("input");
stringparam->setAnimates(false);
stringparam->setIsSecretAndDisabled(true);
stringparam->setParent(*lutexport);
if (page)
page->addChild(*stringparam);

Double2DParamDescriptor* param2D = p_Desc.defineDouble2DParam("range1");
param2D->setLabel("1D Input Range");
param2D->setHint("set input range for LUT");
param2D->setDefault(0.0, 1.0);
param2D->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX);
param2D->setIsSecretAndDisabled(true);
param2D->setParent(*lutexport);
page->addChild(*param2D);

IntParamDescriptor* intparam = p_Desc.defineIntParam("lutsize");
intparam->setLabel("1D size");
intparam->setHint("1D lut size in bytes");
intparam->setDefault(12);
intparam->setRange(1, 14);
intparam->setDisplayRange(8, 14);
intparam->setIsSecretAndDisabled(true);
intparam->setParent(*lutexport);
page->addChild(*intparam);

param2D = p_Desc.defineDouble2DParam("range2");
param2D->setLabel("3D Input Range");
param2D->setHint("set input range for 3D LUT");
param2D->setDefault(0.0, 1.0);
param2D->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX);
param2D->setDisplayRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX);
param2D->setParent(*lutexport);
page->addChild(*param2D);

intparam = p_Desc.defineIntParam("cube");
intparam->setLabel("cube size");
intparam->setHint("3d lut cube size");
intparam->setDefault(33);
intparam->setRange(2, 129);
intparam->setDisplayRange(3, 129);
intparam->setParent(*lutexport);
page->addChild(*intparam);

intparam = p_Desc.defineIntParam("precision");
intparam->setLabel("precision");
intparam->setHint("number of decimal points");
intparam->setDefault(6);
intparam->setRange(1, 12);
intparam->setDisplayRange(3, 10);
intparam->setParent(*lutexport);
page->addChild(*intparam);

pushparam = p_Desc.definePushButtonParam("button3");
pushparam->setLabel("Export LUT");
pushparam->setHint("create 3D LUT");
pushparam->setParent(*lutexport);
page->addChild(*pushparam);
}

ImageEffect* ResolveMathPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum)
{
return new ResolveMathPlugin(p_Handle);
}

void Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static ResolveMathPluginFactory ResolveMathPlugin;
p_FactoryArray.push_back(&ResolveMathPlugin);
}