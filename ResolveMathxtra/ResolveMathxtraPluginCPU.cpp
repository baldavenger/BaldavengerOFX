#include "ResolveMathxtraPlugin.h"
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

#define kPluginName "ResolveMathxtra"
#define kPluginGrouping "BaldavengerOFX"
#define kPluginDescription \
"Useful information : The following references can be applied to the expressions \n" \
"Red, Green, Blue channels: r, g, b // Coordinates: x, y // Operators: +, -, *, /, ^, =\n" \
"Functions: min, max, avg, sum, abs, fmod, ceil, floor, round, pow, exp, log, root\n" \
"sqrt, lerp, sin, cos, tan, asin, acos, atan, hypot // Conditionals: ==, !=, >=, && \n" \
"if(a == b, c, d) : If a equals b then c, else d // a == b ? c : d  If a equals b then c, else d\n" \
"clamp(a,b,c) : a clamped to between b and c // pi : 3.1415926536 // width, height"

#define kPluginIdentifier "BaldavengerOFX.ResolveMathxtra"
#define kPluginVersionMajor 1 
#define kPluginVersionMinor 4

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsRenderScale 1
#define kSupportsMultipleClipPARs false
#define kSupportsMultipleClipDepths true
#define kRenderThreadSafety eRenderFullySafe

static const string  kParamExpr1Name = "expr1";
static const string  kParamExpr1Label= "expr1";
static const string  kParamExpr1Hint = "You can define an expression here and reference it in ResolveMathxtra fields as 'expr1'";
static const string  kParamExpr2Name = "expr2";
static const string  kParamExpr2Label= "expr2";
static const string  kParamExpr2Hint = "Reference in ResolveMathxtra fields as'expr2'";
static const string  kParamExpr3Name = "expr3";
static const string  kParamExpr3Label= "expr3";
static const string  kParamExpr3Hint = "Reference in ResolveMathxtra fields as'expr3'";
static const string  kParamExpr4Name = "expr4";
static const string  kParamExpr4Label= "expr4";
static const string  kParamExpr4Hint = "Reference in ResolveMathxtra fields as'expr4'";
static const string  kParamExpr5Name = "expr5";
static const string  kParamExpr5Label= "expr5";
static const string  kParamExpr5Hint = "Reference in ResolveMathxtra fields as'expr5'";
static const string  kParamExpr6Name = "expr6";
static const string  kParamExpr6Label= "expr6";
static const string  kParamExpr6Hint = "Reference in ResolveMathxtra fields as'expr6'";
static const string  kParamExpr7Name = "expr7";
static const string  kParamExpr7Label= "expr7";
static const string  kParamExpr7Hint = "Reference in ResolveMathxtra fields as'expr7'";
static const string  kParamExpr8Name = "expr8";
static const string  kParamExpr8Label= "expr8";
static const string  kParamExpr8Hint = "Reference in ResolveMathxtra fields as'expr8'";
static const string  kParamExpr9Name = "expr9";
static const string  kParamExpr9Label= "expr9";
static const string  kParamExpr9Hint = "Reference in ResolveMathxtra fields as'expr9'";

static const string  kParamResolveMathxtraR =     "red";
static const string  kParamResolveMathxtraRLabel= "Red Output";
static const string  kParamResolveMathxtraRHint = "Red Channel output";
static const string  kParamResolveMathxtraG     = "green";
static const string  kParamResolveMathxtraGLabel= "Green Output";
static const string  kParamResolveMathxtraGHint = "Green Channel output";
static const string  kParamResolveMathxtraB     = "blue";
static const string  kParamResolveMathxtraBLabel ="Blue Output";
static const string  kParamResolveMathxtraBHint = "Blue Channel output";
static const string  kParamResolveMathxtraLUT   = "lut";
static const string  kParamResolveMathxtraLUTLabel ="1D expr";
static const string  kParamResolveMathxtraLUTHint = "expression for 1D LUT";

static const string  kParamParam1Name = "param1";
static const string  kParamParam1Label= "param1";
static const string  kParamParam1Hint = "Reference in ResolveMathxtra fields as 'param1'";

static const string  kParamParam2Name = "param2";
static const string  kParamParam2Label= "param2";
static const string  kParamParam2Hint = "Reference in ResolveMathxtra fields as 'param2'";

static const string  kParamParam3Name = "param3";
static const string  kParamParam3Label= "param3";
static const string  kParamParam3Hint = "Reference in ResolveMathxtra fields as 'param3'";

static const string  kParamParam4Name = "param4";
static const string  kParamParam4Label= "param4";
static const string  kParamParam4Hint = "Reference in ResolveMathxtra fields as 'param4'";

static const string  kParamParam5Name = "param5";
static const string  kParamParam5Label= "param5";
static const string  kParamParam5Hint = "Reference in ResolveMathxtra fields as 'param5'";

static const string  kParamParam6Name = "param6";
static const string  kParamParam6Label= "param6";
static const string  kParamParam6Hint = "Reference in ResolveMathxtra fields as 'param6'";

static const string  kParamParam7Name = "param7";
static const string  kParamParam7Label= "param7";
static const string  kParamParam7Hint = "Reference in ResolveMathxtra fields as 'param7'";

static const string  kParamParam8Name = "param8";
static const string  kParamParam8Label= "param8";
static const string  kParamParam8Hint = "Reference in ResolveMathxtra fields as 'param8'";

static const string  kParamParam9Name = "param9";
static const string  kParamParam9Label= "param9";
static const string  kParamParam9Hint = "Reference in ResolveMathxtra fields as 'param9'";

static const string  kParamParam10Name = "param10";
static const string  kParamParam10Label= "param10";
static const string  kParamParam10Hint = "Reference in ResolveMathxtra fields as 'param10.r||g||b'";

static const string  kParamParam11Name = "param11";
static const string  kParamParam11Label = "mix";
static const string  kParamParam11Hint = "Mix factor between original and transformed image";

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

struct ResolveMathxtraProperties {
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

class ResolveMathxtra : public OFX::ImageProcessor
{
public:
explicit ResolveMathxtra(OFX::ImageEffect& p_Instance);
virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);
void setSrcImg(OFX::Image* p_SrcImg);

void setScales(const string& expr1, const string& expr2,
const string& expr3, const string& expr4, const string& expr5, const string& expr6,
const string& expr7, const string& expr8, const string& expr9, const string& exprR,
const string& exprG, const string& exprB, double param1, double param2, double param3,
double param4, double param5, double param6, double param7, double param8, double param9, 
const RGBValues& param10, double param11, bool processR, bool processG, bool processB);

private:
OFX::Image *_srcImg;
string _expr1;
string _expr2;
string _expr3;
string _expr4;
string _expr5;
string _expr6;
string _expr7;
string _expr8;
string _expr9;
string _exprR;
string _exprG;
string _exprB;
double _param1;
double _param2;
double _param3;
double _param4;
double _param5;
double _param6;
double _param7;
double _param8;
double _param9;
RGBValues _param10;
double _param11;
bool _processR;
bool _processG;
bool _processB;
};

ResolveMathxtra::ResolveMathxtra(OFX::ImageEffect& p_Instance)
: OFX::ImageProcessor(p_Instance)
{
}

void ResolveMathxtra::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
float temPix[3];
bool doR = _processR;
bool doG = _processG;
bool doB = _processB;
float param1;
float param2;
float param3;
float param4;
float param5;
float param6;
float param7;
float param8;
float param9;
float param10_red;
float param10_green;
float param10_blue;
float param11;
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
symbol_table.add_variable("param4", param4);
symbol_table.add_variable("param5", param5);
symbol_table.add_variable("param6", param6);
symbol_table.add_variable("param7", param7);
symbol_table.add_variable("param8", param8);
symbol_table.add_variable("param9", param9);
symbol_table.add_variable("param10_r",param10_red);
symbol_table.add_variable("param10_g",param10_green);
symbol_table.add_variable("param10_b",param10_blue);
symbol_table.add_variable("x",x_coord);
symbol_table.add_variable("y",y_coord);
symbol_table.add_variable("width",width);
symbol_table.add_variable("height",height);
ResolveMathxtraProperties expr1_props = {kParamExpr1Name, _expr1, true};
ResolveMathxtraProperties expr2_props = {kParamExpr2Name, _expr2, true};
ResolveMathxtraProperties expr3_props = {kParamExpr3Name, _expr3, true};
ResolveMathxtraProperties expr4_props = {kParamExpr4Name, _expr4, true};
ResolveMathxtraProperties expr5_props = {kParamExpr5Name, _expr5, true};
ResolveMathxtraProperties expr6_props = {kParamExpr6Name, _expr6, true};
ResolveMathxtraProperties expr7_props = {kParamExpr7Name, _expr7, true};
ResolveMathxtraProperties expr8_props = {kParamExpr8Name, _expr8, true};
ResolveMathxtraProperties expr9_props = {kParamExpr9Name, _expr9, true};
ResolveMathxtraProperties exprR_props = {kParamResolveMathxtraR, _exprR, true};
ResolveMathxtraProperties exprG_props = {kParamResolveMathxtraG, _exprG, true};
ResolveMathxtraProperties exprB_props = {kParamResolveMathxtraB, _exprB, true};
const int Esize = 12;
ResolveMathxtraProperties E[Esize] = {expr1_props, expr2_props, expr3_props, expr4_props, expr5_props, expr6_props, 
expr7_props, expr8_props, expr9_props, exprR_props, exprG_props, exprB_props};
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
E[i].content = replace_pattern(E[i].content,"param5.","param5_");
E[i].content = replace_pattern(E[i].content,"param6.","param6_");
E[i].content = replace_pattern(E[i].content,"param7.","param7_");
E[i].content = replace_pattern(E[i].content,"param8.","param8_");
E[i].content = replace_pattern(E[i].content,"param9.","param9_");
E[i].content = replace_pattern(E[i].content,"param10.","param10_");
//E[i].content = replace_pattern(E[i].content,"=",":=");
//E[i].content = replace_pattern(E[i].content,":=:=","==");
}
exprtk::function_compositor<float> compositor(symbol_table);
// define function lerp(a,b,c) {a*(c-b)+b}
compositor.add("lerp", " a*(c-b)+b;", "a","b","c");
exprtk::expression<float> expressionR;
expressionR.register_symbol_table(symbol_table);
exprtk::parser<float> parserR;
doR = parserR.compile(E[9].content,expressionR);
exprtk::expression<float> expressionG;
expressionG.register_symbol_table(symbol_table);
exprtk::parser<float> parserG;
doG = parserG.compile(E[10].content,expressionG);
exprtk::expression<float> expressionB;
expressionB.register_symbol_table(symbol_table);
exprtk::parser<float> parserB;
doB = parserB.compile(E[11].content,expressionB);

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
param4 = (float)_param4;
param5 = (float)_param5;
param6 = (float)_param6;
param7 = (float)_param7;
param8 = (float)_param8;
param9 = (float)_param9;
param10_red = (float)_param10.r;
param10_green = (float)_param10.g;
param10_blue = (float)_param10.b;
param11 = (float)_param11;
x_coord = x;
y_coord = y;
width = p_ProcWindow.x2;
height = p_ProcWindow.y2;
for (int c = 0; c < 4; ++c) {
if (doR && c == 0) {
dstPix[0] = (expressionR.value() * param11) + (srcPix[0] * (1.0 - param11));
} else if (doG && c == 1) {
dstPix[1] = (expressionG.value() * param11) + (srcPix[1] * (1.0 - param11));
} else if (doB && c == 2) {
dstPix[2] = (expressionB.value() * param11) + (srcPix[2] * (1.0 - param11));
} else {
dstPix[c] = srcPix[c];
}}
dstPix += 4;
}}}

void ResolveMathxtra::setSrcImg(OFX::Image* p_SrcImg) {
_srcImg = p_SrcImg;
}

void ResolveMathxtra::setScales(const string& expr1, const string& expr2,
const string& expr3, const string& expr4, const string& expr5, const string& expr6,
const string& expr7, const string& expr8, const string& expr9, const string& exprR,
const string& exprG, const string& exprB, double param1, double param2, double param3,
double param4, double param5, double param6, double param7, double param8, double param9, 
const RGBValues& param10, double param11, bool processR, bool processG, bool processB)
{
_expr1 = expr1;
_expr2 = expr2;
_expr3 = expr3;
_expr4 = expr4;
_expr5 = expr5;
_expr6 = expr6;
_expr7 = expr7;
_expr8 = expr8;
_expr9 = expr9;
_exprR = exprR;
_exprG = exprG;
_exprB = exprB;
_param1 = param1;
_param2 = param2;
_param3 = param3;
_param4 = param4;
_param5 = param5;
_param6 = param6;
_param7 = param7;
_param8 = param8;
_param9 = param9;
_param10 = param10;
_param11 = param11;
_processR = processR;
_processG = processG;
_processB = processB;
}

////////////////////////////////////////////////////////////////////////////////

class ResolveMathxtraPlugin : public OFX::ImageEffect
{
public:
explicit ResolveMathxtraPlugin(OfxImageEffectHandle p_Handle);

virtual void render(const OFX::RenderArguments& p_Args);
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
void setupAndProcess(ResolveMathxtra &p_ResolveMathxtra, const OFX::RenderArguments& p_Args);


private:

OFX::Clip* m_DstClip;
OFX::Clip* m_SrcClip;
OFX::StringParam* _expr1;
OFX::StringParam* _expr2;
OFX::StringParam* _expr3;
OFX::StringParam* _expr4;
OFX::StringParam* _expr5;
OFX::StringParam* _expr6;
OFX::StringParam* _expr7;
OFX::StringParam* _expr8;
OFX::StringParam* _expr9;
OFX::StringParam* _exprR;
OFX::StringParam* _exprG;
OFX::StringParam* _exprB;
OFX::StringParam* _exprLUT;
OFX::DoubleParam* _param1;
OFX::DoubleParam* _param2;
OFX::DoubleParam* _param3;
OFX::DoubleParam* _param4;
OFX::DoubleParam* _param5;
OFX::DoubleParam* _param6;
OFX::DoubleParam* _param7;
OFX::DoubleParam* _param8;
OFX::DoubleParam* _param9;
OFX::RGBParam *_param10;
OFX::DoubleParam* _param11;
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
};

ResolveMathxtraPlugin::ResolveMathxtraPlugin(OfxImageEffectHandle p_Handle)
: ImageEffect(p_Handle)
{
m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
_expr1 = fetchStringParam(kParamExpr1Name);
_expr2 = fetchStringParam(kParamExpr2Name);
_expr3 = fetchStringParam(kParamExpr3Name);
_expr4 = fetchStringParam(kParamExpr4Name);
_expr5 = fetchStringParam(kParamExpr5Name);
_expr6 = fetchStringParam(kParamExpr6Name);
_expr7 = fetchStringParam(kParamExpr7Name);
_expr8 = fetchStringParam(kParamExpr8Name);
_expr9 = fetchStringParam(kParamExpr9Name);
_exprR = fetchStringParam(kParamResolveMathxtraR);
_exprG = fetchStringParam(kParamResolveMathxtraG);
_exprB = fetchStringParam(kParamResolveMathxtraB);
_exprLUT = fetchStringParam(kParamResolveMathxtraLUT);
assert(_expr1 && _expr2 && _expr3 && _expr4 && _expr5 && _expr6 && 
_expr7 && _expr8 && _expr9 && _exprR && _exprG && _exprB && _exprLUT);
_param1 = fetchDoubleParam(kParamParam1Name);
assert(_param1);
_param2 = fetchDoubleParam(kParamParam2Name);
assert(_param2);
_param3 = fetchDoubleParam(kParamParam3Name);
assert(_param3);
_param4 = fetchDoubleParam(kParamParam4Name);
assert(_param4);
_param5 = fetchDoubleParam(kParamParam5Name);
assert(_param5);
_param6 = fetchDoubleParam(kParamParam6Name);
assert(_param6);
_param7 = fetchDoubleParam(kParamParam7Name);
assert(_param7);
_param8 = fetchDoubleParam(kParamParam8Name);
assert(_param8);
_param9 = fetchDoubleParam(kParamParam9Name);
assert(_param9);
_param10 = fetchRGBParam(kParamParam10Name);
assert(_param10);
_param11 = fetchDoubleParam(kParamParam11Name);
assert(_param11);
m_Path = fetchStringParam("path");
m_Name = fetchStringParam("name");
m_Info = fetchPushButtonParam("info");
m_Button1 = fetchPushButtonParam("button1");
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
m_Button2 = fetchPushButtonParam("button2");
}

void ResolveMathxtraPlugin::render(const OFX::RenderArguments& p_Args)
{
if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA)) {
ResolveMathxtra ResolveMathxtra(*this);
setupAndProcess(ResolveMathxtra, p_Args);
} else {
OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
}}

bool ResolveMathxtraPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
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

void ResolveMathxtraPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
if (p_ParamName == "info") {
sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
}

if (p_ParamName == "button1") {
string expr1, expr2, expr3, expr4, expr5, expr6, 
expr7, expr8, expr9, exprR, exprG, exprB;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_expr4->getValue(expr4);
_expr5->getValue(expr5);
_expr6->getValue(expr6);
_expr7->getValue(expr7);
_expr8->getValue(expr8);
_expr9->getValue(expr9);
_exprR->getValue(exprR);
_exprG->getValue(exprG);
_exprB->getValue(exprB);
expr1 = expr1.empty() ? "0.000000f" : expr1;
expr2 = expr2.empty() ? "0.000000f" : expr2;
expr3 = expr3.empty() ? "0.000000f" : expr3;
expr4 = expr4.empty() ? "0.000000f" : expr4;
expr5 = expr5.empty() ? "0.000000f" : expr5;
expr6 = expr6.empty() ? "0.000000f" : expr6;
expr7 = expr7.empty() ? "0.000000f" : expr7;
expr8 = expr8.empty() ? "0.000000f" : expr8;
expr9 = expr9.empty() ? "0.000000f" : expr9;
exprR = exprR.empty() ? "r " : exprR;
exprG = exprG.empty() ? "g " : exprG;
exprB = exprB.empty() ? "b " : exprB;
float param1 = _param1->getValueAtTime(p_Args.time);
float param2 = _param2->getValueAtTime(p_Args.time);
float param3 = _param3->getValueAtTime(p_Args.time);
float param4 = _param4->getValueAtTime(p_Args.time);
float param5 = _param5->getValueAtTime(p_Args.time);
float param6 = _param6->getValueAtTime(p_Args.time);
float param7 = _param7->getValueAtTime(p_Args.time);
float param8 = _param8->getValueAtTime(p_Args.time);
float param9 = _param9->getValueAtTime(p_Args.time);
float param11 = _param11->getValueAtTime(p_Args.time);
RGBValues param10;
_param10->getValueAtTime(p_Args.time, param10.r, param10.g, param10.b);
expr1 = replace_pattern(expr1, "param10.r", "param10r");
expr2 = replace_pattern(expr2, "param10.r", "param10r");
expr3 = replace_pattern(expr3, "param10.r", "param10r");
expr4 = replace_pattern(expr4, "param10.r", "param10r");
expr5 = replace_pattern(expr5, "param10.r", "param10r");
expr6 = replace_pattern(expr6, "param10.r", "param10r");
expr7 = replace_pattern(expr7, "param10.r", "param10r");
expr8 = replace_pattern(expr8, "param10.r", "param10r");
expr9 = replace_pattern(expr9, "param10.r", "param10r");
exprR = replace_pattern(exprR, "param10.r", "param10r");
exprG = replace_pattern(exprG, "param10.r", "param10r");
exprB = replace_pattern(exprB, "param10.r", "param10r");
expr1 = replace_pattern(expr1, "param10.g", "param10g");
expr2 = replace_pattern(expr2, "param10.g", "param10g");
expr3 = replace_pattern(expr3, "param10.g", "param10g");
expr4 = replace_pattern(expr4, "param10.g", "param10g");
expr5 = replace_pattern(expr5, "param10.g", "param10g");
expr6 = replace_pattern(expr6, "param10.g", "param10g");
expr7 = replace_pattern(expr7, "param10.g", "param10g");
expr8 = replace_pattern(expr8, "param10.g", "param10g");
expr9 = replace_pattern(expr9, "param10.g", "param10g");
exprR = replace_pattern(exprR, "param10.g", "param10g");
exprG = replace_pattern(exprG, "param10.g", "param10g");
exprB = replace_pattern(exprB, "param10.g", "param10g");
expr1 = replace_pattern(expr1, "param10.b", "param10b");
expr2 = replace_pattern(expr2, "param10.b", "param10b");
expr3 = replace_pattern(expr3, "param10.b", "param10b");
expr4 = replace_pattern(expr4, "param10.b", "param10b");
expr5 = replace_pattern(expr5, "param10.b", "param10b");
expr6 = replace_pattern(expr6, "param10.b", "param10b");
expr7 = replace_pattern(expr7, "param10.b", "param10b");
expr8 = replace_pattern(expr8, "param10.b", "param10b");
expr9 = replace_pattern(expr9, "param10.b", "param10b");
exprR = replace_pattern(exprR, "param10.b", "param10b");
exprG = replace_pattern(exprG, "param10.b", "param10b");
exprB = replace_pattern(exprB, "param10.b", "param10b");
expr1 = replace_pattern(expr1, "pow", "_powf");
expr2 = replace_pattern(expr2, "pow", "_powf");
expr3 = replace_pattern(expr3, "pow", "_powf");
expr4 = replace_pattern(expr4, "pow", "_powf");
expr5 = replace_pattern(expr5, "pow", "_powf");
expr6 = replace_pattern(expr6, "pow", "_powf");
expr7 = replace_pattern(expr7, "pow", "_powf");
expr8 = replace_pattern(expr8, "pow", "_powf");
expr9 = replace_pattern(expr9, "pow", "_powf");
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
fprintf (pFile, "// ResolveMathxtra DCTL export\n" \
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
" \n" \
"float mix = %ff; \n" \
"float param1 = %ff; \n" \
"float param2 = %ff; \n" \
"float param3 = %ff; \n" \
"float param4 = %ff; \n" \
"float param5 = %ff; \n" \
"float param6 = %ff; \n" \
"float param7 = %ff; \n" \
"float param8 = %ff; \n" \
"float param9 = %ff; \n" \
"float param10r = %ff; \n" \
"float param10g = %ff; \n" \
"float param10b = %ff;\n", param11, param1, param2, param3, param4,
param5, param6, param7, param8, param9, param10.r, param10.g, param10.b);
fprintf (pFile, "\n" \
"float expr1 = %s; \n" \
"float expr2 = %s; \n" \
"float expr3 = %s; \n" \
"float expr4 = %s; \n" \
"float expr5 = %s; \n" \
"float expr6 = %s; \n" \
"float expr7 = %s; \n" \
"float expr8 = %s; \n" \
"float expr9 = %s; \n" \
" \n" \
"float R1 = %s; \n" \
"float G1 = %s; \n" \
"float B1 = %s; \n" \
" \n" \
"float R = R1 * (1.0f - mix) + r * mix; \n" \
"float G = G1 * (1.0f - mix) + g * mix; \n" \
"float B = B1 * (1.0f - mix) + b * mix; \n" \
" \n" \
"return make_float3(R, G, B); \n" \
"}\n", expr1.c_str(), expr2.c_str(), expr3.c_str(), expr4.c_str(), expr5.c_str(), expr6.c_str(), 
expr7.c_str(), expr8.c_str(), expr9.c_str(), exprR.c_str(), exprG.c_str(), exprB.c_str());
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to" + PATH  + ". Check Permissions."));
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

if(p_ParamName == "range1" || p_ParamName == kParamResolveMathxtraLUT || p_ParamName == kParamLUT) {
string expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8, expr9, exprR, exprG, exprB, exprLUT;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_expr4->getValue(expr4);
_expr5->getValue(expr5);
_expr6->getValue(expr6);
_expr7->getValue(expr7);
_expr8->getValue(expr8);
_expr9->getValue(expr9);
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
double Param4;
_param4->getValueAtTime(p_Args.time,Param4);
double Param5;
_param5->getValueAtTime(p_Args.time,Param5);
double Param6;
_param6->getValueAtTime(p_Args.time,Param6);
double Param7;
_param7->getValueAtTime(p_Args.time,Param7);
double Param8;
_param8->getValueAtTime(p_Args.time,Param8);
double Param9;
_param9->getValueAtTime(p_Args.time,Param9);
RGBValues Param10;
_param10->getValueAtTime(p_Args.time, Param10.r, Param10.g, Param10.b);
double Param11;
_param11->getValueAtTime(p_Args.time,Param11);
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
expr4 = expr4.empty() ? "0" : expr4;
expr5 = expr5.empty() ? "0" : expr5;
expr6 = expr6.empty() ? "0" : expr6;
expr7 = expr7.empty() ? "0" : expr7;
expr8 = expr8.empty() ? "0" : expr8;
expr9 = expr9.empty() ? "0" : expr9;
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
double param4;
double param5;
double param6;
double param7;
double param8;
double param9;
double param10_red;
double param10_green;
double param10_blue;
double param11;
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
symbol_table.add_variable("param4", param4);
symbol_table.add_variable("param5", param5);
symbol_table.add_variable("param6", param6);
symbol_table.add_variable("param7", param7);
symbol_table.add_variable("param8", param8);
symbol_table.add_variable("param9", param9);
symbol_table.add_variable("param10_r",param10_red);
symbol_table.add_variable("param10_g",param10_green);
symbol_table.add_variable("param10_b",param10_blue);
symbol_table.add_variable("x",x_coord);
symbol_table.add_variable("y",y_coord);
symbol_table.add_variable("width",width);
symbol_table.add_variable("height",height);
ResolveMathxtraProperties expr1_props = {kParamExpr1Name, expr1, true};
ResolveMathxtraProperties expr2_props = {kParamExpr2Name, expr2, true};
ResolveMathxtraProperties expr3_props = {kParamExpr3Name, expr3, true};
ResolveMathxtraProperties expr4_props = {kParamExpr4Name, expr4, true};
ResolveMathxtraProperties expr5_props = {kParamExpr5Name, expr5, true};
ResolveMathxtraProperties expr6_props = {kParamExpr6Name, expr6, true};
ResolveMathxtraProperties expr7_props = {kParamExpr7Name, expr7, true};
ResolveMathxtraProperties expr8_props = {kParamExpr8Name, expr8, true};
ResolveMathxtraProperties expr9_props = {kParamExpr9Name, expr9, true};
ResolveMathxtraProperties exprR_props = {kParamResolveMathxtraR, exprR, true};
ResolveMathxtraProperties exprG_props = {kParamResolveMathxtraG, exprG, true};
ResolveMathxtraProperties exprB_props = {kParamResolveMathxtraB, exprB, true};
ResolveMathxtraProperties exprLUT_props = {kParamResolveMathxtraLUT, exprLUT, true};
const int Esize = 13;
ResolveMathxtraProperties E[Esize] = {expr1_props, expr2_props, expr3_props, 
expr4_props, expr5_props, expr6_props, expr7_props, expr8_props, 
expr9_props, exprR_props, exprG_props, exprB_props, exprLUT_props};
for (int i = 0; i != Esize; ++i) {
for (int k = 0; k != Esize; ++ k) {
if (E[i].content.find(E[i].name) != string::npos){
E[i].content.clear();
E[i].processFlag = false;
}  else if ((i != k) && !E[i].content.empty() && !E[k].content.empty() ) { 
E[i].content = replace_pattern(E[i].content,E[k].name,"("+E[k].content+")");
}}
E[i].content = replace_pattern(E[i].content,"param1.","param1_");
E[i].content = replace_pattern(E[i].content,"param2.","param2_");
E[i].content = replace_pattern(E[i].content,"param3.","param3_");
E[i].content = replace_pattern(E[i].content,"param4.","param4_");
E[i].content = replace_pattern(E[i].content,"param5.","param5_");
E[i].content = replace_pattern(E[i].content,"param6.","param6_");
E[i].content = replace_pattern(E[i].content,"param7.","param7_");
E[i].content = replace_pattern(E[i].content,"param8.","param8_");
E[i].content = replace_pattern(E[i].content,"param9.","param9_");
E[i].content = replace_pattern(E[i].content,"param10.","param10_");
}
exprtk::function_compositor<double> compositor(symbol_table);
compositor.add("lerp", " a*(c-b)+b;", "a","b","c");
exprtk::expression<double> expressionR;
expressionR.register_symbol_table(symbol_table);
exprtk::parser<double> parserR;
doR = parserR.compile(E[9].content,expressionR);
exprtk::expression<double> expressionG;
expressionG.register_symbol_table(symbol_table);
exprtk::parser<double> parserG;
doG = parserG.compile(E[10].content,expressionG);
exprtk::expression<double> expressionB;
expressionB.register_symbol_table(symbol_table);
exprtk::parser<double> parserB;
doB = parserB.compile(E[11].content,expressionB);
exprtk::expression<double> expressionLUT;
expressionLUT.register_symbol_table(symbol_table);
exprtk::parser<double> parserLUT;
doLUT = parserLUT.compile(E[12].content,expressionLUT);
param1 = Param1;
param2 = Param2;
param3 = Param3;
param4 = Param4;
param5 = Param5;
param6 = Param6;
param7 = Param7;
param8 = Param8;
param9 = Param9;
param10_red = Param10.r;
param10_green = Param10.g;
param10_blue = Param10.b;
param11 = Param11;
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

if (p_ParamName == "button2") {
string expr1, expr2, expr3, expr4, expr5, expr6, 
expr7, expr8, expr9, exprR, exprG, exprB, exprLUT;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_expr4->getValue(expr4);
_expr5->getValue(expr5);
_expr6->getValue(expr6);
_expr7->getValue(expr7);
_expr8->getValue(expr8);
_expr9->getValue(expr9);
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
double Param4;
_param4->getValueAtTime(p_Args.time,Param4);
double Param5;
_param5->getValueAtTime(p_Args.time,Param5);
double Param6;
_param6->getValueAtTime(p_Args.time,Param6);
double Param7;
_param7->getValueAtTime(p_Args.time,Param7);
double Param8;
_param8->getValueAtTime(p_Args.time,Param8);
double Param9;
_param9->getValueAtTime(p_Args.time,Param9);
RGBValues Param10;
_param10->getValueAtTime(p_Args.time, Param10.r, Param10.g, Param10.b);
double Param11;
_param11->getValueAtTime(p_Args.time,Param11);
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
expr4 = expr4.empty() ? "0" : expr4;
expr5 = expr5.empty() ? "0" : expr5;
expr6 = expr6.empty() ? "0" : expr6;
expr7 = expr7.empty() ? "0" : expr7;
expr8 = expr8.empty() ? "0" : expr8;
expr9 = expr9.empty() ? "0" : expr9;
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
double param4;
double param5;
double param6;
double param7;
double param8;
double param9;
double param10_red;
double param10_green;
double param10_blue;
double param11;
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
symbol_table.add_variable("param4", param4);
symbol_table.add_variable("param5", param5);
symbol_table.add_variable("param6", param6);
symbol_table.add_variable("param7", param7);
symbol_table.add_variable("param8", param8);
symbol_table.add_variable("param9", param9);
symbol_table.add_variable("param10_r",param10_red);
symbol_table.add_variable("param10_g",param10_green);
symbol_table.add_variable("param10_b",param10_blue);
symbol_table.add_variable("x",x_coord);
symbol_table.add_variable("y",y_coord);
symbol_table.add_variable("width",width);
symbol_table.add_variable("height",height);
ResolveMathxtraProperties expr1_props = {kParamExpr1Name, expr1, true};
ResolveMathxtraProperties expr2_props = {kParamExpr2Name, expr2, true};
ResolveMathxtraProperties expr3_props = {kParamExpr3Name, expr3, true};
ResolveMathxtraProperties expr4_props = {kParamExpr4Name, expr4, true};
ResolveMathxtraProperties expr5_props = {kParamExpr5Name, expr5, true};
ResolveMathxtraProperties expr6_props = {kParamExpr6Name, expr6, true};
ResolveMathxtraProperties expr7_props = {kParamExpr7Name, expr7, true};
ResolveMathxtraProperties expr8_props = {kParamExpr8Name, expr8, true};
ResolveMathxtraProperties expr9_props = {kParamExpr9Name, expr9, true};
ResolveMathxtraProperties exprR_props = {kParamResolveMathxtraR, exprR, true};
ResolveMathxtraProperties exprG_props = {kParamResolveMathxtraG, exprG, true};
ResolveMathxtraProperties exprB_props = {kParamResolveMathxtraB, exprB, true};
ResolveMathxtraProperties exprLUT_props = {kParamResolveMathxtraLUT, exprLUT, true};
const int Esize = 13;
ResolveMathxtraProperties E[Esize] = {expr1_props, expr2_props, expr3_props, 
expr4_props, expr5_props, expr6_props, expr7_props, expr8_props, 
expr9_props, exprR_props, exprG_props, exprB_props, exprLUT_props};
for (int i = 0; i != Esize; ++i) {
for (int k = 0; k != Esize; ++ k) {
if (E[i].content.find(E[i].name) != string::npos){
E[i].content.clear();
E[i].processFlag = false;
}  else if ((i != k) && !E[i].content.empty() && !E[k].content.empty() ) { 
E[i].content = replace_pattern(E[i].content,E[k].name,"("+E[k].content+")");
}}
E[i].content = replace_pattern(E[i].content,"param1.","param1_");
E[i].content = replace_pattern(E[i].content,"param2.","param2_");
E[i].content = replace_pattern(E[i].content,"param3.","param3_");
E[i].content = replace_pattern(E[i].content,"param4.","param4_");
E[i].content = replace_pattern(E[i].content,"param5.","param5_");
E[i].content = replace_pattern(E[i].content,"param6.","param6_");
E[i].content = replace_pattern(E[i].content,"param7.","param7_");
E[i].content = replace_pattern(E[i].content,"param8.","param8_");
E[i].content = replace_pattern(E[i].content,"param9.","param9_");
E[i].content = replace_pattern(E[i].content,"param10.","param10_");
}

exprtk::function_compositor<double> compositor(symbol_table);
compositor.add("lerp", " a*(c-b)+b;", "a","b","c");
exprtk::expression<double> expressionR;
expressionR.register_symbol_table(symbol_table);
exprtk::parser<double> parserR;
doR = parserR.compile(E[9].content,expressionR);
exprtk::expression<double> expressionG;
expressionG.register_symbol_table(symbol_table);
exprtk::parser<double> parserG;
doG = parserG.compile(E[10].content,expressionG);
exprtk::expression<double> expressionB;
expressionB.register_symbol_table(symbol_table);
exprtk::parser<double> parserB;
doB = parserB.compile(E[11].content,expressionB);
exprtk::expression<double> expressionLUT;
expressionLUT.register_symbol_table(symbol_table);
exprtk::parser<double> parserLUT;
doLUT = parserLUT.compile(E[12].content,expressionLUT);
param1 = Param1;
param2 = Param2;
param3 = Param3;
param4 = Param4;
param5 = Param5;
param6 = Param6;
param7 = Param7;
param8 = Param8;
param9 = Param9;
param10_red = Param10.r;
param10_green = Param10.g;
param10_blue = Param10.b;
param11 = Param11;
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
LUTEnum _LUT = (LUTEnum)LUT_i;
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
if(Lut > 0)
fprintf (pFile, "LUT_1D_SIZE %d\n" \
"LUT_1D_INPUT_RANGE %.*f %.*f\n" \
"\n", lutsize, decimal, shaperA, decimal, shaperB);
if(Lut != 1){
fprintf (pFile, "LUT_3D_SIZE %d\n" \
"LUT_3D_INPUT_RANGE %.*f %.*f\n" \
"\n", cube, decimal, inputA, decimal, inputB);
}
if (Lut > 0){
for( int i = 0; i < lutsize; ++i ){
shaper = ((double)i / (lutsize - 1)) * (shaperB - shaperA) + shaperA;
temPix[3] = shaper;
shaper1 = doLUT ? expressionLUT.value() : shaper;
fprintf (pFile, "%.*f %.*f %.*f\n", decimal, shaper1, decimal, shaper1, decimal, shaper1);
}
fprintf (pFile, "\n");
}
if (Lut != 1){
for( int i = 0; i < total; ++i ){
double R = fmod(i, cube) / (cube - 1) * (inputB - inputA) + inputA;
double G = fmod(floor(i / cube), cube) / (cube - 1) * (inputB - inputA) + inputA;
double B = fmod(floor(i / (cube * cube)), cube) / (cube - 1) * (inputB - inputA) + inputA;
temPix[0] = R;
temPix[1] = G;
temPix[2] = B;
x_coord = fmod(i, cube);
y_coord = fmod(floor(i / cube), cube); 
R = doR ? (expressionR.value() * param5) + (R * (1.0 - param5)) : R;
G = doG ? (expressionG.value() * param5) + (G * (1.0 - param5)) : G;
B = doB ? (expressionB.value() * param5) + (B * (1.0 - param5)) : B;
fprintf (pFile, "%.*f %.*f %.*f\n", decimal, R, decimal, G, decimal, B);
}}
fclose (pFile);
} else {
sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".cube to " + PATH + ". Check Permissions."));
}}}}

void ResolveMathxtraPlugin::setupAndProcess(ResolveMathxtra& p_ResolveMathxtra, const OFX::RenderArguments& p_Args)
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
string expr4;
string expr5;
string expr6;
string expr7;
string expr8;
string expr9;
string exprR;
string exprG;
string exprB;
_expr1->getValue(expr1);
_expr2->getValue(expr2);
_expr3->getValue(expr3);
_expr4->getValue(expr4);
_expr5->getValue(expr5);
_expr6->getValue(expr6);
_expr7->getValue(expr7);
_expr8->getValue(expr8);
_expr9->getValue(expr9);
_exprR->getValue(exprR);
_exprG->getValue(exprG);
_exprB->getValue(exprB);
double param1;
_param1->getValueAtTime(p_Args.time,param1);
double param2;
_param2->getValueAtTime(p_Args.time,param2);
double param3;
_param3->getValueAtTime(p_Args.time,param3);
double param4;
_param4->getValueAtTime(p_Args.time,param4);
double param5;
_param5->getValueAtTime(p_Args.time,param5);
double param6;
_param6->getValueAtTime(p_Args.time,param6);
double param7;
_param7->getValueAtTime(p_Args.time,param7);
double param8;
_param8->getValueAtTime(p_Args.time,param8);
double param9;
_param9->getValueAtTime(p_Args.time,param9);
RGBValues param10;
_param10->getValueAtTime(p_Args.time, param10.r, param10.g, param10.b);
double param11;
_param11->getValueAtTime(p_Args.time,param11);
bool processR = !exprR.empty();
bool processG = !exprG.empty();
bool processB = !exprB.empty();

p_ResolveMathxtra.setDstImg(dst.get());
p_ResolveMathxtra.setSrcImg(src.get());

p_ResolveMathxtra.setRenderWindow(p_Args.renderWindow);

p_ResolveMathxtra.setScales(expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8, expr9, 
exprR, exprG, exprB, param1, param2, param3, param4, param5, param6, param7, param8, param9, 
param10, param11, processR, processG, processB);

p_ResolveMathxtra.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ResolveMathxtraPluginFactory::ResolveMathxtraPluginFactory()
: PluginFactoryHelper<ResolveMathxtraPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ResolveMathxtraPluginFactory::describe(ImageEffectDescriptor& p_Desc)
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

void ResolveMathxtraPluginFactory::describeInContext(ImageEffectDescriptor& p_Desc, ContextEnum /*p_Context*/)
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

stringparam = p_Desc.defineStringParam(kParamExpr4Name);
stringparam->setLabel(kParamExpr4Label);
stringparam->setHint(kParamExpr4Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamExpr5Name);
stringparam->setLabel(kParamExpr5Label);
stringparam->setHint(kParamExpr5Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamExpr6Name);
stringparam->setLabel(kParamExpr6Label);
stringparam->setHint(kParamExpr6Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamExpr7Name);
stringparam->setLabel(kParamExpr7Label);
stringparam->setHint(kParamExpr7Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamExpr8Name);
stringparam->setLabel(kParamExpr8Label);
stringparam->setHint(kParamExpr8Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamExpr9Name);
stringparam->setLabel(kParamExpr9Label);
stringparam->setHint(kParamExpr9Hint);
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamResolveMathxtraR);
stringparam->setLabel(kParamResolveMathxtraRLabel);
stringparam->setHint(kParamResolveMathxtraRHint);
stringparam->setDefault("r");
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamResolveMathxtraG);
stringparam->setLabel(kParamResolveMathxtraGLabel);
stringparam->setHint(kParamResolveMathxtraGHint);
stringparam->setDefault("g");
stringparam->setAnimates(false);
if (page)
page->addChild(*stringparam);

stringparam = p_Desc.defineStringParam(kParamResolveMathxtraB);
stringparam->setLabel(kParamResolveMathxtraBLabel);
stringparam->setHint(kParamResolveMathxtraBHint);
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

param = p_Desc.defineDoubleParam(kParamParam4Name);
param->setLabel(kParamParam4Label);
param->setHint(kParamParam4Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

param = p_Desc.defineDoubleParam(kParamParam5Name);
param->setLabel(kParamParam5Label);
param->setHint(kParamParam5Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

param = p_Desc.defineDoubleParam(kParamParam6Name);
param->setLabel(kParamParam6Label);
param->setHint(kParamParam6Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

param = p_Desc.defineDoubleParam(kParamParam7Name);
param->setLabel(kParamParam7Label);
param->setHint(kParamParam7Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

param = p_Desc.defineDoubleParam(kParamParam8Name);
param->setLabel(kParamParam8Label);
param->setHint(kParamParam8Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

param = p_Desc.defineDoubleParam(kParamParam9Name);
param->setLabel(kParamParam9Label);
param->setHint(kParamParam9Hint);
param->setDefault(1.0);
param->setIncrement(0.001);
param->setRange(-100.0, 100.0);
param->setDisplayRange(0.0, 2.0);
param->setAnimates(true);
if (page)
page->addChild(*param);

RGBParamDescriptor *paramRGB = p_Desc.defineRGBParam(kParamParam10Name);
paramRGB->setLabel(kParamParam10Label);
paramRGB->setHint(kParamParam10Hint);
paramRGB->setDefault(1.0, 1.0, 1.0);
paramRGB->setRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
paramRGB->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
param->setAnimates(true);
if (page)
page->addChild(*paramRGB);

param = p_Desc.defineDoubleParam(kParamParam11Name);
param->setLabel(kParamParam11Label);
param->setHint(kParamParam11Hint);
param->setDefault(1.0);
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

GroupParamDescriptor* script = p_Desc.defineGroupParam("DCTL Export");
script->setOpen(false);
script->setHint("export DCTL");
if (page)
page->addChild(*script);

pushparam = p_Desc.definePushButtonParam("button1");
pushparam->setLabel("Export DCTL");
pushparam->setHint("create DCTL version");
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

stringparam = p_Desc.defineStringParam(kParamResolveMathxtraLUT);
stringparam->setLabel(kParamResolveMathxtraLUTLabel);
stringparam->setHint(kParamResolveMathxtraLUTHint);
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

pushparam = p_Desc.definePushButtonParam("button2");
pushparam->setLabel("Export LUT");
pushparam->setHint("create 3D LUT");
pushparam->setParent(*lutexport);
page->addChild(*pushparam);
}

ImageEffect* ResolveMathxtraPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
return new ResolveMathxtraPlugin(p_Handle);
}

void Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
static ResolveMathxtraPluginFactory ResolveMathxtraPlugin;
p_FactoryArray.push_back(&ResolveMathxtraPlugin);
}