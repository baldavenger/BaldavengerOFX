#include "ResolveMathxtra.h"
#include <cstring>
#include <cmath>
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
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#else
#define kPluginScript "/home/resolve/LUT"
#endif

using namespace OFX;
using namespace std;

#define kPluginName "ResolveMathxtra"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"Useful information : The following references can be applied to the expressions \n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Red, Green, Blue channels: r, g, b // Coordinates: x, y // Operators: +, -, *, /, ^, =\n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"Functions: min, max, avg, sum, abs, fmod, ceil, floor, round, pow, exp, log, root\n" \
"sqrt, lerp, sin, cos, tan, asin, acos, atan, hypot // Conditionals: ==, !=, >=, && \n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"if(a == b, c, d) : If a equals b then c, else d // a == b ? c : d  If a equals b then c, else d\n" \
"------------------------------------------------------------------------------------------------------------------ \n" \
"clamp(a,b,c) : a clamped to between b and c // pi : 3.1415926536 // width, height"

#define kPluginIdentifier "OpenFX.Yo.ResolveMathxtra"
#define kPluginVersionMajor 1 
#define kPluginVersionMinor 2

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
string replace_pattern(string text, const string& pattern,
			    const string& replacement) {
     size_t pos = 0;
    while ((pos = text.find(pattern,pos)) != string::npos){
      text.replace(pos,pattern.length(),replacement);
      pos += replacement.length();
    }
    return text;
}
}

class ResolveMathxtra : public OFX::ImageProcessor
{
public:
    explicit ResolveMathxtra(OFX::ImageEffect& p_Instance);
    
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

	void setSrcImg(OFX::Image* p_SrcImg);
	void setScales(const string& expr1, const string& expr2,
	const string& expr3, const string& expr4, const string& expr5, const string& expr6,
	const string& expr7, const string& expr8, const string& expr9, const string& exprR,
	const string& exprG, const string& exprB, float param1, float param2, float param3,
	float param4, float param5, float param6, float param7, float param8, float param9, 
	const RGBValues& param10, float param11, bool processR, bool processG, bool processB);
	   		
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
    float _param1;
    float _param2;
    float _param3;
    float _param4;
    float _param5;
    float _param6;
    float _param7;
    float _param8;
    float _param9;
    RGBValues _param10;
    float _param11;
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
	
	//SYMBOLS FOR EXPRESSIONS
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
	    //if the expression references itself it is invalid and will be deleted for its henious crime
	    if (E[i].content.find(E[i].name) != string::npos){
	      E[i].content.clear();
	      E[i].processFlag = false;
	    }  //otherwise away we go and break down refs to all the other expressions 
	    else if ((i != k) && !E[i].content.empty() && !E[k].content.empty() ) { 
            E[i].content  = replace_pattern(E[i].content,E[k].name,"("+E[k].content+")");
	    }
	  }
	//exprtk does not like dot based naming so use underscores
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
      
    //define custom functions for exprtk to match SeExpr
    exprtk::function_compositor<float> compositor(symbol_table);
      // define function lerp(a,b,c) {a*(c-b)+b}
      compositor
      .add("lerp",
           " a*(c-b)+b;",
           "a","b","c");
      //clamp could not be overloaded so I've modified the exprtk.hpp for that
      
	// load expression for exprR
	exprtk::expression<float> expressionR;
	expressionR.register_symbol_table(symbol_table);
	exprtk::parser<float> parserR;
	doR = parserR.compile(E[9].content,expressionR);
	// load expression for exprG
	exprtk::expression<float> expressionG;
	expressionG.register_symbol_table(symbol_table);
	exprtk::parser<float> parserG;
        doG = parserG.compile(E[10].content,expressionG);
	// load expression for exprB
	exprtk::expression<float> expressionB;
	expressionB.register_symbol_table(symbol_table);
	exprtk::parser<float> parserB;
	doB = parserB.compile(E[11].content,expressionB);
	
	
	// pixelwise
        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
        
       			temPix[0] = srcPix[0];
        		temPix[1] = srcPix[1];
        		temPix[2] = srcPix[2];
        		
                //for the symbol table
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
               
                 // UPDATE ALL THE PIXELS
                for (int c = 0; c < 4; ++c) {
                    if (doR && c == 0) {
                        // RED OUTPUT Resolve
                        dstPix[0] = (expressionR.value() * param11) + (srcPix[0] * (1.0 - param11));
                    } else if (doG && c == 1) {
                        // GREEN OUTPUT Resolve
                        dstPix[1] = (expressionG.value() * param11) + (srcPix[1] * (1.0 - param11));
                    } else if (doB && c == 2) {
                        // BLUE OUTPUT Resolve
                        dstPix[2] = (expressionB.value() * param11) + (srcPix[2] * (1.0 - param11));
                    } else {
                        dstPix[c] = srcPix[c];
                    }
                }
               
                dstPix += 4;
        }
    }
}

void ResolveMathxtra::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ResolveMathxtra::setScales(const string& expr1, const string& expr2,
	const string& expr3, const string& expr4, const string& expr5, const string& expr6,
	const string& expr7, const string& expr8, const string& expr9, const string& exprR,
	const string& exprG, const string& exprB, float param1, float param2, float param3,
	float param4, float param5, float param6, float param7, float param8, float param9, 
	const RGBValues& param10, float param11, bool processR, bool processG, bool processB)
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
/** @brief The plugin that does our work */

class ResolveMathxtraPlugin : public OFX::ImageEffect
{
public:
    explicit ResolveMathxtraPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    
    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(ResolveMathxtra &p_ResolveMathxtra, const OFX::RenderArguments& p_Args);
    

private:
    // Does not own the following pointers
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
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
	OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
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
	assert(_expr1 && _expr2 && _expr3 && _expr4 && _expr5 && _expr6 && _expr7 && _expr8 && _expr9 
	&& _exprR && _exprG && _exprB);

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
             
  }

void ResolveMathxtraPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ResolveMathxtra ResolveMathxtra(*this);
        setupAndProcess(ResolveMathxtra, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool ResolveMathxtraPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    
    string expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8, expr9, exprR,  exprG,  exprB;
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
    RGBValues param10;
    _param10->getValueAtTime(p_Args.time, param10.r, param10.g, param10.b);
    
    if (exprR.empty() && exprG.empty() && exprB.empty()) {
       p_IdentityClip = m_SrcClip;
       p_IdentityTime = p_Args.time;
        return true;
    }
    
    return false;
}

void ResolveMathxtraPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
	if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	 
 	if(p_ParamName == "button1")
    {
	
 	string expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8, expr9, exprR,  exprG,  exprB;
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
	fprintf (pFile, "	float r = p_R;\n" \
	"	float g = p_G;\n" \
	"	float b = p_B;\n" \
	"	float x = p_X;\n" \
	"	float y = p_Y;\n" \
	"	int width = p_Width;\n" \
	"	int height = p_Height;\n" \
	"\n" \
	"	float mix = %ff;\n" \
	"	float param1 = %ff;\n" \
	"	float param2 = %ff;\n" \
	"	float param3 = %ff;\n" \
	"	float param4 = %ff;\n" \
	"	float param5 = %ff;\n" \
	"	float param6 = %ff;\n" \
	"	float param7 = %ff;\n" \
	"	float param8 = %ff;\n" \
	"	float param9 = %ff;\n" \
	"	float param10r = %ff;\n" \
	"	float param10g = %ff;\n" \
	"	float param10b = %ff;\n", param11, param1, param2, param3, param4,
	 param5, param6, param7, param8, param9, param10.r, param10.g, param10.b);
	fprintf (pFile, "\n" \
	"	float expr1 = %s;\n" \
	"	float expr2 = %s;\n" \
	"	float expr3 = %s;\n" \
	"	float expr4 = %s;\n" \
	"	float expr5 = %s;\n" \
	"	float expr6 = %s;\n" \
	"	float expr7 = %s;\n" \
	"	float expr8 = %s;\n" \
	"	float expr9 = %s;\n" \
	"\n" \
	"	float R1 = %s;\n" \
	"	float G1 = %s;\n" \
	"	float B1 = %s;\n" \
	"\n" \
	"	float R = R1 * mix + r * (1.0 - mix);\n" \
	"	float G = G1 * mix + g * (1.0 - mix);\n" \
	"	float B = B1 * mix + b * (1.0 - mix);\n" \
	"\n" \
	"	return make_float3(R, G, B);\n" \
	"}\n", expr1.c_str(), expr2.c_str(), expr3.c_str(), expr4.c_str(), expr5.c_str(), expr6.c_str(), 
	expr7.c_str(), expr8.c_str(), expr9.c_str(), exprR.c_str(), exprG.c_str(), exprB.c_str());
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to" + PATH  + ". Check Permissions."));
	}	
	}
	}
		
}
    
void ResolveMathxtraPlugin::setupAndProcess(ResolveMathxtra& p_ResolveMathxtra, const OFX::RenderArguments& p_Args)
{
    // Get the dst image
    std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
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
    
    //we wont process any Resolve that has a null expression
    //we won't worry about invalid expressions
    bool processR = !exprR.empty();
    bool processG = !exprG.empty();
    bool processB = !exprB.empty();
    
    // Set the images
    p_ResolveMathxtra.setDstImg(dst.get());
    p_ResolveMathxtra.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    //p_ResolveMathxtra.setGPURenderArgs(p_Args);

    // Set the render window
    p_ResolveMathxtra.setRenderWindow(p_Args.renderWindow);
    
    p_ResolveMathxtra.setScales(expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8, expr9, 
    exprR, exprG, exprB, param1, param2, param3, param4, param5, param6, param7, param8, param9, 
    param10, param11, processR, processG, processB);
 
    // Call the base class process member, this will call the derived templated process code
    p_ResolveMathxtra.process();
    

}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ResolveMathxtraPluginFactory::ResolveMathxtraPluginFactory()
    : OFX::PluginFactoryHelper<ResolveMathxtraPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ResolveMathxtraPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
    
}


void ResolveMathxtraPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    
    // make some pages and to things in
    PageParamDescriptor *page = p_Desc.definePageParam("Controls");

    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr1Name);
        param->setLabel(kParamExpr1Label);
        param->setHint(kParamExpr1Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr2Name);
        param->setLabel(kParamExpr2Label);
        param->setHint(kParamExpr2Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr3Name);
        param->setLabel(kParamExpr3Label);
        param->setHint(kParamExpr3Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr4Name);
        param->setLabel(kParamExpr4Label);
        param->setHint(kParamExpr4Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr5Name);
        param->setLabel(kParamExpr5Label);
        param->setHint(kParamExpr5Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr6Name);
        param->setLabel(kParamExpr6Label);
        param->setHint(kParamExpr6Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr7Name);
        param->setLabel(kParamExpr7Label);
        param->setHint(kParamExpr7Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr8Name);
        param->setLabel(kParamExpr8Label);
        param->setHint(kParamExpr8Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamExpr9Name);
        param->setLabel(kParamExpr9Label);
        param->setHint(kParamExpr9Hint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamResolveMathxtraR);
        param->setLabel(kParamResolveMathxtraRLabel);
        param->setHint(kParamResolveMathxtraRHint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamResolveMathxtraG);
        param->setLabel(kParamResolveMathxtraGLabel);
        param->setHint(kParamResolveMathxtraGHint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamResolveMathxtraB);
        param->setLabel(kParamResolveMathxtraBLabel);
        param->setHint(kParamResolveMathxtraBHint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam1Name);
        param->setLabel(kParamParam1Label);
        param->setHint(kParamParam1Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam2Name);
        param->setLabel(kParamParam2Label);
        param->setHint(kParamParam2Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam3Name);
        param->setLabel(kParamParam3Label);
        param->setHint(kParamParam3Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
	
	{
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam4Name);
        param->setLabel(kParamParam4Label);
        param->setHint(kParamParam4Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam5Name);
        param->setLabel(kParamParam5Label);
        param->setHint(kParamParam5Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam6Name);
        param->setLabel(kParamParam6Label);
        param->setHint(kParamParam6Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
	
	{
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam7Name);
        param->setLabel(kParamParam7Label);
        param->setHint(kParamParam7Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam8Name);
        param->setLabel(kParamParam8Label);
        param->setHint(kParamParam8Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor *param = p_Desc.defineDoubleParam(kParamParam9Name);
        param->setLabel(kParamParam9Label);
        param->setHint(kParamParam9Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
	
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam(kParamParam10Name);
        param->setLabel(kParamParam10Label);
        param->setHint(kParamParam10Hint);
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0, 0, 0, 4, 4, 4);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor* param = p_Desc.defineDoubleParam(kParamParam11Name);
        param->setLabel(kParamParam11Label);
        param->setHint(kParamParam11Hint);
        param->setDefault(1.0);
        param->setIncrement(0.001);
        param->setRange(0.0, 1.0);
        param->setDisplayRange(0.0, 1.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("info");
    param->setLabel("Info");
    param->setHint("useful info");
    page->addChild(*param);
    }
	
	{    
    GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
    script->setOpen(false);
    script->setHint("export DCTL and Nuke script");
      if (page) {
            page->addChild(*script);
            }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button1");
    param->setLabel("Export DCTL");
    param->setHint("create DCTL version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("name");
	param->setLabel("Name");
	param->setHint("overwrites if the same");
	param->setDefault("ResolveMathxtra");
	param->setParent(*script);
	page->addChild(*param);
	}
	{
	StringParamDescriptor* param = p_Desc.defineStringParam("path");
	param->setLabel("Directory");
	param->setHint("make sure it's the absolute path");
	param->setStringType(eStringTypeFilePath);
	param->setDefault(kPluginScript);
	param->setFilePathExists(false);
	param->setParent(*script);
	page->addChild(*param);
	}
	}        
    
}

ImageEffect* ResolveMathxtraPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new ResolveMathxtraPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static ResolveMathxtraPluginFactory ResolveMathxtraPlugin;
    p_FactoryArray.push_back(&ResolveMathxtraPlugin);
}
