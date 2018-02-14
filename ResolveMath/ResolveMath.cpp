#include "ResolveMath.h"
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

#define kPluginName "ResolveMath"
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

#define kPluginIdentifier "OpenFX.Yo.ResolveMath"
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
static const string  kParamExpr1Hint = "You can define an expression here and reference it in ResolveMath fields as 'expr1'";
static const string  kParamExpr2Name = "expr2";
static const string  kParamExpr2Label= "expr2";
static const string  kParamExpr2Hint = "Reference in ResolveMath fields as'expr2'";
static const string  kParamExpr3Name = "expr3";
static const string  kParamExpr3Label= "expr3";
static const string  kParamExpr3Hint = "Reference in ResolveMath fields as'expr3'";

static const string  kParamResolveMathR =     "red";
static const string  kParamResolveMathRLabel= "Red Output";
static const string  kParamResolveMathRHint = "Red Channel output";
static const string  kParamResolveMathG     = "green";
static const string  kParamResolveMathGLabel= "Green Output";
static const string  kParamResolveMathGHint = "Green Channel output";
static const string  kParamResolveMathB     = "blue";
static const string  kParamResolveMathBLabel ="Blue Output";
static const string  kParamResolveMathBHint = "Blue Channel output";

static const string  kParamParam1Name = "param1";
static const string  kParamParam1Label= "param1";
static const string  kParamParam1Hint = "Reference in ResolveMath fields as 'param1'";

static const string  kParamParam2Name = "param2";
static const string  kParamParam2Label= "param2";
static const string  kParamParam2Hint = "Reference in ResolveMath fields as 'param2'";

static const string  kParamParam3Name = "param3";
static const string  kParamParam3Label= "param3";
static const string  kParamParam3Hint = "Reference in ResolveMath fields as 'param3'";

static const string  kParamParam4Name = "param4";
static const string  kParamParam4Label= "param4";
static const string  kParamParam4Hint = "Reference in ResolveMath fields as 'param4.r||g||b'";

static const string  kParamParam5Name = "param5";
static const string  kParamParam5Label = "mix";
static const string  kParamParam5Hint = "Mix factor between original and transformed image";

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

class ResolveMath : public OFX::ImageProcessor
{
public:
    explicit ResolveMath(OFX::ImageEffect& p_Instance);
    
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

	void setSrcImg(OFX::Image* p_SrcImg);
	void setScales(const string& expr1, const string& expr2,
	 const string& expr3, const string& exprR, const string& exprG, const string& exprB,
	  double param1, double param2, double param3, const RGBValues& param4, double param5,
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
	
	//SYMBOLS FOR EXPRESSIONS
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
	ResolveMathProperties E[Esize] = {expr1_props, expr2_props, expr3_props, exprR_props,
					 exprG_props, exprB_props};

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
	doR = parserR.compile(E[3].content,expressionR);
	// load expression for exprG
	exprtk::expression<float> expressionG;
	expressionG.register_symbol_table(symbol_table);
	exprtk::parser<float> parserG;
        doG = parserG.compile(E[4].content,expressionG);
	// load expression for exprB
	exprtk::expression<float> expressionB;
	expressionB.register_symbol_table(symbol_table);
	exprtk::parser<float> parserB;
	doB = parserB.compile(E[5].content,expressionB);
	
	
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
                param4_red = (float)_param4.r;
                param4_green = (float)_param4.g;
                param4_blue = (float)_param4.b;
                param5 = (float)_param5;
                x_coord = x;
                y_coord = y;
                width = p_ProcWindow.x2;
                height = p_ProcWindow.y2;
               
                 // UPDATE ALL THE PIXELS
                for (int c = 0; c < 4; ++c) {
                    if (doR && c == 0) {
                        // RED OUTPUT Resolve
                        dstPix[0] = (expressionR.value() * param5) + (srcPix[0] * (1.0 - param5));
                    } else if (doG && c == 1) {
                        // GREEN OUTPUT Resolve
                        dstPix[1] = (expressionG.value() * param5) + (srcPix[1] * (1.0 - param5));
                    } else if (doB && c == 2) {
                        // BLUE OUTPUT Resolve
                        dstPix[2] = (expressionB.value() * param5) + (srcPix[2] * (1.0 - param5));
                    } else {
                        dstPix[c] = srcPix[c];
                    }
                }
               
                dstPix += 4;
        }
    }
}

void ResolveMath::setSrcImg(OFX::Image* p_SrcImg)
{
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
/** @brief The plugin that does our work */

class ResolveMathPlugin : public OFX::ImageEffect
{
public:
    explicit ResolveMathPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    
    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(ResolveMath &p_ResolveMath, const OFX::RenderArguments& p_Args);
    

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
       
	OFX::StringParam* _expr1;
    OFX::StringParam* _expr2;
    OFX::StringParam* _expr3;
    OFX::StringParam* _exprR;
    OFX::StringParam* _exprG;
    OFX::StringParam* _exprB;
    OFX::DoubleParam* _param1;
    OFX::DoubleParam* _param2;
    OFX::DoubleParam* _param3;
    OFX::RGBParam *_param4;
    OFX::DoubleParam* _param5;
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
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
	assert(_expr1 && _expr2 && _expr3 && _exprR && _exprG && _exprB);

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
             
  }

void ResolveMathPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ResolveMath ResolveMath(*this);
        setupAndProcess(ResolveMath, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool ResolveMathPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    
    string expr1, expr2, expr3, exprR,  exprG,  exprB;
    _expr1->getValue(expr1);
    _expr2->getValue(expr2);
    _expr3->getValue(expr3);
    _exprR->getValue(exprR);
    _exprG->getValue(exprG);
    _exprB->getValue(exprB);
   
    RGBValues param4;
    _param4->getValueAtTime(p_Args.time, param4.r, param4.g, param4.b);
    
    if (exprR.empty() && exprG.empty() && exprB.empty()) {
       p_IdentityClip = m_SrcClip;
       p_IdentityTime = p_Args.time;
        return true;
    }
    
    return false;
}

void ResolveMathPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
 	
 	if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
 	
 	if(p_ParamName == "button1")
    {
	
 	string expr1, expr2, expr3, exprR,  exprG,  exprB;
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
	"	float param4r = %ff;\n" \
	"	float param4g = %ff;\n" \
	"	float param4b = %ff;\n", param5, param1, param2, param3, param4.r, param4.g, param4.b);
	fprintf (pFile, "\n" \
	"	float expr1 = %s;\n" \
	"	float expr2 = %s;\n" \
	"	float expr3 = %s;\n" \
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
	"}\n", expr1.c_str(), expr2.c_str(), expr3.c_str(), exprR.c_str(), exprG.c_str(), exprB.c_str());
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to" + PATH  + ". Check Permissions."));
	}	
	}
	}
	
	if(p_ParamName == "button2")
    {
    
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
	}	
	}
	}

}
    
void ResolveMathPlugin::setupAndProcess(ResolveMath& p_ResolveMath, const OFX::RenderArguments& p_Args)
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
    
    //we wont process any Resolve that has a null expression
    //we won't worry about invalid expressions
    bool processR = !exprR.empty();
    bool processG = !exprG.empty();
    bool processB = !exprB.empty();
    
    // Set the images
    p_ResolveMath.setDstImg(dst.get());
    p_ResolveMath.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    //p_ResolveMath.setGPURenderArgs(p_Args);

    // Set the render window
    p_ResolveMath.setRenderWindow(p_Args.renderWindow);
    
    p_ResolveMath.setScales(expr1, expr2, expr3, exprR, exprG, exprB,
    param1, param2, param3, param4, param5, processR, processG, processB);
 
    // Call the base class process member, this will call the derived templated process code
    p_ResolveMath.process();
    

}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

ResolveMathPluginFactory::ResolveMathPluginFactory()
    : OFX::PluginFactoryHelper<ResolveMathPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void ResolveMathPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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


void ResolveMathPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamResolveMathR);
        param->setLabel(kParamResolveMathRLabel);
        param->setHint(kParamResolveMathRHint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamResolveMathG);
        param->setLabel(kParamResolveMathGLabel);
        param->setHint(kParamResolveMathGHint);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::StringParamDescriptor* param = p_Desc.defineStringParam(kParamResolveMathB);
        param->setLabel(kParamResolveMathBLabel);
        param->setHint(kParamResolveMathBHint);
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
        param->setDisplayRange(0.0, 2.0);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }

    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam(kParamParam4Name);
        param->setLabel(kParamParam4Label);
        param->setHint(kParamParam4Hint);
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0, 0, 0, 4, 4, 4);
        param->setAnimates(true); // can animate
        if (page) {
            page->addChild(*param);
        }
    }
    
    {
        DoubleParamDescriptor* param = p_Desc.defineDoubleParam(kParamParam5Name);
        param->setLabel(kParamParam5Label);
        param->setHint(kParamParam5Hint);
        param->setDefault(1.0);
        param->setIncrement(0.01);
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
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button2");
    param->setLabel("Export Nuke script");
    param->setHint("create NUKE version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("name");
	param->setLabel("Name");
	param->setHint("overwrites if the same");
	param->setDefault("ResolveMath");
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

ImageEffect* ResolveMathPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new ResolveMathPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static ResolveMathPluginFactory ResolveMathPlugin;
    p_FactoryArray.push_back(&ResolveMathPlugin);
}
