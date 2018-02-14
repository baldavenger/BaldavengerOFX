#include "HueConvergePlugin.h"

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
#define kPluginScript "/opt/resolve/LUT"
#endif

#define kPluginName "HueConverge"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"HueConverge: Allows for the isolation and convergence of specific ranges of Hue. Also \n" \
"included are Logistic Function controls, Saturation Soft-Clip, and Saturation based Luma \n" \
"control."

#define kPluginIdentifier "OpenFX.Yo.HueConverge"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

namespace {
    struct RGBValues {
        double r,g,b;
        RGBValues(double v) : r(v), g(v), b(v) {}
        RGBValues() : r(0), g(0), b(0) {}
    };
}

float RGBtoHUE(float R, float G, float B)
{

	float mn = fmin(R, fmin(G, B));    
	float Mx = fmax(R, fmax(G, B));    
	float del_Mx = Mx - mn;
	
	float del_R = ((Mx - R) / 6.0f + del_Mx / 2.0f) / del_Mx;
    float del_G = ((Mx - G) / 6.0f + del_Mx / 2.0f) / del_Mx;
    float del_B = ((Mx - B) / 6.0f + del_Mx / 2.0f) / del_Mx;
   
    float h = del_Mx == 0.0f ? 0.0f : R == Mx ? del_B - del_G : G == Mx ? 1.0f / 3.0f + 
    del_R - del_B : 2.0f / 3.0f + del_G - del_R;

    float Hh = h < 0.0f ? h + 1.0f : h > 1.0f ? h - 1.0f : h;
    return Hh;
       
}

class HueConverge : public OFX::ImageProcessor
{
public:
    explicit HueConverge(OFX::ImageEffect& p_Instance);

	virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
	void setScales(float p_SwitchA, float p_SwitchB, float p_SwitchC, float p_SwitchD, float p_SwitchE, float p_SwitchL, 
	float p_Alpha1, float p_Alpha2, float p_LogA, float p_LogB, float p_LogC, float p_LogD, float p_SatA, float p_SatB, 
	float p_SatC, float p_HueA, float p_HueB, float p_HueC, float p_HueR1, float p_HueP1,float p_HueSH1, float p_HueSHP1, 
	float p_HueR2, float p_HueP2, float p_HueSH2, float p_HueSHP2,float p_HueR3, float p_HueP3, float p_HueSH3, 
	float p_HueSHP3, float p_LumaA, float p_LumaB, float p_LumaC);

private:
    OFX::Image* _srcImg;
	float _switch[6];
	float _alpha[2];
    float _log[4];
    float _sat[3];
    float _hue[15];
    float _luma[3];
};

HueConverge::HueConverge(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}


extern void RunCudaKernel(int p_Width, int p_Height, float* p_Switch, float* p_Alpha, float* p_Log,
 float* p_Sat, float* p_Hue, float* p_Luma, const float* p_Input, float* p_Output);

void HueConverge::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, _switch, _alpha, _log, _sat, _hue, _luma, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Switch, float* p_Alpha, float* p_Log, 
float* p_Sat, float* p_Hue, float* p_Luma, const float* p_Input, float* p_Output);

void HueConverge::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _switch, _alpha, _log, _sat, _hue, _luma, input, output);
}

void HueConverge::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
            
            // Euler's Constant e = 2.718281828459045
    
    		double e = 2.718281828459045;
    
  		    // Default expression : 1.0f / (1.0f + _powf(e, -8.9f*(r - 0.435f)))
  		    
   		    // Logistic Function (Sigmoid Curve)
   		    
   		    float Lr = _switch[0] == 1.0f ? _log[0] / (1.0f + pow(e, (-8.9f * _log[1]) * (srcPix[0] - _log[2]))) + _log[3] : srcPix[0];
			float Lg = _switch[0] == 1.0f ? _log[0] / (1.0f + pow(e, (-8.9f * _log[1]) * (srcPix[1] - _log[2]))) + _log[3] : srcPix[1];
			float Lb = _switch[0] == 1.0f ? _log[0] / (1.0f + pow(e, (-8.9f * _log[1]) * (srcPix[2] - _log[2]))) + _log[3] : srcPix[2];
            
            float mn = fmin(Lr, fmin(Lg, Lb));    
			float Mx = fmax(Lr, fmax(Lg, Lb));    
			float del_Mx = Mx - mn;
			
            float L = (Mx + mn) / 2.0f;
			float luma = srcPix[0] * 0.2126f + srcPix[1] * 0.7152f + srcPix[2] * 0.0722f;
			float Sat = del_Mx == 0.0f ? 0.0f : L < 0.5f ? del_Mx / (Mx + mn) : del_Mx / (2.0f - Mx - mn);

            float Hh = RGBtoHUE(Lr, Lg, Lb);   
   		    
   		    // Soft Clip Saturation

			float s = Sat *_sat[0] > 1.0f ? 1.0f : Sat *_sat[0];
			float ss = s > _sat[1] ? (-1.0f / ((s - _sat[1]) / (1.0f - _sat[1]) + 1.0f) + 1.0f) * (1.0f - _sat[1]) + _sat[1] : s;
            float satAlphaA = _sat[2] > 1.0f ? luma + (1.0f - _sat[2]) * (1.0f - luma) : _sat[2] >= 0.0f ? (luma >= _sat[2] ? 
            1.0f : luma / _sat[2]) : _sat[2] < -1.0f ? (1.0f - luma) + (_sat[2] + 1.0f) * luma : luma <= (1.0f + _sat[2]) ? 1.0f : 
            (1.0f - luma) / (1.0f - (_sat[2] + 1.0f));
            float satAlpha = satAlphaA > 1.0 ? 1.0 : satAlphaA;

			float S = _switch[1] == 1.0f ? ss * satAlpha + s * (1.0f - satAlpha) : Sat;

			// Hue Convergence

			float h1 = Hh - (_hue[0] - 0.5f) < 0.0f ? Hh - (_hue[0] - 0.5f) + 1.0f : Hh - (_hue[0] - 0.5f) >
            1.0f ? Hh - (_hue[0] - 0.5f) - 1.0f : Hh - (_hue[0] - 0.5f);

			float Hs1 = _hue[6] >= 1.0f ? 1.0f - pow(1.0f - S, _hue[6]) : pow(S, 1.0f/_hue[6]);
			float cv1 = 1.0f + (_hue[4] - 1.0f) * Hs1;
			float curve1 = _hue[4] + (cv1 - _hue[4]) * _hue[5];
			
			float H1 = _switch[2] != 1.0f ? Hh : h1 > 0.5f - _hue[3] && h1 < 0.5f ? (1.0f - pow(1.0f - (h1 - (0.5f - _hue[3])) *
			(1.0f/_hue[3]), curve1)) * _hue[3] + (0.5f - _hue[3]) + (_hue[0] - 0.5f) : h1 > 0.5f && h1 < 0.5f + 
			_hue[3] ? pow((h1 - 0.5f) * (1.0f/_hue[3]), curve1) * _hue[3] + 0.5f + (_hue[0] - 0.5f) : Hh;
			
   		    float h2 = H1 - (_hue[1] - 0.5f) < 0.0f ? H1 - (_hue[1] - 0.5f) + 1.0f : H1 - (_hue[1] - 0.5f) > 
            1.0f ? H1 - (_hue[1] - 0.5f) - 1.0f : H1 - (_hue[1] - 0.5f);
   		    
   		    float Hs2 = _hue[10] >= 1.0f ? 1.0f - pow(1.0f - S, _hue[10]) : pow(S, 1.0f/_hue[10]);
			float cv2 = 1.0f + (_hue[8] - 1.0f) * Hs2;
			float curve2 = _hue[8] + (cv2 - _hue[8]) * _hue[9];
   		    
			float H2 = _switch[3] != 1.0f ? H1 : h2 > 0.5f - _hue[7] && h2 < 0.5f ? (1.0f - pow(1.0f - (h2 - (0.5f - _hue[7])) *
            (1.0f/_hue[7]), curve2)) * _hue[7] + (0.5f - _hue[7]) + (_hue[1] - 0.5f) : h2 > 0.5f && h2 < 0.5f + 
            _hue[7] ? pow((h2 - 0.5f) * (1.0f/_hue[7]), curve2) * _hue[7] + 0.5f + (_hue[1] - 0.5f) : H1;
   		    
   		    float h3 = H2 - (_hue[2] - 0.5f) < 0.0f ? H2 - (_hue[2] - 0.5f) + 1.0f : H2 - (_hue[2] - 0.5f) > 
            1.0f ? H2 - (_hue[2] - 0.5f) - 1.0f : H2 - (_hue[2] - 0.5f);
   		    
   		    float Hs3 = _hue[14] >= 1.0f ? 1.0f - pow(1.0f - S, _hue[14]) : pow(S, 1.0f/_hue[14]);
			float cv3 = 1.0f + (_hue[12] - 1.0f) * Hs3;
			float curve3 = _hue[12] + (cv3 - _hue[12]) * _hue[13];
   		    
			float H = _switch[4] != 1.0f ? H2 : h3 > 0.5f - _hue[11] && h3 < 0.5f ? (1.0f - pow(1.0f - (h3 - (0.5f - _hue[11])) *
   		    (1.0f/_hue[11]), curve3)) * _hue[11] + (0.5f - _hue[11]) + (_hue[2] - 0.5f) : h3 > 0.5f && h3 < 0.5f + 
   		    _hue[11] ? pow((h3 - 0.5f) * (1.0f/_hue[11]), curve3) * _hue[11] + 0.5f + (_hue[2] - 0.5f) : H2;
   		    
   		    
   		    // HSL to RGB

			float Q = L < 0.5f ? L * (1.0f + S) : L + S - L * S;
			float P = 2.0f * L - Q;

			float RH = H + 1.0f / 3.0f < 0.0f ? H + 1.0f / 3.0f + 1.0f :
			H + 1.0f / 3.0f > 1.0f ? H + 1.0f / 3.0f - 1.0f : H + 1.0f / 3.0f;
	
			float RR = RH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * RH : 
			RH < 1.0f / 2.0f ? Q : RH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - RH) * 6.0f : P;

			float GH = H < 0.0f ? H + 1.0f : H > 1.0f ? H - 1.0f : H;

			float GG = GH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * GH :
			GH < 1.0f / 2.0f ? Q : GH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - GH) * 6.0f : P;

			float BH = H - 1.0f / 3.0f < 0.0f ? H - 1.0f / 3.0f + 1.0f :
			H - 1.0f / 3.0f > 1.0f ? H - 1.0f / 3.0f - 1.0f : H - 1.0f / 3.0f;
	
		    float BB = BH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * BH :
			BH < 1.0f / 2.0f ? Q : BH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - BH) * 6.0f : P;
	
	
			float r1 = S == 0.0f ? L : RR;
			float g1 = S == 0.0f ? L : GG;
			float b1 = S == 0.0f ? L : BB;
    		
    		// Sat vs Luma
   		    
   		    float luma601 = 0.299f * r1 + 0.587f * g1 + 0.114f * b1;
   		    float meanluma = (r1 + g1 + b1) / 3.0f;
   		    float del_luma = luma601 / meanluma;
   		    
            float satluma = _luma[0] >= 1.0f ? meanluma + ((1.0f -_luma[0]) * (1.0f - meanluma)) : _luma[0] >= 0.0f ? (meanluma >= _luma[0] ? 
            1.0f : meanluma / _luma[0]) : _luma[0] < -1.0f ? (1.0f - meanluma) + (_luma[0] + 1.0f) * meanluma : meanluma <= (1.0f + _luma[0]) ? 1.0f : 
            (1.0f - meanluma) / (1.0f - (_luma[0] + 1.0f));
            float satlumaA = satluma <= 0.0f ? 0.0f : satluma >= 1.0f ? 1.0f : satluma; 
            float satlumaB = (_luma[1] > 1.0f ? 1.0f - pow(1.0f - S, _luma[1]) : pow(S, 1.0f / _luma[1])) * satlumaA * del_luma;
            float satalphaL = satlumaB > 1.0f ? 1.0f : satlumaB;
            
            float r = _switch[5] == 1.0f ? (r1 * _luma[2] * satalphaL) + (r1 * (1.0f - satalphaL)) : r1;
            float g = _switch[5] == 1.0f ? (g1 * _luma[2] * satalphaL) + (g1 * (1.0f - satalphaL)) : g1;
            float b = _switch[5] == 1.0f ? (b1 * _luma[2] * satalphaL) + (b1 * (1.0f - satalphaL)) : b1;

            // do we have a source image to scale up
            if (srcPix)
            {
                    dstPix[0] = _alpha[0] == 1.0f ? satAlpha : _alpha[1] == 1.0f ? satalphaL : r;
                    dstPix[1] = _alpha[0] == 1.0f ? satAlpha : _alpha[1] == 1.0f ? satalphaL : g;
                    dstPix[2] = _alpha[0] == 1.0f ? satAlpha : _alpha[1] == 1.0f ? satalphaL : b;
                    dstPix[3] = srcPix[3];
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

void HueConverge::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void HueConverge::setScales(float p_SwitchA, float p_SwitchB, float p_SwitchC, float p_SwitchD, float p_SwitchE, float p_SwitchL, 
float p_Alpha1, float p_Alpha2, float p_LogA, float p_LogB, float p_LogC, float p_LogD, float p_SatA, float p_SatB, float p_SatC, float p_HueA, float p_HueB, 
float p_HueC, float p_HueR1, float p_HueP1, float p_HueSH1, float p_HueSHP1, float p_HueR2, float p_HueP2, float p_HueSH2, 
float p_HueSHP2,float p_HueR3, float p_HueP3, float p_HueSH3, float p_HueSHP3, float p_LumaA, float p_LumaB, float p_LumaC)
{
    _switch[0] = p_SwitchA;
    _switch[1] = p_SwitchB;
    _switch[2] = p_SwitchC;
    _switch[3] = p_SwitchD;
    _switch[4] = p_SwitchE;
    _switch[5] = p_SwitchL;
    _alpha[0] = p_Alpha1;
    _alpha[1] = p_Alpha2;
    _log[0] = p_LogA;
    _log[1] = p_LogB;
    _log[2] = p_LogC;
    _log[3] = p_LogD;
    _sat[0] = p_SatA;
    _sat[1] = p_SatB;
    _sat[2] = p_SatC;
    _hue[0] = p_HueA;
    _hue[1] = p_HueB;
    _hue[2] = p_HueC;
    _hue[3] = p_HueR1;
    _hue[4] = p_HueP1;
    _hue[5] = p_HueSH1;
    _hue[6] = p_HueSHP1;
    _hue[7] = p_HueR2;
    _hue[8] = p_HueP2;
    _hue[9] = p_HueSH2;
    _hue[10] = p_HueSHP2;
    _hue[11] = p_HueR3;
    _hue[12] = p_HueP3;
    _hue[13] = p_HueSH3;
    _hue[14] = p_HueSHP3;
    _luma[0] = p_LumaA;
    _luma[1] = p_LumaB;
    _luma[2] = p_LumaC;
    
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class HueConvergePlugin : public OFX::ImageEffect
{
public:
    explicit HueConvergePlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    
    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(HueConverge &p_HueConverge, const OFX::RenderArguments& p_Args);
    

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::BooleanParam* m_SwitchA;
    OFX::BooleanParam* m_SwitchB;
    OFX::BooleanParam* m_SwitchC;
    OFX::BooleanParam* m_SwitchD;
    OFX::BooleanParam* m_SwitchE;
    OFX::BooleanParam* m_SwitchL;
    OFX::BooleanParam* _displayAlpha1;
    OFX::BooleanParam* _displayAlpha2;
    OFX::RGBParam *m_HueA;
    OFX::RGBParam *m_HueB;
    OFX::RGBParam *m_HueC;
    OFX::DoubleParam* m_Log1;
    OFX::DoubleParam* m_Log2;
    OFX::DoubleParam* m_Log3;
    OFX::DoubleParam* m_Log4;
    OFX::DoubleParam* m_Sat1;
    OFX::DoubleParam* m_Sat2;
    OFX::DoubleParam* m_Sat3;
    OFX::DoubleParam* m_HueR1;
    OFX::DoubleParam* m_HueP1;
    OFX::DoubleParam* m_HueSH1;
    OFX::DoubleParam* m_HueSHP1;
    OFX::DoubleParam* m_HueR2;
    OFX::DoubleParam* m_HueP2;
    OFX::DoubleParam* m_HueSH2;
    OFX::DoubleParam* m_HueSHP2;
    OFX::DoubleParam* m_HueR3;
    OFX::DoubleParam* m_HueP3;
    OFX::DoubleParam* m_HueSH3;
    OFX::DoubleParam* m_HueSHP3;
    OFX::DoubleParam* m_LumaA;
    OFX::DoubleParam* m_LumaB;
    OFX::DoubleParam* m_LumaC;
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;

};

HueConvergePlugin::HueConvergePlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

	m_SwitchA = fetchBooleanParam("switch1");
    m_SwitchB = fetchBooleanParam("switch2");
    m_SwitchC = fetchBooleanParam("hueSwitch1");
    m_SwitchD = fetchBooleanParam("hueSwitch2");
    m_SwitchE = fetchBooleanParam("hueSwitch3");
    m_SwitchL = fetchBooleanParam("lumaSwitch");
    _displayAlpha1 = fetchBooleanParam("displayAlpha1");
    _displayAlpha2 = fetchBooleanParam("displayAlpha2");
    m_HueA = fetchRGBParam("hueA");
    m_HueB = fetchRGBParam("hueB");
    m_HueC = fetchRGBParam("hueC");
    m_Log1 = fetchDoubleParam("peak1");
    m_Log2 = fetchDoubleParam("curvep1");
    m_Log3 = fetchDoubleParam("pivot1");
    m_Log4 = fetchDoubleParam("offset");
    m_Sat1 = fetchDoubleParam("satscale");
    m_Sat2 = fetchDoubleParam("satsoft");
    m_Sat3 = fetchDoubleParam("satsoftluma");
    m_HueR1 = fetchDoubleParam("range1");
    m_HueP1 = fetchDoubleParam("power1");
    m_HueSH1 = fetchDoubleParam("sathue1");
    m_HueSHP1 = fetchDoubleParam("sathuepow1");
    m_HueR2 = fetchDoubleParam("range2");
    m_HueP2 = fetchDoubleParam("power2");
    m_HueSH2 = fetchDoubleParam("sathue2");
    m_HueSHP2 = fetchDoubleParam("sathuepow2");
    m_HueR3 = fetchDoubleParam("range3");
    m_HueP3 = fetchDoubleParam("power3");
    m_HueSH3 = fetchDoubleParam("sathue3");
    m_HueSHP3 = fetchDoubleParam("sathuepow3");
    m_LumaA = fetchDoubleParam("lumaA");
    m_LumaB = fetchDoubleParam("lumaB");
    m_LumaC = fetchDoubleParam("lumaC");
    m_Path = fetchStringParam("path");
    m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");

}

void HueConvergePlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        HueConverge HueConverge(*this);
        setupAndProcess(HueConverge, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool HueConvergePlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    
    bool aSwitch = m_SwitchA->getValueAtTime(p_Args.time);
    bool bSwitch = m_SwitchB->getValueAtTime(p_Args.time);
    bool cSwitch = m_SwitchC->getValueAtTime(p_Args.time);
    bool dSwitch = m_SwitchD->getValueAtTime(p_Args.time);
    bool eSwitch = m_SwitchE->getValueAtTime(p_Args.time);
    bool lSwitch = m_SwitchL->getValueAtTime(p_Args.time);
    
    float hue5 = m_HueP1->getValueAtTime(p_Args.time);
    float hue7 = m_HueSHP1->getValueAtTime(p_Args.time);
    float hue9 = m_HueP2->getValueAtTime(p_Args.time);
    float hue11 = m_HueSHP2->getValueAtTime(p_Args.time);
    float hue13 = m_HueP3->getValueAtTime(p_Args.time);
    float hue15 = m_HueSHP3->getValueAtTime(p_Args.time);
    

        
    
    if (!aSwitch && !bSwitch && !cSwitch && !dSwitch && !eSwitch && !lSwitch && (hue5 == 1.0) && (hue7 == 1.0) && (hue9 == 1.0) && 
    (hue11 == 1.0) && (hue13 == 1.0) && (hue15 == 1.0))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void HueConvergePlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{

	if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if (p_ParamName == "displayAlpha1") {
	
        bool hideAlpha2 = _displayAlpha1->getValueAtTime(p_Args.time);
        _displayAlpha2->setIsSecretAndDisabled(hideAlpha2);
    }
    
    if (p_ParamName == "displayAlpha2") {
	
        bool hideAlpha1 = _displayAlpha2->getValueAtTime(p_Args.time);
        _displayAlpha1->setIsSecretAndDisabled(hideAlpha1);
    }
	
	if(p_ParamName == "button1")
    {
	
	RGBValues hueA;
    m_HueA->getValueAtTime(p_Args.time, hueA.r, hueA.g, hueA.b);
    
    RGBValues hueB;
    m_HueB->getValueAtTime(p_Args.time, hueB.r, hueB.g, hueB.b);
    
    RGBValues hueC;
    m_HueC->getValueAtTime(p_Args.time, hueC.r, hueC.g, hueC.b);
    
    float rHueA = hueA.r;
    float gHueA = hueA.g;
    float bHueA = hueA.b;
    float rHueB = hueB.r;
    float gHueB = hueB.g;
    float bHueB = hueB.b;
    float rHueC = hueC.r;
    float gHueC = hueC.g;
    float bHueC = hueC.b;
    
    float hue1 = RGBtoHUE(rHueA, gHueA, bHueA);
    float hue2 = RGBtoHUE(rHueB, gHueB, bHueB);
    float hue3 = RGBtoHUE(rHueC, gHueC, bHueC);
    
    bool aSwitch = m_SwitchA->getValueAtTime(p_Args.time);
    bool bSwitch = m_SwitchB->getValueAtTime(p_Args.time);
    bool cSwitch = m_SwitchC->getValueAtTime(p_Args.time);
    bool dSwitch = m_SwitchD->getValueAtTime(p_Args.time);
    bool eSwitch = m_SwitchE->getValueAtTime(p_Args.time);
    bool lSwitch = m_SwitchL->getValueAtTime(p_Args.time);
    
	int logswitch = aSwitch ? 1 : 0;
	int satswitch = bSwitch ? 1 : 0;
	int hueswitch1 = cSwitch ? 1 : 0;
	int hueswitch2 = dSwitch ? 1 : 0;
	int hueswitch3 = eSwitch ? 1 : 0;
	int lumaswitch = lSwitch ? 1 : 0;

    float log1 = m_Log1->getValueAtTime(p_Args.time);
    float log2 = m_Log2->getValueAtTime(p_Args.time);
    float log3 = m_Log3->getValueAtTime(p_Args.time);
    float log4 = m_Log4->getValueAtTime(p_Args.time);
    
    float sat1 = m_Sat1->getValueAtTime(p_Args.time);
    float sat2 = m_Sat2->getValueAtTime(p_Args.time);
    float sat3 = m_Sat3->getValueAtTime(p_Args.time);
    
    float hue4 = m_HueR1->getValueAtTime(p_Args.time) / 2.0;
    float hue5 = m_HueP1->getValueAtTime(p_Args.time);
    float hue6 = m_HueSH1->getValueAtTime(p_Args.time);
    float hue7 = m_HueSHP1->getValueAtTime(p_Args.time);

	float hue8 = m_HueR2->getValueAtTime(p_Args.time) / 2.0;
    float hue9 = m_HueP2->getValueAtTime(p_Args.time);
    float hue10 = m_HueSH2->getValueAtTime(p_Args.time);
    float hue11 = m_HueSHP2->getValueAtTime(p_Args.time);

	float hue12 = m_HueR3->getValueAtTime(p_Args.time) / 2.0;
    float hue13 = m_HueP3->getValueAtTime(p_Args.time);
    float hue14 = m_HueSH3->getValueAtTime(p_Args.time);
    float hue15 = m_HueSHP3->getValueAtTime(p_Args.time);
    
    float luma1 = m_LumaA->getValueAtTime(p_Args.time);
    float luma2 = m_LumaB->getValueAtTime(p_Args.time);
    float luma3 = m_LumaC->getValueAtTime(p_Args.time);

	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// HueConvergePlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"	// switches for logistic, saturation, and hues\n" \
	"	int logswitch = %d;\n" \
	"	bool p_SwitchA = logswitch == 1;\n" \
	"	int satswitch = %d;\n" \
	"	bool p_SwitchB = satswitch == 1;\n" \
	"	int hueswitch1 = %d;\n" \
	"	bool p_SwitchC = hueswitch1 == 1;\n" \
	"	int hueswitch2 = %d;\n" \
	"	bool p_SwitchD = hueswitch2 == 1;\n" \
	"	int hueswitch3 = %d;\n" \
	"	bool p_SwitchE = hueswitch3 == 1;\n" \
	"	int lumaswitch = %d;\n" \
	"	bool p_SwitchL = lumaswitch == 1;\n" \
	"\n" \
	"	// logistic parameters peak, curve, pivot, offset\n" \
	"	float p_LogA = %ff;\n" \
	"	float p_LogB = %ff;\n" \
	"	float p_LogC = %ff;\n" \
	"	float p_LogD = %ff;\n" \
	"\n" \
	"	// saturation parameters sat, soft-clip, blend\n" \
	"	float p_SatA = %ff;\n" \
	"	float p_SatB = %ff;\n" \
	"	float p_SatC = %ff;\n" \
	"\n" \
	"	// hue anchor values\n" \
	"	float p_HueA = %ff;\n" \
	"	float p_HueB = %ff;\n" \
	"	float p_HueC = %ff;\n" \
	"\n" \
	"	// hue converge parameters range, power, sat influence, power\n" \
	"	float p_HueR1 = %ff;\n" \
	"	float p_HueP1 = %ff;\n" \
	"	float p_HueSH1 = %ff;\n" \
	"	float p_HueSHP1 = %ff;\n" \
	"\n" \
	"	float p_HueR2 = %ff;\n" \
	"	float p_HueP2 = %ff;\n" \
	"	float p_HueSH2 = %ff;\n" \
	"	float p_HueSHP2 = %ff;\n" \
	"\n" \
	"	float p_HueR3 = %ff;\n" \
	"	float p_HueP3 = %ff;\n" \
	"	float p_HueSH3 = %ff;\n" \
	"	float p_HueSHP3 = %ff;\n" \
	"	\n" \
	"	// sat v luma parameters alpha, power, scale\n" \
	"	float p_LumaA = %ff;\n" \
	"	float p_LumaB = %ff;\n" \
	"	float p_LumaC = %ff;\n" \
	"\n" \
	"	// Euler's Constant e = 2.718281828459045\n" \
	"	float e = 2.718281828459045;\n" \
	"\n" \
	"	// Default expression : 1.0f / (1.0f + _powf(e, -8.9f*(r - 0.435f)))\n" \
	"	// Logistic Function (Sigmoid Curve)\n" \
	"\n" \
	"	float Lr = p_SwitchA ? p_LogA / (1.0f + _powf(e, (-8.9f * p_LogB) * (p_R - p_LogC))) + p_LogD : p_R;\n" \
	"	float Lg = p_SwitchA ? p_LogA / (1.0f + _powf(e, (-8.9f * p_LogB) * (p_G - p_LogC))) + p_LogD : p_G;\n" \
	"	float Lb = p_SwitchA ? p_LogA / (1.0f + _powf(e, (-8.9f * p_LogB) * (p_B - p_LogC))) + p_LogD : p_B; \n" \
	"\n" \
	"	float mn = _fminf(Lr, _fminf(Lg, Lb));    \n" \
	"	float Mx = _fmaxf(Lr, _fmaxf(Lg, Lb));    \n" \
	"	float del_Mx = Mx - mn;\n" \
	"\n" \
	"	float L = (Mx + mn) / 2.0;\n" \
	"	float luma = p_R * 0.2126f + p_G * 0.7152f + p_B * 0.0722f;\n" \
	"	float Sat = del_Mx == 0.0f ? 0.0f : L < 0.5f ? del_Mx / (Mx + mn) : del_Mx / (2.0f - Mx - mn);\n" \
	"\n" \
	"	float del_R = ((Mx - Lr) / 6.0f + del_Mx / 2.0f) / del_Mx;\n" \
	"	float del_G = ((Mx - Lg) / 6.0f + del_Mx / 2.0f) / del_Mx;\n" \
	"	float del_B = ((Mx - Lb) / 6.0f + del_Mx / 2.0f) / del_Mx;\n" \
	"\n" \
	"	float h = del_Mx == 0.0f ? 0.0f : Lr == Mx ? del_B - del_G : Lg == Mx ? 1.0f / 3.0f + \n" \
	"	del_R - del_B : 2.0f / 3.0f + del_G - del_R;\n" \
	"\n" \
	"	float Hh = h < 0.0f ? h + 1.0f : h > 1.0f ? h - 1.0f : h;\n" \
	"\n" \
	"	// Soft Clip Saturation\n" \
	"\n" \
	"	float s = Sat * p_SatA > 1.0f ? 1.0f : Sat * p_SatA;\n" \
	"	float ss = s > p_SatB ? (-1.0f / ((s - p_SatB) / (1.0f - p_SatB) + 1.0f) + 1.0f) * (1.0f - p_SatB) + p_SatB : s;\n" \
	"	float satAlphaA = p_SatC > 1.0f ? luma + (1.0f - p_SatC) * (1.0f - luma) : p_SatC >= 0.0f ? (luma >= p_SatC ? 1.0f : \n" \
	"	luma / p_SatC) : p_SatC < -1.0f ? (1.0f - luma) + (p_SatC + 1.0f) * luma : luma <= (1.0f + p_SatC) ? 1.0f : \n" \
	"	(1.0f - luma) / (1.0f - (p_SatC + 1.0f));\n" \
	"	float satAlpha = satAlphaA > 1.0f ? 1.0f : satAlphaA;\n" \
	"\n" \
	"	float S = p_SwitchB ? ss * satAlpha + s * (1.0f - satAlpha) : Sat;\n" \
	"\n" \
	"	// Hue Convergence\n" \
	"\n" \
	"	float h1 = Hh - (p_HueA - 0.5f) < 0.0 ? Hh - (p_HueA - 0.5f) + 1.0f : Hh - (p_HueA - 0.5f) >\n" \
	"	1.0f ? Hh - (p_HueA - 0.5f) - 1.0f : Hh - (p_HueA - 0.5f);\n" \
	"\n" \
	"	float Hs1 = p_HueSHP1 >= 1.0f ? 1.0f - _powf(1.0f - S, p_HueSHP1) : _powf(S, 1.0f/p_HueSHP1);\n" \
	"	float cv1 = 1.0f + (p_HueP1 - 1.0f) * Hs1;\n" \
	"	float curve1 = p_HueP1 + (cv1 - p_HueP1) * p_HueSH1;\n" \
	"\n" \
	"	float H1 = !p_SwitchC ? Hh : h1 > 0.5f - p_HueR1 && h1 < 0.5f ? (1.0f - _powf(1.0f - (h1 - (0.5f - p_HueR1)) * \n" \
	"	(1.0f/p_HueR1), curve1)) * p_HueR1 + (0.5f - p_HueR1) + (p_HueA - 0.5f) : h1 > 0.5f && h1 < 0.5f + \n" \
	"	p_HueR1 ? _powf((h1 - 0.5f) * (1.0f/p_HueR1), curve1) * p_HueR1 + 0.5f + (p_HueA - 0.5f) : Hh;\n" \
	"\n" \
	"	float h2 = H1 - (p_HueB - 0.5f) < 0.0f ? H1 - (p_HueB - 0.5f) + 1.0f : H1 - (p_HueB - 0.5f) > \n" \
	"	1.0f ? H1 - (p_HueB - 0.5f) - 1.0f : H1 - (p_HueB - 0.5f);\n" \
	"\n" \
	"	float Hs2 = p_HueSHP2 >= 1.0f ? 1.0f - _powf(1.0f - S, p_HueSHP2) : _powf(S, 1.0f/p_HueSHP2);\n" \
	"	float cv2 = 1.0f + (p_HueP2 - 1.0f) * Hs2;\n" \
	"	float curve2 = p_HueP2 + (cv2 - p_HueP2) * p_HueSH2;\n" \
	"\n" \
	"	float H2 = !p_SwitchD ? H1 : h2 > 0.5f - p_HueR2 && h2 < 0.5f ? (1.0f - _powf(1.0f - (h2 - (0.5f - p_HueR2)) * \n" \
	"	(1.0f/p_HueR2), curve2)) * p_HueR2 + (0.5f - p_HueR2) + (p_HueB - 0.5f) : h2 > 0.5f && h2 < 0.5f + \n" \
	"	p_HueR2 ? _powf((h2 - 0.5f) * (1.0f/p_HueR2), curve2) * p_HueR2 + 0.5f + (p_HueB - 0.5f) : H1;\n" \
	"\n" \
	"	float h3 = H2 - (p_HueC - 0.5f) < 0.0f ? H2 - (p_HueC - 0.5f) + 1.0f : H2 - (p_HueC - 0.5f) > \n" \
	"	1.0f ? H2 - (p_HueC - 0.5f) - 1.0f : H2 - (p_HueC - 0.5f);\n" \
	"\n" \
	"	float Hs3 = p_HueSHP3 >= 1.0f ? 1.0f - _powf(1.0f - S, p_HueSHP3) : _powf(S, 1.0f/p_HueSHP3);\n" \
	"	float cv3 = 1.0f + (p_HueP3 - 1.0f) * Hs3;\n" \
	"	float curve3 = p_HueP3 + (cv3 - p_HueP3) * p_HueSH3;\n" \
	"\n" \
	"	float H = !p_SwitchE ? H2 : h3 > 0.5f - p_HueR3 && h3 < 0.5f ? (1.0f - _powf(1.0f - (h3 - (0.5f - p_HueR3)) * \n" \
	"	(1.0f/p_HueR3), curve3)) * p_HueR3 + (0.5f - p_HueR3) + (p_HueC - 0.5f) : h3 > 0.5f && h3 < 0.5f + \n" \
	"	p_HueR3 ? _powf((h3 - 0.5f) * (1.0f/p_HueR3), curve3) * p_HueR3 + 0.5f + (p_HueC - 0.5f) : H2;\n" \
	"\n" \
	"	// HSL to RGB\n" \
	"\n" \
	"	float Q = L < 0.5f ? L * (1.0f + S) : L + S - L * S;\n" \
	"	float P = 2.0f * L - Q;\n" \
	"\n" \
	"	float RH = H + 1.0f / 3.0f < 0.0f ? H + 1.0f / 3.0f + 1.0f :\n" \
	"	H + 1.0f / 3.0f > 1.0f ? H + 1.0f / 3.0f - 1.0f : H + 1.0f / 3.0f;\n" \
	"\n" \
	"	float RR = RH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * RH : \n" \
	"	RH < 1.0f / 2.0f ? Q : RH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - RH) * 6.0f : P;\n" \
	"\n" \
	"	float GH = H < 0.0f ? H + 1.0f :\n" \
	"	H > 1.0f ? H - 1.0f : H;\n" \
	"\n" \
	"	float GG = GH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * GH :\n" \
	"	GH < 1.0f / 2.0f ? Q : GH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - GH) * 6.0f : P;\n" \
	"\n" \
	"	float BH = H - 1.0f / 3.0f < 0.0f ? H - 1.0f / 3.0f + 1.0f :\n" \
	"	H - 1.0f / 3.0f > 1.0f ? H - 1.0f / 3.0f - 1.0f : H - 1.0f / 3.0f;\n" \
	"\n" \
	"	float BB = BH < 1.0f / 6.0f ? P + (Q - P) * 6.0f * BH :\n" \
	"	BH < 1.0f / 2.0f ? Q : BH < 2.0f / 3.0f ? P + (Q - P) * (2.0f / 3.0f - BH) * 6.0f : P;  \n" \
	"\n" \
	"	float r1 = S == 0.0f ? L : RR;\n" \
	"	float g1 = S == 0.0f ? L : GG;\n" \
	"	float b1 = S == 0.0f ? L : BB;\n" \
	"	\n" \
	"	// Sat vs Luma\n" \
	"\n" \
	"	float luma601 = 0.299f * r1 + 0.587f * g1 + 0.114f * b1;\n" \
	"	float meanluma = (r1 + g1 + b1) / 3.0f;\n" \
	"	float del_luma = luma601 / meanluma;\n" \
	"\n" \
	"	float satluma = p_LumaA >= 1.0f ? meanluma + ((1.0f -p_LumaA) * (1.0f - meanluma)) : p_LumaA >= 0.0f ? (meanluma >= p_LumaA ? \n" \
	"	1.0f : meanluma / p_LumaA) : p_LumaA < -1.0f ? (1.0f - meanluma) + (p_LumaA + 1.0f) * meanluma : meanluma <= (1.0f + p_LumaA) ? 1.0f : \n" \
	"	(1.0f - meanluma) / (1.0f - (p_LumaA + 1.0f));\n" \
	"	float satlumaA = satluma <= 0.0f ? 0.0f : satluma >= 1.0f ? 1.0f : satluma; \n" \
	"	float satlumaB = (p_LumaB > 1.0f ? 1.0f - powf(1.0f - S, p_LumaB) : powf(S, 1.0f / p_LumaB)) * satlumaA * del_luma;\n" \
	"	float satalphaL = satlumaB > 1.0f ? 1.0f : satlumaB;\n" \
	"\n" \
	"	float r = p_SwitchL ? (r1 * p_LumaC * satalphaL) + (r1 * (1.0f - satalphaL)) : r1;\n" \
	"	float g = p_SwitchL ? (g1 * p_LumaC * satalphaL) + (g1 * (1.0f - satalphaL)) : g1;\n" \
	"	float b = p_SwitchL ? (b1 * p_LumaC * satalphaL) + (b1 * (1.0f - satalphaL)) : b1;\n" \
	"\n" \
	"\n" \
	"	return make_float3(r, g, b);\n" \
	"}\n", logswitch, satswitch, hueswitch1, hueswitch2, hueswitch3, lumaswitch, log1, log2, log3, log4, sat1, sat2, sat3, hue1, hue2, hue3, 
	hue4, hue5, hue6, hue7, hue8, hue9, hue10, hue11, hue12, hue13, hue14, hue15, luma1, luma2, luma3);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}

	if(p_ParamName == "button2")
    {
    
    RGBValues hueA;
    m_HueA->getValueAtTime(p_Args.time, hueA.r, hueA.g, hueA.b);
    
    RGBValues hueB;
    m_HueB->getValueAtTime(p_Args.time, hueB.r, hueB.g, hueB.b);
    
    RGBValues hueC;
    m_HueC->getValueAtTime(p_Args.time, hueC.r, hueC.g, hueC.b);
    
    float rHueA = hueA.r;
    float gHueA = hueA.g;
    float bHueA = hueA.b;
    float rHueB = hueB.r;
    float gHueB = hueB.g;
    float bHueB = hueB.b;
    float rHueC = hueC.r;
    float gHueC = hueC.g;
    float bHueC = hueC.b;
    
    float hue1 = RGBtoHUE(rHueA, gHueA, bHueA);
    float hue2 = RGBtoHUE(rHueB, gHueB, bHueB);
    float hue3 = RGBtoHUE(rHueC, gHueC, bHueC);
    
    bool aSwitch = m_SwitchA->getValueAtTime(p_Args.time);
    bool bSwitch = m_SwitchB->getValueAtTime(p_Args.time);
    bool cSwitch = m_SwitchC->getValueAtTime(p_Args.time);
    bool dSwitch = m_SwitchD->getValueAtTime(p_Args.time);
    bool eSwitch = m_SwitchE->getValueAtTime(p_Args.time);
    bool lSwitch = m_SwitchL->getValueAtTime(p_Args.time);
    
	int logswitch = aSwitch ? 1 : 0;
	int satswitch = bSwitch ? 1 : 0;
	int hueswitch1 = cSwitch ? 1 : 0;
	int hueswitch2 = dSwitch ? 1 : 0;
	int hueswitch3 = eSwitch ? 1 : 0;
	int lumaswitch = lSwitch ? 1 : 0;

    float log1 = m_Log1->getValueAtTime(p_Args.time);
    float log2 = m_Log2->getValueAtTime(p_Args.time);
    float log3 = m_Log3->getValueAtTime(p_Args.time);
    float log4 = m_Log4->getValueAtTime(p_Args.time);
    
    float sat1 = m_Sat1->getValueAtTime(p_Args.time);
    float sat2 = m_Sat2->getValueAtTime(p_Args.time);
    float sat3 = m_Sat3->getValueAtTime(p_Args.time);
    
    float hue4 = m_HueR1->getValueAtTime(p_Args.time) / 2.0;
    float hue5 = m_HueP1->getValueAtTime(p_Args.time);
    float hue6 = m_HueSH1->getValueAtTime(p_Args.time);
    float hue7 = m_HueSHP1->getValueAtTime(p_Args.time);

	float hue8 = m_HueR2->getValueAtTime(p_Args.time) / 2.0;
    float hue9 = m_HueP2->getValueAtTime(p_Args.time);
    float hue10 = m_HueSH2->getValueAtTime(p_Args.time);
    float hue11 = m_HueSHP2->getValueAtTime(p_Args.time);

	float hue12 = m_HueR3->getValueAtTime(p_Args.time) / 2.0;
    float hue13 = m_HueP3->getValueAtTime(p_Args.time);
    float hue14 = m_HueSH3->getValueAtTime(p_Args.time);
    float hue15 = m_HueSHP3->getValueAtTime(p_Args.time);
    
    float luma1 = m_LumaA->getValueAtTime(p_Args.time);
    float luma2 = m_LumaB->getValueAtTime(p_Args.time);
    float luma3 = m_LumaC->getValueAtTime(p_Args.time);

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
	"name HueConverge\n" \
	"selected true\n" \
	"}\n" \
	"Input {\n" \
	"inputs 0\n" \
	"name Input1\n" \
	"xpos -412\n" \
	"ypos -315\n" \
	"}\n" \
	"set N2bcccce0 [stack 0]\n" \
	"Dot {\n" \
	"name Dot1\n" \
	"xpos -494\n" \
	"ypos -238\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 luma\n" \
	"temp_expr0 \"r * 0.2126 + g * 0.7152 + b * 0.0722\"\n" \
	"temp_name1 SatC\n" \
	"temp_expr1 %f\n" \
	"expr0 0\n" \
	"expr1 0\n" \
	"expr2 \"min(SatC > 1.0 ? luma + (1.0 - SatC) * (1.0 - luma) : SatC >= 0.0 ? (luma >= SatC ? 1.0 : luma / SatC) : SatC < -1.0 ? (1.0 - luma) + (SatC + 1.0) * luma : luma <= (1.0 + SatC) ? 1.0f : (1.0 - luma) / (1.0 - (SatC + 1.0)), 1.0)\"\n" \
	"name SatAlpha\n" \
	"xpos -528\n" \
	"ypos -59\n" \
	"}\n" \
	"push $N2bcccce0\n" \
	"Expression {\n" \
	"temp_name0 LogA\n" \
	"temp_expr0 %f\n" \
	"temp_name1 LogB\n" \
	"temp_expr1 %f\n" \
	"temp_name2 LogC\n" \
	"temp_expr2 %f\n" \
	"temp_name3 LogD\n" \
	"temp_expr3 %f\n" \
	"expr0 \"LogA / (1.0 + pow(2.718281828459045, (-8.9 * LogB) * (r - LogC))) + LogD\"\n" \
	"expr1 \"LogA / (1.0 + pow(2.718281828459045, (-8.9 * LogB) * (g - LogC))) + LogD\"\n" \
	"expr2 \"LogA / (1.0 + pow(2.718281828459045, (-8.9 * LogB) * (b - LogC))) + LogD\"\n" \
	"name Logistic_Function\n" \
	"xpos -412\n" \
	"ypos -183\n" \
	"}\n" \
	"push $N2bcccce0\n" \
	"Switch {\n" \
	"inputs 2\n" \
	"which %d\n" \
	"name Logistic_Function_Switch\n" \
	"xpos -316\n" \
	"ypos -144\n" \
	"}\n" \
	"Colorspace {\n" \
	"colorspace_in sRGB\n" \
	"colorspace_out HSL\n" \
	"name RGB_to_HSL\n" \
	"xpos -316\n" \
	"ypos -108\n" \
	"}\n" \
	"set N2afd14c0 [stack 0]\n" \
	"Expression {\n" \
	"temp_name0 SatA\n" \
	"temp_expr0 %f\n" \
	"expr0 0\n" \
	"expr1 \"g * SatA > 1.0 ? 1.0 : g * SatA\"\n" \
	"expr2 0\n" \
	"name s\n" \
	"xpos -454\n" \
	"ypos -143\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 SatB\n" \
	"temp_expr0 %f\n" \
	"expr0 g\n" \
	"expr1 \"g > SatB ? (-1.0 / ((g - SatB) / (1.0 - SatB) + 1.0) + 1.0) * (1.0 - SatB) + SatB : g\"\n" \
	"expr2 0\n" \
	"name ss\n" \
	"xpos -454\n" \
	"ypos -97\n" \
	"}\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"in rgb\n" \
	"in2 rgb\n" \
	"blue blue\n" \
	"out rgb\n" \
	"name ShuffleCopy1\n" \
	"xpos -454\n" \
	"ypos -20\n" \
	"}\n" \
	"Expression {\n" \
	"expr1 \"g * b + r * (1.0 - b)\"\n" \
	"name Sat_softclip\n" \
	"xpos -454\n" \
	"ypos 20\n" \
	"}\n" \
	"push $N2afd14c0\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"in rgb\n" \
	"in2 rgb\n" \
	"green green\n" \
	"out rgb\n" \
	"name H_Sat_L\n" \
	"xpos -377\n" \
	"ypos 59\n" \
	"}\n" \
	"push $N2afd14c0\n" \
	"Switch {\n" \
	"inputs 2\n" \
	"which %d\n" \
	"name Softclip_switch\n" \
	"xpos -316\n" \
	"ypos 104\n" \
	"}\n" \
	"set N2af608b0 [stack 0]\n" \
	"Expression {\n" \
	"temp_name0 HueA\n" \
	"temp_expr0 %f\n" \
	"temp_name1 HueSHP1\n" \
	"temp_expr1 %f\n" \
	"expr0 \"r - (HueA - 0.5) < 0.0 ? r - (HueA - 0.5) + 1.0 : r - (HueA - 0.5) >\n" \
	"1.0 ? r - (HueA - 0.5) - 1.0 : r - (HueA - 0.5)\"\n" \
	"expr1 \"HueSHP1 >= 1.0 ? 1.0 - pow(1.0 - g, HueSHP1) : pow(g, 1.0/HueSHP1)\"\n" \
	"name h1_Hs1\n" \
	"xpos -522\n" \
	"ypos 87\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 HueP1\n" \
	"temp_expr0 %f\n" \
	"temp_name1 cv1\n" \
	"temp_expr1 \"1.0 + (HueP1 - 1.0) * g\"\n" \
	"temp_name2 HueSH1\n" \
	"temp_expr2 %f\n" \
	"expr2 \"HueP1 + (cv1 - HueP1) * HueSH1\"\n" \
	"name h1_Hs1_curve1\n" \
	"xpos -522\n" \
	"ypos 120\n" \
	"}\n" \
	"push $N2af608b0\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"in rgb\n" \
	"in2 rgb\n" \
	"red red\n" \
	"green red2\n" \
	"blue blue\n" \
	"out rgb\n" \
	"name h1_H_curve1\n" \
	"xpos -522\n" \
	"ypos 158\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 HueA\n" \
	"temp_expr0 %f\n" \
	"temp_name1 HueR1\n" \
	"temp_expr1 %f\n" \
	"expr0 \"r > 0.5 - HueR1 && r < 0.5 ? (1.0 - pow(1.0 - (r - (0.5 - HueR1)) * (1.0/HueR1), b)) * HueR1 + (0.5 - HueR1) + (HueA - 0.5) : r > 0.5 && r < 0.5 + HueR1 ? pow((r - 0.5) * (1.0/HueR1), b) * HueR1 + 0.5 + (HueA - 0.5) : g\"\n" \
	"expr1 0\n" \
	"expr2 0\n" \
	"name H1\n" \
	"xpos -522\n" \
	"ypos 195\n" \
	"}\n" \
	"push $N2af608b0\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"in rgb\n" \
	"in2 rgb\n" \
	"red red\n" \
	"out rgb\n" \
	"name H1_S_L\n" \
	"xpos -467\n" \
	"ypos 228\n" \
	"}\n" \
	"push $N2af608b0\n" \
	"Switch {\n" \
	"inputs 2\n" \
	"which %d\n" \
	"name Hue_switch1\n" \
	"xpos -316\n" \
	"ypos 268\n" \
	"}\n" \
	"set N32015350 [stack 0]\n" \
	"Expression {\n" \
	"temp_name0 HueB\n" \
	"temp_expr0 %f\n" \
	"temp_name1 HueSHP2\n" \
	"temp_expr1 %f\n" \
	"expr0 \"r - (HueB - 0.5) < 0.0 ? r - (HueB - 0.5) + 1.0 : r - (HueB - 0.5) >\n" \
	"1.0 ? r - (HueB - 0.5) - 1.0 : r - (HueB - 0.5)\"\n" \
	"expr1 \"HueSHP2 >= 1.0 ? 1.0 - pow(1.0 - g, HueSHP2) : pow(g, 1.0/HueSHP2)\"\n" \
	"name h2_Hs2\n" \
	"xpos -498\n" \
	"ypos 303\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 HueP2\n" \
	"temp_expr0 %f\n" \
	"temp_name1 cv2\n" \
	"temp_expr1 \"1.0 + (HueP2 - 1.0) * g\"\n" \
	"temp_name2 HueSH2\n" \
	"temp_expr2 %f\n" \
	"expr2 \"HueP2 + (cv2 - HueP2) * HueSH2\"\n" \
	"name h2_Hs2_curve2\n" \
	"xpos -498\n" \
	"ypos 336\n" \
	"}\n" \
	"push $N32015350\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"in rgb\n" \
	"in2 rgb\n" \
	"red red\n" \
	"green red2\n" \
	"blue blue\n" \
	"out rgb\n" \
	"name h2_H_curve2\n" \
	"xpos -498\n" \
	"ypos 374\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 HueB\n" \
	"temp_expr0 %f\n" \
	"temp_name1 HueR2\n" \
	"temp_expr1 %f\n" \
	"expr0 \"r > 0.5 - HueR2 && r < 0.5 ? (1.0 - pow(1.0 - (r - (0.5 - HueR2)) * (1.0/HueR2), b)) * HueR2 + (0.5 - HueR2) + (HueB - 0.5) : r > 0.5 && r < 0.5 + HueR2 ? pow((r - 0.5) * (1.0/HueR2), b) * HueR2 + 0.5 + (HueB - 0.5) : g\"\n" \
	"expr1 0\n" \
	"expr2 0\n" \
	"name H2\n" \
	"xpos -498\n" \
	"ypos 411\n" \
	"}\n" \
	"push $N32015350\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"in rgb\n" \
	"in2 rgb\n" \
	"red red\n" \
	"out rgb\n" \
	"name H2_S_L\n" \
	"xpos -443\n" \
	"ypos 444\n" \
	"}\n" \
	"push $N32015350\n" \
	"Switch {\n" \
	"inputs 2\n" \
	"which %d\n" \
	"name Hue_switch2\n" \
	"xpos -316\n" \
	"ypos 495\n" \
	"}\n" \
	"set N2ae0ed80 [stack 0]\n" \
	"Expression {\n" \
	"temp_name0 HueC\n" \
	"temp_expr0 %f\n" \
	"temp_name1 HueSHP3\n" \
	"temp_expr1 %f\n" \
	"expr0 \"r - (HueC - 0.5) < 0.0 ? r - (HueC - 0.5) + 1.0 : r - (HueC - 0.5) >\n" \
	"1.0 ? r - (HueC - 0.5) - 1.0 : r - (HueC - 0.5)\"\n" \
	"expr1 \"HueSHP3 >= 1.0 ? 1.0 - pow(1.0 - g, HueSHP3) : pow(g, 1.0/HueSHP3)\"\n" \
	"name h3_Hs3\n" \
	"xpos -423\n" \
	"ypos 550\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 HueP3\n" \
	"temp_expr0 %f\n" \
	"temp_name1 cv3\n" \
	"temp_expr1 \"1.0 + (HueP3 - 1.0) * g\"\n" \
	"temp_name2 HueSH3\n" \
	"temp_expr2 %f\n" \
	"expr2 \"HueP3 + (cv3 - HueP3) * HueSH3\"\n" \
	"name h3_Hs3_curve3\n" \
	"xpos -423\n" \
	"ypos 583\n" \
	"}\n" \
	"push $N2ae0ed80\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"in rgb\n" \
	"in2 rgb\n" \
	"red red\n" \
	"green red2\n" \
	"blue blue\n" \
	"out rgb\n" \
	"name h3_H_curve3\n" \
	"xpos -423\n" \
	"ypos 621\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 HueC\n" \
	"temp_expr0 %f\n" \
	"temp_name1 HueR3\n" \
	"temp_expr1 %f\n" \
	"expr0 \"r > 0.5 - HueR3 && r < 0.5 ? (1.0 - pow(1.0 - (r - (0.5 - HueR3)) * (1.0/HueR3), b)) * HueR3 + (0.5 - HueR3) + (HueC - 0.5) : r > 0.5 && r < 0.5 + HueR3 ? pow((r - 0.5) * (1.0/HueR3), b) * HueR3 + 0.5 + (HueC - 0.5) : g\"\n" \
	"expr1 0\n" \
	"expr2 0\n" \
	"name H3\n" \
	"xpos -423\n" \
	"ypos 658\n" \
	"}\n" \
	"push $N2ae0ed80\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"in rgb\n" \
	"in2 rgb\n" \
	"red red\n" \
	"out rgb\n" \
	"name H3_S_L\n" \
	"xpos -368\n" \
	"ypos 689\n" \
	"}\n" \
	"push $N2ae0ed80\n" \
	"Switch {\n" \
	"inputs 2\n" \
	"which %d\n" \
	"name Hue_switch3\n" \
	"xpos -313\n" \
	"ypos 741\n" \
	"}\n" \
	"set N2aff7560 [stack 0]\n" \
	"Colorspace {\n" \
	"colorspace_in HSL\n" \
	"colorspace_out sRGB\n" \
	"name HSL_to_RGB\n" \
	"xpos -474\n" \
	"ypos 741\n" \
	"}\n" \
	"set N2afe7830 [stack 0]\n" \
	"Expression {\n" \
	"temp_name0 luma601\n" \
	"temp_expr0 \"r * 0.299 + g * 0.587 + b * 0.114\"\n" \
	"temp_name1 meanluma\n" \
	"temp_expr1 \"(r + g + b) / 3.0\"\n" \
	"temp_name2 del_luma\n" \
	"temp_expr2 \"luma601 / meanluma\"\n" \
	"temp_name3 LumaA\n" \
	"temp_expr3 %f\n" \
	"expr0 \"LumaA >= 1.0 ? meanluma + ((1.0 - LumaA) * (1.0 - meanluma)) : LumaA >= 0.0 ? (meanluma >= LumaA ? 1.0 : meanluma / LumaA) : LumaA < -1.0 ? (1.0 - meanluma) + (LumaA + 1.0) * meanluma : meanluma <= (1.0 + LumaA) ? 1.0 : (1.0 - meanluma) / (1.0 - (LumaA + 1.0))\"\n" \
	"expr1 del_luma\n" \
	"name satluma\n" \
	"xpos -396\n" \
	"ypos 787\n" \
	"}\n" \
	"Expression {\n" \
	"expr0 \"r <= 0.0 ? 0.0 : r >= 1.0 ? 1.0 : r\"\n" \
	"expr1 g\n" \
	"expr2 0\n" \
	"name satlumaA\n" \
	"xpos -396\n" \
	"ypos 830\n" \
	"}\n" \
	"push $N2aff7560\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"red red\n" \
	"green green\n" \
	"blue green2\n" \
	"name satlumaA_del_luma_S\n" \
	"xpos -313\n" \
	"ypos 853\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 LumaB\n" \
	"temp_expr0 %f\n" \
	"expr0 \"min((LumaB > 1.0 ? 1.0 - pow(1.0 - b, LumaB) : pow(b, 1.0 / LumaB)) * r * g, 1.0)\"\n" \
	"name satAlphaL\n" \
	"xpos -313\n" \
	"ypos 894\n" \
	"}\n" \
	"push $N2afe7830\n" \
	"ShuffleCopy {\n" \
	"inputs 2\n" \
	"alpha red\n" \
	"name RGB_satalpha\n" \
	"xpos -313\n" \
	"ypos 948\n" \
	"}\n" \
	"Expression {\n" \
	"temp_name0 LumaC\n" \
	"temp_expr0 %f\n" \
	"expr0 \"(r * LumaC * a) + (r * (1.0 - a))\"\n" \
	"expr1 \"(g * LumaC * a) + (g * (1.0 - a))\"\n" \
	"expr2 \"(b * LumaC * a) + (b * (1.0 - a))\"\n" \
	"name SatLuma\n" \
	"xpos -313\n" \
	"ypos 999\n" \
	"}\n" \
	"push $N2afe7830\n" \
	"Switch {\n" \
	"inputs 2\n" \
	"which %d\n" \
	"name RGB\n" \
	"xpos -474\n" \
	"ypos 1028\n" \
	"}\n" \
	"Output {\n" \
	"name Output1\n" \
	"xpos -474\n" \
	"ypos 1128\n" \
	"}\n" \
	"end_group\n", sat3, log1, log2, log3, log4, logswitch, sat1, sat2, satswitch, 
	hue1, hue7, hue5, hue6, hue1, hue4, hueswitch1, hue2, hue11, hue9, hue10, hue2, 
	hue8, hueswitch2, hue3, hue15, hue14, hue14, hue3, hue12, hueswitch3,
	luma1, luma2, luma3, lumaswitch);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
	}	
	}
	}

}	
	
void HueConvergePlugin::setupAndProcess(HueConverge& p_HueConverge, const OFX::RenderArguments& p_Args)
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

	RGBValues hueA;
    m_HueA->getValueAtTime(p_Args.time, hueA.r, hueA.g, hueA.b);
    
    RGBValues hueB;
    m_HueB->getValueAtTime(p_Args.time, hueB.r, hueB.g, hueB.b);
    
    RGBValues hueC;
    m_HueC->getValueAtTime(p_Args.time, hueC.r, hueC.g, hueC.b);
    
    float rHueA = hueA.r;
    float gHueA = hueA.g;
    float bHueA = hueA.b;
    float rHueB = hueB.r;
    float gHueB = hueB.g;
    float bHueB = hueB.b;
    float rHueC = hueC.r;
    float gHueC = hueC.g;
    float bHueC = hueC.b;
    
    bool aSwitch = m_SwitchA->getValueAtTime(p_Args.time);
    bool bSwitch = m_SwitchB->getValueAtTime(p_Args.time);
    bool cSwitch = m_SwitchC->getValueAtTime(p_Args.time);
    bool dSwitch = m_SwitchD->getValueAtTime(p_Args.time);
    bool eSwitch = m_SwitchE->getValueAtTime(p_Args.time);
    bool lSwitch = m_SwitchL->getValueAtTime(p_Args.time);
    bool alpha1 = _displayAlpha1->getValueAtTime(p_Args.time);
    bool alpha2 = _displayAlpha2->getValueAtTime(p_Args.time);
    
    
	float aSwitchF = (aSwitch) ? 1.0f : 0.0f;
	float bSwitchF = (bSwitch) ? 1.0f : 0.0f;
	float cSwitchF = (cSwitch) ? 1.0f : 0.0f;
	float dSwitchF = (dSwitch) ? 1.0f : 0.0f;
	float eSwitchF = (eSwitch) ? 1.0f : 0.0f;
	float lSwitchF = (lSwitch) ? 1.0f : 0.0f;
	float Alpha1 = (alpha1) ? 1.0f : 0.0f;
	float Alpha2 = (alpha2) ? 1.0f : 0.0f;

    float log1 = m_Log1->getValueAtTime(p_Args.time);
    float log2 = m_Log2->getValueAtTime(p_Args.time);
    float log3 = m_Log3->getValueAtTime(p_Args.time);
    float log4 = m_Log4->getValueAtTime(p_Args.time);
    
    float sat1 = m_Sat1->getValueAtTime(p_Args.time);
    float sat2 = m_Sat2->getValueAtTime(p_Args.time);
    float sat3 = m_Sat3->getValueAtTime(p_Args.time);
    
    float hue1 = RGBtoHUE(rHueA, gHueA, bHueA);
    float hue2 = RGBtoHUE(rHueB, gHueB, bHueB);
    float hue3 = RGBtoHUE(rHueC, gHueC, bHueC);
    
    float hue4 = m_HueR1->getValueAtTime(p_Args.time) / 2.0;
    float hue5 = m_HueP1->getValueAtTime(p_Args.time);
    float hue6 = m_HueSH1->getValueAtTime(p_Args.time);
    float hue7 = m_HueSHP1->getValueAtTime(p_Args.time);

	float hue8 = m_HueR2->getValueAtTime(p_Args.time) / 2.0;
    float hue9 = m_HueP2->getValueAtTime(p_Args.time);
    float hue10 = m_HueSH2->getValueAtTime(p_Args.time);
    float hue11 = m_HueSHP2->getValueAtTime(p_Args.time);

	float hue12 = m_HueR3->getValueAtTime(p_Args.time) / 2.0;
    float hue13 = m_HueP3->getValueAtTime(p_Args.time);
    float hue14 = m_HueSH3->getValueAtTime(p_Args.time);
    float hue15 = m_HueSHP3->getValueAtTime(p_Args.time);
    
    float luma1 = m_LumaA->getValueAtTime(p_Args.time);
    float luma2 = m_LumaB->getValueAtTime(p_Args.time);
    float luma3 = m_LumaC->getValueAtTime(p_Args.time);


    // Set the images
    p_HueConverge.setDstImg(dst.get());
    p_HueConverge.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_HueConverge.setGPURenderArgs(p_Args);

    // Set the render window
    p_HueConverge.setRenderWindow(p_Args.renderWindow);
    
    // Set the scales
    p_HueConverge.setScales(aSwitchF, bSwitchF, cSwitchF, dSwitchF, eSwitchF, lSwitchF, 
    Alpha1, Alpha2, log1, log2, log3, log4, sat1, sat2, sat3, hue1, hue2, hue3, hue4, hue5, 
    hue6, hue7, hue8, hue9, hue10, hue11, hue12, hue13, hue14, hue15, luma1, luma2, luma3);

    // Call the base class process member, this will call the derived templated process code
    p_HueConverge.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

HueConvergePluginFactory::HueConvergePluginFactory()
    : OFX::PluginFactoryHelper<HueConvergePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void HueConvergePluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
    
    // Setup OpenCL and CUDA render capability flags
    p_Desc.setSupportsOpenCLRender(true);
    p_Desc.setSupportsCudaRender(true);
    
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(1.0);
    param->setRange(0.0, 10.0);
    param->setIncrement(0.1);
    param->setDisplayRange(0.0, 10.0);
    param->setDoubleType(eDoubleTypeScale);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void HueConvergePluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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


    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");
    
    GroupParamDescriptor* log1 = p_Desc.defineGroupParam("Logistic Function");
    log1->setOpen(false);
    log1->setHint("S-Curve");
      if (page) {
            page->addChild(*log1);
    }
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("switch1");
    boolParam->setDefault(false);
    boolParam->setHint("S-Curve switch");
    boolParam->setLabel("Logistic Switch");
    boolParam->setParent(*log1);
    page->addChild(*boolParam);

    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "peak1", "peak", "Curve Peak", log1);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "curvep1", "curve", "S-Curve", log1);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "pivot1", "pivot", "Pivot Point", log1);
    param->setDefault(0.435);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "offset", "offset", "Offset", log1);
    param->setDefault(0.0);
    param->setRange(-1.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-1.0, 1.0);
    page->addChild(*param);
    
    GroupParamDescriptor* sat1 = p_Desc.defineGroupParam("Saturation");
    sat1->setOpen(false);
    sat1->setHint("Sat control");
      if (page) {
            page->addChild(*sat1);
    }
    
    boolParam = p_Desc.defineBooleanParam("switch2");
    boolParam->setDefault(true);
    boolParam->setHint("Saturation switch");
    boolParam->setLabel("Sat Switch");
    boolParam->setParent(*sat1);
    page->addChild(*boolParam);
    
    param = defineScaleParam(p_Desc, "satscale", "sat", "Sat Scale", sat1);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "satsoft", "soft-clip", "Soft-Clip", sat1);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "satsoftluma", "luma limiter", "use luma alpha channel to limit saturation soft-clip", sat1);
    param->setDefault(0.0);
    param->setRange(-2.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-2.0, 2.0);
    page->addChild(*param);
    
	boolParam = p_Desc.defineBooleanParam("displayAlpha1");
	boolParam->setDefault(false);
	boolParam->setHint("display alpha channel");
	boolParam->setLabel("display alpha");
	boolParam->setParent(*sat1);
    page->addChild(*boolParam);

    GroupParamDescriptor* hue1 = p_Desc.defineGroupParam("Hue One");
    hue1->setOpen(true);
    hue1->setHint("Hue Converger");
      if (page) {
            page->addChild(*hue1);
    }

    boolParam = p_Desc.defineBooleanParam("hueSwitch1");
    boolParam->setDefault(true);
    boolParam->setHint("hue1 switch");
    boolParam->setLabel("Hue1 Switch");
    boolParam->setParent(*hue1);
    page->addChild(*boolParam);

    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("hueA");
        param->setLabel("Hue1");
        param->setHint("hue");
        param->setDefault(0.62, 0.25, 0.0);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*hue1);
        page->addChild(*param);
    }
    
    param = defineScaleParam(p_Desc, "range1", "range", "Hue Range", hue1);
    param->setDefault(0.2);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "power1", "converge", "power", hue1);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "sathue1", "sat influence", "sathue", hue1);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "sathuepow1", "sat power", "power", hue1);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    GroupParamDescriptor* hue2 = p_Desc.defineGroupParam("Hue Two");
    hue2->setOpen(false);
    hue2->setHint("Hue Converger");
      if (page) {
            page->addChild(*hue2);
    }

    boolParam = p_Desc.defineBooleanParam("hueSwitch2");
    boolParam->setDefault(true);
    boolParam->setHint("hue2 switch");
    boolParam->setLabel("Hue2 Switch");
    boolParam->setParent(*hue2);
    page->addChild(*boolParam);
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("hueB");
        param->setLabel("Hue2");
        param->setHint("hue");
        param->setDefault(0.0, 0.37, 0.75);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*hue2);
        page->addChild(*param);
    }
    
    param = defineScaleParam(p_Desc, "range2", "range", "Hue Range", hue2);
    param->setDefault(0.2);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "power2", "converge", "power", hue2);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "sathue2", "sat influence", "sathue", hue2);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "sathuepow2", "sat power", "power", hue2);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    GroupParamDescriptor* hue3 = p_Desc.defineGroupParam("Hue Three");
    hue3->setOpen(false);
    hue3->setHint("Hue Converger");
      if (page) {
            page->addChild(*hue3);
    }

    boolParam = p_Desc.defineBooleanParam("hueSwitch3");
    boolParam->setDefault(true);
    boolParam->setHint("hue3 switch");
    boolParam->setLabel("Hue3 Switch");
    boolParam->setParent(*hue3);
    page->addChild(*boolParam);
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("hueC");
        param->setLabel("Hue3");
        param->setHint("hue");
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*hue3);
        page->addChild(*param);
    }
    
    param = defineScaleParam(p_Desc, "range3", "range", "Hue Range", hue3);
    param->setDefault(0.2);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "power3", "converge", "power", hue3);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "sathue3", "sat influence", "sathue", hue3);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "sathuepow3", "sat power", "power", hue3);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    GroupParamDescriptor* luma = p_Desc.defineGroupParam("Sat vs Luma");
    luma->setOpen(false);
    luma->setHint("Saturation vs Luma");
      if (page) {
            page->addChild(*luma);
    }
    
    boolParam = p_Desc.defineBooleanParam("lumaSwitch");
    boolParam->setDefault(true);
    boolParam->setHint("saturation vs luma switch");
    boolParam->setLabel("Sat vs Luma Switch");
    boolParam->setParent(*luma);
    page->addChild(*boolParam);

	param = defineScaleParam(p_Desc, "lumaC", "luma scale", "luma scale", luma);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "lumaB", "sat power", "apply power function to saturation alpha channel", luma);
    param->setDefault(1.0);
    param->setRange(0.1, 10.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.2, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "lumaA", "luma limiter", "shadows < 0 > highlights", luma);
    param->setDefault(0.0);
    param->setRange(-2.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-2.0, 2.0);
    page->addChild(*param);
     
    boolParam = p_Desc.defineBooleanParam("displayAlpha2");
	boolParam->setDefault(false);
	boolParam->setHint("display alpha channel of combined saturation channel and luma limiter");
	boolParam->setLabel("display alpha");
	boolParam->setParent(*luma);
    page->addChild(*boolParam);
    
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("info");
    param->setLabel("Info");
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
	param->setDefault("HueConverge");
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

ImageEffect* HueConvergePluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new HueConvergePlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static HueConvergePluginFactory HueConvergePlugin;
    p_FactoryArray.push_back(&HueConvergePlugin);
}
