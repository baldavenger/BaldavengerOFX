/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of openfx-supportext <https://github.com/devernay/openfx-supportext>,
 * Copyright (C) 2013-2017 INRIA
 *
 * openfx-supportext is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * openfx-supportext is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with openfx-supportext.  If not, see <http://www.gnu.org/licenses/gpl-2.0.html>
 * ***** END LICENSE BLOCK ***** */

/*
 * OFX generic rectangle interact with 4 corner points + center point and 4 mid-points.
 * You can use it to define any rectangle in an image resizable by the user.
 */

#ifndef openfx_supportext_ofxsRamp_h
#define openfx_supportext_ofxsRamp_h

#include <cmath>

#include <ofxsInteract.h>
#include <ofxsImageEffect.h>
#include "ofxsMacros.h"
#include "ofxsOGLTextRenderer.h"

#define kParamRampPoint0 "rampPoint0"
#define kParamRampPoint0Label "Point 0"

#define kParamRampColor0 "rampColor0"
#define kParamRampColor0Label "Color 0"

#define kParamRampPoint1 "rampPoint1"
#define kParamRampPoint1Label "Point 1"

#define kParamRampColor1 "rampColor1"
#define kParamRampColor1Label "Color 1"

#define kParamRampType "rampType"
#define kParamRampTypeLabel "Ramp Type"
#define kParamRampTypeHint "The type of interpolation used to generate the ramp"
#define kParamRampTypeOptionLinear "Linear"
#define kParamRampTypeOptionLinearHint "Linear ramp."
#define kParamRampTypeOptionPLinear "PLinear"
#define kParamRampTypeOptionPLinearHint "Perceptually linear ramp in Rec.709."
#define kParamRampTypeOptionEaseIn "Ease-in"
#define kParamRampTypeOptionEaseInHint "Catmull-Rom spline, smooth start, linear end (a.k.a. smooth0)."
#define kParamRampTypeOptionEaseOut "Ease-out"
#define kParamRampTypeOptionEaseOutHint "Catmull-Rom spline, linear start, smooth end (a.k.a. smooth1)."
#define kParamRampTypeOptionSmooth "Smooth"
#define kParamRampTypeOptionSmoothHint "Traditional smoothstep ramp."
#define kParamRampTypeOptionNone "None"
#define kParamRampTypeOptionNoneHint "No color gradient."

#define kParamRampInteractOpen "rampInteractOpen"
#define kParamRampInteractOpenLabel "Show Interact"
#define kParamRampInteractOpenHint "If checked, the ramp interact is displayed over the image."

#define kParamRampInteractive "rampInteractive"
#define kParamRampInteractiveLabel "Interactive Update"
#define kParamRampInteractiveHint "If checked, update the parameter values during interaction with the image viewer, else update the values when pen is released."

// old names, for the Ramp plugin only
#define kParamRampPoint0Old "point0"
#define kParamRampColor0Old "color0"
#define kParamRampPoint1Old "point1"
#define kParamRampColor1Old "color1"
#define kParamRampTypeOld "type"
#define kParamRampInteractiveOld "interactive"

namespace OFX {
enum RampTypeEnum
{
    eRampTypeLinear = 0,
    eRampTypePLinear,
    eRampTypeEaseIn,
    eRampTypeEaseOut,
    eRampTypeSmooth,
    eRampTypeNone
};

class RampInteractHelper
    : private OFX::InteractAbstract
{
    enum InteractState
    {
        eInteractStateIdle = 0,
        eInteractStateDraggingPoint0,
        eInteractStateDraggingPoint1
    };

    Double2DParam* _point0;
    Double2DParam* _point1;
    ChoiceParam* _type;
    BooleanParam* _interactOpen;
    BooleanParam* _interactive;
    OfxPointD _point0DragPos, _point1DragPos;
    bool _interactiveDrag;
    OfxPointD _lastMousePos;
    InteractState _state;
    OFX::ImageEffect* _effect;
    OFX::Interact* _interact;
    Clip *_dstClip;

public:
    RampInteractHelper(OFX::ImageEffect* effect,
                       OFX::Interact* interact,
                       bool oldParams = false)
        : _point0(0)
        , _point1(0)
        , _type(0)
        , _interactOpen(0)
        , _interactive(0)
        , _point0DragPos()
        , _point1DragPos()
        , _interactiveDrag(false)
        , _lastMousePos()
        , _state(eInteractStateIdle)
        , _effect(effect)
        , _interact(interact)
        , _dstClip(0)
    {
        if (oldParams) {
            _point0 = effect->fetchDouble2DParam(kParamRampPoint0Old);
            _point1 = effect->fetchDouble2DParam(kParamRampPoint1Old);
            _type = effect->fetchChoiceParam(kParamRampTypeOld);
            _interactive = effect->fetchBooleanParam(kParamRampInteractiveOld);
        } else {
            _point0 = effect->fetchDouble2DParam(kParamRampPoint0);
            _point1 = effect->fetchDouble2DParam(kParamRampPoint1);
            _type = effect->fetchChoiceParam(kParamRampType);
            _interactive = effect->fetchBooleanParam(kParamRampInteractive);
        }
        _interactOpen = _effect->fetchBooleanParam(kParamRampInteractOpen);
        assert(_point0 && _point1 && _type && _interactOpen && _interactive);
        assert(_effect && _interact);
        _dstClip = _effect->fetchClip(kOfxImageEffectOutputClipName);
        assert(_dstClip);
    }

    /** @brief virtual destructor */
    virtual ~RampInteractHelper()
    {
        // fetched clips and params are owned and deleted by the ImageEffect and its ParamSet
    }

    /** @brief the function called to draw in the interact */
    virtual bool draw(const DrawArgs &args) OVERRIDE;

    /** @brief the function called to handle pen motion in the interact

       returns true if the interact trapped the action in some sense. This will block the action being passed to
       any other interact that may share the viewer.
     */
    virtual bool penMotion(const PenArgs &args) OVERRIDE;

    /** @brief the function called to handle pen down events in the interact

       returns true if the interact trapped the action in some sense. This will block the action being passed to
       any other interact that may share the viewer.
     */
    virtual bool penDown(const PenArgs &args) OVERRIDE;

    /** @brief the function called to handle pen up events in the interact

       returns true if the interact trapped the action in some sense. This will block the action being passed to
       any other interact that may share the viewer.
     */
    virtual bool penUp(const PenArgs &args) OVERRIDE;

    /** @brief the function called to handle key down events in the interact

       returns true if the interact trapped the action in some sense. This will block the action being passed to
       any other interact that may share the viewer.
     */
    virtual bool keyDown(const KeyArgs & /*args*/) OVERRIDE { return false; };

    /** @brief the function called to handle key up events in the interact

       returns true if the interact trapped the action in some sense. This will block the action being passed to
       any other interact that may share the viewer.
     */
    virtual bool keyUp(const KeyArgs & /*args*/) OVERRIDE { return false; };

    /** @brief the function called to handle key down repeat events in the interact

       returns true if the interact trapped the action in some sense. This will block the action being passed to
       any other interact that may share the viewer.
     */
    virtual bool keyRepeat(const KeyArgs & /*args*/) OVERRIDE { return false; };

    /** @brief Called when the interact is given input focus */
    virtual void gainFocus(const FocusArgs & /*args*/) OVERRIDE {};

    /** @brief Called when the interact is loses input focus */
    virtual void loseFocus(const FocusArgs &args) OVERRIDE;
};

typedef OverlayInteractFromHelper<RampInteractHelper> RampInteract;

class RampOverlayDescriptor
    : public DefaultEffectOverlayDescriptor<RampOverlayDescriptor, RampInteract>
{
};

class RampInteractHelperOldParams
    : public RampInteractHelper
{
public:
    RampInteractHelperOldParams(OFX::ImageEffect* effect,
                                OFX::Interact* interact)
        : RampInteractHelper(effect, interact, true) {}
};

typedef OverlayInteractFromHelper<RampInteractHelperOldParams> RampInteractOldParams;

class RampOverlayDescriptorOldParams
    : public DefaultEffectOverlayDescriptor<RampOverlayDescriptorOldParams, RampInteractOldParams>
{
};


template<RampTypeEnum type>
double
ofxsRampFunc(double t)
{
    if ( (t >= 1.) || (type == eRampTypeNone) ) {
        t = 1.;
    } else if (t <= 0) {
        t = 0.;
    } else {
        // from http://www.comp-fu.com/2012/01/nukes-smooth-ramp-functions/
        // linear
        //y = x
        // plinear: perceptually linear in rec709
        //y = pow(x, 3)
        // smooth: traditional smoothstep
        //y = x*x*(3 - 2*x)
        // smooth0: Catmull-Rom spline, smooth start, linear end
        //y = x*x*(2 - x)
        // smooth1: Catmull-Rom spline, linear start, smooth end
        //y = x*(1 + x*(1 - x))
        switch (type) {
        case eRampTypeLinear:
            break;
        case eRampTypePLinear:
            // plinear: perceptually linear in rec709
            t = t * t * t;
            break;
        case eRampTypeEaseIn:
            //t *= t; // old version, end of curve is too sharp
            // smooth0: Catmull-Rom spline, smooth start, linear end
            t = t * t * (2 - t);
            break;
        case eRampTypeEaseOut:
            //t = - t * (t - 2); // old version, start of curve is too sharp
            // smooth1: Catmull-Rom spline, linear start, smooth end
            t = t * ( 1 + t * (1 - t) );
            break;
        case eRampTypeSmooth:
            /*
               t *= 2.;
               if (t < 1) {
               t = t * t / (2.);
               } else {
               --t;
               t =  -0.5 * (t * (t - 2) - 1);
               }
             */
            // smooth: traditional smoothstep
            t = t * t * (3 - 2 * t);
            break;
        case eRampTypeNone:
            t = 1.;
            break;
        default:
            break;
        }
    }

    return t;
} // ofxsRampFunc

template<RampTypeEnum type>
double
ofxsRampFunc(const OfxPointD& p0,
             double nx,
             double ny,
             const OfxPointD& p)
{
    double t = (p.x - p0.x) * nx + (p.y - p0.y) * ny;

    return ofxsRampFunc<type>(t);
}

inline double
ofxsRampFunc(const OfxPointD& p0,
             double nx,
             double ny,
             RampTypeEnum type,
             const OfxPointD& p)
{
    double t = (p.x - p0.x) * nx + (p.y - p0.y) * ny;

    switch (type) {
    case eRampTypeLinear:

        return ofxsRampFunc<eRampTypeLinear>(t);
        break;
    case eRampTypePLinear:

        return ofxsRampFunc<eRampTypePLinear>(t);
        break;
    case eRampTypeEaseIn:

        return ofxsRampFunc<eRampTypeEaseIn>(t);
        break;
    case eRampTypeEaseOut:

        return ofxsRampFunc<eRampTypeEaseOut>(t);
        break;
    case eRampTypeSmooth:

        return ofxsRampFunc<eRampTypeSmooth>(t);
        break;
    case eRampTypeNone:
        t = 1.;
        break;
    default:
        break;
    }

    return t;
}

void ofxsRampDescribeParams(OFX::ImageEffectDescriptor &desc,
                            OFX::PageParamDescriptor *page,
                            OFX::GroupParamDescriptor *group,
                            RampTypeEnum defaultType,
                            bool isOpen,
                            bool oldParams);
} // namespace OFX

#endif /* defined(openfx_supportext_ofxsRamp_h) */
