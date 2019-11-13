/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of openfx-supportext <https://github.com/devernay/openfx-supportext>,
 * Copyright (C) 2013-2016 INRIA
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

#ifndef openfx_supportext_ofxsRectangleInteract_h
#define openfx_supportext_ofxsRectangleInteract_h

#include <ofxsInteract.h>
#include <ofxsImageEffect.h>
#include "ofxsMacros.h"

// a secret parameter used to enable the interact, if it exists
#define kParamRectangleInteractEnable "rectangleInteractEnable"

#define kParamRectangleInteractBtmLeft "bottomLeft"
#define kParamRectangleInteractBtmLeftLabel "Bottom Left"
#define kParamRectangleInteractBtmLeftHint "Coordinates of the bottom left corner of the rectangle"

#define kParamRectangleInteractSize "size"
#define kParamRectangleInteractSizeLabel "Size"
#define kParamRectangleInteractSizeHint "Width and height of the rectangle"
#define kParamRectangleInteractSizeDim1 "w"
#define kParamRectangleInteractSizeDim2 "h"

#define kParamRectangleInteractInteractive "interactive"
#define kParamRectangleInteractInteractiveLabel "Interactive Update"
#define kParamRectangleInteractInteractiveHint "If checked, update the parameter values during interaction with the image viewer, else update the values when pen is released."

namespace OFX {
/**
 * @brief In order to work the plug-in using this interact must have 2 parameters named after
 * the defines above.
 *
 **/
class RectangleInteract
    : public OFX::OverlayInteract
{
public:
    enum MouseStateEnum
    {
        eMouseStateIdle = 0,
        eMouseStateDraggingTopLeft,
        eMouseStateDraggingTopRight,
        eMouseStateDraggingBtmLeft,
        eMouseStateDraggingBtmRight,
        eMouseStateDraggingCenter,
        eMouseStateDraggingTopMid,
        eMouseStateDraggingMidRight,
        eMouseStateDraggingBtmMid,
        eMouseStateDraggingMidLeft
    };

    enum DrawStateEnum
    {
        eDrawStateInactive = 0,
        eDrawStateHoveringTopLeft,
        eDrawStateHoveringTopRight,
        eDrawStateHoveringBtmLeft,
        eDrawStateHoveringBtmRight,
        eDrawStateHoveringCenter,
        eDrawStateHoveringTopMid,
        eDrawStateHoveringMidRight,
        eDrawStateHoveringBtmMid,
        eDrawStateHoveringMidLeft
    };

public:

    RectangleInteract(OfxInteractHandle handle,
                      OFX::ImageEffect* effect)
        : OFX::OverlayInteract(handle)
        , _effect(effect)
        , _lastMousePos()
        , _mouseState(eMouseStateIdle)
        , _drawState(eDrawStateInactive)
        , _modifierStateCtrl(0)
        , _modifierStateShift(0)
        , _enable(0)
        , _btmLeft(0)
        , _size(0)
    {
        if ( _effect->paramExists(kParamRectangleInteractEnable) ) {
            _enable = effect->fetchBooleanParam(kParamRectangleInteractEnable);
        }
        _btmLeft = effect->fetchDouble2DParam(kParamRectangleInteractBtmLeft);
        _size = effect->fetchDouble2DParam(kParamRectangleInteractSize);
        addParamToSlaveTo(_btmLeft);
        addParamToSlaveTo(_size);
        assert(_btmLeft && _size);
        _interactive = effect->paramExists(kParamRectangleInteractInteractive) ? effect->fetchBooleanParam(kParamRectangleInteractInteractive) : 0;
        _btmLeftDragPos.x = _btmLeftDragPos.y = 0;
        _sizeDrag.x = _sizeDrag.y = 0;
        _interactiveDrag = false;
    }

    // overridden functions from OFX::Interact to do things
    virtual bool draw(const OFX::DrawArgs &args) OVERRIDE;
    virtual bool penMotion(const OFX::PenArgs &args) OVERRIDE;
    virtual bool penDown(const OFX::PenArgs &args) OVERRIDE;
    virtual bool penUp(const OFX::PenArgs &args) OVERRIDE;
    virtual bool keyDown(const OFX::KeyArgs &args) OVERRIDE;
    virtual bool keyUp(const OFX::KeyArgs & args) OVERRIDE;
    virtual void loseFocus(const FocusArgs &args) OVERRIDE;

protected:


    /**
     * @brief This method returns the bottom left point. The base implementation just returns the value
     * of the _btmLeft parameter at the given time.
     * One could override this function to  do more complex stuff based on other parameters state like the Crop plug-in does.
     **/
    virtual OfxPointD getBtmLeft(OfxTime time) const;

    /**
     * @brief This is called right before any call to allowXXX is made.
     * This way you can query values of a parameter and store it away without having to do this
     * at every allowXXX call.
     **/
    virtual void aboutToCheckInteractivity(OfxTime /*time*/)
    {
    }

    /**
     * @brif These can be overriden to disallow interaction with a point.
     **/
    virtual bool allowTopLeftInteraction() const
    {
        return true;
    }

    virtual bool allowTopRightInteraction() const
    {
        return true;
    }

    virtual bool allowBtmRightInteraction() const
    {
        return true;
    }

    virtual bool allowBtmLeftInteraction() const
    {
        return true;
    }

    virtual bool allowTopMidInteraction() const
    {
        return true;
    }

    virtual bool allowMidRightInteraction() const
    {
        return true;
    }

    virtual bool allowBtmMidInteraction() const
    {
        return true;
    }

    virtual bool allowMidLeftInteraction() const
    {
        return true;
    }

    virtual bool allowCenterInteraction() const
    {
        return true;
    }

private:
    void setValue(OfxPointD btmLeft, OfxPointD size, const OfxPointD &pscale);

private:
    OFX::ImageEffect* _effect;
    OfxPointD _lastMousePos;
    MouseStateEnum _mouseState;
    DrawStateEnum _drawState;
    int _modifierStateCtrl;
    int _modifierStateShift;
    OFX::BooleanParam* _enable;
    OFX::Double2DParam* _btmLeft;
    OFX::Double2DParam* _size;
    OFX::BooleanParam* _interactive;
    OfxPointD _btmLeftDragPos;
    OfxPointD _sizeDrag;
    bool _interactiveDrag;
};

class RectangleOverlayDescriptor
    : public OFX::DefaultEffectOverlayDescriptor<RectangleOverlayDescriptor, RectangleInteract>
{
};
} // namespace OFX

#endif /* defined(openfx_supportext_ofxsRectangleInteract_h) */
