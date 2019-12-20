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

#include "ofxsRectangleInteract.h"
#include <cmath>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#define POINT_SIZE 5
#define POINT_TOLERANCE 6
#define CROSS_SIZE 7
#define HANDLE_SIZE 6

using OFX::RectangleInteract;

static bool
isNearby(const OfxPointD & p,
         double x,
         double y,
         double tolerance,
         const OfxPointD & pscale)
{
    return std::fabs(p.x - x) <= tolerance * pscale.x &&  std::fabs(p.y - y) <= tolerance * pscale.y;
}

// round to the closest int, 1/10 int, etc
// this make parameter editing easier
// pscale is args.pixelScale.x / args.renderScale.x;
// pscale10 is the power of 10 below pscale
static double
fround(double val,
       double pscale)
{
    if (pscale == 0) {
        return val;
    }
    double pscale10 = std::pow( 10., std::floor( std::log10(pscale) ) );

    return pscale10 * std::floor(val / pscale10 + 0.5);
}

static void
drawPoint(const OfxRGBColourD &color,
          bool draw,
          double x,
          double y,
          RectangleInteract::DrawStateEnum id,
          RectangleInteract::DrawStateEnum ds,
          bool keepAR,
          int l)
{
    if (draw) {
        if (ds == id) {
            if (keepAR) {
                glColor3f(1.f * l, 0.f * l, 0.f * l);
            } else {
                glColor3f(0.f * l, 1.f * l, 0.f * l);
            }
        } else {
            glColor3f( (float)color.r * l, (float)color.g * l, (float)color.b * l );
        }
        glVertex2d(x, y);
    }
}

bool
RectangleInteract::draw(const OFX::DrawArgs &args)
{
    if ( _btmLeft->getIsSecret() || _size->getIsSecret() ||
         !_btmLeft->getIsEnable() || !_size->getIsEnable() ||
         ( _enable && !_enable->getValueAtTime(args.time) ) ) {
        return false;
    }

    OfxRGBColourD color = { 0.8, 0.8, 0.8 };
    getSuggestedColour(color);
    const OfxPointD& pscale = args.pixelScale;
    GLdouble projection[16];
    glGetDoublev( GL_PROJECTION_MATRIX, projection);
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    OfxPointD shadow; // how much to translate GL_PROJECTION to get exactly one pixel on screen
    shadow.x = 2. / (projection[0] * viewport[2]);
    shadow.y = 2. / (projection[5] * viewport[3]);

    double x1, y1, w, h;
    if (_mouseState != eMouseStateIdle) {
        x1 = _btmLeftDragPos.x;
        y1 = _btmLeftDragPos.y;
        w = _sizeDrag.x;
        h = _sizeDrag.y;
    } else {
        _btmLeft->getValueAtTime(args.time, x1, y1);
        _size->getValueAtTime(args.time, w, h);
    }
    double x2 = x1 + w;
    double y2 = y1 + h;
    double xc = x1 + w / 2;
    double yc = y1 + h / 2;
    const bool keepAR = _modifierStateShift > 0;
    const bool centered = _modifierStateCtrl > 0;

    //glPushAttrib(GL_ALL_ATTRIB_BITS); // caller is responsible for protecting attribs
    aboutToCheckInteractivity(args.time);

    glDisable(GL_LINE_STIPPLE);
    glEnable(GL_LINE_SMOOTH);
    glDisable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE);
    glLineWidth(1.5f);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw everything twice
    // l = 0: shadow
    // l = 1: drawing
    for (int l = 0; l < 2; ++l) {
        // shadow (uses GL_PROJECTION)
        glMatrixMode(GL_PROJECTION);
        int direction = (l == 0) ? 1 : -1;
        // translate (1,-1) pixels
        glTranslated(direction * shadow.x, -direction * shadow.y, 0);
        glMatrixMode(GL_MODELVIEW); // Modelview should be used on Nuke

        glColor3f( (float)color.r * l, (float)color.g * l, (float)color.b * l );

        glBegin(GL_LINE_LOOP);
        glVertex2d(x1, y1);
        glVertex2d(x1, y2);
        glVertex2d(x2, y2);
        glVertex2d(x2, y1);
        glEnd();

        glPointSize(POINT_SIZE);
        glBegin(GL_POINTS);
        drawPoint(color, allowBtmLeftInteraction(),  x1, y1, eDrawStateHoveringBtmLeft,  _drawState, keepAR, l);
        drawPoint(color, allowMidLeftInteraction(),  x1, yc, eDrawStateHoveringMidLeft,  _drawState, false,  l);
        drawPoint(color, allowTopLeftInteraction(),  x1, y2, eDrawStateHoveringTopLeft,  _drawState, keepAR, l);
        drawPoint(color, allowBtmMidInteraction(),   xc, y1, eDrawStateHoveringBtmMid,   _drawState, false,  l);
        drawPoint(color, allowCenterInteraction(),   xc, yc, eDrawStateHoveringCenter,   _drawState, false,  l);
        drawPoint(color, allowTopMidInteraction(),   xc, y2, eDrawStateHoveringTopMid,   _drawState, false,  l);
        drawPoint(color, allowBtmRightInteraction(), x2, y1, eDrawStateHoveringBtmRight, _drawState, keepAR, l);
        drawPoint(color, allowMidRightInteraction(), x2, yc, eDrawStateHoveringMidRight, _drawState, false,  l);
        drawPoint(color, allowTopRightInteraction(), x2, y2, eDrawStateHoveringTopRight, _drawState, keepAR, l);
        glEnd();
        glPointSize(1);

        ///draw center cross hair
        glBegin(GL_LINES);
        if ( (_drawState == eDrawStateHoveringCenter) || ( centered && (_drawState != eDrawStateInactive) ) ) {
            glColor3f(0.f * l, 1.f * l, 0.f * l);
        } else if ( !allowCenterInteraction() ) {
            glColor3f( (float)(color.r / 2) * l, (float)(color.g / 2) * l, (float)(color.b / 2) * l );
        } else {
            glColor3f( (float)color.r * l, (float)color.g * l, (float)color.b * l );
        }
        glVertex2d(xc - CROSS_SIZE * pscale.x, yc);
        glVertex2d(xc + CROSS_SIZE * pscale.x, yc);
        glVertex2d(xc, yc - CROSS_SIZE * pscale.y);
        glVertex2d(xc, yc + CROSS_SIZE * pscale.y);
        glEnd();
    }
    //glPopAttrib();

    return true;
} // draw

bool
RectangleInteract::penMotion(const OFX::PenArgs &args)
{
    if ( _btmLeft->getIsSecret() || _size->getIsSecret() ||
         !_btmLeft->getIsEnable() || !_size->getIsEnable() ||
         ( _enable && !_enable->getValueAtTime(args.time) ) ) {
        return false;
    }

    const OfxPointD& pscale = args.pixelScale;
    double x1, y1, w, h;
    if (_mouseState != eMouseStateIdle) {
        x1 = _btmLeftDragPos.x;
        y1 = _btmLeftDragPos.y;
        w = _sizeDrag.x;
        h = _sizeDrag.y;
    } else {
        _btmLeft->getValueAtTime(args.time, x1, y1);
        _size->getValueAtTime(args.time, w, h);
    }
    double x2 = x1 + w;
    double y2 = y1 + h;
    double xc = x1 + w / 2;
    double yc = y1 + h / 2;
    bool didSomething = false;
    bool valuesChanged = false;
    OfxPointD delta;
    delta.x = args.penPosition.x - _lastMousePos.x;
    delta.y = args.penPosition.y - _lastMousePos.y;

    bool lastStateWasHovered = _drawState != eDrawStateInactive;


    aboutToCheckInteractivity(args.time);
    // test center first
    if ( isNearby(args.penPosition, xc, yc, POINT_TOLERANCE, pscale)  && allowCenterInteraction() ) {
        _drawState = eDrawStateHoveringCenter;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x1, y1, POINT_TOLERANCE, pscale) && allowBtmLeftInteraction() ) {
        _drawState = eDrawStateHoveringBtmLeft;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x2, y1, POINT_TOLERANCE, pscale) && allowBtmRightInteraction() ) {
        _drawState = eDrawStateHoveringBtmRight;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x1, y2, POINT_TOLERANCE, pscale)  && allowTopLeftInteraction() ) {
        _drawState = eDrawStateHoveringTopLeft;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x2, y2, POINT_TOLERANCE, pscale) && allowTopRightInteraction() ) {
        _drawState = eDrawStateHoveringTopRight;
        didSomething = true;
    } else if ( isNearby(args.penPosition, xc, y1, POINT_TOLERANCE, pscale)  && allowBtmMidInteraction() ) {
        _drawState = eDrawStateHoveringBtmMid;
        didSomething = true;
    } else if ( isNearby(args.penPosition, xc, y2, POINT_TOLERANCE, pscale) && allowTopMidInteraction() ) {
        _drawState = eDrawStateHoveringTopMid;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x1, yc, POINT_TOLERANCE, pscale)  && allowMidLeftInteraction() ) {
        _drawState = eDrawStateHoveringMidLeft;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x2, yc, POINT_TOLERANCE, pscale) && allowMidRightInteraction() ) {
        _drawState = eDrawStateHoveringMidRight;
        didSomething = true;
    } else {
        _drawState = eDrawStateInactive;
    }

    const bool keepAR = _modifierStateShift > 0;
    const bool centered = _modifierStateCtrl > 0;
    if ( keepAR && (_sizeDrag.x > 0.) && (_sizeDrag.y > 0.) &&
         ( ( _mouseState == eMouseStateDraggingTopLeft) ||
           ( _mouseState == eMouseStateDraggingTopRight) ||
           ( _mouseState == eMouseStateDraggingBtmLeft) ||
           ( _mouseState == eMouseStateDraggingBtmRight) ) ) {
        double r2 = _sizeDrag.x * _sizeDrag.x + _sizeDrag.y * _sizeDrag.y;
        if ( (_mouseState == eMouseStateDraggingTopRight) ||
             ( _mouseState == eMouseStateDraggingBtmLeft) ) {
            double dotprod = (delta.x * _sizeDrag.y + delta.y * _sizeDrag.x) / r2;
            delta.x = _sizeDrag.x * dotprod;
            delta.y = _sizeDrag.y * dotprod;
        } else {
            double dotprod = (delta.x * _sizeDrag.y - delta.y * _sizeDrag.x) / r2;
            delta.x = _sizeDrag.x * dotprod;
            delta.y = -_sizeDrag.y * dotprod;
        }
    }
    if (_mouseState == eMouseStateDraggingBtmLeft) {
        _drawState = eDrawStateHoveringBtmLeft;
        OfxPointD topRight;
        topRight.x = _btmLeftDragPos.x + _sizeDrag.x;
        topRight.y = _btmLeftDragPos.y + _sizeDrag.y;
        _btmLeftDragPos.x += delta.x;
        _btmLeftDragPos.y += delta.y;
        _sizeDrag.x = topRight.x - _btmLeftDragPos.x;
        _sizeDrag.y = topRight.y - _btmLeftDragPos.y;
        if (centered) {
            _sizeDrag.x -= delta.x;
            _sizeDrag.y -= delta.y;
        }
        valuesChanged = true;
    } else if (_mouseState == eMouseStateDraggingTopLeft) {
        _drawState = eDrawStateHoveringTopLeft;
        OfxPointD btmRight;
        btmRight.x = _btmLeftDragPos.x + _sizeDrag.x;
        btmRight.y = _btmLeftDragPos.y;
        _btmLeftDragPos.x += delta.x;
        _sizeDrag.y += delta.y;
        _sizeDrag.x = btmRight.x - _btmLeftDragPos.x;
        if (centered) {
            _sizeDrag.x -= delta.x;
            _sizeDrag.y += delta.y;
            _btmLeftDragPos.y -= delta.y;
        }
        valuesChanged = true;
    } else if (_mouseState == eMouseStateDraggingTopRight) {
        _drawState = eDrawStateHoveringTopRight;
        _sizeDrag.x += delta.x;
        _sizeDrag.y += delta.y;
        if (centered) {
            _sizeDrag.x += delta.x;
            _btmLeftDragPos.x -= delta.x;
            _sizeDrag.y += delta.y;
            _btmLeftDragPos.y -= delta.y;
        }
        valuesChanged = true;
    } else if (_mouseState == eMouseStateDraggingBtmRight) {
        _drawState = eDrawStateHoveringBtmRight;
        OfxPointD topLeft;
        topLeft.x = _btmLeftDragPos.x;
        topLeft.y = _btmLeftDragPos.y + _sizeDrag.y;
        _sizeDrag.x += delta.x;
        _btmLeftDragPos.y += delta.y;
        _sizeDrag.y = topLeft.y - _btmLeftDragPos.y;
        if (centered) {
            _sizeDrag.x += delta.x;
            _btmLeftDragPos.x -= delta.x;
            _sizeDrag.y -= delta.y;
        }
        valuesChanged = true;
    } else if (_mouseState == eMouseStateDraggingTopMid) {
        _drawState = eDrawStateHoveringTopMid;
        _sizeDrag.y += delta.y;
        if (centered) {
            _sizeDrag.y += delta.y;
            _btmLeftDragPos.y -= delta.y;
        }
        valuesChanged = true;
    } else if (_mouseState == eMouseStateDraggingMidRight) {
        _drawState = eDrawStateHoveringMidRight;
        _sizeDrag.x += delta.x;
        if (centered) {
            _sizeDrag.x += delta.x;
            _btmLeftDragPos.x -= delta.x;
        }
        valuesChanged = true;
    } else if (_mouseState == eMouseStateDraggingBtmMid) {
        _drawState = eDrawStateHoveringBtmMid;
        double top = _btmLeftDragPos.y + _sizeDrag.y;
        _btmLeftDragPos.y += delta.y;
        _sizeDrag.y = top - _btmLeftDragPos.y;
        if (centered) {
            _sizeDrag.y -= delta.y;
        }
        valuesChanged = true;
    } else if (_mouseState == eMouseStateDraggingMidLeft) {
        _drawState = eDrawStateHoveringMidLeft;
        double right = _btmLeftDragPos.x + _sizeDrag.x;
        _btmLeftDragPos.x += delta.x;
        _sizeDrag.x = right - _btmLeftDragPos.x;
        if (centered) {
            _sizeDrag.x -= delta.x;
        }
        valuesChanged = true;
    } else if (_mouseState == eMouseStateDraggingCenter) {
        _drawState = eDrawStateHoveringCenter;
        _btmLeftDragPos.x += delta.x;
        _btmLeftDragPos.y += delta.y;
        valuesChanged = true;
    }


    //if size is negative shift bottom left
    if (_sizeDrag.x < 0) {
        if (_mouseState == eMouseStateDraggingBtmLeft) {
            _mouseState = eMouseStateDraggingBtmRight;
        } else if (_mouseState == eMouseStateDraggingMidLeft) {
            _mouseState = eMouseStateDraggingMidRight;
        } else if (_mouseState == eMouseStateDraggingTopLeft) {
            _mouseState = eMouseStateDraggingTopRight;
        } else if (_mouseState == eMouseStateDraggingBtmRight) {
            _mouseState = eMouseStateDraggingBtmLeft;
        } else if (_mouseState == eMouseStateDraggingMidRight) {
            _mouseState = eMouseStateDraggingMidLeft;
        } else if (_mouseState == eMouseStateDraggingTopRight) {
            _mouseState = eMouseStateDraggingTopLeft;
        }

        _btmLeftDragPos.x += _sizeDrag.x;
        _sizeDrag.x = -_sizeDrag.x;
        valuesChanged = true;
    }
    if (_sizeDrag.y < 0) {
        if (_mouseState == eMouseStateDraggingTopLeft) {
            _mouseState = eMouseStateDraggingBtmLeft;
        } else if (_mouseState == eMouseStateDraggingTopMid) {
            _mouseState = eMouseStateDraggingBtmMid;
        } else if (_mouseState == eMouseStateDraggingTopRight) {
            _mouseState = eMouseStateDraggingBtmRight;
        } else if (_mouseState == eMouseStateDraggingBtmLeft) {
            _mouseState = eMouseStateDraggingTopLeft;
        } else if (_mouseState == eMouseStateDraggingBtmMid) {
            _mouseState = eMouseStateDraggingTopMid;
        } else if (_mouseState == eMouseStateDraggingBtmRight) {
            _mouseState = eMouseStateDraggingTopRight;
        }

        _btmLeftDragPos.y += _sizeDrag.y;
        _sizeDrag.y = -_sizeDrag.y;
        valuesChanged = true;
    }

    ///forbid 0 pixels wide crop rectangles
    if (_sizeDrag.x < 1) {
        _sizeDrag.x = 1;
        valuesChanged = true;
    }
    if (_sizeDrag.y < 1) {
        _sizeDrag.y = 1;
        valuesChanged = true;
    }

    ///repaint if we toggled off a hovered handle
    if (lastStateWasHovered) {
        didSomething = true;
    }

    if ( (_mouseState != eMouseStateIdle) && _interactiveDrag && valuesChanged ) {
        setValue(_btmLeftDragPos, _sizeDrag, args.pixelScale);
        // no need to redraw overlay since it is slave to the paramaters
    } else if (didSomething || valuesChanged) {
        requestRedraw();
    }


    _lastMousePos = args.penPosition;

    return didSomething || valuesChanged;
} // penMotion

bool
RectangleInteract::penDown(const OFX::PenArgs &args)
{
    if ( _btmLeft->getIsSecret() || _size->getIsSecret() ||
         !_btmLeft->getIsEnable() || !_size->getIsEnable() ||
         ( _enable && !_enable->getValueAtTime(args.time) ) ) {
        return false;
    }

    const OfxPointD& pscale = args.pixelScale;
    double x1, y1, w, h;
    if (_mouseState != eMouseStateIdle) {
        x1 = _btmLeftDragPos.x;
        y1 = _btmLeftDragPos.y;
        w = _sizeDrag.x;
        h = _sizeDrag.y;
    } else {
        _btmLeft->getValueAtTime(args.time, x1, y1);
        _size->getValueAtTime(args.time, w, h);
        if ( _interactive && _interactive->getIsEnable() ) {
            _interactive->getValueAtTime(args.time, _interactiveDrag);
        } else {
            _interactiveDrag = false;
        }
    }
    double x2 = x1 + w;
    double y2 = y1 + h;
    double xc = x1 + w / 2;
    double yc = y1 + h / 2;
    bool didSomething = false;

    aboutToCheckInteractivity(args.time);

    // test center first
    if ( isNearby(args.penPosition, xc, yc, POINT_TOLERANCE, pscale)  && allowCenterInteraction() ) {
        _mouseState = eMouseStateDraggingCenter;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x1, y1, POINT_TOLERANCE, pscale) && allowBtmLeftInteraction() ) {
        _mouseState = eMouseStateDraggingBtmLeft;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x2, y1, POINT_TOLERANCE, pscale) && allowBtmRightInteraction() ) {
        _mouseState = eMouseStateDraggingBtmRight;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x1, y2, POINT_TOLERANCE, pscale)  && allowTopLeftInteraction() ) {
        _mouseState = eMouseStateDraggingTopLeft;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x2, y2, POINT_TOLERANCE, pscale) && allowTopRightInteraction() ) {
        _mouseState = eMouseStateDraggingTopRight;
        didSomething = true;
    } else if ( isNearby(args.penPosition, xc, y1, POINT_TOLERANCE, pscale)  && allowBtmMidInteraction() ) {
        _mouseState = eMouseStateDraggingBtmMid;
        didSomething = true;
    } else if ( isNearby(args.penPosition, xc, y2, POINT_TOLERANCE, pscale) && allowTopMidInteraction() ) {
        _mouseState = eMouseStateDraggingTopMid;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x1, yc, POINT_TOLERANCE, pscale)  && allowMidLeftInteraction() ) {
        _mouseState = eMouseStateDraggingMidLeft;
        didSomething = true;
    } else if ( isNearby(args.penPosition, x2, yc, POINT_TOLERANCE, pscale) && allowMidRightInteraction() ) {
        _mouseState = eMouseStateDraggingMidRight;
        didSomething = true;
    } else {
        _mouseState = eMouseStateIdle;
    }

    _btmLeftDragPos.x = x1;
    _btmLeftDragPos.y = y1;
    _sizeDrag.x = w;
    _sizeDrag.y = h;
    _lastMousePos = args.penPosition;
    if (didSomething) {
        requestRedraw();
    }

    return didSomething;
} // penDown

bool
RectangleInteract::penUp(const OFX::PenArgs &args)
{
    if ( _btmLeft->getIsSecret() || _size->getIsSecret() ||
         !_btmLeft->getIsEnable() || !_size->getIsEnable() ||
         ( _enable && !_enable->getValueAtTime(args.time) ) ) {
        return false;
    }

    bool didSmthing = false;

    if ( !_interactiveDrag && (_mouseState != eMouseStateIdle) ) {
        // no need to redraw overlay since it is slave to the paramaters
        setValue(_btmLeftDragPos, _sizeDrag, args.pixelScale);
        didSmthing = true;
    } else if (_mouseState != eMouseStateIdle) {
        requestRedraw();
    }
    _mouseState = eMouseStateIdle;

    return didSmthing;
} // penUp

// keyDown just updates the modifier state
bool
RectangleInteract::keyDown(const OFX::KeyArgs &args)
{
    if ( _btmLeft->getIsSecret() || _size->getIsSecret() ||
         !_btmLeft->getIsEnable() || !_size->getIsEnable() ||
         ( _enable && !_enable->getValueAtTime(args.time) ) ) {
        return false;
    }

    // Note that on the Mac:
    // cmd/apple/cloverleaf is kOfxKey_Control_L
    // ctrl is kOfxKey_Meta_L
    // alt/option is kOfxKey_Alt_L
    bool mustRedraw = false;

    // the two control keys may be pressed consecutively, be aware about this
    if ( (args.keySymbol == kOfxKey_Control_L) || (args.keySymbol == kOfxKey_Control_R) ) {
        mustRedraw = _modifierStateCtrl == 0;
        ++_modifierStateCtrl;
    }
    if ( (args.keySymbol == kOfxKey_Shift_L) || (args.keySymbol == kOfxKey_Shift_R) ) {
        mustRedraw = _modifierStateShift == 0;
        ++_modifierStateShift;
    }
    if (mustRedraw) {
        requestRedraw();
    }
    //std::cout << std::hex << args.keySymbol << std::endl;

    // modifiers are not "caught"
    return false;
}

// keyUp just updates the modifier state
bool
RectangleInteract::keyUp(const OFX::KeyArgs &args)
{
    if ( _btmLeft->getIsSecret() || _size->getIsSecret() ||
         !_btmLeft->getIsEnable() || !_size->getIsEnable() ||
         ( _enable && !_enable->getValueAtTime(args.time) ) ) {
        return false;
    }

    bool mustRedraw = false;

    if ( (args.keySymbol == kOfxKey_Control_L) || (args.keySymbol == kOfxKey_Control_R) ) {
        // we may have missed a keypress
        if (_modifierStateCtrl > 0) {
            --_modifierStateCtrl;
            mustRedraw = _modifierStateCtrl == 0;
        }
    }
    if ( (args.keySymbol == kOfxKey_Shift_L) || (args.keySymbol == kOfxKey_Shift_R) ) {
        if (_modifierStateShift > 0) {
            --_modifierStateShift;
            mustRedraw = _modifierStateShift == 0;
        }
    }
    if (mustRedraw) {
        requestRedraw();
    }

    // modifiers are not "caught"
    return false;
}

/** @brief Called when the interact is loses input focus */
void
RectangleInteract::loseFocus(const OFX::FocusArgs & /*args*/)
{
    // reset the modifiers state
    _modifierStateCtrl = 0;
    _modifierStateShift = 0;
    _interactiveDrag = false;
}

OfxPointD
RectangleInteract::getBtmLeft(OfxTime time) const
{
    OfxPointD ret;

    _btmLeft->getValueAtTime(time, ret.x, ret.y);

    return ret;
}

void
RectangleInteract::setValue(OfxPointD btmLeft,
                            OfxPointD size,
                            const OfxPointD &pscale)
{
    // round newx/y to the closest int, 1/10 int, etc
    // this make parameter editing easier
    switch (_mouseState) {
    case eMouseStateIdle:
        break;
    case eMouseStateDraggingTopLeft:
        btmLeft.x = fround(btmLeft.x, pscale.x);
        size.x = fround(size.x, pscale.x);
        size.y = fround(size.y, pscale.y);
        break;
    case eMouseStateDraggingTopRight:
        size.x = fround(size.x, pscale.x);
        size.y = fround(size.y, pscale.y);
        break;
    case eMouseStateDraggingBtmLeft:
        btmLeft.x = fround(btmLeft.x, pscale.x);
        btmLeft.y = fround(btmLeft.y, pscale.y);
        size.x = fround(size.x, pscale.x);
        size.y = fround(size.y, pscale.y);
        break;
    case eMouseStateDraggingBtmRight:
        size.x = fround(size.x, pscale.x);
        size.y = fround(size.y, pscale.y);
        btmLeft.y = fround(btmLeft.y, pscale.y);
        break;
    case eMouseStateDraggingCenter:
        btmLeft.x = fround(btmLeft.x, pscale.x);
        btmLeft.y = fround(btmLeft.y, pscale.y);
        break;
    case eMouseStateDraggingTopMid:
        size.y = fround(size.y, pscale.y);
        break;
    case eMouseStateDraggingMidRight:
        size.x = fround(size.x, pscale.x);
        break;
    case eMouseStateDraggingBtmMid:
        btmLeft.y = fround(btmLeft.y, pscale.y);
        break;
    case eMouseStateDraggingMidLeft:
        btmLeft.x = fround(btmLeft.x, pscale.x);
        break;
    }
    _effect->beginEditBlock("setRectangle");
    _btmLeft->setValue(btmLeft.x, btmLeft.y);
    _size->setValue(size.x, size.y);
    _effect->endEditBlock();
} // penDown

