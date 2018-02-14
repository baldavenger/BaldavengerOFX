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
 * OFX utilities for tracking.
 */

#ifndef openfx_supportext_ofxsTracking_h
#define openfx_supportext_ofxsTracking_h

#include "ofxsImageEffect.h"
#ifdef OFX_EXTENSIONS_NATRON
#include "ofxNatron.h"
#endif
#include "ofxsMacros.h"

#define kParamTrackingCenterPoint "center"
#define kParamTrackingCenterPointLabel "Center"
#define kParamTrackingCenterPointHint "The center point to track"

#define kParamTrackingOffset "offset"
#define kParamTrackingOffsetLabel "Offset"
#define kParamTrackingOffsetHint "The offset applied to the center point relative to the real tracked position"

#define kParamTrackingReferenceFrame "refFrame"
#define kParamTrackingReferenceFrameLabel "Reference Frame"
#define kParamTrackingReferenceFrameHint "The frame from which the pattern should be sampled"

#define kParamTrackingEnableReferenceFrame "enableRefFrame"
#define kParamTrackingEnableReferenceFrameLabel "Enable Reference Frame"
#define kParamTrackingEnableReferenceFrameHint "When checked, the reference frame will be the frame indicated by the Reference Frame parameter"

#define kParamTrackingCorrelationScore "correlation"
#define kParamTrackingCorrelationScoreLabel "Correlation"
#define kParamTrackingCorrelationScoreHint "The correlation score of the track with respect to the pattern"

#define kParamTrackingPatternBoxBtmLeft "patternBoxBtmLeft"
#define kParamTrackingPatternBoxBtmLeftLabel "Pattern Bottom Left"
#define kParamTrackingPatternBoxBtmLeftHint "The bottom left corner of the inner pattern box. The coordinates are relative to the center point."

#define kParamTrackingPatternBoxTopRight "patternBoxTopRight"
#define kParamTrackingPatternBoxTopRightLabel "Pattern Top Right"
#define kParamTrackingPatternBoxTopRightHint "The top right corner of the inner pattern box. The coordinates are relative to the center point."

#define kParamTrackingSearchBoxBtmLeft "searchBoxBtmLeft"
#define kParamTrackingSearchBoxBtmLeftLabel "Search Area Bottom Left"
#define kParamTrackingSearchBoxBtmLeftHint "The bottom left corner of the search area. The coordinates are relative to the center point."

#define kParamTrackingSearchBoxTopRight "searchBoxTopRight"
#define kParamTrackingSearchBoxTopRightLabel "Search Area Top Right"
#define kParamTrackingSearchBoxTopRightHint "The top right corner of the search area. The coordinates are relative to the center point."

#define kParamTrackingPrevious kNatronParamTrackingPrevious
#define kParamTrackingPreviousLabel "Track Previous"
#define kParamTrackingPreviousHint "Track pattern to previous frame"

#define kParamTrackingNext kNatronParamTrackingNext
#define kParamTrackingNextLabel "Track Next"
#define kParamTrackingNextHint "Track pattern to next frame"

#define kParamTrackingBackward kNatronParamTrackingBackward
#define kParamTrackingBackwardLabel "Track Backward"
#define kParamTrackingBackwardHint "Track pattern to the beginning of the sequence"

#define kParamTrackingForward kNatronParamTrackingForward
#define kParamTrackingForwardLabel "Track Forward"
#define kParamTrackingForwardHint "Track pattern to the end of the sequence"

#define kParamTrackingLabel kNatronOfxParamStringSublabelName // defined in ofxNatron.h
#define kParamTrackingLabelLabel "Track Name"
#define kParamTrackingLabelHint "The name of the track, as it appears in the user interface."
#define kParamTrackingLabelDefault "Track"

namespace OFX {
struct TrackArguments
{
    ///first is not necesserarily lesser than last.
    OfxTime first;     //<! the first frame to track *from*
    OfxTime last;     //<! the last frame to track *from* (can be the same as first)
    bool forward;     //<! tracking direction
    InstanceChangeReason reason;
    OfxPointD renderScale;
};

class GenericTrackerPlugin
    : public OFX::ImageEffect
{
public:
    /** @brief ctor */
    GenericTrackerPlugin(OfxImageEffectHandle handle);

    /**
     * @brief Nothing to do since we're identity. The host should always render the image of the input.
     **/
    virtual void render(const OFX::RenderArguments & /*args*/) OVERRIDE
    {
    }

    /**
     * @brief Returns true always at the same time and for the source clip.
     **/
    virtual bool isIdentity(const OFX::IsIdentityArguments &args, OFX::Clip * &identityClip, double &identityTime) OVERRIDE;

    /**
     * @brief Handles the push buttons actions.
     **/
    virtual void changedParam(const OFX::InstanceChangedArgs &args, const std::string &paramName) OVERRIDE;
    virtual bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments &args, OfxRectD &rod) OVERRIDE;

protected:

    /**
     * @brief Override to track the entire range between [first,last].
     * @param forward If true then it should track from first to last, otherwise it should track
     * from last to first.
     * @param currentTime The current time at which the track has been requested.
     **/
    virtual void trackRange(const OFX::TrackArguments & args) = 0;

    // do not need to delete these, the ImageEffect is managing them for us
    OFX::Clip *_dstClip;
    OFX::Clip *_srcClip;
    OFX::PushButtonParam* _backwardButton;
    OFX::PushButtonParam* _prevButton;
    OFX::PushButtonParam* _nextButton;
    OFX::PushButtonParam* _forwardButton;
    OFX::StringParam* _instanceName;
};

void genericTrackerDescribe(OFX::ImageEffectDescriptor &desc);

OFX::PageParamDescriptor* genericTrackerDescribeInContextBegin(OFX::ImageEffectDescriptor &desc, OFX::ContextEnum context);

void genericTrackerDescribePointParameters(OFX::ImageEffectDescriptor &desc, OFX::PageParamDescriptor* page);

/**
 * @brief This class represents the interact associated with one track.
 * It is composed of the following elements:
 * - A point which is the center point of the pattern to track
 * - An inner rectangle which defines the bounding box of the pattern to track
 * - An outer rectangle which defines the region where we should look for the pattern in the previous/following frames.
 *
 * The inner and outer rectangle are defined respectively by their bottom left corner and their size (width/height).
 * The bottom left corner of these rectangles defines an offset relative to the center point instead of absolute coordinates.
 * It makes it really easier everywhere in the tracker to manipulate coordinates.
 **/
class TrackerRegionInteract
    : public OFX::OverlayInteract
{
    enum MouseStateEnum
    {
        eMouseStateIdle = 0,
        eMouseStateDraggingCenter,
        eMouseStateDraggingOffset,

        eMouseStateDraggingInnerTopLeft,
        eMouseStateDraggingInnerTopRight,
        eMouseStateDraggingInnerBtmLeft,
        eMouseStateDraggingInnerBtmRight,
        eMouseStateDraggingInnerTopMid,
        eMouseStateDraggingInnerMidRight,
        eMouseStateDraggingInnerBtmMid,
        eMouseStateDraggingInnerMidLeft,

        eMouseStateDraggingOuterTopLeft,
        eMouseStateDraggingOuterTopRight,
        eMouseStateDraggingOuterBtmLeft,
        eMouseStateDraggingOuterBtmRight,
        eMouseStateDraggingOuterTopMid,
        eMouseStateDraggingOuterMidRight,
        eMouseStateDraggingOuterBtmMid,
        eMouseStateDraggingOuterMidLeft
    };

    enum DrawStateEnum
    {
        eDrawStateInactive = 0,
        eDrawStateHoveringCenter,

        eDrawStateHoveringInnerTopLeft,
        eDrawStateHoveringInnerTopRight,
        eDrawStateHoveringInnerBtmLeft,
        eDrawStateHoveringInnerBtmRight,
        eDrawStateHoveringInnerTopMid,
        eDrawStateHoveringInnerMidRight,
        eDrawStateHoveringInnerBtmMid,
        eDrawStateHoveringInnerMidLeft,

        eDrawStateHoveringOuterTopLeft,
        eDrawStateHoveringOuterTopRight,
        eDrawStateHoveringOuterBtmLeft,
        eDrawStateHoveringOuterBtmRight,
        eDrawStateHoveringOuterTopMid,
        eDrawStateHoveringOuterMidRight,
        eDrawStateHoveringOuterBtmMid,
        eDrawStateHoveringOuterMidLeft
    };

public:

    TrackerRegionInteract(OfxInteractHandle handle,
                          OFX::ImageEffect* effect)
        : OFX::OverlayInteract(handle)
        , _lastMousePos()
        , _ms(eMouseStateIdle)
        , _ds(eDrawStateInactive)
        , _center(0)
        , _offset(0)
        , _innerBtmLeft(0)
        , _innerTopRight(0)
        , _outerBtmLeft(0)
        , _outerTopRight(0)
        , _name(0)
        , _centerDragPos()
        , _offsetDragPos()
        , _innerBtmLeftDragPos()
        , _innerTopRightDragPos()
        , _outerBtmLeftDragPos()
        , _outerTopRightDragPos()
        , _controlDown(false)
        , _altDown(0)
    {
        _center = effect->fetchDouble2DParam(kParamTrackingCenterPoint);
        _offset = effect->fetchDouble2DParam(kParamTrackingOffset);
        _innerBtmLeft = effect->fetchDouble2DParam(kParamTrackingPatternBoxBtmLeft);
        _innerTopRight = effect->fetchDouble2DParam(kParamTrackingPatternBoxTopRight);
        _outerBtmLeft = effect->fetchDouble2DParam(kParamTrackingSearchBoxBtmLeft);
        _outerTopRight = effect->fetchDouble2DParam(kParamTrackingSearchBoxTopRight);
        _name = effect->fetchStringParam(kNatronOfxParamStringSublabelName);
        addParamToSlaveTo(_center);
        addParamToSlaveTo(_offset);
        addParamToSlaveTo(_innerBtmLeft);
        addParamToSlaveTo(_innerTopRight);
        addParamToSlaveTo(_outerBtmLeft);
        addParamToSlaveTo(_outerTopRight);
        addParamToSlaveTo(_name);
    }

    // overridden functions from OFX::Interact to do things
    virtual bool draw(const OFX::DrawArgs &args);
    virtual bool penMotion(const OFX::PenArgs &args);
    virtual bool penDown(const OFX::PenArgs &args);
    virtual bool penUp(const OFX::PenArgs &args);
    virtual bool keyDown(const OFX::KeyArgs &args);
    virtual bool keyUp(const OFX::KeyArgs &args);
    virtual void loseFocus(const OFX::FocusArgs &args);

private:
    bool isDraggingInnerPoint() const;
    bool isDraggingOuterPoint() const;

    OfxPointD _lastMousePos;
    MouseStateEnum _ms;
    DrawStateEnum _ds;
    OFX::Double2DParam* _center;
    OFX::Double2DParam* _offset;
    OFX::Double2DParam* _innerBtmLeft;
    OFX::Double2DParam* _innerTopRight;
    OFX::Double2DParam* _outerBtmLeft;
    OFX::Double2DParam* _outerTopRight;
    OFX::StringParam* _name;
    OfxPointD _centerDragPos;
    OfxPointD _offsetDragPos;

    ///Here the btm left points are NOT relative to the center
    ///The offset is applied to this points
    OfxPointD _innerBtmLeftDragPos;
    OfxPointD _innerTopRightDragPos;
    OfxPointD _outerBtmLeftDragPos;
    OfxPointD _outerTopRightDragPos;
    int _controlDown;
    int _altDown;
};

class TrackerRegionOverlayDescriptor
    : public OFX::DefaultEffectOverlayDescriptor<TrackerRegionOverlayDescriptor, TrackerRegionInteract>
{
};
} // OFX

#endif /* defined(openfx_supportext_ofxsTracking_h) */
