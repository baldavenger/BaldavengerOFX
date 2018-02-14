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
 * OFX Transform3x3 plugin: a base plugin for 2D homographic transform,
 * represented by a 3x3 matrix.
 */

/*
   Although the indications from nuke/fnOfxExtensions.h were followed, and the
   kFnOfxImageEffectActionGetTransform action was implemented in the Support
   library, that action is never called by the Nuke host.

   The extension was implemented as specified in Natron and in the Support library.

   @see gHostDescription.canTransform, ImageEffectDescriptor::setCanTransform(),
   and ImageEffect::getTransform().

   There is also an open question about how the last plugin in a transform chain
   may get the concatenated transform from upstream, the untransformed source image,
   concatenate its own transform and apply the resulting transform in its render
   action.

   Our solution is to have kFnOfxImageEffectCanTransform set on source clips for which
   a transform can be attached to fetched images.
   @see ClipDescriptor::setCanTransform().

   In this case, images fetched from the host may have a kFnOfxPropMatrix2D attached,
   which must be combined with the transformation applied by the effect (which
   may be any deformation function, not only a homography).
   @see ImageBase::getTransform() and ImageBase::getTransformIsIdentity
 */
// Uncomment the following to enable the experimental host transform code.
#define ENABLE_HOST_TRANSFORM

#include <cfloat> // DBL_MAX
#include <memory>
#include <algorithm>

#include "ofxsTransform3x3.h"
#include "ofxsTransform3x3Processor.h"
#include "ofxsCoords.h"
#include "ofxsShutter.h"


#ifndef ENABLE_HOST_TRANSFORM
#undef OFX_EXTENSIONS_NUKE // host transform is the only nuke extension used
#endif

#ifdef OFX_EXTENSIONS_NUKE
#include "nuke/fnOfxExtensions.h"
#endif

#define kSupportsTiles 1
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 1
#define kRenderThreadSafety eRenderFullySafe

using namespace OFX;

using std::string;

// It would be nice to be able to cache the set of transforms (with motion blur) used to compute the
// current frame between two renders.
// Unfortunately, we cannot rely on the host sending changedParam() when the animation changes
// (Nuke doesn't call the action when a linked animation is changed),
// nor on dst->getUniqueIdentifier (which is "ffffffffffffffff" on Nuke)

#define kTransform3x3MotionBlurCount 1000 // number of transforms used in the motion

namespace OFX {
Transform3x3Plugin::Transform3x3Plugin(OfxImageEffectHandle handle,
                                       bool masked,
                                       Transform3x3ParamsTypeEnum paramsType)
    : ImageEffect(handle)
    , _dstClip(0)
    , _srcClip(0)
    , _maskClip(0)
    , _paramsType(paramsType)
    , _invert(0)
    , _filter(0)
    , _clamp(0)
    , _blackOutside(0)
    , _motionblur(0)
    , _dirBlurAmount(0)
    , _dirBlurCentered(0)
    , _dirBlurFading(0)
    , _directionalBlur(0)
    , _shutter(0)
    , _shutteroffset(0)
    , _shuttercustomoffset(0)
    , _masked(masked)
    , _mix(0)
    , _maskApply(0)
    , _maskInvert(0)
{
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    assert(1 <= _dstClip->getPixelComponentCount() && _dstClip->getPixelComponentCount() <= 4);
    _srcClip = getContext() == eContextGenerator ? NULL : fetchClip(kOfxImageEffectSimpleSourceClipName);
    assert( !_srcClip || !_srcClip->isConnected() || (1 <= _srcClip->getPixelComponentCount() && _srcClip->getPixelComponentCount() <= 4) );
    // name of mask clip depends on the context
    if (masked) {
        _maskClip = fetchClip(getContext() == eContextPaint ? "Brush" : "Mask");
        assert(!_maskClip || !_maskClip->isConnected() || _maskClip->getPixelComponents() == ePixelComponentAlpha);
    }

    if ( paramExists(kParamTransform3x3Invert) ) {
        // Transform3x3-GENERIC
        _invert = fetchBooleanParam(kParamTransform3x3Invert);
        // GENERIC
        _filter = fetchChoiceParam(kParamFilterType);
        _clamp = fetchBooleanParam(kParamFilterClamp);
        _blackOutside = fetchBooleanParam(kParamFilterBlackOutside);
        assert(_invert && _filter && _clamp && _blackOutside);
        if ( paramExists(kParamTransform3x3MotionBlur) ) {
            _motionblur = fetchDoubleParam(kParamTransform3x3MotionBlur); // GodRays may not have have _motionblur
            assert(_motionblur);
        }
        if (paramsType == eTransform3x3ParamsTypeMotionBlur) {
            _directionalBlur = fetchBooleanParam(kParamTransform3x3DirectionalBlur);
            _shutter = fetchDoubleParam(kParamShutter);
            _shutteroffset = fetchChoiceParam(kParamShutterOffset);
            _shuttercustomoffset = fetchDoubleParam(kParamShutterCustomOffset);
            assert(_directionalBlur && _shutter && _shutteroffset && _shuttercustomoffset);
        }
        if (masked) {
            _mix = fetchDoubleParam(kParamMix);
            _maskInvert = fetchBooleanParam(kParamMaskInvert);
            assert(_mix && _maskInvert);
        }

        if (paramsType == eTransform3x3ParamsTypeMotionBlur) {
            bool directionalBlur;
            _directionalBlur->getValue(directionalBlur);
            _shutter->setEnabled(!directionalBlur);
            _shutteroffset->setEnabled(!directionalBlur);
            _shuttercustomoffset->setEnabled(!directionalBlur);
        }
    }
}

Transform3x3Plugin::~Transform3x3Plugin()
{
}

////////////////////////////////////////////////////////////////////////////////
/** @brief render for the filter */

////////////////////////////////////////////////////////////////////////////////
// basic plugin render function, just a skelington to instantiate templates from

/* set up and run a processor */
void
Transform3x3Plugin::setupAndProcess(Transform3x3ProcessorBase &processor,
                                    const RenderArguments &args)
{
    assert(!_invert || _motionblur); // this method should be overridden in GodRays
    const double time = args.time;
    std::auto_ptr<Image> dst( _dstClip->fetchImage(time) );

    if ( !dst.get() ) {
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }
    BitDepthEnum dstBitDepth    = dst->getPixelDepth();
    PixelComponentEnum dstComponents  = dst->getPixelComponents();
    if ( ( dstBitDepth != _dstClip->getPixelDepth() ) ||
         ( dstComponents != _dstClip->getPixelComponents() ) ) {
        setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong depth or components");
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }
    if ( (dst->getRenderScale().x != args.renderScale.x) ||
         ( dst->getRenderScale().y != args.renderScale.y) ||
         ( ( (dst->getField() != eFieldNone) /* for DaVinci Resolve */ && (dst->getField() != args.fieldToRender) ) ) ) {
        setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }
    std::auto_ptr<const Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                    _srcClip->fetchImage(args.time) : 0 );
    size_t invtransformsizealloc = 0;
    size_t invtransformsize = 0;
    std::vector<Matrix3x3> invtransform;
    std::vector<double> invtransformalpha;
    double motionblur = 0.;
    bool directionalBlur = (_paramsType != eTransform3x3ParamsTypeNone);
    double amountFrom = 0.;
    double amountTo = 1.;
    if (_dirBlurAmount) {
        _dirBlurAmount->getValueAtTime(time, amountTo);
    }
    if (_dirBlurCentered) {
        bool centered;
        _dirBlurCentered->getValueAtTime(time, centered);
        if (centered) {
            amountFrom = -amountTo;
        }
    }
    bool blackOutside = false;
    double mix = 1.;

    if ( !src.get() ) {
        // no source image, use a dummy transform
        invtransformsizealloc = 1;
        invtransform.resize(invtransformsizealloc);
        invtransformsize = 1;
        invtransform[0](0,0) = 0.;
        invtransform[0](0,1) = 0.;
        invtransform[0](0,2) = 0.;
        invtransform[0](1,0) = 0.;
        invtransform[0](1,1) = 0.;
        invtransform[0](1,2) = 0.;
        invtransform[0](2,0) = 0.;
        invtransform[0](2,1) = 0.;
        invtransform[0](2,2) = 1.;
    } else {
        BitDepthEnum dstBitDepth       = dst->getPixelDepth();
        PixelComponentEnum dstComponents  = dst->getPixelComponents();
        BitDepthEnum srcBitDepth      = src->getPixelDepth();
        PixelComponentEnum srcComponents = src->getPixelComponents();
        if ( (srcBitDepth != dstBitDepth) || (srcComponents != dstComponents) ) {
            throwSuiteStatusException(kOfxStatFailed);

            return;
        }

        bool invert = false;
        if (_invert) {
            _invert->getValueAtTime(time, invert);
        }

        if (_blackOutside) {
            _blackOutside->getValueAtTime(time, blackOutside);
        }
        if (_masked && _mix) {
            _mix->getValueAtTime(time, mix);
        }
        if (_motionblur) {
            _motionblur->getValueAtTime(time, motionblur);
        }
        if (_directionalBlur) {
            _directionalBlur->getValueAtTime(time, directionalBlur);
        }
        double shutter = 0.;
        if (!directionalBlur) {
            if (_shutter) {
                _shutter->getValueAtTime(time, shutter);
            }
        }
        const bool fielded = args.fieldToRender == eFieldLower || args.fieldToRender == eFieldUpper;
        const double srcpixelAspectRatio = src->getPixelAspectRatio();
        const double dstpixelAspectRatio = _dstClip->getPixelAspectRatio();

        if ( (shutter != 0.) && (motionblur != 0.) ) {
            invtransformsizealloc = kTransform3x3MotionBlurCount;
            invtransform.resize(invtransformsizealloc);
            assert(_shutteroffset);
            ShutterOffsetEnum shutteroffset = (ShutterOffsetEnum)_shutteroffset->getValueAtTime(time);
            double shuttercustomoffset;
            assert(_shuttercustomoffset);
            _shuttercustomoffset->getValueAtTime(time, shuttercustomoffset);

            invtransformsize = getInverseTransforms(time, args.renderView, args.renderScale, fielded, srcpixelAspectRatio, dstpixelAspectRatio, invert, shutter, shutteroffset, shuttercustomoffset, &invtransform.front(), invtransformsizealloc);
        } else if (directionalBlur) {
            invtransformsizealloc = kTransform3x3MotionBlurCount;
            invtransform.resize(invtransformsizealloc);
            invtransformalpha.resize(invtransformsizealloc);
            invtransformsize = getInverseTransformsBlur(time, args.renderView, args.renderScale, fielded, srcpixelAspectRatio, dstpixelAspectRatio, invert, amountFrom, amountTo, &invtransform.front(), &invtransformalpha.front(), invtransformsizealloc);
            // normalize alpha, and apply gamma
            double fading = 0.;
            if (_dirBlurFading) {
                _dirBlurFading->getValueAtTime(time, fading);
            }
            if (fading <= 0.) {
                std::fill(invtransformalpha.begin(), invtransformalpha.end(), 1.);
            } else {
                for (size_t i = 0; i < invtransformalpha.size(); ++i) {
                    invtransformalpha[i] = std::pow(1. - std::abs(invtransformalpha[i]) / amountTo, fading);
                }
            }
        } else {
            invtransformsizealloc = 1;
            invtransform.resize(invtransformsizealloc);
            invtransformsize = 1;
            bool success = getInverseTransformCanonical(time, args.renderView, 1., invert, &invtransform[0]); // virtual function
            if (!success) {
                invtransform[0](0,0) = 0.;
                invtransform[0](0,1) = 0.;
                invtransform[0](0,2) = 0.;
                invtransform[0](1,0) = 0.;
                invtransform[0](1,1) = 0.;
                invtransform[0](1,2) = 0.;
                invtransform[0](2,0) = 0.;
                invtransform[0](2,1) = 0.;
                invtransform[0](2,2) = 1.;
            } else {
                Matrix3x3 canonicalToPixel = ofxsMatCanonicalToPixel(srcpixelAspectRatio, args.renderScale.x,
                                                                     args.renderScale.y, fielded);
                Matrix3x3 pixelToCanonical = ofxsMatPixelToCanonical(dstpixelAspectRatio,  args.renderScale.x,
                                                                     args.renderScale.y, fielded);
                invtransform[0] = canonicalToPixel * invtransform[0] * pixelToCanonical;
            }
        }
        if (invtransformsize == 1) {
            motionblur  = 0.;
        }
#ifdef OFX_EXTENSIONS_NUKE
        // compose with the input transform
        if ( !src->getTransformIsIdentity() ) {
            double srcTransform[9]; // transform to apply to the source image, in pixel coordinates, from source to destination
            src->getTransform(srcTransform);
            Matrix3x3 srcTransformMat;
            srcTransformMat(0,0) = srcTransform[0];
            srcTransformMat(0,1) = srcTransform[1];
            srcTransformMat(0,2) = srcTransform[2];
            srcTransformMat(1,0) = srcTransform[3];
            srcTransformMat(1,1) = srcTransform[4];
            srcTransformMat(1,2) = srcTransform[5];
            srcTransformMat(2,0) = srcTransform[6];
            srcTransformMat(2,1) = srcTransform[7];
            srcTransformMat(2,2) = srcTransform[8];
            // invert it
            Matrix3x3 srcTransformInverse;
            if ( srcTransformMat.inverse(&srcTransformInverse) ) {
                for (size_t i = 0; i < invtransformsize; ++i) {
                    invtransform[i] = srcTransformInverse * invtransform[i];
                }
            }
        }
#endif
    }

    // auto ptr for the mask.
    bool doMasking = ( _masked && ( !_maskApply || _maskApply->getValueAtTime(args.time) ) && _maskClip && _maskClip->isConnected() );
    std::auto_ptr<const Image> mask(doMasking ? _maskClip->fetchImage(args.time) : 0);
    if (doMasking) {
        bool maskInvert = false;
        if (_maskInvert) {
            _maskInvert->getValueAtTime(time, maskInvert);
        }
        // say we are masking
        processor.doMasking(true);

        // Set it in the processor
        processor.setMaskImg(mask.get(), maskInvert);
    }

    // set the images
    processor.setDstImg( dst.get() );
    processor.setSrcImg( src.get() );

    // set the render window
    processor.setRenderWindow(args.renderWindow);
    assert(invtransform.size() && invtransformsize);
    processor.setValues(&invtransform.front(),
                        invtransformalpha.empty() ? 0 : &invtransformalpha.front(),
                        invtransformsize,
                        blackOutside,
                        motionblur,
                        mix);

    // Call the base class process member, this will call the derived templated process code
    processor.process();
} // setupAndProcess

// Compute the bounding box of the transform of four points representing a quadrilateral.
// - If a point has a negative Z, it is not taken into account (thus, if all points have a negative Z, the region is empty)
// - If two consecutive points have a chage of Z-sign, find a point close to the point with the negative z with a Z equal to 0.
// This gives a direction which must be fully included in the rod, and thus a full quadrant must be included, ie an infinite
// value for one of the x bounds and an infinite value for one of the y bounds.
#warning TODO
static void
ofxsTransformRegionFromPoints(const Point3D p[4],
                              OfxRectD &rod)
{
    // extract the x/y bounds
    double x1, y1, x2, y2;
    bool empty = true;
    
    int i = 0, j = 1;
    for (; i < 4; ++i, j = (j + 1) % 4) {
        if (p[i].z > 0) {
            double x = p[i].x / p[i].z;
            double y = p[i].y / p[i].z;

            if (empty) {
                empty = false;
                x1 = x2 = x;
                y1 = y2 = y;
            } else {
                if (x < x1) {
                    x1 = x;
                } else if (x > x2) {
                    x2 = x;
                }
                if (y < y1) {
                    y1 = y;
                } else if (y > y2) {
                    y2 = y;
                }
            }
        } else if (p[j].z > 0) {
            // add next point if set is empty
            if (empty) {
                empty = false;
                x1 = x2 = p[j].x / p[j].z;
                y1 = y2 = p[j].y / p[j].z;
            }
        }
        if ( (p[i].z > 0 && p[j].z <= 0) ||
             (p[i].z <= 0 && p[j].z > 0) ) {
            // find position of the z=0 point on the line
            double a = -p[i].z / (p[j].z - p[i].z);
            // compute infinite direction
            double dx = p[i].x + a * (p[j].x - p[i].x);
            double dy = p[i].y + a * (p[j].y - p[i].y);
            // add next point if set is empty
            if (empty) {
                empty = false;
                x1 = x2 = p[j].x / p[j].z;
                y1 = y2 = p[j].y / p[j].z;
            }
            // add the full quadrant
            if (dx < 0) {
                x1 = kOfxFlagInfiniteMin;
            } else if (dx > 0) {
                x2 = kOfxFlagInfiniteMax;
            }
            if (dy < 0) {
                y1 = kOfxFlagInfiniteMin;
            } else if (dy > 0) {
                y2 = kOfxFlagInfiniteMax;
            }
        }
    }

    if (empty) {
        rod.x1 = rod.x2 = rod.y1 = rod.y2 = 0;
    } else {
        rod.x1 = x1;
        rod.x2 = x2;
        rod.y1 = y1;
        rod.y2 = y2;
    }
    assert(rod.x1 <= rod.x2 && rod.y1 <= rod.y2);
} // ofxsTransformRegionFromPoints

// compute the bounding box of the transform of a rectangle
static void
ofxsTransformRegionFromRoD(const OfxRectD &srcRoD,
                           const Matrix3x3 &transform,
                           Point3D p[4],
                           OfxRectD &rod)
{
    /// now transform the 4 corners of the source clip to the output image
    p[0] = transform * Point3D(srcRoD.x1, srcRoD.y1, 1);
    p[1] = transform * Point3D(srcRoD.x1, srcRoD.y2, 1);
    p[2] = transform * Point3D(srcRoD.x2, srcRoD.y2, 1);
    p[3] = transform * Point3D(srcRoD.x2, srcRoD.y1, 1);

    ofxsTransformRegionFromPoints(p, rod);
}

void
Transform3x3Plugin::transformRegion(const OfxRectD &rectFrom,
                                    double time,
                                    int view,
                                    bool invert,
                                    double motionblur,
                                    bool directionalBlur,
                                    double amountFrom,
                                    double amountTo,
                                    double shutter,
                                    ShutterOffsetEnum shutteroffset,
                                    double shuttercustomoffset,
                                    bool isIdentity,
                                    OfxRectD *rectTo)
{
    // Algorithm:
    // - Compute positions of the four corners at start and end of shutter, and every multiple of 0.25 within this range.
    // - Update the bounding box from these positions.
    // - At the end, expand the bounding box by the maximum L-infinity distance between consecutive positions of each corner.

    OfxRangeD range;
    bool hasmotionblur = ( (shutter != 0. || directionalBlur) && motionblur != 0. );

    if (hasmotionblur && !directionalBlur) {
        shutterRange(time, shutter, shutteroffset, shuttercustomoffset, &range);
    } else {
        ///if is identity return the input rod instead of transforming
        if (isIdentity) {
            *rectTo = rectFrom;

            return;
        }
        range.min = range.max = time;
    }

    // initialize with a super-empty RoD (note that max and min are reversed)
    rectTo->x1 = kOfxFlagInfiniteMax;
    rectTo->x2 = kOfxFlagInfiniteMin;
    rectTo->y1 = kOfxFlagInfiniteMax;
    rectTo->y2 = kOfxFlagInfiniteMin;
    double t = range.min;
    bool first = true;
    bool last = !hasmotionblur; // ony one iteration if there is no motion blur
    bool finished = false;
    double expand = 0.;
    double amount = 1.;
    int dirBlurIter = 0;
    Point3D p_prev[4];
    while (!finished) {
        // compute transformed positions
        OfxRectD thisRoD;
        Matrix3x3 transform;
        bool success = getInverseTransformCanonical(t, view, amountFrom + amount * (amountTo - amountFrom), invert, &transform); // RoD is computed using the *DIRECT* transform, which is why we use !invert
        if (!success) {
            // return infinite region
            rectTo->x1 = kOfxFlagInfiniteMin;
            rectTo->x2 = kOfxFlagInfiniteMax;
            rectTo->y1 = kOfxFlagInfiniteMin;
            rectTo->y2 = kOfxFlagInfiniteMax;

            return;
        }
        Point3D p[4];
        ofxsTransformRegionFromRoD(rectFrom, transform, p, thisRoD);

        // update min/max
        Coords::rectBoundingBox(*rectTo, thisRoD, rectTo);

        // if first iteration, continue
        if (first) {
            first = false;
        } else {
            // compute the L-infinity distance between consecutive tested points
            expand = std::max( expand, std::fabs(p_prev[0].x - p[0].x) );
            expand = std::max( expand, std::fabs(p_prev[0].y - p[0].y) );
            expand = std::max( expand, std::fabs(p_prev[1].x - p[1].x) );
            expand = std::max( expand, std::fabs(p_prev[1].y - p[1].y) );
            expand = std::max( expand, std::fabs(p_prev[2].x - p[2].x) );
            expand = std::max( expand, std::fabs(p_prev[2].y - p[2].y) );
            expand = std::max( expand, std::fabs(p_prev[3].x - p[3].x) );
            expand = std::max( expand, std::fabs(p_prev[3].y - p[3].y) );
        }

        if (last) {
            finished = true;
        } else {
            // prepare for next iteration
            p_prev[0] = p[0];
            p_prev[1] = p[1];
            p_prev[2] = p[2];
            p_prev[3] = p[3];
            if (directionalBlur) {
                const int dirBlurIterMax = 8;
                ++dirBlurIter;
                amount = 1. - dirBlurIter / (double)dirBlurIterMax;
                last = dirBlurIter == dirBlurIterMax;
            } else {
                t = std::floor(t * 4 + 1) / 4; // next quarter-frame
                if (t >= range.max) {
                    // last iteration should be done with range.max
                    t = range.max;
                    last = true;
                }
            }
        }
    }
    // expand to take into account errors due to motion blur
    if (rectTo->x1 > kOfxFlagInfiniteMin) {
        rectTo->x1 -= expand;
    }
    if (rectTo->x2 < kOfxFlagInfiniteMax) {
        rectTo->x2 += expand;
    }
    if (rectTo->y1 > kOfxFlagInfiniteMin) {
        rectTo->y1 -= expand;
    }
    if (rectTo->y2 < kOfxFlagInfiniteMax) {
        rectTo->y2 += expand;
    }
} // transformRegion

// override the rod call
// Transform3x3-GENERIC
// the RoD should at least contain the region of definition of the source clip,
// which will be filled with black or by continuity.
bool
Transform3x3Plugin::getRegionOfDefinition(const RegionOfDefinitionArguments &args,
                                          OfxRectD &rod)
{
    if (!_srcClip || !_srcClip->isConnected()) {
        return false;
    }
    const double time = args.time;
    const OfxRectD& srcRoD = _srcClip->getRegionOfDefinition(time);

    if ( Coords::rectIsInfinite(srcRoD) ) {
        // return an infinite RoD
        rod.x1 = kOfxFlagInfiniteMin;
        rod.x2 = kOfxFlagInfiniteMax;
        rod.y1 = kOfxFlagInfiniteMin;
        rod.y2 = kOfxFlagInfiniteMax;

        return true;
    }

    if ( Coords::rectIsEmpty(srcRoD) ) {
        // return an empty RoD
        rod.x1 = 0.;
        rod.x2 = 0.;
        rod.y1 = 0.;
        rod.y2 = 0.;

        return true;
    }

    double mix = 1.;
    bool doMasking = ( ( !_maskApply || _maskApply->getValueAtTime(args.time) ) && _maskClip && _maskClip->isConnected() );
    if (doMasking && _mix) {
        _mix->getValueAtTime(time, mix);
        if (mix == 0.) {
            // identity transform
            rod = srcRoD;

            return true;
        }
    }

    bool invert = false;
    if (_invert) {
        _invert->getValueAtTime(time, invert);
    }
    invert = !invert; // only for getRegionOfDefinition
    double motionblur = 1.; // default for GodRays
    if (_motionblur) {
        _motionblur->getValueAtTime(time, motionblur);
    }
    bool directionalBlur = (_paramsType != eTransform3x3ParamsTypeNone);
    double amountFrom = 0.;
    double amountTo = 1.;
    if (_dirBlurAmount) {
        _dirBlurAmount->getValueAtTime(time, amountTo);
    }
    if (_dirBlurCentered) {
        bool centered;
        _dirBlurCentered->getValueAtTime(time, centered);
        if (centered) {
            amountFrom = -amountTo;
        }
    }
    double shutter = 0.;
    ShutterOffsetEnum shutteroffset = eShutterOffsetCentered;
    double shuttercustomoffset = 0.;
    if (_directionalBlur) {
        directionalBlur = _directionalBlur->getValueAtTime(time);
        shutter = _shutter->getValueAtTime(time);
        shutteroffset = (ShutterOffsetEnum)_shutteroffset->getValueAtTime(time);
        shuttercustomoffset = _shuttercustomoffset->getValueAtTime(time);
    }

    bool identity = isIdentity(args.time);

    // set rod from srcRoD
#ifdef OFX_EXTENSIONS_NUKE
    const int view = args.view;
#else
    const int view = 0;
#endif

    transformRegion(srcRoD, time, view, invert, motionblur, directionalBlur, amountFrom, amountTo, shutter, shutteroffset, shuttercustomoffset, identity, &rod);

    // If identity do not expand for black outside, otherwise we would never be able to have identity.
    // We want the RoD to be the same as the src RoD when we are identity.
    if (!identity) {
        bool blackOutside = false;
        if (_blackOutside) {
            _blackOutside->getValueAtTime(time, blackOutside);
        }

        ofxsFilterExpandRoD(this, _dstClip->getPixelAspectRatio(), args.renderScale, blackOutside, &rod);
    }

    if ( doMasking && ( (mix != 1.) || _maskClip->isConnected() ) ) {
        // for masking or mixing, we also need the source image.
        // compute the union of both RODs
        Coords::rectBoundingBox(rod, srcRoD, &rod);
    }

    // say we set it
    return true;
} // getRegionOfDefinition

// override the roi call
// Transform3x3-GENERIC
// Required if the plugin requires a region from the inputs which is different from the rendered region of the output.
// (this is the case for transforms)
// It may be difficult to implement for complicated transforms:
// consequently, these transforms cannot support tiles.
void
Transform3x3Plugin::getRegionsOfInterest(const RegionsOfInterestArguments &args,
                                         RegionOfInterestSetter &rois)
{
    if (!_srcClip || !_srcClip->isConnected()) {
        return;
    }
    const double time = args.time;
    const OfxRectD roi = args.regionOfInterest;
    OfxRectD srcRoI;
    double mix = 1.;
    bool doMasking = ( ( !_maskApply || _maskApply->getValueAtTime(args.time) ) && _maskClip && _maskClip->isConnected() );
    if (doMasking) {
        _mix->getValueAtTime(time, mix);
        if (mix == 0.) {
            // identity transform
            srcRoI = roi;
            rois.setRegionOfInterest(*_srcClip, srcRoI);

            return;
        }
    }

    bool invert = false;
    if (_invert) {
        _invert->getValueAtTime(time, invert);
    }
    //invert = !invert; // only for getRegionOfDefinition
    double motionblur = 1; // default for GodRays
    if (_motionblur) {
        _motionblur->getValueAtTime(time, motionblur);
    }
    bool directionalBlur = (_paramsType != eTransform3x3ParamsTypeNone);
    double amountFrom = 0.;
    double amountTo = 1.;
    if (_dirBlurAmount) {
        _dirBlurAmount->getValueAtTime(time, amountTo);
    }
    if (_dirBlurCentered) {
        bool centered;
        _dirBlurCentered->getValueAtTime(time, centered);
        if (centered) {
            amountFrom = -amountTo;
        }
    }
    double shutter = 0.;
    ShutterOffsetEnum shutteroffset = eShutterOffsetCentered;
    double shuttercustomoffset = 0.;
    if (_directionalBlur) {
        directionalBlur = _directionalBlur->getValueAtTime(time);
        shutter = _shutter->getValueAtTime(time);
        shutteroffset = (ShutterOffsetEnum)_shutteroffset->getValueAtTime(time);
        shuttercustomoffset = _shuttercustomoffset->getValueAtTime(time);
    }
#ifdef OFX_EXTENSIONS_NUKE
    const int view = args.view;
#else
    const int view = 0;
#endif

    // set srcRoI from roi
    transformRegion(roi, time, view, invert, motionblur, directionalBlur, amountFrom, amountTo, shutter, shutteroffset, shuttercustomoffset, isIdentity(time), &srcRoI);

    FilterEnum filter = eFilterCubic;
    if (_filter) {
        filter = (FilterEnum)_filter->getValueAtTime(time);
    }

    assert(srcRoI.x1 <= srcRoI.x2 && srcRoI.y1 <= srcRoI.y2);

    ofxsFilterExpandRoI(roi, _srcClip->getPixelAspectRatio(), args.renderScale, filter, doMasking, mix, &srcRoI);

    if ( Coords::rectIsInfinite(srcRoI) ) {
        // RoI cannot be infinite.
        // This is not a mathematically correct solution, but better than nothing: set to the project size
        OfxPointD size = getProjectSize();
        OfxPointD offset = getProjectOffset();

        if (srcRoI.x1 <= kOfxFlagInfiniteMin) {
            srcRoI.x1 = offset.x;
        }
        if (srcRoI.x2 >= kOfxFlagInfiniteMax) {
            srcRoI.x2 = offset.x + size.x;
        }
        if (srcRoI.y1 <= kOfxFlagInfiniteMin) {
            srcRoI.y1 = offset.y;
        }
        if (srcRoI.y2 >= kOfxFlagInfiniteMax) {
            srcRoI.y2 = offset.y + size.y;
        }
    }

    if ( _masked && (mix != 1.) ) {
        // compute the bounding box with the default ROI
        Coords::rectBoundingBox(srcRoI, args.regionOfInterest, &srcRoI);
    }

    // no need to set it on mask (the default ROI is OK)
    rois.setRegionOfInterest(*_srcClip, srcRoI);
} // getRegionsOfInterest

template <class PIX, int nComponents, int maxValue, bool masked>
void
Transform3x3Plugin::renderInternalForBitDepth(const RenderArguments &args)
{
    const double time = args.time;
    FilterEnum filter = args.renderQualityDraft ? eFilterImpulse : eFilterCubic;

    if (!args.renderQualityDraft && _filter) {
        filter = (FilterEnum)_filter->getValueAtTime(time);
    }
    bool clamp = false;
    if (_clamp) {
        clamp = _clamp->getValueAtTime(time);
    }

    // as you may see below, some filters don't need explicit clamping, since they are
    // "clamped" by construction.
    switch (filter) {
    case eFilterImpulse: {
        Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterImpulse, false> fred(*this);
        setupAndProcess(fred, args);
        break;
    }
    case eFilterBilinear: {
        Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterBilinear, false> fred(*this);
        setupAndProcess(fred, args);
        break;
    }
    case eFilterCubic: {
        Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterCubic, false> fred(*this);
        setupAndProcess(fred, args);
        break;
    }
    case eFilterKeys:
        if (clamp) {
            Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterKeys, true> fred(*this);
            setupAndProcess(fred, args);
        } else {
            Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterKeys, false> fred(*this);
            setupAndProcess(fred, args);
        }
        break;
    case eFilterSimon:
        if (clamp) {
            Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterSimon, true> fred(*this);
            setupAndProcess(fred, args);
        } else {
            Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterSimon, false> fred(*this);
            setupAndProcess(fred, args);
        }
        break;
    case eFilterRifman:
        if (clamp) {
            Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterRifman, true> fred(*this);
            setupAndProcess(fred, args);
        } else {
            Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterRifman, false> fred(*this);
            setupAndProcess(fred, args);
        }
        break;
    case eFilterMitchell:
        if (clamp) {
            Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterMitchell, true> fred(*this);
            setupAndProcess(fred, args);
        } else {
            Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterMitchell, false> fred(*this);
            setupAndProcess(fred, args);
        }
        break;
    case eFilterParzen: {
        Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterParzen, false> fred(*this);
        setupAndProcess(fred, args);
        break;
    }
    case eFilterNotch: {
        Transform3x3Processor<PIX, nComponents, maxValue, masked, eFilterNotch, false> fred(*this);
        setupAndProcess(fred, args);
        break;
    }
    } // switch
} // renderInternalForBitDepth

// the internal render function
template <int nComponents, bool masked>
void
Transform3x3Plugin::renderInternal(const RenderArguments &args,
                                   BitDepthEnum dstBitDepth)
{
    switch (dstBitDepth) {
    case eBitDepthUByte:
        renderInternalForBitDepth<unsigned char, nComponents, 255, masked>(args);
        break;
    case eBitDepthUShort:
        renderInternalForBitDepth<unsigned short, nComponents, 65535, masked>(args);
        break;
    case eBitDepthFloat:
        renderInternalForBitDepth<float, nComponents, 1, masked>(args);
        break;
    default:
        throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

// the overridden render function
void
Transform3x3Plugin::render(const RenderArguments &args)
{
    // instantiate the render code based on the pixel depth of the dst clip
    BitDepthEnum dstBitDepth    = _dstClip->getPixelDepth();
    int dstComponentCount  = _dstClip->getPixelComponentCount();

    assert(1 <= dstComponentCount && dstComponentCount <= 4);
    switch (dstComponentCount) {
    case 4:
        if (_masked) {
            renderInternal<4, true>(args, dstBitDepth);
        } else {
            renderInternal<4, false>(args, dstBitDepth);
        }
        break;
    case 3:
        if (_masked) {
            renderInternal<3, true>(args, dstBitDepth);
        } else {
            renderInternal<3, false>(args, dstBitDepth);
        }
        break;
    case 2:
        if (_masked) {
            renderInternal<2, true>(args, dstBitDepth);
        } else {
            renderInternal<2, false>(args, dstBitDepth);
        }
        break;
    case 1:
        if (_masked) {
            renderInternal<1, true>(args, dstBitDepth);
        } else {
            renderInternal<1, false>(args, dstBitDepth);
        }
        break;
    default:
        break;
    }
}

bool
Transform3x3Plugin::isIdentity(const IsIdentityArguments &args,
                               Clip * &identityClip,
                               double & /*identityTime*/)
{
    const double time = args.time;

    if (_dirBlurAmount) {
        double amount = 1.;
        _dirBlurAmount->getValueAtTime(time, amount);
        if (amount == 0.) {
            identityClip = _srcClip;

            return true;
        }
    }

    // if there is motion blur, we suppose the transform is not identity
    double motionblur = _invert ? 1. : 0.; // default is 1 for GodRays, 0 for Mirror
    if (_motionblur) {
        _motionblur->getValueAtTime(time, motionblur);
    }
    double shutter = 0.;
    if (_shutter) {
        _shutter->getValueAtTime(time, shutter);
    }
    bool hasmotionblur = (shutter != 0. && motionblur != 0.);
    if (hasmotionblur) {
        return false;
    }

    if (_clamp) {
        // if image has values above 1., they will be clamped.
        bool clamp;
        _clamp->getValueAtTime(time, clamp);
        if (clamp) {
            return false;
        }
    }

    if ( isIdentity(time) ) { // let's call the Transform-specific one first
        identityClip = _srcClip;

        return true;
    }

    // GENERIC
    if (_masked) {
        double mix = 1.;
        if (_mix) {
            _mix->getValueAtTime(time, mix);
        }
        if (mix == 0.) {
            identityClip = _srcClip;

            return true;
        }

        bool doMasking = ( ( !_maskApply || _maskApply->getValueAtTime(args.time) ) && _maskClip && _maskClip->isConnected() );
        if (doMasking) {
            bool maskInvert;
            _maskInvert->getValueAtTime(args.time, maskInvert);
            if (!maskInvert) {
                OfxRectI maskRoD;
                if (getImageEffectHostDescription()->supportsMultiResolution) {
                    // In Sony Catalyst Edit, clipGetRegionOfDefinition returns the RoD in pixels instead of canonical coordinates.
                    // In hosts that do not support multiResolution (e.g. Sony Catalyst Edit), all inputs have the same RoD anyway.
                    Coords::toPixelEnclosing(_maskClip->getRegionOfDefinition(args.time), args.renderScale, _maskClip->getPixelAspectRatio(), &maskRoD);
                    // effect is identity if the renderWindow doesn't intersect the mask RoD
                    if ( !Coords::rectIntersection<OfxRectI>(args.renderWindow, maskRoD, 0) ) {
                        identityClip = _srcClip;

                        return true;
                    }
                }
            }
        }
    }

    return false;
} // Transform3x3Plugin::isIdentity

#ifdef OFX_EXTENSIONS_NUKE
// overridden getTransform
bool
Transform3x3Plugin::getTransform(const TransformArguments &args,
                                 Clip * &transformClip,
                                 double transformMatrix[9])
{
    //std::cout << "getTransform called!" << std::endl;

    // Even if the plugin advertizes it cannot transform, getTransform() may be called, e.g. to
    // get a transform for the overlays. We thus always return a transform.
    //assert(!_masked); // this should never get called for masked plugins, since they don't advertise that they can transform
    //if (_masked) {
    //    return false;
    //}

    const double time = args.time;

    if (!args.renderQualityDraft) {
        // first, check if effect has blur, see Transform3x3Plugin::setupAndProcess()
        double motionblur = 0.;
        bool directionalBlur = (_paramsType != eTransform3x3ParamsTypeNone);

        if (_motionblur) {
            _motionblur->getValueAtTime(time, motionblur);
        }
        if (_directionalBlur) {
            _directionalBlur->getValueAtTime(time, directionalBlur);
        }
        double shutter = 0.;
        if (!directionalBlur) {
            if (_shutter) {
                _shutter->getValueAtTime(time, shutter);
            }
        }
        if ( ( (shutter != 0.) && (motionblur != 0.) ) || directionalBlur ) {
            // effect has blur
            return false;
        }
    }

    bool invert = false;

    // Transform3x3-GENERIC
    if (_invert) {
        _invert->getValueAtTime(time, invert);
    }

    Matrix3x3 invtransform;
    bool success = getInverseTransformCanonical(time, args.renderView, 1., invert, &invtransform);
    if (!success) {
        return false;
    }


    // invert it
    Matrix3x3 transformCanonical;
    if ( !invtransform.inverse(&transformCanonical) ) {
        return false; // no transform available, render as usual
    }
    double srcpixelaspectratio = ( _srcClip && _srcClip->isConnected() ) ? _srcClip->getPixelAspectRatio() : 1.;
    double dstpixelaspectratio = _dstClip ? _dstClip->getPixelAspectRatio() : 1.;
    bool fielded = args.fieldToRender == eFieldLower || args.fieldToRender == eFieldUpper;
    Matrix3x3 transformPixel = ( ofxsMatCanonicalToPixel(dstpixelaspectratio, args.renderScale.x, args.renderScale.y, fielded) *
                                 transformCanonical *
                                 ofxsMatPixelToCanonical(srcpixelaspectratio, args.renderScale.x, args.renderScale.y, fielded) );
    transformClip = _srcClip;
    transformMatrix[0] = transformPixel(0,0);
    transformMatrix[1] = transformPixel(0,1);
    transformMatrix[2] = transformPixel(0,2);
    transformMatrix[3] = transformPixel(1,0);
    transformMatrix[4] = transformPixel(1,1);
    transformMatrix[5] = transformPixel(1,2);
    transformMatrix[6] = transformPixel(2,0);
    transformMatrix[7] = transformPixel(2,1);
    transformMatrix[8] = transformPixel(2,2);

    return true;
} // Transform3x3Plugin::getTransform

#endif // ifdef OFX_EXTENSIONS_NUKE

size_t
Transform3x3Plugin::getInverseTransforms(double time,
                                         int view,
                                         OfxPointD renderscale,
                                         bool fielded,
                                         double srcpixelAspectRatio,
                                         double dstpixelAspectRatio,
                                         bool invert,
                                         double shutter,
                                         ShutterOffsetEnum shutteroffset,
                                         double shuttercustomoffset,
                                         Matrix3x3* invtransform,
                                         size_t invtransformsizealloc) const
{
    OfxRangeD range;

    shutterRange(time, shutter, shutteroffset, shuttercustomoffset, &range);
    double t_start = range.min;
    double t_end = range.max; // shutter time
    bool allequal = true;
    size_t invtransformsize = invtransformsizealloc;
    Matrix3x3 canonicalToPixel = ofxsMatCanonicalToPixel(srcpixelAspectRatio, renderscale.x, renderscale.y, fielded);
    Matrix3x3 pixelToCanonical = ofxsMatPixelToCanonical(dstpixelAspectRatio, renderscale.x, renderscale.y, fielded);
    Matrix3x3 invtransformCanonical;

    for (size_t i = 0; i < invtransformsize; ++i) {
        double t = (i == 0) ? t_start : ( t_start + i * (t_end - t_start) / (double)(invtransformsizealloc - 1) );
        bool success = getInverseTransformCanonical(t, view, 1., invert, &invtransformCanonical); // virtual function
        if (success) {
            invtransform[i] = canonicalToPixel * invtransformCanonical * pixelToCanonical;
        } else {
            invtransform[i](0,0) = 0.;
            invtransform[i](0,1) = 0.;
            invtransform[i](0,2) = 0.;
            invtransform[i](1,0) = 0.;
            invtransform[i](1,1) = 0.;
            invtransform[i](1,2) = 0.;
            invtransform[i](2,0) = 0.;
            invtransform[i](2,1) = 0.;
            invtransform[i](2,2) = 1.;
        }
        allequal = allequal && (invtransform[i](0,0) == invtransform[0](0,0) &&
                                invtransform[i](0,1) == invtransform[0](0,1) &&
                                invtransform[i](0,2) == invtransform[0](0,2) &&
                                invtransform[i](1,0) == invtransform[0](1,0) &&
                                invtransform[i](1,1) == invtransform[0](1,1) &&
                                invtransform[i](1,2) == invtransform[0](1,2) &&
                                invtransform[i](2,0) == invtransform[0](2,0) &&
                                invtransform[i](2,1) == invtransform[0](2,1) &&
                                invtransform[i](2,2) == invtransform[0](2,2));
    }
    if (allequal) { // there is only one transform, no need to do motion blur!
        invtransformsize = 1;
    }

    return invtransformsize;
}

size_t
Transform3x3Plugin::getInverseTransformsBlur(double time,
                                             int view,
                                             OfxPointD renderscale,
                                             bool fielded,
                                             double srcpixelAspectRatio,
                                             double dstpixelAspectRatio,
                                             bool invert,
                                             double amountFrom,
                                             double amountTo,
                                             Matrix3x3* invtransform,
                                             double *amount,
                                             size_t invtransformsizealloc) const
{
    bool allequal = true;
    Matrix3x3 canonicalToPixel = ofxsMatCanonicalToPixel(srcpixelAspectRatio, renderscale.x, renderscale.y, fielded);
    Matrix3x3 pixelToCanonical = ofxsMatPixelToCanonical(dstpixelAspectRatio, renderscale.x, renderscale.y, fielded);
    Matrix3x3 invtransformCanonical;
    size_t invtransformsize = 0;

    for (size_t i = 0; i < invtransformsizealloc; ++i) {
        //double a = 1. - i / (double)(invtransformsizealloc - 1); // Theoretically better
        double a = 1. - (i + 1) / (double)(invtransformsizealloc); // To be compatible with Nuke (Nuke bug?)
        double amt = amountFrom + (amountTo - amountFrom) * a;
        bool success = getInverseTransformCanonical(time, view, amt, invert, &invtransformCanonical); // virtual function
        if (success) {
            if (amount) {
                amount[invtransformsize] = amt;
            }
            invtransform[invtransformsize] = canonicalToPixel * invtransformCanonical * pixelToCanonical;
            ++invtransformsize;
            allequal = allequal && (invtransform[i](0,0) == invtransform[0](0,0) &&
                                    invtransform[i](0,1) == invtransform[0](0,1) &&
                                    invtransform[i](0,2) == invtransform[0](0,2) &&
                                    invtransform[i](1,0) == invtransform[0](1,0) &&
                                    invtransform[i](1,1) == invtransform[0](1,1) &&
                                    invtransform[i](1,2) == invtransform[0](1,2) &&
                                    invtransform[i](2,0) == invtransform[0](2,0) &&
                                    invtransform[i](2,1) == invtransform[0](2,1) &&
                                    invtransform[i](2,2) == invtransform[0](2,2));
        }
    }
    if ( (invtransformsize != 0) && allequal ) { // there is only one transform, no need to do motion blur!
        invtransformsize = 1;
    }

    return invtransformsize;
}

// override changedParam
void
Transform3x3Plugin::changedParam(const InstanceChangedArgs &args,
                                 const string &paramName)
{
    // must clear persistent message, or render() is not called by Nuke after an error
    clearPersistentMessage();
    if ( (paramName == kParamTransform3x3Invert) ||
         ( paramName == kParamShutter) ||
         ( paramName == kParamShutterOffset) ||
         ( paramName == kParamShutterCustomOffset) ) {
        // Motion Blur is the only parameter that doesn't matter
        assert(paramName != kParamTransform3x3MotionBlur);

        changedTransform(args);
    }
    if (paramName == kParamTransform3x3DirectionalBlur) {
        bool directionalBlur;
        _directionalBlur->getValueAtTime(args.time, directionalBlur);
        _shutter->setEnabled(!directionalBlur);
        _shutteroffset->setEnabled(!directionalBlur);
        _shuttercustomoffset->setEnabled(!directionalBlur);
    }
}

// this method must be called by the derived class when the transform was changed
void
Transform3x3Plugin::changedTransform(const InstanceChangedArgs &args)
{
    (void)args;
}

void
Transform3x3Describe(ImageEffectDescriptor &desc,
                     bool masked)
{
    desc.addSupportedContext(eContextFilter);
    desc.addSupportedContext(eContextGeneral);
    if (masked) {
        desc.addSupportedContext(eContextPaint);
    }
    desc.addSupportedBitDepth(eBitDepthUByte);
    desc.addSupportedBitDepth(eBitDepthUShort);
    desc.addSupportedBitDepth(eBitDepthFloat);

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setTemporalClipAccess(false);
    // each field has to be transformed separately, or you will get combing effect
    // this should be true for all geometric transforms
    desc.setRenderTwiceAlways(true);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(kRenderThreadSafety);
    desc.setSupportsRenderQuality(true);

    // Transform3x3-GENERIC

    // in order to support tiles, the transform plugin must implement the getRegionOfInterest function
    desc.setSupportsTiles(kSupportsTiles);

    // in order to support multiresolution, render() must take into account the pixelaspectratio and the renderscale
    // and scale the transform appropriately.
    // All other functions are usually in canonical coordinates.
    desc.setSupportsMultiResolution(kSupportsMultiResolution);

#ifdef OFX_EXTENSIONS_NUKE
    if (!masked) {
        // Enable transform by the host.
        // It is only possible for transforms which can be represented as a 3x3 matrix.
        desc.setCanTransform(true);
        if (getImageEffectHostDescription()->canTransform) {
            //std::cout << "kFnOfxImageEffectCanTransform (describe) =" << desc.getPropertySet().propGetInt(kFnOfxImageEffectCanTransform) << std::endl;
        }
    }
    // ask the host to render all planes
    desc.setPassThroughForNotProcessedPlanes(ePassThroughLevelRenderAllRequestedPlanes);
#endif
#ifdef OFX_EXTENSIONS_NATRON
    desc.setChannelSelector(ePixelComponentNone);
#endif
}

PageParamDescriptor *
Transform3x3DescribeInContextBegin(ImageEffectDescriptor &desc,
                                   ContextEnum context,
                                   bool masked)
{
    // GENERIC

    // Source clip only in the filter context
    // create the mandated source clip
    // always declare the source clip first, because some hosts may consider
    // it as the default input clip (e.g. Nuke)
    ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);

    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->addSupportedComponent(ePixelComponentRGB);
    srcClip->addSupportedComponent(ePixelComponentXY);
    srcClip->addSupportedComponent(ePixelComponentAlpha);
#ifdef OFX_EXTENSIONS_NATRON
    srcClip->addSupportedComponent(ePixelComponentXY);
#endif
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);
#ifdef OFX_EXTENSIONS_NUKE
    srcClip->setCanTransform(true); // source images can have a transform attached
#endif

    if (masked) {
        // GENERIC (MASKED)
        //
        // if general or paint context, define the mask clip
        // if paint context, it is a mandated input called 'brush'
        ClipDescriptor *maskClip = (context == eContextPaint) ? desc.defineClip("Brush") : desc.defineClip("Mask");
        maskClip->addSupportedComponent(ePixelComponentAlpha);
        maskClip->setTemporalClipAccess(false);
        if (context == eContextGeneral) {
            maskClip->setOptional(true);
        }
        maskClip->setSupportsTiles(kSupportsTiles);
        maskClip->setIsMask(true); // we are a mask input
    }

    // create the mandated output clip
    ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentRGB);
    dstClip->addSupportedComponent(ePixelComponentXY);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
#ifdef OFX_EXTENSIONS_NATRON
    dstClip->addSupportedComponent(ePixelComponentXY);
#endif
    dstClip->setSupportsTiles(kSupportsTiles);


    // make some pages and to things in
    PageParamDescriptor *page = desc.definePageParam("Controls");

    return page;
} // Transform3x3DescribeInContextBegin

void
Transform3x3DescribeInContextEnd(ImageEffectDescriptor &desc,
                                 ContextEnum context,
                                 PageParamDescriptor* page,
                                 bool masked,
                                 Transform3x3Plugin::Transform3x3ParamsTypeEnum paramsType)
{
    // invert
    {
        BooleanParamDescriptor* param = desc.defineBooleanParam(kParamTransform3x3Invert);
        param->setLabel(kParamTransform3x3InvertLabel);
        param->setHint(kParamTransform3x3InvertHint);
        param->setDefault(false);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }
    // GENERIC PARAMETERS
    //

    ofxsFilterDescribeParamsInterpolate2D(desc, page, paramsType == Transform3x3Plugin::eTransform3x3ParamsTypeMotionBlur);

    // motionBlur
    {
        DoubleParamDescriptor* param = desc.defineDoubleParam(kParamTransform3x3MotionBlur);
        param->setLabel(kParamTransform3x3MotionBlurLabel);
        param->setHint(kParamTransform3x3MotionBlurHint);
        param->setDefault(paramsType == Transform3x3Plugin::eTransform3x3ParamsTypeDirBlur ? 1. : 0.);
        param->setIncrement(0.01);
        param->setRange(0., 100.);
        param->setDisplayRange(0., 4.);
        if (page) {
            page->addChild(*param);
        }
    }

    if (paramsType == Transform3x3Plugin::eTransform3x3ParamsTypeDirBlur) {
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamTransform3x3DirBlurAmount);
            param->setLabel(kParamTransform3x3DirBlurAmountLabel);
            param->setHint(kParamTransform3x3DirBlurAmountHint);
            param->setRange(-DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(-1, 2.);
            param->setDefault(1);
            param->setAnimates(true); // can animate
            if (page) {
                page->addChild(*param);
            }
        }

        {
            BooleanParamDescriptor *param = desc.defineBooleanParam(kParamTransform3x3DirBlurCentered);
            param->setLabel(kParamTransform3x3DirBlurCenteredLabel);
            param->setHint(kParamTransform3x3DirBlurCenteredHint);
            param->setAnimates(true); // can animate
            if (page) {
                page->addChild(*param);
            }
        }

        {
            DoubleParamDescriptor *param = desc.defineDoubleParam(kParamTransform3x3DirBlurFading);
            param->setLabel(kParamTransform3x3DirBlurFadingLabel);
            param->setHint(kParamTransform3x3DirBlurFadingHint);
            param->setRange(0., 4.);
            param->setDisplayRange(0., 4.);
            param->setDefault(0.);
            param->setAnimates(true); // can animate
            if (page) {
                page->addChild(*param);
            }
        }
    } else if (paramsType == Transform3x3Plugin::eTransform3x3ParamsTypeMotionBlur) {
        // directionalBlur
        {
            BooleanParamDescriptor* param = desc.defineBooleanParam(kParamTransform3x3DirectionalBlur);
            param->setLabel(kParamTransform3x3DirectionalBlurLabel);
            param->setHint(kParamTransform3x3DirectionalBlurHint);
            param->setDefault(false);
            param->setAnimates(true);
            if (page) {
                page->addChild(*param);
            }
        }

        shutterDescribeInContext(desc, context, page);
    }

    if (masked) {
        // GENERIC (MASKED)
        //
        ofxsMaskMixDescribeParams(desc, page);
#ifdef OFX_EXTENSIONS_NUKE
    } else if (getImageEffectHostDescription()->canTransform) {
        // Transform3x3-GENERIC (NON-MASKED)
        //
        //std::cout << "kFnOfxImageEffectCanTransform in describeincontext(" << context << ")=" << desc.getPropertySet().propGetInt(kFnOfxImageEffectCanTransform) << std::endl;
#endif
    }
} // Transform3x3DescribeInContextEnd
} // namespace OFX
