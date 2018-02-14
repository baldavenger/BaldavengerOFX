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

#ifndef openfx_supportext_ofxsTransform3x3_h
#define openfx_supportext_ofxsTransform3x3_h

#include <memory>

#include "ofxsImageEffect.h"
#include "ofxsTransform3x3Processor.h"
#include "ofxsShutter.h"
#include "ofxsMacros.h"

#define kParamTransform3x3Invert "invert"
#define kParamTransform3x3InvertLabel "Invert"
#define kParamTransform3x3InvertHint "Invert the transform."

#define kParamTransform3x3MotionBlur "motionBlur"
#define kParamTransform3x3MotionBlurLabel "Motion Blur"
#define kParamTransform3x3MotionBlurHint "Quality of motion blur rendering. 0 disables motion blur, 1 is a good value. Increasing this slows down rendering."

// extra parameters for DirBlur:

#define kParamTransform3x3DirBlurAmount "amount"
#define kParamTransform3x3DirBlurAmountLabel "Amount"
#define kParamTransform3x3DirBlurAmountHint "Amount of blur transform to apply. A value of 1 means to apply the full transform range. A value of 0 means to apply no blur at all. Default is 1."

#define kParamTransform3x3DirBlurCentered "centered"
#define kParamTransform3x3DirBlurCenteredLabel "Centered"
#define kParamTransform3x3DirBlurCenteredHint "When checked, apply directional blur symmetrically arount the neutral position."

#define kParamTransform3x3DirBlurFading "fading"
#define kParamTransform3x3DirBlurFadingLabel "Fading"
#define kParamTransform3x3DirBlurFadingHint "Controls the fading function. A value of 1 corresponds to linear fading. A value of 0 disables fading. Default is 0."

// extra parameters for non-DirBlur

#define kParamTransform3x3DirectionalBlur "directionalBlur"
#define kParamTransform3x3DirectionalBlurLabel "Directional Blur Mode"
#define kParamTransform3x3DirectionalBlurHint "Motion blur is computed from the original image to the transformed image, each parameter being interpolated linearly. The motionBlur parameter must be set to a nonzero value, and the blackOutside parameter may have an important effect on the result."

namespace OFX {
////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class Transform3x3Plugin
    : public OFX::ImageEffect
{
protected:
    // do not need to delete these, the ImageEffect is managing them for us
    OFX::Clip *_dstClip;
    OFX::Clip *_srcClip;
    OFX::Clip *_maskClip;

public:

    enum Transform3x3ParamsTypeEnum
    {
        eTransform3x3ParamsTypeNone = 0,
        eTransform3x3ParamsTypeDirBlur,
        eTransform3x3ParamsTypeMotionBlur
    };

    /** @brief ctor */
    Transform3x3Plugin(OfxImageEffectHandle handle,
                       bool masked,
                       Transform3x3ParamsTypeEnum paramsType);

    /** @brief destructor */
    virtual ~Transform3x3Plugin();

    // a default implementation of isIdentity is provided, which may be overridden by the derived class
    virtual bool isIdentity(double /*time*/)
    {
        return false;
    };

    /** @brief recover a transform matrix from an effect */
    virtual bool getInverseTransformCanonical(double time, int view, double amount, bool invert, OFX::Matrix3x3* invtransform) const = 0;


    // The following functions override those is OFX::ImageEffect

    // override the rod call (not final, since overriden by Reformat)
    virtual bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments &args, OfxRectD &rod) OVERRIDE;

    // override the roi call
    virtual void getRegionsOfInterest(const OFX::RegionsOfInterestArguments &args, OFX::RegionOfInterestSetter &rois) OVERRIDE FINAL;

    /* Override the render */
    virtual void render(const OFX::RenderArguments &args) OVERRIDE;

    // override isIdentity
    virtual bool isIdentity(const OFX::IsIdentityArguments &args, OFX::Clip * &identityClip, double &identityTime) OVERRIDE FINAL;

#ifdef OFX_EXTENSIONS_NUKE
    /** @brief recover a transform matrix from an effect */
    virtual bool getTransform(const OFX::TransformArguments & args, OFX::Clip * &transformClip, double transformMatrix[9]) OVERRIDE;
#endif

    // override changedParam. note that the derived class MUST explicitely call this method after handling its own parameter changes
    virtual void changedParam(const OFX::InstanceChangedArgs &args, const std::string &paramName) OVERRIDE;

    // this method must be called by the derived class when the transform was changed
    void changedTransform(const OFX::InstanceChangedArgs &args);

protected:
    size_t getInverseTransforms(double time,
                                int view,
                                OfxPointD renderscale,
                                bool fielded,
                                double srcpixelAspectRatio,
                                double dstpixelAspectRatio,
                                bool invert,
                                double shutter,
                                ShutterOffsetEnum shutteroffset,
                                double shuttercustomoffset,
                                OFX::Matrix3x3* invtransform,
                                size_t invtransformsizealloc) const;

    size_t getInverseTransformsBlur(double time,
                                    int view,
                                    OfxPointD renderscale,
                                    bool fielded,
                                    double srcpixelAspectRatio,
                                    double dstpixelAspectRatio,
                                    bool invert,
                                    double amountFrom,
                                    double amountTo,
                                    OFX::Matrix3x3* invtransform,
                                    double* amount,
                                    size_t invtransformsizealloc) const;

private:
    /* internal render function */
    template <class PIX, int nComponents, int maxValue, bool masked>
    void renderInternalForBitDepth(const OFX::RenderArguments &args);

    template <int nComponents, bool masked>
    void renderInternal(const OFX::RenderArguments &args, OFX::BitDepthEnum dstBitDepth);

    /* set up and run a processor */
    void setupAndProcess(Transform3x3ProcessorBase &, const OFX::RenderArguments &args);

    bool isIdentity(double time, OFX::Clip * &identityClip, double &identityTime);

    void transformRegion(const OfxRectD &rectFrom,
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
                         OfxRectD *rectTo);

protected:
    // Transform3x3-GENERIC
    Transform3x3ParamsTypeEnum _paramsType;
    OFX::BooleanParam* _invert;
    // GENERIC
    OFX::ChoiceParam* _filter;
    OFX::BooleanParam* _clamp;
    OFX::BooleanParam* _blackOutside;
    OFX::DoubleParam* _motionblur;
    OFX::DoubleParam* _dirBlurAmount; // DirBlur only
    OFX::BooleanParam* _dirBlurCentered; // DirBlur only
    OFX::DoubleParam* _dirBlurFading; // DirBlur only
    OFX::BooleanParam* _directionalBlur; // non-DirBlur
    OFX::DoubleParam* _shutter; // non-DirBlur
    OFX::ChoiceParam* _shutteroffset; // non-DirBlur
    OFX::DoubleParam* _shuttercustomoffset; // non-DirBlur
    bool _masked;
    OFX::DoubleParam* _mix;
    OFX::BooleanParam* _maskApply;
    OFX::BooleanParam* _maskInvert;
};

void Transform3x3Describe(OFX::ImageEffectDescriptor &desc, bool masked);

OFX::PageParamDescriptor * Transform3x3DescribeInContextBegin(OFX::ImageEffectDescriptor &desc, OFX::ContextEnum context, bool masked);

void Transform3x3DescribeInContextEnd(OFX::ImageEffectDescriptor &desc, OFX::ContextEnum context, OFX::PageParamDescriptor* page, bool masked, OFX::Transform3x3Plugin::Transform3x3ParamsTypeEnum paramsType);
} // namespace OFX
#endif /* defined(openfx_supportext_ofxsTransform3x3_h) */
