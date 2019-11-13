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
 * OFX Generator plug-in helper
 */

#include "ofxsGenerator.h"

#include <cmath>
#include <cfloat> // DBL_MAX
#include <cassert>
#include <limits>

#include "ofxsFormatResolution.h"
#include "ofxsCoords.h"

// Some hosts (e.g. Resolve) may not support normalized defaults (setDefaultCoordinateSystem(eCoordinatesNormalised))
#define kParamDefaultsNormalised "defaultsNormalisedGenerator"

using namespace OFX;

using std::string;

static bool gHostSupportsDefaultCoordinateSystem = true; // for kParamDefaultsNormalised

GeneratorPlugin::GeneratorPlugin(OfxImageEffectHandle handle,
                                 bool useOutputComponentsAndDepth,
                                 bool supportsBitDepthByte,
                                 bool supportsBitDepthUShort,
                                 bool supportsBitDepthHalf,
                                 bool supportsBitDepthFloat)
    : ImageEffect(handle)
    , _dstClip(0)
    , _extent(0)
    , _format(0)
    , _formatSize(0)
    , _formatPar(0)
    , _reformat(0)
    , _btmLeft(0)
    , _size(0)
    , _interactive(0)
    , _outputComponents(0)
    , _outputBitDepth(0)
    , _range(0)
    , _recenter(0)
    , _useOutputComponentsAndDepth(useOutputComponentsAndDepth)
    , _supportsByte(supportsBitDepthByte)
    , _supportsUShort(supportsBitDepthUShort)
    , _supportsHalf(supportsBitDepthHalf)
    , _supportsFloat(supportsBitDepthFloat)
    , _supportsRGBA(0)
    , _supportsRGB(0)
    , _supportsXY(0)
    , _supportsAlpha(0)
{
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    assert( _dstClip && (!_dstClip->isConnected() || _dstClip->getPixelComponents() == ePixelComponentRGBA ||
                         _dstClip->getPixelComponents() == ePixelComponentRGB ||
                         _dstClip->getPixelComponents() == ePixelComponentXY ||
                         _dstClip->getPixelComponents() == ePixelComponentAlpha) );

    _extent = fetchChoiceParam(kParamGeneratorExtent);
    _format = fetchChoiceParam(kParamGeneratorFormat);
    _formatSize = fetchInt2DParam(kParamGeneratorSize);
    _formatPar = fetchDoubleParam(kParamGeneratorPAR);
    if ( paramExists(kParamGeneratorReformat) ) {
        _reformat = fetchBooleanParam(kParamGeneratorReformat);
    }
    _btmLeft = fetchDouble2DParam(kParamRectangleInteractBtmLeft);
    _size = fetchDouble2DParam(kParamRectangleInteractSize);
    _recenter = fetchPushButtonParam(kParamGeneratorCenter);
    _interactive = fetchBooleanParam(kParamRectangleInteractInteractive);
    assert(_extent && _format && _formatSize && _formatPar && _btmLeft && _size && _interactive && _recenter);

    if (_useOutputComponentsAndDepth) {
        _outputComponents = fetchChoiceParam(kParamGeneratorOutputComponents);

        if (getImageEffectHostDescription()->supportsMultipleClipDepths) {
            _outputBitDepth = fetchChoiceParam(kParamGeneratorOutputBitDepth);
        }
    }
    if (getContext() == eContextGeneral) {
        _range   = fetchInt2DParam(kParamGeneratorRange);
        assert(_range);
    }

    updateParamsVisibility();

    // kOfxImageEffectPropSupportedPixelDepths is not a property of the effect instance
    // (only the host and the plugin descriptor)
    {
        int i = 0;
        if ( _supportsFloat && getImageEffectHostDescription()->supportsBitDepth(eBitDepthFloat) ) {
            _outputBitDepthMap[i] = eBitDepthFloat;
            ++i;
        }
        if ( _supportsHalf && getImageEffectHostDescription()->supportsBitDepth(eBitDepthHalf) ) {
            _outputBitDepthMap[i] = eBitDepthHalf;
            ++i;
        }
        if ( _supportsUShort && getImageEffectHostDescription()->supportsBitDepth(eBitDepthUShort) ) {
            _outputBitDepthMap[i] = eBitDepthUShort;
            ++i;
        }
        if ( _supportsByte && getImageEffectHostDescription()->supportsBitDepth(eBitDepthUByte) ) {
            _outputBitDepthMap[i] = eBitDepthUByte;
            ++i;
        }
        _outputBitDepthMap[i] = eBitDepthNone;
    }

    const PropertySet &dstClipProps = _dstClip->getPropertySet();
    int numComponents = dstClipProps.propGetDimension(kOfxImageEffectPropSupportedComponents);
    for (int i = 0; i < numComponents; ++i) {
        PixelComponentEnum pixelComponents = mapStrToPixelComponentEnum( dstClipProps.propGetString(kOfxImageEffectPropSupportedComponents, i) );
        bool supported = getImageEffectHostDescription()->supportsPixelComponent(pixelComponents);
        switch (pixelComponents) {
        case ePixelComponentRGBA:
            _supportsRGBA  = supported;
            break;
        case ePixelComponentRGB:
            _supportsRGB = supported;
            break;
        case ePixelComponentXY:
            _supportsXY = supported;
            break;
        case ePixelComponentAlpha:
            _supportsAlpha = supported;
            break;
        default:
            // other components are not supported by this plugin
            break;
        }
    }
    int i = 0;
    if (_supportsRGBA) {
        _outputComponentsMap[i] = ePixelComponentRGBA;
        ++i;
    }
    if (_supportsRGB) {
        _outputComponentsMap[i] = ePixelComponentRGB;
        ++i;
    }
    if (_supportsXY) {
        _outputComponentsMap[i] = ePixelComponentXY;
        ++i;
    }
    if (_supportsAlpha) {
        _outputComponentsMap[i] = ePixelComponentAlpha;
        ++i;
    }
    _outputComponentsMap[i] = ePixelComponentNone;

    // honor kParamDefaultsNormalised
    if ( paramExists(kParamDefaultsNormalised) ) {
        // Some hosts (e.g. Resolve) may not support normalized defaults (setDefaultCoordinateSystem(eCoordinatesNormalised))
        // handle these ourselves!
        BooleanParam* param = fetchBooleanParam(kParamDefaultsNormalised);
        assert(param);
        bool normalised = param->getValue();
        if (normalised) {
            OfxPointD size = getProjectExtent();
            OfxPointD origin = getProjectOffset();
            OfxPointD p;
            // we must denormalise all parameters for which setDefaultCoordinateSystem(eCoordinatesNormalised) couldn't be done
            beginEditBlock(kParamDefaultsNormalised);
            p = _btmLeft->getValue();
            _btmLeft->setValue(p.x * size.x + origin.x, p.y * size.y + origin.y);
            p = _size->getValue();
            _size->setValue(p.x * size.x, p.y * size.y);
            param->setValue(false);
            endEditBlock();
        }
    }
}

void
GeneratorPlugin::checkComponents(BitDepthEnum dstBitDepth,
                                 PixelComponentEnum dstComponents)
{
    if (!_useOutputComponentsAndDepth) {
        return;
    }

    // get the components of _dstClip
    PixelComponentEnum outputComponents = _outputComponentsMap[_outputComponents->getValue()];
    if (dstComponents != outputComponents) {
        setPersistentMessage(Message::eMessageError, "", "OFX Host dit not take into account output components");
        throwSuiteStatusException(kOfxStatErrImageFormat);

        return;
    }

    if (getImageEffectHostDescription()->supportsMultipleClipDepths) {
        // get the bitDepth of _dstClip
        BitDepthEnum outputBitDepth = _outputBitDepthMap[_outputBitDepth->getValue()];
        if (dstBitDepth != outputBitDepth) {
            setPersistentMessage(Message::eMessageError, "", "OFX Host dit not take into account output bit depth");
            throwSuiteStatusException(kOfxStatErrImageFormat);

            return;
        }
    }
}

/* override the time domain action, only for the general context */
bool
GeneratorPlugin::getTimeDomain(OfxRangeD &range)
{
    // this should only be called in the general context, ever!
    if (getContext() == eContextGeneral) {
        assert(_range);
        // how many frames on the input clip
        //OfxRangeD srcRange = _srcClip->getFrameRange();

        int min, max;
        _range->getValue(min, max);
        range.min = min;
        range.max = max;

        return true;
    }

    return false;
}

void
GeneratorPlugin::updateParamsVisibility()
{
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();
    bool hasFormat = (extent == eGeneratorExtentFormat);
    bool hasSize = (extent == eGeneratorExtentSize);

    _format->setIsSecretAndDisabled(!hasFormat);
    if (_reformat) {
        _reformat->setIsSecretAndDisabled(!hasSize);
    }
    _size->setIsSecretAndDisabled(!hasSize);
    _recenter->setIsSecretAndDisabled(!hasSize);
    _btmLeft->setIsSecretAndDisabled(!hasSize);
    _interactive->setIsSecretAndDisabled(!hasSize);
}

void
GeneratorPlugin::changedParam(const InstanceChangedArgs &args,
                              const string &paramName)
{
    if ( (paramName == kParamGeneratorExtent) && (args.reason == eChangeUserEdit) ) {
        updateParamsVisibility();
    } else if (paramName == kParamGeneratorFormat) {
        //the host does not handle the format itself, do it ourselves
        EParamFormat format = (EParamFormat)_format->getValue();
        int w = 0, h = 0;
        double par = -1;
        getFormatResolution(format, &w, &h, &par);
        assert(par != -1);
        _formatPar->setValue(par);
        _formatSize->setValue(w, h);
    } else if (paramName == kParamGeneratorCenter) {
        Clip* srcClip = getSrcClip();
        OfxRectD srcRoD;
        if ( srcClip && srcClip->isConnected() ) {
            srcRoD = srcClip->getRegionOfDefinition(args.time);
        } else {
            OfxPointD siz = getProjectSize();
            OfxPointD off = getProjectOffset();
            srcRoD.x1 = off.x;
            srcRoD.x2 = off.x + siz.x;
            srcRoD.y1 = off.y;
            srcRoD.y2 = off.y + siz.y;
        }
        OfxPointD center;
        center.x = (srcRoD.x2 + srcRoD.x1) / 2.;
        center.y = (srcRoD.y2 + srcRoD.y1) / 2.;

        OfxRectD rectangle;
        _size->getValue(rectangle.x2, rectangle.y2);
        _btmLeft->getValue(rectangle.x1, rectangle.y1);
        rectangle.x2 += rectangle.x1;
        rectangle.y2 += rectangle.y1;

        OfxRectD newRectangle;
        newRectangle.x1 = center.x - (rectangle.x2 - rectangle.x1) / 2.;
        newRectangle.y1 = center.y - (rectangle.y2 - rectangle.y1) / 2.;
        newRectangle.x2 = newRectangle.x1 + (rectangle.x2 - rectangle.x1);
        newRectangle.y2 = newRectangle.y1 + (rectangle.y2 - rectangle.y1);

        _size->setValue(newRectangle.x2 - newRectangle.x1, newRectangle.y2 - newRectangle.y1);
        _btmLeft->setValue(newRectangle.x1, newRectangle.y1);
    }
}

bool
GeneratorPlugin::getRegionOfDefinition(double time, OfxRectD &rod)
{
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();

    switch (extent) {
    case eGeneratorExtentFormat: {
        int w, h;
        _formatSize->getValueAtTime(time, w, h);
        double par;
        _formatPar->getValueAtTime(time, par);
        OfxRectI pixelFormat;
        pixelFormat.x1 = pixelFormat.y1 = 0;
        pixelFormat.x2 = w;
        pixelFormat.y2 = h;
        OfxPointD renderScale = {1., 1.};
        Coords::toCanonical(pixelFormat, renderScale, par, &rod);

        return true;
    }
    case eGeneratorExtentSize: {
        _size->getValueAtTime(time, rod.x2, rod.y2);
        _btmLeft->getValueAtTime(time, rod.x1, rod.y1);
        rod.x2 += rod.x1;
        rod.y2 += rod.y1;

        return true;
    }
    case eGeneratorExtentProject: {
        OfxPointD siz = getProjectSize();
        OfxPointD off = getProjectOffset();
        rod.x1 = off.x;
        rod.x2 = off.x + siz.x;
        rod.y1 = off.y;
        rod.y2 = off.y + siz.y;

        return true;
    }
    case eGeneratorExtentDefault:

        return false;
    }

    return false;
}

void
GeneratorPlugin::getClipPreferences(ClipPreferencesSetter &clipPreferences)
{
    double par = 0.;
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();

    switch (extent) {
    case eGeneratorExtentFormat: {
        //specific output format
        par = _formatPar->getValue();
        break;
    }
    case eGeneratorExtentProject:
    case eGeneratorExtentDefault: {
        /// this should be the default value given by the host, no need to set it.
        /// @see Instance::setupClipPreferencesArgs() in HostSupport, it should have
        /// the line:
        /// double inputPar = getProjectPixelAspectRatio();

        par = getProjectPixelAspectRatio();
        break;
    }
    case eGeneratorExtentSize:
        // only set the format if btmLeft and size are not animated 
        if ( _reformat && _reformat->getValue() &&
             _btmLeft->getNumKeys() == 0 &&
             _size->getNumKeys() == 0 ) {
            par = 1;
        }
        break;
    }

    if (_useOutputComponentsAndDepth) {
        // set the components of _dstClip
        PixelComponentEnum outputComponents = _outputComponentsMap[_outputComponents->getValue()];
        clipPreferences.setClipComponents(*_dstClip, outputComponents);

        if (getImageEffectHostDescription()->supportsMultipleClipDepths) {
            // set the bitDepth of _dstClip
            BitDepthEnum outputBitDepth = _outputBitDepthMap[_outputBitDepth->getValue()];
            clipPreferences.setClipBitDepth(*_dstClip, outputBitDepth);
        }
    }
    if (par != 0.) {
        clipPreferences.setPixelAspectRatio(*_dstClip, par);
#ifdef OFX_EXTENSIONS_NATRON
        OfxRectD rod;
        // get the format from time = 0
        if ( getRegionOfDefinition(0, rod) ) { // don't set format if default
            OfxRectI format;
            const OfxPointD rsOne = {1., 1.};
            Coords::toPixelNearest(rod, rsOne, par, &format);
            clipPreferences.setOutputFormat(format);
        }
#endif
    }
}

GeneratorInteract::GeneratorInteract(OfxInteractHandle handle,
                                     ImageEffect* effect)
    : RectangleInteract(handle, effect)
    , _extent(0)
    , _extentValue(eGeneratorExtentDefault)
{
    _extent = effect->fetchChoiceParam(kParamGeneratorExtent);
    assert(_extent);
}

void
GeneratorInteract::aboutToCheckInteractivity(OfxTime /*time*/)
{
    _extentValue = (GeneratorExtentEnum)_extent->getValue();
}

bool
GeneratorInteract::allowTopLeftInteraction() const
{
    return _extentValue == eGeneratorExtentSize;
}

bool
GeneratorInteract::allowBtmRightInteraction() const
{
    return _extentValue == eGeneratorExtentSize;
}

bool
GeneratorInteract::allowBtmLeftInteraction() const
{
    return _extentValue == eGeneratorExtentSize;
}

bool
GeneratorInteract::allowBtmMidInteraction() const
{
    return _extentValue == eGeneratorExtentSize;
}

bool
GeneratorInteract::allowMidLeftInteraction() const
{
    return _extentValue == eGeneratorExtentSize;
}

bool
GeneratorInteract::allowCenterInteraction() const
{
    return _extentValue == eGeneratorExtentSize;
}

bool
GeneratorInteract::draw(const DrawArgs &args)
{
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();

    if (extent != eGeneratorExtentSize) {
        return false;
    }

    return RectangleInteract::draw(args);
}

bool
GeneratorInteract::penMotion(const PenArgs &args)
{
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();

    if (extent != eGeneratorExtentSize) {
        return false;
    }

    return RectangleInteract::penMotion(args);
}

bool
GeneratorInteract::penDown(const PenArgs &args)
{
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();

    if (extent != eGeneratorExtentSize) {
        return false;
    }

    return RectangleInteract::penDown(args);
}

bool
GeneratorInteract::penUp(const PenArgs &args)
{
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();

    if (extent != eGeneratorExtentSize) {
        return false;
    }

    return RectangleInteract::penUp(args);
}

void
GeneratorInteract::loseFocus(const FocusArgs &args)
{
    return RectangleInteract::loseFocus(args);
}

bool
GeneratorInteract::keyDown(const KeyArgs &args)
{
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();

    if (extent != eGeneratorExtentSize) {
        return false;
    }

    return RectangleInteract::keyDown(args);
}

bool
GeneratorInteract::keyUp(const KeyArgs & args)
{
    GeneratorExtentEnum extent = (GeneratorExtentEnum)_extent->getValue();

    if (extent != eGeneratorExtentSize) {
        return false;
    }

    return RectangleInteract::keyUp(args);
}

namespace OFX {
void
generatorDescribe(ImageEffectDescriptor &desc)
{
    desc.setOverlayInteractDescriptor(new GeneratorOverlayDescriptor);
}

void
generatorDescribeInContext(PageParamDescriptor *page,
                           ImageEffectDescriptor &desc,
                           ClipDescriptor &dstClip,
                           GeneratorExtentEnum defaultType,
                           PixelComponentEnum defaultComponents, // either RGBA, RGB, XY or Alpha
                           bool useOutputComponentsAndDepth,
                           ContextEnum context,
                           bool reformat)
{
    // extent
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamGeneratorExtent);
        param->setLabel(kParamGeneratorExtentLabel);
        param->setHint(kParamGeneratorExtentHint);
        assert(param->getNOptions() == eGeneratorExtentFormat);
        param->appendOption(kParamGeneratorExtentOptionFormat, kParamGeneratorExtentOptionFormatHint);
        assert(param->getNOptions() == eGeneratorExtentSize);
        param->appendOption(kParamGeneratorExtentOptionSize, kParamGeneratorExtentOptionSizeHint);
        assert(param->getNOptions() == eGeneratorExtentProject);
        param->appendOption(kParamGeneratorExtentOptionProject, kParamGeneratorExtentOptionProjectHint);
        assert(param->getNOptions() == eGeneratorExtentDefault);
        param->appendOption(kParamGeneratorExtentOptionDefault, kParamGeneratorExtentOptionDefaultHint);
        param->setDefault( (int)defaultType );
        param->setLayoutHint(eLayoutHintNoNewLine);
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    // recenter
    {
        PushButtonParamDescriptor* param = desc.definePushButtonParam(kParamGeneratorCenter);
        param->setLabel(kParamGeneratorCenterLabel);
        param->setHint(kParamGeneratorCenterHint);
        param->setLayoutHint(eLayoutHintNoNewLine);
        if (page) {
            page->addChild(*param);
        }
    }

#ifdef OFX_EXTENSIONS_NATRON
    // reformat
    if (reformat && getImageEffectHostDescription()->isNatron) {
        BooleanParamDescriptor* param = desc.defineBooleanParam(kParamGeneratorReformat);
        param->setLabel(kParamGeneratorReformatLabel);
        param->setHint(kParamGeneratorReformatHint);
        param->setLayoutHint(eLayoutHintNoNewLine);
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }
#endif

    // format
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamGeneratorFormat);
        param->setLabel(kParamGeneratorFormatLabel);
        assert(param->getNOptions() == eParamFormatPCVideo);
        param->appendOption(kParamFormatPCVideoLabel);
        assert(param->getNOptions() == eParamFormatNTSC);
        param->appendOption(kParamFormatNTSCLabel);
        assert(param->getNOptions() == eParamFormatPAL);
        param->appendOption(kParamFormatPALLabel);
        assert(param->getNOptions() == eParamFormatNTSC169);
        param->appendOption(kParamFormatNTSC169Label);
        assert(param->getNOptions() == eParamFormatPAL169);
        param->appendOption(kParamFormatPAL169Label);
        assert(param->getNOptions() == eParamFormatHD720);
        param->appendOption(kParamFormatHD720Label);
        assert(param->getNOptions() == eParamFormatHD);
        param->appendOption(kParamFormatHDLabel);
        assert(param->getNOptions() == eParamFormatUHD4K);
        param->appendOption(kParamFormatUHD4KLabel);
        assert(param->getNOptions() == eParamFormat1kSuper35);
        param->appendOption(kParamFormat1kSuper35Label);
        assert(param->getNOptions() == eParamFormat1kCinemascope);
        param->appendOption(kParamFormat1kCinemascopeLabel);
        assert(param->getNOptions() == eParamFormat2kSuper35);
        param->appendOption(kParamFormat2kSuper35Label);
        assert(param->getNOptions() == eParamFormat2kCinemascope);
        param->appendOption(kParamFormat2kCinemascopeLabel);
        assert(param->getNOptions() == eParamFormat2kDCP);
        param->appendOption(kParamFormat2kDCPLabel);
        assert(param->getNOptions() == eParamFormat4kSuper35);
        param->appendOption(kParamFormat4kSuper35Label);
        assert(param->getNOptions() == eParamFormat4kCinemascope);
        param->appendOption(kParamFormat4kCinemascopeLabel);
        assert(param->getNOptions() == eParamFormat4kDCP);
        param->appendOption(kParamFormat4kDCPLabel);
        assert(param->getNOptions() == eParamFormatSquare256);
        param->appendOption(kParamFormatSquare256Label);
        assert(param->getNOptions() == eParamFormatSquare512);
        param->appendOption(kParamFormatSquare512Label);
        assert(param->getNOptions() == eParamFormatSquare1k);
        param->appendOption(kParamFormatSquare1kLabel);
        assert(param->getNOptions() == eParamFormatSquare2k);
        param->appendOption(kParamFormatSquare2kLabel);
        param->setDefault(eParamFormatPCVideo);
        param->setHint(kParamGeneratorFormatHint);
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    {
        int w = 0, h = 0;
        double par = -1.;
        getFormatResolution(eParamFormatPCVideo, &w, &h, &par);
        assert(par != -1);
        {
            Int2DParamDescriptor* param = desc.defineInt2DParam(kParamGeneratorSize);
            param->setLabel(kParamGeneratorSizeLabel);
            param->setHint(kParamGeneratorSizeHint);
            param->setIsSecretAndDisabled(true);
            param->setDefault(w, h);
            param->setAnimates(false); // does not animate, because we set format
            desc.addClipPreferencesSlaveParam(*param);
            if (page) {
                page->addChild(*param);
            }
        }

        {
            DoubleParamDescriptor* param = desc.defineDoubleParam(kParamGeneratorPAR);
            param->setLabel(kParamGeneratorPARLabel);
            param->setHint(kParamGeneratorPARHint);
            param->setIsSecretAndDisabled(true);
            param->setRange(0., DBL_MAX);
            param->setDisplayRange(0.5, 2.);
            param->setDefault(par);
            param->setAnimates(false); // does not animate, because we set format
            desc.addClipPreferencesSlaveParam(*param);
            if (page) {
                page->addChild(*param);
            }
        }
    }

    // btmLeft
    {
        Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamRectangleInteractBtmLeft);
        param->setLabel(kParamRectangleInteractBtmLeftLabel);
        param->setDoubleType(eDoubleTypeXYAbsolute);
        if ( param->supportsDefaultCoordinateSystem() ) {
            param->setDefaultCoordinateSystem(eCoordinatesNormalised); // no need of kParamDefaultsNormalised
        } else {
            gHostSupportsDefaultCoordinateSystem = false; // no multithread here, see kParamDefaultsNormalised
        }
        param->setDefault(0., 0.);
        param->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
        param->setDisplayRange(-10000, -10000, 10000, 10000); // Resolve requires display range or values are clamped to (-1,1)
        param->setIncrement(1.);
        param->setLayoutHint(eLayoutHintNoNewLine);
        param->setHint("Coordinates of the bottom left corner of the size rectangle.");
        param->setDigits(0);
        // This parameter *can* be animated, because it only sets the format if "Reformat" is checked
        //param->setAnimates(false); // does not animate, because we set format
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    // size
    {
        Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamRectangleInteractSize);
        param->setLabel(kParamRectangleInteractSizeLabel);
        param->setDoubleType(eDoubleTypeXY);
        if ( param->supportsDefaultCoordinateSystem() ) {
            param->setDefaultCoordinateSystem(eCoordinatesNormalised); // no need of kParamDefaultsNormalised
        } else {
            gHostSupportsDefaultCoordinateSystem = false; // no multithread here, see kParamDefaultsNormalised
        }
        param->setDefault(1., 1.);
        param->setRange(0., 0., DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
        param->setDisplayRange(0, 0, 10000, 10000); // Resolve requires display range or values are clamped to (-1,1)
        param->setIncrement(1.);
        param->setDimensionLabels(kParamRectangleInteractSizeDim1, kParamRectangleInteractSizeDim2);
        param->setHint("Width and height of the size rectangle.");
        param->setIncrement(1.);
        param->setDigits(0);
        // This parameter *can* be animated, because it only sets the format if "Reformat" is checked
        //param->setAnimates(false); // does not animate, because we set format
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }


    // interactive
    {
        BooleanParamDescriptor* param = desc.defineBooleanParam(kParamRectangleInteractInteractive);
        param->setLabel(kParamRectangleInteractInteractiveLabel);
        param->setHint(kParamRectangleInteractInteractiveHint);
        param->setEvaluateOnChange(false);
        if (page) {
            page->addChild(*param);
        }
    }

    // range
    if (context == eContextGeneral) {
        Int2DParamDescriptor *param = desc.defineInt2DParam(kParamGeneratorRange);
        param->setLabel(kParamGeneratorRangeLabel);
        param->setHint(kParamGeneratorRangeHint);
        param->setDefault(1, 1);
        param->setDimensionLabels("min", "max");
        param->setAnimates(false); // can not animate, because it defines the time domain
        if (page) {
            page->addChild(*param);
        }
    }

    if (useOutputComponentsAndDepth) {
        bool supportsByte  = false;
        bool supportsUShort = false;
        bool supportsHalf = false;
        bool supportsFloat = false;
        BitDepthEnum outputBitDepthMap[10];
        const PropertySet &effectProps = desc.getPropertySet();
        int numPixelDepths = effectProps.propGetDimension(kOfxImageEffectPropSupportedPixelDepths);
        for (int i = 0; i < numPixelDepths; ++i) {
            BitDepthEnum pixelDepth = mapStrToBitDepthEnum( effectProps.propGetString(kOfxImageEffectPropSupportedPixelDepths, i) );
            bool supported = getImageEffectHostDescription()->supportsBitDepth(pixelDepth);
            switch (pixelDepth) {
            case eBitDepthUByte:
                supportsByte  = supported;
                break;
            case eBitDepthUShort:
                supportsUShort = supported;
                break;
            case eBitDepthHalf:
                supportsHalf = supported;
                break;
            case eBitDepthFloat:
                supportsFloat = supported;
                break;
            default:
                // other bitdepths are not supported by this plugin
                break;
            }
        }
        {
            int i = 0;
            if (supportsFloat) {
                outputBitDepthMap[i] = eBitDepthFloat;
                ++i;
            }
            if (supportsHalf) {
                outputBitDepthMap[i] = eBitDepthHalf;
                ++i;
            }
            if (supportsUShort) {
                outputBitDepthMap[i] = eBitDepthUShort;
                ++i;
            }
            if (supportsByte) {
                outputBitDepthMap[i] = eBitDepthUByte;
                ++i;
            }
            outputBitDepthMap[i] = eBitDepthNone;
        }

        {
            bool supportsRGBA   = false;
            bool supportsRGB    = false;
            bool supportsXY     = false;
            bool supportsAlpha  = false;
            PixelComponentEnum outputComponentsMap[10];
            const PropertySet &dstClipProps = dstClip.getPropertySet();
            int numComponents = dstClipProps.propGetDimension(kOfxImageEffectPropSupportedComponents);
            for (int i = 0; i < numComponents; ++i) {
                PixelComponentEnum pixelComponents = mapStrToPixelComponentEnum( dstClipProps.propGetString(kOfxImageEffectPropSupportedComponents, i) );
                bool supported = getImageEffectHostDescription()->supportsPixelComponent(pixelComponents);
                switch (pixelComponents) {
                case ePixelComponentRGBA:
                    supportsRGBA  = supported;
                    break;
                case ePixelComponentRGB:
                    supportsRGB = supported;
                    break;
                case ePixelComponentXY:
                    supportsXY = supported;
                    break;
                case ePixelComponentAlpha:
                    supportsAlpha = supported;
                    break;
                default:
                    // other components are not supported by this plugin
                    break;
                }
            }
            {
                int i = 0;
                if (supportsRGBA) {
                    outputComponentsMap[i] = ePixelComponentRGBA;
                    ++i;
                }
                if (supportsRGB) {
                    outputComponentsMap[i] = ePixelComponentRGB;
                    ++i;
                }
                if (supportsXY) {
                    outputComponentsMap[i] = ePixelComponentXY;
                    ++i;
                }
                if (supportsAlpha) {
                    outputComponentsMap[i] = ePixelComponentAlpha;
                    ++i;
                }
                outputComponentsMap[i] = ePixelComponentNone;
            }

            // outputComponents
            {
                ChoiceParamDescriptor *param = desc.defineChoiceParam(kParamGeneratorOutputComponents);
                param->setLabel(kParamGeneratorOutputComponentsLabel);
                param->setHint(kParamGeneratorOutputComponentsHint);
                // the following must be in the same order as in describe(), so that the map works
                int defIndex = 0;
                int nOptions = 0;
                if (supportsRGBA) {
                    assert(outputComponentsMap[param->getNOptions()] == ePixelComponentRGBA);
                    param->appendOption(kParamGeneratorOutputComponentsOptionRGBA);
                    if (defaultComponents == ePixelComponentRGBA) {
                        defIndex = nOptions;
                    }
                    ++nOptions;
                }
                if (supportsRGB) {
                    assert(outputComponentsMap[param->getNOptions()] == ePixelComponentRGB);
                    param->appendOption(kParamGeneratorOutputComponentsOptionRGB);
                    if (defaultComponents == ePixelComponentRGB) {
                        defIndex = nOptions;
                    }
                    ++nOptions;
                }
                if (supportsXY) {
                    assert(outputComponentsMap[param->getNOptions()] == ePixelComponentXY);
                    param->appendOption(kParamGeneratorOutputComponentsOptionXY);
                    if (defaultComponents == ePixelComponentXY) {
                        defIndex = nOptions;
                    }
                    ++nOptions;
                }
                if (supportsAlpha) {
                    assert(outputComponentsMap[param->getNOptions()] == ePixelComponentAlpha);
                    param->appendOption(kParamGeneratorOutputComponentsOptionAlpha);
                    if (defaultComponents == ePixelComponentAlpha) {
                        defIndex = nOptions;
                    }
                    ++nOptions;
                }
                param->setDefault(defIndex);
                param->setAnimates(false);
                desc.addClipPreferencesSlaveParam(*param);
                if (page) {
                    page->addChild(*param);
                }
            }
        }

        // ouputBitDepth
        if (getImageEffectHostDescription()->supportsMultipleClipDepths) {
            ChoiceParamDescriptor *param = desc.defineChoiceParam(kParamGeneratorOutputBitDepth);
            param->setLabel(kParamGeneratorOutputBitDepthLabel);
            param->setHint(kParamGeneratorOutputBitDepthHint);
            // the following must be in the same order as in describe(), so that the map works
            if (supportsFloat) {
                // coverity[check_return]
                assert(0 <= param->getNOptions() && param->getNOptions() < 10 && outputBitDepthMap[param->getNOptions()] == eBitDepthFloat);
                param->appendOption(kParamGeneratorOutputBitDepthOptionFloat);
            }
            if (supportsHalf) {
                // coverity[check_return]
                assert(0 <= param->getNOptions() && param->getNOptions() < 10 && outputBitDepthMap[param->getNOptions()] == eBitDepthHalf);
                param->appendOption(kParamGeneratorOutputBitDepthOptionHalf);
            }
            if (supportsUShort) {
                // coverity[check_return]
                assert(0 <= param->getNOptions() && param->getNOptions() < 10 && outputBitDepthMap[param->getNOptions()] == eBitDepthUShort);
                param->appendOption(kParamGeneratorOutputBitDepthOptionShort);
            }
            if (supportsByte) {
                // coverity[check_return]
                assert(0 <= param->getNOptions() && param->getNOptions() < 10 && outputBitDepthMap[param->getNOptions()] == eBitDepthUByte);
                param->appendOption(kParamGeneratorOutputBitDepthOptionByte);
            }
            param->setDefault(0);
#ifndef DEBUG
            // Shuffle only does linear conversion, which is useless for 8-bits and 16-bits formats.
            // Disable it for now (in the future, there may be colorspace conversion options)
            param->setIsSecretAndDisabled(true); // always secret
#endif
            param->setAnimates(false);
            desc.addClipPreferencesSlaveParam(*param);
            if (page) {
                page->addChild(*param);
            }
        }
    } // useOutputComponentsAndDepth


    // Some hosts (e.g. Resolve) may not support normalized defaults (setDefaultCoordinateSystem(eCoordinatesNormalised))
    if (!gHostSupportsDefaultCoordinateSystem) {
        BooleanParamDescriptor* param = desc.defineBooleanParam(kParamDefaultsNormalised);
        param->setDefault(true);
        param->setEvaluateOnChange(false);
        param->setIsSecretAndDisabled(true);
        param->setIsPersistent(true);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }
} // generatorDescribeInContext
} // OFX
