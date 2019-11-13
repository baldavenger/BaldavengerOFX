/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of openfx-io <https://github.com/MrKepzie/openfx-io>,
 * Copyright (C) 2015 INRIA
 *
 * openfx-io is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * openfx-io is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with openfx-io.  If not, see <http://www.gnu.org/licenses/gpl-2.0.html>
 * ***** END LICENSE BLOCK ***** */

/*
 * OFX GenericReader plugin.
 * A base class for all OpenFX-based decoders.
 */

#include "GenericReader.h"

#include <climits>
#include <cmath>
#include <cfloat> // DBL_MAX
#include <memory>
#include <algorithm>
#include <fstream>
#ifdef DEBUG
#include <cstdio>
#define DBG(x) x
#else
#define DBG(x) (void)0
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include "ofxsLog.h"
#include "ofxsCopier.h"
#include "ofxsCoords.h"
#include "ofxsMacros.h"

#ifdef OFX_EXTENSIONS_TUTTLE
#include <tuttle/ofxReadWrite.h>
#endif
//#include <ofxNatron.h>

#include "SequenceParsing/SequenceParsing.h"
#ifdef OFX_IO_USING_OCIO
#include "GenericOCIO.h"
#endif
#include "IOUtility.h"

#ifdef OFX_IO_USING_OCIO
namespace OCIO = OCIO_NAMESPACE;
#endif

using std::string;

NAMESPACE_OFX_ENTER
NAMESPACE_OFX_IO_ENTER

#define kPluginGrouping "Image/Readers"

// in the Reader context, the script name must be kOfxImageEffectFileParamName, @see kOfxImageEffectContextReader
#define kParamFilename kOfxImageEffectFileParamName
#define kParamFilenameLabel "File"
#define kParamFilenameHint "The input image sequence/video stream file(s)."

#define kParamProxy kOfxImageEffectProxyParamName
#define kParamProxyLabel "Proxy File"
#define kParamProxyHint \
    "Filename of the proxy images. They will be used instead of the images read from the File parameter " \
    "when the proxy mode (downscaling of the images) is activated."

#define kParamProxyThreshold "proxyThreshold"
#define kParamProxyThresholdLabel "Proxy threshold"
#define kParamProxyThresholdHint \
    "The scale of the proxy images. By default it will be automatically computed out of the " \
    "images headers when you set the proxy file(s) path. When the render scale (proxy) is set to " \
    "a scale lower or equal to this value then the proxy image files will be used instead of the " \
    "original images. You can change this parameter by checking the \"Custom scale\" checkbox " \
    "so that you can change the scale at which the proxy images should be used instead of the original images."

#define kParamOriginalProxyScale "originalProxyScale"
#define kParamOriginalProxyScaleLabel "Original Proxy Scale"
#define kParamOriginalProxyScaleHint \
    "The original scale of the proxy image."

#define kParamCustomProxyScale "customProxyScale"
#define kParamCustomProxyScaleLabel "Custom Proxy Scale"
#define kParamCustomProxyScaleHint \
    "Check to enable the Proxy scale edition."

#define kParamOnMissingFrame "onMissingFrame"
#define kParamOnMissingFrameLabel "On Missing Frame"
#define kParamOnMissingFrameHint \
    "What to do when a frame is missing from the sequence/stream."

#define kParamFrameMode "frameMode"
#define kParamFrameModeLabel "Frame Mode"
enum FrameModeEnum
{
    eFrameModeStartingTime,
    eFrameModeTimeOffset,
};

#define kParamFrameModeOptionStartingTime "Starting Time"
#define kParamFrameModeOptionTimeOffset "Time Offset"

#define kParamTimeOffset "timeOffset"
#define kParamTimeOffsetLabel "Time Offset"
#define kParamTimeOffsetHint \
    "Offset applied to the sequence in time units (i.e. frames)."

#define kParamStartingTime "startingTime"
#define kParamStartingTimeLabel "Starting Time"
#define kParamStartingTimeHint \
    "At what time (on the timeline) should this sequence/video start."

#define kParamOriginalFrameRange "originalFrameRange"
#define kParamOriginalFrameRangeLabel "Original Range"

#define kParamFirstFrame "firstFrame"
#define kParamFirstFrameLabel "First Frame"
#define kParamFirstFrameHint \
    "The first frame this sequence/video should start at. This cannot be less " \
    " than the first frame of the sequence and cannot be greater than the last" \
    " frame of the sequence."

#define kParamLastFrame "lastFrame"
#define kParamLastFrameLabel "Last Frame"
#define kParamLastFrameHint \
    "The frame this sequence/video should end at. This cannot be lesser " \
    "than the first frame of the sequence and cannot be greater than the last " \
    "frame of the sequence."

#define kParamBefore "before"
#define kParamBeforeLabel "Before"
#define kParamBeforeHint \
    "What to do before the first frame of the sequence."

#define kParamAfter "after"
#define kParamAfterLabel "After"
#define kParamAfterHint \
    "What to do after the last frame of the sequence."

#define kParamTimeDomainUserEdited "timeDomainUserEdited"


enum MissingEnum
{
    eMissingPrevious,
    eMissingNext,
    eMissingNearest,
    eMissingError,
    eMissingBlack,
};

#define kReaderOptionHold "Hold"
#define kReaderOptionHoldHint "While before the sequence, load the first frame."
#define kReaderOptionLoop "Loop"
#define kReaderOptionLoopHint "Repeat the sequence before the first frame"
#define kReaderOptionBounce "Bounce"
#define kReaderOptionBounceHint "Repeat the sequence in reverse before the first frame"
#define kReaderOptionBlack "Black"
#define kReaderOptionBlackHint "Render a black image"
#define kReaderOptionError "Error"
#define kReaderOptionErrorHint "Report an error"
#define kReaderOptionPrevious "Hold previous"
#define kReaderOptionPreviousHint "Try to load the previous frame in the sequence/stream, if any."
#define kReaderOptionNext "Load next"
#define kReaderOptionNextHint "Try to load the next frame in the sequence/stream, if any."
#define kReaderOptionNearest "Load nearest"
#define kReaderOptionNearestHint "Try to load the nearest frame in the sequence/stream, if any."

#define kParamFilePremult "filePremult"
#define kParamFilePremultLabel "File Premult"
#define kParamFilePremultHint \
    "The image file being read is considered to have this premultiplication state.\n" \
    "To get UnPremultiplied (or \"unassociated alpha\") images, set the \"Output Premult\" parameter to Unpremultiplied. \n" \
    "By default the value should be correctly be guessed by the image file, but this parameter can be edited if the metadatas inside the file are wrong.\n" \
    "- Opaque means that the alpha channel is considered to be 1 (one), and it is not taken into account in colorspace conversion.\n" \
    "- Premultiplied, red, green and blue channels are divided by the alpha channel " \
    "before applying the colorspace conversion, and re-multiplied by alpha after colorspace conversion.\n" \
    "- UnPremultiplied, means that red, green and blue channels are not modified " \
    "before applying the colorspace conversion, and are multiplied by alpha after colorspace conversion.\n" \
    "This is set automatically from the image file and the plugin, but can be adjusted if this information is wrong in the file metadata.\n" \
    "RGB images can only be Opaque, and Alpha images can only be Premultiplied (the value of this parameter doesn't matter).\n"
#define kParamFilePremultOptionOpaqueHint \
    "The image is opaque and so has no premultiplication state, as if the alpha component in all pixels were set to the white point."
#define kParamFilePremultOptionPreMultipliedHint \
    "The image is premultiplied by its alpha (also called \"associated alpha\")."
#define kParamFilePremultOptionUnPreMultipliedHint \
    "The image is unpremultiplied (also called \"unassociated alpha\")."

#define kParamOutputPremult "outputPremult"
#define kParamOutputPremultLabel "Output Premult"
#define kParamOutputPremultHint "The alpha premultiplication in output of this node will have this state."

#define kParamOutputComponents "outputComponents"
#define kParamOutputComponentsLabel "Output Components"
#define kParamOutputComponentsHint "What type of components this effect should output when the main color plane is requested." \
    " For the Read node it will map (in number of components) the Output Layer choice to these."
#define kParamOutputComponentsOptionRGBA "RGBA"
#define kParamOutputComponentsOptionRGB "RGB"
#define kParamOutputComponentsOptionXY "RG"
#define kParamOutputComponentsOptionAlpha "Alpha"

#define kParamInputSpaceLabel "File Colorspace"

#define kParamFrameRate "frameRate"
#define kParamFrameRateLabel "Frame rate"
#define kParamFrameRateHint "By default this value is guessed from the file. You can override it by checking the Custom fps parameter. " \
    "The value of this parameter is what will be visible by the effects down-stream."

#define kParamCustomFps "customFps"
#define kParamCustomFpsLabel "Custom FPS"
#define kParamCustomFpsHint "If checked, you can freely force the value of the frame rate parameter. The frame-rate is just the meta-data that will be passed " \
    "downstream to the graph, no retime will actually take place."

#define kParamGuessedParams "ParamExistingInstance" // was guessParamsFromFilename already successfully called once on this instance

#ifdef OFX_IO_USING_OCIO
#define kParamInputSpaceSet "ocioInputSpaceSet" // was the input colorspace set by user?
#endif

#define MISSING_FRAME_NEAREST_RANGE 100

#define kSupportsMultiResolution 1
#define kSupportsRenderScale 1 // GenericReader supports render scale: it scales images and uses proxy image when applicable

#define GENERIC_READER_USE_MULTI_THREAD

static bool gHostIsNatron   = false;
static bool gHostSupportsRGBA   = false;
static bool gHostSupportsRGB    = false;
static bool gHostSupportsXY    = false;
static bool gHostSupportsAlpha  = false;
static const char*
premultString(PreMultiplicationEnum e)
{
    switch (e) {
    case eImageOpaque:

        return "Opaque";
    case eImagePreMultiplied:

        return "PreMultiplied";
    case eImageUnPreMultiplied:

        return "UnPreMultiplied";
    }

    return "Unknown";
}

GenericReaderPlugin::GenericReaderPlugin(OfxImageEffectHandle handle,
                                         const std::vector<string>& extensions,
                                         bool supportsRGBA,
                                         bool supportsRGB,
                                         bool supportsXY,
                                         bool supportsAlpha,
                                         bool supportsTiles,
                                         bool isMultiPlanar)
    : ImageEffect(handle)
    , _missingFrameParam(0)
#ifdef OFX_IO_USING_OCIO
    , _ocio( new GenericOCIO(this) )
#endif
    , _outputClip(0)
    , _fileParam(0)
    , _timeOffset(0)
    , _startingTime(0)
    , _originalFrameRange(0)
    , _proxyFileParam(0)
    , _proxyThreshold(0)
    , _originalProxyScale(0)
    , _enableCustomScale(0)
    , _firstFrame(0)
    , _beforeFirst(0)
    , _lastFrame(0)
    , _afterLast(0)
    , _frameMode(0)
    , _outputComponents(0)
    , _filePremult(0)
    , _outputPremult(0)
    , _timeDomainUserSet(0)
    , _customFPS(0)
    , _fps(0)
    , _sublabel(0)
    , _guessedParams(0)
    , _extensions(extensions)
    , _supportsRGBA(supportsRGBA)
    , _supportsRGB(supportsRGB)
    , _supportsXY(supportsXY)
    , _supportsAlpha(supportsAlpha)
    , _supportsTiles(supportsTiles)
    , _isMultiPlanar(isMultiPlanar)
{
    _outputClip = fetchClip(kOfxImageEffectOutputClipName);

    _fileParam = fetchStringParam(kParamFilename);
    _proxyFileParam = fetchStringParam(kParamProxy);
    _proxyThreshold = fetchDouble2DParam(kParamProxyThreshold);
    _originalProxyScale = fetchDouble2DParam(kParamOriginalProxyScale);
    _enableCustomScale = fetchBooleanParam(kParamCustomProxyScale);
    _missingFrameParam = fetchChoiceParam(kParamOnMissingFrame);
    _firstFrame = fetchIntParam(kParamFirstFrame);
    _beforeFirst = fetchChoiceParam(kParamBefore);
    _lastFrame = fetchIntParam(kParamLastFrame);
    _afterLast = fetchChoiceParam(kParamAfter);
    _frameMode = fetchChoiceParam(kParamFrameMode);
    _timeOffset = fetchIntParam(kParamTimeOffset);
    _startingTime = fetchIntParam(kParamStartingTime);
    _originalFrameRange = fetchInt2DParam(kParamOriginalFrameRange);
    _timeDomainUserSet = fetchBooleanParam(kParamTimeDomainUserEdited);
    _outputComponents = fetchChoiceParam(kParamOutputComponents);
    _filePremult = fetchChoiceParam(kParamFilePremult);
    _outputPremult = fetchChoiceParam(kParamOutputPremult);
    _customFPS = fetchBooleanParam(kParamCustomFps);
    _fps = fetchDoubleParam(kParamFrameRate);
    if (gHostIsNatron) {
        _sublabel = fetchStringParam(kNatronOfxParamStringSublabelName);
        assert(_sublabel);
    }
    _guessedParams = fetchBooleanParam(kParamGuessedParams);


    // must be in sync with GenericReaderDescribeInContextBegin
    int i = 0;

    if (gHostSupportsRGBA && supportsRGBA) {
        _outputComponentsTable[i] = ePixelComponentRGBA;
        ++i;
    }
    if (gHostSupportsRGB && supportsRGB) {
        _outputComponentsTable[i] = ePixelComponentRGB;
        ++i;
    }
    if (gHostSupportsXY && supportsXY) {
        _outputComponentsTable[i] = ePixelComponentXY;
        ++i;
    }

    if (gHostSupportsAlpha && supportsAlpha) {
        _outputComponentsTable[i] = ePixelComponentAlpha;
        ++i;
    }
    _outputComponentsTable[i] = ePixelComponentNone;
}

GenericReaderPlugin::~GenericReaderPlugin()
{
}

void
GenericReaderPlugin::refreshSubLabel(OfxTime time)
{
    assert(_sublabel);
    double sequenceTime;
    GetSequenceTimeRetEnum getTimeRet = getSequenceTimeHold(time, &sequenceTime);
    if ( (getTimeRet == eGetSequenceTimeWithinSequence) ||
         ( getTimeRet == eGetSequenceTimeBeforeSequence) ||
         ( getTimeRet == eGetSequenceTimeAfterSequence) ) {
        string filename;
        GetFilenameRetCodeEnum getFileNameRet = getFilenameAtSequenceTime(sequenceTime, false, false, &filename);
        if (getFileNameRet == eGetFileNameReturnedFullRes) {
            _sublabel->setValue( basename(filename) );
        } else {
            _sublabel->setValue("");
        }
    } else {
        _sublabel->setValue("");
    }
}

/**
 * @brief Restore any state from the parameters set
 * Called from createInstance() and changedParam() (via changedFilename()), must restore the
 * state of the Reader, such as Choice param options, data members and non-persistent param values.
 * We don't do this in the ctor of the plug-in since we can't call virtuals yet.
 * Any derived implementation must call GenericReaderPlugin::restoreStateFromParams() first
 **/
void
GenericReaderPlugin::restoreStateFromParams()
{
    int frameMode_i;

    _frameMode->getValue(frameMode_i);
    FrameModeEnum frameMode = FrameModeEnum(frameMode_i);
    switch (frameMode) {
    case eFrameModeStartingTime:     //starting frame
        _startingTime->setIsSecretAndDisabled(false);
        _timeOffset->setIsSecretAndDisabled(true);
        break;
    case eFrameModeTimeOffset:     //time offset
        _startingTime->setIsSecretAndDisabled(true);
        _timeOffset->setIsSecretAndDisabled(false);
        break;
    }

    ///Detect the scale of the proxy.
    string proxyFile;
    _proxyFileParam->getValue(proxyFile);
    if ( !proxyFile.empty() ) {
        _proxyThreshold->setIsSecretAndDisabled(false);
        _enableCustomScale->setIsSecretAndDisabled(false);
    } else {
        _proxyThreshold->setIsSecretAndDisabled(true);
        _enableCustomScale->setIsSecretAndDisabled(true);
    }

    bool customFps = _customFPS->getValue();
    _fps->setEnabled(customFps);

    if (gHostIsNatron) {
        refreshSubLabel( timeLineGetTime() );
    }
}

bool
GenericReaderPlugin::getTimeDomain(OfxRangeD &range)
{
    OfxRangeI rangeI;
    bool ret = getSequenceTimeDomainInternal(rangeI, false);

    if (ret) {
        ///these are the value held by the "First frame" and "Last frame" param
        OfxRangeI sequenceTimeDomain;
        sequenceTimeDomain.min = _firstFrame->getValue();
        sequenceTimeDomain.max = _lastFrame->getValue();
        int startingTime = _startingTime->getValue();
        timeDomainFromSequenceTimeDomain(sequenceTimeDomain, startingTime, &rangeI);
        range.min = rangeI.min;
        range.max = rangeI.max;
    }

    return ret;
}

bool
GenericReaderPlugin::getSequenceTimeDomainInternal(OfxRangeI& range,
                                                   bool canSetOriginalFrameRange)
{
    ////first-off check if the original frame range param has valid values, in which
    ///case we don't bother calculating the frame range
    int originalMin, originalMax;

    _originalFrameRange->getValue(originalMin, originalMax);

    // test if the host (probably Natron) set kParamOriginalFrameRange
    if ( (originalMin != INT_MIN) && (originalMax != INT_MAX) ) {
        range.min = originalMin;
        range.max = originalMax;

        return true;
    }

    string filename;
    _fileParam->getValue(filename);

    ///call the plugin specific getTimeDomain (if it is a video-stream , it is responsible to
    ///find-out the time domain. If this function return false, it means this is an image sequence
    ///in which case our sequence parser will give us the sequence range
    bool gotRange = getSequenceTimeDomain(filename, range);

    if (!gotRange) {
        // POOR MAN's solution: try to guess the range

        SequenceParsing::FileNameContent content(filename);
        string pattern;
        ///We try to match all the files in the same directory that match the pattern with the frame number
        ///assumed to be in the last part of the filename. This is a harsh assumption but we can't just verify
        ///everything as it would take too much time.
        string noStr;
        int nbFrameNumbers = content.getPotentialFrameNumbersCount();
        content.getNumberByIndex(nbFrameNumbers - 1, &noStr);

        int numHashes = content.getLeadingZeroes();
        string noStrWithoutZeroes;
        for (std::size_t i = 0; i < noStr.size(); ++i) {
            if ( (noStr[i] == '0') && noStrWithoutZeroes.empty() ) {
                continue;
            }
            noStrWithoutZeroes.push_back(noStr[i]);
        }

        if ( (int)noStr.size() > numHashes ) {
            numHashes += noStrWithoutZeroes.size();
        } else {
            numHashes = 1;
        }
        content.generatePatternWithFrameNumberAtIndex(nbFrameNumbers - 1,
                                                      numHashes,
                                                      &pattern);


        SequenceParsing::SequenceFromPattern sequenceFromFiles;
        SequenceParsing::filesListFromPattern_slow(pattern, &sequenceFromFiles);

        range.min = range.max = 1;
        if (sequenceFromFiles.size() > 1) {
            range.min = sequenceFromFiles.begin()->first;
            range.max = sequenceFromFiles.rbegin()->first;
        }
    }


    //// From http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#SettingParams
    //    Plugins are free to set parameters in limited set of circumstances, typically relating to user interaction. You can only set parameters in the following actions passed to the plug-in's main entry function...
    //
    //    The Create Instance Action
    //    The The Begin Instance Changed Action
    //    The The Instance Changed Action
    //    The The End Instance Changed Action
    //    The The Sync Private Data Action
    if (!filename.empty() && canSetOriginalFrameRange) {
        _originalFrameRange->setValue(range.min, range.max);
    }

    return true;
} // GenericReaderPlugin::getSequenceTimeDomainInternal

void
GenericReaderPlugin::timeDomainFromSequenceTimeDomain(const OfxRangeI& sequenceTimeDomain,
                                                      int startingTime,
                                                      OfxRangeI* timeDomain)
{
    timeDomain->min = startingTime;
    timeDomain->max = startingTime + (sequenceTimeDomain.max - sequenceTimeDomain.min);
}

GenericReaderPlugin::GetSequenceTimeRetEnum
GenericReaderPlugin::getSequenceTimeBefore(const OfxRangeI& sequenceTimeDomain,
                                           double t,
                                           BeforeAfterEnum beforeChoice,
                                           double *sequenceTime) const
{
    ///get the offset from the starting time of the sequence in case we bounce or loop
    int timeOffsetFromStart = (int)t -  sequenceTimeDomain.min;

    switch (beforeChoice) {
    case eBeforeAfterHold:     //hold
        *sequenceTime = sequenceTimeDomain.min;

        return eGetSequenceTimeBeforeSequence;

    case eBeforeAfterLoop:     //loop
        timeOffsetFromStart %= (int)(sequenceTimeDomain.max - sequenceTimeDomain.min + 1);
        *sequenceTime = sequenceTimeDomain.max + timeOffsetFromStart;

        return eGetSequenceTimeBeforeSequence;

    case eBeforeAfterBounce: {     //bounce
        int range = sequenceTimeDomain.max - sequenceTimeDomain.min;
        // guard against division by zero
        int sequenceIntervalsCount = range ? (timeOffsetFromStart / range) : 0;
        ///if the sequenceIntervalsCount is odd then do exactly like loop, otherwise do the load the opposite frame
        if (sequenceIntervalsCount % 2 == 0) {
            if (range != 0) {
                timeOffsetFromStart %= range;
            }
            *sequenceTime = sequenceTimeDomain.min - timeOffsetFromStart;
        } else {
            if (range != 0) {
                timeOffsetFromStart %= range;
            }
            *sequenceTime = sequenceTimeDomain.max + timeOffsetFromStart;
        }

        return eGetSequenceTimeBeforeSequence;
    }
    case eBeforeAfterBlack:     //black
        return eGetSequenceTimeBlack;

    case eBeforeAfterError:     //error

        //setPersistentMessage(Message::eMessageError, "", "Out of frame range");
        return eGetSequenceTimeError;
    }

    return eGetSequenceTimeError;
}

GenericReaderPlugin::GetSequenceTimeRetEnum
GenericReaderPlugin::getSequenceTimeAfter(const OfxRangeI& sequenceTimeDomain,
                                          double t,
                                          BeforeAfterEnum afterChoice,
                                          double *sequenceTime) const
{
    ///get the offset from the starting time of the sequence in case we bounce or loop
    int timeOffsetFromStart = (int)t -  sequenceTimeDomain.min;


    switch (afterChoice) {
    case eBeforeAfterHold: {     //hold
        *sequenceTime = sequenceTimeDomain.max;

        return eGetSequenceTimeAfterSequence;
    }
    case eBeforeAfterLoop: {     //loop
        int range = sequenceTimeDomain.max - sequenceTimeDomain.min + 1;
        if (range > 0) {
            timeOffsetFromStart %= range;
        }
        *sequenceTime = sequenceTimeDomain.min + timeOffsetFromStart;

        return eGetSequenceTimeAfterSequence;
    }
    case eBeforeAfterBounce: {     //bounce
        int range = sequenceTimeDomain.max - sequenceTimeDomain.min;
        // guard against division by zero
        int sequenceIntervalsCount = range ? (timeOffsetFromStart / range) : 0;

        ///if the sequenceIntervalsCount is odd then do exactly like loop, otherwise do the load the opposite frame
        if (range > 0) {
            timeOffsetFromStart %= range;
        }
        if (sequenceIntervalsCount % 2 == 0) {
            *sequenceTime = sequenceTimeDomain.min + timeOffsetFromStart;
        } else {
            *sequenceTime = sequenceTimeDomain.max - timeOffsetFromStart;
        }

        return eGetSequenceTimeAfterSequence;
    }
    case eBeforeAfterBlack: {     //black
        return eGetSequenceTimeBlack;
    }
    case eBeforeAfterError: {     //error
        //setPersistentMessage(Message::eMessageError, "", "Out of frame range");

        return eGetSequenceTimeError;
    }
    }

    return eGetSequenceTimeError;
} // GenericReaderPlugin::getSequenceTimeAfter

GenericReaderPlugin::GetSequenceTimeRetEnum
GenericReaderPlugin::getSequenceTimeHold(double t,
                                         double *sequenceTime) const
{
    int timeOffset;

    _timeOffset->getValue(timeOffset);


    ///get the time sequence domain
    OfxRangeI sequenceTimeDomain;
    _firstFrame->getValue(sequenceTimeDomain.min);
    _lastFrame->getValue(sequenceTimeDomain.max);


    ///the return value
    *sequenceTime = t - timeOffset;

    ///if the time given is before the sequence
    if ( (sequenceTimeDomain.min <= *sequenceTime) && (*sequenceTime <= sequenceTimeDomain.max) ) {
        return eGetSequenceTimeWithinSequence;
    } else if (*sequenceTime < sequenceTimeDomain.min) {
        return getSequenceTimeBefore(sequenceTimeDomain, t, eBeforeAfterHold, sequenceTime);
    } else {
        assert(*sequenceTime > sequenceTimeDomain.max); ///the time given is after the sequence
        return getSequenceTimeAfter(sequenceTimeDomain, t, eBeforeAfterHold, sequenceTime);
    }

    return eGetSequenceTimeError;
}

GenericReaderPlugin::GetSequenceTimeRetEnum
GenericReaderPlugin::getSequenceTime(double t,
                                     double *sequenceTime) const
{
    int timeOffset;

    _timeOffset->getValue(timeOffset);


    ///get the time sequence domain
    OfxRangeI sequenceTimeDomain;
    _firstFrame->getValue(sequenceTimeDomain.min);
    _lastFrame->getValue(sequenceTimeDomain.max);


    ///the return value
    *sequenceTime = t - timeOffset;

    ///if the time given is before the sequence
    if ( (sequenceTimeDomain.min <= *sequenceTime) && (*sequenceTime <= sequenceTimeDomain.max) ) {
        return eGetSequenceTimeWithinSequence;
    } else if (*sequenceTime < sequenceTimeDomain.min) {
        /////if we're before the first frame
        int beforeChoice_i;
        _beforeFirst->getValue(beforeChoice_i);
        BeforeAfterEnum beforeChoice = BeforeAfterEnum(beforeChoice_i);

        return getSequenceTimeBefore(sequenceTimeDomain, t, beforeChoice, sequenceTime);
    } else {
        assert(*sequenceTime > sequenceTimeDomain.max); ///the time given is after the sequence
        /////if we're after the last frame
        int afterChoice_i;
        _afterLast->getValue(afterChoice_i);
        BeforeAfterEnum afterChoice = BeforeAfterEnum(afterChoice_i);

        return getSequenceTimeAfter(sequenceTimeDomain, t, afterChoice, sequenceTime);
    }

    return eGetSequenceTimeError;
}

#ifdef _WIN32
std::wstring
utf8ToUtf16 (const string& str)
{
    std::wstring native;

    native.resize( MultiByteToWideChar (CP_UTF8, 0, str.c_str(), -1, NULL, 0) );
    MultiByteToWideChar ( CP_UTF8, 0, str.c_str(), -1, &native[0], (int)native.size() );

    return native;
}

#endif

static bool
checkIfFileExists (const string& path)
{
#ifdef _WIN32
    WIN32_FIND_DATAW FindFileData;
    std::wstring wpath = utf8ToUtf16 (path);
    HANDLE handle = FindFirstFileW(wpath.c_str(), &FindFileData);
    if (handle != INVALID_HANDLE_VALUE) {
        FindClose(handle);

        return true;
    }

    return false;
#else
    // on Unix platforms passing in UTF-8 works
    std::ifstream fs( path.c_str() );

    return fs.is_open() && fs.good();
#endif
}

GenericReaderPlugin::GetFilenameRetCodeEnum
GenericReaderPlugin::getFilenameAtSequenceTime(double sequenceTime,
                                               bool proxyFiles,
                                               bool checkForExistingFile,
                                               string *filename) const
{
    GetFilenameRetCodeEnum ret;
    const MissingEnum missingFrame = (MissingEnum)_missingFrameParam->getValue();
    string filename0;
    bool filenameGood = true;
    int offset = 0;

    sequenceTime = std::floor(sequenceTime + 0.5); // round to the nearest frame

    const bool searchOtherFrame = ( (missingFrame == eMissingPrevious) ||
                                    (missingFrame == eMissingNearest) ||
                                    (missingFrame == eMissingNext) ||
                                    (missingFrame == eMissingNearest) );
    do {
        _fileParam->getValueAtTime(sequenceTime + offset, *filename);
        if (offset == 0) {
            filename0 = *filename; // for error reporting
        }
        if ( filename->empty() ) {
            //filenameGood = false;
            return eGetFileNameBlack; // if filename is empty, just return a black frame. this happens eg when the plugin is created
        } else {
            if (checkForExistingFile) {
                filenameGood = checkIfFileExists(*filename);
            }
        }
        if (filenameGood) {
            ret = eGetFileNameReturnedFullRes;
            // now, try the proxy file
            if (proxyFiles) {
                string proxyFileName;
                bool proxyGood = true;
                _proxyFileParam->getValueAtTime(sequenceTime + offset, proxyFileName);
                if ( proxyFileName.empty() ) {
                    proxyGood = false;
                } else {
                    if (checkForExistingFile) {
                        proxyGood = checkIfFileExists(proxyFileName);
                    }
                }
                if (proxyGood) {
                    // proxy file exists, replace the filename with the proxy name
                    *filename = proxyFileName;
                    ret = eGetFileNameReturnedProxy;
                }
            }
        }
        if (missingFrame == eMissingPrevious) {
            --offset;
        } else if (missingFrame == eMissingNext) {
            ++offset;
        } else if (missingFrame == eMissingNearest) {
            if (offset <= 0) {
                offset = -offset + 1;
            } else if (sequenceTime - offset >= 0) {
                offset = -offset;
            } else {
                ++offset;
            }
        }
    } while ( searchOtherFrame &&                     // only loop if searchOtherFrame
              !filenameGood &&                       // and no valid file was found
              std::abs(offset) <= MISSING_FRAME_NEAREST_RANGE && // and we are within range

              (sequenceTime + offset >= 0) ); // and index is positive
    if (filenameGood) {
        // ret is already set (see above)
    } else {
        *filename = filename0; // reset to the original frame name;
        switch (missingFrame) {
        case eMissingPrevious:     // Hold previous
        case eMissingNext:        // Load next
        case eMissingNearest:     // Load nearest
        case eMissingError:       // Error
            /// For images sequences, we can safely say this is  a missing frame. For video-streams we do not know and the derived class
            // will have to handle the case itself.
            ret = eGetFileNameFailed;
            // return a black image
            break;
        case eMissingBlack:      // Black image
            /// For images sequences, we can safely say this is  a missing frame. For video-streams we do not know and the derived class
            // will have to handle the case itself.
            ret = eGetFileNameBlack;
            break;
        }
    }

    return ret;
} // GenericReaderPlugin::getFilenameAtSequenceTime

OfxStatus
GenericReaderPlugin::getFilenameAtTime(double t,
                                       string *filename) const
{
    double sequenceTime;
    GetSequenceTimeRetEnum getSequenceTimeRet = getSequenceTime(t, &sequenceTime);

    switch (getSequenceTimeRet) {
    case eGetSequenceTimeBlack:

        return kOfxStatReplyDefault;

    case eGetSequenceTimeError:

        return kOfxStatFailed;

    case eGetSequenceTimeBeforeSequence:
    case eGetSequenceTimeWithinSequence:
    case eGetSequenceTimeAfterSequence:
        break;
    }
    GetFilenameRetCodeEnum getFilenameAtSequenceTimeRet = getFilenameAtSequenceTime(sequenceTime, false, true, filename);
    switch (getFilenameAtSequenceTimeRet) {
    case eGetFileNameFailed:

        // do not setPersistentMessage()!
        return kOfxStatFailed;

    case eGetFileNameBlack:

        return kOfxStatReplyDefault;

    case eGetFileNameReturnedFullRes:
    case eGetFileNameReturnedProxy:
        break;
    }

    return kOfxStatOK;
}

int
GenericReaderPlugin::getStartingTime() const
{
    int startingTime;

    _startingTime->getValue(startingTime);

    return startingTime;
}

void
GenericReaderPlugin::copyPixelData(const OfxRectI& renderWindow,
                                   const void *srcPixelData,
                                   const OfxRectI& srcBounds,
                                   PixelComponentEnum srcPixelComponents,
                                   int srcPixelComponentCount,
                                   BitDepthEnum srcBitDepth,
                                   int srcRowBytes,
                                   void *dstPixelData,
                                   const OfxRectI& dstBounds,
                                   PixelComponentEnum dstPixelComponents,
                                   int dstPixelComponentCount,
                                   BitDepthEnum dstBitDepth,
                                   int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);
    assert(srcBounds.y1 <= renderWindow.y1 && renderWindow.y1 <= renderWindow.y2 && renderWindow.y2 <= srcBounds.y2);
    assert(srcBounds.x1 <= renderWindow.x1 && renderWindow.x1 <= renderWindow.x2 && renderWindow.x2 <= srcBounds.x2);

#ifdef GENERIC_READER_USE_MULTI_THREAD
    copyPixels(*this, renderWindow,
               srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
               dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
#else
    copyPixelsNT(*this, renderWindow,
                 srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                 dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
#endif
}

// update the window of dst defined by nextRenderWindow by halving the corresponding area in src.
// proofread and fixed by F. Devernay on 3/10/2014
template <typename PIX>
static void
halveWindow(const OfxRectI& nextRenderWindow,
            const PIX* srcPixels,
            const OfxRectI& srcBounds,
            int srcRowBytes,
            PIX* dstPixels,
            const OfxRectI& dstBounds,
            int dstRowBytes,
            int nComponents)
{
    int srcRowSize = srcRowBytes / sizeof(PIX);
    int dstRowSize = dstRowBytes / sizeof(PIX);
    const PIX* srcData =  srcPixels - (srcBounds.x1 * nComponents + srcRowSize * srcBounds.y1);
    PIX* dstData = dstPixels - (dstBounds.x1 * nComponents + dstRowSize * dstBounds.y1);

    assert(nextRenderWindow.x1 * 2 >= (srcBounds.x1 - 1) && (nextRenderWindow.x2 - 1) * 2 < srcBounds.x2 &&
           nextRenderWindow.y1 * 2 >= (srcBounds.y1 - 1) && (nextRenderWindow.y2 - 1) * 2 < srcBounds.y2);
    for (int y = nextRenderWindow.y1; y < nextRenderWindow.y2; ++y) {
        const PIX* srcLineStart = srcData + y * 2 * srcRowSize;
        PIX* dstLineStart = dstData + y * dstRowSize;
        bool pickNextRow = (y * 2) < (srcBounds.y2 - 1);
        bool pickThisRow = (y * 2) >= (srcBounds.y1);
        int sumH = (int)pickNextRow + (int)pickThisRow;
        assert(sumH == 1 || sumH == 2);
        for (int x = nextRenderWindow.x1; x < nextRenderWindow.x2; ++x) {
            bool pickNextCol = (x * 2) < (srcBounds.x2 - 1);
            bool pickThisCol = (x * 2) >= (srcBounds.x1);
            int sumW = (int)pickThisCol + (int)pickNextCol;
            assert(sumW == 1 || sumW == 2);
            for (int k = 0; k < nComponents; ++k) {
                ///a b
                ///c d

                PIX a = (pickThisCol && pickThisRow) ? srcLineStart[x * 2 * nComponents + k] : 0;
                PIX b = (pickNextCol && pickThisRow) ? srcLineStart[(x * 2 + 1) * nComponents + k] : 0;
                PIX c = (pickThisCol && pickNextRow) ? srcLineStart[(x * 2 * nComponents) + srcRowSize + k] : 0;
                PIX d = (pickNextCol && pickNextRow) ? srcLineStart[(x * 2 + 1) * nComponents + srcRowSize + k] : 0;

                assert( sumW == 2 || ( sumW == 1 && ( (a == 0 && c == 0) || (b == 0 && d == 0) ) ) );
                assert( sumH == 2 || ( sumH == 1 && ( (a == 0 && b == 0) || (c == 0 && d == 0) ) ) );
                int sum = sumH * sumW;
                // guard against division by zero
                assert(sum);
                dstLineStart[x * nComponents + k] = sum ? ( (a + b + c + d) / sum ) : 0;
            }
        }
    }
}

template <typename PIX>
static void
buildMipMapLevelGeneric(ImageEffect* instance,
                        const OfxRectI& originalRenderWindow,
                        const OfxRectI& renderWindowFullRes,
                        unsigned int level,
                        const PIX* srcPixels,
                        const OfxRectI& srcBounds,
                        int srcRowBytes,
                        PIX* dstPixels,
                        const OfxRectI& dstBounds,
                        int dstRowBytes,
                        int nComponents)
{
    assert(level > 0);

    std::auto_ptr<ImageMemory> mem;
    size_t memSize = 0;
    std::auto_ptr<ImageMemory> tmpMem;
    size_t tmpMemSize = 0;
    PIX* nextImg = NULL;
    const PIX* previousImg = srcPixels;
    OfxRectI previousBounds = srcBounds;
    int previousRowBytes = srcRowBytes;
    OfxRectI nextRenderWindow = renderWindowFullRes;

    ///Build all the mipmap levels until we reach the one we are interested in
    for (unsigned int i = 1; i < level; ++i) {
        // loop invariant:
        // - previousImg, previousBounds, previousRowBytes describe the data ate the level before i
        // - nextRenderWindow contains the renderWindow at the level before i
        //
        ///Halve the smallest enclosing po2 rect as we need to render a minimum of the renderWindow
        nextRenderWindow = downscalePowerOfTwoSmallestEnclosing(nextRenderWindow, 1);
#     ifdef DEBUG
        {
            // check that doing i times 1 level is the same as doing i levels
            OfxRectI nrw = downscalePowerOfTwoSmallestEnclosing(renderWindowFullRes, i);
            assert(nrw.x1 == nextRenderWindow.x1 && nrw.x2 == nextRenderWindow.x2 && nrw.y1 == nextRenderWindow.y1 && nrw.y2 == nextRenderWindow.y2);
        }
#     endif
        ///Allocate a temporary image if necessary, or reuse the previously allocated buffer
        int nextRowBytes =  (nextRenderWindow.x2 - nextRenderWindow.x1)  * nComponents * sizeof(PIX);
        size_t newMemSize =  (size_t)(nextRenderWindow.y2 - nextRenderWindow.y1) * (size_t)nextRowBytes;
        if ( tmpMem.get() ) {
            // there should be enough memory: no need to reallocate
            assert(tmpMemSize >= memSize);
        } else {
            tmpMem.reset( new ImageMemory(newMemSize, instance) );
            tmpMemSize = newMemSize;
        }
        nextImg = (float*)tmpMem->lock();

        halveWindow<PIX>(nextRenderWindow, previousImg, previousBounds, previousRowBytes, nextImg, nextRenderWindow, nextRowBytes, nComponents);

        ///Switch for next pass
        previousBounds = nextRenderWindow;
        previousRowBytes = nextRowBytes;
        previousImg = nextImg;
        mem = tmpMem;
        memSize = tmpMemSize;
    }
    // here:
    // - previousImg, previousBounds, previousRowBytes describe the data ate the level before 'level'
    // - nextRenderWindow contains the renderWindow at the level before 'level'

    ///On the last iteration halve directly into the dstPixels
    ///The nextRenderWindow should be equal to the original render window.
    nextRenderWindow = downscalePowerOfTwoSmallestEnclosing(nextRenderWindow, 1);
    assert(originalRenderWindow.x1 == nextRenderWindow.x1 && originalRenderWindow.x2 == nextRenderWindow.x2 &&
           originalRenderWindow.y1 == nextRenderWindow.y1 && originalRenderWindow.y2 == nextRenderWindow.y2);

    halveWindow<PIX>(nextRenderWindow, previousImg, previousBounds, previousRowBytes, dstPixels, dstBounds, dstRowBytes, nComponents);
    // mem and tmpMem are freed at destruction
} // buildMipMapLevelGeneric

// update the window of dst defined by originalRenderWindow by mipmapping the windows of src defined by renderWindowFullRes
// proofread and fixed by F. Devernay on 3/10/2014
template <typename PIX, int nComponents>
static void
buildMipMapLevel(ImageEffect* instance,
                 const OfxRectI& originalRenderWindow,
                 const OfxRectI& renderWindowFullRes,
                 unsigned int level,
                 const PIX* srcPixels,
                 const OfxRectI& srcBounds,
                 int srcRowBytes,
                 PIX* dstPixels,
                 const OfxRectI& dstBounds,
                 int dstRowBytes)
{
    buildMipMapLevelGeneric<PIX>(instance, originalRenderWindow, renderWindowFullRes, level, srcPixels, srcBounds, srcRowBytes
                                 , dstPixels, dstBounds, dstRowBytes, nComponents);
}

void
GenericReaderPlugin::scalePixelData(const OfxRectI& originalRenderWindow,
                                    const OfxRectI& renderWindow,
                                    unsigned int levels,
                                    const void* srcPixelData,
                                    PixelComponentEnum srcPixelComponents,
                                    int srcPixelComponentCount,
                                    BitDepthEnum srcPixelDepth,
                                    const OfxRectI& srcBounds,
                                    int srcRowBytes,
                                    void* dstPixelData,
                                    PixelComponentEnum dstPixelComponents,
                                    int dstPixelComponentCount,
                                    BitDepthEnum dstPixelDepth,
                                    const OfxRectI& dstBounds,
                                    int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);

    // do the rendering
    if ( (dstPixelDepth != eBitDepthFloat) ||
         ( ( dstPixelComponents != ePixelComponentRGBA) &&
           ( dstPixelComponents != ePixelComponentRGB) &&
           ( dstPixelComponents != ePixelComponentXY) &&
           ( dstPixelComponents != ePixelComponentAlpha) &&
           ( dstPixelComponents != ePixelComponentCustom) ) ||
         ( dstPixelDepth != srcPixelDepth) ||
         ( dstPixelComponents != srcPixelComponents) ||
         ( dstPixelComponentCount != srcPixelComponentCount) ) {
        throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }

    if (dstPixelComponents == ePixelComponentRGBA) {
        if (!_supportsRGBA) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        buildMipMapLevel<float, 4>(this, originalRenderWindow, renderWindow, levels, (const float*)srcPixelData,
                                   srcBounds, srcRowBytes, (float*)dstPixelData, dstBounds, dstRowBytes);
    } else if (dstPixelComponents == ePixelComponentRGB) {
        if (!_supportsRGB) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        buildMipMapLevel<float, 3>(this, originalRenderWindow, renderWindow, levels, (const float*)srcPixelData,
                                   srcBounds, srcRowBytes, (float*)dstPixelData, dstBounds, dstRowBytes);
    } else if (dstPixelComponents == ePixelComponentXY) {
        if (!_supportsXY) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        buildMipMapLevel<float, 2>(this, originalRenderWindow, renderWindow, levels, (const float*)srcPixelData,
                                   srcBounds, srcRowBytes, (float*)dstPixelData, dstBounds, dstRowBytes);
    }  else if (dstPixelComponents == ePixelComponentAlpha) {
        if (!_supportsAlpha) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        buildMipMapLevel<float, 1>(this, originalRenderWindow, renderWindow, levels, (const float*)srcPixelData,
                                   srcBounds, srcRowBytes, (float*)dstPixelData, dstBounds, dstRowBytes);
    } else {
        assert(dstPixelComponents == ePixelComponentCustom);

        buildMipMapLevelGeneric<float>(this, originalRenderWindow, renderWindow, levels, (const float*)srcPixelData,
                                       srcBounds, srcRowBytes, (float*)dstPixelData, dstBounds, dstRowBytes, dstPixelComponentCount);
    }
} // GenericReaderPlugin::scalePixelData

/* set up and run a copy processor */
static void
setupAndFillWithBlack(PixelProcessorFilterBase & processor,
                      const OfxRectI &renderWindow,
                      void *dstPixelData,
                      const OfxRectI& dstBounds,
                      PixelComponentEnum dstPixelComponents,
                      int dstPixelComponentCount,
                      BitDepthEnum dstPixelDepth,
                      int dstRowBytes)
{
    // set the images
    processor.setDstImg(dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstPixelDepth, dstRowBytes);

    // set the render window
    processor.setRenderWindow(renderWindow);

    // Call the base class process member, this will call the derived templated process code
    processor.process();
}

void
GenericReaderPlugin::fillWithBlack(const OfxRectI &renderWindow,
                                   void *dstPixelData,
                                   const OfxRectI& dstBounds,
                                   PixelComponentEnum dstPixelComponents,
                                   int dstPixelComponentCount,
                                   BitDepthEnum dstBitDepth,
                                   int dstRowBytes)
{
    BlackFiller<float> fred(*this, dstPixelComponentCount);
    setupAndFillWithBlack(fred, renderWindow, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

static void
setupAndProcess(PixelProcessorFilterBase & processor,
                int premultChannel,
                const OfxRectI &renderWindow,
                const void *srcPixelData,
                const OfxRectI& srcBounds,
                PixelComponentEnum srcPixelComponents,
                int srcPixelComponentCount,
                BitDepthEnum srcPixelDepth,
                int srcRowBytes,
                void *dstPixelData,
                const OfxRectI& dstBounds,
                PixelComponentEnum dstPixelComponents,
                int dstPixelComponentCount,
                BitDepthEnum dstPixelDepth,
                int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);

    // make sure bit depths are sane
    if ( (srcPixelDepth != dstPixelDepth) || (srcPixelComponents != dstPixelComponents) ) {
        throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }

    // set the images
    processor.setDstImg(dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstPixelDepth, dstRowBytes);
    processor.setSrcImg(srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcPixelDepth, srcRowBytes, 0);

    // set the render window
    processor.setRenderWindow(renderWindow);

    processor.setPremultMaskMix(true, premultChannel, 1.);

    // Call the base class process member, this will call the derived templated process code
    processor.process();
}

void
GenericReaderPlugin::unPremultPixelData(const OfxRectI &renderWindow,
                                        const void *srcPixelData,
                                        const OfxRectI& srcBounds,
                                        PixelComponentEnum srcPixelComponents,
                                        int srcPixelComponentCount,
                                        BitDepthEnum srcPixelDepth,
                                        int srcRowBytes,
                                        void *dstPixelData,
                                        const OfxRectI& dstBounds,
                                        PixelComponentEnum dstPixelComponents,
                                        int dstPixelComponentCount,
                                        BitDepthEnum dstBitDepth,
                                        int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);

    // do the rendering
    if ( (dstBitDepth != eBitDepthFloat) || ( (dstPixelComponents != ePixelComponentRGBA) && (dstPixelComponents != ePixelComponentRGB) && (dstPixelComponents != ePixelComponentAlpha) ) ) {
        throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    if (dstPixelComponents == ePixelComponentRGBA) {
        if (!_supportsRGBA) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        PixelCopierUnPremult<float, 4, 1, float, 4, 1> fred(*this);
        setupAndProcess(fred, 3, renderWindow, srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcPixelDepth, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else {
        ///other pixel components means you want to copy only...
        assert(false);
    }
}

void
GenericReaderPlugin::premultPixelData(const OfxRectI &renderWindow,
                                      const void *srcPixelData,
                                      const OfxRectI& srcBounds,
                                      PixelComponentEnum srcPixelComponents,
                                      int srcPixelComponentCount,
                                      BitDepthEnum srcPixelDepth,
                                      int srcRowBytes,
                                      void *dstPixelData,
                                      const OfxRectI& dstBounds,
                                      PixelComponentEnum dstPixelComponents,
                                      int dstPixelComponentCount,
                                      BitDepthEnum dstBitDepth,
                                      int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);

    // do the rendering
    if ( (dstBitDepth != eBitDepthFloat) || ( (dstPixelComponents != ePixelComponentRGBA) && (dstPixelComponents != ePixelComponentRGB) && (dstPixelComponents != ePixelComponentAlpha) ) ) {
        throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }

    if (dstPixelComponents == ePixelComponentRGBA) {
        if (!_supportsRGBA) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        PixelCopierPremult<float, 4, 1, float, 4, 1> fred(*this);
        setupAndProcess(fred, 3, renderWindow, srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcPixelDepth, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else {
        ///other pixel components means you want to copy only...
        assert(false);
    }
}

bool
GenericReaderPlugin::getRegionOfDefinition(const RegionOfDefinitionArguments &args,
                                           OfxRectD &rod)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);

        return false;
    }

    double sequenceTime;
    GetSequenceTimeRetEnum getSequenceTimeRet = getSequenceTime(args.time, &sequenceTime);
    switch (getSequenceTimeRet) {
    case eGetSequenceTimeBlack:

        return false;

    case eGetSequenceTimeError:
        throwSuiteStatusException(kOfxStatFailed);

        return false;

    case eGetSequenceTimeBeforeSequence:
    case eGetSequenceTimeWithinSequence:
    case eGetSequenceTimeAfterSequence:
        break;
    }

    /*We don't want to use the proxy image for the region of definition*/
    string filename;
    GetFilenameRetCodeEnum getFilenameAtSequenceTimeRet = getFilenameAtSequenceTime(sequenceTime, false, true, &filename);
    switch (getFilenameAtSequenceTimeRet) {
    case eGetFileNameFailed:
        setPersistentMessage(Message::eMessageError, "", filename + ": Cannot load frame");
        throwSuiteStatusException(kOfxStatFailed);

        return false;

    case eGetFileNameBlack:
        clearPersistentMessage();

        return false;

    case eGetFileNameReturnedFullRes:
    case eGetFileNameReturnedProxy:
        clearPersistentMessage();
        break;
    }

    string error;
    OfxRectI bounds, format;
    double par = 1.;
    int tile_width, tile_height;
    bool success = getFrameBounds(filename, sequenceTime, &bounds, &format, &par, &error, &tile_width, &tile_height);
    if (!success) {
        setPersistentMessage(Message::eMessageError, "", error);
        throwSuiteStatusException(kOfxStatFailed);

        return false;
    }
    // get the PAR from the clip preferences, since it should be constant over time
    par = _outputClip->getPixelAspectRatio();
    rod.x1 = bounds.x1 * par;
    rod.x2 = bounds.x2 * par;
    rod.y1 = bounds.y1;
    rod.y2 = bounds.y2;

//    if (getFilenameAtSequenceTimeRet == eGetFileNameReturnedProxy) {
//        ///upscale the proxy RoD to be in canonical coords.
//        unsigned int mipmapLvl = getLevelFromScale(args.renderScale.x);
//        rod = upscalePowerOfTwo(rod, (double)mipmapLvl);
//    }

    return true;
} // GenericReaderPlugin::getRegionOfDefinition

class OutputImagesHolder_RAII
{
    std::vector<Image*> dstImages;

public:

    OutputImagesHolder_RAII()
        : dstImages()
    {
    }

    void appendImage(Image* img)
    {
        dstImages.push_back(img);
    }

    const std::vector<Image*>& getOutputPlanes() const
    {
        return dstImages;
    }

    ~OutputImagesHolder_RAII()
    {
        for (std::size_t i = 0; i < dstImages.size(); ++i) {
            delete dstImages[i];
        }
    }
};

void
GenericReaderPlugin::render(const RenderArguments &args)
{
    if (!_outputClip) {
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }

    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }

    assert( kSupportsRenderScale || (args.renderScale.x == 1. && args.renderScale.y == 1.) );
    ///The image will have the appropriate size since we support the render scale (multi-resolution)
    OutputImagesHolder_RAII outputImagesHolder;
#ifdef OFX_EXTENSIONS_NUKE
    if ( !isMultiPlanar() ) {
#endif

    Image* dstImg = _outputClip->fetchImage(args.time);
    if (!dstImg) {
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }
#ifndef NDEBUG
    if (!_supportsTiles) {
        // http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#kOfxImageEffectPropSupportsTiles
        //  If a clip or plugin does not support tiled images, then the host should supply full RoD images to the effect whenever it fetches one.
        OfxRectI dstRod;     // = dst->getRegionOfDefinition(); //  Nuke's image RoDs are wrong
        Coords::toPixelEnclosing(_outputClip->getRegionOfDefinition(args.time), args.renderScale, _outputClip->getPixelAspectRatio(), &dstRod);
        const OfxRectI& dstBounds = dstImg->getBounds();

        assert(dstRod.x1 == dstBounds.x1);
        assert(dstRod.x2 == dstBounds.x2);
        assert(dstRod.y1 == dstBounds.y1);
        assert(dstRod.y2 == dstBounds.y2);     // crashes on Natron if kSupportsTiles=0 & kSupportsMultiResolution=1
    }
#endif

    outputImagesHolder.appendImage(dstImg);
#ifdef OFX_EXTENSIONS_NUKE
} else {
    assert(getImageEffectHostDescription()->isMultiPlanar);

    for (std::list<string>::const_iterator it = args.planes.begin(); it != args.planes.end(); ++it) {
        Image* dstPlane = _outputClip->fetchImagePlane( args.time, args.renderView, it->c_str() );
        if (!dstPlane) {
            throwSuiteStatusException(kOfxStatFailed);

            return;
        }
        outputImagesHolder.appendImage(dstPlane);
    }
}

#endif

    OfxRectI firstBounds = {0, 0, 0, 0};
    BitDepthEnum firstDepth = eBitDepthNone;
    bool firstImageSet = false;

    std::list<PlaneToRender> planes;

    const std::vector<Image*>& outputImages = outputImagesHolder.getOutputPlanes();
    for (std::size_t i = 0; i < outputImages.size(); ++i) {
        if ( (outputImages[i]->getRenderScale().x != args.renderScale.x) ||
             ( outputImages[i]->getRenderScale().y != args.renderScale.y) ||
             ( outputImages[i]->getField() != args.fieldToRender) ) {
            setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
            throwSuiteStatusException(kOfxStatFailed);

            return;
        }

        PlaneToRender plane;
        void* dstPixelData = NULL;
        OfxRectI bounds;
        BitDepthEnum bitDepth;
        getImageData(outputImages[i], &dstPixelData, &bounds, &plane.comps, &bitDepth, &plane.rowBytes);
        if (bitDepth != eBitDepthFloat) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        if (!firstImageSet) {
            firstBounds = bounds;
            firstDepth = bitDepth;
            firstImageSet = true;
        } else {
            if ( (firstBounds.x1 != bounds.x1) || (firstBounds.x2 != bounds.x2) || (firstBounds.y1 != bounds.y1) || (firstBounds.y2 != bounds.y2)
                 || ( firstDepth != bitDepth) ) {
                setPersistentMessage(Message::eMessageError, "", "OFX Host gave image plane with wrong bounds/bitdepth");
                throwSuiteStatusException(kOfxStatFailed);

                return;
            }
        }
        plane.pixelData = (float*)dstPixelData;
        if (!plane.pixelData) {
            setPersistentMessage(Message::eMessageError, "", "OFX Host provided an invalid image buffer");
        }

        plane.rawComps = outputImages[i]->getPixelComponentsProperty();


#ifdef OFX_EXTENSIONS_NUKE
        if (plane.comps != ePixelComponentCustom) {
#endif
        assert(plane.rawComps == kOfxImageComponentAlpha ||
#ifdef OFX_EXTENSIONS_NATRON
               plane.rawComps == kNatronOfxImageComponentXY ||
#endif
               plane.rawComps == kOfxImageComponentRGB || plane.rawComps == kOfxImageComponentRGBA);
        PixelComponentEnum outputComponents = getOutputComponents();
        if (plane.comps != outputComponents) {
            setPersistentMessage(Message::eMessageError, "", "OFX Host dit not take into account output components");
            throwSuiteStatusException(kOfxStatErrImageFormat);

            return;
        }
#ifdef OFX_EXTENSIONS_NUKE
    }
#endif

        plane.numChans = outputImages[i]->getPixelComponentCount();
        planes.push_back(plane);
    }


    // are we in the image bounds
    if ( (args.renderWindow.x1 < firstBounds.x1) || (args.renderWindow.x1 >= firstBounds.x2) || (args.renderWindow.y1 < firstBounds.y1) || (args.renderWindow.y1 >= firstBounds.y2) ||
         ( args.renderWindow.x2 <= firstBounds.x1) || ( args.renderWindow.x2 > firstBounds.x2) || ( args.renderWindow.y2 <= firstBounds.y1) || ( args.renderWindow.y2 > firstBounds.y2) ) {
        throwSuiteStatusException(kOfxStatErrValue);

        //throw std::runtime_error("render window outside of image bounds");
        return;
    }

    double sequenceTime;
    GetSequenceTimeRetEnum getSequenceTimeRet = getSequenceTime(args.time, &sequenceTime);
    switch (getSequenceTimeRet) {
    case eGetSequenceTimeBlack:
        for (std::list<PlaneToRender>::iterator it = planes.begin(); it != planes.end(); ++it) {
            fillWithBlack(args.renderWindow, it->pixelData, firstBounds, it->comps, it->numChans, firstDepth, it->rowBytes);
        }

        return;

    case eGetSequenceTimeError:
        throwSuiteStatusException(kOfxStatFailed);

        return;

    case eGetSequenceTimeBeforeSequence:
    case eGetSequenceTimeWithinSequence:
    case eGetSequenceTimeAfterSequence:
        break;
    }

    bool useProxy = false;
    OfxPointD proxyScaleThreshold;
    _proxyThreshold->getValue(proxyScaleThreshold.x, proxyScaleThreshold.y);

    OfxPointD proxyOriginalScale;
    _originalProxyScale->getValue(proxyOriginalScale.x, proxyOriginalScale.y);

    ///We only support downscaling at a power of two.
    unsigned int renderMipmapLevel = getLevelFromScale( std::min(args.renderScale.x, args.renderScale.y) );
    unsigned int proxyMipMapThresholdLevel = (proxyScaleThreshold.x == 0 || proxyScaleThreshold.y == 0) ? renderMipmapLevel :  getLevelFromScale( std::min(proxyScaleThreshold.x, proxyScaleThreshold.y) );
    unsigned int originalProxyMipMapLevel = (proxyOriginalScale.x == 0 || proxyOriginalScale.y == 0) ? renderMipmapLevel : getLevelFromScale( std::min(proxyOriginalScale.x, proxyOriginalScale.y) );

    if ( kSupportsRenderScale && (renderMipmapLevel >= proxyMipMapThresholdLevel) ) {
        useProxy = true;
    }

    string filename;
    GetFilenameRetCodeEnum getFilenameAtSequenceTimeRet = getFilenameAtSequenceTime(sequenceTime, false, true, &filename);
    switch (getFilenameAtSequenceTimeRet) {
    case eGetFileNameFailed:
        setPersistentMessage(Message::eMessageError, "", filename + ": Cannot load frame");
        throwSuiteStatusException(kOfxStatFailed);

        return;

    case eGetFileNameBlack:
        clearPersistentMessage();
        for (std::list<PlaneToRender>::iterator it = planes.begin(); it != planes.end(); ++it) {
            fillWithBlack(args.renderWindow, it->pixelData, firstBounds, it->comps, it->numChans, firstDepth, it->rowBytes);
        }

        return;

    case eGetFileNameReturnedFullRes:
        clearPersistentMessage();
        break;

    case eGetFileNameReturnedProxy:
        // we didn't ask for proxy!
        assert(false);
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }

    string proxyFile;
    if (useProxy) {
        ///Use the proxy only if getFilenameAtSequenceTime returned a valid proxy filename different from the original file
        GetFilenameRetCodeEnum getFilenameAtSequenceTimeRetPx = getFilenameAtSequenceTime(sequenceTime, true, true, &proxyFile);
        switch (getFilenameAtSequenceTimeRetPx) {
        case eGetFileNameFailed:
            // should never happen: it should return at least the full res frame
            assert(false);
            throwSuiteStatusException(kOfxStatFailed);

            return;

        case eGetFileNameBlack:
            // should never happen: it should return at least the full res frame
            assert(false);
            for (std::list<PlaneToRender>::iterator it = planes.begin(); it != planes.end(); ++it) {
                fillWithBlack(args.renderWindow, it->pixelData, firstBounds, it->comps, it->numChans, firstDepth, it->rowBytes);
            }

            return;

        case eGetFileNameReturnedFullRes:
            assert(proxyFile == filename);
            useProxy = false;
            break;

        case eGetFileNameReturnedProxy:
            assert( !proxyFile.empty() );
            useProxy = (proxyFile != filename);
            break;
        }
    }


    // The args.renderWindow is already in pixels coordinate (render scale is already taken into account).
    // If the filename IS NOT a proxy file we have to make sure the renderWindow is
    // upscaled to a scale of (1,1). On the other hand if the filename IS a proxy we have to determine the actual RoD
    // of the proxy file and adjust the scale so it fits the given scale.
    // When in proxy mode renderWindowFullRes is the render window at the proxy mip map level
    int downscaleLevels = renderMipmapLevel; // the number of mipmap levels from the actual file (proxy or not) to the renderWindow

    if ( useProxy && !proxyFile.empty() ) {
        filename = proxyFile;
        downscaleLevels -= originalProxyMipMapLevel;
    }
    assert(downscaleLevels >= 0);

    if ( filename.empty() || !checkIfFileExists(filename) ) {
        for (std::list<PlaneToRender>::iterator it = planes.begin(); it != planes.end(); ++it) {
            fillWithBlack(args.renderWindow, it->pixelData, firstBounds, it->comps, it->numChans, firstDepth, it->rowBytes);
        }

        return;
    }

    OfxRectI renderWindowFullRes, renderWindowNotRounded;
    OfxRectI frameBounds, format;
    double par = 1.;
    int tile_width, tile_height;
    string error;

    ///if the plug-in doesn't support tiles, just render the full rod
    bool success = getFrameBounds(filename, sequenceTime, &frameBounds, &format, &par, &error, &tile_width, &tile_height);
    ///We shouldve checked above for any failure, now this is too late.
    if (!success) {
        setPersistentMessage(Message::eMessageError, "", error);
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }

    renderWindowFullRes = upscalePowerOfTwo(args.renderWindow, downscaleLevels); // works even if downscaleLevels == 0

    // Intersect the full res renderwindow to the real rod,
    // because we may have gone a few pixels too far (but never more than 2^downscaleLevels-1 pixels)
    assert(renderWindowFullRes.x1 >= frameBounds.x1 - std::pow(2., (double)downscaleLevels) + 1 &&
           renderWindowFullRes.x2 <= frameBounds.x2 + std::pow(2., (double)downscaleLevels) - 1 &&
           renderWindowFullRes.y1 >= frameBounds.y1 - std::pow(2., (double)downscaleLevels) + 1 &&
           renderWindowFullRes.y2 <= frameBounds.y2 + std::pow(2., (double)downscaleLevels) - 1);
    intersect(renderWindowFullRes, frameBounds, &renderWindowFullRes);

    //See below: we round the render window to the tile size
    renderWindowNotRounded = renderWindowFullRes;

    for (std::list<PlaneToRender>::iterator it = planes.begin(); it != planes.end(); ++it) {
        // Read into a temporary image, apply colorspace conversion, then copy
        bool isOCIOIdentity;

        // if components are custom, remap it to a OFX components with the same number of channels
        PixelComponentEnum remappedComponents;

        bool isColor;
        bool isCustom;
        if (it->comps == ePixelComponentCustom) {
            std::vector<string> channelNames = mapPixelComponentCustomToLayerChannels(it->rawComps);
            isColor = channelNames.size() >= 4 && channelNames[1] == "R" && channelNames[2] == "G" && channelNames[3] == "B";
            isCustom = true;
            if (isColor) {
#ifdef OFX_IO_USING_OCIO
                isOCIOIdentity = _ocio->isIdentity(args.time);
#else
                isOCIOIdentity = true;
#endif
            } else {
                isOCIOIdentity = true;
            }

            if (it->numChans == 3) {
                remappedComponents = ePixelComponentRGB;
            } else if (it->numChans == 4) {
                remappedComponents = ePixelComponentRGBA;
            } else if (it->numChans == 2) {
                remappedComponents = ePixelComponentXY;
            } else {
                remappedComponents = ePixelComponentAlpha;
            }
        } else {
            isColor = true;
            isCustom = false;
#ifdef OFX_IO_USING_OCIO
            isOCIOIdentity = _ocio->isIdentity(args.time);
#else
            isOCIOIdentity = true;
#endif
            remappedComponents = it->comps;
        }

        PreMultiplicationEnum filePremult = eImageUnPreMultiplied;
        PreMultiplicationEnum outputPremult = eImageUnPreMultiplied;
        if ( (it->comps == ePixelComponentRGB) || (it->comps == ePixelComponentXY) || ( isCustom && isColor && (remappedComponents == ePixelComponentRGB) ) ) {
            filePremult = outputPremult = eImageOpaque;
        } else if ( (it->comps == ePixelComponentAlpha) || ( isCustom && isColor && (remappedComponents == ePixelComponentAlpha) ) ) {
            filePremult = outputPremult = eImagePreMultiplied;
        } else if ( (it->comps == ePixelComponentRGBA) || ( isCustom && isColor && (remappedComponents == ePixelComponentRGBA) ) ) {
            int premult_i;
            _filePremult->getValue(premult_i);
            filePremult = (PreMultiplicationEnum)premult_i;

            int oPremult_i;
            _outputPremult->getValue(oPremult_i);
            outputPremult = (PreMultiplicationEnum)oPremult_i;
        }


        // we have to do the final premultiplication if:
        // - pixelComponents is RGBA
        //  AND
        //   - buffer is PreMultiplied AND OCIO is not identity (OCIO works only on unpremultiplied data)
        //   OR
        //   - premult is unpremultiplied
        bool mustPremult = ( (remappedComponents == ePixelComponentRGBA) &&
                             ( (filePremult == eImageUnPreMultiplied || !isOCIOIdentity) && outputPremult == eImagePreMultiplied ) );


        if ( !mustPremult && isOCIOIdentity && ( !kSupportsRenderScale || (renderMipmapLevel == 0) ) ) {
            // no colorspace conversion, no premultiplication, no proxy, just read file
            DBG( std::printf("decode (to dst)\n") );

            if (!_isMultiPlanar) {
                decode(filename, sequenceTime, args.renderView, args.sequentialRenderStatus, args.renderWindow, it->pixelData, firstBounds, it->comps, it->numChans, it->rowBytes);
            } else {
                decodePlane(filename, sequenceTime, args.renderView, args.sequentialRenderStatus, args.renderWindow, it->pixelData, firstBounds, it->comps, it->numChans, it->rawComps, it->rowBytes);
            }
        } else {
            int pixelBytes;
            if (it->comps == ePixelComponentCustom) {
                pixelBytes = it->numChans * sizeof(float);
            } else {
                pixelBytes = it->numChans * getComponentBytes(firstDepth);
            }
            assert(pixelBytes > 0);

            /*
               If tile_width and tile_height is set, round the renderWindow to the enclosing tile size to make sure the plug-in has a buffer
               large enough to decode tiles. This is needed for OpenImageIO. Note that
             */
            if ( (tile_width > 0) && (tile_height > 0) ) {
                double frameHeight = frameBounds.y2 - frameBounds.y1;
                if ( isTileOrientationTopDown() ) {
                    //invert Y before rounding

                    renderWindowFullRes.y1 = frameHeight -  renderWindowFullRes.y1;
                    renderWindowFullRes.y2 = frameHeight  - renderWindowFullRes.y2;
                    frameBounds.y1 = frameHeight  - frameBounds.y1;
                    frameBounds.y2 = frameHeight  - frameBounds.y2;

                    renderWindowFullRes.x1 = std::min( (double)std::ceil( (double)renderWindowFullRes.x1 / tile_width ) * tile_width, (double)frameBounds.x1 );
                    renderWindowFullRes.y1 = std::min( (double)std::ceil( (double)renderWindowFullRes.y1 / tile_height ) * tile_height, (double)frameBounds.y1 );
                    renderWindowFullRes.x2 = std::max( (double)std::floor( (double)renderWindowFullRes.x2 / tile_width ) * tile_width, (double)frameBounds.x2 );
                    renderWindowFullRes.y2 = std::max( (double)std::floor( (double)renderWindowFullRes.y2 / tile_height ) * tile_height, (double)frameBounds.y2 );
                } else {
                    renderWindowFullRes.x1 = std::max( (double)std::floor( (double)renderWindowFullRes.x1 / tile_width ) * tile_width, (double)frameBounds.x1 );
                    renderWindowFullRes.y1 = std::max( (double)std::floor( (double)renderWindowFullRes.y1 / tile_height ) * tile_height, (double)frameBounds.y1 );
                    renderWindowFullRes.x2 = std::min( (double)std::ceil( (double)renderWindowFullRes.x2 / tile_width ) * tile_width, (double)frameBounds.x2 );
                    renderWindowFullRes.y2 = std::min( (double)std::ceil( (double)renderWindowFullRes.y2 / tile_height ) * tile_height, (double)frameBounds.y2 );
                }

                if ( isTileOrientationTopDown() ) {
                    //invert back Y
                    renderWindowFullRes.y1 = frameHeight - renderWindowFullRes.y1;
                    renderWindowFullRes.y2 = frameHeight  - renderWindowFullRes.y2;
                    frameBounds.y1 = frameHeight - frameBounds.y1;
                    frameBounds.y2 = frameHeight - frameBounds.y2;
                }
            }

            int tmpRowBytes = (renderWindowFullRes.x2 - renderWindowFullRes.x1) * pixelBytes;
            size_t memSize = (size_t)(renderWindowFullRes.y2 - renderWindowFullRes.y1) * (size_t)tmpRowBytes;
            ImageMemory mem(memSize, this);
            float *tmpPixelData = (float*)mem.lock();

            // read file
            DBG( std::printf("decode (to tmp)\n") );

            if (!_isMultiPlanar) {
                decode(filename, sequenceTime, args.renderView, args.sequentialRenderStatus, renderWindowFullRes, tmpPixelData, renderWindowFullRes, it->comps, it->numChans, tmpRowBytes);
            } else {
                decodePlane(filename, sequenceTime, args.renderView, args.sequentialRenderStatus, renderWindowFullRes, tmpPixelData, renderWindowFullRes, it->comps, it->numChans, it->rawComps, tmpRowBytes);
            }

            if ( abort() ) {
                return;
            }

            ///do the color-space conversion
            if ( !isOCIOIdentity && (it->comps != ePixelComponentAlpha) && (it->comps != ePixelComponentXY) ) {
                if (filePremult == eImagePreMultiplied) {
                    assert(remappedComponents == ePixelComponentRGBA);
                    DBG( std::printf("unpremult (tmp in-place)\n") );
                    //tmpPixelData[0] = tmpPixelData[1] = tmpPixelData[2] = tmpPixelData[3] = 0.5;
                    unPremultPixelData(renderWindowNotRounded, tmpPixelData, renderWindowFullRes, it->comps, it->numChans, firstDepth, tmpRowBytes, tmpPixelData, renderWindowFullRes, remappedComponents, it->numChans, firstDepth, tmpRowBytes);

                    if ( abort() ) {
                        return;
                    }

                    //assert(tmpPixelData[0] == 1. && tmpPixelData[1] == 1. && tmpPixelData[2] == 1. && tmpPixelData[3] == 0.5);
                }
#ifdef OFX_IO_USING_OCIO
                DBG( std::printf("OCIO (tmp in-place)\n") );
                _ocio->apply(args.time, renderWindowFullRes, tmpPixelData, renderWindowFullRes, remappedComponents, it->numChans, tmpRowBytes);
#endif
            }

            if ( kSupportsRenderScale && (downscaleLevels > 0) ) {
                if (!mustPremult) {
                    // we can write directly to dstPixelData
                    /// adjust the scale to match the given output image
                    DBG( std::printf("scale (no premult, tmp to dst)\n") );
                    scalePixelData(args.renderWindow, renderWindowNotRounded, (unsigned int)downscaleLevels, tmpPixelData, remappedComponents,
                                   it->numChans, firstDepth, renderWindowFullRes, tmpRowBytes, it->pixelData,
                                   remappedComponents, it->numChans, firstDepth, firstBounds, it->rowBytes);
                } else {
                    // allocate a temporary image (we must avoid reading from dstPixelData, in case several threads are rendering the same area)
                    int mem2RowBytes = (firstBounds.x2 - firstBounds.x1) * pixelBytes;
                    size_t mem2Size = (size_t)(firstBounds.y2 - firstBounds.y1) * (size_t)mem2RowBytes;
                    ImageMemory mem2(mem2Size, this);
                    float *scaledPixelData = (float*)mem2.lock();

                    /// adjust the scale to match the given output image
                    DBG( std::printf("scale (tmp to scaled)\n") );
                    scalePixelData(args.renderWindow, renderWindowNotRounded, (unsigned int)downscaleLevels, tmpPixelData,
                                   remappedComponents, it->numChans, firstDepth,
                                   renderWindowFullRes, tmpRowBytes, scaledPixelData,
                                   remappedComponents, it->numChans, firstDepth,
                                   firstBounds, mem2RowBytes);

                    if ( abort() ) {
                        return;
                    }

                    // apply premult
                    DBG( std::printf("premult (scaled to dst)\n") );
                    //scaledPixelData[0] = scaledPixelData[1] = scaledPixelData[2] = 1.; scaledPixelData[3] = 0.5;
                    premultPixelData(args.renderWindow, scaledPixelData, firstBounds, remappedComponents,  it->numChans, firstDepth, mem2RowBytes, it->pixelData, firstBounds, remappedComponents, it->numChans, firstDepth, it->rowBytes);
                    //assert(dstPixelDataF[0] == 0.5 && dstPixelDataF[1] == 0.5 && dstPixelDataF[2] == 0.5 && dstPixelDataF[3] == 0.5);
                }
            } else {
                // copy
                if (mustPremult) {
                    DBG( std::printf("premult (no scale, tmp to dst)\n") );
                    //tmpPixelData[0] = tmpPixelData[1] = tmpPixelData[2] = 1.; tmpPixelData[3] = 0.5;
                    premultPixelData(args.renderWindow, tmpPixelData, renderWindowFullRes, remappedComponents, it->numChans, firstDepth, tmpRowBytes, it->pixelData, firstBounds, remappedComponents, it->numChans, firstDepth, it->rowBytes);
                    //assert(dstPixelDataF[0] == 0.5 && dstPixelDataF[1] == 0.5 && dstPixelDataF[2] == 0.5 && dstPixelDataF[3] == 0.5);
                } else {
                    DBG( std::printf("copy (no premult no scale, tmp to dst)\n") );
                    copyPixelData(args.renderWindow, tmpPixelData, renderWindowFullRes, remappedComponents, it->numChans, firstDepth, tmpRowBytes, it->pixelData, firstBounds, remappedComponents, it->numChans, firstDepth, it->rowBytes);
                }
            }
            mem.unlock();
        }
    } // for (std::list<PlaneToRender>::iterator it = planes.begin(); it!=planes.end(); ++it) {
}

void
GenericReaderPlugin::decode(const string& /*filename*/,
                            OfxTime /*time*/,
                            int /*view*/,
                            bool /*isPlayback*/,
                            const OfxRectI& /*renderWindow*/,
                            float */*pixelData*/,
                            const OfxRectI& /*bounds*/,
                            PixelComponentEnum /*pixelComponents*/,
                            int /*pixelComponentCount*/,
                            int /*rowBytes*/)
{
    //does nothing
}

void
GenericReaderPlugin::decodePlane(const string& /*filename*/,
                                 OfxTime /*time*/,
                                 int /*view*/,
                                 bool /*isPlayback*/,
                                 const OfxRectI& /*renderWindow*/,
                                 float */*pixelData*/,
                                 const OfxRectI& /*bounds*/,
                                 PixelComponentEnum /*pixelComponents*/,
                                 int /*pixelComponentCount*/,
                                 const string& /*rawComponents*/,
                                 int /*rowBytes*/)
{
    //does nothing
}

bool
GenericReaderPlugin::checkExtension(const string& ext)
{
    if ( ext.empty() ) {
        // no extension
        return false;
    }

    return std::find(_extensions.begin(), _extensions.end(), ext) != _extensions.end();
}

/**
 * @brief Called from changedParam() when kParamFilename is changed for any reason other than eChangeTime.
 * Calls restoreStateFromParams() to update any non-persistent params that may depend on the filename.
 * If reason is eChangeUserEdit and the params where never guessed (see _guessedParams) also sets these from the file contents.
 * Any derived implementation must call GenericReaderPlugin::changedFilename() first
 **/
void
GenericReaderPlugin::changedFilename(const InstanceChangedArgs &args)
{
    assert(args.reason != eChangeTime);
    if (args.reason == eChangeTime) {
        return;
    }
    string filename;

    _fileParam->getValue(filename);

    clearPersistentMessage();

    if ( filename.empty() ) {
        // if the file name is set to an empty string,
        // reset so that values are automatically set on next call to changedFilename()
        _guessedParams->resetToDefault();

        return;
    }

    OfxRangeI sequenceTimeDomain;
    bool gotSequenceTimeDomain = getSequenceTimeDomainInternal(sequenceTimeDomain, true);
    if (!gotSequenceTimeDomain) {
        return;
    }

    bool customFps;
    _customFPS->getValue(customFps);
    if (!customFps) {
        double fps;
        bool gotFps = getFrameRate(filename, &fps);
        if (gotFps) {
            _fps->setValue(fps);
        }
    }

    OfxRangeI timeDomain;

    timeDomainFromSequenceTimeDomain(sequenceTimeDomain, sequenceTimeDomain.min, &timeDomain);

    if (args.reason == eChangeUserEdit) {
        _firstFrame->setValue(sequenceTimeDomain.min);
        _firstFrame->setDefault(sequenceTimeDomain.min);
        _lastFrame->setValue(sequenceTimeDomain.max);
        _lastFrame->setDefault(sequenceTimeDomain.max);
        _startingTime->setDefault(sequenceTimeDomain.min);
    }

    // restore state! let the derived class (which calls GenericReaderPlugin::restoreStateFromParams()) initialize any other parameters or structures
    restoreStateFromParams();

    // should we guess some parameters? only if it's user-set and it's the first time
    if ( (args.reason == eChangeUserEdit) && !_guessedParams->getValue() ) {
        _startingTime->setValue(timeDomain.min);

        ///We call guessParamsFromFilename with the first frame of the sequence so we're almost sure it will work
        ///unless the user did a mistake. We are also safe to assume that images specs are the same for
        ///all the sequence
        _fileParam->getValueAtTime(timeDomain.min, filename);
        if ( filename.empty() ) {
            // no need to trigger a useless error or a persistent message

            return;
        }

        string colorspace;

#ifdef OFX_IO_USING_OCIO
        _ocio->getInputColorspaceDefault(colorspace);
#else
        colorspace = "default";
#endif
        PixelComponentEnum components = ePixelComponentNone;
        int componentCount = 0;
        PreMultiplicationEnum filePremult = eImageOpaque;

        assert( !_guessedParams->getValue() );
        bool success = guessParamsFromFilename(filename, &colorspace, &filePremult, &components, &componentCount);
        if (!success) {
            return;
        }

# ifdef OFX_IO_USING_OCIO
        // Colorspace parsed from filename overrides the file colorspace,
        // following recommendations from http://opencolorio.org/configurations/spi_pipeline.html
        OCIO::ConstConfigRcPtr ocioConfig = _ocio->getConfig();
        if (ocioConfig) {
            const char* colorSpaceStr = ocioConfig->parseColorSpaceFromString( filename.c_str() );
            if ( colorSpaceStr && _ocio->hasColorspace(colorSpaceStr) ) {
                // we're lucky
                colorspace = colorSpaceStr;
            }
        }

        _ocio->setInputColorspace( colorspace.c_str() );

        // Refreshing the choice menus of ocio is required since the instanceChanged action
        // may not be called recursively during the createInstance action
        _ocio->refreshInputAndOutputState(timeDomain.min);
#endif
        // RGB is always Opaque, Alpha is always PreMultiplied
        if (components == ePixelComponentRGB) {
            filePremult = eImageOpaque;
        } else if (components == ePixelComponentAlpha) {
            filePremult = eImagePreMultiplied;
        }
        if (components != ePixelComponentNone) {
            setOutputComponents(components);
        }
        _filePremult->setValue( (int)filePremult );

        if (components == ePixelComponentRGB) {
            // RGB is always opaque
            _outputPremult->setValue(eImageOpaque);
        } else if (components == ePixelComponentAlpha) {
            // Alpha is always premultiplied
            _outputPremult->setValue(eImagePreMultiplied);
        }

        _guessedParams->setValue(true); // do not try to guess params anymore on this instance
    } // if ( args.reason == eChangeUserEdit && !_guessedParams->getValue() ) {
} // GenericReaderPlugin::changedFilename

void
GenericReaderPlugin::changedParam(const InstanceChangedArgs &args,
                                  const string &paramName)
{
    const double time = args.time;

    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);

        return;
    }

    // please check the reason for each parameter when it makes sense!

    if (paramName == kParamFilename) {
        if (args.reason != eChangeTime) {
            changedFilename(args);
        }
        if ( _sublabel && (args.reason != eChangePluginEdit) ) {
            refreshSubLabel(args.time);
        }
    } else if (paramName == kParamProxy) {
        ///Detect the scale of the proxy.
        string proxyFile, originalFileName;
        double sequenceTime;
        GetSequenceTimeRetEnum getSequenceTimeRet = getSequenceTime(args.time, &sequenceTime);
        switch (getSequenceTimeRet) {
        case eGetSequenceTimeBlack:
        case eGetSequenceTimeError:
            break;

        case eGetSequenceTimeBeforeSequence:
        case eGetSequenceTimeWithinSequence:
        case eGetSequenceTimeAfterSequence: {
            GetFilenameRetCodeEnum getFilenameAtSequenceTimeRet   = getFilenameAtSequenceTime(sequenceTime, false, true, &originalFileName);
            GetFilenameRetCodeEnum getFilenameAtSequenceTimeRetPx = getFilenameAtSequenceTime(sequenceTime, true, true, &proxyFile);

            if ( ( getFilenameAtSequenceTimeRet == eGetFileNameReturnedFullRes) &&
                 ( getFilenameAtSequenceTimeRetPx == eGetFileNameReturnedProxy) &&
                 ( proxyFile != originalFileName) ) {
                assert( !proxyFile.empty() );
                ///show the scale param
                _proxyThreshold->setIsSecretAndDisabled(false);
                _enableCustomScale->setIsSecretAndDisabled(false);

                OfxPointD scale = detectProxyScale(originalFileName, proxyFile, args.time);
                _proxyThreshold->setValue(scale.x, scale.y);
                _originalProxyScale->setValue(scale.x, scale.y);
            } else {
                _proxyThreshold->setIsSecretAndDisabled(true);
                _enableCustomScale->setIsSecretAndDisabled(true);
            }
            break;
        }
        }
    } else if (paramName == kParamCustomProxyScale) {
        bool enabled = _enableCustomScale->getValueAtTime(time);
        _proxyThreshold->setEnabled(enabled);
    } else if (paramName == kParamOriginalFrameRange) {
        int oFirst, oLast;
        string filename;
        _fileParam->getValueAtTime(time, filename);
        if ( isVideoStream(filename) ) {
            return;
        }
        _originalFrameRange->getValue(oFirst, oLast);

        _firstFrame->setRange(oFirst, oLast);
        _firstFrame->setDisplayRange(oFirst, oLast);
        _lastFrame->setRange(oFirst, oLast);
        _lastFrame->setDisplayRange(oFirst, oLast);

        _firstFrame->setValue(oFirst);
        _firstFrame->setDefault(oFirst);
        _lastFrame->setValue(oLast);
        _lastFrame->setDefault(oLast);

        _startingTime->setValue(oFirst);
        _startingTime->setDefault(oFirst);
    } else if ( (paramName == kParamFirstFrame) && (args.reason == eChangeUserEdit) ) {
        int first;
        int oFirst, oLast;
        _originalFrameRange->getValue(oFirst, oLast);
        _firstFrame->getValue(first);
        _lastFrame->setRange(first, oLast);
        _lastFrame->setDisplayRange(first, oLast);

        int offset;
        _timeOffset->getValue(offset);
        _startingTime->setValue(first + offset); // will be called with reason == eChangePluginEdit

        _timeDomainUserSet->setValue(true);
    } else if ( (paramName == kParamLastFrame) && (args.reason == eChangeUserEdit) ) {
        int first;
        int last;
        int oFirst, oLast;
        _originalFrameRange->getValue(oFirst, oLast);
        _firstFrame->getValue(first);
        _lastFrame->getValue(last);
        _firstFrame->setRange(oFirst, last);
        _firstFrame->setDisplayRange(oFirst, last);

        _timeDomainUserSet->setValue(true);
    } else if ( (paramName == kParamFrameMode) && (args.reason == eChangeUserEdit) ) {
        FrameModeEnum frameMode = (FrameModeEnum)_frameMode->getValueAtTime(time);
        switch (frameMode) {
        case eFrameModeStartingTime:     //starting frame
            _startingTime->setIsSecretAndDisabled(false);
            _timeOffset->setIsSecretAndDisabled(true);
            break;
        case eFrameModeTimeOffset:     //time offset
            _startingTime->setIsSecretAndDisabled(true);
            _timeOffset->setIsSecretAndDisabled(false);
            break;
        }
    } else if ( (paramName == kParamStartingTime) && (args.reason == eChangeUserEdit) ) {
        ///recompute the timedomain
        OfxRangeI sequenceTimeDomain;
        getSequenceTimeDomainInternal(sequenceTimeDomain, true);

        //also update the time offset
        int startingTime = _startingTime->getValueAtTime(time);
        int firstFrame = _firstFrame->getValueAtTime(time);

        _timeOffset->setValue(startingTime - firstFrame);
        _timeDomainUserSet->setValue(true);
    } else if ( (paramName == kParamTimeOffset) && (args.reason == eChangeUserEdit) ) {
        //also update the starting frame
        int offset = _timeOffset->getValueAtTime(time);
        int first = _firstFrame->getValueAtTime(time);

        _startingTime->setValue(offset + first);
        _timeDomainUserSet->setValue(true);
    } else if ( (paramName == kParamOutputComponents) && (args.reason == eChangeUserEdit) ) {
        PixelComponentEnum comps = getOutputComponents();
        PreMultiplicationEnum premult = (PreMultiplicationEnum)_outputPremult->getValueAtTime(time);
        if ( (comps == ePixelComponentRGB) && (premult != eImageOpaque) ) {
            // RGB is always opaque
            _outputPremult->setValue(eImageOpaque);
        } else if ( (comps == ePixelComponentAlpha) && (premult != eImagePreMultiplied) ) {
            // Alpha is always premultiplied
            _outputPremult->setValue(eImagePreMultiplied);
        }
    } else if ( (paramName == kParamOutputPremult) && (args.reason == eChangeUserEdit) ) {
        PreMultiplicationEnum premult = (PreMultiplicationEnum)_outputPremult->getValueAtTime(time);
        PixelComponentEnum comps = getOutputComponents();
        // reset to authorized values if necessary
        if ( (comps == ePixelComponentRGB) && (premult != eImageOpaque) ) {
            // RGB is always opaque
            _outputPremult->setValue( (int)eImageOpaque );
        } else if ( (comps == ePixelComponentAlpha) && (premult != eImagePreMultiplied) ) {
            // Alpha is always premultiplied
            _outputPremult->setValue( (int)eImagePreMultiplied );
        }
    } else if (paramName == kParamCustomFps) {
        bool customFps = _customFPS->getValueAtTime(time);
        _fps->setEnabled(customFps);

        if (!customFps) {
            OfxRangeI sequenceTimeDomain;
            if ( getSequenceTimeDomainInternal(sequenceTimeDomain, false) ) {
                OfxRangeI timeDomain;
                ///these are the value held by the "First frame" and "Last frame" param
                sequenceTimeDomain.min = _firstFrame->getValue();
                sequenceTimeDomain.max = _lastFrame->getValue();
                int startingTime = _startingTime->getValue();
                timeDomainFromSequenceTimeDomain(sequenceTimeDomain, startingTime, &timeDomain);
                _startingTime->setValue(timeDomain.min);

                ///We call guessParamsFromFilename with the first frame of the sequence so we're almost sure it will work
                ///unless the user did a mistake. We are also safe to assume that images specs are the same for
                ///all the sequence
                string filename;
                _fileParam->getValueAtTime(timeDomain.min, filename);

                double fps;
                bool gotFps = getFrameRate(filename, &fps);
                if  (gotFps) {
                    _fps->setValue(fps);
                }
            }
        }
    }

#ifdef OFX_IO_USING_OCIO
    _ocio->changedParam(args, paramName);
#endif
} // GenericReaderPlugin::changedParam

PixelComponentEnum
GenericReaderPlugin::getOutputComponents() const
{
    int outputComponents_i = _outputComponents->getValue();

    return _outputComponentsTable[outputComponents_i];
}

void
GenericReaderPlugin::setOutputComponents(PixelComponentEnum comps)
{
    int i;

    for (i = 0; i < 4 && _outputComponentsTable[i] != comps; ++i) {
    }
    if (i >= 4) {
        // not found, set the first supported component
        i = 0;
    }
    assert( i < _outputComponents->getNOptions() );
    _outputComponents->setValue(i);
}

/* Override the clip preferences */
void
GenericReaderPlugin::getClipPreferences(ClipPreferencesSetter &clipPreferences)
{
    // if there is only one frame and before/after behaviour is hold, then
    // the output is not framevarying
    bool frameVarying = true;
    OfxRangeI sequenceTimeDomain;
    bool gotSequenceTimeDomain = getSequenceTimeDomainInternal(sequenceTimeDomain, false);

    if (sequenceTimeDomain.min == sequenceTimeDomain.max) {
        int beforeChoice_i;
        _beforeFirst->getValue(beforeChoice_i);
        BeforeAfterEnum beforeChoice = BeforeAfterEnum(beforeChoice_i);
        int afterChoice_i;
        _afterLast->getValue(afterChoice_i);
        BeforeAfterEnum afterChoice = BeforeAfterEnum(afterChoice_i);
        if ( (beforeChoice == eBeforeAfterHold) && (afterChoice == eBeforeAfterHold) ) {
            frameVarying = false;
        }
    }
    clipPreferences.setOutputFrameVarying(frameVarying); // true for readers and frame-varying generators/effects @see kOfxImageEffectFrameVarying

    PixelComponentEnum outputComponents = getOutputComponents();
    clipPreferences.setClipComponents(*_outputClip, outputComponents);

    int premult_i;
    _outputPremult->getValue(premult_i);
    PreMultiplicationEnum premult = (PreMultiplicationEnum)premult_i;
    switch (outputComponents) {
    case ePixelComponentAlpha:
        if (!_supportsAlpha) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        // alpha is always premultiplied
        premult = eImagePreMultiplied;
        break;

    case ePixelComponentRGBA:
        break;
    default:
        if (!_supportsRGB) {
            throwSuiteStatusException(kOfxStatErrFormat);

            return;
        }
        // RGB is always Opaque
        premult = eImageOpaque;
        break;
    }
    clipPreferences.setOutputPremultiplication(premult);

    // get the pixel aspect ratio from the first frame
    if (gotSequenceTimeDomain) {
        OfxRangeI timeDomain;
        ///these are the value held by the "First frame" and "Last frame" param
        sequenceTimeDomain.min = _firstFrame->getValue();
        sequenceTimeDomain.max = _lastFrame->getValue();
        int startingTime = _startingTime->getValue();
        timeDomainFromSequenceTimeDomain(sequenceTimeDomain, startingTime, &timeDomain);

        string filename;
        GetFilenameRetCodeEnum e = getFilenameAtSequenceTime(timeDomain.min, false, true, &filename);
        if (e == eGetFileNameReturnedFullRes) {
            OfxRectI bounds, format;
            double par = 1.;
            string error;
            int tile_width, tile_height;
            bool success = getFrameBounds(filename, timeDomain.min, &bounds, &format, &par, &error, &tile_width, &tile_height);
            if (success) {
                clipPreferences.setPixelAspectRatio(*_outputClip, par);
                clipPreferences.setOutputFormat(format);
            }

            bool customFPS;
            _customFPS->getValue(customFPS);

            double fps;
            if (customFPS) {
                _fps->getValue(fps);
                clipPreferences.setOutputFrameRate(fps);
            } else {
                success = getFrameRate(filename, &fps);
                if (success) {
                    clipPreferences.setOutputFrameRate(fps);
                }
            }
        }
    }
} // GenericReaderPlugin::getClipPreferences

void
GenericReaderPlugin::purgeCaches()
{
    clearAnyCache();
#ifdef OFX_IO_USING_OCIO
    _ocio->purgeCaches();
#endif
}

bool
GenericReaderPlugin::isIdentity(const IsIdentityArguments &args,
                                Clip * &identityClip,
                                double &identityTime)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);

        return false;
    }

    // must clear persistent message in isIdentity, or render() is not called by Nuke after an error
    clearPersistentMessage();

    if (!gHostIsNatron) {
        // only Natron supports setting the identityClip to the output clip
        return false;
    }

    double sequenceTime;
    GetSequenceTimeRetEnum getSequenceTimeRet = getSequenceTime(args.time, &sequenceTime);
    switch (getSequenceTimeRet) {
    case eGetSequenceTimeBlack:

        return false;

    case eGetSequenceTimeError:
        throwSuiteStatusException(kOfxStatFailed);

        return false;

    case eGetSequenceTimeBeforeSequence:
    case eGetSequenceTimeAfterSequence: {
        ///Transform the sequence time to "real" time
        int timeOffset;
        _timeOffset->getValue(timeOffset);
        identityTime = std::floor(sequenceTime + 0.5) + timeOffset;     // round to the nearest frame
        identityClip = _outputClip;

        return true;
    }
    case eGetSequenceTimeWithinSequence: {
        if (sequenceTime == (int)sequenceTime) {
            return false;
        }
        // fractional time, round to nearest frame
        sequenceTime = std::floor(sequenceTime + 0.5);     // round to the nearest frame
        ///Transform the sequence time to "real" time
        int timeOffset;
        _timeOffset->getValue(timeOffset);
        identityTime = sequenceTime + timeOffset;
        identityClip = _outputClip;

        return true;
    }
    }

    return false;
} // GenericReaderPlugin::isIdentity

OfxPointD
GenericReaderPlugin::detectProxyScale(const string& originalFileName,
                                      const string& proxyFileName,
                                      OfxTime time)
{
    OfxRectI originalBounds, proxyBounds, originalFormat, proxyFormat;
    string error;
    double originalPAR = 1., proxyPAR = 1.;
    int tile_width, tile_height;
    bool success = getFrameBounds(originalFileName, time, &originalBounds, &originalFormat, &originalPAR, &error, &tile_width, &tile_height);

    proxyBounds.x1 = proxyBounds.x2 = proxyBounds.y1 = proxyBounds.y2 = 0.f;
    success = success && getFrameBounds(proxyFileName, time, &proxyBounds, &proxyFormat, &proxyPAR, &error, &tile_width, &tile_height);
    OfxPointD ret;
    if ( !success ||
         (originalBounds.x1 == originalBounds.x2) ||
         (originalBounds.y1 == originalBounds.y2) ||
         (proxyBounds.x1 == proxyBounds.x2) ||
         (proxyBounds.y1 == proxyBounds.y2) ) {
        ret.x = 1.;
        ret.y = 1.;
        setPersistentMessage(Message::eMessageError, "", "Cannot read the proxy file.");

        return ret;
    }
    assert(originalBounds.x2 - originalBounds.x1);
    assert(originalBounds.y2 - originalBounds.y1);
    ret.x = ( (proxyBounds.x2 - proxyBounds.x1)  * proxyPAR ) / ( (originalBounds.x2 - originalBounds.x1) * originalPAR );
    ret.y = (proxyBounds.y2 - proxyBounds.y1) / (double)(originalBounds.y2 - originalBounds.y1);

    return ret;
}

template<typename SRCPIX, int srcMaxValue, int nSrcComp, int nDstComp>
class PixelConverterProcessor
    : public PixelProcessor
{
    const SRCPIX* _srcPixelData;
    int _dstBufferRowBytes;
    int _srcBufferRowBytes;
    OfxRectI _srcBufferBounds;

public:
    // ctor
    PixelConverterProcessor(ImageEffect &instance)
        : PixelProcessor(instance)
        , _srcPixelData(0)
        , _dstBufferRowBytes(0)
        , _srcBufferRowBytes(0)
    {
        assert(srcMaxValue);
        _srcBufferBounds.x1 = _srcBufferBounds.y1 = _srcBufferBounds.x2 = _srcBufferBounds.y2 = 0;
    }

    void setValues(const SRCPIX* srcPixelData,
                   const OfxRectI &srcBufferBounds,
                   int srcBufferRowBytes,
                   float* dstPixelData,
                   int dstBufferRowBytes,
                   const OfxRectI &dstBufferBounds)
    {
        _srcPixelData = srcPixelData;
        _srcBufferBounds = srcBufferBounds;
        _srcBufferRowBytes = srcBufferRowBytes;
        _dstBufferRowBytes = dstBufferRowBytes;

        _dstBounds = dstBufferBounds;
        _dstPixelData = dstPixelData;
    }

    // and do some processing
    void multiThreadProcessImages(OfxRectI procWindow)
    {
        assert(nSrcComp == 1 || nSrcComp == 2 || nSrcComp == 3 || nSrcComp == 4);
        assert(nDstComp == 1 || nDstComp == 2 || nDstComp == 3 || nDstComp == 4);

        for (int dsty = procWindow.y1; dsty < procWindow.y2; ++dsty) {
            if ( _effect.abort() ) {
                break;
            }

            int srcY = _dstBounds.y2 - dsty - 1;
            float* dst_pixels = (float*)( (char*)_dstPixelData + (size_t)_dstBufferRowBytes * (dsty - _dstBounds.y1) )
                                + (_dstBounds.x1 * nDstComp);
            const SRCPIX* src_pixels = (const SRCPIX*)( (const char*)_srcPixelData + (size_t)_srcBufferRowBytes * (srcY - _srcBufferBounds.y1) )
                                       + (_srcBufferBounds.x1 * nSrcComp);


            assert(dst_pixels && src_pixels);

            for (int x = procWindow.x1; x < procWindow.x2; ++x) {
                int srcCol = x * nSrcComp;
                int dstCol = x * nDstComp;

                switch (nSrcComp) {
                case 1:
                    // alpha
                    switch (nDstComp) {
                    case 1:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        break;
                    case 2:
                        dst_pixels[dstCol + 0] = 0.;
                        dst_pixels[dstCol + 1] = 0.;
                        break;

                    case 3:
                        dst_pixels[dstCol + 0] = 0.;
                        dst_pixels[dstCol + 1] = 0.;
                        dst_pixels[dstCol + 2] = 0.;
                        break;

                    case 4:
                        dst_pixels[dstCol + 0] = 0.;
                        dst_pixels[dstCol + 1] = 0.;
                        dst_pixels[dstCol + 2] = 0.;
                        dst_pixels[dstCol + 3] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        break;
                    default:
                        assert(false);

                        return;
                    }
                    break;
                case 2:
                    // XY
                    switch (nDstComp) {
                    case 1:
                        dst_pixels[dstCol + 0] = 0.;
                        break;
                    case 2:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        break;

                    case 3:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        dst_pixels[dstCol + 2] = 0;
                        break;

                    case 4:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        dst_pixels[dstCol + 2] = 0.;
                        dst_pixels[dstCol + 3] = 1.f;
                        break;
                    default:
                        assert(false);

                        return;
                    }

                    break;
                case 3:
                    // RGB
                    switch (nDstComp) {
                    case 1: {
                        dst_pixels[dstCol + 0] = 0.;
                        break;
                    }
                    case 2:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        break;

                    case 3:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        dst_pixels[dstCol + 2] = src_pixels[srcCol + 2] / (float)srcMaxValue;
                        break;

                    case 4:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        dst_pixels[dstCol + 2] = src_pixels[srcCol + 2] / (float)srcMaxValue;
                        dst_pixels[dstCol + 3] = 1.f;
                        break;
                    default:
                        assert(false);

                        return;
                    }

                    break;

                case 4:
                    switch (nDstComp) {
                    case 1:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 3] / (float)srcMaxValue;
                        break;
                    case 2:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        break;

                    case 3:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        dst_pixels[dstCol + 2] = src_pixels[srcCol + 2] / (float)srcMaxValue;
                        break;

                    case 4:
                        dst_pixels[dstCol + 0] = src_pixels[srcCol + 0] / (float)srcMaxValue;
                        dst_pixels[dstCol + 1] = src_pixels[srcCol + 1] / (float)srcMaxValue;
                        dst_pixels[dstCol + 2] = src_pixels[srcCol + 2] / (float)srcMaxValue;
                        dst_pixels[dstCol + 3] = src_pixels[srcCol + 3] / (float)srcMaxValue;
                        break;
                    default:
                        assert(false);

                        return;
                    }

                    break;
                default:
                    assert(false);

                    return;
                } // switch
            }
        }
    } // multiThreadProcessImages
};

template<typename SRCPIX, int srcMaxValue, int nSrcComp, int nDstComp>
void
convertForDstNComps(ImageEffect* effect,
                    const SRCPIX* srcPixelData,
                    const OfxRectI& renderWindow,
                    const OfxRectI& srcBounds,
                    int srcRowBytes,
                    float *dstPixelData,
                    const OfxRectI& dstBounds,
                    int dstRowBytes)
{
    PixelConverterProcessor<SRCPIX, srcMaxValue, nSrcComp, nDstComp> p(*effect);
    p.setValues(srcPixelData, srcBounds, srcRowBytes, dstPixelData,  dstRowBytes,  dstBounds);
    p.setRenderWindow(renderWindow);
    p.process();
}

template<typename SRCPIX, int srcMaxValue, int nSrcComp>
void
convertForSrcNComps(ImageEffect* effect,
                    const SRCPIX* srcPixelData,
                    const OfxRectI& renderWindow,
                    const OfxRectI& srcBounds,
                    int srcRowBytes,
                    float *dstPixelData,
                    const OfxRectI& dstBounds,
                    PixelComponentEnum dstPixelComponents,
                    int dstRowBytes)
{
    switch (dstPixelComponents) {
    case ePixelComponentAlpha: {
        convertForDstNComps<SRCPIX, srcMaxValue, nSrcComp, 1>(effect, srcPixelData, renderWindow, srcBounds, srcRowBytes, dstPixelData, dstBounds, dstRowBytes);
        break;
    }
    case ePixelComponentXY: {
        convertForDstNComps<SRCPIX, srcMaxValue, nSrcComp, 2>(effect, srcPixelData, renderWindow, srcBounds, srcRowBytes, dstPixelData, dstBounds, dstRowBytes);
        break;
    }
    case ePixelComponentRGB: {
        convertForDstNComps<SRCPIX, srcMaxValue, nSrcComp, 3>(effect, srcPixelData, renderWindow, srcBounds, srcRowBytes, dstPixelData, dstBounds, dstRowBytes);
        break;
    }
    case ePixelComponentRGBA: {
        convertForDstNComps<SRCPIX, srcMaxValue, nSrcComp, 4>(effect, srcPixelData, renderWindow, srcBounds, srcRowBytes, dstPixelData, dstBounds, dstRowBytes);
        break;
    }
    default:
        assert(false);
        break;
    }
}

template<typename SRCPIX, int srcMaxValue>
void
convertForDepth(ImageEffect* effect,
                const SRCPIX* srcPixelData,
                const OfxRectI& renderWindow,
                const OfxRectI& srcBounds,
                PixelComponentEnum srcPixelComponents,
                int srcRowBytes,
                float *dstPixelData,
                const OfxRectI& dstBounds,
                PixelComponentEnum dstPixelComponents,
                int dstRowBytes)
{
    switch (srcPixelComponents) {
    case ePixelComponentAlpha:
        convertForSrcNComps<SRCPIX, srcMaxValue, 1>(effect, srcPixelData, renderWindow, srcBounds, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstRowBytes);
        break;
    case ePixelComponentXY:
        convertForSrcNComps<SRCPIX, srcMaxValue, 2>(effect, srcPixelData, renderWindow, srcBounds, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstRowBytes);
        break;
    case ePixelComponentRGB:
        convertForSrcNComps<SRCPIX, srcMaxValue, 3>(effect, srcPixelData, renderWindow, srcBounds, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstRowBytes);
        break;
    case ePixelComponentRGBA:
        convertForSrcNComps<SRCPIX, srcMaxValue, 4>(effect, srcPixelData, renderWindow, srcBounds, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstRowBytes);
        break;
    default:
        assert(false);
        break;
    }
}

void
GenericReaderPlugin::convertDepthAndComponents(const void* srcPixelData,
                                               const OfxRectI& renderWindow,
                                               const OfxRectI& srcBounds,
                                               PixelComponentEnum srcPixelComponents,
                                               BitDepthEnum srcBitDepth,
                                               int srcRowBytes,
                                               float *dstPixelData,
                                               const OfxRectI& dstBounds,
                                               PixelComponentEnum dstPixelComponents,
                                               int dstRowBytes)
{
    switch (srcBitDepth) {
    case eBitDepthFloat:
        convertForDepth<float, 1>(this, (const float*)srcPixelData, renderWindow, srcBounds, srcPixelComponents, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstRowBytes);
        break;
    case eBitDepthUShort:
        convertForDepth<unsigned short, 65535>(this, (const unsigned short*)srcPixelData, renderWindow, srcBounds, srcPixelComponents, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstRowBytes);
        break;
    case eBitDepthUByte:
        convertForDepth<unsigned char, 255>(this, (const unsigned char*)srcPixelData, renderWindow, srcBounds, srcPixelComponents, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstRowBytes);
        break;
    default:
        assert(false);
        break;
    }
}

void
GenericReaderDescribe(ImageEffectDescriptor &desc,
                      const std::vector<string>& extensions,
                      int evaluation,
                      bool supportsTiles,
                      bool multiPlanar)
{
    desc.setPluginGrouping(kPluginGrouping);

#ifdef OFX_EXTENSIONS_TUTTLE
    desc.addSupportedContext(eContextReader);
#endif
    desc.addSupportedContext(eContextGenerator);
    desc.addSupportedContext(eContextGeneral);

    // add supported pixel depths
    //desc.addSupportedBitDepth( eBitDepthUByte );
    //desc.addSupportedBitDepth( eBitDepthUShort );
    desc.addSupportedBitDepth(eBitDepthFloat);

    // set a few flags
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);

    desc.setSupportsMultiResolution(kSupportsMultiResolution);

    desc.setSupportsTiles(supportsTiles);
    desc.setTemporalClipAccess(false); // say we will not be doing random time access on clips
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(true); // plugin may setPixelAspectRatio on output clip
    desc.setRenderThreadSafety(eRenderFullySafe);

#ifdef OFX_EXTENSIONS_NUKE
    if (getImageEffectHostDescription()
        && getImageEffectHostDescription()->isMultiPlanar) {
        desc.setIsMultiPlanar(multiPlanar);
        if (multiPlanar) {
            //We let all un-rendered planes pass-through so that they can be retrieved below by a shuffle node
            desc.setPassThroughForNotProcessedPlanes(ePassThroughLevelPassThroughNonRenderedPlanes);
        }
        desc.setIsViewAware(true);
        desc.setIsViewInvariant(eViewInvarianceAllViewsVariant);
    }
#endif
#ifdef OFX_EXTENSIONS_TUTTLE
    desc.addSupportedExtensions(extensions);
    desc.setPluginEvaluation(evaluation);
#endif
#ifdef OFX_EXTENSIONS_NATRON
    desc.setChannelSelector(ePixelComponentNone);
#endif
}

PageParamDescriptor *
GenericReaderDescribeInContextBegin(ImageEffectDescriptor &desc,
                                    ContextEnum /*context*/,
                                    bool /*isVideoStreamPlugin*/,
                                    bool supportsRGBA,
                                    bool supportsRGB,
                                    bool supportsXY,
                                    bool supportsAlpha,
                                    bool supportsTiles,
                                    bool addSeparatorAfterLastParameter)
{
    gHostIsNatron = (getImageEffectHostDescription()->isNatron);

    for (ImageEffectHostDescription::PixelComponentArray::const_iterator it = getImageEffectHostDescription()->_supportedComponents.begin();
         it != getImageEffectHostDescription()->_supportedComponents.end();
         ++it) {
        switch (*it) {
        case ePixelComponentRGBA:
            gHostSupportsRGBA  = true;
            break;
        case ePixelComponentRGB:
            gHostSupportsRGB = true;
            break;
        case ePixelComponentXY:
            gHostSupportsXY = true;
            break;
        case ePixelComponentAlpha:
            gHostSupportsAlpha = true;
            break;
        default:
            // other components are not supported by this plugin
            break;
        }
    }

    // make some pages and to things in
    PageParamDescriptor *page = desc.definePageParam("Controls");

    // create the optional source clip
    ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    if (supportsRGBA) {
        srcClip->addSupportedComponent(ePixelComponentRGBA);
    }
    if (supportsRGB) {
        srcClip->addSupportedComponent(ePixelComponentRGB);
    }
    if (supportsXY) {
        srcClip->addSupportedComponent(ePixelComponentXY);
    }
    if (supportsAlpha) {
        srcClip->addSupportedComponent(ePixelComponentAlpha);
    }
    srcClip->setSupportsTiles(supportsTiles);
    srcClip->setOptional(true);

    // create the mandated output clip
    ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    if (supportsRGBA) {
        dstClip->addSupportedComponent(ePixelComponentRGBA);
    }
    if (supportsRGB) {
        dstClip->addSupportedComponent(ePixelComponentRGB);
    }
    if (supportsXY) {
        dstClip->addSupportedComponent(ePixelComponentXY);
    }
    if (supportsAlpha) {
        dstClip->addSupportedComponent(ePixelComponentAlpha);
    }
    dstClip->setSupportsTiles(supportsTiles);


    //////////Input file
    {
        StringParamDescriptor* param = desc.defineStringParam(kParamFilename);
        param->setLabel(kParamFilenameLabel);
        param->setStringType(eStringTypeFilePath);
        param->setFilePathExists(true);
        param->setHint(kParamFilenameHint);
        // in the Reader context, the script name must be kOfxImageEffectFileParamName, @see kOfxImageEffectContextReader
        param->setScriptName(kParamFilename);
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    //////////First-frame
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamFirstFrame);
        param->setLabel(kParamFirstFrameLabel);
        param->setHint(kParamFirstFrameHint);
        param->setDefault(0);
        param->setAnimates(false);
        param->setLayoutHint(eLayoutHintNoNewLine, 1);
        if (page) {
            page->addChild(*param);
        }
    }

    ///////////Before first
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamBefore);
        param->setLabel(kParamBeforeLabel);
        param->setHint(kParamBeforeHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterHold);
        param->appendOption(kReaderOptionHold,   kReaderOptionHoldHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterLoop);
        param->appendOption(kReaderOptionLoop,   kReaderOptionLoopHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterBounce);
        param->appendOption(kReaderOptionBounce, kReaderOptionBounceHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterBlack);
        param->appendOption(kReaderOptionBlack,  kReaderOptionBlackHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterError);
        param->appendOption(kReaderOptionError,  kReaderOptionErrorHint);
        param->setAnimates(true);
        param->setDefault(GenericReaderPlugin::eBeforeAfterHold);
        if (page) {
            page->addChild(*param);
        }
    }

    //////////Last-frame
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamLastFrame);
        param->setLabel(kParamLastFrameLabel);
        param->setHint(kParamLastFrameHint);
        param->setDefault(0);
        param->setAnimates(false);
        param->setLayoutHint(eLayoutHintNoNewLine, 1);
        if (page) {
            page->addChild(*param);
        }
    }

    ///////////After first
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamAfter);
        param->setLabel(kParamAfterLabel);
        param->setHint(kParamAfterHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterHold);
        param->appendOption(kReaderOptionHold,   kReaderOptionHoldHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterLoop);
        param->appendOption(kReaderOptionLoop,   kReaderOptionLoopHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterBounce);
        param->appendOption(kReaderOptionBounce, kReaderOptionBounceHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterBlack);
        param->appendOption(kReaderOptionBlack,  kReaderOptionBlackHint);
        assert(param->getNOptions() == GenericReaderPlugin::eBeforeAfterError);
        param->appendOption(kReaderOptionError,  kReaderOptionErrorHint);
        param->setAnimates(true);
        param->setDefault(GenericReaderPlugin::eBeforeAfterHold);
        if (page) {
            page->addChild(*param);
        }
    }

    ///////////Missing frame choice
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamOnMissingFrame);
        param->setLabel(kParamOnMissingFrameLabel);
        param->setHint(kParamOnMissingFrameHint);
        assert(param->getNOptions() == eMissingPrevious);
        param->appendOption(kReaderOptionPrevious,  kReaderOptionPreviousHint);
        assert(param->getNOptions() == eMissingNext);
        param->appendOption(kReaderOptionNext,  kReaderOptionNextHint);
        assert(param->getNOptions() == eMissingNearest);
        param->appendOption(kReaderOptionNearest,  kReaderOptionNearestHint);
        assert(param->getNOptions() == eMissingError);
        param->appendOption(kReaderOptionError,  kReaderOptionErrorHint);
        assert(param->getNOptions() == eMissingBlack);
        param->appendOption(kReaderOptionBlack,  kReaderOptionBlackHint);
        param->setAnimates(true);
        param->setDefault(eMissingError);
        if (page) {
            page->addChild(*param);
        }
    }

    ///////////Frame-mode
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamFrameMode);
        param->setLabel(kParamFrameModeLabel);
        assert(param->getNOptions() == eFrameModeStartingTime);
        param->appendOption(kParamFrameModeOptionStartingTime);
        assert(param->getNOptions() == eFrameModeTimeOffset);
        param->appendOption(kParamFrameModeOptionTimeOffset);
        param->setAnimates(false);
        param->setDefault(eFrameModeStartingTime);
        param->setLayoutHint(eLayoutHintNoNewLine, 1);
        if (page) {
            page->addChild(*param);
        }
    }

    ///////////Starting frame
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamStartingTime);
        param->setLabel(kParamStartingTimeLabel);
        param->setHint(kParamStartingTimeHint);
        param->setDefault(0);
        param->setAnimates(false);
        param->setLayoutHint(eLayoutHintNoNewLine, 1);
        if (page) {
            page->addChild(*param);
        }
    }

    ///////////Time offset
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamTimeOffset);
        param->setLabel(kParamTimeOffsetLabel);
        param->setHint(kParamTimeOffsetHint);
        param->setDefault(0);
        param->setAnimates(false);
        //param->setIsSecretAndDisabled(true); // done in the plugin constructor
        if (page) {
            page->addChild(*param);
        }
    }

    /////////// Secret param set to true if the time domain was edited by the user
    {
        BooleanParamDescriptor* param = desc.defineBooleanParam(kParamTimeDomainUserEdited);
        param->setLabel(kParamTimeDomainUserEdited);
        param->setIsSecretAndDisabled(true); // always secret
        param->setDefault(false);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }

    ///////////Original frame range
    {
        Int2DParamDescriptor* param = desc.defineInt2DParam(kParamOriginalFrameRange);
        param->setLabel(kParamOriginalFrameRangeLabel);
        param->setDefault(INT_MIN, INT_MAX);
        param->setAnimates(true);
        param->setIsSecretAndDisabled(true); // always secret
        param->setIsPersistent(false);
        if (page) {
            page->addChild(*param);
        }
    }

    //////////Input proxy file
    {
        StringParamDescriptor* param = desc.defineStringParam(kParamProxy);
        param->setLabel(kParamProxyLabel);
        param->setStringType(eStringTypeFilePath);
        param->setFilePathExists(true);
        param->setHint(kParamProxyHint);
        // in the Reader context, the script name must be kOfxImageEffectFileParamName, @see kOfxImageEffectContextReader
        param->setScriptName(kParamProxy);
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    ////Proxy original scale
    {
        Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamOriginalProxyScale);
        param->setLabel(kParamOriginalProxyScaleLabel);
        param->setDefault(1., 1.);
        param->setRange(0., 0., 1., 1.);
        param->setDisplayRange(0., 0., 1., 1.);
        param->setIsSecretAndDisabled(true); // always secret
        param->setHint(kParamOriginalProxyScaleHint);
        // param->setLayoutHint(eLayoutHintNoNewLine, 1);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    ////Proxy  scale threshold
    {
        Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamProxyThreshold);
        param->setLabel(kParamProxyThresholdLabel);
        param->setDefault(1., 1.);
        param->setRange(0.01, 0.01, 1., 1.);
        param->setDisplayRange(0.01, 0.01, 1., 1.);
        //param->setIsSecretAndDisabled(true); // done in the plugin constructor
        param->setHint(kParamOriginalProxyScaleHint);
        param->setLayoutHint(eLayoutHintNoNewLine, 1);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    ///Enable custom proxy scale
    {
        BooleanParamDescriptor* param = desc.defineBooleanParam(kParamCustomProxyScale);
        param->setLabel(kParamCustomProxyScaleLabel);
        //param->setIsSecretAndDisabled(true); // done in the plugin constructor
        param->setDefault(false);
        param->setHint(kParamCustomProxyScaleHint);
        param->setAnimates(false);
        param->setEvaluateOnChange(false);
        if (page) {
            page->addChild(*param);
        }
    }

    //// File premult
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamFilePremult);
        param->setLabel(kParamFilePremultLabel);
        param->setHint(kParamFilePremultHint);
        assert(param->getNOptions() == eImageOpaque);
        param->appendOption(premultString(eImageOpaque), kParamFilePremultOptionOpaqueHint);
        if (gHostSupportsRGBA && supportsRGBA) {
            assert(param->getNOptions() == eImagePreMultiplied);
            param->appendOption(premultString(eImagePreMultiplied), kParamFilePremultOptionPreMultipliedHint);
            assert(param->getNOptions() == eImageUnPreMultiplied);
            param->appendOption(premultString(eImageUnPreMultiplied), kParamFilePremultOptionUnPreMultipliedHint);
            param->setDefault(eImagePreMultiplied); // images should be premultiplied in a compositing context
        }
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    //// Output premult
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamOutputPremult);
        param->setLabel(kParamOutputPremultLabel);
        param->setHint(kParamOutputPremultHint);
        assert(param->getNOptions() == eImageOpaque);
        param->appendOption(premultString(eImageOpaque), kParamFilePremultOptionOpaqueHint);
        if (gHostSupportsRGBA && supportsRGBA) {
            assert(param->getNOptions() == eImagePreMultiplied);
            param->appendOption(premultString(eImagePreMultiplied), kParamFilePremultOptionPreMultipliedHint);
            assert(param->getNOptions() == eImageUnPreMultiplied);
            param->appendOption(premultString(eImageUnPreMultiplied), kParamFilePremultOptionUnPreMultipliedHint);
            param->setDefault(eImagePreMultiplied); // images should be premultiplied in a compositing context
        }
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    //// Output components
    {
        ChoiceParamDescriptor *param = desc.defineChoiceParam(kParamOutputComponents);
        param->setLabel(kParamOutputComponentsLabel);
        param->setHint(kParamOutputComponentsHint);

        // must be in sync with the building of _outputComponentsTable in the GenericReaderPlugin constructor
        if (gHostSupportsRGBA && supportsRGBA) {
            param->appendOption(kParamOutputComponentsOptionRGBA);
        }
        if (gHostSupportsRGB && supportsRGB) {
            param->appendOption(kParamOutputComponentsOptionRGB);
        }
        if (gHostSupportsXY && supportsXY) {
            param->appendOption(kParamOutputComponentsOptionXY);
        }

        if (gHostSupportsAlpha && supportsAlpha) {
            param->appendOption(kParamOutputComponentsOptionAlpha);
        }

        param->setDefault(0); // default to the first one available, i.e. the most chromatic
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    ///Frame rate
    {
        DoubleParamDescriptor* param = desc.defineDoubleParam(kParamFrameRate);
        param->setLabel(kParamFrameRateLabel);
        param->setHint(kParamFrameRateHint);
        param->setEvaluateOnChange(false);
        param->setLayoutHint(eLayoutHintNoNewLine, 1);
        //param->setEnabled(false); // done in the restoreStateFromParams()
        param->setDefault(24.);
        param->setRange(0., DBL_MAX);
        param->setDisplayRange(0., 300.);
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (page) {
            page->addChild(*param);
        }
    }

    ///Custom FPS
    {
        BooleanParamDescriptor* param  = desc.defineBooleanParam(kParamCustomFps);
        param->setLabel(kParamCustomFpsLabel);
        param->setHint(kParamCustomFpsHint);
        param->setEvaluateOnChange(false);
        param->setAnimates(false);
        desc.addClipPreferencesSlaveParam(*param);
        if (addSeparatorAfterLastParameter) {
            //param->setLayoutHint(eLayoutHintDivider);
        }
        if (page) {
            page->addChild(*param);
        }
    }

    // sublabel
    if (gHostIsNatron) {
        StringParamDescriptor* param = desc.defineStringParam(kNatronOfxParamStringSublabelName);
        param->setIsSecretAndDisabled(true); // always secret
        param->setIsPersistent(true);
        param->setEvaluateOnChange(false);
        //param->setDefault();
        if (page) {
            page->addChild(*param);
        }
    }

    {
        BooleanParamDescriptor* param  = desc.defineBooleanParam(kParamGuessedParams);
        param->setEvaluateOnChange(false);
        param->setAnimates(false);
        param->setIsSecretAndDisabled(true);
        param->setDefault(false);
        if (page) {
            page->addChild(*param);
        }
    }

    return page;
} // GenericReaderDescribeInContextBegin

void
GenericReaderDescribeInContextEnd(ImageEffectDescriptor &desc,
                                  ContextEnum context,
                                  PageParamDescriptor* page,
                                  const char* inputSpaceNameDefault,
                                  const char* outputSpaceNameDefault)
{
#ifdef OFX_IO_USING_OCIO
    // insert OCIO parameters
    GenericOCIO::describeInContextInput(desc, context, page, inputSpaceNameDefault, kParamInputSpaceLabel);
    {
        BooleanParamDescriptor* param  = desc.defineBooleanParam(kParamInputSpaceSet);
        param->setEvaluateOnChange(false);
        param->setAnimates(false);
        param->setIsSecretAndDisabled(true);
        param->setDefault(false);
        if (page) {
            page->addChild(*param);
        }
    }
    GenericOCIO::describeInContextOutput(desc, context, page, outputSpaceNameDefault);
    GenericOCIO::describeInContextContext(desc, context, page);
    {
        PushButtonParamDescriptor* param = desc.definePushButtonParam(kOCIOHelpButton);
        param->setLabel(kOCIOHelpButtonLabel);
        param->setHint(kOCIOHelpButtonHint);
        if (page) {
            page->addChild(*param);
        }
    }
#endif
}

NAMESPACE_OFX_IO_EXIT
NAMESPACE_OFX_EXIT

