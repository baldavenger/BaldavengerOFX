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

#ifndef Io_GenericReader_h
#define Io_GenericReader_h

#include <memory>
#include <ofxsImageEffect.h>
#include <ofxsMacros.h>
#include "IOUtility.h"

namespace SequenceParsing {
class SequenceFromFiles;
}

NAMESPACE_OFX_ENTER
NAMESPACE_OFX_IO_ENTER

#ifdef OFX_IO_USING_OCIO
class GenericOCIO;
#endif

/**
 * @brief A generic reader plugin, derive this to create a new reader for a specific file format.
 * This class propose to handle the common stuff among readers:
 * - common params
 * - a tiny cache to speed-up the successive getRegionOfDefinition() calls
 * - a way to inform the host about the colour-space of the data.
 **/
class GenericReaderPlugin
    : public OFX::ImageEffect
{
public:

    GenericReaderPlugin(OfxImageEffectHandle handle,
                        const std::vector<std::string>& extensions,
                        bool supportsTiles,
                        bool supportsRGBA,
                        bool supportsRGB,
                        bool supportsXY,
                        bool supportsAlpha,
                        bool isMultiPlanar);

    virtual ~GenericReaderPlugin();

    /**
     * @brief Don't override this function, the GenericReaderPlugin class already does the rendering. The "decoding" of the frame
     * must be done by the pure virtual function decode(...) instead.
     **/
    virtual void render(const OFX::RenderArguments &args) OVERRIDE FINAL;

    /**
     * @brief Don't override this. Basically this function will call getTimeDomainForVideoStream(...),
     * which your reader should implement to read from a video-stream the time range.
     * If the file is not a video stream, the function getTimeDomainForVideoStream() should return false, indicating that
     * we're reading a sequence of images and that the host should get the time domain for us.
     **/
    virtual bool getTimeDomain(OfxRangeD &range) OVERRIDE FINAL;

    /**
     * @brief Don't override this. If the pure virtual function areHeaderAndDataTied() returns true, this
     * function will call decode() to read the region of definition of the image and cache away the decoded image
     * into the _dstImg member.
     * If areHeaderAndDataTied() returns false instead, this function will call the virtual function
     * getFrameBounds() which should read the header of the image to only extract the bounds and PAR of the image.
     **/
    virtual bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments &args, OfxRectD &rod) OVERRIDE FINAL;


    /**
     * @brief You can override this to take actions in response to a param change.
     * Make sure you call the base-class version of this function at the end: i.e:
     *
     * void MyReader::changedParam(const OFX::InstanceChangedArgs &args, const std::string &paramName) {
     *      if (.....) {
     *
     *      } else if(.....) {
     *
     *      } else {
     *          GenericReaderPlugin::changedParam(args,paramName);
     *      }
     * }
     **/
    virtual void changedParam(const OFX::InstanceChangedArgs &args, const std::string &paramName) OVERRIDE;

    /* override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments &args, OFX::Clip * &identityClip, double &identityTime) OVERRIDE;

    /**
     * @brief Set the output components and premultiplication state for the input image automatically.
     * This is filled from directly the info returned by guessParamsFromFilename
     **/
    virtual void getClipPreferences(OFX::ClipPreferencesSetter &clipPreferences) OVERRIDE;

    /**
     * @brief Overriden to clear any OCIO cache.
     * This function calls clearAnyCache() if you have any cache to clear.
     **/
    virtual void purgeCaches(void) OVERRIDE;

    /**
     * @brief Restore any state from the parameters set
     * Called from createInstance() and changedParam() (via changedFilename()), must restore the
     * state of the Reader, such as Choice param options, data members and non-persistent param values.
     * We don't do this in the ctor of the plug-in since we can't call virtuals yet.
     * Any derived implementation must call GenericReaderPlugin::restoreStateFromParams() first
     **/
    virtual void restoreStateFromParams();


    bool isMultiPlanar() const
    {
        return _isMultiPlanar;
    }

    enum BeforeAfterEnum
    {
        eBeforeAfterHold,
        eBeforeAfterLoop,
        eBeforeAfterBounce,
        eBeforeAfterBlack,
        eBeforeAfterError,
    };

protected:
    /**
     * @brief Called from changedParam() when kParamFilename is changed for any reason other than eChangeTime.
     * Calls restoreStateFromParams() to update any non-persistent params that may depend on the filename.
     * If reason is eChangeUserEdit and the params where never guessed (see _guessedParams) also sets these from the file contents.
     * Any derived implementation must call GenericReaderPlugin::changedFilename() first
     **/
    virtual void changedFilename(const OFX::InstanceChangedArgs &args);


    OFX::ChoiceParam* _missingFrameParam; //< what to do on missing frame

    OfxStatus getFilenameAtTime(double t, std::string *filename) const;

    int getStartingTime() const;

    // get the value of kParamOutputComponents as a OFX::PixelComponentEnum
    OFX::PixelComponentEnum getOutputComponents() const;

    // sets the value of kParamOutputComponents
    void setOutputComponents(OFX::PixelComponentEnum comps);

    struct PlaneToRender
    {
        float* pixelData;
        int rowBytes;
        int numChans;
        OFX::PixelComponentEnum comps;
        std::string rawComps;
    };

    void convertDepthAndComponents(const void* srcPixelData,
                                   const OfxRectI& renderWindow,
                                   const OfxRectI& srcBounds,
                                   OFX::PixelComponentEnum srcPixelComponents,
                                   OFX::BitDepthEnum srcBitDepth,
                                   int srcRowBytes,
                                   float *dstPixelData,
                                   const OfxRectI& dstBounds,
                                   OFX::PixelComponentEnum dstPixelComponents,
                                   int dstRowBytes);

private:
    /**
     * @brief Called when the input image/video file changed.
     *
     * returns true if file exists and parameters successfully guessed, false in case of error.
     *
     * This function is only called once: when the filename is first set.
     *
     * Besides returning colorspace, premult, components, and componentcount, if it returns true
     * this function may also set extra format-specific parameters using OFX::Param::setValue.
     * The parameters must not be animated, since their value must remain the same for a whole sequence.
     *
     * You shouldn't do any strong processing as this is called on the main thread and
     * the getRegionOfDefinition() and  decode() should open the file in a separate thread.
     *
     * The colorspace may be set if available, else a default colorspace is used.
     *
     * You must also return the premultiplication state and pixel components of the image.
     * When reading an image sequence, this is called only for the first image when the user actually selects the new sequence.
     **/
    virtual bool guessParamsFromFilename(const std::string& newFile,
                                         std::string *colorspace,
                                         OFX::PreMultiplicationEnum *filePremult,
                                         OFX::PixelComponentEnum *components,
                                         int *componentCount) = 0;

    /**
     * @brief Override to clear any cache you may have.
     **/
    virtual void clearAnyCache() {}


    /**
     * @brief Overload this function to extract the bound of the pixel data
     * in pixel coordinates and the pixel aspect ratio out of the header
     * of the image targeted by the filename.
     **/
    virtual bool getFrameBounds(const std::string& filename,
                                OfxTime time,
                                OfxRectI *bounds,
                                OfxRectI *format,
                                double *par,
                                std::string *error,
                                int* tile_width,
                                int* tile_height) = 0;

    /*
     * In case of plug-ins that can read tiled files, determines whether the image is oriented bottom up or top down.
     * This is so that the rounding to the tile size is applied correctly. Currently this is only useful for the OIIO plug-in.
     */
    virtual bool isTileOrientationTopDown() const { return true; }

    virtual bool getFrameRate(const std::string& /*filename*/,
                              double* /*fps*/) const { return false; }

    /**
     * @brief Override this function to actually decode the image contained in the file pointed to by filename.
     * If the file is a video-stream then you should decode the frame at the time given in parameters.
     * You must write the decoded image into dstImg. This function should convert the read pixels into the
     * bitdepth of the dstImg. You can inform the host of the bitdepth you support in the describe() function.
     * You can always skip the color-space conversion, but for all linear hosts it would produce either
     * false colors or sub-par performances in the case the end-user has to append a color-space conversion
     * effect her/himself.
     **/
    virtual void decode(const std::string& filename, OfxTime time, int view, bool isPlayback, const OfxRectI& renderWindow, float *pixelData, const OfxRectI& bounds, OFX::PixelComponentEnum pixelComponents, int pixelComponentCount, int rowBytes);
    virtual void decodePlane(const std::string& filename, OfxTime time, int view, bool isPlayback, const OfxRectI& renderWindow, float *pixelData, const OfxRectI& bounds,
                             OFX::PixelComponentEnum pixelComponents, int pixelComponentCount, const std::string& rawComponents, int rowBytes);


    /**
     * @brief Override to indicate the time domain. Return false if you know that the
     * file isn't a video-stream, true when you can find-out the frame range.
     **/
    virtual bool getSequenceTimeDomain(const std::string& /*filename*/,
                                       OfxRangeI & /*range*/) { return false; }

    /**
     * @brief Called internally by getTimeDomain(...)
     **/
    bool getSequenceTimeDomainInternal(OfxRangeI& range, bool canSetOriginalFrameRange);

    /**
     * @brief Used internally by the GenericReader.
     **/
    void timeDomainFromSequenceTimeDomain(const OfxRangeI& sequenceTimeDomain, int startingTime, OfxRangeI* timeDomain);

    /**
     * @brief Should return true if the file indicated by filename is a video-stream and not
     * a single image file.
     **/
    virtual bool isVideoStream(const std::string& filename) = 0;
    enum GetSequenceTimeRetEnum
    {
        eGetSequenceTimeWithinSequence = 0,
        eGetSequenceTimeBeforeSequence,
        eGetSequenceTimeAfterSequence,
        eGetSequenceTimeBlack,
        eGetSequenceTimeError,
    };


    /**
     * @brief compute the sequence/file time from time
     */
    GetSequenceTimeRetEnum getSequenceTime(double t, double *sequenceTime) const WARN_UNUSED_RETURN;
    GetSequenceTimeRetEnum getSequenceTimeHold(double t, double *sequenceTime) const WARN_UNUSED_RETURN;
    GetSequenceTimeRetEnum getSequenceTimeBefore(const OfxRangeI& sequenceTimeDomain, double t, BeforeAfterEnum beforeChoice, double *sequenceTime) const;
    GetSequenceTimeRetEnum getSequenceTimeAfter(const OfxRangeI& sequenceTimeDomain, double t, BeforeAfterEnum afterChoice, double *sequenceTime) const;

    enum GetFilenameRetCodeEnum
    {
        eGetFileNameFailed = 0,
        eGetFileNameReturnedFullRes,
        eGetFileNameReturnedProxy,
        eGetFileNameBlack,
    };

    /**
     * @brief Returns the filename of the image at the sequence time t.
     **/
    GetFilenameRetCodeEnum getFilenameAtSequenceTime(double t,
                                                     bool proxyFiles,
                                                     bool checkForExistingFile,
                                                     std::string *filename) const WARN_UNUSED_RETURN;


    void copyPixelData(const OfxRectI &renderWindow,
                       const void *srcPixelData,
                       const OfxRectI& srcBounds,
                       OFX::PixelComponentEnum srcPixelComponents,
                       int srcPixelComponentCount,
                       OFX::BitDepthEnum srcPixelDepth,
                       int srcRowBytes,
                       void *dstPixelData,
                       const OfxRectI& dstBounds,
                       OFX::PixelComponentEnum dstPixelComponents,
                       int dstPixelComponentCount,
                       OFX::BitDepthEnum dstBitDepth,
                       int dstRowBytes);

    void scalePixelData(const OfxRectI& originalRenderWindow,
                        const OfxRectI& renderWindow,
                        unsigned int levels,
                        const void* srcPixelData,
                        OFX::PixelComponentEnum srcPixelComponents,
                        int srcPixelComponentCount,
                        OFX::BitDepthEnum srcPixelDepth,
                        const OfxRectI& srcBounds,
                        int srcRowBytes,
                        void* dstPixelData,
                        OFX::PixelComponentEnum dstPixelComponents,
                        int dstPixelComponentCount,
                        OFX::BitDepthEnum dstPixelDepth,
                        const OfxRectI& dstBounds,
                        int dstRowBytes);

    void fillWithBlack(const OfxRectI &renderWindow,
                       void *dstPixelData,
                       const OfxRectI& dstBounds,
                       OFX::PixelComponentEnum dstPixelComponents,
                       int dstPixelComponentCount,
                       OFX::BitDepthEnum dstBitDepth,
                       int dstRowBytes);


    void premultPixelData(const OfxRectI &renderWindow,
                          const void *srcPixelData,
                          const OfxRectI& srcBounds,
                          OFX::PixelComponentEnum srcPixelComponents,
                          int srcPixelComponentCount,
                          OFX::BitDepthEnum srcPixelDepth,
                          int srcRowBytes,
                          void *dstPixelData,
                          const OfxRectI& dstBounds,
                          OFX::PixelComponentEnum dstPixelComponents,
                          int dstPixelComponentCount,
                          OFX::BitDepthEnum dstBitDepth,
                          int dstRowBytes);

    void unPremultPixelData(const OfxRectI &renderWindow,
                            const void *srcPixelData,
                            const OfxRectI& srcBounds,
                            OFX::PixelComponentEnum srcPixelComponents,
                            int srcPixelComponentCount,
                            OFX::BitDepthEnum srcPixelDepth,
                            int srcRowBytes,
                            void *dstPixelData,
                            const OfxRectI& dstBounds,
                            OFX::PixelComponentEnum dstPixelComponents,
                            int dstPixelComponentCount,
                            OFX::BitDepthEnum dstBitDepth,
                            int dstRowBytes);

    OfxPointD detectProxyScale(const std::string& originalFileName, const std::string& proxyFileName, OfxTime time);

    void setSequenceFromFile(const std::string& filename);

    void refreshSubLabel(OfxTime time);

    bool checkExtension(const std::string& ext);

protected:
#ifdef OFX_IO_USING_OCIO
    std::auto_ptr<GenericOCIO> _ocio;
#endif

    OFX::Clip * _outputClip; //< Mandated output clip
    OFX::StringParam  *_fileParam; //< The input file

    OFX::IntParam* _timeOffset; //< the time offset applied to the sequence
    OFX::IntParam* _startingTime; //< the starting frame of the sequence

    OFX::Int2DParam* _originalFrameRange; //< the original frame range computed the first time by getSequenceTimeDomainInternal

private:
    // the following params should not be needed in derived classes.

    OFX::StringParam  *_proxyFileParam; //< the proxy input files
    OFX::Double2DParam *_proxyThreshold; //< the proxy  images scale threshold
    OFX::Double2DParam *_originalProxyScale; //< the original proxy image scale
    OFX::BooleanParam *_enableCustomScale; //< is custom proxy scale enabled

    OFX::IntParam* _firstFrame; //< the first frame in the sequence (clamped to the time domain)
    OFX::ChoiceParam* _beforeFirst;//< what to do before the first frame
    OFX::IntParam* _lastFrame; //< the last frame in the sequence (clamped to the time domain)
    OFX::ChoiceParam* _afterLast; //< what to do after the last frame

    OFX::ChoiceParam* _frameMode;//< do we use a time offset or an absolute starting frame

    OFX::ChoiceParam* _outputComponents;
    OFX::ChoiceParam* _filePremult;
    OFX::ChoiceParam* _outputPremult;

    OFX::BooleanParam* _timeDomainUserSet; //< true when the time domain has bee nuser edited

    OFX::BooleanParam* _customFPS;
    OFX::DoubleParam* _fps;

    OFX::StringParam* _sublabel;
    OFX::BooleanParam* _guessedParams;//!< was guessParamsFromFilename already successfully called once on this instance

    const std::vector<std::string>& _extensions;

private:
    const bool _supportsRGBA;
    const bool _supportsRGB;
    const bool _supportsXY;
    const bool _supportsAlpha;
    const bool _supportsTiles;
    const bool _isMultiPlanar;

    OFX::PixelComponentEnum _outputComponentsTable[5];
};


void GenericReaderDescribe(OFX::ImageEffectDescriptor &desc,
                           const std::vector<std::string>& extensions, // list of supported extensions
                           int evaluation, // plugin quality from 0 (bad) to 100 (perfect) or -1 if not evaluated
                           bool supportsTiles, bool multiPlanar);

OFX::PageParamDescriptor* GenericReaderDescribeInContextBegin(OFX::ImageEffectDescriptor &desc, OFX::ContextEnum context, bool isVideoStreamPlugin, bool supportsRGBA, bool supportsRGB, bool supportsXY, bool supportsAlpha, bool supportsTiles, bool addSeparatorAfterLastParameter);
void GenericReaderDescribeInContextEnd(OFX::ImageEffectDescriptor &desc, OFX::ContextEnum context, OFX::PageParamDescriptor* page, const char* inputSpaceNameDefault, const char* outputSpaceNameDefault);

#define mDeclareReaderPluginFactory(CLASS, UNLOADFUNCDEF, ISVIDEOSTREAM) \
    class CLASS \
        : public OFX::PluginFactoryHelper<CLASS>                       \
    {                                                                     \
public:                                                                \
        CLASS(const std::string & id, unsigned int verMaj, unsigned int verMin) \
            : OFX::PluginFactoryHelper<CLASS>(id, verMaj, verMin) {} \
        virtual void load();                                   \
        virtual void unload() UNLOADFUNCDEF;                               \
        virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum context); \
        bool isVideoStreamPlugin() const { return ISVIDEOSTREAM; }  \
        virtual void describe(OFX::ImageEffectDescriptor & desc);      \
        virtual void describeInContext(OFX::ImageEffectDescriptor & desc, OFX::ContextEnum context); \
        std::vector<std::string> _extensions; \
    };

NAMESPACE_OFX_IO_EXIT
    NAMESPACE_OFX_EXIT

#endif // ifndef Io_GenericReader_h
