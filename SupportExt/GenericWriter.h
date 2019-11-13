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
 * OFX GenericWriter plugin.
 * A base class for all OpenFX-based encoders.
 */

#ifndef Io_GenericWriter_h
#define Io_GenericWriter_h

#include <memory>
#include <ofxsImageEffect.h>
#include <ofxsMultiPlane.h>
#include "IOUtility.h"
#include "ofxsMacros.h"
#include "ofxsPixelProcessor.h" // for getImageData
#include "ofxsCopier.h" // for copyPixels

namespace OFX {
class PixelProcessorFilterBase;

namespace IO {
#ifdef OFX_IO_USING_OCIO
class GenericOCIO;
#endif

enum LayerViewsPartsEnum
{
    eLayerViewsSinglePart = 0,
    eLayerViewsSplitViews,
    eLayerViewsSplitViewsLayers
};

#define kGenericWriterViewDefault -2 // Indicates that we want to render what the host request via the render action (the default)
#define kGenericWriterViewAll -1 // the write will write all views when rendering view 0

/**
 * @brief A generic writer plugin, derive this to create a new writer for a specific file format.
 * This class propose to handle the common stuff among writers:
 * - common params
 * - a way to inform the host about the colour-space of the data.
 **/
class GenericWriterPlugin
    : public OFX::MultiPlane::MultiPlaneEffect
{
public:

    GenericWriterPlugin(OfxImageEffectHandle handle,
                        const std::vector<std::string>& extensions,
                        bool supportsRGBA, bool supportsRGB, bool supportsXY, bool supportsAlpha);

    virtual ~GenericWriterPlugin();

    /**
     * @brief Don't override this function, the GenericWriterPlugin class already does the rendering. The "encoding" of the frame
     * must be done by the pure virtual function encode(...) instead.
     * The render function also copies the image from the input clip to the output clip (only if the effect is connected downstream)
     * in order to be able to branch this effect in the middle of an effect tree.
     **/
    virtual void render(const OFX::RenderArguments &args) OVERRIDE FINAL;

    /* override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments &args, OFX::Clip * &identityClip, double &identityTime) OVERRIDE;

    /** @brief client begin sequence render function */
    virtual void beginSequenceRender(const OFX::BeginSequenceRenderArguments &args) OVERRIDE;

    /** @brief client end sequence render function */
    virtual void endSequenceRender(const OFX::EndSequenceRenderArguments &args) OVERRIDE;

    /**
     * @brief Don't override this. It returns the projects region of definition.
     **/
    virtual bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments &args, OfxRectD &rod) OVERRIDE FINAL;

    /**
     * @brief Don't override this. It returns the source region of definition.
     **/
    //virtual void getRegionsOfInterest(const OFX::RegionsOfInterestArguments &args, OFX::RegionOfInterestSetter &rois) OVERRIDE FINAL;

    /**
     * @brief Don't override this. It returns the frame range to render.
     **/
    virtual bool getTimeDomain(OfxRangeD &range) OVERRIDE FINAL;

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

    /**
     * @brief Overriden to handle premultiplication parameter given the input clip.
     * Make sure you call the base class implementation if you override it.
     **/
    virtual void changedClip(const OFX::InstanceChangedArgs &args, const std::string &clipName) OVERRIDE;

    /**
     * @brief Overriden to set the clips premultiplication according to the user and plug-ins wishes.
     * It also set the output components from the output components parameter
     **/
    virtual void getClipPreferences(OFX::ClipPreferencesSetter &clipPreferences) OVERRIDE;


    /**
     * @brief Overriden to request the views needed to render.
     **/
    virtual void getFrameViewsNeeded(const OFX::FrameViewsNeededArguments& args, OFX::FrameViewsNeededSetter& frameViews) OVERRIDE FINAL;

    /**
     * @brief Overriden to clear any OCIO cache.
     * This function calls clearAnyCache() if you have any cache to clear.
     **/
    virtual void purgeCaches(void) OVERRIDE FINAL;

    /**
     * @brief Restore any state from the parameters set
     * Called from createInstance() and changedParam() (via outputFileChanged()), must restore the
     * state of the Reader, such as Choice param options, data members and non-persistent param values.
     * We don't do this in the ctor of the plug-in since we can't call virtuals yet.
     * Any derived implementation must call GenericWriterPlugin::restoreStateFromParams() first
     **/
    virtual void restoreStateFromParams();

protected:

    void setOutputComponentsParam(OFX::PixelComponentEnum components);


    /**
     * @brief Override this function to actually encode the image in the file pointed to by filename.
     * If the file is a video-stream then you should encode the frame at the time given in parameters.
     * You must write the decoded image into dstImg. This function should convert the  pixels from srcImg
     * into the color-space and bitdepths of the newly created images's file.
     * You can inform the host of the bitdepth you support in input in the describe() function.
     * You can always skip the color-space conversion, but for all linear hosts it would produce either
     * false colors or sub-par performances in the case the end-user has to prepend a color-space conversion
     * effect her/himself.
     *
     * @param filename The output file to write to
     * @param time The frame number
     * @param viewName The name of the view to render
     * @param pixelData Pointer to the start of the input image
     * @param bounds The bounds of the pixelData buffer
     * @param pixelAspectRatio The PAR of the source image
     * @param pixelDataNComps The number of components per pixel in pixelData
     * @param dstNCompsStartIndex The start index where the first component of dstNComps is to be read (in the range of pixelDataNComps)
     * @param dstNComps The desired number of components in the written file
     * @param rowBytes The number of bytes in a row of pixelData.
     * The following assert should hold true:
     * assert(((bounds.x2 - bounds.x1) * pixelDataNComps * sizeof(float)) == rowBytes);
     *
     * @pre The filename has been validated against the supported file extensions.
     * You don't need to check this yourself.
     * The source image has been correctly color-converted
     **/
    virtual void encode(const std::string& filename,
                        const OfxTime time,
                        const std::string& viewName,
                        const float *pixelData,
                        const OfxRectI& bounds,
                        const float pixelAspectRatio,
                        const int pixelDataNComps,
                        const int dstNCompsStartIndex,
                        const int dstNComps,
                        const int rowBytes);
    virtual void beginEncode(const std::string& /*filename*/,
                             const OfxRectI& /*rodPixel*/,
                             float /*pixelAspectRatio*/,
                             const OFX::BeginSequenceRenderArguments & /*args*/) {}

    virtual void endEncode(const OFX::EndSequenceRenderArguments & /*args*/) {}

    friend class EncodePlanesLocalData_RAII;
    ///Used to allocate/free userdata passed to beginEncodePlanes,endEncodePlanes and encodePlane
    virtual void* allocateEncodePlanesUserData() { return (void*)0; }

    virtual void destroyEncodePlanesUserData(void* /*data*/) {}

    /**
     * @brief When writing multiple planes, should allocate data that are shared amongst all planes
     **/
    virtual void beginEncodeParts(void* user_data,
                                  const std::string& filename,
                                  OfxTime time,
                                  float pixelAspectRatio,
                                  LayerViewsPartsEnum partsSplitting,
                                  const std::map<int, std::string>& viewsToRender,
                                  const std::list<std::string>& planes,
                                  const bool packingRequired,
                                  const std::vector<int>& packingMapping,
                                  const OfxRectI& bounds);
    virtual void endEncodeParts(void* /*user_data*/) {}

    virtual void encodePart(void* user_data, const std::string& filename, const float *pixelData, int pixelDataNComps, int planeIndex, int rowBytes);

    /**
     * @brief Should return the view index needed to render.
     * Possible return values:
     * -2 or kGenericWriterViewDefault: Indicates that we want to render what the host request via the render action (the default)
     * -1 or kGenericWriterViewAll: Indicates that we want to render all views in a single file. In this case, rendering view 0 requires all views.
     * >= 0: Indicates the view index to render
     **/
    virtual int getViewToRender() const { return -2; }

    virtual LayerViewsPartsEnum getPartsSplittingPreference() const { return eLayerViewsSinglePart; }

    /**
     * @brief Overload to return false if the given file extension is a video file extension or
     * true if this is an image file extension.
     **/
    virtual bool isImageFile(const std::string& fileExtension) const = 0;
    virtual void setOutputFrameRate(double /*fps*/) {}

    /**
     * @brief Must return whether your plug-in expects an input stream to be premultiplied or unpremultiplied to encode
     * properly into the file.
     **/
    virtual OFX::PreMultiplicationEnum getExpectedInputPremultiplication() const = 0;

    /**
     * @brief To implement if you added supportsDisplayWindow = true to GenericWriterDescribe().
     * Basically only EXR file format can handle this.
     **/
    virtual bool displayWindowSupportedByFormat(const std::string& /*filename*/) const { return false; }


    OFX::Clip* _inputClip; //< Mantated input clip
    OFX::Clip *_outputClip; //< Mandated output clip
    OFX::StringParam  *_fileParam; //< The output file
    OFX::ChoiceParam *_frameRange; //<The frame range type
    OFX::IntParam* _firstFrame; //< the first frame if the frame range type is "Manual"
    OFX::IntParam* _lastFrame; //< the last frame if the frame range type is "Manual"
    OFX::ChoiceParam* _outputFormatType; //< the type of output format
    OFX::ChoiceParam* _outputFormat; //< the output format to render
    OFX::Int2DParam* _outputFormatSize;
    OFX::DoubleParam* _outputFormatPar;
    OFX::ChoiceParam* _premult;
    OFX::BooleanParam* _clipToRoD;

    OFX::StringParam* _sublabel;
    OFX::BooleanParam* _processChannels[4];
    OFX::ChoiceParam* _outputComponents;
    OFX::BooleanParam* _guessedParams; //!< was guessParamsFromFilename already successfully called once on this instance

#ifdef OFX_IO_USING_OCIO
    OFX::BooleanParam* _outputSpaceSet;
    std::auto_ptr<GenericOCIO> _ocio;
#endif

    const std::vector<std::string>& _extensions;
    bool _supportsRGBA;
    bool _supportsRGB;
    bool _supportsXY;
    bool _supportsAlpha;

    std::vector<OFX::PixelComponentEnum> _outputComponentsTable;

private:


    class InputImagesHolder
    {
        std::list<const OFX::Image*> _imgs;
        std::list<OFX::ImageMemory*> _mems;

public:

        InputImagesHolder();
        void addImage(const OFX::Image* img);
        void addMemory(OFX::ImageMemory* mem);
        ~InputImagesHolder();
    };

    /*
     * @brief Fetch the given plane for the given view at the given time and convert into suited color-space
     * using OCIO if needed.
     *
     * If view == renderRequestedView the output image will be fetched on the output clip
     * and written to aswell.
     *
     * Post-condition:
     * - srcImgsHolder had the srcImg appended to it so it gets correctly released when it is
     * destroyed.
     * - tmpMemPtr is never NULL and points to either srcImg buffer or tmpMem buffer
     * - If a color-space conversion occured, tmpMem/tmpMemPtr is non-null and tmpMem was added to srcImgsHolder
     * so it gets correctly released upon destruction.
     *
     * This function MAY throw exceptions aborting the action, that is why we use the InputImagesHolder RAII style class
     * that will properly release resources.
     */
    void fetchPlaneConvertAndCopy(const std::string& plane,
                                  bool failIfNoSrcImg,
                                  int view,
                                  int renderRequestedView,
                                  double time,
                                  const OfxRectI& renderWindow,
                                  const OfxPointD& renderScale,
                                  OFX::FieldEnum fieldToRender,
                                  OFX::PreMultiplicationEnum pluginExpectedPremult,
                                  OFX::PreMultiplicationEnum userPremult,
                                  const bool isOCIOIdentity,
                                  const bool doAnyPacking,
                                  const bool packingContiguous,
                                  const std::vector<int>& packingMapping,
                                  InputImagesHolder* srcImgsHolder,
                                  OfxRectI* bounds,
                                  OFX::ImageMemory** tmpMem,
                                  const OFX::Image** inputImage,
                                  float** tmpMemPtr,
                                  int* rowBytes,
                                  OFX::PixelComponentEnum* mappedComponents,
                                  int* mappedComponentsCount);

    /**
     * @brief Checks if the extension is supported.
     **/
    bool checkExtension(const std::string& filename);

    /**
     * @brief Override if you want to do something when the output image/video file changed.
     * You shouldn't do any strong processing as this is called on the main thread and
     * the getRegionOfDefinition() and  decode() should open the file in a separate thread.
     **/
    virtual void onOutputFileChanged(const std::string& newFile, bool setColorSpace) = 0;

    /**
     * @brief Override to clear any cache you may have.
     **/
    virtual void clearAnyCache() {}

    void getOutputRoD(OfxTime time, int view, OfxRectD* rod, double* par);

protected:

    void getSelectedOutputFormat(OfxRectI* format, double* par);

private:

    void copyPixelData(const OfxRectI &renderWindow,
                       const OFX::Image* srcImg,
                       OFX::Image* dstImg)
    {
        const void* srcPixelData;
        OfxRectI srcBounds;

        OFX::PixelComponentEnum srcPixelComponents;
        OFX::BitDepthEnum srcBitDepth;
        int srcRowBytes;
        getImageData(srcImg, &srcPixelData, &srcBounds, &srcPixelComponents, &srcBitDepth, &srcRowBytes);
        int srcPixelComponentCount = srcImg->getPixelComponentCount();
        void* dstPixelData;
        OfxRectI dstBounds;
        OFX::PixelComponentEnum dstPixelComponents;
        OFX::BitDepthEnum dstBitDepth;
        int dstRowBytes;
        getImageData(dstImg, &dstPixelData, &dstBounds, &dstPixelComponents, &dstBitDepth, &dstRowBytes);
        int dstPixelComponentCount = dstImg->getPixelComponentCount();
        copyPixels(*this,
                   renderWindow,
                   srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                   dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    }

    void copyPixelData(const OfxRectI &renderWindow,
                       const void *srcPixelData,
                       const OfxRectI& srcBounds,
                       OFX::PixelComponentEnum srcPixelComponents,
                       int srcPixelComponentCount,
                       OFX::BitDepthEnum srcBitDepth,
                       int srcRowBytes,
                       OFX::Image* dstImg)
    {
        void* dstPixelData;
        OfxRectI dstBounds;

        OFX::PixelComponentEnum dstPixelComponents;
        OFX::BitDepthEnum dstBitDepth;
        int dstRowBytes;
        getImageData(dstImg, &dstPixelData, &dstBounds, &dstPixelComponents, &dstBitDepth, &dstRowBytes);
        int dstPixelComponentCount = dstImg->getPixelComponentCount();
        copyPixels(*this,
                   renderWindow,
                   srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                   dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    }

    void copyPixelData(const OfxRectI &renderWindow,
                       const OFX::Image* srcImg,
                       void *dstPixelData,
                       const OfxRectI& dstBounds,
                       OFX::PixelComponentEnum dstPixelComponents,
                       int dstPixelComponentCount,
                       OFX::BitDepthEnum dstBitDepth,
                       int dstRowBytes)
    {
        const void* srcPixelData;
        OfxRectI srcBounds;

        OFX::PixelComponentEnum srcPixelComponents;
        OFX::BitDepthEnum srcBitDepth;
        int srcRowBytes;
        getImageData(srcImg, &srcPixelData, &srcBounds, &srcPixelComponents, &srcBitDepth, &srcRowBytes);
        int srcPixelComponentCount = srcImg->getPixelComponentCount();
        copyPixels(*this,
                   renderWindow,
                   srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                   dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    }

    void packPixelBuffer(const OfxRectI& renderWindow,
                         const void *srcPixelData,
                         const OfxRectI& bounds,
                         OFX::BitDepthEnum bitDepth,
                         int srcRowBytes,
                         OFX::PixelComponentEnum srcPixelComponents,
                         const std::vector<int>& channelsMapping, //maps dst channels to input channels
                         int dstRowBytes,
                         void* dstPixelData);

    void interleavePixelBuffers(const OfxRectI& renderWindow,
                                const void *srcPixelData,
                                const OfxRectI& bounds,
                                const OFX::PixelComponentEnum srcPixelComponents,
                                const int srcPixelComponentCount,
                                const int srcNCompsStartIndex,
                                const int desiredSrcNComps,
                                const OFX::BitDepthEnum bitDepth,
                                const int srcRowBytes,
                                const OfxRectI& dstBounds,
                                const OFX::PixelComponentEnum dstPixelComponents,
                                const int dstPixelComponentStartIndex,
                                const int dstPixelComponentCount,
                                const int dstRowBytes,
                                void* dstPixelData);

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

    void getPackingOptions(bool *allCheckboxHidden, std::vector<int>* packingMapping) const;

    void outputFileChanged(OFX::InstanceChangeReason reason, bool restoreExistingWriter, bool throwErrors);
};

class EncodePlanesLocalData_RAII
{
    GenericWriterPlugin* _w;
    void* data;

public:

    EncodePlanesLocalData_RAII(GenericWriterPlugin* w)
        : _w(w)
        , data(0)
    {
        data = w->allocateEncodePlanesUserData();
    }

    ~EncodePlanesLocalData_RAII()
    {
        _w->destroyEncodePlanesUserData(data);
    }

    void* getData() const { return data; }
};

void GenericWriterDescribe(OFX::ImageEffectDescriptor &desc,
                           OFX::RenderSafetyEnum safety,
                           const std::vector<std::string>& extensions,
                           int evaluation,
                           bool isMultiPlanar,
                           bool isMultiView);

OFX::PageParamDescriptor* GenericWriterDescribeInContextBegin(OFX::ImageEffectDescriptor &desc,
                                                              OFX::ContextEnum context,
                                                              bool supportsRGBA,
                                                              bool supportsRGB,
                                                              bool supportsXY,
                                                              bool supportsAlpha,
                                                              const char* inputSpaceNameDefault,
                                                              const char* outputSpaceNameDefault,
                                                              bool supportsDisplayWindow);

void GenericWriterDescribeInContextEnd(OFX::ImageEffectDescriptor &desc,
                                       OFX::ContextEnum context,
                                       OFX::PageParamDescriptor* defaultPage);

// the load() member has to be provided, and it should fill the _extensions list of valid file extensions
#define mDeclareWriterPluginFactory(CLASS, UNLOADFUNCDEF, ISVIDEOSTREAM) \
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
private: \
        std::vector<std::string> _extensions; \
    };
} // namespace IO
} // namespace OFX

#endif // ifndef Io_GenericWriter_h
