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
 * OFX Masking/Mixing help functions
 */

#ifndef Misc_ofxsMaskMix_h
#define Misc_ofxsMaskMix_h

#include <cfloat> // FLT_EPSILON

#include <ofxsImageEffect.h>

#define kParamPremult "premult"
#define kParamPremultLabel "(Un)premult"
#define kParamPremultHint \
    "Divide the image by the alpha channel before processing, and re-multiply it afterwards. " \
    "Use if the input images are premultiplied."

#define kParamPremultChannel "premultChannel"
#define kParamPremultChannelLabel "By"
#define kParamPremultChannelHint \
    "The channel to use for (un)premult."
#define kParamPremultChannelR "R"
#define kParamPremultChannelRHint "R channel from input"
#define kParamPremultChannelG "G"
#define kParamPremultChannelGHint "G channel from input"
#define kParamPremultChannelB "B"
#define kParamPremultChannelBHint "B channel from input"
#define kParamPremultChannelA "A"
#define kParamPremultChannelAHint "A channel from input"

#define kParamMix "mix"
#define kParamMixLabel "Mix"
#define kParamMixHint "Mix factor between the original and the transformed image."
#define kParamMaskApply "mask"
#define kParamMaskApplyLabel "Mask"
#define kParamMaskApplyHint "When checked, mask is applied."
#define kParamMaskInvert "maskInvert"
#define kParamMaskInvertLabel "Invert Mask"
#define kParamMaskInvertHint "When checked, the effect is fully applied where the mask is 0."

namespace OFX {
inline
void
ofxsPremultDescribeParams(OFX::ImageEffectDescriptor &desc,
                          OFX::PageParamDescriptor *page)
{
    {
        OFX::BooleanParamDescriptor* param = desc.defineBooleanParam(kParamPremult);
        param->setLabel(kParamPremultLabel);
        param->setHint(kParamPremultHint);
#ifdef OFX_EXTENSIONS_NUKE
        param->setLayoutHint(eLayoutHintNoNewLine, 1);
#endif
        if (page) {
            page->addChild(*param);
        }
    }
    {
        // not yet implemented, for future use (whenever deep compositing is supported)
        OFX::ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamPremultChannel);
        param->setLabel(kParamPremultChannelLabel);
        param->setHint(kParamPremultChannelHint);
        param->appendOption(kParamPremultChannelR, kParamPremultChannelRHint);
        param->appendOption(kParamPremultChannelG, kParamPremultChannelGHint);
        param->appendOption(kParamPremultChannelB, kParamPremultChannelBHint);
        param->appendOption(kParamPremultChannelA, kParamPremultChannelAHint);
        param->setDefault(3); // alpha
        param->setIsSecret(true); // not yet implemented
        if (page) {
            page->addChild(*param);
        }
    }
}

inline
bool
ofxsMaskIsAlwaysConnected(OFX::ImageEffectHostDescription *desc)
{
    return (desc->hostName.compare(0, 14, "DaVinciResolve") == 0);
}

inline
void
ofxsMaskDescribeParams(OFX::ImageEffectDescriptor &desc,
                       OFX::PageParamDescriptor *page)
{
    // If the host always sees mask clips are connected, this is a problem because
    // mask will appear as black and transparent, although it is not connected
    if ( ofxsMaskIsAlwaysConnected( OFX::getImageEffectHostDescription() ) ) {
        OFX::BooleanParamDescriptor* param = desc.defineBooleanParam(kParamMaskApply);
        param->setLabel(kParamMaskApplyLabel);
        param->setHint(kParamMaskApplyHint);
        if (page) {
            page->addChild(*param);
        }
    }
    {
        OFX::BooleanParamDescriptor* param = desc.defineBooleanParam(kParamMaskInvert);
        param->setLabel(kParamMaskInvertLabel);
        param->setHint(kParamMaskInvertHint);
        if (page) {
            page->addChild(*param);
        }
    }
}

inline
void
ofxsMixDescribeParams(OFX::ImageEffectDescriptor &desc,
                          OFX::PageParamDescriptor *page)
{
    // GENERIC (MASKED)
    //
    {
        OFX::DoubleParamDescriptor* param = desc.defineDoubleParam(kParamMix);
        param->setLabel(kParamMixLabel);
        param->setHint(kParamMixHint);
        param->setDefault(1.);
        param->setIncrement(0.01);
        param->setRange(0., 1.);
        param->setDisplayRange(0., 1.);
        if (page) {
            page->addChild(*param);
        }
    }
}

inline
void
ofxsMaskMixDescribeParams(OFX::ImageEffectDescriptor &desc,
                          OFX::PageParamDescriptor *page)
{
    // GENERIC (MASKED)
    //
    ofxsMaskDescribeParams(desc, page);
    ofxsMixDescribeParams(desc, page);
}


template <class T>
inline
T
ofxsClamp(T v,
          int min,
          int max)
{
    if ( v < T(min) ) {
        return T(min);
    }
    if ( v > T(max) ) {
        return T(max);
    }

    return v;
}

template <typename PIX, int maxValue>
inline
PIX
ofxsClampIfInt(float v,
               int min,
               int max)
{
    if (maxValue == 1) {
        return v;
    }

    return ofxsClamp(v, min, max) * maxValue + 0.5;
}

// normalize in [0,1]
template <class PIX, int nComponents, int maxValue>
void
ofxsToRGBA(const PIX *srcPix,
           float unpPix[4])
{
    if (!srcPix) {
        // no src pixel here, be black and transparent
        for (int c = 0; c < 4; ++c) {
            unpPix[c] = 0.f;
        }

        return;
    }

    if (nComponents == 1) {
        unpPix[0] = 0.f;
        unpPix[1] = 0.f;
        unpPix[2] = 0.f;
        unpPix[3] = srcPix[0] / (float)maxValue;

        return;
    }

    if (nComponents == 2) {
        unpPix[0] = srcPix[0] / (float)maxValue;
        unpPix[1] = srcPix[1] / (float)maxValue;
        unpPix[2] = 0.f;
        unpPix[3] = 1.0f;

        return;
    }

    unpPix[0] = srcPix[0] / (float)maxValue;
    unpPix[1] = srcPix[1] / (float)maxValue;
    unpPix[2] = srcPix[2] / (float)maxValue;
    unpPix[3] = (nComponents == 4) ? (srcPix[3] / (float)maxValue) : 1.0f;
}

// normalize in [0,1] and unpremultiply srcPix
// if premult is false, just normalize
// unpremult by alpha <= 0 gives identity
// premult(unpremult(p)) or premult(unpremult(p)) is thus not identity for alpha <= 0
template <class PIX, int nComponents, int maxValue>
void
ofxsUnPremult(const PIX *srcPix,
              float unpPix[4],
              bool premult,
              int /*premultChannel*/)
{
    if (!srcPix) {
        // no src pixel here, be black and transparent
        for (int c = 0; c < 4; ++c) {
            unpPix[c] = 0.f;
        }

        return;
    }

    if (nComponents == 1) {
        unpPix[0] = 0.f;
        unpPix[1] = 0.f;
        unpPix[2] = 0.f;
        unpPix[3] = srcPix[0] / (float)maxValue;

        return;
    }

    if (nComponents == 2) {
        unpPix[0] = srcPix[0] / (float)maxValue;
        unpPix[1] = srcPix[1] / (float)maxValue;
        unpPix[2] = 0.f;
        unpPix[3] = 1.0f;

        return;
    }

    // unpremult by alpha <= 0 gives identity
    if ( !premult || (nComponents == 3) || (srcPix[3] <= 0) ) {
        unpPix[0] = srcPix[0] / (float)maxValue;
        unpPix[1] = srcPix[1] / (float)maxValue;
        unpPix[2] = srcPix[2] / (float)maxValue;
        unpPix[3] = (nComponents == 4) ? (srcPix[3] / (float)maxValue) : 1.0f;

        return;
    }

    assert(nComponents == 4);
    PIX alpha = srcPix[3];
    if ( alpha > (PIX)(FLT_EPSILON * maxValue) ) {
        unpPix[0] = srcPix[0] / (float)alpha;
        unpPix[1] = srcPix[1] / (float)alpha;
        unpPix[2] = srcPix[2] / (float)alpha;
    } else {
        unpPix[0] = srcPix[0] / (float)maxValue;
        unpPix[1] = srcPix[1] / (float)maxValue;
        unpPix[2] = srcPix[2] / (float)maxValue;
    }
    unpPix[3] = srcPix[3] / (float)maxValue;
} // ofxsUnPremult

// unpPix is in [0, 1]
// premultiply and denormalize in [0, maxValue]
// if premult is false, just denormalize
// premult by alpha <= 0 gives 0
// premult(unpremult(p)) or premult(unpremult(p)) is thus not identity for alpha <= 0
template <class PIX, int nComponents, int maxValue>
void
ofxsPremult(const float unpPix[4],
            float *tmpPix,
            bool premult,
            int /*premultChannel*/)
{
    if (nComponents == 1) {
        tmpPix[0] = unpPix[3] * maxValue;

        return;
    }

    // premult by alpha <= 0 gives 0
    float alpha = std::max(0.f, unpPix[3]);

    if ( !premult ) {
        tmpPix[0] = unpPix[0] * maxValue;
        if (nComponents >= 2) {
            tmpPix[1] = unpPix[1] * maxValue;
        }
        if (nComponents >= 3) {
            tmpPix[2] = unpPix[2] * maxValue;
        }
        if (nComponents >= 4) {
            tmpPix[3] = alpha * maxValue;
        }

        return;
    }

    tmpPix[0] = unpPix[0] * alpha * maxValue;
    if (nComponents >= 2) {
        tmpPix[1] = unpPix[1] * alpha * maxValue;
    }
    if (nComponents >= 3) {
        tmpPix[2] = unpPix[2] * alpha * maxValue;
    }
    if (nComponents >= 4) {
        tmpPix[3] = alpha * maxValue;
    }
}

// tmpPix is not normalized, it is within [0,maxValue]
template <class PIX, int nComponents, int maxValue>
void
ofxsPix(const float *tmpPix, //!< interpolated pixel
        PIX *dstPix) //!< destination pixel
{
    // no mask, no mix
    for (int c = 0; c < nComponents; ++c) {
        dstPix[c] = ofxsClampIfInt<PIX, maxValue>(tmpPix[c], 0, maxValue);
    }
} // ofxsMixPix

// unpPix is normalized between [0,1]
template <class PIX, int nComponents, int maxValue>
void
ofxsPremultPix(const float unpPix[4], //!< interpolated unpremultiplied pixel
               bool premult,
               int premultChannel,
               PIX *dstPix) //!< destination pixel
{
    float tmpPix[nComponents];

    // unpPix is in [0..1]
    ofxsPremult<PIX, nComponents, maxValue>(unpPix, tmpPix, premult, premultChannel);
    // tmpPix is in [0..maxValue]
    ofxsPix<PIX, nComponents, maxValue>(tmpPix, dstPix);
}


// tmpPix is not normalized, it is within [0,maxValue]
template <class PIX, int nComponents, int maxValue>
void
ofxsMixPix(const float *tmpPix, //!< interpolated pixel
           const PIX *srcPix, //!< the background image (the output is srcImg where maskImg=0, else it is tmpPix)
           float mix, //!< mix factor between the output and bkImg
           PIX *dstPix) //!< destination pixel
{
    if (mix == 1.) {
        ofxsPix<PIX, nComponents, maxValue>(tmpPix, dstPix);
    } else {
        // just mix
        float alpha = mix;
        if (alpha == 0.) {
            if (srcPix) {
                for (int c = 0; c < nComponents; ++c) {
                    dstPix[c] = ofxsClampIfInt<PIX, maxValue>(srcPix[c], 0, maxValue);
                }
            } else {
                for (int c = 0; c < nComponents; ++c) {
                    dstPix[c] = 0;
                }
            }
        } else if (alpha == 1) {
            for (int c = 0; c < nComponents; ++c) {
                dstPix[c] = ofxsClampIfInt<PIX, maxValue>(tmpPix[c], 0, maxValue);
            }
        } else {
            if (srcPix) {
                for (int c = 0; c < nComponents; ++c) {
                    float v = tmpPix[c] * alpha + (1.f - alpha) * srcPix[c];
                    dstPix[c] = ofxsClampIfInt<PIX, maxValue>(v, 0, maxValue);
                }
            } else {
                for (int c = 0; c < nComponents; ++c) {
                    float v = tmpPix[c] * alpha;
                    dstPix[c] = ofxsClampIfInt<PIX, maxValue>(v, 0, maxValue);
                }
            }
        }
    }
} // ofxsMixPix

// tmpPix is not normalized, it is within [0,maxValue]
template <class PIX, int nComponents, int maxValue, bool masked>
void
ofxsMaskMixPix(const float *tmpPix, //!< interpolated pixel
               int x, //!< coordinates for the pixel to be computed (PIXEL coordinates)
               int y,
               const PIX *srcPix, //!< the background image (the output is srcImg where maskImg=0, else it is tmpPix)
               bool domask, //!< apply the mask?
               const OFX::Image *maskImg, //!< the mask image (ignored if masked=false or domask=false), which must be Alpha
               float mix, //!< mix factor between the output and bkImg
               bool maskInvert, //<! invert mask behavior
               PIX *dstPix) //!< destination pixel
{
    assert(!domask || !maskImg || maskImg->getPixelComponents() == ePixelComponentAlpha);
    const PIX *maskPix = NULL;
    float maskScale = 1.f;

    // are we doing masking
    if (!masked) {
        ofxsMixPix<PIX, nComponents, maxValue>(tmpPix, srcPix, mix, dstPix);
    } else {
        if (domask) {
            // we do, get the pixel from the mask
            maskPix = maskImg ? (const PIX *)maskImg->getPixelAddress(x, y) : 0;
            // figure the scale factor from that pixel
            if (maskPix == 0) {
                maskScale = maskInvert ? 1.f : 0.f;
            } else {
                maskScale = *maskPix / float(maxValue);
                if (maskInvert) {
                    maskScale = 1.f - maskScale;
                }
            }
        }
        float alpha = maskScale * mix;
        if (alpha == 0.) {
            if (srcPix) {
                for (int c = 0; c < nComponents; ++c) {
                    dstPix[c] = ofxsClampIfInt<PIX, maxValue>(srcPix[c], 0, maxValue);
                }
            } else {
                for (int c = 0; c < nComponents; ++c) {
                    dstPix[c] = 0;
                }
            }
        } else if (alpha == 1.) {
            for (int c = 0; c < nComponents; ++c) {
                dstPix[c] = ofxsClampIfInt<PIX, maxValue>(tmpPix[c], 0, maxValue);
            }
        } else {
            if (srcPix) {
                for (int c = 0; c < nComponents; ++c) {
                    float v = tmpPix[c] * alpha + (1.f - alpha) * srcPix[c];
                    dstPix[c] = ofxsClampIfInt<PIX, maxValue>(v, 0, maxValue);
                }
            } else {
                for (int c = 0; c < nComponents; ++c) {
                    float v = tmpPix[c] * alpha;
                    dstPix[c] = ofxsClampIfInt<PIX, maxValue>(v, 0, maxValue);
                }
            }
        }
    }
} // ofxsMaskMixPix

// unpPix is normalized between [0,1]
template <class PIX, int nComponents, int maxValue, bool masked>
void
ofxsPremultMaskMixPix(const float unpPix[4], //!< interpolated unpremultiplied pixel
                      bool premult,
                      int premultChannel,
                      int x, //!< coordinates for the pixel to be computed (PIXEL coordinates)
                      int y,
                      const PIX *srcPix, //!< the background image (the output is srcImg where maskImg=0, else it is tmpPix)
                      bool domask, //!< apply the mask?
                      const OFX::Image *maskImg, //!< the mask image (ignored if masked=false or domask=false)
                      float mix, //!< mix factor between the output and bkImg
                      bool maskInvert, //<! invert mask behavior
                      PIX *dstPix) //!< destination pixel
{
    assert(!domask || !maskImg || maskImg->getPixelComponents() == ePixelComponentAlpha);
    float tmpPix[nComponents];

    // unpPix is in [0..1]
    ofxsPremult<PIX, nComponents, maxValue>(unpPix, tmpPix, premult, premultChannel);
    // tmpPix is in [0..maxValue]
    ofxsMaskMixPix<PIX, nComponents, maxValue, masked>(tmpPix, x, y, srcPix, domask, maskImg, mix, maskInvert, dstPix);
}

// unpPix is normalized between [0,1]
template <class PIX, int nComponents, int maxValue>
void
ofxsPremultMixPix(const float unpPix[4], //!< interpolated unpremultiplied pixel
                  bool premult,
                  int premultChannel,
                  const PIX *srcPix, //!< the background image (the output is srcImg where maskImg=0, else it is tmpPix)
                  float mix, //!< mix factor between the output and bkImg
                  PIX *dstPix) //!< destination pixel
{
    float tmpPix[nComponents];

    // unpPix is in [0..1]
    ofxsPremult<PIX, nComponents, maxValue>(unpPix, tmpPix, premult, premultChannel);
    // tmpPix is in [0..maxValue]
    ofxsMixPix<PIX, nComponents, maxValue>(tmpPix, srcPix, mix, dstPix);
}

// tmpPix is not normalized, it is within [0,maxValue]
template <class PIX, int nComponents, int maxValue, bool masked>
void
ofxsMaskMix(const float *tmpPix, //!< interpolated pixel
            int x, //!< coordinates for the pixel to be computed (PIXEL coordinates)
            int y,
            const OFX::Image *srcImg, //!< the background image (the output is srcImg where maskImg=0, else it is tmpPix)
            bool domask, //!< apply the mask?
            const OFX::Image *maskImg, //!< the mask image (ignored if masked=false or domask=false)
            float mix, //!< mix factor between the output and bkImg
            bool maskInvert, //<! invert mask behavior
            PIX *dstPix) //!< destination pixel
{
    assert(!domask || !maskImg || maskImg->getPixelComponents() == ePixelComponentAlpha);
    const PIX *srcPix = NULL;

    // are we doing masking/mixing? in this case, retrieve srcPix
    if (masked && srcImg) {
        if ( (domask /*&& maskImg*/) || (mix != 1.) ) {
            srcPix = (const PIX *)srcImg->getPixelAddress(x, y);
        }
    }

    return ofxsMaskMixPix<PIX, nComponents, maxValue, masked>(tmpPix, x, y, srcPix, domask, maskImg, mix, maskInvert, dstPix);
}
} // OFX

#endif // ifndef Misc_ofxsMaskMix_h
