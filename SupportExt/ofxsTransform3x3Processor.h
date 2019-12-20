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
 * OFX Transform plugin.
 */

#ifndef MISC_TRANSFORMPROCESSOR_H
#define MISC_TRANSFORMPROCESSOR_H

#include <algorithm>

#include "ofxsProcessing.H"
#include "ofxsMatrix2D.h"
#include "ofxsFilter.h"
#include "ofxsMaskMix.h"
#include "ofxsMacros.h"

// constants for the motion blur algorithm (may depend on _motionblur)
#define kTransform3x3ProcessorMotionBlurMaxError (_motionblur * maxValue / 1000.)
#define kTransform3x3ProcessorMotionBlurMinIterations ( std::max( 13, (int)(kTransform3x3ProcessorMotionBlurMaxIterations / 3) ) )
#define kTransform3x3ProcessorMotionBlurMaxIterations ( (int)(_motionblur * 40) )

namespace OFX {
class Transform3x3ProcessorBase
    : public OFX::ImageProcessor
{
protected:
    const OFX::Image *_srcImg;
    const OFX::Image *_maskImg;
    // NON-GENERIC PARAMETERS:
    const OFX::Matrix3x3* _invtransform; // the set of transforms to sample from (in PIXEL coords)
    const double* _invtransformalpha; // blending factor for each tranform, or NULL for uniform blending
    size_t _invtransformsize;
    // GENERIC PARAMETERS:
    bool _blackOutside;
    double _motionblur; // quality of the motion blur. 0 means disabled
    bool _domask;
    double _mix;
    bool _maskInvert;

public:

    Transform3x3ProcessorBase(OFX::ImageEffect &instance)
        : OFX::ImageProcessor(instance)
        , _srcImg(0)
        , _maskImg(0)
        , _invtransform()
        , _invtransformalpha(0)
        , _invtransformsize(0)
        , _blackOutside(false)
        , _motionblur(0.)
        , _domask(false)
        , _mix(1.0)
        , _maskInvert(false)
    {
    }

    virtual FilterEnum getFilter() const = 0;
    virtual bool getClamp() const = 0;

    /** @brief set the src image */
    void setSrcImg(const OFX::Image *v)
    {
        _srcImg = v;
    }

    /** @brief set the optional mask image */
    void setMaskImg(const OFX::Image *v,
                    bool maskInvert)
    {
        _maskImg = v; _maskInvert = maskInvert;
    }

    // Are we masking. We can't derive this from the mask image being set as NULL is a valid value for an input image
    void doMasking(bool v)
    {
        _domask = v;
    }

    void setValues(const OFX::Matrix3x3* invtransform, //!< non-generic - must be in PIXEL coords
                   double* invtransformalpha,
                   size_t invtransformsize,
                   // all generic parameters below
                   bool blackOutside, //!< generic
                   double motionblur,
                   double mix)          //!< generic
    {
        // NON-GENERIC
        assert(invtransform);
        _invtransform = invtransform;
        _invtransformalpha = invtransformalpha;
        _invtransformsize = invtransformsize;
        // GENERIC
        _blackOutside = blackOutside;
        _motionblur = motionblur;
        _mix = mix;
    }
};


// The "masked", "filter" and "clamp" template parameters allow filter-specific optimization
// by the compiler, using the same generic code for all filters.
template <class PIX, int nComponents, int maxValue, bool masked, FilterEnum filter, bool clamp>
class Transform3x3Processor
    : public Transform3x3ProcessorBase
{
public:
    Transform3x3Processor(OFX::ImageEffect &instance)
        : Transform3x3ProcessorBase(instance)
    {
    }

private:
    virtual FilterEnum getFilter() const OVERRIDE FINAL
    {
        return filter;
    }

    virtual bool getClamp() const OVERRIDE FINAL
    {
        return clamp;
    }

    void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE
    {
        assert(_invtransform);
        if (_motionblur == 0.) { // no motion blur
            return multiThreadProcessImagesNoBlur(procWindow);
        } else { // motion blur
            return multiThreadProcessImagesMotionBlur(procWindow);
        }
    } // multiThreadProcessImages

private:
    void multiThreadProcessImagesNoBlur(const OfxRectI &procWindow)
    {
        float tmpPix[nComponents];
        const OFX::Matrix3x3 & H = _invtransform[0];
        const int x1 = _srcImg ? _srcImg->getBounds().x1 : 0;
        const int x2 = _srcImg ? _srcImg->getBounds().x2 : 0;
        const int y1 = _srcImg ? _srcImg->getBounds().y1 : 0;
        const int y2 = _srcImg ? _srcImg->getBounds().y2 : 0;

        for (int y = procWindow.y1; y < procWindow.y2; ++y) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);

            // the coordinates of the center of the pixel in canonical coordinates
            // see http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#CanonicalCoordinates
            OFX::Point3D canonicalCoords;
            canonicalCoords.z = 1;
            canonicalCoords.y = (double)y + 0.5;

            for (int x = procWindow.x1; x < procWindow.x2; ++x, dstPix += nComponents) {
                // NON-GENERIC TRANSFORM

                // the coordinates of the center of the pixel in canonical coordinates
                // see http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#CanonicalCoordinates
                canonicalCoords.x = (double)x + 0.5;
                OFX::Point3D transformed = H * canonicalCoords;
                if ( !_srcImg || (transformed.z <= 0.) ) {
                    // the back-transformed point is at infinity (==0) or behind the camera (<0)
                    for (int c = 0; c < nComponents; ++c) {
                        tmpPix[c] = 0;
                    }
                } else {
                    double fx = transformed.z != 0 ? transformed.x / transformed.z : transformed.x;
                    double fy = transformed.z != 0 ? transformed.y / transformed.z : transformed.y;
                    if (filter == eFilterImpulse) {
                        ofxsFilterInterpolate2D<PIX, nComponents, filter, clamp>(fx, fy, _srcImg, _blackOutside, tmpPix);
                    } else {
                        bool xinside = (x1 <= fx + 0.5 && fx - 0.5 < x2);
                        bool yinside = (y1 <= fy + 0.5 && fy - 0.5 < y2);
                        if ( _blackOutside && !(xinside && yinside) ) {
                            xinside = yinside = false;
                        }

                        double Jxx = xinside ? (H(0,0) * transformed.z - transformed.x * H(2,0)) / (transformed.z * transformed.z) : 0.;
                        double Jxy = xinside ? (H(0,1) * transformed.z - transformed.x * H(2,1)) / (transformed.z * transformed.z) : 0.;
                        double Jyx = yinside ? (H(1,0) * transformed.z - transformed.y * H(2,0)) / (transformed.z * transformed.z) : 0;
                        double Jyy = yinside ? (H(1,1) * transformed.z - transformed.y * H(2,1)) / (transformed.z * transformed.z) : 0.;
                        ofxsFilterInterpolate2DSuper<PIX, nComponents, filter, clamp>(fx, fy, Jxx, Jxy, Jyx, Jyy, _srcImg, _blackOutside, tmpPix);
                    }
                }

                ofxsMaskMix<PIX, nComponents, maxValue, masked>(tmpPix, x, y, _srcImg, _domask, _maskImg, (float)_mix, _maskInvert, dstPix);
            }
        }
    } // multiThreadProcessImagesNoBlur

    void multiThreadProcessImagesMotionBlur(const OfxRectI &procWindow)
    {
        float tmpPix[nComponents];
        const double maxErr2 = kTransform3x3ProcessorMotionBlurMaxError * kTransform3x3ProcessorMotionBlurMaxError; // maximum expected squared error
        const int maxIt = kTransform3x3ProcessorMotionBlurMaxIterations; // maximum number of iterations
        const int x1 = _srcImg ? _srcImg->getBounds().x1 : 0;
        const int x2 = _srcImg ? _srcImg->getBounds().x2 : 0;
        const int y1 = _srcImg ? _srcImg->getBounds().y1 : 0;
        const int y2 = _srcImg ? _srcImg->getBounds().y2 : 0;

        // Monte Carlo integration, starting with at least 13 regularly spaced samples, and then low discrepancy
        // samples from the van der Corput sequence.
        for (int y = procWindow.y1; y < procWindow.y2; ++y) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);

            // the coordinates of the center of the pixel in canonical coordinates
            // see http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#CanonicalCoordinates
            OFX::Point3D canonicalCoords;
            canonicalCoords.z = 1;
            canonicalCoords.y = (double)y + 0.5;

            for (int x = procWindow.x1; x < procWindow.x2; ++x, dstPix += nComponents) {
                double acc;
                double accPix[nComponents];
                double accPix2[nComponents];
                double mean[nComponents];
                double var[nComponents];
                for (int c = 0; c < nComponents; ++c) {
                    acc = 0.;
                    accPix[c] = 0;
                    accPix2[c] = 0;
                    mean[c] = 0.;
                    var[c] = (double)maxValue * maxValue;
                }
                unsigned int seed = (unsigned int)( hash(hash( x + (unsigned int)(0x10000 * _motionblur) ) + y) );
                int sample = 0;
                const int minsamples = kTransform3x3ProcessorMotionBlurMinIterations; // minimum number of samples (at most maxIt/3
                int maxsamples = minsamples;
                while (sample < maxsamples) {
                    for (; sample < maxsamples; ++sample, ++seed) {
                        //int t = 0.5*(van_der_corput<2>(seed1) + van_der_corput<3>(seed2)) * _invtransform.size();
                        int t;
                        if (sample < minsamples) {
                            // distribute the first samples evenly over the interval
                            t = (int)( ( sample  + van_der_corput<2>(seed) ) * _invtransformsize / (double)minsamples );
                        } else {
                            t = (int)(van_der_corput<2>(seed) * _invtransformsize);
                        }
                        // NON-GENERIC TRANSFORM

                        // the coordinates of the center of the pixel in canonical coordinates
                        // see http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#CanonicalCoordinates
                        canonicalCoords.x = (double)x + 0.5;
                        const OFX::Matrix3x3& H = _invtransform[t];
                        OFX::Point3D transformed = H * canonicalCoords;
                        if ( !_srcImg || (transformed.z <= 0.) ) {
                            // the back-transformed point is at infinity (==0) or behind the camera (<0)
                            for (int c = 0; c < nComponents; ++c) {
                                tmpPix[c] = 0;
                            }
                        } else {
                            double fx = transformed.z != 0 ? transformed.x / transformed.z : transformed.x;
                            double fy = transformed.z != 0 ? transformed.y / transformed.z : transformed.y;
                            if (filter == eFilterImpulse) {
                                ofxsFilterInterpolate2D<PIX, nComponents, filter, clamp>(fx, fy, _srcImg, _blackOutside, tmpPix);
                            } else {
                                bool xinside = (x1 <= fx + 0.5 && fx - 0.5 < x2);
                                bool yinside = (y1 <= fy + 0.5 && fy - 0.5 < y2);
                                if ( _blackOutside && !(xinside && yinside) ) {
                                    xinside = yinside = false;
                                }

                                double Jxx = xinside ? (H(0,0) * transformed.z - transformed.x * H(2,0)) / (transformed.z * transformed.z) : 0.;
                                double Jxy = xinside ? (H(0,1) * transformed.z - transformed.x * H(2,1)) / (transformed.z * transformed.z) : 0.;
                                double Jyx = yinside ? (H(1,0) * transformed.z - transformed.y * H(2,0)) / (transformed.z * transformed.z) : 0;
                                double Jyy = yinside ? (H(1,1) * transformed.z - transformed.y * H(2,1)) / (transformed.z * transformed.z) : 0.;
                                ofxsFilterInterpolate2DSuper<PIX, nComponents, filter, clamp>(fx, fy, Jxx, Jxy, Jyx, Jyy, _srcImg, _blackOutside, tmpPix);
                            }
                        }
                        if (!_invtransformalpha) {
                            for (int c = 0; c < nComponents; ++c) {
                                accPix[c] += tmpPix[c];
                                accPix2[c] += tmpPix[c] * tmpPix[c];
                            }
                        } else {
                            acc += _invtransformalpha[t];
                            for (int c = 0; c < nComponents; ++c) {
                                accPix[c] += tmpPix[c] * _invtransformalpha[t];
                                accPix2[c] += tmpPix[c] * tmpPix[c] * _invtransformalpha[t];
                            }
                        }
                    }
                    if (!_invtransformalpha) {
                        // compute mean and variance (unbiased)
                        for (int c = 0; c < nComponents; ++c) {
                            mean[c] = accPix[c] / sample;
                            if (sample <= 1) {
                                var[c] = (double)maxValue * maxValue;
                            } else {
                                var[c] = (accPix2[c] - mean[c] * mean[c] * sample) / (sample - 1);
                                // the variance of the mean is var[c]/n, so compute n so that it falls below some threashold (maxErr2).
                                // Note that this could be improved/optimized further by variance reduction and importance sampling
                                // http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-17-monte-carlo-methods-in-practice/variance-reduction-methods-a-quick-introduction-to-importance-sampling/
                                // http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-xx-introduction-to-importance-sampling/
                                // The threshold is computed by a simple rule of thumb:
                                // - the error should be less than motionblur*maxValue/100
                                // - the total number of iterations should be less than motionblur*100
                                if (maxsamples < maxIt) {
                                    maxsamples = std::max( maxsamples, std::min( (int)(var[c] / maxErr2), maxIt ) );
                                }
                            }
                        }
                    } else if (acc > 0.) {
                        // compute mean and variance (biased)
                        for (int c = 0; c < nComponents; ++c) {
                            mean[c] = accPix[c] / acc;
                            if (sample <= 1) {
                                var[c] = (double)maxValue * maxValue;
                            } else {
                                var[c] = accPix2[c] / acc - mean[c] * mean[c];
                                // the variance of the mean is var[c]/n, so compute n so that it falls below some threashold (maxErr2).
                                // Note that this could be improved/optimized further by variance reduction and importance sampling
                                // http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-17-monte-carlo-methods-in-practice/variance-reduction-methods-a-quick-introduction-to-importance-sampling/
                                // http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-xx-introduction-to-importance-sampling/
                                // The threshold is computed by a simple rule of thumb:
                                // - the error should be less than motionblur*maxValue/100
                                // - the total number of iterations should be less than motionblur*100
                                if (maxsamples < maxIt) {
                                    maxsamples = std::max( maxsamples, std::min( (int)(var[c] / maxErr2), maxIt ) );
                                }
                            }
                        }
                    }
                }
                for (int c = 0; c < nComponents; ++c) {
                    tmpPix[c] = (float)mean[c];
                }
                ofxsMaskMix<PIX, nComponents, maxValue, masked>(tmpPix, x, y, _srcImg, _domask, _maskImg, (float)_mix, _maskInvert, dstPix);
            }
        }
    } // multiThreadProcessImagesMotionBlur

    // Compute the /seed/th element of the van der Corput sequence
    // see http://en.wikipedia.org/wiki/Van_der_Corput_sequence
    template <int base>
    double van_der_corput(unsigned int seed)
    {
        double base_inv;
        int digit;
        double r;

        r = 0.0;

        base_inv = 1.0 / ( (double)base );

        while (seed != 0) {
            digit = seed % base;
            r = r + ( (double)digit ) * base_inv;
            base_inv = base_inv / ( (double)base );
            seed = seed / base;
        }

        return r;
    }

    unsigned int hash(unsigned int a)
    {
        a = (a ^ 61) ^ (a >> 16);
        a = a + (a << 3);
        a = a ^ (a >> 4);
        a = a * 0x27d4eb2d;
        a = a ^ (a >> 15);

        return a;
    }
};
} // namespace OFX

#endif // MISC_TRANSFORM_H
