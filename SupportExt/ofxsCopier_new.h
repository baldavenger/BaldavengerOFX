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

//
//  ofxsCopier.h
//

#ifndef IO_ofxsCopier_h
#define IO_ofxsCopier_h

#include <cstring>
#include <algorithm>

#include "ofxsPixelProcessor.h"
#include "ofxsMaskMix.h"

namespace OFX {
// Base class for the RGBA and the Alpha processor

template <class PIX, int nComponents>
class PixelCopier
    : public OFX::PixelProcessorFilterBase
{
public:
    // ctor
    PixelCopier(OFX::ImageEffect &instance)
        : OFX::PixelProcessorFilterBase(instance)
    {
    }

    // and do some processing
    void multiThreadProcessImages(OfxRectI procWindow)
    {
        assert(_dstBounds.x1 <= procWindow.x1 && procWindow.x2 <= _dstBounds.x2 && _dstBounds.y1 <= procWindow.y1 && procWindow.y2 <= _dstBounds.y2);
        // for more safety, make sure procWindow is within dstBounds (as covered by the above assert)
        if (_dstBounds.x1 > procWindow.x1) {
            procWindow.x1 = _dstBounds.x1;
        }
        if (_dstBounds.x2 < procWindow.x2) {
            procWindow.x2 = _dstBounds.x2;
        }
        if (_dstBounds.y1 > procWindow.y1) {
            procWindow.y1 = _dstBounds.y1;
        }
        if (_dstBounds.y2 < procWindow.y2) {
            procWindow.y2 = _dstBounds.y2;
        }

        int rowBytes = sizeof(PIX) * nComponents * (procWindow.x2 - procWindow.x1);

        for (int dsty = procWindow.y1; dsty < procWindow.y2; ++dsty) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) getDstPixelAddress(procWindow.x1, dsty);
            assert(dstPix);
            if (!dstPix) {
                // coverity[dead_error_line]
                continue;
            }

            int srcy = dsty;

            if (_srcBoundary == 1) {
                if (_srcBounds.y2 <= srcy) {
                    srcy = _srcBounds.y2 - 1;
                }
                if (srcy < _srcBounds.y1) {
                    srcy = _srcBounds.y1;
                }
            } else if (_srcBoundary == 2) {
                if ( (srcy < _srcBounds.y1) || (_srcBounds.y2 <= srcy) ) {
                    srcy = _srcBounds.y1 + positive_modulo(srcy - _srcBounds.y1, _srcBounds.y2 - _srcBounds.y1);
                }
            }

            if ( (srcy < _srcBounds.y1) || (_srcBounds.y2 <= srcy) || (_srcBounds.y2 <= _srcBounds.y1) ) {
                assert(_srcBoundary == 0);
                std::memset(dstPix, 0, rowBytes);
            } else {
                int x1 = std::max(_srcBounds.x1, procWindow.x1);
                int x2 = std::min(_srcBounds.x2, procWindow.x2);
                // start of line may be black
                if (procWindow.x1 < x1) {
                    if ( (_srcBoundary != 1) && (_srcBoundary != 2) ) {
                        std::memset( dstPix, 0, sizeof(PIX) * nComponents * (x1 - procWindow.x1) );
                        dstPix += nComponents * (x1 - procWindow.x1);
                    } else if (_srcBoundary == 1) {
                        const PIX *srcPix = (const PIX *) getSrcPixelAddress(x1, srcy);
                        assert(srcPix);
                        if (!srcPix) {
                            std::memset( dstPix, 0, sizeof(PIX) * nComponents * (x1 - procWindow.x1) );
                            dstPix += nComponents * (x1 - procWindow.x1);
                        } else {
#                        ifdef DEBUG
                            for (int c = 0; c < nComponents; ++c) {
                                assert(srcPix[c] == srcPix[c]); // check for NaN
                            }
#                        endif
                            for (int x = procWindow.x1; x < x1; ++x) {
                                std::copy(srcPix, srcPix + nComponents, dstPix);
                                dstPix += nComponents;
                            }
                        }
                    } else if (_srcBoundary == 2) {
                        int srcx = procWindow.x1;
                        if ( (srcx < _srcBounds.x1) || (_srcBounds.x2 <= srcx) ) {
                            srcx = _srcBounds.x1 + positive_modulo(srcx - _srcBounds.x1, _srcBounds.x2 - _srcBounds.x1);
                        }

                        const PIX *srcPix = (const PIX *) getSrcPixelAddress(srcx, srcy);
                        assert(srcPix);
                        if (!srcPix) {
                            std::memset( dstPix, 0, sizeof(PIX) * nComponents * (x1 - procWindow.x1) );
                            dstPix += nComponents * (x1 - procWindow.x1);
                        } else {
                            for (int x = procWindow.x1; x < x1; ++x) {
#                             ifdef DEBUG
                                for (int c = 0; c < nComponents; ++c) {
                                    assert(srcPix[c] == srcPix[c]); // check for NaN
                                }
#                             endif
                                std::copy(srcPix, srcPix + nComponents, dstPix);
                                dstPix += nComponents;
                                ++srcx;
                                if (_srcBounds.x2 <= srcx) {
                                    srcx -= (_srcBounds.x2 - _srcBounds.x1);
                                    srcPix -= (_srcBounds.x2 - _srcBounds.x1) * nComponents;
                                }
                            }
                        }
                    }
                }
                // then, copy the relevant fraction of src
                if ( (x1 < x2) && (procWindow.x1 <= x1) && (x2 <= procWindow.x2) ) {
                    const PIX *srcPix = (const PIX *) getSrcPixelAddress(x1, srcy);
                    assert(srcPix);
                    if (!srcPix) {
                        std::memset( dstPix, 0, sizeof(PIX) * nComponents * (x2 - x1) );
                    } else {
#                     ifdef DEBUG
                        for (int c = 0; c < nComponents * (x2 - x1); ++c) {
                            assert(srcPix[c] == srcPix[c]); // check for NaN
                        }
#                     endif
                        std::memcpy( dstPix, srcPix, sizeof(PIX) * nComponents * (x2 - x1) );
                    }
                    dstPix += nComponents * (x2 - x1);
                }
                // end of line may be black
                if (x2 < procWindow.x2) {
                    if ( (_srcBoundary != 1) && (_srcBoundary != 2) ) {
                        std::memset( dstPix, 0, sizeof(PIX) * nComponents * (procWindow.x2 - x2) );
                        dstPix += nComponents * (procWindow.x2 - x2);
                    } else if (_srcBoundary == 1) {
                        const PIX *srcPix = (const PIX *) getSrcPixelAddress(x2 - 1, srcy);
                        assert(srcPix);
                        if (!srcPix) {
                            std::memset( dstPix, 0, sizeof(PIX) * nComponents * (procWindow.x2 - x2) );
                        } else {
                            for (int x = x2; x < procWindow.x2; ++x) {
                                std::memcpy( dstPix, srcPix, sizeof(PIX) * nComponents );
                                dstPix += nComponents;
                            }
                        }
                    } else if (_srcBoundary == 2) {
                        int srcx = x2;
                        while (_srcBounds.x2 <= srcx) {
                            srcx -= (_srcBounds.x2 - _srcBounds.x1);
                        }

                        const PIX *srcPix = (const PIX *) getSrcPixelAddress(srcx, srcy);
                        assert(srcPix);
                        if (!srcPix) {
                            std::memset( dstPix, 0, sizeof(PIX) * nComponents * (procWindow.x2 - x2) );
                            dstPix += nComponents * (procWindow.x2 - x2);
                        } else {
                            for (int x = x2; x < procWindow.x2; ++x) {
#                             ifdef DEBUG
                                for (int c = 0; c < nComponents; ++c) {
                                    assert(srcPix[c] == srcPix[c]); // check for NaN
                                }
#                             endif
                                std::copy(srcPix, srcPix + nComponents, dstPix);
                                dstPix += nComponents;
                                ++srcx;
                                if (_srcBounds.x2 <= srcx) {
                                    srcx -= (_srcBounds.x2 - _srcBounds.x1);
                                    srcPix -= (_srcBounds.x2 - _srcBounds.x1) * nComponents;
                                }
                            }
                        }
                    }
                }
            }
        }
    } // multiThreadProcessImages
};

/*
 * @brief Same as PixelCopier except that the alpha channel is set to maxValue instead of being copied
 */
template <class PIX, int nComponents, int maxValue>
class PixelCopierOpaque
    : public OFX::PixelProcessorFilterBase
{
public:
    // ctor
    PixelCopierOpaque(OFX::ImageEffect &instance)
        : OFX::PixelProcessorFilterBase(instance)
    {
    }

    // and do some processing
    void multiThreadProcessImages(OfxRectI procWindow)
    {
        assert(_dstBounds.x1 <= procWindow.x1 && procWindow.x2 <= _dstBounds.x2 && _dstBounds.y1 <= procWindow.y1 && procWindow.y2 <= _dstBounds.y2);
        // for more safety, make sure procWindow is within dstBounds (as covered by the above assert)
        if (_dstBounds.x1 > procWindow.x1) {
            procWindow.x1 = _dstBounds.x1;
        }
        if (_dstBounds.x2 < procWindow.x2) {
            procWindow.x2 = _dstBounds.x2;
        }
        if (_dstBounds.y1 > procWindow.y1) {
            procWindow.y1 = _dstBounds.y1;
        }
        if (_dstBounds.y2 < procWindow.y2) {
            procWindow.y2 = _dstBounds.y2;
        }
        assert(nComponents == 4 || nComponents == 1);
        for (int dsty = procWindow.y1; dsty < procWindow.y2; ++dsty) {
            if ( _effect.abort() ) {
                break;
            }

            int srcy = dsty;

            if (_srcBoundary == 1) {
                if (_srcBounds.y2 <= srcy) {
                    srcy = _srcBounds.y2 - 1;
                }
                if (srcy < _srcBounds.y1) {
                    srcy = _srcBounds.y1;
                }
            } else if (_srcBoundary == 2) {
                if ( (srcy < _srcBounds.y1) || (_srcBounds.y2 <= srcy) ) {
                    srcy = _srcBounds.y1 + positive_modulo(srcy - _srcBounds.y1, _srcBounds.y2 - _srcBounds.y1);
                }
            }

            PIX *dstPix = (PIX *) getDstPixelAddress(procWindow.x1, dsty);
            assert(dstPix);
            if (!dstPix) {
                // coverity[dead_error_line]
                continue;
            }

            for (int dstx = procWindow.x1; dstx < procWindow.x2; ++dstx) {
                int srcx = dstx;

                if (_srcBoundary == 1) {
                    if (_srcBounds.x2 <= srcx) {
                        srcx = _srcBounds.x2 - 1;
                    }
                    if (srcx < _srcBounds.x1) {
                        srcx = _srcBounds.x1;
                    }
                } else if (_srcBoundary == 2) {
                    if ( (srcx < _srcBounds.x1) || (_srcBounds.x2 <= srcx) ) {
                        srcx = _srcBounds.x1 + positive_modulo(srcx - _srcBounds.x1, _srcBounds.x2 - _srcBounds.x1);
                    }
                }

                // origPix is at dstx,dsty
                const PIX *srcPix = (const PIX *) getSrcPixelAddress(srcx, srcy);
                if (srcPix) {
                    std::copy(srcPix, srcPix + nComponents - 1, dstPix);
                    dstPix[nComponents - 1] = maxValue;
                } else {
                    std::fill(dstPix, dstPix + nComponents, 0); // no src pixel here, be black and transparent
                }
                // increment the dst pixel
                dstPix += nComponents;
            }
        }
    } // multiThreadProcessImages
};


template <class PIX, int nComponents, int maxValue, bool masked>
class PixelCopierMaskMix
    : public OFX::PixelProcessorFilterBase
{
public:
    // ctor
    PixelCopierMaskMix(OFX::ImageEffect &instance)
        : OFX::PixelProcessorFilterBase(instance)
    {
    }

    // and do some processing
    void multiThreadProcessImages(OfxRectI procWindow)
    {
        assert(_dstBounds.x1 <= procWindow.x1 && procWindow.x2 <= _dstBounds.x2 && _dstBounds.y1 <= procWindow.y1 && procWindow.y2 <= _dstBounds.y2);
        // for more safety, make sure procWindow is within dstBounds (as covered by the above assert)
        if (_dstBounds.x1 > procWindow.x1) {
            procWindow.x1 = _dstBounds.x1;
        }
        if (_dstBounds.x2 < procWindow.x2) {
            procWindow.x2 = _dstBounds.x2;
        }
        if (_dstBounds.y1 > procWindow.y1) {
            procWindow.y1 = _dstBounds.y1;
        }
        if (_dstBounds.y2 < procWindow.y2) {
            procWindow.y2 = _dstBounds.y2;
        }
        float tmpPix[nComponents];

        for (int dsty = procWindow.y1; dsty < procWindow.y2; ++dsty) {
            if ( _effect.abort() ) {
                break;
            }

            int srcy = dsty;

            if (_srcBoundary == 1) {
                if (_srcBounds.y2 <= srcy) {
                    srcy = _srcBounds.y2 - 1;
                }
                if (srcy < _srcBounds.y1) {
                    srcy = _srcBounds.y1;
                }
            } else if (_srcBoundary == 2) {
                if ( (srcy < _srcBounds.y1) || (_srcBounds.y2 <= srcy) ) {
                    srcy = _srcBounds.y1 + positive_modulo(srcy - _srcBounds.y1, _srcBounds.y2 - _srcBounds.y1);
                }
            }

            PIX *dstPix = (PIX *) getDstPixelAddress(procWindow.x1, dsty);
            assert(dstPix);
            if (!dstPix) {
                // coverity[dead_error_line]
                continue;
            }

            for (int dstx = procWindow.x1; dstx < procWindow.x2; ++dstx) {
                int srcx = dstx;

                if (_srcBoundary == 1) {
                    if (_srcBounds.x2 <= srcx) {
                        srcx = _srcBounds.x2 - 1;
                    }
                    if (srcx < _srcBounds.x1) {
                        srcx = _srcBounds.x1;
                    }
                } else if (_srcBoundary == 2) {
                    if ( (srcx < _srcBounds.x1) || (_srcBounds.x2 <= srcx) ) {
                        srcx = _srcBounds.x1 + positive_modulo(srcx - _srcBounds.x1, _srcBounds.x2 - _srcBounds.x1);
                    }
                }

                // origPix is at dstx,dsty
                const PIX *origPix = (const PIX *)  (_origImg ? _origImg->getPixelAddress(dstx, dsty) : 0);
                const PIX *srcPix = (const PIX *) getSrcPixelAddress(srcx, srcy);
                if (srcPix) {
                    std::copy(srcPix, srcPix + nComponents, tmpPix);
                } else {
                    std::fill(tmpPix, tmpPix + nComponents, 0.); // no src pixel here, be black and transparent
                }
                // dstx,dsty are the mask image coordinates (no boundary conditions)
                ofxsMaskMixPix<PIX, nComponents, maxValue, masked>(tmpPix, dstx, dsty, origPix, _doMasking, _maskImg, (float)_mix, _maskInvert, dstPix);
                // increment the dst pixel
                dstPix += nComponents;
            }
        }
    } // multiThreadProcessImages
};

template <class SRCPIX, int srcNComponents, int srcMaxValue, class DSTPIX, int dstNComponents, int dstMaxValue>
class PixelCopierUnPremult
    : public OFX::PixelProcessorFilterBase
{
public:
    // ctor
    PixelCopierUnPremult(OFX::ImageEffect &instance)
        : OFX::PixelProcessorFilterBase(instance)
    {
        assert( (srcNComponents == 3 || srcNComponents == 4) && (dstNComponents == 3 || dstNComponents == 4) );
    }

    // and do some processing
    void multiThreadProcessImages(OfxRectI procWindow)
    {
        assert(_dstBounds.x1 <= procWindow.x1 && procWindow.x2 <= _dstBounds.x2 && _dstBounds.y1 <= procWindow.y1 && procWindow.y2 <= _dstBounds.y2);
        // for more safety, make sure procWindow is within dstBounds (as covered by the above assert)
        if (_dstBounds.x1 > procWindow.x1) {
            procWindow.x1 = _dstBounds.x1;
        }
        if (_dstBounds.x2 < procWindow.x2) {
            procWindow.x2 = _dstBounds.x2;
        }
        if (_dstBounds.y1 > procWindow.y1) {
            procWindow.y1 = _dstBounds.y1;
        }
        if (_dstBounds.y2 < procWindow.y2) {
            procWindow.y2 = _dstBounds.y2;
        }
        float unpPix[4];

        for (int dsty = procWindow.y1; dsty < procWindow.y2; ++dsty) {
            if ( _effect.abort() ) {
                break;
            }

            int srcy = dsty;

            if (_srcBoundary == 1) {
                if (_srcBounds.y2 <= srcy) {
                    srcy = _srcBounds.y2 - 1;
                }
                if (srcy < _srcBounds.y1) {
                    srcy = _srcBounds.y1;
                }
            } else if (_srcBoundary == 2) {
                if ( (srcy < _srcBounds.y1) || (_srcBounds.y2 <= srcy) ) {
                    srcy = _srcBounds.y1 + positive_modulo(srcy - _srcBounds.y1, _srcBounds.y2 - _srcBounds.y1);
                }
            }

            DSTPIX *dstPix = (DSTPIX *) getDstPixelAddress(procWindow.x1, dsty);
            assert(dstPix);
            if (!dstPix) {
                // coverity[dead_error_line]
                continue;
            }
            
            for (int dstx = procWindow.x1; dstx < procWindow.x2; ++dstx) {
                int srcx = dstx;

                if (_srcBoundary == 1) {
                    if (_srcBounds.x2 <= srcx) {
                        srcx = _srcBounds.x2 - 1;
                    }
                    if (srcx < _srcBounds.x1) {
                        srcx = _srcBounds.x1;
                    }
                } else if (_srcBoundary == 2) {
                    if ( (srcx < _srcBounds.x1) || (_srcBounds.x2 <= srcx) ) {
                        srcx = _srcBounds.x1 + positive_modulo(srcx - _srcBounds.x1, _srcBounds.x2 - _srcBounds.x1);
                    }
                }

                const SRCPIX *srcPix = (const SRCPIX *) getSrcPixelAddress(srcx, srcy);
                ofxsUnPremult<SRCPIX, srcNComponents, srcMaxValue>(srcPix, unpPix, _premult, _premultChannel);
                for (int c = 0; c < dstNComponents; ++c) {
                    float v = unpPix[c] * dstMaxValue;
                    dstPix[c] = ofxsClampIfInt<DSTPIX, dstMaxValue>(v, 0, dstMaxValue);
                }
                // increment the dst pixel
                dstPix += dstNComponents;
            }
        }
    } // multiThreadProcessImages
};

template <class SRCPIX, int srcNComponents, int srcMaxValue, class DSTPIX, int dstNComponents, int dstMaxValue>
class PixelCopierPremult
    : public OFX::PixelProcessorFilterBase
{
public:
    // ctor
    PixelCopierPremult(OFX::ImageEffect &instance)
        : OFX::PixelProcessorFilterBase(instance)
    {
        assert(srcMaxValue);
        assert( (srcNComponents == 3 || srcNComponents == 4) && (dstNComponents == 3 || dstNComponents == 4) );
    }

    // and do some processing
    void multiThreadProcessImages(OfxRectI procWindow)
    {
        assert(_dstBounds.x1 <= procWindow.x1 && procWindow.x2 <= _dstBounds.x2 && _dstBounds.y1 <= procWindow.y1 && procWindow.y2 <= _dstBounds.y2);
        // for more safety, make sure procWindow is within dstBounds (as covered by the above assert)
        if (_dstBounds.x1 > procWindow.x1) {
            procWindow.x1 = _dstBounds.x1;
        }
        if (_dstBounds.x2 < procWindow.x2) {
            procWindow.x2 = _dstBounds.x2;
        }
        if (_dstBounds.y1 > procWindow.y1) {
            procWindow.y1 = _dstBounds.y1;
        }
        if (_dstBounds.y2 < procWindow.y2) {
            procWindow.y2 = _dstBounds.y2;
        }
        for (int dsty = procWindow.y1; dsty < procWindow.y2; ++dsty) {
            if ( _effect.abort() ) {
                break;
            }

            int srcy = dsty;

            if (_srcBoundary == 1) {
                if (_srcBounds.y2 <= srcy) {
                    srcy = _srcBounds.y2 - 1;
                }
                if (srcy < _srcBounds.y1) {
                    srcy = _srcBounds.y1;
                }
            } else if (_srcBoundary == 2) {
                if ( (srcy < _srcBounds.y1) || (_srcBounds.y2 <= srcy) ) {
                    srcy = _srcBounds.y1 + positive_modulo(srcy - _srcBounds.y1, _srcBounds.y2 - _srcBounds.y1);
                }
            }

            DSTPIX *dstPix = (DSTPIX *) getDstPixelAddress(procWindow.x1, dsty);
            assert(dstPix);
            if (!dstPix) {
                // coverity[dead_error_line]
                continue;
            }

            for (int dstx = procWindow.x1; dstx < procWindow.x2; ++dstx) {
                int srcx = dstx;

                if (_srcBoundary == 1) {
                    if (_srcBounds.x2 <= srcx) {
                        srcx = _srcBounds.x2 - 1;
                    }
                    if (srcx < _srcBounds.x1) {
                        srcx = _srcBounds.x1;
                    }
                } else if (_srcBoundary == 2) {
                    if ( (srcx < _srcBounds.x1) || (_srcBounds.x2 <= srcx) ) {
                        srcx = _srcBounds.x1 + positive_modulo(srcx - _srcBounds.x1, _srcBounds.x2 - _srcBounds.x1);
                    }
                }

                const SRCPIX *srcPix = (const SRCPIX *) getSrcPixelAddress(srcx, srcy);
                if (!srcPix) {
                    // no source, be black and transparent
                    for (int c = 0; c < dstNComponents; ++c) {
                        dstPix[c] = DSTPIX();
                    }
                } else {
                    float unpPix[4];
                    if (srcNComponents == 1) {
                        unpPix[0] = 0.f;
                        unpPix[1] = 0.f;
                        unpPix[2] = 0.f;
                        unpPix[3] = srcPix[0] / (float)srcMaxValue;
                    } else {
                        unpPix[0] = srcPix[0] / (float)srcMaxValue;
                        unpPix[1] = srcPix[1] / (float)srcMaxValue;
                        unpPix[2] = srcPix[2] / (float)srcMaxValue;
                        unpPix[3] = (srcNComponents == 4) ? (srcPix[3] / (float)srcMaxValue) : 1.0f;
                    }
                    float pPix[dstNComponents];
                    // unpPix is in [0, 1]
                    // premultiply and denormalize in [0, maxValue]
                    // if premult is false, just denormalize
                    ofxsPremult<DSTPIX, dstNComponents, dstMaxValue>(unpPix, pPix, _premult, _premultChannel);
                    for (int c = 0; c < dstNComponents; ++c) {
                        dstPix[c] = ofxsClampIfInt<DSTPIX, dstMaxValue>(pPix[c], 0, dstMaxValue);
                    }
                }
                // increment the dst pixel
                dstPix += dstNComponents;
            }
        }
    } // multiThreadProcessImages
};

// _srcBoundarys The border condition type { 0=zero |  1=dirichlet | 2=periodic }.
// template to do the RGBA processing
template <class SRCPIX, int srcNComponents, int srcMaxValue, class DSTPIX, int dstNComponents, int dstMaxValue>
class PixelCopierPremultMaskMix
    : public OFX::PixelProcessorFilterBase
{
public:
    // ctor
    PixelCopierPremultMaskMix(OFX::ImageEffect &instance)
        : OFX::PixelProcessorFilterBase(instance)
    {
        assert( (srcNComponents == 3 || srcNComponents == 4) && (dstNComponents == 3 || dstNComponents == 4) );
    }

    // and do some processing
    void multiThreadProcessImages(OfxRectI procWindow)
    {
        assert(_dstBounds.x1 <= procWindow.x1 && procWindow.x2 <= _dstBounds.x2 && _dstBounds.y1 <= procWindow.y1 && procWindow.y2 <= _dstBounds.y2);
        // for more safety, make sure procWindow is within dstBounds (as covered by the above assert)
        if (_dstBounds.x1 > procWindow.x1) {
            procWindow.x1 = _dstBounds.x1;
        }
        if (_dstBounds.x2 < procWindow.x2) {
            procWindow.x2 = _dstBounds.x2;
        }
        if (_dstBounds.y1 > procWindow.y1) {
            procWindow.y1 = _dstBounds.y1;
        }
        if (_dstBounds.y2 < procWindow.y2) {
            procWindow.y2 = _dstBounds.y2;
        }
        float unpPix[4];

        if (srcNComponents == 3) {
            unpPix[3] = 1.f;
        }
        for (int dsty = procWindow.y1; dsty < procWindow.y2; ++dsty) {
            if ( _effect.abort() ) {
                break;
            }

            int srcy = dsty;

            if (_srcBoundary == 1) {
                if (_srcBounds.y2 <= srcy) {
                    srcy = _srcBounds.y2 - 1;
                }
                if (srcy < _srcBounds.y1) {
                    srcy = _srcBounds.y1;
                }
            } else if (_srcBoundary == 2) {
                if ( (srcy < _srcBounds.y1) || (_srcBounds.y2 <= srcy) ) {
                    srcy = _srcBounds.y1 + positive_modulo(srcy - _srcBounds.y1, _srcBounds.y2 - _srcBounds.y1);
                }
            }

            DSTPIX *dstPix = (DSTPIX *) getDstPixelAddress(procWindow.x1, dsty);
            assert(dstPix);
            if (!dstPix) {
                // coverity[dead_error_line]
                continue;
            }

            for (int dstx = procWindow.x1; dstx < procWindow.x2; ++dstx) {
                int srcx = dstx;

                if (_srcBoundary == 1) {
                    if (_srcBounds.x2 <= srcx) {
                        srcx = _srcBounds.x2 - 1;
                    }
                    if (srcx < _srcBounds.x1) {
                        srcx = _srcBounds.x1;
                    }
                } else if (_srcBoundary == 2) {
                    if ( (srcx < _srcBounds.x1) || (_srcBounds.x2 <= srcx) ) {
                        srcx = _srcBounds.x1 + positive_modulo(srcx - _srcBounds.x1, _srcBounds.x2 - _srcBounds.x1);
                    }
                }
                // origPix is at dstx,dsty
                const DSTPIX *origPix = (const DSTPIX *)  (_origImg ? _origImg->getPixelAddress(dstx, dsty) : 0);
                const SRCPIX *srcPix = (const SRCPIX *) getSrcPixelAddress(srcx, srcy);
                for (int c = 0; c < srcNComponents; ++c) {
                    unpPix[c] = (srcPix ? (srcPix[c] / (float)srcMaxValue) : 0.f);
                }
                // dstx,dsty are the mask image coordinates (no boundary conditions)
                ofxsPremultMaskMixPix<DSTPIX, dstNComponents, dstMaxValue, true>(unpPix, _premult, _premultChannel, dstx, dsty, origPix, _doMasking, _maskImg, (float)_mix, _maskInvert, dstPix);
                // increment the dst pixel
                dstPix += dstNComponents;
            }
        }
    } // multiThreadProcessImages
};

template <class PIX>
class BlackFiller
    : public OFX::PixelProcessorFilterBase
{
public:
    // ctor
    BlackFiller(OFX::ImageEffect &instance,
                int comps)
        : OFX::PixelProcessorFilterBase(instance)
        , _nComponents(comps)
    {
    }

    // and do some processing
    void multiThreadProcessImages(OfxRectI procWindow)
    {
        assert(_dstBounds.x1 <= procWindow.x1 && procWindow.x2 <= _dstBounds.x2 && _dstBounds.y1 <= procWindow.y1 && procWindow.y2 <= _dstBounds.y2);
        // for more safety, make sure procWindow is within dstBounds (as covered by the above assert)
        if (_dstBounds.x1 > procWindow.x1) {
            procWindow.x1 = _dstBounds.x1;
        }
        if (_dstBounds.x2 < procWindow.x2) {
            procWindow.x2 = _dstBounds.x2;
        }
        if (_dstBounds.y1 > procWindow.y1) {
            procWindow.y1 = _dstBounds.y1;
        }
        if (_dstBounds.y2 < procWindow.y2) {
            procWindow.y2 = _dstBounds.y2;
        }
        int rowSize =  _nComponents * (procWindow.x2 - procWindow.x1);

        for (int y = procWindow.y1; y < procWindow.y2; ++y) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) getDstPixelAddress(procWindow.x1, y);
            assert(dstPix);
            if (!dstPix) {
                // coverity[dead_error_line]
                continue;
            }
            std::fill( dstPix, dstPix + rowSize, PIX() );
        }
    }

private:
    int _nComponents;
};

// black fillers, non-threaded versions
template<class PIX>
void
fillBlackNTForDepth(const OfxRectI & renderWindow,
                    void *dstPixelData,
                    const OfxRectI & dstBounds,
                    int dstPixelComponentCount,
                    int dstRowBytes)
{
    assert(dstPixelData);
    // do the rendering
    int dstRowElements = dstRowBytes / sizeof(PIX);
    int x1 = std::max(renderWindow.x1, dstBounds.x1);
    int x2 = std::min(renderWindow.x2, dstBounds.x2);
    int y1 = std::max(renderWindow.y1, dstBounds.y1);
    int y2 = std::min(renderWindow.y2, dstBounds.y2);
    PIX* dstPixels = (PIX*)dstPixelData + (size_t)(y1 - dstBounds.y1) * dstRowElements + (x1 - dstBounds.x1) * dstPixelComponentCount;
    int rowElements = dstPixelComponentCount * (x2 - renderWindow.x1);

    for (int y = y1; y < y2; ++y, dstPixels += dstRowElements) {
        std::fill( dstPixels, dstPixels + rowElements, PIX() ); // no src pixel here, be black and transparent
    }
}

inline void
fillBlackNT(const OfxRectI & renderWindow,
            void *dstPixelData,
            const OfxRectI & dstBounds,
            int dstPixelComponentCount,
            OFX::BitDepthEnum dstBitDepth,
            int dstRowBytes)
{
    assert(dstPixelData);
    if (!dstPixelData) {
        // coverity[dead_error_line]
        return;
    }
    // do the rendering
    if ( (dstBitDepth != OFX::eBitDepthUByte) && (dstBitDepth != OFX::eBitDepthUShort) && (dstBitDepth != OFX::eBitDepthHalf) && (dstBitDepth != OFX::eBitDepthFloat) ) {
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    if (dstBitDepth == OFX::eBitDepthUByte) {
        fillBlackNTForDepth<unsigned char>(renderWindow,
                                           dstPixelData, dstBounds, dstPixelComponentCount, dstRowBytes);
    } else if ( (dstBitDepth == OFX::eBitDepthUShort) || (dstBitDepth == OFX::eBitDepthHalf) ) {
        fillBlackNTForDepth<unsigned short>(renderWindow,
                                            dstPixelData, dstBounds, dstPixelComponentCount, dstRowBytes);
    } else if (dstBitDepth == OFX::eBitDepthFloat) {
        fillBlackNTForDepth<float>(renderWindow,
                                   dstPixelData, dstBounds, dstPixelComponentCount, dstRowBytes);
    } // switch
}

inline void
fillBlackNT(const OfxRectI & renderWindow,
            OFX::Image* dstImg)
{
    void* dstPixelData;
    OfxRectI dstBounds;
    OFX::PixelComponentEnum dstPixelComponents;
    OFX::BitDepthEnum dstBitDepth;
    int dstRowBytes;

    assert(dstImg);
    if (!dstImg) {
        // coverity[dead_error_line]
        return;
    }
    getImageData(dstImg, &dstPixelData, &dstBounds, &dstPixelComponents, &dstBitDepth, &dstRowBytes);
    int dstPixelComponentCount = dstImg->getPixelComponentCount();

    return fillBlackNT(renderWindow, dstPixelData, dstBounds, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

#if 0 // Don't use threaded version: very probably less efficient

// black fillers, threaded versions
template<class PIX, int nComponents>
void
fillBlackForDepthAndComponents(OFX::ImageEffect &instance,
                               const OfxRectI & renderWindow,
                               PIX *dstPixelData,
                               const OfxRectI & dstBounds,
                               OFX::PixelComponentEnum dstPixelComponents,
                               int dstPixelComponentCount,
                               OFX::BitDepthEnum dstBitDepth,
                               int dstRowBytes)
{
    (void)dstPixelComponents;
    (void)dstBitDepth;

    OFX::BlackFiller<PIX> processor(instance, nComponents);
    // set the images
    processor.setDstImg(dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);

    // set the render window
    processor.setRenderWindow(renderWindow);

    // Call the base class process member, this will call the derived templated process code
    processor.process();
}

template<class PIX>
void
fillBlackForDepth(OFX::ImageEffect &instance,
                  const OfxRectI & renderWindow,
                  void *dstPixelData,
                  const OfxRectI & dstBounds,
                  OFX::PixelComponentEnum dstPixelComponents,
                  int dstPixelComponentCount,
                  OFX::BitDepthEnum dstBitDepth,
                  int dstRowBytes)
{
    assert(dstPixelData);
    if (!dstPixelData) {
        // coverity[dead_error_line]
        return;
    }
    // do the rendering
    if ( (dstPixelComponentCount < 0) || (4 < dstPixelComponentCount) ) {
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    if (dstPixelComponentCount == 4) {
        fillBlackForDepthAndComponents<PIX, 4>(instance, renderWindow,
                                               (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if (dstPixelComponentCount == 3) {
        fillBlackForDepthAndComponents<PIX, 3>(instance, renderWindow,
                                               (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if (dstPixelComponentCount == 2) {
        fillBlackForDepthAndComponents<PIX, 2>(instance, renderWindow,
                                               (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    }  else if (dstPixelComponentCount == 1) {
        fillBlackForDepthAndComponents<PIX, 1>(instance, renderWindow,
                                               (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } // switch
}

inline void
fillBlack(OFX::ImageEffect &instance,
          const OfxRectI & renderWindow,
          void *dstPixelData,
          const OfxRectI & dstBounds,
          OFX::PixelComponentEnum dstPixelComponents,
          int dstPixelComponentCount,
          OFX::BitDepthEnum dstBitDepth,
          int dstRowBytes)
{
    assert(dstPixelData);
    if (!dstPixelData) {
        // coverity[dead_error_line]
        return;
    }
    // do the rendering
    if ( (dstBitDepth != OFX::eBitDepthUByte) && (dstBitDepth != OFX::eBitDepthUShort) && (dstBitDepth != OFX::eBitDepthHalf) && (dstBitDepth != OFX::eBitDepthFloat) ) {
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    if (dstBitDepth == OFX::eBitDepthUByte) {
        fillBlackForDepth<unsigned char>(instance, renderWindow,
                                         dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if ( (dstBitDepth == OFX::eBitDepthUShort) || (dstBitDepth == OFX::eBitDepthHalf) ) {
        fillBlackForDepth<unsigned short>(instance, renderWindow,
                                          dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if (dstBitDepth == OFX::eBitDepthFloat) {
        fillBlackForDepth<float>(instance, renderWindow,
                                 dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } // switch
}

inline void
fillBlack(OFX::ImageEffect &instance,
          const OfxRectI & renderWindow,
          OFX::Image* dstImg)
{
    void* dstPixelData;
    OfxRectI dstBounds;
    OFX::PixelComponentEnum dstPixelComponents;
    OFX::BitDepthEnum dstBitDepth;
    int dstRowBytes;

    assert(dstImg);
    if (!dstImg) {
        // coverity[dead_error_line]
        return;
    }
    getImageData(dstImg, &dstPixelData, &dstBounds, &dstPixelComponents, &dstBitDepth, &dstRowBytes);
    int dstPixelComponentCount = dstImg->getPixelComponentCount();

    return fillBlack(instance, renderWindow, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

#else // if 0

// Use non-threaded version: probably more efficient

inline void
fillBlack(OFX::ImageEffect &instance,
          const OfxRectI & renderWindow,
          void *dstPixelData,
          const OfxRectI & dstBounds,
          OFX::PixelComponentEnum dstPixelComponents,
          int dstPixelComponentCount,
          OFX::BitDepthEnum dstBitDepth,
          int dstRowBytes)
{
    (void)instance;
    (void)dstPixelComponents;

    return fillBlackNT(renderWindow, dstPixelData, dstBounds, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

inline void
fillBlack(OFX::ImageEffect &instance,
          const OfxRectI & renderWindow,
          OFX::Image* dstImg)
{
    (void)instance;

    return fillBlackNT(renderWindow, dstImg);
}

#endif // if 0

// pixel copiers, non-threaded versions
template<class PIX, int nComponents>
void
copyPixelsNTForDepthAndComponents(OFX::ImageEffect &instance,
                                  const OfxRectI & renderWindow,
                                  const PIX *srcPixelData,
                                  const OfxRectI & srcBounds,
                                  OFX::PixelComponentEnum srcPixelComponents,
                                  int srcPixelComponentCount,
                                  OFX::BitDepthEnum srcBitDepth,
                                  int srcRowBytes,
                                  PIX *dstPixelData,
                                  const OfxRectI & dstBounds,
                                  OFX::PixelComponentEnum dstPixelComponents,
                                  int dstPixelComponentCount,
                                  OFX::BitDepthEnum dstBitDepth,
                                  int dstRowBytes)
{
    assert(srcBitDepth == dstBitDepth);
    assert(srcPixelComponentCount == dstPixelComponentCount);
    assert(srcPixelComponentCount == nComponents);
    (void)srcPixelComponents;
    (void)srcPixelComponentCount;
    (void)srcBitDepth;
    (void)dstPixelComponents;
    (void)dstPixelComponentCount;
    (void)dstBitDepth;
    (void)instance;

    int srcRowElements = srcRowBytes / sizeof(PIX);
    assert(srcBounds.y1 <= renderWindow.y1 && renderWindow.y1 <= renderWindow.y2 && renderWindow.y2 <= srcBounds.y2);
    assert(srcBounds.x1 <= renderWindow.x1 && renderWindow.x1 <= renderWindow.x2 && renderWindow.x2 <= srcBounds.x2);
    int x1 = std::max( renderWindow.x1, std::max(dstBounds.x1, srcBounds.x1) );
    int x2 = std::min( renderWindow.x2, std::min(dstBounds.x2, srcBounds.x2) );
    int y1 = std::max( renderWindow.y1, std::max(dstBounds.y1, srcBounds.y1) );
    int y2 = std::min( renderWindow.y2, std::min(dstBounds.y2, srcBounds.y2) );
    const PIX* srcPixels = srcPixelData + (size_t)(y1 - srcBounds.y1) * srcRowElements + (x1 - srcBounds.x1) * nComponents;
    int dstRowElements = dstRowBytes / sizeof(PIX);
    PIX* dstPixels = dstPixelData + (size_t)(y1 - dstBounds.y1) * dstRowElements + (x1 - dstBounds.x1) * nComponents;
    int rowBytes = sizeof(PIX) * nComponents * (x2 - x1);

    for (int y = y1; y < y2; ++y, srcPixels += srcRowElements, dstPixels += dstRowElements) {
        std::memcpy(dstPixels, srcPixels, rowBytes);
    }
}

template<class PIX>
void
copyPixelsNTForDepth(OFX::ImageEffect &instance,
                     const OfxRectI & renderWindow,
                     const void *srcPixelData,
                     const OfxRectI & srcBounds,
                     OFX::PixelComponentEnum srcPixelComponents,
                     int srcPixelComponentCount,
                     OFX::BitDepthEnum srcBitDepth,
                     int srcRowBytes,
                     void *dstPixelData,
                     const OfxRectI & dstBounds,
                     OFX::PixelComponentEnum dstPixelComponents,
                     int dstPixelComponentCount,
                     OFX::BitDepthEnum dstBitDepth,
                     int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);
    assert(srcBitDepth == dstBitDepth);
    assert(srcPixelComponentCount == dstPixelComponentCount);
    // do the rendering
    if ( (dstPixelComponents != OFX::ePixelComponentRGBA) && (dstPixelComponents != OFX::ePixelComponentRGB) && (dstPixelComponents != OFX::ePixelComponentAlpha) ) {
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    if (dstPixelComponents == OFX::ePixelComponentRGBA) {
        copyPixelsNTForDepthAndComponents<PIX, 4>(instance, renderWindow,
                                                  (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                  (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if (dstPixelComponents == OFX::ePixelComponentRGB) {
        copyPixelsNTForDepthAndComponents<PIX, 3>(instance, renderWindow,
                                                  (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                  (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    }  else if (dstPixelComponents == OFX::ePixelComponentAlpha) {
        copyPixelsNTForDepthAndComponents<PIX, 1>(instance, renderWindow,
                                                  (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                  (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } // switch
}

inline void
copyPixelsNT(OFX::ImageEffect &instance,
             const OfxRectI & renderWindow,
             const void *srcPixelData,
             const OfxRectI & srcBounds,
             OFX::PixelComponentEnum srcPixelComponents,
             int srcPixelComponentCount,
             OFX::BitDepthEnum srcBitDepth,
             int srcRowBytes,
             void *dstPixelData,
             const OfxRectI & dstBounds,
             OFX::PixelComponentEnum dstPixelComponents,
             int dstPixelComponentCount,
             OFX::BitDepthEnum dstBitDepth,
             int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);
    assert(srcBitDepth == dstBitDepth);
    assert(srcPixelComponentCount == dstPixelComponentCount);

    // do the rendering
    if ( (dstBitDepth != OFX::eBitDepthUByte) && (dstBitDepth != OFX::eBitDepthUShort) && (dstBitDepth != OFX::eBitDepthHalf) && (dstBitDepth != OFX::eBitDepthFloat) ) {
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    if (dstBitDepth == OFX::eBitDepthUByte) {
        copyPixelsNTForDepth<unsigned char>(instance, renderWindow,
                                            srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                            dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if ( (dstBitDepth == OFX::eBitDepthUShort) || (dstBitDepth == OFX::eBitDepthHalf) ) {
        copyPixelsNTForDepth<unsigned short>(instance, renderWindow,
                                             srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                             dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if (dstBitDepth == OFX::eBitDepthFloat) {
        copyPixelsNTForDepth<float>(instance, renderWindow,
                                    srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                    dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } // switch
}

// pixel copiers, threaded versions
template<class PIX, int nComponents>
void
copyPixelsForDepthAndComponents(OFX::ImageEffect &instance,
                                const OfxRectI & renderWindow,
                                const PIX *srcPixelData,
                                const OfxRectI & srcBounds,
                                OFX::PixelComponentEnum srcPixelComponents,
                                int srcPixelComponentCount,
                                OFX::BitDepthEnum srcBitDepth,
                                int srcRowBytes,
                                PIX *dstPixelData,
                                const OfxRectI & dstBounds,
                                OFX::PixelComponentEnum dstPixelComponents,
                                int dstPixelComponentCount,
                                OFX::BitDepthEnum dstBitDepth,
                                int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);
    //assert(srcBounds.y1 <= renderWindow.y1 && renderWindow.y1 <= renderWindow.y2 && renderWindow.y2 <= srcBounds.y2); // not necessary, PixelCopier should handle this
    //assert(srcBounds.x1 <= renderWindow.x1 && renderWindow.x1 <= renderWindow.x2 && renderWindow.x2 <= srcBounds.x2); // not necessary, PixelCopier should handle this
    assert(srcBitDepth == dstBitDepth);
    assert(srcPixelComponentCount == dstPixelComponentCount);
    (void)srcPixelComponents;
    (void)srcBitDepth;
    (void)dstPixelComponents;
    (void)dstBitDepth;

    OFX::PixelCopier<PIX, nComponents> processor(instance);
    // set the images
    processor.setDstImg(dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    processor.setSrcImg(srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes, 0);

    // set the render window
    processor.setRenderWindow(renderWindow);

    // Call the base class process member, this will call the derived templated process code
    processor.process();
}

template<class PIX>
void
copyPixelsForDepth(OFX::ImageEffect &instance,
                   const OfxRectI & renderWindow,
                   const void *srcPixelData,
                   const OfxRectI & srcBounds,
                   OFX::PixelComponentEnum srcPixelComponents,
                   int srcPixelComponentCount,
                   OFX::BitDepthEnum srcBitDepth,
                   int srcRowBytes,
                   void *dstPixelData,
                   const OfxRectI & dstBounds,
                   OFX::PixelComponentEnum dstPixelComponents,
                   int dstPixelComponentCount,
                   OFX::BitDepthEnum dstBitDepth,
                   int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);
    assert(srcBitDepth == dstBitDepth);
    assert(srcPixelComponentCount == dstPixelComponentCount);
    // do the rendering
    if ( (dstPixelComponentCount < 0) || (4 < dstPixelComponentCount) ) {
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    if (dstPixelComponentCount == 4) {
        copyPixelsForDepthAndComponents<PIX, 4>(instance, renderWindow,
                                                (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if (dstPixelComponentCount == 3) {
        copyPixelsForDepthAndComponents<PIX, 3>(instance, renderWindow,
                                                (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if (dstPixelComponentCount == 2) {
        copyPixelsForDepthAndComponents<PIX, 2>(instance, renderWindow,
                                                (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    }  else if (dstPixelComponentCount == 1) {
        copyPixelsForDepthAndComponents<PIX, 1>(instance, renderWindow,
                                                (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } // switch
}

inline void
copyPixels(OFX::ImageEffect &instance,
           const OfxRectI & renderWindow,
           const void *srcPixelData,
           const OfxRectI & srcBounds,
           OFX::PixelComponentEnum srcPixelComponents,
           int srcPixelComponentCount,
           OFX::BitDepthEnum srcBitDepth,
           int srcRowBytes,
           void *dstPixelData,
           const OfxRectI & dstBounds,
           OFX::PixelComponentEnum dstPixelComponents,
           int dstPixelComponentCount,
           OFX::BitDepthEnum dstBitDepth,
           int dstRowBytes)
{
    assert(dstPixelData);
    if (!srcPixelData) {
        // no input, be black and transparent
        return fillBlack(instance, renderWindow,
                         dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    }
    assert(srcPixelComponentCount == dstPixelComponentCount && srcBitDepth == dstBitDepth);
    // do the rendering
    if ( (dstBitDepth != OFX::eBitDepthUByte) && (dstBitDepth != OFX::eBitDepthUShort) && (dstBitDepth != OFX::eBitDepthHalf) && (dstBitDepth != OFX::eBitDepthFloat) ) {
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    if (dstBitDepth == OFX::eBitDepthUByte) {
        copyPixelsForDepth<unsigned char>(instance, renderWindow,
                                          srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                          dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if ( (dstBitDepth == OFX::eBitDepthUShort) || (dstBitDepth == OFX::eBitDepthHalf) ) {
        copyPixelsForDepth<unsigned short>(instance, renderWindow,
                                           srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                           dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } else if (dstBitDepth == OFX::eBitDepthFloat) {
        copyPixelsForDepth<float>(instance, renderWindow,
                                  srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                  dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    } // switch
}

inline void
copyPixels(OFX::ImageEffect &instance,
           const OfxRectI & renderWindow,
           const OFX::Image* srcImg,
           void *dstPixelData,
           const OfxRectI & dstBounds,
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

    return copyPixels(instance, renderWindow, srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

inline void
copyPixels(OFX::ImageEffect &instance,
           const OfxRectI & renderWindow,
           const OFX::Image* srcImg,
           OFX::Image* dstImg)
{
    void* dstPixelData;
    OfxRectI dstBounds;
    OFX::PixelComponentEnum dstPixelComponents;
    OFX::BitDepthEnum dstBitDepth;
    int dstRowBytes;

    getImageData(dstImg, &dstPixelData, &dstBounds, &dstPixelComponents, &dstBitDepth, &dstRowBytes);
    int dstPixelComponentCount = dstImg->getPixelComponentCount();

    return copyPixels(instance, renderWindow, srcImg, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

inline void
copyPixels(OFX::ImageEffect &instance,
           const OfxRectI & renderWindow,
           const void *srcPixelData,
           const OfxRectI & srcBounds,
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

    return copyPixels(instance, renderWindow, srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

// pixel copiers, threaded versions
template<class PIX, int nComponents, int maxValue>
void
copyPixelsOpaqueForDepthAndComponents(OFX::ImageEffect &instance,
                                      const OfxRectI & renderWindow,
                                      const PIX *srcPixelData,
                                      const OfxRectI & srcBounds,
                                      OFX::PixelComponentEnum srcPixelComponents,
                                      int srcPixelComponentCount,
                                      OFX::BitDepthEnum srcBitDepth,
                                      int srcRowBytes,
                                      PIX *dstPixelData,
                                      const OfxRectI & dstBounds,
                                      OFX::PixelComponentEnum dstPixelComponents,
                                      int dstPixelComponentCount,
                                      OFX::BitDepthEnum dstBitDepth,
                                      int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);
    //assert(srcBounds.y1 <= renderWindow.y1 && renderWindow.y1 <= renderWindow.y2 && renderWindow.y2 <= srcBounds.y2); // not necessary, PixelCopier should handle this
    //assert(srcBounds.x1 <= renderWindow.x1 && renderWindow.x1 <= renderWindow.x2 && renderWindow.x2 <= srcBounds.x2); // not necessary, PixelCopier should handle this
    assert(srcPixelComponents == dstPixelComponents && srcBitDepth == dstBitDepth);
    assert(srcPixelComponentCount == dstPixelComponentCount);
    (void)srcPixelComponents;
    (void)srcBitDepth;
    (void)dstPixelComponents;
    (void)dstBitDepth;

    OFX::PixelCopierOpaque<PIX, nComponents, maxValue> processor(instance);
    // set the images
    processor.setDstImg(dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    processor.setSrcImg(srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes, 0);

    // set the render window
    processor.setRenderWindow(renderWindow);

    // Call the base class process member, this will call the derived templated process code
    processor.process();
}

template<class PIX, int maxValue>
void
copyPixelsOpaqueForDepth(OFX::ImageEffect &instance,
                         const OfxRectI & renderWindow,
                         const void *srcPixelData,
                         const OfxRectI & srcBounds,
                         OFX::PixelComponentEnum srcPixelComponents,
                         int srcPixelComponentCount,
                         OFX::BitDepthEnum srcBitDepth,
                         int srcRowBytes,
                         void *dstPixelData,
                         const OfxRectI & dstBounds,
                         OFX::PixelComponentEnum dstPixelComponents,
                         int dstPixelComponentCount,
                         OFX::BitDepthEnum dstBitDepth,
                         int dstRowBytes)
{
    if ( (dstPixelComponentCount < 0) || (4 < dstPixelComponentCount) ) {
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    }
    assert(srcPixelData && dstPixelData);
    assert(srcPixelComponents == dstPixelComponents && srcBitDepth == dstBitDepth);
    assert(srcPixelComponentCount == dstPixelComponentCount);
    // do the rendering
    if ( (dstPixelComponentCount == 4) || (dstPixelComponentCount == 1) ) {
        if (dstPixelComponentCount == 4) {
            copyPixelsOpaqueForDepthAndComponents<PIX, 4, maxValue>(instance, renderWindow,
                                                                    (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                                    (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
        } else {
            copyPixelsOpaqueForDepthAndComponents<PIX, 1, maxValue>(instance, renderWindow,
                                                                    (const PIX*)srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                                    (PIX *)dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
        }
    } else {
        copyPixelsForDepth<PIX>(instance, renderWindow,
                                srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
    }
}

inline void
copyPixelsOpaque(OFX::ImageEffect &instance,
                 const OfxRectI & renderWindow,
                 const void *srcPixelData,
                 const OfxRectI & srcBounds,
                 OFX::PixelComponentEnum srcPixelComponents,
                 int srcPixelComponentCount,
                 OFX::BitDepthEnum srcBitDepth,
                 int srcRowBytes,
                 void *dstPixelData,
                 const OfxRectI & dstBounds,
                 OFX::PixelComponentEnum dstPixelComponents,
                 int dstPixelComponentCount,
                 OFX::BitDepthEnum dstBitDepth,
                 int dstRowBytes)
{
    if ( ( (dstPixelComponents != ePixelComponentRGBA) && (dstPixelComponents != ePixelComponentAlpha) ) ||
         !srcPixelData ) {
        assert(dstPixelComponentCount != 4 || !srcPixelData);

        return copyPixels(instance,
                          renderWindow,
                          srcPixelData,
                          srcBounds,
                          srcPixelComponents,
                          srcPixelComponentCount,
                          srcBitDepth,
                          srcRowBytes,
                          dstPixelData,
                          dstBounds,
                          dstPixelComponents,
                          dstPixelComponentCount,
                          dstBitDepth,
                          dstRowBytes);
    }
    assert(dstPixelData && srcPixelData);
    assert(srcPixelComponents == dstPixelComponents && srcBitDepth == dstBitDepth);
    // do the rendering
    switch (dstBitDepth) {
    case OFX::eBitDepthUByte: {
        copyPixelsOpaqueForDepth<unsigned char, 255>(instance, renderWindow,
                                                     srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                     dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
        break;
    }
    case OFX::eBitDepthUShort: {
        copyPixelsOpaqueForDepth<unsigned short, 65535>(instance, renderWindow,
                                                        srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                        dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
        break;
    }
    case OFX::eBitDepthHalf: {
        // the unsigned short representation of 1.h is 15360
        // #include <OpenEXR/half.h>
        // half one(1.);
        // cout << "the unsigned short representation of 1.h is " << one.bits() << endl;
        copyPixelsOpaqueForDepth<unsigned short, 15360>(instance, renderWindow,
                                                        srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                                        dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
        break;
    }
    case OFX::eBitDepthFloat: {
        copyPixelsOpaqueForDepth<float, 1>(instance, renderWindow,
                                           srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes,
                                           dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
        break;
    }
    default:
        OFX::throwSuiteStatusException(kOfxStatErrFormat);

        return;
    } // switch
} // copyPixelsOpaque

inline void
copyPixelsOpaque(OFX::ImageEffect &instance,
                 const OfxRectI & renderWindow,
                 const OFX::Image* srcImg,
                 void *dstPixelData,
                 const OfxRectI & dstBounds,
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

    return copyPixelsOpaque(instance, renderWindow, srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

inline void
copyPixelsOpaque(OFX::ImageEffect &instance,
                 const OfxRectI & renderWindow,
                 const OFX::Image* srcImg,
                 OFX::Image* dstImg)
{
    void* dstPixelData;
    OfxRectI dstBounds;
    OFX::PixelComponentEnum dstPixelComponents;
    OFX::BitDepthEnum dstBitDepth;
    int dstRowBytes;

    getImageData(dstImg, &dstPixelData, &dstBounds, &dstPixelComponents, &dstBitDepth, &dstRowBytes);
    int dstPixelComponentCount = dstImg->getPixelComponentCount();

    return copyPixelsOpaque(instance, renderWindow, srcImg, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}

inline void
copyPixelsOpaque(OFX::ImageEffect &instance,
                 const OfxRectI & renderWindow,
                 const void *srcPixelData,
                 const OfxRectI & srcBounds,
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

    return copyPixelsOpaque(instance, renderWindow, srcPixelData, srcBounds, srcPixelComponents, srcPixelComponentCount, srcBitDepth, srcRowBytes, dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
}
} // OFX

#endif // ifndef IO_ofxsCopier_h
