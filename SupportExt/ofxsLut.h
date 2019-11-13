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
 * OFX color-spaces transformations support as-well as bit-depth conversions.
 */

#ifndef openfx_supportext_ofxsLut_h
#define openfx_supportext_ofxsLut_h

#include <string>
#include <map>
#include <cmath>
#include <cassert>
#include <cstring> // for memcpy
#include <cstdlib> // for rand
#include <memory> // for auto_ptr

#include "ofxCore.h"
#include "ofxsImageEffect.h"
#include "ofxsMacros.h"
#include "ofxsPixelProcessor.h"
#include "ofxsMultiThread.h"
#include "ofxsThreadSuite.h"

#define OFXS_HUE_CIRCLE 1. // if hue should be between 0 and 1
//#define OFXS_HUE_CIRCLE 360. // if hue should be in degrees

namespace OFX {
namespace Color {
/// numvals should be 256 for byte, 65536 for 16-bits, etc.

/// maps 0-(numvals-1) to 0.-1.
template<int numvals>
float
intToFloat(int value)
{
    return value / (float)(numvals - 1);
}

/// maps °.-1. to 0-(numvals-1)
template<int numvals>
int
floatToInt(float value)
{
    if (value <= 0) {
        return 0;
    } else if (value >= 1.) {
        return numvals - 1;
    }

    return value * (numvals - 1) + 0.5;
}

/// maps 0x0-0xffff to 0x0-0xff
inline unsigned char
uint16ToChar(unsigned short quantum)
{
    // see ScaleQuantumToChar() in ImageMagick's magick/quantum.h

    /* test:
       for(int i=0; i < 0x10000; ++i) {
       printf("%x -> %x,%x\n", i, uint16ToChar(i), floatToInt<256>(intToFloat<65536>(i)));
       assert(uint16ToChar(i) == floatToInt<256>(intToFloat<65536>(i)));
       }
     */
    return (unsigned char) ( ( (quantum + 128UL) - ( (quantum + 128UL) >> 8 ) ) >> 8 );
}

/// maps 0x0-0xff to 0x0-0xffff
inline unsigned short
charToUint16(unsigned char quantum)
{
    /* test:
       for(int i=0; i < 0x100; ++i) {
       printf("%x -> %x,%x\n", i, charToUint16(i), floatToInt<65536>(intToFloat<256>(i)));
       assert(charToUint16(i) == floatToInt<65536>(intToFloat<256>(i)));
       assert(i == uint16ToChar(charToUint16(i)));
       }
     */
    return (unsigned short) ( (quantum << 8) | quantum );
}

// maps 0x0-0xff00 to 0x0-0xff
inline unsigned char
uint8xxToChar(unsigned short quantum)
{
    /* test:
       for(int i=0; i < 0xff01; ++i) {
       printf("%x -> %x,%x, err=%d\n", i, uint8xxToChar(i), floatToInt<256>(intToFloat<0xff01>(i)),i - charToUint8xx(uint8xxToChar(i)));
       assert(uint8xxToChar(i) == floatToInt<256>(intToFloat<0xff01>(i)));
       }
     */
    return (unsigned char) ( (quantum + 0x80) >> 8 );
}

// maps 0x0-0xff to 0x0-0xff00
inline unsigned short
charToUint8xx(unsigned char quantum)
{
    /* test:
       for(int i=0; i < 0x100; ++i) {
       printf("%x -> %x,%x\n", i, charToUint8xx(i), floatToInt<0xff01>(intToFloat<256>(i)));
       assert(charToUint8xx(i) == floatToInt<0xff01>(intToFloat<256>(i)));
       assert(i == uint8xxToChar(charToUint8xx(i)));
       }
     */
    return (unsigned short) (quantum << 8);
}

/* @brief Converts a float ranging in [0 - 1.f] in the desired color-space to linear color-space also ranging in [0 - 1.f]*/
typedef float (*fromColorSpaceFunctionV1)(float v);

/* @brief Converts a float ranging in [0 - 1.f] in  linear color-space to the desired color-space to also ranging in [0 - 1.f]*/
typedef float (*toColorSpaceFunctionV1)(float v);


/**
 * @brief A Lut (look-up table) used to speed-up color-spaces conversions.
 * If you plan on doing linear conversion, you should just use the Linear class instead.
 **/
class Lut
{
    template<class MUTEX>
    friend class LutManager;

    std::string _name;                 ///< name of the lut
    fromColorSpaceFunctionV1 _fromFunc;
    toColorSpaceFunctionV1 _toFunc;

    /// the fast lookup tables are mutable, because they are automatically initialized post-construction,
    /// and never change afterwards
    mutable unsigned short toFunc_hipart_to_uint8xx[0x10000];                 /// contains  2^16 = 65536 values between 0-255
    mutable float fromFunc_uint8_to_float[256];                 /// values between 0-1.f

private:
    // Luts should be allocated and destroyed  through the LutManager
    Lut(const std::string & name,
        fromColorSpaceFunctionV1 fromFunc,
        toColorSpaceFunctionV1 toFunc)
        : _name(name)
        , _fromFunc(fromFunc)
        , _toFunc(toFunc)
    {
        fillTables();
    }

    virtual ~Lut()
    {
    }


    ///init luts
    ///it uses fromColorSpaceFloatToLinearFloat(float) and toColorSpaceFloatFromLinearFloat(float)
    ///Called by validate()
    void fillTables() const
    {
        // fill all
        for (int i = 0; i < 0x10000; ++i) {
            float inp = index_to_float( (unsigned short)i );
            float f = _toFunc(inp);
            toFunc_hipart_to_uint8xx[i] = Color::floatToInt<0xff01>(f);
        }
        // fill fromFunc_uint8_to_float, and make sure that
        // the entries of toFunc_hipart_to_uint8xx corresponding
        // to the transform of each byte value contain the same value,
        // so that toFunc(fromFunc(b)) is identity
        //
        for (int b = 0; b < 256; ++b) {
            float f = _fromFunc( Color::intToFloat<256>(b) );
            fromFunc_uint8_to_float[b] = f;
            int i = hipart(f);
            toFunc_hipart_to_uint8xx[i] = Color::charToUint8xx(b);
        }
    }

public:

    /* @brief Converts a float ranging in [0 - 1.f] in the desired color-space to linear color-space also ranging in [0 - 1.f]
     * This function is not fast!
     * @see fromColorSpaceFloatToLinearFloatFast(float)
     */
    float fromColorSpaceFloatToLinearFloat(float v) const WARN_UNUSED_RETURN
    {
        return _fromFunc(v);
    }

    /* @brief Converts a float ranging in [0 - 1.f] in  linear color-space to the desired color-space to also ranging in [0 - 1.f]
     * This function is not fast!
     */
    float toColorSpaceFloatFromLinearFloat(float v) const WARN_UNUSED_RETURN
    {
        return _toFunc(v);
    }

    /* @brief Converts a float ranging in [0 - 1.f] in linear color-space using the look-up tables.
     * @return A float in [0 - 1.f] in the destination color-space.
     */
    // It is not recommended to use this function, because the output is quantized
    // If one really needs float, one has to use the full function (or OpenColorIO)

    /* @brief Converts a float ranging in [0 - 1.f] in linear color-space using the look-up tables.
     * @return A byte in [0 - 255] in the destination color-space.
     */
    unsigned char toColorSpaceUint8FromLinearFloatFast(float v) const WARN_UNUSED_RETURN
    {
        return Color::uint8xxToChar(toFunc_hipart_to_uint8xx[hipart(v)]);
    }

    /* @brief Converts a float ranging in [0 - 1.f] in linear color-space using the look-up tables.
     * @return An unsigned short in [0 - 0xff00] in the destination color-space.
     */
    unsigned short toColorSpaceUint8xxFromLinearFloatFast(float v) const WARN_UNUSED_RETURN
    {
        return toFunc_hipart_to_uint8xx[hipart(v)];
    }

    // the following only works for increasing LUTs

    /* @brief Converts a float ranging in [0 - 1.f] in linear color-space using the look-up tables.
     * @return An unsigned short in [0 - 65535] in the destination color-space.
     * This function uses localluy linear approximations of the transfer function.
     */
    unsigned short toColorSpaceUint16FromLinearFloatFast(float v) const WARN_UNUSED_RETURN
    {
        // algorithm:
        // - convert to 8 bits -> val8u
        // - convert val8u-1, val8u and val8u+1 to float
        // - interpolate linearly in the right interval
        unsigned char v8u = toColorSpaceUint8FromLinearFloatFast(v);
        unsigned char v8u_next, v8u_prev;
        float v32f_next, v32f_prev;
        if (v8u == 0) {
            v8u_prev = 0;
            v8u_next = 1;
            v32f_prev = fromColorSpaceUint8ToLinearFloatFast(0);
            v32f_next = fromColorSpaceUint8ToLinearFloatFast(1);
        } else if (v8u == 255) {
            v8u_prev = 254;
            v8u_next = 255;
            v32f_prev = fromColorSpaceUint8ToLinearFloatFast(254);
            v32f_next = fromColorSpaceUint8ToLinearFloatFast(255);
        } else {
            float v32f = fromColorSpaceUint8ToLinearFloatFast(v8u);
            // we suppose the LUT is an increasing func
            if (v < v32f) {
                v8u_prev = v8u - 1;
                v32f_prev = fromColorSpaceUint8ToLinearFloatFast(v8u_prev);
                v8u_next = v8u;
                v32f_next = v32f;
            } else {
                v8u_prev = v8u;
                v32f_prev = v32f;
                v8u_next = v8u + 1;
                v32f_next = fromColorSpaceUint8ToLinearFloatFast(v8u_next);
            }
        }

        // interpolate linearly
        return (v8u_prev << 8) + v8u_prev + (v - v32f_prev) * ( ( (v8u_next - v8u_prev) << 8 ) + (v8u_next + v8u_prev) ) / (v32f_next - v32f_prev) + 0.5;
    }

    /* @brief Converts a byte ranging in [0 - 255] in the destination color-space using the look-up tables.
     * @return A float in [0 - 1.f] in linear color-space.
     */
    float fromColorSpaceUint8ToLinearFloatFast(unsigned char v) const WARN_UNUSED_RETURN
    {
        return fromFunc_uint8_to_float[v];
    }

    /* @brief Converts a short ranging in [0 - 65535] in the destination color-space using the look-up tables.
     * @return A float in [0 - 1.f] in linear color-space.
     */
    float fromColorSpaceUint16ToLinearFloatFast(unsigned short v) const WARN_UNUSED_RETURN
    {
        // the following is from ImageMagick's quantum.h
        unsigned char v8u_prev = ( v - (v >> 8) ) >> 8;
        unsigned char v8u_next = v8u_prev + 1;
        unsigned short v16u_prev = (v8u_prev << 8) + v8u_prev;
        unsigned short v16u_next = (v8u_next << 8) + v8u_next;
        float v32f_prev = fromColorSpaceUint8ToLinearFloatFast(v8u_prev);
        float v32f_next = fromColorSpaceUint8ToLinearFloatFast(v8u_next);

        // interpolate linearly
        return v32f_prev + (v - v16u_prev) * (v32f_next - v32f_prev) / (v16u_next - v16u_prev);
    }

    /* @brief convert from float to byte with dithering (error diffusion).
     It uses random numbers for error diffusion, and thus the result is different at each function call. */
    void to_byte_packed_dither(const void* pixelData,
                               const OfxRectI & bounds,
                               OFX::PixelComponentEnum pixelComponents,
                               int pixelComponentCount,
                               OFX::BitDepthEnum bitDepth,
                               int rowBytes,
                               const OfxRectI & renderWindow,
                               void* dstPixelData,
                               const OfxRectI & dstBounds,
                               OFX::PixelComponentEnum dstPixelComponents,
                               int dstPixelComponentCount,
                               OFX::BitDepthEnum dstBitDepth,
                               int dstRowBytes) const
    {
        assert(bitDepth == eBitDepthFloat && dstBitDepth == eBitDepthUByte && pixelComponents == dstPixelComponents);
        assert(bounds.x1 <= renderWindow.x1 && renderWindow.x2 <= bounds.x2 &&
               bounds.y1 <= renderWindow.y1 && renderWindow.y2 <= bounds.y2 &&
               dstBounds.x1 <= renderWindow.x1 && renderWindow.x2 <= dstBounds.x2 &&
               dstBounds.y1 <= renderWindow.y1 && renderWindow.y2 <= dstBounds.y2);
        if (pixelComponents == ePixelComponentAlpha) {
            // alpha: no dither
            return to_byte_packed_nodither(pixelData, bounds, pixelComponents, pixelComponentCount, bitDepth, rowBytes,
                                           renderWindow,
                                           dstPixelData, dstBounds, dstPixelComponents, dstPixelComponentCount, dstBitDepth, dstRowBytes);
        }
        //validate();

        const int nComponents = dstPixelComponentCount;
        assert(dstPixelComponentCount == 3 || dstPixelComponentCount == 4);

        for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
            // coverity[dont_call]
            int xstart = renderWindow.x1 + std::rand() % (renderWindow.x2 - renderWindow.x1);
            unsigned error[3] = {
                0x80, 0x80, 0x80
            };
            const float *src_pixels = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, xstart, y);
            unsigned char *dst_pixels = (unsigned char*)OFX::getPixelAddress(dstPixelData, dstBounds, dstPixelComponents, dstBitDepth, dstRowBytes, xstart, y);

            /* go forward from starting point to end of line: */
            const float *src_end = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, renderWindow.x2, y, false);

            while (src_pixels < src_end) {
                for (int k = 0; k < 3; ++k) {
                    error[k] = (error[k] & 0xff) + toColorSpaceUint8xxFromLinearFloatFast(src_pixels[k]);
                    assert(error[k] < 0x10000);
                    dst_pixels[k] = (unsigned char)(error[k] >> 8);
                }
                if (nComponents == 4) {
                    // alpha channel: no dithering
                    dst_pixels[3] = floatToInt<256>(src_pixels[3]);
                }
                dst_pixels += nComponents;
                src_pixels += nComponents;
            }

            if (xstart > 0) {
                /* go backward from starting point to start of line: */
                src_pixels = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, xstart - 1, y);
                src_end = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, 0, y);
                dst_pixels = (unsigned char*)OFX::getPixelAddress(dstPixelData, dstBounds, dstPixelComponents, dstBitDepth, dstRowBytes, xstart - 1, y);

                for (int i = 0; i < 3; ++i) {
                    error[i] = 0x80;
                }

                while (src_pixels >= src_end) {
                    for (int k = 0; k < 3; ++k) {
                        error[k] = (error[k] & 0xff) + toColorSpaceUint8xxFromLinearFloatFast(src_pixels[k]);
                        assert(error[k] < 0x10000);
                        dst_pixels[k] = (unsigned char)(error[k] >> 8);
                    }
                    if (nComponents == 4) {
                        // alpha channel: no colorspace conversion & no dithering
                        dst_pixels[3] = floatToInt<256>(src_pixels[3]);
                    }
                    dst_pixels -= nComponents;
                    src_pixels -= nComponents;
                }
            }
        }
    } // to_byte_packed_dither

    /* @brief convert from float to byte without dithering. */
    void to_byte_packed_nodither(const void* pixelData,
                                 const OfxRectI & bounds,
                                 OFX::PixelComponentEnum pixelComponents,
                                 int pixelComponentCount,
                                 OFX::BitDepthEnum bitDepth,
                                 int rowBytes,
                                 const OfxRectI & renderWindow,
                                 void* dstPixelData,
                                 const OfxRectI & dstBounds,
                                 OFX::PixelComponentEnum dstPixelComponents,
                                 int dstPixelComponentCount,
                                 OFX::BitDepthEnum dstBitDepth,
                                 int dstRowBytes) const
    {
        assert(bitDepth == eBitDepthFloat && dstBitDepth == eBitDepthUByte);
        assert(pixelComponents == ePixelComponentRGBA || pixelComponents == ePixelComponentRGB || pixelComponents == ePixelComponentAlpha);
        assert(dstPixelComponents == ePixelComponentRGBA || dstPixelComponents == ePixelComponentRGB || dstPixelComponents == ePixelComponentAlpha);
        assert(bounds.x1 <= renderWindow.x1 && renderWindow.x2 <= bounds.x2 &&
               bounds.y1 <= renderWindow.y1 && renderWindow.y2 <= bounds.y2 &&
               dstBounds.x1 <= renderWindow.x1 && renderWindow.x2 <= dstBounds.x2 &&
               dstBounds.y1 <= renderWindow.y1 && renderWindow.y2 <= dstBounds.y2);
        //validate();

        const int srcComponents = pixelComponentCount;
        const int dstComponents = dstPixelComponentCount;

        for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
            const float *src_pixels = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, 0, y);
            unsigned char *dst_pixels = (unsigned char*)OFX::getPixelAddress(dstPixelData, dstBounds, dstPixelComponents, dstBitDepth, dstRowBytes, 0, y);
            const float *src_end = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, renderWindow.x2, y, false);
            unsigned char tmpPixel[4] = {0, 0, 0, 0};
            while (src_pixels != src_end) {
                if (srcComponents == 1) {
                    // alpha channel: no colorspace conversion
                    tmpPixel[3] = floatToInt<256>(src_pixels[0]);
                } else {
                    for (int k = 0; k < 3; ++k) {
                        tmpPixel[k] = toColorSpaceUint8FromLinearFloatFast(src_pixels[k]);
                    }
                    if (srcComponents == 4) {
                        // alpha channel: no colorspace conversion
                        tmpPixel[3] = floatToInt<256>(src_pixels[3]);
                    }
                }
                if (dstComponents == 1) {
                    dst_pixels[0] = tmpPixel[3];
                } else {
                    for (int k = 0; k < dstComponents; ++k) {
                        dst_pixels[k] = tmpPixel[k];
                    }
                }
                dst_pixels += dstComponents;
                src_pixels += srcComponents;
            }
        }
    } // to_byte_packed_nodither

    /* @brief uses Rec.709 to convert from color to grayscale. */
    void to_byte_grayscale_nodither(const void* pixelData,
                                    const OfxRectI & bounds,
                                    OFX::PixelComponentEnum pixelComponents,
                                    int pixelComponentCount,
                                    OFX::BitDepthEnum bitDepth,
                                    int rowBytes,
                                    const OfxRectI & renderWindow,
                                    void* dstPixelData,
                                    const OfxRectI & dstBounds,
                                    OFX::PixelComponentEnum dstPixelComponents,
                                    int dstPixelComponentCount,
                                    OFX::BitDepthEnum dstBitDepth,
                                    int dstRowBytes) const
    {
        assert(bitDepth == eBitDepthFloat && dstBitDepth == eBitDepthUByte &&
               (pixelComponents == ePixelComponentRGB || pixelComponents == ePixelComponentRGBA) &&
               dstPixelComponents == ePixelComponentAlpha &&
               (pixelComponentCount == 3 || pixelComponentCount == 4) &&
               dstPixelComponentCount == 1);
        assert(bounds.x1 <= renderWindow.x1 && renderWindow.x2 <= bounds.x2 &&
               bounds.y1 <= renderWindow.y1 && renderWindow.y2 <= bounds.y2 &&
               dstBounds.x1 <= renderWindow.x1 && renderWindow.x2 <= dstBounds.x2 &&
               dstBounds.y1 <= renderWindow.y1 && renderWindow.y2 <= dstBounds.y2);
        //validate();

        const int srcComponents = pixelComponentCount;

        for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
            const float *src_pixels = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, 0, y);
            unsigned char *dst_pixels = (unsigned char*)OFX::getPixelAddress(dstPixelData, dstBounds, dstPixelComponents, dstBitDepth, dstRowBytes, 0, y);
            const float *src_end = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, renderWindow.x2, y, false);

            while (src_pixels != src_end) {
                float l = 0.2126 * src_pixels[0] + 0.7152 * src_pixels[1] + 0.0722 * src_pixels[2]; // Rec.709 luminance formula
                dst_pixels[0] = toColorSpaceUint8FromLinearFloatFast(l);
                ++dst_pixels;
                src_pixels += srcComponents;
            }
        }
    } // to_byte_packed_nodither

    /* @brief convert from float to short without dithering. */
    void to_short_packed(const void* pixelData,
                         const OfxRectI & bounds,
                         OFX::PixelComponentEnum pixelComponents,
                         int pixelComponentCount,
                         OFX::BitDepthEnum bitDepth,
                         int rowBytes,
                         const OfxRectI & renderWindow,
                         void* dstPixelData,
                         const OfxRectI & dstBounds,
                         OFX::PixelComponentEnum dstPixelComponents,
                         int dstPixelComponentCount,
                         OFX::BitDepthEnum dstBitDepth,
                         int dstRowBytes) const
    {
        assert(bitDepth == eBitDepthFloat && dstBitDepth == eBitDepthUShort && pixelComponents == dstPixelComponents && pixelComponentCount == dstPixelComponentCount);
        assert(bounds.x1 <= renderWindow.x1 && renderWindow.x2 <= bounds.x2 &&
               bounds.y1 <= renderWindow.y1 && renderWindow.y2 <= bounds.y2 &&
               dstBounds.x1 <= renderWindow.x1 && renderWindow.x2 <= dstBounds.x2 &&
               dstBounds.y1 <= renderWindow.y1 && renderWindow.y2 <= dstBounds.y2);
        //validate();

        const int nComponents = pixelComponentCount;

        for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
            const float *src_pixels = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, 0, y);
            unsigned char *dst_pixels = (unsigned char*)OFX::getPixelAddress(dstPixelData, dstBounds, dstPixelComponents, dstBitDepth, dstRowBytes, 0, y);
            const float *src_end = (const float*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, renderWindow.x2, y, false);

            while (src_pixels != src_end) {
                if (nComponents == 1) {
                    // alpha channel: no colorspace conversion
                    dst_pixels[0] = floatToInt<65536>(src_pixels[0]);
                } else {
                    for (int k = 0; k < 3; ++k) {
                        dst_pixels[k] = toColorSpaceUint16FromLinearFloatFast(src_pixels[k]);
                    }
                    if (nComponents == 4) {
                        // alpha channel: no colorspace conversion
                        dst_pixels[3] = floatToInt<65536>(src_pixels[3]);
                    }
                }
                dst_pixels += nComponents;
                src_pixels += nComponents;
            }
        }
    }

    void from_byte_packed(const void* pixelData,
                          const OfxRectI & bounds,
                          OFX::PixelComponentEnum pixelComponents,
                          int pixelComponentCount,
                          OFX::BitDepthEnum bitDepth,
                          int rowBytes,
                          const OfxRectI & renderWindow,
                          void* dstPixelData,
                          const OfxRectI & dstBounds,
                          OFX::PixelComponentEnum dstPixelComponents,
                          int dstPixelComponentCount,
                          OFX::BitDepthEnum dstBitDepth,
                          int dstRowBytes) const
    {
        assert(bitDepth == eBitDepthUByte && dstBitDepth == eBitDepthFloat && pixelComponents == dstPixelComponents && pixelComponentCount == dstPixelComponentCount);
        assert(bounds.x1 <= renderWindow.x1 && renderWindow.x2 <= bounds.x2 &&
               bounds.y1 <= renderWindow.y1 && renderWindow.y2 <= bounds.y2 &&
               dstBounds.x1 <= renderWindow.x1 && renderWindow.x2 <= dstBounds.x2 &&
               dstBounds.y1 <= renderWindow.y1 && renderWindow.y2 <= dstBounds.y2);
        //validate();

        const int nComponents = pixelComponentCount;

        for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
            const unsigned char *src_pixels = (const unsigned char*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, 0, y);
            float *dst_pixels = (float*)OFX::getPixelAddress(dstPixelData, dstBounds, dstPixelComponents, dstBitDepth, dstRowBytes, 0, y);
            const unsigned char *src_end = (const unsigned char*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, renderWindow.x2, y, false);


            while (src_pixels != src_end) {
                if (nComponents == 1) {
                    *dst_pixels++ = intToFloat<256>(*src_pixels++);
                } else {
                    for (int k = 0; k < 3; ++k) {
                        dst_pixels[k] = fromColorSpaceUint8ToLinearFloatFast(src_pixels[k]);
                    }
                    if (nComponents == 4) {
                        // alpha channel: no colorspace conversion
                        dst_pixels[3] = intToFloat<256>(src_pixels[3]);
                    }
                }
                dst_pixels += nComponents;
                src_pixels += nComponents;
            }
        }
    }

    void from_short_packed(const void* pixelData,
                           const OfxRectI & bounds,
                           OFX::PixelComponentEnum pixelComponents,
                           int pixelComponentCount,
                           OFX::BitDepthEnum bitDepth,
                           int rowBytes,
                           const OfxRectI & renderWindow,
                           void* dstPixelData,
                           const OfxRectI & dstBounds,
                           OFX::PixelComponentEnum dstPixelComponents,
                           int dstPixelComponentCount,
                           OFX::BitDepthEnum dstBitDepth,
                           int dstRowBytes) const
    {
        assert(bitDepth == eBitDepthUShort && dstBitDepth == eBitDepthFloat && pixelComponents == dstPixelComponents && pixelComponentCount == dstPixelComponentCount);
        assert(bounds.x1 <= renderWindow.x1 && renderWindow.x2 <= bounds.x2 &&
               bounds.y1 <= renderWindow.y1 && renderWindow.y2 <= bounds.y2 &&
               dstBounds.x1 <= renderWindow.x1 && renderWindow.x2 <= dstBounds.x2 &&
               dstBounds.y1 <= renderWindow.y1 && renderWindow.y2 <= dstBounds.y2);
        //validate();

        const int nComponents = pixelComponentCount;

        for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
            const unsigned short *src_pixels = (const unsigned short*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, 0, y);
            float *dst_pixels = (float*)OFX::getPixelAddress(dstPixelData, dstBounds, dstPixelComponents, dstBitDepth, dstRowBytes, 0, y);
            const unsigned short *src_end = (const unsigned short*)OFX::getPixelAddress(pixelData, bounds, pixelComponents, bitDepth, rowBytes, renderWindow.x2, y, false);


            while (src_pixels != src_end) {
                if (nComponents == 1) {
                    *dst_pixels++ = intToFloat<65536>(*src_pixels++);
                } else {
                    for (int k = 0; k < 3; ++k) {
                        dst_pixels[k] = fromColorSpaceUint16ToLinearFloatFast(src_pixels[k]);
                    }
                    if (nComponents == 4) {
                        // alpha channel: no colorspace conversion
                        dst_pixels[3] = intToFloat<65536>(src_pixels[3]);
                    }
                }
                dst_pixels += nComponents;
                src_pixels += nComponents;
            }
        }
    }

private:
    static float index_to_float(const unsigned short i);
    static unsigned short hipart(const float f);
};


////////////////////////////////////////////////////////////////
// Transfer functions
//
// from_func_*: EOTF (Electro-Optical Transfer Function)
// to_func_*: OETF (Opto-Electronic Transfer Function)
//
// more can be found at:
// https://github.com/colour-science/colour/tree/develop/colour/models/rgb/transfer_functions
//
////////////////////////////////////////////////////////////////

inline float
from_func_linear(float v)
{
    return v;
}

inline float
to_func_linear(float v)
{
    return v;
}

/// from sRGB to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_srgb(float v)
{
    if (v < 0.04045f) {
        return (v < 0.0f) ? 0.0f : v * (1.0f / 12.92f);
    } else {
        return std::pow( (v + 0.055f) * (1.0f / 1.055f), 2.4f );
    }
}

/// to sRGB from Linear Opto-Electronic Transfer Function (OETF)
inline float
to_func_srgb(float v)
{
    if (v < 0.0031308f) {
        return (v < 0.0f) ? 0.0f : v * 12.92f;
    } else {
        return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }
}

// Rec.709 and Rec.2020 share the same transfer function (and illuminant), except that
// Rec.2020 is more precise.
// https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-S!!PDF-E.pdf
// Since this is float, we use the coefficients from Rec.2020

/// From Rec.709 to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_Rec709(float v)
{
    //if (v < 0.081f) {
    if (v < 0.08145f) {
        return (v < 0.0f) ? 0.0f : v * (1.0f / 4.5f);
    } else {
        //return std::pow( (v + 0.099f) * (1.0f / 1.099f), (1.0f / 0.45f) );
        return std::pow( (v + 0.0993f) * (1.0f / 1.0993f), (1.0f / 0.45f) );
    }
}

// see above comment
/// to Rec.709 from Linear Opto-Electronic Transfer Function (OETF)
inline float
to_func_Rec709(float v)
{
    //if (v < 0.018f) {
    if (v < 0.0181f) {
        return (v < 0.0f) ? 0.0f : v * 4.5f;
    } else {
        //return 1.099f * std::pow(v, 0.45f) - 0.099f;
        return 1.0993f * std::pow(v, 0.45f) - (1.0993f - 1.f);
    }
}

/*
   Following the formula:
   offset = pow(10,(blackpoint - whitepoint) * 0.002 / gamma)
   gain = 1/(1-offset)
   linear = gain * (pow(10,(1023*v - whitepoint)*0.002/gamma) - offset)
   cineon = (log10((v + offset) /gain)/ (0.002 / gamma) + whitepoint)/1023
   Here we're using: blackpoint = 95.0
   whitepoint = 685.0
   gammasensito = 0.6
 */
/// from Cineon to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_Cineon(float v)
{
    //return ( 1.f / ( 1.f - std::pow(10.f, -1.97f) ) ) * std::pow(10.f, ( (1023.f * v) - 685.f ) * 0.002f / 0.6f);
    //float offset = std::pow(10.f, (95.f - 685.f)*0.002f/0.6f);
    //float offset = 0.01079775161f;
    return ( 1.f / ( 1.f - 0.01079775161f ) ) * ( std::pow(10.f, ( (1023.f * v) - 685.f ) * 0.002f / 0.6f) - 0.01079775161f);
}

/// to Cineon from Linear Opto-Electronic Transfer Function (OETF)
inline float
to_func_Cineon(float v)
{
    //float offset = std::pow(10.f, -1.97f);
    //float offset = std::pow(10.f, (95.f - 685.f)*0.002f/0.6f);
    //float offset = 0.01079775161f;

    //return (std::log10( (v + offset) / ( 1.f / (1.f - offset) ) ) / 0.0033f + 685.0f) / 1023.f;
    return (std::log10( (v + 0.01079775161f) / ( 1.f / (1.f - 0.01079775161f) ) ) / (0.002f / 0.6f) + 685.0f) / 1023.f;
}

/// from Gamma 1.8 to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_Gamma1_8(float v)
{
    return (v < 0.0f) ? 0.0f : std::pow(v, 1.8f);
}

/// to Gamma 1.8 from Linear Opto-Electronic Transfer Function (OETF)
inline float
to_func_Gamma1_8(float v)
{
    return (v < 0.0f) ? 0.0f : std::pow(v, 0.55f);
}

/// from Gamma 2.2 to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_Gamma2_2(float v)
{
    return (v < 0.0f) ? 0.0f : std::pow(v, 2.2f);
}

/// to Gamma 2.2 from Linear Opto-Electronic Transfer Function (OETF)
inline float
to_func_Gamma2_2(float v)
{
    return (v < 0.0f) ? 0.0f : std::pow(v, 0.45f);
}

/// from Panalog to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_Panalog(float v)
{
    return (std::pow(10.f, (1023.f * v - 681.f) / 444.f) - 0.0408f) / (1.0f - 0.0408f);
}

/// to Panalog from Linear Opto-Electronic Transfer Function (OETF)
inline float
to_func_Panalog(float v)
{
    return (444.f * std::log10(0.0408f + (1.0f - 0.0408f) * v) + 681.f) / 1023.f;
}

/// from REDLog to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_REDLog(float v)
{
    return (std::pow(10.f, (1023.f * v - 1023.f) / 511.f) - 0.01) / (1.0f - 0.01f);
}

/// to REDLog from Linear Opto-Electronic Transfer Function (OETF)
    inline float
to_func_REDLog(float v)
{
    return (511.f * std::log10(0.01f + (1.0f - 0.01f) * v) + 1023.) / 1023.f;
}

/// from ViperLog to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_ViperLog(float v)
{
    return std::pow(10.f, (1023.f * v - 1023.f) / 500.f);
}

/// to ViperLog from Linear Opto-Electronic Transfer Function (OETF)
inline float
to_func_ViperLog(float v)
{
    return (500.f * std::log10(v) + 1023.f) / 1023.f;
}

/// from AlexaV3LogC to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_AlexaV3LogC(float v)
{
    // ref: "ALEXA LOG C Curve-Usage in VFX" PDF, p9
    return v > 0.1496582f ? std::pow(10.f, (v - 0.385537f) / 0.2471896f) * 0.18f - 0.00937677f
           : ( v / 0.9661776f - 0.04378604f) * 0.18f - 0.00937677f;
}

/// from Linear to AlexaV3LogC (EI=800) Opto-Electronic Transfer Function (OETF)
inline float
to_func_AlexaV3LogC(float v)
{
    // ref: "ALEXA LOG C Curve-Usage in VFX" PDF, p9
    return v > 0.010591f ?  0.247190f * std::log10(5.555556f * v + 0.052272f) + 0.385537f
           : v * 5.367655f + 0.092809f;
}

/// from SLog1 to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_SLog1(float v)
{
    // ref: https://pro.sony.com/bbsccms/assets/files/micro/dmpc/training/S-Log2_Technical_PaperV1_0.pdf
    return v >= 90./1023. ? (std::pow( 10., (((v*1023.0-64.0)/(940.0-64.0)-0.616596-0.03)/0.432699))-0.037584)*0.9
           : ((v*1023.0-64.0)/(940.0-64.0)-0.030001222851889303)/5.*0.9;
}

/// from Linear to SLog1 Opto-Electronic Transfer Function (OETF)
inline float
to_func_SLog1(float v)
{
    // ref: https://pro.sony.com/bbsccms/assets/files/micro/dmpc/training/S-Log2_Technical_PaperV1_0.pdf
    return v >= -0.00008153227156 ? ((std::log10((v / 0.9) + 0.037584) * 0.432699 +0.616596+0.03)*(940.0-64.0) + 64.)/1023.
           : (((v / 0.9) * 5. + 0.030001222851889303)*(940.0-64.0) + 64.)/1023;
}

/// from SLog2 to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_SLog2(float v)
{
    // http://community.thefoundry.co.uk/discussion/topic.aspx?f=189&t=100372
    // nuke.root().knob('luts').addCurve("SLog2-Ref", "{ (t>=90.0/1023.0)? 219.0*(pow(10.0, (((t*1023.0-64.0)/(940.0-64.0)-0.616596-0.03)/0.432699))-0.037584)/155.0*0.9 : ((t*1023.0-64.0)/(940.0-64.0)-0.030001222851889303)/3.53881278538813*0.9 }")
    // ref: https://pro.sony.com/bbsccms/assets/files/micro/dmpc/training/S-Log2_Technical_PaperV1_0.pdf
    return v >= 90./1023. ? 219.0 * (std::pow( 10., (((v*1023.0-64.0)/(940.0-64.0)-0.616596-0.03)/0.432699))-0.037584)/155.0*0.9
          : ((v*1023.0-64.0)/(940.0-64.0)-0.030001222851889303)/3.53881278538813*0.9;
}

/// from Linear to SLog2 Opto-Electronic Transfer Function (OETF)
inline float
to_func_SLog2(float v)
{
    // ref: https://pro.sony.com/bbsccms/assets/files/micro/dmpc/training/S-Log2_Technical_PaperV1_0.pdf
    return v >= -0.00008153227156 ? ((std::log10((v / 0.9) * 155. / 219. + 0.037584) * 0.432699 +0.616596+0.03)*(940.0-64.0) + 64.)/1023.
            : (((v / 0.9) * 3.53881278538813 + 0.030001222851889303)*(940.0-64.0) + 64.)/1023;
}

/// from SLog3 to Linear Electro-Optical Transfer Function (EOTF)
inline float
from_func_SLog3(float v)
{
    // http://www.sony.co.uk/pro/support/attachment/1237494271390/1237494271406/technical-summary-for-s-gamut3-cine-s-log3-and-s-gamut3-s-log3.pdf
    return v >= 171.2102946929 / 1023.0 ? std::pow(10.0, ((v * 1023.0 - 420.0) / 261.5)) * (0.18 + 0.01) - 0.01
          : (v * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0);
}

/// from Linear to SLog3 Opto-Electronic Transfer Function (OETF)
inline float
to_func_SLog3(float v)
{
    // http://www.sony.co.uk/pro/support/attachment/1237494271390/1237494271406/technical-summary-for-s-gamut3-cine-s-log3-and-s-gamut3-s-log3.pdf
    return v >= 0.01125000 ? (420.0 + std::log10((v + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0
         : (v * (171.2102946929 - 95.0)/0.01125000 + 95.0) / 1023.0;
}

/// convert RGB to HSV
/// In Nuke's viewer, sRGB values are used (apply to_func_srgb to linear
/// RGB values before calling this fuunction)
// r,g,b values are from 0 to 1
/// h = [0,360], s = [0,1], v = [0,1]
///		if s == 0, then h = -1 (undefined)
void rgb_to_hsv( float r, float g, float b, float *h, float *s, float *v );
void hsv_to_rgb( float h, float s, float v, float *r, float *g, float *b );

void rgb_to_hsl( float r, float g, float b, float *h, float *s, float *l );
void hsl_to_rgb( float h, float s, float l, float *r, float *g, float *b );

void rgb_to_hsi( float r, float g, float b, float *h, float *s, float *i );
void hsi_to_rgb( float h, float s, float i, float *r, float *g, float *b );

void rgb_to_ycbcr601( float r, float g, float b, float *y, float *cb, float *cr );
void ycbcr_to_rgb601( float y, float cb, float cr, float *r, float *g, float *b );

void rgb_to_ycbcr709( float r, float g, float b, float *y, float *cb, float *cr );
void ycbcr_to_rgb709( float y, float cb, float cr, float *r, float *g, float *b );

void rgb_to_ypbpr601( float r, float g, float b, float *y, float *pb, float *pr );
void ypbpr_to_rgb601( float y, float pb, float pr, float *r, float *g, float *b );

void rgb_to_ypbpr709( float r, float g, float b, float *y, float *pb, float *pr );
void ypbpr_to_rgb709( float y, float pb, float pr, float *r, float *g, float *b );

void rgb_to_ypbpr2020( float r, float g, float b, float *y, float *pb, float *pr );
void ypbpr_to_rgb2020( float y, float pb, float pr, float *r, float *g, float *b );

void rgb_to_yuv601( float r, float g, float b, float *y, float *u, float *v );
void yuv_to_rgb601( float y, float u, float v, float *r, float *g, float *b );

void rgb_to_yuv709( float r, float g, float b, float *y, float *u, float *v );
void yuv_to_rgb709( float y, float u, float v, float *r, float *g, float *b );

// r,g,b values are from 0 to 1
// Convert pixel values from linear RGB_709 or sRGB to XYZ color spaces.
// Uses the standard D65 white point.
template<typename T>
T rgb709_to_y( T r, T g, T b )
{
    // coefficients are those of http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    // from 07 Apr 2017
    return 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
}

template<typename T>
void
rgb709_to_xyz(T r,
              T g,
              T b,
              T *x,
              T *y,
              T *z)
{
    //*x = 0.412453f * r + 0.357580f * g + 0.180423f * b;
    //*y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
    //*z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
    // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    //> with(linalg):
    //> M:=matrix([[3.2409699419, -1.5373831776, -0.4986107603],[-0.9692436363, 1.8759675015, 0.0415550574],[ 0.0556300797, -0.2039769589,  1.0569715142]]);
    //> inverse(M);

    // coefficients are those of http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    // from 07 Apr 2017
    *x = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
    *y = rgb709_to_y(r, g, b);
    *z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;
}

// Convert pixel values from XYZ to linear RGB_709 or sRGB color spaces.
// Uses the standard D65 white point.
template<typename T>
void
xyz_to_rgb709(T x,
              T y,
              T z,
              T *r,
              T *g,
              T *b)
{
    //*r =  3.240479f * x - 1.537150f * y - 0.498535f * z;
    //*g = -0.969256f * x + 1.875992f * y + 0.041556f * z;
    //*b =  0.055648f * x - 0.204043f * y + 1.057311f * z;
    // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    //*r =  3.2409699419 * x + -1.5373831776 * y + -0.4986107603 * z;
    //*g = -0.9692436363 * x +  1.8759675015 * y +  0.0415550574 * z;
    //*b =  0.0556300797 * x + -0.2039769589 * y +  1.0569715142 * z;

    // coefficients are those of http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    // from 07 Apr 2017
    *r =  3.2404542f * x + -1.5371385f * y + -0.4985314f * z;
    *g = -0.9692660f * x +  1.8760108f * y +  0.0415560f * z;
    *b =  0.0556434f * x + -0.2040259f * y +  1.0572252f * z;
}

// r,g,b values are from 0 to 1
// Convert pixel values from RGB_2020 to XYZ color spaces.
// Uses the standard D65 white point.
template<typename T>
T rgb2020_to_y( T r, T g, T b )
{
    return 0.2627002119 * r + 0.6779980711 * g + 0.0593017165 * b;
}

template<typename T>
void
rgb2020_to_xyz(T r,
               T g,
               T b,
               T *x,
               T *y,
               T *z)
{
    //> with(linalg):
    //> P:=matrix([[1.7166511880,-0.3556707838,-0.2533662814],[-0.6666843518,1.6164812366,0.0157685458],[0.0176398574,-0.0427706133,0.9421031212]]);
    //> inverse(P);

    *x = 0.6369580481 * r + 0.1446169036 * g + 0.1688809752 * b;
    *y = rgb2020_to_y(r, g, b);
    *z = 0.0000000000 * r + 0.0280726931 * g + 1.060985058  * b;
}

// Convert pixel values from XYZ to RGB_2020 color spaces.
// Uses the standard D65 white point.
template<typename T>
void
xyz_to_rgb2020(T x,
               T y,
               T z,
               T *r,
               T *g,
               T *b)
{
    // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    *r =  1.7166511880 * x + -0.3556707838 * y + -0.2533662814 * z;
    *g = -0.6666843518 * x +  1.6164812366 * y +  0.0157685458 * z;
    *b =  0.0176398574 * x + -0.0427706133 * y +  0.9421031212 * z;
}

// r,g,b values are from 0 to 1
// Convert pixel values from RGB_ACES_AP0 to XYZ color spaces.
// Uses the ACES white point (approx. D60).
template<typename T>
T rgbACESAP0_to_y( T r, T g, T b )
{
    return 0.3439664498 * r + 0.7281660966 * g + -0.0721325464 * b;
}

template<typename T>
void
rgbACESAP0_to_xyz(T r,
                  T g,
                  T b,
                  T *x,
                  T *y,
                  T *z)
{
    // https://en.wikipedia.org/wiki/Academy_Color_Encoding_System#Converting_ACES_RGB_values_to_CIE_XYZ_values
    // and
    // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    *x = 0.9525523959 * r + 0.0000000000 * g +  0.0000936786 * b;
    *y = rgbACESAP0_to_y(r, g, b);
    *z = 0.0000000000 * r + 0.0000000000 * g +  1.0088251844 * b;
}

// Convert pixel values from XYZ to RGB_ACES_AP0 (with the ACES illuminant) color spaces.
// Uses the ACES white point (approx. D60).
template<typename T>
void
xyz_to_rgbACESAP0(T x,
                  T y,
                  T z,
                  T *r,
                  T *g,
                  T *b)
{
    // https://en.wikipedia.org/wiki/Academy_Color_Encoding_System#Converting_ACES_RGB_values_to_CIE_XYZ_values
    // and
    // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    *r =  1.0498110175 * x +  0.0000000000 * y + -0.0000974845 * z;
    *g = -0.4959030231 * x +  1.3733130458 * y +  0.0982400361 * z;
    *b =  0.0000000000 * x +  0.0000000000 * y +  0.9912520182 * z;
}

// r,g,b values are from 0 to 1
// Convert pixel values from RGB_ACES_AP1 to XYZ color spaces.
// Uses the ACES white point (approx. D60).
template<typename T>
T rgbACESAP1_to_y( T r, T g, T b )
{
    return 0.2722287168 * r +  0.6740817658 * g +  0.0536895174 * b;
}

template<typename T>
void
rgbACESAP1_to_xyz(T r,
                  T g,
                  T b,
                  T *x,
                  T *y,
                  T *z)
{
    // https://en.wikipedia.org/wiki/Academy_Color_Encoding_System#Converting_ACES_RGB_values_to_CIE_XYZ_values
    // and
    // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    *x =  0.6624541811 * r +  0.1340042065 * g +  0.1561876870 * b;
    *y =  rgbACESAP1_to_y(r, g, b);
    *z = -0.0055746495 * r +  0.0040607335 * g +  1.0103391003 * b;
}

// Convert pixel values from XYZ to RGB_ACES_AP1 color spaces.
// Uses the ACES white point (approx. D60).
template<typename T>
void
xyz_to_rgbACESAP1(T x,
                  T y,
                  T z,
                  T *r,
                  T *g,
                  T *b)
{
    // https://en.wikipedia.org/wiki/Academy_Color_Encoding_System#Converting_ACES_RGB_values_to_CIE_XYZ_values
    // and
    // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    *r =  1.6410233797 * x + -0.3248032942 * y + -0.2364246952 * z;
    *g = -0.6636628587 * x +  1.6153315917 * y +  0.0167563477 * z;
    *b =  0.0117218943 * x + -0.0082844420 * y +  0.9883948585 * z;
}

void xyz_to_lab( float x, float y, float z, float *l, float *a, float *b );
void lab_to_xyz( float l, float a, float b, float *x, float *y, float *z );

void rgb709_to_lab( float r, float g, float b, float *l, float *a, float *b_ );
void lab_to_rgb709( float l, float a, float b, float *r, float *g, float *b_ );


// an object that holds precomputed LUTs for the whole application.
// The LutManager object should be constructed in the plugin factory's load() function, and destructed in the unload() function
// Luts are allocated on request, and destructed either on request, or when the LutManager is destroyed
template <class MUTEX>
class LutManager
{
    typedef OFX::MultiThread::AutoMutexT<MUTEX> AutoMutex;

    typedef std::map<std::string, const Lut* > LutsMap;

public:
    LutManager()
    : _lock()
    , _luts()
    {
    }

    ~LutManager()
    {
        for (typename LutsMap::iterator it = _luts.begin(); it != _luts.end(); ++it) {
            delete it->second;
        }
    }

    /**
     * @brief Returns a pointer to a lut with the given name and the given from and to functions.
     * If a lut with the same name didn't already exist, then it will create one.
     * Ownership of the returned pointer remains to the LutManager.
     * You must release the lut when you are done using it.
     * @WARNING: Not thread-safe. You should call it in the load() action of your plug-in
     **/
    const Lut* getLut(const std::string & name,
                                 fromColorSpaceFunctionV1 fromFunc,
                                 toColorSpaceFunctionV1 toFunc)
    {
        AutoMutex l(_lock);
        typename LutsMap::iterator found = _luts.find(name);

        if ( found != _luts.end() ) {

            return found->second;
        } else {
            Lut* lut = new Lut(name, fromFunc, toFunc);;
            //lut->validate();
            _luts[name] = lut;

            return lut;
        }

        return NULL;
    }

    /**
     * @brief Release a lut previously retrieved with getLut()
     **/
    void releaseLut(const std::string& name)
    {
        AutoMutex l(_lock);
        typename LutsMap::iterator found = _luts.find(name);
        if ( found != _luts.end() ) {
            delete found->second;
            _luts.erase(found);
        }
    }

    ///buit-ins color-spaces
    const Lut* linearLut()
    {
        return getLut("Linear", from_func_linear, to_func_linear);
    }

    const Lut* sRGBLut()
    {
        return getLut("sRGB", from_func_srgb, to_func_srgb);
    }

    const Lut* Rec709Lut()
    {
        return getLut("Rec709", from_func_Rec709, to_func_Rec709);
    }

    const Lut* CineonLut()
    {
        return getLut("Cineon", from_func_Cineon, to_func_Cineon);
    }

    const Lut* Gamma1_8Lut()
    {
        return getLut("Gamma1_8", from_func_Gamma1_8, to_func_Gamma1_8);
    }

    const Lut* Gamma2_2Lut()
    {
        return getLut("Gamma2_2", from_func_Gamma2_2, to_func_Gamma2_2);
    }

    const Lut* PanalogLut()
    {
        return getLut("Panalog", from_func_Panalog, to_func_Panalog);
    }

    const Lut* ViperLogLut()
    {
        return getLut("ViperLog", from_func_ViperLog, to_func_ViperLog);
    }

    const Lut* REDLogLut()
    {
        return getLut("REDLog", from_func_REDLog, to_func_REDLog);
    }

    const Lut* AlexaV3LogCLut()
    {
        return getLut("AlexaV3LogC", from_func_AlexaV3LogC, to_func_AlexaV3LogC);
    }

private:
    LutManager &operator= (const LutManager &);
    LutManager(const LutManager &);


    mutable MUTEX _lock;                 ///< protects _luts
    LutsMap _luts;
};

}         //namespace Color
}     //namespace OFX

/*
 RGB reference primaries.
 
 more can be found at https://github.com/colour-science/colour/tree/develop/colour/models/rgb/dataset

 ACES-Gamut
 http://www.digitalpreservation.gov/partners/documents/IIF_Overview_August_2010.pdf
 -      |  CIE x  |  CIE y
 Red    | 0.73470 | 0.26530
 Green  | 0.00000 | 1.00000
 Blue   | 0.00010 | -0.07700
 White  | 0.32168 | 0.33767 (approx. D60)
 
 Sony S-Gamut3.Cine
 http://www.sony.co.uk/pro/support/attachment/1237494271390/1237494271406/technical-summary-for-s-gamut3-cine-s-log3-and-s-gamut3-s-log3.pdf
 -      |  CIE x  |  CIE y
 Red    | 0.76600 | 0.27500
 Green  | 0.22500 | 0.80000
 Blue   | 0.08900 | -0.08700
 White  | 0.31270 | 0.32900 (D65)

 Sony S-Gamut/S-Gamut3
 http://www.sony.co.uk/pro/support/attachment/1237494271390/1237494271406/technical-summary-for-s-gamut3-cine-s-log3-and-s-gamut3-s-log3.pdf
 -      |  CIE x  |  CIE y
 Red    | 0.73000 | 0.28000
 Green  | 0.14000 | 0.85500
 Blue   | 0.10000 | -0.05000
 White  | 0.31270 | 0.32900 (D65)

 DCI-P3
 http://www.sony.co.uk/pro/support/attachment/1237494271390/1237494271406/technical-summary-for-s-gamut3-cine-s-log3-and-s-gamut3-s-log3.pdf
 -      |  CIE x  |  CIE y
 Red    | 0.68000 | 0.32000
 Green  | 0.26500 | 0.69000
 Blue   | 0.15000 | 0.06000
 White  | 0.31400 | 0.35100 (DCI)
 
 sRGB/Rec709
 http://en.wikipedia.org/wiki/SRGB
 -      |  CIE x  |  CIE y
 Red    | 0.64000 | 0.33000
 Green  | 0.30000 | 0.60000
 Blue   | 0.15000 | 0.06000
 White  | 0.31270 | 0.32900 (D65)

 */

#endif // ifndef openfx_supportext_ofxsLut_h
