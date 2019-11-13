/*
   OFX I/O utility functions.
   Adds OpenColorIO functionality to any plugin.

   Copyright (C) 2014 INRIA
   Author: Frederic Devernay <frederic.devernay@inria.fr>

   Redistribution and use in source and binary forms, with or without modification,
   are permitted provided that the following conditions are met:

   Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

   Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or
   other materials provided with the distribution.

   Neither the name of the {organization} nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   INRIA
   Domaine de Voluceau
   Rocquencourt - B.P. 105
   78153 Le Chesnay Cedex - France

 */

#ifndef IO_Utility_h
#define IO_Utility_h


#include <cmath>
#include <cassert>
#include <algorithm>
#include <string>
#include <functional>
#include <locale>

#include "ofxsImageEffect.h"

#define NAMESPACE_OFX_ENTER namespace OFX {
#define NAMESPACE_OFX_EXIT }

#define NAMESPACE_OFX_IO_ENTER namespace IO {
#define NAMESPACE_OFX_IO_EXIT }


NAMESPACE_OFX_ENTER
NAMESPACE_OFX_IO_ENTER

inline std::string
basename( std::string const& pathname )
{
#if defined(_WIN32) || defined(WIN64)
    std::size_t found = pathname.find_last_of("/\\");
#else
    std::size_t found = pathname.find_last_of("/");
#endif

    return pathname.substr(found + 1);
}

inline std::string
dirname( std::string const& pathname )
{
#if defined(_WIN32) || defined(WIN64)
    std::size_t found = pathname.find_last_of("/\\");
#else
    std::size_t found = pathname.find_last_of("/");
#endif

    return pathname.substr(0, found);
}

inline std::string
extension(const std::string& filename)
{
    std::string::const_reverse_iterator pivot = std::find( filename.rbegin(), filename.rend(), '.' );
    if ( pivot == filename.rend() ) {
        return "";
    }
    std::string ext;
    std::locale loc;
    for (std::string::const_iterator it = pivot.base(); it != filename.end(); ++it) {
        ext.append( 1, std::tolower(*it, loc) );
    }

    return ext;
}

/// numvals should be 256 for byte, 65536 for 16-bits, etc.
template<int numvals>
float
intToFloat(int value)
{
    return value / (float)(numvals - 1);
}

template<int numvals>
int
floatToInt(float value)
{
    if (value <= 0) {
        return 0;
    } else if (value >= 1.) {
        return numvals - 1;
    }

    return (int)(value * (numvals - 1) + 0.5);
}

/**
 * @brief Upscales the bounds assuming this rectangle is the Nth level of mipmap
 **/
inline OfxRectI
upscalePowerOfTwo(const OfxRectI& r,
                  unsigned int thisLevel)
{
    if (thisLevel == 0) {
        return r;
    }
    OfxRectI ret;
    ret.x1 = r.x1 << thisLevel;
    ret.x2 = r.x2 << thisLevel;
    ret.y1 = r.y1 << thisLevel;
    ret.y2 = r.y2 << thisLevel;

    return ret;
}

/**
 * @brief Upscales the bounds assuming this rectangle is the Nth level of mipmap
 **/
inline OfxRectD
upscalePowerOfTwo(const OfxRectD& r,
                  double thisLevel)
{
    if (thisLevel == 0) {
        return r;
    }
    OfxRectD ret;
    ret.x1 = std::pow(r.x1, thisLevel);
    ret.x2 = std::pow(r.x2, thisLevel);
    ret.y1 = std::pow(r.y1, thisLevel);
    ret.y2 = std::pow(r.y2, thisLevel);

    return ret;
}

/**
 * @brief Scales down the rectangle by the given power of 2
 **/
inline OfxRectI
downscalePowerOfTwo(const OfxRectI& r,
                    unsigned int thisLevel)
{
    if (thisLevel == 0) {
        return r;
    }
    OfxRectI ret;
    assert(r.x1 % (1 << thisLevel) == 0 && r.x2 % (1 << thisLevel) == 0 && r.y1 % (1 << thisLevel) == 0 && r.y2 % (1 << thisLevel) == 0);
    ret.x1 = r.x1 >> thisLevel;
    ret.x2 = r.x2 >> thisLevel;
    ret.y1 = r.y1 >> thisLevel;
    ret.y2 = r.y2 >> thisLevel;

    return ret;
}

inline bool
isRectNull(const OfxRectI& r)
{
    return (r.x2 <= r.x1) || (r.y2 <= r.y1);
}

inline bool
intersect(const OfxRectI& r1,
          const OfxRectI& r2,
          OfxRectI* intersection)
{
    if ( isRectNull(r1) || isRectNull(r2) ) {
        return false;
    }

    if ( (r1.x1 > r2.x2) || (r2.x1 > r1.x2) || (r1.y1 > r2.y2) || (r2.y1 > r1.y2) ) {
        return false;
    }

    intersection->x1 = std::max(r1.x1, r2.x1);
    intersection->x2 = std::min(r1.x2, r2.x2);
    intersection->y1 = std::max(r1.y1, r2.y1);
    intersection->y2 = std::min(r1.y2, r2.y2);

    return true;
}

/*
   test program for rounding integer to the next/previous pot:
 #include <stdio.h>
   int main()
   {
   int i;
   int pot = 3;
   int scale = 1 << pot;
   int scalem1 = scale - 1;
   for(i=-100; i<100; ++i)
   {
   printf("%d => %d,%d %d,%d\n", i, i & ~scalem1, i+scalem1 & ~scalem1, (i >> pot) << pot, ((i+scalem1)>>pot) << pot);
   }
   }
 */
/**
 * @brief round the rectangle by the given power of 2, and return the largest *enclosed* (inside) rectangle
 **/
inline OfxRectI
roundPowerOfTwoLargestEnclosed(const OfxRectI& r,
                               unsigned int thisLevel)
{
    if (thisLevel == 0) {
        return r;
    }
    OfxRectI ret;
    int pot = (1 << thisLevel);
    int pot_minus1 = pot - 1;
    ret.x1 = (r.x1 + pot_minus1) & ~pot_minus1;
    ret.x2 = r.x2 & ~pot_minus1;
    ret.y1 = (r.y1 + pot_minus1) & ~pot_minus1;
    ret.y2 = r.y2 & ~pot_minus1;
    // check that it's enclosed
    assert(ret.x1 >= r.x1 && ret.x2 <= r.x2 && ret.y1 >= r.y1 && ret.y2 <= r.y2);

    return ret;
}

/**
 * @brief round the rectangle by the given power of 2, and return the smallest *enclosing* rectangle
 **/
inline OfxRectI
roundPowerOfTwoSmallestEnclosing(const OfxRectI& r,
                                 unsigned int thisLevel)
{
    if (thisLevel == 0) {
        return r;
    }
    OfxRectI ret;
    int pot = (1 << thisLevel);
    int pot_minus1 = pot - 1;
    ret.x1 = r.x1 & ~pot_minus1;
    ret.x2 = (r.x2 + pot_minus1) & ~pot_minus1;
    ret.y1 = r.y1 & ~pot_minus1;
    ret.y2 = (r.y2 + pot_minus1) & ~pot_minus1;
    // check that it's enclosing
    assert(ret.x1 <= r.x1 && ret.x2 >= r.x2 && ret.y1 <= r.y1 && ret.y2 >= r.y2);

    return ret;
}

/**
 * @brief Scales down the rectangle by the given power of 2, and return the largest *enclosed* (inside) rectangle
 **/
inline OfxRectI
downscalePowerOfTwoLargestEnclosed(const OfxRectI& r,
                                   unsigned int thisLevel)
{
    if (thisLevel == 0) {
        return r;
    }
    OfxRectI ret;
    int pot = (1 << thisLevel);
    int pot_minus1 = pot - 1;
    ret.x1 = (r.x1 + pot_minus1) >> thisLevel;
    ret.x2 = r.x2 >> thisLevel;
    ret.y1 = (r.y1 + pot_minus1) >> thisLevel;
    ret.y2 = r.y2 >> thisLevel;
    // check that it's enclosed
    assert(ret.x1 * pot >= r.x1 && ret.x2 * pot <= r.x2 && ret.y1 * pot >= r.y1 && ret.y2 * pot <= r.y2);

    return ret;
}

/**
 * @brief Scales down the rectangle by the given power of 2, and return the smallest *enclosing* rectangle
 **/
inline OfxRectI
downscalePowerOfTwoSmallestEnclosing(const OfxRectI& r,
                                     unsigned int thisLevel)
{
    if (thisLevel == 0) {
        return r;
    }
    OfxRectI ret;
    int pot = (1 << thisLevel);
    int pot_minus1 = pot - 1;
    ret.x1 = r.x1 >> thisLevel;
    ret.x2 = (r.x2 + pot_minus1) >> thisLevel;
    ret.y1 = r.y1 >> thisLevel;
    ret.y2 = (r.y2 + pot_minus1) >> thisLevel;
    // check that it's enclosing
    assert(ret.x1 * pot <= r.x1 && ret.x2 * pot >= r.x2 && ret.y1 * pot <= r.y1 && ret.y2 * pot >= r.y2);

    return ret;
}

inline OfxRectI
nextRectLevel(const OfxRectI& r)
{
    OfxRectI ret = r;

    ret.x1 /= 2;
    ret.y1 /= 2;
    ret.x2 /= 2;
    ret.y2 /= 2;

    return ret;
}

inline double
getScaleFromMipMapLevel(unsigned int level)
{
    return 1. / (1 << level);
}

#ifndef M_LN2
#define M_LN2       0.693147180559945309417232121458176568  /* loge(2)        */
#endif
inline unsigned int
getLevelFromScale(double s)
{
    assert(0. < s && s <= 1.);
    int retval = -(int)std::floor(std::log(s) / M_LN2 + 0.5);
    assert(retval >= 0);

    return retval;
}

/**
 * @brief Helper class to make fast buffers that are ensured to be deallocated in a RAII style
 **/
class RamBuffer
{
    unsigned char* data;

public:

    RamBuffer(std::size_t nBytes)
        : data(0)
    {
        data = (unsigned char*)malloc(nBytes);
    }

    unsigned char* getData() const
    {
        return data;
    }

    ~RamBuffer()
    {
        if (data) {
            free(data);
        }
    }
};

NAMESPACE_OFX_IO_EXIT
    NAMESPACE_OFX_EXIT

#endif // ifndef IO_Utility_h
