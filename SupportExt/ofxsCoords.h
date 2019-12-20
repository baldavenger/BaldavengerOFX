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
 * OFX Coords helpers
 */

#ifndef openfx_supportext_ofxsCoords_h
#define openfx_supportext_ofxsCoords_h

#include <cmath>
#include <cfloat>
#include <algorithm>

#include "ofxsImageEffect.h"

#ifndef M_LN2
#define M_LN2       0.693147180559945309417232121458176568  /* loge(2)        */
#endif

namespace OFX {
namespace Coords {
template <typename Rect>
bool
rectIsEmpty(const Rect & r)
{
    return (r.x2 <= r.x1) || (r.y2 <= r.y1);
}

/// Bounding box of two rectangles
/// bbox may be aliased to a or b
template <typename Rect>
void
rectBoundingBox(const Rect & a,
                const Rect & b,
                Rect* bbox)
{
    if ( rectIsEmpty(a) ) {
        *bbox = b;

        return;
    }
    if ( rectIsEmpty(b) ) {
        *bbox = a;

        return;
    }
    bbox->x1 = std::min(a.x1, b.x1);
    bbox->x2 = std::max( bbox->x1, std::max(a.x2, b.x2) );
    bbox->y1 = std::min(a.y1, b.y1);
    bbox->y2 = std::max( bbox->y1, std::max(a.y2, b.y2) );
}

template <typename Rect>
bool
rectIsInfinite(const Rect & r)
{
    return (r.x1 <= kOfxFlagInfiniteMin) || (r.x2 >= kOfxFlagInfiniteMax) ||
           (r.y1 <= kOfxFlagInfiniteMin) || (r.y2 >= kOfxFlagInfiniteMax);
}

/// compute the intersection of two rectangles, and return true if they intersect
/// intersection may be aliased to r1 or r2
template <typename Rect>
bool
rectIntersection(const Rect & r1,
                 const Rect & r2,
                 Rect* intersection)
{
    if ( rectIsEmpty(r1) || rectIsEmpty(r2) ) {
        if (intersection) {
            intersection->x1 = 0;
            intersection->x2 = 0;
            intersection->y1 = 0;
            intersection->y2 = 0;
        }

        return false;
    }

    if ( ( r1.x1 > r2.x2) || ( r2.x1 > r1.x2) || ( r1.y1 > r2.y2) || ( r2.y1 > r1.y2) ) {
        if (intersection) {
            intersection->x1 = 0;
            intersection->x2 = 0;
            intersection->y1 = 0;
            intersection->y2 = 0;
        }

        return false;
    }

    if (intersection) {
        intersection->x1 = std::max(r1.x1, r2.x1);
        // the region must be *at least* empty, thus the maximin.
        intersection->x2 = std::max( intersection->x1, std::min(r1.x2, r2.x2) );
        intersection->y1 = std::max(r1.y1, r2.y1);
        // the region must be *at least* empty, thus the maximin.
        intersection->y2 = std::max( intersection->y1, std::min(r1.y2, r2.y2) );
    }

    return true;
}

/**
 * @brief Scales down the rectangle in pixel coordinates by the given power of 2, and return the smallest *enclosing* rectangle in pixel coordinates
 * Never use this with canonical coordinates, or never round canonical coordinates to use this: use toPixelEnclosing instead.
 **/
/*inline
OfxRectI
downscalePowerOfTwoSmallestEnclosing(const OfxRectI & r,
                                     unsigned int thisLevel)
{
    if (thisLevel == 0) {
        return r;
    }
    OfxRectI ret;
    int pot = (1 << thisLevel);
    int pot_minus1 = pot - 1;
    if (r.x1 <= kOfxFlagInfiniteMin) {
        ret.x1 = kOfxFlagInfiniteMin;
    } else {
        ret.x1 = r.x1 >> thisLevel;
        assert(ret.x1 * pot <= r.x1);
    }
    if (r.x2 >= kOfxFlagInfiniteMax) {
        ret.x2 = kOfxFlagInfiniteMax;
    } else {
        ret.x2 = (r.x2 + pot_minus1) >> thisLevel;
        assert(ret.x2 * pot >= r.x2);
    }
    if (r.y1 <= kOfxFlagInfiniteMin) {
        ret.y1 = kOfxFlagInfiniteMin;
    } else {
        ret.y1 = r.y1 >> thisLevel;
        assert(ret.y1 * pot <= r.y1);
    }
    if (r.y2 >= kOfxFlagInfiniteMax) {
        ret.y2 = kOfxFlagInfiniteMax;
    } else {
        ret.y2 = (r.y2 + pot_minus1) >> thisLevel;
        assert(ret.y2 * pot >= r.y2);
    }

    return ret;
}
*/
inline
double
scaleFromMipmapLevel(unsigned int level)
{
    return 1. / (1 << level);
}

inline void
toPixelEnclosing(const OfxRectD & regionOfInterest,
                 const OfxPointD & renderScale,
                 double par,
                 OfxRectI *rect)
{
    assert(par);
    if ( rectIsEmpty(regionOfInterest) ) {
        rect->x1 = rect->y1 = rect->x2 = rect->y2 = 0;

        return;
    }
    rect->x1 = (int)std::floor(regionOfInterest.x1 * renderScale.x / par);
    rect->y1 = (int)std::floor(regionOfInterest.y1 * renderScale.y);
    rect->x2 = (int)std::ceil(regionOfInterest.x2 * renderScale.x / par);
    rect->y2 = (int)std::ceil(regionOfInterest.y2 * renderScale.y);
}

inline void
toPixelNearest(const OfxRectD & regionOfInterest,
               const OfxPointD & renderScale,
               double par,
               OfxRectI *rect)
{
    assert(par);
    if ( rectIsEmpty(regionOfInterest) ) {
        rect->x1 = rect->y1 = rect->x2 = rect->y2 = 0;

        return;
    }
    rect->x1 = (int)std::floor(regionOfInterest.x1 * renderScale.x / par + 0.5);
    rect->y1 = (int)std::floor(regionOfInterest.y1 * renderScale.y + 0.5);
    rect->x2 = (int)std::ceil(regionOfInterest.x2 * renderScale.x / par - 0.5);
    rect->y2 = (int)std::ceil(regionOfInterest.y2 * renderScale.y - 0.5);
}

inline void
toPixel(const OfxPointD & p_canonical,
        const OfxPointD & renderScale,
        double par,
        OfxPointI *p_pixel)
{
    assert(par);
    p_pixel->x = (int)std::floor(p_canonical.x * renderScale.x / par);
    p_pixel->y = (int)std::floor(p_canonical.y * renderScale.y);
}

// subpixel version (no rounding)
inline void
toPixelSub(const OfxPointD & p_canonical,
           const OfxPointD & renderScale,
           double par,
           OfxPointD *p_pixel)
{
    assert(par);
    p_pixel->x = p_canonical.x * renderScale.x / par - 0.5;
    p_pixel->y = p_canonical.y * renderScale.y - 0.5;
}

// transforms the middle of the given pixel to canonical coordinates
inline void
toCanonical(const OfxPointI & p_pixel,
            const OfxPointD & renderScale,
            double par,
            OfxPointD *p_canonical)
{
    assert(par);
    p_canonical->x = (p_pixel.x + 0.5) * par / renderScale.x;
    p_canonical->y = (p_pixel.y + 0.5) / renderScale.y;
}

// subpixel version (no rounding)
inline void
toCanonicalSub(const OfxPointD & p_pixel,
               const OfxPointD & renderScale,
               double par,
               OfxPointD *p_canonical)
{
    assert(par);
    p_canonical->x = (p_pixel.x + 0.5) * par / renderScale.x;
    p_canonical->y = (p_pixel.y + 0.5) / renderScale.y;
}

inline void
toCanonical(const OfxRectI & rect,
            const OfxPointD & renderScale,
            double par,
            OfxRectD *regionOfInterest)
{
    assert(par);
    if ( rectIsEmpty(rect) ) {
        regionOfInterest->x1 = regionOfInterest->y1 = regionOfInterest->x2 = regionOfInterest->y2 = 0;

        return;
    }
    regionOfInterest->x1 = rect.x1 * par / renderScale.x;
    regionOfInterest->y1 = rect.y1 / renderScale.y;
    regionOfInterest->x2 = rect.x2 * par / renderScale.x;
    regionOfInterest->y2 = rect.y2 / renderScale.y;
}

inline void
toCanonical(const OfxRectD & rect,
            const OfxPointD & renderScale,
            double par,
            OfxRectD *regionOfInterest)
{
    assert(par);
    if ( rectIsEmpty(rect) ) {
        regionOfInterest->x1 = regionOfInterest->y1 = regionOfInterest->x2 = regionOfInterest->y2 = 0;

        return;
    }
    regionOfInterest->x1 = rect.x1 * par / renderScale.x;
    regionOfInterest->y1 = rect.y1 / renderScale.y;
    regionOfInterest->x2 = rect.x2 * par / renderScale.x;
    regionOfInterest->y2 = rect.y2 / renderScale.y;
}

inline
unsigned int
mipmapLevelFromScale(double s)
{
    assert(0. < s && s <= 1.);
    int retval = -(int)std::floor(std::log(s) / M_LN2 + 0.5);
    assert(retval >= 0);

    return retval;
}
} // Coords
} // OFX


#endif // openfx_supportext_ofxsCoords_h
