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
 * OFX mipmapping help functions
 */

#ifndef openfx_supportext_ofxsMipmap_h
#define openfx_supportext_ofxsMipmap_h

#include <cmath>
#include <cassert>
#include <vector>

#include "ofxsImageEffect.h"

namespace OFX {
void ofxsScalePixelData(OFX::ImageEffect* instance,
                        const OfxRectI & originalRenderWindow,
                        const OfxRectI & renderWindow,
                        unsigned int levels,
                        const void* srcPixelData,
                        OFX::PixelComponentEnum srcPixelComponents,
                        OFX::BitDepthEnum srcPixelDepth,
                        const OfxRectI & srcBounds,
                        int srcRowBytes,
                        void* dstPixelData,
                        OFX::PixelComponentEnum dstPixelComponents,
                        OFX::BitDepthEnum dstPixelDepth,
                        const OfxRectI & dstBounds,
                        int dstRowBytes);

struct MipMap
{
    std::size_t memSize;
    OFX::ImageMemory* data;
    OfxRectI bounds;

    MipMap()
        : memSize(0)
        , data(0)
        , bounds()
    {
    }

    ~MipMap()
    {
        delete data;
        data = 0;
    }
};

//Contains all levels of details > 0, sort by decreasing LoD
typedef std::vector<MipMap> MipMapsVector;

/**
   @brief Given the original image, this function builds all mipmap levels
   up to maxLevel and stores them in the mipmaps vector, in decreasing LoD.
   The original image will not be stored in the mipmaps vector.
   @param mipmaps[out] The mipmaps vector should contains at least maxLevel
   entries
 **/
void ofxsBuildMipMaps(OFX::ImageEffect* instance,
                      const OfxRectI & renderWindow,
                      const void* srcPixelData,
                      OFX::PixelComponentEnum srcPixelComponents,
                      OFX::BitDepthEnum srcPixelDepth,
                      const OfxRectI & srcBounds,
                      int srcRowBytes,
                      unsigned int maxLevel,
                      MipMapsVector & mipmaps);
} // OFX

#endif // ifndef openfx_supportext_ofxsMipmap_h
