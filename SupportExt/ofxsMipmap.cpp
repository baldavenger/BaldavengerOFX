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

#include "ofxsMipMap.h"

namespace OFX {
// update the window of dst defined by dstRoI by halving the corresponding area in src.
// proofread and fixed by F. Devernay on 3/10/2014
template <typename PIX, int nComponents>
static void
halveWindow(const OfxRectI & dstRoI,
            const PIX* srcPixels,
            const OfxRectI & srcBounds,
            int srcRowBytes,
            PIX* dstPixels,
            const OfxRectI & dstBounds,
            int dstRowBytes)
{
    assert(srcPixels && dstPixels);
    if (!srcPixels || !dstPixels) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    assert(dstRoI.x1 * 2 >= (srcBounds.x1 - 1) && (dstRoI.x2 - 1) * 2 < srcBounds.x2 &&
           dstRoI.y1 * 2 >= (srcBounds.y1 - 1) && (dstRoI.y2 - 1) * 2 < srcBounds.y2);
    int srcRowSize = srcRowBytes / sizeof(PIX);
    int dstRowSize = dstRowBytes / sizeof(PIX);

    // offset pointers so that srcData and dstData correspond to pixel (0,0)
    const PIX* const srcData = srcPixels - (srcBounds.x1 * nComponents + srcRowSize * srcBounds.y1);
    PIX* const dstData       = dstPixels - (dstBounds.x1 * nComponents + dstRowSize * dstBounds.y1);

    for (int y = dstRoI.y1; y < dstRoI.y2; ++y) {
        const PIX* const srcLineStart    = srcData + y * 2 * srcRowSize;
        PIX* const dstLineStart          = dstData + y     * dstRowSize;

        // The current dst row, at y, covers the src rows y*2 (thisRow) and y*2+1 (nextRow).
        // Check that if are within srcBounds.
        int srcy = y * 2;
        bool pickThisRow = srcBounds.y1 <= (srcy + 0) && (srcy + 0) < srcBounds.y2;
        bool pickNextRow = srcBounds.y1 <= (srcy + 1) && (srcy + 1) < srcBounds.y2;
        const int sumH = (int)pickNextRow + (int)pickThisRow;
        assert(sumH == 1 || sumH == 2);

        for (int x = dstRoI.x1; x < dstRoI.x2; ++x) {
            const PIX* const srcPixStart    = srcLineStart   + x * 2 * nComponents;
            PIX* const dstPixStart          = dstLineStart   + x * nComponents;

            // The current dst col, at y, covers the src cols x*2 (thisCol) and x*2+1 (nextCol).
            // Check that if are within srcBounds.
            int srcx = x * 2;
            bool pickThisCol = srcBounds.x1 <= (srcx + 0) && (srcx + 0) < srcBounds.x2;
            bool pickNextCol = srcBounds.x1 <= (srcx + 1) && (srcx + 1) < srcBounds.x2;
            const int sumW = (int)pickThisCol + (int)pickNextCol;
            assert(sumW == 1 || sumW == 2);
            const int sum = sumW * sumH;
            assert(0 < sum && sum <= 4);

            for (int k = 0; k < nComponents; ++k) {
                ///a b
                ///c d

                const PIX a = (pickThisCol && pickThisRow) ? *(srcPixStart + k) : 0;
                const PIX b = (pickNextCol && pickThisRow) ? *(srcPixStart + k + nComponents) : 0;
                const PIX c = (pickThisCol && pickNextRow) ? *(srcPixStart + k + srcRowSize) : 0;
                const PIX d = (pickNextCol && pickNextRow) ? *(srcPixStart + k + srcRowSize  + nComponents)  : 0;

                assert( sumW == 2 || ( sumW == 1 && ( (a == 0 && c == 0) || (b == 0 && d == 0) ) ) );
                assert( sumH == 2 || ( sumH == 1 && ( (a == 0 && b == 0) || (c == 0 && d == 0) ) ) );
                dstPixStart[k] = (a + b + c + d) / sum;
            }
        }
    }
} // halveWindow

// update the window of dst defined by originalRenderWindow by mipmapping the windows of src defined by renderWindowFullRes
// proofread and fixed by F. Devernay on 3/10/2014
template <typename PIX, int nComponents>
static void
buildMipMapLevel(ImageEffect* instance,
                 const OfxRectI & originalRenderWindow,
                 const OfxRectI & renderWindowFullRes,
                 unsigned int level,
                 const PIX* srcPixels,
                 const OfxRectI & srcBounds,
                 int srcRowBytes,
                 PIX* dstPixels,
                 const OfxRectI & dstBounds,
                 int dstRowBytes)
{
    assert(level > 0);
    assert(srcPixels && dstPixels);
    if (!srcPixels || !dstPixels) {
        throwSuiteStatusException(kOfxStatFailed);
    }

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
        size_t newMemSize =  (nextRenderWindow.y2 - nextRenderWindow.y1) * nextRowBytes;
        if ( tmpMem.get() ) {
            // there should be enough memory: no need to reallocate
            assert(tmpMemSize >= memSize);
        } else {
            tmpMem.reset( new ImageMemory(newMemSize, instance) );
            tmpMemSize = newMemSize;
        }
        nextImg = (float*)tmpMem->lock();

        halveWindow<PIX, nComponents>(nextRenderWindow, previousImg, previousBounds, previousRowBytes, nextImg, nextRenderWindow, nextRowBytes);

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

    halveWindow<PIX, nComponents>(nextRenderWindow, previousImg, previousBounds, previousRowBytes, dstPixels, dstBounds, dstRowBytes);
    // mem and tmpMem are freed at destruction
} // buildMipMapLevel

void
ofxsScalePixelData(ImageEffect* instance,
                   const OfxRectI & originalRenderWindow,
                   const OfxRectI & renderWindow,
                   unsigned int levels,
                   const void* srcPixelData,
                   PixelComponentEnum srcPixelComponents,
                   BitDepthEnum srcPixelDepth,
                   const OfxRectI & srcBounds,
                   int srcRowBytes,
                   void* dstPixelData,
                   PixelComponentEnum dstPixelComponents,
                   BitDepthEnum dstPixelDepth,
                   const OfxRectI & dstBounds,
                   int dstRowBytes)
{
    assert(srcPixelData && dstPixelData);
    if (!srcPixelData || !dstPixelData) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    // do the rendering
    if ( ( dstPixelDepth != eBitDepthFloat) ||
         ( ( dstPixelComponents != ePixelComponentRGBA) &&
           ( dstPixelComponents != ePixelComponentRGB) &&
           ( dstPixelComponents != ePixelComponentAlpha) ) ||
         ( dstPixelDepth != srcPixelDepth) ||
         ( dstPixelComponents != srcPixelComponents) ) {
        throwSuiteStatusException(kOfxStatErrFormat);
    }

    if (dstPixelComponents == ePixelComponentRGBA) {
        buildMipMapLevel<float, 4>(instance, originalRenderWindow, renderWindow, levels, (const float*)srcPixelData,
                                   srcBounds, srcRowBytes, (float*)dstPixelData, dstBounds, dstRowBytes);
    } else if (dstPixelComponents == ePixelComponentRGB) {
        buildMipMapLevel<float, 3>(instance, originalRenderWindow, renderWindow, levels, (const float*)srcPixelData,
                                   srcBounds, srcRowBytes, (float*)dstPixelData, dstBounds, dstRowBytes);
    }  else if (dstPixelComponents == ePixelComponentAlpha) {
        buildMipMapLevel<float, 1>(instance, originalRenderWindow, renderWindow, levels, (const float*)srcPixelData,
                                   srcBounds, srcRowBytes, (float*)dstPixelData, dstBounds, dstRowBytes);
    }     // switch
}

template <typename PIX, int nComponents>
static void
ofxsBuildMipMapsForComponents(ImageEffect* instance,
                              const OfxRectI & renderWindow,
                              const PIX* srcPixelData,
                              const OfxRectI & srcBounds,
                              int srcRowBytes,
                              unsigned int maxLevel,
                              MipMapsVector & mipmaps)
{
    assert(srcPixelData);
    if (!srcPixelData) {
        throwSuiteStatusException(kOfxStatFailed);
    }
    const PIX* previousImg = srcPixelData;
    OfxRectI previousBounds = srcBounds;
    int previousRowBytes = srcRowBytes;
    OfxRectI nextRenderWindow = renderWindow;

    ///Build all the mipmap levels until we reach the one we are interested in
    for (unsigned int i = 1; i <= maxLevel; ++i) {
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
        assert(i - 1 >= 0);

        ///Allocate a temporary image if necessary, or reuse the previously allocated buffer
        int nextRowBytes = (nextRenderWindow.x2 - nextRenderWindow.x1)  * nComponents * sizeof(PIX);
        mipmaps[i - 1].memSize = (nextRenderWindow.y2 - nextRenderWindow.y1) * nextRowBytes;
        mipmaps[i - 1].bounds = nextRenderWindow;

        mipmaps[i - 1].data = new ImageMemory(mipmaps[i - 1].memSize, instance);
        tmpMemSize = newMemSize;

        float* nextImg = (float*)tmpMem->lock();

        halveWindow<PIX, nComponents>(nextRenderWindow, previousImg, previousBounds, previousRowBytes, nextImg, nextRenderWindow, nextRowBytes);

        ///Switch for next pass
        previousBounds = nextRenderWindow;
        previousRowBytes = nextRowBytes;
        previousImg = nextImg;
    }
}

void
ofxsBuildMipMaps(ImageEffect* instance,
                 const OfxRectI & renderWindow,
                 const void* srcPixelData,
                 PixelComponentEnum srcPixelComponents,
                 BitDepthEnum srcPixelDepth,
                 const OfxRectI & srcBounds,
                 int srcRowBytes,
                 unsigned int maxLevel,
                 MipMapsVector & mipmaps)
{
    assert(srcPixelData && mipmaps->size() == maxLevel);
    if ( !srcPixelData || (mipmaps->size() != maxLevel) ) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    // do the rendering
    if ( srcPixelData && ( ( srcPixelDepth != eBitDepthFloat) ||
                           ( ( srcPixelComponents != ePixelComponentRGBA) &&
                             ( srcPixelComponents != ePixelComponentRGB) &&
                             ( srcPixelComponents != ePixelComponentAlpha) ) ) ) {
        throwSuiteStatusException(kOfxStatErrFormat);
    }

    if (dstPixelComponents == ePixelComponentRGBA) {
        ofxsBuildMipMapsForComponents<float, 4>(instance, renderWindow, srcPixelData, srcBounds,
                                                srcRowBytes, maxLevel, mipmaps);
    } else if (dstPixelComponents == ePixelComponentRGB) {
        ofxsBuildMipMapsForComponents<float, 3>(instance, renderWindow, srcPixelData, srcBounds,
                                                srcRowBytes, maxLevel, mipmaps);
    }  else if (dstPixelComponents == ePixelComponentAlpha) {
        ofxsBuildMipMapsForComponents<float, 1>(instance, renderWindow, srcPixelData, srcBounds,
                                                srcRowBytes, maxLevel, mipmaps);
    }
}
} // OFX
