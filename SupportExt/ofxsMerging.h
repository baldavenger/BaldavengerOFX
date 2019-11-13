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
 * OFX Merge helpers
 */

#ifndef Misc_Merging_helper_h
#define Misc_Merging_helper_h

#include <cmath>
#include <cfloat>
#include <algorithm>

#include "ofxsImageEffect.h"

#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

namespace OFX {
// References:
//
// SVG Compositing Specification:
//   http://www.w3.org/TR/SVGCompositing/
// PDF Reference v1.7:
//   http://www.adobe.com/content/dam/Adobe/en/devnet/acrobat/pdfs/pdf_reference_1-7.pdf
//   http://www.adobe.com/devnet/pdf/pdf_reference_archive.html
// Adobe photoshop blending modes:
//   http://helpx.adobe.com/en/photoshop/using/blending-modes.html
//   http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
// ImageMagick:
//   http://www.imagemagick.org/Usage/compose/
//
// Note about the Soft-Light operation:
// Soft-light as implemented in Nuke comes from the SVG 2004 specification, which is wrong.
// In SVG 2004, 'Soft_Light' did not work as expected, producing a brightening for any non-gray shade
// image overlay.
// It was fixed in the March 2009 SVG specification, which was used for this implementation.

namespace MergeImages2D {
// please keep this long list sorted alphabetically
enum MergingFunctionEnum
{
    eMergeATop = 0,
    eMergeAverage,
    eMergeColor,
    eMergeColorBurn,
    eMergeColorDodge,
    eMergeConjointOver,
    eMergeCopy,
    eMergeDifference,
    eMergeDisjointOver,
    eMergeDivide,
    eMergeExclusion,
    eMergeFreeze,
    eMergeFrom,
    eMergeGeometric,
    eMergeGrainExtract,
    eMergeGrainMerge,
    eMergeHardLight,
    eMergeHue,
    eMergeHypot,
    eMergeIn,
    //eMergeInterpolated,
    eMergeLuminosity,
    eMergeMask,
    eMergeMatte,
    eMergeMax,
    eMergeMin,
    eMergeMinus,
    eMergeMultiply,
    eMergeOut,
    eMergeOver,
    eMergeOverlay,
    eMergePinLight,
    eMergePlus,
    eMergeReflect,
    eMergeSaturation,
    eMergeScreen,
    eMergeSoftLight,
    eMergeStencil,
    eMergeUnder,
    eMergeXOR,
};

inline bool
isMaskable(MergingFunctionEnum operation)
{
    switch (operation) {
    case eMergeAverage:
    case eMergeColorBurn:
    case eMergeColorDodge:
    case eMergeDifference:
    case eMergeDivide:
    case eMergeExclusion:
    case eMergeFrom:
    case eMergeFreeze:
    case eMergeGeometric:
    case eMergeGrainExtract:
    case eMergeGrainMerge:
    case eMergeHardLight:
    case eMergeHypot:
    //case eMergeInterpolated:
    case eMergeMax:
    case eMergeMin:
    case eMergeMinus:
    case eMergeMultiply:
    case eMergeOverlay:
    case eMergePinLight:
    case eMergePlus:
    case eMergeReflect:
    case eMergeSoftLight:

        return true;
    case eMergeATop:
    case eMergeConjointOver:
    case eMergeCopy:
    case eMergeDisjointOver:
    case eMergeIn:
    case eMergeMask:
    case eMergeMatte:
    case eMergeOut:
    case eMergeOver:
    case eMergeScreen:
    case eMergeStencil:
    case eMergeUnder:
    case eMergeXOR:
    case eMergeHue:
    case eMergeSaturation:
    case eMergeColor:
    case eMergeLuminosity:

        return false;
    }

    return true;
} // isMaskable

// is the operator separable for R,G,B components, or do they have to be processed simultaneously?
inline bool
isSeparable(MergingFunctionEnum operation)
{
    switch (operation) {
    case eMergeHue:
    case eMergeSaturation:
    case eMergeColor:
    case eMergeLuminosity:

        return false;

    default:

        return true;
    }
}

inline std::string
getOperationString(MergingFunctionEnum operation)
{
    switch (operation) {
    case eMergeATop:

        return "atop";

    case eMergeAverage:

        return "average";

    case eMergeColor:

        return "color";

    case eMergeColorBurn:

        return "color-burn";

    case eMergeColorDodge:

        return "color-dodge";

    case eMergeConjointOver:

        return "conjoint-over";

    case eMergeCopy:

        return "copy";

    case eMergeDifference:

        return "difference";

    case eMergeDisjointOver:

        return "disjoint-over";

    case eMergeDivide:

        return "divide";

    case eMergeExclusion:

        return "exclusion";

    case eMergeFreeze:

        return "freeze";

    case eMergeFrom:

        return "from";

    case eMergeGeometric:

        return "geometric";

    case eMergeGrainExtract:

        return "grain-extract";

    case eMergeGrainMerge:

        return "grain-merge";

    case eMergeHardLight:

        return "hard-light";

    case eMergeHue:

        return "hue";

    case eMergeHypot:

        return "hypot";

    case eMergeIn:

        return "in";

    //case eMergeInterpolated:
    //    return "interpolated";

    case eMergeLuminosity:

        return "luminosity";

    case eMergeMask:

        return "mask";

    case eMergeMatte:

        return "matte";

    case eMergeMax:

        return "max";

    case eMergeMin:

        return "min";

    case eMergeMinus:

        return "minus";

    case eMergeMultiply:

        return "multiply";

    case eMergeOut:

        return "out";

    case eMergeOver:

        return "over";

    case eMergeOverlay:

        return "overlay";

    case eMergePinLight:

        return "pinlight";

    case eMergePlus:

        return "plus";

    case eMergeReflect:

        return "reflect";

    case eMergeSaturation:

        return "saturation";

    case eMergeScreen:

        return "screen";

    case eMergeSoftLight:

        return "soft-light";

    case eMergeStencil:

        return "stencil";

    case eMergeUnder:

        return "under";

    case eMergeXOR:

        return "xor";
    } // switch

    return "unknown";
} // getOperationString

inline std::string
getOperationDescription(MergingFunctionEnum operation)
{
    switch (operation) {
    case eMergeATop:

        return "Ab + B(1 - a) (a.k.a. src-atop)";

    case eMergeAverage:

        return "(A + B) / 2";

    case eMergeColor:

        return "SetLum(A, Lum(B))";

    case eMergeColorBurn:

        return "darken B towards A";

    case eMergeColorDodge:

        return "brighten B towards A";

    case eMergeConjointOver:

        return "A + B(1-a)/b, A if a > b";

    case eMergeCopy:

        return "A (a.k.a. src)";

    case eMergeDifference:

        return "abs(A-B) (a.k.a. absminus)";

    case eMergeDisjointOver:

        return "A+B(1-a)/b, A+B if a+b < 1";

    case eMergeDivide:

        return "A/B, 0 if A < 0 and B < 0";

    case eMergeExclusion:

        return "A+B-2AB";

    case eMergeFreeze:

        return "1-sqrt(1-A)/B";

    case eMergeFrom:

        return "B-A (a.k.a. subtract)";

    case eMergeGeometric:

        return "2AB/(A+B)";

    case eMergeGrainExtract:

        return "B - A + 0.5";

    case eMergeGrainMerge:

        return "B + A - 0.5";

    case eMergeHardLight:

        return "multiply if A < 0.5, screen if A > 0.5";

    case eMergeHue:

        return "SetLum(SetSat(A, Sat(B)), Lum(B))";

    case eMergeHypot:

        return "sqrt(A*A+B*B)";

    case eMergeIn:

        return "Ab (a.k.a. src-in)";

    //case eMergeInterpolated:
    //    return "(like average but better and slower)";

    case eMergeLuminosity:

        return "SetLum(B, Lum(A))";

    case eMergeMask:

        return "Ba (a.k.a dst-in)";

    case eMergeMatte:

        return "Aa + B(1-a) (unpremultiplied over)";

    case eMergeMax:

        return "max(A, B) (a.k.a. lighten only)";

    case eMergeMin:

        return "min(A, B) (a.k.a. darken only)";

    case eMergeMinus:

        return "A-B";

    case eMergeMultiply:

        return "AB, 0 if A < 0 and B < 0";

    case eMergeOut:

        return "A(1-b) (a.k.a. src-out)";

    case eMergeOver:

        return "A+B(1-a) (a.k.a. src-over)";

    case eMergeOverlay:

        return "multiply if B < 0.5, screen if B > 0.5";

    case eMergePinLight:

        return "if B >= 0.5 then max(A, 2*B - 1), min(A, B * 2.0 ) else";

    case eMergePlus:

        return "A+B (a.k.a. add)";

    case eMergeReflect:

        return "A*A / (1 - B)";

    case eMergeSaturation:

        return "SetLum(SetSat(B, Sat(A)), Lum(B))";

    case eMergeScreen:

        return "A+B-AB if A or B <= 1, otherwise max(A, B)";

    case eMergeSoftLight:

        return "burn-in if A < 0.5, lighten if A > 0.5";

    case eMergeStencil:

        return "B(1-a) (a.k.a. dst-out)";

    case eMergeUnder:

        return "A(1-b)+B (a.k.a. dst-over)";

    case eMergeXOR:

        return "A(1-b)+B(1-a)";
    } // switch

    return "unknown";
} // getOperationString

inline std::string
getOperationHelp(MergingFunctionEnum operation, bool markdown)
{
    if (!markdown) {
        return getOperationString(operation) + ": " + getOperationDescription(operation);
    }
    std::string escaped = getOperationString(operation) + ": ";
    std::string plain = getOperationDescription(operation);
    // the following chars must be backslash-escaped in markdown:
    // \    backslash
    // `    backtick
    // *    asterisk
    // _    underscore
    // {}   curly braces
    // []   square brackets
    // ()   parentheses
    // #    hash mark
    // +    plus sign
    // -    minus sign (hyphen)
    // .    dot
    // !    exclamation mark
    for (unsigned i = 0; i < plain.size(); ++i) {
        if (plain[i] == '\\' ||
            plain[i] == '`' ||
            plain[i] == '*' ||
            plain[i] == '_' ||
            plain[i] == '{' ||
            plain[i] == '}' ||
            plain[i] == '[' ||
            plain[i] == ']' ||
            plain[i] == '(' ||
            plain[i] == ')' ||
            plain[i] == '#' ||
            plain[i] == '+' ||
            plain[i] == '-' ||
            plain[i] == '.' ||
            plain[i] == '!') {
            escaped += '\\';
        }
        escaped += plain[i];
    }
    return escaped;
}

inline std::string
getOperationGroupString(MergingFunctionEnum operation)
{
    switch (operation) {
    // Porter Duff Compositing Operators
    // missing: clear
    case eMergeCopy:     // src
    // missing: dst
    case eMergeOver:     // src-over
    case eMergeUnder:     // dst-over
    case eMergeIn:     // src-in
    case eMergeMask:     // dst-in
    case eMergeOut:     // src-out
    case eMergeStencil:     // dst-out
    case eMergeATop:     // src-atop
    case eMergeXOR:     // xor
        return "Operator";

    // Blend modes, see https://en.wikipedia.org/wiki/Blend_modes

    // Multiply and screen
    case eMergeMultiply:
    case eMergeScreen:
    case eMergeOverlay:
    case eMergeHardLight:
    case eMergeSoftLight:

        return "Multiply and Screen";

    // Dodge and burn
    case eMergeColorDodge:
    case eMergeColorBurn:
    case eMergePinLight:
    //case eMergeDifference:
    case eMergeExclusion:

        //case eMergeDivide:
        return "Dodge and Burn";

    // Simple arithmetic blend modes
    case eMergeDivide:
    case eMergePlus:
    case eMergeFrom:
    case eMergeMinus:
    case eMergeDifference:
    case eMergeMin:
    case eMergeMax:

        return "HSL";

    // Hue, saturation, luminosity
    case eMergeHue:
    case eMergeSaturation:
    case eMergeColor:
    case eMergeLuminosity:

        return "HSL";

    case eMergeAverage:
    case eMergeConjointOver:
    case eMergeDisjointOver:
    case eMergeFreeze:
    case eMergeGeometric:
    case eMergeGrainExtract:
    case eMergeGrainMerge:
    case eMergeHypot:
    //case eMergeInterpolated:
    case eMergeMatte:
    case eMergeReflect:

        return "Other";
    } // switch

    return "unknown";
} // getOperationGroupString

template <typename PIX>
PIX
averageFunctor(PIX A,
               PIX B)
{
    return (A + B) / 2;
}

template <typename PIX>
PIX
copyFunctor(PIX A,
            PIX /*B*/)
{
    return A;
}

template <typename PIX>
PIX
plusFunctor(PIX A,
            PIX B)
{
    return A + B;
}

template <typename PIX, int maxValue>
PIX
grainExtractFunctor(PIX A,
                    PIX B)
{
    return (B - A + (PIX)maxValue / 2);
}

template <typename PIX, int maxValue>
PIX
grainMergeFunctor(PIX A,
                  PIX B)
{
    return (B + A - (PIX)maxValue / 2);
}

template <typename PIX>
PIX
differenceFunctor(PIX A,
                  PIX B)
{
    return std::abs(A - B);
}

template <typename PIX>
PIX
divideFunctor(PIX A,
              PIX B)
{
    if (B <= 0) {
        return 0;
    }

    return A / B;
}

template <typename PIX, int maxValue>
PIX
exclusionFunctor(PIX A,
                 PIX B)
{
    return PIX(A + B - 2 * A * B / (double)maxValue);
}

template <typename PIX>
PIX
fromFunctor(PIX A,
            PIX B)
{
    return B - A;
}

template <typename PIX>
PIX
geometricFunctor(PIX A,
                 PIX B)
{
    double sum = (double)A + (double)B;

    if (sum == 0) {
        return 0;
    } else {
        return 2 * A * B / sum;
    }
}

template <typename PIX, int maxValue>
PIX
multiplyFunctor(PIX A,
                PIX B)
{
    return PIX(A * B / (double)maxValue);
}

template <typename PIX, int maxValue>
PIX
screenFunctor(PIX A,
              PIX B)
{
    if ( (A <= maxValue) || (B <= maxValue) ) {
        return PIX( (double)A + B - (double)A * B );
    } else {
        return std::max(A, B);
    }
}

template <typename PIX, int maxValue>
PIX
hardLightFunctor(PIX A,
                 PIX B)
{
    if ( A < ( (double)maxValue / 2. ) ) {
        return PIX(2 * A * B / (double)maxValue);
    } else {
        return PIX( maxValue * ( 1. - 2 * (1. - A / (double)maxValue) * (1. - B / (double)maxValue) ) );
    }
}

template <typename PIX, int maxValue>
PIX
softLightFunctor(PIX A,
                 PIX B)
{
    double An = A / (double)maxValue;
    double Bn = B / (double)maxValue;

    if (2 * An <= 1) {
        return PIX( maxValue * ( Bn - (1 - 2 * An) * Bn * (1 - Bn) ) );
    } else if (4 * Bn <= 1) {
        return PIX( maxValue * ( Bn + (2 * An - 1) * (4 * Bn * (4 * Bn + 1) * (Bn - 1) + 7 * Bn) ) );
    } else {
        return PIX( maxValue * ( Bn + (2 * An - 1) * (sqrt(Bn) - Bn) ) );
    }
}

template <typename PIX>
PIX
hypotFunctor(PIX A,
             PIX B)
{
    return PIX( std::sqrt( (double)(A * A + B * B) ) );
}

template <typename PIX>
PIX
minusFunctor(PIX A,
             PIX B)
{
    return A - B;
}

template <typename PIX>
PIX
darkenFunctor(PIX A,
              PIX B)
{
    return std::min(A, B);
}

template <typename PIX>
PIX
lightenFunctor(PIX A,
               PIX B)
{
    return std::max(A, B);
}

template <typename PIX, int maxValue>
PIX
overlayFunctor(PIX A,
               PIX B)
{
    double An = A / (double)maxValue;
    double Bn = B / (double)maxValue;

    if (2 * Bn <= 1.) {
        // multiply
        return PIX( maxValue * (2 * An * Bn) );
    } else {
        // screen
        return PIX( maxValue * ( 1 - 2 * (1 - Bn) * (1 - An) ) );
    }
}

template <typename PIX, int maxValue>
PIX
colorDodgeFunctor(PIX A,
                  PIX B)
{
    if (A >= maxValue) {
        return A;
    } else {
        return PIX( maxValue * std::min( 1., B / (maxValue - (double)A) ) );
    }
}

template <typename PIX, int maxValue>
PIX
colorBurnFunctor(PIX A,
                 PIX B)
{
    if (A <= 0) {
        return A;
    } else {
        return PIX( maxValue * ( 1. - std::min(1., (maxValue - B) / (double)A) ) );
    }
}

template <typename PIX, int maxValue>
PIX
pinLightFunctor(PIX A,
                PIX B)
{
    PIX max2 = PIX( (double)maxValue / 2. );

    return A >= max2 ? std::max(B, (A - max2) * 2) : std::min(B, A * 2);
}

template <typename PIX, int maxValue>
PIX
reflectFunctor(PIX A,
               PIX B)
{
    if (B >= maxValue) {
        return maxValue;
    } else {
        return PIX( std::min( (double)maxValue, A * A / (double)(maxValue - B) ) );
    }
}

template <typename PIX, int maxValue>
PIX
freezeFunctor(PIX A,
              PIX B)
{
    if (B <= 0) {
        return 0;
    } else {
        double An = A / (double)maxValue;
        double Bn = B / (double)maxValue;

        return PIX( std::max( 0., maxValue * (1 - std::sqrt( std::max(0., 1. - An) ) / Bn) ) );
    }
}

// This functions seems wrong. Is it a confusion with cosine interpolation?
// see http://paulbourke.net/miscellaneous/interpolation/
template <typename PIX, int maxValue>
PIX
interpolatedFunctor(PIX A,
                    PIX B)
{
    double An = A / (double)maxValue;
    double Bn = B / (double)maxValue;

    return PIX( maxValue * ( 0.5 - 0.25 * ( std::cos(M_PI * An) - std::cos(M_PI * Bn) ) ) );
}

template <typename PIX, int maxValue>
PIX
atopFunctor(PIX A,
            PIX B,
            PIX alphaA,
            PIX alphaB)
{
    return PIX( A * alphaB / (double)maxValue + B * (1. - alphaA / (double)maxValue) );
}

template <typename PIX, int maxValue>
PIX
conjointOverFunctor(PIX A,
                    PIX B,
                    PIX alphaA,
                    PIX alphaB)
{
    if (alphaA > alphaB) {
        return A;
    } else if (alphaB <= 0) {
        return A + B;
    } else {
        return A + B * ( 1. - (alphaA / (double)alphaB) );
    }
}

template <typename PIX, int maxValue>
PIX
disjointOverFunctor(PIX A,
                    PIX B,
                    PIX alphaA,
                    PIX alphaB)
{
    if (alphaA >= maxValue) {
        return A;
    } else if ( (alphaA + alphaB) < maxValue ) {
        return A + B;
    } else if (alphaB <= 0) {
        return A + B * (1 - alphaA / (double)maxValue);
    } else {
        return A + B * (maxValue - alphaA) / alphaB;
    }
}

template <typename PIX, int maxValue>
PIX
inFunctor(PIX A,
          PIX /*B*/,
          PIX /*alphaA*/,
          PIX alphaB)
{
    return PIX(A * alphaB / (double)maxValue);
}

template <typename PIX, int maxValue>
PIX
matteFunctor(PIX A,
             PIX B,
             PIX alphaA,
             PIX /*alphaB*/)
{
    return PIX( A * alphaA / (double)maxValue + B * (1. - alphaA / (double)maxValue) );
}

template <typename PIX, int maxValue>
PIX
maskFunctor(PIX /*A*/,
            PIX B,
            PIX alphaA,
            PIX /*alphaB*/)
{
    return PIX(B * alphaA / (double)maxValue);
}

template <typename PIX, int maxValue>
PIX
outFunctor(PIX A,
           PIX /*B*/,
           PIX /*alphaA*/,
           PIX alphaB)
{
    return PIX( A * (1. - alphaB / (double)maxValue) );
}

template <typename PIX, int maxValue>
PIX
overFunctor(PIX A,
            PIX B,
            PIX alphaA,
            PIX /*alphaB*/)
{
    return PIX( A + B * (1 - alphaA / (double)maxValue) );
}

template <typename PIX, int maxValue>
PIX
stencilFunctor(PIX /*A*/,
               PIX B,
               PIX alphaA,
               PIX /*alphaB*/)
{
    return PIX( B * (1 - alphaA / (double)maxValue) );
}

template <typename PIX, int maxValue>
PIX
underFunctor(PIX A,
             PIX B,
             PIX /*alphaA*/,
             PIX alphaB)
{
    return PIX(A * (1 - alphaB / (double)maxValue) + B);
}

template <typename PIX, int maxValue>
PIX
xorFunctor(PIX A,
           PIX B,
           PIX alphaA,
           PIX alphaB)
{
    return PIX( A * (1 - alphaB / (double)maxValue) + B * (1 - alphaA / (double)maxValue) );
}

///////////////////////////////////////////////////////////////////////////////
//
// Code from pixman-combine-float.c
// START
/*
 * Copyright © 2010, 2012 Soren Sandmann Pedersen
 * Copyright © 2010, 2012 Red Hat, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Author: Soren Sandmann Pedersen (sandmann@cs.au.dk)
 */
/*
 * PDF nonseperable blend modes are implemented using the following functions
 * to operate in Hsl space, with Cmax, Cmid, Cmin referring to the max, mid
 * and min value of the red, green and blue components.
 *
 * LUM (C) = 0.3 × Cred + 0.59 × Cgreen + 0.11 × Cblue
 *
 * clip_color (C):
 *     l = LUM (C)
 *     min = Cmin
 *     max = Cmax
 *     if n < 0.0
 *         C = l + (((C – l) × l) ⁄ (l – min))
 *     if x > 1.0
 *         C = l + (((C – l) × (1 – l) ) ⁄ (max – l))
 *     return C
 *
 * set_lum (C, l):
 *     d = l – LUM (C)
 *     C += d
 *     return clip_color (C)
 *
 * SAT (C) = CH_MAX (C) - CH_MIN (C)
 *
 * set_sat (C, s):
 *     if Cmax > Cmin
 *         Cmid = ( ( ( Cmid – Cmin ) × s ) ⁄ ( Cmax – Cmin ) )
 *         Cmax = s
 *     else
 *         Cmid = Cmax = 0.0
 *         Cmin = 0.0
 *     return C
 */

/* For premultiplied colors, we need to know what happens when C is
 * multiplied by a real number. LUM and SAT are linear:
 *
 *     LUM (r × C) = r × LUM (C)	SAT (r * C) = r * SAT (C)
 *
 * If we extend clip_color with an extra argument a and change
 *
 *     if x >= 1.0
 *
 * into
 *
 *     if x >= a
 *
 * then clip_color is also linear:
 *
 *     r * clip_color (C, a) = clip_color (r * C, r * a);
 *
 * for positive r.
 *
 * Similarly, we can extend set_lum with an extra argument that is just passed
 * on to clip_color:
 *
 *       r * set_lum (C, l, a)
 *
 *     = r × clip_color (C + l - LUM (C), a)
 *
 *     = clip_color (r * C + r × l - r * LUM (C), r * a)
 *
 *     = set_lum (r * C, r * l, r * a)
 *
 * Finally, set_sat:
 *
 *       r * set_sat (C, s) = set_sat (x * C, r * s)
 *
 * The above holds for all non-zero x, because the x'es in the fraction for
 * C_mid cancel out. Specifically, it holds for x = r:
 *
 *       r * set_sat (C, s) = set_sat (r * C, r * s)
 *
 */
typedef struct
{
    float r;
    float g;
    float b;
} rgb_t;
inline bool
float_is_zero(float f)
{
    return (-FLT_MIN < (f) && (f) < FLT_MIN);
}

inline float
channel_min (const rgb_t *c)
{
    return std::min(std::min(c->r, c->g), c->b);
}

inline float
channel_max (const rgb_t *c)
{
    return std::max(std::max(c->r, c->g), c->b);
}

inline float
get_lum (const rgb_t *c)
{
    return c->r * 0.3f + c->g * 0.59f + c->b * 0.11f;
}

inline float
get_sat (const rgb_t *c)
{
    return channel_max(c) - channel_min(c);
}

inline void
clip_color (rgb_t *color,
            float a)
{
    float l = get_lum(color);
    float n = channel_min(color);
    float x = channel_max(color);
    float t;

    if (n < 0.0f) {
        t = l - n;
        if ( float_is_zero(t) ) {
            color->r = 0.0f;
            color->g = 0.0f;
            color->b = 0.0f;
        } else {
            color->r = l + ( ( (color->r - l) * l ) / t );
            color->g = l + ( ( (color->g - l) * l ) / t );
            color->b = l + ( ( (color->b - l) * l ) / t );
        }
    }
    if (x > a) {
        t = x - l;
        if ( float_is_zero(t) ) {
            color->r = a;
            color->g = a;
            color->b = a;
        } else {
            color->r = l + ( ( (color->r - l) * (a - l) / t ) );
            color->g = l + ( ( (color->g - l) * (a - l) / t ) );
            color->b = l + ( ( (color->b - l) * (a - l) / t ) );
        }
    }
}

static void
set_lum (rgb_t *color,
         float sa,
         float l)
{
    float d = l - get_lum(color);

    color->r = color->r + d;
    color->g = color->g + d;
    color->b = color->b + d;

    clip_color(color, sa);
}

inline void
set_sat (rgb_t *src,
         float sat)
{
    float *max, *mid, *min;
    float t;

    if (src->r > src->g) {
        if (src->r > src->b) {
            max = &(src->r);

            if (src->g > src->b) {
                mid = &(src->g);
                min = &(src->b);
            } else {
                mid = &(src->b);
                min = &(src->g);
            }
        } else {
            max = &(src->b);
            mid = &(src->r);
            min = &(src->g);
        }
    } else {
        if (src->r > src->b) {
            max = &(src->g);
            mid = &(src->r);
            min = &(src->b);
        } else {
            min = &(src->r);

            if (src->g > src->b) {
                max = &(src->g);
                mid = &(src->b);
            } else {
                max = &(src->b);
                mid = &(src->g);
            }
        }
    }

    t = *max - *min;

    if ( float_is_zero(t) ) {
        *mid = *max = 0.0f;
    } else {
        *mid = ( (*mid - *min) * sat ) / t;
        *max = sat;
    }

    *min = 0.0f;
} // set_sat

/* Hue:
 *
 *       as * ad * B(s/as, d/as)
 *     = as * ad * set_lum (set_sat (s/as, SAT (d/ad)), LUM (d/ad), 1)
 *     = set_lum (set_sat (ad * s, as * SAT (d)), as * LUM (d), as * ad)
 *
 */
inline void
blend_hsl_hue (rgb_t *res,
               const rgb_t *dest,
               float da,
               const rgb_t *src,
               float sa)
{
    res->r = src->r * da;
    res->g = src->g * da;
    res->b = src->b * da;

    set_sat(res, get_sat(dest) * sa);
    set_lum(res, sa * da, get_lum(dest) * sa);
}

/*
 * Saturation
 *
 *     as * ad * B(s/as, d/ad)
 *   = as * ad * set_lum (set_sat (d/ad, SAT (s/as)), LUM (d/ad), 1)
 *   = set_lum (as * ad * set_sat (d/ad, SAT (s/as)),
 *                                       as * LUM (d), as * ad)
 *   = set_lum (set_sat (as * d, ad * SAT (s), as * LUM (d), as * ad))
 */
inline void
blend_hsl_saturation (rgb_t *res,
                      const rgb_t *dest,
                      float da,
                      const rgb_t *src,
                      float sa)
{
    res->r = dest->r * sa;
    res->g = dest->g * sa;
    res->b = dest->b * sa;

    set_sat(res, get_sat(src) * da);
    set_lum(res, sa * da, get_lum(dest) * sa);
}

/*
 * Color
 *
 *     as * ad * B(s/as, d/as)
 *   = as * ad * set_lum (s/as, LUM (d/ad), 1)
 *   = set_lum (s * ad, as * LUM (d), as * ad)
 */
inline void
blend_hsl_color (rgb_t *res,
                 const rgb_t *dest,
                 float da,
                 const rgb_t *src,
                 float sa)
{
    res->r = src->r * da;
    res->g = src->g * da;
    res->b = src->b * da;

    set_lum(res, sa * da, get_lum(dest) * sa);
}

/*
 * Luminosity
 *
 *     as * ad * B(s/as, d/ad)
 *   = as * ad * set_lum (d/ad, LUM (s/as), 1)
 *   = set_lum (as * d, ad * LUM (s), as * ad)
 */
inline void
blend_hsl_luminosity (rgb_t *res,
                      const rgb_t *dest,
                      float da,
                      const rgb_t *src,
                      float sa)
{
    res->r = dest->r * sa;
    res->g = dest->g * sa;
    res->b = dest->b * sa;

    set_lum (res, sa * da, get_lum (src) * da);
}

// END
// Code from pixman-combine-float.c
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Global wrapper templated by the blending operator.
 * A and B are respectively the color of the image A and B and is assumed to of size nComponents, 
 * nComponents being at most 4
 **/
template <MergingFunctionEnum f, typename PIX, int nComponents, int maxValue>
void
mergePixel(bool doAlphaMasking,
           const PIX *A,
           PIX a,
           const PIX *B,
           PIX b,
           PIX* dst)
{
    doAlphaMasking = (f == eMergeMatte) || (doAlphaMasking && isMaskable(f));

    ///When doAlphaMasking is enabled and we're in RGBA the output alpha is set to alphaA+alphaB-alphaA*alphaB
    int maxComp = nComponents;
    if ( !isSeparable(f) ) {
        // HSL modes
        rgb_t src, dest, res;
        if (a == 0 || nComponents < 3) {
            src.r = src.g = src.b = 0;
        } else {
            src.r = A[0] / (float)a;
            src.g = A[1] / (float)a;
            src.b = A[2] / (float)a;
        }
        if (b == 0 || nComponents < 3) {
            dest.r = dest.g = dest.b = 0;
        } else {
            dest.r = B[0] / (float)b;
            dest.g = B[1] / (float)b;
            dest.b = B[2] / (float)b;
        }
        float sa = a / (float)maxValue;
        float da = b / (float)maxValue;

        switch (f) {
        case eMergeHue:
            blend_hsl_hue(&res, &dest, da, &src, sa);
            break;

        case eMergeSaturation:
            blend_hsl_saturation(&res, &dest, da, &src, sa);
            break;

        case eMergeColor:
            blend_hsl_color(&res, &dest, da, &src, sa);
            break;

        case eMergeLuminosity:
            blend_hsl_luminosity(&res, &dest, da, &src, sa);
            break;

        default:
            res.r = res.g = res.b = 0.f;
            assert(false);
            break;
        }
        float R[3] = { res.r, res.g, res.b };
        for (int i = 0; i < std::min(nComponents, 3); ++i) {
            dst[i] = PIX( (1 - sa) * B[i] + (1 - da) * A[i] + R[i] * maxValue );
        }
        if (nComponents == 4) {
            dst[3] = PIX(a + b - a * b / (double)maxValue);
        }

        return;
    }

    // separable modes
    if ( doAlphaMasking && (nComponents == 4) ) {
        maxComp = 3;
        dst[3] = PIX(a + b - a * b / (double)maxValue);
    }
    for (int i = 0; i < maxComp; ++i) {
        switch (f) {
        case eMergeATop:
            dst[i] = atopFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeAverage:
            dst[i] = averageFunctor(A[i], B[i]);
            break;
        case eMergeColorBurn:
            dst[i] = colorBurnFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeColorDodge:
            dst[i] = colorDodgeFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeConjointOver:
            dst[i] = conjointOverFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeCopy:
            dst[i] = copyFunctor(A[i], B[i]);
            break;
        case eMergeDifference:
            dst[i] = differenceFunctor(A[i], B[i]);
            break;
        case eMergeDisjointOver:
            dst[i] = disjointOverFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeDivide:
            dst[i] = divideFunctor(A[i], B[i]);
            break;
        case eMergeExclusion:
            dst[i] = exclusionFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeFreeze:
            dst[i] = freezeFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeFrom:
            dst[i] = fromFunctor(A[i], B[i]);
            break;
        case eMergeGeometric:
            dst[i] = geometricFunctor(A[i], B[i]);
            break;
        case eMergeGrainExtract:
            dst[i] = grainExtractFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeGrainMerge:
            dst[i] = grainMergeFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeHardLight:
            dst[i] = hardLightFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeHypot:
            dst[i] = hypotFunctor(A[i], B[i]);
            break;
        case eMergeIn:
            dst[i] = inFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        //case eMergeInterpolated:
        //    dst[i] = interpolatedFunctor<PIX, maxValue>(A[i], B[i]);
        //    break;
        case eMergeMask:
            dst[i] = maskFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeMatte:
            dst[i] = matteFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeMax:
            dst[i] = lightenFunctor(A[i], B[i]);
            break;
        case eMergeMin:
            dst[i] = darkenFunctor(A[i], B[i]);
            break;
        case eMergeMinus:
            dst[i] = minusFunctor(A[i], B[i]);
            break;
        case eMergeMultiply:
            dst[i] = multiplyFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeOut:
            dst[i] = outFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeOver:
            dst[i] = overFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeOverlay:
            dst[i] = overlayFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergePinLight:
            dst[i] = pinLightFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergePlus:
            dst[i] = plusFunctor(A[i], B[i]);
            break;
        case eMergeReflect:
            dst[i] = reflectFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeScreen:
            dst[i] = screenFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeSoftLight:
            dst[i] = softLightFunctor<PIX, maxValue>(A[i], B[i]);
            break;
        case eMergeStencil:
            dst[i] = stencilFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeUnder:
            dst[i] = underFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        case eMergeXOR:
            dst[i] = xorFunctor<PIX, maxValue>(A[i], B[i], a, b);
            break;
        default:
            dst[i] = 0;
            assert(false);
            break;
        } // switch
    }
} // mergePixel
} // MergeImages2D
} // OFX


#endif // Misc_Merging_helper_h
