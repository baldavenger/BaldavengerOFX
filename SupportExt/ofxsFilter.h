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
 * OFX Filter/Interpolation help functions
 */

#ifndef openfx_supportext_ofxsFilter_h
#define openfx_supportext_ofxsFilter_h

#include <cmath>
#include <cassert>
#include <algorithm>

#include "ofxsImageEffect.h"

namespace OFX {
// GENERIC
#define kParamFilterType "filter"
#define kParamFilterTypeLabel "Filter"
#define kParamFilterTypeHint "Filtering algorithm - some filters may produce values outside of the initial range (*) or modify the values even if there is no movement (+)."
#define kParamFilterClamp "clamp"
#define kParamFilterClampLabel "Clamp"
#define kParamFilterClampHint "Clamp filter output within the original range - useful to avoid negative values in mattes"
#define kParamFilterBlackOutside "black_outside"
#define kParamFilterBlackOutsideLabel "Black outside"
#define kParamFilterBlackOutsideHint "Fill the area outside the source image with black"

enum FilterEnum
{
    eFilterImpulse,
    eFilterBilinear,
    eFilterCubic,
    eFilterKeys,
    eFilterSimon,
    eFilterRifman,
    eFilterMitchell,
    eFilterParzen,
    eFilterNotch,
};

#define kFilterImpulse "Impulse"
#define kFilterImpulseHint "(nearest neighbor / box) Use original values"
#define kFilterBilinear "Bilinear"
#define kFilterBilinearHint "(tent / triangle) Bilinear interpolation between original values"
#define kFilterCubic "Cubic"
#define kFilterCubicHint "(cubic spline) Some smoothing"
#define kFilterKeys "Keys"
#define kFilterKeysHint "(Catmull-Rom / Hermite spline) Some smoothing, plus minor sharpening (*)"
#define kFilterSimon "Simon"
#define kFilterSimonHint "Some smoothing, plus medium sharpening (*)"
#define kFilterRifman "Rifman"
#define kFilterRifmanHint "Some smoothing, plus significant sharpening (*)"
#define kFilterMitchell "Mitchell"
#define kFilterMitchellHint "Some smoothing, plus blurring to hide pixelation (*+)"
#define kFilterParzen "Parzen"
#define kFilterParzenHint "(cubic B-spline) Greatest smoothing of all filters (+)"
#define kFilterNotch "Notch"
#define kFilterNotchHint "Flat smoothing (which tends to hide moire' patterns) (+)"

inline
void
ofxsFilterDescribeParamsInterpolate2D(OFX::ImageEffectDescriptor &desc,
                                      OFX::PageParamDescriptor *page,
                                      bool blackOutsideDefault = true)
{
    // GENERIC PARAMETERS
    //
    {
        OFX::ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamFilterType);

        param->setLabel(kParamFilterTypeLabel);
        param->setHint(kParamFilterTypeHint);
        assert(param->getNOptions() == eFilterImpulse);
        param->appendOption(kFilterImpulse, kFilterImpulseHint);
        assert(param->getNOptions() == eFilterBilinear);
        param->appendOption(kFilterBilinear, kFilterBilinearHint);
        assert(param->getNOptions() == eFilterCubic);
        param->appendOption(kFilterCubic, kFilterCubicHint);
        assert(param->getNOptions() == eFilterKeys);
        param->appendOption(kFilterKeys, kFilterKeysHint);
        assert(param->getNOptions() == eFilterSimon);
        param->appendOption(kFilterSimon, kFilterSimonHint);
        assert(param->getNOptions() == eFilterRifman);
        param->appendOption(kFilterRifman, kFilterRifmanHint);
        assert(param->getNOptions() == eFilterMitchell);
        param->appendOption(kFilterMitchell, kFilterMitchellHint);
        assert(param->getNOptions() == eFilterParzen);
        param->appendOption(kFilterParzen, kFilterParzenHint);
        assert(param->getNOptions() == eFilterNotch);
        param->appendOption(kFilterNotch, kFilterNotchHint);
        param->setDefault(eFilterCubic);
        param->setAnimates(true);
#ifdef OFX_EXTENSIONS_NUKE
        param->setLayoutHint(OFX::eLayoutHintNoNewLine, 1);
#endif
        if (page) {
            page->addChild(*param);
        }
    }

    // clamp
    {
        OFX::BooleanParamDescriptor* param = desc.defineBooleanParam(kParamFilterClamp);
        param->setLabel(kParamFilterClampLabel);
        param->setHint(kParamFilterClampHint);
        param->setDefault(false);
        param->setAnimates(true);
#ifdef OFX_EXTENSIONS_NUKE
        param->setLayoutHint(OFX::eLayoutHintNoNewLine, 1);
#endif
        if (page) {
            page->addChild(*param);
        }
    }

    // blackOutside
    {
        OFX::BooleanParamDescriptor* param = desc.defineBooleanParam(kParamFilterBlackOutside);
        param->setLabel(kParamFilterBlackOutsideLabel);
        param->setHint(kParamFilterBlackOutsideHint);
        param->setDefault(blackOutsideDefault);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }
} // ofxsFilterDescribeParamsInterpolate2D

/*
   Maple code to compute the filters.

 # Mitchell, D. and A. Netravali, "Reconstruction Filters in Computer Graphics."
 # http://www.cs.utexas.edu/users/fussell/courses/cs384g/lectures/mitchell/Mitchell.pdf
 # Computer Graphics, Vol. 22, No. 4, pp. 221-228.
 # (B, C)
 # (1/3, 1/3) - Defaults recommended by Mitchell and Netravali
 # (1, 0) - Equivalent to the Cubic B-Spline
 # (0, 0.5) - Equivalent to the Catmull-Rom Spline
 # (0, C) - The family of Cardinal Cubic Splines
 # (B, 0) - Duff's tensioned B-Splines.
   unassign('Ip'):unassign('Ic'):unassign('In'):unassign('Ia'):
   unassign('Jp'):unassign('Jc'):unassign('Jn'):unassign('Ja'):
   P:= x -> ((12-9*B-6*C)*x**3 + (-18+12*B+6*C)*x**2+(6-2*B))/6;
   Q:= x -> ((-B-6*C)*x**3 + (6*B+30*C)*x**2 + (-12*B-48*C)*x + (8*B+24*C))/6;

   R := d -> Q(d+1)*Ip + P(d)*Ic + P(1-d) * In + Q(2-d)*Ia;

 # how does it perform on a linear function?
   R0 :=  d -> Q(d+1)*(Ic-1) + P(d)*Ic + P(1-d) * (Ic+1) + Q(2-d)*(Ic+2);

 # Cubic (cubic splines - depends only on In and Ic, derivatives are 0 at the center of each sample)
   collect(subs({B=0,C=0},R(d)),d);
   collect(subs({B=0,C=0},R0(d)),d);

 # Catmull-Rom / Keys / Hermite spline - gives linear func if input is linear
   collect(subs({B=0,C=0.5},R(d)),d);
   collect(subs({B=0,C=0.5},R0(d)),d);

 # Simon
   collect(subs({B=0,C=0.75},R(d)),d);
   collect(subs({B=0,C=0.75},R0(d)),d);

 # Rifman
   collect(subs({B=0,C=1.},R(d)),d);
   collect(subs({B=0,C=1.},R0(d)),d);

 # Mitchell - gives linear func if input is linear
   collect(subs({B=1/3, C=1/3},R(d)),d);
   collect(subs({B=1/3, C=1/3},R0(d)),d);

 # Parzen (Cubic B-spline) - gives linear func if input is linear
   collect(subs({B=1,C=0},R(d)),d);
   collect(subs({B=1,C=0},R0(d)),d);

 # Notch - gives linear func if input is linear
   collect(subs({B=3/2,C=-1/4},R(d)),d);
   collect(subs({B=3/2,C=-1/4},R0(d)),d);
 */
inline double
ofxsFilterClampVal(double I,
                   double Ic,
                   double In)
{
    double Imin = std::min(Ic, In);

    if (I < Imin) {
        return Imin;
    }
    double Imax = std::max(Ic, In);
    if (I > Imax) {
        return Imax;
    }

    return I;
}

inline
double
ofxsFilterLinear(double Ic,
                 double In,
                 double d)
{
    return Ic + d * (In - Ic);
}

static inline
double
ofxsFilterCubic(double Ic,
                double In,
                double d,
                bool clamp)
{
    double I = Ic + d * d * ( (-3 * Ic + 3 * In ) + d * (2 * Ic - 2 * In ) );

    if (clamp) {
        I = ofxsFilterClampVal(I, Ic, In);
    }

    return I;
}

inline
double
ofxsFilterKeys(double Ip,
               double Ic,
               double In,
               double Ia,
               double d,
               bool clamp)
{
    double I = Ic  + d * ( (-Ip + In ) + d * ( (2 * Ip - 5 * Ic + 4 * In - Ia ) + d * (-Ip + 3 * Ic - 3 * In + Ia ) ) ) / 2;

    if (clamp) {
        I = ofxsFilterClampVal(I, Ic, In);
    }

    return I;
}

inline
double
ofxsFilterSimon(double Ip,
                double Ic,
                double In,
                double Ia,
                double d,
                bool clamp)
{
    double I = Ic  + d * ( (-3 * Ip + 3 * In ) + d * ( (6 * Ip - 9 * Ic + 6 * In - 3 * Ia ) + d * (-3 * Ip + 5 * Ic - 5 * In + 3 * Ia ) ) ) / 4;

    if (clamp) {
        I = ofxsFilterClampVal(I, Ic, In);
    }

    return I;
}

inline
double
ofxsFilterRifman(double Ip,
                 double Ic,
                 double In,
                 double Ia,
                 double d,
                 bool clamp)
{
    double I = Ic  + d * ( (-Ip + In ) + d * ( (2 * Ip - 2 * Ic + In - Ia ) + d * (-Ip + Ic - In + Ia ) ) );

    if (clamp) {
        I = ofxsFilterClampVal(I, Ic, In);
    }

    return I;
}

inline
double
ofxsFilterMitchell(double Ip,
                   double Ic,
                   double In,
                   double Ia,
                   double d,
                   bool clamp)
{
    double I = ( Ip + 16 * Ic + In + d * ( (-9 * Ip + 9 * In ) + d * ( (15 * Ip - 36 * Ic + 27 * In - 6 * Ia ) + d * (-7 * Ip + 21 * Ic - 21 * In + 7 * Ia ) ) ) ) / 18;

    if (clamp) {
        I = ofxsFilterClampVal(I, Ic, In);
    }

    return I;
}

inline
double
ofxsFilterParzen(double Ip,
                 double Ic,
                 double In,
                 double Ia,
                 double d,
                 bool /*clamp*/)
{
    double I = ( Ip + 4 * Ic + In + d * ( (-3 * Ip + 3 * In ) + d * ( (3 * Ip - 6 * Ic + 3 * In ) + d * (-Ip + 3 * Ic - 3 * In + Ia ) ) ) ) / 6;

    // clamp is not necessary for Parzen
    return I;
}

inline
double
ofxsFilterNotch(double Ip,
                double Ic,
                double In,
                double Ia,
                double d,
                bool /*clamp*/)
{
    double I = ( Ip + 2 * Ic + In + d * ( (-2 * Ip + 2 * In ) + d * ( (Ip - Ic - In + Ia ) ) ) ) / 4;

    // clamp is not necessary for Notch
    return I;
}

#define OFXS_APPLY4(f, j) double I ## j = f(Ip ## j, Ic ## j, In ## j, Ia ## j, dx, clamp)

#define OFXS_CUBIC2D(f)                                      \
    inline                                           \
    double                                                  \
    f ## 2D (double Ipp, double Icp, double Inp, double Iap, \
             double Ipc, double Icc, double Inc, double Iac, \
             double Ipn, double Icn, double Inn, double Ian, \
             double Ipa, double Ica, double Ina, double Iaa, \
             double dx, double dy, bool clamp)               \
    {                                                       \
        OFXS_APPLY4(f, p); OFXS_APPLY4(f, c); OFXS_APPLY4(f, n); OFXS_APPLY4(f, a); \
        return f(Ip, Ic, In, Ia, dy, clamp);            \
    }

OFXS_CUBIC2D(ofxsFilterKeys);
OFXS_CUBIC2D(ofxsFilterSimon);
OFXS_CUBIC2D(ofxsFilterRifman);
OFXS_CUBIC2D(ofxsFilterMitchell);
OFXS_CUBIC2D(ofxsFilterParzen);
OFXS_CUBIC2D(ofxsFilterNotch);

#undef OFXS_CUBIC2D
#undef OFXS_APPLY4

template <class PIX>
PIX
ofxsGetPixComp(const PIX* p,
               int c)
{
    return p ? p[c] : PIX();
}

// Macros used in ofxsFilterInterpolate2D
#define OFXS_CLAMPXY(m) \
    m ## x = std::max( srcImg->getBounds().x1, std::min(m ## x, srcImg->getBounds().x2 - 1) ); \
    m ## y = std::max( srcImg->getBounds().y1, std::min(m ## y, srcImg->getBounds().y2 - 1) )

#define OFXS_GETPIX(i, j) PIX * P ## i ## j = (PIX *)srcImg->getPixelAddress(i ## x, j ## y)

#define OFXS_GETI(i, j)   const double I ## i ## j = ofxsGetPixComp(P ## i ## j, c)

#define OFXS_GETPIX4(i)  OFXS_GETPIX(i, p); OFXS_GETPIX(i, c); OFXS_GETPIX(i, n); OFXS_GETPIX(i, a);

#define OFXS_GETI4(i)    OFXS_GETI(i, p); OFXS_GETI(i, c); OFXS_GETI(i, n); OFXS_GETI(i, a);


#define OFXS_I44         Ipp, Icp, Inp, Iap, \
    Ipc, Icc, Inc, Iac, \
    Ipn, Icn, Inn, Ian, \
    Ipa, Ica, Ina, Iaa

// note that the center of pixel (0,0) has pixel coordinates (0.5,0.5)
template <class PIX, int nComponents, FilterEnum filter, bool clamp>
bool
ofxsFilterInterpolate2D(double fx,
                        double fy,            //!< coordinates of the pixel to be interpolated in srcImg in pixel coordinates
                        const OFX::Image *srcImg, //!< image to be transformed
                        bool blackOutside,
                        float *tmpPix) //!< destination pixel in float format
{
    if (!srcImg) {
        for (int c = 0; c < nComponents; ++c) {
            tmpPix[c] = 0;
        }

        return false;
    }
    bool inside = true; // return true, except if outside and black
    // GENERIC TRANSFORM
    // from here on, everything is generic, and should be moved to a generic transform class
    // Important: (0,0) is the *corner*, not the *center* of the first pixel (see OpenFX specs)
    switch (filter) {
    case eFilterImpulse: {
        ///nearest neighboor
        // the center of pixel (0,0) has coordinates (0.5,0.5)
        int mx = (int)std::floor(fx);     // don't add 0.5
        int my = (int)std::floor(fy);     // don't add 0.5

        if (!blackOutside) {
            OFXS_CLAMPXY(m);
        }
        OFXS_GETPIX(m, m);
        if (Pmm) {
            for (int c = 0; c < nComponents; ++c) {
                tmpPix[c] = Pmm[c];
            }
        } else {
            for (int c = 0; c < nComponents; ++c) {
                tmpPix[c] = 0;
            }
            inside = false;
        }
        break;
    }
    case eFilterBilinear:
    case eFilterCubic: {
        // bilinear or cubic
        // the center of pixel (0,0) has coordinates (0.5,0.5)
        int cx = (int)std::floor(fx - 0.5);
        int cy = (int)std::floor(fy - 0.5);
        int nx = cx + 1;
        int ny = cy + 1;
        if (!blackOutside) {
            OFXS_CLAMPXY(c);
            OFXS_CLAMPXY(n);
        }

        const double dx = std::max( 0., std::min(fx - 0.5 - cx, 1.) );
        const double dy = std::max( 0., std::min(fy - 0.5 - cy, 1.) );

        OFXS_GETPIX(c, c); OFXS_GETPIX(n, c); OFXS_GETPIX(c, n); OFXS_GETPIX(n, n);
        if (Pcc || Pnc || Pcn || Pnn) {
            for (int c = 0; c < nComponents; ++c) {
                OFXS_GETI(c, c); OFXS_GETI(n, c); OFXS_GETI(c, n); OFXS_GETI(n, n);
                if (filter == eFilterBilinear) {
                    double Ic = ofxsFilterLinear(Icc, Inc, dx);
                    double In = ofxsFilterLinear(Icn, Inn, dx);
                    tmpPix[c] = (float)ofxsFilterLinear(Ic, In, dy);
                } else if (filter == eFilterCubic) {
                    double Ic = ofxsFilterCubic(Icc, Inc, dx, clamp);
                    double In = ofxsFilterCubic(Icn, Inn, dx, clamp);
                    tmpPix[c] = (float)ofxsFilterCubic(Ic, In, dy, clamp);
                } else {
                    assert(0);
                }
            }
        } else {
            for (int c = 0; c < nComponents; ++c) {
                tmpPix[c] = 0;
            }
            inside = false;
        }
        break;
    }

    // (B,C) cubic filters
    case eFilterKeys:
    case eFilterSimon:
    case eFilterRifman:
    case eFilterMitchell:
    case eFilterParzen:
    case eFilterNotch: {
        // the center of pixel (0,0) has coordinates (0.5,0.5)
        int cx = (int)std::floor(fx - 0.5);
        int cy = (int)std::floor(fy - 0.5);
        int px = cx - 1;
        int py = cy - 1;
        int nx = cx + 1;
        int ny = cy + 1;
        int ax = cx + 2;
        int ay = cy + 2;
        if (!blackOutside) {
            OFXS_CLAMPXY(c);
            OFXS_CLAMPXY(p);
            OFXS_CLAMPXY(n);
            OFXS_CLAMPXY(a);
        }
        const double dx = std::max( 0., std::min(fx - 0.5 - cx, 1.) );
        const double dy = std::max( 0., std::min(fy - 0.5 - cy, 1.) );

        OFXS_GETPIX4(p); OFXS_GETPIX4(c); OFXS_GETPIX4(n); OFXS_GETPIX4(a);
        if (Ppp || Pcp || Pnp || Pap || Ppc || Pcc || Pnc || Pac || Ppn || Pcn || Pnn || Pan || Ppa || Pca || Pna || Paa) {
            for (int c = 0; c < nComponents; ++c) {
                //double Ipp = get(Ppp,c);, etc.
                OFXS_GETI4(p); OFXS_GETI4(c); OFXS_GETI4(n); OFXS_GETI4(a);
                double I = 0.;
                switch (filter) {
                case eFilterKeys:
                    I = ofxsFilterKeys2D(OFXS_I44, dx, dy, clamp);
                    break;
                case eFilterSimon:
                    I = ofxsFilterSimon2D(OFXS_I44, dx, dy, clamp);
                    break;
                case eFilterRifman:
                    I = ofxsFilterRifman2D(OFXS_I44, dx, dy, clamp);
                    break;
                case eFilterMitchell:
                    I = ofxsFilterMitchell2D(OFXS_I44, dx, dy, clamp);
                    break;
                case eFilterParzen:
                    I = ofxsFilterParzen2D(OFXS_I44, dx, dy, false);
                    break;
                case eFilterNotch:
                    I = ofxsFilterNotch2D(OFXS_I44, dx, dy, false);
                    break;
                default:
                    assert(0);
                }
                tmpPix[c] = (float)I;
            }
        } else {
            for (int c = 0; c < nComponents; ++c) {
                tmpPix[c] = 0;
            }
            inside = false;
        }
        break;
    }

    default:
        assert(0);
        break;
    } // switch

    return inside;
} // ofxsFilterInterpolate2D

/*
 * Interpolation with SuperSampling, to avoid moire artifacts when minimizing.
 *

   ofxsFilterInterpolate2D() does not take into account scaling or distortion effects.
   A consequence is that the transform nodes may produce aliasing artifacts when downscaling by a factor of 2 or more.

   There are several solutions to this problem is the case where the same texture has to be mapped *several times*:

 * Trilinear mipmapping (as implemented by OpenGL) still produces artifacts when scaling is anisotropic (i.e. the scaling factor is different along two directions)
 * [Feline (McCormack, 1999)](http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-99-1.pdf), which is close to what is proposed in [OpenGL's anisotropic texture filter](http://www.opengl.org/registry/specs/EXT/texture_filter_anisotropic.txt) is probably 4-5 times slower than mipmapping, but produces less artifacts
 * [EWA (Heckbert 1989)](https://www.cs.cmu.edu/~ph/texfund/texfund.pdf) would give the highest quality, but is probably 20 times slower than mipmapping.

   A sample implementation of the three methods is given in [Mesa 3D](http://mesa3d.org/)'s [software rasterizer, src/mesa/swrast/s_texfilter.c](http://cgit.freedesktop.org/mesa/mesa/tree/src/mesa/swrast/s_texfilter.c).

 * However*, in our case, the texture usually has to be mapped only once. Thus it is more appropriate to use one of the techniques described in this document: <http://people.cs.clemson.edu/~dhouse/courses/405/notes/antialiasing.pdf>.

 # Our implementation:

   We chose to use a standard supersampling method without jitter (to avoid randomness in rendered images).

   The first implementation was interpolating accross scales between supersampled pixels (see OFX_FILTER_SUPERSAMPLING_TRILINEAR below), but since we noticed that using the highest scale produces less moire, and it even costs a bit less (less tests in the processing).

   We also noticed that supersampled pixels don't need to use anything better than bilinear filter. The impulse filter still produces moire, and other filters are overkill or may even produce more moire.

 */

#ifdef DEBUG
#define _DBG_COUNT(x) (x)
#else
#define _DBG_COUNT(x) ( (void)0 )
#endif

// Internal function for supersampling (should never be called by the user)
// note that the center of pixel (0,0) has pixel coordinates (0.5,0.5)
template <class PIX, int nComponents, FilterEnum filter, int subx, int suby>
void
ofxsFilterInterpolate2DSuperInternal(double fx,
                                     double fy,            //!< coordinates of the pixel to be interpolated in srcImg in pixel coordinates
                                     double Jxx, //!< derivative of fx over x
                                     double Jxy, //!< derivative of fx over y
                                     double Jyx, //!< derivative of fy over x
                                     double Jyy, //!< derivative of fy over y
                                     double sx, //!< scale over x as a power of 3
                                     double sy, //!< scale over y as a power of 3
                                     int isx, //!< floor(sx)
                                     int isy,  //!< floor(sy)
                                     const OFX::Image *srcImg, //!< image to be transformed
                                     bool blackOutside,
                                     float *tmpPix) //!< input: interpolated center filter. output: destination pixel in float format
{
    // do supersampling.
    // All values are computed using nearest neighbor interpolation, except for the center value

    // compute number of samples over each dimension, i.e. pow(nis*,3)
    // see http://stackoverflow.com/a/101613/2607517
    int nisx;
    {
        int base = 3;
        int exp = isx;
        int result = 1;
        while (exp) {
            if (exp & 1) {
                result *= base;
            }
            exp >>= 1;
            base *= base;
        }
        nisx = result;
    }
    /// linear version:
    //nisx = 1;
    //for (int p = 0; p < isx; ++p) {
    //    nisx *= 3;
    //}
    int nisy;
    {
        int base = 3;
        int exp = isy;
        int result = 1;
        while (exp) {
            if (exp & 1) {
                result *= base;
            }
            exp >>= 1;
            base *= base;
        }
        nisy = result;
    }

    /// linear version:
    //nisy = 1;
    //for (int p = 0; p < isy; ++p) {
    //    nisy *= 3;
    //}
    assert( nisx == std::pow( (double)3, (double)isx ) && nisy == std::pow( (double)3, (double)isy ) );

    // compute the pixel value at scales (isx,isy) (nsx,isy) (isx,nsy) (nsx,nsy), and interpolate bilinearly using sx,sy
    float *pii = tmpPix;
    float pni[nComponents];
    float pin[nComponents];
    float pnn[nComponents];
#ifdef DEBUG
    int piicount = 1;
    int pnicount = 0;
    int pincount = 0;
    int pnncount = 0;
#endif

    // initialize to value of center pixel
    if (subx) {
        std::copy(tmpPix, tmpPix + nComponents, pni);
        _DBG_COUNT(pnicount = 1);
        if (suby) {
            std::copy(tmpPix, tmpPix + nComponents, pnn);
            _DBG_COUNT(pnncount = 1);
        }
    }
    if (suby) {
        std::copy(tmpPix, tmpPix + nComponents, pin);
        _DBG_COUNT(pincount = 1);
    }

    // accumulate
    for (int y = -nisy / 2; y <= nisy / 2; ++y) {
        for (int x = -nisx / 2; x <= nisx / 2; ++x) {
            // subsample position
            double sfx = fx + (Jxx * x) / nisx + (Jxy * y) / nisy;
            double sfy = fy + (Jyx * x) / nisx + (Jyy * y) / nisy;
            if ( (x != 0) || (y != 0) ) { // center pixel was already accumulated
                float tmp[nComponents];
                ofxsFilterInterpolate2D<PIX, nComponents, filter, false>(sfx, sfy, srcImg, blackOutside, tmp);
                for (int c = 0; c < nComponents; ++c) {
                    pii[c] += tmp[c];
                    _DBG_COUNT( piicount += (c == 0) );
                    // other scales
                    if (subx) {
                        pni[c] += tmp[c];
                        _DBG_COUNT( pnicount += (c == 0) );
                        if (suby) {
                            pnn[c] += tmp[c];
                            _DBG_COUNT( pnncount += (c == 0) );
                        }
                    }
                    if (suby) {
                        pin[c] += tmp[c];
                        _DBG_COUNT( pincount += (c == 0) );
                    }
                }
            }
            // subsamples from next scales
            for (int j = -suby; j <= suby; ++j) {
                for (int i = -subx; i <= subx; ++i) {
                    if ( (i != 0) || (j != 0) ) { // center subsample was already accumulated
                        double subfx = sfx + (Jxx * i) / (nisx * 3) + (Jxy * j) / (nisy * 3);
                        double subfy = sfy + (Jyx * i) / (nisx * 3) + (Jyy * j) / (nisy * 3);
                        {
                            float tmp[nComponents];
                            ofxsFilterInterpolate2D<PIX, nComponents, filter, false>(subfx, subfy, srcImg, blackOutside, tmp);
                            for (int c = 0; c < nComponents; ++c) {
                                // other scales
                                if (subx) {
                                    if (j == 0) {
                                        pni[c] += tmp[c];
                                        _DBG_COUNT( pnicount += (c == 0) );
                                    }
                                    if (suby) {
                                        pnn[c] += tmp[c];
                                        _DBG_COUNT( pnncount += (c == 0) );
                                    }
                                }
                                if (suby) {
                                    if (i == 0) {
                                        pin[c] += tmp[c];
                                        _DBG_COUNT( pincount += (c == 0) );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // divide by number of values
    int insamples = nisx * nisy;

#ifdef DEBUG
    assert(piicount == insamples);
    if (subx) {
        assert(pnicount == insamples * 3);
        if (suby) {
            assert(pnncount == insamples * 9);
        }
    }
    if (suby) {
        assert(pincount == insamples * 3);
    }
#endif

    for (int c = 0; c < nComponents; ++c) {
        pii[c] /= insamples;
        if (subx) {
            pni[c] /= insamples * 3;
            if (suby) {
                pnn[c] /= insamples * 9;
            }
        }
        if (suby) {
            pin[c] /= insamples * 3;
        }
    }
    if (subx) {
        // interpolate linearly over sx
        float alpha = sx - isx;
        for (int c = 0; c < nComponents; ++c) {
            pii[c] = pii[c] + alpha * (pni[c] - pii[c]);
        }
        if (suby) {
            for (int c = 0; c < nComponents; ++c) {
                pin[c] = pin[c] + alpha * (pnn[c] - pin[c]);
            }
        }
    }
    if (suby) {
        // interpolate linearly over sy
        float alpha = sy - isy;
        for (int c = 0; c < nComponents; ++c) {
            pii[c] = pii[c] + alpha * (pin[c] - pii[c]);
        }
    }

    // pii is actually an alias to tmpPix, so everything is done
} // ofxsFilterInterpolate2DSuperInternal

#undef _DBG_COUNT

inline bool
ofxsFilterOutside(double x,
                  double y,
                  const OfxRectI &bounds)
{
    return x < bounds.x1 || bounds.x2 <= x || y < bounds.y1 || bounds.y2 <= y;
}

// Interpolation using the given filter and supersampling for minification
// note that the center of pixel (0,0) has pixel coordinates (0.5,0.5)
template <class PIX, int nComponents, FilterEnum filter, bool clamp>
void
ofxsFilterInterpolate2DSuper(double fx,
                             double fy,            //!< coordinates of the pixel to be interpolated in srcImg in pixel coordinates
                             double Jxx, //!< derivative of fx over x
                             double Jxy, //!< derivative of fx over y
                             double Jyx, //!< derivative of fy over x
                             double Jyy, //!< derivative of fy over y
                             const OFX::Image *srcImg, //!< image to be transformed
                             bool blackOutside,
                             float *tmpPix) //!< destination pixel in float format
{
    // first, compute the center value
    bool inside = ofxsFilterInterpolate2D<PIX, nComponents, filter, clamp>(fx, fy, srcImg, blackOutside, tmpPix);

    if (!inside) {
        if (!srcImg) {
            return;
        }
        // Center of the pixel is outside.
        // no supersampling if we're outside (we don't want to supersample black and transparent areas)
        // ... but we still have to check wether the entire pixel is outside
        const OfxRectI &bounds = srcImg->getBounds();
        // we check the four corners of the pixel
        if ( ofxsFilterOutside(fx - Jxx * 0.5 - Jxy * 0.5, fy - Jyx * 0.5 - Jyy * 0.5, bounds) &&
             ofxsFilterOutside(fx + Jxx * 0.5 - Jxy * 0.5, fy + Jyx * 0.5 - Jyy * 0.5, bounds) &&
             ofxsFilterOutside(fx - Jxx * 0.5 + Jxy * 0.5, fy - Jyx * 0.5 + Jyy * 0.5, bounds) &&
             ofxsFilterOutside(fx + Jxx * 0.5 + Jxy * 0.5, fy + Jyx * 0.5 + Jyy * 0.5, bounds) ) {
            return;
        }
    }

    double dx = Jxx * Jxx + Jyx * Jyx; // squared norm of the derivative over x
    double dy = Jxy * Jxy + Jyy * Jyy; // squared norm of the derivative over x

    if ( (dx <= 1.) && (dy <= 1.) ) {
        // no minificationin either direction, means no supersampling
        return;
    }

    // maximum scale is 4, which is 81x81 pixels for a scale factor < 1/81
    // rather than taking sqrt(dx), we divide its log by 2
    double sx = (dx <= 1.) ? 0. : std::min(std::log(dx) / ( 2 * std::log(3.) ), 4.); // scale over x as a power of 3
    double sy = (dy <= 1.) ? 0. : std::min(std::log(dy) / ( 2 * std::log(3.) ), 4.); // scale over y as a power of 3
//#define OFX_FILTER_SUPERSAMPLING_TRILINEAR
#ifdef OFX_FILTER_SUPERSAMPLING_TRILINEAR
    // produces artifacts
    int isx = std::floor(sx);
    int isy = std::floor(sy);
    int subx = (sx > isx);
    int suby = (sy > isy);

    // we use bilinear filtering for the supersamples (except for the center point).
    if (subx) {
        if (suby) {
            return ofxsFilterInterpolate2DSuperInternal<PIX, nComponents, eFilterBilinear, true, true>(fx, fy, Jxx, Jxy, Jyx, Jyy, sx, sy, isx, isy, srcImg, blackOutside, tmpPix);
        } else {
            return ofxsFilterInterpolate2DSuperInternal<PIX, nComponents, eFilterBilinear, true, false>(fx, fy, Jxx, Jxy, Jyx, Jyy, sx, sy, isx, isy, srcImg, blackOutside, tmpPix);
        }
    } else {
        if (suby) {
            return ofxsFilterInterpolate2DSuperInternal<PIX, nComponents, eFilterBilinear, false, true>(fx, fy, Jxx, Jxy, Jyx, Jyy, sx, sy, isx, isy, srcImg, blackOutside, tmpPix);
        } else {
            return ofxsFilterInterpolate2DSuperInternal<PIX, nComponents, eFilterBilinear, false, false>(fx, fy, Jxx, Jxy, Jyx, Jyy, sx, sy, isx, isy, srcImg, blackOutside, tmpPix);
        }
    }
#else
    // always use the supersampled data
    // produces less artifacts, costs less
    // the problem is that sx = 1.0001 is supersampled, which gives a result very different from sx=1
    //int isx = std::ceil(sx);
    //int isy = std::ceil(sy);
    // This is why we prefer rounding. The jump will be at sx=sqrt(3)=1.732.
    // This produces quicker renders too, since we supersample less.
    int isx = std::ceil(sx-0.5);
    int isy = std::ceil(sy-0.5);

    return ofxsFilterInterpolate2DSuperInternal<PIX, nComponents, eFilterBilinear, false, false>(fx, fy, Jxx, Jxy, Jyx, Jyy, isx, isy, isx, isy, srcImg, blackOutside, tmpPix);
#endif
} // ofxsFilterInterpolate2DSuper

#undef OFXS_CLAMPXY
#undef OFXS_GETPIX
#undef OFXS_GETI
#undef OFXS_GETPIX4
#undef OFXS_GETI
#undef OFXS_I44


inline void
ofxsFilterExpandRoD(OFX::ImageEffect* /*effect*/,
                    double pixelAspectRatio,
                    const OfxPointD & renderScale,
                    bool blackOutside,
                    OfxRectD *rod)
{
    if ( (rod->x2 <= rod->x1) || (rod->y2 <= rod->y1) ) {
        // empty rod
        
        return;
    }

    // No need to round things up here, we must give the *actual* RoD

    if (!blackOutside) {
        // if it's not black outside, the RoD should contain the project (we can't rely on the host to fill it).
        // [FD] 2014/09/01: disabled this. The transformed object may have a size which is very different from the project size
        /*
           OfxPointD size = effect->getProjectSize();
           OfxPointD offset = effect->getProjectOffset();

           rod->x1 = std::min(rod->x1, offset.x);
           rod->x2 = std::max(rod->x2, offset.x + size.x);
           rod->y1 = std::min(rod->y1, offset.y);
           rod->y2 = std::max(rod->y2, offset.y + size.y);
         */
    } else {
        // expand the RoD to get at least one black pixel
        double pixelSizeX = pixelAspectRatio / renderScale.x;
        double pixelSizeY = 1. / renderScale.y;
        if (rod->x1 > kOfxFlagInfiniteMin) {
            rod->x1 = rod->x1 - pixelSizeX;
        }
        if (rod->x2 < kOfxFlagInfiniteMax) {
            rod->x2 = rod->x2 + pixelSizeX;
        }
        if (rod->y1 > kOfxFlagInfiniteMin) {
            rod->y1 = rod->y1 - pixelSizeY;
        }
        if (rod->y2 < kOfxFlagInfiniteMax) {
            rod->y2 = rod->y2 + pixelSizeY;
        }
    }
#if 0
    // The following code may be needed for hosts which do not
    // round correctly the RoD/RoI
    // This should correspond to pixel boundaries at the given renderScale,
    // which is why we have to round things up/down.
    if (rod->x1 > kOfxFlagInfiniteMin) {
        rod->x1 = ( std::floor(rod->x1 / pixelSizeX) ) * pixelSizeX;
    }
    if (rod->x2 < kOfxFlagInfiniteMax) {
        rod->x2 = ( std::ceil(rod->x2 / pixelSizeX) )  * pixelSizeX;
    }
    if (rod->y1 > kOfxFlagInfiniteMin) {
        rod->y1 = ( std::floor(rod->y1 / pixelSizeY) ) * pixelSizeY;
    }
    if (rod->y2 < kOfxFlagInfiniteMax) {
        rod->y2 = ( std::ceil(rod->y2 / pixelSizeY) )  * pixelSizeY;
    }
#endif
    assert(rod->x1 <= rod->x2 && rod->y1 <= rod->y2);
}

inline void
ofxsFilterExpandRoI(const OfxRectD &roi,
                    double pixelAspectRatio,
                    const OfxPointD & renderScale,
                    FilterEnum filter,
                    bool doMasking,
                    double mix,
                    OfxRectD *srcRoI)
{
    // No need to round things up here, we must give the *actual* RoI,
    // the host should compute the right image region from it (by rounding it up/down).

    if ( (roi.x2 <= roi.x1) || (roi.y2 <= roi.y1) ) {
        *srcRoI = roi;

        return;
    }

    double pixelSizeX = pixelAspectRatio / renderScale.x;
    double pixelSizeY = 1. / renderScale.y;

    switch (filter) {
    case eFilterImpulse:
        // nearest neighbor, the exact region is OK
        break;
    case eFilterBilinear:
    case eFilterCubic:
        // bilinear or cubic, expand by 0.5 pixels
        if (srcRoI->x1 > kOfxFlagInfiniteMin) {
            srcRoI->x1 -= 0.5 * pixelSizeX;
        }
        if (srcRoI->x2 < kOfxFlagInfiniteMax) {
            srcRoI->x2 += 0.5 * pixelSizeX;
        }
        if (srcRoI->y1 > kOfxFlagInfiniteMin) {
            srcRoI->y1 -= 0.5 * pixelSizeY;
        }
        if (srcRoI->y2 < kOfxFlagInfiniteMax) {
            srcRoI->y2 += 0.5 * pixelSizeY;
        }
        break;
    case eFilterKeys:
    case eFilterSimon:
    case eFilterRifman:
    case eFilterMitchell:
    case eFilterParzen:
    case eFilterNotch:
        // bicubic, expand by 1.5 pixels
        if (srcRoI->x1 > kOfxFlagInfiniteMin) {
            srcRoI->x1 -= 1.5 * pixelSizeX;
        }
        if (srcRoI->x2 < kOfxFlagInfiniteMax) {
            srcRoI->x2 += 1.5 * pixelSizeX;
        }
        if (srcRoI->y1 > kOfxFlagInfiniteMin) {
            srcRoI->y1 -= 1.5 * pixelSizeY;
        }
        if (srcRoI->y2 < kOfxFlagInfiniteMax) {
            srcRoI->y2 += 1.5 * pixelSizeY;
        }
        break;
    }
    if ( doMasking || (mix != 1.) ) {
        // for masking or mixing, we also need the source image for that same roi.
        // compute the union of both ROIs
        srcRoI->x1 = std::min(srcRoI->x1, roi.x1);
        srcRoI->x2 = std::max(srcRoI->x2, roi.x2);
        srcRoI->y1 = std::min(srcRoI->y1, roi.y1);
        srcRoI->y2 = std::max(srcRoI->y2, roi.y2);
    }
#if 0
    // The following code may be needed for hosts which do not
    // round correctly the RoD/RoI
    // This should correspond to pixel boundaries at the given renderScale,
    // which is why we have to round things up/down.
    if (srcRoI->x1 > kOfxFlagInfiniteMin) {
        srcRoI->x1 = std::floor(srcRoI->x1 / pixelSizeX) * pixelSizeX;
    }
    if (srcRoI->x2 < kOfxFlagInfiniteMax) {
        srcRoI->x2 = std::ceil(srcRoI->x2 / pixelSizeX)  * pixelSizeX;
    }
    if (srcRoI->y1 > kOfxFlagInfiniteMin) {
        srcRoI->y1 = std::floor(srcRoI->y1 / pixelSizeY) * pixelSizeY;
    }
    if (srcRoI->y2 < kOfxFlagInfiniteMax) {
        srcRoI->y2 = std::ceil(srcRoI->y2 / pixelSizeY)  * pixelSizeY;
    }
#endif
    assert(srcRoI->x1 < srcRoI->x2 && srcRoI->y1 < srcRoI->y2);
} // ofxsFilterExpandRoI
} // OFX

#endif // ifndef openfx_supportext_ofxsFilter_h
