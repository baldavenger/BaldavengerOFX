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

#include "ofxsLut.h"

#include <algorithm>
#ifdef _WIN32
typedef unsigned __int32 uint32_t;
typedef unsigned char uint8_t;
#else
#include <stdint.h>
#endif
#include <limits>
#include <cmath>

#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

namespace OFX {
namespace Color {
// compile-time endianness checking found on:
// http://stackoverflow.com/questions/2100331/c-macro-definition-to-determine-big-endian-or-little-endian-machine
// if(O32_HOST_ORDER == O32_BIG_ENDIAN) will always be optimized by gcc -O2
enum
{
    O32_LITTLE_ENDIAN = 0x03020100ul,
    O32_BIG_ENDIAN = 0x00010203ul,
    O32_PDP_ENDIAN = 0x01000302ul
};

union o32_host_order_t
{
    uint8_t bytes[4];
    uint32_t value;
};

static const o32_host_order_t o32_host_order = {
    { 0, 1, 2, 3 }
};
#define O32_HOST_ORDER (o32_host_order.value)
unsigned short
Lut::hipart(const float f)
{
    union
    {
        float f;
        unsigned short us[2];
    }

    tmp;

    tmp.us[0] = tmp.us[1] = 0;
    tmp.f = f;

    if (O32_HOST_ORDER == O32_BIG_ENDIAN) {
        return tmp.us[0];
    } else if (O32_HOST_ORDER == O32_LITTLE_ENDIAN) {
        return tmp.us[1];
    } else {
        assert( (O32_HOST_ORDER == O32_LITTLE_ENDIAN) || (O32_HOST_ORDER == O32_BIG_ENDIAN) );

        return 0;
    }
}

float
Lut::index_to_float(const unsigned short i)
{
    union
    {
        float f;
        unsigned short us[2];
    }

    tmp;

    /* positive and negative zeros, and all gradual underflow, turn into zero: */
    if ( ( i < 0x80) || ( ( i >= 0x8000) && ( i < 0x8080) ) ) {
        return 0;
    }
    /* All NaN's and infinity turn into the largest possible legal float: */
    if ( ( i >= 0x7f80) && ( i < 0x8000) ) {
        return std::numeric_limits<float>::max();
    }
    if (i >= 0xff80) {
        return -std::numeric_limits<float>::max();
    }
    if (O32_HOST_ORDER == O32_BIG_ENDIAN) {
        tmp.us[0] = i;
        tmp.us[1] = 0x8000;
    } else if (O32_HOST_ORDER == O32_LITTLE_ENDIAN) {
        tmp.us[0] = 0x8000;
        tmp.us[1] = i;
    } else {
        assert( (O32_HOST_ORDER == O32_LITTLE_ENDIAN) || (O32_HOST_ORDER == O32_BIG_ENDIAN) );
    }

    return tmp.f;
}

// r,g,b values are from 0 to 1
// h = [0,OFXS_HUE_CIRCLE], s = [0,1], v = [0,1]
//		if s == 0, then h = 0 (undefined)
void
rgb_to_hsv( float r,
            float g,
            float b,
            float *h,
            float *s,
            float *v )
{
    float min = std::min(std::min(r, g), b);
    float max = std::max(std::max(r, g), b);

    *v = max;                               // v

    float delta = max - min;

    if (max != 0.) {
        *s = delta / max;                       // s
    } else {
        // r = g = b = 0		// s = 0, v is undefined
        *s = 0.f;
        *h = 0.f;

        return;
    }

    if (delta == 0.) {
        *h = 0.f;                 // gray
    } else if (r == max) {
        *h = (g - b) / delta;                       // between yellow & magenta
    } else if (g == max) {
        *h = 2 + (b - r) / delta;                   // between cyan & yellow
    } else {
        *h = 4 + (r - g) / delta;                   // between magenta & cyan
    }
    *h *= OFXS_HUE_CIRCLE / 6.;
    if (*h < 0) {
        *h += OFXS_HUE_CIRCLE;
    }
}

void
hsv_to_rgb(float h,
           float s,
           float v,
           float *r,
           float *g,
           float *b)
{
    if (s == 0) {
        // achromatic (grey)
        *r = *g = *b = v;

        return;
    }

    h *= 6. / OFXS_HUE_CIRCLE;            // sector 0 to 5
    int i = std::floor(h);
    float f = h - i;          // factorial part of h
    i = (i >= 0) ? (i % 6) : (i % 6) + 6; // take h modulo 360
    float p = v * ( 1 - s );
    float q = v * ( 1 - s * f );
    float t = v * ( 1 - s * ( 1 - f ) );

    switch (i) {
    case 0:
        *r = v;
        *g = t;
        *b = p;
        break;
    case 1:
        *r = q;
        *g = v;
        *b = p;
        break;
    case 2:
        *r = p;
        *g = v;
        *b = t;
        break;
    case 3:
        *r = p;
        *g = q;
        *b = v;
        break;
    case 4:
        *r = t;
        *g = p;
        *b = v;
        break;
    default:                // case 5:
        *r = v;
        *g = p;
        *b = q;
        break;
    }
} // hsv_to_rgb

void
rgb_to_hsl( float r,
            float g,
            float b,
            float *h,
            float *s,
            float *l )
{
    float min = std::min(std::min(r, g), b);
    float max = std::max(std::max(r, g), b);

    *l = (min + max) / 2;

    float delta = max - min;

    if (max != 0.) {
        *s = (*l <= 0.5) ? ( delta / (max + min) ) : ( delta / (2 - max - min) ); // s = delta/(1-abs(2L-1))
    } else {
        // r = g = b = 0		// s = 0
        *s = 0.f;
        *h = 0.f;

        return;
    }

    if (delta == 0.) {
        *h = 0.f;                 // gray
    } else if (r == max) {
        *h = (g - b) / delta;                       // between yellow & magenta
    } else if (g == max) {
        *h = 2 + (b - r) / delta;                   // between cyan & yellow
    } else {
        *h = 4 + (r - g) / delta;                   // between magenta & cyan
    }
    *h *= OFXS_HUE_CIRCLE / 6.;
    if (*h < 0) {
        *h += OFXS_HUE_CIRCLE;
    }
}

void
hsl_to_rgb(float h,
           float s,
           float l,
           float *r,
           float *g,
           float *b)
{
    if (s == 0) {
        // achromatic (grey)
        *r = *g = *b = l;

        return;
    }

    h *= 6. / OFXS_HUE_CIRCLE;            // sector 0 to 5
    int i = std::floor(h);
    float f = h - i;          // factorial part of h
    i = (i >= 0) ? (i % 6) : (i % 6) + 6; // take h modulo 360
    float v = (l <= 0.5f) ? ( l * (1.0f + s) ) : (l + s - l * s);
    float p = l + l - v;
    float sv = (v - p ) / v;
    float vsf = v * sv * f;
    float t = p + vsf;
    float q = v - vsf;

    switch (i) {
    case 0:
        *r = v;
        *g = t;
        *b = p;
        break;
    case 1:
        *r = q;
        *g = v;
        *b = p;
        break;
    case 2:
        *r = p;
        *g = v;
        *b = t;
        break;
    case 3:
        *r = p;
        *g = q;
        *b = v;
        break;
    case 4:
        *r = t;
        *g = p;
        *b = v;
        break;
    default:                // case 5:
        *r = v;
        *g = p;
        *b = q;
        break;
    }
} // hsl_to_rgb

//! Convert pixel values from RGB to HSI color spaces.
void
rgb_to_hsi( float r,
            float g,
            float b,
            float *h,
            float *s,
            float *i )
{
    float nR = r; //(r < 0 ? 0 : (r > 1. ? 1. : r));
    float nG = g; //(g < 0 ? 0 : (g > 1. ? 1. : g));
    float nB = b; //(b < 0 ? 0 : (b > 1. ? 1. : b));
    float m = std::min(std::min(nR, nG), nB);
    float theta = (float)(std::acos( 0.5f * ( (nR - nG) + (nR - nB) ) / std::sqrt( std::max( 0.f, (nR - nG) * (nR - nG) + (nR - nB) * (nG - nB) ) ) ) * (OFXS_HUE_CIRCLE / 2) / M_PI);
    float sum = nR + nG + nB;

    if (theta > 0) {
        *h = (nB <= nG) ? theta : (OFXS_HUE_CIRCLE - theta);
    } else {
        *h = 0.;
    }
    if (sum > 0) {
        *s = 1 - 3 / sum * m;
    } else {
        *s = 0.;
    }
    *i = sum / 3;
}

void
hsi_to_rgb(float h,
           float s,
           float i,
           float *r,
           float *g,
           float *b)
{
    float a = i * (1 - s);

    if ( h < (OFXS_HUE_CIRCLE / 3) ) {
        *b = a;
        *r = (float)( i * ( 1 + s * std::cos( h * M_PI / (OFXS_HUE_CIRCLE / 2) ) / std::cos( ( (OFXS_HUE_CIRCLE / 6) - h ) * M_PI / (OFXS_HUE_CIRCLE / 2) ) ) );
        *g = 3 * i - (*r + *b);
    } else if ( h < (OFXS_HUE_CIRCLE * 2 / 3) ) {
        h -= OFXS_HUE_CIRCLE / 3;
        *r = a;
        *g = (float)( i * ( 1 + s * std::cos( h * M_PI / (OFXS_HUE_CIRCLE / 2) ) / std::cos( ( (OFXS_HUE_CIRCLE / 6) - h ) * M_PI / (OFXS_HUE_CIRCLE / 2) ) ) );
        *b = 3 * i - (*r + *g);
    } else {
        h -= OFXS_HUE_CIRCLE * 2 / 3;
        *g = a;
        *b = (float)( i * ( 1 + s * std::cos( h * M_PI / (OFXS_HUE_CIRCLE / 2) ) / std::cos( ( (OFXS_HUE_CIRCLE / 6) - h ) * M_PI / (OFXS_HUE_CIRCLE / 2) ) ) );
        *r = 3 * i - (*g + *b);
    }
} // hsi_to_rgb

// R'G'B' in the range 0-1 to Y'CbCr in the video range
// (Y' = 16/255 to 235/255, CbCr = 16/255 to 240/255)
void
rgb_to_ycbcr601(float r,
                float g,
                float b,
                float *y,
                float *cb,
                float *cr)
{
    /// ref: CImg (BT.601)
    //*y  = ((255*(66*r + 129*g + 25*b) + 128)/256 + 16)/255;
    //*cb = ((255*(-38*r - 74*g + 112*b) + 128)/256 + 128)/255,
    //*cr = ((255*(112*r - 94*g - 18*b) + 128)/256 + 128)/255;

    /// ref: http://www.equasys.de/colorconversion.html (BT.601)
    /// also http://www.intersil.com/data/an/AN9717.pdf
    *y  =  0.257 * r + 0.504 * g + 0.098 * b + 16 / 255.;
    *cb = -0.148 * r - 0.291 * g + 0.439 * b + 128 / 255.;
    *cr =  0.439 * r - 0.368 * g - 0.071 * b + 128 / 255.;
}

// Y'CbCr in the video range (Y' = 16/255 to 235/255, CbCr = 16/255 to 240/255)
// to R'G'B' in the range 0-1
void
ycbcr_to_rgb601(float y,
                float cb,
                float cr,
                float *r,
                float *g,
                float *b)
{
    /// ref: CImg (BT.601)
    //y  = y * 255 - 16;
    //cb = cb * 255 - 128;
    //cr = cr * 255 - 128;
    //*r = (298 * y + 409 * cr + 128)/256/255;
    //*g = (298 * y - 100 * cb - 208 * cr + 128)/256/255;
    //*b = (298 * y + 516 * cb + 128)/256/255;

    /// ref: http://www.equasys.de/colorconversion.html (BT.601)
    /// also http://www.intersil.com/data/an/AN9717.pdf
    *r = 1.164 * (y - 16 / 255.) + 1.596 * (cr - 128 / 255.);
    *g = 1.164 * (y - 16 / 255.) - 0.813 * (cr - 128 / 255.) - 0.392 * (cb - 128 / 255.);
    *b = 1.164 * (y - 16 / 255.) + 2.017 * (cb - 128 / 255.);
} // ycbcr_to_rgb

// R'G'B' in the range 0-1 to Y'CbCr in the video range
// (Y' = 16/255 to 235/255, CbCr = 16/255 to 240/255)
void
rgb_to_ycbcr709(float r,
                float g,
                float b,
                float *y,
                float *cb,
                float *cr)
{
    // ref: http://www.poynton.com/PDFs/coloureq.pdf (BT.709)
    //*y  =  0.2215 * r +0.7154 * g +0.0721 * b;
    //*cb = -0.1145 * r -0.3855 * g +0.5000 * b + 128./255;
    //*cr =  0.5016 * r -0.4556 * g -0.0459 * b + 128./255;

    // ref: http://www.equasys.de/colorconversion.html (BT.709)
    *y  =  0.183 * r + 0.614 * g + 0.062 * b + 16 / 255.;
    *cb = -0.101 * r - 0.339 * g + 0.439 * b + 128 / 255.;
    *cr =  0.439 * r - 0.399 * g - 0.040 * b + 128 / 255.;
}

// Y'CbCr in the video range (Y' = 16/255 to 235/255, CbCr = 16/255 to 240/255)
// to R'G'B' in the range 0-1
void
ycbcr_to_rgb709(float y,
                float cb,
                float cr,
                float *r,
                float *g,
                float *b)
{
    // ref: http://www.equasys.de/colorconversion.html (BT.709)
    *r = 1.164 * (y - 16 / 255.) + 1.793 * (cr - 128 / 255.);
    *g = 1.164 * (y - 16 / 255.) - 0.533 * (cr - 128 / 255.) - 0.213 * (cb - 128 / 255.);
    *b = 1.164 * (y - 16 / 255.) + 2.112 * (cb - 128 / 255.);
} // ycbcr_to_rgb

// R'G'B' in the range 0-1 to Y'CbCr Analog (Y' in the range 0-1, PbPr in the range -0.5 - 0.5)
void
rgb_to_ypbpr601(float r,
                float g,
                float b,
                float *y,
                float *pb,
                float *pr)
{
    // ref: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
    // also http://www.equasys.de/colorconversion.html (BT.601)
    // and http://public.kitware.com/vxl/doc/release/core/vil/html/vil__colour__space_8cxx_source.html
    //*y  =  0.299f    * r + 0.587f    * g + 0.114f * b;
    //*pb = -0.168736f * r - 0.331264f * g + 0.500f * b;
    //*pr =  0.500f    * r - 0.418688f * g - 0.081312f * b;

#define Kb 0.114
#define Kr 0.299
    *y  =  Kr * r + (1 - Kr - Kb) * g + Kb * b;
    *pb =  (b - *y) / ( 2 * (1 - Kb) );
    *pr =  (r - *y) / ( 2 * (1 - Kr) );
}

// Y'CbCr Analog (Y' in the range 0-1, PbPr in the range -0.5 - 0.5) to R'G'B' in the range 0-1
void
ypbpr_to_rgb601(float y,
                float pb,
                float pr,
                float *r,
                float *g,
                float *b)
{
    // https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
    // also ref: http://www.equasys.de/colorconversion.html (BT.601)
    // and http://public.kitware.com/vxl/doc/release/core/vil/html/vil__colour__space_8cxx_source.html
    //*r = y                + 1.402f    * pr,
    //*g = y - 0.344136 * pb - 0.714136f * pr;
    //*b = y + 1.772f   * pb;

    *b = pb * ( 2 * (1 - Kb) ) + y;
    *r = pr * ( 2 * (1 - Kr) ) + y;
    *g = (y - Kr * *r - Kb * *b) / (1 - Kr - Kb);
#undef Kb
#undef Kr
} // yuv_to_rgb

// R'G'B' in the range 0-1 to Y'CbCr Analog (Y' in the range 0-1, PbPr in the range -0.5 - 0.5)
void
rgb_to_ypbpr709(float r,
                float g,
                float b,
                float *y,
                float *pb,
                float *pr)
{
    // ref: http://www.equasys.de/colorconversion.html (BT.709)
    //*y  =  0.2126f * r + 0.7152f * g + 0.0722f * b;
    //*pb = -0.115f  * r - 0.385f  * g + 0.500f  * b; // or (b - y)/1.8556
    //*pr =  0.500f  * r - 0.454f  * g - 0.046f  * b; // or (r - y)/1.5748

    // ref: http://www.poynton.com/PDFs/coloureq.pdf (10.5)
    //*y  =  0.2215f * r + 0.7154f * g + 0.0721f * b;
    //*pb = -0.1145f * r - 0.3855f * g + 0.5000f  * b;
    //*pr =  0.5016f * r - 0.4556f * g - 0.0459f  * b;

    //*y  =  0.2126390058 * r + 0.7151686783 * g + 0.07219231534 * b;
    //*pb =  (b - *y) / 1.8556;
    //*pr =  (r - *y) / 1.5748;
#define Kb 0.07219231534
#define Kr 0.2126390058
    *y  =  Kr * r + (1 - Kr - Kb) * g + Kb * b;
    *pb =  (b - *y) / ( 2 * (1 - Kb) );
    *pr =  (r - *y) / ( 2 * (1 - Kr) );
}

// Y'CbCr Analog (Y in the range 0-1, PbPr in the range -0.5 - 0.5) to R'G'B' in the range 0-1
void
ypbpr_to_rgb709(float y,
                float pb,
                float pr,
                float *r,
                float *g,
                float *b)
{
    // ref: http://www.equasys.de/colorconversion.html (BT.709)
    //*r = y               + 1.575f * pr,
    //*g = y - 0.187f * pb - 0.468f * pr;
    //*b = y + 1.856f * pb;

    // ref: http://www.poynton.com/PDFs/coloureq.pdf (10.5)
    // (there is a sign error on the R' coeff for Cr in Poynton's doc)
    //*r = y                + 1.5701f * pr,
    //*g = y - 0.1870f * pb - 0.4664f * pr;
    //*b = y + 1.8556f * pb;

    //*b = pb * 1.8556 + y;
    //*r = pr * 1.5748 + y;
    ////*g = (y - 0.2126f * *r - 0.0722f * *b) / 0.7152f;
    //*g = (y - 0.2126390058 * *r - 0.07219231534 * *b) / 0.7151686783;
    *b = pb * ( 2 * (1 - Kb) ) + y;
    *r = pr * ( 2 * (1 - Kr) ) + y;
    *g = (y - Kr * *r - Kb * *b) / (1 - Kr - Kb);
#undef Kb
#undef Kr
} // yuv_to_rgb

// R'G'B' in the range 0-1 to Y'CbCr Analog (Y' in the range 0-1, PbPr in the range -0.5 - 0.5)
void
rgb_to_ypbpr2020(float r,
                 float g,
                 float b,
                 float *y,
                 float *pb,
                 float *pr)
{
    // ref: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-S!!PDF-E.pdf
    // (Rec.2020, table 4 p4)
    //
    //*y  =  0.2627f * r + 0.6780f * g + 0.0593f * b;
    //*pb =  (b - *y) / 1.8814;
    //*pr =  (r - *y) / 1.4746;

    // ref: http://www.poynton.com/PDFs/coloureq.pdf (10.5)
    //*y  =  0.2215f * r + 0.7154f * g + 0.0721f * b;
    //*pb = -0.1145f * r - 0.3855f * g + 0.5000f  * b;
    //*pr =  0.5016f * r - 0.4556f * g - 0.0459f  * b;
#define Kb 0.0593
#define Kr 0.2627
    *y  =  Kr * r + (1 - Kr - Kb) * g + Kb * b;
    *pb =  (b - *y) / ( 2 * (1 - Kb) );
    *pr =  (r - *y) / ( 2 * (1 - Kr) );
}

// Y'CbCr Analog (Y in the range 0-1, PbPr in the range -0.5 - 0.5) to R'G'B' in the range 0-1
void
ypbpr_to_rgb2020(float y,
                 float pb,
                 float pr,
                 float *r,
                 float *g,
                 float *b)
{
    // ref: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-S!!PDF-E.pdf
    // (Rec.2020, table 4 p4)
    //
    //*b = pb * 1.8814 + y;
    //*r = pr * 1.4746 + y;
    //*g = (y - 0.2627f * *r - 0.0593f * *b) / 0.6780f;
    *b = pb * ( 2 * (1 - Kb) ) + y;
    *r = pr * ( 2 * (1 - Kr) ) + y;
    *g = (y - Kr * *r - Kb * *b) / (1 - Kr - Kb);
#undef Kb
#undef Kr
} // yuv_to_rgb

// R'G'B' in the range 0-1 to Y'UV (Y' in the range 0-1, U in the range -0.436 - 0.436,
// V in the range -0.615 - 0.615)
void
rgb_to_yuv601(float r,
              float g,
              float b,
              float *y,
              float *u,
              float *v)
{
    /// ref: https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
    *y =  0.299f   * r + 0.587f   * g + 0.114f  * b;
    *u = -0.14713f * r - 0.28886f * g + 0.436f  * b;
    *v =  0.615f   * r - 0.51499f * g - 0.10001 * b;
}

// Y'UV (Y' in the range 0-1, U in the range -0.436 - 0.436,
// V in the range -0.615 - 0.615) to R'G'B' in the range 0-1
void
yuv_to_rgb601(float y,
              float u,
              float v,
              float *r,
              float *g,
              float *b)
{
    /// ref: https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
    *r = y                + 1.13983f * v,
    *g = y - 0.39465f * u - 0.58060f * v;
    *b = y + 2.03211f * u;
} // yuv_to_rgb

// R'G'B' in the range 0-1 to Y'UV (Y' in the range 0-1, U in the range -0.436 - 0.436,
// V in the range -0.615 - 0.615)
void
rgb_to_yuv709(float r,
              float g,
              float b,
              float *y,
              float *u,
              float *v)
{
    /// ref: https://en.wikipedia.org/wiki/YUV#HDTV_with_BT.709
    *y =  0.2126f  * r + 0.7152f  * g + 0.0722f  * b;
    *u = -0.09991f * r - 0.33609f * g + 0.436f   * b;
    *v =  0.615f   * r - 0.55861f * g - 0.05639f * b;
}

// Y'UV (Y in the range 0-1, U in the range -0.436 - 0.436,
// V in the range -0.615 - 0.615) to R'G'B' in the range 0-1
void
yuv_to_rgb709(float y,
              float u,
              float v,
              float *r,
              float *g,
              float *b)
{
    /// ref: https://en.wikipedia.org/wiki/YUV#HDTV_with_BT.709
    *r = y               + 1.28033f * v,
    *g = y - 0.21482f * u - 0.38059f * v;
    *b = y + 2.12798f * u;
} // yuv_to_rgb

static inline
float
labf(float x)
{
    return ( (x) >= 0.008856f ? ( std::pow(x, (float)1 / 3) ) : (7.787f * x + 16.0f / 116) );
}

// Convert pixel values from XYZ to Lab color spaces.
// Uses the standard D65 white point.
void
xyz_to_lab(float x,
           float y,
           float z,
           float *l,
           float *a,
           float *b)
{
    const float fx = labf( x / (0.412453f + 0.357580f + 0.180423f) );
    const float fy = labf( y / (0.212671f + 0.715160f + 0.072169f) );
    const float fz = labf( z / (0.019334f + 0.119193f + 0.950227f) );

    *l = 116 * fy - 16;
    *a = 500 * (fx - fy);
    *b = 200 * (fy - fz);
}

static inline
float
labfi(float x)
{
    return ( x >= 0.206893f ? (x * x * x) : ( (x - 16.0f / 116) / 7.787f ) );
}

// Convert pixel values from Lab to XYZ color spaces.
// Uses the standard D65 white point.
void
lab_to_xyz(float l,
           float a,
           float b,
           float *x,
           float *y,
           float *z)
{
    const float cy = (l + 16) / 116;

    *y = (0.212671f + 0.715160f + 0.072169f) * labfi(cy);
    const float cx = a / 500 + cy;
    *x = (0.412453f + 0.357580f + 0.180423f) * labfi(cx);
    const float cz = cy - b / 200;
    *z = (0.019334f + 0.119193f + 0.950227f) * labfi(cz);
}

// Convert pixel values from RGB to Lab color spaces.
// Uses the standard D65 white point.
void
rgb709_to_lab(float r,
              float g,
              float b,
              float *l,
              float *a,
              float *b_)
{
    float x, y, z;

    rgb709_to_xyz(r, g, b, &x, &y, &z);
    xyz_to_lab(x, y, z, l, a, b_);
}

// Convert pixel values from RGB to Lab color spaces.
// Uses the standard D65 white point.
void
lab_to_rgb709(float l,
              float a,
              float b,
              float *r,
              float *g,
              float *b_)
{
    float x, y, z;

    lab_to_xyz(l, a, b, &x, &y, &z);
    xyz_to_rgb709(x, y, z, r, g, b_);
}
}         // namespace Color
} //namespace OFX

