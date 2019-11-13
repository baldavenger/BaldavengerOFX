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
 * Small utility to draw text using OpenGL.
 * This code is based on the free glut source code.
 *
 * Copyright (c) 1999-2000 Pawel W. Olszta. All Rights Reserved.
 * Written by Pawel W. Olszta, <olszta@sourceforge.net>
 * Creation date: Thu Dec 2 1999
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * PAWEL W. OLSZTA BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#include "ofxsOGLTextRenderer.h"
#include "ofxsOGLFontUtils.h"

#include <cstdlib>

namespace  {
const OFX::SFG_Font*
getFont(OFX::TextRenderer::Font font)
{
    switch (font) {
    case OFX::TextRenderer::FONT_FIXED_8_X_13:

        return &OFX::fgFontFixed8x13;
        break;
    case OFX::TextRenderer::FONT_FIXED_9_X_15:

        return &OFX::fgFontFixed9x15;
        break;
    case OFX::TextRenderer::FONT_HELVETICA_10:

        return &OFX::fgFontHelvetica10;
        break;
    case OFX::TextRenderer::FONT_HELVETICA_12:

        return &OFX::fgFontHelvetica12;
        break;
    case OFX::TextRenderer::FONT_HELVETICA_18:

        return &OFX::fgFontHelvetica18;
        break;
    case OFX::TextRenderer::FONT_TIMES_ROMAN_10:

        return &OFX::fgFontTimesRoman10;
        break;
    case OFX::TextRenderer::FONT_TIMES_ROMAN_24:

        return &OFX::fgFontTimesRoman24;
        break;
    default:

        return (const OFX::SFG_Font*)NULL;
        break;
    }
}
}

void
OFX::TextRenderer::bitmapString(const char *string,
                                TextRenderer::Font f)
{
    unsigned char c;
    float x = 0.0f;
    const SFG_Font* font = getFont(f);

    if (!font) {
        return;
    }
    if (!string || !*string) {
        return;
    }

    glPushClientAttrib( GL_CLIENT_PIXEL_STORE_BIT );
    glPixelStorei( GL_UNPACK_SWAP_BYTES,  GL_FALSE );
    glPixelStorei( GL_UNPACK_LSB_FIRST,   GL_FALSE );
    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0        );
    glPixelStorei( GL_UNPACK_SKIP_ROWS,   0        );
    glPixelStorei( GL_UNPACK_SKIP_PIXELS, 0        );
    glPixelStorei( GL_UNPACK_ALIGNMENT,   1        );

    /*
     * Step through the string, drawing each character.
     * A newline will simply translate the next character's insertion
     * point back to the start of the line and down one line.
     */
    while ( ( c = *string++) ) {
        if (c == '\n') {
            glBitmap ( 0, 0, 0, 0, -x, (float) -font->Height, NULL );
            x = 0.0f;
        } else {   /* Not an EOL, draw the bitmap character */
            const GLubyte* face = font->Characters[ c ];

            glBitmap(
                face[ 0 ], font->Height,          /* Bitmap's width and height    */
                font->xorig, font->yorig,         /* The origin in the font glyph */
                ( float )( face[ 0 ] ), 0.0f,     /* The raster advance; inc. x,y */
                ( face + 1 )                      /* The packed bitmap data...    */
                );

            x += ( float )( face[ 0 ] );
        }
    }

    glPopClientAttrib( );
}

void
OFX::TextRenderer::bitmapString(double x,
                                double y,
                                const char*string,
                                TextRenderer::Font font)
{
    //glPushAttrib(GL_CURRENT_BIT); // caller is responsible for protecting attribs
    glRasterPos2d(x, y);
    bitmapString(string, font);
    //glPopAttrib();
}
