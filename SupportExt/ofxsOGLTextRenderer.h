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


#ifndef openfx_supportext_ofxsOGLTextRenderer_h
#define openfx_supportext_ofxsOGLTextRenderer_h


namespace OFX {
namespace TextRenderer {
enum Font
{
    FONT_FIXED_8_X_13 = 0,
    FONT_FIXED_9_X_15,
    FONT_HELVETICA_10,
    FONT_HELVETICA_12,
    FONT_HELVETICA_18,
    FONT_TIMES_ROMAN_10,
    FONT_TIMES_ROMAN_24
};

/**
 * @brief Draws the text contained in string. This must be a NULL terminated string.
 * @param font The font to use to render. If it doesn't correspond to one of the enum
 * this function will not draw anything.
 **/
void bitmapString(const char *string, TextRenderer::Font font = FONT_HELVETICA_12);

/**
 *@brief Same as strokeString() but translates the OpenGL matrix to the (x,y) position before drawing.
 **/
void bitmapString(double x, double y, const char*string, TextRenderer::Font font = FONT_HELVETICA_12);
} // TextRendered
} // OFX


#endif /* defined(openfx_supportext_ofxsOGLTextRenderer_h) */
