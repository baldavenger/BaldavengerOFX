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

#ifndef openfx_supportext_ofxsOGLDebug_h
#define openfx_supportext_ofxsOGLDebug_h

inline const char*
glErrorString(GLenum errorCode)
{
    static const struct
    {
        GLenum code;
        const char *string;
    }

    errors[] =
    {
        /* GL */
        {GL_NO_ERROR, "no error"},
        {GL_INVALID_ENUM, "invalid enumerant"},
        {GL_INVALID_VALUE, "invalid value"},
        {GL_INVALID_OPERATION, "invalid operation"},
        {GL_STACK_OVERFLOW, "stack overflow"},
        {GL_STACK_UNDERFLOW, "stack underflow"},
        {GL_OUT_OF_MEMORY, "out of memory"},
#ifdef GL_EXT_histogram
        {GL_TABLE_TOO_LARGE, "table too large"},
#endif
#ifdef GL_EXT_framebuffer_object
        {GL_INVALID_FRAMEBUFFER_OPERATION_EXT, "invalid framebuffer operation"},
#endif

        {0, NULL }
    };
    int i;

    for (i = 0; errors[i].string; i++) {
        if (errors[i].code == errorCode) {
            return errors[i].string;
        }
    }

    return NULL;
}

/* *INDENT-OFF* */

inline const char*
glGetEnumString(GLenum v)
{
    switch (v) {
        case 0x0000: return "GL_ZERO/GL_NO_ERROR/GL_POINTS";
        case 0x0001: return "GL_ONE/GL_LINES";
        case 0x0002: return "GL_LINE_LOOP";
        case 0x0003: return "GL_LINE_STRIP";
        case 0x0004: return "GL_TRIANGLES";
        case 0x0005: return "GL_TRIANGLE_STRIP";
        case 0x0006: return "GL_TRIANGLE_FAN";
        case 0x0100: return "GL_DEPTH_BUFFER_BIT";
        case 0x0104: return "GL_ADD";
        case 0x0200: return "GL_NEVER";
        case 0x0201: return "GL_LESS";
        case 0x0202: return "GL_EQUAL";
        case 0x0203: return "GL_LEQUAL";
        case 0x0204: return "GL_GREATER";
        case 0x0205: return "GL_NOTEQUAL";
        case 0x0206: return "GL_GEQUAL";
        case 0x0207: return "GL_ALWAYS";
        case 0x0300: return "GL_SRC_COLOR";
        case 0x0301: return "GL_ONE_MINUS_SRC_COLOR";
        case 0x0302: return "GL_SRC_ALPHA";
        case 0x0303: return "GL_ONE_MINUS_SRC_ALPHA";
        case 0x0304: return "GL_DST_ALPHA";
        case 0x0305: return "GL_ONE_MINUS_DST_ALPHA";
        case 0x0306: return "GL_DST_COLOR";
        case 0x0307: return "GL_ONE_MINUS_DST_COLOR";
        case 0x0308: return "GL_SRC_ALPHA_SATURATE";
        case 0x0400: return "GL_STENCIL_BUFFER_BIT";
        case 0x0404: return "GL_FRONT";
        case 0x0405: return "GL_BACK";
        case 0x0408: return "GL_FRONT_AND_BACK";
        case 0x0500: return "GL_INVALID_ENUM";
        case 0x0501: return "GL_INVALID_VALUE";
        case 0x0502: return "GL_INVALID_OPERATION";
        case 0x0503: return "GL_STACK_OVERFLOW";
        case 0x0504: return "GL_STACK_UNDERFLOW";
        case 0x0505: return "GL_OUT_OF_MEMORY";
        case 0x0506: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case 0x0800: return "GL_EXP";
        case 0x0801: return "GL_EXP2";
        case 0x0900: return "GL_CW";
        case 0x0901: return "GL_CCW";
        case 0x0B00: return "GL_CURRENT_COLOR";
        case 0x0B02: return "GL_CURRENT_NORMAL";
        case 0x0B03: return "GL_CURRENT_TEXTURE_COORDS";
        case 0x0B10: return "GL_POINT_SMOOTH";
        case 0x0B11: return "GL_POINT_SIZE";
        case 0x0B12: return "GL_SMOOTH_POINT_SIZE_RANGE";
        case 0x0B20: return "GL_LINE_SMOOTH";
        case 0x0B21: return "GL_LINE_WIDTH";
        case 0x0B22: return "GL_SMOOTH_LINE_WIDTH_RANGE";
        case 0x0B44: return "GL_CULL_FACE";
        case 0x0B45: return "GL_CULL_FACE_MODE";
        case 0x0B46: return "GL_FRONT_FACE";
        case 0x0B50: return "GL_LIGHTING";
        case 0x0B52: return "GL_LIGHT_MODEL_TWO_SIDE";
        case 0x0B53: return "GL_LIGHT_MODEL_AMBIENT";
        case 0x0B54: return "GL_SHADE_MODEL";
        case 0x0B57: return "GL_COLOR_MATERIAL";
        case 0x0B60: return "GL_FOG";
        case 0x0B62: return "GL_FOG_DENSITY";
        case 0x0B63: return "GL_FOG_START";
        case 0x0B64: return "GL_FOG_END";
        case 0x0B65: return "GL_FOG_MODE";
        case 0x0B66: return "GL_FOG_COLOR";
        case 0x0B70: return "GL_DEPTH_RANGE";
        case 0x0B71: return "GL_DEPTH_TEST";
        case 0x0B72: return "GL_DEPTH_WRITEMASK";
        case 0x0B73: return "GL_DEPTH_CLEAR_VALUE";
        case 0x0B74: return "GL_DEPTH_FUNC";
        case 0x0B90: return "GL_STENCIL_TEST";
        case 0x0B91: return "GL_STENCIL_CLEAR_VALUE";
        case 0x0B92: return "GL_STENCIL_FUNC";
        case 0x0B93: return "GL_STENCIL_VALUE_MASK";
        case 0x0B94: return "GL_STENCIL_FAIL";
        case 0x0B95: return "GL_STENCIL_PASS_DEPTH_FAIL";
        case 0x0B96: return "GL_STENCIL_PASS_DEPTH_PASS";
        case 0x0B97: return "GL_STENCIL_REF";
        case 0x0B98: return "GL_STENCIL_WRITEMASK";
        case 0x0BA0: return "GL_MATRIX_MODE";
        case 0x0BA1: return "GL_NORMALIZE";
        case 0x0BA2: return "GL_VIEWPORT";
        case 0x0BA3: return "GL_MODELVIEW_STACK_DEPTH";
        case 0x0BA4: return "GL_PROJECTION_STACK_DEPTH";
        case 0x0BA5: return "GL_TEXTURE_STACK_DEPTH";
        case 0x0BA6: return "GL_MODELVIEW_MATRIX";
        case 0x0BA7: return "GL_PROJECTION_MATRIX";
        case 0x0BA8: return "GL_TEXTURE_MATRIX";
        case 0x0BC0: return "GL_ALPHA_TEST";
        case 0x0BC1: return "GL_ALPHA_TEST_FUNC";
        case 0x0BC2: return "GL_ALPHA_TEST_REF";
        case 0x0BD0: return "GL_DITHER";
        case 0x0BE0: return "GL_BLEND_DST";
        case 0x0BE1: return "GL_BLEND_SRC";
        case 0x0BE2: return "GL_BLEND";
        case 0x0BF0: return "GL_LOGIC_OP_MODE";
        case 0x0BF2: return "GL_COLOR_LOGIC_OP";
        case 0x0C10: return "GL_SCISSOR_BOX";
        case 0x0C11: return "GL_SCISSOR_TEST";
        case 0x0C22: return "GL_COLOR_CLEAR_VALUE";
        case 0x0C23: return "GL_COLOR_WRITEMASK";
        case 0x0C50: return "GL_PERSPECTIVE_CORRECTION_HINT";
        case 0x0C51: return "GL_POINT_SMOOTH_HINT";
        case 0x0C52: return "GL_LINE_SMOOTH_HINT";
        case 0x0C54: return "GL_FOG_HINT";
        case 0x0CF5: return "GL_UNPACK_ALIGNMENT";
        case 0x0D05: return "GL_PACK_ALIGNMENT";
        case 0x0D1C: return "GL_ALPHA_SCALE";
        case 0x0D31: return "GL_MAX_LIGHTS";
        case 0x0D32: return "GL_MAX_CLIP_PLANES";
        case 0x0D33: return "GL_MAX_TEXTURE_SIZE";
        case 0x0D36: return "GL_MAX_MODELVIEW_STACK_DEPTH";
        case 0x0D38: return "GL_MAX_PROJECTION_STACK_DEPTH";
        case 0x0D39: return "GL_MAX_TEXTURE_STACK_DEPTH";
        case 0x0D3A: return "GL_MAX_VIEWPORT_DIMS";
        case 0x0D50: return "GL_SUBPIXEL_BITS";
        case 0x0D52: return "GL_RED_BITS";
        case 0x0D53: return "GL_GREEN_BITS";
        case 0x0D54: return "GL_BLUE_BITS";
        case 0x0D55: return "GL_ALPHA_BITS";
        case 0x0D56: return "GL_DEPTH_BITS";
        case 0x0D57: return "GL_STENCIL_BITS";
        case 0x0DE1: return "GL_TEXTURE_2D";
        case 0x1100: return "GL_DONT_CARE";
        case 0x1101: return "GL_FASTEST";
        case 0x1102: return "GL_NICEST";
        case 0x1200: return "GL_AMBIENT";
        case 0x1201: return "GL_DIFFUSE";
        case 0x1202: return "GL_SPECULAR";
        case 0x1203: return "GL_POSITION";
        case 0x1204: return "GL_SPOT_DIRECTION";
        case 0x1205: return "GL_SPOT_EXPONENT";
        case 0x1206: return "GL_SPOT_CUTOFF";
        case 0x1207: return "GL_CONSTANT_ATTENUATION";
        case 0x1208: return "GL_LINEAR_ATTENUATION";
        case 0x1209: return "GL_QUADRATIC_ATTENUATION";
        case 0x1400: return "GL_BYTE";
        case 0x1401: return "GL_UNSIGNED_BYTE";
        case 0x1402: return "GL_SHORT";
        case 0x1403: return "GL_UNSIGNED_SHORT";
        case 0x1404: return "GL_INT";
        case 0x1405: return "GL_UNSIGNED_INT";
        case 0x1406: return "GL_FLOAT";
        case 0x140C: return "GL_FIXED";
        case 0x1500: return "GL_CLEAR";
        case 0x1501: return "GL_AND";
        case 0x1502: return "GL_AND_REVERSE";
        case 0x1503: return "GL_COPY";
        case 0x1504: return "GL_AND_INVERTED";
        case 0x1505: return "GL_NOOP";
        case 0x1506: return "GL_XOR";
        case 0x1507: return "GL_OR";
        case 0x1508: return "GL_NOR";
        case 0x1509: return "GL_EQUIV";
        case 0x150A: return "GL_INVERT";
        case 0x150B: return "GL_OR_REVERSE";
        case 0x150C: return "GL_COPY_INVERTED";
        case 0x150D: return "GL_OR_INVERTED";
        case 0x150E: return "GL_NAND";
        case 0x150F: return "GL_SET";
        case 0x1600: return "GL_EMISSION";
        case 0x1601: return "GL_SHININESS";
        case 0x1602: return "GL_AMBIENT_AND_DIFFUSE";
        case 0x1700: return "GL_MODELVIEW";
        case 0x1701: return "GL_PROJECTION";
        case 0x1702: return "GL_TEXTURE";
        case 0x1901: return "GL_STENCIL_INDEX";
        case 0x1902: return "GL_DEPTH_COMPONENT";
        case 0x1906: return "GL_ALPHA";
        case 0x1907: return "GL_RGB";
        case 0x1908: return "GL_RGBA";
        case 0x1909: return "GL_LUMINANCE";
        case 0x190A: return "GL_LUMINANCE_ALPHA";
        case 0x1D00: return "GL_FLAT";
        case 0x1D01: return "GL_SMOOTH";
        case 0x1E00: return "GL_KEEP";
        case 0x1E01: return "GL_REPLACE";
        case 0x1E02: return "GL_INCR";
        case 0x1E03: return "GL_DECR";
        case 0x1F00: return "GL_VENDOR";
        case 0x1F01: return "GL_RENDERER";
        case 0x1F02: return "GL_VERSION";
        case 0x1F03: return "GL_EXTENSIONS";
        case 0x2100: return "GL_MODULATE";
        case 0x2101: return "GL_DECAL";
        case 0x2200: return "GL_TEXTURE_ENV_MODE";
        case 0x2201: return "GL_TEXTURE_ENV_COLOR";
        case 0x2300: return "GL_TEXTURE_ENV";
        case 0x2500: return "GL_TEXTURE_GEN_MODE_OES";
        case 0x2600: return "GL_NEAREST";
        case 0x2601: return "GL_LINEAR";
        case 0x2700: return "GL_NEAREST_MIPMAP_NEAREST";
        case 0x2701: return "GL_LINEAR_MIPMAP_NEAREST";
        case 0x2702: return "GL_NEAREST_MIPMAP_LINEAR";
        case 0x2703: return "GL_LINEAR_MIPMAP_LINEAR";
        case 0x2800: return "GL_TEXTURE_MAG_FILTER";
        case 0x2801: return "GL_TEXTURE_MIN_FILTER";
        case 0x2802: return "GL_TEXTURE_WRAP_S";
        case 0x2803: return "GL_TEXTURE_WRAP_T";
        case 0x2901: return "GL_REPEAT";
        case 0x2A00: return "GL_POLYGON_OFFSET_UNITS";
        case 0x3000: return "GL_CLIP_PLANE0";
        case 0x3001: return "GL_CLIP_PLANE1";
        case 0x3002: return "GL_CLIP_PLANE2";
        case 0x3003: return "GL_CLIP_PLANE3";
        case 0x3004: return "GL_CLIP_PLANE4";
        case 0x3005: return "GL_CLIP_PLANE5";
        case 0x300E: return "GL_CONTEXT_LOST";
        case 0x4000: return "GL_LIGHT0";
        case 0x4001: return "GL_LIGHT1";
        case 0x4002: return "GL_LIGHT2";
        case 0x4003: return "GL_LIGHT3";
        case 0x4004: return "GL_LIGHT4";
        case 0x4005: return "GL_LIGHT5";
        case 0x4006: return "GL_LIGHT6";
        case 0x4007: return "GL_LIGHT7";
        case 0x8001: return "GL_CONSTANT_COLOR";
        case 0x8002: return "GL_ONE_MINUS_CONSTANT_COLOR";
        case 0x8003: return "GL_CONSTANT_ALPHA";
        case 0x8004: return "GL_ONE_MINUS_CONSTANT_ALPHA";
        case 0x8005: return "GL_BLEND_COLOR";
        case 0x8006: return "GL_FUNC_ADD";
        case 0x8009: return "GL_BLEND_EQUATION";
        case 0x800A: return "GL_FUNC_SUBTRACT";
        case 0x800B: return "GL_FUNC_REVERSE_SUBTRACT";
        case 0x8033: return "GL_UNSIGNED_SHORT_4_4_4_4";
        case 0x8034: return "GL_UNSIGNED_SHORT_5_5_5_1";
        case 0x8037: return "GL_POLYGON_OFFSET_FILL";
        case 0x8038: return "GL_POLYGON_OFFSET_FACTOR";
        case 0x803A: return "GL_RESCALE_NORMAL";
        case 0x8056: return "GL_RGBA4";
        case 0x8057: return "GL_RGB5_A1";
        case 0x8069: return "GL_TEXTURE_BINDING_2D";
        case 0x8074: return "GL_VERTEX_ARRAY";
        case 0x8075: return "GL_NORMAL_ARRAY";
        case 0x8076: return "GL_COLOR_ARRAY";
        case 0x8078: return "GL_TEXTURE_COORD_ARRAY";
        case 0x807A: return "GL_VERTEX_ARRAY_SIZE";
        case 0x807B: return "GL_VERTEX_ARRAY_TYPE";
        case 0x807C: return "GL_VERTEX_ARRAY_STRIDE";
        case 0x807E: return "GL_NORMAL_ARRAY_TYPE";
        case 0x807F: return "GL_NORMAL_ARRAY_STRIDE";
        case 0x8081: return "GL_COLOR_ARRAY_SIZE";
        case 0x8082: return "GL_COLOR_ARRAY_TYPE";
        case 0x8083: return "GL_COLOR_ARRAY_STRIDE";
        case 0x8088: return "GL_TEXTURE_COORD_ARRAY_SIZE";
        case 0x8089: return "GL_TEXTURE_COORD_ARRAY_TYPE";
        case 0x808A: return "GL_TEXTURE_COORD_ARRAY_STRIDE";
        case 0x808E: return "GL_VERTEX_ARRAY_POINTER";
        case 0x808F: return "GL_NORMAL_ARRAY_POINTER";
        case 0x8090: return "GL_COLOR_ARRAY_POINTER";
        case 0x8092: return "GL_TEXTURE_COORD_ARRAY_POINTER";
        case 0x809D: return "GL_MULTISAMPLE";
        case 0x809E: return "GL_SAMPLE_ALPHA_TO_COVERAGE";
        case 0x809F: return "GL_SAMPLE_ALPHA_TO_ONE";
        case 0x80A0: return "GL_SAMPLE_COVERAGE";
        case 0x80A8: return "GL_SAMPLE_BUFFERS";
        case 0x80A9: return "GL_SAMPLES";
        case 0x80AA: return "GL_SAMPLE_COVERAGE_VALUE";
        case 0x80AB: return "GL_SAMPLE_COVERAGE_INVERT";
        case 0x80C8: return "GL_BLEND_DST_RGB";
        case 0x80C9: return "GL_BLEND_SRC_RGB";
        case 0x80CA: return "GL_BLEND_DST_ALPHA";
        case 0x80CB: return "GL_BLEND_SRC_ALPHA";
        case 0x8126: return "GL_POINT_SIZE_MIN";
        case 0x8127: return "GL_POINT_SIZE_MAX";
        case 0x8128: return "GL_POINT_FADE_THRESHOLD_SIZE";
        case 0x8129: return "GL_POINT_DISTANCE_ATTENUATION";
        case 0x812F: return "GL_CLAMP_TO_EDGE";
        case 0x8191: return "GL_GENERATE_MIPMAP";
        case 0x8192: return "GL_GENERATE_MIPMAP_HINT";
        case 0x81A5: return "GL_DEPTH_COMPONENT16";
        case 0x81A6: return "GL_DEPTH_COMPONENT24_OES";
        case 0x81A7: return "GL_DEPTH_COMPONENT32_OES";
        case 0x8363: return "GL_UNSIGNED_SHORT_5_6_5";
        case 0x8370: return "GL_MIRRORED_REPEAT";
        case 0x846D: return "GL_ALIASED_POINT_SIZE_RANGE";
        case 0x846E: return "GL_ALIASED_LINE_WIDTH_RANGE";
        case 0x84C0: return "GL_TEXTURE0";
        case 0x84C1: return "GL_TEXTURE1";
        case 0x84C2: return "GL_TEXTURE2";
        case 0x84C3: return "GL_TEXTURE3";
        case 0x84C4: return "GL_TEXTURE4";
        case 0x84C5: return "GL_TEXTURE5";
        case 0x84C6: return "GL_TEXTURE6";
        case 0x84C7: return "GL_TEXTURE7";
        case 0x84C8: return "GL_TEXTURE8";
        case 0x84C9: return "GL_TEXTURE9";
        case 0x84CA: return "GL_TEXTURE10";
        case 0x84CB: return "GL_TEXTURE11";
        case 0x84CC: return "GL_TEXTURE12";
        case 0x84CD: return "GL_TEXTURE13";
        case 0x84CE: return "GL_TEXTURE14";
        case 0x84CF: return "GL_TEXTURE15";
        case 0x84D0: return "GL_TEXTURE16";
        case 0x84D1: return "GL_TEXTURE17";
        case 0x84D2: return "GL_TEXTURE18";
        case 0x84D3: return "GL_TEXTURE19";
        case 0x84D4: return "GL_TEXTURE20";
        case 0x84D5: return "GL_TEXTURE21";
        case 0x84D6: return "GL_TEXTURE22";
        case 0x84D7: return "GL_TEXTURE23";
        case 0x84D8: return "GL_TEXTURE24";
        case 0x84D9: return "GL_TEXTURE25";
        case 0x84DA: return "GL_TEXTURE26";
        case 0x84DB: return "GL_TEXTURE27";
        case 0x84DC: return "GL_TEXTURE28";
        case 0x84DD: return "GL_TEXTURE29";
        case 0x84DE: return "GL_TEXTURE30";
        case 0x84DF: return "GL_TEXTURE31";
        case 0x84E0: return "GL_ACTIVE_TEXTURE";
        case 0x84E1: return "GL_CLIENT_ACTIVE_TEXTURE";
        case 0x84E2: return "GL_MAX_TEXTURE_UNITS";
        case 0x84E7: return "GL_SUBTRACT";
        case 0x84E8: return "GL_MAX_RENDERBUFFER_SIZE";
        case 0x8507: return "GL_INCR_WRAP";
        case 0x8508: return "GL_DECR_WRAP";
        case 0x8511: return "GL_NORMAL_MAP_OES";
        case 0x8512: return "GL_REFLECTION_MAP_OES";
        case 0x8513: return "GL_TEXTURE_CUBE_MAP_OES";
        case 0x8514: return "GL_TEXTURE_BINDING_CUBE_MAP_OES";
        case 0x8515: return "GL_TEXTURE_CUBE_MAP_POSITIVE_X_OES";
        case 0x8516: return "GL_TEXTURE_CUBE_MAP_NEGATIVE_X_OES";
        case 0x8517: return "GL_TEXTURE_CUBE_MAP_POSITIVE_Y_OES";
        case 0x8518: return "GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_OES";
        case 0x8519: return "GL_TEXTURE_CUBE_MAP_POSITIVE_Z_OES";
        case 0x851A: return "GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_OES";
        case 0x851C: return "GL_MAX_CUBE_MAP_TEXTURE_SIZE_OES";
        case 0x8570: return "GL_COMBINE";
        case 0x8571: return "GL_COMBINE_RGB";
        case 0x8572: return "GL_COMBINE_ALPHA";
        case 0x8573: return "GL_RGB_SCALE";
        case 0x8574: return "GL_ADD_SIGNED";
        case 0x8575: return "GL_INTERPOLATE";
        case 0x8576: return "GL_CONSTANT";
        case 0x8577: return "GL_PRIMARY_COLOR";
        case 0x8578: return "GL_PREVIOUS";
        case 0x8580: return "GL_SRC0_RGB";
        case 0x8581: return "GL_SRC1_RGB";
        case 0x8582: return "GL_SRC2_RGB";
        case 0x8588: return "GL_SRC0_ALPHA";
        case 0x8589: return "GL_SRC1_ALPHA";
        case 0x858A: return "GL_SRC2_ALPHA";
        case 0x8590: return "GL_OPERAND0_RGB";
        case 0x8591: return "GL_OPERAND1_RGB";
        case 0x8592: return "GL_OPERAND2_RGB";
        case 0x8598: return "GL_OPERAND0_ALPHA";
        case 0x8599: return "GL_OPERAND1_ALPHA";
        case 0x859A: return "GL_OPERAND2_ALPHA";
        case 0x8622: return "GL_VERTEX_ATTRIB_ARRAY_ENABLED";
        case 0x8623: return "GL_VERTEX_ATTRIB_ARRAY_SIZE";
        case 0x8624: return "GL_VERTEX_ATTRIB_ARRAY_STRIDE";
        case 0x8625: return "GL_VERTEX_ATTRIB_ARRAY_TYPE";
        case 0x8626: return "GL_CURRENT_VERTEX_ATTRIB";
        case 0x8645: return "GL_VERTEX_ATTRIB_ARRAY_POINTER";
        case 0x86A2: return "GL_NUM_COMPRESSED_TEXTURE_FORMATS";
        case 0x86A3: return "GL_COMPRESSED_TEXTURE_FORMATS";
        case 0x86AE: return "GL_DOT3_RGB";
        case 0x86AF: return "GL_DOT3_RGBA";
        case 0x8764: return "GL_BUFFER_SIZE";
        case 0x8765: return "GL_BUFFER_USAGE";
        case 0x8800: return "GL_STENCIL_BACK_FUNC";
        case 0x8801: return "GL_STENCIL_BACK_FAIL";
        case 0x8802: return "GL_STENCIL_BACK_PASS_DEPTH_FAIL";
        case 0x8803: return "GL_STENCIL_BACK_PASS_DEPTH_PASS";
        case 0x883D: return "GL_BLEND_EQUATION_ALPHA";
        case 0x8861: return "GL_POINT_SPRITE_OES";
        case 0x8862: return "GL_COORD_REPLACE_OES";
        case 0x8869: return "GL_MAX_VERTEX_ATTRIBS";
        case 0x886A: return "GL_VERTEX_ATTRIB_ARRAY_NORMALIZED";
        case 0x8872: return "GL_MAX_TEXTURE_IMAGE_UNITS";
        case 0x8892: return "GL_ARRAY_BUFFER";
        case 0x8893: return "GL_ELEMENT_ARRAY_BUFFER";
        case 0x8894: return "GL_ARRAY_BUFFER_BINDING";
        case 0x8895: return "GL_ELEMENT_ARRAY_BUFFER_BINDING";
        case 0x8896: return "GL_VERTEX_ARRAY_BUFFER_BINDING";
        case 0x8897: return "GL_NORMAL_ARRAY_BUFFER_BINDING";
        case 0x8898: return "GL_COLOR_ARRAY_BUFFER_BINDING";
        case 0x889A: return "GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING";
        case 0x889F: return "GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING";
        case 0x88E0: return "GL_STREAM_DRAW";
        case 0x88E4: return "GL_STATIC_DRAW";
        case 0x88E8: return "GL_DYNAMIC_DRAW";
        case 0x898A: return "GL_POINT_SIZE_ARRAY_TYPE_OES";
        case 0x898B: return "GL_POINT_SIZE_ARRAY_STRIDE_OES";
        case 0x898C: return "GL_POINT_SIZE_ARRAY_POINTER_OES";
        case 0x8B30: return "GL_FRAGMENT_SHADER";
        case 0x8B31: return "GL_VERTEX_SHADER";
        case 0x8B4C: return "GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS";
        case 0x8B4D: return "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS";
        case 0x8B4F: return "GL_SHADER_TYPE";
        case 0x8B50: return "GL_FLOAT_VEC2";
        case 0x8B51: return "GL_FLOAT_VEC3";
        case 0x8B52: return "GL_FLOAT_VEC4";
        case 0x8B53: return "GL_INT_VEC2";
        case 0x8B54: return "GL_INT_VEC3";
        case 0x8B55: return "GL_INT_VEC4";
        case 0x8B56: return "GL_BOOL";
        case 0x8B57: return "GL_BOOL_VEC2";
        case 0x8B58: return "GL_BOOL_VEC3";
        case 0x8B59: return "GL_BOOL_VEC4";
        case 0x8B5A: return "GL_FLOAT_MAT2";
        case 0x8B5B: return "GL_FLOAT_MAT3";
        case 0x8B5C: return "GL_FLOAT_MAT4";
        case 0x8B5E: return "GL_SAMPLER_2D";
        case 0x8B60: return "GL_SAMPLER_CUBE";
        case 0x8B80: return "GL_DELETE_STATUS";
        case 0x8B81: return "GL_COMPILE_STATUS";
        case 0x8B82: return "GL_LINK_STATUS";
        case 0x8B83: return "GL_VALIDATE_STATUS";
        case 0x8B84: return "GL_INFO_LOG_LENGTH";
        case 0x8B85: return "GL_ATTACHED_SHADERS";
        case 0x8B86: return "GL_ACTIVE_UNIFORMS";
        case 0x8B87: return "GL_ACTIVE_UNIFORM_MAX_LENGTH";
        case 0x8B88: return "GL_SHADER_SOURCE_LENGTH";
        case 0x8B89: return "GL_ACTIVE_ATTRIBUTES";
        case 0x8B8A: return "GL_ACTIVE_ATTRIBUTE_MAX_LENGTH";
        case 0x8B8C: return "GL_SHADING_LANGUAGE_VERSION";
        case 0x8B8D: return "GL_CURRENT_PROGRAM";
        case 0x8B90: return "GL_PALETTE4_RGB8_OES";
        case 0x8B91: return "GL_PALETTE4_RGBA8_OES";
        case 0x8B92: return "GL_PALETTE4_R5_G6_B5_OES";
        case 0x8B93: return "GL_PALETTE4_RGBA4_OES";
        case 0x8B94: return "GL_PALETTE4_RGB5_A1_OES";
        case 0x8B95: return "GL_PALETTE8_RGB8_OES";
        case 0x8B96: return "GL_PALETTE8_RGBA8_OES";
        case 0x8B97: return "GL_PALETTE8_R5_G6_B5_OES";
        case 0x8B98: return "GL_PALETTE8_RGBA4_OES";
        case 0x8B99: return "GL_PALETTE8_RGB5_A1_OES";
        case 0x8B9A: return "GL_IMPLEMENTATION_COLOR_READ_TYPE";
        case 0x8B9B: return "GL_IMPLEMENTATION_COLOR_READ_FORMAT";
        case 0x8B9C: return "GL_POINT_SIZE_ARRAY_OES";
        case 0x8B9D: return "GL_POINT_SIZE_ARRAY_OES";
        case 0x8B9F: return "GL_POINT_SIZE_ARRAY_BUFFER_BINDING_OES";
        case 0x8CA3: return "GL_STENCIL_BACK_REF";
        case 0x8CA4: return "GL_STENCIL_BACK_VALUE_MASK";
        case 0x8CA5: return "GL_STENCIL_BACK_WRITEMASK";
        case 0x8CA6: return "GL_FRAMEBUFFER_BINDING";
        case 0x8CA7: return "GL_RENDERBUFFER_BINDING";
        case 0x8CD0: return "GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE";
        case 0x8CD1: return "GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME";
        case 0x8CD2: return "GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL";
        case 0x8CD3: return "GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE";
        case 0x8CD5: return "GL_FRAMEBUFFER_COMPLETE";
        case 0x8CD6: return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
        case 0x8CD7: return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
        case 0x8CD9: return "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS";
        case 0x8CDD: return "GL_FRAMEBUFFER_UNSUPPORTED";
        case 0x8CE0: return "GL_COLOR_ATTACHMENT0";
        case 0x8D00: return "GL_DEPTH_ATTACHMENT";
        case 0x8D20: return "GL_STENCIL_ATTACHMENT";
        case 0x8D40: return "GL_FRAMEBUFFER";
        case 0x8D41: return "GL_RENDERBUFFER";
        case 0x8D42: return "GL_RENDERBUFFER_WIDTH";
        case 0x8D43: return "GL_RENDERBUFFER_HEIGHT";
        case 0x8D44: return "GL_RENDERBUFFER_INTERNAL_FORMAT";
        case 0x8D48: return "GL_STENCIL_INDEX8";
        case 0x8D50: return "GL_RENDERBUFFER_RED_SIZE";
        case 0x8D51: return "GL_RENDERBUFFER_GREEN_SIZE";
        case 0x8D52: return "GL_RENDERBUFFER_BLUE_SIZE";
        case 0x8D53: return "GL_RENDERBUFFER_ALPHA_SIZE";
        case 0x8D54: return "GL_RENDERBUFFER_DEPTH_SIZE";
        case 0x8D55: return "GL_RENDERBUFFER_STENCIL_SIZE";
        case 0x8D60: return "GL_TEXTURE_GEN_STR_OES";
        case 0x8D62: return "GL_RGB565";
        case 0x8D64: return "GL_ETC1_RGB8_OES";
        case 0x8D65: return "GL_TEXTURE_EXTERNAL_OES";
        case 0x8D67: return "GL_TEXTURE_BINDING_EXTERNAL_OES";
        case 0x8D68: return "GL_REQUIRED_TEXTURE_IMAGE_UNITS_OES";
        case 0x8DF0: return "GL_LOW_FLOAT";
        case 0x8DF1: return "GL_MEDIUM_FLOAT";
        case 0x8DF2: return "GL_HIGH_FLOAT";
        case 0x8DF3: return "GL_LOW_INT";
        case 0x8DF4: return "GL_MEDIUM_INT";
        case 0x8DF5: return "GL_HIGH_INT";
        case 0x8DF8: return "GL_SHADER_BINARY_FORMATS";
        case 0x8DF9: return "GL_NUM_SHADER_BINARY_FORMATS";
        case 0x8DFA: return "GL_SHADER_COMPILER";
        case 0x8DFB: return "GL_MAX_VERTEX_UNIFORM_VECTORS";
        case 0x8DFC: return "GL_MAX_VARYING_VECTORS";
        case 0x8DFD: return "GL_MAX_FRAGMENT_UNIFORM_VECTORS";
        default: return "<unknown>";
    }
}

/* *INDENT-ON* */

#ifndef DEBUG
#define glCheckError() ( (void)0 )
#else

#include <iostream>

// put a breakpoint in glError to halt the debugger
inline void
glError() {}


#define glCheckError()                                                  \
    {                                                                   \
        GLenum _glerror_ = glGetError();                                \
        if (_glerror_ != GL_NO_ERROR) {                                 \
            std::cout << "GL_ERROR: " << __FILE__ << ":" << __LINE__ << " " << glErrorString(_glerror_) << std::endl; \
            glError();                                                  \
        }                                                               \
    }

#endif // ifndef DEBUG


#endif /* defined(openfx_supportext_ofxsOGLDebug_h) */
