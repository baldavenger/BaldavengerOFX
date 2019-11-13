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

#ifndef openfx_supportext_ofxsOGLUtilities_h
#define openfx_supportext_ofxsOGLUtilities_h

namespace OFX {
/**
 * @brief Loads OpenGL functions using GLAD so that they are available if using glad.h or ofxsOGLFunctions.h
 * Note: this function will loads them once only, subsequent calls to this function do nothing.
 * This is thread-safe (protected by a mutex).
 * An OpenGL context MUST be bound when calling this function. A good place to call it is in the
 * draw function of an interact (if you only use interacts) or in the contextAttached action of your
 * plug-in if you support OpenGL rendering.
 * Note that the OpenGL version loaded is the one specified when generating glad.h
 * @returns true if OpenGL was load successfully, false otherwise.
 * Reasons for failure might be:
 * - opengl32.dll was not found, or libGL.so was not found or OpenGL.framework was not found
 * - glGetString does not return a valid version
 * Note: It does NOT check that required extensions and functions have actually been found,
 * nor that the OpenGL version of the driver matches the version at which glad was generated.
 * Use functions below for that.
 **/
bool ofxsLoadOpenGLOnce();

/**
 * @brief Returns the OpenGL major version loaded by GLAD. This is the version of the client driver
 * and may differ from the version for which GLAD was generated.
 * Note: ofxsLoadOpenGLOnce() must have been called at least once prior to calling this function.
 **/
int getOpenGLMajorVersion();

/**
 * @brief Returns the OpenGL minor version loaded by GLAD. This is the version of the client driver
 * and may differ from the version for which GLAD was generated.
 * Note: ofxsLoadOpenGLOnce() must have been called at least once prior to calling this function.
 **/
int getOpenGLMinorVersion();

/**
 * @brief Returns whether the OpenGL driver of the client support the GL_ARB_texture_float extension.
 * Note: ofxsLoadOpenGLOnce() must have been called at least once prior to calling this function.
 **/
bool getOpenGLSupportsTextureFloat();

/**
 * @brief Returns whether the OpenGL driver of the client support the GL_ARB_framebuffer_object extension.
 * Note that if it is unsupported but GL_EXT_framebuffer_object is supported this function will return true
 * and all functions of the GL_ARB_framebuffer_object extension will be in fact using the functions of
 * GL_EXT_framebuffer_object.
 * Note: ofxsLoadOpenGLOnce() must have been called at least once prior to calling this function.
 **/
bool getOpenGLSupportFramebuffer();

/**
 * @brief Returns whether the OpenGL driver of the client support the GL_ARB_pixel_buffer_object extension.
 * Note that if it is unsupported but GLAD_GL_APPLE_vertex_array_object is supported this function will return true
 * and all functions of the GLAD_GL_ARB_vertex_array_object extension will be in fact using the functions of
 * GLAD_GL_APPLE_vertex_array_object.
 * Note: ofxsLoadOpenGLOnce() must have been called at least once prior to calling this function.
 **/
bool getOpenGLSupportPixelbuffer();

/**
 * @brief Returns whether the OpenGL driver of the client support the GL_ARB_vertex_array_object extension.
 * Note: ofxsLoadOpenGLOnce() must have been called at least once prior to calling this function.
 **/
bool getOpenGLSupportVertexArray();
} // namespace OFX

#endif /* defined(openfx_supportext_ofxsOGLDebug_h) */
