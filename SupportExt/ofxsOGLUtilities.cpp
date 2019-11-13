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

#include "ofxsOGLUtilities.h"

#include "ofxsOGLFunctions.h"

#include "ofxsMultiThread.h"
#ifndef OFX_USE_MULTITHREAD_MUTEX
// some OFX hosts do not have mutex handling in the MT-Suite (e.g. Sony Catalyst Edit)
// prefer using the fast mutex by Marcus Geelnard http://tinythreadpp.bitsnbites.eu/
#include "fast_mutex.h"
#endif
#ifdef OFX_USE_MULTITHREAD_MUTEX
typedef OFX::MultiThread::Mutex Mutex;
typedef OFX::MultiThread::AutoMutex AutoMutex;
#else
typedef tthread::fast_mutex Mutex;
typedef OFX::MultiThread::AutoMutexT<tthread::fast_mutex> AutoMutex;
#endif

static Mutex g_glLoadOnceMutex;
static bool g_glLoaded = false;

extern "C" {
extern int gladLoadGL(void);
struct gladGLversionStruct
{
    int major;
    int minor;
};

typedef void (* GLADcallback)(const char *name, void *funcptr, int len_args, ...);
extern void glad_set_pre_callback(GLADcallback);
extern void glad_set_post_callback(GLADcallback);
extern gladGLversionStruct GLVersion;
extern int GLAD_GL_ARB_vertex_buffer_object;
extern int GLAD_GL_ARB_framebuffer_object;
extern int GLAD_GL_ARB_pixel_buffer_object;
extern int GLAD_GL_ARB_vertex_array_object;
extern int GLAD_GL_ARB_texture_float;
extern int GLAD_GL_EXT_framebuffer_object;
extern int GLAD_GL_APPLE_vertex_array_object;

typedef GLboolean (* PFNGLISRENDERBUFFEREXTPROC)(GLuint renderbuffer);
extern PFNGLISRENDERBUFFEREXTPROC glad_glIsRenderbufferEXT;
extern PFNGLISRENDERBUFFEREXTPROC glad_glIsRenderbuffer;

typedef void (* PFNGLBINDRENDERBUFFEREXTPROC)(GLenum target, GLuint renderbuffer);
extern PFNGLBINDRENDERBUFFEREXTPROC glad_glBindRenderbufferEXT;
extern PFNGLBINDRENDERBUFFERPROC glad_glBindRenderbuffer;

typedef void (* PFNGLDELETERENDERBUFFERSEXTPROC)(GLsizei n, const GLuint* renderbuffers);
extern PFNGLDELETERENDERBUFFERSEXTPROC glad_glDeleteRenderbuffersEXT;
extern PFNGLDELETERENDERBUFFERSEXTPROC glad_glDeleteRenderbuffers;


typedef void (* PFNGLGENRENDERBUFFERSEXTPROC)(GLsizei n, GLuint* renderbuffers);
extern PFNGLGENRENDERBUFFERSEXTPROC glad_glGenRenderbuffersEXT;
extern PFNGLGENRENDERBUFFERSEXTPROC glad_glGenRenderbuffers;

typedef void (* PFNGLRENDERBUFFERSTORAGEEXTPROC)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
extern PFNGLRENDERBUFFERSTORAGEEXTPROC glad_glRenderbufferStorageEXT;
extern PFNGLRENDERBUFFERSTORAGEEXTPROC glad_glRenderbufferStorage;

typedef void (* PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC)(GLenum target, GLenum pname, GLint* params);
extern PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC glad_glGetRenderbufferParameterivEXT;
extern PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC glad_glGetRenderbufferParameteriv;


typedef void (* PFNGLBINDFRAMEBUFFEREXTPROC)(GLenum target, GLuint framebuffer);
extern PFNGLBINDFRAMEBUFFEREXTPROC glad_glBindFramebufferEXT;
extern PFNGLBINDFRAMEBUFFEREXTPROC glad_glBindFramebuffer;


typedef GLboolean (* PFNGLISFRAMEBUFFEREXTPROC)(GLuint framebuffer);
extern PFNGLISFRAMEBUFFEREXTPROC glad_glIsFramebufferEXT;
extern PFNGLISFRAMEBUFFEREXTPROC glad_glIsFramebuffer;


typedef void (* PFNGLDELETEFRAMEBUFFERSEXTPROC)(GLsizei n, const GLuint* framebuffers);
extern PFNGLDELETEFRAMEBUFFERSEXTPROC glad_glDeleteFramebuffersEXT;
extern PFNGLDELETEFRAMEBUFFERSEXTPROC glad_glDeleteFramebuffers;


typedef void (* PFNGLGENFRAMEBUFFERSEXTPROC)(GLsizei n, GLuint* framebuffers);
extern PFNGLGENFRAMEBUFFERSEXTPROC glad_glGenFramebuffersEXT;
extern PFNGLGENFRAMEBUFFERSEXTPROC glad_glGenFramebuffers;


typedef GLenum (* PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC)(GLenum target);
extern PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC glad_glCheckFramebufferStatusEXT;
extern PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC glad_glCheckFramebufferStatus;


typedef void (* PFNGLFRAMEBUFFERTEXTURE1DEXTPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
extern PFNGLFRAMEBUFFERTEXTURE1DEXTPROC glad_glFramebufferTexture1DEXT;
extern PFNGLFRAMEBUFFERTEXTURE1DEXTPROC glad_glFramebufferTexture1D;


typedef void (* PFNGLFRAMEBUFFERTEXTURE2DEXTPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
extern PFNGLFRAMEBUFFERTEXTURE2DEXTPROC glad_glFramebufferTexture2DEXT;
extern PFNGLFRAMEBUFFERTEXTURE2DEXTPROC glad_glFramebufferTexture2D;


typedef void (* PFNGLFRAMEBUFFERTEXTURE3DEXTPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset);
extern PFNGLFRAMEBUFFERTEXTURE3DEXTPROC glad_glFramebufferTexture3DEXT;
extern PFNGLFRAMEBUFFERTEXTURE3DEXTPROC glad_glFramebufferTexture3D;


typedef void (* PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
extern PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC glad_glFramebufferRenderbufferEXT;
extern PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC glad_glFramebufferRenderbuffer;


typedef void (* PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC)(GLenum target, GLenum attachment, GLenum pname, GLint* params);
extern PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC glad_glGetFramebufferAttachmentParameterivEXT;
extern PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC glad_glGetFramebufferAttachmentParameteriv;

typedef void (* PFNGLGENERATEMIPMAPEXTPROC)(GLenum target);
extern PFNGLGENERATEMIPMAPEXTPROC glad_glGenerateMipmapEXT;
extern PFNGLGENERATEMIPMAPEXTPROC glad_glGenerateMipmap;

typedef void (* PFNGLBINDVERTEXARRAYAPPLEPROC)(GLuint array);
extern PFNGLBINDVERTEXARRAYAPPLEPROC glad_glBindVertexArrayAPPLE;
extern PFNGLBINDVERTEXARRAYAPPLEPROC glad_glBindVertexArray;

typedef void (* PFNGLDELETEVERTEXARRAYSAPPLEPROC)(GLsizei n, const GLuint* arrays);
extern PFNGLDELETEVERTEXARRAYSAPPLEPROC glad_glDeleteVertexArraysAPPLE;
extern PFNGLDELETEVERTEXARRAYSAPPLEPROC glad_glDeleteVertexArrays;

typedef void (* PFNGLGENVERTEXARRAYSAPPLEPROC)(GLsizei n, GLuint* arrays);
extern PFNGLGENVERTEXARRAYSAPPLEPROC glad_glGenVertexArraysAPPLE;
extern PFNGLGENVERTEXARRAYSAPPLEPROC glad_glGenVertexArrays;

typedef GLboolean (* PFNGLISVERTEXARRAYAPPLEPROC)(GLuint array);
extern PFNGLISVERTEXARRAYAPPLEPROC glad_glIsVertexArrayAPPLE;
extern PFNGLISVERTEXARRAYAPPLEPROC glad_glIsVertexArray;
} // extern "C"

namespace OFX {
bool
ofxsLoadOpenGLOnce()
{
    // Ensure that OpenGL functions loading is thread-safe
    AutoMutex locker(&g_glLoadOnceMutex);

    if (g_glLoaded) {
        // Already loaded, don't do it again
        return true;
    }

    // Reasons for failure might be:
    // - opengl32.dll was not found, or libGL.so was not found or OpenGL.framework was not found
    // - glGetString does not return a valid version
    // Note: It does NOT check that required extensions and functions have actually been found
    bool glLoaded = gladLoadGL();

    g_glLoaded = glLoaded;

    // If only EXT_framebuffer is present and not ARB link functions
    if (glLoaded && GLAD_GL_EXT_framebuffer_object && !GLAD_GL_ARB_framebuffer_object) {
        glad_glIsRenderbuffer = glad_glIsRenderbufferEXT;
        glad_glBindRenderbuffer = glad_glBindRenderbufferEXT;
        glad_glDeleteRenderbuffers = glad_glDeleteRenderbuffersEXT;
        glad_glGenRenderbuffers = glad_glGenRenderbuffersEXT;
        glad_glRenderbufferStorage = glad_glRenderbufferStorageEXT;
        glad_glGetRenderbufferParameteriv = glad_glGetRenderbufferParameterivEXT;
        glad_glBindFramebuffer = glad_glBindFramebufferEXT;
        glad_glIsFramebuffer = glad_glIsFramebufferEXT;
        glad_glDeleteFramebuffers = glad_glDeleteFramebuffersEXT;
        glad_glGenFramebuffers = glad_glGenFramebuffersEXT;
        glad_glCheckFramebufferStatus = glad_glCheckFramebufferStatusEXT;
        glad_glFramebufferTexture1D = glad_glFramebufferTexture1DEXT;
        glad_glFramebufferTexture2D = glad_glFramebufferTexture2DEXT;
        glad_glFramebufferTexture3D = glad_glFramebufferTexture3DEXT;
        glad_glFramebufferRenderbuffer = glad_glFramebufferRenderbufferEXT;
        glad_glGetFramebufferAttachmentParameteriv = glad_glGetFramebufferAttachmentParameterivEXT;
        glad_glGenerateMipmap = glad_glGenerateMipmapEXT;
    }

    if (glLoaded && GLAD_GL_APPLE_vertex_array_object && !GLAD_GL_ARB_vertex_buffer_object) {
        glad_glBindVertexArray = glad_glBindVertexArrayAPPLE;
        glad_glDeleteVertexArrays = glad_glDeleteVertexArraysAPPLE;
        glad_glGenVertexArrays = glad_glGenVertexArraysAPPLE;
        glad_glIsVertexArray = glad_glIsVertexArrayAPPLE;
    }

    return g_glLoaded;
} // ofxsLoadGLOnce

int
getOpenGLMajorVersion()
{
    return GLVersion.major;
}

int
getOpenGLMinorVersion()
{
    return GLVersion.minor;
}

bool
getOpenGLSupportsTextureFloat()
{
    return GLAD_GL_ARB_texture_float;
}

bool
getOpenGLSupportFramebuffer()
{
    return GLAD_GL_ARB_framebuffer_object || GLAD_GL_EXT_framebuffer_object;
}

bool
getOpenGLSupportPixelbuffer()
{
    return GLAD_GL_ARB_pixel_buffer_object;
}

bool
getOpenGLSupportVertexArray()
{
    return GLAD_GL_ARB_vertex_array_object || GLAD_GL_APPLE_vertex_array_object;
}
} // namespace OFX
