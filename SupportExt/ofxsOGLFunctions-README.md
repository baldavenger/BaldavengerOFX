Here are the rules to use OpenGL in your plug-in:

- If your plug-in is going to use OSMesa to handle CPU rendering, you need to ensure you are not going to call regular OpenGL functions but the one provided by mesa. In that case, it's very simple, just:
        
    #include <ofxsGLFunctions.h>

and then in the code prefix every gl function call by either **GL_CPU** to use OSMesa or **GL_GPU** to use regular OpenGL, e.g:

    GL_CPU::glBegin(GL_POINTS);
    GL_CPU::glVertex2d(50., 50.);
    GL_CPU::glEnd();

If you have code that will be executed both in OSMesa code path and OpenGL code path, you can template your functions so that the same code is used:

    template <typename GL>
    void myCrossPlatformPoint()
    {
        GL::glBegin(GL_POINTS);
        GL::glVertex2d(50., 50.);
        GL::glEnd();
    }

Functions used by GL_CPU will be available only if HAVE_OSMESA is defined otherwise they will point to NULL.

- If your plug-in is just going to use regular OpenGL for it's rendering or for interacts and does not need OSMesa at all, you may use directly

    #include <glad.h>

- Either way, you must include 

    #include <ofxsOGLUtilities.h>

and you must initialize the OpenGL functions at least once using the function OFX::ofxsLoadOpenGLOnce(). To do so, an OpenGL context must be bound by the host application, so you can only do that in 2 different places:

    * For overlay interacts, you can do it in the draw action
    * For OpenGL render plug-ins, you can do it in the contextAttached function

The function *ofxsLoadOpenGLOnce()* is thread-safe and will do the work only once.


Note that in that case you don't need to prefix your gl calls by **GL_GPU** you will directly use the functions loaded by GLAD. Using the **GL_GPU** prefix would just require 1 function pointer dereference which is slower than calling the function directly.


Warning: It is very important that you DO NOT mix environments with glad.h functions or ofxsGLFunctions.h or directly by including gl.h. Make sure that only one of them is included in your code.

