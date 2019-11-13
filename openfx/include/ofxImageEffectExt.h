#ifndef _ofxImageEffectExt_h_
#define _ofxImageEffectExt_h_

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Says whether the clip is for thumbnail.
   - Type - int X 1
   - Property Set - clip instance (read only)
   - Valid Values - This must be one of 0 or 1
 */
#define kOfxImageClipPropThumbnail "kOfxImageClipPropThumbnail"

/** @brief Indicates which Resolve Page we are currently at
   - Type - string X 1
   - Property Set - inArgs property set of the kOfxActionCreateInstance action
   - Default - "Color"
   - Valid Values - This must be "Edit", "Color" or "Fusion"
 */
#define kOfxImageEffectPropResolvePage "OfxImageEffectPropResolvePage"

/** @brief Indicates whether a host or plugin can support Cuda render

    - Type - string X 1
    - Property Set - plugin descriptor (read/write), host descriptor (read only)
    - Default - "false"
    - Valid Values - This must be one of
      - "false"  - in which case the host or plugin does not support Cuda render
      - "true"   - which means a host or plugin can support Cuda render,
                   in the case of plug-ins this also means that it is
                   capable of CPU based rendering in the absence of a GPU
 */
#define kOfxImageEffectPropCudaRenderSupported "OfxImageEffectPropCudaRenderSupported"

/** @brief Indicates that an image effect SHOULD use Cuda render in
the current action

   When a plugin and host have established they can both use Cuda renders
   then when this property has been set, the host expects the plugin to render
   its result into the buffer it has setup before calling the render. The plugin
   should also handle the situation if the plugin is running on the same device
   as the host.

   - Type - int X 1
   - Property Set - inArgs property set of the kOfxImageEffectActionRender action
   - Valid Values
      - 0 indicates that the effect should assume that the buffers reside on
          the CPU.
      - 1 indicates that the effect should assume that the buffers reside on
          the device.

\note Once this property is set, the host and plug-in have agreed to
use Cuda render, so the effect SHOULD access all its images directly
using the buffer pointers.

*/
#define kOfxImageEffectPropCudaEnabled "OfxImageEffectPropCudaEnabled"

/** @brief Indicates whether a host or plugin can support Cuda render

    - Type - string X 1
    - Property Set - plugin descriptor (read/write), host descriptor (read only)
    - Default - "false"
    - Valid Values - This must be one of
      - "false"  - in which case the host or plugin does not support Cuda render
      - "true"   - which means a host or plugin can support Cuda render,
                   in the case of plug-ins this also means that it is
                   capable of CPU based rendering in the absence of a GPU
 */

#define kOfxImageEffectPropMetalRenderSupported "OfxImageEffectPropMetalRenderSupported"

/** @brief Indicates that an image effect SHOULD use Metal render in
the current action

   When a plugin and host have established they can both use Metal renders
   then when this property has been set, the host expects the plugin to render
   its result into the buffer it has setup before calling the render. The plugin
   should also handle the situation if the plugin is running on the same device
   as the host.

   - Type - int X 1
   - Property Set - inArgs property set of the kOfxImageEffectActionRender action
   - Valid Values
      - 0 indicates that the effect should assume that the buffers reside on
          the CPU.
      - 1 indicates that the effect should assume that the buffers reside on
          the device.

\note Once this property is set, the host and plug-in have agreed to
use Metal render, so the effect SHOULD access all its images directly
using the buffer pointers.

*/
#define kOfxImageEffectPropMetalEnabled "OfxImageEffectPropMetalEnabled"

/** @brief Indicates whether a host or plugin can support Metal render

    - Type - string X 1
    - Property Set - plugin descriptor (read/write), host descriptor (read only)
    - Default - "false"
    - Valid Values - This must be one of
      - "false"  - in which case the host or plugin does not support Metal render
      - "true"   - which means a host or plugin can support Metal render,
                   in the case of plug-ins this also means that it is
                   capable of CPU based rendering in the absence of a GPU
 */
 
 #define kOfxImageEffectPropMetalCommandQueue "OfxImageEffectPropMetalCommandQueue"

/** @brief Indicates whether a host or plugin can support Metal render

    - Type - string X 1
    - Property Set - plugin descriptor (read/write), host descriptor (read only)
    - Default - "false"
    - Valid Values - This must be one of
      - "false"  - in which case the host or plugin does not support Metal render
      - "true"   - which means a host or plugin can support Metal render,
                   in the case of plug-ins this also means that it is
                   capable of CPU based rendering in the absence of a GPU
 */

#define kOfxImageEffectPropOpenCLRenderSupported "OfxImageEffectPropOpenCLRenderSupported"

/** @brief Indicates that an image effect SHOULD use OpenCL render in
the current action

   When a plugin and host have established they can both use OpenCL renders
   then when this property has been set, the host expects the plugin to render
   its result into the buffer it has setup before calling the render. The plugin
   should also handle the situation if the plugin is running on the same device
   as the host.

   - Type - int X 1
   - Property Set - inArgs property set of the kOfxImageEffectActionRender action
   - Valid Values
      - 0 indicates that the effect should assume that the buffers reside on
          the CPU.
      - 1 indicates that the effect should assume that the buffers reside on
          the device.

\note Once this property is set, the host and plug-in have agreed to
use OpenCL render, so the effect SHOULD access all its images directly
using the buffer pointers.

*/
#define kOfxImageEffectPropOpenCLEnabled "OfxImageEffectPropOpenCLEnabled"

/**  @brief The command queue of OpenCL render

    - Type - pointer X 1
    - Property Set - plugin descriptor (read only), host descriptor (read/write)

This property contains a pointer to the command queue of OpenCL render (cl_command_queue type).
In order to use it, reinterpret_cast<cl_command_queue>(pointer) is needed.

*/
#define kOfxImageEffectPropOpenCLCommandQueue "OfxImageEffectPropOpenCLCommandQueue"

/** @brief Indicates a plugin output does not depend on location or neighbours of a given pixel.
If the plugin is with no spatial awareness, it will be executed during LUT generation. Otherwise,
it will be bypassed during LUT generation.

    - Type - string X 1
    - Property Set - plugin descriptor (read/write)
    - Default - "false"
    - Valid Values - This must be one of
      - "false"  - the plugin is with spatial awareness, it will be bypassed during LUT generation
      - "true"   - the plugin is with no spatial awareness, it will be executed during LUT generation
 */
#define kOfxImageEffectPropNoSpatialAwareness "OfxImageEffectPropNoSpatialAwareness"

#ifdef __cplusplus
}
#endif

#endif
