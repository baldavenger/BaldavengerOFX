#ifndef _ofxParamExt_h_
#define _ofxParamExt_h_

#include <cstdarg>

#include "ofxCore.h"
#include "ofxProperty.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief String to identify a param as a Single string-valued, 'one-of-many' parameter */
#define kOfxParamTypeStrChoice "OfxParamTypeStrChoice"

/** @brief Set a enumeration string in a choice parameter.

    - Type - UTF8 C string X N
    - Property Set - plugin parameter descriptor (read/write) and instance (read/write),
    - Default - the property is empty with no options set.

This property contains the set of enumeration strings corresponding to the options that will be presented to a user from a choice parameter. See @ref ParametersChoice for more details..
*/
#define kOfxParamPropChoiceEnum "OfxParamPropChoiceEnum"

/** @brief Indicates if the host supports animation of string choice params.

    - Type - int X 1
    - Property Set - host descriptor (read only)
    - Valid Values - 0 or 1
*/
#define kOfxParamHostPropSupportsStrChoiceAnimation "OfxParamHostPropSupportsStrChoiceAnimation"

#ifdef __cplusplus
}
#endif

#endif
