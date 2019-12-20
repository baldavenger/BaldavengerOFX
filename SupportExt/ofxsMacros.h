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
 * Useful macros.
 */

#ifndef openfx_supportext_ofxsMacros_h
#define openfx_supportext_ofxsMacros_h

/* *INDENT-OFF* */

#define OFXS_NAMESPACE_OFX_ENTER namespace OFX {
#define OFXS_NAMESPACE_OFX_EXIT }

// anonymous namespace macros, mainly to avoid crazy intentation by some IDEs (e.g. Xcode)
#define OFXS_NAMESPACE_ANONYMOUS_ENTER namespace {
#define OFXS_NAMESPACE_ANONYMOUS_EXIT }

// compiler_warning.h
#define STRINGISE_IMPL(x) # x
#define STRINGISE(x) STRINGISE_IMPL(x)

// Use: #pragma message WARN("My message")
#if _MSC_VER
#   define FILE_LINE_LINK __FILE__ "(" STRINGISE(__LINE__) ") : "
#   define WARN(exp) (FILE_LINE_LINK "WARNING: " exp)
#else //__GNUC__ - may need other defines for different compilers
#   define WARN(exp) ("WARNING: " exp)
#endif

// The following was grabbed from WTF/wtf/Compiler.h (where WTF was replaced by OFXS)

/* COMPILER() - the compiler being used to build the project */
#define COMPILER(OFXS_FEATURE) (defined OFXS_COMPILER_ ## OFXS_FEATURE && OFXS_COMPILER_ ## OFXS_FEATURE)

/* COMPILER_SUPPORTS() - whether the compiler being used to build the project supports the given feature. */
#define COMPILER_SUPPORTS(OFXS_COMPILER_FEATURE) (defined OFXS_COMPILER_SUPPORTS_ ## OFXS_COMPILER_FEATURE && OFXS_COMPILER_SUPPORTS_ ## OFXS_COMPILER_FEATURE)

/* COMPILER_QUIRK() - whether the compiler being used to build the project requires a given quirk. */
#define COMPILER_QUIRK(OFXS_COMPILER_QUIRK) (defined OFXS_COMPILER_QUIRK_ ## OFXS_COMPILER_QUIRK && OFXS_COMPILER_QUIRK_ ## OFXS_COMPILER_QUIRK)

/* ==== COMPILER() - the compiler being used to build the project ==== */

/* COMPILER(CLANG) - Clang */
#if defined(__clang__)
#define OFXS_COMPILER_CLANG 1

#ifndef __has_extension
#define __has_extension __has_feature /* Compatibility with older versions of clang */
#endif

#define CLANG_PRAGMA(PRAGMA) _Pragma(PRAGMA)

/* Specific compiler features */
#define OFXS_COMPILER_SUPPORTS_CXX_VARIADIC_TEMPLATES __has_extension(cxx_variadic_templates)

/* There is a bug in clang that comes with Xcode 4.2 where AtomicStrings can't be implicitly converted to Strings
   in the presence of move constructors and/or move assignment operators. This bug has been fixed in Xcode 4.3 clang, so we
   check for both cxx_rvalue_references as well as the unrelated cxx_nonstatic_member_init feature which we know was added in 4.3 */
#define OFXS_COMPILER_SUPPORTS_CXX_RVALUE_REFERENCES __has_extension(cxx_rvalue_references) && __has_extension(cxx_nonstatic_member_init)

#define OFXS_COMPILER_SUPPORTS_CXX_DELETED_FUNCTIONS __has_extension(cxx_deleted_functions)
#define OFXS_SUPPORTS_CXX_NULLPTR __has_feature(cxx_nullptr)
#define OFXS_COMPILER_SUPPORTS_CXX_EXPLICIT_CONVERSIONS __has_feature(cxx_explicit_conversions)
#define OFXS_COMPILER_SUPPORTS_BLOCKS __has_feature(blocks)
#define OFXS_COMPILER_SUPPORTS_C_STATIC_ASSERT __has_extension(c_static_assert)
#define OFXS_COMPILER_SUPPORTS_CXX_OVERRIDE_CONTROL __has_extension(cxx_override_control)
#define OFXS_COMPILER_SUPPORTS_HAS_TRIVIAL_DESTRUCTOR __has_extension(has_trivial_destructor)

#endif

#ifndef CLANG_PRAGMA
#define CLANG_PRAGMA(PRAGMA)
#endif

/* COMPILER(MSVC) - Microsoft Visual C++ */
/* COMPILER(MSVC7_OR_LOWER) - Microsoft Visual C++ 2003 or lower*/
/* COMPILER(MSVC9_OR_LOWER) - Microsoft Visual C++ 2008 or lower*/
#if defined(_MSC_VER)
#define OFXS_COMPILER_MSVC 1
#if _MSC_VER < 1400
#define OFXS_COMPILER_MSVC7_OR_LOWER 1
#elif _MSC_VER < 1600
#define OFXS_COMPILER_MSVC9_OR_LOWER 1
#endif

/* Specific compiler features */
#if !COMPILER(CLANG) && _MSC_VER >= 1600
#define OFXS_SUPPORTS_CXX_NULLPTR 1
#endif

#if !COMPILER(CLANG)
#define OFXS_COMPILER_SUPPORTS_CXX_OVERRIDE_CONTROL 1
#define OFXS_COMPILER_QUIRK_FINAL_IS_CALLED_SEALED 1
#endif

#endif

/* COMPILER(RVCT) - ARM RealView Compilation Tools */
/* COMPILER(RVCT4_OR_GREATER) - ARM RealView Compilation Tools 4.0 or greater */
#if defined(__CC_ARM) || defined(__ARMCC__)
#define OFXS_COMPILER_RVCT 1
#define RVCT_VERSION_AT_LEAST(major, minor, patch, build) ( __ARMCC_VERSION >= (major * 100000 + minor * 10000 + patch * 1000 + build) )
#else
/* Define this for !RVCT compilers, just so we can write things like RVCT_VERSION_AT_LEAST(3, 0, 0, 0). */
#define RVCT_VERSION_AT_LEAST(major, minor, patch, build) 0
#endif

/* COMPILER(GCCE) - GNU Compiler Collection for Embedded */
#if defined(__GCCE__)
#define OFXS_COMPILER_GCCE 1
#define GCCE_VERSION (__GCCE__ * 10000 + __GCCE_MINOR__ * 100 + __GCCE_PATCHLEVEL__)
#define GCCE_VERSION_AT_LEAST(major, minor, patch) ( GCCE_VERSION >= (major * 10000 + minor * 100 + patch) )
#endif

/* COMPILER(GCC) - GNU Compiler Collection */
/* --gnu option of the RVCT compiler also defines __GNUC__ */
#if defined(__GNUC__) && !COMPILER(RVCT)
#define OFXS_COMPILER_GCC 1
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#define GCC_VERSION_AT_LEAST(major, minor, patch) ( GCC_VERSION >= (major * 10000 + minor * 100 + patch) )
#else
/* Define this for !GCC compilers, just so we can write things like GCC_VERSION_AT_LEAST(4, 1, 0). */
#define GCC_VERSION_AT_LEAST(major, minor, patch) 0
#endif

/* Specific compiler features */
#if COMPILER(GCC) && !COMPILER(CLANG)
#if GCC_VERSION_AT_LEAST(4, 7, 0) && defined(__cplusplus) && __cplusplus >= 201103L
#define OFXS_COMPILER_SUPPORTS_CXX_RVALUE_REFERENCES 1
#define OFXS_COMPILER_SUPPORTS_CXX_DELETED_FUNCTIONS 1
#define OFXS_SUPPORTS_CXX_NULLPTR 1
#define OFXS_COMPILER_SUPPORTS_CXX_OVERRIDE_CONTROL 1
#define OFXS_COMPILER_QUIRK_GCC11_GLOBAL_ISINF_ISNAN 1

#elif GCC_VERSION_AT_LEAST(4, 6, 0) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#define OFXS_SUPPORTS_CXX_NULLPTR 1
#define OFXS_COMPILER_QUIRK_GCC11_GLOBAL_ISINF_ISNAN 1
#endif

#endif

/* COMPILER(MINGW) - MinGW GCC */
/* COMPILER(MINGW64) - mingw-w64 GCC - only used as additional check to exclude mingw.org specific functions */
#if defined(__MINGW32__)
#define OFXS_COMPILER_MINGW 1
#include <_mingw.h> /* private MinGW header */
#if defined(__MINGW64_VERSION_MAJOR) /* best way to check for mingw-w64 vs mingw.org */
#define OFXS_COMPILER_MINGW64 1
#endif /* __MINGW64_VERSION_MAJOR */
#endif /* __MINGW32__ */

/* COMPILER(INTEL) - Intel C++ Compiler */
#if defined(__INTEL_COMPILER)
#define OFXS_COMPILER_INTEL 1
#endif

/* COMPILER(SUNCC) */
#if defined(__SUNPRO_CC) || defined(__SUNPRO_C)
#define OFXS_COMPILER_SUNCC 1
#endif

/* ==== Compiler features ==== */


/* ALWAYS_INLINE */

#ifndef ALWAYS_INLINE
#if COMPILER(GCC) && defined(NDEBUG) && !COMPILER(MINGW)
#define ALWAYS_INLINE inline __attribute__( (__always_inline__) )
#elif (COMPILER(MSVC) || COMPILER(RVCT ) ) && defined(NDEBUG)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE inline
#endif
#endif


/* NEVER_INLINE */

#ifndef NEVER_INLINE
#if COMPILER(GCC)
#define NEVER_INLINE __attribute__( (__noinline__) )
#elif COMPILER(RVCT)
#define NEVER_INLINE __declspec(noinline)
#else
#define NEVER_INLINE
#endif
#endif


/* UNLIKELY */

#ifndef UNLIKELY
#if COMPILER(GCC) || (RVCT_VERSION_AT_LEAST(3, 0, 0, 0) && defined(__GNUC__ ) )
#define UNLIKELY(x) __builtin_expect( (x), 0 )
#else
#define UNLIKELY(x) (x)
#endif
#endif


/* LIKELY */

#ifndef LIKELY
#if COMPILER(GCC) || (RVCT_VERSION_AT_LEAST(3, 0, 0, 0) && defined(__GNUC__ ) )
#define LIKELY(x) __builtin_expect( (x), 1 )
#else
#define LIKELY(x) (x)
#endif
#endif


/* NO_RETURN */


#ifndef NO_RETURN
#if COMPILER(GCC)
#define NO_RETURN __attribute( (__noreturn__) )
#elif COMPILER(MSVC) || COMPILER(RVCT)
#define NO_RETURN __declspec(noreturn)
#else
#define NO_RETURN
#endif
#endif


/* NO_RETURN_WITH_VALUE */

#ifndef NO_RETURN_WITH_VALUE
#if !COMPILER(MSVC)
#define NO_RETURN_WITH_VALUE NO_RETURN
#else
#define NO_RETURN_WITH_VALUE
#endif
#endif


/* WARN_UNUSED_RETURN */

#if COMPILER(GCC)
#define WARN_UNUSED_RETURN __attribute__ ( (warn_unused_result) )
#else
#define WARN_UNUSED_RETURN
#endif

/* OVERRIDE and FINAL */

#if COMPILER_SUPPORTS(CXX_OVERRIDE_CONTROL) &&  !COMPILER(MSVC) //< patch so msvc 2010 ignores the override and final keywords.
#define OVERRIDE override

#if COMPILER_QUIRK(FINAL_IS_CALLED_SEALED)
#define FINAL sealed
#else
#define FINAL final
#endif

#else
#define OVERRIDE
#define FINAL
#endif

/* REFERENCED_FROM_ASM */

#ifndef REFERENCED_FROM_ASM
#if COMPILER(GCC)
#define REFERENCED_FROM_ASM __attribute__( (used) )
#else
#define REFERENCED_FROM_ASM
#endif
#endif

/* OBJC_CLASS */

#ifndef OBJC_CLASS
#ifdef __OBJC__
#define OBJC_CLASS @class
#else
#define OBJC_CLASS class
#endif
#endif

/* ABI */
#if defined(__ARM_EABI__) || defined(__EABI__)
#define OFXS_COMPILER_SUPPORTS_EABI 1
#endif

// Warning control from https://svn.boost.org/trac/boost/wiki/Guidelines/WarningsGuidelines
#if ( ( __GNUC__ * 100) + __GNUC_MINOR__) >= 402
#define GCC_DIAG_STR(s) # s
#define GCC_DIAG_JOINSTR(x,y) GCC_DIAG_STR(x ## y)
# define GCC_DIAG_DO_PRAGMA(x) _Pragma ( # x)
# define GCC_DIAG_PRAGMA(x) GCC_DIAG_DO_PRAGMA(GCC diagnostic x)
# if ( ( __GNUC__ * 100) + __GNUC_MINOR__) >= 406
#  define GCC_DIAG_OFF(x) GCC_DIAG_PRAGMA(push) \
    GCC_DIAG_PRAGMA( ignored GCC_DIAG_JOINSTR(-W,x) )
#  define GCC_DIAG_ON(x) GCC_DIAG_PRAGMA(pop)
# else
#  define GCC_DIAG_OFF(x) GCC_DIAG_PRAGMA( ignored GCC_DIAG_JOINSTR(-W,x) )
#  define GCC_DIAG_ON(x)  GCC_DIAG_PRAGMA( warning GCC_DIAG_JOINSTR(-W,x) )
# endif
#else
# define GCC_DIAG_OFF(x)
# define GCC_DIAG_ON(x)
#endif

#ifdef __clang__
#  define CLANG_DIAG_STR(s) # s
// stringize s to "no-unused-variable"
#  define CLANG_DIAG_JOINSTR(x,y) CLANG_DIAG_STR(x ## y)
//  join -W with no-unused-variable to "-Wno-unused-variable"
#  define CLANG_DIAG_DO_PRAGMA(x) _Pragma ( # x)
// _Pragma is unary operator  #pragma ("")
#  define CLANG_DIAG_PRAGMA(x) CLANG_DIAG_DO_PRAGMA(clang diagnostic x)
#    define CLANG_DIAG_OFF(x) CLANG_DIAG_PRAGMA(push) \
    CLANG_DIAG_PRAGMA( ignored CLANG_DIAG_JOINSTR(-W,x) )
// For example: #pragma clang diagnostic ignored "-Wno-unused-variable"
#   define CLANG_DIAG_ON(x) CLANG_DIAG_PRAGMA(pop)
// For example: #pragma clang diagnostic warning "-Wno-unused-variable"
#else // Ensure these macros so nothing for other compilers.
#  define CLANG_DIAG_OFF(x)
#  define CLANG_DIAG_ON(x)
#  define CLANG_DIAG_PRAGMA(x)
#endif

/* Usage:
   CLANG_DIAG_OFF(unused-variable)
   CLANG_DIAG_OFF(unused-parameter)
   CLANG_DIAG_OFF(uninitialized)
 */

#if COMPILER_SUPPORTS(CXX_OVERRIDE_CONTROL)
// we want to use override & final, and get no warnings even if not compiling in c++11 mode
CLANG_DIAG_OFF(c++11-extensions)
GCC_DIAG_OFF(c++11-extensions)
#endif

/* *INDENT-ON* */

#endif // ifndef openfx_supportext_ofxsMacros_h
