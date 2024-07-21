# `.\pytorch\c10\macros\Export.h`

```py
#ifndef C10_MACROS_EXPORT_H_
#define C10_MACROS_EXPORT_H_

/* Header file to define the common scaffolding for exported symbols.
 *
 * Export is by itself a quite tricky situation to deal with, and if you are
 * hitting this file, make sure you start with the background here:
 * - Linux: https://gcc.gnu.org/wiki/Visibility
 * - Windows:
 * https://docs.microsoft.com/en-us/cpp/cpp/dllexport-dllimport?view=vs-2017
 *
 * Do NOT include this file directly. Instead, use c10/macros/Macros.h
 */

// You do not need to edit this part of file unless you are changing the core
// pytorch export abstractions.
//
// This part defines the C10 core export and import macros. This is controlled
// by whether we are building shared libraries or not, which is determined
// during build time and codified in c10/core/cmake_macros.h.
// When the library is built as a shared lib, EXPORT and IMPORT will contain
// visibility attributes. If it is being built as a static lib, then EXPORT
// and IMPORT basically have no effect.

// As a rule of thumb, you should almost NEVER mix static and shared builds for
// libraries that depend on c10. AKA, if c10 is built as a static library, we
// recommend everything dependent on c10 to be built statically. If c10 is built
// as a shared library, everything dependent on it should be built as shared. In
// the PyTorch project, all native libraries shall use the macro
// C10_BUILD_SHARED_LIB to check whether pytorch is building shared or static
// libraries.

// For build systems that do not directly depend on CMake and directly build
// from the source directory (such as Buck), one may not have a cmake_macros.h
// file at all. In this case, the build system is responsible for providing
// correct macro definitions corresponding to the cmake_macros.h.in file.
//
// In such scenarios, one should define the macro
//     C10_USING_CUSTOM_GENERATED_MACROS
// to inform this header that it does not need to include the cmake_macros.h
// file.

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include <c10/macros/cmake_macros.h>
#endif // C10_USING_CUSTOM_GENERATED_MACROS

#ifdef _WIN32
#define C10_HIDDEN
#if defined(C10_BUILD_SHARED_LIBS)
// If building on Windows and shared libraries are enabled, define C10_EXPORT
// as dllexport and C10_IMPORT as dllimport
#define C10_EXPORT __declspec(dllexport)
#define C10_IMPORT __declspec(dllimport)
#else
// If building on Windows but not as shared library, C10_EXPORT and C10_IMPORT
// have no effect
#define C10_EXPORT
#define C10_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
// If building on non-Windows platform with GCC, define C10_EXPORT with default
// visibility attribute and C10_HIDDEN as hidden visibility attribute
#define C10_EXPORT __attribute__((__visibility__("default")))
#define C10_HIDDEN __attribute__((__visibility__("hidden")))
#else // defined(__GNUC__)
// For non-Windows platforms without GCC, C10_EXPORT and C10_HIDDEN have no effect
#define C10_EXPORT
#define C10_HIDDEN
#endif // defined(__GNUC__)
// C10_IMPORT is defined as C10_EXPORT for non-Windows platforms
#define C10_IMPORT C10_EXPORT
#endif // _WIN32

#ifdef NO_EXPORT
// If NO_EXPORT is defined, undefine C10_EXPORT
#undef C10_EXPORT
#define C10_EXPORT
#endif

// Definition of an adaptive XX_API macro, that depends on whether you are
// building the library itself or not, routes to XX_EXPORT and XX_IMPORT.
// Basically, you will need to do this for each shared library that you are
// building, and the instruction is as follows: assuming that you are building

#endif // C10_MACROS_EXPORT_H_
// Define the C10_API macro based on whether C10_BUILD_MAIN_LIB is defined.
// If C10_BUILD_MAIN_LIB is defined, C10_API is set to C10_EXPORT; otherwise, it is set to C10_IMPORT.
#define C10_API C10_BUILD_MAIN_LIB ? C10_EXPORT : C10_IMPORT

// Define the TORCH_API macro based on whether CAFFE2_BUILD_MAIN_LIB is defined.
// If CAFFE2_BUILD_MAIN_LIB is defined, TORCH_API is set to C10_EXPORT; otherwise, it is set to C10_IMPORT.
#define TORCH_API CAFFE2_BUILD_MAIN_LIB ? C10_EXPORT : C10_IMPORT

// Explanation about splitting torch_cuda into two libraries due to issues with linking big binaries in CUDA 11.1.
// Define TORCH_CUDA_CU_API based on whether TORCH_CUDA_CU_BUILD_MAIN_LIB is defined or BUILD_SPLIT_CUDA is defined.
// If TORCH_CUDA_CU_BUILD_MAIN_LIB is defined, TORCH_CUDA_CU_API is set to C10_EXPORT.
// If BUILD_SPLIT_CUDA is defined, TORCH_CUDA_CU_API is set to C10_IMPORT.
#define TORCH_CUDA_CU_API TORCH_CUDA_CU_BUILD_MAIN_LIB ? C10_EXPORT : (BUILD_SPLIT_CUDA ? C10_IMPORT : /* undefined */)

// Define TORCH_CUDA_CPP_API similarly based on whether TORCH_CUDA_CPP_BUILD_MAIN_LIB is defined or BUILD_SPLIT_CUDA is defined.
#define TORCH_CUDA_CPP_API TORCH_CUDA_CPP_BUILD_MAIN_LIB ? C10_EXPORT : (BUILD_SPLIT_CUDA ? C10_IMPORT : /* undefined */)

// Define TORCH_CUDA_CPP_API and TORCH_CUDA_CU_API based on whether TORCH_CUDA_BUILD_MAIN_LIB is defined or BUILD_SPLIT_CUDA is not defined.
// If TORCH_CUDA_BUILD_MAIN_LIB is defined, both TORCH_CUDA_CPP_API and TORCH_CUDA_CU_API are set to C10_EXPORT.
// If BUILD_SPLIT_CUDA is not defined, both are set to C10_IMPORT.
#define TORCH_CUDA_CPP_API TORCH_CUDA_BUILD_MAIN_LIB ? C10_EXPORT : (BUILD_SPLIT_CUDA ? C10_IMPORT : /* undefined */)
#define TORCH_CUDA_CU_API TORCH_CUDA_BUILD_MAIN_LIB ? C10_EXPORT : (BUILD_SPLIT_CUDA ? C10_IMPORT : /* undefined */)

// Define TORCH_HIP_API based on whether TORCH_HIP_BUILD_MAIN_LIB is defined.
// If TORCH_HIP_BUILD_MAIN_LIB is defined, TORCH_HIP_API is set to C10_EXPORT; otherwise, it is set to C10_IMPORT.
#define TORCH_HIP_API TORCH_HIP_BUILD_MAIN_LIB ? C10_EXPORT : C10_IMPORT

// Define TORCH_XPU_API based on whether TORCH_XPU_BUILD_MAIN_LIB is defined.
// If TORCH_XPU_BUILD_MAIN_LIB is defined, TORCH_XPU_API is set to C10_EXPORT; otherwise, it is set to C10_IMPORT.
#define TORCH_XPU_API TORCH_XPU_BUILD_MAIN_LIB ? C10_EXPORT : C10_IMPORT

// Define C10_API_ENUM for enums depending on whether _WIN32 and __CUDACC__ are defined.
// If both are defined (indicating Windows and CUDA environment), C10_API_ENUM is set to C10_API.
// Otherwise, it is left empty.
#if defined(_WIN32) && defined(__CUDACC__)
#define C10_API_ENUM C10_API
#else
#define C10_API_ENUM
#endif

// End of the header guard and comments explaining the purpose of the macros.
#endif // C10_MACROS_MACROS_H_
```