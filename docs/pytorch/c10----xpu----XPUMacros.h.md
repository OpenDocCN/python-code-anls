# `.\pytorch\c10\xpu\XPUMacros.h`

```py
#pragma once

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include <c10/xpu/impl/xpu_cmake_macros.h>
#endif

// 以下宏定义用于控制符号的导出和导入，具体功能请参考 c10/macros/Export.h
// 每个构建的库都需要使用一组这样的宏定义。

#ifdef _WIN32
// 如果是在 Windows 平台，并且定义了 C10_XPU_BUILD_SHARED_LIBS 宏
#if defined(C10_XPU_BUILD_SHARED_LIBS)
// 则将 C10_XPU_EXPORT 定义为 __declspec(dllexport)
#define C10_XPU_EXPORT __declspec(dllexport)
// 将 C10_XPU_IMPORT 定义为 __declspec(dllimport)
#define C10_XPU_IMPORT __declspec(dllimport)
#else
// 否则，将 C10_XPU_EXPORT 和 C10_XPU_IMPORT 都定义为空
#define C10_XPU_EXPORT
#define C10_XPU_IMPORT
#endif
#else // _WIN32
// 如果不是在 Windows 平台
#if defined(__GNUC__)
// 并且定义了 __GNUC__ 宏，则使用 __attribute__((__visibility__("default"))) 将 C10_XPU_EXPORT 定义为默认可见性
#define C10_XPU_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
// 否则，将 C10_XPU_EXPORT 定义为空
#define C10_XPU_EXPORT
#endif // defined(__GNUC__)
// 将 C10_XPU_IMPORT 定义为 C10_XPU_EXPORT，保持一致性
#define C10_XPU_IMPORT C10_XPU_EXPORT
#endif // _WIN32

// 下面的宏根据 C10_XPU_BUILD_MAIN_LIB 的定义来确定 C10_XPU_API 的定义
#ifdef C10_XPU_BUILD_MAIN_LIB
// 如果定义了 C10_XPU_BUILD_MAIN_LIB 宏，则将 C10_XPU_API 定义为 C10_XPU_EXPORT
#define C10_XPU_API C10_XPU_EXPORT
#else
// 否则，将 C10_XPU_API 定义为 C10_XPU_IMPORT
#define C10_XPU_API C10_XPU_IMPORT
#endif
```