# `.\pytorch\c10\util\win32-headers.h`

```py
#pragma once
// 如果未定义 WIN32_LEAN_AND_MEAN 宏，则定义它，用于减少 windows.h 的包含内容
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// 如果未定义 NOMINMAX 宏，则定义它，用于避免 Windows 头文件中的 min 和 max 宏定义
#ifndef NOMINMAX
#define NOMINMAX
#endif
// 如果未定义 NOKERNEL 宏，则定义它，用于排除 kernel 相关的头文件和函数
#ifndef NOKERNEL
#define NOKERNEL
#endif
// 如果未定义 NOUSER 宏，则定义它，用于排除 user 相关的头文件和函数
#ifndef NOUSER
#define NOUSER
#endif
// 如果未定义 NOSERVICE 宏，则定义它，用于排除 service 相关的头文件和函数
#ifndef NOSERVICE
#define NOSERVICE
#endif
// 如果未定义 NOSOUND 宏，则定义它，用于排除 sound 相关的头文件和函数
#ifndef NOSOUND
#define NOSOUND
#endif
// 如果未定义 NOMCX 宏，则定义它，用于排除 modem communications 相关的头文件和函数
#ifndef NOMCX
#define NOMCX
#endif
// 如果未定义 NOGDI 宏，则定义它，用于排除 GDI 相关的头文件和函数
#ifndef NOGDI
#define NOGDI
#endif
// 如果未定义 NOMSG 宏，则定义它，用于排除 message 相关的头文件和函数
#ifndef NOMSG
#define NOMSG
#endif
// 如果未定义 NOMB 宏，则定义它，用于排除 MB 相关的定义
#ifndef NOMB
#define NOMB
#endif
// 如果未定义 NOCLIPBOARD 宏，则定义它，用于排除 clipboard 相关的头文件和函数
#ifndef NOCLIPBOARD
#define NOCLIPBOARD
#endif

// dbghelp 需要包含 windows.h。
// clang-format off
#include <windows.h>    // 包含 Windows API 头文件
#include <dbghelp.h>    // 包含调试帮助 API 头文件
// clang-format on

// 清除可能与宏定义重名的定义，以避免冲突
#undef VOID
#undef DELETE
#undef IN
#undef THIS
#undef CONST
#undef NAN
#undef UNKNOWN
#undef NONE
#undef ANY
#undef IGNORE
#undef STRICT
#undef GetObject
#undef CreateSemaphore
#undef Yield
#undef RotateRight32
#undef RotateLeft32
#undef RotateRight64
#undef RotateLeft64
```