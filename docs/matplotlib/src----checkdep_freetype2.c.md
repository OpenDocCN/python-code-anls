# `D:\src\scipysrc\matplotlib\src\checkdep_freetype2.c`

```py
#ifdef __has_include
  #if !__has_include(<ft2build.h>)
    #error "FreeType version 2.3 or higher is required. \
You may set the system-freetype Meson build option to false to let Matplotlib download it."
  #endif
#endif

#include <ft2build.h>
#include FT_FREETYPE_H

#define XSTR(x) STR(x)
#define STR(x) #x

#pragma message("Compiling with FreeType version " \
  XSTR(FREETYPE_MAJOR) "." XSTR(FREETYPE_MINOR) "." XSTR(FREETYPE_PATCH) ".")
#if FREETYPE_MAJOR << 16 + FREETYPE_MINOR << 8 + FREETYPE_PATCH < 0x020300
  #error "FreeType version 2.3 or higher is required. \
You may set the system-freetype Meson build option to false to let Matplotlib download it."
#endif



#ifdef __has_include
  // 检查是否支持__has_include，用于检测是否可以包含特定头文件
  #if !__has_include(<ft2build.h>)
    // 如果不包含<ft2build.h>，则输出错误信息，要求安装 FreeType 版本 2.3 或更高
    #error "FreeType version 2.3 or higher is required. \
You may set the system-freetype Meson build option to false to let Matplotlib download it."
  #endif
#endif

#include <ft2build.h>   // 包含 FreeType 的构建文件
#include FT_FREETYPE_H   // 包含 FreeType 的头文件

#define XSTR(x) STR(x)   // 定义宏 XSTR，用于将参数转换为字符串
#define STR(x) #x        // 定义宏 STR，用于将参数转换为字符串

#pragma message("Compiling with FreeType version " \
  XSTR(FREETYPE_MAJOR) "." XSTR(FREETYPE_MINOR) "." XSTR(FREETYPE_PATCH) ".")
// 使用#pragma message 输出正在编译的 FreeType 版本信息

#if FREETYPE_MAJOR << 16 + FREETYPE_MINOR << 8 + FREETYPE_PATCH < 0x020300
  // 检查 FreeType 版本是否低于 2.3
  #error "FreeType version 2.3 or higher is required. \
You may set the system-freetype Meson build option to false to let Matplotlib download it."
#endif
// 如果 FreeType 版本低于 2.3，则输出错误信息
```