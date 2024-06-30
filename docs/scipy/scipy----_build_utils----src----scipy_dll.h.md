# `D:\src\scipysrc\scipy\scipy\_build_utils\src\scipy_dll.h`

```
#pragma once

// SCIPY_DLL
// 从 https://github.com/abseil/abseil-cpp/blob/20240116.2/absl/base/config.h#L736-L753 得到灵感
//
// 如果构建 sf_error_state 作为 DLL，此宏扩展为 `__declspec(dllexport)`，
// 以便我们可以适当地注释导出的符号。当在消费 DLL 的头文件中使用时，此宏扩展为
// `__declspec(dllimport)`，使消费者知道该符号在 DLL 内定义。在所有其他情况下，
// 此宏扩展为空。
// 注意：SCIPY_DLL_{EX,IM}PORTS 在 scipy/special/meson.build 中设置
#if defined(SCIPY_DLL_EXPORTS)
    #define SCIPY_DLL __declspec(dllexport)
#elif defined(SCIPY_DLL_IMPORTS)
    #define SCIPY_DLL __declspec(dllimport)
#else
    #define SCIPY_DLL
#endif
```