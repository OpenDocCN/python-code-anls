# `.\pytorch\c10\test\util\Macros.h`

```py
#ifndef C10_TEST_CORE_MACROS_MACROS_H_
// 如果未定义 C10_TEST_CORE_MACROS_MACROS_H_，则执行以下内容

#ifdef _WIN32
// 如果定义了 _WIN32，表示当前编译环境为 Windows
#define DISABLED_ON_WINDOWS(x) DISABLED_##x
// 定义一个宏 DISABLED_ON_WINDOWS(x)，如果在 Windows 环境下，则替换为 DISABLED_x
#else
// 如果不是在 Windows 环境下
#define DISABLED_ON_WINDOWS(x) x
// 定义一个宏 DISABLED_ON_WINDOWS(x)，直接返回 x，即在非 Windows 环境下不做替换
#endif

#endif // C10_MACROS_MACROS_H_
// 结束条件编译指令，标记 C10_TEST_CORE_MACROS_MACROS_H_ 的结束
```