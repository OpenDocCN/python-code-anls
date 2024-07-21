# `.\pytorch\c10\core\alignment.h`

```
#pragma once
// 声明此头文件只需包含一次

#include <cstddef>
// 包含标准库中的 cstddef 头文件，提供 size_t 等类型定义

namespace c10 {

#ifdef C10_MOBILE
// 如果定义了 C10_MOBILE 宏，则在移动平台上使用 16 字节对齐
// - 适用于 ARM NEON AArch32 和 AArch64 架构
// - 以及 x86[-64] 但不支持 AVX 指令集的平台
constexpr size_t gAlignment = 16;
#else
// 否则使用 64 字节对齐，适用于支持 AVX512 指令集的平台
constexpr size_t gAlignment = 64;
#endif

constexpr size_t gPagesize = 4096;
// 定义页面大小为 4096 字节（即 4KB）

// 由于默认的 thp 页面大小为 2MB，仅对大小为 2MB 或更大的缓冲区启用 thp，
// 以避免内存膨胀
constexpr size_t gAlloc_threshold_thp = static_cast<size_t>(2) * 1024 * 1024;
// 阈值为 2MB，以字节为单位，用于启用 thp（Transparent Huge Pages）
} // namespace c10
```