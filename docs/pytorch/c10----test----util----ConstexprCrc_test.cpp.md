# `.\pytorch\c10\test\util\ConstexprCrc_test.cpp`

```
#include <c10/util/ConstexprCrc.h>

// 使用 c10::util 命名空间中的 crc64 和 crc64_t 类型
using c10::util::crc64;
using c10::util::crc64_t;

// 对通用的测试进行静态断言
static_assert(
    // 检查相同字符串的 CRC64 值是否相等，验证 CRC64 的确定性
    crc64("MyTestString") == crc64("MyTestString"),
    "crc64 is deterministic");
static_assert(
    // 检查不同字符串的 CRC64 值是否不相等，验证 CRC64 的不同输入产生不同结果
    crc64("MyTestString1") != crc64("MyTestString2"),
    "different strings, different result");

// 检查具体的预期值（对于 CRC64，使用 Jones 系数和初始值为 0）
static_assert(crc64_t{0} == crc64(""),
    // 空字符串的 CRC64 值应为 0
    "CRC64 of empty string should be 0");
static_assert(crc64_t{0xe9c6d914c4b8d9ca} == crc64("123456789"),
    // 特定输入字符串 "123456789" 的预期 CRC64 值
    "CRC64 of '123456789' should match expected value");
```