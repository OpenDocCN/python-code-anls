# `D:\src\scipysrc\scikit-learn\sklearn\utils\src\MurmurHash3.h`

```
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#ifndef _MURMURHASH3_H_
#define _MURMURHASH3_H_

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER)
// 定义特定于 Microsoft Visual Studio 的数据类型别名
typedef unsigned char uint8_t;
typedef unsigned long uint32_t;
typedef unsigned __int64 uint64_t;

// Other compilers

#else    // defined(_MSC_VER)
// 使用标准头文件定义的数据类型别名
#include <stdint.h>

#endif // !defined(_MSC_VER)

//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// 函数声明：32位 x86 架构的 MurmurHash3 哈希函数
void MurmurHash3_x86_32  ( const void * key, int len, uint32_t seed, void * out );

// 函数声明：128位 x86 架构的 MurmurHash3 哈希函数
void MurmurHash3_x86_128 ( const void * key, int len, uint32_t seed, void * out );

// 函数声明：128位 x64 架构的 MurmurHash3 哈希函数
void MurmurHash3_x64_128 ( const void * key, int len, uint32_t seed, void * out );

#ifdef __cplusplus
}
#endif

//-----------------------------------------------------------------------------

#endif // _MURMURHASH3_H_
```