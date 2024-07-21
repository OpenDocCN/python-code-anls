# `.\pytorch\torch\csrc\utils\byte_order.h`

```
#pragma once

#include <c10/util/BFloat16.h>  // 包含BFloat16类型的头文件
#include <c10/util/Float8_e4m3fn.h>  // 包含Float8_e4m3fn类型的头文件
#include <c10/util/Float8_e4m3fnuz.h>  // 包含Float8_e4m3fnuz类型的头文件
#include <c10/util/Float8_e5m2.h>  // 包含Float8_e5m2类型的头文件
#include <c10/util/Float8_e5m2fnuz.h>  // 包含Float8_e5m2fnuz类型的头文件
#include <c10/util/Half.h>  // 包含Half类型的头文件
#include <torch/csrc/Export.h>  // 包含导出API的头文件
#include <cstddef>  // 包含size_t类型的头文件
#include <cstdint>  // 包含int16_t, int32_t, int64_t等整数类型的头文件

#ifdef __FreeBSD__  // 如果目标平台是FreeBSD
#include <sys/endian.h>  // 包含字节序转换函数的头文件
#include <sys/types.h>  // 包含基本类型的头文件
#define thp_bswap16(x) bswap16(x)  // 定义16位大小端转换宏
#define thp_bswap32(x) bswap32(x)  // 定义32位大小端转换宏
#define thp_bswap64(x) bswap64(x)  // 定义64位大小端转换宏
#elif defined(__APPLE__)  // 如果目标平台是苹果操作系统
#include <libkern/OSByteOrder.h>  // 包含苹果平台字节序转换函数的头文件
#define thp_bswap16(x) OSSwapInt16(x)  // 定义16位大小端转换宏
#define thp_bswap32(x) OSSwapInt32(x)  // 定义32位大小端转换宏
#define thp_bswap64(x) OSSwapInt64(x)  // 定义64位大小端转换宏
#elif defined(__GNUC__) && !defined(__MINGW32__)  // 如果目标平台是GNU编译器，且不是MinGW环境
#include <byteswap.h>  // 包含字节序转换函数的头文件
#define thp_bswap16(x) bswap_16(x)  // 定义16位大小端转换宏
#define thp_bswap32(x) bswap_32(x)  // 定义32位大小端转换宏
#define thp_bswap64(x) bswap_64(x)  // 定义64位大小端转换宏
#elif defined _WIN32 || defined _WIN64  // 如果目标平台是Windows
#define thp_bswap16(x) _byteswap_ushort(x)  // 定义16位大小端转换宏
#define thp_bswap32(x) _byteswap_ulong(x)  // 定义32位大小端转换宏
#define thp_bswap64(x) _byteswap_uint64(x)  // 定义64位大小端转换宏
#endif

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__  // 如果当前平台是小端序
#define to_be16(x) thp_bswap16(x)  // 定义16位大小端转换宏（大端到小端）
#define from_be16(x) thp_bswap16(x)  // 定义16位大小端转换宏（小端到大端）
#define to_be32(x) thp_bswap32(x)  // 定义32位大小端转换宏（大端到小端）
#define from_be32(x) thp_bswap32(x)  // 定义32位大小端转换宏（小端到大端）
#define to_be64(x) thp_bswap64(x)  // 定义64位大小端转换宏（大端到小端）
#define from_be64(x) thp_bswap64(x)  // 定义64位大小端转换宏（小端到大端）
#define to_le16(x) (x)  // 定义16位小端转换宏（小端到小端，无需转换）
#define from_le16(x) (x)  // 定义16位小端转换宏（小端到小端，无需转换）
#define to_le32(x) (x)  // 定义32位小端转换宏（小端到小端，无需转换）
#define from_le32(x) (x)  // 定义32位小端转换宏（小端到小端，无需转换）
#define to_le64(x) (x)  // 定义64位小端转换宏（小端到小端，无需转换）
#define from_le64(x) (x)  // 定义64位小端转换宏（小端到小端，无需转换）
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__  // 如果当前平台是大端序
#define to_be16(x) (x)  // 定义16位大小端转换宏（大端到大端，无需转换）
#define from_be16(x) (x)  // 定义16位大小端转换宏（大端到大端，无需转换）
#define to_be32(x) (x)  // 定义32位大小端转换宏（大端到大端，无需转换）
#define from_be32(x) (x)  // 定义32位大小端转换宏（大端到大端，无需转换）
#define to_be64(x) (x)  // 定义64位大小端转换宏（大端到大端，无需转换）
#define from_be64(x) (x)  // 定义64位大小端转换宏（大端到大端，无需转换）
#define to_le16(x) thp_bswap16(x)  // 定义16位小端转换宏（大端到小端）
#define from_le16(x) thp_bswap16(x)  // 定义16位小端转换宏（小端到大端）
#define to_le32(x) thp_bswap32(x)  // 定义32位小端转换宏（大端到小端）
#define from_le32(x) thp_bswap32(x)  // 定义32位小端转换宏（小端到大端）
#define to_le64(x) thp_bswap64(x)  // 定义64位小端转换宏（大端到小端）
#define from_le64(x) thp_bswap64(x)  // 定义64位小端转换宏（小端到大端）
#else
#error Unexpected or undefined __BYTE_ORDER__  // 如果字节序未定义或不符合预期，抛出错误
#endif

namespace torch::utils {

enum THPByteOrder { THP_LITTLE_ENDIAN = 0, THP_BIG_ENDIAN = 1 };  // 定义枚举类型THPByteOrder，表示小端序和大端序

TORCH_API THPByteOrder THP_nativeByteOrder();  // 声明获取本机字节序的API函数

TORCH_API void THP_decodeInt16Buffer(  // 声明解码int16_t类型缓冲区的API函数
    int16_t* dst,  // 目标缓冲区指针
    const uint8_t* src,  // 源缓冲区指针（字节流）
    bool do_byte_swap,  // 是否进行字节序转换的布尔标志
    size_t len);  // 缓冲区长度

TORCH_API void THP_decodeInt32Buffer(  // 声明解码int32_t类型缓冲区的API函数
    int32_t* dst,  // 目标缓冲区指针
    const uint8_t* src,  // 源缓冲区指针（字节流）
    bool do_byte_swap,  // 是否进行字节序转换的布尔标志
    size_t len);  // 缓冲区长度

TORCH_API void THP_decodeInt64Buffer(  // 声明解码int64_t类型缓冲区的API函数
    int64_t* dst,  // 目标缓冲区指针
    const uint8_t* src,  // 源缓冲区指针（字节流）
    bool do_byte_swap,  // 是否进行字节序
    c10::complex<float>* dst,
    // 参数 dst：指向复数数组的指针，存储处理后的结果

    const uint8_t* src,
    // 参数 src：指向 uint8_t 类型数组的指针，存储输入数据

    bool do_byte_swap,
    // 参数 do_byte_swap：布尔值，指示是否需要进行字节交换

    size_t len);
    // 参数 len：表示数组的长度，即要处理的元素个数
// 解码复杂双精度缓冲区的函数定义
TORCH_API void THP_decodeComplexDoubleBuffer(
    c10::complex<double>* dst,  // 目标数组指针，用于存储解码后的复杂双精度数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    bool do_byte_swap,          // 是否进行字节交换的标志位，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码 int16 类型数据缓冲区的函数定义
TORCH_API void THP_decodeInt16Buffer(
    int16_t* dst,               // 目标数组指针，用于存储解码后的 int16 数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码 int32 类型数据缓冲区的函数定义
TORCH_API void THP_decodeInt32Buffer(
    int32_t* dst,               // 目标数组指针，用于存储解码后的 int32 数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码 int64 类型数据缓冲区的函数定义
TORCH_API void THP_decodeInt64Buffer(
    int64_t* dst,               // 目标数组指针，用于存储解码后的 int64 数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码半精度数据缓冲区的函数定义
TORCH_API void THP_decodeHalfBuffer(
    c10::Half* dst,             // 目标数组指针，用于存储解码后的半精度数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码单精度浮点数缓冲区的函数定义
TORCH_API void THP_decodeFloatBuffer(
    float* dst,                 // 目标数组指针，用于存储解码后的单精度浮点数数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码双精度浮点数缓冲区的函数定义
TORCH_API void THP_decodeDoubleBuffer(
    double* dst,                // 目标数组指针，用于存储解码后的双精度浮点数数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码布尔类型数据缓冲区的函数定义
TORCH_API void THP_decodeBoolBuffer(
    bool* dst,                  // 目标数组指针，用于存储解码后的布尔类型数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码 BFloat16 类型数据缓冲区的函数定义
TORCH_API void THP_decodeBFloat16Buffer(
    at::BFloat16* dst,          // 目标数组指针，用于存储解码后的 BFloat16 数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码 Float8_e5m2 类型数据缓冲区的函数定义
TORCH_API void THP_decodeFloat8_e5m2Buffer(
    at::Float8_e5m2* dst,       // 目标数组指针，用于存储解码后的 Float8_e5m2 数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    size_t len);                // 待解码数据的长度

// 解码 Float8_e4m3fn 类型数据缓冲区的函数定义
TORCH_API void THP_decodeFloat8_e4m3fnBuffer(
    at::Float8_e4m3fn* dst,     // 目标数组指针，用于存储解码后的 Float8_e4m3fn 数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    size_t len);                // 待解码数据的长度

// 解码 Float8_e5m2fnuz 类型数据缓冲区的函数定义
TORCH_API void THP_decodeFloat8_e5m2fnuzBuffer(
    at::Float8_e5m2fnuz* dst,   // 目标数组指针，用于存储解码后的 Float8_e5m2fnuz 数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    size_t len);                // 待解码数据的长度

// 解码 Float8_e4m3fnuz 类型数据缓冲区的函数定义
TORCH_API void THP_decodeFloat8_e4m3fnuzBuffer(
    at::Float8_e4m3fnuz* dst,   // 目标数组指针，用于存储解码后的 Float8_e4m3fnuz 数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    size_t len);                // 待解码数据的长度

// 解码复杂单精度缓冲区的函数定义
TORCH_API void THP_decodeComplexFloatBuffer(
    c10::complex<float>* dst,   // 目标数组指针，用于存储解码后的复杂单精度数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 解码复杂双精度缓冲区的函数定义
TORCH_API void THP_decodeComplexDoubleBuffer(
    c10::complex<double>* dst,  // 目标数组指针，用于存储解码后的复杂双精度数据
    const uint8_t* src,         // 源数据缓冲区的指针，包含待解码的数据
    THPByteOrder order,         // 数据的字节序，用于处理数据端序问题
    size_t len);                // 待解码数据的长度

// 编码 int16 类型数据缓冲区的函数定义
TORCH_API void THP_encodeInt16Buffer(
    uint8_t* dst,               // 目标数据缓冲区的指针，用于存储编码后的数据
    const int16_t* src,         //
```