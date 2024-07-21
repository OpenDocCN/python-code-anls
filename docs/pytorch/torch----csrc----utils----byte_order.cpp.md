# `.\pytorch\torch\csrc\utils\byte_order.cpp`

```
// 包含 BFloat16 数据类型定义
#include <c10/util/BFloat16.h>
// 包含范围处理的工具函数
#include <c10/util/irange.h>
// 包含字节顺序处理的工具函数
#include <torch/csrc/utils/byte_order.h>

// 包含 C 标准库的字符串操作函数
#include <cstring>
// 包含 STL 容器向量
#include <vector>

// 如果编译器为 Microsoft Visual Studio
#if defined(_MSC_VER)
// 包含标准库函数定义
#include <stdlib.h>
#endif

// 匿名命名空间，定义静态内联函数
namespace {

// 交换两个字节的顺序（16位）
static inline void swapBytes16(void* ptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint16_t output;
    // 将内存中的数据复制到 output
    memcpy(&output, ptr, sizeof(uint16_t));
    // 根据编译器定义选择不同的字节交换方式
    #if defined(_MSC_VER) && !defined(_DEBUG)
    output = _byteswap_ushort(output);
    // 对于 LLVM 和 GCC 编译器，使用内建函数进行字节交换
    #elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
    output = __builtin_bswap16(output);
    // 其他情况手动进行字节交换
    #else
    uint16_t Hi = output >> 8;
    uint16_t Lo = output << 8;
    output = Hi | Lo;
    #endif
    // 将交换后的数据写回原始内存地址
    memcpy(ptr, &output, sizeof(uint16_t));
}

// 交换四个字节的顺序（32位）
static inline void swapBytes32(void* ptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint32_t output;
    // 将内存中的数据复制到 output
    memcpy(&output, ptr, sizeof(uint32_t));
    // 根据编译器定义选择不同的字节交换方式
    #if defined(_MSC_VER) && !defined(_DEBUG)
    output = _byteswap_ulong(output);
    // 对于 LLVM 和 GCC 编译器，使用内建函数进行字节交换
    #elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
    output = __builtin_bswap32(output);
    // 其他情况手动进行字节交换
    #else
    uint32_t Byte0 = output & 0x000000FF;
    uint32_t Byte1 = output & 0x0000FF00;
    uint32_t Byte2 = output & 0x00FF0000;
    uint32_t Byte3 = output & 0xFF000000;
    output = (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24);
    #endif
    // 将交换后的数据写回原始内存地址
    memcpy(ptr, &output, sizeof(uint32_t));
}

// 交换八个字节的顺序（64位）
static inline void swapBytes64(void* ptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint64_t output;
    // 将内存中的数据复制到 output
    memcpy(&output, ptr, sizeof(uint64_t));
    // 根据编译器定义选择不同的字节交换方式
    #if defined(_MSC_VER)
    output = _byteswap_uint64(output);
    // 对于 LLVM 和 GCC 编译器，使用内建函数进行字节交换
    #elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
    output = __builtin_bswap64(output);
    // 其他情况手动进行字节交换
    #else
    uint64_t Byte0 = output & 0x00000000000000FF;
    uint64_t Byte1 = output & 0x000000000000FF00;
    uint64_t Byte2 = output & 0x0000000000FF0000;
    uint64_t Byte3 = output & 0x00000000FF000000;
    uint64_t Byte4 = output & 0x000000FF00000000;
    uint64_t Byte5 = output & 0x0000FF0000000000;
    uint64_t Byte6 = output & 0x00FF000000000000;
    uint64_t Byte7 = output & 0xFF00000000000000;
    output = (Byte0 << (7 * 8)) | (Byte1 << (5 * 8)) | (Byte2 << (3 * 8)) |
             (Byte3 << (1 * 8)) | (Byte7 >> (7 * 8)) | (Byte6 >> (5 * 8)) |
             (Byte5 >> (3 * 8)) | (Byte4 >> (1 * 8));
    #endif
    // 将交换后的数据写回原始内存地址
    memcpy(ptr, &output, sizeof(uint64_t));
}

// 解码一个字节序为本地顺序的 16 位无符号整数
static inline uint16_t decodeUInt16(const uint8_t* data) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint16_t output;
    // 将内存中的数据复制到 output
    memcpy(&output, data, sizeof(uint16_t));
    // 返回解码后的整数值
    return output;
}

// 解码一个字节序为本地顺序的 16 位无符号整数并交换字节顺序
static inline uint16_t decodeUInt16ByteSwapped(const uint8_t* data) {
    // 解码获取原始数据
    uint16_t output = decodeUInt16(data);
    // 交换字节顺序
    swapBytes16(&output);
    // 返回交换后的整数值
    return output;
}

// 解码一个字节序为本地顺序的 32 位无符号整数
static inline uint32_t decodeUInt32(const uint8_t* data) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint32_t output;
    // 将内存中的数据复制到 output
    memcpy(&output, data, sizeof(uint32_t));
    // 返回解码后的整数值
    return output;
}

// 解码一个字节序为本地顺序的 32 位无符号整数并交换字节顺序
static inline uint32_t decodeUInt32ByteSwapped(const uint8_t* data) {
    // 解码获取原始数据
    uint32_t output = decodeUInt32(data);
    // 交换字节顺序
    swapBytes32(&output);
    // 返回交换后的整数值
    return output;
}

} // namespace
// 解码一个 uint64_t 类型的数据，从给定的 uint8_t 数据中读取
static inline uint64_t decodeUInt64(const uint8_t* data) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint64_t output;
  // 将 data 中的 sizeof(uint64_t) 字节复制到 output 中
  memcpy(&output, data, sizeof(uint64_t));
  // 返回解码后的 uint64_t 数据
  return output;
}

// 解码一个字节顺序已经反转的 uint64_t 类型数据
static inline uint64_t decodeUInt64ByteSwapped(const uint8_t* data) {
  // 调用 decodeUInt64 函数解码数据
  uint64_t output = decodeUInt64(data);
  // 使用 swapBytes64 函数反转 output 的字节顺序
  swapBytes64(&output);
  // 返回反转字节顺序后的 uint64_t 数据
  return output;
}

} // anonymous namespace

namespace torch::utils {

// 确定本机的字节序，并返回对应的 THPByteOrder 枚举值
THPByteOrder THP_nativeByteOrder() {
  uint32_t x = 1;
  // 判断 x 的低字节是否为 1，以确定本机字节序
  return *(uint8_t*)&x ? THP_LITTLE_ENDIAN : THP_BIG_ENDIAN;
}

// 解码一个 int16_t 类型的缓冲区
void THP_decodeInt16Buffer(
    int16_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // 根据需要进行字节顺序转换，然后将数据存入目标缓冲区
    dst[i] = (int16_t)(do_byte_swap ? decodeUInt16ByteSwapped(src)
                                    : decodeUInt16(src));
    // 移动到下一个 int16_t 的起始位置
    src += sizeof(int16_t);
  }
}

// 解码一个 int32_t 类型的缓冲区
void THP_decodeInt32Buffer(
    int32_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // 根据需要进行字节顺序转换，然后将数据存入目标缓冲区
    dst[i] = (int32_t)(do_byte_swap ? decodeUInt32ByteSwapped(src)
                                    : decodeUInt32(src));
    // 移动到下一个 int32_t 的起始位置
    src += sizeof(int32_t);
  }
}

// 解码一个 int64_t 类型的缓冲区
void THP_decodeInt64Buffer(
    int64_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // 根据需要进行字节顺序转换，然后将数据存入目标缓冲区
    dst[i] = (int64_t)(do_byte_swap ? decodeUInt64ByteSwapped(src)
                                    : decodeUInt64(src));
    // 移动到下一个 int64_t 的起始位置
    src += sizeof(int64_t);
  }
}

// 解码一个 c10::Half 类型的缓冲区
void THP_decodeHalfBuffer(
    c10::Half* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint16_t x;
      c10::Half f;
    };
    // 根据需要进行字节顺序转换，然后将数据存入目标缓冲区
    x = (do_byte_swap ? decodeUInt16ByteSwapped(src) : decodeUInt16(src));
    dst[i] = f;
    // 移动到下一个 c10::Half 的起始位置
    src += sizeof(uint16_t);
  }
}

// 解码一个 at::BFloat16 类型的缓冲区
void THP_decodeBFloat16Buffer(
    at::BFloat16* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    uint16_t x =
        (do_byte_swap ? decodeUInt16ByteSwapped(src) : decodeUInt16(src));
    // 将解码后的数据复制到目标缓冲区
    std::memcpy(&dst[i], &x, sizeof(dst[i]));
    // 移动到下一个 at::BFloat16 的起始位置
    src += sizeof(uint16_t);
  }
}

// 解码一个 bool 类型的缓冲区
void THP_decodeBoolBuffer(
    bool* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // 将 uint8_t 类型的数据转换为 bool 类型，存入目标缓冲区
    dst[i] = (int)src[i] != 0 ? true : false;
  }
}

// 解码一个 float 类型的缓冲区
void THP_decodeFloatBuffer(
    float* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t x;
      float f;
    };
    // 根据需要进行字节顺序转换，然后将数据存入目标缓冲区
    x = (do_byte_swap ? decodeUInt32ByteSwapped(src) : decodeUInt32(src));
    dst[i] = f;
    // 移动到下一个 float 的起始位置
    src += sizeof(float);
  }
}

// 解码一个 double 类型的缓冲区
void THP_decodeDoubleBuffer(
    double* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    # 定义一个联合体（union），在同一内存空间中存储一个64位整数和一个双精度浮点数
    union {
      uint64_t x;    # 定义联合体中的64位整数成员x
      double d;       # 定义联合体中的双精度浮点数成员d
    };
    # 根据条件选择使用不同的函数解码src指向的数据为64位整数，并赋值给联合体的x或d成员
    x = (do_byte_swap ? decodeUInt64ByteSwapped(src) : decodeUInt64(src));
    # 将联合体中的双精度浮点数d赋值给目标数组dst的第i个元素
    dst[i] = d;
    # 将src指针向后移动一个双精度浮点数的大小
    src += sizeof(double);
  }
void THP_decodeComplexFloatBuffer(
    c10::complex<float>* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t x;
      float re;
    };
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t y;
      float im;
    };

    // 根据字节顺序解码单精度浮点数，并存储到联合体 x 中
    x = (do_byte_swap ? decodeUInt32ByteSwapped(src) : decodeUInt32(src));
    src += sizeof(float);
    // 根据字节顺序解码单精度浮点数，并存储到联合体 y 中
    y = (do_byte_swap ? decodeUInt32ByteSwapped(src) : decodeUInt32(src));
    src += sizeof(float);

    // 使用 re 和 im 创建复数对象，并存储到 dst 数组中的第 i 个位置
    dst[i] = c10::complex<float>(re, im);
  }
}
    // 调用THP_decodeComplexFloatBuffer函数，解码源缓冲区中的复杂浮点数数据到目标缓冲区
    THP_decodeComplexFloatBuffer(dst, src,
        // 检查给定的字节顺序是否与本机字节顺序不同，如果不同则需要反转字节顺序
        (order != THP_nativeByteOrder()),
        // 数据的长度，即要处理的数据元素数目
        len);
}

void THP_decodeComplexDoubleBuffer(
    c10::complex<double>* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  // 调用另一个重载的函数，用于反序列化复杂双精度数缓冲区
  THP_decodeComplexDoubleBuffer(
      dst, src, (order != THP_nativeByteOrder()), len);
}

void THP_encodeInt16Buffer(
    uint8_t* dst,
    const int16_t* src,
    THPByteOrder order,
    size_t len) {
  // 将 int16_t 数组拷贝到目标缓冲区
  memcpy(dst, src, sizeof(int16_t) * len);
  // 如果字节顺序不是本地字节顺序，则交换每个 int16_t 数据的字节顺序
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i; // 抑制未使用变量警告
      swapBytes16(dst); // 交换当前位置的 int16_t 数据的字节顺序
      dst += sizeof(int16_t); // 移动指针到下一个 int16_t 数据的位置
    }
  }
}

void THP_encodeInt32Buffer(
    uint8_t* dst,
    const int32_t* src,
    THPByteOrder order,
    size_t len) {
  // 将 int32_t 数组拷贝到目标缓冲区
  memcpy(dst, src, sizeof(int32_t) * len);
  // 如果字节顺序不是本地字节顺序，则交换每个 int32_t 数据的字节顺序
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i; // 抑制未使用变量警告
      swapBytes32(dst); // 交换当前位置的 int32_t 数据的字节顺序
      dst += sizeof(int32_t); // 移动指针到下一个 int32_t 数据的位置
    }
  }
}

void THP_encodeInt64Buffer(
    uint8_t* dst,
    const int64_t* src,
    THPByteOrder order,
    size_t len) {
  // 将 int64_t 数组拷贝到目标缓冲区
  memcpy(dst, src, sizeof(int64_t) * len);
  // 如果字节顺序不是本地字节顺序，则交换每个 int64_t 数据的字节顺序
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i; // 抑制未使用变量警告
      swapBytes64(dst); // 交换当前位置的 int64_t 数据的字节顺序
      dst += sizeof(int64_t); // 移动指针到下一个 int64_t 数据的位置
    }
  }
}

void THP_encodeFloatBuffer(
    uint8_t* dst,
    const float* src,
    THPByteOrder order,
    size_t len) {
  // 将 float 数组拷贝到目标缓冲区
  memcpy(dst, src, sizeof(float) * len);
  // 如果字节顺序不是本地字节顺序，则交换每个 float 数据的字节顺序
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i; // 抑制未使用变量警告
      swapBytes32(dst); // 交换当前位置的 float 数据的字节顺序
      dst += sizeof(float); // 移动指针到下一个 float 数据的位置
    }
  }
}

void THP_encodeDoubleBuffer(
    uint8_t* dst,
    const double* src,
    THPByteOrder order,
    size_t len) {
  // 将 double 数组拷贝到目标缓冲区
  memcpy(dst, src, sizeof(double) * len);
  // 如果字节顺序不是本地字节顺序，则交换每个 double 数据的字节顺序
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i; // 抑制未使用变量警告
      swapBytes64(dst); // 交换当前位置的 double 数据的字节顺序
      dst += sizeof(double); // 移动指针到下一个 double 数据的位置
    }
  }
}

template <typename T>
std::vector<T> complex_to_float(const c10::complex<T>* src, size_t len) {
  // 将复数数组转换为浮点数数组
  std::vector<T> new_src;
  new_src.reserve(2 * len);
  for (const auto i : c10::irange(len)) {
    auto elem = src[i];
    new_src.emplace_back(elem.real()); // 添加实部到新数组
    new_src.emplace_back(elem.imag()); // 添加虚部到新数组
  }
  return new_src;
}

void THP_encodeComplexFloatBuffer(
    uint8_t* dst,
    const c10::complex<float>* src,
    THPByteOrder order,
    size_t len) {
  // 将复杂浮点数数组转换为浮点数数组，并拷贝到目标缓冲区
  auto new_src = complex_to_float(src, len);
  memcpy(dst, static_cast<void*>(&new_src), 2 * sizeof(float) * len);
  // 如果字节顺序不是本地字节顺序，则交换每个浮点数数据的字节顺序
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(2 * len)) {
      (void)i; // 抑制未使用变量警告
      swapBytes32(dst); // 交换当前位置的浮点数数据的字节顺序
      dst += sizeof(float); // 移动指针到下一个浮点数数据的位置
    }
  }
}

void THP_encodeComplexDoubleBuffer(
    uint8_t* dst,
    const c10::complex<double>* src,
    THPByteOrder order,
    size_t len) {
  // 将复杂双精度浮点数数组转换为浮点数数组，并拷贝到目标缓冲区
  auto new_src = complex_to_float(src, len);
  memcpy(dst, static_cast<void*>(&new_src), 2 * sizeof(double) * len);
  // 如果字节顺序不是本地字节顺序，则交换每个浮点数数据的字节顺序
  if (order != THP_nativeByteOrder()) {
    // 使用 C10 库中的 irange 函数生成一个范围为 [0, 2 * len) 的迭代器，遍历这个范围
    for (const auto i : c10::irange(2 * len)) {
      // (void)i; // 抑制未使用变量的警告，i 在循环体内没有实际使用
      // 交换 dst 指向的内存块中的 64 位数据的字节顺序
      swapBytes64(dst);
      // 将 dst 指针向后移动 sizeof(double) 个字节，指向下一个 double 类型的内存块
      dst += sizeof(double);
    }
  }
}

} // namespace torch::utils
```