# `.\pytorch\c10\util\SmallVector.cpp`

```
/// LLVM/ADT/SmallVector.cpp - 实现了 SmallVector 类
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 该文件实现了 SmallVector 类。
//
//===----------------------------------------------------------------------===//

// ATen: 从 llvm::SmallVector 修改而来。
// 替换 llvm::safe_malloc 为 std::bad_alloc
// 删除了 LLVM_ENABLE_EXCEPTIONS

#include <c10/util/SmallVector.h> // 包含 SmallVector 头文件
#include <cstdint> // 包含 C++ 标准整数类型头文件
#include <stdexcept> // 包含 C++ 标准异常头文件
#include <string> // 包含 C++ 标准字符串头文件
using namespace c10; // 使用 c10 命名空间

// 检查没有浪费字节且所有内容都是良好对齐的。
namespace {
// 这些结构可能会在 AIX 上引起二进制兼容性警告。忽略警告，因为我们只在下面的静态断言中使用这些类型。
#if defined(_AIX)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waix-compat"
#endif
struct Struct16B {
  alignas(16) void* X;
};
struct Struct32B {
  alignas(32) void* X;
};
#if defined(_AIX)
#pragma GCC diagnostic pop
#endif
} // namespace

// 静态断言，检查 SmallVector<void*, 0> 大小和对齐
static_assert(
    sizeof(SmallVector<void*, 0>) == sizeof(unsigned) * 2 + sizeof(void*),
    "SmallVector size 0 中存在浪费空间");
static_assert(
    alignof(SmallVector<Struct16B, 0>) >= alignof(Struct16B),
    "16字节对齐的 T 类型对齐错误");
static_assert(
    alignof(SmallVector<Struct32B, 0>) >= alignof(Struct32B),
    "32字节对齐的 T 类型对齐错误");
static_assert(
    sizeof(SmallVector<Struct16B, 0>) >= alignof(Struct16B),
    "16字节对齐的 T 类型缺少填充");
static_assert(
    sizeof(SmallVector<Struct32B, 0>) >= alignof(Struct32B),
    "32字节对齐的 T 类型缺少填充");
static_assert(
    sizeof(SmallVector<void*, 1>) == sizeof(unsigned) * 2 + sizeof(void*) * 2,
    "SmallVector size 1 中存在浪费空间");

static_assert(
    sizeof(SmallVector<char, 0>) == sizeof(void*) * 2 + sizeof(void*),
    "1字节元素的大小和容量使用了字大小的类型");

/// 当 MinSize 超出此向量的大小类型范围时报告。抛出 std::length_error 或调用 report_fatal_error。
[[noreturn]] static void report_size_overflow(size_t MinSize, size_t MaxSize);

/// 报告向量已达到最大容量。抛出 std::length_error 或调用 report_fatal_error。
[[noreturn]] static void report_at_maximum_capacity(size_t MaxSize);


这些注释将每行代码解释其作用，包括静态断言的验证和异常处理函数的定义。
// 报告达到最大容量时的异常，抛出长度错误并包含详细原因信息
static void report_at_maximum_capacity(size_t MaxSize) {
  // 构造异常信息字符串，指出 SmallVector 容量无法增长，已经达到最大尺寸 MaxSize
  std::string Reason =
      "SmallVector capacity unable to grow. Already at maximum size " +
      std::to_string(MaxSize);
  // 抛出长度错误异常，将详细原因信息传入异常对象
  throw std::length_error(Reason);
}

// 注意：将此函数移到头文件可能会导致性能回归。
// 根据传入的 MinSize、TSize 和 OldCapacity 计算新的容量 NewCapacity
template <class Size_T>
static size_t getNewCapacity(size_t MinSize, size_t TSize, size_t OldCapacity) {
  // 获取 Size_T 类型的最大值
  constexpr size_t MaxSize = std::numeric_limits<Size_T>::max();

  // 确保能够容纳新容量
  // 仅当容量为 32 位时才会适用此检查
  if (MinSize > MaxSize)
    report_size_overflow(MinSize, MaxSize);

  // 确保至少有空间保证至少可以容纳一个新元素
  // 仅当容量为 32 位时才会适用此检查
  if (OldCapacity == MaxSize)
    report_at_maximum_capacity(MaxSize);

  // 根据旧容量计算新容量，确保能够容纳更多元素
  size_t NewCapacity = 2 * OldCapacity + 1; // 总是增长
  return std::min(std::max(NewCapacity, MinSize), MaxSize);
}

// 注意：将此函数移到头文件可能会导致性能回归。
// 分配内存以进行扩展，根据 MinSize、TSize 和当前容量计算新的容量
template <class Size_T>
void* SmallVectorBase<Size_T>::mallocForGrow(
    size_t MinSize,
    size_t TSize,
    size_t& NewCapacity) {
  // 计算新的容量
  NewCapacity = getNewCapacity<Size_T>(MinSize, TSize, this->capacity());
  // 使用 malloc 分配新容量大小的内存空间
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  auto Result = std::malloc(NewCapacity * TSize);
  // 如果分配失败，抛出 bad_alloc 异常
  if (Result == nullptr) {
    throw std::bad_alloc();
  }
  return Result;
}

// 注意：将此函数移到头文件可能会导致性能回归。
// 增加 SmallVector 的容量，根据 MinSize 和 TSize 计算新的容量
template <class Size_T>
void SmallVectorBase<Size_T>::grow_pod(
    const void* FirstEl,
    size_t MinSize,
    size_t TSize) {
  // 计算新的容量
  size_t NewCapacity = getNewCapacity<Size_T>(MinSize, TSize, this->capacity());
  void* NewElts = nullptr;
  // 如果当前的 BeginX 等于传入的 FirstEl 指针，表示从 inline 复制增长
  if (BeginX == FirstEl) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    // 使用 malloc 分配新容量大小的内存空间
    NewElts = std::malloc(NewCapacity * TSize);
    // 如果分配失败，抛出 bad_alloc 异常
    if (NewElts == nullptr) {
      throw std::bad_alloc();
    }

    // 复制元素到新的内存空间，对 POD 类型元素不需要调用析构函数
    memcpy(NewElts, this->BeginX, size() * TSize);
  } else {
    // 如果不是从内联复制增长，则调整已分配空间的大小
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    NewElts = std::realloc(this->BeginX, NewCapacity * TSize);
    // 如果重新分配失败，抛出 bad_alloc 异常
    if (NewElts == nullptr) {
      throw std::bad_alloc();
    }
  }

  // 更新 BeginX 和 Capacity 到新的值
  this->BeginX = NewElts;
  this->Capacity = NewCapacity;
}

// 实例化 SmallVectorBase 模板类，使用 uint32_t 类型作为 Size_T
template class c10::SmallVectorBase<uint32_t>;

// 禁用在 32 位构建中使用 uint64_t 实例化
// 在 64 位构建中，需要同时使用 uint32_t 和 uint64_t 的实例化
// 32 位构建中永远不会使用此实例化，并且会导致
// 当 sizeof(Size_T) > sizeof(size_t) 时发出警告。
#if SIZE_MAX > UINT32_MAX
// 实例化模板类 SmallVectorBase<uint64_t>，用于处理大尺寸的 SmallVector。
template class c10::SmallVectorBase<uint64_t>;

// 确保此 #if 与 SmallVectorSizeType 保持同步的断言。
static_assert(
    sizeof(SmallVectorSizeType<char>) == sizeof(uint64_t),
    "Expected SmallVectorBase<uint64_t> variant to be in use.");
#else
// 如果 sizeof(Size_T) 不大于 sizeof(size_t)，则实例化模板类 SmallVectorBase<uint32_t>。
static_assert(
    sizeof(SmallVectorSizeType<char>) == sizeof(uint32_t),
    "Expected SmallVectorBase<uint32_t> variant to be in use.");
#endif
```