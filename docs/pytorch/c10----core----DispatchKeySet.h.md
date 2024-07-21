# `.\pytorch\c10\core\DispatchKeySet.h`

```py
#pragma once
// 防止头文件被多次包含

#include <c10/core/DispatchKey.h>
// 包含 DispatchKey 相关的头文件，用于定义 DispatchKey 类

#include <c10/macros/Export.h>
// 包含导出宏定义的头文件，用于定义 C10_API 宏

#include <c10/macros/Macros.h>
// 包含通用宏定义的头文件，用于定义 C10_ALWAYS_INLINE 等宏

#include <c10/util/Exception.h>
// 包含异常处理相关的头文件，用于异常处理工具

#include <c10/util/Metaprogramming.h>
// 包含元编程相关的头文件，用于实现元编程功能

#include <c10/util/TypeList.h>
// 包含类型列表相关的头文件，用于类型列表工具

#include <c10/util/llvmMathExtras.h>
// 包含 LLVM 数学扩展相关的头文件，用于数学工具

#include <array>
// 包含数组标准库头文件，用于定义 std::array

#include <cstddef>
// 包含标准库头文件，用于定义 std::size_t

#include <cstdint>
// 包含标准整数类型头文件，用于定义标准整数类型

#include <initializer_list>
// 包含初始化列表头文件，用于初始化列表工具

#include <iterator>
// 包含迭代器头文件，用于定义迭代器相关工具

#include <ostream>
// 包含输出流头文件，用于定义输出流工具

#include <string>
// 包含字符串头文件，用于定义字符串工具

#include <type_traits>
// 包含类型特性头文件，用于定义类型特性工具

namespace c10 {

struct FunctionalityOffsetAndMask {
  // 表示功能偏移量和掩码的结构体

  // 空的构造函数不应该被使用；仅在初始化数组之前使用。
  FunctionalityOffsetAndMask() = default;

  // 构造函数，初始化 offset 和 mask
  FunctionalityOffsetAndMask(uint16_t offset, uint16_t mask)
      : offset(offset), mask(mask) {}

  // 这个大小需要足够覆盖操作表的大小。
  uint16_t offset{};  // 偏移量，用于操作表
  // 参见注释 [No More Than 16 Backends]
  // 这个掩码需要足够大，以掩盖所有后端位。
  // 我们可能不会有超过 16 个后端位，所以 uint16_t 应该足够了。
  uint16_t mask{};    // 掩码，用于屏蔽后端位
};

static_assert(
    c10::num_runtime_entries < 65536,
    "The dispatcher currently only supports up to 2^16 runtime entries");
// 静态断言，确保运行时条目数小于 65536，因为调度器目前最多支持 2^16 个运行时条目

C10_API std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks();
// 声明初始化功能偏移量和掩码的函数，返回一个包含 FunctionalityOffsetAndMask 的数组

C10_ALWAYS_INLINE static const std::
    array<FunctionalityOffsetAndMask, num_functionality_keys>&
    offsetsAndMasks() {
  // 声明一个内联函数，返回一个静态的 offsets_and_masks_ 数组
  static auto offsets_and_masks_ = initializeFunctionalityOffsetsAndMasks();
  return offsets_and_masks_;
}

// 一个 DispatchKeySet 的表示。DispatchKeySet 包含功能位和后端位，
// 每个张量都有自己的 DispatchKeySet。调度器通过获取每个输入张量的 keyset，
// 将它们进行或运算，并调度到特定的功能块。功能位是*有序*的。
// 当设置多个功能位时，我们使用最高优先级的功能。类似地，如果使用多个张量从不同设备调用操作
// （例如 CPU 和 CUDA），理论上可以设置多个后端位，尽管混合设备调度的支持有限
// （目前唯一能够优雅处理混合设备输入的内核是接受标量 CPU 张量的 CUDA 内核）。

// DispatchKeySet 的集合表示。一个张量可能有多个张量类型 id，例如，变量张量也可以是 CPU 张量；
// DispatchKeySet 指定了哪些类型 id 适用。内部表示为一个 64 位位集
// （这意味着仅支持 64 个张量类型 id）。

// 如上所述，DispatchKey 是有序的；因此，我们可以问诸如“集合中的最高优先级 DispatchKey 是什么”？
// （集合本身是无序的；具有相同 id 的两个集合将始终以相同的方式排序 id。）

// 注释 [DispatchKeySet Internal Representation]
// Internally, dispatch keys are packed into 64-bit DispatchKeySet objects
// that get passed around at runtime.
// However, there isn't necessarily a 1-to-1 mapping between bits in the keyset
// and individual dispatch keys.
//
// First: why do we have this distinction, and why not map every dispatch key
// directly to a bit? This is mostly because we have several types of
// functionalities that different backends would like to customize. For example,
// we have:
// - "Dense":     CPU, CUDA, XLA, ... (~12 keys)
// - "Sparse":    SparseCPU, SparseCUDA, ...
// - "SparseCsr": SparseCsrCPU, SparseCsrCUDA, ...
// - "Quantized": QuantizedCPU, QuantizedCUDA, QuantizedXLA, ...
// - "Autograd":  AutogradCPU, AutogradCUDA, Autograd XLA, ...
// The problem is that total number of keys grows quadratically with [#
// backends] x [# functionalities], making it very difficult to map each key
// directly to a bit in a bitset without dramatically increasing the size of the
// bitset over time.
//
// The two enums (BackendComponent and DispatchKey) can be divided roughly into
// 5 categories.
//
// (1) "Building block" keys
//    (a) backends: Everything in the BackendComponent enum (e.g. CPUBit,
//    CUDABit) (b) functionalities: (per-backend) functionality-bit DispatchKeys
//    (e.g. AutogradFunctionality, SparseCsr, Sparse, Dense)
// (2) "Runtime" keys
//    (a) "non-customizable backends" (e.g. FPGA)
//    (b) "non-customizable functionalities" (e.g. Functionalize)
//    (c) "per-backend instances of customizable functionalities" (e.g. CPU,
//    SparseCPU, AutogradCPU)
// (3) "Alias" DispatchKeys (see Note [Alias Dispatch Keys])
//
// (1) Building block keys always correspond to individual bits in a
// DispatchKeySet. They can also be combined in a DispatchKeySet to form actual
// runtime keys. e.g.
//     auto dense_cpu_ks = DispatchKeySet({DispatchKey::CPUBit,
//     DispatchKey::Dense});
//     // The keyset has the runtime dense-cpu key.
//     dense_cpu_ks.has(DispatchKey::CPU);
//     // And it contains the building block keys too.
//     dense_cpu_ks.has(DispatchKey::CPUBit);
//     dense_cpu_ks.has(DispatchKey::Dense);
//
// Not every backend and not every functionality counts as a "building block
// key". This is mostly to give us more levers to pull in the design space.
// Backend keys and functionality keys that count as "building blocks" will
// contribute to a full cross product of functionality that can be overriden.
//
// For example, right now we have at least 12 "backend" building
// blocks (CPU, CUDA, XLA, ...) and at least 5 "functionality"
// building blocks (Dense, Sparse, SparseCsr, Quantized,
// AutogradFunctionality, ...). These keys together allow every
// dispatcher operator to be customized in up to 12*4 different
// ways. Each of those requires a slot in the operator table of every
// dispatcher operator.  Not every piece of functionality necessarily
// needs to be customizable per-backend, and not every backend
// An undefined tensor is one with an empty tensor type set.
class DispatchKeySet final {
 public:
  enum Full { FULL };  // 枚举类型 Full，表示完整的 DispatchKeySet
  enum FullAfter { FULL_AFTER };  // 枚举类型 FullAfter，表示在某个 DispatchKey 之后的完整集合

  // NB: default constructor representation as zero is MANDATORY as
  // use of DispatchKeySet in TLS requires this.
  // 默认构造函数，将 repr_ 设为零，这是因为在 TLS 中使用 DispatchKeySet 必须要有这种表示方式。
  constexpr DispatchKeySet() = default;

  // 构造函数，创建一个包含所有功能的 DispatchKeySet
  constexpr DispatchKeySet(Full)
      : repr_((1ULL << (num_backends + num_functionality_keys - 1)) - 1) {}

  // 构造函数，创建一个在指定 DispatchKey 之后的完整集合 DispatchKeySet
  constexpr DispatchKeySet(FullAfter, DispatchKey t)
      // LSB after t are OK, but not t itself.
      // "functionalities" have a notion of ordering (e.g. Autograd > Sparse >
      // Quantized > Dense). But backends don't really have an ordering.
      // Therefore, we're enforcing that FullAfter can only be used on
      // "functionality" keys.
      // 根据给定的 DispatchKey t 创建一个在其后的完整集合，要求只能在功能键上使用 FullAfter。
      : repr_(
            (1ULL
             << (num_backends + static_cast<uint8_t>(toFunctionalityKey(t)) -
                 1)) -
            1) {
    *this = add(DispatchKey::PythonDispatcher);  // 添加 PythonDispatcher 到当前集合中
  }

  // 公共构造函数，根据原始值 x 创建 DispatchKeySet
  constexpr DispatchKeySet(Raw, uint64_t x) : repr_(x) {}

  // 显式构造函数，根据 BackendComponent k 创建 DispatchKeySet
  constexpr explicit DispatchKeySet(BackendComponent k) {
    if (k == BackendComponent::InvalidBit) {
      repr_ = 0;  // 如果 k 是 InvalidBit，则将 repr_ 设为零
    } else {
      repr_ = 1ULL << (static_cast<uint8_t>(k) - 1);  // 否则根据 k 的值设定 repr_
    }
  }

  // 显式构造函数，根据 DispatchKey k 创建 DispatchKeySet
  constexpr explicit DispatchKeySet(DispatchKey k) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    // 根据给定的 DispatchKey k 创建 DispatchKeySet

      repr_ = 1ULL << static_cast<uint8_t>(k);
  }

 private:
  uint64_t repr_;  // 存储 DispatchKeySet 的位表示形式
};
    if (k == DispatchKey::Undefined) {
      // Case 1: 处理 Undefined 的情况
      // 设置 repr_ 为 0
      repr_ = 0;
    } else if (k <= DispatchKey::EndOfFunctionalityKeys) {
      // Case 2: 处理仅包含功能位的键
      // 这些键具有功能位设置，但没有后端位
      // 可能是有效的运行时键或“构建块”键
      // 功能性值为 num_backends + static_cast<uint8_t>(k) - 1 对应位的值
      uint64_t functionality_val = 1ULL
          << (num_backends + static_cast<uint8_t>(k) - 1);
      repr_ = functionality_val;
    } else if (k <= DispatchKey::EndOfRuntimeBackendKeys) {
      // Case 3: 同时具有功能位和后端位的运行时键
      // 首先计算功能位的翻转
      auto functionality_k = toFunctionalityKey(k);
      // 减 1 是因为 Undefined 在技术上是一个不在位集中的“功能”，所以 Dense 技术上是第二个功能，但是最低的功能位
      uint64_t functionality_val = 1ULL
          << (num_backends + static_cast<uint8_t>(functionality_k) - 1);

      // 然后计算后端位的翻转
      // Case 4a: 处理“每个后端功能”的运行时实例
      // 例如，给定 DispatchKey::CPU，应设置：
      // - Dense 功能位
      // - CPUBit 后端位
      // 首先计算要翻转的后端位
      auto backend_k = toBackendComponent(k);
      uint64_t backend_val = backend_k == BackendComponent::InvalidBit
          ? 0
          : 1ULL << (static_cast<uint8_t>(backend_k) - 1);
      // 将功能位和后端位的值加在一起，得到 repr_
      repr_ = functionality_val + backend_val;
    } else {
      // 到达此处，应该已经覆盖了除了别名键之外的所有情况
      // 现在，将 repr_ 设置为 0
      repr_ = 0;
    }
  }

  // 将一组 DispatchKey 转换为对应的 repr 值
  constexpr uint64_t keys_to_repr(std::initializer_list<DispatchKey> ks) {
    uint64_t repr = 0;
    for (auto k : ks) {
      // 将每个 DispatchKey 对应的 DispatchKeySet 的 repr_ 值进行按位或操作
      repr |= DispatchKeySet(k).repr_;
    }
    return repr;
  }

  // 将一组 BackendComponent 转换为对应的 repr 值
  constexpr uint64_t backend_bits_to_repr(
      std::initializer_list<BackendComponent> ks) {
    uint64_t repr = 0;
    for (auto k : ks) {
      // 将每个 BackendComponent 对应的 DispatchKeySet 的 repr_ 值进行按位或操作
      repr |= DispatchKeySet(k).repr_;
    }
    return repr;
  }
  // 返回 repr 变量，即当前对象的内部表示
  return repr;
}

explicit constexpr DispatchKeySet(std::initializer_list<DispatchKey> ks)
    : repr_(keys_to_repr(ks)) {}

explicit constexpr DispatchKeySet(std::initializer_list<BackendComponent> ks)
    // 注意：由于某些原因，直接在构造函数中放置此逻辑似乎无法在 CUDA 10.1 上编译通过。
    // 详见内部失败示例 https://www.internalfb.com/intern/skycastle/run/76561193669136035/artifact/actionlog.76561193742069401.stderr
    : repr_(backend_bits_to_repr(ks)) {}

// 测试 DispatchKey 是否存在于集合中
inline bool has(DispatchKey t) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t != DispatchKey::Undefined);
  return has_all(DispatchKeySet(t));
}
constexpr bool has_backend(BackendComponent t) const {
  return has_all(DispatchKeySet(t));
}

// 测试集合是否包含所有给定的 DispatchKeySet
constexpr bool has_all(DispatchKeySet ks) const {
  return static_cast<bool>((repr_ & ks.repr_) == ks.repr_);
}

// 测试集合是否包含任意给定的 DispatchKeySet
inline bool has_any(DispatchKeySet ks) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      // 输入的 keyset 中要么没有后端位
      ((ks.repr_ & full_backend_mask) == 0) ||
      // 要么没有每个后端功能位
      // 参见 [Note: Per-Backend Functionality Dispatch Keys]
      ((ks &
        DispatchKeySet({
                           DispatchKey::Dense,
                           DispatchKey::Quantized,
                           DispatchKey::Sparse,
                           DispatchKey::SparseCsr,
                           DispatchKey::AutogradFunctionality,
                       })
            .repr_) == 0));
  return static_cast<bool>((repr_ & ks.repr_) != 0);
}
// 测试 DispatchKeySet 是否是 ks 的超集
bool isSupersetOf(DispatchKeySet ks) const {
  return (repr_ & ks.repr_) == ks.repr_;
}
// 执行集合的并运算
constexpr DispatchKeySet operator|(DispatchKeySet other) const {
  return DispatchKeySet(repr_ | other.repr_);
}
// 执行集合的交运算
constexpr DispatchKeySet operator&(DispatchKeySet other) const {
  }
  // 计算集合的差集 self - other，
  // 但仅适用于功能键。
  // self 上的任何后端位都将保持不变。
  // 参见注释 [仅影响功能键的 DispatchKeySet 中删除键]
  constexpr DispatchKeySet operator-(DispatchKeySet other) const {
    // 使用按位与操作符计算差集
    return DispatchKeySet(repr_ & (full_backend_mask | ~other.repr_));
  }

  // 计算集合的对称差集 self ^ other
  constexpr DispatchKeySet operator^(DispatchKeySet other) const {
    // 使用按位异或操作符计算对称差集
    return DispatchKeySet(repr_ ^ other.repr_);
  }

  // 比较两个 DispatchKeySet 是否相等
  bool operator==(DispatchKeySet other) const {
    // 使用相等操作符比较内部表示
    return repr_ == other.repr_;
  }

  // 比较两个 DispatchKeySet 是否不相等
  bool operator!=(DispatchKeySet other) const {
    // 使用不等操作符比较内部表示
    return repr_ != other.repr_;
  }

  // 向 DispatchKeySet 添加一个 DispatchKey。不会改变原集合，返回扩展后的 DispatchKeySet！
  C10_NODISCARD constexpr DispatchKeySet add(DispatchKey t) const {
    // 使用按位或操作符将一个 DispatchKey 添加到集合中
    return *this | DispatchKeySet(t);
  }

  // 向 DispatchKeySet 添加另一个 DispatchKeySet
  C10_NODISCARD constexpr DispatchKeySet add(DispatchKeySet ks) const {
    // 使用按位或操作符将另一个 DispatchKeySet 添加到集合中
    return *this | ks;
  }

  // 从 DispatchKeySet 中移除一个 DispatchKey
  // 通常不建议执行此操作（用于实现打印重载 operator<<）
  //
  // 注释 [仅影响功能键的 DispatchKeySet 中删除键]
  // 只允许从键集中移除功能位。
  // 目前，我们只允许从键集中移除“功能位”，这是计算 fallthrough 键逻辑所必需的。
  // 移除后端位为何会有问题？考虑以下示例：
  //
  // DispatchKeySet([DispatchKey.CPU, DispatchKey.AutogradCUDA,
  // DispatchKey.CUDA]).remove(DispatchKey.AutogradCUDA)
  // DispatchKeySet([DispatchKey.CPU,
  // DispatchKey.AutogradCUDA]).remove(DispatchKey.AutogradCUDA)
  //
  // 我们希望发生什么？
  // 从技术上讲，我们希望在移除后，第一个键集仍然具有 CUDA 调度键，而第二个键集则没有。
  // 不幸的是，没有办法表示这一点，因为这两个键集在内部表示上是相同的：
  // 功能位：Autograd，Dense；后端位：CPU，CUDA
  //
  // 因此，remove(DispatchKey.AutogradCPU) 只会从位集中移除 “Autograd” 位。
  C10_NODISCARD constexpr DispatchKeySet remove(DispatchKey t) const {
    // 使用按位与和按位非操作符从集合中移除指定的 DispatchKey
    return DispatchKeySet(
        repr_ & ~(DispatchKeySet(t).repr_ & ~full_backend_mask));
  }

  // 允许从 DispatchKeySet 中移除后端位，
  // 但必须明确说明（使用 remove_backend() 而不是 remove()）。
  constexpr DispatchKeySet remove_backend(BackendComponent b) const {
    // 使用按位与和按位非操作符从集合中移除指定的后端组件
    return DispatchKeySet(repr_ & ~(DispatchKeySet(b).repr_));
  }

  // 集合是否为空？（即未定义的张量）
  bool empty() const {
    // 检查内部表示是否为零
    return repr_ == 0;
  }

  // 返回原始的内部表示
  uint64_t raw_repr() {
    // 返回当前实例的内部表示
    return repr_;
  }

  DispatchKey highestFunctionalityKey() const {
    // 找到表示最高位功能性的索引
    auto functionality_idx = indexOfHighestBit();
    // 如果功能性索引小于后端数目，表示没有设置任何功能位
    if (functionality_idx < num_backends)
      // 返回未定义的分发键
      return DispatchKey::Undefined;
    // 前 num_backends 位不对应真实的分发键
    // 计算并返回对应的分发键
    return static_cast<DispatchKey>(functionality_idx - num_backends);
  }

  // 这个函数类似于 toBackendComponent(DispatchKey)，但不那么限制性。
  // toBackendComponent() 如果传入的键没有后端位，则报错，用于错误检查。
  // 这里我们需要一个版本，能处理像 FPGA 这样的“假”后端，它们需要映射到 AutogradOther 键。
  // 对于这些后端，我们返回 BackendComponent::InvalidBit。
  BackendComponent highestBackendKey() const {
    // 创建掩码以屏蔽功能性位
    auto backend_idx =
        DispatchKeySet(repr_ & full_backend_mask).indexOfHighestBit();
    // 如果所有后端位都是零，表示没有设置任何后端位
    if (backend_idx == 0)
      // 返回无效的后端位
      return BackendComponent::InvalidBit;
    // 返回对应的后端组件
    return static_cast<BackendComponent>(backend_idx);
  }

  // 返回集合中优先级最高的 DispatchKey。
  DispatchKey highestPriorityTypeId() const {
    // 获取最高功能性键
    auto functionality_k = highestFunctionalityKey();
    // 如果是每个后端功能性键
    if (isPerBackendFunctionalityKey(functionality_k)) {
      // 转换到运行时每个后端功能性键
      return toRuntimePerBackendFunctionalityKey(
          functionality_k, highestBackendKey());
    }
    // 直接返回功能性键
    return functionality_k;
  }

  // 返回集合中最高位的索引。
  // 这用于计算操作表中的部分，以获取：
  // - 集合中最高的“功能性”位。
  // - 集合中最高的“后端”位。
  uint8_t indexOfHighestBit() const {
    // 使用 LLVM 的函数计算前导零的数量，然后从 64 减去得到最高位索引
    return 64 - llvm::countLeadingZeros(repr_);
  }
#if defined(C10_MOBILE_TRIM_DISPATCH_KEYS)
  // [Note: Trimmed Mobile Dispatch Keys]
  /**
   * The method below maps the dispatch key in the enum DispatchKey to an
   * integer index in the dispatchTable_ array in OperatorEntry. The array
   * is trimmed for mobile to reduce peak memory usage since it's
   * unnecessary to reserve additional space for dispatch keys that will
   * never be used on mobile.
   */
  // 根据最高优先级的调度键确定其在 dispatchTable_ 数组中的整数索引
  int getDispatchTableIndexForDispatchKeySet() const {
    // 获取最高优先级的调度键
    auto dk = highestPriorityTypeId();
    switch (dk) {
      // 根据不同的调度键返回对应的索引
      case DispatchKey::Undefined:
        return 0;
      case DispatchKey::CPU:
        return 1;
      case DispatchKey::QuantizedCPU:
        return 2;
      case DispatchKey::SparseCPU:
        return 3;
      case DispatchKey::BackendSelect:
        return 4;
      case DispatchKey::ADInplaceOrView:
        return 5;
      case DispatchKey::AutogradOther:
        return 6;
      case DispatchKey::AutogradCPU:
        return 7;
      default:
        return -1;  // 如果调度键未知，则返回 -1
    }
  }
#else
  // returns the index in the operator table of highest priority key in the the
  // keyset Note that we could in theory implement this using
  // highestPriorityTypeId(), but this code is very hotpath and we can do it
  // faster without it.
  // 返回关键集中优先级最高键的运算符表中的索引
  // 注意，理论上我们可以使用 highestPriorityTypeId() 来实现这一点，
  // 但这段代码非常热门，我们可以不用它更快地完成。
  int getDispatchTableIndexForDispatchKeySet() const {
    // 计算功能索引，即关键集中优先级最高的索引
    auto functionality_idx =
        DispatchKeySet(repr_ >> num_backends).indexOfHighestBit();
    auto offset_and_mask = offsetsAndMasks()[functionality_idx];
    // 首先屏蔽功能位，然后右移 1 位。
    // 右移 1 位是因为所有都是从零开始索引。
    // 例如，000001（CPU）应该给我们一个偏移量为 0，000010（CUDA）应该
    // 给我们一个偏移量为 1，依此类推。
    auto backend_idx =
        DispatchKeySet((repr_ & offset_and_mask.mask) >> 1).indexOfHighestBit();
    return offset_and_mask.offset + backend_idx;
  }
#endif

  // returns the "index" of the highest priority backend in the keyset.
  // This is pretty similar to getBackendKey(), but:
  // - It's hotpath code (part of the runtime bitset calculation)
  // - I's returns an integer index, not an enum value
  // - Everything is shifted to the right by 1.
  //   BackendComponent::InvalidBit is technically the lowest enum value,
  //   but it isn't included in the runtime table. So CPUBit = 1, CUDABit = 2,
  //   etc.
  // 返回关键集中优先级最高后端的“索引”。
  // 这与 getBackendKey() 类似，但是：
  // - 这是热门路径代码（运行时位集计算的一部分）
  // - 它返回一个整数索引，而不是枚举值
  // - 所有都右移 1 位。
  //   BackendComponent::InvalidBit 在技术上是最低的枚举值，
  //   但它不包括在运行时表中。因此 CPUBit = 1，CUDABit = 2，等等。
  uint64_t getBackendIndex() const {
    // 返回存储在 DispatchKeySet 中的最高位的索引值，使用按位与和位移操作计算
    return DispatchKeySet((repr_ & full_backend_mask) >> 1).indexOfHighestBit();
  }

 private:
  // 使用给定的 uint64_t 类型的 repr 参数，初始化 DispatchKeySet 对象
  constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
  // 存储 DispatchKeySet 的内部表示，默认初始化为 0
  uint64_t repr_ = 0;

 public:
  // STL 迭代器类，用于遍历 DispatchKeySet 中的所有运行时 DispatchKey
  // 迭代器只在底层 DispatchKeySet 销毁时失效，因为迭代器存储指向 DispatchKeySet 原始表示的指针。
  // 注意：当遇到每个后端功能（如 Dense 或 Sparse）时，我们会遍历该功能在 keyset 中的每个后端。
  // 例如，如果下一个功能键要迭代的是 Autograd，并且 keyset 中的后端位对应于 [BackendComponent::CPUBit, BackendComponent::CUDABit]，
  // 则返回的下两个键将是 DispatchKey::AutogradCPU、DispatchKey::AutogradCUDA（CPU 先，因为在 DispatchKey.h 中，CPU 的优先级低于 CUDA）。
  class iterator {
   public:
    using self_type = iterator;
    using iterator_category = std::input_iterator_tag;
    using value_type = DispatchKey;
    using difference_type = ptrdiff_t;
    using reference = value_type&;
    using pointer = value_type*;
    // end_iter_mask_val 应该掩码掉整个 keyset
    static const uint8_t end_iter_mask_val =
        num_backends + num_functionality_keys;
    // end_iter_key_val 应该是最后一个 DispatchKey
    static const uint8_t end_iter_key_val = num_functionality_keys;

    // current_dispatchkey_idx_ 将遍历所有功能位
    // current_backendcomponent_idx_ 将遍历所有后端位
    explicit iterator(
        const uint64_t* data_ptr,
        uint8_t next_functionality = num_backends,
        uint8_t next_backend = 0)
        : data_ptr_(data_ptr),
          next_functionality_(next_functionality),
          next_backend_(next_backend),
          // 在构造时处于无效状态，并由第一个增量调用设置
          current_dispatchkey_idx_(end_iter_key_val),
          current_backendcomponent_idx_(end_iter_key_val) {
      // 转到集合中的第一个键
      TORCH_INTERNAL_ASSERT(
          next_functionality_ >= num_backends,
          "num_backends=",
          static_cast<uint32_t>(num_backends),
          "next_functionality_=",
          static_cast<uint32_t>(next_functionality_));
      ++(*this);
    }

    // 前缀递增运算符：用于迭代器的前进操作
    C10_API self_type& operator++();

    // 后缀递增运算符：用于迭代器的前进操作，返回之前的迭代器状态
    self_type operator++(int) {
      self_type previous_iterator = *this;
      ++(*this);
      return previous_iterator;
    }

    // 相等比较运算符：用于比较两个迭代器是否相等
    bool operator==(const self_type& rhs) const {
      return next_functionality_ == rhs.next_functionality_ &&
          current_dispatchkey_idx_ == rhs.current_dispatchkey_idx_ &&
          next_backend_ == rhs.next_backend_ &&
          current_backendcomponent_idx_ == rhs.current_backendcomponent_idx_;
    }
    // 比较当前对象和另一个对象是否不相等，返回布尔值
    bool operator!=(const self_type& rhs) const {
      // 比较四个成员变量是否都不相等，只要有一个不相等即返回true
      return next_functionality_ != rhs.next_functionality_ ||
          current_dispatchkey_idx_ != rhs.current_dispatchkey_idx_ ||
          next_backend_ != rhs.next_backend_ ||
          current_backendcomponent_idx_ != rhs.current_backendcomponent_idx_;
    }
    
    // 返回当前对象指向的 DispatchKey
    DispatchKey operator*() const {
      // 将 current_dispatchkey_idx_ 转换为 DispatchKey 类型
      auto functionality_key =
          static_cast<DispatchKey>(current_dispatchkey_idx_);
      
      // 如果是每个后端的功能键，则转换为相应的运行时后端功能键
      if (isPerBackendFunctionalityKey(functionality_key)) {
        // 根据当前索引和后端组件索引获取下一个键
        auto next_key = toRuntimePerBackendFunctionalityKey(
            functionality_key,
            static_cast<BackendComponent>(current_backendcomponent_idx_));
        
        // 断言：确保转换后的运行时键和当前的后端组件键一致
        TORCH_INTERNAL_ASSERT(
            toBackendComponent(next_key) ==
                static_cast<BackendComponent>(current_backendcomponent_idx_),
            "Tried to map functionality key ",
            toString(functionality_key),
            " and backend bit ",
            toString(
                static_cast<BackendComponent>(current_backendcomponent_idx_)),
            " to a runtime key, but ended up with ",
            toString(next_key),
            ". This can happen if the order of the backend dispatch keys in DispatchKey.h isn't consistent.",
            " Please double check that enum for inconsistencies.");
        
        // 返回计算得到的下一个键
        return next_key;
      } else {
        // 如果不是每个后端的功能键，则直接返回功能键本身
        return functionality_key;
      }
    }

   private:
    // 指向数据的指针
    const uint64_t* data_ptr_;
    // 下一个功能键的索引
    uint8_t next_functionality_;
    // 下一个后端的索引
    uint8_t next_backend_;
    // 当前调度键的索引
    uint8_t current_dispatchkey_idx_;
    // 当前后端组件的索引
    uint8_t current_backendcomponent_idx_;
  };

 public:
  // 返回指向集合中第一个键的迭代器，如果集合为空，则返回末尾迭代器
  iterator begin() const {
    return iterator(&repr_);
  }

  // 将 EndOfFunctionalityKeys 视为集合的末尾迭代器
  iterator end() const {
    return iterator(&repr_, iterator::end_iter_mask_val);
  }
};

// 声明一个函数 toString，接受 DispatchKeySet 类型参数，返回 std::string
C10_API std::string toString(DispatchKeySet);
// 声明一个函数 operator<<，接受 std::ostream 和 DispatchKeySet 类型参数，返回 std::ostream 引用
C10_API std::ostream& operator<<(std::ostream&, DispatchKeySet);

// 内联函数，根据给定的 DispatchKey 返回对应的 DispatchTable 索引
C10_API inline int getDispatchTableIndexForDispatchKey(DispatchKey k) {
  return DispatchKeySet(k).getDispatchTableIndexForDispatchKeySet();
}

// 声明一个常量表达式 DispatchKeySet autograd_dispatch_keyset，包含自动求导相关的 DispatchKey
// 这些键不包括后端位 (BackendComponent::CPUBit 等)
// 见注释 [autograd_dispatch_keyset Does Not Include Backend Bits] 了解详情
constexpr DispatchKeySet autograd_dispatch_keyset = DispatchKeySet({
    DispatchKey::AutogradFunctionality,
    DispatchKey::AutogradOther,
    DispatchKey::AutogradNestedTensor,
});

// 声明一个常量表达式 DispatchKeySet autocast_dispatch_keyset，包含自动转换相关的 DispatchKey
constexpr DispatchKeySet autocast_dispatch_keyset = DispatchKeySet({
    DispatchKey::AutocastCPU,
    DispatchKey::AutocastCUDA,
    DispatchKey::AutocastXPU,
    DispatchKey::AutocastIPU,
    DispatchKey::AutocastHPU,
    DispatchKey::AutocastXLA,
    DispatchKey::AutocastPrivateUse1,
});

// 声明一个常量表达式 DispatchKeySet default_included_set，包含默认包含的 DispatchKey
constexpr DispatchKeySet default_included_set = DispatchKeySet({
    DispatchKey::BackendSelect,
    DispatchKey::ADInplaceOrView,
});

// 声明一个常量表达式 DispatchKeySet default_excluded_set，包含默认排除的 DispatchKey
constexpr DispatchKeySet default_excluded_set = DispatchKeySet({
    DispatchKey::AutocastCPU,
    DispatchKey::AutocastCUDA,
    DispatchKey::AutocastXPU,
    DispatchKey::AutocastIPU,
    DispatchKey::AutocastHPU,
    DispatchKey::AutocastXLA,
    DispatchKey::AutocastPrivateUse1,
});

// 声明一个常量表达式 DispatchKeySet autograd_dispatch_keyset_with_ADInplaceOrView，
// 合并 autograd_dispatch_keyset 和 DispatchKey::ADInplaceOrView
constexpr DispatchKeySet autograd_dispatch_keyset_with_ADInplaceOrView =
    autograd_dispatch_keyset | DispatchKeySet(DispatchKey::ADInplaceOrView);

// 声明一个常量表达式 DispatchKeySet python_ks，包含 Python 相关的 DispatchKey
constexpr DispatchKeySet python_ks = DispatchKeySet({
    DispatchKey::Python,
    DispatchKey::PythonTLSSnapshot,
});

// 声明一个常量表达式 DispatchKeySet sparse_ks，包含 Sparse 相关的 DispatchKey
constexpr DispatchKeySet sparse_ks = DispatchKeySet(DispatchKey::Sparse);

// 声明一个常量表达式 DispatchKeySet sparse_csr_ks，包含 SparseCsr 相关的 DispatchKey
constexpr DispatchKeySet sparse_csr_ks = DispatchKeySet(DispatchKey::SparseCsr);

// 声明一个常量表达式 DispatchKeySet mkldnn_ks，包含 MkldnnCPU 相关的 DispatchKey
constexpr DispatchKeySet mkldnn_ks = DispatchKeySet(DispatchKey::MkldnnCPU);

// 声明一个常量表达式 DispatchKeySet autogradother_backends，
// 包含与 DispatchKey::AutogradOther 对应的后端 DispatchKey
constexpr DispatchKeySet autogradother_backends =
    DispatchKeySet(
        // 声明一个调度键集合，用于存储支持的后端类型，除了 HIP 和 VE，它们有自己的后端位表示
        // 这意味着它们可以拥有自己的自动求导键。
        // 技术上讲，HIP 现在会重新分派到其自定义的 AutogradHIP 槽位在运行时表中。
        {DispatchKey::FPGA,                // 添加 FPGA 后端键
         DispatchKey::MAIA,                // 添加 MAIA 后端键
         DispatchKey::Vulkan,              // 添加 Vulkan 后端键
         DispatchKey::Metal,               // 添加 Metal 后端键
         DispatchKey::CustomRNGKeyId,      // 添加 CustomRNGKeyId 后端键
         DispatchKey::MkldnnCPU,           // 添加 MkldnnCPU 后端键
         // 稀疏和量化后端也在此处
         DispatchKey::Sparse,              // 添加 Sparse 后端键
         DispatchKey::SparseCsr,           // 添加 SparseCsr 后端键
         DispatchKey::Quantized})          // 添加 Quantized 后端键
    // 包括后端位，因为这个键集在操作注册过程中使用，需要循环遍历所有运行时自动求导和其他后端键
    | DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);
// 定义一个包含 AutogradOther 之后所有调度键的集合
// 注意：这依赖于 AutogradOther 目前是最低的 Autograd 键
constexpr DispatchKeySet after_autograd_keyset =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, c10::DispatchKey::AutogradOther);

// 定义一个包含 ADInplaceOrView 之后所有调度键的集合
constexpr DispatchKeySet after_ADInplaceOrView_keyset = DispatchKeySet(
    DispatchKeySet::FULL_AFTER,
    c10::DispatchKey::ADInplaceOrView);

// 定义一个包含 Functionalize 之后所有调度键的集合，并从中移除 ADInplaceOrView 键
constexpr DispatchKeySet after_func_keyset =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, c10::DispatchKey::Functionalize)
        .remove(
            // 注意：在重新分派后函数内核时，需要从键集中移除 ADInplaceOrView。
            // 这是因为我们不再调用相同的原位操作；我们最初调用的是一个原位操作，
            // 而现在不是了。原始的键计算根据原位操作确定哪些键是 Fallthrough。
            // 这意味着它没有将 ADInplaceOrView 内核作为 Fallthrough 键。然而，
            // 我们希望现在调用一个非原位操作时忽略 ADInplaceOrView 内核。
            // 通过在这里显式移除该键，我们可以更高效地使用 at::redispatch。
            c10::DispatchKey::ADInplaceOrView);

// 定义一个后端比特掩码集合，用于表示所有后端的调度键
constexpr DispatchKeySet backend_bitset_mask =
    DispatchKeySet(DispatchKeySet::RAW, (1ULL << num_backends) - 1);

// 定义各个 Autograd 相关的调度键集合
constexpr auto inplace_or_view_ks =
    DispatchKeySet(DispatchKey::ADInplaceOrView);
constexpr auto autograd_cpu_ks = DispatchKeySet(DispatchKey::AutogradCPU);
constexpr auto autograd_ipu_ks = DispatchKeySet(DispatchKey::AutogradIPU);
constexpr auto autograd_xpu_ks = DispatchKeySet(DispatchKey::AutogradXPU);
constexpr auto autograd_cuda_ks = DispatchKeySet(DispatchKey::AutogradCUDA);
constexpr auto autograd_xla_ks = DispatchKeySet(DispatchKey::AutogradXLA);
constexpr auto autograd_lazy_ks = DispatchKeySet(DispatchKey::AutogradLazy);
constexpr auto autograd_meta_ks = DispatchKeySet(DispatchKey::AutogradMeta);
constexpr auto autograd_mps_ks = DispatchKeySet(DispatchKey::AutogradMPS);
constexpr auto autograd_hpu_ks = DispatchKeySet(DispatchKey::AutogradHPU);
constexpr auto autograd_privateuse1_ks =
    DispatchKeySet(DispatchKey::AutogradPrivateUse1);
constexpr auto autograd_privateuse2_ks =
    DispatchKeySet(DispatchKey::AutogradPrivateUse2);
constexpr auto autograd_privateuse3_ks =
    DispatchKeySet(DispatchKey::AutogradPrivateUse3);
constexpr auto autograd_other_ks = DispatchKeySet(DispatchKey::AutogradOther);
constexpr auto autograd_nested =
    DispatchKeySet(DispatchKey::AutogradNestedTensor);
// 与 Functorch 键对应的键集，这些键具有自己的专用
// 定义一个包含多个 DispatchKey 的常量表达式集合，用于标识 TensorImpl 的功能转换
constexpr auto functorch_transforms_ks = DispatchKeySet(
    {DispatchKey::FuncTorchBatched,
     DispatchKey::FuncTorchVmapMode,
     DispatchKey::Batched,
     DispatchKey::VmapMode,
     DispatchKey::FuncTorchGradWrapper});

// 定义一个仅包含 DispatchKey::FuncTorchBatched 的常量表达式集合
constexpr auto functorch_batched_ks =
    DispatchKeySet({DispatchKey::FuncTorchBatched});

// 定义一个 DispatchKeySet，包含后端功能位 (dense, sparse, quantized) 对应的 DispatchKey，
// 以及所有后端位的组合
constexpr DispatchKeySet backend_functionality_keys =
    DispatchKeySet({
        DispatchKey::Dense,
        DispatchKey::Quantized,
        DispatchKey::Sparse,
        DispatchKey::SparseCsr,
    }) |
    DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);

// 定义一个结构体 OpTableOffsetAndMask，包含 offset 和 backend_mask 两个 uint16_t 类型的成员
struct OpTableOffsetAndMask {
  uint16_t offset; // 偏移量
  uint16_t backend_mask; // 后端掩码
};

// 静态断言，确保后端数量不超过 16
static_assert(
    num_backends <= 16,
    "Right now we expect the number of backends not to exceed 16. In the (unlikely) event"
    " that this changes, the size of OpTableOffsetAndMask::backend_mask needs to be increased too.");

// 函数声明：判断给定的 DispatchKey 是否是后端 DispatchKey
C10_API bool isBackendDispatchKey(DispatchKey t);

// 函数声明：根据别名 DispatchKey 解析为 DispatchKeySet（如果适用）
C10_API DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t);

// 函数声明：根据别名 DispatchKey 解析为 DispatchKeySet（如果适用），
// 并检查 k 是否属于该集合
C10_API bool runtimeDispatchKeySetHas(DispatchKey t, DispatchKey k);

// 函数声明：返回一个包含所有与 Autograd DispatchKey t 相关的后端键的 DispatchKeySet，
// 如果 t 不是 DispatchKey::Autograd 的别名，则返回空的 DispatchKeySet。
C10_API DispatchKeySet getBackendKeySetFromAutograd(DispatchKey t);

// 函数声明：返回一个与后端相关的 Autograd 键集合，
// 对于给定的后端键，使用关联的 Autograd 键。
// 对于非后端键，使用 AutogradOther 作为默认值。
// 注意：在这里返回一个默认值更为方便和快速，而不是返回 optional<DispatchKey> 或抛出异常。
// 但调用者需要负责确保只传递后端键作为参数，或者仔细解释返回值。
inline DispatchKeySet getAutogradRelatedKeySetFromBackend(BackendComponent t) {
  switch (t) {
    case BackendComponent::CPUBit:
      return inplace_or_view_ks | autograd_cpu_ks;
    case BackendComponent::IPUBit:
      return inplace_or_view_ks | autograd_ipu_ks;
    case BackendComponent::XPUBit:
      return inplace_or_view_ks | autograd_xpu_ks;
    case BackendComponent::CUDABit:
      return inplace_or_view_ks | autograd_cuda_ks;
    case BackendComponent::XLABit:
      return inplace_or_view_ks | autograd_xla_ks;
    case BackendComponent::LazyBit:
      return inplace_or_view_ks | autograd_lazy_ks;
    case BackendComponent::MetaBit:
      return inplace_or_view_ks | autograd_meta_ks;
    case BackendComponent::MPSBit:
      return inplace_or_view_ks | autograd_mps_ks;
    // 没有匹配的情况，默认返回 inplace_or_view_ks
    default:
      return inplace_or_view_ks;
  }
}
    # 根据不同的后端组件类型进行不同的位操作，并返回结果
    case BackendComponent::HPUBit:
      return inplace_or_view_ks | autograd_hpu_ks;
    # 对于 HPUBit 后端组件，返回 inplace_or_view_ks 与 autograd_hpu_ks 的按位或结果
    case BackendComponent::PrivateUse1Bit:
      return inplace_or_view_ks | autograd_privateuse1_ks;
    # 对于 PrivateUse1Bit 后端组件，返回 inplace_or_view_ks 与 autograd_privateuse1_ks 的按位或结果
    case BackendComponent::PrivateUse2Bit:
      return inplace_or_view_ks | autograd_privateuse2_ks;
    # 对于 PrivateUse2Bit 后端组件，返回 inplace_or_view_ks 与 autograd_privateuse2_ks 的按位或结果
    case BackendComponent::PrivateUse3Bit:
      return inplace_or_view_ks | autograd_privateuse3_ks;
    # 对于 PrivateUse3Bit 后端组件，返回 inplace_or_view_ks 与 autograd_privateuse3_ks 的按位或结果
    default:
      return inplace_or_view_ks | autograd_other_ks;
    # 对于默认情况（未匹配到上述任何后端组件类型），返回 inplace_or_view_ks 与 autograd_other_ks 的按位或结果
// 返回一个由与自动转换相关的键映射到后端的 DispatchKeySet。
inline DispatchKeySet getAutocastRelatedKeySetFromBackend(BackendComponent t) {
  // 定义不同后端组件的自动转换 DispatchKeySet 常量
  constexpr auto autocast_cpu_ks = DispatchKeySet(DispatchKey::AutocastCPU);
  constexpr auto autocast_xpu_ks = DispatchKeySet(DispatchKey::AutocastXPU);
  constexpr auto autocast_ipu_ks = DispatchKeySet(DispatchKey::AutocastIPU);
  constexpr auto autocast_hpu_ks = DispatchKeySet(DispatchKey::AutocastHPU);
  constexpr auto autocast_cuda_ks = DispatchKeySet(DispatchKey::AutocastCUDA);
  constexpr auto autocast_xla_ks = DispatchKeySet(DispatchKey::AutocastXLA);
  constexpr auto autocast_privateuse1_ks =
      DispatchKeySet(DispatchKey::AutocastPrivateUse1);
  
  // 根据给定的后端组件 t 返回对应的自动转换 DispatchKeySet
  switch (t) {
    case BackendComponent::CPUBit:
      return autocast_cpu_ks;
    case BackendComponent::XPUBit:
      return autocast_xpu_ks;
    case BackendComponent::IPUBit:
      return autocast_ipu_ks;
    case BackendComponent::HPUBit:
      return autocast_hpu_ks;
    case BackendComponent::CUDABit:
      return autocast_cuda_ks;
    case BackendComponent::XLABit:
      return autocast_xla_ks;
    case BackendComponent::PrivateUse1Bit:
      return autocast_privateuse1_ks;
    default:
      return DispatchKeySet(); // 默认情况下返回空的 DispatchKeySet
  }
}

// 返回集合中优先级最高的“后端” DispatchKey。
// 这类似于 highestBackendKey()，但我们还有一些对应于后端的“功能性”位（如 Sparse、Quantized）。
inline DispatchKey highestPriorityBackendTypeId(DispatchKeySet ks) {
  // 返回集合中与后端功能键相交的最高优先级的类型 ID
  return (ks & backend_functionality_keys).highestPriorityTypeId();
}

// 此 API 的存在是因为我们在 OperatorEntry.cpp 中有一种情况需要检查 getRuntimeDispatchKeySet(alias).has(DispatchKey::Undefined)，
// 但我们不允许在 has() API 中使用它。
C10_API bool isIncludedInAlias(DispatchKey k, DispatchKey alias);

// 从历史上看，每个张量只有一个 DispatchKey，并且它总是类似于 CPU 的某种类型，并且 TLS 不能导致张量的 DispatchKey 更改。
// 但是我们仍然有一些遗留代码仍在使用 DispatchKey 进行诸如 instanceof 检查等操作；如果可能的话，请重构代码以停止在这些情况下使用 DispatchKey。
inline DispatchKey legacyExtractDispatchKey(DispatchKeySet s) {
  // 注意：如果你添加了任何额外的可以存储在 TensorImpl 中的键，例如 autograd 键和 ADInplaceOrView 键，则需要在此处添加。
  return (s - autograd_dispatch_keyset_with_ADInplaceOrView -
          autocast_dispatch_keyset -
          DispatchKeySet(
              {DispatchKey::Functionalize,
               DispatchKey::PythonTLSSnapshot,
               DispatchKey::Python}))
      .highestPriorityTypeId();
}

// 用于检查模板类型 T 是否不是 DispatchKeySet。
template <class T>
using is_not_DispatchKeySet = std::negation<std::is_same<DispatchKeySet, T>>;
// 给定一个函数类型，构造一个函数特性类型 function_traits，如果第一个参数是 DispatchKeySet 类型，
// 则从函数类型中去除第一个参数类型。注意：当前情况下，DispatchKeySet 被显式地隐藏在 JIT 中
// （主要是为了避免在堆栈上推送不必要的参数 - 详见“Note [ Plumbing Keys Through the Dispatcher]”）。如果将来需要将此类型暴露给 JIT，
// 则需要重新审视此类型别名的使用。
template <class FuncType>
using remove_DispatchKeySet_arg_from_func = guts::make_function_traits_t<
    typename guts::infer_function_traits_t<FuncType>::return_type,
    typename std::conditional_t<
        std::is_same_v<
            DispatchKeySet,
            typename guts::typelist::head_with_default_t<
                void,
                typename guts::infer_function_traits_t<
                    FuncType>::parameter_types>>,
        guts::typelist::drop_if_nonempty_t<
            typename guts::infer_function_traits_t<FuncType>::parameter_types,
            1>,
        typename guts::infer_function_traits_t<FuncType>::parameter_types>>;
} // namespace c10
```