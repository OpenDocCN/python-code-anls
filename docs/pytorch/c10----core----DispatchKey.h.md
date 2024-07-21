# `.\pytorch\c10\core\DispatchKey.h`

```
// 预处理指令，确保本文件只被编译一次
#pragma once

// 包含 C10 库中的头文件
#include <c10/core/DeviceType.h>
#include <c10/macros/Export.h>
#include <cstddef> // 包含标准库头文件 <cstddef>
#include <cstdint> // 包含标准库头文件 <cstdint>
#include <functional> // 包含标准库头文件 <functional>
#include <ostream> // 包含标准库头文件 <ostream>
#include <string> // 包含标准库头文件 <string>

// 定义命名空间 c10，用于封装库中的功能
namespace c10 {

// BackendComponent 枚举类型用于标识分发时的后端组件
// 每个值代表一种后端，允许不同功能注册各自的处理器
// 用于确定分发时要使用的后端实现
#define C10_FORALL_BACKEND_COMPONENTS(_, extra) \
  _(CPU, extra)                                 \
  _(CUDA, extra)                                \
  _(HIP, extra)                                 \
  _(XLA, extra)                                 \
  _(MPS, extra)                                 \
  _(IPU, extra)                                 \
  _(XPU, extra)                                 \
  _(HPU, extra)                                 \
  _(VE, extra)                                  \
  _(Lazy, extra)                                \
  _(MTIA, extra)                                \
  _(PrivateUse1, extra)                         \
  _(PrivateUse2, extra)                         \
  _(PrivateUse3, extra)                         \
  _(Meta, extra)

// WARNING! 如果在此列表末尾添加新的后端组件，请确保在 Meta 之前注册它。
// Meta 必须位于末尾，以便在 tls 中的 meta 键触发 meta 内核。
// （但不应该这样做：私有使用键应该比所有内置键的优先级都高）

// 如果在此处添加新的（非私有使用）后端，请确保在 aten/src/ATen/core/VariableFallbackKernel.cpp 中添加 Autograd<Backend> 的 fallthrough 内核

// WARNING! 如果我们添加了一个新的每个后端功能键，其优先级高于 Autograd，请确保更新 EndOfRuntimeBackendKeys

// 定义所有功能键的枚举
#define C10_FORALL_FUNCTIONALITY_KEYS(_) \
  _(Dense, )                             \
  _(Quantized, Quantized)                \
  _(Sparse, Sparse)                      \
  _(SparseCsr, SparseCsr)                \
  _(NestedTensor, NestedTensor)          \
  _(AutogradFunctionality, Autograd)

} // namespace c10
// 定义一个枚举类 BackendComponent，底层类型为 uint8_t
enum class BackendComponent : uint8_t {

  // A "backend" is colloquially used to refer to handlers for dispatch
  // which actually implement the numerics of an operation in question.
  // 备用机制通常用于指代实现操作数学运算的调度处理程序。

  // Due to the nature of the enum, these backends are specified in
  // an ordered way, but for most backends this order is not semantically
  // meaningful (e.g., it's valid to reorder these backends without changing
  // semantics).  The only situation when backend ordering is meaningful
  // is when the backend participates in multiple dispatch with another
  // backend; e.g., CPU and CUDA (cuda must have higher priority).
  // 由于枚举的性质，这些后端以有序的方式指定，但对于大多数后端来说，这个顺序并没有语义上的意义（例如，重新排序这些后端而不改变语义是有效的）。唯一有意义的情况是，当后端与另一个后端参与多次分派时，后端的顺序才有意义；例如，CPU 和 CUDA（CUDA 必须具有更高的优先级）。

  // These keys don't correspond to individual kernels.
  // Instead, they represent the backends that are allowed to override specific
  // pieces of functionality:
  // - dense kernels (e.g. DispatchKey::CPU)
  // - sparse kernels (e.g. DispatchKey::SparseCPU)
  // - quantized kernels (e.g. DispatchKey::QuantizedCPU)
  // - autograd kernels (e.g. DispatchKey::AutogradCPU)
  // We reserve space in the runtime operator table for this full cross product
  // of
  // [backends in this enum] x [keys below that are explicitly marked as having
  // per-backend functionality]
  // 这些键不对应于单独的内核。它们代表允许覆盖特定功能的后端：
  // - 密集内核（例如 DispatchKey::CPU）
  // - 稀疏内核（例如 DispatchKey::SparseCPU）
  // - 量化内核（例如 DispatchKey::QuantizedCPU）
  // - 自动微分内核（例如 DispatchKey::AutogradCPU）
  // 我们在运行时操作表中为这个枚举和下面标记为具有每个后端功能的键的交叉乘积预留空间。

  // A meta tensor is a tensor without any data associated with it.  (They
  // have also colloquially been referred to as tensors on the "null" device).
  // A meta tensor can be used to dry run operators without actually doing any
  // computation, e.g., add on two meta tensors would give you another meta
  // tensor with the output shape and dtype, but wouldn't actually add anything.
  // 元张量是一种没有与之关联数据的张量（它们也俗称为“null”设备上的张量）。元张量可用于在不进行任何计算的情况下运行操作符，例如，对两个元张量进行加法运算将给出具有输出形状和数据类型的另一个元张量，但实际上不会添加任何内容。

  InvalidBit = 0,  // 无效位，值为 0

#define DEFINE_BACKEND_COMPONENT(n, _) n##Bit,
  C10_FORALL_BACKEND_COMPONENTS(DEFINE_BACKEND_COMPONENT, unused)
#undef DEFINE_BACKEND_COMPONENT

  // Define an alias to represent end of backend dispatch keys.
  // If you add new backend keys after PrivateUse3, please also update it here.
  EndOfBackendKeys = MetaBit,  // 定义一个别名来表示后端调度键的末尾，值为 MetaBit
};

// Semantically, a dispatch key identifies a possible "level" in our
// dispatch, for which a handler may be registered. Each handler corresponds
// to a type of functionality.
// 在语义上，调度键标识我们调度中可能的“级别”，可以为其注册处理程序。每个处理程序对应于一种功能类型。

// In implementation terms, the dispatch key identifies a specific "bit" in a
// DispatchKeySet.  Higher bit indexes get handled by dispatching first (because
// we "count leading zeros" when we extract the highest priority dispatch
// key.)
// 在实现术语中，调度键标识调度键集合中的特定“位”。更高的位索引优先进行分派（因为在提取最高优先级调度键时“计算领先零位”）。

// Note [DispatchKey Classification]
// This enum actually contains several types of keys, which are explained
// in more detail further down:
// (1) non-customizable backends (e.g. FPGA)
// (2) non-customizable functionalities (e.g. Functionalize)
// (3) functionalized that are customizable per backend (e.g. Dense, Sparse,
// AutogradFunctionality) (4) per-backend instances of customizable
// functionalities (e.g. CPU, SparseCPU, AutogradCPU) (5) alias keys (e.g.
// CompositeImplicitAutograd)
// 注释 [DispatchKey Classification]
// 此枚举实际上包含几种类型的键，后文将更详细地解释：
// (1) 非定制后端（例如 FPGA）
// (2) 非定制功能（例如 Functionalize）
// (3) 每个后端可定制的功能化（例如 Dense、Sparse、AutogradFunctionality）
// (4) 可定制功能的每个后端实例（例如 CPU、SparseCPU、AutogradCPU）
// (5) 别名键（例如 CompositeImplicitAutograd）

// Of the categories above, it's important to note:
// (a) which keys are assigned individual bits in a DispatchKeySet
// 在上述类别中，重要的是注意：
// (a) 哪些键在 DispatchKeySet 中被分配了独立的位数
// (b) which keys are assigned individual slots in the runtime operator table
// ("Runtime keys")
//
// (1), (2) and (3) all get their own dedicated bits in the DispatchKeySet.
// (1), (2) and (4) all get their own dedicated slots in the runtime operator
// table.

// See Note [DispatchKeySet Internal Representation] for more details.
//
// NOTE: Keep the list in sync with `DispatchKey` in torchgen/model.py
// ~~~~~~~~~~~~~~ "Dense" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~~~~ //
// Here are backends which you think of as traditionally specifying
// how to implement operations on some device.

#define DEFINE_PER_BACKEND_KEYS_FOR_BACKEND(n, prefix) prefix##n,

#define DEFINE_PER_BACKEND_KEYS(fullname, prefix)      \
  StartOf##fullname##Backends,                         \
      C10_FORALL_BACKEND_COMPONENTS(                   \
          DEFINE_PER_BACKEND_KEYS_FOR_BACKEND, prefix) \
          EndOf##fullname##Backends = prefix##Meta,

  C10_FORALL_FUNCTIONALITY_KEYS(DEFINE_PER_BACKEND_KEYS)

#undef DEFINE_PER_BACKEND_KEYS
// 定义宏 `DEFINE_PER_BACKEND_KEYS_FOR_BACKEND` 并取消定义

EndOfRuntimeBackendKeys = EndOfAutogradFunctionalityBackends,

// ~~~~~~~~~~~~~~~~~~~~~~ 别名调度键 ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// 注意 [别名调度键]
// 别名调度键是合成的调度键，映射到多个运行时调度键。别名键具有优先级，但始终低于运行时键。
// 可以向别名键注册一个内核，该内核可能会在调度表计算期间填充到映射的运行时键中。
// 如果一个运行时调度键有多个来自别名键的内核，内核的胜出是基于别名键的优先级（但运行时键始终优先于别名键）。
// 别名键在运行时不会直接调用。

// 参见 注意 [别名调度键 : Autograd]
Autograd,
CompositeImplicitAutograd, // 注册在 build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp

// 注意：FuncTorchBatchedDecomposition 的别名键集合与所有其他别名键集合不相交，因此优先顺序无关紧要
FuncTorchBatchedDecomposition, // 注册在 build/aten/src/ATen/RegisterFuncTorchBatchedDecomposition.cpp

// 注意：CompositeImplicitAutogradNestedTensor 的别名键集合与所有其他别名键集合不相交
CompositeImplicitAutogradNestedTensor, // 注册在 build/aten/src/ATen/RegisterCompositeImplicitAutogradNestedTensor.cpp
CompositeExplicitAutograd, // 注册在 build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp

// 参见 注意 [CompositeExplicitAutogradNonFunctional Key]
CompositeExplicitAutogradNonFunctional, // 注册在 build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp

// 定义一个别名键来表示别名调度键的结束位置。
// 如果在 Autograd 后添加新的别名键，请在此处更新
StartOfAliasKeys = Autograd,
EndOfAliasKeys = CompositeExplicitAutogradNonFunctional, //

// ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// 这些别名是为了向后兼容而存在的，不应该使用
CPUTensorId = CPU,
CUDATensorId = CUDA,
DefaultBackend = CompositeExplicitAutograd,
PrivateUse1_PreAutograd = AutogradPrivateUse1,
PrivateUse2_PreAutograd = AutogradPrivateUse2,
PrivateUse3_PreAutograd = AutogradPrivateUse3,
Autocast = AutocastCUDA,
};

// 注意 [私有使用调度键]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 私有使用张量 ID 是预分配的张量类型 ID，供用户应用程序使用。
// 类似于 HTTP 中的私有使用字段，它们可以由最终用户用于实验或私有应用程序，而无需需要“标准化”张量 ID（可以通过提交 PR 到 PyTorch 添加您的类型 ID 来完成标准化）。
//
// 如果您希望进行实验，则适合使用私有使用张量 ID
// 确保所有的 BackendComponent 和 DispatchKey 枚举值加起来不超过 64，
// 因为它们映射到一个 64 位的位掩码中，用于表示后端和功能类型。
static_assert(
    (static_cast<uint8_t>(BackendComponent::EndOfBackendKeys) +
     static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys)) <= 64,
    "The BackendComponent and DispatchKey enums (below EndOfFunctionalityKeys)"
    " both map to backend and functionality bits"
    " into a 64-bit bitmask; you must have less than 64 total entries between them");

// 检查一个 DispatchKey 是否是一个别名映射到其他运行时键的 DispatchKey。
constexpr bool isAliasDispatchKey(DispatchKey k) {
  return k >= DispatchKey::StartOfAliasKeys && k <= DispatchKey::EndOfAliasKeys;
}

// [注：每个后端功能性 DispatchKey]
// 检查一个 DispatchKey 是否是一个每个后端功能性的键。
// 任何可以根据后端定制的功能应该在这里添加。
// 这些键对应于可以单独根据后端定制的功能。
// 尽管它们在 `DispatchKeySet` 位集中只占用一个位，但它们映射到操作表中的 (# backends) 个槽位。
// 每个这样的键在 dispatch key 枚举中都有一个单独的运行时键，用于映射到各自的操作表槽位。
// 例如，"Sparse" 键映射到 DispatchKeySet 中的一个单独位，
// 而 `SparseCPU`、`SparseCUDA` 等则映射到运行时操作表中的单独槽位。
constexpr bool isPerBackendFunctionalityKey(DispatchKey k) {
  if (k == DispatchKey::Dense || k == DispatchKey::Quantized ||
      k == DispatchKey::Sparse || k == DispatchKey::SparseCsr ||
      k == DispatchKey::AutogradFunctionality ||
      k == DispatchKey::NestedTensor) {
    return true;
  } else {
    return false;
  }
}

// 注意这里包括 Undefined 在内的总计数。
// 但 EndOfFunctionalityKeys 是它自己的（占位符）键。
// 例如 Undefined=0, Dense=1, Sparse=2, EndOfFunctionalityKeys=3。
// 在上面的例子中，总共有 3 个功能性键。
constexpr uint8_t num_functionality_keys =
    static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);
// 定义一个常量，表示后端组件的数量，使用枚举值的末尾作为值
constexpr uint8_t num_backends =
    static_cast<uint8_t>(BackendComponent::EndOfBackendKeys);

// 注意事项 [不超过16个后端]
// 在代码中搜索此注释可以找到"不超过16个后端"不变式的具体位置。
static_assert(
    static_cast<uint8_t>(BackendComponent::EndOfBackendKeys) <= 16,
    "BackendComponent目前仅支持不超过16个后端。如果确实需要扩展，有几个地方在这个不变式中是硬编码的");

// 定义一个constexpr函数，计算每个后端功能键的数量
constexpr uint8_t numPerBackendFunctionalityKeys() {
  uint8_t count = 0;
  for (uint8_t k = 0; k <= num_functionality_keys; ++k) {
    if (isPerBackendFunctionalityKey(static_cast<DispatchKey>(k)))
      ++count;
  }
  return count;
}

#if defined(C10_MOBILE_TRIM_DISPATCH_KEYS)
// 参见注释 [注意：精简移动调度键]
constexpr uint16_t num_runtime_entries = 8;
#else
// 根据后端功能键数量和后端数量计算运行时条目的数量
constexpr uint16_t num_runtime_entries = num_functionality_keys +
    (numPerBackendFunctionalityKeys() * (num_backends - 1));
#endif

// 参见注意事项 [不超过16个后端]
// 定义一个掩码，用于标识所有后端的位模式
constexpr uint16_t full_backend_mask =
    (static_cast<uint16_t>(1) << num_backends) - 1;

// 定义C10_API接口函数的声明
C10_API const char* toString(DispatchKey);
C10_API const char* toString(BackendComponent);
C10_API std::ostream& operator<<(std::ostream&, DispatchKey);
C10_API std::ostream& operator<<(std::ostream&, BackendComponent);

// 根据后端组件获取自动微分键的函数声明
C10_API DispatchKey getAutogradKeyFromBackend(BackendComponent k);

// 解析字符串为调度键的函数声明
// 如果无法正确解析，会抛出异常
C10_API c10::DispatchKey parseDispatchKey(const std::string& k);

// 这些是一些便利的调度键标识符，比它们的长名称更短且更易于输入。
// 注意，其中一些调度键直接对应DeviceType；大多数接受DispatchKey的API也接受DeviceType；例如，
// torch::dispatch(torch::kCPU, ...)也是有效的。
constexpr DispatchKey kAutograd = DispatchKey::Autograd;

// 参见注释 [每个后端调度键的排序很重要！]
// 此函数依赖于这样的不变式：在StartOfDenseBackends和EndOfRuntimeBackendKeys之间的调度键按照BackendComponent的相同顺序排序。
constexpr BackendComponent toBackendComponent(DispatchKey k) {
  if (k >= DispatchKey::StartOfDenseBackends &&
      k <= DispatchKey::EndOfDenseBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfDenseBackends));
  } else if (
      k >= DispatchKey::StartOfQuantizedBackends &&
      k <= DispatchKey::EndOfQuantizedBackends) {
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfQuantizedBackends));
  } else if (
      k >= DispatchKey::StartOfSparseBackends &&
      k <= DispatchKey::EndOfSparseBackends) {
    // 如果给定的 k 值在稀疏后端范围内
    return static_cast<BackendComponent>(
        static_cast<uint8_t>(k) -
        static_cast<uint8_t>(DispatchKey::StartOfSparseBackends));
    // 否则，如果 k 值在稀疏 CSR 后端范围内
    } else if (
        k >= DispatchKey::StartOfSparseCsrBackends &&
        k <= DispatchKey::EndOfSparseCsrBackends) {
      // 返回对应的稀疏 CSR 后端组件
      return static_cast<BackendComponent>(
          static_cast<uint8_t>(k) -
          static_cast<uint8_t>(DispatchKey::StartOfSparseCsrBackends));
    // 否则，如果 k 值在嵌套张量后端范围内
    } else if (
        k >= DispatchKey::StartOfNestedTensorBackends &&
        k <= DispatchKey::EndOfNestedTensorBackends) {
      // 返回对应的嵌套张量后端组件
      return static_cast<BackendComponent>(
          static_cast<uint8_t>(k) -
          static_cast<uint8_t>(DispatchKey::StartOfNestedTensorBackends));
    // 否则，如果 k 值在自动微分功能后端范围内
    } else if (
        k >= DispatchKey::StartOfAutogradFunctionalityBackends &&
        k <= DispatchKey::EndOfAutogradFunctionalityBackends) {
      // 返回对应的自动微分功能后端组件
      return static_cast<BackendComponent>(
          static_cast<uint8_t>(k) -
          static_cast<uint8_t>(
              DispatchKey::StartOfAutogradFunctionalityBackends));
    // 若 k 值不在以上任何范围内，则返回无效的后端组件
    } else {
      return BackendComponent::InvalidBit;
    }
}

// 将给定的 DispatchKey 转换为对应的功能性 DispatchKey
constexpr DispatchKey toFunctionalityKey(DispatchKey k) {
  // 如果给定的 DispatchKey 在功能性 DispatchKey 的范围内，直接返回
  if (k <= DispatchKey::EndOfFunctionalityKeys) {
    return k;
  // 如果给定的 DispatchKey 在 Dense 后端范围内，返回 DispatchKey::Dense
  } else if (k <= DispatchKey::EndOfDenseBackends) {
    return DispatchKey::Dense;
  // 如果给定的 DispatchKey 在 Quantized 后端范围内，返回 DispatchKey::Quantized
  } else if (k <= DispatchKey::EndOfQuantizedBackends) {
    return DispatchKey::Quantized;
  // 如果给定的 DispatchKey 在 Sparse 后端范围内，返回 DispatchKey::Sparse
  } else if (k <= DispatchKey::EndOfSparseBackends) {
    return DispatchKey::Sparse;
  // 如果给定的 DispatchKey 在 SparseCsr 后端范围内，返回 DispatchKey::SparseCsr
  } else if (k <= DispatchKey::EndOfSparseCsrBackends) {
    return DispatchKey::SparseCsr;
  // 如果给定的 DispatchKey 在 NestedTensor 后端范围内，返回 DispatchKey::NestedTensor
  } else if (k <= DispatchKey::EndOfNestedTensorBackends) {
    return DispatchKey::NestedTensor;
  // 如果给定的 DispatchKey 在 AutogradFunctionality 后端范围内，返回 DispatchKey::AutogradFunctionality
  } else if (k <= DispatchKey::EndOfAutogradFunctionalityBackends) {
    return DispatchKey::AutogradFunctionality;
  // 其它情况返回 DispatchKey::Undefined
  } else {
    return DispatchKey::Undefined;
  }
}

// 将功能性 DispatchKey 和后端组件转换为运行时后端功能性 DispatchKey
// 当 DispatchKey::Dense 和 BackendComponent::CUDABit 时，返回 DispatchKey::CUDA
// 见注释 "The Ordering of Per-Backend Dispatch Keys Matters!" 
constexpr DispatchKey toRuntimePerBackendFunctionalityKey(
    DispatchKey functionality_k,
    BackendComponent backend_k) {
  // 根据功能性 DispatchKey 返回对应的运行时后端功能性 DispatchKey
  if (functionality_k == DispatchKey::Dense) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfDenseBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::Sparse) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfSparseBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::SparseCsr) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfSparseCsrBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::Quantized) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfQuantizedBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::NestedTensor) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfNestedTensorBackends) +
        static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::AutogradFunctionality) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(
            DispatchKey::StartOfAutogradFunctionalityBackends) +
        static_cast<uint8_t>(backend_k));
  }
  // 默认情况返回 DispatchKey::Undefined
  return DispatchKey::Undefined;
}

} // namespace c10

namespace torch {
// 公开常量 kAutograd，但不公开类型（DispatchKey 是实现细节！）
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::kAutograd;
} // namespace torch

// 注意：你确实不应该使用此实例；这个枚举保证非常小，因此常规数组应该是可以接受的。
namespace std {
template <>
# 定义一个结构体 hash，用于将 c10::DispatchKey 类型的对象映射为 size_t 类型的哈希值
struct hash<c10::DispatchKey> {
  # result_type 是哈希函数的返回类型，这里是 size_t
  typedef size_t result_type;
  # argument_type 是哈希函数的参数类型，这里是 c10::DispatchKey
  typedef c10::DispatchKey argument_type;

  # 哈希函数的重载运算符，将 c10::DispatchKey 转换为其对应的 size_t 值作为哈希结果
  size_t operator()(c10::DispatchKey x) const {
    return static_cast<size_t>(x);
  }
};
# 结束 std 命名空间
} // namespace std
```