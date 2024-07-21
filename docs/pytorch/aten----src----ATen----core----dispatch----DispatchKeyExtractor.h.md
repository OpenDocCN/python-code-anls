# `.\pytorch\aten\src\ATen\core\dispatch\DispatchKeyExtractor.h`

```
#pragma once

#include <cstdint>  // 包含标准整数类型定义
#include <ATen/core/function_schema.h>  // 包含 ATen 函数模式的头文件
#include <ATen/core/jit_type.h>  // 包含 ATen JIT 类型的头文件
#include <c10/util/Bitset.h>  // 包含 c10 Bitset 的头文件
#include <c10/core/DispatchKeySet.h>  // 包含 c10 DispatchKeySet 的头文件
#include <c10/util/irange.h>  // 包含 c10 irange 的头文件
#include <ATen/core/Variadic.h>  // 包含 ATen 可变参数的头文件
#include <ATen/core/stack.h>  // 包含 ATen 栈操作的头文件

namespace c10 {

namespace impl {

// 获取给定 Tensor 的 DispatchKeySet，并确定实际的 DispatchKey。
// 考虑 TLS，并跳过通过的后端。
//
// 与 Tensor::key_set() 不同，此函数的返回值可能受 TLS 影响而变化。
//
// 注意：如果没有有效的 DispatchKey，将返回 Undefined。
inline DispatchKeySet computeDispatchKeySet(
    DispatchKeySet ks,
    // key_mask 用于排除不应考虑的键。有两种情况会使用它：
    //
    // - 如果操作符的分发表包含 fallthrough 条目，则在查找键时应完全跳过它
    // - 如果用户使用 redispatch 调用，此 mask 允许我们将用户要求停止的键置为零
    //
    // 这些排除的后端不受 TLS 跟踪，但必须在 TLS 之后应用（因为后端可能已经通过包含的 TLS 而被引入考虑），
    // 这就是为什么你必须将它们传递给此函数（而不仅应用于输入的 'ks'）。
    DispatchKeySet key_mask
) {
  // 获取当前线程局部存储中的 DispatchKeySet
  c10::impl::LocalDispatchKeySet local = c10::impl::tls_local_dispatch_key_set();
  // TODO: 这里需要进行逻辑或运算，稍显繁琐。希望只需进行一次操作。
  // 是否总是包含可以折叠到 TLS 中？这有点麻烦，因为快速路径 TLS 访问要求零初始化 TLS 类型，
  // 所以在这种情况下实际上不会获得任何优势。
  return (((ks | local.included_) - local.excluded_) & key_mask);
}

}

namespace detail {
  // 从已知具有 DispatchKeySet 的类型中提取 DispatchKeySet 的小工具。
  // 用于从未装箱调用中提取分派键。
  struct MultiDispatchKeySet : at::IterArgs<MultiDispatchKeySet> {
    DispatchKeySet ts;
    void operator()(const at::Tensor& x) {
      ts = ts | x.key_set();  // 获取 Tensor x 的 DispatchKeySet 并合并到 ts 中
    }
    void operator()(const std::optional<at::Tensor>& x) {
      if (x.has_value()) {
        ts = ts | x->key_set();  // 如果 x 有值，则获取其 DispatchKeySet 并合并到 ts 中
      }
    }
    void operator()(at::ArrayRef<at::Tensor> xs) {
      for (const auto& x : xs) {
        ts = ts | x.key_set();  // 遍历 Tensor 数组 xs，获取每个 Tensor 的 DispatchKeySet 并合并到 ts 中
      }
    }
    // Tensor?[] 对应这种情况
    void operator()(const c10::List<std::optional<at::Tensor>>& xs) {
      for (std::optional<at::Tensor> x : xs) {
        if (x.has_value()) {
          ts = ts | x.value().key_set();  // 遍历 Tensor 可选列表 xs，如果 x 有值，则获取其 DispatchKeySet 并合并到 ts 中
        }
      }
    }
    // 结构化的 Tensor[] 对应这种情况
    void operator()(const at::ITensorListRef& xs) {
      for (const auto& x : xs) {
        ts = ts | x.key_set();  // 遍历 ITensorListRef xs，获取每个 Tensor 的 DispatchKeySet 并合并到 ts 中
      }
    }
    [[noreturn]] void operator()(at::ArrayRef<std::optional<at::Tensor>>) {
      // 核验对 Tensor?[] 的处理未变
      // 当执行到这里时，断言应该失败，因为这个函数被声明为不返回的
      TORCH_INTERNAL_ASSERT(false);
    }
    
    void operator()(const at::Generator& gen) {
      // 如果生成器对象已定义
      if (gen.defined()) {
        // 将生成器的键集合合并到当前对象的键集合中
        ts = ts | gen.key_set();
      }
    }
    
    void operator()(const std::optional<at::Generator>& gen) {
      // 如果 std::optional 包含值并且生成器对象已定义
      if (gen.has_value() && gen->defined()) {
        // 将生成器的键集合合并到当前对象的键集合中
        ts = ts | gen->key_set();
      }
    }
    
    template <typename T>
    void operator()(const T&) {
      // 什么都不做
      // 通用的模板，用于处理除上述特定类型以外的所有类型，不执行任何操作
    }
    
    // 注意：通过常量引用传递参数（不要在这里使用通用转发！你不希望将参数移动进入这个函数！）
    template <typename... Args>
    DispatchKeySet multi_dispatch_key_set(const Args&... args) {
      // 创建 MultiDispatchKeySet 对象，并对参数列表应用操作符 ()，返回其包含的键集合
      return MultiDispatchKeySet().apply(args...).ts;
    }
}

/**
 * An instance of DispatchKeyExtractor knows how to get a dispatch key given
 * a list of arguments for an operator call.
 *
 * The instance is specific for a certain operator as:
 *  - In boxed dispatch, different operators have different ways to extract
 *    the dispatch key (e.g. different numbers of arguments), and we precompute
 *    the stack locations we should look at; and
 *  - In all dispatch, some backends should be excluded from dispatch because
 *    they have been registered as fallthrough.  The set of excluded backends
 *    varies from operator, as some operators may have overridden the
 *    fallthrough with custom behavior.
 *
 *   Note - this should maintain identical impl to the py dispatcher key extraction logic
 *   at pytorch/torch/dispatcher.py
 */
struct TORCH_API DispatchKeyExtractor final {
public:
  /**
   * Create a DispatchKeyExtractor instance for a given FunctionSchema.
   *
   * @param schema The FunctionSchema for which dispatch key extraction is needed.
   * @return DispatchKeyExtractor The constructed instance.
   */
  static DispatchKeyExtractor make(const FunctionSchema& schema) {
    return DispatchKeyExtractor(makeBitsetForDispatchArgs(schema));
  }

  /**
   * Create an uninitialized DispatchKeyExtractor instance.
   *
   * @return DispatchKeyExtractor The uninitialized instance.
   */
  static DispatchKeyExtractor makeUninitialized() {
    return DispatchKeyExtractor(c10::utils::bitset());
  }

  /**
   * Register a FunctionSchema to set up dispatch key extraction.
   *
   * @param schema The FunctionSchema to register.
   */
  void registerSchema(const FunctionSchema& schema) {
    TORCH_INTERNAL_ASSERT(dispatch_arg_indices_reverse_.is_entirely_unset());
    dispatch_arg_indices_reverse_ = makeBitsetForDispatchArgs(schema);
  }

  /**
   * Deregister the current schema, resetting dispatch key extraction state.
   */
  void deregisterSchema() {
    dispatch_arg_indices_reverse_ = c10::utils::bitset();
  }

  /**
   * Get the DispatchKeySet for boxed dispatch using the provided stack.
   *
   * @param stack Pointer to the Torch stack containing arguments.
   * @return DispatchKeySet The computed dispatch key set.
   */
  DispatchKeySet getDispatchKeySetBoxed(const torch::jit::Stack* stack) const {
    DispatchKeySet ks;
    dispatch_arg_indices_reverse_.for_each_set_bit([&] (size_t reverse_arg_index) {
      const auto& ivalue = torch::jit::peek(*stack, 0, reverse_arg_index + 1);
      if (C10_LIKELY(ivalue.isTensor())) {
        // NB: Take care not to introduce a refcount bump (there's
        // no safe toTensorRef method, alas)
        ks = ks | ivalue.unsafeToTensorImpl()->key_set();
      } else if (C10_UNLIKELY(ivalue.isTensorList())) {
        for (const at::Tensor& tensor : ivalue.toTensorList()) {
          ks = ks | tensor.key_set();
        }
      }
      // Tensor?[] translates to a c10::List<IValue> so we need to peek inside
      else if (C10_UNLIKELY(ivalue.isList())) {
        for (const auto& elt : ivalue.toListRef()) {
          if (elt.isTensor()) {
            ks = ks | elt.toTensor().key_set();
          }
        }
      }
    });
    // Keys that are fallthrough should be skipped
    if (requiresBitsetPerBackend_) {
      auto backend_idx = ks.getBackendIndex();
      return impl::computeDispatchKeySet(ks, nonFallthroughKeysPerBackend_[backend_idx]);
    } else {
      return impl::computeDispatchKeySet(ks, nonFallthroughKeys_);
    }
  }

  /**
   * Get the DispatchKeySet for unboxed dispatch using variadic arguments.
   *
   * @tparam Args Variadic template parameters representing arguments.
   * @param args Arguments for which dispatch key set is computed.
   * @return DispatchKeySet The computed dispatch key set.
   */
  template<class... Args>
  DispatchKeySet getDispatchKeySetUnboxed(const Args&... args) const {
    auto ks = detail::multi_dispatch_key_set(args...);
    // Keys that are fallthrough should be skipped
    // 如果 requiresBitsetPerBackend_ 为真，则执行以下逻辑
    if (requiresBitsetPerBackend_) {
      // 获取当前的后端索引
      auto backend_idx = ks.getBackendIndex();
      // 根据指定后端索引从 nonFallthroughKeysPerBackend_ 中获取对应的非 fallthrough keys，并计算调度键集合
      return impl::computeDispatchKeySet(ks, nonFallthroughKeysPerBackend_[backend_idx]);
    } else {
      // 如果 requiresBitsetPerBackend_ 为假，则计算使用全局的 nonFallthroughKeys_ 计算调度键集合
      return impl::computeDispatchKeySet(ks, nonFallthroughKeys_);
    }
  }

  // 设置指定调度键 k 是否有 fallthrough
  void setOperatorHasFallthroughForKey(DispatchKey k, bool has_fallthrough);

  // 返回当前对象的状态信息的字符串表示
  std::string dumpState() const;
  // 检查当前对象的不变量是否满足给定的函数模式 schema
  void checkInvariants(const FunctionSchema& schema) const;
private:
  // 创建一个位集合，用于存储需要分派的参数索引
  static c10::utils::bitset makeBitsetForDispatchArgs(const FunctionSchema& schema) {
    // 检查函数模式的参数数量是否不超过位集合的位数
    TORCH_CHECK(schema.arguments().size() <= c10::utils::bitset::NUM_BITS(),
        "The function schema has ", schema.arguments().size(),
        " arguments but this PyTorch build only supports ", c10::utils::bitset::NUM_BITS());
    
    // 创建一个位集合，用于存储反向的分派参数索引
    c10::utils::bitset dispatch_arg_indices_reverse;
    
    // 遍历函数模式的每个参数索引
    for (const auto index : c10::irange(schema.arguments().size())) {
      // 如果参数类型是张量类型或张量列表类型之一，设置对应位为真
      if (schema.arguments()[index].type()->isSubtypeOf(*TensorType::get()) ||
          schema.arguments()[index].type()->isSubtypeOf(
              *ListType::ofTensors()) ||
          schema.arguments()[index].type()->isSubtypeOf(
              *ListType::ofOptionalTensors()) ||
          schema.arguments()[index].type()->isSubtypeOf(
              *OptionalType::ofTensor())) {
        dispatch_arg_indices_reverse.set(schema.arguments().size() - 1 - index);
      }
    }
    
    // 返回反向分派参数索引的位集合
    return dispatch_arg_indices_reverse;
  }

  // 显式构造函数，初始化成员变量
  explicit DispatchKeyExtractor(c10::utils::bitset dispatch_arg_indices_reverse)
  : dispatch_arg_indices_reverse_(dispatch_arg_indices_reverse)
  , nonFallthroughKeys_(DispatchKeySet::FULL)  // 设置非跨越内核的功能键集合为全集
  , requiresBitsetPerBackend_(false) {  // 初始化是否需要每个后端位集合的标志为假
    // 遍历非跨越内核的功能键集合数组，设置每个元素为全集
    for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size())) {
      nonFallthroughKeysPerBackend_[i] = DispatchKeySet::FULL;
    }
  }

  // 表示需要进行分派的参数索引的位集合
  c10::utils::bitset dispatch_arg_indices_reverse_;

  // 一组不支持跨越内核的功能键集合
  DispatchKeySet nonFallthroughKeys_;

  // 每个后端定义的不支持跨越内核的功能键集合数组
  // 只有在运算符对某些后端有不同的跨越内核定义时才需要这些集合
  std::array<DispatchKeySet, num_backends> nonFallthroughKeysPerBackend_;

  // 标志，告知是否可以使用单一的nonFallthroughKeys_集合（快速路径）
  // 或者是否需要回退到更慢的路径并检查nonFallthroughKeysPerBackend_
  bool requiresBitsetPerBackend_;
};
```