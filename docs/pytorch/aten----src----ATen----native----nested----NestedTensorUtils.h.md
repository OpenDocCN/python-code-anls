# `.\pytorch\aten\src\ATen\native\nested\NestedTensorUtils.h`

```
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/ones_native.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/stack_native.h>
#include <ATen/ops/tensor.h>
#endif

#include <utility>
#include <vector>

namespace at {
namespace native {
struct NestedTensorImpl;

// The following functions are used to construct nested tensors from buffers and
// metadata.

// 包装给定的 buffer 和 nested_sizes 以创建一个嵌套张量
inline at::Tensor wrap_buffer(at::Tensor buffer, at::Tensor nested_sizes) {
  // 检查 buffer 必须是一维的
  TORCH_CHECK(
      buffer.dim() == 1,
      "Expected given buffer to be 1dim, but got ",
      buffer.dim(),
      " instead.");
  // 检查 buffer 必须是连续的
  TORCH_CHECK(
      buffer.is_contiguous(), "Expected given buffer to be contiguous.");
  // 调用 make_tensor 创建 NestedTensorImpl 的张量
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), std::move(nested_sizes));
}

// TODO: Figure out if we need a non-moving wrap_buffer()
// 包装给定的 buffer、nested_sizes、nested_strides 和 storage_offsets 以创建嵌套张量
inline at::Tensor wrap_buffer(
    at::Tensor buffer,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets) {
  // 调试模式下检查 buffer 必须是连续的
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer.is_contiguous(), "Given buffer must be contiguous.");
  // 调用 make_tensor 创建 NestedTensorImpl 的张量
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer),
      std::move(nested_sizes),
      std::move(nested_strides),
      std::move(storage_offsets));
}

// 获取给定张量的 buffer，用于嵌套张量
inline at::Tensor get_buffer(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_buffer();
}

/**
 * Create a new nested tensor that is a view of a base nested tensor
 *
 * create_view_tensor calls a specialized constructor that copys the
 * the keys from base onto the new view tensor being created.
 * The storage is shared between the base and the returned view tensor
 *
 * All callers of this helper must:
 * - Only return a view of the input
 * - Must be explicit and define a derivative
 *
 * @param base Base tensor to construct view from.
 * @param nested_sizes View tensors' sizes.
 * @param nested_strides View tensors' strides.
 * @param storage_offsets View tensors' offsets.
 * @return A newly constructed view tensor
 */
inline at::Tensor create_nested_view_tensor(
    const at::Tensor& base,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets) {
  // 调用 make_tensor 创建 NestedTensorImpl 的视图张量
  // 共享 base 和返回的视图张量之间的存储
  return at::detail::make_tensor<NestedTensorImpl>(
      base,
      std::move(nested_sizes),
      std::move(nested_strides),
      std::move(storage_offsets));
}

} // namespace native
} // namespace at
    // 断言确保基本张量 `base` 是嵌套的，即它是一个嵌套张量视图
    TORCH_INTERNAL_ASSERT(
        base.is_nested(),
        "This function can only be used to create nested tensor views");
    
    // 断言确保当前没有设置自动求导功能的分发键，因为在 CompositeImplicit 函数中创建非可微嵌套张量视图是不允许的
    TORCH_INTERNAL_ASSERT(
        c10::impl::tls_local_dispatch_key_set().excluded_.has(
            c10::DispatchKey::AutogradFunctionality),
        "Creating a non differentiable nested tensor view in a CompositeImplicit function is not allowed.");
    
    // 调用 make_tensor 函数创建一个嵌套张量视图，并返回其对应的 TensorImpl 对象
    return at::detail::make_tensor<NestedTensorImpl>(
        c10::TensorImpl::VIEW,   // 使用 VIEW 类型创建张量实现
        base,                    // 基础张量
        nested_sizes,            // 嵌套张量视图的大小
        nested_strides,          // 嵌套张量视图的步长
        storage_offsets);        // 嵌套张量视图的存储偏移量
// Helper functions for getting information about a nested tensor's shape.

// 获取嵌套张量最后一个维度的一致大小
int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt);

// 获取嵌套张量的尺寸
inline std::vector<IntArrayRef> NestedTensor_get_sizes(
    const NestedTensorImpl* self_ptr) {
  // 获取嵌套张量中包含的张量数量
  int64_t ntensors = self_ptr->size(0);
  std::vector<IntArrayRef> sizes(ntensors);
  // 如果嵌套张量为空，则直接返回空尺寸向量
  if (ntensors == 0) {
    return sizes;
  }
  // 获取嵌套尺寸矩阵
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  // 获取原始维度大小
  int64_t orig_dim = sizemat.size(1);
  // 如果原始维度为0，表示嵌套标量，也返回空尺寸向量
  if (orig_dim == 0) {
    return sizes;
  }
  // 获取尺寸矩阵数据指针
  const int64_t* sizemat_ptr = sizemat.const_data_ptr<int64_t>();

  // 遍历每个张量，从尺寸矩阵中获取对应的尺寸信息
  for (const auto i : c10::irange(ntensors)) {
    sizes[i] = IntArrayRef(sizemat_ptr, sizemat_ptr + orig_dim);
    sizemat_ptr += orig_dim;
  }
  return sizes;
}

// 获取嵌套张量的最大尺寸
TORCH_API std::vector<int64_t> NestedTensor_get_max_size(
    const NestedTensorImpl& nt);

// 从尺寸张量获取嵌套张量的最大尺寸
std::vector<int64_t> NestedTensor_get_max_size_from_size_tensor(
    const Tensor& sizes);

// 获取嵌套张量的尺寸，通过转发到具体的实现
inline std::vector<IntArrayRef> NestedTensor_get_sizes(const at::Tensor& self) {
  const NestedTensorImpl* self_ptr = get_nested_tensor_impl(self);
  return NestedTensor_get_sizes(self_ptr);
}

// 获取嵌套张量的步幅
inline std::vector<IntArrayRef> NestedTensor_get_strides(
    const NestedTensorImpl* self_ptr) {
  // 获取嵌套张量中包含的张量数量
  int64_t ntensors = self_ptr->size(0);
  std::vector<IntArrayRef> strides(ntensors);
  // 如果嵌套张量为空，则直接返回空步幅向量
  if (ntensors == 0) {
    return strides;
  }
  // 获取嵌套步幅矩阵
  const Tensor& stridemat = self_ptr->get_nested_strides();
  // 获取原始维度大小
  int64_t orig_dim = stridemat.size(1);
  // 如果原始维度为0，表示嵌套标量，也返回空步幅向量
  if (orig_dim == 0) {
    return strides;
  }
  // 获取步幅矩阵数据指针
  const int64_t* stridemat_ptr = stridemat.const_data_ptr<int64_t>();
  // 遍历每个张量，从步幅矩阵中获取对应的步幅信息
  for (const auto i : c10::irange(ntensors)) {
    strides[i] = IntArrayRef(stridemat_ptr, stridemat_ptr + orig_dim);
    stridemat_ptr += orig_dim;
  }
  return strides;
}

// 获取嵌套张量的步幅，通过转发到具体的实现
inline std::vector<IntArrayRef> NestedTensor_get_strides(
    const at::Tensor& self) {
  const NestedTensorImpl* self_ptr = get_nested_tensor_impl(self);
  return NestedTensor_get_strides(self_ptr);
}

// 检查嵌套张量的元素数量与缓冲区大小是否相等
inline void check_numel_equals_buffer_size(const at::Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  // 使用TORCH_CHECK进行断言检查
  TORCH_CHECK(
      self.numel() == static_cast<int64_t>(self_impl->get_buffer_size()),
      "Number of elements in nested tensor must match number of elements in buffer.");
}

// 检查嵌套张量的元素数量与缓冲区大小是否相等，通过转发到具体的实现
inline void check_numel_equals_buffer_size(const NestedTensorImpl* self_ptr) {
  // 使用TORCH_CHECK进行断言检查
  TORCH_CHECK(
      self_ptr->numel() == static_cast<int64_t>(self_ptr->get_buffer_size()),
      "Number of elements in nested tensor must match number of elements in buffer.");
}

// 获取嵌套/普通张量的尺寸/步幅/偏移的辅助函数
inline IntArrayRef get_size_for_index(const Tensor& tensor, int i) {
  // 如果张量是嵌套的
  if (tensor.is_nested()) {
    // 如果输入的张量是嵌套张量（nested tensor）
    std::vector<IntArrayRef> tensor_sizes =
        NestedTensor_get_sizes(get_nested_tensor_impl(tensor));
    // 返回嵌套张量的第 i 个元素的大小（shape）
    return tensor_sizes[i];
  } else {
    // 如果输入的张量不是嵌套张量，则返回从第一个维度（索引为1）开始的所有维度大小（shape）
    return tensor.sizes().slice(1);
  }
}

// 返回指定张量的索引处的步长数组引用
inline IntArrayRef get_stride_for_index(const Tensor& tensor, int i) {
  // 如果张量是嵌套的
  if (tensor.is_nested()) {
    // 获取嵌套张量的步长数组
    std::vector<IntArrayRef> tensor_strides =
        NestedTensor_get_strides(get_nested_tensor_impl(tensor));
    // 返回指定索引处的步长数组引用
    return tensor_strides[i];
  } else {
    // 如果张量不是嵌套的，返回从第二维开始的步长数组引用
    return tensor.strides().slice(1);
  }
}

// 返回指定张量的索引处的偏移量
inline int64_t get_offset_for_index(const Tensor& tensor, int i) {
  // 如果张量是嵌套的
  if (tensor.is_nested()) {
    // 获取嵌套张量的存储偏移量指针
    int64_t* offsets_ptr = get_nested_tensor_impl(tensor)
                               ->get_storage_offsets()
                               .data_ptr<int64_t>();
    // 返回指定索引处的偏移量
    return offsets_ptr[i];
  } else {
    // 如果张量不是嵌套的，计算并返回指定索引处的偏移量
    int64_t offset = tensor.storage_offset();
    return offset + tensor.strides()[0] * i;
  }
}
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 用于在嵌套张量上通用应用函数的数据结构和函数。
namespace impl {

// 嵌套节点结构模板
template <typename T>
struct NestedNode {
  NestedNode() = delete;
  // 构造函数，接受子节点的移动语义
  explicit NestedNode(std::vector<T>&& children)
      : _is_leaf(false), _children(children) {}
  // 构造函数，接受张量列表作为子节点
  explicit NestedNode(TensorList children)
      : _is_leaf(false), _children(children.vec()) {}
  // 构造函数，接受负载作为叶子节点
  explicit NestedNode(T payload)
      : _is_leaf(true), _payload(std::move(payload)) {}

  // 返回节点是否是叶子节点
  inline bool is_leaf() const {
    return _is_leaf;
  }
  // 返回节点的度（子节点个数）
  inline size_t degree() const {
    return _children.size();
  }
  // 解绑节点的子节点
  inline const std::vector<T> unbind() const {
    return _children;
  }
  // 返回指定索引处的子节点
  inline T children(size_t i) const {
    return _children[i];
  }
  // 返回节点的负载（如果是叶子节点）
  inline const T& payload() const {
    return _payload;
  }
  // 返回节点的负载（如果是叶子节点），可用于修改
  inline T& payload() {
    return _payload;
  }

 private:
  bool _is_leaf;        // 节点是否是叶子节点的标志
  std::vector<T> _children;  // 子节点数组
  T _payload;           // 负载数据
};

// 使用Lambda函数应用于嵌套节点的映射类模板
using TensorNode = NestedNode<at::Tensor>;

// 用于嵌套节点的映射类模板特化，处理给定类型列表的映射操作
template <class F, class A, class TypeList>
class _map;

template <class F, class A, class... Args>
class _map<F, A, c10::guts::typelist::typelist<Args...>> {
 public:
  // 应用单个函数于给定参数的映射操作
  static A function_one(F&& fn, const Args&... nested_node) {
    return std::forward<F>(fn)(nested_node...);
  }
  
  // 应用函数于嵌套节点的映射操作
  // 注意：必须移动F以避免Lambda函数捕获时的对象复制
  static NestedNode<A> function(
      F&& fn,
      const NestedNode<Args>&... nested_node) {
    size_t degree = 0;
    bool all_leaf = true;
    // 遍历嵌套节点，并更新degree和all_leaf标志
    c10::guts::tuple_map(
        std::forward_as_tuple(nested_node...), [&all_leaf, &degree](auto n) {
          all_leaf = all_leaf && (n.is_leaf());
          if (degree > 1 && n.degree() > 1) {
            TORCH_CHECK(
                degree == n.degree(), "NestedNodes must match in degree.");
          }
          if (n.degree() > degree) {
            degree = n.degree();
          }
          return nullptr;
        });
    // 如果所有的嵌套节点都是叶子节点，直接应用函数于负载并返回
    if (all_leaf) {
      return NestedNode<A>(std::forward<F>(fn)(nested_node.payload()...));
    }
    // 否则，构造一个新的嵌套节点
    // 这里没有完整的代码，只有注释
    // Some NestedNodes wrap regular Tensors, some NestedTensors and some other
    // types.
    // 定义一个存储类型为 A 的向量 result，用于存储最终的结果
    std::vector<A> result;
    // 遍历每一个子节点，子节点的数量由 degree 决定
    for (size_t i = 0; i < degree; i++) {
      // 通过 tuple_map 将 nested_node 转换为 std::tuple<Args...> 类型的 children
      std::tuple<Args...> children = c10::guts::tuple_map(
          std::forward_as_tuple(nested_node...), [&i](auto a) {
            static_assert(
                c10::guts::is_instantiation_of<NestedNode, decltype(a)>::value,
                "Internal error.");
            // 如果当前节点是叶子节点，则返回其 payload
            if (a.is_leaf()) {
              return a.payload();
            }
            // 如果当前节点的度为 1 且不是叶子节点，则返回其第一个子节点
            // 用于将 NestedTensors 广播到单一成员
            if (a.degree() == 1 && !a.is_leaf()) {
              return a.children(0);
            }
            // 对于其它情况，确保节点的度大于 0，否则触发断言错误
            TORCH_CHECK(a.degree() > 0, "Internal assert.");
            // 返回当前节点的第 i 个子节点
            return a.children(i);
          });
      // 对 children 应用函数 fn，并将结果存储到 result 中
      c10::guts::apply(
          [&result, &fn](Args... filtered) {
            result.emplace_back(function_one(std::forward<F>(fn), filtered...));
          },
          std::move(children));
    }
    // 返回一个新的 NestedNode<A>，其包含 result 中的内容
    return NestedNode<A>(std::move(result));
};

// 结束匿名命名空间或静态代码块的标记

// TODO: Add static assert to verify lambda arguments match nested_node types
// 添加静态断言以验证 lambda 参数与 nested_node 类型匹配

template <class F, class... B>
// 定义 map 函数模板，接受一个函数对象和多个 NestedNode 对象，并返回一个 NestedNode
static inline NestedNode<
    typename c10::guts::infer_function_traits<F>::type::return_type>
map(F&& fn, const NestedNode<B>&... nested_node) {
  // 调用 _map 函数模板，将参数 fn 和 nested_node 转发给 _map 函数
  return _map<
      F,
      typename c10::guts::infer_function_traits<F>::type::return_type,
      typename c10::guts::infer_function_traits<F>::type::parameter_types>::
      function(std::forward<F>(fn), nested_node...);
}

// 获取嵌套张量结构的函数，接受一个张量参数，并返回一个 TensorNode 对象
inline TensorNode get_nested_tensor_structure(at::Tensor tensor) {
  // 如果张量没有嵌套实现，则返回包装后的 TensorNode
  if (get_nested_tensor_impl_or_null(tensor) == nullptr) {
    return TensorNode(std::move(tensor));
  }
  // 否则，返回解绑定后的 TensorNode
  return TensorNode(tensor.unbind());
}

// 包装 TensorNode 的张量函数，接受 TensorNode、可选的数据类型、布局、设备和锁页内存参数，并返回一个张量
inline Tensor wrap_tensor_node(
    TensorNode tensor_node,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 检查 TensorNode 是否是叶子节点，如果是则抛出错误
  TORCH_CHECK(
      !tensor_node.is_leaf(), "Expected TensorNode to wrap a list of Tensors.");
  // 构造张量选项对象 options_
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  // 如果 TensorNode 的度为 0，返回包装后的空张量
  if (tensor_node.degree() == 0) {
    return wrap_buffer(ones({0}, dtype, layout, device), ones({}));
  }

  // 快速路径：如果所有张量都在 CPU 上、内存连续且数据类型相同，则可以更快地进行复制
  bool all_tensors_cpu = true;
  bool all_tensors_contiguous = true;
  bool all_tensors_same_dtype = true;
  auto first_dtype = tensor_node.children(0).dtype();
  std::vector<long> start_offsets(tensor_node.degree());
  start_offsets[0] = 0;
  long total_size = 0;
  // 遍历 TensorNode 的子节点
  for (const auto i : c10::irange(tensor_node.degree())) {
    // 检查所有条件
    all_tensors_cpu = all_tensors_cpu && tensor_node.children(i).is_cpu();
    all_tensors_contiguous =
        all_tensors_contiguous && tensor_node.children(i).is_contiguous();
    all_tensors_same_dtype = all_tensors_same_dtype &&
        (first_dtype == tensor_node.children(i).dtype());
    // 如果有条件不满足，则退出循环
    if (!(all_tensors_cpu && all_tensors_contiguous &&
          all_tensors_same_dtype)) {
      break;
    }
    // 计算每个张量的起始偏移量和总大小
    if (i > 0) {
      start_offsets[i] =
          start_offsets[i - 1] + tensor_node.children(i - 1).numel();
    }
    total_size += tensor_node.children(i).numel();
  }

  // 构造张量选项对象 options
  TensorOptions options;
  Tensor nt_buffer, nt_sizes;
  // 如果所有张量都在 CPU 上、内存连续且数据类型相同，则使用快速路径
  if (all_tensors_cpu && all_tensors_contiguous && all_tensors_same_dtype) {
    // 创建空张量作为缓冲区和大小张量
    nt_buffer = at::empty({total_size}, tensor_node.children(0).options());
    nt_sizes = at::empty(
        {static_cast<long>(tensor_node.degree()),
         static_cast<long>(tensor_node.children(0).sizes().size())},
        TensorOptions().dtype(kLong));
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        c10::typeMetaToScalarType(first_dtype),
        "create_nt_buffer",
        [&]() {
          at::parallel_for(
              0, tensor_node.degree(), 1, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                  // 只有在某个张量包含的元素数量大于0时才尝试复制内存
                  if (tensor_node.children(i).numel() > 0) {
                    // 使用 memcpy 复制内存，从 tensor_node 中的子张量到 nt_buffer 中
                    memcpy(
                        nt_buffer.mutable_data_ptr<scalar_t>() + start_offsets[i],
                        tensor_node.children(i).const_data_ptr<scalar_t>(),
                        tensor_node.children(i).numel() * sizeof(scalar_t));
                  }
                }
              });
        });
    // 初始化 sizes_offset 变量为0
    long sizes_offset = 0;
    // 遍历每个子张量，获取其尺寸并存储到 nt_sizes 中
    for (size_t i = 0; i < tensor_node.degree(); ++i) {
      auto tensor_sizes = tensor_node.children(i).sizes();
      for (int64_t tensor_size : tensor_sizes) {
        // 将每个张量的尺寸存储到 nt_sizes 中
        nt_sizes.mutable_data_ptr<int64_t>()[sizes_offset++] = tensor_size;
      }
    }
    // 将 options_ 合并到 nt_buffer 的选项中，并赋给 options 变量
    options = nt_buffer.options().merge_in(options_);
  } else { // Slow path
    // 如果不满足快速路径条件，执行慢速路径
    std::vector<Tensor> flat_tensors;
    std::vector<Tensor> sizes;
    // 对于每个子张量，将其展平并转换为连续张量，存储在 flat_tensors 中，并获取其尺寸存储在 sizes 中
    for (const auto i : c10::irange(tensor_node.degree())) {
      flat_tensors.push_back(tensor_node.children(i).reshape(-1).contiguous());
      sizes.push_back(
          tensor(c10::IntArrayRef(tensor_node.children(i).sizes())));
    }
    // 将 flat_tensors 的选项合并到 options_ 中，并赋给 options 变量
    options = flat_tensors[0].options().merge_in(options_);
    // 将 flat_tensors 连接成一个张量，并赋给 nt_buffer 变量
    nt_buffer = at::cat(flat_tensors);
    // 使用 at::native::stack 将 sizes 合并成一个张量，并赋给 nt_sizes 变量
    nt_sizes = at::native::stack(sizes);
  }

  // 将 nt_buffer 和 nt_sizes 封装成一个封装后的缓冲区并返回
  return wrap_buffer(nt_buffer.to(options), nt_sizes);
} // 结束命名空间 impl

} // 结束命名空间 namespace

// 这个函数旨在为 NestedTensor 内核提供快速的操作符覆盖。
// 它并非旨在效率上优化。请谨慎使用。
template <class F, class... A>
inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  return wrap_tensor_node(
      // 使用 fn 函数对给定的 NestedTensor 结构进行映射操作
      impl::map(std::forward<F>(fn), impl::get_nested_tensor_structure(a)...),
      // 包装成 Tensor 节点，以下参数都为默认空值
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);
}

} // 结束命名空间 native
} // 结束命名空间 at
```