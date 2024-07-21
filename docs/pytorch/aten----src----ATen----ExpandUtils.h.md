# `.\pytorch\aten\src\ATen\ExpandUtils.h`

```py
#pragma once

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/view.h>
#include <ATen/ops/view_copy.h>
#endif

#include <ATen/Tensor.h>
#include <ATen/core/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/irange.h>

#include <functional>
#include <sstream>
#include <tuple>
#include <utility>

// 命名空间：at，包含了所有的 ATen 库相关内容
namespace at {

// 推断两个整数数组的尺寸
TORCH_API std::vector<int64_t> infer_size(IntArrayRef a, IntArrayRef b);

// 推断两个符号整数数组的尺寸
TORCH_API std::vector<SymInt> infer_size_symint(
    SymIntArrayRef a,
    SymIntArrayRef b);

// 推断两个整数数组的尺寸并返回 DimVector
TORCH_API DimVector infer_size_dimvector(IntArrayRef a, IntArrayRef b);

// 推断两个符号整数数组的尺寸并返回 SymDimVector
TORCH_API SymDimVector
infer_size_symdimvector(SymIntArrayRef a, SymIntArrayRef b);

// 结构体模板：InferExpandGeometryResult
// 用于返回尺寸和步长的容器，确保通过 NRVO 进行优化
template <typename Container>
struct InferExpandGeometryResult {
  Container sizes;   // 尺寸容器
  Container strides; // 步长容器

  // 构造函数：根据给定的维度数 ndim 构造 sizes 和 strides
  explicit InferExpandGeometryResult(size_t ndim)
      : sizes(ndim), strides(ndim) {}

  // 构造函数：根据给定的 sizes_ 和 ndim 构造 sizes 和 strides
  explicit InferExpandGeometryResult(IntArrayRef sizes_, size_t ndim)
      : sizes(sizes_.begin(), sizes_.end()), strides(ndim) {}
};

// 推断扩展的几何结构并返回尺寸和步长
TORCH_API std::tuple<std::vector<int64_t>, std::vector<int64_t>>
inferExpandGeometry(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

// 推断扩展的几何结构并返回 DimVector 尺寸和步长
TORCH_API InferExpandGeometryResult<DimVector> inferExpandGeometry_dimvector(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

// 推断密集步长
TORCH_API std::vector<int64_t> infer_dense_strides(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides);

// 检查输入形状是否可扩展的函数
// 注意：如果需要更改，请与 infer_size 保持同步
inline bool are_expandable(IntArrayRef shape1, IntArrayRef shape2) {
  size_t ndim1 = shape1.size(); // 获取 shape1 的维度数
  size_t ndim2 = shape2.size(); // 获取 shape2 的维度数
  size_t ndim = ndim1 < ndim2 ? ndim1 : ndim2; // 取较小的维度数

  // 逆序比较各维度是否可扩展
  for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
    // 如果两个维度相等，或其中一个维度为 1，则可扩展
    if (shape1[--ndim1] == shape2[--ndim2] || shape1[ndim1] == 1 ||
        shape2[ndim2] == 1) {
      continue;
    }
    return false; // 否则不可扩展
  }
  return true; // 可扩展
}

// 避免使用引用包装进行张量的复制构造
inline void check_defined(
    std::initializer_list<std::reference_wrapper<const Tensor>> tensors,
    const char* api_name) {
  for (auto& t : tensors) {
    if (!t.get().defined()) {
      AT_ERROR(api_name, "(...) called with an undefined Tensor"); // 若张量未定义，则抛出异常
    }
  }
}

// 注意事项 [ ExpandUtils Borrowing ]
//
// ExpandUtils 中的函数返回 `c10::MaybeOwned<Tensor>`，因为
// 可能不需要实际进行扩展，可以通过 `c10::MaybeOwned<Tensor>::borrowed(to_expand)`
// 提高效率。但需要注意的是：返回的 `c10::MaybeOwned<Tensor>`
// 不能超出原始 `Tensor` 对象 `to_expand` 的生存周期！
// 这些函数的右值引用重载被删除了。
// 在-place 扩展张量的内联函数，返回一个 MaybeOwned<Tensor> 对象，可以是借用或拥有的状态

inline c10::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand) {
  // 检查张量的符号尺寸是否相等，如果相等，则返回一个借用状态的 to_expand
  if (tensor.sym_sizes().equals(to_expand.sym_sizes())) {
    return c10::MaybeOwned<Tensor>::borrowed(to_expand);
  }
  // 否则，返回一个拥有状态的 to_expand，通过 expand_symint 方法扩展成 tensor 的符号整型
  return c10::MaybeOwned<Tensor>::owned(
      to_expand.expand_symint(tensor.sym_sizes()));
}

// 删除右值引用版本的 expand_inplace 函数
inline c10::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand) = delete;

// 在-place 扩展张量的内联函数，带有额外的 API 名称参数
inline c10::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand,
    const char* api_name) {
  // 使用 check_defined 函数检查 tensor 和 to_expand 是否已定义
  check_defined({tensor, to_expand}, api_name);
  // 调用前一个版本的 expand_inplace 函数
  return expand_inplace(tensor, to_expand);
}

// 删除带有右值引用版本的带有 API 名称参数的 expand_inplace 函数
inline c10::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand,
    const char* api_name) = delete;

// 在-place 扩展两个张量的内联函数，返回一个元组，每个元素都是 MaybeOwned<Tensor> 对象
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand1,
    const Tensor& to_expand2) {
  // 检查两个张量的尺寸是否相等
  if (tensor.sizes().equals(to_expand1.sizes()) &&
      tensor.sizes().equals(to_expand2.sizes())) {
    // 如果相等，返回一个包含两个借用状态的元组
    return std::make_tuple(
        c10::MaybeOwned<Tensor>::borrowed(to_expand1),
        c10::MaybeOwned<Tensor>::borrowed(to_expand2));
  }

  // 否则，返回一个包含两个拥有状态的元组，分别通过 expand 方法扩展成 tensor 的尺寸
  return std::make_tuple(
      c10::MaybeOwned<Tensor>::owned(to_expand1.expand(tensor.sizes())),
      c10::MaybeOwned<Tensor>::owned(to_expand2.expand(tensor.sizes())));
}

// 删除右值引用版本的 expand_inplace 函数，其中第一个参数是右值引用
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand1,
    const Tensor& to_expand2) = delete;

// 删除带有右值引用版本的 expand_inplace 函数，其中第二个参数是右值引用
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand1,
    Tensor&& to_expand2) = delete;

// 删除带有两个右值引用版本的 expand_inplace 函数
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(const Tensor& tensor, Tensor&& to_expand1, Tensor&& to_expand2) =
    delete;

// 在-place 扩展三个张量的内联函数，返回一个元组，每个元素都是 MaybeOwned<Tensor> 对象
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    const char* api_name) {
  // 使用 check_defined 函数检查 tensor 和 to_expand1、to_expand2 是否已定义
  check_defined({tensor, to_expand1, to_expand2}, api_name);
  // 调用前一个版本的 expand_inplace 函数
  return expand_inplace(tensor, to_expand1, to_expand2);
}

// 删除带有右值引用版本的带有 API 名称参数的 expand_inplace 函数，其中第一个参数是右值引用
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    const char* api_name) = delete;

// 删除带有右值引用版本的带有 API 名称参数的 expand_inplace 函数，其中第二个参数是右值引用
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    const char* api_name) = delete;

// 删除带有两个右值引用版本的带有 API 名称参数的 expand_inplace 函数
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    const char* api_name) = delete;
// 根据上面的说明 [ ExpandUtils Borrowing ]，解释 `MaybeOwned` 的用法
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(const Tensor& to_expand1, const Tensor& to_expand2) {
  // 获取第一个和第二个输入张量的符号化大小
  auto s1 = to_expand1.sym_sizes();
  auto s2 = to_expand2.sym_sizes();
  // 如果两个张量的符号化大小相等，则直接返回它们的引用
  if (s1.equals(s2)) {
    return std::make_tuple(
        c10::MaybeOwned<Tensor>::borrowed(to_expand1),
        c10::MaybeOwned<Tensor>::borrowed(to_expand2));
  }

  // 推断扩展后的大小
  auto expanded_size = infer_size_symdimvector(s1, s2);
  // 返回扩展后的张量，作为拥有的对象
  return std::make_tuple(
      c10::MaybeOwned<Tensor>::owned(to_expand1.expand_symint(expanded_size)),
      c10::MaybeOwned<Tensor>::owned(to_expand2.expand_symint(expanded_size)));
}

// 禁止通过右值引用扩展张量的不同重载
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(Tensor&& to_expand1, const Tensor& to_expand2) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(const Tensor& to_expand1, Tensor&& to_expand2) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(Tensor&& to_expand1, Tensor&& to_expand2) = delete;

// 带有 API 名称参数的扩展张量函数重载
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    const char* api_name) {
  // 检查张量是否已定义，使用给定的 API 名称进行检查
  check_defined({to_expand1, to_expand2}, api_name);
  // 调用无 API 名称参数版本的扩展函数，并返回结果
  return expand_outplace(to_expand1, to_expand2);
}

// 禁止通过右值引用扩展张量的带有 API 名称参数的重载
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    const char* api_name) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    const char* api_name) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    const char* api_name) = delete;

// 三个张量的扩展函数重载
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    const Tensor& to_expand3) {
  // 如果三个张量的大小相同，则返回它们的引用
  if (to_expand1.sizes().equals(to_expand2.sizes()) &&
      to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(
        c10::MaybeOwned<Tensor>::borrowed(to_expand1),
        c10::MaybeOwned<Tensor>::borrowed(to_expand2),
        c10::MaybeOwned<Tensor>::borrowed(to_expand3));
  }

  // 推断扩展后的大小
  auto expanded_size12 =
      infer_size_dimvector(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size =
      infer_size_dimvector(expanded_size12, to_expand3.sizes());
  // 返回扩展后的张量，作为拥有的对象
  return std::make_tuple(
      c10::MaybeOwned<Tensor>::owned(to_expand1.expand(expanded_size)),
      c10::MaybeOwned<Tensor>::owned(to_expand2.expand(expanded_size)),
      c10::MaybeOwned<Tensor>::owned(to_expand3.expand(expanded_size)));
}
    // 删除了带有特定参数类型的移动构造函数和多个常量引用参数的构造函数重载
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    const Tensor& to_expand3) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,     // 第一个输入参数，以常量引用方式传入，表示要扩展的第一个张量
    Tensor&& to_expand2,          // 第二个输入参数，以右值引用方式传入，表示要扩展的第二个张量
    const Tensor& to_expand3) = delete;
                                // 第三个输入参数，以常量引用方式传入，表示要扩展的第三个张量
                                // 该函数被删除，不可用

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,          // 第一个输入参数，以右值引用方式传入，表示要扩展的第一个张量
    Tensor&& to_expand2,          // 第二个输入参数，以右值引用方式传入，表示要扩展的第二个张量
    const Tensor& to_expand3) = delete;
                                // 第三个输入参数，以常量引用方式传入，表示要扩展的第三个张量
                                // 该函数被删除，不可用

// 下面几个函数都类似，它们的作用是在给定的三个张量上进行扩展操作，但是由于右值引用的使用方式不同，
// 且它们被标记为删除，因此在编译时不可使用这些函数。

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,     // 第一个输入参数，以常量引用方式传入，表示要扩展的第一个张量
    const Tensor& to_expand2,     // 第二个输入参数，以常量引用方式传入，表示要扩展的第二个张量
    Tensor&& to_expand3) = delete;
                                // 第三个输入参数，以右值引用方式传入，表示要扩展的第三个张量
                                // 该函数被删除，不可用

// 以下几个函数同样被标记为删除，因此不可使用，其功能与前述函数类似，仅右值引用参数的使用方式不同。

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    Tensor&& to_expand3) = delete;

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    Tensor&& to_expand3) = delete;

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    Tensor&& to_expand3) = delete;

// 在这里，定义了一个可用的扩展函数，其参数包括三个张量和一个 API 名称。
// 函数首先检查所提供的张量是否已定义，然后调用上述被删除的函数进行实际的扩展操作。

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,     // 第一个输入参数，以常量引用方式传入，表示要扩展的第一个张量
    const Tensor& to_expand2,     // 第二个输入参数，以常量引用方式传入，表示要扩展的第二个张量
    const Tensor& to_expand3,     // 第三个输入参数，以常量引用方式传入，表示要扩展的第三个张量
    const char* api_name) {      // API 名称，用于检查已定义的张量
  check_defined({to_expand1, to_expand2, to_expand3}, api_name);   // 检查三个张量是否已定义
  return expand_outplace(to_expand1, to_expand2, to_expand3);      // 调用被删除的扩展函数，实际执行扩展操作
}

// 下面的函数都被标记为删除，因此不可使用，其功能与前述函数类似，仅右值引用参数的使用方式不同。

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    const Tensor& to_expand3,
    const char* api_name) = delete;

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    const Tensor& to_expand3,
    const char* api_name) = delete;

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    const Tensor& to_expand3,
    const char* api_name) = delete;

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    Tensor&& to_expand3,
    const char* api_name) = delete;

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    Tensor&& to_expand3,
    const char* api_name) = delete;
    // 声明了一个名为 delete 的特殊成员函数（deleted function），表示禁止使用这个函数进行调用或实例化
    const Tensor& to_expand1,  // 第一个参数，传入时不可修改，是一个常量引用
    Tensor&& to_expand2,       // 第二个参数，传入时是一个右值引用，可以接受临时对象或移动语义
    Tensor&& to_expand3,       // 第三个参数，同样是右值引用，接受临时对象或移动语义
    const char* api_name       // 第四个参数，是一个指向常量字符的指针，用于标识 API 名称
    ) = delete;                // 将整个函数声明为删除的，禁止使用该函数
// 删除了函数 expand_outplace，该函数接受三个 Tensor 参数和一个 API 名称，返回三个 MaybeOwned<Tensor> 类型的元组。
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    Tensor&& to_expand3,
    const char* api_name) = delete;

// 根据给定的 Tensor 和 sizes 参数，如果 Tensor 的尺寸与 sizes 相同，则返回 Tensor 的 borrowed 版本；否则返回扩展后的 owned 版本。
inline c10::MaybeOwned<Tensor> expand_size(
    const Tensor& to_expand,
    IntArrayRef sizes) {
  // 检查 Tensor 的尺寸是否与给定的 sizes 相同，如果相同则直接返回 borrowed 版本
  if (to_expand.sizes().equals(sizes)) {
    return c10::MaybeOwned<Tensor>::borrowed(to_expand);
  }
  // 否则返回 owned 版本，使用 expand 方法扩展 Tensor 到给定的 sizes
  return c10::MaybeOwned<Tensor>::owned(to_expand.expand(sizes));
}

// 删除了函数 expand_size，该函数接受一个移动语义的 Tensor 和 sizes 参数。
inline c10::MaybeOwned<Tensor> expand_size(
    Tensor&& to_expand,
    IntArrayRef sizes) = delete;

// 根据给定的 Tensor、sizes 参数和 API 名称，检查 Tensor 是否已定义，然后调用 expand_size 函数进行尺寸扩展。
inline c10::MaybeOwned<Tensor> expand_size(
    const Tensor& to_expand,
    IntArrayRef sizes,
    const char* api_name) {
  // 检查输入的 Tensor 是否已定义，如果未定义会抛出异常
  check_defined({to_expand}, api_name);
  // 调用上面定义的 expand_size 函数进行尺寸扩展，并返回结果
  return expand_size(to_expand, sizes);
}

// 删除了函数 expand_size，该函数接受一个移动语义的 Tensor、sizes 参数和 API 名称。
inline c10::MaybeOwned<Tensor> expand_size(
    Tensor&& to_expand,
    IntArrayRef sizes,
    const char* api_name) = delete;

// 对输入的 TensorList 进行扩展操作，返回扩展后的 Tensor 向量。
inline std::vector<Tensor> expand_outplace(TensorList to_expand) {
  // 初始化变量，用于记录是否为第一个有效的 Tensor 和记录尺寸信息的向量
  bool first = true;
  DimVector sizes;
  // 遍历输入的 TensorList
  for (const auto i : c10::irange(to_expand.size())) {
    // 如果当前 Tensor 未定义，则跳过
    if (!to_expand[i].defined()) {
      continue;
    } else if (first) {
      // 如果是第一个定义的 Tensor，则记录其尺寸
      sizes = to_expand[i].sizes();
      first = false;
    } else {
      // 否则根据当前 Tensor 的尺寸更新 sizes 向量
      sizes = infer_size_dimvector(sizes, to_expand[i].sizes());
    }
  }

  // 初始化结果向量，大小与输入 TensorList 相同
  std::vector<Tensor> result(to_expand.size());
  // 再次遍历输入的 TensorList
  for (const auto i : c10::irange(to_expand.size())) {
    // 如果当前 Tensor 未定义，则跳过
    if (!to_expand[i].defined()) {
      continue;
    } else if (to_expand[i].sizes().equals(sizes)) {
      // 如果当前 Tensor 尺寸与 sizes 相同，则直接赋值给结果向量
      result[i] = to_expand[i];
    } else {
      // 否则使用 expand 方法将当前 Tensor 扩展到 sizes 尺寸，并赋值给结果向量
      result[i] = to_expand[i].expand(sizes);
    }
  }
  // 返回结果向量
  return result;
}

// 根据给定的 Tensor 和 shape，对其进行求和操作，并根据 always_return_non_view 的值返回 Tensor 或其视图。
template <typename T>
inline Tensor _sum_to(
    Tensor tensor,
    const c10::ArrayRef<T> shape,
    bool always_return_non_view = false) {
  // 如果 shape 的大小为 0，则对输入的 tensor 进行求和并返回结果
  if (shape.size() == 0) {
    return tensor.sum();
  }

  // 获取 tensor 的尺寸信息
  auto sizes = at::symint::sizes<T>(tensor);
  // 初始化 reduce_dims，用于记录需要减少的维度
  c10::SmallVector<int64_t, 8> reduce_dims;
  // 计算 leading_dims，即 tensor 尺寸的维度数减去 shape 的维度数
  const int64_t leading_dims = sizes.size() - shape.size();
  // 遍历 leading_dims，将前 leading_dims 维度加入 reduce_dims
  for (const auto i : c10::irange(leading_dims)) {
    reduce_dims.push_back(i);
  }
  // 遍历 shape 的维度
  for (int64_t i = leading_dims; i < static_cast<int64_t>(sizes.size()); ++i) {
    // 如果当前维度满足条件，则将其加入 reduce_dims
    if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(shape[i - leading_dims], 1)) &&
        TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(sizes[i], 1))) {
      reduce_dims.push_back(i);
    }
  }

  // 如果 reduce_dims 不为空，则使用 sum 方法对 tensor 进行求和，并保持维度不变
  if (!reduce_dims.empty()) {
    tensor = tensor.sum(reduce_dims, /*keepdim=*/true);
  }

  // 根据 always_return_non_view 的值决定返回 tensor 或其视图
  if (always_return_non_view) {
    // 如果 always_return_non_view 为 true，则确保返回 tensor 的副本（非视图）
    return leading_dims > 0 ? at::symint::view_copy<T>(tensor, shape)
                            : tensor.clone();
  } else {
    // 否则返回 tensor 或其视图
    return leading_dims > 0 ? at::symint::view<T>(tensor, shape) : tensor;
  }
}

// 对输入的 Tensor 进行求和，并将结果的尺寸变换为给定的 shape
inline Tensor sum_to(
    Tensor tensor,
    const c10::SymIntArrayRef shape,
    # 定义一个布尔变量，控制是否始终返回非视图对象，默认为 False
    bool always_return_non_view = false;
    # 调用 _sum_to 函数，并将 tensor 对象以移动语义传递进去，
    # 传递 shape 参数和 always_return_non_view 变量作为参数，
    # 返回 _sum_to 函数的结果
    return _sum_to(std::move(tensor), shape, always_return_non_view);
// 结束命名空间 `at` 的定义

// 将 `tensor` 多次求和以生成形状为 `shape` 的张量。
// 先决条件：is_expandable_to(shape, tensor.sizes()) 必须为真
inline Tensor sum_to(
    Tensor tensor,
    const IntArrayRef shape,
    bool always_return_non_view = false) {
  // 调用内部函数 `_sum_to` 来执行实际的求和操作
  return _sum_to(std::move(tensor), shape, always_return_non_view);
}

// 检查 `shape` 是否可以扩展到 `desired` 的形状
static inline bool is_expandable_to(
    SymIntArrayRef shape,
    c10::SymIntArrayRef desired) {
  // 获取 `shape` 和 `desired` 的维度数
  size_t ndim = shape.size();
  size_t target_dim = desired.size();
  // 如果 `shape` 的维度数大于 `desired` 的维度数，则不可扩展
  if (ndim > target_dim) {
    return false;
  }
  // 逐个比较 `shape` 和 `desired` 的每个维度
  for (const auto i : c10::irange(ndim)) {
    // 获取当前维度的大小
    const auto& size = shape[ndim - i - 1];
    const auto& target = desired[target_dim - i - 1];
    // 如果当前维度的大小不等于目标维度的大小且不为1，则不可扩展
    if (size != target && size != 1) {
      return false;
    }
  }
  // 所有维度都匹配，可以扩展
  return true;
}

// 重载的 `is_expandable_to` 函数，处理普通整数数组的情况
static inline bool is_expandable_to(IntArrayRef shape, IntArrayRef desired) {
  // 将普通整数数组转换为符号整数数组并调用相应的函数
  auto sym_shape = c10::SymIntArrayRef(
      reinterpret_cast<const c10::SymInt*>(shape.data()), shape.size());
  auto sym_desired = c10::SymIntArrayRef(
      reinterpret_cast<const c10::SymInt*>(desired.data()), desired.size());
  return is_expandable_to(sym_shape, sym_desired);
}

// 结束命名空间 `at` 的定义
} // namespace at
```