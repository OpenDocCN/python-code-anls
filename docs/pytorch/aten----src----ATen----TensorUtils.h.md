# `.\pytorch\aten\src\ATen\TensorUtils.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/DimVector.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/Utils.h>
// 包含必要的头文件，用于张量和几何结构的操作和定义

#include <utility>
// 包含用于 std::move 和其他实用工具的头文件

// 这些函数没有放在 Utils.h 中，因为该文件依赖于 Tensor.h

#define TORCH_CHECK_TENSOR_ALL(cond, ...) \
  TORCH_CHECK((cond)._is_all_true().item<bool>(), __VA_ARGS__);
// 定义宏 TORCH_CHECK_TENSOR_ALL，用于检查张量是否满足给定条件

namespace at {

// 下面是用于检查参数合理性的实用函数。对于原生函数尤为有用，因为它们默认不进行参数检查。

struct TORCH_API TensorArg {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const Tensor& tensor;
  const char* name;
  int pos; // 1-indexed，参数在参数列表中的位置（从1开始）
  TensorArg(const Tensor& tensor, const char* name, int pos)
      : tensor(tensor), name(name), pos(pos) {}
  // 尝试减少对临时对象悬挂引用的可能性。
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  TensorArg(Tensor&& tensor, const char* name, int pos) = delete;
  const Tensor* operator->() const {
    return &tensor;
  }
  const Tensor& operator*() const {
    return tensor;
  }
};

struct TORCH_API TensorGeometryArg {
  TensorGeometry tensor;
  const char* name;
  int pos; // 1-indexed，参数在参数列表中的位置（从1开始）
  /* implicit */ TensorGeometryArg(TensorArg arg)
      : tensor(TensorGeometry{arg.tensor}), name(arg.name), pos(arg.pos) {}
  TensorGeometryArg(TensorGeometry tensor, const char* name, int pos)
      : tensor(std::move(tensor)), name(name), pos(pos) {}
  const TensorGeometry* operator->() const {
    return &tensor;
  }
  const TensorGeometry& operator*() const {
    return tensor;
  }
};

// 描述哪个函数对其输入参数进行了检查的字符串。
// TODO: 考虑将其推广为调用堆栈的一部分。
using CheckedFrom = const char*;

// 未定义的约定：单一运算符假定其参数已定义，但多个张量的函数将隐式过滤掉未定义的张量
// （以便更轻松地执行仅在张量定义时应用的测试，否则不应该应用）。
//
// 注意：这意味着 n 元运算符接受 TensorArg 列表，而不是 TensorGeometryArg，
// 因为如果存在未定义的张量，Tensor 到 TensorGeometry 的转换将导致失败。

TORCH_API std::ostream& operator<<(
    std::ostream& out,
    const TensorGeometryArg& t);
// 重载流操作符，用于将 TensorGeometryArg 输出到流中

TORCH_API void checkDim(
    CheckedFrom c,
    const Tensor& tensor,
    const char* name,
    int pos, // 1-indexed，张量维度的位置
    int64_t dim);
// 检查给定张量的指定维度是否为给定值

TORCH_API void checkDim(CheckedFrom c, const TensorGeometryArg& t, int64_t dim);
// 检查给定张量几何结构对象的指定维度是否为给定值

// 注意：这是一个包含-不包含的范围
TORCH_API void checkDimRange(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim_start,
    int64_t dim_end);
// 检查给定张量几何结构对象的指定维度范围是否在给定的开始和结束之间

TORCH_API void checkSameDim(
    CheckedFrom c,
    const TensorGeometryArg& t1,
    const TensorGeometryArg& t2);
// 检查两个张量的几何结构对象是否具有相同的维度

TORCH_API void checkContiguous(CheckedFrom c, const TensorGeometryArg& t);
// 检查给定张量的几何结构对象是否是连续的
// 检查所有张量是否都是连续的
TORCH_API void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts);

// 检查张量的尺寸是否匹配给定的尺寸
TORCH_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    IntArrayRef sizes);

// 检查张量的尺寸是否匹配给定的尺寸（支持符号整数）
TORCH_API void checkSize_symint(
    CheckedFrom c,
    const TensorGeometryArg& t,
    c10::SymIntArrayRef sizes);

// 检查张量的指定维度的尺寸是否匹配给定的尺寸
TORCH_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    int64_t size);

// 检查张量的指定维度的尺寸是否匹配给定的尺寸（支持符号整数）
TORCH_API void checkSize_symint(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    const c10::SymInt& size);

// 检查张量的元素数量是否与给定值相等
TORCH_API void checkNumel(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t numel);

// 检查两个张量是否具有相同的元素数量
TORCH_API void checkSameNumel(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);

// 检查一组张量是否都具有相同的元素数量
TORCH_API void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors);

// 检查张量的标量类型是否匹配给定的标量类型
TORCH_API void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType s);

// 检查张量的标量类型是否与给定的一组标量类型中的任意一个匹配
TORCH_API void checkScalarTypes(
    CheckedFrom c,
    const TensorArg& t,
    at::ArrayRef<ScalarType> l);

// 检查两个张量是否位于相同的 GPU 上
TORCH_API void checkSameGPU(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);

// 检查一组张量是否都位于相同的 GPU 上
TORCH_API void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors);

// 检查两个张量是否具有相同的类型
TORCH_API void checkSameType(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);

// 检查一组张量是否都具有相同的类型
TORCH_API void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors);

// 检查两个张量是否具有相同的尺寸
TORCH_API void checkSameSize(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);

// 检查一组张量是否都具有相同的尺寸
TORCH_API void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors);

// 检查张量是否已定义（非空）
TORCH_API void checkDefined(CheckedFrom c, const TensorArg& t);

// 检查一组张量是否都已定义（非空）
TORCH_API void checkAllDefined(CheckedFrom c, at::ArrayRef<TensorArg> t);

// 修复：TensorArg 是否会拖慢速度？

// 检查一组张量是否具有相同的后端（Backend）
TORCH_API void checkBackend(
    CheckedFrom c,
    at::ArrayRef<Tensor> t,
    at::Backend backend);

// 检查一组张量是否都属于相同的设备类型（DeviceType）
TORCH_API void checkDeviceType(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::DeviceType device_type);

// 检查张量的布局是否与给定的布局匹配
TORCH_API void checkLayout(CheckedFrom c, const Tensor& t, Layout layout);

// 检查一组张量的布局是否都与给定的布局匹配
TORCH_API void checkLayout(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::Layout layout);

// 如果张量已定义，返回其数据指针
TORCH_API void* maybe_data_ptr(const Tensor& tensor);

// 如果张量已定义，返回其数据指针
TORCH_API void* maybe_data_ptr(const TensorArg& tensor);

// 检查张量的指定维度的大小是否与给定的大小匹配
TORCH_API void check_dim_size(
    const Tensor& tensor,
    int64_t dim,
    int64_t dim_size,
    int64_t size);

// 包含了一些详细实现的命名空间
namespace detail {

// 计算默认的步长数组，给定尺寸数组
TORCH_API std::vector<int64_t> defaultStrides(IntArrayRef sizes);

// 计算新形状相对于旧形状的步长数组（支持符号整数）
TORCH_API std::optional<std::vector<int64_t>> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape);

// 计算新形状相对于旧形状的步长数组（支持符号整数）
TORCH_API std::optional<SymDimVector> computeStride(
    c10::SymIntArrayRef oldshape,
    c10::SymIntArrayRef oldstride,
    c10::SymIntArrayRef newshape);

// 计算新形状相对于旧形状的步长数组
TORCH_API std::optional<DimVector> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    const DimVector& newshape);

} // namespace detail

} // namespace at
```