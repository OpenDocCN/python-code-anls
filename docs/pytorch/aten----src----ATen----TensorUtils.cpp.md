# `.\pytorch\aten\src\ATen\TensorUtils.cpp`

```
// 引入 ATen 库中所需的头文件
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/TensorUtils.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

// 引入输出流和字符串流的头文件
#include <ostream>
#include <sstream>

// ATen 命名空间
namespace at {

// 定义重载运算符 <<，用于输出 TensorGeometryArg 对象到输出流
std::ostream& operator<<(std::ostream & out, const TensorGeometryArg& t) {
  // 如果位置为 0，输出单引号加参数名，通常表示 'self' 或返回的张量
  if (t.pos == 0) {
    out << "'" << t.name << "'";
  } else {
    // 否则输出参数位置和参数名
    out << "argument #" << t.pos << " '" << t.name << "'";
  }
  return out;
}

// 检查张量的维度是否符合预期
void checkDim(
    CheckedFrom c,
    const Tensor& tensor,
    const char* name,
    int pos, // 1-indexed
    int64_t dim) {
  // 使用 TORCH_CHECK 检查张量的维度是否等于给定的 dim
  TORCH_CHECK(
      tensor.dim() == dim,
      "Expected ",
      dim,
      "-dimensional tensor, but got ",
      tensor.dim(),
      "-dimensional tensor for ",
      TensorGeometryArg(TensorArg({tensor, name, pos})),
      " (while checking arguments for ",
      c,
      ")");
}

// 重载 checkDim 函数，接受 TensorGeometryArg 作为参数
void checkDim(CheckedFrom c, const TensorGeometryArg& t, int64_t dim) {
  // 使用 TORCH_CHECK 检查张量的维度是否等于给定的 dim
  TORCH_CHECK(t->dim() == dim,
    "Expected ", dim, "-dimensional tensor, but got ", t->dim(),
    "-dimensional tensor for ", t," (while checking arguments for ", c, ")");
}

// 检查张量的维度是否在给定的范围内
void checkDimRange(CheckedFrom c, const TensorGeometryArg& t, int64_t dim_start, int64_t dim_end) {
  // 使用 TORCH_CHECK 检查张量的维度是否在 dim_start 和 dim_end 之间
  TORCH_CHECK(
    t->dim() >= dim_start && t->dim() < dim_end,
    "Expected ", dim_start, " to ", (dim_end - 1), " dimensions, but got ",
    t->dim(), "-dimensional tensor for ", t, " (while checking arguments for ",
    c, ")");
}

// 检查张量是否是连续的
void checkContiguous(CheckedFrom c, const TensorGeometryArg& t) {
  // 使用 TORCH_CHECK 检查张量是否是连续的
  TORCH_CHECK(
    t->is_contiguous(),
    "Expected contiguous tensor, but got non-contiguous tensor for ", t,
     " (while checking arguments for ", c, ")");
}

// 检查所有张量是否都是连续的
void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts) {
  // 遍历所有张量，并检查它们是否是连续的
  for (auto& t : ts) {
    if (!t->defined()) continue;
    checkContiguous(c, t);
  }
}

// 检查张量的尺寸是否符合预期
void checkSize(CheckedFrom c, const TensorGeometryArg& t, IntArrayRef sizes) {
  // 检查张量的维度是否与 sizes 的长度相同，并且检查张量的具体尺寸是否与 sizes 相匹配
  checkDim(c, t, static_cast<int64_t>(sizes.size()));
  TORCH_CHECK(
    t->sizes().equals(sizes),
    "Expected tensor of size ", sizes, ", but got tensor of size ", t->sizes(),
    " for ", t, " (while checking arguments for ", c, ")");
}

// 重载 checkSize 函数，接受 c10::SymIntArrayRef 类型的 sizes
void checkSize_symint(CheckedFrom c, const TensorGeometryArg& t, c10::SymIntArrayRef sizes) {
  // 检查张量的维度是否与 sizes 的长度相同，并且检查张量的具体尺寸是否与 sizes 相匹配
  checkDim(c, t, static_cast<int64_t>(sizes.size()));
  TORCH_CHECK(
    t->sym_sizes().equals(sizes),
    "Expected tensor of size ", sizes, ", but got tensor of size ", t->sizes(),
    " for ", t, " (while checking arguments for ", c, ")");
}

// 检查张量的某一维度的尺寸是否符合预期
void checkSize(CheckedFrom c, const TensorGeometryArg& t, int64_t dim, int64_t size) {
  // 使用 TORCH_CHECK 检查张量的特定维度的尺寸是否符合预期
  TORCH_CHECK(
    t->size(dim) == size,
    "Expected tensor to have size ", size, " at dimension ", dim,
    ", but got size ", t->size(dim), " for ", t,
    " (while checking arguments for ", c, ")");
}

// 重载 checkSize 函数，接受 c10::SymInt 类型的 size
void checkSize_symint(CheckedFrom c, const TensorGeometryArg& t, int64_t dim, const c10::SymInt& size) {
  // 使用 TORCH_CHECK 检查张量的特定维度的尺寸是否符合预期
  TORCH_CHECK(
    t->sym_size(dim) == size,
    // 未完成的注释
    "Expected tensor to have size ", size, " at dimension ", dim,
    ", but got size ", t->size(dim), " for ", t,
    " (while checking arguments for ", c, ")");
}

// 检查所有张量是否相同的辅助函数，执行指定操作函数 fn
static void checkAllSame(CheckedFrom c, ArrayRef<TensorArg> tensors, void(*fn)(CheckedFrom, const TensorArg&, const TensorArg&)) {
  // 初始化第一个张量指针为 nullptr
  const TensorArg* t0 = nullptr;
  // 遍历传入的张量数组
  for (auto& t : tensors) {
    // 如果当前张量未定义，则跳过
    if (!t->defined()) continue;
    // 如果第一个张量指针不为空，则执行操作函数 fn，传入第一个张量和当前张量
    if (t0 != nullptr) {
      fn(c, *t0, t);
    } else {
      // 否则，将第一个张量指针指向当前张量
      t0 = &t;
    }
  }
}

// 检查两个张量的尺寸是否相同
void checkSameSize(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  // 使用 TORCH_CHECK 来检查两个张量的尺寸是否相同，并输出错误信息
  TORCH_CHECK(
    t1->sizes().equals(t2->sizes()),
    "Expected tensor for ", t1, " to have same size as tensor for ", t2,
    "; but ", t1->sizes(), " does not equal ", t2->sizes(),
    " (while checking arguments for ", c, ")");
}

// 检查数组中所有张量的尺寸是否相同
void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  // 调用 checkAllSame 函数，传入检查尺寸的辅助函数 checkSameSize
  checkAllSame(c, tensors, checkSameSize);
}

// 检查张量的元素个数是否为指定值
void checkNumel(CheckedFrom c, const TensorGeometryArg& t, int64_t numel) {
  // 使用 TORCH_CHECK 来检查张量的元素个数是否符合预期，并输出错误信息
  TORCH_CHECK(
    t->numel() == numel,
    "Expected tensor for ", t, " to have ", numel,
    " elements; but it actually has ", t->numel(), " elements",
    " (while checking arguments for ", c, ")");
}

// 检查两个张量的元素个数是否相同
void checkSameNumel(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  // 使用 TORCH_CHECK 来检查两个张量的元素个数是否相同，并输出错误信息
  TORCH_CHECK(
    t1->numel() == t2->numel(),
    "Expected tensor for ", t1,
    " to have same number of elements as tensor for ", t2, "; but ",
    t1->numel(), " does not equal ", t2->numel(),
    " (while checking arguments for ", c, ")");
}

// 检查数组中所有张量的元素个数是否相同
void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  // 调用 checkAllSame 函数，传入检查元素个数的辅助函数 checkSameNumel
  checkAllSame(c, tensors, checkSameNumel);
}

// 检查两个张量是否位于相同的 GPU 上
void checkSameGPU(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  // 如果其中任意一个张量位于 CPU，则输出错误信息
  if (t1->is_cpu() || t2->is_cpu()) {
    std::ostringstream oss;
    if (t1->is_cpu()) {
      oss << "Tensor for " << t1 << " is on CPU, ";
    }
    if (t2->is_cpu()) {
      oss << "Tensor for " << t2 << " is on CPU, ";
    }
    oss << "but expected " << ((!t1->is_cpu() && !t2->is_cpu()) ? "them" : "it")
        << " to be on GPU (while checking arguments for " << c << ")";
    // 输出错误信息并抛出异常
    AT_ERROR(oss.str());
  }
  // 使用 TORCH_CHECK 来检查两个张量是否位于相同的 GPU 上，并输出错误信息
  TORCH_CHECK(
    t1->get_device() == t2->get_device(),
    "Expected tensor for ", t1, " to have the same device as tensor for ", t2,
    "; but device ", t1->get_device(), " does not equal ", t2->get_device(),
    " (while checking arguments for ", c, ")");
}

// 检查数组中所有张量是否位于相同的 GPU 上
void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  // 调用 checkAllSame 函数，传入检查 GPU 的辅助函数 checkSameGPU
  checkAllSame(c, tensors, checkSameGPU);
}

// 检查两个张量是否具有相同的数据类型
void checkSameType(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  // 使用 TORCH_CHECK 来检查两个张量的数据类型是否相同，并输出错误信息
  TORCH_CHECK(
    t1->options().type_equal(t2->options()),
    "Expected tensor for ", t1, " to have the same type as tensor for ", t2,
    "; but type ", t1->toString(), " does not equal ", t2->toString(),
    " (while checking arguments for ", c, ")");
}

// 检查张量是否具有指定的标量类型
void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType ty) {
  // 使用 TORCH_CHECK 来检查张量的标量类型是否符合预期，并输出错误信息
  TORCH_CHECK(
    t->scalar_type() == ty,
    "Expected tensor for ", t, " to have scalar type ", toString(ty),
    "; but got ", t->toString(), " instead (while checking arguments for ", c,
    ")");
}
// 检查给定张量参数的标量类型是否在指定列表中
void checkScalarTypes(CheckedFrom c, const TensorArg& t,
                      at::ArrayRef<ScalarType> l) {
    // 如果张量参数的标量类型不在列表 l 中
    if (std::find(l.begin(), l.end(), t->scalar_type()) == l.end()) {
      // 构造一个字符串流 oss 用于生成错误消息
      std::ostringstream oss;
      // 添加错误消息的前缀
      oss << "Expected tensor for " << t << " to have one of the following "
          << "scalar types: ";
      size_t i = 0;
      // 遍历标量类型列表 l
      for (auto ty : l) {
        if (i != 0) {
          oss << ", ";
        }
        // 将标量类型转换为字符串并添加到错误消息中
        oss << toString(ty);
        i++;
      }
      // 添加张量实际标量类型到错误消息中
      oss << "; but got " << t->toString()
          << " instead (while checking arguments for " << c << ")";
      // 抛出带有 oss 字符串流内容的错误
      AT_ERROR(oss.str());
    }
}

// 检查给定张量参数数组中的所有张量是否具有相同的类型
void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  // 调用 checkAllSame 函数，检查所有张量是否具有相同的类型
  checkAllSame(c, tensors, checkSameType);
}

// 检查两个张量参数是否具有相同的维度
void checkSameDim(CheckedFrom c, const TensorGeometryArg& t1, const TensorGeometryArg& t2) {
  TORCH_CHECK(
    // 检查第一个张量的维度是否等于第二个张量的维度
    t1->dim() == t2->dim(),
    "Expected tensor for ", t1, " to have the same dimension as tensor for ",
    t2, "; but ", t1->dim(), " does not equal ", t2->dim(),
    " (while checking arguments for ", c, ")");
}

// 检查给定张量参数是否已定义（非空）
void checkDefined(CheckedFrom c, const TensorArg& t) {
  TORCH_CHECK(
    // 检查张量是否已定义（非空）
    t->defined(),
    "Expected tensor for ", t, " to be non-null, but it was undefined ",
    " (while checking arguments for ", c, ")");
}

// 检查给定张量参数数组中的所有张量是否均已定义（非空）
void checkAllDefined(CheckedFrom c, ArrayRef<TensorArg> ts) {
  // 遍历所有张量参数，逐个调用 checkDefined 函数检查是否已定义
  for (auto t : ts) {
    checkDefined(c, t);
  }
}

// 检查给定张量是否具有指定的后端类型
static void checkBackend(CheckedFrom c, const Tensor& t, Backend backend) {
  TORCH_CHECK(
    // 检查张量是否未定义或其后端类型与指定的后端类型相同
    !t.defined() || t.options().backend() == backend,
    "Expected tensor to have ", toString(backend),
    " Backend, but got tensor with ", toString(t.options().backend()), " Backend ",
    "(while checking arguments for ", c, ")");
}

// 检查给定张量数组中的所有张量是否具有指定的后端类型
void checkBackend(CheckedFrom c, at::ArrayRef<Tensor> tensors, at::Backend backend) {
  // 遍历所有张量，逐个调用 checkBackend 函数检查后端类型
  for (auto &t : tensors) {
    checkBackend(c, t, backend);
  }
}

// 检查给定张量是否具有指定的设备类型
static void checkDeviceType(CheckedFrom c, const Tensor& t, DeviceType device_type) {
  TORCH_CHECK(
      // 检查张量是否未定义或其设备类型与指定的设备类型相同
      !t.defined() || t.device().type() == device_type,
      "Expected tensor to have ", device_type,
      " DeviceType, but got tensor with ", t.device().type(), " DeviceType ",
      "(while checking arguments for ", c, ")");
}

// 检查给定张量数组中的所有张量是否具有指定的设备类型
void checkDeviceType(CheckedFrom c, at::ArrayRef<Tensor> tensors, at::DeviceType device_type) {
  // 遍历所有张量，逐个调用 checkDeviceType 函数检查设备类型
  for (auto &t : tensors) {
    checkDeviceType(c, t, device_type);
  }
}

// 检查给定张量是否具有指定的布局类型
void checkLayout(CheckedFrom c, const Tensor& t, Layout layout) {
  TORCH_CHECK(
    // 检查张量是否未定义或其布局类型与指定的布局类型相同
    !t.defined() || t.layout() == layout,
    "Expected tensor to have ", layout,
    " Layout, but got tensor with ", t.layout(), " Layout ",
    "(while checking arguments for ", c, ")");
}

// 检查给定张量数组中的所有张量是否具有指定的布局类型
void checkLayout(CheckedFrom c, at::ArrayRef<Tensor> tensors, at::Layout layout) {
  // 遍历所有张量，逐个调用 checkLayout 函数检查布局类型
  for (auto &t : tensors) {
    checkLayout(c, t, layout);
  }
}

// 返回张量的数据指针，如果张量未定义，则返回空指针
void * maybe_data_ptr(const Tensor& tensor) {
  // 如果张量已定义，则返回其数据指针；否则返回空指针
  return tensor.defined() ? (void *)tensor.data_ptr() : nullptr;
}
// 返回一个指向数据的指针，如果张量已定义，则返回其数据指针；否则返回空指针
void * maybe_data_ptr(const TensorArg& tensor) {
  return tensor->defined() ? (void *)tensor->data_ptr() : nullptr;
}

// 检查张量的特定维度大小
void check_dim_size(
    const Tensor& tensor,
    int64_t dim,
    int64_t dim_size,
    int64_t size) {
  /* 检查张量的维度大小 */
  TORCH_CHECK(
      tensor.dim() == dim && tensor.size(dim_size) == size,
      "Expected a tensor of dimension ",
      dim,
      " and tensor.size[",
      dim_size,
      "] == ",
      size,
      " but got: dimension ",
      tensor.dim(),
      " and tensor.size[",
      dim_size,
      "] = ",
      tensor.size(dim_size));
}

namespace detail {

// 返回默认的步长向量，用于给定大小向量的张量
std::vector<int64_t> defaultStrides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for(size_t i = sizes.size(); i > 0; --i) {
    strides[i-1] = stride;
    stride *= sizes[i-1];
  }
  return strides;
}

// 计算张量的步长向量，以匹配新的形状
// 1. 将 `oldshape` 分成维度的块，其中每个块中的维度是“连续”的，即 oldstride[i] = oldshape[i+1] * oldstride[i+1]
// 2. `newshape` 必须能够分成与 `oldshape` 相同数量的块，其中每个 newshape 的块具有与相应 oldshape 的块相匹配的“numel”，即子空间的数量。
// 用于 DimVector 和 IntArrayRef 的模板化用例，参见下面 computeStride() 的重载。
template <typename ResultVec, typename NewShapeVec, typename Numel>
inline std::optional<ResultVec> computeStride_impl(
    const NewShapeVec& oldshape,
    const NewShapeVec& oldstride,
    const NewShapeVec& newshape,
    ResultVec toResult(const NewShapeVec&)
) {
  if (oldshape.empty()) {
    return ResultVec(newshape.size(), 1);
  }

  // 当 numel() == 0 时，步长是任意的；
  // 为了匹配 NumPy 的行为，如果尺寸匹配，则复制步长，否则像调整大小一样使用步长。
  const Numel numel = c10::multiply_integers(oldshape);
  bool zero_numel = TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(numel, 0));
  if (zero_numel && oldshape.equals(newshape)) {
    return toResult(oldstride);
  }

  ResultVec newstride(newshape.size());
  if (zero_numel) {
    for (int64_t view_d = newshape.size() - 1; view_d >= 0; view_d--) {
      if (view_d == (int64_t)(newshape.size() - 1)) {
        newstride[view_d] = 1;
      } else {
        newstride[view_d] =
          std::max<Numel>(newshape[view_d+1], Numel(1)) * newstride[view_d+1];
      }
    }
    return newstride;
  }

  int64_t view_d = (int64_t)newshape.size() - 1;
  // 块中每个子空间的步长
  Numel chunk_base_stride = oldstride.back();
  // 当前块的 numel
  Numel tensor_numel = 1;
  Numel view_numel = 1;
  for (int64_t tensor_d = oldshape.size() - 1; tensor_d >= 0; tensor_d--) {
    tensor_numel *= oldshape[tensor_d];
    // 如果当前张量大小的块结束，检查视图
    if ((tensor_d == 0) ||
        (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(oldshape[tensor_d - 1], 1)) &&
         oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      // 当视图维度大于等于0且满足以下条件时执行循环
      while (view_d >= 0 &&
            (TORCH_GUARD_SIZE_OBLIVIOUS(sym_lt(view_numel, tensor_numel)) || TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(newshape[view_d], 1)))) {
        // 设置新的步长为视图元素数量乘以块的基础步长
        newstride[view_d] = view_numel * chunk_base_stride;
        // 更新视图元素数量
        view_numel *= newshape[view_d];
        // 减少视图维度
        view_d--;
      }
      // 如果视图元素数量不等于张量元素数量，则返回空值
      if (view_numel != tensor_numel) {
        return c10::nullopt;
      }
      // 如果张量维度大于0，则重置块的基础步长、张量元素数量和视图元素数量
      if (tensor_d > 0) {
        chunk_base_stride = oldstride[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  // 如果视图维度不等于-1，则返回空值
  if (view_d != -1) {
    return c10::nullopt;
  }
  // 返回新的步长数组
  return newstride;
}

std::optional<std::vector<int64_t>> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape) {
  auto toResult = [](const IntArrayRef& a) { return a.vec(); };
  // 调用通用的 computeStride_impl 函数，使用 std::vector<int64_t> 作为结果类型
  return computeStride_impl<std::vector<int64_t>, IntArrayRef, int64_t>(oldshape, oldstride, newshape, toResult);
}

std::optional<SymDimVector> computeStride(
    c10::SymIntArrayRef oldshape,
    c10::SymIntArrayRef oldstride,
    c10::SymIntArrayRef newshape) {
  auto toResult = [](const SymIntArrayRef& a) { return SymDimVector(a); };
  // 调用通用的 computeStride_impl 函数，使用 SymDimVector 作为结果类型
  return computeStride_impl<SymDimVector, c10::SymIntArrayRef, c10::SymInt>(oldshape, oldstride, newshape, toResult);
}

std::optional<DimVector> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    const DimVector& newshape) {
  auto toResult = [](const IntArrayRef& a) { return DimVector(a); };
  // 调用通用的 computeStride_impl 函数，使用 DimVector 作为结果类型
  return computeStride_impl<DimVector, IntArrayRef, int64_t>(oldshape, oldstride, newshape, toResult);
}

}  // namespace detail
}  // namespace at


注释：
```