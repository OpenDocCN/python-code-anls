# `.\pytorch\torch\csrc\utils\tensor_apply.cpp`

```py
// 包含 Torch 库中的 tensor_apply.h 文件

#include <torch/csrc/utils/tensor_apply.h>

// 包含 ATen 库中的 ExpandUtils.h 和 TensorUtils.h 文件
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
// 包含 c10 库中的 irange.h 文件
#include <c10/util/irange.h>

// 包含 Torch 中的 Exceptions.h 和 python_numbers.h, python_scalars.h 文件
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_scalars.h>

// 使用 at 命名空间
using namespace at;

// 定义 torch::utils 命名空间
namespace torch::utils {

// 定义 StridedData 结构体，用于存储张量的数据、步幅和元素大小
struct StridedData {
  // 构造函数，初始化数据成员
  StridedData(const Tensor& tensor)
      : data(tensor.data_ptr()),
        strides(tensor.strides()),
        elementSize(tensor.element_size()) {}

  // 数据指针
  void* data;
  // 步幅数组的引用
  IntArrayRef strides;
  // 元素大小
  int64_t elementSize;

  // 在指定维度上进行步进
  void step(int dim) {
    data = (char*)data + (strides[dim] * elementSize);
  }
};

// 递归应用函数，对张量进行逐元素操作
template <size_t N>
static void recursive_apply(
    IntArrayRef sizes,
    ScalarType scalarType,
    int64_t dim,
    PyObject* fn,
    std::array<StridedData, N> strided_data) {
  // 获取张量的维度数
  int64_t ndim = static_cast<int64_t>(sizes.size());
  // 如果当前维度等于张量维度数，则递归结束，调用 Python 函数并返回
  if (dim == ndim) {
    // 创建 Python 元组作为函数参数
    auto args = THPObjectPtr(PyTuple_New(N));
    if (!args)
      throw python_error();
    // 将每个张量元素转换为 Python 对象，并设置到元组中
    for (const auto i : c10::irange(N)) {
      PyObject* arg = load_scalar(strided_data[i].data, scalarType);
      if (!arg)
        throw python_error();
      PyTuple_SET_ITEM(args.get(), i, arg);
    }
    // 调用 Python 函数
    auto ret = THPObjectPtr(PyObject_CallObject(fn, args.get()));
    if (!ret)
      throw python_error();
    // 将函数返回值存储回张量的第一个元素位置
    store_scalar(strided_data[0].data, scalarType, ret.get());
    return;
  }

  // 获取当前维度的大小
  auto n = sizes[dim];
  // 递归遍历当前维度上的每个元素
  for (const auto i : c10::irange(n)) {
    (void)i; // 抑制未使用变量的警告
    // 递归调用，处理下一个维度
    recursive_apply(sizes, scalarType, dim + 1, fn, strided_data);
    // 对每个 StridedData 对象在当前维度上进行步进
    for (auto& td : strided_data) {
      td.step(dim);
    }
  }
}

// apply_ 函数，对张量应用指定的 Python 函数
const Tensor& apply_(const Tensor& self, PyObject* fn) {
  // 如果张量为元信息张量，直接返回
  if (self.is_meta()) {
    return self; // Just skip
  }
  // 检查张量是否在 CPU 上
  TORCH_CHECK_TYPE(
      self.device().is_cpu(), "apply_ is only implemented on CPU tensors");
  // 获取张量的标量类型
  auto scalarType = self.scalar_type();
  // 调用 recursive_apply 函数，处理张量中的每个元素
  recursive_apply<1>(self.sizes(), scalarType, 0, fn, {{self}});
  // 返回处理后的张量
  return self;
}

// map_ 函数，对两个张量中的对应元素应用指定的 Python 函数
const Tensor& map_(const Tensor& self, const Tensor& other_, PyObject* fn) {
  // 检查两个张量是否具有相同的类型和设备
  TORCH_CHECK_TYPE(
      other_.options().type_equal(self.options()),
      "map_: expected ",
      self.toString(),
      " for 'other' (got ",
      other_.toString(),
      ")");
  // 如果张量为元信息张量，直接返回
  if (self.is_meta()) {
    return self; // Just skip
  }
  // 检查张量是否在 CPU 上
  TORCH_CHECK_TYPE(
      self.device().is_cpu(), "map_ is only implemented on CPU tensors");
  // 在 inplace 扩展操作中使用 other_ 张量
  c10::MaybeOwned<Tensor> other = expand_inplace(self, other_, "map_");
  // 获取张量的标量类型
  auto scalarType = self.scalar_type();
  // 调用 recursive_apply 函数，处理两个张量中的每对元素
  recursive_apply<2>(self.sizes(), scalarType, 0, fn, {{self, *other}});
  // 返回处理后的张量
  return self;
}

// map2_ 函数，对三个张量中的对应元素应用指定的 Python 函数
const Tensor& map2_(
    const Tensor& self,
    const Tensor& x_,
    const Tensor& y_,
    // 检查张量 `x_` 的数据类型是否与 `self` 相同
    TORCH_CHECK_TYPE(
        x_.options().type_equal(self.options()),
        "map2_: expected ",
        self.toString(),
        " for argument 'x' (got ",
        x_.toString(),
        ")");
    // 检查张量 `y_` 的数据类型是否与 `self` 相同
    TORCH_CHECK_TYPE(
        y_.options().type_equal(self.options()),
        "map2_: expected ",
        self.toString(),
        " for argument 'y' (got ",
        y_.toString(),
        ")");
    // 如果 `self` 是元数据张量，则直接返回自身，跳过后续操作
    if (self.is_meta()) {
        return self; // Just skip
    }
    // 检查 `self`、`x_`、`y_` 是否均在 CPU 设备上，因为 `map2_` 目前仅支持 CPU 张量
    TORCH_CHECK_TYPE(
        (self.device().is_cpu() && x_.device().is_cpu() && y_.device().is_cpu()),
        "map2_ is only implemented on CPU tensors");
    // 扩展 `self`、`x_`、`y_` 张量，以确保它们具有相同的形状和大小
    auto others = expand_inplace(self, x_, y_, "map2_");
    // 获取 `self` 张量的标量类型
    auto scalarType = self.scalar_type();
    // 对 `self`、`x_`、`y_` 张量进行递归应用函数 `fn`
    recursive_apply<3>(
        self.sizes(),
        scalarType,
        0,
        fn,
        {{self, *std::get<0>(others), *std::get<1>(others)}});
    // 返回处理后的 `self` 张量
    return self;
}

} // namespace torch::utils
```