# `.\pytorch\aten\src\ATen\native\TensorTransformations.cpp`

```
// 定义宏以限制只使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含张量转换的头文件
#include <ATen/native/TensorTransformations.h>
// 包含 flip_stub 函数的头文件，用于索引操作
#include <ATen/native/IndexKernel.h>

// 包含并行处理相关的头文件
#include <ATen/Parallel.h>
// 包含张量迭代器相关的头文件
#include <ATen/TensorIterator.h>
// 包含多维度操作的辅助函数
#include <ATen/WrapDimUtilsMulti.h>
// 包含维度向量的头文件
#include <ATen/core/DimVector.h>
// 包含异常处理相关的头文件
#include <c10/util/Exception.h>
// 包含范围操作的头文件
#include <c10/util/irange.h>

// 如果未定义每个运算符的单独头文件，则包含以下函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了每个运算符的单独头文件，则包含以下函数头文件
#else
#include <ATen/ops/atleast_1d_native.h>
#include <ATen/ops/atleast_2d_native.h>
#include <ATen/ops/atleast_3d_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chalf_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/flip_native.h>
#include <ATen/ops/fliplr_native.h>
#include <ATen/ops/flipud_native.h>
#include <ATen/ops/roll_native.h>
#include <ATen/ops/rot90_native.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

// 包含标准库中的算法
#include <algorithm>
// 包含实用工具
#include <utility>
// 包含向量
#include <vector>

// 命名空间 at::native 中的函数定义
namespace at::native {

// flip 函数的实现，翻转给定的张量 self 沿指定的维度 dims
Tensor flip(const Tensor& self, IntArrayRef dims) {
  // 获取张量 self 的总维度数
  const int64_t total_dims = self.dim();
  // 将 dims 转换为位集，检查并包装维度，并确保没有重复的维度
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);

  // 创建一个与 self 相同尺寸和数据类型的空张量 out_tensor
  Tensor out_tensor = at::empty_like(self, MemoryFormat::Preserve);

  // 计算需要处理的维度数 n
  int n = 0;
  // 获取 self 的步长 strides
  auto strides = DimVector(self.strides());
  // 遍历所有维度
  for (const auto i : c10::irange(total_dims)) {
    // 如果 flip_dims_b 中标记了当前维度需要翻转，并且 self 在当前维度上的尺寸大于 1 且步长不为 0
    if (flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      // 增加需要处理的维度计数，并将该维度的步长设为 0
      n++;
      strides[i] = 0;
    }
  }

  // 如果没有需要处理的维度或者 self 的元素个数小于等于 1，直接将 self 复制到 out_tensor 并返回
  if (n == 0 || self.numel() <= 1) {
    out_tensor.copy_(self);
    return out_tensor;
  }

  // 根据修改后的步长创建一个虚拟的张量 restrided_self，用于在迭代器中防止合并翻转的维度
  const auto restrided_self = self.as_strided(self.sizes(), strides);
  // 创建张量迭代器 iter
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // 不检查内存重叠
    .check_all_same_dtype(false)   // 不检查所有输入是否具有相同的数据类型
    .declare_static_dtype_and_device(self.scalar_type(), self.device())  // 声明静态的数据类型和设备类型
    .add_output(out_tensor)        // 添加输出张量
    .add_const_input(self)         // 添加输入张量 self
    .add_const_input(restrided_self)  // 添加虚拟张量 restrided_self
    .build();                      // 构建迭代器

  // 获取 iter 的数据指针
  auto* data = reinterpret_cast<char*>(iter.data_ptr(0));
  // 获取 iter 的形状尺寸
  const auto sizes = iter.shape();
  // 获取 iter 的步长（以字节为单位）
  auto strides_bytes = DimVector(iter.strides(0));
  // 获取 iter 的输入 self 的步长
  const auto strides_self = iter.strides(1);
  // 获取 iter 的虚拟输入 restrided_self 的步长
  const auto strides_dummy = iter.strides(2);

  // 为了理解这种转换，请考虑一个三维立方体。
  //   - 数据指针指向立方体的最低左侧顶点
  //   - 步长告诉我们如何在每个维度移动，
  //     即 data + stride[i] 在维度 i 中前进一个元素
  // 要翻转一个维度：
  //   - 将指针移动到立方体的对角顶点
  //   - 在相反的方向迭代（反转步长）

  // 遍历迭代器的所有维度
  for (const auto i : c10::irange(iter.ndim())) {
    // 我们知道一个维度有零步长而 self[i] 没有，正如我们上面定义的
    // 如果 strides_dummy[i] 等于 0 而 strides_self[i] 不等于 0，这可能是因为 strides_self[i] 本身为 0 而不是我们手动设置的情况。
    // 我们不希望在这种情况下执行任何操作。
    if (strides_dummy[i] == 0 && strides_self[i] != 0) {
      // 将 data 增加到末尾元素的位置
      data += strides_bytes[i] * (sizes[i]-1);
      // 将 strides_bytes[i] 取反
      strides_bytes[i] *= -1;
    }
  }
  // 设置迭代器的参数步长为新的 strides_bytes 数组
  iter._unsafe_set_arg_strides(0, strides_bytes);
  // 将数据指针设置为指向 data 的重新解释地址
  iter._unsafe_set_arg_data(0, reinterpret_cast<void*>(data));

  // 调用 flip_stub 函数翻转张量的内容
  flip_stub(iter.device_type(), iter, self.is_quantized());

  // 返回翻转后的输出张量
  return out_tensor;
`
}

Tensor roll(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) { // 定义名为 roll 的函数，用于实现张量的滚动操作，接受张量 self 和整数数组 shifts、dims 作为参数
  if (dims.size() != 1 || shifts.size() != 1) { // 如果 dims 和 shifts 的大小不等于 1
    return roll_common(self, shifts, dims); // 调用 roll_common 函数处理滚动操作
  }
  // 避免除以零错误
  if (self.numel() == 0) { // 如果张量元素数为零
    return self.clone(at::MemoryFormat::Preserve); // 返回保留内存格式的自身克隆
  }
  int64_t dim = dims[0]; // 获取维度 dim
  int64_t size = self.size(dim); // 获取维度 dim 的大小
  int64_t start = (size - shifts[0]) % size; // 计算起始位置 start
  // 修正 C++ 中 % 运算符对负数的行为，以与 Python 一致
  if (start < 0) {
    start = start + size;
  }
  auto t0 = self.narrow(dim, start, size-start); // 对维度 dim 进行窄化操作，获取子张量 t0
  auto t1 = self.narrow(dim, 0, start); // 对维度 dim 进行窄化操作，获取子张量 t1
  return at::cat({std::move(t0), std::move(t1)}, dim); // 在维度 dim 上连接 t0 和 t1，并返回结果
}

Tensor rot90(const Tensor& self, int64_t k, IntArrayRef dims) { // 定义名为 rot90 的函数，用于实现张量的旋转操作，接受张量 self、整数 k 和整数数组 dims 作为参数
  const int64_t total_dims = self.dim(), total_rot_dims = dims.size(); // 获取张量的总维度数和旋转维度数

  TORCH_CHECK(total_rot_dims == 2, // 断言旋转维度数为 2
    "expected total rotation dims == 2, but got dims = ", total_rot_dims);

  TORCH_CHECK(total_dims >= 2, // 断言张量总维度数至少为 2
    "expected total dims >= 2, but got total dims = ", total_dims);

  TORCH_CHECK(dims[0] != dims[1] && std::abs(dims[0] - dims[1]) != total_dims, // 断言旋转维度不相同且它们的绝对差不等于张量总维度数
    "expected rotation dims to be different, but got dim0 = ", dims[0],
    " and dim1 = ", dims[1]);

  // 检查维度范围
  TORCH_CHECK(dims[0] < total_dims && dims[0] >= -total_dims, // 断言旋转维度 dim0 在有效范围内
    "Rotation dim0 out of range, dim0 = ", dims[0]);

  TORCH_CHECK(dims[1] < total_dims && dims[1] >= -total_dims, // 断言旋转维度 dim1 在有效范围内
    "Rotation dim1 out of range, dim1 = ", dims[1]);

  // 处理负 k 的模运算
  k = (4 + (k % 4)) % 4;

  switch(k) {
    case 1:
      return self.flip({dims[1]}).transpose_(dims[0], dims[1]); // 对维度 dims[1] 进行翻转并转置维度 dims[0] 和 dims[1]
    case 2:
      return self.flip(dims); // 对指定维度数组 dims 进行翻转操作
    case 3:
      return self.flip({dims[0]}).transpose_(dims[0], dims[1]); // 对维度 dims[0] 进行翻转并转置维度 dims[0] 和 dims[1]
    default:
      return self.clone(at::MemoryFormat::Contiguous); // 返回张量的连续内存格式的克隆
  }
}

Tensor fliplr(const Tensor& self) { // 定义名为 fliplr 的函数，用于实现张量的左右翻转操作，接受张量 self 作为参数
  TORCH_CHECK(self.dim() >= 2, "Input must be >= 2-d."); // 断言输入张量至少为二维

  return self.flip({1}); // 对维度 1 进行翻转操作
}

Tensor flipud(const Tensor& self) { // 定义名为 flipud 的函数，用于实现张量的上下翻转操作，接受张量 self 作为参数
  TORCH_CHECK(self.dim() >= 1, "Input must be >= 1-d."); // 断言输入张量至少为一维

  return self.flip({0}); // 对维度 0 进行翻转操作
}

Tensor atleast_1d(const Tensor& self) { // 定义名为 atleast_1d 的函数，用于确保张量至少为一维，接受张量 self 作为参数
  switch (self.dim()) {
    case 0:
      return self.reshape({1}); // 若张量为零维，则改变形状为一维
    default:
      return self; // 其他情况直接返回自身
  }
}

std::vector<Tensor> atleast_1d(TensorList tensors) { // 定义名为 atleast_1d 的函数重载，用于处理张量列表，确保每个张量至少为一维，接受张量列表 tensors 作为参数
  std::vector<Tensor> result(tensors.size()); // 创建与输入张量列表大小相同的结果向量
  auto transform_lambda = [](const Tensor& input) -> Tensor { // 定义转换 lambda 函数，确保每个输入张量至少为一维
    return at::native::atleast_1d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda); // 对张量列表应用转换 lambda 函数
  return result; // 返回处理后的结果向量
}

Tensor atleast_2d(const Tensor& self) { // 定义名为 atleast_2d 的函数，用于确保张量至少为二维，接受张量 self 作为参数
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1}); // 若张量为零维，则改变形状为二维
    case 1: {
      return self.unsqueeze(0); // 若张量为一维，则在第 0 维上增加维度
    }
    default:
      return self; // 其他情况直接返回自身
  }
}

std::vector<Tensor> atleast_2d(TensorList tensors) { // 定义名为 atleast_2d 的函数重载，用于处理张量列表，确保每个张量至少为二维，接受张量列表 tensors 作为参数
  std::vector<Tensor> result(tensors.size()); // 创建与输入张量列表大小相同的结果向量
  auto transform_lambda = [](const Tensor& input) -> Tensor { // 定义转换 lambda 函数，确保每个输入张量至少为二维
    return at::native::atleast_2d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda); // 对张量列表应用转换 lambda 函数
  return result; // 返回处理后的结果向量
}
    // 调用PyTorch的native命名空间下的atleast_2d函数，将输入张量转换为至少二维的形式
    return at::native::atleast_2d(input);
  };
  // 使用std::transform算法，对tensors容器中的每个元素应用transform_lambda函数，并将结果存储到result容器中
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  // 返回经过std::transform处理后的结果容器result，其中包含了每个输入张量经过transform_lambda函数转换后的结果
  return result;
}

Tensor atleast_3d(const Tensor& self) {
  // 根据输入张量的维度进行处理，确保至少为三维
  switch (self.dim()) {
    case 0:
      // 对于0维张量，reshape为1x1x1的三维张量
      return self.reshape({1, 1, 1});
    case 1: {
      // 对于1维张量，在两端各增加一个维度，变为1xNx1的三维张量
      return self.unsqueeze(0).unsqueeze(-1);
    }
    case 2: {
      // 对于2维张量，在最后一个维度上增加一个维度，变为NxMx1的三维张量
      return self.unsqueeze(-1);
    }
    default:
      // 其他情况下，保持张量不变，返回原始张量
      return self;
  }
}

std::vector<Tensor> atleast_3d(TensorList tensors) {
  // 对输入的张量列表进行处理，确保每个张量至少为三维
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    // 调用atleast_3d函数处理单个张量，确保其至少为三维
    return at::native::atleast_3d(input);
  };
  // 使用transform将输入张量列表中的每个张量都转换为至少为三维的张量
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  // 返回处理后的结果张量列表
  return result;
}

Tensor chalf(const Tensor& self, std::optional<MemoryFormat> memory_format) {
  // 将输入张量转换为复数类型的半精度张量
  return self.to(kComplexHalf, false, false, memory_format);
}

DEFINE_DISPATCH(flip_stub);

} // namespace at::native
```