# `.\pytorch\aten\src\ATen\native\TensorAdvancedIndexingUtils.h`

```py
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native {

// 定义一个匿名命名空间，用于封装本地函数和静态变量，避免全局作用域污染
namespace {
// 将一组张量的形状转换为字符串表示
static std::string shapes_as_str(TensorList tensors) {
  // 创建一个输出字符串流
  std::ostringstream os;
  // 是否是第一个张量的标志
  bool first = true;
  // 遍历张量列表
  for (auto& tensor : tensors) {
    // 如果张量被定义
    if (tensor.defined()) {
      // 如果不是第一个张量，则在形状字符串之前添加逗号和空格
      if (!first) {
        os << ", ";
      }
      // 将张量的形状追加到输出流中
      os << tensor.sizes();
      first = false;
    }
  }
  // 返回形状字符串
  return os.str();
}
} // anonymous namespace

// 检查是否可以分派到 masked_fill 操作，并返回操作结果和生成的掩码张量
static std::tuple<bool, Tensor> canDispatchToMaskedFill(const Tensor& self, const torch::List<std::optional<at::Tensor>>& indices,
const Tensor& value){
  // 检查值张量是否只有一个元素且在 CPU 设备上
  if (!(value.numel() == 1 && value.device().is_cpu())){
    // 如果条件不满足，则返回不支持分派和空张量的元组
    return std::make_tuple(false, Tensor());
  }
  // 初始化索引数和掩码张量
  int64_t num_ind = 0;
  Tensor mask;
  auto self_device = self.device();
  // 遍历索引列表
  for (const std::optional<Tensor>& i: indices) {
    // 如果索引未定义或为空
    if (!i.has_value() || !(*i).defined()){
      // 增加未定义索引计数
      num_ind++;
    } else {
      // 获取索引张量
      const Tensor &index = *i;
      // 如果索引类型不是 kByte 或 kBool，或者设备与 self 不匹配，或者 mask 已定义
      if ((index.scalar_type() != kByte && index.scalar_type() != kBool) ||
          index.device() != self_device || mask.defined()){
        // 返回不支持分派和空张量的元组
        return std::make_tuple(false, Tensor());
      } else {
        // 将 mask 设置为当前索引张量
        mask = index;
        // 验证掩码张量的形状与被索引张量的形状是否匹配
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = num_ind + j;
          // 检查索引张量在指定维度上的形状是否与被索引张量的形状相匹配
          TORCH_CHECK_INDEX(index.size(j) == self.size(srcIdx), "The shape of the mask ", index.sizes(), " at index ", j,
  " does not match the shape of the indexed tensor ", self.sizes(), " at index ", srcIdx);
        }
        // 更新索引数，增加掩码张量的维度数量
        num_ind += mask.ndimension();
      }
    }
  }
  // 如果索引数小于被索引张量的维度数
  for (C10_UNUSED const auto i : c10::irange(num_ind, self.ndimension())) {
    // 在掩码张量的最后一个维度上增加一个维度
    mask = mask.unsqueeze(-1);
  }
  // 返回支持分派和生成的掩码张量的元组
  return std::make_tuple(true, mask);
}

// 创建并返回高级索引信息结构体
static AdvancedIndex make_info(Tensor self, IOptTensorListRef orig) {
  // 检查索引张量类型，并允许整数类型的索引
  checkIndexTensorTypes(orig, /*allow_int*/ true);
  // 将原始索引张量扩展为 LongTensor 类型的索引列表
  auto indices = expandTensors(self, orig);
  // 尝试将所有索引张量进行广播扩展
  try {
    indices = expand_outplace(indices);
  } catch (std::exception& e) {
    // 如果广播失败，则抛出错误信息
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
                   " with shapes ", shapes_as_str(indices));
  }
  // 如果索引列表长度小于被索引张量的维度数
  while (indices.size() < (size_t)self.dim()) {
    // 添加空张量，使索引列表与被索引张量的维度数相匹配
    indices.emplace_back();
  }
  // 如果非空索引不是全部相邻的，则对 self 和 indices 进行转置，使其前置相邻
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  // 确保索引张量与被索引张量在相同的设备上
  for (auto & indice : indices) {
    if (indice.defined() && indice.device() != self.device()) {
      indice = indice.to(self.device());
    }
  }
  // 如果索引张量已定义且数据类型为 kInt，则将其转换为 kLong
  for (auto & indice : indices) {
    if (indice.defined() && indice.dtype() == at::kInt) {
      indice = indice.to(at::kLong);
    }
  }
  // 返回创建的高级索引信息结构体
  return {self, std::move(indices)};
}

} // namespace at::native
    }
  }



    # 结束嵌套的两个控制流语句（循环或条件语句），返回到外层控制流的结尾
    # 这里的代码片段没有具体语境，假设它是在一个函数或方法内部
    # 在这里可能表示一个条件语句或循环的结束
    # 不过缺少上下文的话很难确切地判断其用途
  # 返回一个 AdvancedIndex 类的实例，使用当前对象和指定的 indices 参数
  return AdvancedIndex(self, indices);


这段代码片段中，第一个注释解释了两个控制流语句（循环或条件语句）的结束，而第二个注释则解释了整个函数或方法的返回语句。
}

// 结束 at::native 命名空间
} // namespace at::native
```