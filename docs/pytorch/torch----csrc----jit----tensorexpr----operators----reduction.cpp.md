# `.\pytorch\torch\csrc\jit\tensorexpr\operators\reduction.cpp`

```
// 引入 Torch 库中的张量表达式操作的头文件
#include <torch/csrc/jit/tensorexpr/operators/reduction.h>

// 使用 torch::jit::tensorexpr 命名空间
using namespace torch::jit::tensorexpr;

// 从索引列表中移除所有指定的轴
static std::vector<VarHandle> squeezeIndices(
    const ParameterList& indices,          // 输入的参数列表，包含索引变量
    const std::vector<size_t>& axes) {     // 需要移除的轴的索引列表
  std::vector<VarHandle> indices_squeezed; // 用于存储移除后的索引变量
  // 遍历索引列表
  for (size_t dim = 0; dim < indices.size(); ++dim) {
    // 如果当前维度不在需要移除的轴列表中，则保留该索引变量
    if (!std::count(axes.begin(), axes.end(), dim)) {
      indices_squeezed.push_back(indices[dim]);
    }
  }
  return indices_squeezed; // 返回移除指定轴后的索引变量列表
}

namespace torch {
namespace jit {
namespace tensorexpr {

// 计算张量的和操作
Tensor computeSum(
    const std::vector<ArgValue>& inputs,   // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出张量的形状表达式
    const std::vector<ExprHandle>& outputStrides,  // 输出张量的步长表达式
    const std::optional<ScalarType>& outputType,   // 输出张量的数据类型（可选）
    at::Device device) {    // 输出张量的设备类型
  std::vector<size_t> axes;   // 需要进行求和的轴列表
  bool keepdim = false;       // 是否保留求和后的维度信息

  // 获取输入张量的形状信息
  auto sizes = valueShape(inputs[0]);

  size_t rank = sizes.size(); // 张量的秩（维度数）

  // 如果输入参数超过两个
  if (inputs.size() > 2) {
    // 检查第二个参数是否为空列表
    if (auto emptyAxes = std::get_if<BufList>(&inputs[1])) {
      // 如果是空列表，则需要对所有轴进行求和
      TORCH_INTERNAL_ASSERT(emptyAxes->empty());
      axes.resize(rank);
      std::iota(axes.begin(), axes.end(), 0);
    } else if (rank > 0) {
      auto nodeAxes = std::get<IntList>(inputs[1]);
      // 规范化轴：处理负数索引，排序并去除重复项
      for (auto axis : nodeAxes) {
        axes.push_back(at::maybe_wrap_dim(axis, rank));
      }
      std::sort(axes.begin(), axes.end());
      axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    }
    // 获取是否保留维度信息的标志位
    keepdim = std::get<bool>(inputs[2]);
  } else {
    // 如果输入参数只有一个，则对所有维度进行求和
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
  }

  // 创建用于存储求和维度的表达式列表
  std::vector<ExprHandle> reductionDims;
  reductionDims.reserve(rank);
  for (size_t axis : axes) {
    reductionDims.emplace_back(sizes[axis]);
  }

  // 创建用于存储输出维度的表达式列表
  std::vector<ExprHandle> outputDims;
  // 输出维度是未被求和的维度。当 keepdim 为 true 时，为每个轴插入大小为一的维度
  for (size_t dim = 0; dim < rank; ++dim) {
    if (!std::count(axes.begin(), axes.end(), dim)) {
      outputDims.emplace_back(sizes[dim]);
    } else if (keepdim) {
      outputDims.emplace_back(1);
    }


这段代码的注释至此结束，后续部分的注释可以继续补充。
    }
  }



// 结束 lambda 表达式的定义

  return Reduce(
      "sum",
      outputDims,
      outputStrides,
      Sum(),
      [&](ParameterList& indices) {
        // 在设置 keepdim 时，移除插入的索引。
        auto indices_squeezed =
            keepdim ? squeezeIndices(indices, axes) : indices;
        // 断言轴的数量不超过压缩后的索引数量。
        TORCH_INTERNAL_ASSERT(axes.size() <= indices_squeezed.size());
        // 将最内层的索引移动到指定的轴位置：
        //   1. 首先填充最外层的索引。
        //   2. 将最内层的索引插入到正确的轴位置，按需移动外层的索引。
        std::vector<ExprHandle> indices_exprs;
        size_t i = 0;
        for (; i < indices_squeezed.size() - axes.size(); ++i) {
          indices_exprs.push_back(indices_squeezed[i]);
        }
        for (auto axis : axes) {
          indices_exprs.insert(
              indices_exprs.begin() + axis, indices_squeezed[i]);
          ++i;
        }
        // 对输入张量或常量进行索引操作。
        auto indexed = tensorOrConstant(inputs[0], indices_exprs);
        // 如果有指定输出类型，则将结果转换为该类型。
        if (outputType) {
          return Cast::make(ToDtype(*outputType), indexed);
        } else {
          return indexed;
        }
      },
      reductionDims);



// 返回 Reduce 对象，执行指定的求和操作，并进行维度规约。
} // 结束命名空间 torch

Tensor computeMean(
    const std::vector<ArgValue>& inputs,  // 输入参数列表，包含了要处理的数据以及附加参数
    const std::vector<ExprHandle>& outputShape,  // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides,  // 输出张量的步长描述
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device device) {  // 数据处理所需的设备

  Dtype dtype = kFloat;  // 默认数据类型为浮点数
  if (outputType) {  // 如果指定了输出类型，则使用指定的类型
    dtype = Dtype(*outputType);
  }
  bool keepdim = false;  // 是否保持维度，初始值为 false
  BufHandle ResultBuf("mean", outputShape, dtype);  // 创建用于结果的缓冲区
  BufHandle InputBuf = std::get<BufHandle>(inputs[0]);  // 获取输入数据的缓冲区
  std::vector<ExprHandle> extra_args;  // 额外的参数列表

  if (inputs.size() > 2) {  // 如果输入参数个数大于2
    keepdim = std::get<bool>(inputs[2]);  // 获取是否保持维度的设置
  }

  if (auto mean_dims = std::get_if<IntList>(&inputs[1])) {  // 如果第二个参数是整数列表
    extra_args = c10::fmap<ExprHandle>(*mean_dims);  // 使用列表中的维度作为额外的参数
  } else {
    // 当未指定维度参数时，对所有维度进行求均值操作
    for (int64_t idx = 0; idx < static_cast<int64_t>(InputBuf.ndim()); ++idx) {
      extra_args.emplace_back(idx);
    }
  }
  extra_args.push_back(LongImm::make(static_cast<int64_t>(keepdim)));  // 将保持维度的设置加入额外参数列表中
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(ResultBuf, "nnc_aten_mean", {InputBuf}, extra_args));  // 返回基于输入数据进行求均值的张量
}

Tensor computeMax(
    const std::vector<ArgValue>& inputs,  // 输入参数列表，包含了要处理的数据以及附加参数
    const std::vector<ExprHandle>& outputShape,  // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides,  // 输出张量的步长描述
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device device) {  // 数据处理所需的设备

  Dtype dtype = kFloat;  // 默认数据类型为浮点数
  if (outputType) {  // 如果指定了输出类型，则使用指定的类型
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("max", outputShape, dtype);  // 创建用于结果的缓冲区
  BufHandle InputBuf = std::get<BufHandle>(inputs[0]);  // 获取输入数据的缓冲区
  std::vector<ExprHandle> max_dims_expr;  // 存储最大化操作的维度表达式列表
  auto max_dim = std::get<int64_t>(inputs[1]);  // 获取指定的最大化维度
  auto keep_dim = std::get<bool>(inputs[2]);  // 获取是否保持维度的设置
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_max_red",
          {InputBuf},
          {max_dim, (int64_t)keep_dim}));  // 返回基于输入数据进行最大化操作的张量
}

Tensor computeAdaptiveAvgPool2d(
    const std::vector<ArgValue>& inputs,  // 输入参数列表，包含了要处理的数据以及附加参数
    const std::vector<ExprHandle>& outputShape,  // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides,  // 输出张量的步长描述
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device device) {  // 数据处理所需的设备

  Dtype dtype = kFloat;  // 默认数据类型为浮点数
  if (outputType) {  // 如果指定了输出类型，则使用指定的类型
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("adaptive_avgpool2d", outputShape, dtype);  // 创建用于结果的缓冲区
  auto out_size_param = std::get<IntList>(inputs[1]);  // 获取自适应平均池化操作的输出尺寸参数
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_adaptive_avg_pool2d",
          {std::get<BufHandle>(inputs[0])},  // 传入输入数据的缓冲区
          c10::fmap<ExprHandle>(out_size_param)));  // 使用输出尺寸参数进行自适应平均池化操作
}

} // 结束命名空间 jit
} // 结束命名空间 tensorexpr
```