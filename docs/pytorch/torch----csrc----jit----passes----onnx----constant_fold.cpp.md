# `.\pytorch\torch\csrc\jit\passes\onnx\constant_fold.cpp`

```
// 包含头文件 torch/csrc/jit/jit_log.h，用于 JIT 日志记录
#include <torch/csrc/jit/jit_log.h>
// 包含头文件 torch/csrc/jit/passes/onnx/constant_fold.h，用于 ONNX 常量折叠
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
// 包含头文件 torch/csrc/jit/passes/onnx/helper.h，提供 ONNX 相关辅助功能
#include <torch/csrc/jit/passes/onnx/helper.h>

// 包含 ATen 库中的 Functions.h，提供 ATen 库的函数
#include <ATen/Functions.h>

// 包含 c10/util/Exception.h，提供 C10 库的异常处理功能
#include <c10/util/Exception.h>
// 包含 c10/util/Optional.h，提供 C10 库的可选值类型
#include <c10/util/Optional.h>
// 包含 c10/util/irange.h，提供 C10 库的范围迭代器
#include <c10/util/irange.h>

// 包含标准库中的算法函数
#include <algorithm>

// 定义命名空间 torch::jit
namespace torch {
namespace jit {

// 定义命名空间 onnx，引入 c10::onnx 命名空间
namespace onnx {
using namespace ::c10::onnx;
}

// 定义命名空间 onnx_constant_fold
namespace onnx_constant_fold {

// 枚举类型 OnnxType，定义了一些 ONNX 数据类型对应的整数值
enum OnnxType : int {
  ONNX_FLOAT = 1,
  ONNX_UINT8,
  ONNX_INT8,
  ONNX_UINT16,
  ONNX_INT16,
  ONNX_INT32,
  ONNX_INT64,
  ONNX_FLOAT16 = 10,
  ONNX_DOUBLE,
  ONNX_UINT32,
};

// 定义映射表 onnxTypeToScalarTypeMap，将 ONNX 数据类型映射到 ATen 标量类型
std::unordered_map<int, at::ScalarType> onnxTypeToScalarTypeMap = {
    // 只包括 ONNX 数值类型的转换
    // 无符号的 ONNX 类型映射到下一个更高的有符号 ScalarType 类型
    {ONNX_FLOAT, at::kFloat},
    {ONNX_UINT8, at::kByte},
    {ONNX_INT8, at::kChar},
    {ONNX_UINT16, at::kInt},
    {ONNX_INT16, at::kShort},
    {ONNX_INT32, at::kInt},
    {ONNX_INT64, at::kLong},
    {ONNX_FLOAT16, at::kFloat},
    {ONNX_DOUBLE, at::kDouble},
    {ONNX_UINT32, at::kLong},
};

// 处理负的起始和结束索引，确保它们在有效范围内
void handleNegativeStartEndIndex(
    int64_t& start,
    int64_t& end,
    int64_t& axis,
    c10::IntArrayRef tensorSizes) {
  // 如果起始索引为负数，修正为相对于轴长度的正索引
  if (start < 0) {
    start = tensorSizes[axis] + start;
  }
  // 如果结束索引为负数，修正为相对于轴长度的正索引
  if (end < 0) {
    end = tensorSizes[axis] + end;
  }
  // 索引超出维度长度的情况，将结束索引修正为维度长度
  if (end > tensorSizes[axis]) {
    end = tensorSizes[axis];
  }
}

// 在 opset 9 中运行 Torch Slice 操作，对输入张量进行常量折叠
std::optional<at::Tensor> runTorchSlice_opset9(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues) {
  // 断言输入张量数组大小为1
  assert(inputTensorValues.size() == 1);
  // 如果输入张量数组大小不为1，则发出警告并返回空值
  if (inputTensorValues.size() != 1) {
    TORCH_WARN(
        "Constant folding - Invalid number of inputs found for opset 9 "
        "onnx::Slice op. Constant folding not applied.");
    return c10::nullopt;
  }
  // 如果节点缺少 starts 或 ends 属性，则返回空值
  if (!(node->hasAttributeS("starts") && node->hasAttributeS("ends"))) {
    return c10::nullopt;
  }
  // 获取 starts 和 ends 属性的值
  auto startsAttr = node->is(attr::starts);
  auto endsAttr = node->is(attr::ends);
  // 如果 starts 和 ends 属性数组大小不一致，则返回空值
  if (startsAttr.size() != endsAttr.size()) {
    return c10::nullopt;
  }
  // 获取 axes 属性的值，如果不存在则默认为从 0 开始的索引数组
  std::vector<int64_t> axesAttr;
  if (node->hasAttributeS("axes")) {
    axesAttr = node->is(attr::axes);
  } else {
    axesAttr.resize(startsAttr.size());
    std::iota(axesAttr.begin(), axesAttr.end(), 0);
  }
  // 初始化更新后的张量值为输入张量的第一个元素
  auto updated_val = inputTensorValues[0];
  // 遍历 axes 属性数组
  for (const auto i : c10::irange(axesAttr.size())) {
    // 获取当前轴索引、起始和结束索引
    int64_t axis = axesAttr[i], start = startsAttr[i], end = endsAttr[i];
    // 如果轴索引为负数，修正为相对于输入张量维度数的正索引
    axis += axis < 0 ? inputTensorValues[0].sizes().size() : 0;
    // 处理负的起始和结束索引，确保在有效范围内
    handleNegativeStartEndIndex(start, end, axis, updated_val.sizes());
    // 计算切片的长度
    int64_t length = end - start;
    // 如果长度小于0或者起始索引超出有效范围，则返回空值
    if (length < 0 || start > updated_val.sizes()[axis] - length)
      return c10::nullopt;
    // 对更新后的张量进行缩窄操作
    updated_val = at::narrow(updated_val, axis, start, length);
  }
  // 返回更新后的张量值的可选值
  return std::optional<at::Tensor>(updated_val);
}

} // namespace onnx_constant_fold
} // namespace jit
} // namespace torch
// 运行 torch::onnx::Slice 操作的函数，支持 opset >= 10 版本
std::optional<at::Tensor> runTorchSlice_opset10(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues) {
  // 最大和最小的 slice 输入数量限制
  const int maxSliceInputCount = 5;
  const int minSliceInputCount = 3;
  // 检查输入张量值的数量是否在有效范围内
  if (inputTensorValues.size() < minSliceInputCount ||
      inputTensorValues.size() > maxSliceInputCount) {
    // 如果数量无效，发出警告并返回空值
    TORCH_WARN(
        "Constant folding - Invalid number of inputs found for opset opset >= 10 onnx::Slice op. "
        "Constant folding not applied.");
    return c10::nullopt;
  }
  // 检查 'starts' 和 'ends' 输入的有效性
  if (inputTensorValues[1].sizes().size() != 1 ||
      inputTensorValues[2].sizes().size() != 1) {
    // 如果 'starts' 或 'ends' 输入无效，发出警告并返回空值
    TORCH_WARN(
        "Constant folding - Invalid 'starts' or 'ends' inputs found for opset >= 10 onnx::Slice op. "
        "Constant folding not applied.");
    return c10::nullopt;
  }
  // 检查 'starts' 和 'ends' 的元素数量是否相同
  if (inputTensorValues[1].sizes()[0] != inputTensorValues[2].sizes()[0]) {
    // 如果不同，返回空值
    return c10::nullopt;
  }
  // 检查 'axes' 输入的有效性，如果可用
  std::vector<int64_t> axes;
  if (inputTensorValues.size() > 3) {
    if (inputTensorValues[3].sizes().size() != 1) {
      // 如果 'axes' 输入无效，发出警告并返回空值
      TORCH_WARN(
          "Constant folding - Invalid 'axes' input found for opset >= 10 onnx::Slice op. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
    // 检查 'axes' 和 'ends' 输入的元素数量是否相同
    if (inputTensorValues[3].sizes()[0] != inputTensorValues[1].sizes()[0]) {
      // 如果不同，发出警告并返回空值
      TORCH_WARN(
          "Constant folding - Invalid 'axes' or 'ends' inputs found for opset >= 10 onnx::Slice op. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
    // 获取 'axes' 的访问器，并处理负轴值，修复为正值
    auto axes_a = inputTensorValues[3].accessor<int64_t, 1>();
    axes.resize(inputTensorValues[3].sizes()[0]);
    for (const auto i : c10::irange(inputTensorValues[3].sizes()[0])) {
      axes[i] = axes_a[i] < 0 ? axes_a[i] + inputTensorValues[0].sizes().size()
                              : axes_a[i];
    }
  } else {
    // 如果没有提供 'axes' 输入，则使用默认值 0
    axes = std::vector<int64_t>(inputTensorValues[1].sizes()[0], 0);
  }
  // 检查 'steps' 输入的有效性，如果可用
  if (inputTensorValues.size() > 4) {
    if (inputTensorValues[4].sizes().size() != 1) {
      // 如果 'steps' 输入无效，发出警告并返回空值
      TORCH_WARN(
          "Constant folding - Invalid 'steps' input found for opset >= 10 onnx::Slice op. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
    // 检查 'steps' 和 'ends' 输入的元素数量是否相同
    if (inputTensorValues[4].sizes()[0] != inputTensorValues[1].sizes()[0]) {
      // 如果不同，发出警告并返回空值
      TORCH_WARN(
          "Constant folding - Invalid 'steps' or 'ends' inputs found for opset >= 10 onnx::Slice op. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
    // 获取 'steps' 的访问器，暂时不做处理
    auto steps_a = inputTensorValues[4].accessor<int64_t, 1>();
    // 这里省略了 steps_a 的处理部分
    // 对输入张量的第五维度大小范围进行迭代
    for (const auto i : c10::irange(inputTensorValues[4].sizes()[0])) {
      // 只支持步长为1的常量折叠
      if (steps_a[i] != 1) {
        // 如果步长不为1，则发出警告并返回空的可选类型
        TORCH_WARN(
            "Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. "
            "Constant folding not applied.");
        return c10::nullopt;
      }
    }
  }
  // 获取起始点数组的访问器
  auto starts_a = inputTensorValues[1].accessor<int64_t, 1>();
  // 获取结束点数组的访问器
  auto ends_a = inputTensorValues[2].accessor<int64_t, 1>();
  // 复制输入的更新后的张量
  auto updated_val = inputTensorValues[0];
  // 对起始点数组的大小范围进行迭代
  for (const auto i : c10::irange(inputTensorValues[1].sizes()[0])) {
    // ONNX 的 Slice 操作接受负的起始点和结束点值
    int64_t start = starts_a[i], end = ends_a[i], axis = axes[i];
    // 处理负的起始点和结束点索引，根据张量维度更新值
    handleNegativeStartEndIndex(start, end, axis, updated_val.sizes());
    // 计算切片的长度
    int64_t length = end - start;
    // 如果长度小于0或者起始点大于更新后的张量在当前轴上的大小减去长度，则返回空的可选类型
    if (length < 0 || start > updated_val.sizes()[axis] - length)
      return c10::nullopt;
    // 在指定轴上进行缩小操作，更新更新后的值
    updated_val = at::narrow(updated_val, axis, start, length);
  }
  // 返回更新后的张量作为包含在可选类型中的结果
  return std::optional<at::Tensor>(updated_val);
}

// 运行 Torch Arange 操作（opset 11 版本）
at::Tensor runTorchArange_opset11(
    const Node* node,
    const std::vector<at::Tensor>& inputTensorValues) {
  // 断言输入张量值的数量为 3
  TORCH_INTERNAL_ASSERT(inputTensorValues.size() == 3);
  // 获取第一个输入张量的数据类型
  auto dtype = inputTensorValues[0].scalar_type();
  // 定义更新后的张量
  at::Tensor updated_val;
  // 根据数据类型执行不同的操作
  switch (dtype) {
    case at::ScalarType::Float: {
      // 获取 float 类型的起始值、结束值和步长
      auto start = inputTensorValues[0].item<float>();
      auto end = inputTensorValues[1].item<float>();
      auto step = inputTensorValues[2].item<float>();
      // 调用 Torch 的 arange 函数生成张量
      updated_val = at::arange(start, end, step);
      break;
    }
    case at::ScalarType::Double: {
      // 获取 double 类型的起始值、结束值和步长
      auto start = inputTensorValues[0].item<double>();
      auto end = inputTensorValues[1].item<double>();
      auto step = inputTensorValues[2].item<double>();
      // 调用 Torch 的 arange 函数生成张量
      updated_val = at::arange(start, end, step);
      break;
    }
    case at::ScalarType::Short: {
      // 获取 short 类型的起始值、结束值和步长
      auto start = inputTensorValues[0].item<int16_t>();
      auto end = inputTensorValues[1].item<int16_t>();
      auto step = inputTensorValues[2].item<int16_t>();
      // 调用 Torch 的 arange 函数生成张量
      updated_val = at::arange(start, end, step);
      break;
    }
    case at::ScalarType::Int: {
      // 获取 int 类型的起始值、结束值和步长
      auto start = inputTensorValues[0].item<int>();
      auto end = inputTensorValues[1].item<int>();
      auto step = inputTensorValues[2].item<int>();
      // 调用 Torch 的 arange 函数生成张量
      updated_val = at::arange(start, end, step);
      break;
    }
    case at::ScalarType::Long: {
      // 获取 long 类型的起始值、结束值和步长
      auto start = inputTensorValues[0].item<int64_t>();
      auto end = inputTensorValues[1].item<int64_t>();
      auto step = inputTensorValues[2].item<int64_t>();
      // 调用 Torch 的 arange 函数生成张量
      updated_val = at::arange(start, end, step);
      break;
    }
    default: {
      // 如果数据类型不支持，发出警告
      TORCH_WARN(
          "Constant folding - ONNX Range type: ", dtype, " is not supported.");
    }
  }
  // 返回更新后的张量
  return updated_val;
}

// 将 int64_t 类型的整数转换为 Torch 张量
at::Tensor IntToTensor(int64_t value) {
  // 定义张量的选项，数据类型为 long，设备为 CPU
  auto options = c10::TensorOptions().dtype(at::kLong).device(at::kCPU);
  // 创建存储数据的向量
  std::vector<int64_t> size_data = {value};
  // 从数据指针创建 Torch 张量，并设置为 CPU 上的张量
  auto f = at::from_blob(size_data.data(), {1}, at::kLong).to(at::kCPU);
  // 需要复制数据到新的张量中
  at::Tensor f_copy = at::empty({1}, options);
  f_copy.copy_(f);
  // 返回压缩后的张量
  return at::squeeze(f_copy, 0);
}

// 运行 Torch 后端处理 ONNX 操作
std::optional<at::Tensor> runTorchBackendForOnnx(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues,
    int opset_version) {
  // 定义更新后的张量
  at::Tensor updated_val;
  // 如果节点的种类是 Slice
  if (node->kind() == onnx::Slice) {
    // 根据 ONNX 的操作集版本调用不同的函数处理 Slice
    if (opset_version == ONNX_OPSET_9) {
      return runTorchSlice_opset9(node, inputTensorValues);
    } else if (opset_version >= ONNX_OPSET_10) {
      return runTorchSlice_opset10(node, inputTensorValues);
    } else {
      // 如果操作集版本不支持，发出警告
      TORCH_WARN(
          "Constant folding - unsupported opset version. Constant folding not applied.");
      return c10::nullopt;
    }
  } else if (node->kind() == onnx::Concat) {
    // 如果节点的种类是 Concat
    // 检查节点是否具有 "axis" 属性
    if (!node->hasAttributeS("axis")) {
      return c10::nullopt;
    }
    // 调用 Torch 的 cat 函数执行 Concat 操作
    updated_val =
        at::cat(at::TensorList(inputTensorValues), node->i(attr::axis));
    // 如果节点类型为onnx::Sqrt，计算输入张量的平方根并返回作为可选的张量
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Sqrt) {
    // 计算输入张量的平方根并返回作为可选的张量
    updated_val = at::sqrt(inputTensorValues[0]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Div) {
    // 若一个例子表明at::div(CPULongType, CPULongType) = CPUFloatType，
    // 因此我们在下面添加一个类型转换。
    updated_val = at::div(inputTensorValues[0], inputTensorValues[1]);
    // 如果输入张量的标量类型相同，则将结果张量转换为相同的类型
    if (inputTensorValues[0].scalar_type() ==
        inputTensorValues[1].scalar_type()) {
      updated_val = updated_val.to(inputTensorValues[0].scalar_type());
    }
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Mul) {
    // 计算输入张量的乘积并返回作为可选的张量
    updated_val = at::mul(inputTensorValues[0], inputTensorValues[1]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Sub) {
    // 计算输入张量的差并返回作为可选的张量
    updated_val = at::sub(inputTensorValues[0], inputTensorValues[1]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Add) {
    // 计算输入张量的和并返回作为可选的张量
    updated_val = at::add(inputTensorValues[0], inputTensorValues[1]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Unsqueeze) {
    // 如果操作集版本大于等于ONNX_OPSET_13
    if (opset_version >= ONNX_OPSET_13) {
      assert(inputTensorValues.size() == 2);
      // 检查'axes'输入的有效性
      if (inputTensorValues[1].sizes().size() != 1) {
        // 对于opset 13的onnx::Unsqueeze操作，发现无效的'axes'输入。
        // 不应用常量折叠。
        TORCH_WARN(
            "Constant folding - Invalid 'axes' inputs found for opset 13 onnx::Unsqueeze op. "
            "Constant folding not applied.");
        return c10::nullopt;
      }
      auto axes_a = inputTensorValues[1].accessor<int64_t, 1>();
      std::vector<int64_t> axes;
      for (int64_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
        // ONNX的unsqueeze接受负的axes
        // 来自https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        // 负的维度将对应于在维度=维度+input.dim()+1处应用unsqueeze()。
        axes_a[i] +=
            axes_a[i] < 0 ? inputTensorValues[0].sizes().size() + 1 : 0;
        axes.push_back(axes_a[i]);
      }
      // 对axes进行排序
      std::sort(axes.begin(), axes.end());
      updated_val = inputTensorValues[0];
      for (int64_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
        // 在指定的维度上对输入张量进行unsqueeze操作
        updated_val = at::unsqueeze(updated_val, axes[i]);
      }
      return std::optional<at::Tensor>(updated_val);
    } else if (opset_version >= ONNX_OPSET_9) {
      assert(inputTensorValues.size() == 1);
      // 如果节点未定义属性'sxes'
      if (!node->hasAttributeS("axes")) {
        return c10::nullopt;
      }
      updated_val = inputTensorValues[0];
      std::vector<int64_t> axesAttr = node->is(attr::axes);
      // 对axes属性进行排序
      std::sort(axesAttr.begin(), axesAttr.end());
      for (auto axis : axesAttr) {
        // 在指定的维度上对输入张量进行unsqueeze操作
        updated_val = at::unsqueeze(updated_val, axis);
      }
      return std::optional<at::Tensor>(updated_val);
    } else {
      // 如果操作集版本不支持常量折叠，则发出警告并返回空值
      TORCH_WARN(
          "Constant folding - unsupported opset version. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
  } else if (node->kind() == onnx::Squeeze) {
    // 断言输入张量值的大小为2或1
    assert(inputTensorValues.size() == 2 || inputTensorValues.size() == 1);
    if (opset_version >= ONNX_OPSET_13) {
      // 如果操作集版本大于等于13，则执行以下操作
      // 当Squeeze版本为13时，输入的axes是可选的，inputTensorValues.size() == 1 表示axes等于None
      updated_val = inputTensorValues[0];
      if (inputTensorValues.size() == 2) {
        // 检查'axes'输入的有效性
        if (inputTensorValues[1].sizes().size() != 1) {
          // 如果'axes'输入无效，则发出警告并返回空值
          TORCH_WARN(
              "Constant folding - Invalid 'axes' inputs found for opset 13 onnx::Squeeze op. "
              "Constant folding not applied.");
          return c10::nullopt;
        }
        auto axes_a = inputTensorValues[1].accessor<int64_t, 1>();
        std::vector<int64_t> axes;
        for (int64_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
          // ONNX Squeeze接受负的axes值
          axes_a[i] += axes_a[i] < 0 ? inputTensorValues[0].sizes().size() : 0;
          axes.push_back(axes_a[i]);
        }
        std::sort(axes.begin(), axes.end());
        for (int64_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
          updated_val = at::squeeze(updated_val, axes[i]);
        }
      }
      return std::optional<at::Tensor>(updated_val);
    } else if (opset_version >= ONNX_OPSET_9) {
      // 如果操作集版本大于等于9，则执行以下操作
      assert(inputTensorValues.size() == 1);
      updated_val = inputTensorValues[0];
      if (node->hasAttributeS("axes")) {
        // 获取'axes'属性并进行排序
        std::vector<int64_t> axesAttr = node->is(attr::axes);
        std::sort(axesAttr.begin(), axesAttr.end());
        for (auto axis : axesAttr) {
          updated_val = at::squeeze(updated_val, axis);
        }
      }
      return std::optional<at::Tensor>(updated_val);
    } else {
      // 如果操作集版本不支持常量折叠，则发出警告并返回空值
      TORCH_WARN(
          "Constant folding - unsupported opset version. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
  } else if (node->kind() == onnx::Transpose) {
    // 断言输入张量值的大小为1
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("perm")) {
      // 如果没有'perm'属性，则返回空值
      return c10::nullopt;
    }
    // 对输入张量进行维度变换，并返回更新后的值
    updated_val = inputTensorValues[0].permute(node->is(attr::perm));
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Cast) {
    // 断言输入张量值的大小为1
    assert(inputTensorValues.size() == 1);
    if (node->hasAttributeS("to") && ONNXTypeToATenType(node->i(attr::to))) {
      // 如果有'to'属性并且可以转换为ATen类型，则将输入张量值转换为指定类型
      updated_val = inputTensorValues[0].to(
          ONNXTypeToATenType(node->i(attr::to)).value());
      return std::optional<at::Tensor>(updated_val);
    }
    // 否则返回空值
    return c10::nullopt;
  } else if (node->kind() == onnx::Reshape) {
    // 断言输入张量值的大小为2
    assert(inputTensorValues.size() == 2);
    updated_val = inputTensorValues[0];
    // 创建用于重塑的形状向量，并初始化为0
    std::vector<int64_t> shape(inputTensorValues[1].sizes()[0], 0);
    auto shape_a = inputTensorValues[1].accessor<int64_t, 1>();
    // 检查输入张量的第二个元素的大小是否大于等于0
    assert(inputTensorValues[1].sizes()[0] >= 0);
    // 设置 allowzero 的值
    int64_t allowzero = 0;
    // 如果节点具有 allowzero 属性，则将其值赋给 allowzero
    if (node->hasAttributeS("allowzero")) {
      allowzero = node->i(attr::allowzero);
    }
    // 遍历输入张量的第二个元素的大小
    for (size_t i = 0; i < (size_t)(inputTensorValues[1].sizes()[0]); ++i) {
      // 所有的 shape 维度值应该大于等于 -1
      // onnx::Reshape 支持 shape 维度值为零，此时实际维度值保持不变。然而，
      // at::reshape 不支持 shape 维度值为零
      assert(shape_a[i] >= -1);
      // 如果 shape_a[i] 为零且 allowzero 为 false，则抛出异常
      if (shape_a[i] == 0 && !allowzero) {
        // 如果 i 超过了输入张量的维度大小，则抛出维度异常
        if (i >= inputTensorValues[0].sizes().size()) {
          throw std::runtime_error(
              "Dimension with value 0 exceeds the input size dimensions.");
        }
        // 将 shape[i] 设置为输入张量的第一个元素的维度值
        shape[i] = inputTensorValues[0].sizes()[i];
      } else {
        // 否则，将 shape[i] 设置为 shape_a[i]
        shape[i] = shape_a[i];
      }
    }
    // 返回使用更新后的值和 shape 调整维度后的可选张量
    return std::optional<at::Tensor>(at::reshape(updated_val, shape));
  } else if (node->kind() == onnx::Shape) {
    // 断言输入张量的大小为 1
    TORCH_INTERNAL_ASSERT(inputTensorValues.size() == 1);
    // 更新值为将输入张量转换为形状张量的结果
    updated_val = at::_shape_as_tensor(inputTensorValues[0]);
    // 返回更新后的值作为可选张量
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::ReduceL1 || node->kind() == onnx::ReduceL2) {
    // 断言输入张量的大小为 1
    assert(inputTensorValues.size() == 1);
    // 如果节点没有 axes 属性，则返回空值
    if (!node->hasAttributeS("axes")) {
      return c10::nullopt;
    }
    // 如果节点没有 keepdims 属性，则返回空值
    if (!node->hasAttributeS("keepdims")) {
      return c10::nullopt;
    }
    // 根据节点类型确定 p 的值
    int p = node->kind() == onnx::ReduceL1 ? 1 : 2;
    // 更新值为根据指定参数计算的 L1 或 L2 范数
    updated_val = at::norm(
        inputTensorValues[0], p, node->is(attr::axes), node->i(attr::keepdims));
    // 返回更新后的值作为可选张量
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::ReduceProd) {
    // 计算输入张量的秩
    int64_t rank = inputTensorValues[0].sizes().size();
    // 定义轴数组
    std::vector<int64_t> axes;
    // 如果节点没有 axes 属性，则设定默认的轴顺序
    if (!node->hasAttributeS("axes")) {
      axes = std::vector<int64_t>(rank);
      std::iota(axes.rbegin(), axes.rend(), 0);
    } else {
      // 否则，根据节点的 axes 属性设置轴顺序，并按降序排序
      for (const auto& axis : node->is(attr::axes)) {
        axes.emplace_back(axis < 0 ? axis + rank : axis);
      }
      std::sort(axes.begin(), axes.end(), std::greater<>());
    }

    // 确定是否保持维度
    bool keepdims =
        node->hasAttributeS("keepdims") ? node->i(attr::keepdims) : true;
    // 更新值为按指定轴计算的积
    updated_val = inputTensorValues[0];
    for (const auto& axis : axes) {
      updated_val = at::prod(updated_val, axis, keepdims);
    }
    // 返回更新后的值作为可选张量
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Gather) {
    // 断言输入张量的大小为 2
    assert(inputTensorValues.size() == 2);
    // 默认轴为 0
    int64_t axis = 0;
    // 如果节点具有 axis 属性，则将其值赋给 axis
    if (node->hasAttributeS("axis")) {
      axis = node->i(attr::axis);
    }
    // 如果 onnx::Gather 的 axis 属性值小于 0，则需要调整为有效的维度值
    axis += axis < 0 ? inputTensorValues[0].sizes().size() : 0;
    // 获取 indices 张量
    at::Tensor indices = inputTensorValues[1];
    // 获取 indices 张量的维度
    auto q = indices.dim();
    // at::index_select 只支持维度小于等于 1 的索引
    // 如果 q 大于 1，则返回空 optional
    if (q > 1) {
      return c10::nullopt;
    }
    // 如果 indices 张量的设备与输入张量的设备不同，将其移动到输入张量的设备上
    if (inputTensorValues[0].device() != indices.device()) {
      indices = indices.to(inputTensorValues[0].device());
    }
    // 如果 onnx::Gather 的 indices 输入有小于 0 的值，需要进行调整（+= 维度的值）以适应 aten 操作
    auto less_mask = at::lt(indices, 0);
    auto indices_corr = at::add(indices, inputTensorValues[0].sizes()[axis]);
    auto indices_masked = at::where(less_mask, indices_corr, indices);
    // 使用 index_select 函数根据给定轴和修正后的 indices 获取更新后的值
    updated_val = at::index_select(inputTensorValues[0], axis, indices_masked);
    // 如果 indices 的秩为 0，则输出张量的秩应为输入张量的秩 - 1
    if (q < 1) {
      updated_val = updated_val.squeeze(axis);
    }
    // 返回更新后的值作为 std::optional<at::Tensor>
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Range) {
    // 如果节点类型是 onnx::Range，则调用 runTorchArange_opset11 函数处理，并返回结果作为 std::optional<at::Tensor>
    updated_val = runTorchArange_opset11(node, inputTensorValues);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Where) {
    // 如果节点类型是 onnx::Where，则调用 at::where 函数处理，并返回结果作为 std::optional<at::Tensor>
    updated_val = at::where(
        inputTensorValues[0], inputTensorValues[1], inputTensorValues[2]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Equal) {
    // 如果节点类型是 onnx::Equal，则调用 at::eq 函数处理，并返回结果作为 std::optional<at::Tensor>
    updated_val = at::eq(inputTensorValues[0], inputTensorValues[1]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Greater) {
    // 如果节点类型是 onnx::Greater，则调用 at::greater 函数处理，并返回结果作为 std::optional<at::Tensor>
    updated_val = at::greater(inputTensorValues[0], inputTensorValues[1]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Less) {
    // 如果节点类型是 onnx::Less，则调用 at::less 函数处理，并返回结果作为 std::optional<at::Tensor>
    updated_val = at::less(inputTensorValues[0], inputTensorValues[1]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Neg) {
    // 如果节点类型是 onnx::Neg，则调用 at::neg 函数处理，并返回结果作为 std::optional<at::Tensor>
    updated_val = at::neg(inputTensorValues[0]);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Not) {
    // 如果节点类型是 onnx::Not，则创建一个全为 1 的张量，调用 at::ne 函数处理，并返回结果作为 std::optional<at::Tensor>
    auto ones =
        at::ones(inputTensorValues[0].sizes(), inputTensorValues[0].dtype());
    updated_val = at::ne(inputTensorValues[0], ones);
    return std::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Size) {
    // 如果节点类型是 onnx::Size，则计算输入张量的总大小，并返回结果作为 std::optional<at::Tensor>
    int64_t total_size = 1;
    for (auto size : inputTensorValues[0].sizes()) {
      total_size *= size;
    }
    return std::optional<at::Tensor>(IntToTensor(total_size));
  } else if (node->kind() == onnx::Softmax) {
    // 如果节点类型是 onnx::Softmax，则根据指定的轴调用 at::softmax 函数处理，并返回结果作为 std::optional<at::Tensor>
    int64_t axis = node->hasAttributeS("axis") ? node->i(attr::axis) : -1;
    updated_val = at::softmax(inputTensorValues[0], axis);
    return std::optional<at::Tensor>(updated_val);
  } else {
    // 如果节点类型未被识别，则返回空 optional
    return c10::nullopt;
  }
} // namespace onnx_constant_fold



// 指定命名空间结束，结束了对命名空间 onnx_constant_fold 的定义



// 检查给定的值是否为常量
bool isConstant(Value* val, const ValueToParamPairMap& valsToParamsMap) {
  auto parentNode = val->node();
  // 如果值的父节点是 prim::Param 并且值存在于参数映射中，则被认为是常量
  return (parentNode->kind() == prim::Param &&
          valsToParamsMap.find(val) !=
              valsToParamsMap
                  .end()) || // 检查值是否为参数，并且不是真实输入
      (parentNode->kind() == onnx::Constant && !parentNode->mustBeNone() &&
       parentNode->kindOf(attr::value) ==
           AttributeKind::t); // 检查其他类型？
}



// 检查节点是否有参数输入
bool hasParamInput(Node* n, const ValueToParamPairMap& valsToParamsMap) {
  for (auto input : n->inputs()) {
    // 如果节点的输入在参数映射中存在，则返回 true
    if (valsToParamsMap.find(input) != valsToParamsMap.end()) {
      return true;
    }
  }
  return false;
}



// 获取节点输入的值作为张量的向量
std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      auto itr = valsToParamsMap.find(val);
      // 如果在参数映射中找不到输入值，则抛出运行时错误
      if (itr == valsToParamsMap.end()) {
        throw std::runtime_error(
            "getValues: Input value not found amongst constant parameters.");
      }
      inputTensorValues.push_back(itr->second.second.toTensor());
    } else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      // 如果找到不支持的常量节点类型，则抛出运行时错误
      throw std::runtime_error(
          "getValues: Unsupported kind of constant node found.");
    }
  }
  // 断言输入张量的数量与节点输入数量相同
  TORCH_INTERNAL_ASSERT(inputTensorValues.size() == numInputs);
  return inputTensorValues;
}



// 检查节点的所有输入是否都是常量
bool areNodeInputsConstant(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  return std::all_of(
      node->inputs().begin(),
      node->inputs().end(),
      [&valsToParamsMap](Value* v) { return isConstant(v, valsToParamsMap); });
}



// 获取所有的 ONNX 常量父节点以移除
std::vector<Node*> getOnnxConstParentsToRemove(Node* node) {
  std::vector<Node*> parentNodes;
  for (auto val : node->inputs()) {
    // 如果节点的父节点是 onnx::Constant 并且该节点只服务于当前节点，则将其添加到待移除的列表中
    if (val->node()->kind() == onnx::Constant && val->uses().size() == 1) {
      parentNodes.push_back(val->node());
    }
  }
  return parentNodes;
}



// 此方法在原地更新块，将所有基于一次常量的计算/操作折叠到初始化节点中
//
// 注意：这不是传统意义上的常量折叠，因为我们不会特别努力评估常量节点上的操作。
// 这更像是部分评估分析，其中对常量节点的操作可以提前执行，以便在通常的参数已知之前运行它们。
void ConstantFoldONNX(Block* b, ParamMap& paramsDict, int opset_version) {
  if (opset_version < ONNX_OPSET_9) {



// 块内部的常量折叠方法结束
    // 发出警告信息，指出常量折叠仅支持 opsets >= 9，当前不适用常量折叠
    TORCH_WARN(
        "Constant folding supported for only opsets >= 9. "
        "Constant folding not applied.");
    // 函数返回，结束执行
    return;
  }
  // 内部断言，确保块 b 具有参数节点
  TORCH_INTERNAL_ASSERT(b->param_node());
  // 构建值到参数映射表，将参数字典 paramsDict 传入
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  // 只有根块进行常量折叠，不支持对嵌套块进行常量折叠
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto node = *it;
    if (node->outputs().size() > 1) {
      // 不支持多输出节点的常量折叠，跳过该节点
      continue;
    }
    if (!onnx_constant_fold::areNodeInputsConstant(node, valsToParamsMap)) {
      // 如果该节点的所有输入既不是参数也不是 onnx::Constant，则跳过该节点
      continue;
    }

    auto inputTensorValues =
        onnx_constant_fold::getValues(node, valsToParamsMap);
    if (inputTensorValues.empty()) {
      // 该节点是一个没有输入的终端节点，比如 onnx::Constant，跳过该节点
      continue;
    }
    // 运行 Torch 后端对 ONNX 节点进行常量折叠，并获取更新后的值
    auto updatedValWrapped = onnx_constant_fold::runTorchBackendForOnnx(
        node, inputTensorValues, opset_version);
    if (updatedValWrapped == c10::nullopt) {
      // 不支持对此操作进行常量折叠，跳过该节点
      continue;
    }

    // 获取更新后的张量值
    at::Tensor updatedVal = *updatedValWrapped;
    // 创建新的源节点输出
    auto newSourceNodeOutput = [&]() -> Value* {
      if (onnx_constant_fold::hasParamInput(node, valsToParamsMap)) {
        // 如果节点具有参数输入，创建块的新输入（prim::Param 节点输出），并在 valsToParamsMap 中添加对应的条目
        auto newSourceNodeOutput = b->addInput();
        valsToParamsMap.insert(
            {newSourceNodeOutput,
             std::make_pair(newSourceNodeOutput->debugName(), updatedVal)});
        return newSourceNodeOutput;
      } else {
        // 如果节点没有参数输入，创建一个新的 ONNX 常量节点
        auto newSourceNode =
            createONNXConstant(node->owningGraph(), node, updatedVal);
        newSourceNode->copyMetadata(node);
        return newSourceNode->output();
      }
    }();
    // 根据更新后的值推断新源节点输出的类型
    newSourceNodeOutput->inferTypeFrom(updatedVal);
    // 将当前节点的输出替换为新的源节点输出
    node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);
    // 移除当前节点，替换为初始化器
    auto onnxConstParents =
        onnx_constant_fold::getOnnxConstParentsToRemove(node);
    node->removeAllInputs();
    // 移除所有父节点是 onnx::Constant 的节点
    for (auto* n : onnxConstParents) {
      n->destroy();
    }
    it.destroyCurrent();
  }
  # 销毁当前迭代器指向的对象
  eraseUnusedValuesFromMap(valsToParamsMap);
  # 从映射中删除未使用的值
  eraseUnusedBlockInputs(b);
  # 删除块 b 中未使用的输入
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  # 根据值到参数映射构建参数映射
  return;
  # 返回，结束函数执行
}

void ConstantFoldONNX(
    std::shared_ptr<Graph>& g,
    ParamMap& paramsDict,
    int opset_version) {
  // 调用 ConstantFoldONNX 函数，传入当前图的块、参数字典和操作集版本号
  ConstantFoldONNX(g->block(), paramsDict, opset_version);
  // 调用 GRAPH_DUMP 宏，打印常量折叠后的图的信息
  GRAPH_DUMP("After ConstantFoldONNX:", g);
}

} // namespace jit
} // namespace torch
```