# `.\pytorch\torch\csrc\jit\passes\graph_rewrite_helper.cpp`

```
// 包含 Torch 的图形重写助手相关头文件
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

// 包含 Torch 的子图匹配器相关头文件
#include <torch/csrc/jit/ir/subgraph_matcher.h>

// 包含 Torch 的常量传播相关头文件
#include <torch/csrc/jit/passes/constant_propagation.h>

// 包含 Torch 的子图重写相关头文件
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

// Torch 命名空间
namespace torch {
namespace jit {
namespace graph_rewrite_helper {

// 获取函数名称，根据给定的函数值
std::string getFuncName(Value* func_value) {
  // 期望函数类型的引用，并获取其函数对象
  auto func = func_value->type()->expectRef<FunctionType>().function();
  // 获取函数的限定名称
  const auto& qname = func->qualname();
  const auto& name = qname.qualifiedName();
  // 查找限定名称中最后一个点号的位置
  auto rdot_idx = name.rfind('.');
  // 如果找到点号，则返回点号后的字符串作为函数名称
  if (rdot_idx != std::string::npos) {
    return name.substr(rdot_idx + 1, name.length());
  } else {
    // 否则返回完整的限定名称作为函数名称
    return name;
  }
}

// 获取值的函数，根据名称在匹配值映射和值映射中查找
Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 使用名称从值映射中获取值对应的值
  return match_vmap.at(vmap.at(name));
}

// 获取 IValue 的函数，根据名称获取匹配值映射和值映射中的值，并转换为 IValue
std::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 转换获取到的值为 IValue 类型，并返回可选的 IValue
  return toIValue(getValue(name, match_vmap, vmap));
}

// 静态函数，获取卷积参数的函数，根据匹配和值映射获取参数
static std::unordered_map<std::string, c10::IValue> getConvParams(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 创建用于存储计算值的映射
  std::unordered_map<std::string, c10::IValue> calc_values;
  // 获取匹配值映射
  const auto& match_vmap = match.values_map;
  // 获取转置参数的 IValue 并存储到计算值映射中
  auto transposed_value = getIValue("transposed", match_vmap, vmap).value();
  calc_values["transposed"] = transposed_value;
  // 获取输出填充参数的 IValue 并存储到计算值映射中
  auto output_padding_value =
      getIValue("output_padding", match_vmap, vmap).value();
  calc_values["output_padding"] = output_padding_value;
  // 获取步幅参数的 IValue 并存储到计算值映射中
  auto stride_value = getIValue("stride", match_vmap, vmap).value();
  calc_values["stride"] = stride_value;
  // 获取填充参数的 IValue 并存储到计算值映射中
  auto padding_value = getIValue("padding", match_vmap, vmap).value();
  calc_values["padding"] = padding_value;
  // 获取扩展参数的 IValue 并存储到计算值映射中
  auto dilation_value = getIValue("dilation", match_vmap, vmap).value();
  calc_values["dilation"] = dilation_value;
  // 返回计算值映射
  return calc_values;
}

// 匿名函数，检查是否为卷积操作的辅助函数，根据匹配和值映射检查参数维度是否正确
auto check_conv2d = [](const Match& match,
                       const std::unordered_map<std::string, Value*>& vmap) {
  // 获取卷积参数的计算值映射
  auto calc_value_map = getConvParams(match, vmap);
  // 检查输出填充、步幅、填充和扩展参数的维度是否为 1
  if (calc_value_map["output_padding"].toIntList().size() != 1 ||
      calc_value_map["stride"].toIntList().size() != 1 ||
      calc_value_map["padding"].toIntList().size() != 1 ||
      calc_value_map["dilation"].toIntList().size() != 1) {
    return false;
  }
  // 返回转置参数的布尔值是否为假
  return !calc_value_map["transposed"].toBool();
};

// 匿名函数，过滤卷积操作的辅助函数，根据匹配和值映射检查参数维度是否正确
auto filter_conv2d = [](const Match& match,
                        const std::unordered_map<std::string, Value*>& vmap) {
  // 获取卷积参数的计算值映射
  auto calc_value_map = getConvParams(match, vmap);
  // 检查输出填充、步幅、填充和扩展参数的维度是否为 2
  if (calc_value_map["output_padding"].toIntList().size() != 2 ||
      calc_value_map["stride"].toIntList().size() != 2 ||
      calc_value_map["padding"].toIntList().size() != 2 ||
      calc_value_map["dilation"].toIntList().size() != 2) {
    return false;
  }
  // 返回真，表示通过过滤条件
  return true;
};
    // 返回一个布尔值，指示是否不为真(transposed 不为真)
    return !calc_value_map["transposed"].toBool();
  };

  // 定义一个 lambda 表达式 filter_conv3d，接受 match 和 vmap 两个参数
  auto filter_conv3d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    // 调用 getConvParams 函数，获取计算参数的映射 calc_value_map
    auto calc_value_map = getConvParams(match, vmap);

    // 如果输出填充、步幅、填充、扩展的整数列表的大小不是 3，则返回 false
    if (calc_value_map["output_padding"].toIntList().size() != 3 ||
        calc_value_map["stride"].toIntList().size() != 3 ||
        calc_value_map["padding"].toIntList().size() != 3 ||
        calc_value_map["dilation"].toIntList().size() != 3) {
      return false;
    }
} // 结束命名空间 torch

namespace graph_rewrite_helper {
namespace jit {
namespace torch {

bool isClampFusable(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  // 从匹配对象中获取值映射
  const auto& match_vmap = match.values_map;
  // 确保在要替换的子图中能找到名为 dummy_min_max 的值
  TORCH_CHECK(
      vmap.find("dummy_min_max") != vmap.end(),
      "Expected to find dummy_min_max Value in the subgraph to be replaced.");
  // 获取名为 dummy_min_max 的值
  auto dummy_min_max =
      graph_rewrite_helper::getIValue("dummy_min_max", match_vmap, vmap);

  // 判断 dummy_min_max 是否为空或者为 None
  auto is_fusable = !dummy_min_max || dummy_min_max.value().isNone();

  // 还需检查 output_min 和 output_max 的值是否为常量
  // 如果 hardtanh 的 min/max 值不是常量，则无法删除预包装操作
  if (vmap.find("output_min") != vmap.end()) {
    // aten::relu 模式没有 output_min/output_max
    // aten::hardtanh/_ 模式有这些值
    TORCH_CHECK(
        vmap.find("output_max") != vmap.end(),
        "Expected to find output_max as well given "
        "output_min exist in pattern graph.");
    // 获取 output_min 和 output_max 的值
    auto output_min =
        graph_rewrite_helper::getIValue("output_min", match_vmap, vmap);
    auto output_max =
        graph_rewrite_helper::getIValue("output_max", match_vmap, vmap);
    // 更新 is_fusable，确保 output_min 和 output_max 都是常量
    is_fusable =
        is_fusable && (output_min.has_value() && output_max.has_value());
  }

  return is_fusable;
}

} // namespace torch::jit::graph_rewrite_helper
```