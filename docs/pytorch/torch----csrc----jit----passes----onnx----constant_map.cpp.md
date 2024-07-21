# `.\pytorch\torch\csrc\jit\passes\onnx\constant_map.cpp`

```
// 包含 C++ 头文件
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/constant_map.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

// 定义命名空间 torch::jit
namespace torch {
namespace jit {

// 定义命名空间 onnx，并引入 c10::onnx 命名空间
namespace onnx {
using namespace ::c10::onnx;
}

// Meyer's 单例模式实现，用于获取 ConstantValueMap 的单例实例
ConstantValueMap& ConstantValueMap::getInstance() {
  static ConstantValueMap s;
  return s;
}

// 设置张量的秩信息
void ConstantValueMap::SetRank(
    const std::string& tensorName,
    size_t rankValue) {
  // 设置张量名到秩值的映射
  ConstantValueMap::getInstance().rankMap[tensorName] = rankValue;
  // 标记张量名使用推断类型
  ConstantValueMap::getInstance().useInferredTypeMap[tensorName] = true;
}

// 检查张量是否有秩信息
bool ConstantValueMap::HasRank(const std::string& tensorName) {
  // 检查张量名是否在 rankMap 中
  return ConstantValueMap::getInstance().rankMap.find(tensorName) !=
      ConstantValueMap::getInstance().rankMap.end();
}

// 获取张量的秩信息，如果不存在返回空值
std::optional<size_t> ConstantValueMap::GetRank(const std::string& tensorName) {
  if (!HasRank(tensorName)) {
    return c10::nullopt;
  }
  // 返回张量名对应的秩值
  return ConstantValueMap::getInstance().rankMap[tensorName];
}

// 设置所有图输入为静态类型
void ConstantValueMap::SetAllGraphInputsStatic(bool all_static) {
  // 设置所有图输入是否静态的可选值
  ConstantValueMap::getInstance().allGraphInputsStatic =
      c10::make_optional(all_static);
}

// 获取所有图输入是否静态的可选值
std::optional<bool> ConstantValueMap::GetAllGraphInputsStatic() {
  return ConstantValueMap::getInstance().allGraphInputsStatic;
}

// 设置所有图输入可靠计算状态
void ConstantValueMap::SetAllGraphInputsReliableComputed(bool computed) {
  // 设置所有图输入可靠计算状态
  ConstantValueMap::getInstance().allGraphInputsReliableComputed = computed;
}

// 获取所有图输入可靠计算状态
bool ConstantValueMap::GetAllGraphInputsReliableComputed() {
  return ConstantValueMap::getInstance().allGraphInputsReliableComputed;
}

// 设置张量的形状信息
void ConstantValueMap::SetShape(
    const std::string& tensorName,
    const c10::SymbolicShape& shapeValue) {
  // 设置张量名到形状的映射
  ConstantValueMap::getInstance().shapeMap[tensorName] = shapeValue;
  // 标记张量名使用推断类型
  ConstantValueMap::getInstance().useInferredTypeMap[tensorName] = true;
}

// 检查张量是否有形状信息
bool ConstantValueMap::HasShape(const std::string& tensorName) {
  // 检查张量名是否在 shapeMap 中
  return ConstantValueMap::getInstance().shapeMap.find(tensorName) !=
      ConstantValueMap::getInstance().shapeMap.end();
}

// 获取张量的形状信息，如果不存在返回空值
std::optional<c10::SymbolicShape> ConstantValueMap::GetShape(
    const std::string& tensorName) {
  if (!HasShape(tensorName)) {
    return c10::nullopt;
  }
  // 返回张量名对应的形状
  return ConstantValueMap::getInstance().shapeMap[tensorName];
}

// 设置张量的值
void ConstantValueMap::SetValue(
    const std::string& tensorName,
    const at::Tensor& value) {
  // 设置张量名到值的映射
  ConstantValueMap::getInstance().tensorValueMap[tensorName] = value;
}

// 检查张量是否有值
bool ConstantValueMap::HasValue(const std::string& tensorName) {
  // 检查张量名是否在 tensorValueMap 中
  return ConstantValueMap::getInstance().tensorValueMap.find(tensorName) !=
      ConstantValueMap::getInstance().tensorValueMap.end();
}

// 获取张量的值，如果不存在返回空值
std::optional<at::Tensor> ConstantValueMap::GetValue(
    const std::string& tensorName) {
  if (!HasValue(tensorName)) {
    return c10::nullopt;
  }
  // 返回张量名对应的值
  return ConstantValueMap::getInstance().tensorValueMap[tensorName];
}
// 从 ConstantValueMap 单例实例中移除指定张量名对应的常量值
void ConstantValueMap::EraseValue(const std::string& tensorName) {
  ConstantValueMap::getInstance().tensorValueMap.erase(tensorName);
}

// 将 SymbolicShape 转换为完整的一维 int64_t 向量
std::vector<int64_t> ConstantValueMap::GetCompleteShapeInto1DInt64Vector(
    const c10::SymbolicShape& shape) {
  // 内部断言，确保 shape 是完整的
  TORCH_INTERNAL_ASSERT(shape.isComplete());
  std::vector<int64_t> shape_value;
  // 获取 shape 中的 sizes，并预留空间
  auto shape_symbol_list = shape.sizes().value();
  shape_value.reserve(shape_symbol_list.size());
  // 遍历 shape 中的每个符号尺寸，并将其静态大小添加到 shape_value 中
  for (const auto& v : shape_symbol_list) {
    shape_value.emplace_back(v.static_size());
  }
  return shape_value;
}

// 获取指定值的形状作为一维 int64_t 向量的可选值
std::optional<std::vector<int64_t>> ConstantValueMap::GetShapeInto1DInt64Vector(
    const std::string& value_name) {
  // 如果 ConstantValueMap 中存在指定值的形状
  if (ConstantValueMap::HasShape(value_name)) {
    auto shape_size = ConstantValueMap::GetShape(value_name).value();
    // 如果形状是完整的
    if (shape_size.isComplete()) {
      // 获取完整形状的一维 int64_t 向量
      auto shape_value =
          ConstantValueMap::GetCompleteShapeInto1DInt64Vector(shape_size);
      return shape_value;
    }
  }
  return c10::nullopt;
}

// 获取带有一个未知尺寸的形状作为一维 int64_t 向量的可选值
std::optional<std::vector<int64_t>> ConstantValueMap::
    GetShapeInto1DInt64VectorWithOneUnknown(const std::string& value_name) {
  // 如果 ConstantValueMap 中存在指定值的形状
  if (ConstantValueMap::HasShape(value_name)) {
    auto shape_size = ConstantValueMap::GetShape(value_name).value();
    std::vector<int64_t> shape_value;
    // 如果形状是完整的
    if (shape_size.isComplete()) {
      // 获取完整形状的一维 int64_t 向量
      shape_value =
          ConstantValueMap::GetCompleteShapeInto1DInt64Vector(shape_size);
      return shape_value;
    } else {
      size_t count_unknown = 0;
      auto shape_size_sizes = shape_size.sizes();
      // 如果形状中有尺寸值
      if (shape_size_sizes.has_value()) {
        auto shape_symbol_list = shape_size_sizes.value();
        // 遍历形状中的每个符号尺寸
        for (const auto& v : shape_symbol_list) {
          if (v.is_static()) {
            // 如果尺寸是静态的，则将其静态大小添加到 shape_value 中
            shape_value.emplace_back(v.static_size());
          } else {
            // 如果尺寸是动态的，则添加 -1 表示未知，并增加未知计数
            shape_value.emplace_back(-1);
            count_unknown += 1;
          }
        }
        // 如果只有一个未知尺寸，则返回 shape_value
        if (count_unknown == 1) {
          return shape_value;
        }
      }
    }
  }
  return c10::nullopt;
}

// 对于 1 维 int64_t 情况的 accessor<int64_t, 1> 访问器
std::vector<int64_t> ConstantValueMap::GetValueInto1DInt64Vector(
    const std::string& value_name) {
  // 获取指定值的常量值，并将其转换为 Long 类型
  auto value = ConstantValueMap::GetValue(value_name).value();
  auto value_int64_t = value.toType(at::ScalarType::Long);
  std::vector<int64_t> value_vector;
  // 预留空间以容纳值的大小
  value_vector.reserve(value_int64_t.size(0));
  // 获取值的 accessor<int64_t, 1>，并将其元素添加到 value_vector 中
  auto value_size_a = value_int64_t.accessor<int64_t, 1>();
  for (const auto i : c10::irange(value_int64_t.size(0))) {
    value_vector.emplace_back(static_cast<int64_t>(value_size_a[i]));
  }
  return value_vector;
}

// 设置指定张量名的类型可靠性
void ConstantValueMap::SetTypeReliable(
    const std::string& tensorName,
    bool value) {
  ConstantValueMap::getInstance().typeReliableMap[tensorName] = value;
}

// 检查指定张量名是否具有类型可靠性的标记
bool ConstantValueMap::HasTypeReliable(const std::string& tensorName) {
  // 检查 ConstantValueMap 中是否存在指定张量名的类型可靠性标记
  return ConstantValueMap::getInstance().typeReliableMap.find(tensorName) !=
      ConstantValueMap::getInstance().typeReliableMap.end();
}
    # 检查给定的张量名称是否在可靠类型映射中存在，如果不存在则返回空值
    const std::string& tensorName) {
        if (!HasTypeReliable(tensorName)) {
            # 如果张量名称不存在于可靠类型映射中，则返回空的optional对象
            return c10::nullopt;
        }
        # 如果存在，返回该张量名称在常量数值映射中对应的值
        return ConstantValueMap::getInstance().typeReliableMap[tensorName];
// 设置在推断类型中使用的张量名称及其对应的布尔值
void ConstantValueMap::SetUseInferredType(
    const std::string& tensorName,
    bool value) {
  // 使用单例模式获取常量值映射对象，并设置张量名称对应的推断类型使用布尔值
  ConstantValueMap::getInstance().useInferredTypeMap[tensorName] = value;
}

// 检查给定张量名称是否在推断类型映射中存在
bool ConstantValueMap::HasUseInferredType(const std::string& tensorName) {
  // 使用单例模式获取常量值映射对象，并检查张量名称是否存在于推断类型映射中
  return ConstantValueMap::getInstance().useInferredTypeMap.find(tensorName) !=
      ConstantValueMap::getInstance().useInferredTypeMap.end();
}

// 获取给定张量名称的推断类型使用情况的可选布尔值
std::optional<bool> ConstantValueMap::GetUseInferredType(
    const std::string& tensorName) {
  // 如果张量名称不存在于推断类型映射中，则返回空
  if (!HasUseInferredType(tensorName)) {
    return c10::nullopt;
  }
  // 使用单例模式获取常量值映射对象，并返回张量名称对应的推断类型使用布尔值
  return ConstantValueMap::getInstance().useInferredTypeMap[tensorName];
}

// 设置张量的形状值
void ConstantValueMap::SetShapeValue(
    const std::string& tensorName,
    const c10::SymbolicShape& shapeValue) {
  // 使用单例模式获取常量值映射对象，并设置张量名称对应的形状值
  ConstantValueMap::getInstance().shapeValueMap[tensorName] = shapeValue;
}

// 检查给定张量名称是否在形状值映射中存在
bool ConstantValueMap::HasShapeValue(const std::string& tensorName) {
  // 使用单例模式获取常量值映射对象，并检查张量名称是否存在于形状值映射中
  return ConstantValueMap::getInstance().shapeValueMap.find(tensorName) !=
      ConstantValueMap::getInstance().shapeValueMap.end();
}

// 获取给定张量名称的形状值的可选符号形状
std::optional<c10::SymbolicShape> ConstantValueMap::GetShapeValue(
    const std::string& tensorName) {
  // 如果张量名称不存在于形状值映射中，则返回空
  if (!HasShapeValue(tensorName)) {
    return c10::nullopt;
  }
  // 使用单例模式获取常量值映射对象，并返回张量名称对应的形状值
  return ConstantValueMap::getInstance().shapeValueMap[tensorName];
}

// 获取推断形状数据的引用，这些数据通过ONNX数据传播获得
ShapeDataMap& ConstantValueMap::GetInferredShapeData() {
  // 返回常量值映射对象中的推断形状数据映射
  return ConstantValueMap::getInstance().inferredShapeData;
}

// 获取符号维度映射的引用
SymbolDimMap& ConstantValueMap::GetSymbolDimMap() {
  // 返回常量值映射对象中的符号维度映射
  return ConstantValueMap::getInstance().symbolDimMap;
}

// 获取维度符号映射的引用
DimSymbolMap& ConstantValueMap::GetDimSymbolMap() {
  // 返回常量值映射对象中的维度符号映射
  return ConstantValueMap::getInstance().dimSymbolMap;
}

// 更新字符串键的映射，将旧键更新为新键
template <typename Map>
void UpdateStrKey(
    Map& map,
    const std::string& old_key,
    const std::string& new_key) {
  // 内部断言，确保旧键不等于新键
  TORCH_INTERNAL_ASSERT(old_key != new_key);
  // 如果映射中不存在旧键，则直接返回
  if (map.find(old_key) == map.end()) {
    return;
  }
  // 将映射中的旧键对应的值复制到新键上，并删除旧键
  map[new_key] = map[old_key];
  map.erase(old_key);
}

// 更新值的名称
void ConstantValueMap::UpdateValueName(
    const std::string& old_name,
    const std::string& new_name) {
  // 如果旧名称等于新名称，则直接返回
  if (old_name == new_name) {
    return;
  }
  // 分别更新常量值映射对象中的多个映射，将旧名称更新为新名称
  UpdateStrKey<decltype(rankMap)>(
      ConstantValueMap::getInstance().rankMap, old_name, new_name);
  UpdateStrKey<decltype(shapeMap)>(
      ConstantValueMap::getInstance().shapeMap, old_name, new_name);
  UpdateStrKey<decltype(tensorValueMap)>(
      ConstantValueMap::getInstance().tensorValueMap, old_name, new_name);
  UpdateStrKey<decltype(typeReliableMap)>(
      ConstantValueMap::getInstance().typeReliableMap, old_name, new_name);
  UpdateStrKey<decltype(useInferredTypeMap)>(
      ConstantValueMap::getInstance().useInferredTypeMap, old_name, new_name);
  UpdateStrKey<decltype(shapeValueMap)>(
      ConstantValueMap::getInstance().shapeValueMap, old_name, new_name);
  UpdateStrKey<decltype(inferredShapeData)>(
      ConstantValueMap::getInstance().inferredShapeData, old_name, new_name);
}
// 清空 ConstantValueMap 单例中的各个映射和属性
void ConstantValueMap::ClearMaps() {
  // 清空排名映射
  ConstantValueMap::getInstance().rankMap.clear();
  // 清空形状映射
  ConstantValueMap::getInstance().shapeMap.clear();
  // 清空张量值映射
  ConstantValueMap::getInstance().tensorValueMap.clear();
  // 清空类型可靠性映射
  ConstantValueMap::getInstance().typeReliableMap.clear();
  // 清空使用推断类型映射
  ConstantValueMap::getInstance().useInferredTypeMap.clear();
  // 清空形状值映射
  ConstantValueMap::getInstance().shapeValueMap.clear();
  // 清空推断形状数据
  ConstantValueMap::getInstance().inferredShapeData.clear();
  // 清空符号维度映射
  ConstantValueMap::getInstance().symbolDimMap.clear();
  // 清空维度符号映射
  ConstantValueMap::getInstance().dimSymbolMap.clear();
  // 重置所有图输入是否静态的状态为未确定
  ConstantValueMap::getInstance().allGraphInputsStatic = c10::nullopt;
  // 重置所有图输入是否可靠计算的状态为未计算
  ConstantValueMap::getInstance().allGraphInputsReliableComputed = false;
}

// 仅用于调试。
void ConstantValueMap::PrintMaps() {
  std::cout << "Rank/Shape Map:" << std::endl;
  // 遍历排名映射并打印
  for (const auto& x : ConstantValueMap::getInstance().rankMap) {
    std::stringstream ss;
    // 如果形状映射中存在该节点名，则获取其形状符号
    if (ConstantValueMap::getInstance().shapeMap.find(x.first) !=
        ConstantValueMap::getInstance().shapeMap.end()) {
      auto shape_symbols =
          ConstantValueMap::getInstance().shapeMap[x.first].sizes();
      if (shape_symbols.has_value()) {
        // 打印形状信息
        for (const auto& shape_symbol : shape_symbols.value()) {
          if (shape_symbol.is_static()) {
            ss << shape_symbol.static_size() << ", ";
          } else {
            ss << "*, ";
          }
        }
      }
    }
    ss << " (rank = " << x.second << ")";
    std::cout << "node " << x.first << ": " << ss.str() << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Value Map:" << std::endl;
  // 打印张量值映射
  for (const auto& x : ConstantValueMap::getInstance().tensorValueMap) {
    std::cout << "node " << x.first << ": " << x.second << std::endl;
  }
  std::cout << std::endl;
  std::cout << "TypeReliable Map:" << std::endl;
  size_t count = 0;
  // 打印类型可靠性映射，每行打印10个
  for (const auto& x : ConstantValueMap::getInstance().typeReliableMap) {
    std::cout << "(node " << x.first << ": " << x.second << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  std::cout << "UseInferredType Map:" << std::endl;
  count = 0;
  // 打印使用推断类型映射，每行打印10个
  for (const auto& x : ConstantValueMap::getInstance().useInferredTypeMap) {
    std::cout << "(node " << x.first << ": " << x.second << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  std::cout << "ShapeValue Map:" << std::endl;
  count = 0;
  // 打印形状值映射，每行打印10个
  for (const auto& x : ConstantValueMap::getInstance().shapeValueMap) {
    std::cout << "(node " << x.first << ": " << x.second << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  std::cout << "InferredShape Map:" << std::endl;
  count = 0;
  // 打印推断形状数据，每行打印10个
  for (const auto& x : ConstantValueMap::getInstance().inferredShapeData) {
    std::cout << "(node " << x.first << ": ";
    // 遍历 x.second 中的每一个 dim 对象
    for (const auto& dim : x.second.dim()) {
      // 检查 dim 是否有维度参数，如果有，则输出维度参数
      if (dim.has_dim_param()) {
        std::cout << dim.dim_param() << " ";
      } else {
        // 否则输出维度值
        std::cout << dim.dim_value() << " ";
      }
    }
    // 输出结束符号
    std::cout << "), ";
    // 增加计数器
    count++;
    // 每输出 10 个元素换行一次
    if (count % 10 == 0) {
      std::cout << std::endl;
    }
  }
  // 输出一个空行
  std::cout << std::endl;
  // 输出标题 "SymbolDim Map:"
  std::cout << "SymbolDim Map:" << std::endl;
  // 重置计数器
  count = 0;
  // 遍历 ConstantValueMap 的 symbolDimMap
  for (const auto& x : ConstantValueMap::getInstance().symbolDimMap) {
    // 输出每个键值对，格式为 (键: 值),
    std::cout << "(" << x.first << ": " << x.second << "), ";
    // 增加计数器
    count++;
    // 每输出 10 个元素换行一次
    if (count % 10 == 0) {
      std::cout << std::endl;
    }
  }
  // 输出一个空行
  std::cout << std::endl;
  // 输出标题 "DimSymbol Map:"
  std::cout << "DimSymbol Map:" << std::endl;
  // 重置计数器
  count = 0;
  // 遍历 ConstantValueMap 的 dimSymbolMap
  for (const auto& x : ConstantValueMap::getInstance().dimSymbolMap) {
    // 输出每个键值对，格式为 (键: 值),
    std::cout << "(" << x.first << ": " << x.second << "), ";
    // 增加计数器
    count++;
    // 每输出 10 个元素换行一次
    if (count % 10 == 0) {
      std::cout << std::endl;
    }
  }
}

} // namespace jit
} // namespace torch
```