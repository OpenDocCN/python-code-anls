# `.\pytorch\torch\csrc\lazy\core\shape.cpp`

```py
// 包含所需的头文件：范围处理、张量形状和张量核心
#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/tensor.h>

// 定义一个布尔标志，用于控制是否启用符号形状的计算
C10_DEFINE_bool(
    ltc_enable_symbolic_shapes,
    false,
    "Enables calculation of if dims are symbolic");

// 命名空间 torch 中的命名空间 lazy
namespace torch {
namespace lazy {

// 构造函数：初始化形状对象的标量类型、尺寸和是否为符号形状的可选值
Shape::Shape(
    at::ScalarType scalar_type,
    c10::ArrayRef<int64_t> sizes,
    std::optional<std::vector<bool>> is_symbolic)
    : scalar_type_(scalar_type),
      sizes_(sizes.begin(), sizes.end()), // 初始化尺寸
      is_symbolic_(std::move(is_symbolic)) {} // 初始化是否为符号形状

// 将形状对象转换为字符串表示
std::string Shape::to_string() const {
  return c10::str(toString(scalar_type_), "[", c10::Join(",", sizes_), "]");
}

// 形状对象的相等比较运算符重载
bool Shape::operator==(const Shape& other) const {
  return scalar_type_ == other.scalar_type_ && sizes_ == other.sizes_;
}

// 输出形状对象的字符串表示
std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.to_string();
}

// 计算形状对象中元素的总数
size_t Shape::numel() const {
  size_t elts = 1;
  for (auto size : sizes_) {
    elts *= size;
  }
  return elts;
}

// 计算形状对象的哈希值
hash_t Shape::hash(bool bakeInSizes) const {
  if (bakeInSizes) {
    return HashCombine(
        Hash(scalar_type_),
        DataHash(sizes_.data(), sizes_.size() * sizeof(int64_t)));
  } else {
    return HashCombine(Hash(scalar_type_), Hash(sizes_.size()));
  }
}

// 返回具有指定符号维度的新形状对象副本
Shape Shape::with_symbolic_dims(
    std::optional<std::vector<bool>> symbolic_dims) const {
  Shape copy = *this;
  copy.is_symbolic_ = symbolic_dims;
  return copy;
}

// 检查是否启用符号形状计算
bool symbolicShapeEnabled() {
  static bool enabled = std::getenv("LTC_ENABLE_SYMBOLIC_SHAPES") != nullptr;
  return enabled || FLAGS_ltc_enable_symbolic_shapes;
}

// 获取张量的符号形状
static c10::SymbolicShape get_symbolic_shape(at::Tensor& tensor) {
  auto ltc_tensor = TryGetLtcTensor(tensor);
  if (!ltc_tensor) {
    // 对于具体张量设置具体尺寸
    return c10::SymbolicShape(tensor.sizes());
  }
  const Shape& input_shape = ltc_tensor->GetIrValue()->shape();
  auto& is_symbolic = input_shape.is_symbolic();
  if (!is_symbolic.has_value()) {
    return c10::SymbolicShape();
  }
  auto sizes = input_shape.sizes();
  TORCH_INTERNAL_ASSERT(
      sizes.size() == is_symbolic->size(),
      "Dims of two values are not consistent");
  std::vector<std::optional<int64_t>> symbolic_dims;
  for (size_t i = 0; i < sizes.size(); i++) {
    if (is_symbolic->at(i)) {
      symbolic_dims.emplace_back(c10::nullopt);
    } else {
      symbolic_dims.emplace_back(sizes.at(i));
    }
  }
  return c10::SymbolicShape(symbolic_dims);
}

// 在 LazyTensor 上应用符号形状
void applySymbolicShapesOnLT(
    const char* schema_str,
    std::vector<c10::IValue> args,
    std::vector<Shape>& result_shapes) {
  std::vector<jit::SSAInput> converted_args;
  // TODO: 确定 LazyTensor 中是否有未知值

  // 获取操作符架构，根据字面量获取函数架构
  const c10::FunctionSchema& schema =
      jit::getOperatorForLiteral(schema_str)->schema();

  // 遍历参数列表
  for (auto& arg : args) {
    // 处理张量列表
    // 如果参数是一个张量列表
    if (arg.isTensorList()) {
      // 将参数转换为张量列表
      at::List<at::Tensor> tensor_list = arg.toTensorList();
      // 遍历张量列表中的每个张量
      for (at::Tensor tensor : tensor_list) {
        // 获取张量的符号化形状并添加到转换后的参数列表中
        converted_args.emplace_back(get_symbolic_shape(tensor));
      }
    } else if (arg.isTensor()) {
      // 如果参数是单个张量
      auto ss = get_symbolic_shape(arg.toTensor());
      // 获取张量的符号化形状并添加到转换后的参数列表中
      converted_args.emplace_back(ss);
    } else {
      // 如果需要支持符号整数，则在此处添加支持
      // 否则，将参数直接添加到转换后的参数列表中
      converted_args.emplace_back(arg);
    }
  }
  // 调用 JIT 函数来计算操作的符号化形状
  auto res_symbolic = jit::calculateSymbolicShapesOnOp(&schema, converted_args);
  // 如果符号化形状计算失败
  if (!res_symbolic) {
    // 将每个结果形状设为没有符号维度
    for (auto& result_shape : result_shapes) {
      result_shape = result_shape.with_symbolic_dims(c10::nullopt);
    }
  } else {
    // 否则，检查结果形状的大小是否一致
    TORCH_INTERNAL_ASSERT(
        res_symbolic->size() == result_shapes.size(),
        "Result shape size is not consistent");
    // 遍历每个计算出的符号化形状
    for (size_t i = 0; i < res_symbolic->size(); i++) {
      // 获取当前结果的符号维度（如果有）
      auto sym_dims = res_symbolic->at(i).symbolicDims();
      // 如果存在符号维度，则更新结果形状的符号维度
      if (sym_dims.has_value()) {
        result_shapes[i] = result_shapes[i].with_symbolic_dims(*sym_dims);
      }
    }
  }
}

} // namespace lazy
} // namespace torch
```