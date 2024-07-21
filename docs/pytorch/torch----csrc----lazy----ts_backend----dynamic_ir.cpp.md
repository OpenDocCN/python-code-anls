# `.\pytorch\torch\csrc\lazy\ts_backend\dynamic_ir.cpp`

```
// 引入 Torch 动态 IR 头文件
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>

// 定义函数 DimCast，用于将输出节点转换为 DimensionNode 类型指针
static const torch::lazy::DimensionNode* DimCast(torch::lazy::Output output) {
  return dynamic_cast<const torch::lazy::DimensionNode*>(output.node);
}

// 命名空间 torch::lazy 的开始
namespace torch {
namespace lazy {

// SizeNode 类的 Lower 方法实现，将 SizeNode 转换为 Torch Script 操作向量
TSOpVector SizeNode::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {
  // 创建参数向量和关键字参数向量
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(2);
  // 插入常量表示维度索引
  auto index = loctx->graph()->insertConstant(static_cast<int64_t>(this->dim_));
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(index);
  // 调用 LowerTSBuiltin 函数，返回 Lower 的 Torch Script 内置操作向量
  torch::lazy::TSOpVector size_out =
      torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
  // 检查操作向量的大小是否为 1
  TORCH_CHECK_EQ(size_out.size(), 1);
  // 返回操作向量
  return size_out;
}

// SizeNode 类的构造函数实现，初始化 TsNode 基类和 dim_ 成员
SizeNode::SizeNode(Value input, size_t dim)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::size")},
          {input},
          std::vector<Shape>{},
          1,
          MHash(dim)),
      dim_(dim){};

// SizeNode 类的 getStaticValue 方法实现，返回静态值
int64_t SizeNode::getStaticValue() const {
  return dynamic_cast<const TsNode*>(operand(0).node)->shape(0).size(dim_);
}

// SizeNode 类的 isSymbolic 方法实现，检查是否为符号化
bool SizeNode::isSymbolic() const {
  auto symbolic_vec =
      dynamic_cast<const TsNode*>(operand(0).node)->shape(0).is_symbolic();
  if (!symbolic_vec.has_value()) {
    return true;
  }
  return symbolic_vec->at(dim_);
}

// SizeNode 类的 ToString 方法实现，返回描述字符串
std::string SizeNode::ToString() const {
  return "SizeNode";
}

// SizeAdd 类的构造函数实现，初始化 TsNode 基类
SizeAdd::SizeAdd(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::add")},
          {a, b},
          std::vector<Shape>{},
          1){};

// SizeAdd 类的 getStaticValue 方法实现，返回两个操作数的静态值之和
int64_t SizeAdd::getStaticValue() const {
  return DimCast(operand(0))->getStaticValue() +
      DimCast(operand(1))->getStaticValue();
}

// SizeAdd 类的 isSymbolic 方法实现，检查两个操作数是否有符号化
bool SizeAdd::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

// SizeAdd 类的 ToString 方法实现，返回描述字符串
std::string SizeAdd::ToString() const {
  return "SizeAdd";
}

// SizeMul 类的构造函数实现，初始化 TsNode 基类
SizeMul::SizeMul(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::mul")},
          {a, b},
          std::vector<Shape>{},
          1){};

// SizeMul 类的 getStaticValue 方法实现，返回两个操作数的静态值之积
int64_t SizeMul::getStaticValue() const {
  return DimCast(operand(0))->getStaticValue() *
      DimCast(operand(1))->getStaticValue();
}

// SizeMul 类的 isSymbolic 方法实现，检查两个操作数是否有符号化
bool SizeMul::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

// SizeMul 类的 ToString 方法实现，返回描述字符串
std::string SizeMul::ToString() const {
  return "SizeMul";
}

// SizeDiv 类的构造函数实现，初始化 TsNode 基类
SizeDiv::SizeDiv(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::div")},
          {a, b},
          std::vector<Shape>{},
          1){};

// SizeDiv 类的 getStaticValue 方法实现，返回两个操作数的静态值之商
int64_t SizeDiv::getStaticValue() const {
  // 检查分母是否为零
  TORCH_CHECK(
      DimCast(operand(1))->getStaticValue() != 0,
      "Can't divide a dimension by zero");
  return DimCast(operand(0))->getStaticValue() /
      DimCast(operand(1))->getStaticValue();
}
# 判断 SizeDiv 对象的第一个操作数或第二个操作数是否为符号表达式，返回布尔值
bool SizeDiv::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

# 返回 "SizeDiv" 字符串，用于表示 SizeDiv 对象的文本表示
std::string SizeDiv::ToString() const {
  return "SizeDiv";
}

# 结束 lazy 命名空间
} // namespace lazy

# 结束 torch 命名空间
} // namespace torch
```