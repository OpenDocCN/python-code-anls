# `.\pytorch\torch\csrc\jit\ir\constants.cpp`

```
// 引入 Torch 的头文件 constants.h
#include <torch/csrc/jit/ir/constants.h>

// 引入 ATen 的头文件 functional.h
#include <ATen/core/functional.h>
// 引入 Torch 的自动微分变量头文件
#include <torch/csrc/autograd/variable.h>
// 引入 Torch 的 IR 头文件 ir.h
#include <torch/csrc/jit/ir/ir.h>
// 引入 Torch 的自定义运算符运行时头文件
#include <torch/csrc/jit/runtime/custom_operator.h>
// 引入 Torch 的运算符运行时头文件
#include <torch/csrc/jit/runtime/operator.h>
// 引入 Torch 的运算符注册工具函数头文件
#include <torch/csrc/jit/runtime/register_ops_utils.h>

// Torch 的命名空间声明
namespace torch::jit {

// 检查是否可以插入张量的静态函数
static bool insertableTensor(const at::Tensor& ten) {
  // 如果张量没有梯度、有存储，并且不是嵌套张量，则可以插入
  return !ten.requires_grad() && ten.has_storage() && !ten.is_nested();
}

// 检查是否可以插入 IValue 的静态函数
static bool insertableIValue(const IValue& ivalue) {
  // 如果是基本类型之一，直接返回 true
  if (ivalue.isInt() || ivalue.isNone() || ivalue.isBool() ||
      ivalue.isDouble() || ivalue.isComplexDouble() || ivalue.isString() ||
      ivalue.isDevice() || ivalue.isEnum()) {
    return true;
  }
  // 如果是张量类型，调用 insertableTensor 函数检查是否可以插入
  if (ivalue.isTensor()) {
    return insertableTensor(ivalue.toTensor());
  }
  // 如果是列表或元组类型，递归检查每个元素是否可以插入
  if (ivalue.isList() || ivalue.isTuple()) {
    c10::ArrayRef<IValue> elems;
    if (ivalue.isTuple()) {
      elems = ivalue.toTupleRef().elements();
    } else {
      elems = ivalue.toListRef();
    }
    return std::all_of(elems.begin(), elems.end(), [](const IValue& tup_elem) {
      return insertableIValue(tup_elem);
    });
  }
  // 如果是通用字典类型，递归检查每对键值是否可以插入
  if (ivalue.isGenericDict()) {
    const auto& dict = ivalue.toGenericDict();
    return std::all_of(dict.begin(), dict.end(), [](const auto& entry) {
      return insertableIValue(entry.key()) && insertableIValue(entry.value());
    });
  }

  return false;
}

// 插入常量节点到图中的函数
Value* insertConstant(
    Graph& g,
    const IValue& val,
    std::optional<SourceRange> loc,
    std::optional<ScopePtr> scope) {
  auto value = tryInsertConstant(g, val, std::move(loc), std::move(scope));
  // 如果插入成功，返回插入的值
  if (value) {
    return *value;
  }
  // 否则抛出常量不支持的错误
  throw constant_not_supported_error(
      "Unsupported value kind: " + val.tagKind());
}

// 尝试插入常量节点到图中的函数
std::optional<Value*> tryInsertConstant(
    Graph& g,
    const IValue& val,
    std::optional<SourceRange> loc,
    std::optional<ScopePtr> scope) {
  // 创建一个常量节点
  Node* n = g.create(prim::Constant);
  // 如果值是张量类型
  if (val.isTensor()) {
    at::Tensor ref = val.toTensor();
    // 如果张量不可插入，销毁节点并返回空
    if (!insertableTensor(val.toTensor())) {
      n->destroy();
      return c10::nullopt;
    }
    // 如果张量未定义，返回一个空节点输出
    if (!ref.defined()) {
      n->destroy();
      return g.insertNode(g.createNone())->output();
    }
    // 断言张量没有梯度
    TORCH_INTERNAL_ASSERT(!ref.requires_grad());
    // 推断输出节点类型为张量，并设置节点的值属性
    n->output()->inferTypeFrom(
        ref); // 注意：在 std::move(ref) 之前因为 t_ 之前
    n->t_(attr::value, std::move(ref));
  } else if (val.isInt()) {  // 如果值是整数类型
    n->i_(attr::value, val.toInt());
    n->output()->setType(IntType::get());
  } else if (val.isDouble()) {  // 如果值是双精度浮点数类型
    n->f_(attr::value, val.toDouble());
    n->output()->setType(FloatType::get());
  } else if (val.isComplexDouble()) {  // 如果值是复数双精度浮点数类型
    n->c_(attr::value, val.toComplexDouble());
    n->output()->setType(ComplexType::get());
  } else if (val.isBool()) {  // 如果值是布尔类型
    n->i_(attr::value, val.toBool());
    n->output()->setType(BoolType::get());
  } else {  // 如果值类型不支持，销毁节点并返回空
    n->destroy();
    return c10::nullopt;
  }

  // 返回节点的输出值指针
  return n->output();
}
    // 如果值是布尔类型
    n->output()->setType(BoolType::get());
  } else if (val.isList()) {
    // 如果值是列表类型
    bool fast_path_list =
        val.isBoolList() || val.isIntList() || val.isDoubleList();
    // 如果值是快速路径列表或可插入的IValue
    if (fast_path_list || insertableIValue(val)) {
      // 设置节点的属性值为val
      n->ival_(attr::value, val);
      // 设置节点的输出类型为val的类型
      n->output()->setType(val.type());
    } else {
      // 销毁节点n并返回空optional
      n->destroy();
      return c10::nullopt;
    }
  } else if (val.isString()) {
    // 如果值是字符串类型
    n->s_(attr::value, val.toStringRef());
    // 设置节点的输出类型为字符串类型
    n->output()->setType(StringType::get());
  } else if (val.isDevice()) {
    // 如果值是设备类型
    std::stringstream ss;
    ss << val.toDevice();
    // 将设备信息转换为字符串并设置节点的属性值为该字符串
    n->s_(attr::value, ss.str());
    // 设置节点的输出类型为设备对象类型
    n->output()->setType(DeviceObjType::get());
  } else if (val.isGenerator()) {
    // 如果值是生成器类型
    auto generator = val.toGenerator();
    // 设置节点的属性值为生成器对象
    n->ival_(attr::value, generator);
    // 设置节点的输出类型为生成器类型
    n->output()->setType(GeneratorType::get());
  } else if (val.isStream()) {
    // 如果值是流对象类型
    // 省略了将流对象打包为int64_t的步骤
    n->ival_(attr::value, val);
    // 设置节点的输出类型为流对象类型
    n->output()->setType(StreamObjType::get());
  } else if (val.isNone()) {
    // 如果值是None类型
    // 设置节点的输出类型为None类型
    n->output()->setType(NoneType::get());
  } else if (val.isTuple()) {
    // 如果值是元组类型
    if (insertableIValue(val)) {
      // 设置节点的属性值为val
      n->ival_(attr::value, val);
      // 设置节点的输出类型为val的类型
      n->output()->setType(val.type());
    } else {
      // 销毁节点n并返回空optional
      n->destroy();
      return c10::nullopt;
    };
  } else if (val.isObject()) {
    // 如果值是对象类型
    const auto& ref = val.toObjectRef();
    // 见：[Constant Object Weak CompilationUnit Reference]
    // 如果对象不是模块类型且是弱编译引用或空强引用字符串
    if (!ref.type()->is_module() &&
        (ref.is_weak_compilation_ref() ||
         ref.is_empty_strong_compilation_ref())) {
      // 设置节点的属性值为val
      n->ival_(attr::value, val);
      // 设置节点的输出类型为val的类型
      n->output()->setType(val.type());
    } else {
      // 销毁节点n并返回空optional
      n->destroy();
      return c10::nullopt;
    }
  } else if ((val.isGenericDict() && insertableIValue(val)) || (val.isEnum())) {
    // 如果值是泛型字典类型且可插入IValue或者是枚举类型
    // 设置节点的属性值为val
    n->ival_(attr::value, val);
    // 设置节点的输出类型为val的类型
    n->output()->setType(val.type());
  } else {
    // 如果以上条件均不满足，销毁节点n并返回空optional
    n->destroy();
    return c10::nullopt;
  }
  // 如果loc不为空，设置节点的源范围
  if (loc)
    n->setSourceRange(*loc);
  // 如果scope不为空，设置节点的作用域
  if (scope)
    n->setScope(*scope);
  // 在图中插入节点n，并返回其输出
  return g.insertNode(n)->output();
} // 结束函数定义

// 将 Value 指针转换为 IValue 的可选类型
std::optional<IValue> toIValue(const Value* v) {
    // 检查节点不是常量或者类型是函数类型，则返回空的可选类型
    if (v->node()->kind() != prim::Constant || v->type()->cast<FunctionType>()) {
        return c10::nullopt;
    }
    const Node* node = v->node();
    const TypePtr& type = v->type();
    // 如果类型是 TensorType，则返回节点的 Tensor 值
    if (type->isSubtypeOf(*TensorType::get())) {
        return node->t(attr::value);
    } else if (type->isSubtypeOf(*BoolType::get())) {
        // 如果类型是 BoolType，则返回节点的布尔值
        return (bool)node->i(attr::value);
    } else if (
        type->isSubtypeOf(*NumberType::get()) &&
        node->kindOf(attr::value) == AttributeKind::i) {
        // 如果类型是 NumberType，并且值是整数，则返回节点的整数值
        return node->i(attr::value);
    } else if (
        type->isSubtypeOf(*NumberType::get()) &&
        node->kindOf(attr::value) == AttributeKind::f) {
        // 如果类型是 NumberType，并且值是浮点数，则返回节点的浮点数值
        return node->f(attr::value);
    } else if (
        type->isSubtypeOf(*NumberType::get()) &&
        node->kindOf(attr::value) == AttributeKind::c) {
        // 如果类型是 NumberType，并且值是复数，则返回节点的复数值
        return node->c(attr::value);
    } else if (
        type->cast<ListType>() &&
        node->kindOf(attr::value) == AttributeKind::ival) {
        // 如果类型是 ListType，并且值是列表，则返回节点的列表值
        const auto& list = node->ival(attr::value);
        TORCH_INTERNAL_ASSERT(list.isList());
        return list;
    } else if (
        type->cast<DictType>() &&
        node->kindOf(attr::value) == AttributeKind::ival) {
        // 如果类型是 DictType，并且值是字典，则返回节点的字典值
        const auto& dict = node->ival(attr::value);
        TORCH_INTERNAL_ASSERT(dict.isGenericDict());
        return dict;
    } else if (
        type->cast<TupleType>() &&
        node->kindOf(attr::value) == AttributeKind::ival) {
        // 如果类型是 TupleType，并且值是元组，则返回节点的元组值
        const auto& tup = node->ival(attr::value);
        TORCH_INTERNAL_ASSERT(tup.isTuple());
        return tup;
    } else if (type == StringType::get()) {
        // 如果类型是字符串类型，则返回节点的字符串值
        const auto& s = node->s(attr::value);
        return s;
    } else if (type == DeviceObjType::get()) {
        // 如果类型是 DeviceObjType，则返回节点的设备对象值
        auto d = c10::Device(node->s(attr::value));
        return d;
    } else if (type == GeneratorType::get()) {
        // 如果类型是 GeneratorType，则返回节点的生成器对象值
        auto generator = node->ival(attr::value).toGenerator();
        return generator;
    } else if (type == StreamObjType::get()) {
        // 如果类型是 StreamObjType，则返回节点的流对象值
        // 移除了 int64_t 打包的操作
        auto s = node->ival(attr::value).toStream();
        return s;
    } else if (node->mustBeNone()) {
        // 如果节点必须是 None 类型，则返回空的 IValue
        return IValue();
    } else if (type->cast<EnumType>()) {
        // 如果类型是枚举类型，则返回节点的枚举值
        const auto& enum_val = node->ival(attr::value);
        return enum_val;
    } else if (type->cast<ClassType>() && !type->is_module()) {
        // 如果类型是类类型且不是模块类型，则返回节点的类对象值
        const auto& class_val = node->ival(attr::value);
        return class_val;
    } else {
        // 对于不支持的常量字面值类型，抛出运行时错误
        std::stringstream ss;
        ss << "constant literal not supported for: " << type->str();
        throw std::runtime_error(ss.str());
    }
}

} // 结束命名空间 torch::jit
```