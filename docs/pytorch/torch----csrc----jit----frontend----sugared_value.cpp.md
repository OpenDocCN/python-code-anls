# `.\pytorch\torch\csrc\jit\frontend\sugared_value.cpp`

```
// 引入 Torch JIT 前端的相关头文件
#include <torch/csrc/jit/frontend/sugared_value.h>

// 引入 C++ 标准库和 Torch 库的其他头文件
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>

// 定义命名空间 torch::jit
namespace torch::jit {

// 定义 NoneValue 结构体，实现 SugaredValue 接口
struct NoneValue : SugaredValue {
  NoneValue() = default;

  // 返回对象类型的字符串表示
  std::string kind() const override {
    return "None";
  }
};

// 实现 PrintValue 类的 call 方法
std::shared_ptr<SugaredValue> PrintValue::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  auto& g = *m.graph();

  // 如果有关键字参数，则抛出错误
  if (!kwargs.empty())
    throw ErrorReport(loc) << "print doesn't accept any keyword arguments";

  // 将输入参数转换为值列表
  std::vector<Value*> lowered_inputs = toValues(*m.graph(), args);

  // 在图中插入一个打印节点，将其位置设置为 loc
  g.insertNode(g.create(prim::Print, lowered_inputs, 0)->setSourceRange(loc));

  // 返回一个空值的共享指针，表示成功调用结束
  return std::make_shared<NoneValue>();
}

// 定义静态函数 builtin_cast_method_to_scalar_type，返回一个常量引用
static const std::unordered_map<std::string, at::ScalarType>&
builtin_cast_method_to_scalar_type() {
  // 静态局部变量 mapping，映射字符串到 Torch 标量类型
  static std::unordered_map<std::string, at::ScalarType> mapping = {
      {"byte", at::kByte},
      {"char", at::kChar},
      {"double", at::kDouble},
      {"float", at::kFloat},
      {"cfloat", at::kComplexFloat},
      {"cdouble", at::kComplexDouble},
      {"int", at::kInt},
      {"long", at::kLong},
      {"short", at::kShort},
      {"half", at::kHalf}};
  return mapping; // 返回映射表
}

// 实现 BuiltinFunction 类的 call 方法
std::shared_ptr<SugaredValue> BuiltinFunction::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  // 调用 emitBuiltinCall 函数，生成内置调用表达式，并返回 SimpleValue 包装结果
  return std::make_shared<SimpleValue>(
      emitBuiltinCall(loc, *m.graph(), symbol, args, kwargs, self));
}

// 定义 EnumClassHash 结构体，实现哈希函数对象
struct EnumClassHash {
  // 模板函数，接受任意类型参数 T，将其转换为 size_t 返回
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

// 实现 SimpleValue 类的 hasAttr 方法
bool SimpleValue::hasAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // 如果值类型是 ClassType
  if (auto class_type = value_->type()->cast<ClassType>()) {
    // 检查是否具有方法、属性或常量
    return class_type->hasMethod(field) || class_type->hasAttribute(field) ||
        class_type->hasConstant(field);
  }
  // 如果值类型是 TupleType
  else if (auto tuple_type = value_->type()->cast<TupleType>()) {
    // 如果存在 schema
    if (tuple_type->schema()) {
      // 遍历 schema 的参数，检查是否存在指定名称的参数
      for (const auto& arg : tuple_type->schema()->arguments()) {
        if (arg.name() == field) {
          return true;
        }
      }
      return false; // 找不到指定名称的参数
    } else {
      // 抛出错误，要求第一个参数必须是对象或 NamedTuple 类型，但是得到了普通 Tuple 类型
      throw ErrorReport(loc) << "hasattr's first argument must be a object "
                             << "or NamedTuple, but got a normal Tuple "
                             << value_->type()->repr_str() << " instead";
    }
  }
  // 如果类型不是 ClassType 或 TupleType，则默认返回 false
  return false;
}
    }
  }
  throw ErrorReport(loc) << "hasattr's first argument must be an object or "
                         << "NamedTuple, got " << value_->type()->repr_str()
                         << " instead";



    // 结束上一个if语句块
    }
  // 结束上一个for语句块
  // 抛出错误报告，指示hasattr的第一个参数必须是对象或NamedTuple，实际传入的是value_的类型描述字符串
  throw ErrorReport(loc) << "hasattr's first argument must be an object or "
                         << "NamedTuple, got " << value_->type()->repr_str()
                         << " instead";
}

// 支持语法糖，使得 x.foo(y, z) 可以通过允许 x.foo 返回一个可调用的值来解析为 foo(x, y, z)。
std::shared_ptr<SugaredValue> SimpleValue::attr(
    const SourceRange& loc,                    // 属性访问位置的源范围
    GraphFunction& m,                          // 图函数的引用
    const std::string& field) {                // 属性字段名称

  // 允许在 Tensor 类型上进行方法式的类型转换，例如 x.int()
  if (value_->type()->isSubtypeOf(*TensorType::get())) {
    if (builtin_cast_method_to_scalar_type().count(field)) {
      return std::make_shared<TensorCastValue>(
          builtin_cast_method_to_scalar_type().at(field),
          NamedValue(loc, "self", value_));
    }
  }

  // 访问作为 prim:: 或 aten:: 操作符实现的 Tensor 和 Device 的属性
  using PropertiesLookup = std::unordered_map<
      TypeKind,
      std::unordered_map<std::string, std::string>,
      EnumClassHash>;
  static const PropertiesLookup builtin_properties = {
      {TypeKind::OptionalType,
       {
           {"unchecked_unwrap_optional", "prim"},
       }},
      {TypeKind::TensorType,
       {
           {"dtype", "prim"},
           {"device", "prim"},
           {"grad", "prim"},
           {"data", "prim"},
           {"shape", "prim"},
           {"is_cuda", "prim"},
           {"is_cpu", "prim"},
           {"is_xla", "prim"},
           {"is_xpu", "prim"},
           {"is_sparse", "prim"},
           {"is_sparse_csr", "prim"},
           {"is_mkldnn", "prim"},
           {"is_mps", "prim"},
           {"is_mtia", "prim"},
           {"is_quantized", "prim"},
           {"is_vulkan", "prim"},
           {"is_ipu", "prim"},
           {"is_meta", "prim"},
           {"is_leaf", "aten"},
           {"is_nested", "prim"},
           {"requires_grad", "prim"},
           {"layout", "prim"},
           {"T", "prim"},
           {"H", "prim"},
           {"mT", "aten"},
           {"mH", "aten"},
           {"is_maia", "prim"},
           {"itemsize", "prim"},
           {"nbytes", "prim"},
           {"ndim", "prim"},
           {"name", "prim"},
           {"real", "aten"},
           {"imag", "aten"},
           {"retains_grad", "aten"},
       }},
      {TypeKind::DeviceObjType, {{"type", "prim"}, {"index", "prim"}}}};
  
  auto kind = value_->type()->kind();          // 获取值的类型种类
  auto types_for_builtin = builtin_properties.find(kind);   // 查找类型种类对应的属性映射
  if (types_for_builtin != builtin_properties.end()) {
    auto builtin_entry = types_for_builtin->second.find(field);   // 查找属性映射中的具体字段
    if (builtin_entry != types_for_builtin->second.end()) {
      // 如果找到内置属性，则将其添加到图中
      auto the_namespace = builtin_entry->second;
      auto r = m.graph()->insert(
          Symbol::fromQualString(the_namespace + "::" + field), {value_});
      return std::make_shared<SimpleValue>(r);
    }
  }

  // 访问命名元组的字段
  if (auto tuple_type = value_->type()->cast<TupleType>()) {
    // 检查 tuple_type 是否存在有效的 schema
    if (tuple_type->schema()) {
      // 获取 tuple_type 的所有属性
      auto attrs = tuple_type->schema()->arguments();
      // 遍历属性列表
      for (const auto i : c10::irange(attrs.size())) {
        // 检查属性是否与指定的 field 匹配
        if (attrs[i].name() == field) {
          // 在计算图中插入常量节点，表示属性的索引值
          auto idx = m.graph()->insertConstant(IValue(static_cast<int64_t>(i)));
          // 获取指定索引位置的元素类型
          auto out_type = tuple_type->elements().at(i);
          // 在计算图中插入节点，表示从元组中获取指定索引的元素
          auto r = m.graph()
                       ->insertNode(
                           m.graph()->createTupleIndex(value_, idx, out_type))
                       ->output();
          // 返回表示结果的 SimpleValue 共享指针
          return std::make_shared<SimpleValue>(r);
        }
      }
    }
  } else if (auto awaitType = value_->type()->cast<AwaitType>()) {
    // 如果值的类型是 AwaitType，则处理等待类型的情况
    auto elType = awaitType->getElementType();
    auto& g = *m.graph();
    // 在计算图中插入 awaitable_wait 操作节点
    auto v = g.insert(prim::awaitable_wait, {value_}, {}, loc);
    // 创建并返回表示结果的 SimpleValue 共享指针
    auto sv = std::make_shared<SimpleValue>(v);
    return sv->attr(loc, m, field);
  } else if (auto classType = value_->type()->cast<ClassType>()) {
    // 如果值的类型是 ClassType，则处理类的属性访问
    // 检查是否存在与 field 匹配的方法
    if (classType->findMethod(field)) {
      // 返回表示方法的 MethodValue 共享指针
      return std::make_shared<MethodValue>(getValue(), field);
    }
    // 检查是否存在与 field 匹配的属性
    if (classType->hasAttribute(field)) {
      auto& g = *m.graph();
      // 在计算图中插入 get 属性节点
      auto n = g.insertNode(g.createGetAttr(value_, field));
      // 创建并返回表示结果的 SimpleValue 共享指针
      return std::make_shared<SimpleValue>(n->output());
    }
    // 检查是否存在与 field 匹配的 getter 属性
    auto prop = classType->getProperty(field);
    if (prop) {
      // 调用 MethodValue 的 call 方法，返回方法调用结果
      return MethodValue(value_, prop->getter->name())
          .call(loc, m, {}, {}, /*n_binders=*/1);
    }
  } else if (auto iface = value_->type()->cast<InterfaceType>()) {
    // 如果值的类型是 InterfaceType，则处理接口的方法访问
    // 检查是否存在与 field 匹配的方法
    if (iface->getMethod(field)) {
      // 返回表示方法的 MethodValue 共享指针
      return std::make_shared<MethodValue>(getValue(), field);
    }
  } else if (auto enum_type = value_->type()->cast<EnumType>()) {
    // 如果值的类型是 EnumType，则处理枚举类型的属性访问
    auto& g = *m.graph();
    // 检查是否访问枚举的 name 属性
    if (field == "name") {
      // 在计算图中插入 enum 名称节点
      auto n = g.insertNode(g.createEnumName(value_));
      // 创建并返回表示结果的 SimpleValue 共享指针
      return std::make_shared<SimpleValue>(n->output());
    }
    // 检查是否访问枚举的 value 属性
    if (field == "value") {
      // 在计算图中插入 enum 值节点
      auto n = g.insertNode(g.createEnumValue(value_));
      // 创建并返回表示结果的 SimpleValue 共享指针
      return std::make_shared<SimpleValue>(n->output());
    }
  }

  // 如果以上所有更具体的情况都不适用，则尝试处理内置方法或函数调用
  // 如果 field 是 "type"，则调用 aten::to 操作
  if (field == "type") {
    if (auto builtin = BuiltinFunction::tryCreate(
            Symbol::aten("to"), NamedValue(loc, "self", value_))) {
      // 返回内置函数的结果
      return builtin;
    }
  }

  // 尝试调用具有给定 field 名称的内置函数
  if (auto builtin = BuiltinFunction::tryCreate(
          Symbol::aten(field), NamedValue(loc, "self", value_))) {
    // 返回内置函数的结果
    return builtin;
  }

  // 处理在 Tensor 上调用 tolist() 方法的情况
  if (value_->type()->isSubtypeOf(*TensorType::get()) && field == "tolist") {
    // 如果字段是 `prim::tolist`，返回一个表示特殊形式的值
    return SpecialFormValue::create(prim::tolist);
  }

  // 当在 Tensor 上直接调用 `__getitem__()` 时，需要特殊处理，
  // 因为所需的方法名 (`__getitem__`) 与 `aten` 运算符名 `aten::index` 不匹配
  if (value_->type()->isSubtypeOf(*TensorType::get()) &&
      field == "__getitem__") {
    // 返回一个表示特殊形式的值，对应于 `aten::index`
    return SpecialFormValue::create(aten::index);
  }

  if (auto generator_type = value_->type()->cast<GeneratorType>()) {
    // 处理对 Generator 的 `manual_seed`、`initial_seed` 和 `seed` 属性的访问
    if (field == "manual_seed" || field == "initial_seed" || field == "seed") {
      // 尝试创建一个内置函数对象，对应于 `aten::field`，并返回该对象
      if (auto builtin = BuiltinFunction::tryCreate(
              Symbol::aten(field), NamedValue(loc, "self", value_))) {
        return builtin;
      }
    }
  }

  // 如果以上条件都不满足，则报告错误
  ErrorReport report(loc);
  report << "'" << value_->type()->repr_str()
         << "' object has no attribute or method '" << field << "'.";
  if (auto classType = value_->type()->cast<ClassType>()) {
    // 如果是未解析的类属性，则提供相应的建议
    if (classType->isUnresolvedClassAttribute(field)) {
      report
          << " '" << field
          << "' is defined as a class attribute which currently is not"
             " supported. Consider converting this to an instance attribute.";
    } else {
      report << " Did you forget to initialize an attribute in __init__()?";
    }
  }
  // 抛出错误报告异常
  throw report;
}

std::vector<std::shared_ptr<SugaredValue>> SimpleValue::asTuple(
    const SourceRange& loc,                           // 函数参数：源代码范围
    GraphFunction& m,                                 // 函数参数：图函数对象引用
    const std::optional<size_t>& size_hint) {         // 函数参数：可选的大小提示

  static const auto make_simple_value =
      [](Value* v) -> std::shared_ptr<SugaredValue> { // 静态 lambda 函数：将 Value 指针封装成 SimpleValue 的共享指针
    return std::make_shared<SimpleValue>(v);
  };

  if (value_->type()->kind() == TypeKind::TupleType) { // 如果 value_ 的类型是元组类型
    auto outputs = createTupleUnpack(value_);          // 创建元组展开操作
    return fmap(outputs, make_simple_value);           // 将输出映射为 SimpleValue 的共享指针并返回
  } else if (value_->type()->kind() == TypeKind::ListType) { // 如果 value_ 的类型是列表类型
    if (!size_hint) {
      throw ErrorReport(loc)                          // 抛出错误：无法在此上下文中静态推断列表的预期大小
          << "cannot statically infer the expected size of a "
          << "list in this context";
    }
    auto graph = value_->owningGraph();
    Node* unpack =                                   // 创建列表展开节点
        graph->insertNode(graph->createListUnpack(value_, *size_hint));
    return fmap(unpack->outputs(), make_simple_value); // 将节点输出映射为 SimpleValue 的共享指针并返回
  } else if (value_->type()->kind() == TypeKind::AnyTupleType) { // 如果 value_ 的类型是任意元组类型
    throw ErrorReport(loc)                           // 抛出错误：提供的元组未完全定义或未完全精炼其元素类型，请提供类似 Tuple[int, int] 类型的值
        << "Provided tuple is not fully defined/refined including its element types, please provide a value of type like Tuple[int, int]";
  }
  throw ErrorReport(loc) << value_->type()->repr_str()  // 抛出错误：value_ 的类型不能作为元组使用
                         << " cannot be used as a tuple";
}

static bool isRecursive(const TypePtr& classType, const TypePtr& attrType) {
  if (attrType->isSubtypeOf(*classType)) {            // 如果 attrType 是 classType 的子类型
    return true;                                     // 返回 true，表示存在递归
  }

  // 递归检查包含的类型。这是因为用户可能会创建 A -> B -> A 的递归结构。
  for (const auto& type : attrType->containedTypes()) { // 遍历 attrType 包含的类型
    if (isRecursive(classType, type)) {               // 递归调用 isRecursive 函数检查类型
      return true;                                   // 如果存在递归，则返回 true
    }
  }
  return false;                                      // 如果没有找到递归，则返回 false
}

void SimpleValue::setAttr(
    const SourceRange& loc,                          // 函数参数：源代码范围
    GraphFunction& m,                                // 函数参数：图函数对象引用
    const std::string& field,                        // 函数参数：字段名称
    Value* newValue) {                               // 函数参数：新值

  const auto classType = value_->type()->cast<ClassType>();  // 获取 value_ 的类型，并尝试将其转换为 ClassType
  if (!classType) {
    throw ErrorReport(loc) << "Tried to set an attribute: " << field  // 抛出错误：试图在非类上设置属性
                           << " on a non-class: " << value_->type()->repr_str();
  }
  auto expectedType = classType->findAttribute(field);   // 查找字段在类中的预期类型
  if (!expectedType) {
    // 如果我们仍在编译此类的 __init__ 方法，则设置一个未知属性将其添加到类的定义中。

    // 如果我们正在初始化：
    const auto isInitializing =
        // 1. 当前正在插入的方法是一个 init 方法
        // TODO 可以是一个合格的名称检查
        m.name() == "__init__" &&
        // 2. self 参数与该值的类型匹配（即我们正在该类的 init 方法中，而不是其他类的方法）
        !m.graph()->inputs().empty() &&
        m.graph()->inputs().at(0)->type() == classType;
    // 如果正在初始化阶段
    if (isInitializing) {
      // 检查是否存在递归类型
      if (isRecursive(classType, newValue->type())) {
        // 抛出错误，不能将属性赋值给包含类类型的对象
        throw ErrorReport(loc)
            << "Assignment to attribute '" << field
            << "' cannot be of a type that contains class "
            << "'" << classType->repr_str() << "'.\n"
            << "Classes that recursively contain instances of themselves"
            << " are not yet supported";
      }

      // 向类类型中添加属性，并设置期望的类型为新值的类型
      classType->addAttribute(field, newValue->type());
      expectedType = newValue->type();

      // 获取当前插入点和顶级代码块
      const auto insertPoint = m.graph()->insertPoint();
      const auto topLevelBlock = m.graph()->block();
      // 如果插入点不在顶级代码块中，则抛出错误
      if (insertPoint->owningBlock() != topLevelBlock) {
        throw ErrorReport(loc)
            << "First assignment cannot be in a control-flow block. "
            << "Initialize the field at the top level first";
      }
    } else {
      // 如果不是初始化阶段，检查是否为 setter 属性
      auto prop = classType->getProperty(field);
      if (prop && prop->setter) {
        // 如果存在 setter，则调用 setter 方法设置属性值并返回
        MethodValue(value_, prop->setter->name())
            .call(loc, m, {newValue}, {}, /*n_binders=*/1);
        return;
      }

      if (prop && !prop->setter) {
        // 如果存在属性但没有 setter，抛出只读属性设置错误
        throw ErrorReport(loc) << "Tried to set read-only attribute: " << field;
      }

      // 如果属性不存在，则抛出错误提示
      throw ErrorReport(loc)
          << "Tried to set nonexistent attribute: " << field
          << ". Did you forget to initialize it in __init__()?";
    }
  }

  // 确保期望类型非空
  AT_ASSERT(expectedType);

  // 检查类型正确性
  const auto newType = newValue->type();
  // 如果新值类型不是期望类型的子类型，则抛出类型错误
  if (!newType->isSubtypeOf(*expectedType)) {
    throw ErrorReport(loc) << "Wrong type for attribute assignment. Expected "
                           << expectedType->repr_str() << " but got "
                           << newType->repr_str();
  }

  // 获取当前图，并在图中插入设置属性的节点
  auto& g = *m.graph();
  g.insertNode(g.createSetAttr(value_, field, newValue));
}

// 实现 SimpleValue 类中的 call 方法，用于调用对象的函数
std::shared_ptr<SugaredValue> SimpleValue::call(
    const SourceRange& loc,  // 函数调用位置信息
    GraphFunction& m,         // 图函数对象的引用
    at::ArrayRef<NamedValue> args,  // 函数调用的位置参数
    at::ArrayRef<NamedValue> kwargs,  // 函数调用的关键字参数
    size_t n_binders) {       // 绑定器的数量

  // 允许调用“假”的闭包，目前主要用于 fork 序列化，以后可能扩展
  Node* self = getValue()->node();  // 获取对象的节点
  if (self->kind() == prim::TupleConstruct && self->inputs().size() == 2 &&
      self->inputs().at(0)->node()->kind() == prim::Closure) {
    std::shared_ptr<Graph> graph =
        self->inputs().at(0)->node()->g(attr::Subgraph);  // 获取闭包的子图
    Value* context = self->inputs().at(1);  // 获取上下文值
    AT_ASSERT(context->node()->kind() == prim::TupleConstruct);  // 断言上下文为元组构造

    // 在 fork 块中发出节点，但不简化跨块的元组构造
    // 为确保在 fork 块中清理元组构造，创建元组构造的另一个副本
    Value* close_context =
        m.graph()
            ->insertNode(m.graph()->createTuple(context->node()->inputs()))
            ->output();
    // TODO 这需要放入 m 的编译单元中
    auto cu = std::make_shared<CompilationUnit>();  // 创建编译单元
    auto fn = cu->create_function(QualifiedName("anon"), graph);  // 创建函数
    auto ret = StrongFunctionPtr(std::move(cu), fn);  // 创建强函数指针

    std::vector<NamedValue> ctx_inputs = {close_context};
    ctx_inputs.insert(ctx_inputs.end(), args.begin(), args.end());
    return FunctionValue(ret).call(loc, m, ctx_inputs, kwargs, n_binders);  // 调用函数值
  }

  if (auto class_type = getValue()->type()->cast<ClassType>()) {
    return attr(loc, m, "__call__")->call(loc, m, args, kwargs, n_binders);  // 调用 __call__ 方法
  }

  return SugaredValue::call(loc, m, args, kwargs, n_binders);  // 调用基类的 call 方法
}

// 实现 SimpleValue 类中的 len 方法，返回对象的长度
Value* SimpleValue::len(const SourceRange& loc, GraphFunction& m) {
  // List, Tuple, Tensor, fill in missing information desugaring
  Value* val = getValue();  // 获取对象的值
  TypePtr val_type = val->type();  // 获取对象的类型
  Graph& g = *m.graph();  // 获取图对象的引用
  if (val_type->cast<ListType>() || val_type->cast<StringType>() ||
      val_type->isSubtypeOf(*TensorType::get())) {
    return g.insert(aten::len, {val}, {}, loc);  // 插入长度操作节点到图中
  } else {
    throw ErrorReport(loc) << "'" << val_type->repr_str() << "'"
                           << " object is not iterable";  // 抛出类型不可迭代的错误
  }
}

// 实现 SimpleValue 类中的 getitem 方法，返回对象的指定项
SugaredValuePtr SimpleValue::getitem(
    const SourceRange& loc,  // 获取项位置信息
    GraphFunction& m,         // 图函数对象的引用
    Value* idx,               // 索引值
    TypePtr type_hint) {      // 类型提示

  Value* val = getValue();    // 获取对象的值
  TypePtr val_type = val->type();  // 获取对象的类型
  Graph& g = *m.graph();      // 获取图对象的引用

  // 如果是 List/String/Dict 类型，则插入常规的 __getitem__ 操作
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (val_type->cast<ListType>() || val_type->cast<StringType>()) {
    return std::make_shared<SimpleValue>(
        g.insert(aten::__getitem__, {val, idx}, {}, loc));  // 插入 __getitem__ 操作节点到图中
  } else if (auto dict_type = val_type->cast<DictType>()) {
    return std::make_shared<SimpleValue>(
        g.insert(aten::__getitem__, {val, idx}, {}, loc));  // 插入 __getitem__ 操作节点到图中
  } else if (val_type->isSubtypeOf(*TensorType::get())) {
    // 省略部分代码，未完成的逻辑
    // 返回一个包含 SimpleValue 共享指针的对象，其值为 g.insert(aten::select, {val, 0, idx}, {}, loc) 的结果
    return std::make_shared<SimpleValue>(
        g.insert(aten::select, {val, 0, idx}, {}, loc));
  } else if (auto class_type = val_type->cast<ClassType>()) {
    // 检查是否可以进行索引操作，通过类型提示进行启用
    // 在 IR 生成期间已经检查过 ModuleDict，以确保其内容实现了由 type_hint 引用的模块接口
    if (class_type->is_module() && type_hint) {
      // 插入 prim::ModuleContainerIndex 操作，并设置结果类型为 type_hint
      auto res = g.insert(prim::ModuleContainerIndex, {val, idx}, {}, loc);
      res->setType(type_hint);
      // 返回一个包含 SimpleValue 共享指针的对象，其值为 res
      return std::make_shared<SimpleValue>(res);
    }

    // 委托给类的 __getitem__ 属性
    return attr(loc, m, "__getitem__")->call(loc, m, {idx}, {}, 1);
  } else {
    // 如果 val_type 不是 ClassType，则抛出错误报告
    throw ErrorReport(loc) << "'" << val_type->repr_str() << "'"
                           << " object is not subscriptable";
  }
}

// 在 SimpleValue 类中定义 iter 方法，返回一个 SugaredValuePtr，表示该对象的迭代器
SugaredValuePtr SimpleValue::iter(const SourceRange& loc, GraphFunction& m) {
  // 获取当前 SimpleValue 对象持有的值
  auto value = getValue();
  // 获取该值的类型
  auto type = value->type();

  // 对于内置的可迭代类型，返回一个新的 SimpleValue 对象
  // 这些类型包括 ListType、StringType 和 TensorType
  if (type->cast<ListType>() || type->cast<StringType>() ||
      type->cast<TensorType>()) {
    return std::make_shared<SimpleValue>(value);
  }

  // 对于字典类型，返回一个新的 SimpleValue 对象，表示其键的迭代器
  if (type->cast<DictType>()) {
    return std::make_shared<SimpleValue>(
        m.graph()->insert(aten::keys, {value}, {}, loc));
  }

  // 对于元组类型，将元组解包为其各个成员，并返回一个 SugaredTupleValue 对象
  if (auto tup = type->cast<TupleType>()) {
    auto tup_values = createTupleUnpack(value);
    std::vector<SugaredValuePtr> tup_sugared;
    for (Value* v : tup_values) {
      tup_sugared.push_back(std::make_shared<SimpleValue>(v));
    }
    return std::make_shared<SugaredTupleValue>(tup_sugared);
  } else {
    // 若类型不属于可迭代类型，则抛出错误报告
    throw ErrorReport(loc) << "'" << type->repr_str() << "'"
                           << " object is not iterable";
  }
}

// 构造函数 RangeValue 的实现，用于处理 range 类型的值
RangeValue::RangeValue(
    const SourceRange& loc,
    GraphFunction& m,
    std::vector<Value*> inputs,
    std::optional<int64_t> static_len) {
  // 遍历输入的值，检查它们是否都是整数类型
  for (const auto i : c10::irange(inputs.size())) {
    auto typ = inputs[i]->type();
    if (!typ->cast<IntType>()) {
      throw ErrorReport(loc)
          << "all inputs of range must be ints, found " << typ->repr_str()
          << " in argument " << std::to_string(i);
    }
  }

  // 获取当前 Graph 对象
  Graph& g = *m.graph();

  // 处理不同数量的输入参数
  if (inputs.empty()) {
    throw ErrorReport(loc) << "range expected at least 1 arguments, got 0";
  } else if (inputs.size() == 1) {
    // 若只有一个参数，则表示 range() 函数只包含终止值
    end_ = inputs[0];
    start_ = g.insertConstant(0, loc);
    step_ = g.insertConstant(1, loc);
    // 设置标志表明只有终止值
    has_only_end_ = true;
  } else if (inputs.size() <= 3) {
    // 若参数数量在 2 到 3 之间，则处理起始值、终止值和步长
    start_ = inputs[0];
    end_ = inputs[1];
    if (inputs.size() == 3) {
      step_ = inputs[2];
    } else {
      step_ = g.insertConstant(1, loc);
    }
    // 设置标志表明不只有终止值
    has_only_end_ = false;
  } else {
    // 若参数数量超过 3，则抛出错误报告
    throw ErrorReport(loc) << "range expected at most 3 arguments, got "
                           << inputs.size();
  }

  // 设置静态长度（如果提供了）
  static_len_ = static_len;
}

// 在 RangeValue 类中定义 iter 方法，返回对象自身的 shared_ptr，表示其迭代器
SugaredValuePtr RangeValue::iter(const SourceRange& loc, GraphFunction& m) {
  return shared_from_this();
};

// 在 RangeValue 类中定义 len 方法，返回 range 对象的长度
Value* RangeValue::len(const SourceRange& loc, GraphFunction& m) {
  // 如果有静态长度，则返回静态长度的常量值
  if (static_len_) {
    return insertConstant(*m.graph(), *static_len_, loc);
  }
  // 如果只有终止值，则直接返回终止值
  if (has_only_end_) {
    return end_;
  } else {
    // 否则，插入一个计算 range 长度的操作节点，并返回其结果值
    Graph& g = *m.graph();
    return g.insert(aten::__range_length, {start_, end_, step_}, {}, loc);
  }
}

// 在 RangeValue 类中定义 getitem 方法，用于获取 range 对象的指定元素
SugaredValuePtr RangeValue::getitem(
    const SourceRange& loc,
    GraphFunction& m,
    Value* idx,
    TypePtr type_hint) {
  // 如果只有终止值，则直接返回 idx 作为 SimpleValue 对象
  if (has_only_end_) {
    return std::make_shared<SimpleValue>(idx);
  } else {
    // 否则，插入一个计算索引位置的操作节点，并返回其结果作为 SimpleValue 对象
    auto& g = *m.graph();
    return std::make_shared<SimpleValue>(
        g.insert(aten::__derive_index, {idx, start_, step_}, {}, loc));
  }
}
// 获取当前 IterableTree 实例中所有子节点的基本可迭代对象
std::vector<SugaredValuePtr> IterableTree::get_base_iterables() {
  // 初始化一个空的基本可迭代对象列表
  std::vector<SugaredValuePtr> base_iters{};

  // 遍历当前节点的子节点列表
  for (SugaredValuePtr& sv : children_) {
    // 检查子节点是否是 IterableTree 的实例
    if (auto iv = std::dynamic_pointer_cast<IterableTree>(sv)) {
      // 如果是 IterableTree 的实例，递归获取其基本可迭代对象
      std::vector<SugaredValuePtr> child_iters = iv->get_base_iterables();
      // 将子节点的可迭代对象合并到当前节点的基本可迭代对象列表中
      base_iters.insert(
          base_iters.end(),
          std::make_move_iterator(child_iters.begin()),
          std::make_move_iterator(child_iters.end()));

    } else {
      // 如果子节点不是 IterableTree 的实例，直接将其加入基本可迭代对象列表中
      base_iters.emplace_back(sv);
    }
  }

  // 返回当前节点的基本可迭代对象列表
  return base_iters;
}

// 计算当前 IterableTree 实例的长度
Value* IterableTree::len(const SourceRange& loc, GraphFunction& m) {
  // 断言当前实例不是展开长度的情况
  TORCH_INTERNAL_ASSERT(!unroll_length_);

  // 获取当前函数的图对象
  Graph& g = *m.graph();

  // 获取当前实例的基本可迭代对象列表
  std::vector<SugaredValuePtr> base_iters = get_base_iterables();
  
  // 用于存储每个可迭代对象的长度
  std::vector<Value*> lengths;
  lengths.reserve(base_iters.size());

  // 遍历每个基本可迭代对象，计算其长度并存储在 lengths 列表中
  for (const SugaredValuePtr& base_iter : base_iters) {
    lengths.emplace_back(base_iter->len(loc, m));
  }

  // 创建一个新的列表节点，将 lengths 中的值作为元素添加到列表中
  Node* list_node = g.insertNode(g.createList(IntType::get(), lengths));

  // 返回一个新的节点，该节点调用 prim::min 操作，参数为列表节点的输出值，位置信息为 loc
  return g.insert(prim::min, {list_node->output()}, {}, loc);
}

// 获取当前 IterableTree 实例的子项
SugaredValuePtr IterableTree::getitem(
    const SourceRange& loc,
    GraphFunction& m,
    Value* idx,
    TypePtr type_hint) {
  // 初始化一个空的子项列表
  std::vector<SugaredValuePtr> child_items;
  child_items.reserve(children_.size());

  // 遍历当前节点的所有子节点，获取每个子节点的子项，并添加到 child_items 列表中
  for (const SugaredValuePtr& child : children_) {
    child_items.emplace_back(child->getitem(loc, m, idx));
  }

  // 返回一个新的 SugaredTupleValue，该值包含了所有子项
  return std::make_shared<SugaredTupleValue>(child_items);
}

// 向当前 IterableTree 实例添加子节点
void IterableTree::addChild(
    const SourceRange& range,
    GraphFunction& m,
    const SugaredValuePtr& iter_value) {
  // 获取 iter_value 的静态长度
  std::optional<int64_t> child_len = iter_value->staticLen();

  // 如果当前节点没有子节点
  if (children_.empty()) {
    // 将当前节点的展开长度设置为 iter_value 的长度
    unroll_length_ = child_len;
  } else {
    // 如果当前节点已经有子节点
    if ((unroll_length_ && !child_len) || (child_len && !unroll_length_)) {
      // 如果当前节点的展开长度和 iter_value 的长度不一致，抛出错误
      throw ErrorReport(range)
          << "Can not iterate over a module list or tuple with a value "
             "that does not have a statically determinable length\n";
    }
    // 如果当前节点的展开长度和 iter_value 的长度都存在
    if (unroll_length_ && child_len) {
      // 将当前节点的展开长度设置为当前节点展开长度和 iter_value 长度的最小值
      unroll_length_ = std::min(*child_len, *unroll_length_);
    } else {
      // 如果展开长度不确定，则设置为无效值
      unroll_length_ = c10::nullopt;
    }
  }

  // 向当前节点的子节点列表中添加 iter_value
  children_.push_back(iter_value);
}

// 调用魔术方法的函数
std::shared_ptr<SugaredValue> MagicMethod::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  // 如果参数不为空
  if (!args.empty()) {
    // 获取参数的 self 值
    Value* self = args[0].value(*m.graph());

    // 如果 self 的类型可以转换为 ClassType
    if (auto class_ptr = self->type()->cast<ClassType>()) {
      // 创建一个 SimpleValue 对象，并调用其属性方法 desugared_name_
      return SimpleValue(self)
          .attr(loc, m, desugared_name_)
          ->call(loc, m, args.slice(1), kwargs, n_binders);
    }
  }
  // 调用 TORCH_INTERNAL_ASSERT 宏，确保 base_value_ 不为空
  TORCH_INTERNAL_ASSERT(base_value_);
  // 调用 base_value_ 对象的 call 方法，传入 loc, m, args, kwargs, n_binders 参数，并返回结果
  return base_value_->call(loc, m, args, kwargs, n_binders);
}

std::shared_ptr<SugaredValue> ClassValue::call(
    const SourceRange& loc,
    GraphFunction& m,
    // note: names for args will be 'argument 0', 'argument 1', etc..
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  AT_ASSERT(n_binders <= 1);

  // Generate a new object of the right type, then call `__init__` on it
  auto& g = *m.graph();
  // 创建一个新的对象，并插入图中，类型为当前类的类型
  auto self = g.insertNode(g.createObject(type_))->output();
  // 设置新对象的源代码范围
  self->node()->setSourceRange(loc);
  // 如果当前类没有定义 `__init__` 方法，则抛出错误
  if (!type_->findMethod("__init__")) {
    throw ErrorReport(loc) << "Class " << type_->name()->name()
                           << " does not have an __init__ function defined";
  }

  // 调用 `__init__` 方法初始化新对象
  MethodValue(self, "__init__").call(loc, m, args, kwargs, n_binders);

  // 返回一个包装了新对象的 SimpleValue 指针
  return std::make_shared<SimpleValue>(self);
}

std::shared_ptr<SugaredValue> ClassValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // 允许 import_source.cpp 解析对子模块的钩子调用，这是一个边界情况，
  // 因为通常不允许模块调用子模块的函数
  if (Function* hook = type_->findHook(field)) {
    // 如果找到了对应字段的钩子函数，返回一个 FunctionValue 指针
    return std::make_shared<FunctionValue>(hook);
  }

  // 如果字段不是 "__new__"，则抛出错误，表示在当前类上查找了未知属性
  if (field != "__new__") {
    throw ErrorReport(loc) << "Tried to lookup unknown attribute on class "
                           << type_->annotation_str();
  }
  // 返回一个表示创建对象特殊形式的 SpecialFormValue 指针
  return SpecialFormValue::create(prim::CreateObject);
}

std::shared_ptr<SugaredValue> NamedTupleConstructor::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  auto& g = *m.graph();

  // 获取当前命名元组的模式
  auto schema = type_->schema();
  TORCH_INTERNAL_ASSERT(schema);
  // 获取当前类型的限定名称
  auto qualname = type_->name();
  // 使用传入的参数和模式匹配，返回匹配后的结果
  auto matched_schema = matchSchema(*schema, loc, g, args, kwargs);

  // 创建一个元组节点，并将其插入图中，设置源代码范围，并将类型设置为当前类型
  auto self =
      g.insertNode(
           g.createTuple(matched_schema.inputs, type_)->setSourceRange(loc))
          ->output();
  self->setType(type_);

  // 返回一个包装了新元组的 SimpleValue 指针
  return std::make_shared<SimpleValue>(self);
}

std::shared_ptr<BuiltinFunction> BuiltinFunction::tryCreate(
    Symbol symbol,
    std::optional<NamedValue> self) {
  // 遍历所有与给定符号相关的运算符
  for (const std::shared_ptr<Operator>& op : getAllOperatorsFor(symbol)) {
    // 如果没有 self 参数，创建一个没有 self 参数的内置函数对象
    if (!self) {
      return std::make_shared<BuiltinFunction>(symbol, nullptr);
    }
    // 如果运算符的模式中有名为 "self" 的参数
    if (auto index = op->schema().argumentIndexWithName("self")) {
      std::unordered_map<std::string, TypePtr> type_env;
      // 获取形式参数的类型
      TypePtr formal_type = op->schema().arguments().at(*index).type();
      // 尝试匹配类型变量
      const MatchTypeReturn matched =
          matchTypeVariables(formal_type, self->type(), type_env);
      // 如果匹配不成功，继续下一个运算符
      if (!matched.success()) {
        continue;
      }
      // 尝试评估类型变量得到具体类型
      const auto concrete_type = tryEvalTypeVariables(formal_type, type_env);
      // 如果没有具体类型或者 self 的类型不是具体类型的子类型，则继续下一个运算符
      if (!concrete_type || !self->type()->isSubtypeOf(*concrete_type)) {
        continue;
      }
      // 返回一个具有 self 参数的内置函数对象
      return std::make_shared<BuiltinFunction>(symbol, self);
    }
  }
  // 如果没有找到匹配的内置函数，返回空指针
  return nullptr;
}
# 返回指定字段的属性值的包装类
std::shared_ptr<SugaredValue> SugaredEnumClass::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  # 获取枚举类型的所有名称和对应的值
  const auto& names_values = enum_type_->enumNamesValues();
  # 在名称-值对列表中查找与指定字段匹配的条目
  auto it = std::find_if(
      names_values.begin(),
      names_values.end(),
      [&field](const at::EnumNameValue& nv) { return nv.first == field; });
  # 如果未找到指定字段的条目，则抛出错误报告
  if (it == names_values.end()) {
    throw ErrorReport(loc) << enum_type_->repr_str() << "'"
                           << " has no attribute '" << field << "'";
  }
  # 创建枚举值的包装类对象
  auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
      enum_type_, it->first, it->second);
  # 将包装类对象转换为简单值，并插入计算图中作为常量
  return std::make_shared<SimpleValue>(
      m.graph()->insertConstant(IValue(enum_holder), loc));
}

# 返回枚举类型所有值的迭代器的包装类
SugaredValuePtr SugaredEnumClass::iter(
    const SourceRange& loc,
    GraphFunction& m) {
  # 获取枚举类型的所有名称和对应的值
  const auto& names_values = enum_type_->enumNamesValues();
  # 创建一个通用列表，用于存储枚举值的包装类
  auto enum_value_ivalues = c10::impl::GenericList(enum_type_);
  enum_value_ivalues.reserve(names_values.size());
  # 遍历所有枚举名称和对应的值，创建并添加枚举值的包装类对象到通用列表中
  for (const auto& name_value : names_values) {
    auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
        enum_type_, name_value.first, name_value.second);
    enum_value_ivalues.emplace_back(enum_holder);
  }

  # 将枚举值的通用列表转换为简单值，并插入计算图中作为常量
  auto enum_values_list_constant = std::make_shared<SimpleValue>(
      m.graph()->insertConstant(enum_value_ivalues, loc));
  return enum_values_list_constant;
}

} // namespace torch::jit
```