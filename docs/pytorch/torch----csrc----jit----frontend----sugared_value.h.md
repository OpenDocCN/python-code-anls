# `.\pytorch\torch\csrc\jit\frontend\sugared_value.h`

```py
// 一旦这个头文件被包含，确保当前头文件内容只被编译一次
#pragma once

// 包含以下库头文件
#include <c10/util/Optional.h>  // 包含 c10 库中的 Optional 类
#include <functional>           // C++ 标准库中的 functional 库
#include <memory>               // C++ 标准库中的内存管理相关功能
#include <string>               // C++ 标准库中的字符串处理功能
#include <utility>              // C++ 标准库中的实用工具功能

// 包含以下 ATen 库和 Torch 库中的头文件
#include <ATen/core/symbol.h>                // ATen 库中的 symbol 头文件
#include <caffe2/serialize/versions.h>       // Caffe2 序列化版本控制相关头文件
#include <torch/csrc/jit/api/module.h>       // Torch 中用于模块操作的头文件
#include <torch/csrc/jit/frontend/error_report.h>  // Torch 中用于错误报告的头文件
#include <torch/csrc/jit/frontend/schema_matching.h>  // Torch 中用于模式匹配的头文件
#include <torch/csrc/jit/frontend/versioned_symbols.h>  // Torch 中用于版本符号管理的头文件
#include <torch/csrc/jit/ir/ir.h>            // Torch 中的 IR 抽象表示头文件

// 命名空间声明，所有的内容都在 torch::jit 命名空间下
namespace torch {
namespace jit {

// 使用 std::shared_ptr 来定义 SugaredValuePtr 类型
using SugaredValuePtr = std::shared_ptr<SugaredValue>;

// SugaredValue 用于表示 AST 中的一些特殊节点，如 self、self.b 或 python_fn，
// 这些节点在图表示中不是一级值，而是根据其在 AST 中的使用进行展开。
struct TORCH_API SugaredValue
    : public std::enable_shared_from_this<SugaredValue> {
  // 返回当前节点的类型，用于错误报告（例如 Module、python function 等）
  virtual std::string kind() const = 0;

  // 将当前节点视为值使用，例如 `this + 4`
  virtual Value* asValue(const SourceRange& loc, GraphFunction& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as a value";
  }

  // 获取当前节点的属性，例如 `this.field`
  virtual std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) {
    throw ErrorReport(loc) << "attribute lookup is not defined on " << kind();
  }

  // 检查当前节点是否具有某个属性
  virtual bool hasAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) {
    throw ErrorReport(loc) << "attribute lookup is not defined on " << kind();
  }

  // 给当前节点设置属性，例如 `this.field = newValue`
  virtual void setAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field,
      Value* newValue) {
    throw ErrorReport(loc) << "attribute assignment is not defined on "
                           << kind();
  }

  // 将当前节点视为值向量使用，例如作为方法调用返回的值元组
  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      GraphFunction& m,
      const std::optional<size_t>& size_hint = {}) {
    throw ErrorReport(loc) << kind() << " cannot be used as a tuple";
  }

  // 用于特定重构的 API，将当前节点视为元组值使用
  virtual SugaredValuePtr asTupleValue(
      const SourceRange& loc,
      GraphFunction& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as a tuplevalue";
  }

  // 将当前节点视为类型使用
  virtual std::vector<std::shared_ptr<SugaredValue>> asType(
      const SourceRange& loc,
      Method& m) {
  // 抛出错误报告，指示给定位置 loc 处的类型错误
  throw ErrorReport(loc) << kind() << " cannot be used as a type";
}

// 将其视为函数调用，例如 `outputs = this(inputs)`
virtual std::shared_ptr<SugaredValue> call(
    const SourceRange& loc,
    GraphFunction& m,
    // 注意：args 的名称将是 'argument 0'、'argument 1' 等
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  // n_binders 始终设置为表达式在语法上绑定的变量数：
  //     a = foo() # 1 个绑定器（注意，在此情况下，单个绑定器可能是一个元组）
  //     a, *b = foo() # 1 个绑定器
  //     a, b = foo() # 2 个绑定器
  //     foo() # 0 个绑定器
  //
  // 在子表达式（例如 foo(bar()) 中的 bar()）中，n_binders 始终设置为 1。
  // n_binders 被用作子表达式的提示，以确定静态上不明确的情况下它们应该返回多少个值。
  // 特别地，它当前用于决定调用 Python 函数时应返回多少个张量。这只是一个提示，函数不必检查 n_binders 是否与它们返回的事物数量匹配，分配逻辑将在任何情况下都会执行该检查。
  
  throw ErrorReport(loc) << "cannot call a " << kind();
}

// 当将 SugaredValue 转换为其迭代器时调用此函数。
// 例如，当遍历字典时，我们遍历其键。
virtual std::shared_ptr<SugaredValue> iter(
    const SourceRange& loc,
    GraphFunction& m) {
  throw ErrorReport(loc) << kind() << " cannot be used as an iterable";
}

// 如果正在迭代 SugaredValue 并且它从此函数返回一个值，则发出对变量的展开循环。
// 这允许我们支持异构类型的容器，例如模块容器和元组。
virtual std::optional<int64_t> staticLen() {
  return c10::nullopt;
}

// 当迭代此 SugaredValue 时，是否应将 for 循环展开为展开的循环。
bool shouldEmitUnrolled() {
  return staticLen() != c10::nullopt;
}

// 返回此对象的长度，如果不能确定静态长度，则无法进行迭代。
// 如果具有静态确定的长度，则必须返回一个常量值 *。
virtual Value* len(const SourceRange& loc, GraphFunction& m) {
  throw ErrorReport(loc) << "'" << kind() << "' object is not iterable";
}

// 用于可迭代值的第 idx 个元素的表达式。
virtual std::shared_ptr<SugaredValue> getitem(
    const SourceRange& loc,
    GraphFunction& m,
    Value* idx,
    TypePtr type_hint = nullptr) {
  throw ErrorReport(loc) << "'" << kind() << "' object is not subscriptable";
}

// 默认析构函数
virtual ~SugaredValue() = default;
};

// 环境中的大部分内容只是简单的值类型，而不是特殊的 Python 语法糖类型
// SimpleValue 结构体，继承自 SugaredValue
struct TORCH_API SimpleValue : public SugaredValue {
  // 构造函数，接受一个值类型指针作为参数
  SimpleValue(Value* value) : value_(value) {}
  // 返回对象类型的字符串表示
  std::string kind() const override {
    std::stringstream ss;
    // 构建类型信息字符串
    ss << "value of type '" << value_->type()->annotation_str() << "'";
    return ss.str();
  }
  // 返回原始值对象
  Value* asValue(const SourceRange& range, GraphFunction& m) override {
    return value_;
  }
  // 将对象解释为元组
  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      GraphFunction& m,
      const std::optional<size_t>& size_hint = {}) override;
  // 获取对象的属性
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // 判断对象是否具有指定属性
  bool hasAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // 设置对象的属性
  void setAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field,
      Value* newValue) override;

  // 调用对象
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& m,
      // 注意：参数的名称将会是 'argument 0', 'argument 1', 等等
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  // 迭代对象
  std::shared_ptr<SugaredValue> iter(const SourceRange& loc, GraphFunction& m)
      override;

  // 获取对象的长度
  Value* len(const SourceRange& loc, GraphFunction& m) override;
  // 获取对象的指定项
  SugaredValuePtr getitem(
      const SourceRange& loc,
      GraphFunction& m,
      Value* idx,
      TypePtr type_hint = nullptr) override;

 private:
  // 内部存储的值对象指针
  Value* value_;
};

// 内置函数类型，继承自 SugaredValue
struct TORCH_API BuiltinFunction : public SugaredValue {
  // 构造函数，接受函数符号和可选的 self 参数
  BuiltinFunction(Symbol symbol, std::optional<NamedValue> self)
      : symbol(symbol), self(std::move(self)) {}

  // 函数符号 (例如 `aten::relu`)
  Symbol symbol;

  // 如果是方法，则这是 self 参数
  std::optional<NamedValue> self;
  // 返回对象类型的字符串表示
  std::string kind() const override {
    return "builtin";
  }
  // 调用函数
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& m,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  // 尝试创建内置函数，如果不存在或者 self 参数不匹配，则返回 nullptr
  static std::shared_ptr<BuiltinFunction> tryCreate(
      Symbol symbol,
      std::optional<NamedValue> self);
};

// 元组值类型，继承自 SugaredValue
struct TORCH_API SugaredTupleValue : public SugaredValue {
  // 构造函数，接受一个共享指针的向量作为参数
  explicit SugaredTupleValue(std::vector<std::shared_ptr<SugaredValue>> tup)
      : tup_(std::move(tup)){};

  // 将对象解释为元组
  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      GraphFunction& m,
      const std::optional<size_t>& size_hint = {}) override {
  // 返回 tup_ 成员变量
  return tup_;
};

Value* asValue(const SourceRange& loc, GraphFunction& m) override {
  // 创建一个空的值向量，预留足够的空间以容纳 tup_ 的大小
  std::vector<Value*> vec;
  vec.reserve(tup_.size());
  // 遍历 tup_ 中的每个元素，并将其转换为值对象，添加到 vec 中
  for (const auto& sv : tup_) {
    vec.push_back(sv->asValue(loc, m));
  }
  // 获取 Graph 对象的引用
  Graph& g = *m.graph();
  // 创建一个元组节点，将 vec 中的值作为元组的元素，并返回该节点的输出
  return g.insertNode(g.createTuple(vec))->output();
}

std::string kind() const override {
  // 返回字符串 "Tuple"，表示这个对象的类型是元组
  return "Tuple";
}

SugaredValuePtr getitem(
    const SourceRange& loc,
    GraphFunction& m,
    Value* idx,
    TypePtr type_hint = nullptr) override {
  // 检查 idx 是否是整数类型的常量，并获取其整数值
  if (!(idx->type()->cast<IntType>() && toIValue(idx))) {
    // 如果 idx 不是整数常量，则抛出错误报告
    throw ErrorReport(loc)
        << "Expected integer literal for index but got a variable or non-integer. "
        << "ModuleList/Sequential indexing is only supported with integer literals. "
        << "For example, 'i = 4; self.layers[i](x)' will fail because i is not a literal. "
        << "Enumeration is supported, e.g. 'for index, v in enumerate(self): out = v(inp)'";
  }
  // 将 idx 转换为整数值
  auto index = toIValue(idx)->toInt();
  // 计算调整后的索引，支持负数索引
  int64_t adj_index =
      (index < 0) ? index + static_cast<int64_t>(tup_.size()) : index;
  // 检查调整后的索引是否在有效范围内
  if (!(adj_index >= 0 && adj_index < static_cast<int64_t>(tup_.size()))) {
    // 如果索引超出范围，则抛出错误报告
    throw ErrorReport(loc)
        << "Index " << index << " out of range of length " << tup_.size();
  }
  // 返回 tup_ 中调整后索引位置的元素
  return tup_.at(adj_index);
}

// 当将 SugaredValue 用作其迭代器时调用此函数，例如在字典上进行迭代时迭代其键
std::shared_ptr<SugaredValue> iter(const SourceRange& loc, GraphFunction& m)
    override {
  // 返回当前对象的共享指针，表示它自身可以作为其迭代器
  return shared_from_this();
};

// 因为此对象包含异构类型的 SugaredValues，所以定义 staticLen() 方法以便在迭代时作为展开的循环进行处理
std::optional<int64_t> staticLen() override {
  // 返回 tup_ 的大小作为可选的整数，表示其长度
  return static_cast<int64_t>(tup_.size());
}

// 元组对象的成员变量，包含多个共享指针指向 SugaredValue 对象
std::vector<std::shared_ptr<SugaredValue>> tup_;
};

// 表示一个内置模块，继承自 SugaredValue
struct TORCH_API BuiltinModule : public SugaredValue {
  // 构造函数，接受模块名和可选的版本号作为参数
  BuiltinModule(std::string name, std::optional<int64_t> version = at::nullopt)
      : name(std::move(name)), version(version) {}

  // 返回对象类型的字符串表示，这里是 "builtin module"
  std::string kind() const override {
    return "builtin module";
  }

  // 返回模块的属性，可以是函数或其他模块
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override {
    if (field == "autograd") {
      // 当访问 torch.autograd 时，也视为内置模块，并调度到对应模块下的 aten 操作符
      return std::make_shared<BuiltinModule>("aten", version);
    }

    // 从限定字符串中创建符号对象，例如 "name::field"
    auto sym = Symbol::fromQualString(name + "::" + field);
    // 创建一个内置函数对象，并返回
    return std::make_shared<BuiltinFunction>(sym, c10::nullopt);
  }

 private:
  std::string name;
  // 运算符版本号，用于添加运算符版本控制
  std::optional<int64_t> version;
};

// 表示一个类，类似于 int 或 dict，类的实例表示为 SimpleValues
struct TORCH_API ClassValue : public SugaredValue {
  explicit ClassValue(ClassTypePtr type) : type_(std::move(type)) {}

  // 调用类型的构造函数，例如 n = Foo(constructor_arg)
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& m,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  // 返回类的属性
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // 返回对象类型的字符串表示
  std::string kind() const override {
    return type_->str();
  }

  ClassTypePtr type_;
};

// 表示一个具名元组构造函数，接受元组类型作为参数
struct TORCH_API NamedTupleConstructor : public SugaredValue {
  explicit NamedTupleConstructor(TupleTypePtr type) : type_(std::move(type)) {}

  // 调用具名元组构造函数
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& m,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  // 返回对象类型的字符串表示
  std::string kind() const override {
    return type_->str();
  }

  TupleTypePtr type_;
};

// 表示一个函数值对象
struct FunctionValue : public SugaredValue {
  // 构造函数，接受一个函数指针
  FunctionValue(Function* callee) : callees_({callee}) {}
  // 构造函数，接受一个强引用的函数指针
  FunctionValue(const StrongFunctionPtr& p)
      : callees_({p.function_}), cu_(p.cu_) {}
  // 构造函数，接受一个函数指针的向量
  FunctionValue(const std::vector<StrongFunctionPtr>& callees) {
    for (const StrongFunctionPtr& callee : callees) {
      cu_ = cu_ ? cu_ : callee.cu_;
      TORCH_INTERNAL_ASSERT(callee.cu_ == cu_);
      callees_.push_back(callee.function_);
    }
  }

  // 返回对象类型的字符串表示，这里是 "function"
  std::string kind() const override {
    return "function";
  }

  // 调用函数对象
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& f,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    std::vector<const FunctionSchema*> schemas;
    // ...
  }

 private:
  std::vector<Function*> callees_; // 函数指针的向量
  std::shared_ptr<CompilationUnit> cu_; // 引用的编译单元
};
    for (Function* callee : callees_) {
      // 遍历调用函数列表，对每个函数进行以下操作
      try {
        // 尝试确保调用的函数已定义
        callee->ensure_defined();
      } catch (const RecursiveMethodCallError&) {
        // 捕获递归方法调用错误，抛出错误报告
        throw ErrorReport(loc)
            << " function '" << callee->name() << "' is called recursively. "
            << "Recursive calls are not supported";
      }
      // 将调用函数的模式指针添加到schemas向量中
      schemas.push_back(&callee->getSchema());
    }
    // 使用schemas向量和其他参数匹配函数调用模式，获取匹配结果
    auto match = matchSchemas(schemas, loc, *f.graph(), args, kwargs);
    // 在函数图中插入调用函数的调用，并获取其输出值
    Value* output =
        f.graph()->insertFunctionCall(callees_[match.first], match.second);
    // 设置输出值的源范围
    output->node()->setSourceRange(loc);
    // 返回输出值的简单值的共享指针
    return std::make_shared<SimpleValue>(output);
  }

  // 返回调用函数列表的常量引用
  const std::vector<Function*>& callees() {
    return callees_;
  }

 private:
  // 存储调用函数的指针的向量
  std::vector<Function*> callees_;
  // 持有此对象似乎有些怪异（TODO: 或许需要进一步解释为什么持有这个对象会让人感到奇怪）
  std::shared_ptr<CompilationUnit> cu_;
};

// 表示闭包值的类，继承自SugaredValue
struct TORCH_API ClosureValue : public SugaredValue {
  // 构造函数，接受一个值作为参数并存储在成员变量中
  ClosureValue(Value* value) : value_(value) {
    // 断言值的节点类型是 prim::Closure
    TORCH_INTERNAL_ASSERT(value_->node()->kind() == prim::Closure);
  }
  // 返回字符串 "closure"
  std::string kind() const override {
    return "closure";
  }
  // 将存储的值作为节点值返回
  Value* asValue(const SourceRange& range, GraphFunction& m) override {
    return value_;
  }
  // 存储的值
  Value* value_;
};

// 定义从模块/类/接口中获取的方法在脚本中的行为方式
struct MethodValue : public SugaredValue {
  // 构造函数，接受一个self值和方法名的vector
  MethodValue(Value* self, std::vector<std::string> method_names)
      : self_(self), method_names_(std::move(method_names)) {}
  // 构造函数，接受一个self值和单个方法名
  MethodValue(Value* self, std::string method_name)
      : MethodValue(self, std::vector<std::string>({std::move(method_name)})) {}

  // 返回字符串 "method"
  std::string kind() const override {
    return "method";
  }

  // 调用方法的实现，返回一个SugaredValue的共享指针
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& f,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    // 创建包含self的参数列表
    std::vector<NamedValue> argsWithSelf = {self_};
    argsWithSelf.insert(argsWithSelf.end(), args.begin(), args.end());
    // 存储函数架构的指针
    std::vector<const FunctionSchema*> schemas;
    // 遍历方法名列表
    for (const std::string& method_name : method_names_) {
      // 如果self的类型是ClassType
      if (auto class_type = self_->type()->cast<ClassType>()) {
        // 获取类中对应方法的函数对象
        Function& method = class_type->getMethod(method_name);
        try {
          // 确保方法已定义，捕获递归方法调用错误
          method.ensure_defined();
        } catch (const RecursiveMethodCallError&) {
          throw ErrorReport(loc)
              << " method '" << method.name() << "' is called recursively. "
              << "Recursive calls are not supported";
        }
        // 获取方法的函数架构并添加到schemas中
        schemas.push_back(&method.getSchema());
      } else if (auto interface_type = self_->type()->cast<InterfaceType>()) {
        // 如果self的类型是InterfaceType，则直接获取方法的函数架构
        schemas.push_back(interface_type->getMethod(method_name));
      } else {
        // 如果self的类型既不是ClassType也不是InterfaceType，则断言失败
        TORCH_INTERNAL_ASSERT(
            false, "method constructed that is not a class or interface");
      }
    }
    // 使用matchSchemas函数匹配schemas和输入的参数列表，获取匹配的索引和参数
    auto match = matchSchemas(schemas, loc, *f.graph(), argsWithSelf, kwargs);
    // 插入方法调用的节点到图中，并返回输出值
    Value* output =
        f.graph()->insertMethodCall(method_names_[match.first], match.second);
    output->node()->setSourceRange(loc);
    // 返回一个SimpleValue的共享指针，封装输出值
    return std::make_shared<SimpleValue>(output);
  }

 private:
  // 存储self值的成员变量
  Value* self_;
  // 存储方法名列表的成员变量
  std::vector<std::string> method_names_;
};

// 表示打印操作的类，继承自SugaredValue
struct TORCH_API PrintValue : public SugaredValue {
  // 返回字符串 "print"
  std::string kind() const override {
    return "print";
  }
  // 调用打印操作的实现，返回一个SugaredValue的共享指针
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& m,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;
};

// 表示如 int(x) 这样的表达式
// 这些与调用 prim::Int 或等效操作相同，但当输入是 'type' 的子类型时为无操作
// 定义一个结构体 CastValue，继承自 BuiltinFunction 类
// 该结构体表示一个类型转换函数
struct TORCH_API CastValue : public BuiltinFunction {
  // 构造函数，初始化类型和方法符号
  CastValue(TypePtr type, c10::Symbol method)
      : BuiltinFunction(method, c10::nullopt), type_(std::move(type)) {}

  // 重写 call 方法，执行函数调用
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,                   // 调用位置信息
      GraphFunction& m,                         // 图函数
      at::ArrayRef<NamedValue> args,            // 位置参数
      at::ArrayRef<NamedValue> kwargs,          // 关键字参数
      size_t n_binders) override {              // 绑定器数量
    // 如果只有一个位置参数且无关键字参数
    if (args.size() == 1 && kwargs.empty()) {
      // 创建 len_op 和 gt_op 操作符
      auto len_op = std::make_shared<BuiltinFunction>(aten::len, at::nullopt);
      auto gt_op = std::make_shared<BuiltinFunction>(aten::gt, at::nullopt);
      // 在图中插入常量 0
      auto zero = m.graph()->insertConstant(0);

      // 获取参数的值
      auto v = args[0].value(*m.graph());
      // 如果参数的类型是 type_ 的子类型
      if (v->type()->isSubtypeOf(*type_)) {
        // 返回一个包装了参数值的 SimpleValue 对象
        return std::make_shared<SimpleValue>(v);
      } else if (
          *type_ == *BoolType::get() &&
          (v->type()->isSubtypeOf(*AnyListType::get()) ||
           v->type()->isSubtypeOf(*StringType::get()) ||
           v->type()->cast<DictType>())) {
        // 如果目标类型是布尔型，且参数类型是列表、字符串或字典
        // 调用 len_op 获取参数的长度
        auto len = len_op->call(loc, m, {v}, {}, 1);
        // 调用 gt_op 判断参数长度是否大于零
        return gt_op->call(loc, m, {len->asValue(loc, m), zero}, {}, 1);
      }
    }
    // 如果不符合上述条件，则调用基类 BuiltinFunction 的 call 方法
    return BuiltinFunction::call(loc, m, args, kwargs, n_binders);
  }

 private:
  TypePtr type_;  // 目标类型
};

// 定义一个结构体 TensorCastValue，继承自 SugaredValue 类
// 该结构体表示张量类型转换函数
struct TORCH_API TensorCastValue : public SugaredValue {
  // 构造函数，初始化数据类型和自身参数
  TensorCastValue(at::ScalarType type, NamedValue self)
      : dtype_(type), self_(std::move(self)) {}

  // 返回对象种类的字符串表示
  std::string kind() const override {
    return "Cast";
  }

  // 重写 call 方法，执行函数调用
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,                   // 调用位置信息
      GraphFunction& m,                         // 图函数
      at::ArrayRef<NamedValue> args,            // 位置参数
      at::ArrayRef<NamedValue> kwargs,          // 关键字参数
      size_t n_binders) override {              // 绑定器数量
    TORCH_INTERNAL_ASSERT(args.empty() && kwargs.empty());
    // 在图中插入数据类型常量
    Value* dtype_const = m.graph()->insertConstant(dtype_, loc);
    // 构建关键字参数列表
    std::vector<NamedValue> kwargs_{
        self_, NamedValue(loc, "dtype", dtype_const)};
    // 在图中插入转换操作，返回转换后的值
    Value* casted_val = m.graph()->insert(
        /*opname=*/Symbol::fromQualString("aten::to"),
        /*args=*/args,
        /*kwargs=*/kwargs_,
        /*range=*/loc);
    // 返回一个包装了转换后值的 SimpleValue 对象
    return std::make_shared<SimpleValue>(casted_val);
  }

  at::ScalarType dtype_;  // 数据类型
  NamedValue self_;       // 自身参数
};

// 定义一个结构体 MagicMethod，继承自 SugaredValue 类
// 该结构体表示魔法方法
// 用于调用类类型的方法，如 'len(x)' 和 'x + y'
struct TORCH_API MagicMethod : public SugaredValue {
  // 构造函数，初始化解析后名称和基础值
  MagicMethod(std::string desugared_name, SugaredValuePtr base)
      : base_value_(std::move(base)),
        desugared_name_(std::move(desugared_name)) {}

  // 返回对象种类的字符串表示
  std::string kind() const override {
    return desugared_name_;
  }

  // 重写 call 方法，执行函数调用
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,                   // 调用位置信息
      GraphFunction& m,                         // 图函数
      at::ArrayRef<NamedValue> args,            // 位置参数
      at::ArrayRef<NamedValue> kwargs,          // 关键字参数
      size_t n_binders) override;               // 绑定器数量

 private:
  SugaredValuePtr base_value_;  // 基础值
  std::string desugared_name_;  // 解析后的名称
};

// 定义一个结构体 SpecialFormValue，继承自 SugaredValue 类
// 表示执行非标准评估的特殊形式值，如 isinstance(x, int) 和 fork(fn)
// 通常作为函数应用的表示
// 定义一个结构体 SpecialFormValue，继承自 SugaredValue，用于表示特殊形式的值
struct TORCH_API SpecialFormValue : public SugaredValue {
  // 构造函数，接受一个 Symbol 参数作为特殊形式的标识符
  SpecialFormValue(Symbol form) : form_(form) {}
  // 返回特殊形式的标识符字符串
  std::string kind() const override {
    return form_.toUnqualString();
  }
  // 返回特殊形式的标识符
  Symbol form() const {
    return form_;
  }
  // 创建一个 SpecialFormValue 的智能指针，传入特殊形式的标识符
  static std::shared_ptr<SpecialFormValue> create(Symbol form) {
    return std::make_shared<SpecialFormValue>(form);
  }

 private:
  // 特殊形式的标识符
  Symbol form_;
};

// 定义一个结构体 LegacyTensorConstructor，继承自 SpecialFormValue，用于表示旧版张量的构造函数
struct TORCH_API LegacyTensorConstructor : public SpecialFormValue {
  // 构造函数，接受特殊形式的标识符、数据类型和设备类型作为参数
  LegacyTensorConstructor(Symbol form, at::ScalarType dtype, at::Device device)
      : SpecialFormValue(form), device_(device), dtype_(dtype) {}

  // 创建一个 LegacyTensorConstructor 的智能指针，传入特殊形式的标识符、数据类型和设备类型
  static std::shared_ptr<LegacyTensorConstructor> create(
      Symbol form,
      at::ScalarType dtype,
      at::Device device) {
    return std::make_shared<LegacyTensorConstructor>(form, dtype, device);
  }
  // 返回张量的数据类型
  at::ScalarType dtype() const {
    return dtype_;
  }

 private:
  // 张量的设备类型
  at::Device device_;
  // 张量的数据类型
  at::ScalarType dtype_;
};

// 用于匹配和处理 range 表达式的结构体 RangeValue，继承自 SugaredValue
struct TORCH_API RangeValue : SugaredValue {
  // 构造函数，接受源代码范围、图函数、输入值向量和静态长度（可选）作为参数
  RangeValue(
      const SourceRange& loc,
      GraphFunction& m,
      std::vector<Value*> input,
      std::optional<int64_t> static_len = c10::nullopt);

  // 返回 range 的类型名称
  std::string kind() const override {
    return "range";
  }
  // 返回 range 的长度
  Value* len(const SourceRange& loc, GraphFunction& m) override;
  // 获取 range 的指定元素
  SugaredValuePtr getitem(
      const SourceRange& loc,
      GraphFunction& m,
      Value* idx,
      TypePtr type_hint = nullptr) override;
  // 返回 range 的迭代器
  std::shared_ptr<SugaredValue> iter(const SourceRange& loc, GraphFunction& m)
      override;

  // 当 Range 是通过 enumerate(iterable_with_static_len) 实例化时，返回其静态长度
  std::optional<int64_t> staticLen() override {
    return static_len_;
  }

 private:
  // range 的起始值
  Value* start_{};
  // range 的结束值
  Value* end_{};
  // range 的步长
  Value* step_{};
  // 一个标志，用于确定是否仅包含来自参数的简单 range() 调用的 end_
  // 如果为 true，则不会插入长度计算和索引推导节点，以简化图并启用更多可能的优化
  bool has_only_end_{};
  // 静态长度，当 Range 是通过 enumerate(iterable_with_static_len) 实例化时使用
  std::optional<int64_t> static_len_;
};

// 专门设计用于匹配和处理内置函数迭代表达式（如 zip()、enumerate() 等）的树结构
// zip 和 enumerate 可以被建模为 SimpleValue/RangeValue 的树：
//    zip(x, y) ->  (x, y)，并将元组分配给每个循环目标
//    enumerate(x) -> (range(0, math.inf, 1), x)
// 因此，像 zip(a, enumerate(b), range(0, 100)) 这样的复杂表达式将是：
// (a, (range(0, math.inf, 1), b), range(0, 100))
// 我们使用这些基本的可迭代对象来填充循环信息，如 max_trip_count，并设置循环目标的值表
// 可迭代对象可以包含 SugaredValues 的列表，如 ModuleLists。如果是这样，则我们展开它并要求它包含的所有值
// Emit it unrolled 且要求它包含的所有值
// 一个可迭代树结构，继承自 SugaredValue
struct TORCH_API IterableTree : SugaredValue {
  // 默认构造函数
  IterableTree() = default;
  // 构造函数，接受源范围、图函数和子节点数组，将每个子节点添加到树中
  IterableTree(
      const SourceRange& range,
      GraphFunction& m,
      at::ArrayRef<SugaredValuePtr> children) {
    // 遍历子节点数组，逐个添加到树中
    for (const auto& child : children) {
      addChild(range, m, child);
    }
  }
  // 返回类型名称为 "iterabletree"
  std::string kind() const override {
    return "iterabletree";
  }

  // 返回共享指针到自身，表示该对象是可迭代的
  std::shared_ptr<SugaredValue> iter(const SourceRange& loc, GraphFunction& m)
      override {
    return shared_from_this();
  }

  // 添加子节点到树中，接受源范围、图函数和子节点的 SugaredValue 指针
  void addChild(
      const SourceRange& range,
      GraphFunction& m,
      const SugaredValuePtr& iter_value);

  // 返回树的所有子节点
  std::vector<SugaredValuePtr> get_children() {
    return children_;
  }

  // 如果此可迭代树包含 ModuleList 或 Tuple，则具有静态长度，返回其长度
  std::optional<int64_t> staticLen() override {
    return unroll_length_;
  }

  // 给定一个 IterableTree 节点，获取所有基本的可迭代对象/叶子节点
  // 这使得我们能够获取所有包含有效循环信息（如 len() 和 getitem()）的基本 SugaredValue
  std::vector<SugaredValuePtr> get_base_iterables();

  // 返回树的长度，接受源范围和图函数
  Value* len(const SourceRange& loc, GraphFunction& m) override;
  
  // 获取树的索引项，接受源范围、图函数、索引值和可选的类型提示
  SugaredValuePtr getitem(
      const SourceRange& loc,
      GraphFunction& m,
      Value* idx,
      TypePtr type_hint = nullptr) override;

 private:
  // 静态长度，可选类型，初始为无
  std::optional<int64_t> unroll_length_ = c10::nullopt;
  // 子节点数组
  std::vector<SugaredValuePtr> children_;
};

// 将 NamedValue 数组转换为 Value 指针数组，接受图和 NamedValue 数组
static inline std::vector<Value*> toValues(
    Graph& g,
    at::ArrayRef<NamedValue> nvs) {
  // 使用 fmap 将 NamedValue 转换为 Value 指针
  return fmap(nvs, [&](const NamedValue& v) { return v.value(g); });
}

// 简单的 Self 类，继承自 Self 类
struct SimpleSelf : public Self {
  // 显式构造函数，接受类类型指针
  explicit SimpleSelf(ClassTypePtr classType)
      : Self(), classType_(std::move(classType)) {}
  // 返回 SugaredValue 指针，设置值类型为类类型
  std::shared_ptr<SugaredValue> makeSugared(Value* v) const override {
    v->setType(classType_);
    return std::make_shared<SimpleValue>(v);
  }
  // 返回类类型指针
  ClassTypePtr getClassType() const override {
    return classType_;
  }

 private:
  // 类类型指针
  ClassTypePtr classType_;
};

// ExceptionMessageValue 是一个 SugaredValue 类型，表示异常信息值
// 该类型无法通过期望 SimpleValue 作为 SugaredValue 的代码路径
struct TORCH_API ExceptionMessageValue : public SugaredValue {
  // 显式构造函数，接受值和限定类名（可选）
  explicit ExceptionMessageValue(
      Value* value,
      Value* qualified_class_name = nullptr)
      : value_(value), qualified_class_name_(qualified_class_name) {}

  // 返回类型名称为 "exception message"
  std::string kind() const override {
    return "exception message";
  }

  // 返回值
  Value* getValue() {
    return value_;
  }

  // 返回限定的 Python 类名
  Value* getQualifiedClassName() {
    return qualified_class_name_;
  }

 private:
  // 值
  Value* value_;
  // 限定的 Python 类名
  Value* qualified_class_name_;
};

// ExceptionValue 是一个 SugaredValue 类型，表示异常值
struct TORCH_API ExceptionValue : public SugaredValue {
  // 显式构造函数，接受异常信息字符串
  explicit ExceptionValue(std::string message) : message_(std::move(message)) {}

  // 返回类型名称为 "exception value"
  std::string kind() const override {
    return "exception value";
  }

  // 异常信息
  std::string getMessage() {
    return message_;
  }

 private:
  // 异常信息字符串
  std::string message_;
};
  return "exception";

# 返回一个字符串 "exception"
  

}

std::shared_ptr<SugaredValue> call(

# 定义一个名为 `call` 的函数，返回类型为 `std::shared_ptr<SugaredValue>`，接受以下参数：


    const SourceRange& loc,

# 引用参数 `loc`，类型为 `const SourceRange&`


    GraphFunction& m,

# 引用参数 `m`，类型为 `GraphFunction&`


    at::ArrayRef<NamedValue> args,

# 引用参数 `args`，类型为 `at::ArrayRef<NamedValue>`


    at::ArrayRef<NamedValue> /*attributes*/,

# 忽略名为 `attributes` 的参数


    size_t /*n_binders*/) override {

# 接受一个无名参数 `n_binders`，返回类型为 `override`


  auto exception_message = insertConstant(*m.graph(), message_ + ": ", loc);

# 在图 `m.graph()` 中插入一个常量，内容为 `message_ + ": "`，结果保存在 `exception_message` 中


  for (auto& input : args) {

# 遍历参数 `args` 中的每一个元素，将每个元素依次赋值给 `input`


    auto input_str = input.value(*m.graph());

# 调用 `input` 的 `value` 方法，将结果保存在 `input_str` 中


    if (!input_str->type()->isSubtypeOf(*StringType::get())) {

# 如果 `input_str` 的类型不是 `StringType` 的子类型，则执行以下操作：


      input_str =
          emitBuiltinCall(loc, *m.graph(), aten::str, {input_str}, {});

# 调用 `emitBuiltinCall` 函数，将结果赋值给 `input_str`


    }
    exception_message = emitBuiltinCall(
        loc, *m.graph(), aten::add, {exception_message, input_str}, {});

# 调用 `emitBuiltinCall` 函数，将结果赋值给 `exception_message`


  }
  return std::make_shared<ExceptionMessageValue>(exception_message);

# 返回一个指向 `ExceptionMessageValue` 类型对象的 `std::shared_ptr`，其参数为 `exception_message`


}

std::string message_;

# 定义一个名为 `message_` 的字符串变量
};

// 定义一个名为 SugaredEnumClass 的结构体，继承自 SugaredValue
struct TORCH_API SugaredEnumClass : public SugaredValue {
  // 构造函数，接受 EnumTypePtr 类型参数 enum_type
  explicit SugaredEnumClass(EnumTypePtr enum_type)
      : enum_type_(std::move(enum_type)) {}

  // 返回字符串 "EnumClass"，覆盖父类的虚函数 kind()
  std::string kind() const override {
    return "EnumClass";
  }

  // 返回值为 SugaredValuePtr 类型的指针，接受 loc、m 和 field 三个参数
  SugaredValuePtr attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // 返回值为 SugaredValuePtr 类型的指针，接受 loc 和 m 两个参数
  SugaredValuePtr iter(const SourceRange& loc, GraphFunction& m) override;

 private:
  EnumTypePtr enum_type_;  // 私有成员变量，类型为 EnumTypePtr，存储枚举类型指针
};

// 定义一个名为 SliceValue 的结构体，继承自 SugaredValue
struct TORCH_API SliceValue : public SugaredValue {
  // 构造函数，接受三个参数：Value* 类型的 start、stop 和 step
  explicit SliceValue(Value* start, Value* stop, Value* step)
      : start_(start), stop_(stop), step_(step) {}

  // 返回字符串 "Python slice value"，覆盖父类的虚函数 kind()
  std::string kind() const override {
    return "Python slice value";
  }

  // 返回 start_ 成员变量，类型为 Value*，用于获取起始值
  Value* start() {
    return start_;
  };
  // 返回 stop_ 成员变量，类型为 Value*，用于获取停止值
  Value* stop() {
    return stop_;
  };
  // 返回 step_ 成员变量，类型为 Value*，用于获取步长值
  Value* step() {
    return step_;
  };

 private:
  Value* start_;  // 私有成员变量，类型为 Value*，存储起始值
  Value* stop_;   // 私有成员变量，类型为 Value*，存储停止值
  Value* step_;   // 私有成员变量，类型为 Value*，存储步长值
};

} // namespace jit
} // namespace torch
```