# `.\pytorch\torch\csrc\jit\python\python_sugared_value.h`

```
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/concrete_module_type.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// 命名空间 torch::jit 内的声明和定义
namespace torch::jit {

// 将 Python 对象的类型转换为字符串表示
std::string typeString(py::handle h);

// 将一个 JIT Value 指针封装成简单的 SugaredValue 共享指针
inline std::shared_ptr<SugaredValue> toSimple(Value* v) {
  return std::make_shared<SimpleValue>(v);
}

// 这应该是从 Python 对象实例化 SugaredValue 的唯一入口点
// 如果要添加对新 Python 类型的支持，应该在此函数的实现中添加
std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    GraphFunction& m,
    const SourceRange& loc,
    bool is_constant = false);

// 尝试将 Python 对象转换为 StrongFunctionPtr
std::optional<StrongFunctionPtr> as_function(const py::object& obj);

// PythonValue 是 SugaredValue 的子类，代表一个 Python 对象
struct VISIBILITY_HIDDEN PythonValue : public SugaredValue {
  PythonValue(
      py::object the_self,
      std::optional<py::object> rcb = c10::nullopt,
      Value* module_self = nullptr)
      : self(std::move(the_self)),
        rcb(std::move(rcb)),
        moduleSelf_(module_self) {}

  // 获取此 PythonValue 的函数模式
  FunctionSchema getSchema(
      const size_t n_args,
      const size_t n_binders,
      const SourceRange& loc);

  // 调用 PythonValue，如 `outputs = this(inputs)`
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& m,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  // 返回 PythonValue 对象的类型描述
  std::string kind() const override;

  // 将 PythonValue 对象转换为元组形式
  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      GraphFunction& m,
      const std::optional<size_t>& size_hint = {}) override;

  // 获取 PythonValue 对象的属性
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // 将 PythonValue 对象转换为 JIT Value，如果无法转换则抛出异常
  Value* asValue(const SourceRange& loc, GraphFunction& m) override {
    throw ErrorReport(loc)
        << kind() << " cannot be used as a value. "
        << "Perhaps it is a closed over global variable? If so, please "
        << "consider passing it in as an argument or use a local varible "
        << "instead.";
  }

 protected:
  // 获取 Python 对象的属性值
  py::object getattr(const SourceRange& loc, const std::string& name);

  // 检查是否存在将常量错误添加到常量列表中的情况
  void checkForAddToConstantsError(std::stringstream& ss);

  // PythonValue 的成员变量，代表 Python 对象本身、rcb 和 moduleSelf_
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  py::object self;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::optional<py::object> rcb;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Value* moduleSelf_ = nullptr;
};

// PythonModuleValue 是 PythonValue 的子类，表示一个 Python 模块对象
struct VISIBILITY_HIDDEN PythonModuleValue : public PythonValue {
  explicit PythonModuleValue(py::object mod) : PythonValue(std::move(mod)) {}

  // 获取 Python 模块对象的属性
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;
};

} // namespace torch::jit
// 用于展开 torch.cuda 模块的使用。所有的 CUDA API，如 torch.cuda.*，都使用 CUDAPythonModuleValue 解析。
struct VISIBILITY_HIDDEN CUDAPythonModuleValue : public PythonValue {
  explicit CUDAPythonModuleValue(py::object mod)
      : PythonValue(std::move(mod)) {}

  // 返回属性的 SugaredValue，用于处理模块中的字段访问
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;
};

// 将模块的所有参数表示为 List[Tensor]
struct VISIBILITY_HIDDEN ConstantParameterList : public SugaredValue {
  ConstantParameterList(Value* the_list) : the_list_(the_list) {}

  // 返回 SugaredValue 的类型描述
  std::string kind() const override {
    return "constant parameter list";
  }

  // 调用函数，返回参数列表的简单形式
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& caller,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    return toSimple(the_list_);
  }

 private:
  Value* the_list_;
};

// 表示模块字典方法的 SugaredValue
struct VISIBILITY_HIDDEN ModuleDictMethod : public SugaredValue {
  explicit ModuleDictMethod(SugaredValuePtr iterable, std::string name)
      : iterable_(std::move(iterable)), name_(std::move(name)){};

  // 返回 SugaredValue 的类型描述，即方法名称
  std::string kind() const override {
    return name_;
  }

  // 调用方法，若有参数则抛出错误
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& f,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    if (!args.empty() || !kwargs.empty()) {
      throw ErrorReport(loc)
          << name_ << " method does not accept any arguments";
    }
    return iterable_;
  }

  SugaredValuePtr iterable_;
  const std::string name_;
};

struct SugaredDict;

// 定义模块或方法在脚本子集中的行为。目前与 Python 没有交互。
// 将来，我们将添加将 `self.foo` 解析为 Python {函数、模块、常量} 的功能，
// 因此这里定义的 SugaredValue 预期最终将 Module 替换为持有实际 nn.Module 类的 py::object。
struct VISIBILITY_HIDDEN ModuleValue : public SugaredValue {
  ModuleValue(Value* self, std::shared_ptr<ConcreteModuleType> concreteType)
      : self_(self), concreteType_(std::move(concreteType)) {}

  // 返回 SugaredValue 的类型描述
  std::string kind() const override {
  // 返回字符串 "module"
  return "module";
}

// 根据源范围和图函数返回一个值对象指针
Value* asValue(const SourceRange& loc, GraphFunction& m) override;

// 根据源范围和图函数返回一个元组值对象的指针
SugaredValuePtr asTupleValue(const SourceRange& loc, GraphFunction& m) override;

// 尝试获取对象的属性，例如 `this.field`
std::shared_ptr<SugaredValue> tryGetAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field);

// 获取对象的属性，例如 `this.field`
std::shared_ptr<SugaredValue> attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) override;

// 检查对象是否具有属性，例如 `this.field`
bool hasAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) override;

// 调用 module.forward，并考虑 pre_hooks 和 hooks
std::shared_ptr<SugaredValue> call(
    const SourceRange& loc,
    GraphFunction& caller,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) override;

// 获取一个SugaredDict对象，用于处理字典操作
std::shared_ptr<SugaredDict> getSugaredDict(
    const SourceRange& loc,
    GraphFunction& m);

// 获取一个SugaredDict对象，用于处理命名缓冲区字典操作
std::shared_ptr<SugaredDict> getSugaredNamedBufferDict(
    const SourceRange& loc,
    GraphFunction& m);

// 获取一个SugaredDict对象，用于处理命名参数列表字典操作
std::shared_ptr<SugaredDict> getSugaredNamedParameterList(
    const SourceRange& loc,
    GraphFunction& m);

// 获取一个SugaredDict对象，用于处理命名参数字典操作
std::shared_ptr<SugaredDict> getSugaredNamedParameterDict(
    const SourceRange& loc,
    GraphFunction& m);

// 设置对象的属性值
void setAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field,
    Value* newValue) override;

// 返回一个迭代器，用于迭代对象
SugaredValuePtr iter(const SourceRange& loc, GraphFunction& m) override;

// 获取对象的元素值，通过索引 idx，并考虑类型提示 type_hint
std::shared_ptr<SugaredValue> getitem(
    const SourceRange& loc,
    GraphFunction& m,
    Value* idx,
    TypePtr type_hint) override;

// 私有方法：检查所有子模块的类型是否是 ty 的子类型，如果不是则打印详细信息到 why_not
bool areAllSubmodulesSubtypeOf(
    const TypePtr& ty,
    std::ostream* why_not = nullptr) const;

Value* self_;  // 对象自身的值
std::shared_ptr<ConcreteModuleType> concreteType_;  // 具体模块类型的共享指针
// 结构体定义结束
};

// 判断对象是否为命名元组类
bool isNamedTupleClass(const py::object& obj);

// 注册命名元组类，并返回类型指针
TypePtr registerNamedTuple(
    const py::object& obj,
    const SourceRange& loc,
    const ResolutionCallback& rcb);

// 递归遍历嵌套模块的函数
void recurseThroughNestedModules(
    const SourceRange& loc,
    GraphFunction& m,
    std::vector<SugaredValuePtr>& keys,
    std::vector<SugaredValuePtr>& values,
    std::shared_ptr<ModuleValue>& self,
    const std::string& prefix,
    const std::string& field);

// 用于支持 named_modules() 函数
struct VISIBILITY_HIDDEN SugaredDict : public SugaredValue {
  explicit SugaredDict(
      std::shared_ptr<ModuleValue> self,
      std::shared_ptr<SugaredTupleValue> keys,
      std::shared_ptr<SugaredTupleValue> modules)
      : self_(std::move(self)),
        keys_(std::move(keys)),
        modules_(std::move(modules)) {}

  std::string kind() const override {
    return "ModuleDict";
  }

  std::shared_ptr<SugaredTupleValue> getKeys() {
    return keys_;
  }

  std::shared_ptr<SugaredTupleValue> getModules() {
    return modules_;
  }

  // 获取属性的值
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // 返回迭代器，这里是 keys_
  SugaredValuePtr iter(const SourceRange& loc, GraphFunction& m) override {
    return keys_;
  };

  std::shared_ptr<ModuleValue> self_;
  std::shared_ptr<SugaredTupleValue> keys_;
  std::shared_ptr<SugaredTupleValue> modules_;
};

// 用于布尔值分发的结构体
struct VISIBILITY_HIDDEN BooleanDispatchValue : public SugaredValue {
  BooleanDispatchValue(py::dict dispatched_fn)
      : dispatched_fn_(std::move(dispatched_fn)) {}

  std::string kind() const override {
    return "boolean dispatch";
  }

  // 调用函数的方法
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& caller,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

 private:
  py::dict dispatched_fn_;
};

// Python 类型的值
struct VISIBILITY_HIDDEN PythonClassValue : public ClassValue {
  PythonClassValue(ClassTypePtr type, py::object py_type)
      : ClassValue(std::move(type)), py_type_(std::move(py_type)) {}

  std::string kind() const override {
    return "Python type";
  }

  // 获取属性的值
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // 判断是否有属性
  bool hasAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

 private:
  py::object py_type_;
};

// Python 异常类型的值
struct VISIBILITY_HIDDEN PythonExceptionValue : public ExceptionValue {
  explicit PythonExceptionValue(const py::object& exception_class)
      : ExceptionValue(
            py::str(py::getattr(exception_class, "__name__", py::str("")))),
        exception_class_qualified_name_(
            py::str(py::module::import("torch._jit_internal")
                        .attr("_qualified_name")(
                            exception_class,
                            /*mangle_name=*/false))) {}

  std::string kind() const override {
    // 返回异常类型的名称
    return "Python Exception";
  }

  std::string exception_class_qualified_name_;
};
    return "Python exception";
  }



// 返回一个字符串 "Python exception"，表示异常情况下的返回值
return "Python exception";



  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& caller,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;



// 实现 call 方法，用于执行函数调用操作
std::shared_ptr<SugaredValue> call(
    const SourceRange& loc,       // 调用位置的源范围
    GraphFunction& caller,        // 调用的图函数对象
    at::ArrayRef<NamedValue> args, // 参数列表
    at::ArrayRef<NamedValue> kwargs, // 关键字参数列表
    size_t n_binders) override;   // 绑定器的数量



 private:
  std::string exception_class_qualified_name_;



// 私有成员变量，存储异常类的限定名称
std::string exception_class_qualified_name_;
};

// Python Slice class.
// 定义一个名为 PythonSliceClass 的结构体，它继承自 SugaredValue 类
struct VISIBILITY_HIDDEN PythonSliceClass : public SugaredValue {
  // 默认构造函数
  explicit PythonSliceClass() = default;

  // 返回当前对象的类型名称
  std::string kind() const override {
    return "Python slice class";
  }

  // 调用函数，返回一个 SugaredValue 类型的共享指针
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,               // 调用位置信息
      GraphFunction& caller,                // 调用者的图函数对象
      at::ArrayRef<NamedValue> args,        // 位置参数数组
      at::ArrayRef<NamedValue> kwargs,      // 关键字参数数组
      size_t n_binders) override;           // 绑定器数量
};

} // namespace torch::jit
```