# `.\pytorch\torch\csrc\jit\python\script_init.cpp`

```py
// 包含 PyTorch 的 C++ 头文件
#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/utils/pybind.h>

// 包含 Caffe2 的版本信息头文件
#include <caffe2/serialize/versions.h>

// 包含 PyTorch 的设备和动态类型相关的头文件
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>

// 包含 PyTorch 的模块和 IR 相关的头文件
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/sugared_value.h>

// 包含 PyTorch 移动端相关的头文件
#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/compatibility/backport.h>
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/quantization.h>

// 包含 PyTorch 运算符升级相关的头文件
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>

// 包含 PyTorch 的 Python 接口相关的头文件
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_sugared_value.h>

// 包含 PyTorch 的序列化和反序列化相关的头文件
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/import.h>

// 包含 PyTorch 测试相关的头文件
#include <torch/csrc/jit/testing/file_check.h>

// 包含 C10 库的异常处理和指针相关头文件
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>

// 包含 PyTorch 的前端解析器和跟踪器相关的头文件
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/tracer.h>

// 包含 PyTorch 的 IR 常量和图形工具相关的头文件
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/graph_utils.h>
#include <torch/csrc/jit/ir/irparser.h>

// 包含 PyTorch 的 passes 和形状分析相关的头文件
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

// 包含 PyTorch 的 Python 绑定工具相关的头文件
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <torch/csrc/jit/python/python_list.h>
#include <torch/csrc/jit/python/python_tracer.h>

// 包含 PyTorch 运行时执行器和指令相关的头文件
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/logging.h>

// 包含 PyTorch 的序列化相关的头文件
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>

// 包含 PyTorch 测试钩子相关的头文件
#include <torch/csrc/jit/testing/hooks_for_testing.h>

// 包含 PyTorch API 中的有序字典头文件
#include <torch/csrc/api/include/torch/ordered_dict.h>

// 包含 ATen 库的核心头文件
#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>

// 包含 PyBind11 的功能和 STL 相关的头文件
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// 包含 PyTorch 移动端训练数据导出相关的头文件
#include <torch/csrc/jit/mobile/train/export_data.h>

// 包含 C++ 标准库的时间处理、大小、内存管理、字符串等头文件
#include <chrono>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>  // 包含标准库头文件<vector>

#include <fmt/format.h>  // 包含fmt库中的format头文件

namespace torch::jit {  // 进入torch::jit命名空间

using ::c10::Argument;  // 使用c10命名空间中的Argument
using ::c10::FunctionSchema;  // 使用c10命名空间中的FunctionSchema

using FunctionDefaults = std::unordered_map<std::string, py::object>;  // 定义FunctionDefaults为string到py::object的无序映射
using ClassMethodDefaults = std::unordered_map<std::string, FunctionDefaults>;  // 定义ClassMethodDefaults为string到FunctionDefaults的无序映射

namespace {  // 匿名命名空间，限定作用域为当前文件

// 一个解析器，将检查外部Python作用域以查找`name`
struct PythonResolver : public Resolver {  // PythonResolver继承自Resolver类
  explicit PythonResolver(ResolutionCallback rcb) : rcb_(std::move(rcb)) {}  // 显式构造函数，初始化rcb_

  /**
   * 在编译类时，因为尚未定义该类，所以类类型在Python中不可用。因此，为了使类类型对其自己的方法可用，我们需要显式解析它。
   *
   * @param rcb 解析名称到其在封闭范围中Python对象的Python函数
   * @param classname 当前编译的类的未限定类名
   * @param classType 类的类型
   */
  explicit PythonResolver(
      ResolutionCallback rcb,
      std::string classname,
      ClassTypePtr classType)
      : rcb_(std::move(rcb)),
        classname_(std::move(classname)),
        classType_(std::move(classType)) {}  // 显式构造函数，初始化rcb_, classname_, classType_

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override {  // 解析值的方法重写
    pybind11::gil_scoped_acquire ag;  // 获取全局解释器锁的作用域
    py::object obj = rcb_(name);  // 使用rcb_函数解析name对应的Python对象
    if (obj.is_none()) {  // 如果解析得到的对象为空
      return nullptr;  // 返回空指针
    }
    return toSugaredValue(obj, m, loc);  // 将Python对象转换为SugaredValue类型并返回
  }

  static bool isNamedTupleClass(py::object obj) {  // 判断是否为命名元组类的静态方法
    auto tuple_type = reinterpret_cast<PyObject*>(&PyTuple_Type);  // 获取Python的元组类型
    return PyObject_IsSubclass(obj.ptr(), tuple_type) &&  // 判断obj是否为元组类型的子类，并且
        py::hasattr(obj, "_fields");  // 判断obj是否有_fields属性
  }

  TypePtr resolveTypeFromObject(const py::object& obj, const SourceRange& loc) {  // 从Python对象解析类型的方法
    if (py::isinstance<ScriptClass>(obj)) {  // 如果obj是ScriptClass的实例
      auto script_class = py::cast<ScriptClass>(obj);  // 将obj转换为ScriptClass类型
      return script_class.class_type_.type_;  // 返回script_class的类类型
    }

    py::bool_ isClass = py::module::import("inspect").attr("isclass")(obj);  // 判断obj是否为类
    if (!py::cast<bool>(isClass)) {  // 如果不是类
      return nullptr;  // 返回空指针
    }

    if (isNamedTupleClass(obj)) {  // 如果obj是命名元组类
      return registerNamedTuple(obj, loc, rcb_);  // 注册命名元组并返回其类型
    }

    auto qualifiedName = c10::QualifiedName(  // 获取obj的限定名称
        py::cast<std::string>(py::module::import("torch._jit_internal")
                                  .attr("_qualified_name")(obj)));

    return get_python_cu()->get_type(qualifiedName);  // 返回Python编译单元中的obj类型
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)  // 解析类型的方法
      override {
    if (classType_ && name == classname_) {  // 如果classType_存在并且name等于classname_
      return classType_;  // 返回classType_
    }
    pybind11::gil_scoped_acquire ag;  // 获取全局解释器锁的作用域
    py::object obj = rcb_(name);  // 使用rcb_函数解析name对应的Python对象
    if (obj.is_none()) {  // 如果解析得到的对象为空
      return nullptr;  // 返回空指针
    }

    auto annotation_type =  // 获取obj的注释类型
        py::module::import("torch.jit.annotations")
            .attr("try_ann_to_type")(obj, loc, py::cpp_function(rcb_));
    if (!annotation_type.is_none()) {  // 如果注释类型不为空
      return py::cast<TypePtr>(annotation_type);  // 将注释类型转换为TypePtr类型并返回
    }
    }
    // 返回 resolveTypeFromObject 函数的调用结果
    return resolveTypeFromObject(obj, loc);
  }

 private:
  // ResolutionCallback 类型的私有成员变量 rcb_
  ResolutionCallback rcb_;
  // std::string 类型的私有成员变量 classname_
  std::string classname_;
  // ClassTypePtr 类型的私有成员变量 classType_
  ClassTypePtr classType_;
};

// 创建一个 PythonResolver 的共享指针，使用给定的回调函数作为参数
std::shared_ptr<PythonResolver> pythonResolver(const ResolutionCallback& rcb) {
  return std::make_shared<PythonResolver>(rcb);
}

// 创建一个 PythonResolver 的共享指针，使用给定的回调函数、类名和类类型作为参数
std::shared_ptr<PythonResolver> pythonResolver(
    const ResolutionCallback& rcb,
    std::string classname,
    ClassTypePtr classType) {
  return std::make_shared<PythonResolver>(
      rcb, std::move(classname), std::move(classType));
}

// 检查重载声明的参数是否一致
void checkOverloadDecl(const Decl& new_decl, const Decl& old_decl) {
  const auto& new_params = new_decl.params();
  const auto& old_params = old_decl.params();

  // 断言新旧声明的参数数量相同
  TORCH_INTERNAL_ASSERT(
      new_params.size() == old_params.size(),
      "Overload must have same number of parameters\n",
      new_decl.range(),
      old_decl.range());

  // 遍历参数列表，断言对应位置的参数名称相同
  for (const auto i : c10::irange(new_decl.params().size())) {
    TORCH_INTERNAL_ASSERT(
        new_params[i].ident().name() == old_params[i].ident().name(),
        "Overload parameters must have the same names\n",
        new_params[i].ident(),
        old_params[i].ident());
  }
}

// 尝试计算默认参数的值，并返回可选的 IValue
std::optional<IValue> tryCalculateDefaultParam(
    const Argument& arg,
    const py::object& def_value) {
  auto n = arg.N();
  auto list_type = arg.type()->cast<ListType>();
  try {
    if (n && *n > 0 && list_type) {
      // 如果是 BroadcastingList 类型，允许使用列表元素的默认值
      return toIValue(def_value, list_type->getElementType());
    } else {
      // 否则，使用参数的类型计算默认值
      return toIValue(def_value, arg.type());
    }
  } catch (...) {
    // 异常时返回空值
    return c10::nullopt;
  }
}

// 计算重载函数的默认参数
FunctionDefaults calcOverloadedFunctionDefaults(
    const FunctionSchema& schema,
    const FunctionDefaults& defaults) {
  FunctionDefaults updated_defaults;

  // 遍历函数模式的参数列表
  for (const auto& arg : schema.arguments()) {
    const std::string& arg_name = arg.name();
    auto value = defaults.find(arg_name);
    if (value == defaults.end()) {
      continue;  // 如果默认值中没有该参数，则跳过
    }
    // 尝试计算参数的默认值，并更新到更新后的默认值列表中
    auto maybe_ivalue = tryCalculateDefaultParam(arg, value->second);
    if (maybe_ivalue) {
      updated_defaults[arg_name] = value->second;
    }
  }
  return updated_defaults;
}

// 检查函数默认参数是否为可变类型
bool checkMutableFunctionDefault(const py::object& def_arg) {
  if (py::isinstance<py::list>(def_arg) || py::isinstance<py::dict>(def_arg)) {
    return true;  // 如果是列表或者字典类型，认为是可变的
  }
  if (py::isinstance<py::tuple>(def_arg)) {
    auto pytuple = def_arg.cast<py::tuple>();
    for (py::handle t : pytuple) {
      py::object obj = py::reinterpret_borrow<py::object>(t);
      if (checkMutableFunctionDefault(obj)) {
        return true;  // 如果元组中有可变类型，则认为是可变的
      }
    }
  }
  return false;  // 否则认为是不可变的
}

// 检查函数默认参数是否为可变类型，并指定源代码范围和参数
void checkMutableFunctionDefault(
    const SourceRange& range,
    const Argument& arg,
    const py::object& def_arg) {
  if (checkMutableFunctionDefault(def_arg) || arg.type()->cast<ClassType>()) {
    // 如果参数是可变类型或者是类类型，处理为可变的情况
    // （这里的具体实现可能需要进一步的上下文理解）
    // 此处缺失了部分代码，需要完整上下文才能提供准确的注释
    # 抛出错误报告，指出不支持可变的默认参数，因为Python将它们绑定到函数上并且跨函数调用保持不变。
    # 可以通过将默认参数设为None，并在函数体内实例化默认值来解决此问题。在参数
    # arg.name() 上发现 def_arg.get_type()。
    throw ErrorReport(range)
        << "Mutable default parameters are not supported because Python binds them to the function"
        << " and they persist across function calls.\n As a workaround, make the default None and instantiate"
        << " the default parameter within the body of the function. Found "
        << def_arg.get_type() << " on parameter " << arg.name();
  }
// 合并函数声明的默认参数和额外参数到重载声明中
static Decl mergeDefaultsAndExtraParametersToOverloadDecl(
    // 重载声明
    const Decl& overload_decl,
    // 实现声明
    const Decl& impl_decl,
    // 函数默认参数映射
    const FunctionDefaults& defaults) {
  // 调整后的参数列表
  std::vector<Param> adjusted_params;
  // 获取重载声明和实现声明的参数列表
  const auto& overload_params = overload_decl.params();
  const auto& impl_params = impl_decl.params();

  // 检查重载声明的参数数量不应大于实现函数的参数数量，遵循 PEP 规范
  TORCH_CHECK(
      overload_params.size() <= impl_params.size(),
      "Overload should not have more parameters than implementation function",
      overload_decl.range(),
      impl_decl.range());

  // 遍历重载声明和实现声明的参数
  for (const auto i : c10::irange(overload_params.size())) {
    auto overload_name = overload_params[i].ident().name();
    auto impl_name = impl_params[i].ident().name();
    // 检查参数名是否相同
    if (overload_name != impl_name) {
      throw ErrorReport(overload_decl.range())
          << "Overload parameters must have the same names. "
          << "Found " << overload_name << " and " << impl_name
          << " on argument " << i;
    }
    // 将参数添加到调整后的参数列表中
    adjusted_params.push_back(overload_params[i]);
  }

  // 处理实现声明中超出重载声明的参数，应用默认参数
  for (size_t i = overload_params.size(); i < impl_params.size(); ++i) {
    // 检查在默认参数映射中是否存在当前参数的默认值
    if (!defaults.count(impl_params[i].ident().name())) {
      throw ErrorReport(impl_decl.range())
          << "Expected to find default parameter on argument"
          << impl_params[i].ident().name()
          << " because it is not defined on the overloaded declaration";
    }
    // 添加参数到调整后的参数列表中
    adjusted_params.push_back(impl_params[i]);
  }

  // 返回包含调整后参数的新声明
  return Decl(overload_decl.kind(), overload_decl.range(), adjusted_params);
}
    # 检查当前重载函数的参数类型是否已经指定
    if (!impl_params[i].type().present()) {
      # 如果未指定参数类型，则抛出错误报告，指明实现函数中未指定类型的参数必须有类型注释
      throw ErrorReport(impl_decl.range())
          << "Parameters not specified on the overloaded declaration must have a type annotation in the implementation function."
          << " Did not find type for param " << impl_params[i].ident().name();
    }
    # 将经过调整的参数添加到调整后的参数列表中
    adjusted_params.push_back(impl_params[i]);
  }
  # 创建一个新的声明，包括重载声明的范围、调整后的参数列表和重载声明的返回类型
  return Decl::create(
      overload_decl.range(),
      List<Param>::create(overload_decl.range(), adjusted_params),
      overload_decl.return_type());
}

static StrongFunctionPtr script_compile_overloaded_function(
    const c10::QualifiedName& name,
    const Decl& overload_decl,
    const Def& implementation_def,
    const ResolutionCallback& rcb,
    const FunctionDefaults& implementation_defaults,
    const py::object& signature) {
  // 如果函数签名为 None，抛出错误报告，要求显式添加类型注解到重载函数
  if (signature.is_none()) {
    throw ErrorReport(overload_decl.range())
        << "Must explicitly add type annotations to overloaded functions";
  }

  // 合并默认值和额外参数到重载声明中
  auto adjusted_decl = mergeDefaultsAndExtraParametersToOverloadDecl(
      overload_decl, implementation_def.decl(), implementation_defaults);
  // 使用调整后的声明创建新的定义
  auto new_def = implementation_def.withDecl(adjusted_decl);
  // 获取 Python 编译单元
  auto cu = get_python_cu();
  // 定义函数并返回已定义的函数列表
  auto defined_functions = cu->define(
      QualifiedName(name.prefix()),
      /*properties=*/{},
      /*propResolvers=*/{},
      {new_def},
      {pythonResolver(rcb)},
      nullptr,
      true);
  // 断言已定义函数的数量为 1
  TORCH_INTERNAL_ASSERT(defined_functions.size() == 1);
  // 获取并更新重载函数的默认值
  auto& defined = defined_functions[0];
  FunctionDefaults updated_defaults = calcOverloadedFunctionDefaults(
      defined->getSchema(), implementation_defaults);
  defined->setSchema(getSchemaWithNameAndDefaults(
      new_def.range(),
      defined->getSchema(),
      new_def.name().name(),
      updated_defaults));
  // 创建并返回强引用指向函数的指针
  StrongFunctionPtr ret(std::move(cu), defined);
  // 完成函数的发射操作
  didFinishEmitFunction(ret);
  return ret;
}

static StrongFunctionPtr script_compile_function(
    const c10::QualifiedName& name,
    const Def& def,
    const FunctionDefaults& defaults,
    const ResolutionCallback& rcb) {
  // 获取 Python 编译单元
  auto cu = get_python_cu();
  // 定义函数并返回已定义的函数列表
  auto defined_functions = cu->define(
      QualifiedName(name.prefix()),
      /*properties=*/{},
      /*propResolvers=*/{},
      {def},
      {pythonResolver(rcb)},
      nullptr,
      true);
  // 断言已定义函数的数量为 1
  TORCH_INTERNAL_ASSERT(defined_functions.size() == 1);
  // 获取并设置函数的模式和默认值
  auto& defined = defined_functions[0];
  defined->setSchema(getSchemaWithNameAndDefaults(
      def.range(), defined->getSchema(), def.name().name(), defaults));
  // 创建并返回强引用指向函数的指针
  StrongFunctionPtr ret(std::move(cu), defined);
  // 完成函数的发射操作
  didFinishEmitFunction(ret);
  return ret;
}

struct VISIBILITY_HIDDEN ModuleSelf : public Self {
  // 构造函数，初始化 ModuleSelf 类
  ModuleSelf(std::shared_ptr<ConcreteModuleType> concreteType)
      : Self(), concreteType_(std::move(concreteType)) {}

  // 生成与值关联的糖值的方法
  std::shared_ptr<SugaredValue> makeSugared(Value* v) const override {
    v->setType(getClassType());
    return std::make_shared<ModuleValue>(v, concreteType_);
  }

  // 获取类类型的方法
  ClassTypePtr getClassType() const override {
    return concreteType_->getJitType()->expect<ClassType>();
  }

 private:
  std::shared_ptr<ConcreteModuleType> concreteType_;
};

static std::shared_ptr<Graph> _propagate_shapes(
    Graph& graph,
    std::vector<at::Tensor> inputs,
    bool with_grad = false) {
  // 使用输入张量创建堆栈
  Stack stack(inputs.begin(), inputs.end());
  // 复制输入图形并返回复制后的图形指针
  auto retval = graph.copy();
  // 设置输入张量的类型到复制的图形中
  setInputTensorTypes(*retval, stack, /*complete=*/false);
  // 传播输入形状
  PropagateInputShapes(retval);
  // 返回传播形状后的图形指针
  return retval;
}

static std::shared_ptr<Graph> _propagate_and_assign_input_shapes(
    Graph& graph,
    std::vector<at::Tensor> inputs,
    bool with_grad = false) {
    const std::vector<at::Tensor>& inputs,

# 定义一个常量引用，类型为 `std::vector<at::Tensor>`，表示输入张量的向量（即输入张量的数组）。


    const std::vector<int>& param_count_list,

# 定义一个常量引用，类型为 `std::vector<int>`，表示参数计数列表的向量（即整数的数组）。


    bool with_grad = false,

# 布尔类型参数，默认为 `false`，指示是否需要梯度信息。


    bool propagate = true) {

# 布尔类型参数，默认为 `true`，指示是否需要传播输入形状信息。


  auto retval = graph.copy();

# 复制输入的图形 `graph`，并将结果存储在变量 `retval` 中。


  setInputTensorTypes(
      *retval, fmap<IValue>(inputs), /*complete=*/true, param_count_list);

# 调用函数 `setInputTensorTypes`，传递参数 `*retval`（图形的指针），`fmap<IValue>(inputs)`（输入张量向量的映射结果），`true`（完整标志位），`param_count_list`（参数计数列表）。


  if (propagate) {

# 如果 `propagate` 参数为真，则执行以下代码块。


    PropagateInputShapes(retval);

# 调用函数 `PropagateInputShapes`，传递参数 `retval`（图形的指针），用于传播输入形状信息。


  }

# 结束条件语句块。


  return retval;

# 返回变量 `retval`，它是输入图形的副本，可能已更新了输入形状信息。
}

// 向模块中添加函数的方法
void addFunctionToModule(Module& module, const StrongFunctionPtr& func) {
  // 创建一个带有虚拟 self 参数的图形
  auto graph = toGraphFunction(*func.function_).graph()->copy();
  // 在图形中插入一个输入节点 "self"
  auto v = graph->insertInput(0, "self");
  // 设置节点的类型为模块的类型
  v->setType(module._ivalue()->type());
  // 构建一个新的方法名，通常是模块类型名加上 "forward"
  const auto name = QualifiedName(*module.type()->name(), "forward");
  // 创建一个函数方法并将其添加到模块的类型中
  auto method =
      module._ivalue()->compilation_unit()->create_function(name, graph);
  module.type()->addMethod(method);
}

// 这个函数用于在测试套件中检查我们是否正确保留了类型标签
bool ivalue_tags_match(const Module& lhs, const Module& rhs) {
  // 定义一个结构体用于处理工作项
  struct Work {
    IValue a;
    IValue b;
  };
  // 使用无序集合来记录已访问过的对象指针，避免重复处理
  std::unordered_set<const void*> visited;
  // 准备工作列表，初始包含左右两个模块的 _ivalue()
  std::vector<Work> work = {{lhs._ivalue(), rhs._ivalue()}};
  // 处理工作列表，直到为空
  while (!work.empty()) {
    // 取出最后一个工作项
    Work item = work.back();
    work.pop_back();
    // 如果 item.a 是指针类型，则记录其指针，避免重复处理
    if (item.a.isPtrType()) {
      if (visited.count(item.a.internalToPointer())) {
        continue;
      }
      visited.emplace(item.a.internalToPointer());
    }
    // 检查 item.b 的类型是否是 item.b 的子类型，如果不是则返回 false
    if (!unshapedType(item.b.type())
             ->isSubtypeOf(unshapedType(item.b.type()))) {
      // 因为命名类型在保存和加载时可能会有差异，所以不应强制它们相等
      if (!item.a.type()->cast<c10::NamedType>()) {
        return false;
      }
    }
    // 如果 item.a 是对象类型，则逐一检查其槽位
    if (item.a.isObject()) {
      auto ao = item.a.toObject();
      auto bo = item.b.toObject();
      for (size_t i = 0; i < ao->slots().size(); ++i) {
        // 将对象的每个槽位作为新的工作项加入到工作列表中
        work.emplace_back(Work{ao->slots().at(i), bo->slots().at(i)});
      }
    } else if (item.a.isTuple()) {  // 如果 item.a 是元组类型
      auto at = item.a.toTuple();
      auto bt = item.b.toTuple();
      for (size_t i = 0; i < at->elements().size(); ++i) {
        // 将元组的每个元素作为新的工作项加入到工作列表中
        work.emplace_back(Work{at->elements().at(i), bt->elements().at(i)});
      }
    } else if (item.a.isList()) {  // 如果 item.a 是列表类型
      auto al = item.a.toList();
      auto bl = item.b.toList();
      for (const auto i : c10::irange(al.size())) {
        // 将列表的每个元素作为新的工作项加入到工作列表中
        work.emplace_back(Work{al.get(i), bl.get(i)});
      }
    } else if (item.a.isGenericDict()) {  // 如果 item.a 是通用字典类型
      auto ad = item.a.toGenericDict();
      auto bd = item.b.toGenericDict();
      for (auto& item : ad) {
        // 将字典的每个键值对的值作为新的工作项加入到工作列表中
        work.emplace_back(Work{item.value(), bd.at(item.key())});
      }
    }
    } else if (item.a.isFuture()) {
      // 如果 item.a 是一个 future 对象，则执行以下操作
      auto af = item.a.toFuture();  // 将 item.a 转换为 future 对象 af
      auto bf = item.b.toFuture();  // 将 item.b 转换为 future 对象 bf
      af->wait();  // 等待 future 对象 af 的完成
      bf->wait();  // 等待 future 对象 bf 的完成
      // 将 af 和 bf 对象的值构造成 Work 对象，并添加到 work 容器中
      work.emplace_back(Work{af->value(), bf->value()});
    }
  }
  
  // 返回 true，表示函数成功执行
  return true;
// helper used to implement ._parameters, ._buffers, ._modules dicts
// inside of script nn.Module
template <typename Policy>
struct slot_dict_impl {
  // 构造函数，接受一个 ModulePtr 参数并初始化 module_
  slot_dict_impl(ModulePtr module) : module_(std::move(module)) {}

  // 检查字典中是否包含指定名称的项
  bool contains(const std::string& name) const {
    // 查找指定名称在模块类型中的属性槽位
    if (auto slot = module_->type()->findAttributeSlot(name)) {
      // 如果找到槽位并且策略允许，则返回 true
      if (Policy::valid(module_->type(), *slot, module_->getSlot(*slot))) {
        return true;
      }
    }
    return false;
  }

  // 返回字典中所有项的名称和对应的 Python 对象的 vector
  std::vector<std::pair<std::string, py::object>> items() const {
    std::vector<std::pair<std::string, py::object>> result;
    // 遍历模块类型的所有属性
    for (size_t i = 0, N = module_->type()->numAttributes(); i < N; ++i) {
      // 如果策略允许，则将属性名和对应的 Python 对象转换后添加到 result 中
      if (Policy::valid(module_->type(), i, module_->getSlot(i))) {
        result.emplace_back(
            module_->type()->getAttributeName(i),
            toPyObject(module_->getSlot(i)));
      }
    }
    return result;
  }

  // 设置字典中指定名称的属性值为给定的 Python 对象
  void setattr(const std::string& name, py::object value) {
    // 获取属性的类型信息
    const TypePtr& type = module_->type()->getAttribute(name);
    // 调用 Module 的 setattr 方法设置属性值
    Module(module_).setattr(name, toIValue(std::move(value), type));
  }

  // 获取字典中指定名称的属性值，并转换为 Python 对象返回
  py::object getattr(const std::string& name) {
    return toPyObject(Module(module_).attr(name));
  }

  // 绑定 slot_dict_impl 类到 Python 模块中，以便 Python 可以使用
  static void bind(const py::module& m, const char* name) {
    py::class_<slot_dict_impl<Policy>>(m, name)
        .def(py::init(
            [](Module& m) { return slot_dict_impl<Policy>(m._ivalue()); }))
        .def("contains", &slot_dict_impl<Policy>::contains)
        .def("items", &slot_dict_impl<Policy>::items)
        .def("setattr", &slot_dict_impl<Policy>::setattr)
        .def("getattr", &slot_dict_impl<Policy>::getattr);
  }

 private:
  ModulePtr module_;
};

// 将 C++ 列表转换为 Python 列表
template <typename T>
py::list debugMakeList(const T& list) {
  py::list result;
  for (const auto& elem : list) {
    result.append(py::cast(elem));
  }
  return result;
}

// 将带有名称和值的结构体转换为 Python 的名值对列表
template <typename T>
py::list debugMakeNamedList(const T& list) {
  py::list result;
  for (auto elem : list) {
    result.append(py::cast(std::make_pair(elem.name, elem.value)));
  }
  return result;
}

// 将 C++ 集合转换为 Python 集合
template <typename T>
py::set debugMakeSet(const T& list) {
  py::set result;
  for (const auto& elem : list) {
    result.add(py::cast(elem));
  }
  return result;
}
static py::dict _jit_debug_module_iterators(Module& module) {
  // 创建一个空的 Python 字典，用于存储调试信息
  py::dict result;
  // 将模块的子模块列表转换为 Python 列表并存入字典中
  result["children"] = debugMakeList(module.children());
  // 将模块的命名子模块列表转换为 Python 字典并存入字典中
  result["named_children"] = debugMakeNamedList(module.named_children());
  // 将模块的所有模块列表转换为 Python 列表并存入字典中
  result["modules"] = debugMakeList(module.modules());
  // 将模块的命名所有模块列表转换为 Python 字典并存入字典中
  result["named_modules"] = debugMakeNamedList(module.named_modules());

  // 将模块的参数列表转换为 Python 列表并存入字典中
  result["parameters"] = debugMakeList(module.parameters(false));
  // 将模块的命名参数列表转换为 Python 字典并存入字典中
  result["named_parameters"] =
      debugMakeNamedList(module.named_parameters(false));
  // 将模块的递归参数列表转换为 Python 列表并存入字典中
  result["parameters_r"] = debugMakeList(module.parameters(true));
  // 将模块的递归命名参数列表转换为 Python 字典并存入字典中
  result["named_parameters_r"] =
      debugMakeNamedList(module.named_parameters(true));

  // 将模块的缓冲区列表转换为 Python 列表并存入字典中
  result["buffers"] = debugMakeList(module.buffers(false));
  // 将模块的命名缓冲区列表转换为 Python 字典并存入字典中
  result["named_buffers"] = debugMakeNamedList(module.named_buffers(false));
  // 将模块的递归缓冲区列表转换为 Python 列表并存入字典中
  result["buffers_r"] = debugMakeList(module.buffers(true));
  // 将模块的递归命名缓冲区列表转换为 Python 字典并存入字典中
  result["named_buffers_r"] = debugMakeNamedList(module.named_buffers(true));

  // 将模块的命名属性列表转换为 Python 字典并存入字典中
  result["named_attributes"] =
      debugMakeNamedList(module.named_attributes(false));
  // 将模块的递归命名属性列表转换为 Python 字典并存入字典中
  result["named_attributes_r"] =
      debugMakeNamedList(module.named_attributes(true));
  // 返回包含所有调试信息的 Python 字典
  return result;
}

static constexpr std::array<const char*, 48> magic_method_names = {
    // 定义包含特殊方法名称的常量数组
    "__lt__",      "__le__",      "__eq__",        "__ne__",
    "__ge__",      "__gt__",      "__not__",       "__abs__",
    "__add__",     "__and__",     "__floordiv__",  "__index__",
    "__inv__",     "__invert__",  "__lshift__",    "__mod__",
    "__mul__",     "__matmul__",  "__neg__",       "__or__",
    "__pos__",     "__pow__",     "__rshift__",    "__sub__",
    "__truediv__", "__xor__",     "__concat__",    "__contains__",
    "__delitem__", "__getitem__", "__setitem__",   "__iadd__",
    "__iand__",    "__iconcat__", "__ifloordiv__", "__ilshift__",
    "__imod__",    "__imul__",    "__imatmul__",   "__ior__",
    "__ipow__",    "__irshift__", "__isub__",      "__itruediv__",
    "__ixor__",    "__str__",     "__len__",       "__repr__",
};

struct DeepCopyMemoTable {
  std::shared_ptr<IValue::HashIdentityIValueMap> map;
};

IValue pyIValueDeepcopy(const IValue& ivalue, const py::dict& memo) {
  // 深拷贝 Python 对象 ivalue，使用 memo 表来避免循环引用
  if (!memo.contains(py::str("__torch_script_memo_table"))) {
    // 如果 memo 中不存在名为 "__torch_script_memo_table" 的项，则创建并初始化之
    memo["__torch_script_memo_table"] =
        DeepCopyMemoTable{std::make_shared<IValue::HashIdentityIValueMap>()};
  }
  auto& ivalue_memo =
      *py::cast<DeepCopyMemoTable>(memo["__torch_script_memo_table"]).map;
  // 返回深拷贝后的 IValue 对象
  return ivalue.deepcopy(ivalue_memo);
}

ExtraFilesMap extra_files_from_python(const py::dict& pydict) {
  // 从 Python 字典中提取额外文件的信息，初始化为一个空映射
  ExtraFilesMap r;
  // 遍历 Python 字典中的每一项，将键转换为字符串作为映射的键
  for (const auto& it : pydict) {
    r[py::cast<std::string>(it.first)] = "";
  }
  // 返回包含额外文件信息的映射
  return r;
}

void extra_files_to_python(const ExtraFilesMap& m, const py::dict& pydict) {
  // 将额外文件信息映射 m 转换为 Python 字典 pydict
  // py::dict 是类似指针的类型，尽管是 const 引用，也可以修改其内容
  for (const auto& it : m) {
    // 将映射 m 中的每一项转换为 Python 字典中的键值对
    pydict[py::str(it.first)] = py::bytes(it.second);
  }
}

void pyCompilationUnitDefine(
    CompilationUnit& cu,
    const std::string& src,
    const ResolutionCallback* rcb,
    // 如果传入的指针 rcb 不为空，并且指针指向的对象也不为空
    if (rcb && *rcb) {
        // 使用 pythonResolver 函数创建解析回调对象，并使用其定义上下文
        cu.define(c10::nullopt, src, pythonResolver(*rcb), nullptr);
    } else {
        // 否则，从 torch._jit_internal 模块中导入 createResolutionCallbackFromFrame 函数
        py::object py_default_rcb =
            py::module::import("torch._jit_internal")
                .attr("createResolutionCallbackFromFrame")(_frames_up);
        // 将 Python 对象 py_default_rcb 转换为 ResolutionCallback 类型
        auto default_rcb = py_default_rcb.cast<ResolutionCallback>();
        // 使用默认的解析回调对象定义上下文
        cu.define(c10::nullopt, src, pythonResolver(default_rcb), nullptr);
    }
// } 表示匹配前面的 if 或者 else 语句块的结束

// 将 bytes 字符串复制到一个按 kFlatbufferDataAlignmentBytes 边界对齐的 shared_ptr<char> 中（当前为 16 字节）
// 这是必需的，因为张量需要按 16 字节边界对齐
static std::shared_ptr<char> copyStr(const std::string& bytes) {
  // 计算所需内存大小，使得大小是 kFlatbufferDataAlignmentBytes 的倍数
  size_t size = (bytes.size() / kFlatbufferDataAlignmentBytes + 1) * kFlatbufferDataAlignmentBytes;
  
  // 根据操作系统不同选择不同的内存分配方式
#ifdef _WIN32
  std::shared_ptr<char> bytes_copy(
      static_cast<char*>(_aligned_malloc(size, kFlatbufferDataAlignmentBytes)),
      _aligned_free);
#elif defined(__APPLE__)
  void* p;
  // 在 macOS 上使用 posix_memalign 进行内存分配
  ::posix_memalign(&p, kFlatbufferDataAlignmentBytes, size);
  TORCH_INTERNAL_ASSERT(p, "Could not allocate memory for flatbuffer");
  std::shared_ptr<char> bytes_copy(static_cast<char*>(p), free);
#else
  // 在其他操作系统上使用 aligned_alloc 进行内存分配
  std::shared_ptr<char> bytes_copy(
      static_cast<char*>(aligned_alloc(kFlatbufferDataAlignmentBytes, size)),
      free);
#endif
  
  // 将 bytes 字符串的内容复制到分配的内存中
  memcpy(bytes_copy.get(), bytes.data(), bytes.size());
  
  // 返回包含复制数据的 shared_ptr
  return bytes_copy;
}

// 如果 special_magic_methods 中包含 mm_name，则将对应的方法定义到 object_class 中
// 否则，定义一个 lambda 函数作为 mm_name 方法的默认实现
if (special_magic_methods.count(mm_name)) {
  object_class.def(mm_name, special_magic_methods[mm_name]);
} else {
  object_class.def(
      mm_name,
      [mm_name](const Object& self, py::args args, py::kwargs kwargs) {
        auto method = self.find_method(mm_name);
        if (!method) {
          // 如果方法未实现，则抛出 NotImplementedError 异常
          std::string msg = fmt::format(
              "'{}' is not implemented for {}",
              mm_name,
              self.type()->str());
          throw c10::NotImplementedError(msg);
        }
        // 调用 Python 中的方法来执行脚本方法
        return invokeScriptMethodFromPython(
            *method,
            // NOLINTNEXTLINE(performance-move-const-arg)
            std::move(args),
            // NOLINTNEXTLINE(performance-move-const-arg)
            std::move(kwargs));
      });
}

// 创建一个 Parser 对象，解析给定的源码字符串 src 中的函数定义，并返回解析结果
Parser p(std::make_shared<Source>(src));
return Def(p.parseFunction(/*is_method=*/true));
  return _get_model_bytecode_version(filename);
  });


  // 返回模型字节码版本号
  m.def(
      "_get_model_extra_files",
      [](const std::string& filename, const py::dict& py_extra_files) {
        std::optional<at::Device> optional_device;
        ExtraFilesMap cpp_extra_files = ExtraFilesMap();
        // 调用_load_for_mobile函数加载模型文件到cpp_extra_files
        _load_for_mobile(filename, optional_device, cpp_extra_files);
        // 将C++端的额外文件映射转换为Python字典
        extra_files_to_python(cpp_extra_files, py_extra_files);

        return py_extra_files;
      });


  // 从缓冲区中获取模型字节码版本号
  m.def(
      "_get_model_bytecode_version_from_buffer", [](const std::string& buffer) {
        std::istringstream in(buffer);
        // 调用_get_model_bytecode_version函数获取模型字节码版本号
        return _get_model_bytecode_version(in);
      });


  // 从缓冲区中获取模型的额外文件
  m.def(
      "_get_model_extra_files_from_buffer",
      [](const std::string& buffer, const py::dict& py_extra_files) {
        std::optional<at::Device> optional_device;
        ExtraFilesMap cpp_extra_files = ExtraFilesMap();
        std::istringstream in(buffer);
        // 调用_load_for_mobile函数加载模型文件到cpp_extra_files
        _load_for_mobile(in, optional_device, cpp_extra_files);
        // 将C++端的额外文件映射转换为Python字典
        extra_files_to_python(cpp_extra_files, py_extra_files);

        return py_extra_files;
      });


  // 获取移动端模型中包含的类型信息
  m.def("_get_mobile_model_contained_types", [](const std::string& filename) {
    return _get_mobile_model_contained_types(filename);
  });


  // 从缓冲区中获取移动端模型中包含的类型信息
  m.def(
      "_get_mobile_model_contained_types_from_buffer",
      [](const std::string& buffer) {
        std::istringstream in(buffer);
        // 调用_get_mobile_model_contained_types函数获取移动端模型中包含的类型信息
        return _get_mobile_model_contained_types(in);
      });


  // 将普通的PyTorch模块转换为移动端模型
  m.def("_nn_module_to_mobile", [](const Module& module) {
    CompilationOptions options;
    // 调用jitModuleToMobile函数将模块转换为移动端模型
    return jitModuleToMobile(module, options);
  });


  // 定义Python类OperatorInfo，表示操作符信息
  py::class_<OperatorInfo>(m, "OperatorInfo")
      .def_readonly("num_schema_args", &OperatorInfo::num_schema_args);


  // 获取模型的操作符和信息
  m.def("_get_model_ops_and_info", [](const std::string& filename) {
    return _get_model_ops_and_info(filename);
  });


  // 从缓冲区中获取模型的操作符和信息
  m.def("_get_model_ops_and_info_from_buffer", [](const std::string& buffer) {
    std::istringstream in(buffer);
    // 调用_get_model_ops_and_info函数获取模型的操作符和信息
    return _get_model_ops_and_info(in);
  });


  // 导出运算符列表
  m.def("_export_operator_list", [](torch::jit::mobile::Module& sm) {
    // 调用debugMakeSet函数导出移动端模型的运算符列表
    return debugMakeSet(torch::jit::mobile::_export_operator_list(sm));
  });


  // 在设备上执行动态PTQ量化
  m.def(
      "_quantize_ondevice_ptq_dynamic",
      [](mobile::Module& m, const std::string& method_name) {
        mobile::quantization::PTQQuanizationHelper ptq_helper;
        // 调用ptq_helper的quantize_dynamic方法执行动态PTQ量化
        ptq_helper.quantize_dynamic(m, method_name);
      });


  // 设置JIT的发射钩子函数
  m.def("_jit_set_emit_hooks", setEmitHooks);


  // 获取JIT的发射钩子函数
  m.def("_jit_get_emit_hooks", getEmitHooks);


  // 清除JIT类注册表
  m.def("_jit_clear_class_registry", []() {
    get_python_cu()->_clear_python_cu();
  });


    // 调用 get_python_cu() 函数获取当前 Python 编译单元，并清除其中的 Python 编译单元
    // 注：该函数可能用于清除当前 Python 编译单元的状态或缓存信息
    m.def(


      "_debug_set_autodiff_subgraph_inlining",
      debugSetAutodiffSubgraphInlining);


      // 定义名为 "_debug_set_autodiff_subgraph_inlining" 的 Python 绑定函数，映射到 C++ 函数 debugSetAutodiffSubgraphInlining
      // 该函数可能用于设置自动微分子图的内联行为
      m.def("_debug_set_fusion_group_inlining", debugSetFusionGroupInlining);


  m.def("_debug_get_fusion_group_inlining", getFusionGroupInlining);


  // 定义名为 "_debug_get_fusion_group_inlining" 的 Python 绑定函数，映射到 C++ 函数 getFusionGroupInlining
  // 该函数可能用于获取融合组的内联信息
  m.def("_propagate_shapes", _propagate_shapes);


  // 定义名为 "_propagate_shapes" 的 Python 绑定函数，映射到 C++ 函数 _propagate_shapes
  // 该函数可能用于推广形状信息
  m.def(


      "_propagate_and_assign_input_shapes", _propagate_and_assign_input_shapes);


      // 定义名为 "_propagate_and_assign_input_shapes" 的 Python 绑定函数，映射到 C++ 函数 _propagate_and_assign_input_shapes
      // 该函数可能用于推广和分配输入形状信息
      m.def(


      "_last_executed_optimized_graph",
      []() { return lastExecutedOptimizedGraph(); },
      "Retrieve the optimized graph that was run the last time the graph executor ran on this thread");


      // 定义名为 "_last_executed_optimized_graph" 的 Python 绑定函数，使用 lambda 表达式实现
      // 函数功能为获取上次图执行器在当前线程上运行时所使用的优化图
      m.def(


      "_create_function_from_graph",
      [](const std::string& qualname, std::shared_ptr<Graph> graph) {
        // TODO this should go in the global Python CU
        auto cu = std::make_shared<CompilationUnit>();
        c10::QualifiedName name(qualname);
        auto fn = cu->create_function(std::move(name), std::move(graph));
        return StrongFunctionPtr(std::move(cu), fn);
      });


      // 定义名为 "_create_function_from_graph" 的 Python 绑定函数，使用 lambda 表达式实现
      // 函数接受一个限定名和图对象，创建一个新函数，并返回一个 StrongFunctionPtr
      // 注：cu 可能指代 CompilationUnit（编译单元），用于管理函数的创建和存储
      m.def("_ivalue_tags_match", ivalue_tags_match);


  // 定义名为 "_ivalue_tags_match" 的 Python 绑定函数，映射到 C++ 函数 ivalue_tags_match
  // 该函数可能用于比较 IValue 的标签信息是否匹配
  m.def("_ivalue_debug_python_object", [](py::object py_obj) {


    // 定义名为 "_ivalue_debug_python_object" 的 Python 绑定函数，使用 lambda 表达式实现
    // 函数接受一个 Python 对象作为参数，将其转换为 IValue，并增加引用计数
    // 返回原始的 PyObject 对象，引用计数加一
    IValue pyobj_ivalue = toIValue(std::move(py_obj), PyObjectType::get());


    // 将 Python 对象转换为 IValue，IValue 通过 py::object 增加引用计数
    // convert back to PyObject by borrowing the reference, which also
    // incref, after the return of this function, IValue is out of scope
    // which decref, so the return value is original refcount + 1
    // 将 IValue 转换回 PyObject，通过借用引用来获取，同时增加引用计数
    // 函数返回后，IValue 超出作用域，引用计数减少，因此返回值是原始引用计数加一
    py::object ret = toPyObject(pyobj_ivalue);
  return ret;
});
m.def("_jit_debug_module_iterators", _jit_debug_module_iterators);


  // 返回 ret 变量
  return ret;
});
// 定义 Python 绑定函数 _jit_debug_module_iterators
m.def("_jit_debug_module_iterators", _jit_debug_module_iterators);



py::class_<testing::FileCheck>(m, "FileCheck")
    .def(py::init<>())
    .def("check", &testing::FileCheck::check)
    .def("check_not", &testing::FileCheck::check_not)
    .def("check_same", &testing::FileCheck::check_same)
    .def("check_next", &testing::FileCheck::check_next)
    .def("check_count", &testing::FileCheck::check_count)
    .def("check_dag", &testing::FileCheck::check_dag)
    .def(
        "check_source_highlighted",
        &testing::FileCheck::check_source_highlighted)
    .def("check_regex", &testing::FileCheck::check_regex)
    .def(
        "check_count",
        [](testing::FileCheck& f,
           const std::string& str,
           size_t count,
           bool exactly) { return f.check_count(str, count, exactly); },
        "Check Count",
        py::arg("str"),
        py::arg("count"),
        py::arg("exactly") = false)
    .def(
        "run",
        [](testing::FileCheck& f, const std::string& str) {
          return f.run(str);
        })
    .def(
        "run", [](testing::FileCheck& f, const Graph& g) { return f.run(g); })
    .def(
        "run",
        [](testing::FileCheck& f,
           const std::string& input,
           const std::string& output) { return f.run(input, output); },
        "Run",
        py::arg("checks_file"),
        py::arg("test_file"))
    .def(
        "run",
        [](testing::FileCheck& f, const std::string& input, const Graph& g) {
          return f.run(input, g);
        },
        "Run",
        py::arg("checks_file"),
        py::arg("graph"));


py::class_<testing::FileCheck>(m, "FileCheck")
    // 定义 Python 类型 FileCheck，并注册初始化函数
    .def(py::init<>())
    // 将 C++ 方法 check 映射为 Python 方法
    .def("check", &testing::FileCheck::check)
    // 将 C++ 方法 check_not 映射为 Python 方法
    .def("check_not", &testing::FileCheck::check_not)
    // 将 C++ 方法 check_same 映射为 Python 方法
    .def("check_same", &testing::FileCheck::check_same)
    // 将 C++ 方法 check_next 映射为 Python 方法
    .def("check_next", &testing::FileCheck::check_next)
    // 将 C++ 方法 check_count 映射为 Python 方法
    .def("check_count", &testing::FileCheck::check_count)
    // 将 C++ 方法 check_dag 映射为 Python 方法
    .def("check_dag", &testing::FileCheck::check_dag)
    // 将 C++ 方法 check_source_highlighted 映射为 Python 方法
    .def("check_source_highlighted", &testing::FileCheck::check_source_highlighted)
    // 将 C++ 方法 check_regex 映射为 Python 方法
    .def("check_regex", &testing::FileCheck::check_regex)
    // 将 C++ 方法 check_count 以 lambda 函数形式映射为 Python 方法
    .def(
        "check_count",
        [](testing::FileCheck& f,
           const std::string& str,
           size_t count,
           bool exactly) { return f.check_count(str, count, exactly); },
        "Check Count",
        py::arg("str"),
        py::arg("count"),
        py::arg("exactly") = false)
    // 将 C++ 方法 run 以 lambda 函数形式映射为 Python 方法
    .def(
        "run",
        [](testing::FileCheck& f, const std::string& str) {
          return f.run(str);
        })
    // 将 C++ 方法 run 以 lambda 函数形式映射为 Python 方法，接受 Graph 类型参数
    .def(
        "run", [](testing::FileCheck& f, const Graph& g) { return f.run(g); })
    // 将 C++ 方法 run 以 lambda 函数形式映射为 Python 方法，接受两个字符串参数
    .def(
        "run",
        [](testing::FileCheck& f,
           const std::string& input,
           const std::string& output) { return f.run(input, output); },
        "Run",
        py::arg("checks_file"),
        py::arg("test_file"))
    // 将 C++ 方法 run 以 lambda 函数形式映射为 Python 方法，接受字符串和 Graph 参数
    .def(
        "run",
        [](testing::FileCheck& f, const std::string& input, const Graph& g) {
          return f.run(input, g);
        },
        "Run",
        py::arg("checks_file"),
        py::arg("graph"));



m.def(
    "_logging_set_logger",
    [](logging::LoggerBase* logger) { return logging::setLogger(logger); },
    py::return_value_policy::reference);
m.def("_set_graph_executor_optimize", [](bool optimize) {
  setGraphExecutorOptimize(optimize);
});


// 定义 Python 绑定函数 _logging_set_logger，接受 logging::LoggerBase* 类型参数
m.def(
    "_logging_set_logger",
    [](logging::LoggerBase* logger) { return logging::setLogger(logger); },
    py::return_value_policy::reference);
// 定义 Python 绑定函数 _set_graph_executor_optimize，接受布尔类型参数
m.def("_set_graph_executor_optimize", [](bool optimize) {
  setGraphExecutorOptimize(optimize);
});



m.def(
    "_get_graph_executor_optimize",
    [](std::optional<bool> new_setting = c10::nullopt) {
      bool old_value = getGraphExecutorOptimize();
      if (new_setting) {
        setGraphExecutorOptimize(*new_setting);
      }
      return old_value;
    },
    py::arg("new_settings") = nullptr);


// 定义 Python 绑定函数 _get_graph_executor_optimize，接受可选的布尔类型参数
m.def(
    "_get_graph_executor_optimize",
    [](std::optional<bool> new_setting = c10::nullopt) {
      // 获取当前的 GraphExecutor 优化设置
      bool old_value = getGraphExecutorOptimize();
      // 如果有新的设置值传入，则更新 GraphExecutor 优化设置
      if (new_setting) {
        setGraphExecutorOptimize(*new_setting);
      }
      // 返回旧的设置值
      return old_value;
    },
    py::arg("new_settings") = nullptr);



m.def(
    "_enable_mobile_interface_call_export",
    &torch::jit::enableMobileInterfaceCallExport);


// 定义 Python 绑定函数 _enable_mobile_interface_call_export，调用 Torch JIT 的函数 enableMobileInterfaceCallExport
m.def(
    "_enable_mobile_interface_call_export",
    &torch::jit::enableMobileInterfaceCallExport);



m.def("_create_module_with_type", [](const ClassTypePtr& type) {
   return Module(get_python_cu(), type);
 }).def("_create_object_with_type", [](const ClassTypePtr& type) {
  return Object(get_python_cu(), type);
});


// 定义 Python 绑定函数 _create_module_with_type，接受 ClassTypePtr 类型参数
m.def("_create_module_with_type", [](const ClassTypePtr& type) {
   // 使用给定的 type 创建 Module 对象，并返回
   return Module(get_python_cu(), type);
}).def("_create_object_with_type", [](const ClassTypePtr& type) {
  // 使用给定的 type 创建 Object 对象，并返回
  return Object(get_python_cu(), type);
});



m.def("_export_opnames", [](Module& sm) {
  return py::isinstance<Object>(obj);
});


// 定义 Python 绑定函数 _export_opnames，接受 Module 对象参数 sm
m.def("_export_opnames", [](Module& sm) {
  // 检查 obj 是否为 Object 类型的实例，并返回结果
  return py::isinstance<Object>(obj);
});



m.def("_get_file_format", [](const std::string& path) {


// 定义 Python 绑定函数 _get_file_format，接受 std::string 类型参数 path
m.def("_get_file_format", [](const std::string& path) {
  switch (getFileFormat(path)) {
    case FileFormat::FlatbufferFileFormat:
      // 如果文件格式是 Flatbuffer，则返回字符串 "flatbuffer"
      return "flatbuffer";
    case FileFormat::ZipFileFormat:
      // 如果文件格式是 Zip 文件，则返回字符串 "zipfile"
      return "zipfile";
    default:
      // 如果文件格式不符合预期，则返回字符串 "invalid"
      return "invalid";
  }
});

m.def(
    "_save_parameters",
    [](const std::map<std::string, at::Tensor>& map,
       const std::string& filename,
       bool use_flatbuffer = false) {
      // 调用 _save_parameters 函数，保存参数到文件
      _save_parameters(map, filename, use_flatbuffer);
    });

m.def("_load_mobile_module_from_file", [](const std::string& filename) {
  // 调用 torch::jit::load_mobile_module_from_file 函数，从文件加载移动模块
  return torch::jit::load_mobile_module_from_file(filename);
});
m.def("_load_mobile_module_from_bytes", [](const std::string& bytes) {
  // 复制字节内容，调用 torch::jit::parse_and_initialize_mobile_module 函数解析和初始化移动模块
  auto bytes_copy = copyStr(bytes);
  return torch::jit::parse_and_initialize_mobile_module(
      bytes_copy, bytes.size());
});
m.def("_load_jit_module_from_file", [](const std::string& filename) {
  // 准备额外文件映射，调用 torch::jit::load_jit_module_from_file 函数加载 JIT 模块
  ExtraFilesMap extra_files = ExtraFilesMap();
  return torch::jit::load_jit_module_from_file(filename, extra_files);
});
m.def("_load_jit_module_from_bytes", [](const std::string& bytes) {
  // 复制字节内容，准备额外文件映射，调用 torch::jit::parse_and_initialize_jit_module 函数解析和初始化 JIT 模块
  auto bytes_copy = copyStr(bytes);
  ExtraFilesMap extra_files = ExtraFilesMap();
  return torch::jit::parse_and_initialize_jit_module(
      bytes_copy, bytes.size(), extra_files);
});
m.def(
    "_save_mobile_module",
    [](const torch::jit::mobile::Module& module,
       const std::string& filename,
       const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
      // 调用 torch::jit::save_mobile_module 函数，保存移动模块到文件
      return torch::jit::save_mobile_module(module, filename, _extra_files);
    });
m.def(
    "_save_jit_module",
    [](const torch::jit::Module& module,
       const std::string& filename,
       const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
      // 调用 torch::jit::save_jit_module 函数，保存 JIT 模块到文件
      return torch::jit::save_jit_module(module, filename, _extra_files);
    });
m.def(
    "_save_mobile_module_to_bytes",
    [](const torch::jit::mobile::Module& module,
       const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
      // 调用 torch::jit::save_mobile_module_to_bytes 函数，将移动模块保存为字节流
      auto detached_buffer =
          torch::jit::save_mobile_module_to_bytes(module, _extra_files);
      return py::bytes(
          reinterpret_cast<char*>(detached_buffer->data()),
          detached_buffer->size());
    });
m.def(
    "_save_jit_module_to_bytes",
    [](const torch::jit::Module& module,
       const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
      // 调用 torch::jit::save_jit_module_to_bytes 函数，将 JIT 模块保存为字节流
      auto detached_buffer =
          torch::jit::save_jit_module_to_bytes(module, _extra_files);
      return py::bytes(
          reinterpret_cast<char*>(detached_buffer->data()),
          detached_buffer->size());
    });
m.def("_get_module_info_from_flatbuffer", [](std::string flatbuffer_content) {
  py::gil_scoped_acquire acquire;
  py::dict result;
  // 调用 torch::jit::get_module_info_from_flatbuffer 函数，获取 Flatbuffer 格式模块信息
  mobile::ModuleInfo minfo =
      torch::jit::get_module_info_from_flatbuffer(&flatbuffer_content[0]);
  result["bytecode_version"] = minfo.bytecode_version;
  result["operator_version"] = minfo.operator_version;
    result["function_names"] = minfo.function_names;
    // 将 minfo 中的函数名列表赋值给 result 字典的 "function_names" 键
    result["type_names"] = minfo.type_names;
    // 将 minfo 中的类型名列表赋值给 result 字典的 "type_names" 键
    result["opname_to_num_args"] = minfo.opname_to_num_args;
    // 将 minfo 中的操作名到参数个数的映射赋值给 result 字典的 "opname_to_num_args" 键
    return result;
  });
  // 返回包含 minfo 中函数、类型和操作名映射的 result 字典

  m.def("_pickle_save", [](IValue v) {
    // 匿名 Lambda 函数，接收一个 IValue 类型参数 v
    auto bytes = torch::jit::pickle_save(std::move(v));
    // 调用 torch::jit::pickle_save 函数序列化参数 v，并将结果保存到 bytes 变量中
    return py::bytes(bytes.data(), bytes.size());
    // 返回一个 py::bytes 对象，包含序列化后的数据
  });

  initScriptDictBindings(module);
  // 调用初始化脚本字典绑定的函数，初始化 module 的字典绑定
  initScriptListBindings(module);
  // 调用初始化脚本列表绑定的函数，初始化 module 的列表绑定
}

} // namespace torch::jit
```