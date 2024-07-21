# `.\pytorch\torch\csrc\utils\python_arg_parser.cpp`

```py
// 包含 Torch C++ 库中的头文件，用于处理 Python 函数参数解析
#include <torch/csrc/utils/python_arg_parser.h>

// Torch C++ 库中的异常处理相关头文件
#include <torch/csrc/Exceptions.h>

// Torch C++ 库中的张量布局相关头文件
#include <torch/csrc/Layout.h>

// Torch C++ 库中的内存格式相关头文件
#include <torch/csrc/MemoryFormat.h>

// Torch C++ 库中的自动求导相关 Python 变量处理头文件
#include <torch/csrc/autograd/python_variable.h>

// Torch C++ 库中的无效参数处理相关头文件
#include <torch/csrc/utils/invalid_arguments.h>

// Torch C++ 库中的 Python 字符串处理头文件
#include <torch/csrc/utils/python_strings.h>

// Torch C++ 库中的 Torch 函数模式相关头文件
#include <torch/csrc/utils/python_torch_function_mode.h>

// Torch C++ 库中的 Torch 分发模式相关头文件
#include <torch/csrc/utils/torch_dispatch_mode.h>

// ATen 张量库的主要头文件
#include <ATen/ATen.h>

// ATen Python Torch Function 线程局部存储相关头文件
#include <ATen/PythonTorchFunctionTLS.h>

// ATen 跟踪模式相关头文件
#include <ATen/TracerMode.h>

// C10 工具库的循环范围处理头文件
#include <c10/util/irange.h>

// 标准库头文件
#include <sstream>         // 字符串流
#include <stdexcept>       // 标准异常类
#include <string>          // 字符串处理
#include <unordered_map>   // 无序映射容器
#include <vector>          // 向量容器

// Torch 命名空间
namespace torch {

// 参数类型映射表，将字符串映射为对应的参数类型枚举值
static std::unordered_map<std::string, ParameterType> type_map = {
    {"Tensor", ParameterType::TENSOR},
    {"Scalar", ParameterType::SCALAR},
    {"int64_t", ParameterType::INT64},
    {"SymInt", ParameterType::SYM_INT},
    {"double", ParameterType::DOUBLE},
    {"complex", ParameterType::COMPLEX},
    {"TensorList", ParameterType::TENSOR_LIST},
    {"c10::List<::std::optional<Tensor>>", ParameterType::TENSOR_LIST},
    {"IntArrayRef", ParameterType::INT_LIST},
    {"SymIntArrayRef", ParameterType::SYM_INT_LIST},
    {"ArrayRef<double>", ParameterType::FLOAT_LIST},
    {"Generator", ParameterType::GENERATOR},
    {"bool", ParameterType::BOOL},
    {"Storage", ParameterType::STORAGE},
    {"PyObject*", ParameterType::PYOBJECT},
    {"ScalarType", ParameterType::SCALARTYPE},
    {"Layout", ParameterType::LAYOUT},
    {"MemoryFormat", ParameterType::MEMORY_FORMAT},
    {"QScheme", ParameterType::QSCHEME},
    {"Device", ParameterType::DEVICE},
    {"DeviceIndex", ParameterType::INT64},
    {"Stream", ParameterType::STREAM},
    {"std::string", ParameterType::STRING},
    {"c10::string_view", ParameterType::STRING},
    {"Dimname", ParameterType::DIMNAME},
    {"DimnameList", ParameterType::DIMNAME_LIST},
    {"ScalarList", ParameterType::SCALAR_LIST},
    {"DispatchKeySet", ParameterType::DISPATCH_KEY_SET},
};

// NumPy 兼容性参数名称映射表，用于 Torch 函数参数和 NumPy 函数参数的对应关系
// 例如，"dim" 对应于 "axis"
static const std::unordered_map<std::string, std::vector<std::string>>
    numpy_compatibility_arg_names = {
        {"dim", {"axis"}},           // 维度参数对应关系
        {"keepdim", {"keepdims"}},   // 保持维度参数对应关系
        {"input", {"x", "a", "x1"}}, // 输入参数对应关系
        {"other", {"x2"}},           // 其他参数对应关系
};

// TODO: remove this. This is a temporary list of functions that allow Python
// numbers to bind to Tensors. Some binary ops have separate Tensor and Scalar
// 判断是否应该允许将数字视为张量的函数
bool should_allow_numbers_as_tensors(const std::string& name) {
  // 允许将以下方法名作为张量处理的白名单集合
  static std::unordered_set<std::string> allowed = {
      "add",          "add_",          "add_out",
      "div",          "div_",          "div_out",
      "divide",       "divide_",       "divide_out", // div的别名
      "mul",          "mul_",          "mul_out",
      "multiply",     "multiply_",     "multiply_out", // mul的别名
      "sub",          "sub_",          "sub_out",
      "subtract",     "subtract_",     "subtract_out", // sub的别名
      "true_divide",  "true_divide_",  "true_divide_out",
      "to",           "_to_copy",      "copy_",
      "floor_divide", "floor_divide_", "floor_divide_out"};
  // 判断给定的方法名是否在白名单内
  return allowed.find(name) != allowed.end();
}

// 构造函数参数对象，根据给定格式字符串fmt和关键字标志keyword_only初始化
FunctionParameter::FunctionParameter(const std::string& fmt, bool keyword_only)
    : optional(false), // 是否可选，默认为false
      allow_none(false), // 是否允许为None，默认为false
      keyword_only(keyword_only), // 是否仅限关键字参数
      size(0), // 初始化参数大小为0
      default_scalar(0) { // 默认标量为0
  auto space = fmt.find(' '); // 查找空格位置，分隔类型和名称
  if (space == std::string::npos) { // 如果找不到空格，抛出异常
    throw std::runtime_error("FunctionParameter(): missing type: " + fmt);
  }

  auto type_str = fmt.substr(0, space); // 提取类型字符串

  auto question = type_str.find('?'); // 查找是否有问号，表示可为None
  if (question != std::string::npos) {
    allow_none = true; // 如果有问号，设置允许为None
    type_str = type_str.substr(0, question); // 截取问号之前的部分作为类型
  }

  // 解析并移除类型字符串中的方括号
  auto bracket = type_str.find('[');
  if (bracket != std::string::npos) {
    auto size_str =
        type_str.substr(bracket + 1, type_str.length() - bracket - 2); // 提取方括号中的大小
    size = atoi(size_str.c_str()); // 将字符串转换为整数，作为参数大小
    type_str = type_str.substr(0, bracket); // 截取方括号之前的部分作为类型
  }

  auto name_str = fmt.substr(space + 1); // 提取空格之后的名称字符串

  auto it = type_map.find(type_str); // 在类型映射中查找类型字符串
  if (it == type_map.end()) { // 如果找不到对应的类型，抛出异常
    throw std::runtime_error(
        "FunctionParameter(): invalid type string: " + type_str);
  }
  type_ = it->second; // 设置类型

  auto eq = name_str.find('='); // 查找是否有等号，表示默认值
  if (eq != std::string::npos) {
    name = name_str.substr(0, eq); // 提取等号之前的部分作为名称
    optional = true; // 设置为可选参数
    set_default_str(name_str.substr(eq + 1)); // 设置默认值字符串
  } else {
    name = name_str; // 没有默认值，则名称为name_str
  }
  python_name = THPUtils_internString(name); // 将名称字符串转换为Python字符串对象

  // 检查是否存在与名称兼容的NumPy兼容参数
  auto np_compat_it = numpy_compatibility_arg_names.find(name);
  if (np_compat_it != numpy_compatibility_arg_names.end()) {
    // 将NumPy兼容参数的Python名称转换为Python字符串对象列表
    for (const auto& str : np_compat_it->second) {
      numpy_python_names.push_back(THPUtils_internString(str));
    }
  }
}

// 处理torch函数获取器的函数，接受THPVariable*类型的self参数
auto handle_torch_function_getter(
    THPVariable* self,
    const std::string& property_name) -> PyObject* {

# 定义一个函数，接受一个常量引用参数 property_name，并返回一个 PyObject 指针。


  py::object torch_api = PyObject_FastGetAttrString(
      THPVariableClass, (char*)property_name.c_str());

# 使用 PyObject_FastGetAttrString 函数从 THPVariableClass 中获取一个属性名为 property_name 的属性，将其封装为 py::object 对象 torch_api。


  std::string module_name = "torch.Tensor." + property_name;

# 创建一个名为 module_name 的字符串，其内容是 "torch.Tensor." 后面跟着 property_name 的值。


  return handle_torch_function(
      (PyObject*)self,
      "__get__",
      nullptr,
      nullptr,
      torch_api.ptr(),
      module_name);

# 调用 handle_torch_function 函数，传递以下参数：
# - (PyObject*)self：当前对象的 PyObject 指针。
# - "__get__"：字符串 "__get__"，表示要调用的函数名。
# - nullptr：空指针，表示无需传递第三个参数。
# - nullptr：空指针，表示无需传递第四个参数。
# - torch_api.ptr()：torch_api 对象的底层 PyObject 指针。
# - module_name：前面构建的 module_name 字符串。
// 自动处理torch_function设置器的函数
auto handle_torch_function_setter(
    THPVariable* self,  // THPVariable类型指针，表示当前对象
    const std::string& property_name,  // 要设置的属性名称
    PyObject* value) -> int {  // 要设置的属性值
  // 获取THPVariableClass中属性名称对应的Python对象
  py::object torch_api = PyObject_FastGetAttrString(
      THPVariableClass, (char*)property_name.c_str());
  // 构造属性的完整模块名称
  std::string module_name = "torch.Tensor." + property_name;
  if (value != nullptr) {
    // 如果值不为空，调用handle_torch_function处理__set__操作
    py::tuple args_ = py::make_tuple(py::handle(value));
    handle_torch_function(
        (PyObject*)self,
        "__set__",
        args_.ptr(),
        nullptr,
        torch_api.ptr(),
        module_name);
  } else {
    // 如果值为空，调用handle_torch_function处理__delete__操作
    handle_torch_function(
        (PyObject*)self,
        "__delete__",
        nullptr,
        nullptr,
        torch_api.ptr(),
        module_name);
  }
  // 返回0，表示成功处理
  return 0;
}

// 将self和args合并为一个元组
static auto combine_self_args(PyObject* self, PyObject* args) -> py::tuple {
  if (args == nullptr) {
    // 如果args为空，返回只包含self的元组
    return py::make_tuple(py::handle(self));
  } else if (self == nullptr) {
    // 如果self为空，将args作为元组返回
    return py::reinterpret_borrow<py::tuple>(args);
  }

  // 否则，args不为空，且self也不为空
  auto py_args = py::reinterpret_borrow<py::tuple>(args);
  size_t n = py_args.size();
  auto args_ = py::tuple(n + 1);  // 创建一个长度为n+1的元组
  args_[0] = py::handle(self);  // 第一个元素是self
  for (const auto i : c10::irange(n)) {
    args_[i + 1] = py_args[i];  // 后续元素是args中的每个元素
  }
  return args_;  // 返回合并后的元组
}

// TODO: 我不确定是否应该称其为__torch_function__还是torch_function。
// 前者使得将现有的类似Tensor的__torch_function__对象转换为模式更容易；
// 但是一般情况下，模式不必类似于Tensor（我们会错误地接受模式对象作为参数，而不应该以这种方式传递它们）。
// 定义torch_function模式名称常量
const char* torch_function_mode_name = "__torch_function__";

// 处理torch_function的函数
auto handle_torch_function(
    PyObject* self,  // 表示当前对象
    const std::string& func_name,  // 函数名
    PyObject* args,  // 参数
    PyObject* kwargs,  // 关键字参数
    PyObject* torch_api,  // torch API对象
    const std::string& module_name) -> PyObject* {
  // 获取torch_api对象的func_name属性
  py::object torch_api_function =
      PyObject_FastGetAttrString(torch_api, (char*)func_name.c_str());
  // 断言torch_api_function不为空，即torch API函数必须存在
  TORCH_INTERNAL_ASSERT(
      torch_api_function.ptr() != nullptr, "torch API function must exist");
  // 合并self和args为一个元组
  py::tuple args_ = combine_self_args(self, args);
  // 调用handle_torch_function_no_python_arg_parser处理torch_function
  return handle_torch_function_no_python_arg_parser(
      {self},
      args_.ptr(),
      kwargs,
      func_name.c_str(),
      torch_api_function.ptr(),
      module_name.c_str(),
      TorchFunctionName::TorchFunction);
}

// 注释: [Overloaded args]
// 过载的参数可能是以下之一：
// - 具有__torch_function__方法的对象实例
// - 具有__torch_dispatch__类方法的对象实例
// - 具有__torch_dispatch__类方法的类类型
//
// 此函数返回参数的类型（如果参数是实例），否则返回参数本身。
static PyObject* get_type_of_overloaded_arg(PyObject* obj_or_type) {
  if (PyType_Check(obj_or_type)) {
    // 如果是类型对象，直接返回
    return obj_or_type;
  }
  // 否则返回对象的类型
  return (PyObject*)Py_TYPE(obj_or_type);
}

// 在子类上分发的函数
static py::object dispatch_on_subclass(
    PyObject* args,
    // 定义一个 Python 对象 ret
    py::object ret;
    // 遍历重载的参数列表
    for (auto& arg : overloaded_args) {
        // 获取参数对象 arg 中名为 torch_function_name_str 的属性 torch_function
        py::object torch_function =
            PyObject_FastGetAttrString(arg, torch_function_name_str);
        // 如果获取失败，断言错误
        if (!torch_function) {
            TORCH_INTERNAL_ASSERT(0);
        }
        // 如果 torch_function 是被禁用的 torch_dispatch 实现，则跳过当前循环
        if (torch_function.ptr() == torch::disabled_torch_dispatch_impl()) {
            // 在 __torch_dispatch__ 过程中，对于带有禁用 torch_dispatch 的参数不进行调度
            // 这段代码运行在基础模式之前，因此确保基础模式可以先运行
            continue;
        }

        // 查看是否为 torch_function，并且具有 __self__ 属性指向当前参数对象
        if (is_torch_function &&
            PyObject_FastGetAttrString(torch_function.ptr(), "__self__")
                .is(py::handle(arg)) &&
            torch_function.ptr() != torch::disabled_torch_function_impl()) {
            // 如果是 torch_function，且 __self__ 属性指向当前参数对象，并且不是被禁用的 torch_function 实现，则发出警告
            TORCH_WARN(
                "Defining your `",
                torch_function_name_str,
                "` as a plain method is deprecated ",
                "and will be an error in future, please define it as a classmethod.");
        }

        // 调用 torch_function，并传递给定的参数
        ret = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
            torch_function.ptr(),
            torch_api_function,
            py_types.ptr(),
            args,
            kwargs,
            NULL));
        // 如果调用返回空指针，则抛出 Python 错误
        if (ret.ptr() == nullptr) {
            throw python_error();
        }
        // 如果返回值不是 Py_NotImplemented，则退出循环
        if (ret.ptr() != Py_NotImplemented) {
            // 返回结果引用
            // 这也包括 ret 为空且 __torch_function__/__torch_dispatch 抛出异常的情况，我们在下面抛出异常
            break;
        }
    }
    // 返回结果对象 ret
    return ret;
}

static std::tuple<py::object, py::object> dispatch_on_mode(
    PyObject* args,                            // 参数 args：传递给函数的位置参数
    PyObject* kwargs,                          // 参数 kwargs：传递给函数的关键字参数
    py::tuple py_types,                        // 参数 py_types：Python 元组，包含类型信息
    PyObject* torch_api_function,              // 参数 torch_api_function：Torch API 函数对象
    bool is_torch_function,                    // 参数 is_torch_function：是否为 Torch 函数的标志
    const char* torch_function_name_str) {     // 参数 torch_function_name_str：Torch 函数名字符串
  // 在内部禁用模式；如果尝试例如打印张量，这将提升用户友好性。
  at::optional<torch::overrides::StashTorchFunctionModeGuard> tf_g;  // 可选类型，保存当前 Torch 函数模式的保护对象
  at::optional<torch_dispatch_mode::StashTorchDispatchModeGuard> td_g; // 可选类型，保存当前 Torch 分发模式的保护对象
  py::object mode_obj;                        // Python 对象，用于保存当前模式的引用
  // 注意：如果函数调用失败需要错误报告，我们实际上只需要保持 mode_obj 存活，但 Python 引用计数是廉价的
  if (is_torch_function) {
    tf_g.emplace();                           // 创建并保存 Torch 函数模式保护对象
    mode_obj = py::reinterpret_borrow<py::object>(
        tf_g->get_cur_mode()->ptr(getPyInterpreter()));  // 获取当前模式对象的 Python 引用
  } else {
    td_g.emplace();                           // 创建并保存 Torch 分发模式保护对象
    mode_obj = py::reinterpret_borrow<py::object>(
        td_g->get_cur_mode()->ptr(getPyInterpreter()));  // 获取当前模式对象的 Python 引用
  }
  py::object torch_function =
      PyObject_FastGetAttrString(mode_obj.ptr(), torch_function_name_str);  // 获取当前模式对象上的 Torch 函数对象
  if (!torch_function) {
    TORCH_INTERNAL_ASSERT(0);                 // 断言：如果获取 Torch 函数对象失败，抛出内部错误
  }
  TORCH_INTERNAL_ASSERT(py_types.ptr() != nullptr);  // 断言：确保 py_types 不为空
  TORCH_INTERNAL_ASSERT(args != nullptr);      // 断言：确保 args 不为空

  TORCH_CHECK(
      PyObject_FastGetAttrString(torch_function.ptr(), "__self__").is(mode_obj),
      "Defining your mode's `",               // 检查：确保 Torch 函数被正确定义为模式的方法而不是类方法
      torch_function_name_str,
      "` as a classmethod is not supported, please make it a plain method");

  // 令人讨厌的事情。下面的 PyObject_CallFunctionObjArgs 中因为 nullptr 终止了参数列表，所以无意中起作用了。
  py::object ret;
  if (kwargs == nullptr) {
    ret = py::reinterpret_steal<py::object>(PyObject_CallMethod(
        mode_obj.ptr(),
        torch_function_name_str,
        "OOO",
        torch_api_function,
        py_types.ptr(),
        args));                                // 调用模式对象上的 Torch 函数，传递位置参数和类型信息
  } else {
    ret = py::reinterpret_steal<py::object>(PyObject_CallMethod(
        mode_obj.ptr(),
        torch_function_name_str,
        "OOOO",
        torch_api_function,
        py_types.ptr(),
        args,
        kwargs));                              // 调用模式对象上的 Torch 函数，传递位置参数、类型信息和关键字参数
  }
  if (ret.ptr() == nullptr) {
    throw python_error();                     // 如果返回值为空，抛出 Python 错误
  }
  return std::make_tuple(ret, mode_obj);       // 返回调用结果和模式对象的元组
}

// See Note: [Overloaded args] for what they hold
auto handle_torch_function_no_python_arg_parser(
    at::ArrayRef<PyObject*> overloaded_args,    // 参数 overloaded_args：传递给函数的重载参数的引用
    PyObject* args,                            // 参数 args：传递给函数的位置参数
    PyObject* kwargs,                          // 参数 kwargs：传递给函数的关键字参数
    const char* func_name,                     // 参数 func_name：函数名字符串
    PyObject* torch_api_function,              // 参数 torch_api_function：Torch API 函数对象
    const char* module_name,                   // 参数 module_name：模块名字符串
    TorchFunctionName torch_function_name) -> PyObject* {  // 参数 torch_function_name：Torch 函数名称枚举
  const char* torch_function_name_str = nullptr;  // Torch 函数名字符串指针初始化为空
  switch (torch_function_name) {
    case TorchFunctionName::TorchFunction:
      torch_function_name_str = "__torch_function__";  // 设置为 Torch 函数名字符串
      break;
    case TorchFunctionName::TorchDispatch:
      torch_function_name_str = "__torch_dispatch__";  // 设置为 Torch 分发名字符串
      break;
    default:
      // 如果没有匹配的模式，使用 TORCH_INTERNAL_ASSERT 抛出错误
      TORCH_INTERNAL_ASSERT(0, static_cast<int>(torch_function_name));
  }
  // overloaded_args 已经全部具有唯一的类型
  // 注意：模式不会放在重载类型列表中，因为它们并不是类型
  std::vector<py::object> overloaded_types;
  overloaded_types.reserve(overloaded_args.size());
  for (auto& arg : overloaded_args) {
    // 将每个重载参数的类型转换为 Python 对象，并加入到 overloaded_types 中
    overloaded_types.push_back(
        py::reinterpret_borrow<py::object>(get_type_of_overloaded_arg(arg)));
  }
  // 将 overloaded_types 转换为 Python 元组
  py::tuple py_types = py::cast(overloaded_types);
  py::object ret;
  py::object mode_obj;

  // Step 1: Try to dispatch based on the mode stack, *ignoring* infra
  // torch_dispatch modes.
  const bool is_torch_function =
      // 检查是否为 TorchFunction 模式
      torch_function_name == TorchFunctionName::TorchFunction;
  const auto is_mode_active = [&]() {
    return is_torch_function
        // 如果是 TorchFunction 模式，则检查是否启用 torch_function_mode
        ? at::impl::torch_function_mode_enabled()
        // 否则检查是否启用 dispatch_mode
        : c10::impl::dispatch_mode_enabled();
  };
  // Note [__torch_dispatch__ dispatching order]
  // 下面的分发顺序的主要思想是：
  // (1) 模式优先于子类
  // (2) “用户”模式/子类优先于“基础设施”模式/子类

  // 例如，当 mode_stack 包含 ModeA 和一些基础设施模式时，根据这些模式的优先级进行分发

  if (is_mode_active()) {
    // Step 1: 尝试根据用户定义的 TorchDispatchModes 进行调度（包括基础设施模式，始终位于模式堆栈底部）。
    auto ret_ = dispatch_on_mode(
        args,                               // 函数参数
        kwargs,                             // 关键字参数
        py_types,                           // Python 类型列表
        torch_api_function,                 // Torch API 函数对象
        is_torch_function,                  // 是否为 torch function
        torch_function_name_str);           // Torch 函数名称字符串
    ret = std::get<0>(ret_);                // 调度结果
    mode_obj = std::get<1>(ret_);           // 模式对象

  }

  // Step 2: 尝试根据用户子类进行调度，忽略带有 _mode_key 字段的基础设施子类
  // 注意：用户子类应始终在基础设施模式之前运行，如代理/虚拟模式。这是通过在遇到未知的用户子类时，
  // 使代理/虚拟模式返回 NotImplemented 来处理的。
  if (ret.ptr() == nullptr || ret.ptr() == Py_NotImplemented) {
    auto curr_ret = dispatch_on_subclass(
        args,                               // 函数参数
        kwargs,                             // 关键字参数
        overloaded_args,                    // 过载的参数
        py_types,                           // Python 类型列表
        torch_api_function,                 // Torch API 函数对象
        is_torch_function,                  // 是否为 torch function
        torch_function_name_str);           // Torch 函数名称字符串
    if (curr_ret.ptr() != nullptr) {
      ret = curr_ret;                       // 更新返回值
    }
  }

  if (ret.ptr() == nullptr) {
    // 如果在用户的 __torch_function__ 实现中发生异常，则抛出异常
    throw python_error();
  } else if (ret.ptr() == Py_NotImplemented) {
    // 所有 overloaded_args 中的 __torch_function__ 实现均返回 NotImplemented，
    // 因此我们抛出一个 TypeError。
    std::stringstream ss;
    ss << "Multiple dispatch failed for '";
    if (module_name && func_name) {
      ss << module_name << "." << func_name;
    } else {
      py::handle fn = torch_api_function;
      ss << py::str(fn.attr("__module__")) << "."
         << py::str(fn.attr("__name__"));
    }
    ss << "'; all " << torch_function_name_str
       << " handlers returned NotImplemented:\n\n";
    if (mode_obj) {
      ss << "  - mode object " << py::repr(mode_obj) << "\n";
    }
    for (auto& arg : overloaded_args) {
      ss << "  - tensor subclass " << py::repr(get_type_of_overloaded_arg(arg))
         << "\n";
    }
    ss << "\nFor more information, try re-running with TORCH_LOGS=not_implemented";
    const std::string& tmp = ss.str();
    PyErr_SetString(PyExc_TypeError, tmp.c_str());    // 设置异常信息
    throw python_error();                            // 抛出异常
  }
  return ret.release().ptr();  // 返回处理后的结果指针
auto handle_torch_function(
    PythonArgs& r,                    // PythonArgs 类的引用 r，用于封装 Python 函数调用的参数和状态
    PyObject* self,                   // 指向调用对象的 PyObject 指针
    PyObject* args,                   // Python 函数调用的位置参数
    PyObject* kwargs,                 // Python 函数调用的关键字参数
    PyObject* torch_api,              // 指向 Torch API 对象的 PyObject 指针
    const char* module_name,          // 模块名称字符串
    const char* func_name_override    // 函数名覆盖字符串（可选）
) -> PyObject* {
  py::object torch_api_function = PyObject_FastGetAttrString(
      torch_api,
      (char*)(func_name_override ? func_name_override
                                 : r.get_func_name().c_str()));  // 获取 Torch API 中的函数对象
  TORCH_INTERNAL_ASSERT(
      torch_api_function.ptr() != nullptr, "torch API function must exist");  // 断言 Torch API 函数对象不为空
  py::tuple args_ = combine_self_args(self, args);  // 将 self 和 args 合并为一个 tuple
  return handle_torch_function_no_python_arg_parser(
      r.overloaded_args,             // PythonArgs 中的重载参数列表
      args_.ptr(),                   // 组合后的参数元组
      kwargs,                        // 关键字参数
      r.get_func_name().c_str(),     // 获取函数名
      torch_api_function.ptr(),      // Torch API 函数对象指针
      module_name                    // 模块名称
  );
}

auto handle_torch_function(
    PythonArgs& r,                    // PythonArgs 类的引用 r，用于封装 Python 函数调用的参数和状态
    PyObject* args,                   // Python 函数调用的位置参数
    PyObject* kwargs,                 // Python 函数调用的关键字参数
    PyObject* torch_api,              // 指向 Torch API 对象的 PyObject 指针
    const char* module_name,          // 模块名称字符串
    const char* func_name_override    // 函数名覆盖字符串（可选）
) -> PyObject* {
  return handle_torch_function(
      r, nullptr, args, kwargs, torch_api, module_name, func_name_override);  // 调用上一个函数的重载，传递 nullptr 作为 self
}

auto handle_torch_function_indexing(
    PyObject* self,                   // 指向调用对象的 PyObject 指针
    PyObject* index,                  // 索引参数
    PyObject* val                     // 值参数（可选）
) -> PyObject* {
  const char* func_name = (val == nullptr) ? "__getitem__" : "__setitem__";  // 确定操作类型的函数名字符串
  py::object index_tup;
  if (PyTuple_Check(index)) {
    index_tup = py::reinterpret_borrow<py::object>(index);  // 如果索引是元组，则转换为 Python 元组对象
  } else {
    index_tup = py::make_tuple(py::handle(index));           // 否则，创建包含 index 的 Python 元组对象
  }
  std::vector<PyObject*> overridable_args;  // 存储可重载参数的 PyObject 指针向量
  is_tensor_and_append_overloaded(self, &overridable_args);  // 检查并将 self 添加到可重载参数列表中
  auto size = PyTuple_GET_SIZE(index_tup.ptr());  // 获取索引元组的大小
  for (auto i : c10::irange(size)) {
    auto* obj = PyTuple_GetItem(index_tup.ptr(), i);  // 遍历索引元组中的每个项
    is_tensor_and_append_overloaded(obj, &overridable_args);  // 检查并将每个项添加到可重载参数列表中
  }
  if (val != nullptr) {
    is_tensor_and_append_overloaded(val, &overridable_args);  // 如果存在值参数，则将其添加到可重载参数列表中
  }
  py::object func =
      PyObject_FastGetAttrString(THPVariableClass, (char*)func_name);  // 获取 Torch 变量类的指定函数对象
  py::object args = (val == nullptr)
      ? py::make_tuple(py::handle(self), py::handle(index))  // 创建调用参数的元组（包含 self 和 index）
      : py::make_tuple(py::handle(self), py::handle(index), py::handle(val));  // 创建调用参数的元组（包含 self、index 和 val）
  return handle_torch_function_no_python_arg_parser(
      overridable_args,             // 可重载参数列表
      args.ptr(),                   // 调用参数的元组对象指针
      nullptr,                      // 不包含关键字参数
      func_name,                    // 函数名字符串
      func.ptr(),                   // Torch 变量类的指定函数对象指针
      "torch.Tensor"                // 模块名称字符串
  );
}
/*
 *  obj has a __torch_function__ implementation and may either be a
 *  subclass of Tensor or a Tensor-like duck type. We may need to
 *  append this object to the overloaded_args vector, which tracks all
 *  of the arguments with distinct __torch_function__ implementations
 *  we've seen so far.
 *
 *  If this is the first argument we've seen with __torch_function__
 *  defined, we unconditionally add obj to the overloaded_args vector.
 *
 *  If we've already seen arguments with __torch_function__ defined,
 *  then we first need to check if obj is the same type as any of the
 *  entries in overloaded_args.  If so, we can ignore obj since we
 *  already have an entry in overloaded_args with the same
 *  __torch_function__ implementation.
 *
 *  If it's a different type, we then need to check if it's a subclass
 *  of one of the types we've already seen. If so, we need to insert an
 *  entry in overloaded_args for this type with higher precedence than
 *  the superclass.
 *
 *  See torch._overrides._get_overloaded_args for the equivalent
 *  function in the Python __torch_function__ implementation.
 *
 *  The precedence-determining algorithm implemented in this function is
 *  described in NEP-0018:
 *  https://numpy.org/neps/nep-0018-array-function-protocol.html
 *
 *  'overloaded_args' is a raw pointer to a vector of pybind11 handles
 *  that have distinct __torch_function__ implementations, in order of calling
 *  precedence.
 *
 *  'obj' is an object to check for a __torch_function__ implementation
 *
 * If changing this file in a way that can affect the __torch_function__
 * overhead, please report the benchmarks in 'benchmarks/overrides_benchmark'.
 * See the instructions in the 'README.md' in that directory.
 *
 */

static void append_overloaded_arg(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj,
    bool obj_is_type) {
  // Initialize a flag to track if the object's type hasn't been seen before
  bool class_not_seen_yet = true;
  
  // Determine the type of the object to be added to overloaded_args
  PyObject* obj_type = obj_is_type ? obj : (PyObject*)Py_TYPE(obj);
  
  // Iterate over existing entries in overloaded_args to check for type matches
  for (auto& arg : *overloaded_args) {
    // Check if the type of obj matches the type of the current argument in overloaded_args
    if (obj_type == get_type_of_overloaded_arg(arg)) {
      // Skip adding obj to overloaded_args because its type has already been recorded
      class_not_seen_yet = false;
      break;
    }
  }
  
  // If obj's type hasn't been recorded yet, determine its insertion index
  if (class_not_seen_yet) {
    auto arg_index = overloaded_args->size();
    // Check if obj is a subclass of any existing types in overloaded_args
    for (const auto j : c10::irange(arg_index)) {
      if (PyObject_IsSubclass(
              obj_type, get_type_of_overloaded_arg((*overloaded_args)[j]))) {
        // Insert obj before the superclass in overloaded_args
        arg_index = j;
        break;
      }
    }
    // Add obj to overloaded_args at the determined index
    // If it's a subclass of another class, it should have higher precedence
    // 如果已经发现将要插入的位置是在超类之前，
    // 那么将在超类之前插入对象
    // 否则，将在数组末尾插入对象
    overloaded_args->insert(
        overloaded_args->begin() + static_cast<long>(arg_index), obj);
}

void append_overloaded_tensor(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj) {
  // 调用通用函数，将 obj 加入到重载参数列表中，标志为非类型对象
  append_overloaded_arg(overloaded_args, obj, /*obj_is_type*/ false);
}

void append_overloaded_type(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj) {
  // 调用通用函数，将 obj 加入到重载参数列表中，标志为类型对象
  append_overloaded_arg(overloaded_args, obj, /*obj_is_type*/ true);
}

bool is_tensor_and_append_overloaded(
    PyObject* obj,
    std::vector<PyObject*>* overloaded_args) {
  if (THPVariable_CheckExact(obj)) {
    // 检查是否为 torch.Tensor 的实例（非子类，除了 Parameter）
    return true;
  }

  if (check_has_torch_function(obj, /*ignore_mode*/ true)) {
    // 检查是否有 __torch_function__ 方法，可能是 tensor 子类或者其他支持的对象
    append_overloaded_tensor(overloaded_args, obj);
    return true;
  } else if (THPVariable_Check(obj)) {
    // 检查是否为 tensor 子类（没有 __torch_function__ 方法）
    return true;
  }

  return false;
}

static bool is_scalar_list(PyObject* obj) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return false;
  }
  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  for (const auto idx : c10::irange(size)) {
    PyObject* iobj =
        tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (!THPUtils_checkScalar(iobj)) {
      return false;
    }
  }
  return true;
}

bool is_tensor_list_and_append_overloaded(
    PyObject* obj,
    std::vector<PyObject*>* overloaded_args,
    int argnum,
    bool throw_error) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return false;
  }
  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  for (long idx = 0; idx < size; idx++) {
    PyObject* iobj =
        tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (!is_tensor_and_append_overloaded(iobj, overloaded_args)) {
      if (throw_error) {
        TORCH_CHECK_TYPE(
            false,
            "expected Tensor as element ",
            idx,
            " in argument ",
            argnum,
            ", but got ",
            Py_TYPE(iobj)->tp_name);
      }
      return false;
    }
  }
  return true;
}

static bool is_float_or_complex_list(PyObject* obj) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  if (size > 0) {
    PyObject* iobj = tuple ? PyTuple_GET_ITEM(obj, 0) : PyList_GET_ITEM(obj, 0);
    if (!THPUtils_checkDouble(iobj) && !PyComplex_Check(iobj)) {
      return false;
    }
  }

  return true;
}
static bool is_int_or_symint(PyObject* obj) {
    // THPUtils_checkIndex 可能调用 __index__ 或 __int__
    // 如果 obj 是 symint 节点，则可能有副作用
    // 因此我们首先检查是否为 symint
    // TODO: 或许我们应该在这里使用 checkLong？
    if (torch::is_symint(py::handle(obj))) {
        return true;
    }

    // FakeTensor(..., size=()) 适合 SymInt 参数，
    // 但我们不能像对待常规张量那样通过 __index__（下文）处理，
    // 因为 __index__ 首先强制将其转换为 int，
    // 而对于未支持的 SymInt，这通常是不可能的。
    // 因此，这个快速通道确保我们在这种情况下仍允许虚张量，
    // 但对于常规张量，它与下面的测试是多余的。
    if (THPVariable_Check(obj)) {
        auto& var = THPVariable_Unpack(obj);
        if (TORCH_GUARD_SIZE_OBLIVIOUS(var.sym_numel().sym_eq(1)) &&
            at::isIntegralType(var.dtype().toScalarType(), /*include_bool*/ true)) {
            return true;
        }
    }

    // 如果 THPUtils_checkIndex 返回 true，则返回 true
    if (THPUtils_checkIndex(obj)) {
        return true;
    }

    // 否则返回 false
    return false;
}

static bool is_int_or_symint_list(
    PyObject* obj,
    int broadcast_size,
    int64_t* failed_idx = nullptr) {
    if (PyTuple_Check(obj) || PyList_Check(obj)) {
        // 如果是元组或列表且长度为 0，则返回 true
        if (PySequence_Size(obj) == 0) {
            return true;
        }
        // 获取序列中的第一个项目
        auto item = py::reinterpret_steal<py::object>(PySequence_GetItem(obj, 0));

        // 如果第一个项目是 int 或 symint，则返回 true
        if (is_int_or_symint(item.ptr())) {
            return true;
        }

        // 注意：JIT 追踪器允许任意标量张量充当 intlist 参数中的 int。
        // 即使是浮点数或复数标量张量也可以。
        bool r =
            (jit::tracer::isTracing() && THPVariable_Check(item.ptr()) &&
             THPVariable_Unpack(item.ptr()).sizes().empty());
        // 如果 r 是 false 且 failed_idx 不为 nullptr，则将 *failed_idx 设置为 0
        if (!r && failed_idx != nullptr) {
            *failed_idx = 0;
        }
        return r;
    }

    // 如果指定了大小（例如 IntArrayRef[2]），则允许传递单个 int
    return broadcast_size > 0 && is_int_or_symint(obj);
}

// argnum 用于在引发 TypeError 时提供上下文信息，它用于错误消息。
auto FunctionParameter::check(
    PyObject* obj,
    std::vector<PyObject*>& overloaded_args,
    int argnum,
    int64_t* failed_idx) -> bool {
    switch (type_) {
        case ParameterType::TENSOR: {
            // 如果是 tensor 并添加到 overloaded_args，则返回 true
            if (is_tensor_and_append_overloaded(obj, &overloaded_args)) {
                return true;
            }
            // 如果允许将数字作为 tensor，则返回 THPUtils_checkScalar(obj) 的结果
            if (allow_numbers_as_tensors) {
                return THPUtils_checkScalar(obj);
            }
            // 否则返回 false
            return false;
        }
        case ParameterType::SCALAR:
            // 如果是标量，则返回 THPUtils_checkScalar(obj) 的结果
            if (THPUtils_checkScalar(obj)) {
                return true;
            }
            [[fallthrough]];
        case ParameterType::COMPLEX:
            // 如果是复数，则返回 PyComplex_Check(obj) 的结果
            if (PyComplex_Check(obj)) {
                return true;
            }
            [[fallthrough]];
    // 当参数类型为 DOUBLE 时的处理逻辑
    case ParameterType::DOUBLE: {
      // 检查 obj 是否为 double 类型，如果是则返回 true
      if (THPUtils_checkDouble(obj)) {
        return true;
      }
      // 如果 obj 是 torch.Tensor 对象
      if (THPVariable_Check(obj)) {
        // 将 obj 解包为 torch::autograd::Variable 引用 var
        const auto& var = THPVariable_Unpack(obj);
        // 返回条件：不要求梯度且维度为 0 的标量张量
        return !var.requires_grad() && var.dim() == 0;
      }
      // 如果 obj 是 torch 的符号浮点数或符号整数
      if (torch::is_symfloat(py::handle(obj)) ||
          torch::is_symint(py::handle(obj))) {
        // 这会引发一个保护
        return true;
      }
      // 其他情况返回 false
      return false;
    }

    // 当参数类型为 INT64 时的处理逻辑
    case ParameterType::INT64: {
      // 检查 obj 是否为 long 类型，如果是则返回 true
      if (THPUtils_checkLong(obj)) {
        return true;
      }
      // 如果 obj 是 torch.Tensor 对象
      if (THPVariable_Check(obj)) {
        // 将 obj 解包为 torch::autograd::Variable 引用 var
        const auto& var = THPVariable_Unpack(obj);
        // 返回条件：张量的标量类型为整数（不包括 bool）、不要求梯度且维度为 0
        return at::isIntegralType(var.scalar_type(), /*includeBool=*/false) &&
            !var.requires_grad() && var.dim() == 0;
      }
      // 如果 obj 是 torch 的符号整数
      if (torch::is_symint(py::handle(obj))) {
        // 这会引发一个保护
        return true;
      }
      // 其他情况返回 false
      return false;
    }

    // 当参数类型为 DIMNAME 时，调用 THPUtils_checkDimname 函数检查 obj
    case ParameterType::DIMNAME:
      return THPUtils_checkDimname(obj);

    // 当参数类型为 DIMNAME_LIST 时的处理逻辑
    case ParameterType::DIMNAME_LIST: {
      // 检查 obj 是否为 DimnameList 类型，如果是则返回 true
      if (THPUtils_checkDimnameList(obj)) {
        return true;
      }
      // 如果 size 为 1 并且 obj 是单个 Dimname，则也允许通过
      return size == 1 && THPUtils_checkDimname(obj);
    }

    // 当参数类型为 TENSOR_LIST 时，调用 is_tensor_list_and_append_overloaded 函数检查 obj
    case ParameterType::TENSOR_LIST: {
      return is_tensor_list_and_append_overloaded(
          obj, &overloaded_args, argnum, true /* throw_error */);
    }

    // 当参数类型为 FLOAT_LIST 时，调用 is_float_or_complex_list 函数检查 obj
    case ParameterType::FLOAT_LIST:
      return is_float_or_complex_list(obj);

    // 当参数类型为 GENERATOR 时，调用 THPGenerator_Check 函数检查 obj
    case ParameterType::GENERATOR:
      return THPGenerator_Check(obj);

    // 当参数类型为 BOOL 时，调用 PyBool_Check 函数检查 obj
    case ParameterType::BOOL:
      return PyBool_Check(obj);

    // 当参数类型为 STORAGE 时，调用 isStorage 函数检查 obj
    case ParameterType::STORAGE:
      return isStorage(obj);

    // 当参数类型为 PYOBJECT 时，直接返回 true
    case ParameterType::PYOBJECT:
      return true;

    // 当参数类型为 SCALARTYPE 时，调用 THPDtype_Check 或 THPPythonScalarType_Check 函数检查 obj
    case ParameterType::SCALARTYPE:
      return THPDtype_Check(obj) || THPPythonScalarType_Check(obj);

    // 当参数类型为 LAYOUT 时，调用 THPLayout_Check 函数检查 obj
    case ParameterType::LAYOUT:
      return THPLayout_Check(obj);

    // 当参数类型为 MEMORY_FORMAT 时，调用 THPMemoryFormat_Check 函数检查 obj
    case ParameterType::MEMORY_FORMAT:
      return THPMemoryFormat_Check(obj);

    // 当参数类型为 QSCHEME 时，调用 THPQScheme_Check 函数检查 obj
    case ParameterType::QSCHEME:
      return THPQScheme_Check(obj);

    // 当参数类型为 DEVICE 时，检查 obj 是否为 long、string 或 torch.device 类型
    case ParameterType::DEVICE:
      return THPUtils_checkLong(obj) || THPUtils_checkString(obj) ||
          THPDevice_Check(obj);

    // 当参数类型为 STREAM 时，调用 THPStream_Check 函数检查 obj
    case ParameterType::STREAM:
      return THPStream_Check(obj);

    // 当参数类型为 STRING 时，调用 THPUtils_checkString 函数检查 obj
    case ParameterType::STRING:
      return THPUtils_checkString(obj);

    // 当参数类型为 SCALAR_LIST 时，调用 is_scalar_list 函数检查 obj
    case ParameterType::SCALAR_LIST:
      return is_scalar_list(obj);

    // 当参数类型为 SYM_INT 时，调用 is_int_or_symint 函数检查 obj
    case ParameterType::SYM_INT:
      return is_int_or_symint(obj);

    // 当参数类型为 INT_LIST 或 SYM_INT_LIST 时，调用 is_int_or_symint_list 函数检查 obj
    // 如果 size 和 failed_idx 满足条件则返回 true
    case ParameterType::INT_LIST:
    case ParameterType::SYM_INT_LIST:
      return is_int_or_symint_list(obj, size, failed_idx);

    // 当参数类型为 DISPATCH_KEY_SET 时，检查 obj 是否为 c10::DispatchKeySet 类型
    case ParameterType::DISPATCH_KEY_SET:
      return py::isinstance<c10::DispatchKeySet>(py::handle(obj));

    // 默认情况下，抛出运行时错误，表示未知的参数类型
    default:
      throw std::runtime_error("unknown parameter type");
  }
}

// WARNING: these strings are parsed invalid_arguments.cpp
// 返回参数类型的字符串表示
std::string FunctionParameter::type_name() const {
  switch (type_) {
    case ParameterType::TENSOR:
      return "Tensor";
    case ParameterType::SCALAR:
      return "Number";
    case ParameterType::INT64:
    // 注意: SymInt 没有在这里列出，因为一般用户只知道 int 类型
    case ParameterType::SYM_INT:
      return "int";
    case ParameterType::DOUBLE:
      return "float";
    case ParameterType::COMPLEX:
      return "complex";
    case ParameterType::TENSOR_LIST:
      return "tuple of Tensors";
    case ParameterType::INT_LIST:
      return "tuple of ints";
    case ParameterType::FLOAT_LIST:
      return "tuple of floats";
    case ParameterType::GENERATOR:
      return "torch.Generator";
    case ParameterType::BOOL:
      return "bool";
    case ParameterType::STORAGE:
      return "torch.Storage";
    case ParameterType::PYOBJECT:
      return "object";
    case ParameterType::SCALARTYPE:
      return "torch.dtype";
    case ParameterType::LAYOUT:
      return "torch.layout";
    case ParameterType::MEMORY_FORMAT:
      return "torch.memory_format";
    case ParameterType::QSCHEME:
      return "torch.qscheme";
    case ParameterType::DEVICE:
      return "torch.device";
    case ParameterType::STRING:
      return "str";
    case ParameterType::DIMNAME:
      return "name";
    case ParameterType::DIMNAME_LIST:
      return "tuple of names";
    case ParameterType::SCALAR_LIST:
      return "tuple of Scalars";
    case ParameterType::SYM_INT_LIST:
      return "tuple of ints";
    case ParameterType::DISPATCH_KEY_SET:
      return "DispatchKeySet";
    default:
      throw std::runtime_error("unknown parameter type");
  }
}

static inline std::optional<int64_t> parse_as_integer(const std::string& s) {
  if (s.empty())
    return c10::nullopt;
  char* str_end = nullptr;
  // 将字符串解析为整数
  long ans = strtol(s.c_str(), &str_end, 0);
  // 如果整个字符串都被解析为整数，则返回解析的整数值
  // *str_end == 0 表示整个字符串都被解析
  return (*str_end == 0) ? std::optional<int64_t>(ans) : c10::nullopt;
}

/*
Parse default value of IntArrayRef declared at native_functions.yaml

There are two kinds of default values:
1. IntArrayRef[2] x=1 (where size=2, value={1,1}
2. IntArrayRef x={1,2,3} (where size=3, value={1,2,3}, note that there cannot be
space after comma since native_parse.py uses ', ' to split args)
*/
// 解析在 native_functions.yaml 中声明的 IntArrayRef 的默认值
// 有两种默认值的情况：
// 1. IntArrayRef[2] x=1 (大小为2，值为{1,1})
// 2. IntArrayRef x={1,2,3} (大小为3，值为{1,2,3}，注意逗号后不能有空格，因为 native_parse.py 使用 ', ' 分割参数)
static inline std::vector<int64_t> parse_intlist_args(
    const std::string& s,
    int64_t size) {
  size_t n = s.size();

  if (s.empty())
    return std::vector<int64_t>();

  // 如果 s 是一个整数 (例如，s=2)
  if (s[0] != '{') {
    TORCH_CHECK(size > 0, "Incorrect size of IntArrayRef: ", size);
    // 将字符串 s 转换为包含 size 个 int64_t 元素的 vector，并返回
    return std::vector<int64_t>(size, std::stol(s));
  }

  // case 2. s is a list of dims (e.g., s={1,2})

  // 因为在上面已经检查了左大括号 '{'，这里只需要检查右大括号 '}'
  TORCH_CHECK(
      s[n - 1] == '}',
      "Default value of IntArrayRef is missing right brace '}', found ",
      s[n - 1]);

  auto args = std::vector<int64_t>();  // 创建一个空的 int64_t 类型的 vector
  std::istringstream ss(s.substr(1, s.length() - 2)); // 截取字符串 s，去除开头的 '{' 和末尾的 '}'
  std::string tok;  // 用于存储每次从 ss 中读取的 token

  // 使用 ',' 作为分隔符，从 ss 中逐行读取 token，并将其转换为 int64_t 后加入 args
  while (std::getline(ss, tok, ',')) {
    args.emplace_back(std::stol(tok));
  }
  return args;  // 返回存储了从字符串 s 中解析得到的 int64_t 元素的 vector
// } 这是一个 C++ 函数 set_default_str 的结束大括号

// 将输入的字符串字面值解析为实际字符串，去除引号和转义序列
static std::string parse_string_literal(c10::string_view str) {
  // 检查字符串长度至少为2，确保字符串有引号包裹
  TORCH_CHECK(str.length() >= 2, "String defaults must be quoted");

  // 如果字符串以双引号开头
  if (str.front() == '"') {
    // 检查字符串以双引号结尾，确保引号匹配
    TORCH_CHECK(
        str.back() == '"', "Mismatched quotes in string default: ", str);
  } else {
    // 否则，字符串应以单引号开头和结尾
    TORCH_CHECK(
        str.front() == '\'' && str.back() == '\'',
        "Invalid quotes in string default: ",
        str)
  }

  // 创建一个用于存储解析后字符串的变量
  std::string parsed;
  parsed.reserve(str.size());
  // 遍历字符串中的每个字符（去除首尾引号后的部分）
  for (size_t i = 1; i < str.size() - 1;) {
    // 如果当前字符不是反斜杠，则直接加入解析后的字符串中
    if (str[i] != '\\') {
      parsed.push_back(str[i]);
      ++i;
      continue;
    }

    // 处理转义序列
    TORCH_CHECK(
        i < str.size() - 2, "String ends with escaped final quote: ", str)
    char c = str[i + 1];
    switch (c) {
      case '\\':
      case '\'':
      case '\"':
        break;
      case 'a':
        c = '\a';
        break;
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'n':
        c = '\n';
        break;
      case 'v':
        c = '\v';
        break;
      case 't':
        c = '\t';
        break;
      default:
        // 抛出异常，不支持的转义序列
        TORCH_CHECK(
            false,
            "Unsupported escape sequence in string default: \\",
            str[i + 1]);
    }
    // 将处理后的字符加入解析后的字符串中
    parsed.push_back(c);
    i += 2;
  }
  return parsed;
}

// 设置函数参数的默认字符串值
void FunctionParameter::set_default_str(const std::string& str) {
  // 如果输入的字符串是 "None"，则允许默认值为 None
  if (str == "None") {
    allow_none = true;
  }

  // 如果参数类型为 TENSOR 或 DISPATCH_KEY_SET，则默认值必须为 None
  if (type_ == ParameterType::TENSOR ||
      type_ == ParameterType::DISPATCH_KEY_SET) {
    if (str != "None") {
      // 抛出异常，张量类型的默认值必须是 None
      throw std::runtime_error(
          "default value for Tensor must be none, got: " + str);
    }
  } else if (type_ == ParameterType::INT64 || type_ == ParameterType::SYM_INT) {
    // 如果参数类型为 INT64 或 SYM_INT，则将字符串转换为长整型
    default_int = atol(str.c_str());
  } else if (type_ == ParameterType::BOOL) {
    // 如果参数类型为 BOOL，则判断字符串是否为 "True" 或 "true"
    default_bool = (str == "True" || str == "true");
  } else if (type_ == ParameterType::DOUBLE) {
    // 如果参数类型为 DOUBLE，则将字符串转换为双精度浮点数
    default_double = atof(str.c_str());
  } else if (type_ == ParameterType::COMPLEX) {
    // 如果参数类型为 COMPLEX，则尝试解析字符串为复数
    default_complex[0] = atof(str.c_str()); // TODO: parse "x + xj"?
    default_complex[1] = 0;
  } else if (type_ == ParameterType::SCALAR) {
    // 如果参数类型为 SCALAR，则根据字符串的内容确定数值类型
    if (str != "None") {
      // 尝试将字符串解析为整数或浮点数，作为标量的默认值
      const auto as_integer = parse_as_integer(str);
      default_scalar = as_integer.has_value() ? at::Scalar(as_integer.value())
                                              : at::Scalar(atof(str.c_str()));
    }
  } else if (
      type_ == ParameterType::INT_LIST ||
      type_ == ParameterType::SYM_INT_LIST) {
    // 如果参数类型为 INT_LIST 或 SYM_INT_LIST，则将字符串解析为整数列表
    if (str != "None") {
      default_intlist = parse_intlist_args(str, size);
    }
  } else if (type_ == ParameterType::FLOAT_LIST) {
    // 如果参数类型为 FLOAT_LIST，则抛出异常，不支持 float 列表的默认值
    if (str != "None") {
      throw std::runtime_error("Defaults not supported for float[]");
    }
  } else if (type_ == ParameterType::SCALARTYPE) {
    if (str == "None") {
      // 如果字符串为 "None"，则将默认标量类型设为未定义
      default_scalartype = at::ScalarType::Undefined;
    } else if (str == "torch.int64") {
      // 如果字符串为 "torch.int64"，则将默认标量类型设为长整型
      default_scalartype = at::ScalarType::Long;
    } else {
      // 如果字符串既不是 "None" 也不是 "torch.int64"，抛出运行时错误
      throw std::runtime_error("invalid default value for ScalarType: " + str);
    }
  } else if (type_ == ParameterType::LAYOUT) {
    if (str == "None") {
      // 如果字符串为 "None"，在调试模式下断言允许为空
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(allow_none);
    } else if (str == "torch.strided") {
      // 如果字符串为 "torch.strided"，则将默认布局设为步进布局
      default_layout = at::Layout::Strided;
    } else if (str == "torch.sparse_coo") {
      // 如果字符串为 "torch.sparse_coo"，则将默认布局设为稀疏 COO 布局
      default_layout = at::Layout::Sparse;
    } else {
      // 如果字符串既不是 "None" 也不是已知布局选项，抛出运行时错误
      throw std::runtime_error("invalid default value for layout: " + str);
    }
  } else if (type_ == ParameterType::DEVICE) {
    if (str != "None") {
      // 如果字符串不是 "None"，抛出运行时错误，表示设备无效
      throw std::runtime_error("invalid device: " + str);
    }
  } else if (type_ == ParameterType::STREAM) {
    if (str != "None") {
      // 如果字符串不是 "None"，抛出运行时错误，表示流无效
      throw std::runtime_error("invalid stream: " + str);
    }
  } else if (type_ == ParameterType::STRING) {
    if (str != "None") {
      // 如果字符串不是 "None"，解析字符串字面量并将其作为默认字符串值
      default_string = parse_string_literal(str);
    }
  }
  // 以下这些类型之前未在此处处理过。添加默认错误会导致大量测试失败，所以现在跳过它们。
  // 不过，我们应该正确处理它们，因为可能会导致静默失败。
  else if (type_ == ParameterType::TENSOR_LIST) { // NOLINT
    // throw std::runtime_error("Invalid Tensor List");
  } else if (type_ == ParameterType::GENERATOR) { // NOLINT
    // throw std::runtime_error("ParameterType::GENERATOR");
  } else if (type_ == ParameterType::PYOBJECT) { // NOLINT
    // throw std::runtime_error("ParameterType::PYOBJECT");
  } else if (type_ == ParameterType::MEMORY_FORMAT) { // NOLINT
    // throw std::runtime_error("ParameterType::MEMORY_FORMAT");
  } else if (type_ == ParameterType::DIMNAME) { // NOLINT
    // throw std::runtime_error("ParameterType::DIMNAME");
  } else if (type_ == ParameterType::DIMNAME_LIST) { // NOLINT
    // throw std::runtime_error("ParameterType::DIMNAME_LIST");
  } else if (type_ == ParameterType::SCALAR_LIST) { // NOLINT
    // throw std::runtime_error("ParameterType::SCALAR_LIST");
  } else if (type_ == ParameterType::STORAGE) { // NOLINT
    // throw std::runtime_error("ParameterType::STORAGE");
  } else if (type_ == ParameterType::QSCHEME) { // NOLINT
    // throw std::runtime_error("ParameterType::QSCHEME");
  } else {
    // 如果参数类型未知，抛出运行时错误
    throw std::runtime_error("unknown parameter type");
  }
  // 将处理过的字符串值赋给默认值
  default_value = str;
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 根据给定的格式字符串和索引，初始化函数签名对象
FunctionSignature::FunctionSignature(const std::string& fmt, int index)
    : min_args(0),                   // 最小参数个数初始化为0
      max_args(0),                   // 最大参数个数初始化为0
      max_pos_args(0),               // 最大位置参数个数初始化为0
      index(index),                  // 设置索引
      hidden(false),                 // 初始化隐藏标志为false
      deprecated(false) {            // 初始化废弃标志为false

  auto open_paren = fmt.find('(');   // 查找开括号位置
  if (open_paren == std::string::npos) {
    throw std::runtime_error("missing opening parenthesis: " + fmt);
  }
  name = fmt.substr(0, open_paren);  // 截取函数名

  bool allow_numbers_as_tensors = should_allow_numbers_as_tensors(name);  // 检查是否允许将数字视为张量

  auto last_offset = open_paren + 1;  // 上一个偏移量从开括号后开始
  bool keyword_only = false;
  bool done = false;
  while (!done) {
    auto offset = fmt.find(", ", last_offset);  // 查找下一个参数分隔符的位置
    auto next_offset = offset + 2;
    if (offset == std::string::npos) {
      offset = fmt.find(')', last_offset);     // 查找闭括号的位置
      done = true;
      next_offset = offset + 1;
      // 如果参数列表为空，即 fn()
      if (offset == last_offset) {
        last_offset = next_offset;
        break;
      }
    }
    if (offset == std::string::npos) {
      throw std::runtime_error("missing closing parenthesis: " + fmt);
    }
    if (offset == last_offset) {
      throw std::runtime_error("malformed signature: " + fmt);
    }

    auto param_str = fmt.substr(last_offset, offset - last_offset);  // 提取参数字符串
    last_offset = next_offset;
    if (param_str == "*") {
      keyword_only = true;   // 标记为只关键字参数
    } else {
      params.emplace_back(param_str, keyword_only);  // 添加参数到参数列表
      params.back().allow_numbers_as_tensors = allow_numbers_as_tensors;  // 设置参数是否允许数字作为张量
    }
  }

  if (fmt.substr(last_offset) == "|deprecated") {
    hidden = true;            // 标记为隐藏函数
    // TODO: 解析废弃签名时提出警告
    deprecated = true;        // 标记为废弃函数
  } else if (fmt.substr(last_offset) == "|hidden") {
    hidden = true;            // 标记为隐藏函数
  }

  max_args = params.size();   // 设置最大参数个数为参数列表的大小

  // 计算非可选参数的个数
  for (auto& param : params) {
    if (!param.optional) {
      min_args++;             // 非可选参数数量加一
    }
    if (!param.keyword_only) {
      max_pos_args++;         // 非关键字参数数量加一
    }
  }
}

// 返回函数签名的字符串表示
std::string FunctionSignature::toString() const {
  // 处理可选参数等
  std::ostringstream ss;
  bool keyword_already = false;
  ss << "(";
  int i = 0;
  for (auto& param : params) {
    if (i != 0) {
      ss << ", ";
    }
    if (param.keyword_only && !keyword_already) {
      ss << "*, ";            // 插入关键字参数分隔符
      keyword_already = true;
    }
    ss << param.type_name() << " " << param.name;  // 添加参数类型和名称
    if (param.optional) {
      ss << " = " << param.default_value;  // 添加可选参数的默认值
    }
    i++;
  }
  ss << ")";
  return ss.str();            // 返回字符串表示
}

// 根据函数签名和实际参数个数，抛出额外参数异常
[[noreturn]] static void extra_args(
    const FunctionSignature& signature,
    Py_ssize_t nargs) {
  const auto max_pos_args = signature.max_pos_args;  // 获取最大位置参数个数
  const auto min_args = signature.min_args;          // 获取最小参数个数
  const long nargs_ = nargs;                         // 将参数个数转换为长整型
  if (min_args != max_pos_args) {                    // 如果最小参数个数不等于最大位置参数个数，则
    // 进行额外参数异常处理
    // 暂无具体实现
  }
}
  throw TypeError(
      "%s() takes from %zu to %zu positional arguments but %ld were given",
      signature.name.c_str(),
      min_args,
      max_pos_args,
      nargs_);

抛出一个类型错误异常，其中包含格式化字符串，说明函数 `%s()` 接受从 `%zu` 到 `%zu` 个位置参数，但实际给出了 `%ld` 个参数。


  }
  throw TypeError(
      "%s() takes %zu positional argument%s but %ld %s given",
      signature.name.c_str(),
      max_pos_args,
      max_pos_args == 1 ? "" : "s",
      nargs_,
      nargs == 1 ? "was" : "were");

抛出一个类型错误异常，其中包含格式化字符串，说明函数 `%s()` 接受 `%zu` 个位置参数，但实际给出了 `%ld` 个参数，用于给出详细信息。
}

[[noreturn]] static void missing_args(
    const FunctionSignature& signature,
    int idx) {
  int num_missing = 0;
  std::stringstream ss;

  auto& params = signature.params;
  // 从参数列表中检查缺失的必需位置参数
  for (auto it = params.begin() + idx; it != params.end(); ++it) {
    if (!it->optional) {
      if (num_missing > 0) {
        ss << ", ";
      }
      ss << '"' << it->name << '"';
      num_missing++;
    }
  }

  // 抛出类型错误异常，指示缺少的位置参数
  throw TypeError(
      "%s() missing %d required positional argument%s: %s",
      signature.name.c_str(),
      num_missing,
      num_missing == 1 ? "s" : "",
      ss.str().c_str());
}

// 在函数签名中查找给定名称的参数索引
static Py_ssize_t find_param(FunctionSignature& signature, PyObject* name) {
  Py_ssize_t i = 0;
  for (auto& param : signature.params) {
    int cmp = PyObject_RichCompareBool(name, param.python_name, Py_EQ);
    if (cmp < 0) {
      throw python_error();
    } else if (cmp) {
      return i;
    }
    i++;
  }
  return -1;
}

[[noreturn]] static void extra_kwargs(
    FunctionSignature& signature,
    PyObject* kwargs,
    Py_ssize_t num_pos_args) {
  PyObject* key = nullptr;
  PyObject* value = nullptr;
  Py_ssize_t pos = 0;

  // 遍历关键字参数字典，检查不期望的关键字参数或重复参数
  while (PyDict_Next(kwargs, &pos, &key, &value)) {
    if (!THPUtils_checkString(key)) {
      throw TypeError("keywords must be strings");
    }

    // 查找关键字参数在函数签名中的索引
    auto param_idx = find_param(signature, key);
    if (param_idx < 0) {
      throw TypeError(
          "%s() got an unexpected keyword argument '%s'",
          signature.name.c_str(),
          THPUtils_unpackString(key).c_str());
    }

    if (param_idx < num_pos_args) {
      throw TypeError(
          "%s() got multiple values for argument '%s'",
          signature.name.c_str(),
          THPUtils_unpackString(key).c_str());
    }
  }

  // 如果执行到这里，说明出现了未预期的关键字参数，抛出类型错误异常
  throw TypeError("invalid keyword arguments");
}

// 解析函数的参数和关键字参数
bool FunctionSignature::parse(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* dst[], // NOLINT
    std::vector<PyObject*>& overloaded_args,
    bool raise_exception) {
  Py_ssize_t nargs = args ? PyTuple_GET_SIZE(args) : 0;
  auto remaining_kwargs = kwargs ? PyDict_Size(kwargs) : 0;
  size_t arg_pos = 0;
  bool allow_varargs_intlist = false;

  // 如果只有一个位置参数且为 IntArrayRef 类型或 SymIntArrayRef 类型，则允许变长参数风格的 IntArrayRef
  if (max_pos_args == 1 &&
      (params[0].type_ == ParameterType::INT_LIST ||
       params[0].type_ == ParameterType::SYM_INT_LIST)) {
    allow_varargs_intlist = true;
  }

  // 检查传入的位置参数数量是否超出预期，并根据需要抛出异常
  if (static_cast<size_t>(nargs) > max_pos_args && !allow_varargs_intlist) {
    if (raise_exception) {
      // 抛出异常，指示传入的位置参数数量多于预期
      extra_args(*this, nargs);
    }
    return false;
  }

  int i = 0;
  // 如果 self 不为空并且具有 torch function，将其添加到重载参数列表中
  if (self != nullptr && check_has_torch_function(self, /*ignore_mode*/ true)) {
    append_overloaded_tensor(&overloaded_args, self);
  }
  // 遍历函数参数列表
  for (auto& param : params) {
    PyObject* obj = nullptr;
    bool is_kwd = false;
    if (arg_pos < static_cast<size_t>(nargs)) {
      // 如果当前位置参数的索引小于参数总数，说明还有未处理的位置参数
      // 在单个位置参数 IntArrayRef 之后给出额外的位置参数
      if (param.keyword_only) {
        // 如果参数标记为仅限关键字参数，处理额外的位置参数
        if (raise_exception) {
          // 如果需要抛出异常，调用 extra_args 函数处理多余的参数
          extra_args(*this, nargs);
        }
        return false;
      }
      // 获取当前位置参数的 Python 对象
      obj = PyTuple_GET_ITEM(args, arg_pos);
    } else if (kwargs) {
      // 如果已经处理完所有位置参数，检查是否有关键字参数
      obj = PyDict_GetItem(kwargs, param.python_name);
      // 检查额外的关键字参数，可能有多个对应的命名
      for (PyObject* numpy_name : param.numpy_python_names) {
        if (obj) {
          break;
        }
        obj = PyDict_GetItem(kwargs, numpy_name);
      }
      // 标记当前处理的是关键字参数
      is_kwd = true;
    }

    // 初始化失败索引为 -1
    int64_t failed_idx = -1;
    // 检查是否允许接受变长参数的整数列表，并且当前处理的是第一个位置参数且不是关键字参数
    bool varargs_eligible = allow_varargs_intlist && arg_pos == 0 && !is_kwd;
    // 如果参数值为空并且参数是可选的，或者参数值为 None 且允许参数为 None
    if ((!obj && param.optional) || (obj == Py_None && param.allow_none)) {
      // 将目标数组中的当前索引位置置为 nullptr
      dst[i++] = nullptr;
    } else if (!obj) {
      // 如果参数值为空且不可选，根据条件抛出异常或返回 false
      if (raise_exception) {
        // 抛出异常，提示缺少必需的位置参数
        // 示例：foo() missing 1 required positional argument: "b"
        missing_args(*this, i);
      }
      return false;
    } else if (param.check(obj, overloaded_args, i, &failed_idx)) {
      // 检查参数值是否符合预期的类型和条件，通过检查后将其添加到目标数组中
      dst[i++] = obj;
      // XXX: Variable 检查是必要的，因为在启用跟踪器时，大小会变成张量。
      // 这种行为很容易导致歧义，我们应避免使用复杂的签名来利用它...
      // （此处是对特定行为的技术性说明）
    } else if (
        varargs_eligible &&
        (is_int_or_symint_list(args, param.size, &failed_idx))) {
      // 如果符合条件允许接受变长参数的整数列表，且参数为整数或符号整数列表
      // 将所有位置参数作为此参数处理，例如 permute(1, 2, 3) -> permute((1, 2, 3))
      dst[i++] = args;
      // 更新参数位置索引到参数总数，继续下一个循环
      arg_pos = nargs;
      continue;
    // 如果参数验证失败，并且设置了抛出异常的选项
    } else if (raise_exception) {
      // 如果参数是关键字参数
      if (is_kwd) {
        // 抛出类型错误异常，格式化错误信息包括函数名、参数名、期望类型和实际类型
        throw TypeError(
            "%s(): argument '%s' must be %s, not %s",
            name.c_str(),
            param.name.c_str(),
            param.type_name().c_str(),
            Py_TYPE(obj)->tp_name);
      } else {
        // 如果参数是位置参数，并且设置了失败索引
        if (failed_idx != -1) {
          // 如果对象不是元组或列表，则内部断言它是变长参数的一部分
          if (!(PyTuple_Check(obj) || PyList_Check(obj))) {
            TORCH_INTERNAL_ASSERT(varargs_eligible);
            obj = args;
          }
          // 内部断言失败索引小于对象的长度
          TORCH_INTERNAL_ASSERT(failed_idx < PySequence_Size(obj));
          // 抛出类型错误异常，格式化错误信息包括函数名、参数名、位置、期望类型和实际类型及位置信息
          throw TypeError(
              "%s(): argument '%s' (position %ld) must be %s, but found element of type %s at pos %ld",
              name.c_str(),
              param.name.c_str(),
              static_cast<long>(arg_pos + 1),
              param.type_name().c_str(),
              Py_TYPE(py::reinterpret_steal<py::object>(
                          PySequence_GetItem(obj, failed_idx))
                          .ptr())
                  ->tp_name,
              static_cast<long>(failed_idx));
        }
        // 如果没有失败索引，则抛出类型错误异常，格式化错误信息包括函数名、参数名、位置、期望类型和实际类型
        throw TypeError(
            "%s(): argument '%s' (position %ld) must be %s, not %s",
            name.c_str(),
            param.name.c_str(),
            static_cast<long>(arg_pos + 1),
            param.type_name().c_str(),
            Py_TYPE(obj)->tp_name);
      }
    } else {
      // 如果没有设置抛出异常选项，则返回 false
      return false;
    }

    // 如果不是关键字参数，则增加参数位置
    if (!is_kwd) {
      arg_pos++;
    } else if (obj) {
      // 如果是关键字参数并且对象存在，则减少剩余关键字参数计数
      remaining_kwargs--;
    }
  }

  // 如果还有剩余的关键字参数
  if (remaining_kwargs > 0) {
    // 如果设置了抛出异常选项，则抛出异常，说明有未预期的关键字参数
    if (raise_exception) {
      extra_kwargs(*this, kwargs, nargs);
    }
    // 返回 false
    return false;
  }
  // 参数验证通过，返回 true
  return true;
}

PythonArgParser::PythonArgParser(
    const std::vector<std::string>& fmts,
    bool traceable)
    : max_args(0), traceable(traceable) {
  // 初始化最大参数为0，跟踪性设置为给定的值
  int index = 0;
  // 遍历给定的格式字符串列表
  for (auto& fmt : fmts) {
    // 将格式字符串和索引作为函数签名对象添加到签名列表中
    signatures_.emplace_back(fmt, index);
    ++index;
  }
  // 找出签名列表中最大的参数个数
  for (auto& signature : signatures_) {
    if (signature.max_args > max_args) {
      max_args = signature.max_args;
    }
  }
  // 如果签名列表非空，设置函数名为第一个签名的名字
  if (!signatures_.empty()) {
    function_name = signatures_[0].name;
  }

  // 将已废弃的签名移动到签名列表的末尾
  std::stable_partition(
      signatures_.begin(), signatures_.end(), [](const FunctionSignature& sig) {
        return !sig.deprecated;
      });
}

void PythonArgParser::check_deprecated(const FunctionSignature& signature) {
  // 如果签名已经废弃
  if (signature.deprecated) {
    // 构造废弃签名的警告消息
    auto msg = c10::str(
        "This overload of ",
        signature.name,
        " is deprecated:\n\t",
        signature.name,
        signature.toString());
    auto signatures = get_signatures();
    // 如果存在可用的签名，添加建议使用的签名信息到警告消息中
    if (!signatures.empty()) {
      msg += "\nConsider using one of the following signatures instead:";
      for (const auto& sig : signatures) {
        msg += "\n\t";
        msg += signature.name;
        msg += sig;
      }
    }
    // 发出一次性废弃警告消息
    TORCH_WARN_ONCE(msg);
  }
}

PythonArgs PythonArgParser::raw_parse(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* parsed_args[]) { // NOLINT
  // 如果只有一个签名
  if (signatures_.size() == 1) {
    auto& signature = signatures_[0];
    std::vector<PyObject*> overloaded_args;
    // 解析参数并检查废弃状态
    signature.parse(self, args, kwargs, parsed_args, overloaded_args, true);
    check_deprecated(signature);
    // 返回解析后的参数对象
    return PythonArgs(
        traceable, signature, parsed_args, std::move(overloaded_args));
  }

  // 对所有签名进行解析，找到第一个匹配的签名
  for (auto& signature : signatures_) {
    std::vector<PyObject*> overloaded_args;
    if (signature.parse(
            self, args, kwargs, parsed_args, overloaded_args, false)) {
      check_deprecated(signature);
      // 返回解析后的参数对象
      return PythonArgs(
          traceable, signature, parsed_args, std::move(overloaded_args));
    }
  }

  // 如果没有匹配的签名，打印错误信息
  print_error(self, args, kwargs, parsed_args);
}

void PythonArgParser::print_error(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* parsed_args[]) { // NOLINT
  // 计算传入的参数个数
  size_t num_args =
      (args ? PyTuple_GET_SIZE(args) : 0) + (kwargs ? PyDict_Size(kwargs) : 0);
  std::vector<unsigned> plausible_idxs;
  unsigned i = 0;
  // 找出可能的签名索引
  for (auto& signature : signatures_) {
    if (num_args >= signature.min_args && num_args <= signature.max_args &&
        !signature.hidden) {
      plausible_idxs.push_back(i);
    }
    i++;
  }

  // 如果只有一个可能的签名匹配
  if (plausible_idxs.size() == 1) {
    auto& signature = signatures_[plausible_idxs[0]];
    std::vector<PyObject*> overloaded_args;
    // 解析参数并检查废弃状态
    signature.parse(self, args, kwargs, parsed_args, overloaded_args, true);
  }

  // 获取所有签名选项
  auto options = get_signatures();
  // 格式化错误消息，包括无效参数
  auto msg =
      torch::format_invalid_args(args, kwargs, function_name + "()", options);
  // 抛出类型错误异常，包含错误消息
  throw TypeError("%s", msg.c_str());
}
// 返回 PythonArgParser 类中存储的所有非隐藏签名的字符串表示形式
std::vector<std::string> PythonArgParser::get_signatures() const {
  std::vector<std::string> options;
  // 遍历所有签名
  for (auto& signature : signatures_) {
    // 如果签名不是隐藏的，则将其字符串表示形式加入选项列表中
    if (!signature.hidden) {
      options.push_back(signature.toString());
    }
  }
  // 返回所有非隐藏签名的字符串列表
  return options;
}

// 返回 PythonArgs 对象中索引为 i 的参数对应的 Tensor
at::Tensor PythonArgs::tensor_slow(int i) {
  PyObject* obj = args[i];
  // 如果参数为 NULL，则返回一个空 Tensor
  if (!obj) {
    return at::Tensor();
  }
  // 如果参数是 THPVariable 类型，则解包为 Tensor 并返回
  if (THPVariable_Check(obj)) {
    return THPVariable_Unpack(obj);
  }

  // 以下根据不同的 Python 对象类型，转换为对应的 Scalar 类型
  bool save_symint = false;
  at::Scalar scalar;
  if (PyBool_Check(obj)) {
    scalar = at::Scalar(THPUtils_unpackBool(obj));
  } else if (THPUtils_checkLong(obj)) {
    // 如果是整数类型，根据情况转换为 int64_t 或 uint64_t 类型的 Scalar
    int overflow = -1;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (value == -1 && PyErr_Occurred()) {
      throw python_error();
    }
    if (overflow != 0) {
      // 尝试无符号整数类型转换
      unsigned long long value = PyLong_AsUnsignedLongLong(obj);
      if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        throw python_error();
      }
      scalar = at::Scalar(static_cast<uint64_t>(value));
    } else {
      scalar = at::Scalar(static_cast<int64_t>(value));
    }
  } else if (PyComplex_Check(obj)) {
    // 如果是复数类型，转换为对应的复数 Scalar
    scalar = at::Scalar(THPUtils_unpackComplexDouble(obj));
  } else if (THPUtils_checkDouble(obj)) {
    // 如果是浮点数类型，转换为浮点数 Scalar
    scalar = at::Scalar(THPUtils_unpackDouble(obj));
    // 注意：不将符号整数/浮点数放入 Scalar 本身，因为后续转换为 Tensor 时不支持
  } else if (torch::is_symint(py::handle(obj))) {
    // 如果是符号整数类型，标记为需要保存
    save_symint = true;
    // 这个 Scalar 的实际值并不重要，不会被读取到，只是为了标记
    scalar = at::Scalar(7777777);
  } else if (torch::is_symfloat(py::handle(obj))) {
    // 如果是符号浮点数类型，标记为需要保存，并使用 NaN 初始化 Scalar
    save_symint = true;
    scalar = at::Scalar(std::numeric_limits<double>::quiet_NaN());
  } else if (torch::is_symbool(py::handle(obj))) {
    // 如果是符号布尔类型，标记为需要保存，并使用 true 初始化 Scalar
    save_symint = true;
    scalar = at::Scalar(true);
  } else {
    // 如果参数类型不匹配预期类型，抛出 TypeError 异常
    throw TypeError(
        "expected Tensor as argument %d, but got %s", i, Py_TYPE(obj)->tp_name);
  }

  // 设置自动分发和跟踪器的保护
  at::AutoDispatchBelowADInplaceOrView guard; // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;

  // 将 Scalar 转换为 Tensor
  at::Tensor tensor = scalar_to_tensor(scalar);
  // 设置 Tensor 的 wrapped_number 标志为 true
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);

  // 如果需要保存符号整数/浮点数信息，则将原始 Python 对象保存在 Tensor 的 _wrapped_number 属性中
  if (save_symint) {
    auto py_tensor = py::cast(tensor);
    if (PyObject_SetAttrString(py_tensor.ptr(), "_wrapped_number", obj) < 0) {
      throw python_error();
    }
  }

  // 返回转换后的 Tensor 对象
  return tensor;
}
// 返回 PythonArgs 对象中索引为 i 的参数的标量值
at::Scalar PythonArgs::scalar_slow(int i) {
  // 如果正在追踪且参数是 THPVariable 类型，则解包并存储值
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, c10::NumberType::get());
  }

  // 调用 scalar_slow(PyObject* arg) 处理参数 args[i]
  return scalar_slow(args[i]);
}

// 处理 Python 对象 arg，并返回对应的标量值
at::Scalar PythonArgs::scalar_slow(PyObject* arg) {
  // 将零维张量直接转换为标量。注意，此处未处理除 np.float64 外的大多数 NumPy 标量类型。
  if (THPVariable_Check(arg)) {
    return THPVariable_Unpack(arg).item();
  }

  // 检查是否为长整型
  if (THPUtils_checkLong(arg)) {
    int overflow = -1;
    long long value = PyLong_AsLongLongAndOverflow(arg, &overflow);
    if (value == -1 && PyErr_Occurred()) {
      throw python_error();
    }
    if (overflow != 0) {
      // 尝试转换为无符号长整型
      unsigned long long value = PyLong_AsUnsignedLongLong(arg);
      if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        throw python_error();
      }
      return at::Scalar(static_cast<uint64_t>(value));
    } else {
      return at::Scalar(static_cast<int64_t>(value));
    }
  }

  // 检查是否为布尔型
  if (PyBool_Check(arg)) {
    return at::Scalar(THPUtils_unpackBool(arg));
  }

  // 检查是否为复数型
  if (PyComplex_Check(arg)) {
    return at::Scalar(THPUtils_unpackComplexDouble(arg));
  }

  // 检查是否为符号整型
  if (torch::is_symint(arg)) {
    return at::Scalar(py::cast<c10::SymInt>(arg));
  }

  // 检查是否为符号浮点型
  if (torch::is_symfloat(arg)) {
    return at::Scalar(py::cast<c10::SymFloat>(arg));
  }

  // 检查是否为符号布尔型
  if (torch::is_symbool(arg)) {
    // 在 Windows 构建时，使用 py::handle(arg).cast<c10::SymBool>() 会导致错误 C2440
    auto sym_bool = py::handle(arg).cast<c10::SymBool>();
    return at::Scalar(sym_bool);
  }

  // 默认情况下将 Python 对象转换为双精度浮点数并返回
  return at::Scalar(THPUtils_unpackDouble(arg));
}

} // namespace torch


这些注释为给定的 C++ 代码提供了详细的解释，涵盖了每行代码的功能和处理逻辑。
```