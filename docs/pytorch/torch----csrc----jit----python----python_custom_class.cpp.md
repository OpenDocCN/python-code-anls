# `.\pytorch\torch\csrc\jit\python\python_custom_class.cpp`

```
/// 包含 Torch JIT 的 C++ 绑定和自定义类相关头文件

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_custom_class.h>

/// 包含 Torch JIT 的前端语法糖相关头文件
#include <torch/csrc/jit/frontend/sugared_value.h>

/// 包含格式化输出库 fmt 的头文件
#include <fmt/format.h>

/// Torch JIT 的命名空间
namespace torch::jit {

/// 定义一个结构体 ScriptClass，表示 Torch JIT 中的脚本类
struct CustomMethodProxy;
struct CustomObjectProxy;

/// ScriptClass 类的 __call__ 方法实现
py::object ScriptClass::__call__(py::args args, py::kwargs kwargs) {
  // 创建一个 Torch 的对象实例，使用给定的类类型和预留的槽数量为 1
  auto instance = Object(at::ivalue::Object::create(class_type_, /*numSlots=*/1));

  // 查找对象实例类型中的 __init__ 方法
  Function* init_fn = instance.type()->findMethod("__init__");
  // 检查是否找到了 __init__ 方法，如果未找到则抛出异常
  TORCH_CHECK(
      init_fn,
      fmt::format(
          "Custom C++ class: '{}' does not have an '__init__' method bound. "
          "Did you forget to add '.def(torch::init<...>)' to its registration?",
          instance.type()->repr_str()));

  // 创建 Method 对象，用于调用初始化方法
  Method init_method(instance._ivalue(), init_fn);
  // 调用 Python 传递过来的脚本方法，并传递参数 args 和 kwargs
  invokeScriptMethodFromPython(init_method, std::move(args), std::move(kwargs));
  
  // 将 Torch 对象实例转换为 Python 对象并返回
  return py::cast(instance);
}

/// ScriptClassFunctionPtr 结构体用于表示自定义类静态方法的指针
/// 由于这些方法不属于编译单元，因此不能使用 StrongFunctionPtr
/// 尽管直接携带原始指针通常是不安全的，但拥有该指针的自定义类方法注册表不会被销毁
struct ScriptClassFunctionPtr {
  /// 构造函数，接受一个 Function 指针作为参数
  ScriptClassFunctionPtr(Function* function) : function_(function) {
    // 内部断言，确保传入的 function 指针不为空
    TORCH_INTERNAL_ASSERT(function_);
  }
  
  /// 存储 Function 指针的成员变量
  Function* function_;
};
void initPythonCustomClassBindings(PyObject* module) {
  // 将传入的 Python 模块对象转换为 pybind11 模块
  auto m = py::handle(module).cast<py::module>();

  // 定义 ScriptClassFunction 类型，支持动态属性
  py::class_<ScriptClassFunctionPtr>(
      m, "ScriptClassFunction", py::dynamic_attr())
      .def("__call__", [](py::args args, const py::kwargs& kwargs) {
        // 从参数中获取 ScriptClassFunctionPtr，并转换为强引用指针
        auto strongPtr = py::cast<ScriptClassFunctionPtr>(args[0]);
        // 获取函数对象并调用，将结果返回给 Python
        Function& callee = *strongPtr.function_;
        py::object result = invokeScriptFunctionFromPython(
            callee, tuple_slice(std::move(args), 1), kwargs);
        return result;
      });

  // 定义 ScriptClass 类型
  py::class_<ScriptClass>(m, "ScriptClass")
      .def("__call__", &ScriptClass::__call__)  // 定义 __call__ 方法
      .def(
          "__getattr__",
          [](ScriptClass& self, const std::string& name) {
            // 定义 __getattr__ 方法以便在常规 Python 中使用自定义类的静态函数
            auto type = self.class_type_.type_->castRaw<ClassType>();
            TORCH_INTERNAL_ASSERT(type);
            auto* fn = type->findStaticMethod(name);  // 查找静态方法
            if (fn) {
              return ScriptClassFunctionPtr(fn);  // 返回静态方法对应的指针
            }

            throw AttributeError("%s does not exist", name.c_str());
          })
      .def_property_readonly("__doc__", [](const ScriptClass& self) {
        // 返回类的文档字符串
        return self.class_type_.type_->expectRef<ClassType>().doc_string();
      });

  // 返回一个 ScriptClass，包装了给定类的构造函数
  // 通过传入的限定名创建类的实例，模拟 Python 中类的实例化行为
  // 在 Python 中，实例化类是对类的代码对象的调用，而该代码对象又调用 __init__
  // 我们需要一个包装器，至少返回实例而不是 __init__ 的 None 返回值
  m.def(
      "_get_custom_class_python_wrapper",
      [](const std::string& ns, const std::string& qualname) {
        // 构建完整的限定名
        std::string full_qualname =
            "__torch__.torch.classes." + ns + "." + qualname;
        // 获取自定义类的类型
        auto named_type = getCustomClass(full_qualname);
        TORCH_CHECK(
            named_type,
            fmt::format(
                "Tried to instantiate class '{}.{}', but it does not exist! "
                "Ensure that it is registered via torch::class_",
                ns,
                qualname));
        c10::ClassTypePtr class_type = named_type->cast<ClassType>();
        // 返回一个包装了类类型的 ScriptClass 对象
        return ScriptClass(c10::StrongTypePtr(
            std::shared_ptr<CompilationUnit>(), std::move(class_type)));
      });
}

} // namespace torch::jit
```