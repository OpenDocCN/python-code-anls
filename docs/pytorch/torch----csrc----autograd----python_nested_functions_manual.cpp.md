# `.\pytorch\torch\csrc\autograd\python_nested_functions_manual.cpp`

```py
// 定义 torch::autograd 命名空间内的静态函数 THPVariable_nested_tensor
static PyObject* THPVariable_nested_tensor(
    PyObject* /*self*/,    // self 参数，通常用于方法，这里忽略
    PyObject* args,        // 位置参数元组
    PyObject* kwargs) {    // 关键字参数字典
  HANDLE_TH_ERRORS        // 处理 Torch 异常的宏开始

  // 定义静态的 PythonArgParser 对象，用于解析 Python 参数
  static PythonArgParser parser({
      "nested_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
  });

  constexpr int ctor_num_args = 5;
  // 创建 ParsedArgs 对象用于存储解析后的参数
  ParsedArgs<ctor_num_args> parsed_args;
  // 解析 Python 的 args 和 kwargs，将结果存储在 parsed_args 中
  auto r = parser.parse(args, kwargs, parsed_args);

  // 发出警告消息，指示使用过时的构造函数
  jit::tracer::warn(
      "torch.nested.nested_tensor", jit::tracer::WARN_CONSTRUCTOR);

  // 调用 torch::utils::nested_tensor_ctor 创建一个 THPVariable 对象，并包装成 PyObject*
  return THPVariable_Wrap(torch::utils::nested_tensor_ctor(
      torch::tensors::get_default_dispatch_key(),   // 获取默认的分发键
      torch::tensors::get_default_scalar_type(),    // 获取默认的标量类型
      r));    // 使用解析后的参数 r 进行构造
  END_HANDLE_TH_ERRORS    // 处理 Torch 异常的宏结束
}

// 定义 PyMethodDef 结构体数组 nested_functions_manual
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef nested_functions_manual[] = {
    // 注册 nested_tensor 方法到 Python 中，使用 castPyCFunctionWithKeywords 将 C++ 函数转换成 Python 方法
    {"nested_tensor",
     castPyCFunctionWithKeywords(THPVariable_nested_tensor),
     METH_VARARGS | METH_KEYWORDS,    // 方法接受位置参数和关键字参数
     nullptr},    // 方法文档字符串为空
};

// 返回 nested_functions_manual 数组，用于注册到 Python 模块中
PyMethodDef* get_nested_functions_manual() {
  return nested_functions_manual;
}
```