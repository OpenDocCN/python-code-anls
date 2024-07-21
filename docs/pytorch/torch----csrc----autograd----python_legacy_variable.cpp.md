# `.\pytorch\torch\csrc\autograd\python_legacy_variable.cpp`

```py
  // 引入 Torch 自动求导 Python 变量的头文件
  #include <torch/csrc/autograd/python_legacy_variable.h>

  // 引入 ATen 的头文件
  #include <ATen/ATen.h>

  // 引入 Torch 异常处理和自动求导 Python 功能、变量的头文件
  #include <torch/csrc/Exceptions.h>
  #include <torch/csrc/autograd/python_function.h>
  #include <torch/csrc/autograd/python_variable.h>
  #include <torch/csrc/jit/frontend/tracer.h>
  #include <torch/csrc/tensor/python_tensor.h>

  // 使用 ATen 命名空间
  using namespace at;

  // Torch 自动求导命名空间
  namespace torch {
  namespace autograd {

  // 定义静态函数 THPVariable_pynew，用于创建 Python 新变量对象
  static PyObject* THPVariable_pynew(
      PyTypeObject* type,
      PyObject* args,
      PyObject* kwds) {
    HANDLE_TH_ERRORS

    // 定义 Python 对象指针和变量
    THPObjectPtr _data;
    PyObject* data = nullptr;
    PyObject* grad_fn = nullptr;
    char is_volatile = 0;
    char requires_grad = 0;
    const char* name = nullptr;

    // 定义接受的参数数组，并解析传入的 Python 参数
    constexpr const char* accepted_args[] = {
        "data", "requires_grad", "volatile", "_grad_fn", "name", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "|ObbOz",
            // 禁止 Lint 检查数组的使用
            const_cast<char**>(accepted_args),
            &data,
            &requires_grad,
            &is_volatile,
            &grad_fn,
            &name))
      return nullptr;

    // 如果 grad_fn 是 None，则置为 nullptr
    if (grad_fn == Py_None)
      grad_fn = nullptr;

    // 如果 is_volatile 为真，则发出警告并抛出异常
    if (is_volatile) {
      auto r = PyErr_WarnEx(
          PyExc_UserWarning,
          "volatile was removed and now has no effect. Use `with torch.no_grad():` "
          "instead.",
          1);
      if (r != 0)
        throw python_error();
    }

    // 检查 volatile 和 requires_grad 不能同时为真
    TORCH_CHECK_VALUE(
        !is_volatile || !requires_grad,
        "Variable can't be volatile and require_grad at the same time!");

    // 检查 grad_fn 必须是 Function 对象或 None
    if (grad_fn && !THPFunction_Check(grad_fn)) {
      throw TypeError(
          "_grad_fn has to be a Function object or None, but got %s",
          Py_TYPE(grad_fn)->tp_name);
    }

    // 定义 Variable 变量
    Variable var;

    // 如果 data 为空或者为 None，则创建一个空张量
    if (!data || data == Py_None) {
      // 为了兼容旧版序列化代码，创建一个空张量
      auto dispatch_key = torch::tensors::get_default_dispatch_key();
      auto scalar_type = torch::tensors::get_default_scalar_type();
      auto options = TensorOptions(scalar_type)
                         .device(dispatchKeyToDeviceType(dispatch_key))
                         .layout(dispatchKeyToLayout(dispatch_key));
      var = at::empty({0}, options);
    } else if (THPVariable_Check(data)) {
      // 如果 data 是 THPVariable 类型，则解包并分离出 Variable
      var = THPVariable_Unpack(data).detach();
    } else {
    // 抛出类型错误异常，指示变量 `data` 必须是张量类型，给出实际类型信息
    throw torch::TypeError(
        "Variable data has to be a tensor, but got %s", Py_TYPE(data)->tp_name);
  }
  // 在此处将 `tensor` 的 `allow_tensor_metadata_change` 设置为 true，
  // 因为我们希望允许以下的用例以保持向后兼容性：
  //
  // ```python
  // var = Variable(torch.randn(2, 3))
  // var.resize_(4, 5)
  // ```py
  var.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);

  // 检查是否存在梯度函数 `grad_fn`，不支持在旧的 Variable 构造函数中使用 `_grad_fn`
  TORCH_CHECK(
      !grad_fn,
      "_grad_fn argument to legacy Variable constructor is no longer supported.  "
      "Instead, please invoke your _grad_fn to produce a variable with it as the "
      "_grad_fn.");
  // 设置变量是否需要梯度计算
  var.set_requires_grad(requires_grad);

  // 如果指定了变量名 `name`，则将其设置为变量 `var` 的名称
  if (name) {
    impl::set_name(var, name);
  }

  // 如果正在进行 TorchScript 的跟踪，并且 `data` 不为空且是有效的 PyTorch 变量，
  // 则尝试复制跟踪信息到新的变量 `var`
  if (jit::tracer::isTracing() && data && data != Py_None &&
      THPVariable_Check(data)) {
    if (auto* v = jit::tracer::getValueTrace(THPVariable_Unpack(data))) {
      jit::tracer::setValueTrace(var, v);
    }
  }

  // 返回封装后的 THPVariable 对象，移动操作
  return THPVariable_Wrap(std::move(var));
  END_HANDLE_TH_ERRORS
} // 结束 autograd 命名空间

PyTypeObject THPLegacyVariableType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch._C._LegacyVariableBase", /* tp_name */
    0, /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPVariable_pynew /* tp_new */
}; // 定义了一个名为 THPLegacyVariableType 的 PyTypeObject 结构体

void init_legacy_variable(PyObject* module) {
  if (PyType_Ready(&THPLegacyVariableType) < 0) {
    throw python_error(); // 如果 PyType_Ready 函数调用失败，抛出 python_error 异常
  }
  auto obj = (PyObject*)&THPLegacyVariableType;
  Py_INCREF(obj); // 增加对 obj 的引用计数
  if (PyModule_AddObject(module, "_LegacyVariableBase", obj) < 0) {
    throw python_error(); // 如果 PyModule_AddObject 函数调用失败，抛出 python_error 异常
  }
} // 初始化 legacy variable，并将其添加到给定的 Python 模块中

} // 结束 torch 命名空间
```