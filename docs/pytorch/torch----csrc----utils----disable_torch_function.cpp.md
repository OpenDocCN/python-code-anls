# `.\pytorch\torch\csrc\utils\disable_torch_function.cpp`

```
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    DisableTorchFunctionSubclass_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
    nullptr, /* tp_free */
    0, /* tp_is_gc */
    nullptr, /* tp_bases */
    nullptr, /* tp_mro */
    nullptr, /* tp_cache */
    nullptr, /* tp_subclasses */
    nullptr, /* tp_weaklist */
    nullptr, /* tp_del */
    0, /* tp_version_tag */
    nullptr /* tp_finalize */
};
    nullptr, /* tp_richcompare */
    // 定义比较运算符函数指针为空，表示不支持富比较
    
    0, /* tp_weaklistoffset */
    // 弱引用列表的偏移量为0，表示该类型对象不支持弱引用
    
    nullptr, /* tp_iter */
    // 迭代器函数指针为空，表示该类型对象不支持迭代器协议
    
    nullptr, /* tp_iternext */
    // 迭代器的下一个函数指针为空，表明该类型对象没有迭代器行为
    
    DisableTorchFunctionSubclass_methods, /* tp_methods */
    // 指定类型对象的方法集合为 DisableTorchFunctionSubclass_methods
    
    nullptr, /* tp_members */
    // 该类型对象没有成员变量
    
    nullptr, /* tp_getset */
    // 该类型对象没有属性
    
    nullptr, /* tp_base */
    // 没有基类，即该类型对象不继承自其它类型
    
    nullptr, /* tp_dict */
    // 该类型对象没有实例字典
    
    nullptr, /* tp_descr_get */
    // 描述器的获取函数为空，表明该类型对象没有描述器
    
    nullptr, /* tp_descr_set */
    // 描述器的设置函数为空，表明该类型对象没有描述器
    
    0, /* tp_dictoffset */
    // 实例字典偏移量为0，表示该类型对象的实例没有字典
    
    nullptr, /* tp_init */
    // 初始化函数为空，即不需要初始化操作
    
    PyType_GenericAlloc, /* tp_alloc */
    // 分配内存的函数指针为 PyType_GenericAlloc
    
    PyType_GenericNew, /* tp_new */
    // 新建实例的函数指针为 PyType_GenericNew
};

// 函数：THPModule_DisableTorchFunctionSubclassType，用于初始化 DisableTorchFunctionSubclassType 类型对象
PyObject* THPModule_DisableTorchFunctionSubclassType() {
  // 检查并准备 DisableTorchFunctionSubclassType 类型对象
  if (PyType_Ready(&DisableTorchFunctionSubclassType) < 0) {
    return nullptr;
  }

  // 返回指向 DisableTorchFunctionSubclassType 的 PyObject 指针
  return (PyObject*)(&DisableTorchFunctionSubclassType);
}

// 结构体定义：DisableTorchFunction，包含 Python 对象头和 at::impl::TorchFunctionDisabledState 类型的旧状态
typedef struct {
  PyObject_HEAD
      /* Type-specific fields go here. */
      at::impl::TorchFunctionDisabledState old_state;
} DisableTorchFunction;

// 函数：DisableTorchFunction__enter，用于进入 DisableTorchFunction 上下文管理器
PyObject* DisableTorchFunction__enter(PyObject* self, PyObject* unused) {
  // 将当前 TorchFunction 的禁用状态保存到上下文对象中
  ((DisableTorchFunctionSubclass*)self)->old_state =
      at::impl::PythonTorchFunctionTLS::get_disabled_state();
  // 禁用所有 TorchFunction
  at::impl::PythonTorchFunctionTLS::set_disabled_state(
      at::impl::TorchFunctionDisabledState::ALL_DISABLED);
  // 返回 None 对象表示成功
  Py_RETURN_NONE;
}

// 函数：DisableTorchFunction__exit，用于退出 DisableTorchFunction 上下文管理器
PyObject* DisableTorchFunction__exit(PyObject* self, PyObject* unused) {
  // 恢复之前保存的 TorchFunction 的禁用状态
  at::impl::PythonTorchFunctionTLS::set_disabled_state(
      ((DisableTorchFunctionSubclass*)self)->old_state);
  // 返回 None 对象表示成功
  Py_RETURN_NONE;
}

// 静态方法定义数组：DisableTorchFunction_methods，包含两个特殊方法及其处理函数
static PyMethodDef DisableTorchFunction_methods[] = { // NOLINT
    {"__enter__", DisableTorchFunction__enter, METH_NOARGS, nullptr}, // 进入上下文管理器方法
    {"__exit__", DisableTorchFunction__exit, METH_VARARGS, nullptr}, // 退出上下文管理器方法
    {nullptr, nullptr, 0, nullptr}};

// 类型对象定义：DisableTorchFunctionType，用于描述 DisableTorchFunction 类型
PyTypeObject DisableTorchFunctionType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch._C.DisableTorchFunction", /* tp_name */  // 类型名称
    sizeof(DisableTorchFunction), /* tp_basicsize */       // 基本大小
    0, /* tp_itemsize */                                   // 单个项目大小
    nullptr, /* tp_dealloc */                              // 释放函数
    0, /* tp_vectorcall_offset */                          // Vectorcall 偏移
    nullptr, /* tp_getattr */                              // 获取属性函数
    nullptr, /* tp_setattr */                              // 设置属性函数
    nullptr, /* tp_reserved */                             // 保留字段
    nullptr, /* tp_repr */                                 // 字符串表示函数
    nullptr, /* tp_as_number */                            // 数字类型相关函数
    nullptr, /* tp_as_sequence */                          // 序列类型相关函数
    nullptr, /* tp_as_mapping */                           // 映射类型相关函数
    nullptr, /* tp_hash  */                                // 哈希函数
    nullptr, /* tp_call */                                 // 调用函数
    nullptr, /* tp_str */                                  // 字符串转换函数
    nullptr, /* tp_getattro */                             // 获取属性函数
    nullptr, /* tp_setattro */                             // 设置属性函数
    nullptr, /* tp_as_buffer */                            // 缓冲区接口
    Py_TPFLAGS_DEFAULT, /* tp_flags */                     // 类型标志
    nullptr, /* tp_doc */                                  // 文档字符串
    nullptr, /* tp_traverse */                             // 遍历函数
    nullptr, /* tp_clear */                                // 清理函数
    nullptr, /* tp_richcompare */                          // 对象比较函数
    0, /* tp_weaklistoffset */                             // 弱引用偏移
    nullptr, /* tp_iter */                                 // 迭代器
    nullptr, /* tp_iternext */                             // 迭代函数
    DisableTorchFunction_methods, /* tp_methods */         // 方法列表
    nullptr, /* tp_members */                              // 成员列表
    nullptr, /* tp_getset */                               // 获取设置函数
    nullptr, /* tp_base */                                 // 基类类型
    nullptr, /* tp_dict */                                 // 字典
    nullptr, /* tp_descr_get */                            // 描述符获取函数
    nullptr, /* tp_descr_set */                            // 描述符设置函数
    0, /* tp_dictoffset */                                 // 字典偏移
    nullptr, /* tp_init */                                 // 初始化函数
    PyType_GenericAlloc, /* tp_alloc */                    // 分配函数
    PyType_GenericNew, /* tp_new */                        // 新建函数
};

// 函数：THPModule_DisableTorchFunctionType，用于初始化 DisableTorchFunctionType 类型对象
PyObject* THPModule_DisableTorchFunctionType() {
  // 检查并准备 DisableTorchFunctionType 类型对象
  if (PyType_Ready(&DisableTorchFunctionType) < 0) {
    return nullptr;
  }

  // 返回指向 DisableTorchFunctionType 的 PyObject 指针
  return (PyObject*)(&DisableTorchFunctionType);
}

// 函数：THPModule_disable_torch_function，用于禁用 Torch Function 的 Python 函数实现
PyObject* THPModule_disable_torch_function(PyObject* self, PyObject* a) {
  HANDLE_TH_ERRORS
  PyObject *func = nullptr, *types = nullptr, *args = nullptr,
           *kwargs = nullptr;
  // 解析函数参数，获取 func、types、args 和 kwargs
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  // 如果 args 为空，则创建一个空的 tuple
  if (args == nullptr) {
    py_args = py::make_tuple();
  } else if (PyList_Check(args)) { // 如果 args 是列表
    // 转换 args 到 py::tuple
    py_args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(args));

# 将 Python 的列表 `args` 转换为 Python C++ 扩展库中的元组 `py_args`。

  } else if (PyTuple_Check(args)) {

# 如果 `args` 是 Python 的元组类型：
    py_args = py::reinterpret_borrow<py::tuple>(args);

# 则直接将 `args` 引用为 Python C++ 扩展库中的元组 `py_args`。

  } else {

# 如果 `args` 不是列表也不是元组，则抛出类型错误异常。
    throw torch::TypeError(
        "expected List or Tuple (got %s)", Py_TYPE(args)->tp_name);
  }

  // These are all C-API calls so no exceptions will be raised
  // and therefore no need for RAII approach to storing
  // the old value.
  auto old_value = at::impl::PythonTorchFunctionTLS::get_disabled_state();

# 获取当前 Python Torch 函数的禁用状态 `old_value`，由于这些是 C-API 调用，不会引发异常，因此不需要使用 RAII 方法来存储旧值。

  if (old_value == at::impl::TorchFunctionDisabledState::ENABLED) {

# 如果当前禁用状态 `old_value` 为启用状态：
    at::impl::PythonTorchFunctionTLS::set_disabled_state(
        at::impl::TorchFunctionDisabledState::SUBCLASSES_DISABLED);
  }

# 则将 Python Torch 函数的禁用状态设置为子类禁用状态。

  // kwargs can safely be nullptr here.

# 这里 `kwargs` 可以安全地为 nullptr（空指针）。
  PyObject* result = PyObject_Call(func, py_args.ptr(), kwargs);

# 使用给定的函数对象 `func`、参数元组 `py_args` 和关键字参数 `kwargs` 来调用 Python 对象，并将结果存储在 `result` 中。

  at::impl::PythonTorchFunctionTLS::set_disabled_state(old_value);

# 恢复之前保存的 Python Torch 函数的禁用状态 `old_value`。

  return result;

# 返回 Python 对象调用的结果 `result`。

  END_HANDLE_TH_ERRORS

# 结束对 Torch 错误处理的代码块。
}

PyObject* THPModule_disable_torch_dispatch(PyObject* self, PyObject* a) {
  HANDLE_TH_ERRORS
  PyObject *func = nullptr, *types = nullptr, *args = nullptr,
           *kwargs = nullptr;
  // 解析传入的参数 a，期望接收两个对象和两个可选对象
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  // 如果参数 args 是 nullptr，则创建一个空的 Python 元组
  if (args == nullptr) {
    py_args = py::make_tuple();
  } else if (PyList_Check(args)) {
    // 如果 args 是列表，则将其转换为元组
    py_args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(args));
  } else if (PyTuple_Check(args)) {
    // 如果 args 已经是元组，则直接使用
    py_args = py::reinterpret_borrow<py::tuple>(args);
  } else {
    // 如果 args 不是列表也不是元组，则抛出类型错误异常
    throw torch::TypeError(
        "expected List or Tuple (got %s)", Py_TYPE(args)->tp_name);
  }

  // 以下是注释部分
  // 此实现并不完全正确。这个函数的主要目的是在 PythonKey 之后重新调度。
  // 但是在这里我们没有调度器调用，而是一个不透明的 Python 对象。
  //
  // 我们在这里实现了一个近似值：不是调度重新调度()，而是排除 Python 以及所有其之前的键，
  // 这样我们就会继续到 Python 之后的下一个键。然而，不同之处在于我们现在永久地在 Python 之后。
  // 我们认为没有任何合法的情况需要在整个调度器键集上再进行一轮，但如果有的话，
  // 那么我们将不得不在这里执行其他操作。
  c10::impl::ExcludeDispatchKeyGuard guard_(
      // TODO: 添加这个特定构造函数
      c10::DispatchKeySet(c10::DispatchKeySet::FULL) -
      c10::DispatchKeySet(
          c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Python)
      // 注意：这里可能存在一些误差，但是它能正常工作：Python 键不包含在 AFTER 中，
      // 因此它包含在否定中（这是正确的：我们想要排除 Python 键和其之前的一切。）
  );
  // 调用传入的函数 func，传递参数 py_args 和 kwargs
  auto r = PyObject_Call(func, py_args.ptr(), kwargs);
  if (r == nullptr)
    // 如果调用失败，则抛出 Python 错误
    throw python_error();
  return r;
  END_HANDLE_TH_ERRORS
}

// 确保我们不会在基本的 Python 类型上检查 __torch_function__
static bool is_basic_python_type(PyTypeObject* tp) {
  return (
      /* 基本的数值类型 */
      tp == &PyBool_Type ||

      tp == &PyLong_Type || tp == &PyFloat_Type || tp == &PyComplex_Type ||

      /* 基本的序列类型 */
      tp == &PyList_Type || tp == &PyTuple_Type || tp == &PyDict_Type ||
      tp == &PySet_Type || tp == &PyFrozenSet_Type || tp == &PyUnicode_Type ||
      tp == &PyBytes_Type ||

      /* 其他内置类型 */
      tp == &PySlice_Type || tp == Py_TYPE(Py_None) ||
      tp == Py_TYPE(Py_Ellipsis) || tp == Py_TYPE(Py_NotImplemented) ||

      PyModule_Check(tp) ||
      /* 用于占位，防止结尾的 || */
      false);
}

// 检查对象是否具有 __torch_function__ 属性
inline bool has_torch_function_attr(PyObject* obj) {
  auto attr = PyObject_FastGetAttrString(obj, "__torch_function__");
  return (
      attr.ptr() != nullptr && attr.ptr() != torch::disabled_torch_function);
}

namespace torch {
// 检查给定对象是否具有 Torch 函数重载，考虑是否忽略模式
auto check_has_torch_function(PyObject* obj, bool ignore_mode) -> bool {
  // 如果不忽略模式且 Torch 函数模式已启用，则返回 true
  if (!ignore_mode && at::impl::torch_function_mode_enabled())
    return true;
  
  // 获取对象的类型信息
  PyTypeObject* tp = Py_TYPE(obj);
  
  // 返回以下条件的逻辑与结果：
  //   - 对象不是 THPVariable 类型
  //   - 对象不是基本的 Python 类型
  //   - Torch 函数已启用
  //   - 对象具有 torch_function 属性
  return (
      !THPVariable_CheckTypeExact(tp) && !is_basic_python_type(tp) &&
      torch::torch_function_enabled() && has_torch_function_attr(obj));
}
} // namespace torch

// 检查序列中是否有对象具有 Torch 函数重载
inline bool sequence_has_torch_function(PyObject* args) {
  // 获取序列的长度
  // NOLINTNEXTLINE(bugprone-branch-clone)
  Py_ssize_t nargs = PySequence_Fast_GET_SIZE(args);
  
  // 遍历序列中的每个对象
  for (Py_ssize_t i = 0; i < nargs; i++) {
    // 获取序列中的第 i 个对象
    PyObject* obj = PySequence_Fast_GET_ITEM(args, i);
    
    // 如果对象具有 Torch 函数重载，则返回 true
    if (torch::check_has_torch_function(obj)) {
      return true;
    }
  }
  
  // 如果序列中没有对象具有 Torch 函数重载，则返回 false
  return false;
}

// 检查数组中是否有对象具有 Torch 函数重载
inline bool array_has_torch_function(PyObject* const* args, Py_ssize_t nargs) {
  // 遍历数组中的每个对象
  for (Py_ssize_t i = 0; i < nargs; i++) {
    // 如果数组中的第 i 个对象具有 Torch 函数重载，则返回 true
    if (torch::check_has_torch_function(args[i])) {
      return true;
    }
  }
  
  // 如果数组中没有对象具有 Torch 函数重载，则返回 false
  return false;
}

// Python 绑定函数：检查给定对象或序列是否具有 Torch 函数重载
PyObject* THPModule_has_torch_function(PyObject*, PyObject* arg) {
  bool result; // NOLINT(cppcoreguidelines-init-variables)
  
  // 如果参数是元组或列表，使用快速路径进行处理
  if (PyTuple_CheckExact(arg) || PyList_CheckExact(arg)) {
    // 快速路径：
    //   如果我们知道参数是元组或列表，可以跳过对 PySequence_Fast 的 INCREF 和 DECREF 操作
    //   核心函数几乎总是遵循这种约定（几乎总是元组），这样做可以减少大约 3.5% 的检查成本
    result = sequence_has_torch_function(arg);
  } else {
    // 否则，将参数转换为 Python 对象并处理
    auto args = py::reinterpret_steal<py::object>(
        PySequence_Fast(arg, "expected a sequence"));
    
    // 如果转换失败，则返回空指针
    if (!args) {
      return nullptr;
    }
    
    // 检查转换后的对象中是否有 Torch 函数重载
    result = sequence_has_torch_function(args.ptr());
  }
  
  // 如果结果为 true，则返回 Python 中的 True；否则返回 False
  if (result) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

// Python 绑定函数：检查给定单个对象是否具有 Torch 函数重载
PyObject* THPModule_has_torch_function_unary(PyObject*, PyObject* obj) {
  // 对单个对象调用 check_has_torch_function 进行检查
  // 如果对象具有 Torch 函数重载，则返回 Python 中的 True；否则返回 False
  if (torch::check_has_torch_function(obj)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

// Python 绑定函数：检查给定数组中的对象是否有 Torch 函数重载
PyObject* THPModule_has_torch_function_variadic(
    PyObject*,
    PyObject* const* args,
    Py_ssize_t nargs) {
  // 检查数组中的对象是否有 Torch 函数重载
  // 如果有任意对象具有 Torch 函数重载，则返回 Python 中的 True；否则返回 False
  if (array_has_torch_function(args, nargs)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}
```