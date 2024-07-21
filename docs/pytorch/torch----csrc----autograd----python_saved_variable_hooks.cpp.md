# `.\pytorch\torch\csrc\autograd\python_saved_variable_hooks.cpp`

```py
namespace py = pybind11;  // 引入 pybind11 命名空间，用于与 Python 的绑定交互

namespace torch::autograd {
PySavedVariableHooks::PySavedVariableHooks(
    py::function& pack_hook,  // 构造函数，接收打包钩子函数的引用参数
    py::function& unpack_hook)  // 接收解包钩子函数的引用参数
    : pack_hook_(pack_hook.release().ptr()),  // 使用传入的打包钩子函数并释放其引用
      unpack_hook_(unpack_hook.release().ptr()) {}  // 使用传入的解包钩子函数并释放其引用

// 避免使用 pybind 处理 call_pack_hook 和 call_unpack_hook，以避免问题
void PySavedVariableHooks::call_pack_hook(const at::Tensor& tensor) {
  py::gil_scoped_acquire acquire;  // 获取全局解释器锁，以确保线程安全
  THPObjectPtr obj(THPVariable_Wrap(tensor));  // 将张量 tensor 包装为 Python 对象
  THPObjectPtr packed(
      PyObject_CallFunctionObjArgs(pack_hook_, obj.get(), nullptr));  // 调用打包钩子函数并传递参数
  if (!packed) {
    throw python_error();  // 如果调用失败，抛出 Python 异常
  }
  data_ = packed.release();  // 存储打包后的数据，释放 packed 的所有权
}

at::Tensor PySavedVariableHooks::call_unpack_hook() {
  py::gil_scoped_acquire acquire;  // 获取全局解释器锁，以确保线程安全
  THPObjectPtr res(PyObject_CallFunctionObjArgs(unpack_hook_, data_, nullptr));  // 调用解包钩子函数并传递参数
  if (!res) {
    throw python_error();  // 如果调用失败，抛出 Python 异常
  }
  TORCH_CHECK_TYPE(
      THPVariable_Check(res),
      "Output of saved tensor unpack_hook expected to be a Tensor but got result of type ",
      THPUtils_typename(res));  // 检查返回结果是否为 Tensor 类型
  return THPVariable_Unpack(res);  // 解包结果并返回张量
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PySavedVariableHooks::~PySavedVariableHooks() {
  // 如果 Python 已经关闭，则泄露包装的 Python 对象
  if (Py_IsInitialized()) {
    py::gil_scoped_acquire gil;  // 获取全局解释器锁
    Py_XDECREF(pack_hook_);  // 释放打包钩子函数的引用
    Py_XDECREF(unpack_hook_);  // 释放解包钩子函数的引用
    Py_XDECREF(data_);  // 释放数据对象的引用
  }
}

void PyDefaultSavedVariableHooks::push_hooks(
    py::function& pack_hook,  // 推入默认的打包钩子函数
    py::function& unpack_hook) {  // 推入默认的解包钩子函数
  at::SavedTensorDefaultHooks::lazy_initialize();  // 惰性初始化默认的张量保存钩子
  at::SavedTensorDefaultHooks::push_hooks(
      pack_hook.release().ptr(), unpack_hook.release().ptr());  // 推入钩子函数到默认的张量保存钩子
}

void PyDefaultSavedVariableHooks::pop_hooks() {
  auto [pack_hook, unpack_hook] = at::SavedTensorDefaultHooks::pop_hooks();  // 弹出默认的打包和解包钩子函数
  TORCH_INTERNAL_ASSERT(pack_hook != nullptr && unpack_hook != nullptr);  // 断言钩子函数非空
  if (Py_IsInitialized()) {
    py::gil_scoped_acquire gil;  // 获取全局解释器锁
    Py_XDECREF(pack_hook);  // 释放打包钩子函数的引用
    Py_XDECREF(unpack_hook);  // 释放解包钩子函数的引用
  }
}

std::unique_ptr<SavedVariableHooks> PyDefaultSavedVariableHooks::get_hooks() {
  auto [pack_hook, unpack_hook] = at::SavedTensorDefaultHooks::get_hooks();  // 获取默认的打包和解包钩子函数
  if (!pack_hook || !unpack_hook) {
    return nullptr;  // 如果没有有效的钩子函数，则返回空指针
  }
  py::gil_scoped_acquire gil;  // 获取全局解释器锁
  py::function pack_hook_ = py::reinterpret_borrow<py::function>(pack_hook);  // 转换为 Python 函数对象
  py::function unpack_hook_ = py::reinterpret_borrow<py::function>(unpack_hook);  // 转换为 Python 函数对象
  return std::make_unique<PySavedVariableHooks>(pack_hook_, unpack_hook_);  // 创建 PySavedVariableHooks 对象并返回
}
} // namespace torch::autograd
```