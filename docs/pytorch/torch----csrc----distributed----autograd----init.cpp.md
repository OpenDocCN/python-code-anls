# `.\pytorch\torch\csrc\distributed\autograd\init.cpp`

```
// 包含 Torch C++ 自动微分相关的头文件
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/distributed/autograd/autograd.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

namespace torch {
namespace distributed {
namespace autograd {

// 匿名命名空间，用于限定作用域
namespace {

// 定义模块初始化函数，初始化 torch.distributed.autograd 模块
PyObject* dist_autograd_init(PyObject* _unused, PyObject* noargs) {
  // 导入 torch.distributed.autograd 模块
  auto autograd_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.autograd"));
  if (!autograd_module) {
    // 如果导入失败，抛出 Python 异常
    throw python_error();
  }

  // 导入 torch._C 模块
  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    // 如果导入失败，抛出 Python 异常
    throw python_error();
  }

  // 返回初始化后的对象，无需返回值，直接传递 nullptr
  Py_RETURN_NONE;
}

} // namespace

} // namespace autograd
} // namespace distributed
} // namespace torch


这段代码是一个 C++ 的命名空间和函数定义，用于初始化分布式自动微分相关的 Torch 模块。
// 定义 Python 模块方法 "_dist_autograd_init"，用于初始化分布式自动求导功能
static PyObject* dist_autograd_init(PyObject* self, PyObject* args) {
  // 使用 pybind11::gil_scoped_release 调用保护来释放全局解释器锁（GIL）
  py::gil_scoped_release gil_release;

  // 将 C++ 函数注册为 Python 方法 "_dist_autograd_init"，该方法无参数
  auto module = py::module::import("torch.distributed.autograd");
  module.def("_dist_autograd_init", []() {
    // 调用 C++ 函数 DistAutogradContainer::init() 初始化分布式自动求导容器
    DistAutogradContainer::init();
  });

  // 将 C++ 函数注册为 Python 方法 "backward"
  module.def(
      "backward",
      [](int64_t contextId, const py::list& roots, bool retain_graph) {
        // 从全局分布式自动求导容器中检索指定 contextId 的自动求导上下文
        const auto& autogradContext =
            DistAutogradContainer::getInstance().retrieveContext(contextId);
        // 使用传入的 roots 参数调用自动求导的 backward 方法
        autogradContext->backward(roots, retain_graph);
      },
      R"(
backward(context_id: int, roots: List[Tensor], retain_graph: bool = False)

执行分布式自动求导的反向传播，累积梯度到指定 context_id 对应的自动求导上下文中。

参数:
    context_id(int): 要检索梯度的自动求导上下文的ID。
    roots(List[Tensor]): 作为反向传播起点的张量列表。
    retain_graph(bool, 可选): 是否保留计算图用于多次梯度计算，默认为 False。

示例::
    >>> import torch.distributed.autograd as dist_autograd
    >>> with dist_autograd.context() as context_id:
    >>>     t1 = torch.rand((3, 3), requires_grad=True)
    >>>     t2 = torch.rand((3, 3), requires_grad=True)
    >>>     loss = t1 + t2
    >>>     dist_autograd.backward(context_id, [loss.sum()])
)",
      py::arg("contextId"),
      py::arg("roots"),
      py::arg("retain_graph") = false,
      py::call_guard<py::gil_scoped_release>());

  // 将 C++ 函数注册为 Python 方法 "get_gradients"
  module.def(
      "get_gradients",
      [](int64_t contextId) -> py::dict {
        // 获取指定 contextId 的自动求导上下文
        const auto& autogradContext =
            DistAutogradContainer::getInstance().retrieveContext(contextId);
        // 获取梯度信息并转换为 IValue
        auto ival = IValue(autogradContext->getGradients());

        // 仅为 Python 对象转换时获取全局解释器锁（GIL）
        pybind11::gil_scoped_acquire ag;
        // 将 IValue 转换为 PyObject 返回
        return torch::jit::toPyObject(ival);
      },
      R"(
get_gradients(context_id: int) -> Dict[Tensor, Tensor]

检索在分布式自动求导的反向传播过程中，根据给定的 ``context_id`` 累积的 Tensor 到梯度的映射。

参数:
    context_id(int): 要检索梯度的自动求导上下文的ID。

返回:
    一个映射，其中键是 Tensor，值是对应 Tensor 的梯度。

示例::
    >>> import torch.distributed.autograd as dist_autograd
    >>> with dist_autograd.context() as context_id:
    >>>     t1 = torch.rand((3, 3), requires_grad=True)
    >>>     t2 = torch.rand((3, 3), requires_grad=True)
    >>>     loss = t1 + t2
    >>>     dist_autograd.backward(context_id, [loss.sum()])
    >>>     grads = dist_autograd.get_gradients(context_id)
    >>>     print(grads[t1])
    >>>     print(grads[t2])
)",
      py::arg("context_id"),
      py::call_guard<py::gil_scoped_release>());

  // 返回 Python 中的 True 值
  Py_RETURN_TRUE;
}
} // namespace

// 定义静态方法数组 methods
static PyMethodDef methods[] = { // NOLINT
    // 注册 C++ 函数 "_dist_autograd_init" 到 Python 方法 "_dist_autograd_init"
    {"_dist_autograd_init", dist_autograd_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

// 返回方法数组 methods
PyMethodDef* python_functions() {
  return methods;
}

// namespace autograd
} // namespace distributed
} // namespace torch
```