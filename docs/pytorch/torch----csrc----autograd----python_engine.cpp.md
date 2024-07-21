# `.\pytorch\torch\csrc\autograd\python_engine.cpp`

```py
// 包含 Torch 的自动求导 Python 引擎头文件
#include <torch/csrc/autograd/python_engine.h>

// 包含 LegacyBatchedTensorImpl 的头文件
#include <ATen/LegacyBatchedTensorImpl.h>

// 包含 LegacyVmapMode 的头文件
#include <ATen/LegacyVmapMode.h>

// 包含 c10 的实用工具 irange 头文件
#include <c10/util/irange.h>

// 包含 pybind11 的头文件
#include <pybind11/pybind11.h>

// 包含 Torch 的动态类型头文件
#include <torch/csrc/DynamicTypes.h>

// 包含 Torch 的 THP 头文件
#include <torch/csrc/THP.h>

// 包含 Torch 的自动求导边缘头文件
#include <torch/csrc/autograd/edge.h>

// 包含 Torch 的自动求导引擎头文件
#include <torch/csrc/autograd/engine.h>

// 包含 Torch 的自动求导函数头文件
#include <torch/csrc/autograd/function.h>

// 包含 Torch 的自动求导基本操作函数头文件
#include <torch/csrc/autograd/functions/basic_ops.h>

// 包含 Torch 的自动求导 Python 异常模式头文件
#include <torch/csrc/autograd/python_anomaly_mode.h>

// 包含 Torch 的自动求导 Python C++ 函数头文件
#include <torch/csrc/autograd/python_cpp_function.h>

// 包含 Torch 的自动求导 Python 函数头文件
#include <torch/csrc/autograd/python_function.h>

// 包含 Torch 的自动求导 Python 保存的变量钩子头文件
#include <torch/csrc/autograd/python_saved_variable_hooks.h>

// 包含 Torch 的工具函数 pybind 头文件
#include <torch/csrc/utils/pybind.h>

// 包含 Torch 的工具函数 pycfunction_helpers 头文件
#include <torch/csrc/utils/pycfunction_helpers.h>

// 如果不在 Windows 系统下，则包含线程操作头文件 pthread.h
#ifndef _WIN32
#include <pthread.h>
#endif

// 包含内存管理头文件，用于 unique_ptr
#include <memory>

// 使用 Torch 的自动求导命名空间
using namespace torch::autograd;

// 定义 THPEngine 结构体，继承自 PyObject
struct THPEngine {
  PyObject_HEAD
};

// 定义全局变量，标记是否需要重新初始化引擎
static bool _reinitialize_engine = false;

// Torch 的自动求导命名空间
namespace torch {
namespace autograd {
namespace python {

// PythonEngine 类的默认构造函数
PythonEngine::PythonEngine() = default;

// 获取 Python 引擎的实例
Engine& PythonEngine::get_python_engine() {
  // 静态局部变量，保证 PythonEngine 单例模式
  static PythonEngine engine;

  // 如果需要重新初始化引擎
  if (_reinitialize_engine) {
    // 释放工作线程资源
    engine.release_workers();
    // 显式调用析构函数
    engine.~PythonEngine();
    // 使用 placement new 构造新的 PythonEngine 对象
    new (&engine) torch::autograd::python::PythonEngine();
    // 重置重新初始化标志
    _reinitialize_engine = false;
  }

  // 返回 PythonEngine 实例
  return engine;
}

// PythonEngine 类的析构函数
PythonEngine::~PythonEngine() {
  // 停止引擎
  Engine::stop();
}

// 如果 Python 版本大于等于 3.9，则定义宏 IS_PYTHON_3_9_PLUS
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 9
#define IS_PYTHON_3_9_PLUS
#endif

// 线程初始化函数
void PythonEngine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {
  // 在获取 GIL 前增加非可重入线程计数
  if (should_increment) {
    increment_non_reentrant_thread_count();
  }

  // 创建 PyThreadState，并释放 GIL。这样可以在 thread_main 中使用 pybind11::gil_scoped_acquire
  // 调用来获取 GIL，而不必每次都创建新的 PyThreadState。
#if defined(IS_PYTHON_3_9_PLUS)
  auto gil = std::make_unique<pybind11::gil_scoped_acquire>();
#else
  pybind11::gil_scoped_acquire gil;
#endif
  pybind11::gil_scoped_release no_gil;
  
  // 调用 Engine 的线程初始化方法
  Engine::thread_init(device, ready_queue, false);

  // 在关闭时如果之前增加过线程计数，则减少计数
  if (should_increment) {
    decrement_non_reentrant_thread_count();
  }

  // 如果 Python 版本大于等于 3.9，则不调用 PyEval_RestoreThread、PyThreadState_Clear 或 PyThreadState_DeleteCurrent
#if defined(IS_PYTHON_3_9_PLUS)
  if (!Py_IsInitialized()) {
    no_gil.disarm();
    // TODO: 等 PyThreadState_Clear 能够安全地从中调用时调用 disarm 一次
#endif
}
    // 释放 GIL（全局解释器锁）的所有权并获取其指针
    auto ptr = gil.release();
    // 调用 operator delete 释放 GIL 指针对应的内存
    operator delete(ptr);
#endif
}

// 处理线程异常，包括 Python 异常和其他异常
void PythonEngine::thread_on_exception(
    std::shared_ptr<GraphTask> graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {
  // 查看是否为 Python 错误，如果是，则持久化该错误状态
  auto python_err = dynamic_cast<python_error*>(&e);
  if (python_err) {
    python_err->persist();
  }
  // 调用基类 Engine 的异常处理方法
  Engine::thread_on_exception(std::move(graph_task), fn, e);
}

// 创建异常元数据的具体实现
std::unique_ptr<AnomalyMetadata> PythonEngine::make_anomaly_metadata() {
  return std::make_unique<PyAnomalyMetadata>();
}

// 获取默认的保存变量钩子的具体实现
std::unique_ptr<SavedVariableHooks> PythonEngine::
    get_default_saved_variable_hooks() {
  return PyDefaultSavedVariableHooks::get_hooks();
}

// 执行计算图的具体实现，处理多个参数和计算图边
variable_list PythonEngine::execute(
    const edge_list& roots,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) {
  // 检查是否持有 GIL，不应在持有 GIL 时调用自动求导引擎
  TORCH_CHECK(
      !PyGILState_Check(),
      "The autograd engine was called while holding the GIL. If you are using the C++ "
      "API, the autograd engine is an expensive operation that does not require the "
      "GIL to be held so you should release it with 'pybind11::gil_scoped_release no_gil;'"
      ". If you are not using the C++ API, please report a bug to the pytorch team.")
  try {
    // 调用基类 Engine 的执行方法来执行计算图
    return Engine::execute(
        roots, inputs, keep_graph, create_graph, accumulate_grad, outputs);
  } catch (python_error& e) {
    // 恢复 Python 错误状态并重新抛出异常
    e.restore();
    throw;
  }
}

// 执行带有计算图任务的具体实现，处理多个参数和计算图边
c10::intrusive_ptr<at::ivalue::Future> PythonEngine::execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {
  try {
    // 调用基类 Engine 的带有计算图任务的执行方法
    return Engine::execute_with_graph_task(
        graph_task, std::move(graph_root), std::move(input_buffer));
  } catch (python_error& e) {
    // 在捕获到 Python 错误时，需要重新获取 GIL 并恢复错误状态
    pybind11::gil_scoped_acquire gil;
    if (!PyErr_Occurred()) {
      // 只在错误状态未被设置时设置错误指示器
      e.restore();
    }
    throw;
  }
}
} // namespace python
} // namespace autograd
} // namespace torch

// THPEngineClass 用于实现 torch._C._EngineBase.run_backward
PyObject* THPEngineClass = nullptr;

// 实现 torch._C._EngineBase.run_backward 的具体方法
PyObject* THPEngine_run_backward(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  PyObject* tensors = nullptr;
  PyObject* grad_tensors = nullptr;
  unsigned char keep_graph = 0;
  unsigned char create_graph = 0;
  PyObject* inputs = nullptr;
  unsigned char allow_unreachable = 0;
  unsigned char accumulate_grad =
      0; // 标志是否累积梯度到叶子张量或捕获
  constexpr const char* accepted_kwargs[] = {// NOLINT
                                             "tensors",           // 接受张量参数
                                             "grad_tensors",      // 接受梯度张量参数
                                             "keep_graph",        // 是否保持计算图
                                             "create_graph",      // 是否创建计算图
                                             "inputs",            // 输入参数
                                             "allow_unreachable", // 是否允许不可达张量
                                             "accumulate_grad",   // 是否累积梯度
                                             nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OObb|Obb",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast,-warnings-as-errors)
          const_cast<char**>(accepted_kwargs),
          &tensors,
          &grad_tensors,
          &keep_graph,
          &create_graph,
          &inputs,
          &allow_unreachable,
          &accumulate_grad))
    return nullptr;
  TORCH_CHECK(
      PyTuple_Check(tensors),
      "tensors argument is expected to "
      "be a tuple, but got ",
      THPUtils_typename(tensors));  // 检查张量参数是否为元组类型
  TORCH_CHECK(
      PyTuple_Check(grad_tensors),
      "grad_tensors argument is "
      "expected to be a tuple, but got ",
      THPUtils_typename(grad_tensors));  // 检查梯度张量参数是否为元组类型

  Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors);
  Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_tensors);
  TORCH_CHECK(
      num_tensors == num_gradients,
      "got ",
      num_tensors,
      " tensors and ",
      num_gradients,
      " gradients");  // 检查张量和梯度张量的数量是否一致

  // 用户调用 autograd.backward(...) 或 autograd.grad(...) 来到这里
  bool backward_api_called = accumulate_grad;
  TORCH_CHECK(
      !backward_api_called || at::impl::VmapMode::current_vmap_level() == 0,
      "backward() called inside torch.vmap. This is not supported, "
      "please call backward() outside torch.vmap or instead use "
      "torch.autograd.grad inside torch.vmap");  // 检查是否在 torch.vmap 内部调用 backward()

  edge_list roots;
  roots.reserve(num_tensors);  // 预留存储空间以容纳边缘列表的大小为张量数量
  variable_list grads;
  grads.reserve(num_tensors);  // 预留存储空间以容纳变量列表的大小为张量数量
  for (const auto i : c10::irange(num_tensors)) {
    PyObject* _tensor = PyTuple_GET_ITEM(tensors, i);
    TORCH_CHECK(
        THPVariable_Check(_tensor),
        "element ",
        i,
        " of tensors tuple is not a Tensor");  // 检查张量元组中的第 i 个元素是否为 Tensor 类型
    const auto& variable = THPVariable_Unpack(_tensor);
    // 检查是否为批量张量，如果是则抛出错误信息，不支持在 torch.vmap 中使用 autograd.grad
    TORCH_CHECK(
        !isBatchedTensor(variable),
        "torch.autograd.grad(outputs, inputs, grad_outputs) called inside ",
        "torch.vmap. We do not support the case where any outputs are ",
        "vmapped tensors (output ",
        i,
        " is being vmapped over). Please "
        "call autograd.grad() outside torch.vmap or file a bug report "
        "with your use case.")

    // 获取变量的梯度边缘信息
    auto gradient_edge = torch::autograd::impl::gradient_edge(variable);

    // 检查梯度边缘的函数是否存在，如果不存在则抛出错误信息
    TORCH_CHECK(
        gradient_edge.function,
        "element ",
        i,
        " of tensors does not require grad and does not have a grad_fn");

    // 将梯度边缘信息移动到 roots 向量中
    roots.push_back(std::move(gradient_edge));

    // 获取梯度元组中第 i 个元素
    PyObject* grad = PyTuple_GET_ITEM(grad_tensors, i);

    // 如果是 THPVariable 类型的对象，则将其解包为 Variable，并检查是否具有命名
    if (THPVariable_Check(grad)) {
      const Variable& grad_var = THPVariable_Unpack(grad);
      // 如果梯度变量有命名，则发出警告，因为 autograd 不支持命名张量语义
      if (grad_var.has_names()) {
        TORCH_WARN(
            "Autograd was passed a named grad tensor with dims ",
            grad_var.names(),
            ". Autograd does not yet support named tensor semantics, so all names ",
            "will be ignored. In practice all computed gradients will still be correct "
            "according to regular tensor semantics.");
      }
      // 将梯度变量添加到 grads 向量中
      grads.push_back(grad_var);
    } else {
      // 如果梯度不是 Tensor 或者 None，则抛出错误信息
      TORCH_CHECK(
          grad == Py_None,
          "element ",
          i,
          " of gradients tuple is not a Tensor or None");
      // 如果变量要求梯度，但梯度为 None，则抛出错误信息
      TORCH_CHECK(
          !variable.requires_grad(),
          "element ",
          i,
          " of gradients tuple is None, but the corresponding Tensor requires grad");
    }
  }

  // 初始化输出边缘的向量
  std::vector<Edge> output_edges;
  // 如果输入不为空，则进行进一步检查
  if (inputs != nullptr) {
    // 检查输入是否为元组
    TORCH_CHECK(
        PyTuple_CheckExact(inputs), "inputs to run_backward must be a tuple");
    // 获取元组中元素的数量
    int num_inputs = PyTuple_GET_SIZE(inputs);
    // 预留足够的空间以容纳输出边缘的数量
    output_edges.reserve(num_inputs);
    for (const auto i : c10::irange(num_inputs)) {
      // 遍历输入的所有项
      PyObject* input = PyTuple_GET_ITEM(inputs, i);
      // 获取元组中第 i 个输入对象

      if (THPVariable_Check(input)) {
        // 检查输入是否为 THPVariable 类型
        const auto& tensor = THPVariable_Unpack(input);
        // 将 THPVariable 对象解包成 tensor
        TORCH_CHECK(
            !isBatchedTensor(tensor),
            // 检查 tensor 是否为批处理张量，如果是则抛出异常
            "torch.autograd.grad(outputs, inputs, grad_outputs) called inside ",
            "torch.vmap. We do not support the case where any inputs are ",
            "vmapped tensors (input ",
            i,
            " is being vmapped over). Please "
            "call autograd.grad() outside torch.vmap or file a bug report "
            "with your use case.")
        const auto output_nr = tensor.output_nr();
        // 获取 tensor 的输出编号
        auto grad_fn = tensor.grad_fn();
        // 获取 tensor 的梯度函数

        if (!grad_fn) {
          // 如果 grad_fn 为空
          // NOTE [ Autograd Unreachable Input ]
          // 因为输入没有梯度累加器，它保证是不可达的。我们初始化一个指向非空节点的边，
          // 以防止在执行信息中，对图中的节点（例如，当操作数为标量时的乘法）误分配 `needed = True`。
          output_edges.emplace_back(std::make_shared<Identity>(), 0);
          // 向输出边列表添加一个指向 Identity 的共享指针，编号为 0
        } else {
          // 如果 grad_fn 不为空
          output_edges.emplace_back(grad_fn, output_nr);
          // 向输出边列表添加一个指向 grad_fn 的边，输出编号为 output_nr
        }

        if (accumulate_grad) {
          // 如果需要累加梯度
          tensor.retain_grad();
          // 保留 tensor 的梯度
        }

        TORCH_CHECK(
            tensor.requires_grad(),
            // 检查 tensor 是否需要梯度，如果不需要则抛出异常
            "One of the differentiated Tensors does not require grad");
      } else if (PyObject_IsInstance(input, THPGradientEdgeClass)) {
        // 如果输入是 THPGradientEdgeClass 的实例
        auto node = PyTuple_GetItem(input, 0);
        // 获取元组中第一个对象作为节点
        bool isTHPFunction = THPFunction_Check(node);
        // 检查节点是否为 THPFunction 类型
        bool isTHPCppFunction = THPCppFunction_Check(node);
        // 检查节点是否为 THPCppFunction 类型

        TORCH_CHECK(
            isTHPFunction || isTHPCppFunction,
            // 检查节点类型是否为 THPFunction 或 THPCppFunction，否则抛出异常
            "GradientEdge first object must be an autograd.graph.Node "
            "but got ",
            THPUtils_typename(node));
        std::shared_ptr<torch::autograd::Node> node_sp;
        // 声明节点的共享指针

        if (isTHPFunction) {
          // 如果节点是 THPFunction 类型
          node_sp = ((THPFunction*)node)->cdata.lock();
          // 将 THPFunction 转换为节点的共享指针并加锁
        } else {
          // 如果节点是 THPCppFunction 类型
          node_sp = ((torch::autograd::THPCppFunction*)node)->cdata;
          // 将 THPCppFunction 转换为节点的共享指针
        }

        auto output_nr = THPUtils_unpackUInt32(PyTuple_GetItem(input, 1));
        // 解包元组中第二个对象作为输出编号
        output_edges.emplace_back(node_sp, output_nr);
        // 向输出边列表添加一个指向节点共享指针的边，输出编号为 output_nr
      } else {
        // 如果输入既不是 THPVariable 也不是 THPGradientEdgeClass
        TORCH_CHECK(
            false,
            // 抛出异常，要求所有的输入必须是 Tensors 或 GradientEdges
            "all inputs have to be Tensors or GradientEdges, but got ",
            THPUtils_typename(input));
      }
    }
  }

  variable_list outputs;
  // 声明变量列表 outputs

  {
    pybind11::gil_scoped_release no_gil;
    // 释放全局解释器锁
    auto& engine = python::PythonEngine::get_python_engine();
    // 获取 Python 引擎的实例
    outputs = engine.execute(
        roots, grads, keep_graph, create_graph, accumulate_grad, output_edges);
    // 执行 Python 引擎的 execute 方法，并将结果赋给 outputs
  }

  if (!backward_api_called && inputs != nullptr) {
    // 如果没有调用 backward_api 并且 inputs 不为空
    int num_inputs = PyTuple_GET_SIZE(inputs);
    // 获取元组 inputs 的大小
    THPObjectPtr py_outputs{PyTuple_New(num_inputs)};
    // 创建一个大小为 num_inputs 的 Python 元组对象指针
    // 如果 py_outputs 为空指针，则返回空指针
    if (!py_outputs)
      return nullptr;
    // 遍历 num_inputs 范围内的所有输入
    for (const auto i : c10::irange(num_inputs)) {
      // 检查是否允许不可达或者输出张量已定义，否则抛出错误信息
      TORCH_CHECK(
          allow_unreachable || outputs[i].defined(),
          "One of the "
          "differentiated Tensors appears to not have been used "
          "in the graph. Set allow_unused=True if this is the "
          "desired behavior.");
      // 将 outputs[i] 封装为 Python 对象，并设置为 py_outputs 的第 i 项
      PyTuple_SET_ITEM(py_outputs.get(), i, THPVariable_Wrap(outputs[i]));
    }
    // 释放 py_outputs 的所有权并返回
    return py_outputs.release();
  } else {
    // 如果条件不满足，则返回 Python 中的 None 对象
    Py_RETURN_NONE;
  }
  // 结束处理 Torch 错误的宏，通常用于异常处理的尾部
  END_HANDLE_TH_ERRORS
// 定义 THPEngineType 结构体，表示一个 Python 类型对象 torch._C._EngineBase
PyTypeObject THPEngineType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._EngineBase", /* tp_name */
    sizeof(THPEngine), /* tp_basicsize */
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


这段代码定义了一个 Python 类型对象 `THPEngineType`，表示为 `torch._C._EngineBase`。
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
    THPEngine_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPEngine_new /* tp_new */



    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    // 指针成员 tp_as_buffer 设为 nullptr，表示该类型不支持 buffer 接口
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    // 使用默认标志和基类型标志组合，这是一个 Python 类型对象的标志位
    nullptr, /* tp_doc */
    // 不提供文档字符串
    nullptr, /* tp_traverse */
    // 不提供遍历函数
    nullptr, /* tp_clear */
    // 不提供清除函数
    nullptr, /* tp_richcompare */
    // 不提供富比较函数
    0, /* tp_weaklistoffset */
    // 弱引用偏移量设为 0
    nullptr, /* tp_iter */
    // 不提供迭代器
    nullptr, /* tp_iternext */
    // 不提供迭代器的下一个函数
    THPEngine_methods, /* tp_methods */
    // 使用 THPEngine_methods 作为类型对象的方法描述列表
    nullptr, /* tp_members */
    // 不提供成员变量
    nullptr, /* tp_getset */
    // 不提供属性访问
    nullptr, /* tp_base */
    // 不继承于其它类型
    nullptr, /* tp_dict */
    // 不提供字典
    nullptr, /* tp_descr_get */
    // 不提供描述符的获取函数
    nullptr, /* tp_descr_set */
    // 不提供描述符的设置函数
    0, /* tp_dictoffset */
    // 字典偏移量设为 0
    nullptr, /* tp_init */
    // 不提供初始化函数
    nullptr, /* tp_alloc */
    // 不提供内存分配函数
    THPEngine_new /* tp_new */
    // 使用 THPEngine_new 作为类型对象的新建函数
};

// 子进程在 fork 后重新初始化引擎的函数
static void child_atfork() {
  _reinitialize_engine = true;
}

// 初始化 THPEngine 模块的函数
bool THPEngine_initModule(PyObject* module) {
#ifndef _WIN32
  // 注册在 fork 后调用的处理函数，用于重新初始化引擎
  if (pthread_atfork(nullptr, nullptr, child_atfork) != 0) {
    throw std::runtime_error("unable to set pthread_atfork handler");
  }
#endif

  // 准备 THPEngineType 类型对象
  if (PyType_Ready(&THPEngineType) < 0)
    return false;

  // 增加 THPEngineType 类型对象的引用计数
  Py_INCREF(&THPEngineType);

  // 将 THPEngineType 对象添加到模块中
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject*)&THPEngineType);

  // 设置默认的引擎存根为 PythonEngine::get_python_engine
  set_default_engine_stub(python::PythonEngine::get_python_engine);

  // 初始化模块成功，返回 true
  return true;
}
```