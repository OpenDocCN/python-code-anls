# `.\pytorch\torch\csrc\cuda\Graph.cpp`

```py
// 引入 Torch Python 头文件
#include <torch/csrc/python_headers.h>

// 引入 pybind11 对于时间处理的头文件
#include <pybind11/chrono.h>

// 引入 Torch JIT Python 绑定工具的头文件
#include <torch/csrc/jit/python/pybind_utils.h>

// 引入 Torch 工具函数的 Python 绑定头文件
#include <torch/csrc/utils/pybind.h>

// 引入 ATen CUDA 图形处理的头文件
#include <ATen/cuda/CUDAGraph.h>

// 引入 C10 CUDA 图形处理工具函数的头文件
#include <c10/cuda/CUDAGraphsC10Utils.h>

// 从 csrc/distributed/c10d/init.cpp 和 csrc/cuda/Stream.cpp 部分复制而来的代码。
// THCPStream_init 也在全局范围声明。

// 因为 THCPGraph_init 在唯一的使用者 (csrc/Module.cpp) 中被前向声明，
// 我们不认为需要 Graph.h。

// 定义一个模板，用于创建共享指针类的 pybind11 类型别名
template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;
// 初始化 THCPGraph 类，将函数注册到给定的 Python 模块中
void THCPGraph_init(PyObject* module) {
  // 将 Python 模块对象转换为 pybind11 的模块对象
  auto torch_C_m = py::handle(module).cast<py::module>();

  // 将 _graph_pool_handle 函数注册到 torch_C_m 模块中
  torch_C_m.def("_graph_pool_handle", &::at::cuda::graph_pool_handle);

  // 使用 shared_ptr_class_ 封装 ::at::cuda::CUDAGraph 类，并定义其初始化方法
  shared_ptr_class_<::at::cuda::CUDAGraph>(torch_C_m, "_CUDAGraph")
      .def(py::init<>())  // 定义默认构造函数
      .def(
          "capture_begin",  // 定义 capture_begin 方法
          [](::at::cuda::CUDAGraph& self,
             std::optional<c10::cuda::MempoolId_t> pool_opt,
             std::string capture_error_mode) {
            cudaStreamCaptureMode capture_mode;
            c10::cuda::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::cuda::MempoolId_t{0, 0};
            // 根据 capture_error_mode 设置 capture_mode
            if (capture_error_mode == "global") {
              capture_mode = cudaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cudaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cudaStreamCaptureModeRelaxed;
            } else {
              // 抛出错误，如果 capture_error_mode 不是预期的值
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            // 调用 CUDAGraph 对象的 capture_begin 方法
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),  // capture_begin 方法的参数说明
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())  // 设置 GIL 保护方式

      // 将 capture_end 方法注册到 _CUDAGraph 类中
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::capture_end))

      // 将 register_generator_state 方法注册到 _CUDAGraph 类中
      .def(
          "register_generator_state",
          [](::at::cuda::CUDAGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // 解除 GIL，调用 C++ 方法前释放 Python GIL
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))

      // 将 replay 方法注册到 _CUDAGraph 类中
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::replay))

      // 将 reset 方法注册到 _CUDAGraph 类中
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::reset))

      // 将 pool 方法注册到 _CUDAGraph 类中
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::pool))

      // 将 debug_dump 方法注册到 _CUDAGraph 类中
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump))

      // 将 enable_debug_mode 方法注册到 _CUDAGraph 类中
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::enable_debug_mode))

      // 将 debug_dump 方法注册到 _CUDAGraph 类中，带有 debug_path 参数
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump),
          py::arg("debug_path"));
}
```