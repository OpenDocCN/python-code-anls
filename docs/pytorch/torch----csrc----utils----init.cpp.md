# `.\pytorch\torch\csrc\utils\init.cpp`

```
#include <ATen/core/ivalue.h>
#include <torch/csrc/utils/init.h>
#include <torch/csrc/utils/throughput_benchmark.h>

#include <pybind11/functional.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::throughput_benchmark {

void initThroughputBenchmarkBindings(PyObject* module) {
  // 将传入的 Python 模块对象转换为 py::module 类型
  auto m = py::handle(module).cast<py::module>();
  // 使用 torch::throughput_benchmark 命名空间
  using namespace torch::throughput_benchmark;
  // 定义 BenchmarkConfig 类的 Python 绑定
  py::class_<BenchmarkConfig>(m, "BenchmarkConfig")
      .def(py::init<>())
      .def_readwrite(
          "num_calling_threads", &BenchmarkConfig::num_calling_threads)
      .def_readwrite("num_worker_threads", &BenchmarkConfig::num_worker_threads)
      .def_readwrite("num_warmup_iters", &BenchmarkConfig::num_warmup_iters)
      .def_readwrite("num_iters", &BenchmarkConfig::num_iters)
      .def_readwrite(
          "profiler_output_path", &BenchmarkConfig::profiler_output_path);

  // 定义 BenchmarkExecutionStats 类的 Python 绑定
  py::class_<BenchmarkExecutionStats>(m, "BenchmarkExecutionStats")
      .def_readonly("latency_avg_ms", &BenchmarkExecutionStats::latency_avg_ms)
      .def_readonly("num_iters", &BenchmarkExecutionStats::num_iters);

  // 定义 ThroughputBenchmark 类的 Python 绑定
  py::class_<ThroughputBenchmark>(m, "ThroughputBenchmark", py::dynamic_attr())
      .def(py::init<jit::Module>())
      .def(py::init<py::object>())
      .def(
          "add_input",
          [](ThroughputBenchmark& self, py::args args, py::kwargs kwargs) {
            self.addInput(std::move(args), std::move(kwargs));
          })
      .def(
          "run_once",
          [](ThroughputBenchmark& self,
             py::args args,
             const py::kwargs& kwargs) {
            // 根据是否为 ScriptModule 或 nn.Module 释放 GIL
            return self.runOnce(std::move(args), kwargs);
          })
      .def(
          "benchmark",
          [](ThroughputBenchmark& self, const BenchmarkConfig& config) {
            // 始终在不使用 GIL 的情况下运行基准测试
            // 在 nn.Module 模式下操作输入和运行实际推理时将使用 GIL
            pybind11::gil_scoped_release no_gil_guard;
            return self.benchmark(config);
          });
}

} // namespace torch::throughput_benchmark
```