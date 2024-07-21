# `.\pytorch\torch\csrc\utils\throughput_benchmark.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/ivalue.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/jit/python/pybind_utils.h>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace torch::throughput_benchmark {

/**
 * The struct is used to provide results of a benchmark to the caller
 * In the future all additional statics should be added here.
 */
struct BenchmarkExecutionStats {
  float latency_avg_ms{-1}; // 平均延迟，初始化为-1
  int64_t num_iters{-1}; // 迭代次数，初始化为-1
};

std::ostream& operator<<(
    std::ostream& os,
    const BenchmarkExecutionStats& value);
// 流输出运算符重载，用于将 BenchmarkExecutionStats 结构体输出到流中

/**
 * Use this struct in order to configure a throughput benchmark run.
 * This struct should include parameters related to threading, batching, number
 * of iterations, warm-up, etc. More configs can be added as needed.
 * General rule here is that only things that c++ must(!) to be aware of should
 * be here. If we can keep other parts in python, we should keep them there.
 * This is typical for things that are not perf critical and don't affect
 * execution statistics benchmark returns.
 */
struct BenchmarkConfig {
 public:
  // Calling threads are those threads that are calling into a module in
  // parallel.
  int num_calling_threads{1}; // 调用线程数，默认为1
  // Worker threads are not supported yet. This is just an example that we plan
  // to support some sort of multi-threaded forward calls. We may change this
  // setting in the future to support different intra and inter op parallelism
  // which is not available in PyTorch yet
  int num_worker_threads{1}; // 工作线程数，默认为1
  // Warmup iters are used to make sure we run a module a few times before
  // actually measuring things. This way we avoid cold caches and any other
  // similar problems
  int num_warmup_iters{1}; // 预热迭代次数，默认为1
  // Number of iterations the benchmark should run with. This number is separate
  // from the warmup iterations
  int64_t num_iters{100}; // 基准测试迭代次数，默认为100
  // If set autograd profiler will be enabled. I.e. this variable would be
  // created before the main benchmark loop (but after the warmup):
  // RecordProfile guard(profiler_output_path);
  std::string profiler_output_path{""}; // 自动求导分析器输出路径，默认为空字符串
};

namespace detail {

/**
 * A helper class to abstract out different models we test throughput of
 */
template <class Input, class Output, class Model>
// 模板类，用于抽象出我们测试吞吐量的不同模型
class BenchmarkHelper {
 public:
  BenchmarkHelper();  // 默认构造函数声明

  // 显式构造函数，接受一个模型对象作为参数
  explicit BenchmarkHelper(Model model)
      : model_(std::move(model)), initialized_(true) {}

  // 用于在 benchmark() 方法中调用的方法
  // 注意，此方法没有返回结果。即使在 nn.Module 模式下运行时，也无需在 GIL 下调用此方法。
  // 否则，结果的析构函数可能会与 Python 中的其他操作竞争资源。
  void runOnce(Input&&) const;

  // 用于直接从 Python 调用时使用的方法
  Output runOnce(py::args&&, const py::kwargs&) const;

  // 格式化输入，使其符合模型的要求，避免在基准测试时进行进一步的转换
  void addInput(py::args&&, py::kwargs&&);
  void addInput(Input&&);

  // 执行基准测试，并返回基准测试的执行统计信息
  BenchmarkExecutionStats benchmark(const BenchmarkConfig& config) const;

  // 返回 initialized_ 的状态
  bool initialized() const {
    return initialized_;
  }

  // 析构函数不需要 GIL，因为它将在 Python 线程中执行
  std::vector<Input> inputs_;  // 输入数据的向量
  Model model_;  // 模型对象
  bool initialized_{false};  // 初始化状态
};

struct C10_HIDDEN ModuleInput {
  ModuleInput(ModuleInput&& other) = default;

  ModuleInput(const ModuleInput&) = delete;  // 禁用复制构造函数
  ModuleInput& operator=(ModuleInput& other) = delete;  // 禁用赋值运算符（拷贝赋值）
  ModuleInput& operator=(ModuleInput&& other) = delete;  // 禁用移动赋值运算符

  // 使用传入的 args 和 kwargs 构造函数
  ModuleInput(py::args&& args, py::kwargs&& kwargs)
      : args(std::move(args)), kwargs(std::move(kwargs)) {}

  py::args args;  // Python 的位置参数
  py::kwargs kwargs;  // Python 的关键字参数
};

typedef py::object ModuleOutput;  // 模块输出的 Python 对象类型
typedef std::vector<at::IValue> ScriptModuleInput;  // 脚本模块的输入类型
typedef at::IValue ScriptModuleOutput;  // 脚本模块的输出类型

// 模板函数，用于克隆输入参数
template <class Input>
Input cloneInput(const Input& input);

// ScriptModuleBenchmark 类型的别名
typedef BenchmarkHelper<ScriptModuleInput, at::IValue, jit::Module>
    ScriptModuleBenchmark;

// ScriptModuleBenchmark 类的默认构造函数的特化实现
template <>
inline BenchmarkHelper<ScriptModuleInput, at::IValue, jit::Module>::
    BenchmarkHelper()
    : model_("Module", std::make_shared<jit::CompilationUnit>()),
      initialized_(false) {}

// ModuleBenchmark 类型的别名
typedef BenchmarkHelper<ModuleInput, py::object, py::object> ModuleBenchmark;

// ModuleBenchmark 类的默认构造函数的特化实现
template <>
inline BenchmarkHelper<ModuleInput, py::object, py::object>::BenchmarkHelper()
    : initialized_(false) {}

// 下面是特化实现，为 ScriptModuleBenchmark 类添加方法的定义

// runOnce 方法的特化实现，接受 ScriptModuleInput 类型的参数
template <>
void ScriptModuleBenchmark::runOnce(ScriptModuleInput&& input) const;

// runOnce 方法的特化实现，返回 ScriptModuleOutput 类型的结果
template <>
ScriptModuleOutput ScriptModuleBenchmark::runOnce(
    py::args&& args,
    const py::kwargs& kwargs) const;

// addInput 方法的特化实现，接受 py::args 和 py::kwargs 类型的参数
template <>
void ScriptModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs);

// addInput 方法的特化实现，接受 ScriptModuleInput 类型的参数
template <>
void ScriptModuleBenchmark::addInput(ScriptModuleInput&& input);

// 下面是特化实现，为 ModuleBenchmark 类添加方法的定义

// runOnce 方法的特化实现，接受 ModuleInput 类型的参数
template <>
void ModuleBenchmark::runOnce(ModuleInput&& input) const;

// runOnce 方法的特化实现，返回 ModuleOutput 类型的结果
template <>
ModuleOutput ModuleBenchmark::runOnce(
    py::args&& args, const py::kwargs& kwargs) const;

// addInput 方法的特化实现，接受 py::args 和 py::kwargs 类型的参数
template <>
void ModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs);

} // namespace detail
/**
 * This class is a small C++ component responsible for executing a PyTorch
 * module under an inference server like load. It can emulate multiple calling
 * threads to a single module provided. In the future we plan to enhance this
 * component to support inter and intra-op parallelism as well as multiple
 * models running in a single process.
 *
 * For current available configurations refer to the BenchmarkConfig
 * documentation
 *
 * The class supports working with either nn.Module or ScriptModule.
 * Under the hood it just dispatches to corresponding specialization of
 * class BenchmarkHelper<Input, Output, Model>
 */
class C10_HIDDEN ThroughputBenchmark {
 public:
  /**
   * Constructor accepting a PyTorch module in the form of jit::Module.
   */
  explicit ThroughputBenchmark(const jit::Module& module);

  /**
   * Constructor accepting a PyTorch module as a Python object.
   */
  explicit ThroughputBenchmark(py::object module);

  /**
   * Add one more input example. This input example should be in the exact
   * format the module under test expects. It is responsibility of the module to
   * perform any such format checks, the benchmark doesn't perform any
   * validation of its own.
   */
  void addInput(py::args args, py::kwargs kwargs);

  /**
   * Equivalent to just running the model directly on the given input.
   * Returns the output of the model execution.
   */
  py::object runOnce(py::args&& args, const py::kwargs& kwargs);

  /**
   * The main method of the class allows to perform a multi-threaded benchmark.
   * It returns BenchmarkExecutionStats object with a lot of useful statistics
   * about runtime execution. We can enhance this class in the future to provide
   * more information to the user.
   */
  BenchmarkExecutionStats benchmark(const BenchmarkConfig& config) const;

 private:
  detail::ScriptModuleBenchmark script_module_; // Instance for handling ScriptModule benchmarks
  detail::ModuleBenchmark module_; // Instance for handling nn::Module benchmarks
};
} // namespace torch::throughput_benchmark

#include <torch/csrc/utils/throughput_benchmark-inl.h>
```