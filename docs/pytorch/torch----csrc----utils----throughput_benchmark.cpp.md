# `.\pytorch\torch\csrc\utils\throughput_benchmark.cpp`

```
// 包含 Torch 的性能基准工具的头文件
#include <torch/csrc/utils/throughput_benchmark.h>

// 包含 Pybind11 库的头文件
#include <pybind11/pybind11.h>

// 包含 Torch JIT 模块与 Python 绑定的工具函数的头文件
#include <torch/csrc/jit/python/pybind_utils.h>

// 包含 Torch 的 Python 绑定工具函数的头文件
#include <torch/csrc/utils/pybind.h>

// 定义了 torch::throughput_benchmark 命名空间
namespace torch::throughput_benchmark {

// 实现了流操作符重载，用于输出 BenchmarkExecutionStats 结构的信息
std::ostream& operator<<(
    std::ostream& os,
    const BenchmarkExecutionStats& value) {
  return os << "Average latency / iter (ms): " << value.latency_avg_ms
            << "\n Total number of iters: " << value.num_iters;
}

// 添加输入数据到模型中的方法
void ThroughputBenchmark::addInput(py::args args, py::kwargs kwargs) {
  // 检查只有一个脚本模块或者一个模块被初始化
  CHECK(script_module_.initialized() ^ module_.initialized());
  if (script_module_.initialized()) {
    // 如果是脚本模块，调用脚本模块的 addInput 方法
    script_module_.addInput(std::move(args), std::move(kwargs));
  } else {
    // 如果是普通模块，调用模块的 addInput 方法
    CHECK(module_.initialized());
    module_.addInput(std::move(args), std::move(kwargs));
  }
}

// 运行一次模型推断的方法，返回 Python 对象
py::object ThroughputBenchmark::runOnce(
    py::args&& args,
    const py::kwargs& kwargs) {
  // 检查只有一个脚本模块或者一个模块被初始化
  CHECK(script_module_.initialized() ^ module_.initialized());
  if (script_module_.initialized()) {
    c10::IValue result;
    {
      // 释放 GIL，以便在没有主线程 GIL 控制的情况下运行推断
      pybind11::gil_scoped_release no_gil_guard;
      // 调用脚本模块的 runOnce 方法进行推断
      result = script_module_.runOnce(std::move(args), kwargs);
    }
    // 将推断结果转换为 Python 对象并返回
    return jit::toPyObject(std::move(result));
  } else {
    // 如果是普通模块，直接调用模块的 runOnce 方法进行推断并返回结果
    CHECK(module_.initialized());
    return module_.runOnce(std::move(args), kwargs);
  }
}

// 使用脚本模块初始化的构造函数
ThroughputBenchmark::ThroughputBenchmark(const jit::Module& script_module)
    : script_module_(script_module) {}

// 使用普通模块初始化的构造函数
ThroughputBenchmark::ThroughputBenchmark(py::object module)
    : module_(std::move(module)) {}

// 对模型进行性能基准测试的方法
BenchmarkExecutionStats ThroughputBenchmark::benchmark(
    const BenchmarkConfig& config) const {
  // 检查只有一个脚本模块或者一个模块被初始化
  CHECK(script_module_.initialized() ^ module_.initialized());
  // 主要的基准测试线程在调度工作线程后不持有 GIL
  // 但是目前我们没有释放它，因为在 nn.Module 基准测试的情况下，我们将隐式地操作 py::object 的引用计数。
  if (script_module_.initialized()) {
    // 如果是脚本模块，调用脚本模块的 benchmark 方法进行基准测试
    return script_module_.benchmark(config);
  } else {
    // 如果是普通模块，调用模块的 benchmark 方法进行基准测试
    CHECK(module_.initialized());
    TORCH_WARN(
        "Starting benchmark on an nn.Module. This can be slow due "
        "to Python GIL.For proper inference simulation you might want to switch to "
        "a ScriptModule instead");
    return module_.benchmark(config);
  }
}

// Torch 脚本模块的详细信息命名空间
namespace detail {

// 对 ScriptModuleBenchmark 类模板特化，运行一次推断的方法
template <>
void ScriptModuleBenchmark::runOnce(ScriptModuleInput&& input) const {
  // 检查模块已初始化
  CHECK(initialized_);
  // TODO: 提供编译器不会优化此代码的保证
  // 调用模型的 forward 方法进行推断
  model_.get_method("forward").function()(std::move(input));
}

// 对 ScriptModuleBenchmark 类模板特化，运行一次推断的方法，返回输出
template <>
ScriptModuleOutput ScriptModuleBenchmark::runOnce(
    py::args&& args,
    const py::kwargs& kwargs) const {
  // 检查模块已初始化
  CHECK(initialized_);
  // 获取 forward 方法的函数对象
  auto& function = model_.get_method("forward").function();
  // 创建推断的输入栈
  ScriptModuleInput stack = jit::createStackForSchema(
      function.getSchema(), std::move(args), kwargs, model_._ivalue());
  // 调用 forward 方法进行推断并返回结果
  return function(std::move(stack));
}

// 对模板特化的终止标记
template <>
void ModuleBenchmark::runOnce(ModuleInput&& input) const {
```  
// 确保模块已经初始化
CHECK(initialized_);
// 获取全局解释器锁（GIL）以确保线程安全
pybind11::gil_scoped_acquire gil_guard;
// 调用模型的运行函数，传入参数和关键字参数
model_(*input.args, **input.kwargs);



template <>
ModuleOutput ModuleBenchmark::runOnce(py::args&& args, const py::kwargs& kwargs) const {
```  
// 确保模块已经初始化
CHECK(initialized_);
// 获取全局解释器锁（GIL）以确保线程安全
pybind11::gil_scoped_acquire gil_guard;
// 调用模型的运行函数，传入参数和关键字参数，并返回输出结果
return model_(*args, **kwargs);



template <>
void ScriptModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs) {
```  
// 为模型创建执行栈，用于调用前向方法
jit::Stack stack = jit::createStackForSchema(
    // 获取模型的前向方法的模式
    model_.get_method("forward").function().getSchema(),
    // 移动参数 args
    std::move(args),
    // 传入关键字参数 kwargs
    kwargs,
    // 获取模型的 ivalue
    model_._ivalue());
// 将创建的栈添加到输入列表中
inputs_.emplace_back(std::move(stack));



template <>
void ScriptModuleBenchmark::addInput(ScriptModuleInput&& input) {
```  
// 在输入的开头插入模型的 ivalue
input.insert(input.begin(), model_._ivalue());
// 将处理后的输入添加到输入列表中
inputs_.emplace_back(std::move(input));



template <>
void ModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs) {
```  
// 将传入的参数和关键字参数作为新的模块输入添加到输入列表中
inputs_.emplace_back(std::move(args), std::move(kwargs));



template <>
ModuleInput cloneInput<ModuleInput>(const ModuleInput& input) {
```  
// 获取全局解释器锁（GIL）以确保线程安全
pybind11::gil_scoped_acquire gil_guard;
// 克隆模块输入的参数和关键字参数
py::args args = input.args;
py::kwargs kwargs = input.kwargs;
// 返回克隆后的新的模块输入
return {std::move(args), std::move(kwargs)};



template <>
ScriptModuleInput cloneInput<ScriptModuleInput>(const ScriptModuleInput& input) {
```  
// 直接返回脚本模块输入的克隆
return input;
```