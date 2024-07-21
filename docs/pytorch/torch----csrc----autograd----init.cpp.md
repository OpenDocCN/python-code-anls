# `.\pytorch\torch\csrc\autograd\init.cpp`

```
#include <torch/csrc/python_headers.h>
// 包含 Torch C++ API 的 Python 头文件

#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/SequenceNumber.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/PythonFallbackKernel.h>
#include <ATen/record_function.h>
#include <c10/core/DeviceType.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/profiler_python.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_saved_variable_hooks.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_raii.h>
#include <torch/csrc/utils/python_torch_function_mode.h>
// 包含各种 Torch 和 ATen 模块的头文件

#include <set>
#include <unordered_set>
#include <utility>
// 包含标准库中的集合和实用工具

using torch::impl::py_context_manager;
using torch::impl::py_context_manager_DEPRECATED;
// 使用 Torch 实现的 Python 上下文管理器

namespace {
// 匿名命名空间，限定符仅在当前文件内可见

struct DisableFuncTorch {
  DisableFuncTorch()
      : front_guard_(c10::DispatchKey::FuncTorchDynamicLayerFrontMode),
        back_guard_(c10::DispatchKey::FuncTorchDynamicLayerBackMode) {}
  c10::impl::ExcludeDispatchKeyGuard front_guard_;
  c10::impl::ExcludeDispatchKeyGuard back_guard_;
};
// 结构体 DisableFuncTorch，用于禁用 FuncTorch 动态层模式的调度键

struct DisableAutocast {
  c10::impl::ExcludeDispatchKeyGuard guard_{c10::autocast_dispatch_keyset};
};
// 结构体 DisableAutocast，用于禁用自动混合精度转换的调度键

struct EnableTorchFunction {
  EnableTorchFunction()
      : old_(at::impl::PythonTorchFunctionTLS::get_disabled_state()) {
    at::impl::PythonTorchFunctionTLS::set_disabled_state(
        at::impl::TorchFunctionDisabledState::ENABLED);
  }
  ~EnableTorchFunction() {
    at::impl::PythonTorchFunctionTLS::set_disabled_state(old_);
  }
  at::impl::TorchFunctionDisabledState old_;
};
// 结构体 EnableTorchFunction，用于启用 Torch 函数的调度状态

struct EnablePythonDispatcher {
  EnablePythonDispatcher() : old_(c10::impl::PythonDispatcherTLS::get_state()) {
    c10::impl::PythonDispatcherTLS::set_state(getPyInterpreter());
  }
  ~EnablePythonDispatcher() {
    c10::impl::PythonDispatcherTLS::set_state(old_);
  }
  c10::impl::PyInterpreter* old_;
};
// 结构体 EnablePythonDispatcher，用于启用 Python 分发器的状态

struct EnablePreDispatch {
  EnablePreDispatch() : guard_(c10::DispatchKey::PreDispatch) {}
  c10::impl::IncludeDispatchKeyGuard guard_;
};
// 结构体 EnablePreDispatch，用于启用预调度的调度键

} // namespace
// 匿名命名空间结束
// 导入 Torch 的自动微分模块中的 profiler 命名空间
using namespace torch::autograd::profiler;
// 导入 Torch 的 profiler 实现命名空间
using namespace torch::profiler::impl;

// 导入并初始化 torch._tensor 模块，如果失败则返回 nullptr
auto tensor_module = THPObjectPtr(PyImport_ImportModule("torch._tensor"));
if (!tensor_module)
  return nullptr;

// 获取 THPVariableClass，即 torch._tensor.Tensor 对应的 Python 类
// 注意：此处可能会产生内存泄漏
THPVariableClass = PyObject_GetAttrString(tensor_module, "Tensor");
if (!THPVariableClass)
  return nullptr;

// 导入并初始化 torch.autograd 模块，如果失败则返回 nullptr
auto autograd_module = THPObjectPtr(PyImport_ImportModule("torch.autograd"));
if (!autograd_module)
  return nullptr;

// 获取 THPFunctionClass，即 torch.autograd.Function 对应的 Python 类
// 注意：此处可能会产生内存泄漏
THPFunctionClass = PyObject_GetAttrString(autograd_module, "Function");
if (!THPFunctionClass)
  return nullptr;

// 导入并初始化 torch.autograd.graph 模块，如果失败则返回 nullptr
auto autograd_graph_mod = THPObjectPtr(PyImport_ImportModule("torch.autograd.graph"));
THPGradientEdgeClass = PyObject_GetAttrString(autograd_graph_mod, "GradientEdge");
if (!THPGradientEdgeClass)
  return nullptr;

// 导入并初始化 torch._C 模块，如果失败则返回 nullptr
auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
if (!torch_C_module)
  return nullptr;

// 在 torch._C 模块中定义子模块 "_autograd"，用于自动微分绑定
auto _C_m = py::handle(torch_C_module).cast<py::module>();
auto m = _C_m.def_submodule("_autograd", "autograd bindings");

// 导入并初始化 torch.nn.parameter 模块，如果失败则返回 nullptr
auto parameter_module = THPObjectPtr(PyImport_ImportModule("torch.nn.parameter"));
if (!parameter_module)
  return nullptr;

// 获取 ParameterClass，即 torch.nn.parameter.Parameter 对应的 Python 类
// 注意：此处可能会产生内存泄漏
ParameterClass = PyObject_GetAttrString(parameter_module, "Parameter");
if (!ParameterClass)
#ifdef USE_KINETO
    .def("save", &ProfilerResult::save)
  // 结束条件：如果未定义 USE_KINETO，则直接结束，无需执行任何操作
#endif // USE_KINETO
      ;

  // 定义 Python 绑定的函数 "_enable_profiler"，接受 config、activities 和 scopes 参数
  m.def(
      "_enable_profiler",
      &enableProfiler,
      py::arg("config"),
      py::arg("activities"),
      py::arg("scopes") = std::unordered_set<at::RecordScope>());

  // 定义 Python 绑定的函数 "_disable_profiler"，用于禁用分析器
  m.def("_disable_profiler", disableProfiler);

  // 定义 Python 绑定的函数 "_prepare_profiler"，调用 prepareProfiler 函数
  // 使用 py::gil_scoped_release 调用保护
  m.def(
      "_prepare_profiler",
      prepareProfiler,
      py::call_guard<py::gil_scoped_release>());

  // 如果定义了 USE_KINETO，则定义 Python 绑定的函数 "_add_metadata_json"，用于添加元数据 JSON
  m.def("_add_metadata_json", addMetadataJson); // Only if `USE_KINETO` is set

  // 如果定义了 USE_KINETO，则定义 Python 绑定的函数 "_kineto_step"，用于执行 Kineto 分步
  m.def("_kineto_step", profilerStep); // Only if `USE_KINETO` is set

  // 定义 Python 函数 "kineto_available"，返回 Kineto 是否可用的状态
  m.def("kineto_available", []() { return torch::profiler::kKinetoAvailable; });

  // 注意事项：这些记录函数不是 Torch 操作符，可能不会出现在 TorchScript 追踪、FX 转换或操作符序列化中。
  // 对于这些用例，请使用 `torch.profiler.record_function`。
  // 创建新的使用 RecordFunction 的分析作用域，并调用其起始回调。
  m.def(
      "_record_function_with_args_enter",
      [](const std::string& name, const py::args& args) {
        using torch::autograd::profiler::PythonRecordFunction;
        auto python_rec = c10::make_intrusive<PythonRecordFunction>(
            at::RecordScope::USER_SCOPE);
        auto* rec = &python_rec->record;
        if (rec->isActive()) {
          if (rec->needsInputs()) {
            auto iv_inputs = std::vector<c10::IValue>();
            for (const auto& arg : args) {
              iv_inputs.push_back(torch::jit::toTypeInferredIValue(arg));
            }
            rec->before(
                name,
                c10::ArrayRef<const c10::IValue>(
                    iv_inputs.data(), iv_inputs.size()));
          } else {
            rec->before(name);
          }
        }
        return torch::jit::toPyObject(std::move(python_rec));
      });

  // 结束使用 record_function_with_param_enter 创建的分析作用域。
  m.def("_record_function_with_args_exit", [](const py::object& obj) {
    using torch::autograd::profiler::PythonRecordFunction;
    auto python_record = torch::jit::toCustomClass<PythonRecordFunction>(obj);

    // 实际上我们不需要对 handle 做任何事情，只需要保持它的生命周期直到现在。
    python_record->record.end();
  });

  // 定义函数 "_supported_activities"，返回支持的活动类型集合
  m.def("_supported_activities", []() {
    std::set<torch::profiler::impl::ActivityType> activities{
        torch::profiler::impl::ActivityType::CPU};

    // 如果定义了 USE_KINETO，并且未定义 LIBKINETO_NOCUPTI 或 LIBKINETO_NOROCTRACER，则添加 CUDA 活动类型
#if defined(USE_KINETO) && \
    (!defined(LIBKINETO_NOCUPTI) || !defined(LIBKINETO_NOROCTRACER))
    if (at::getNumGPUs() > 0) {
      activities.insert(torch::profiler::impl::ActivityType::CUDA);
    }
// 如果定义了 USE_KINETO，则根据系统支持情况添加 XPU、MTIA 和 PrivateUse1 活动类型
#elif defined(USE_KINETO)
    if (at::hasXPU()) {
      activities.insert(torch::profiler::impl::ActivityType::XPU);
    }
    if (at::hasMTIA()) {
      activities.insert(torch::profiler::impl::ActivityType::MTIA);
    }
    if (c10::get_privateuse1_backend() != "privateuseone") {
      activities.insert(torch::profiler::impl::ActivityType::PrivateUse1);
    }
#endif
  return activities;
});

m.def("_unsafe_set_version_counter", [](const at::Tensor& t, int64_t i) {
  // 获取张量 t 的版本计数器
  auto vc = torch::autograd::impl::version_counter(t);
  // 设置版本计数器的版本号为 i
  vc.set_version(i);
});

m.def("_enable_profiler_legacy", enableProfilerLegacy);
py::class_<ProfilerDisableOptions>(m, "_ProfilerDisableOptions")
    .def(py::init<bool, bool>());
m.def(
    "_disable_profiler_legacy",
    disableProfilerLegacy,
    py::arg("profiler_disable_options") = ProfilerDisableOptions());
m.def("_profiler_enabled", profilerEnabled);
m.def("_profiler_type", torch::profiler::impl::profilerType);
m.def("_enable_record_function", [](bool enable) {
  // 开启或关闭记录函数
  at::enableRecordFunction(enable);
});
m.def("_set_empty_test_observer", [](bool is_global, double sampling_prob) {
  // 设置空的测试观察者回调函数
  auto cb =
      at::RecordFunctionCallback(nullptr).needsInputs(true).samplingProb(
          sampling_prob);
  if (is_global) {
    // 如果是全局的，则添加全局回调函数
    at::addGlobalCallback(cb);
  } else {
    // 否则添加线程局部的回调函数
    at::addThreadLocalCallback(cb);
  }
});
m.def("_clear_callbacks", []() { 
  // 清除所有回调函数
  at::clearCallbacks(); 
});
m.def(
    "_saved_tensors_hooks_is_enabled",
    at::SavedTensorDefaultHooks::is_enabled);
m.def("_saved_tensors_hooks_enable", at::SavedTensorDefaultHooks::enable);
m.def("_saved_tensors_hooks_disable", at::SavedTensorDefaultHooks::disable);
m.def(
    "_saved_tensors_hooks_set_tracing",
    at::SavedTensorDefaultHooks::set_tracing);
m.def(
    "_saved_tensors_hooks_get_disabled_error_message",
    at::SavedTensorDefaultHooks::get_disabled_error_message);
m.def(
    "_push_saved_tensors_default_hooks",
    [](py::function& pack_hook, py::function& unpack_hook) {
      // 压入默认的张量保存钩子函数
      torch::autograd::PyDefaultSavedVariableHooks::push_hooks(
          pack_hook, unpack_hook);
    });
m.def("_pop_saved_tensors_default_hooks", []() {
  // 弹出默认的张量保存钩子函数
  torch::autograd::PyDefaultSavedVariableHooks::pop_hooks();
});

m.def("_get_creation_meta", [](const at::Tensor& t) {
  // 获取张量 t 的视图自动求导元数据
  auto* meta = torch::autograd::impl::get_view_autograd_meta(t);
  TORCH_CHECK(meta != nullptr);
  return meta->get_creation_meta();
});

m.def(
    "_set_creation_meta",
    [](const at::Tensor& t, CreationMeta new_creation_meta) {
      // 设置张量 t 的视图自动求导元数据的创建元数据
      auto* meta = torch::autograd::impl::get_view_autograd_meta(t);
      TORCH_CHECK(meta != nullptr);
      meta->set_creation_meta(new_creation_meta);
    });

_C_m.def(
    "_register_py_class_for_device",
    [](const std::string& device, py::object python_type_class) {
      // 注册 Python 类型的张量类到指定设备
      auto cls = python_type_class.ptr();
      registerPythonTensorClass(device, cls);
    });
_C_m.def("_set_autograd_fallback_mode", [](const std::string& mode) {
  if (mode == "nothing") {
    // 设置自动求导的回退模式为什么也不做
    torch::autograd::setAutogradFallbackMode(
        torch::autograd::AutogradFallbackMode::Nothing);
    return;
  }
  if (mode == "warn") {
    // 设置自动求导的回退模式为警告
    torch::autograd::setAutogradFallbackMode(
        torch::autograd::AutogradFallbackMode::Warn);
    return;
  }
});
    // 如果模式为 "error"，设置自动求导回退模式为错误并返回
    if (mode == "error") {
      torch::autograd::setAutogradFallbackMode(
          torch::autograd::AutogradFallbackMode::Error);
      return;
    }
    // 否则，断言失败并输出不支持的自动求导回退模式
    TORCH_INTERNAL_ASSERT(false, "Unsupported AutogradFallbackMode: ", mode);
  });

  // 定义一个函数 "_get_autograd_fallback_mode"，返回当前的自动求导回退模式
  _C_m.def("_get_autograd_fallback_mode", []() {
    auto mode = torch::autograd::getAutogradFallbackMode();
    switch (mode) {
      case torch::autograd::AutogradFallbackMode::Nothing:
        return "nothing";
      case torch::autograd::AutogradFallbackMode::Warn:
        return "warn";
      case torch::autograd::AutogradFallbackMode::Error:
        return "error";
      default:
        // 断言失败，输出不支持的自动求导回退模式
        TORCH_INTERNAL_ASSERT(false, "Unsupported AutogradFallbackMode");
    }
  });

  // 定义一个函数 "_activate_gpu_trace"，激活 GPU 追踪
  _C_m.def("_activate_gpu_trace", []() { activateGPUTrace(); });

  // 使用上下文管理器在 Python 中注册 "_InferenceMode"
  py_context_manager_DEPRECATED<c10::InferenceMode, bool>(
      _C_m, "_InferenceMode");

  // 使用上下文管理器在 Python 中注册 "_RestorePythonTLSSnapshot"
  py_context_manager<at::impl::RestorePythonTLSSnapshot>(
      _C_m, "_RestorePythonTLSSnapshot");

  // 使用上下文管理器在 Python 中注册 "_DisableTorchDispatch"（已废弃）
  py_context_manager_DEPRECATED<torch::DisableTorchDispatch>(
      _C_m, "_DisableTorchDispatch");

  // 使用上下文管理器在 Python 中注册 "_EnableTorchFunction"（已废弃）
  py_context_manager_DEPRECATED<EnableTorchFunction>(
      _C_m, "_EnableTorchFunction");

  // 使用上下文管理器在 Python 中注册 "_EnablePythonDispatcher"（已废弃）
  py_context_manager_DEPRECATED<EnablePythonDispatcher>(
      _C_m, "_EnablePythonDispatcher");

  // 使用上下文管理器在 Python 中注册 "_DisablePythonDispatcher"
  py_context_manager<c10::impl::DisablePythonDispatcher>(
      _C_m, "_DisablePythonDispatcher");

  // 使用上下文管理器在 Python 中注册 "_EnablePreDispatch"
  py_context_manager<EnablePreDispatch>(_C_m, "_EnablePreDispatch");

  // 使用上下文管理器在 Python 中注册 "_DisableFuncTorch"（已废弃）
  py_context_manager_DEPRECATED<DisableFuncTorch>(_C_m, "_DisableFuncTorch");

  // 使用上下文管理器在 Python 中注册 "_DisableAutocast"
  py_context_manager<DisableAutocast>(_C_m, "_DisableAutocast");

  // 在 Python 中定义名为 "SavedTensor" 的类，注册其构造函数和方法
  py::class_<torch::autograd::SavedVariable>(std::move(m), "SavedTensor")
      .def(py::init([]() -> torch::autograd::SavedVariable {
        // 禁止从 Python 中创建 SavedTensor 对象，触发断言失败
        TORCH_CHECK(
            false,
            "Trying to create a SavedTensor object from Python is forbidden.");
      }))
      .def(
          "register_hooks",
          [](torch::autograd::SavedVariable& s,
             py::function& pack_hook,
             py::function& unpack_hook) {
            // 注册钩子函数，使用 py::object ，pybind 会自动增加引用计数
            s.register_hooks(
                std::make_unique<torch::autograd::PySavedVariableHooks>(
                    pack_hook, unpack_hook));
          });

  // 初始化 Python 追踪器的配置
  torch::autograd::profiler::python_tracer::init();
  
  // 返回 Python 中的 True
  Py_RETURN_TRUE;
// 定义一个静态函数，用于设置自动类型转换的开启状态
static PyObject* set_autocast_enabled(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，支持两种不同的调用方式
  static PythonArgParser parser(
      {"set_autocast_enabled(c10::string_view device_type, bool enabled)",
       "set_autocast_enabled(bool enabled)"}); // this signature is depracated.
  ParsedArgs<2> parsed_args;
  // 解析传入的参数
  auto r = parser.parse(args, kwargs, parsed_args);
  // 默认将 at::kCUDA 设为设备类型，以防止破坏现有代码兼容性
  at::DeviceType device_type = at::kCUDA;
  int enabled_id = 0;
  // 根据解析结果选择合适的参数索引
  if (r.idx == 0) {
    device_type = at::Device(r.string(0)).type();
    enabled_id = 1;
  }
  // 获取 bool 值，表示是否开启自动类型转换
  auto enabled = r.toBool(enabled_id);
  // 调用 Torch 的自动类型转换设置函数
  at::autocast::set_autocast_enabled(device_type, enabled);
  // 返回 Python 中的 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}



// 定义一个静态函数，用于查询指定设备上是否开启了自动类型转换
static PyObject* is_autocast_enabled(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，支持两种不同的调用方式
  static PythonArgParser parser(
      {"is_autocast_enabled(c10::string_view device_type)",
       "is_autocast_enabled()"}); // this signature is depracated.
  ParsedArgs<1> parsed_args;
  // 解析传入的参数
  auto r = parser.parse(args, kwargs, parsed_args);
  // 默认将 at::kCUDA 设为设备类型，以防止破坏现有代码兼容性
  at::DeviceType device_type = at::kCUDA;
  // 根据解析结果选择合适的参数索引
  if (r.idx == 0) {
    device_type = at::Device(r.string(0)).type();
  }
  // 查询指定设备上是否开启了自动类型转换，并返回相应的 Python 布尔值
  if (at::autocast::is_autocast_enabled(device_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}



// 定义一个静态函数，用于获取指定设备上当前的自动类型转换的数据类型
static PyObject* get_autocast_dtype(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，支持一种调用方式
  static PythonArgParser parser(
      {"get_autocast_dtype(c10::string_view device_type)"});
  ParsedArgs<1> parsed_args;
  // 解析传入的参数
  auto r = parser.parse(args, kwargs, parsed_args);
  // 获取设备类型
  auto device_type = at::Device(r.string(0)).type();
  // 获取指定设备上当前的自动类型转换的数据类型
  at::ScalarType current_dtype = at::autocast::get_autocast_dtype(device_type);
  // 将获取到的数据类型包装成 Python 对象返回
  return utils::wrap(current_dtype);
  END_HANDLE_TH_ERRORS
}



// 定义一个静态函数，用于设置指定设备上的自动类型转换的数据类型
static PyObject* set_autocast_dtype(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，支持一种调用方式
  static PythonArgParser parser(
      {"set_autocast_dtype(c10::string_view device_type, ScalarType dtype)"});
  ParsedArgs<2> parsed_args;
  // 解析传入的参数
  auto r = parser.parse(args, kwargs);
  // 获取设备类型
  auto device_type = at::Device(r.string(0)).type();
  // 获取自动类型转换的目标数据类型
  auto dtype = r.scalartype(1);
  // 设置指定设备上的自动类型转换的数据类型
  at::autocast::set_autocast_dtype(device_type, dtype);
  // 返回 Python 中的 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}



// 定义一个静态函数，用于检查任意设备上是否开启了自动类型转换
static PyObject* is_any_autocast_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查是否有任意设备开启了自动类型转换
  if (at::autocast::is_autocast_enabled(at::kCPU) ||
      at::autocast::is_autocast_enabled(at::kCUDA) ||
      at::autocast::is_autocast_enabled(at::kXPU) ||
      at::autocast::is_autocast_enabled(at::kIPU) ||
      at::autocast::is_autocast_enabled(at::kXLA) ||
      at::autocast::is_autocast_enabled(at::kHPU) ||
      at::autocast::is_autocast_enabled(at::kPrivateUse1)) {
    // 如果有设备开启了自动类型转换，则返回 Python 中的 True
    Py_RETURN_TRUE;
  } else {
    // 如果没有设备开启自动类型转换，则返回 Python 中的 False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}
    // 返回 Python 中的 False 对象，表示函数执行失败
    Py_RETURN_FALSE;
  }
  // 结束错误处理块
  END_HANDLE_TH_ERRORS
// 检查是否自动类型转换功能在指定设备类型上可用
static PyObject* is_autocast_available(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 创建 PythonArgParser 对象，用于解析参数列表和关键字参数
  static PythonArgParser parser(
      {"_is_autocast_available(c10::string_view device_type)"});
  // 解析传入的参数，期望一个字符串参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // 获取设备类型字符串并转换为 ATen 中的设备类型
  auto device_type = at::Device(r.string(0)).type();
  // 检查给定设备类型上是否支持自动类型转换，返回相应的 Python 值
  if (at::autocast::is_autocast_available(device_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// 设置 CPU 上的自动类型转换功能是否启用
static PyObject* set_autocast_cpu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数类型是否为布尔型
  TORCH_CHECK_TYPE(
      PyBool_Check(arg),
      "enabled must be a bool (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  // 发出警告，提示该函数已经废弃，推荐使用新的函数替代
  TORCH_WARN_DEPRECATION(
      "torch.set_autocast_cpu_enabled(enabled) is deprecated. Please use torch.set_autocast_enabled('cpu', enabled) instead.")
  // 设置 CPU 上的自动类型转换功能是否启用
  at::autocast::set_autocast_enabled(at::kCPU, arg == Py_True);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 检查 CPU 上的自动类型转换功能是否启用
static PyObject* is_autocast_cpu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 发出警告，提示该函数已经废弃，推荐使用新的函数替代
  TORCH_WARN_DEPRECATION(
      "torch.is_autocast_cpu_enabled() is deprecated. Please use torch.is_autocast_enabled('cpu') instead.")
  // 检查 CPU 上的自动类型转换功能是否启用，返回相应的 Python 值
  if (at::autocast::is_autocast_enabled(at::kCPU)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// 设置 IPU 上的自动类型转换功能是否启用
static PyObject* set_autocast_ipu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数类型是否为布尔型
  TORCH_CHECK_TYPE(
      PyBool_Check(arg),
      "enabled must be a bool (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  // 发出警告，提示该函数已经废弃，推荐使用新的函数替代
  TORCH_WARN_DEPRECATION(
      "torch.set_autocast_ipu_enabled(enabled) is deprecated. Please use torch.set_autocast_enabled('ipu', enabled) instead.")
  // 设置 IPU 上的自动类型转换功能是否启用
  at::autocast::set_autocast_enabled(at::kIPU, arg == Py_True);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 检查 IPU 上的自动类型转换功能是否启用
static PyObject* is_autocast_ipu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 发出警告，提示该函数已经废弃，推荐使用新的函数替代
  TORCH_WARN_DEPRECATION(
      "torch.is_autocast_ipu_enabled() is deprecated. Please use torch.is_autocast_enabled('ipu') instead.")
  // 检查 IPU 上的自动类型转换功能是否启用，返回相应的 Python 值
  if (at::autocast::is_autocast_enabled(at::kIPU)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// 设置 XLA 上的自动类型转换功能是否启用
static PyObject* set_autocast_xla_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数类型是否为布尔型
  TORCH_CHECK_TYPE(
      PyBool_Check(arg),
      "enabled must be a bool (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  // 发出警告，提示该函数已经废弃，推荐使用新的函数替代
  TORCH_WARN_DEPRECATION(
      "torch.set_autocast_xla_enabled(enabled) is deprecated. Please use torch.set_autocast_enabled('xla', enabled) instead.")
  // 设置 XLA 上的自动类型转换功能是否启用
  at::autocast::set_autocast_enabled(at::kXLA, arg == Py_True);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 检查 XLA 上的自动类型转换功能是否启用
static PyObject* is_autocast_xla_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 发出警告，提示该函数已经废弃，推荐使用新的函数替代
  TORCH_WARN_DEPRECATION(
      "torch.is_autocast_xla_enabled() is deprecated. Please use torch.is_autocast_enabled('xla') instead.")
  // 检查 XLA 上的自动类型转换功能是否启用，返回相应的 Python 值
  if (at::autocast::is_autocast_enabled(at::kXLA)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}
    // 返回 Python 中的 False 值，这是一个宏，用于快速返回 False
    Py_RETURN_FALSE;
  }
  // 结束异常处理块，此处会捕获并处理所有由 TH 库抛出的异常
  END_HANDLE_TH_ERRORS
static PyObject* set_autocast_gpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数类型是否为 torch.dtype
  TORCH_CHECK_TYPE(
      THPDtype_Check(arg),
      "dtype must be a torch.dtype (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  // 发出警告信息，提醒用户该函数已经废弃
  TORCH_WARN_DEPRECATION(
      "torch.set_autocast_gpu_dtype(dtype) is deprecated. Please use torch.set_autocast_dtype('cuda', dtype) instead.")
  // 从传入的参数中获取目标数据类型
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  // 设置 GPU 自动混合精度的数据类型
  at::autocast::set_autocast_dtype(at::kCUDA, targetType);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_cpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数类型是否为 torch.dtype
  TORCH_CHECK_TYPE(
      THPDtype_Check(arg),
      "dtype must be a torch.dtype (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  // 发出警告信息，提醒用户该函数已经废弃
  TORCH_WARN_DEPRECATION(
      "torch.set_autocast_cpu_dtype(dtype) is deprecated. Please use torch.set_autocast_dtype('cpu', dtype) instead.")
  // 从传入的参数中获取目标数据类型
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  // 设置 CPU 自动混合精度的数据类型
  at::autocast::set_autocast_dtype(at::kCPU, targetType);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_ipu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数类型是否为 torch.dtype
  TORCH_CHECK_TYPE(
      THPDtype_Check(arg),
      "dtype must be a torch.dtype (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  // 发出警告信息，提醒用户该函数已经废弃
  TORCH_WARN_DEPRECATION(
      "torch.set_autocast_ipu_dtype(dtype) is deprecated. Please use torch.set_autocast_dtype('ipu', dtype) instead.")
  // 从传入的参数中获取目标数据类型
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  // 设置 IPU 自动混合精度的数据类型
  at::autocast::set_autocast_dtype(at::kIPU, targetType);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_xla_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数类型是否为 torch.dtype
  TORCH_CHECK_TYPE(
      THPDtype_Check(arg),
      "dtype must be a torch.dtype (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  // 发出警告信息，提醒用户该函数已经废弃
  TORCH_WARN_DEPRECATION(
      "torch.set_autocast_xla_dtype(dtype) is deprecated. Please use torch.set_autocast_dtype('xla', dtype) instead.")
  // 从传入的参数中获取目标数据类型
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  // 设置 XLA 自动混合精度的数据类型
  at::autocast::set_autocast_dtype(at::kXLA, targetType);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_gpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 发出警告信息，提醒用户该函数已经废弃
  TORCH_WARN_DEPRECATION(
      "torch.get_autocast_gpu_dtype() is deprecated. Please use torch.get_autocast_dtype('cuda') instead.")
  // 获取当前 GPU 自动混合精度的数据类型
  at::ScalarType current_dtype = at::autocast::get_autocast_dtype(at::kCUDA);
  // 将数据类型包装为 Python 对象并返回
  return utils::wrap(current_dtype);
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_cpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 发出警告信息，提醒用户该函数已经废弃
  TORCH_WARN_DEPRECATION(
      "torch.get_autocast_cpu_dtype() is deprecated. Please use torch.get_autocast_dtype('cpu') instead.")
  // 获取当前 CPU 自动混合精度的数据类型
  at::ScalarType current_dtype = at::autocast::get_autocast_dtype(at::kCPU);
  // 将数据类型包装为 Python 对象并返回
  return utils::wrap(current_dtype);
  END_HANDLE_TH_ERRORS
}
static PyObject* get_autocast_ipu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 发出警告信息，提示该函数已被弃用，请使用新的函数替代
  TORCH_WARN_DEPRECATION(
      "torch.get_autocast_ipu_dtype() is deprecated. Please use torch.get_autocast_dtype('ipu') instead.")
  // 获取当前IPU设备的自动混合精度数据类型
  at::ScalarType current_dtype = at::autocast::get_autocast_dtype(at::kIPU);
  // 将当前数据类型封装成Python对象并返回
  return utils::wrap(current_dtype);
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_xla_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 发出警告信息，提示该函数已被弃用，请使用新的函数替代
  TORCH_WARN_DEPRECATION(
      "torch.get_autocast_xla_dtype() is deprecated. Please use torch.get_autocast_dtype('xla') instead.")
  // 获取当前XLA设备的自动混合精度数据类型
  at::ScalarType current_dtype = at::autocast::get_autocast_dtype(at::kXLA);
  // 将当前数据类型封装成Python对象并返回
  return utils::wrap(current_dtype);
  END_HANDLE_TH_ERRORS
}

static PyObject* clear_autocast_cache(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS {
    // 释放全局解释器锁，允许在无GIL状态下执行代码
    pybind11::gil_scoped_release no_gil;
    // 清除自动混合精度的缓存
    at::autocast::clear_cache();
  }
  // 返回Python中的None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* autocast_increment_nesting(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 增加自动混合精度的嵌套层数，并将结果封装成Python的整数对象返回
  return THPUtils_packInt64(at::autocast::increment_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject* autocast_decrement_nesting(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 减少自动混合精度的嵌套层数，并将结果封装成Python的整数对象返回
  return THPUtils_packInt64(at::autocast::decrement_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_cache_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查自动混合精度缓存是否启用，返回Python中的True或False对象
  if (at::autocast::is_autocast_cache_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_cache_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数类型是否为布尔型
  TORCH_CHECK_TYPE(
      PyBool_Check(arg),
      "enabled must be a bool (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  // 设置自动混合精度缓存是否启用
  at::autocast::set_autocast_cache_enabled(arg == Py_True);
  // 返回Python中的None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_grad_enabled(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义Python函数参数的解析器
  static PythonArgParser parser({
      "set_grad_enabled(bool enabled)",
  });
  // 解析传入的参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  // 如果启用了torch函数模式，则调用相关处理函数
  if (at::impl::torch_function_mode_enabled()) {
    auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
    return handle_torch_function(
        r, args, kwargs, torch_C_module, "torch._C", "_set_grad_enabled");
  }
  // 否则设置梯度是否启用
  auto grad_enabled = r.toBool(0);
  GradMode::set_enabled(grad_enabled);
  // 返回Python中的None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_grad_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查梯度是否启用，并返回Python中的True或False对象
  if (GradMode::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}
# 设置是否启用前向梯度的函数
static PyObject* set_fwd_grad_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  # 检查参数类型是否为布尔型
  TORCH_CHECK_TYPE(
      PyBool_Check(arg),
      "enabled must be a bool (got ",
      Py_TYPE(arg)->tp_name,
      ")");
  # 设置当前线程的前向梯度模式
  c10::AutogradState::get_tls_state().set_fw_grad_mode(arg == Py_True);
  # 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

# 查询是否启用前向梯度的函数
static PyObject* is_fwd_grad_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  # 如果前向梯度模式已启用，则返回 True
  if (c10::AutogradState::get_tls_state().get_fw_grad_mode()) {
    Py_RETURN_TRUE;
  } else {
    # 否则返回 False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

# 设置是否启用多线程的函数
static PyObject* set_multithreading_enabled(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  # 定义参数解析器
  static PythonArgParser parser({
      "set_multithreading_enabled(bool enabled)",
  });
  ParsedArgs<1> parsed_args;
  # 解析参数
  auto r = parser.parse(args, kwargs, parsed_args);

  # 如果启用了 torch function 模式
  if (at::impl::torch_function_mode_enabled()) {
    # 导入 torch._C 模块并处理 torch function
    auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
    return handle_torch_function(
        r,
        args,
        kwargs,
        torch_C_module,
        "torch._C",
        "_set_multithreading_enabled");
  }

  # 否则设置多线程模式的状态
  auto multithreading_enabled = r.toBool(0);
  c10::AutogradState::get_tls_state().set_multithreading_enabled(
      multithreading_enabled);
  # 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

# 查询是否启用多线程的函数
static PyObject* is_multithreading_enabled(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  # 如果多线程模式已启用，则返回 True
  if (c10::AutogradState::get_tls_state().get_multithreading_enabled()) {
    Py_RETURN_TRUE;
  } else {
    # 否则返回 False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

# 设置是否启用视图重播的函数
static PyObject* set_view_replay_enabled(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  # 定义参数解析器
  static PythonArgParser parser({
      "set_view_replay_enabled(bool enabled)",
  });
  ParsedArgs<1> parsed_args;
  # 解析参数
  auto r = parser.parse(args, kwargs, parsed_args);

  # 如果启用了 torch function 模式
  if (at::impl::torch_function_mode_enabled()) {
    # 导入 torch._C 模块并处理 torch function
    auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
    return handle_torch_function(
        r,
        args,
        kwargs,
        torch_C_module,
        "torch._C",
        "_set_view_replay_enabled");
  }

  # 否则设置视图重播模式的状态
  auto view_replay_enabled = r.toBool(0);
  c10::AutogradState::get_tls_state().set_view_replay_enabled(
      view_replay_enabled);
  # 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

# 查询是否启用视图重播的函数
static PyObject* is_view_replay_enabled(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  # 如果视图重播模式已启用，则返回 True
  if (c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    Py_RETURN_TRUE;
  } else {
    # 否则返回 False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

# 查询是否启用推断模式的函数
static PyObject* is_inference_mode_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  # 如果推断模式已启用，则返回 True
  if (c10::InferenceMode::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    # 否则返回 False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

# 设置是否启用异常模式的函数
static PyObject* set_anomaly_mode_enabled(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
    PyObject* kwargs) {  // 定义一个函数，接受Python对象和关键字参数
      HANDLE_TH_ERRORS  // 处理 Torch 框架的错误和异常
    
      // 静态 PythonArgParser 对象，用于解析参数
      static PythonArgParser parser({
          "set_anomaly_enabled(bool enabled, bool check_nan=True)",
      });
    
      // 解析函数参数并获取解析结果
      ParsedArgs<2> parsed_args;
      auto r = parser.parse(args, kwargs, parsed_args);
    
      // 调用 AnomalyMode 类的静态方法 set_enabled，设置异常检测模式
      AnomalyMode::set_enabled(r.toBool(0), r.toBool(1));
    
      // 返回 Python 的 None 对象
      Py_RETURN_NONE;
    
      // 结束 Torch 框架错误处理
      END_HANDLE_TH_ERRORS
    }
static PyObject* is_anomaly_mode_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查异常模式是否启用
  if (AnomalyMode::is_enabled()) {
    // 如果异常模式启用，返回Python中的True
    Py_RETURN_TRUE;
  } else {
    // 如果异常模式未启用，返回Python中的False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* is_anomaly_check_nan_enabled(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查是否应该检测NaN异常
  if (AnomalyMode::should_check_nan()) {
    // 如果应该检测NaN异常，返回Python中的True
    Py_RETURN_TRUE;
  } else {
    // 如果不需要检测NaN异常，返回Python中的False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* python_enter_dual_level(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 由于前向嵌套的深度溢出int64_t的可能性极低，因此在此进行静态类型转换
  return utils::wrap(static_cast<int64_t>(forward_ad::enter_dual_level()));
  END_HANDLE_TH_ERRORS
}

static PyObject* python_exit_dual_level(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"exit_dual_level(int64_t level)"});

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  auto idx = _r.toInt64(0);
  // 确保给定的索引是有效的，然后进行类型转换
  TORCH_CHECK(idx >= 0, "Dual level must be a positive number.");
  forward_ad::exit_dual_level(static_cast<uint64_t>(idx));
  // 返回Python中的None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_torch_function_mode_enabled(
    PyObject* _unused,
    PyObject* _unused2) {
  HANDLE_TH_ERRORS
  // 检查Torch函数模式是否启用
  if (at::impl::torch_function_mode_enabled()) {
    // 如果Torch函数模式启用，返回Python中的True
    Py_RETURN_TRUE;
  } else {
    // 如果Torch函数模式未启用，返回Python中的False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* push_on_torch_function_stack(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 如果参数不是None，则增加其引用计数，并将其推入Torch函数堆栈
  if (arg != Py_None) {
    Py_INCREF(arg);
    at::impl::PythonTorchFunctionTLS::push_onto_stack(
        std::make_shared<c10::SafePyObject>(arg, getPyInterpreter()));
  }
  // 返回Python中的None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* pop_torch_function_stack(
    PyObject* _unused,
    PyObject* _unused2) {
  HANDLE_TH_ERRORS
  // 弹出Torch函数堆栈中的模式，并返回其指针
  const auto& mode = at::impl::PythonTorchFunctionTLS::pop_stack();
  auto* r = mode->ptr(getPyInterpreter());
  Py_INCREF(r);
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_function_stack_at(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"get_stack_at(int64_t level)"});

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  auto idx = _r.toInt64(0);
  // 获取Torch函数堆栈中特定索引处的模式，并返回其指针
  const auto& mode = at::impl::PythonTorchFunctionTLS::get_stack_at(idx);
  auto* r = mode->ptr(getPyInterpreter());
  Py_INCREF(r);
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* len_torch_function_stack(
    PyObject* _unused,
    PyObject* _unused2) {
  HANDLE_TH_ERRORS
  // 获取Torch函数堆栈的长度，并将其转换为int64_t类型后返回
  const auto len = at::impl::PythonTorchFunctionTLS::stack_len();
  return utils::wrap(static_cast<int64_t>(len));
  END_HANDLE_TH_ERRORS
}

static PyObject* push_on_torch_dispatch_stack(
    PyObject* _unused,
    //
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否不为空
  if (arg != Py_None) {
    // 引入 TorchDispatchModeKey 类型的别名
    using c10::impl::TorchDispatchModeKey;
    
    // 尝试获取参数对象的 _mode_key 属性
    std::optional<c10::impl::TorchDispatchModeKey> mode_key = c10::nullopt;
    py::object maybe_mode_key_obj =
        PyObject_FastGetAttrString(arg, "_mode_key");
    
    // 如果成功获取到 _mode_key 属性
    if (maybe_mode_key_obj) {
      // 将获取到的 _mode_key 属性转换为 TorchDispatchModeKey 类型
      mode_key = py::cast<c10::impl::TorchDispatchModeKey>(maybe_mode_key_obj);
      
      // 设置 TorchDispatchModeTLS 的模式
      c10::impl::TorchDispatchModeTLS::set_mode(
          // 创建 PyObject_TorchDispatchMode 的共享指针
          std::make_shared<c10::impl::PyObject_TorchDispatchMode>(
              arg, getPyInterpreter()),
          // 使用获取到的 _mode_key 属性作为参数
          mode_key.value());
    } else {
      // 将非 infra 模式推入模式栈
      c10::impl::TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
          // 创建 PyObject_TorchDispatchMode 的共享指针
          std::make_shared<c10::impl::PyObject_TorchDispatchMode>(
              arg, getPyInterpreter()));
    }
    
    // 增加参数对象的引用计数
    Py_INCREF(arg);
  }
  
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject* pop_torch_dispatch_stack(
    PyObject* _unused,
    PyObject* maybe_mode_key) {
  HANDLE_TH_ERRORS
  std::optional<c10::impl::TorchDispatchModeKey> mode_key = c10::nullopt;
  PyObject* r = nullptr;
  if (maybe_mode_key != Py_None) {
    // 将 Python 对象转换为 TorchDispatchModeKey 类型
    mode_key = py::cast<c10::impl::TorchDispatchModeKey>(maybe_mode_key);
    // 尝试取消指定的 Torch 分发模式
    auto maybe_mode =
        c10::impl::TorchDispatchModeTLS::unset_mode(mode_key.value());
    // 检查是否成功取消模式，并报错处理
    TORCH_CHECK(
        maybe_mode.has_value(),
        "Attempted to unset ",
        c10::impl::to_string(mode_key.value()),
        ", but there wasn't one active.");
    auto mode = maybe_mode.value();
    // 获取模式对应的 Python 对象，并增加其引用计数
    r = mode->ptr(getPyInterpreter());
  } else {
    // 从 Torch 分发模式栈中弹出模式
    auto mode = c10::impl::TorchDispatchModeTLS::pop_stack();
    // 获取模式对应的 Python 对象，并增加其引用计数
    r = mode->ptr(getPyInterpreter());
  }
  // 增加返回值的引用计数
  Py_INCREF(r);
  // 返回 Python 对象
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_dispatch_stack_at(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"get_stack_at(int64_t level)"});

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // 解析传入的参数，获取栈的索引值
  auto idx = _r.toInt64(0);
  // 获取指定索引处的 Torch 分发模式
  const auto& mode = c10::impl::TorchDispatchModeTLS::get_stack_at(idx);
  // 获取模式对应的 Python 对象，并增加其引用计数
  auto* r = mode->ptr(getPyInterpreter());
  // 增加返回值的引用计数
  Py_INCREF(r);
  // 返回 Python 对象
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_dispatch_mode(PyObject* _unused, PyObject* mode) {
  HANDLE_TH_ERRORS
  // 检查传入的模式对象不为 None
  TORCH_CHECK(mode != Py_None);

  // 从模式对象中获取 _mode_key 属性
  py::object maybe_mode_key_obj = PyObject_FastGetAttrString(mode, "_mode_key");
  // 检查获取属性是否成功
  TORCH_CHECK(
      maybe_mode_key_obj,
      "set_dispatch_mode() called with a mode that does not contain a _mode_key attribute!");
  // 将 _mode_key 属性转换为 TorchDispatchModeKey 类型
  auto mode_key = py::cast<c10::impl::TorchDispatchModeKey>(maybe_mode_key_obj);

  // 增加模式对象的引用计数
  Py_INCREF(mode);
  // 设置 Torch 分发模式，并关联 Python 对象
  c10::impl::TorchDispatchModeTLS::set_mode(
      std::make_shared<c10::impl::PyObject_TorchDispatchMode>(
          mode, getPyInterpreter()),
      mode_key);

  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_dispatch_mode(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数不为 None
  TORCH_CHECK(arg != Py_None);
  // 将参数转换为 TorchDispatchModeKey 类型
  auto mode_key = py::cast<c10::impl::TorchDispatchModeKey>(arg);

  // 获取指定的 Torch 分发模式
  auto maybe_mode = c10::impl::TorchDispatchModeTLS::get_mode(mode_key);
  // 如果模式不存在，则返回 None
  if (maybe_mode == c10::nullopt) {
    Py_RETURN_NONE;
  }
  // 获取模式对应的 Python 对象，并增加其引用计数
  auto* r = maybe_mode.value()->ptr(getPyInterpreter());
  // 增加返回值的引用计数
  Py_INCREF(r);
  // 返回 Python 对象
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* unset_dispatch_mode(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数不为 None
  TORCH_CHECK(arg != Py_None);
  // 将参数转换为 TorchDispatchModeKey 类型
  auto mode_key = py::cast<c10::impl::TorchDispatchModeKey>(arg);

  // 尝试取消指定的 Torch 分发模式
  const auto maybe_mode = c10::impl::TorchDispatchModeTLS::unset_mode(mode_key);
  // 如果模式不存在，则返回 None
  if (maybe_mode == c10::nullopt) {
    Py_RETURN_NONE;
  }
  // 获取模式对应的 Python 对象，并增加其引用计数
  auto* r = maybe_mode.value()->ptr(getPyInterpreter());
  // 增加返回值的引用计数
  Py_INCREF(r);
  // 返回 Python 对象
  return r;
  END_HANDLE_TH_ERRORS
}
static PyObject* len_torch_dispatch_stack(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 获取当前 Torch 分发模式堆栈的长度
  const auto len = c10::impl::TorchDispatchModeTLS::stack_len();
  // 将长度转换为 Python 对象并返回
  return utils::wrap(static_cast<int64_t>(len));
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_increment_version(PyObject* _unused, PyObject* tensor) {
  HANDLE_TH_ERRORS
  // 检查输入的对象是否为 Tensor，若不是则抛出错误
  TORCH_CHECK(
      THPVariable_Check(tensor), "increment_version expect a Tensor as input");
  // 增加 Tensor 的版本号
  torch::autograd::increment_version((THPVariable_Unpack(tensor)));
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_set_grad_enabled",
     // 设置 grad 是否启用的函数
     castPyCFunctionWithKeywords(set_grad_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"is_grad_enabled", // 查询 grad 是否启用的函数
     is_grad_enabled, METH_NOARGS, nullptr},
    {"_set_fwd_grad_enabled", // 设置前向梯度是否启用的函数
     set_fwd_grad_enabled, METH_O, nullptr},
    {"_is_fwd_grad_enabled", // 查询前向梯度是否启用的函数
     is_fwd_grad_enabled, METH_NOARGS, nullptr},
    {"is_inference_mode_enabled", // 查询推断模式是否启用的函数
     is_inference_mode_enabled, METH_NOARGS, nullptr},
    {"set_autocast_enabled", // 设置自动类型转换是否启用的函数
     castPyCFunctionWithKeywords(set_autocast_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"is_autocast_enabled", // 查询自动类型转换是否启用的函数
     castPyCFunctionWithKeywords(is_autocast_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"set_autocast_dtype", // 设置自动类型转换的数据类型的函数
     castPyCFunctionWithKeywords(set_autocast_dtype),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"get_autocast_dtype", // 获取自动类型转换的数据类型的函数
     castPyCFunctionWithKeywords(get_autocast_dtype),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_any_autocast_enabled", // 查询是否有任何自动类型转换启用的函数
     is_any_autocast_enabled, METH_NOARGS, nullptr},
    {"_is_autocast_available", // 查询自动类型转换是否可用的函数
     castPyCFunctionWithKeywords(is_autocast_available),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"clear_autocast_cache", // 清除自动类型转换缓存的函数
     clear_autocast_cache, METH_NOARGS, nullptr},
    {"set_autocast_cpu_enabled", // 设置 CPU 上的自动类型转换是否启用的函数
     set_autocast_cpu_enabled, METH_O, nullptr},
    {"is_autocast_cpu_enabled", // 查询 CPU 上的自动类型转换是否启用的函数
     is_autocast_cpu_enabled, METH_NOARGS, nullptr},
    {"set_autocast_cpu_dtype", // 设置 CPU 上的自动类型转换的数据类型的函数
     set_autocast_cpu_dtype, METH_O, nullptr},
    {"get_autocast_cpu_dtype", // 获取 CPU 上的自动类型转换的数据类型的函数
     get_autocast_cpu_dtype, METH_NOARGS, nullptr},
    {"set_autocast_gpu_dtype", // 设置 GPU 上的自动类型转换的数据类型的函数
     set_autocast_gpu_dtype, METH_O, nullptr},
    {"get_autocast_gpu_dtype", // 获取 GPU 上的自动类型转换的数据类型的函数
     get_autocast_gpu_dtype, METH_NOARGS, nullptr},
    {"set_autocast_xla_enabled", // 设置 XLA 上的自动类型转换是否启用的函数
     set_autocast_xla_enabled, METH_O, nullptr},
    {"is_autocast_xla_enabled", // 查询 XLA 上的自动类型转换是否启用的函数
     is_autocast_xla_enabled, METH_NOARGS, nullptr},
    {"set_autocast_xla_dtype", // 设置 XLA 上的自动类型转换的数据类型的函数
     set_autocast_xla_dtype, METH_O, nullptr},
    {"get_autocast_xla_dtype", // 获取 XLA 上的自动类型转换的数据类型的函数
     get_autocast_xla_dtype, METH_NOARGS, nullptr},
    {"set_autocast_ipu_enabled", // 设置 IPU 上的自动类型转换是否启用的函数
     set_autocast_ipu_enabled, METH_O, nullptr},
    {"is_autocast_ipu_enabled", // 查询 IPU 上的自动类型转换是否启用的函数
     is_autocast_ipu_enabled, METH_NOARGS, nullptr},
    {"set_autocast_ipu_dtype", // 设置 IPU 上的自动类型转换的数据类型的函数
     set_autocast_ipu_dtype, METH_O, nullptr},
    {"get_autocast_ipu_dtype", // 获取 IPU 上的自动类型转换的数据类型的函数
     get_autocast_ipu_dtype, METH_NOARGS, nullptr},
    // 注册一个名为 "autocast_increment_nesting" 的 C 函数，无参数
    {"autocast_increment_nesting",
     autocast_increment_nesting,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "autocast_decrement_nesting" 的 C 函数，无参数
    {"autocast_decrement_nesting",
     autocast_decrement_nesting,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "is_autocast_cache_enabled" 的 C 函数，无参数
    {"is_autocast_cache_enabled",
     is_autocast_cache_enabled,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "set_autocast_cache_enabled" 的 C 函数，带一个参数
    {"set_autocast_cache_enabled", set_autocast_cache_enabled, METH_O, nullptr},
    // 注册一个名为 "_increment_version" 的 C 函数，带一个参数
    {"_increment_version", THPModule_increment_version, METH_O, nullptr},
    // 注册一个名为 "set_anomaly_enabled" 的 C 函数，带变长参数和关键字参数
    {"set_anomaly_enabled",
     castPyCFunctionWithKeywords(set_anomaly_mode_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // 注册一个名为 "is_anomaly_enabled" 的 C 函数，无参数
    {"is_anomaly_enabled", is_anomaly_mode_enabled, METH_NOARGS, nullptr},
    // 注册一个名为 "is_anomaly_check_nan_enabled" 的 C 函数，无参数
    {"is_anomaly_check_nan_enabled",
     is_anomaly_check_nan_enabled,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "_is_multithreading_enabled" 的 C 函数，无参数
    {"_is_multithreading_enabled",
     is_multithreading_enabled,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "_set_multithreading_enabled" 的 C 函数，带变长参数和关键字参数
    {"_set_multithreading_enabled",
     castPyCFunctionWithKeywords(set_multithreading_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // 注册一个名为 "_is_view_replay_enabled" 的 C 函数，无参数
    {"_is_view_replay_enabled", is_view_replay_enabled, METH_NOARGS, nullptr},
    // 注册一个名为 "_set_view_replay_enabled" 的 C 函数，带变长参数和关键字参数
    {"_set_view_replay_enabled",
     castPyCFunctionWithKeywords(set_view_replay_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // 注册一个名为 "_enter_dual_level" 的 C 函数，无参数
    {"_enter_dual_level", python_enter_dual_level, METH_NOARGS, nullptr},
    // 注册一个名为 "_exit_dual_level" 的 C 函数，带变长参数和关键字参数
    {"_exit_dual_level",
     castPyCFunctionWithKeywords(python_exit_dual_level),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // 注册一个名为 "_is_torch_function_mode_enabled" 的 C 函数，无参数
    {"_is_torch_function_mode_enabled",
     is_torch_function_mode_enabled,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "_push_on_torch_function_stack" 的 C 函数，带一个参数
    {"_push_on_torch_function_stack",
     push_on_torch_function_stack,
     METH_O,
     nullptr},
    // 注册一个名为 "_pop_torch_function_stack" 的 C 函数，无参数
    {"_pop_torch_function_stack",
     pop_torch_function_stack,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "_get_function_stack_at" 的 C 函数，带变长参数和关键字参数
    {"_get_function_stack_at",
     castPyCFunctionWithKeywords(get_function_stack_at),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // 注册一个名为 "_len_torch_function_stack" 的 C 函数，无参数
    {"_len_torch_function_stack",
     len_torch_function_stack,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "_push_on_torch_dispatch_stack" 的 C 函数，带一个参数
    {"_push_on_torch_dispatch_stack",
     push_on_torch_dispatch_stack,
     METH_O,
     nullptr},
    // 注册一个名为 "_pop_torch_dispatch_stack" 的 C 函数，带一个参数
    {"_pop_torch_dispatch_stack", pop_torch_dispatch_stack, METH_O, nullptr},
    // 注册一个名为 "_get_dispatch_stack_at" 的 C 函数，带变长参数和关键字参数
    {"_get_dispatch_stack_at",
     castPyCFunctionWithKeywords(get_dispatch_stack_at),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // 注册一个名为 "_len_torch_dispatch_stack" 的 C 函数，无参数
    {"_len_torch_dispatch_stack",
     len_torch_dispatch_stack,
     METH_NOARGS,
     nullptr},
    // 注册一个名为 "_set_dispatch_mode" 的 C 函数，带一个参数
    {"_set_dispatch_mode", set_dispatch_mode, METH_O, nullptr},
    // 注册一个名为 "_get_dispatch_mode" 的 C 函数，带一个参数
    {"_get_dispatch_mode", get_dispatch_mode, METH_O, nullptr},
    // 注册一个名为 "_unset_dispatch_mode" 的 C 函数，带一个参数
    {"_unset_dispatch_mode", unset_dispatch_mode, METH_O, nullptr},

    // 结束函数列表的标记
    {nullptr, nullptr, 0, nullptr}};
# 返回指向 PyMethodDef 结构体数组的指针，该数组用于定义 Python 扩展模块中的函数
PyMethodDef* python_functions() {
    return methods;
}

# 结束 torch::autograd 命名空间的定义
} // namespace torch::autograd
```