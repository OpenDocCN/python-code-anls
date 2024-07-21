# `.\pytorch\test\custom_backend\custom_backend.h`

```
// 包含 Torch 的 JIT 后端接口头文件
#include <torch/csrc/jit/backends/backend.h>
// 包含 Torch 的 JIT 后端详细信息头文件
#include <torch/csrc/jit/backends/backend_detail.h>
// 包含 Torch 的 JIT 模块 API 头文件
#include <torch/csrc/jit/api/module.h>

// 定义自定义后端命名空间
namespace torch {
namespace custom_backend {

// 自定义 JIT 后端类，用于测试 JIT 后端注册端点和代码生成是否正确工作的最小化实现
class CustomBackend : public torch::jit::PyTorchBackendInterface {
 public:
  // 构造函数
  explicit CustomBackend() {}
  // 虚析构函数
  virtual ~CustomBackend() = default;

  // 检查后端是否可用
  bool is_available() override {
    return true;
  }

  // 编译函数，返回一个处理过的字典
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto spec =
        c10::impl::toTypedDict<std::string, at::IValue>(method_compile_spec);

    // 为每个键返回相同的字符串作为值
    auto handles = c10::Dict<std::string, std::string>();
    for (auto it = spec.begin(), end = spec.end(); it != end; ++it) {
      handles.insert(it->key(), it->key());
    }
    return c10::impl::toGenericDict(handles);
  }

  // 执行函数，处理输入列表并返回输出列表
  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    TORCH_INTERNAL_ASSERT(handle.isString());
    TORCH_INTERNAL_ASSERT(inputs.size() > 0);

    // 输出列表
    c10::List<at::Tensor> output_list;

    // 简单实现累加器和负累加器操作，根据句柄返回一个或两个输出张量
    c10::IValue value = inputs[0];
    at::Tensor accum = value.toTensor();
    accum = accum.clone();
    at::Tensor sub_accum = value.toTensor();
    sub_accum = sub_accum.clone();

    for (size_t i = 1, e = inputs.size(); i < e; ++i) {
      value = inputs[i];
      accum.add_(value.toTensor(), 1.0);
      sub_accum.sub_(value.toTensor(), 1.0);
    }

    if (handle.toStringRef() == "accum") {
      output_list.emplace_back(accum);
    } else if (handle.toStringRef() == "sub_accum") {
      output_list.emplace_back(sub_accum);
    } else if (handle.toStringRef() == "forward") {
      output_list.emplace_back(accum);
      output_list.emplace_back(sub_accum);
    }

    return c10::impl::toList(output_list);
  }
};

// 预处理函数，返回模块的 IValue 表示
c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  return mod._ivalue();
}

// 定义导出符号的宏，根据平台定义不同的导出方式
// Windows 下使用 __declspec(dllexport) 或 __declspec(dllimport)，其他平台为空
// clang-format off
#  if defined(_WIN32)
#    if defined(custom_ops_EXPORTS)
#      define CUSTOM_BACKEND_API __declspec(dllexport)
#    else
#      define CUSTOM_BACKEND_API __declspec(dllimport)
#    endif
#  else
#    define CUSTOM_BACKEND_API
#  endif
// clang-format on

// 返回后端名称的函数声明
CUSTOM_BACKEND_API std::string getBackendName();

} // namespace custom_backend
} // namespace torch
```