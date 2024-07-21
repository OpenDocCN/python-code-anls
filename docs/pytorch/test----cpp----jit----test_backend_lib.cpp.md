# `.\pytorch\test\cpp\jit\test_backend_lib.cpp`

```py
// 包含 Torch JIT 后端的相关头文件
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>

// Torch JIT 命名空间
namespace torch {
namespace jit {

// 这个测试 JIT 后端旨在执行最小化的工作，以测试 JIT 后端注册端点和代码生成的正确性。
// 不打算生成数值上正确的结果。
template <bool isAvailable>
class TestBackend : public PyTorchBackendInterface {
 public:
  // 默认构造函数
  // NOLINTNEXTLINE(modernize-use-equals-default)
  explicit TestBackend() {}
  virtual ~TestBackend() override = default;

  // 检查后端是否可用
  bool is_available() override {
    return isAvailable;
  }

  // 编译处理后的输入和方法编译规格，返回处理结果字典
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto spec =
        c10::impl::toTypedDict<std::string, at::IValue>(method_compile_spec);

    // 使用 method_compile_spec 中的每个键返回相同的字符串作为值
    auto handles = c10::Dict<std::string, std::string>();
    for (const auto& it : spec) {
      handles.insert(it.key(), it.key());
    }
    return c10::impl::toGenericDict(handles);
  }

  // 执行操作，根据句柄返回一个或多个输出张量列表
  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    TORCH_INTERNAL_ASSERT(handle.isString());
    TORCH_INTERNAL_ASSERT(inputs.size() > 0);

    c10::List<at::Tensor> output_list;

    // 实现简单的累加器和负累加器操作，根据句柄返回一个或两个张量
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

// 匿名命名空间，定义预处理函数
namespace {
c10::IValue preprocess(
    const Module& mod,
    const c10::Dict<IValue, IValue>& method_compile_spec,
    const BackendDebugHandleGenerator& generate_debug_handles) {
  return mod._ivalue();
}

// 定义测试后端名称及其可用性
constexpr auto backend_name = "test_backend";
static auto cls_available =
    torch::jit::backend<TestBackend<true>>(backend_name);
static auto pre_reg = backend_preprocess_register(backend_name, preprocess);

// 定义不可用的测试后端及其名称
constexpr auto backend_unavailable_name = "test_backend_unavailable";
static auto cls_unavailable =
    torch::jit::backend<TestBackend<false>>(backend_unavailable_name);
static auto pre_reg_unavailable =
    backend_preprocess_register(backend_unavailable_name, preprocess);
    # 使用函数 backend_preprocess_register 注册预处理函数 preprocess 到名为 backend_unavailable_name 的后端
    backend_preprocess_register(backend_unavailable_name, preprocess);
} // 关闭 torch 命名空间
} // 关闭 jit 命名空间
} // 关闭全局命名空间
```