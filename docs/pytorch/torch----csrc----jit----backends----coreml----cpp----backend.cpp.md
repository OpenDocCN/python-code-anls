# `.\pytorch\torch\csrc\jit\backends\coreml\cpp\backend.cpp`

```
// 包含 Torch 的后端接口和脚本支持
#include <torch/csrc/jit/backends/backend.h>
#include <torch/script.h>

// 命名空间开始
namespace {

// 定义 CoreMLBackend 类，继承自 torch::jit::PyTorchBackendInterface
class CoreMLBackend : public torch::jit::PyTorchBackendInterface {
 public:
  // 编译函数，接收处理后的值和方法编译规范作为参数，并返回泛型字典
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    // 断言：CoreML 后端在服务器端不受支持，抛出错误信息
    TORCH_CHECK(false, "The CoreML backend is not supported on server side!");
    // 创建一个空的字符串到字符串的字典
    auto handles = c10::Dict<std::string, std::string>();
    // 将该字典转换为泛型字典并返回
    return c10::impl::toGenericDict(handles);
  }

  // 执行函数，接收处理句柄和输入列表作为参数，并返回泛型列表
  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    // 断言：CoreML 后端在服务器端不受支持，抛出错误信息
    TORCH_CHECK(false, "The CoreML backend is not supported on server side!");
    // 创建一个空的 Tensor 列表
    c10::List<at::Tensor> output_list;
    // 将该列表转换为泛型列表并返回
    return c10::impl::toList(output_list);
  }

  // 检查后端是否可用的函数，始终返回 false
  bool is_available() override {
    return false;
  }
};

// 注册 CoreMLBackend 类为 Torch 的后端，并命名为 "coreml"
static auto cls = torch::jit::backend<CoreMLBackend>("coreml");

} // 命名空间结束
```