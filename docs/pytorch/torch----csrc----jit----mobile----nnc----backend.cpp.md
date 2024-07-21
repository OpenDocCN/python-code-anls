# `.\pytorch\torch\csrc\jit\mobile\nnc\backend.cpp`

```
#include <vector>  // 包含标准库中的 vector 头文件

#include <torch/csrc/jit/backends/backend.h>  // 包含 Torch 后端接口头文件
#include <torch/csrc/jit/mobile/nnc/context.h>  // 包含 Torch 移动端 NNCompiler 上下文头文件

namespace torch {  // 进入 Torch 命名空间
namespace jit {  // 进入 JIT 命名空间
namespace mobile {  // 进入移动端命名空间
namespace nnc {  // 进入 NNCompiler 命名空间

class NNCBackend : public PyTorchBackendInterface {  // 定义 NNCompiler 后端类，继承自 PyTorch 后端接口
 public:
  explicit NNCBackend() = default;  // 构造函数，默认实现
  ~NNCBackend() override = default;  // 析构函数，默认实现

  bool is_available() override {  // 判断 NNCompiler 后端是否可用的函数
    return true;  // 始终返回 true 表示可用
  }

  c10::impl::GenericDict compile(  // 编译函数，接收处理后的 IValue 和方法编译规范
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    cu_ = std::make_shared<CompilationUnit>(processed);  // 创建编译单元，使用处理后的 IValue

    // 输入 method_compile_spec:
    //   键: 方法名
    //   值: 每个方法的编译规范
    // 输出:
    //   键: 方法名
    //   值: 每个方法的后端句柄
    auto spec =
        c10::impl::toTypedDict<std::string, at::IValue>(method_compile_spec);  // 将 method_compile_spec 转换为类型化的字典
    auto handles = c10::Dict<std::string, std::string>();  // 创建存储方法名及其后端句柄的字典
    for (const auto& it : spec) {  // 遍历编译规范中的每个项
      // 每个方法的后端句柄即为方法名本身
      handles.insert(it.key(), it.key());  // 将方法名作为键和值插入到 handles 字典中
    }
    return c10::impl::toGenericDict(handles);  // 返回转换为通用字典类型的 handles
  }

  c10::impl::GenericList execute(  // 执行函数，接收句柄和输入列表
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    const std::string& method_name = handle.toStringRef();  // 获取句柄所表示的方法名
    auto function_name = c10::QualifiedName(method_name);  // 使用方法名创建 QualifiedName 对象
    return cu_->run(function_name, inputs);  // 调用编译单元的 run 方法执行方法，并返回结果列表
  }

 private:
  std::shared_ptr<CompilationUnit> cu_;  // 编译单元的智能指针
};

namespace {
// TODO(mvz): temporarily disable NNC backend in mobile builds.
// 临时禁用移动端构建中的 NNC 后端。
// static const auto cls = torch::jit::backend<NNCBackend>("nnc");
} // namespace

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
```