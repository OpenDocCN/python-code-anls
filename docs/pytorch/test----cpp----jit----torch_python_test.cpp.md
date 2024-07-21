# `.\pytorch\test\cpp\jit\torch_python_test.cpp`

```py
// 导入必要的头文件，ATen 库用于处理张量，c10 异常处理，torch 库导出，jit 模块相关接口，torch 脚本运行支持
#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>

// 声明 torch 命名空间内的 jit 命名空间
namespace torch {
namespace jit {

// 根据不同编译器定义 JIT_TEST_API 宏
#ifdef _MSC_VER
#define JIT_TEST_API
#else
#define JIT_TEST_API TORCH_API
#endif

// 匿名命名空间，包含一些辅助函数和测试用例

// 检查是否在 Sandcastle 环境中运行
bool isSandcastle() {
  return (
      (std::getenv("SANDCASTLE")) ||
      (std::getenv("TW_JOB_USER") &&
       std::string(std::getenv("TW_JOB_USER")) == "sandcastle"));
}

// 测试加载的模型是否能正确设置和获取训练模式
void testEvalModeForLoadedModule() {
  if (isSandcastle())
    return; // 不在 Sandcastle 环境中生成模型文件
  std::string module_path = "dropout_model.pt";
  torch::jit::Module module = torch::jit::load(module_path);
  AT_ASSERT(module.attr("dropout").toModule().is_training());
  module.eval();
  AT_ASSERT(!module.attr("dropout").toModule().is_training());
  module.train();
  AT_ASSERT(module.attr("dropout").toModule().is_training());
}

// 测试 Torch 的序列化和反序列化功能是否正常工作
void testTorchSaveError() {
  if (isSandcastle()) {
    // 在 Sandcastle 环境中不生成需要加载的文件
    return;
  }

  // 通过 test/cpp/jit/tests_setup.py 生成的序列化文件加载并进行测试
  bool passed = true;
  try {
    torch::jit::load("eager_value.pt");
    passed = false;
  } catch (const std::exception& c) {
  }
  // 确保 torch::jit::load 没有运行
  AT_ASSERT(passed);
}

} // namespace

// 导出的函数，用于运行所有的 JIT C++ 测试
JIT_TEST_API void runJITCPPTests() {
  // 调用测试函数，包括 testEvalModeForLoadedModule 和 testTorchSaveError
  testEvalModeForLoadedModule();
  testTorchSaveError();
}

} // namespace jit
} // namespace torch
```