# `.\pytorch\test\mobile\custom_build\predictor.cpp`

```py
// This is a simple predictor binary that loads a TorchScript CV model and runs
// a forward pass with fixed input `torch::ones({1, 3, 224, 224})`.
// It's used for end-to-end integration test for custom mobile build.

#include <iostream>
#include <string>
#include <c10/util/irange.h>
#include <torch/script.h>

using namespace std;

namespace {

// 包装器结构体，用于设置推理模式和禁用图优化器
struct MobileCallGuard {
  // 设置推理模式以进行推理
  c10::InferenceMode guard;
  // 禁用图优化器，确保自定义移动构建中未使用的操作列表不被改变
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
};

// 加载 TorchScript 模型的函数
torch::jit::Module loadModel(const std::string& path) {
  // 创建 MobileCallGuard 对象，设置推理模式和禁用图优化器
  MobileCallGuard guard;
  // 使用给定路径加载 TorchScript 模型
  auto module = torch::jit::load(path);
  // 将模型设置为评估模式
  module.eval();
  return module;
}

} // namespace

// 主函数入口
int main(int argc, const char* argv[]) {
  // 检查命令行参数是否少于 2 个，若少于则打印用法信息并退出
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>\n";
    return 1;
  }
  // 加载指定路径下的模型
  auto module = loadModel(argv[1]);
  // 创建输入张量，大小为 {1, 3, 224, 224}，元素值为 1
  auto input = torch::ones({1, 3, 224, 224});
  // 运行模型的前向传播，并获取输出张量
  auto output = [&]() {
    // 创建新的 MobileCallGuard 对象，设置推理模式和禁用图优化器
    MobileCallGuard guard;
    // 执行模型的前向传播，并将输出转换为张量返回
    return module.forward({input}).toTensor();
  }();

  // 设置输出精度为小数点后三位
  std::cout << std::setprecision(3) << std::fixed;
  // 遍历输出张量的前五个元素，并逐行打印它们的值
  for (const auto i : c10::irange(5)) {
    std::cout << output.data_ptr<float>()[i] << std::endl;
  }
  return 0;
}
```