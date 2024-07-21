# `.\pytorch\test\custom_backend\test_custom_backend.cpp`

```
// 包含 Torch CUDA 相关头文件
#include <torch/cuda.h>
// 包含 Torch 脚本模块相关头文件
#include <torch/script.h>

// 包含标准字符串库
#include <string>

// 包含自定义后端头文件
#include "custom_backend.h"

// 从指定路径加载降低为自定义后端的模块，并测试其执行和结果正确性
void load_serialized_lowered_module_and_execute(const std::string& path) {
  // 使用 Torch 加载指定路径的模块
  torch::jit::Module module = torch::jit::load(path);
  // 自定义后端被硬编码为计算 f(a, b) = (a + b, a - b)。
  auto tensor = torch::ones(5);
  // 准备输入向量，包含两个相同的张量
  std::vector<torch::jit::IValue> inputs{tensor, tensor};
  // 执行模块的前向传播
  auto output = module.forward(inputs);
  // 断言输出是一个元组
  AT_ASSERT(output.isTuple());
  // 获取元组的元素列表
  auto output_elements = output.toTupleRef().elements();
  // 断言每个元素是张量
  for (auto& e : output_elements) {
    AT_ASSERT(e.isTensor());
  }
  // 断言元组中应有两个元素
  AT_ASSERT(output_elements.size(), 2);
  // 断言第一个元素的张量应接近于 tensor + tensor
  AT_ASSERT(output_elements[0].toTensor().allclose(tensor + tensor));
  // 断言第二个元素的张量应接近于 tensor - tensor
  AT_ASSERT(output_elements[1].toTensor().allclose(tensor - tensor));
}

// 主函数，程序入口
int main(int argc, const char* argv[]) {
  // 检查命令行参数数量是否正确
  if (argc != 2) {
    std::cerr
        << "usage: test_custom_backend <path-to-exported-script-module>\n";
    return -1;
  }
  // 获取导出脚本模块的路径
  const std::string path_to_exported_script_module = argv[1];

  // 输出自定义后端的名称进行测试
  std::cout << "Testing " << torch::custom_backend::getBackendName() << "\n";
  // 载入降低为自定义后端的模块并执行测试
  load_serialized_lowered_module_and_execute(path_to_exported_script_module);

  // 打印测试通过消息
  std::cout << "OK\n";
  return 0;
}
```