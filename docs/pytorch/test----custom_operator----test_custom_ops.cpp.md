# `.\pytorch\test\custom_operator\test_custom_ops.cpp`

```
// 引入C++标准库头文件和Torch相关头文件
#include <c10/util/irange.h>
#include <torch/script.h>
#include <torch/cuda.h>

// 引入自定义操作的头文件
#include "op.h"

// 引入C++标准库
#include <memory>
#include <string>
#include <vector>

// 引入输入输出流库
#include <iostream>

// 定义帮助函数的命名空间
namespace helpers {

// 检查模块中所有参数是否符合给定的谓词条件
template <typename Predicate>
void check_all_parameters(
    const torch::jit::Module& module,
    Predicate predicate) {
  // 遍历模块中的所有参数，并使用谓词进行检查
  for (at::Tensor parameter : module.parameters()) {
    AT_ASSERT(predicate(parameter));
  }
}

// 从操作注册表中获取并执行操作
template<class Result, class... Args>
Result get_operator_from_registry_and_execute(const char* op_name, Args&&... args) {
  // 获取指定操作名对应的所有操作符
  auto& ops = torch::jit::getAllOperatorsFor(
      torch::jit::Symbol::fromQualString(op_name));
  // 确保找到的操作符数量为1
  TORCH_INTERNAL_ASSERT(ops.size() == 1);

  // 获取第一个操作符
  auto& op = ops.front();
  // 确保操作符的名称与指定的操作名一致
  TORCH_INTERNAL_ASSERT(op->schema().name() == op_name);

  // 创建堆栈用于操作执行
  torch::jit::Stack stack;
  // 将参数推送到堆栈中
  torch::jit::push(stack, std::forward<Args>(args)...);
  // 执行操作
  op->getOperation()(stack);

  // 确保堆栈中的大小为1
  TORCH_INTERNAL_ASSERT(1 == stack.size());
  // 弹出结果并转换为指定类型并返回
  return torch::jit::pop(stack).to<Result>();
}
} // namespace helpers

// 从注册表中获取自定义操作并执行
void get_operator_from_registry_and_execute() {
  // 调用帮助函数获取自定义操作的输出向量
  std::vector<torch::Tensor> output =
    helpers::get_operator_from_registry_and_execute<std::vector<torch::Tensor>>("custom::op", torch::ones(5), 2.0, 3);

  // 手动执行自定义操作
  const auto manual = custom_op(torch::ones(5), 2.0, 3);

  // 确保输出向量的大小为3
  TORCH_INTERNAL_ASSERT(output.size() == 3);
  // 遍历输出向量并逐一检查是否接近期望的结果
  for (const auto i : c10::irange(output.size())) {
    TORCH_INTERNAL_ASSERT(output[i].allclose(torch::ones(5) * 2));
    TORCH_INTERNAL_ASSERT(output[i].allclose(manual[i]));
  }
}

// 从注册表中获取带自动求导的操作并执行
void get_autograd_operator_from_registry_and_execute() {
  // 创建需要自动求导的张量
  torch::Tensor x = torch::randn({5,5}, torch::requires_grad());
  torch::Tensor y = torch::randn({5,5}, torch::requires_grad());
  torch::Tensor z = torch::randn({5,5}, torch::requires_grad());

  // 调用帮助函数获取带自动求导的操作的输出张量
  torch::Tensor output =
    helpers::get_operator_from_registry_and_execute<torch::Tensor>("custom::op_with_autograd", x, 2, y, std::optional<torch::Tensor>());

  // 确保输出张量接近预期的结果
  TORCH_INTERNAL_ASSERT(output.allclose(x + 2*y + x*y));

  // 创建梯度张量用于反向传播
  auto go = torch::ones({}, torch::requires_grad());
  // 计算输出张量的和的反向传播
  output.sum().backward(go, false, true);

  // 确保计算出的梯度张量与预期结果接近
  TORCH_INTERNAL_ASSERT(torch::allclose(x.grad(), y + torch::ones({5,5})));
  TORCH_INTERNAL_ASSERT(torch::allclose(y.grad(), x + torch::ones({5,5})*2));

  // 使用可选参数测试操作
  at::zero_(x.mutable_grad());
  at::zero_(y.mutable_grad());
  // 再次调用帮助函数获取带自动求导的操作的输出张量
  output = helpers::get_operator_from_registry_and_execute<torch::Tensor>(
      "custom::op_with_autograd", x, 2, y, z);

  // 确保输出张量接近包括可选参数在内的预期结果
  TORCH_INTERNAL_ASSERT(output.allclose(x + 2*y + x*y + z));
  // 计算输出张量的和的反向传播
  go = torch::ones({}, torch::requires_grad());
  output.sum().backward(go, false, true);

  // 确保计算出的梯度张量与预期结果接近
  TORCH_INTERNAL_ASSERT(torch::allclose(x.grad(), y + torch::ones({5,5})));
  TORCH_INTERNAL_ASSERT(torch::allclose(y.grad(), x + torch::ones({5,5})*2));
  TORCH_INTERNAL_ASSERT(torch::allclose(z.grad(), torch::ones({5,5})));
}
// 在自动微分模式下执行注册表中的运算符
void get_autograd_operator_from_registry_and_execute_in_nograd_mode() {
  // 进入自动微分以下的自动调度保护区域
  at::AutoDispatchBelowAutograd guard;

  // 创建需要梯度的随机张量 x 和 y
  torch::Tensor x = torch::randn({5,5}, torch::requires_grad());
  torch::Tensor y = torch::randn({5,5}, torch::requires_grad());

  // 从注册表中获取自定义的运算符并执行，返回输出张量
  torch::Tensor output =
    helpers::get_operator_from_registry_and_execute<torch::Tensor>("custom::op_with_autograd", x, 2, y, std::optional<torch::Tensor>());

  // 内部断言，确保输出张量与预期结果 x + 2*y + x*y 在数值上接近
  TORCH_INTERNAL_ASSERT(output.allclose(x + 2*y + x*y));
}

// 加载序列化模块并执行带有自定义运算符的脚本
void load_serialized_module_with_custom_op_and_execute(
    const std::string& path_to_exported_script_module) {
  // 加载导出的脚本模块
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);
  
  // 准备输入张量
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones(5));

  // 执行模块的前向传播并获取输出张量
  auto output = module.forward(inputs).toTensor();

  // 内部断言，确保输出张量在数值上接近预期结果 torch::ones(5) + 1
  AT_ASSERT(output.allclose(torch::ones(5) + 1));
}

// 测试序列化模块的参数检查
void test_argument_checking_for_serialized_modules(
    const std::string& path_to_exported_script_module) {
  // 加载导出的脚本模块
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);

  // 测试期望抛出异常的情况：输入参数个数超过预期
  try {
    module.forward({torch::jit::IValue(1), torch::jit::IValue(2)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("Expected at most 2 argument(s) for operator 'forward', "
                  "but received 3 argument(s)") == 0);
  }

  // 测试期望抛出异常的情况：输入参数类型不符合预期
  try {
    module.forward({torch::jit::IValue(5)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("forward() Expected a value of type 'Tensor' "
                  "for argument 'input' but instead found type 'int'") == 0);
  }

  // 测试期望抛出异常的情况：缺少必要的输入参数
  try {
    module.forward({});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("forward() is missing value for argument 'input'") == 0);
  }
}

// 测试模块移动到不同设备上
void test_move_to_device(const std::string& path_to_exported_script_module) {
  // 加载导出的脚本模块
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);

  // 检查所有参数是否都在 CPU 上
  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cpu();
  });

  // 将模块移动到 CUDA 设备
  module.to(torch::kCUDA);

  // 检查所有参数是否都在 CUDA 设备上
  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cuda();
  });

  // 将模块移动回 CPU 设备
  module.to(torch::kCPU);

  // 检查所有参数是否都在 CPU 上
  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cpu();
  });
}

// 测试模块移动到不同数据类型
void test_move_to_dtype(const std::string& path_to_exported_script_module) {
  // 加载导出的脚本模块
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);

  // 将模块移动到 Float16 数据类型
  module.to(torch::kFloat16);

  // 检查所有参数是否都为 Float16 数据类型
  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    return tensor.dtype() == torch::kFloat16;
  });

  // 将模块移动到 Double 数据类型
  module.to(torch::kDouble);

  // 检查所有参数是否都为 Double 数据类型
  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    // 检查张量的数据类型是否为双精度（double）
    return tensor.dtype() == torch::kDouble;
  });
}

// 主函数，程序的入口点，接收命令行参数
int main(int argc, const char* argv[]) {
  // 检查命令行参数数量，确保只有一个参数
  if (argc != 2) {
    // 如果参数数量不正确，输出使用说明并退出程序
    std::cerr << "usage: test_custom_ops <path-to-exported-script-module>\n";
    return -1;
  }
  // 从命令行参数中获取导出脚本模块的路径
  const std::string path_to_exported_script_module = argv[1];

  // 调用注册表中的操作符并执行
  get_operator_from_registry_and_execute();
  // 调用注册表中的自动微分操作符并执行
  get_autograd_operator_from_registry_and_execute();
  // 在无梯度模式下，调用注册表中的自动微分操作符并执行
  get_autograd_operator_from_registry_and_execute_in_nograd_mode();
  // 载入包含自定义操作的序列化模块并执行
  load_serialized_module_with_custom_op_and_execute(
      path_to_exported_script_module);
  // 测试序列化模块的参数检查
  test_argument_checking_for_serialized_modules(path_to_exported_script_module);
  // 测试将操作移至指定数据类型
  test_move_to_dtype(path_to_exported_script_module);

  // 如果存在 CUDA 设备，测试将操作移至设备
  if (torch::cuda::device_count() > 0) {
    test_move_to_device(path_to_exported_script_module);
  }

  // 输出测试通过信息
  std::cout << "ok\n";
}
```