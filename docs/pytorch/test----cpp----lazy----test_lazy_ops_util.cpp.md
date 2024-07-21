# `.\pytorch\test\cpp\lazy\test_lazy_ops_util.cpp`

```
// 包含头文件用于测试懒惰操作工具
#include <test/cpp/lazy/test_lazy_ops_util.h>

// 包含懒惰计算的背景降低上下文的头文件
#include <torch/csrc/lazy/backend/lowering_context.h>
// 包含懒惰计算的IR构建器的头文件
#include <torch/csrc/lazy/core/ir_builder.h>
// 包含懒惰计算的IR转储工具的头文件
#include <torch/csrc/lazy/core/ir_dump_util.h>
// 包含懒惰计算的张量实现的头文件
#include <torch/csrc/lazy/core/tensor_impl.h>

// 包含标准输入输出流的头文件
#include <iostream>
// 包含字符串操作的头文件
#include <string>

// 使用torch命名空间下的lazy命名空间
namespace torch {
namespace lazy {
// 匿名命名空间，用于实现私有函数和静态变量
namespace {

// 创建一个无序集合指针，用于存储被忽略的计数器名称
std::unordered_set<std::string>* CreateIgnoredCounters() {
  // 分配一个新的无序集合指针对象
  std::unordered_set<std::string>* icounters =
      new std::unordered_set<std::string>();
  // 向集合中添加需要在执行"是否有计数器改变"断言时被忽略的计数器名称
  icounters->insert("aten::rand");
  return icounters;
}

} // namespace

// 获取被忽略计数器名称集合的常量指针
const std::unordered_set<std::string>* GetIgnoredCounters() {
  // 静态局部变量，保证函数返回相同的集合指针
  static const std::unordered_set<std::string>* icounters =
      CreateIgnoredCounters();
  return icounters;
}

// 将张量转移到CPU设备的函数
at::Tensor ToCpuTensor(const at::Tensor& tensor) {
  // 如果张量位于torch::kLazy设备，tensor.to()会隐式触发同步操作
  return tensor.to(torch::kCPU);
}

// 复制张量到指定设备的函数
torch::Tensor CopyToDevice(
    const torch::Tensor& tensor,
    const torch::Device& device) {
  // 克隆张量并将其拷贝到指定设备，非阻塞操作，拷贝张量内容
  return tensor.clone().to(device, /*non_blocking=*/false, /*copy=*/true);
}

// 比较两个张量的值是否相等的函数
bool EqualValues(at::Tensor tensor1, at::Tensor tensor2) {
  // 将张量1和张量2都转移到CPU设备
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);

  // 如果张量1中存在NaN值，进行额外的处理
  if (torch::isnan(tensor1).any().item<bool>()) {
    EXPECT_TRUE(EqualValues(torch::isnan(tensor1), torch::isnan(tensor2)));
    tensor1.nan_to_num_();
    tensor2.nan_to_num_();
  }

  // 检查两个张量的形状和数据类型是否相同
  if (tensor1.sizes() != tensor2.sizes() ||
      tensor1.dtype() != tensor2.dtype()) {
    // 输出不同的形状信息到标准错误流
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }

  // 如果张量1和张量2的数据类型不同，将张量1转换为张量2的数据类型
  at::ScalarType type1 = tensor1.scalar_type();
  at::ScalarType type2 = tensor2.scalar_type();
  if (type1 != type2) {
    tensor1 = tensor1.toType(type2);
  }

  // 比较两个张量是否相等
  bool equal = tensor1.equal(tensor2);
  return equal;
}

// 比较两个张量的值是否相等的函数，不检查元素类型
bool EqualValuesNoElementTypeCheck(at::Tensor tensor1, at::Tensor tensor2) {
  // 将张量1和张量2都转移到CPU设备
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);

  // 检查两个张量的形状是否相同
  if (tensor1.sizes() != tensor2.sizes()) {
    // 输出不同的形状信息到标准错误流
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }

  // 如果张量1和张量2的数据类型不同，将张量1转换为张量2的数据类型
  at::ScalarType type1 = tensor1.scalar_type();
  at::ScalarType type2 = tensor2.scalar_type();
  if (type1 != type2) {
    tensor1 = tensor1.toType(type2);
  }

  // 比较两个张量是否相等
  bool equal = tensor1.equal(tensor2);
  return equal;
}
void ForEachDevice(const std::function<void(const torch::Device&)>& devfn) {
    // 遍历每个设备并执行给定函数
    // 当前 TorchScript 后端仅支持进程中一种硬件类型，由环境设置。
    // 分布式训练/多设备尚未支持，因此始终使用序数 0。
    
    // 获取当前硬件设备
    auto device = torch::lazy::BackendDevice();
    // 将后端设备转换为 ATen 设备类型
    torch::Device torch_device = torch::lazy::backendDeviceToAtenDevice(device);
    // 执行给定的设备函数
    devfn(torch_device);
}

bool CloseValues(
    at::Tensor tensor1,
    at::Tensor tensor2,
    double rtol,
    double atol) {
    // 将张量1和张量2移动到 CPU 上进行比较
    tensor1 = ToCpuTensor(tensor1);
    tensor2 = ToCpuTensor(tensor2);
    
    // 如果张量1中存在 NaN 值，则处理 NaN 值
    if (torch::isnan(tensor1).any().item<bool>()) {
        EXPECT_TRUE(EqualValues(torch::isnan(tensor1), torch::isnan(tensor2)));
        tensor1.nan_to_num_();
        tensor2.nan_to_num_();
    }
    
    // 检查张量形状和数据类型是否相同
    if (tensor1.sizes() != tensor2.sizes() ||
        tensor1.dtype() != tensor2.dtype()) {
        // 输出不同形状的错误信息
        std::cerr << "Different shape:\n"
                  << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
                  << tensor2.dtype() << " " << tensor2.sizes() << "\n";
        return false;
    }
    
    // 使用相对容差和绝对容差比较两个张量的近似相等性
    bool equal = tensor1.allclose(tensor2, rtol, atol);
    return equal;
}

std::string GetTensorTextGraph(at::Tensor tensor) {
    // 尝试获取懒惰张量，并将其转换为文本格式的计算图
    torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
    return torch::lazy::DumpUtil::ToText({lazy_tensor->GetIrValue().node.get()});
}

std::string GetTensorDotGraph(at::Tensor tensor) {
    // 尝试获取懒惰张量，并将其转换为 DOT 格式的计算图
    torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
    return torch::lazy::DumpUtil::ToDot({lazy_tensor->GetIrValue().node.get()});
}

void TestBackward(
    const std::vector<torch::Tensor>& inputs,
    const torch::Device& device,
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>&
        testfn,
    double rtol,
    double atol,
    int derivative_level) {
    // 初始化用于反向传播测试的变量
    std::vector<torch::Tensor> input_vars;
    std::vector<torch::Tensor> xinput_vars;
    std::vector<torch::Tensor> inputs_w_grad;
    std::vector<torch::Tensor> xinputs_w_grad;
    
    // 遍历输入张量并进行相应处理
    for (size_t i = 0; i < inputs.size(); ++i) {
        const torch::Tensor& input = inputs[i];
        if (input.defined()) {
            // 克隆输入张量并设置梯度要求
            torch::Tensor oinput =
                input.clone().detach().set_requires_grad(input.requires_grad());
            input_vars.push_back(oinput);
            
            // 将输入张量复制到指定设备并设置梯度要求
            torch::Tensor xinput = CopyToDevice(input, device)
                                       .detach()
                                       .set_requires_grad(input.requires_grad());
            xinput_vars.push_back(xinput);
            
            // 如果张量需要梯度，加入梯度列表
            if (input.requires_grad()) {
                inputs_w_grad.push_back(oinput);
                xinputs_w_grad.push_back(xinput);
            }
        } else {
            // 未定义的输入张量处理
            input_vars.emplace_back();
            xinput_vars.emplace_back();
        }
    }
    
    // 执行测试函数并获取输出张量
    torch::Tensor output = testfn(input_vars);
    torch::Tensor xoutput = testfn(xinput_vars);
    
    // 检查输出张量在两个设备上的近似相等性
    torch::lazy::AllClose(output, xoutput, rtol, atol);
    
    // 初始化输出张量列表
    std::vector<torch::Tensor> outs = {output};
    std::vector<torch::Tensor> xouts = {xoutput};
    
    // 遍历计算指定导数级别的反向传播
    for (int d = 1; d <= derivative_level; ++d) {
        // To be continued in actual code...
    // 计算 sum(outs) 相对于 inputs_w_grad 的梯度
    torch::Tensor sum = torch::zeros_like(outs[0]).sum();
    // 计算 sum(xouts) 相对于 xinputs_w_grad 的梯度
    torch::Tensor xsum = torch::zeros_like(xouts[0]).sum();
    // 遍历所有的输出张量
    for (size_t i = 0; i < outs.size(); ++i) {
      // 检查当前张量是否需要梯度
      if (outs[i].requires_grad()) {
        // 如果需要梯度，则累加其所有元素的和到 sum 和 xsum 中
        sum += outs[i].sum();
        xsum += xouts[i].sum();
      }
    }
    // 计算高阶导数时需要设置 create_graph=true
    bool create_graph = d != derivative_level;
    // 计算 sum 对 inputs_w_grad 中张量的梯度
    outs = torch::autograd::grad(
        {sum},
        inputs_w_grad,
        /*grad_outputs=*/{},
        /*retain_graph=*/c10::nullopt,
        /*create_graph=*/create_graph,
        /*allow_unused=*/true);
    // 计算 xsum 对 xinputs_w_grad 中张量的梯度
    xouts = torch::autograd::grad(
        {xsum},
        xinputs_w_grad,
        /*grad_outputs=*/{},
        /*retain_graph=*/c10::nullopt,
        /*create_graph=*/create_graph,
        /*allow_unused=*/true);
    // 遍历所有输出张量，确保其定义性并进行数值相等性检查
    for (size_t i = 0; i < outs.size(); ++i) {
      // 使用 ASSERT_EQ 确保 outs[i] 和 xouts[i] 的定义性相同
      ASSERT_EQ(outs[i].defined(), xouts[i].defined());
      // 如果 outs[i] 定义了，则使用 AllClose 函数检查其值与 xouts[i] 的接近程度
      if (outs[i].defined()) {
        AllClose(outs[i], xouts[i], rtol, atol);
      }
    }
  }
}

} // namespace lazy
} // namespace torch
```