# `.\pytorch\torch\csrc\api\include\torch\nn\parallel\data_parallel.h`

```py
// 防止头文件被多次包含
#pragma once

// 引入相关的 CUDA 和 Torch 头文件
#include <torch/cuda.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

// 引入 ATen 库中的功能和函数
#include <ATen/core/functional.h>
#include <torch/csrc/autograd/functions/comm.h>
#include <torch/csrc/autograd/functions/utils.h>

// 引入 ATen 库中的设备和并行功能
#include <ATen/Device.h>
#include <ATen/Parallel.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 引入标准库头文件
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

// Torch 命名空间开始
namespace torch {
namespace nn {

// 匿名命名空间内的注释区块
namespace {

// 注意事项 [Replicating Modules]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// 模块复制实现的两个步骤：
// 1) 使用 Module.clone() 在每个目标设备上创建模块副本。
// 2) 手动添加一个梯度边，从每个模块副本中的每个参数 X 指向原始模块中相同的参数 X，使用 ReduceAdd 作为 grad_fn。
//
// ReduceAdd 只能在数据并行的反向传播过程中使用。前向传播不能使用此函数，因为它根本不设置梯度函数和历史记录。不要尝试将 ReduceAdd 用于任何其他目的。
//
// 注意：一个替代方法是将 Broadcast 和 ReduceAddCoalesce 添加到 torch/csrc/autograd/functions/comm.cpp 中作为正常的自动求导函数，
// 实现一个 Replicatable 类（类似可克隆类），并在 Module.h 中将其添加为友元类。在前向传播中，Replicatable 可以使用 Broadcast 函数来复制每个模块参数，
// 并使用 ReduceAddCoalesce 设置梯度函数（类似于 Python 中的实现方式）。然而，与 Python 不同的是，对 Linear._parameters["weight"] 的更改也会应用于 Linear.weight（以 Linear 为例），
// 而 Linear.weight 和 Linear.parameters_["weight"] 是指向同一 TensorImpl 的两个张量对象。将新张量分配给 Linear.parameters_["weight"] 不会更改 Linear.weight。
// 为使其工作，我们将不得不：
// 1) 强制每个模块也继承自 Replicatable
// 2) 强制每个模块实现一个额外的函数，例如 Replicatable::load_params()，以从 parameters_ 中获取到其自己的成员字段的更改。
// 这将是一种过度设计，因为 Replicatable 仅在数据并行中使用，甚至不是 ddp。

// 用于数据并行中复制步骤的自动求导函数。仅在数据并行中使用，不应作为用户 API 暴露出去。
struct ReduceAdd : public autograd::Node {
  explicit ReduceAdd(const at::Device& destination_device)
      : destination_device_(destination_device){};
  ~ReduceAdd() override {}

  // 在该函数中应用 ReduceAdd 的操作
  autograd::variable_list apply(autograd::variable_list&& inputs) override {
    // 检查是否需要计算梯度
    TORCH_CHECK(
        !torch::autograd::compute_requires_grad(inputs),
        "ReduceAdd can only be used during the backward pass of data parallel.");

    // 在目标设备上创建与 inputs[0] 形状相同的零张量
    Tensor output = torch::zeros_like(inputs[0], {destination_device_});
    // 对于输入的每个张量进行遍历
    for (auto& input : inputs) {
      // 检查所有输入张量的尺寸必须相同
      TORCH_CHECK(
          input.sizes() == inputs[0].sizes(),
          "All inputs of ReduceAdd must have the same size, but got ",
          input.sizes(),
          " and ",
          inputs[0].sizes());

      // 检查所有输入张量的数据类型必须相同
      TORCH_CHECK(
          input.dtype() == inputs[0].dtype(),
          "All inputs of ReduceAdd must have the same dtype, but got ",
          input.dtype(),
          " and ",
          inputs[0].dtype());

      // TODO: 使用 nccl 进行 reduce 操作（尚未实现）
      // 将每个输入张量转移到目标设备并累加到输出张量上
      output.add_(input.to(destination_device_));
    }

    // 返回包含输出张量的数组
    return {output};
  }

 private:
  // 目标设备，用于确定输出张量的存储位置
  at::Device destination_device_;
};

} // namespace

// A friend function to Module, it recursively sets gradient edges pointing from
// every parameter X in every module replica to the same parameter X in the
// original module. See [Replicating Modules]
template <typename ModuleType>
void replicate_grad_edges(
    const std::shared_ptr<Module>& module,
    const std::vector<std::shared_ptr<ModuleType>>& replicas,
    const std::vector<Device>& devices) {
  // 遍历原始模块的所有命名参数
  for (auto& parameter : module->named_parameters(/*recurse=*/false)) {
    // 创建一个 ReduceAdd 的共享指针，设定其设备
    auto grad_fn = std::make_shared<ReduceAdd>((*parameter).device());
    // 设置 grad_fn 的下一个边缘，收集参数的下一个边缘
    grad_fn->set_next_edges(autograd::collect_next_edges(*parameter));

    // 遍历设备列表
    for (const auto i : c10::irange(devices.size())) {
      // 在每个复制品的参数历史中设置 grad_fn
      autograd::set_history(replicas[i]->parameters_[parameter.key()], grad_fn);
    }
  }

  // 遍历原始模块的所有命名缓冲区
  for (auto& buffer : module->named_buffers(/*recurse=*/false)) {
    // 如果缓冲区需要梯度
    if (buffer.value().requires_grad()) {
      // 创建一个 ReduceAdd 的共享指针，设定其设备
      auto grad_fn = std::make_shared<ReduceAdd>((*buffer).device());
      // 设置 grad_fn 的下一个边缘，收集缓冲区的下一个边缘
      grad_fn->set_next_edges(autograd::collect_next_edges(*buffer));

      // 遍历设备列表
      for (const auto i : c10::irange(devices.size())) {
        // 在每个复制品的缓冲区历史中设置 grad_fn
        autograd::set_history(replicas[i]->buffers_[buffer.key()], grad_fn);
      }
    }
  }

  // 遍历原始模块的所有子模块
  for (auto& child : module->children_) {
    // 准备子模块的复制品列表
    std::vector<std::shared_ptr<Module>> child_replicas;
    child_replicas.reserve(devices.size());
    // 为每个设备复制子模块并加入列表
    for (auto& replica : replicas) {
      child_replicas.push_back(replica->children_[child.key()]);
    }

    // 递归为所有子模块设置梯度边缘
    replicate_grad_edges(*child, child_replicas, devices);
  }
}

namespace parallel {

/// Replicates a module on the given list of devices.
/// A replica is created by calling `clone()` on the module. For this, the
/// module must inherit from `nn::Cloneable`, or define its own `clone()`
/// method, which is expected to perform a deep copy of the module.
template <typename ModuleType>
std::vector<std::shared_ptr<ModuleType>> replicate(
    const std::shared_ptr<ModuleType>& module,
    const std::vector<Device>& devices) {
  // 准备复制品列表
  std::vector<std::shared_ptr<ModuleType>> replicas;
  replicas.reserve(devices.size());
  // 遍历设备列表，为每个设备创建模块的克隆体并加入复制品列表
  for (const auto& device : devices) {
    replicas.push_back(
        std::dynamic_pointer_cast<ModuleType>(module->clone(device)));
  }
  // 配置梯度边缘，使复制品参数指向原始模块参数。见 [Replicating Modules]
  replicate_grad_edges(module, replicas, devices);
  // 返回复制品列表
  return replicas;
}

/// Replicates a module holder on the given list of devices.
/// This method allows calling `replicate()` with a module holder, such as
/// `Linear`.
template <typename ModuleType>
std::vector<ModuleHolder<ModuleType>> replicate(
    const ModuleHolder<ModuleType>& module,
    const std::vector<Device>& devices) {
  // 调用上面的 replicate 函数获取模块的复制品指针，并将其转换为 ModuleHolder 的列表
  auto ptrs = replicate(module.ptr(), devices);
  return std::vector<ModuleHolder<ModuleType>>(ptrs.begin(), ptrs.end());
}

/// Applies the given inputs to the given modules in a parallel fashion.
/// Conceptually, a thread is spawned for each `(module, input)` pair, in which
/// `forward()` is called on the module with its corresponding input. The
/// outputs of the individual calls are stored in a vector and returned.
///
/// The first exception caught by any thread is stashed and rethrown after all
/// threads have completed their operation.
///
/// Further remarks:
/// 1. The length of the module container must match the length of the inputs.
/// 2. If a list of devices is supplied, it must match the list of modules in
/// length. Each device will be set to the current default device during the
/// invocation of the respective module. This means any tensors allocated on the
/// default device inside the module will be constructed on this device.
template <typename ModuleType>
std::vector<Tensor> parallel_apply(
    std::vector<ModuleType>& modules,                      // 输入：模块类型的向量
    const std::vector<Tensor>& inputs,                     // 输入：张量的向量
    const optional<std::vector<Device>>& devices = nullopt // 可选输入：设备的向量，默认为无
) {
  TORCH_CHECK(
      modules.size() == inputs.size(), "Must have as many inputs as modules");  // 检查：模块数量必须与输入数量相等

  if (devices) {
    TORCH_CHECK(
        modules.size() == devices->size(),
        "Must have as many devices as modules");  // 如果提供了设备列表，则检查设备数量必须与模块数量相等
  }

  std::vector<Tensor> outputs(modules.size());  // 初始化：与模块数量相等的输出张量向量
  std::mutex mutex;  // 初始化：互斥锁，用于保护共享资源

  // std::exception_ptr can be passed between threads:
  // > An instance of std::exception_ptr may be passed to another function,
  // > possibly on another thread, where the exception may be rethrown [...].
  // https://en.cppreference.com/w/cpp/error/exception_ptr
  std::exception_ptr exception;  // 初始化：用于存储捕获的异常指针

  at::parallel_for(
      /*begin=*/0,
      /*end=*/modules.size(),
      /*grain_size=*/1,
      [&modules, &inputs, &devices, &outputs, &mutex, &exception](
          int64_t index, int64_t stop) {
        for (; index < stop; ++index) {
          try {
            auto output = modules[index]->forward(inputs[index]);  // 调用：模块的forward方法，使用对应的输入
            output =
                output.to(devices ? (*devices)[index] : inputs[index].device());  // 如果提供了设备列表，则将输出移动到相应的设备上
            std::lock_guard<std::mutex> lock(mutex);  // 锁定：互斥锁，保护对共享资源的访问
            outputs[index] = output;  // 将输出保存到输出张量向量中
          } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);  // 锁定：互斥锁，保护对共享资源的访问
            if (!exception) {
              exception = std::current_exception();  // 捕获异常，并将异常指针存储起来
            }
          }
        }
      });

  if (exception) {
    std::rethrow_exception(exception);  // 如果有捕获的异常，则重新抛出该异常
  }

  return outputs;  // 返回：所有模块的输出张量向量
}

/// Evaluates `module(input)` in parallel across the given `devices`. If
/// `devices` is not supplied, the invocation is parallelized across all
/// available CUDA devices. If `output_device` is supplied, the final, combined
/// tensor will be placed on this device. If not, it defaults to the first
/// device in `devices`.
///
/// In detail, this method performs the following four distinct steps:
/// 1. *Scatter* the input to the given devices,
/// 2. *Replicate* (deep clone) the model on each device,
/// 3. *Evaluate* each module with its input on its device,
/// 4. *Gather* the outputs of each replica into a single output tensor, located
/// on the `output_device`.
template <typename ModuleType>
Tensor data_parallel(
    ModuleType module,
    Tensor input,
    optional<std::vector<Device>> devices = nullopt,
    optional<Device> output_device = nullopt,
    int64_t dim = 0) {
  // 如果未提供设备列表，则获取当前可用的 CUDA 设备列表
  if (!devices) {
    const auto device_count = torch::cuda::device_count();
    TORCH_CHECK(
        device_count > 0, "Expected at least one CUDA device to be available");
    devices = std::vector<Device>();
    devices->reserve(device_count);
    // 填充设备列表
    for (const auto index : c10::irange(device_count)) {
      devices->emplace_back(kCUDA, static_cast<torch::DeviceIndex>(index));
    }
  }
  // 如果未提供输出设备，则使用设备列表中的第一个设备
  if (!output_device) {
    output_device = devices->front();
  }

  // 如果只有一个设备，将模块和输入数据移到该设备上执行前向传播
  if (devices->size() == 1) {
    module->to(devices->front());
    input = input.to(devices->front());
    return module->forward(std::move(input)).to(*output_device);
  }

  // 使用 Scatter 将输入张量分散到所有设备上
  autograd::Scatter scatter(*devices, /*chunk_sizes=*/nullopt, dim);
  auto scattered_inputs = fmap<Tensor>(scatter.apply({std::move(input)}));
  // 如果分散后的输入张量数目小于设备数目，则调整设备列表大小
  if (scattered_inputs.size() < devices->size()) {
    devices->resize(
        scattered_inputs.size(),
        Device(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES));
  }

  // 复制模块到所有设备上，并并行应用模块到分散的输入上
  auto replicas = replicate(module, *devices);
  auto outputs = parallel_apply(replicas, scattered_inputs, *devices);
  // 使用 Gather 将输出从各个设备上收集到单个设备上
  return autograd::Gather(*output_device, dim)
      .apply(fmap<autograd::Variable>(std::move(outputs)))
      .front();
}
```