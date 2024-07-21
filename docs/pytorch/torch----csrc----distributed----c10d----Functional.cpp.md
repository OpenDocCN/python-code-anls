# `.\pytorch\torch\csrc\distributed\c10d\Functional.cpp`

```
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 ATen 核心操作注册相关的头文件
#include <ATen/core/op_registration/op_registration.h>
// 包含 C10 核心调度键相关的头文件
#include <c10/core/DispatchKey.h>
// 包含 Torch 自动求导自定义函数相关的头文件
#include <torch/csrc/autograd/custom_function.h>
// 包含 Torch 自动求导函数相关的头文件
#include <torch/csrc/autograd/function.h>
// 包含 Torch 分布式模块的 GroupRegistry 头文件
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
// 包含 Torch 分布式模块的 ProcessGroup 头文件
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
// 包含 Torch 分布式模块的 RankLocal 头文件
#include <torch/csrc/distributed/c10d/RankLocal.hpp>
// 包含标准库的头文件
#include <utility>

// 声明一个匿名命名空间
namespace {

// 定义一个工作注册类
class WorkRegistry {
 public:
  // 注册工作函数，将张量和工作对象关联起来
  void register_work(
      const at::Tensor& tensor,
      const c10::intrusive_ptr<c10d::Work>& work) {
    // 获取张量的存储对象
    auto storage = tensor.storage().getWeakStorageImpl();
    // 使用互斥锁锁住当前代码块
    std::unique_lock lock(lock_);
    // 尝试将存储对象和工作对象插入注册表中
    auto [it, inserted] = registry_.try_emplace(std::move(storage), work);
    // 检查是否插入成功或者插入的工作对象与已有工作对象不同
    TORCH_CHECK(
        inserted || it->second != work,
        "The tensor storage is already associated with another work.");
  }

  // 弹出与张量关联的工作对象
  c10::intrusive_ptr<c10d::Work> pop_work(const at::Tensor& tensor) {
    // 获取张量的存储对象
    const auto storage = tensor.storage().getWeakStorageImpl();
    // 使用互斥锁锁住当前代码块
    std::unique_lock lock(lock_);
    // 在注册表中查找存储对象对应的工作对象
    auto it = registry_.find(storage);
    // 如果未找到，则返回空指针
    if (it == registry_.end()) {
      return nullptr;
    }
    // 获取存储对象对应的工作对象
    auto work = it->second;
    // 从注册表中移除存储对象和其对应的工作对象
    registry_.erase(it);
    // 返回工作对象
    return work;
  }

  // 析构函数，在对象销毁时调用
  ~WorkRegistry() {
    // 如果仍有未等待的工作对象，其对应的进程组应该已经在此阶段被销毁
    // 任何尝试等待或销毁这些工作对象都将导致混乱的错误
    // 因此，我们发出警告并故意允许这些未等待的工作对象泄漏
    if (!registry_.empty()) {
      TORCH_WARN(
          "At the time of process termination, there are still ",
          registry_.size(),
          " unwaited c10d_functional collective calls. "
          "Please review your program to ensure c10d_functional.wait_tensor() "
          "is invoked on all tensors returned from c10d_functional collective "
          "ops before they are used.");
    }
    // 释放注册表中所有工作对象
    for (auto& it : registry_) {
      it.second.release();
    }
  }

 private:
  // 存储对象到工作对象的映射表
  std::unordered_map<
      c10::weak_intrusive_ptr<c10::StorageImpl>,
      c10::intrusive_ptr<c10d::Work>>
      registry_;
  // 互斥锁，用于保护注册表的并发访问
  std::mutex lock_;
};

// 创建静态的进程注册表对象
static WorkRegistry process_registry;

// 将字符串到 ReduceOp 枚举值的映射关系定义为静态常量映射表
const std::unordered_map<std::string, c10d::ReduceOp> str_to_reduce_op = {
    {"sum", c10d::ReduceOp(c10d::ReduceOp::RedOpType::SUM)},
    {"avg", c10d::ReduceOp(c10d::ReduceOp::RedOpType::AVG)},
    {"product", c10d::ReduceOp(c10d::ReduceOp::RedOpType::PRODUCT)},
    {"min", c10d::ReduceOp(c10d::ReduceOp::RedOpType::MIN)},
    {"max", c10d::ReduceOp(c10d::ReduceOp::RedOpType::MAX)},
    {"band", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BAND)},
    {"bor", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BOR)},
    {"bxor", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BXOR)},
    // TODO: 支持 premul_sum
    // {"premul_sum", c10d::ReduceOp(c10d::ReduceOp::RedOpType::PREMUL_SUM)},
    {"unused", c10d::ReduceOp(c10d::ReduceOp::RedOpType::UNUSED)}};
c10d::ReduceOp to_reduce_op(const std::string& reduce_op) {
  // 查找给定字符串表示的 reduce_op 对应的枚举值
  auto it = str_to_reduce_op.find(reduce_op);
  // 如果找不到对应的 reduce_op，则抛出错误信息
  TORCH_CHECK(
      it != str_to_reduce_op.end(), "Unrecognized reduce_op: ", reduce_op);
  // 返回找到的 reduce_op 对应的枚举值
  return it->second;
}

at::Tensor& all_reduce_(
    at::Tensor& input,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  // 创建 allreduce 的选项对象 opts
  c10d::AllreduceOptions opts;
  // 将输入的 reduce_op 字符串转换为对应的枚举值，并赋给 opts
  opts.reduceOp = to_reduce_op(reduce_op);

  // 将输入的 Tensor 放入 vector 中
  std::vector<at::Tensor> inputs{input};
  // 解析给定的 group_name，并获取与之关联的进程组
  auto group = c10d::resolve_process_group(group_name);
  // 执行 allreduce 操作，返回异步执行的工作对象
  auto work = group->allreduce(inputs, opts);
  // 将 Tensor 和其对应的工作对象注册到本地的 WorkRegistry 中
  c10d::RankLocal<WorkRegistry>::get().register_work(input, work);
  // 返回原始输入的 Tensor 引用
  return input;
}

at::Tensor all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  // 根据输入 Tensor 创建一个副本 output，并设置为连续内存格式
  auto output = input.clone(at::MemoryFormat::Contiguous);
  // 调用 all_reduce_ 函数执行实际的 allreduce 操作，并返回结果
  return all_reduce_(output, std::move(reduce_op), std::move(group_name));
}

std::vector<at::Tensor> all_reduce_coalesced_(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  // 创建 allreduce_coalesced 的选项对象 opts
  c10d::AllreduceCoalescedOptions opts;
  // 将输入的 reduce_op 字符串转换为对应的枚举值，并赋给 opts
  opts.reduceOp = to_reduce_op(reduce_op);

  // 解析给定的 group_name，并获取与之关联的进程组
  auto group = c10d::resolve_process_group(group_name);
  // 执行 allreduce_coalesced 操作，返回异步执行的工作对象
  auto work = group->allreduce_coalesced(inputs, opts);
  // 将每个输入 Tensor 及其对应的工作对象注册到本地的 WorkRegistry 中
  for (const auto& tensor : inputs) {
    c10d::RankLocal<WorkRegistry>::get().register_work(tensor, work);
  }
  // 返回原始输入的 Tensor vector
  return inputs;
}

std::vector<at::Tensor> all_reduce_coalesced(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::vector<at::Tensor> inputs,
    std::string reduce_op,
    std::string group_name) {
  // 创建一个输出 Tensor 的 vector
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  // 对于每个输入的 Tensor，创建一个连续内存格式的副本，并放入 outputs 中
  for (const auto& tensor : inputs) {
    outputs.push_back(tensor.clone(at::MemoryFormat::Contiguous));
  }
  // 调用 all_reduce_coalesced_ 函数执行实际的 allreduce_coalesced 操作，并返回结果
  return all_reduce_coalesced_(
      outputs, std::move(reduce_op), std::move(group_name));
}

at::Tensor allocate_all_gather_output(
    const at::Tensor& input,
    int64_t group_size) {
  // 获取输入 Tensor 的尺寸信息，并调整为符合 all gather 要求的输出尺寸
  auto output_size = input.sizes().vec();
  output_size[0] *= group_size;
  // 创建一个空的 Tensor，具有与输入相同的数据类型和设备，并返回
  return at::empty(
      output_size,
      at::TensorOptions().dtype(input.dtype()).device(input.device()));
}

std::vector<at::Tensor> all_gather_into_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    int64_t group_size,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  // 创建一个输出 Tensor 的 vector
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  // 对于每个输入的 Tensor，根据 group_size 分配相应的 all gather 输出 Tensor
  for (const auto& tensor : inputs) {
    outputs.push_back(allocate_all_gather_output(tensor, group_size));
  }

  // 解析给定的 group_name，并获取与之关联的进程组
  auto group = c10d::resolve_process_group(group_name);
  // 执行 allgather_into_tensor_coalesced 操作，返回异步执行的工作对象
  auto work = group->allgather_into_tensor_coalesced(outputs, inputs);
  // 将每个输出 Tensor 及其对应的工作对象注册到本地的 WorkRegistry 中
  for (const auto& tensor : outputs) {
    c10d::RankLocal<WorkRegistry>::get().register_work(tensor, work);
  }
  // 返回包含所有输出 Tensor 的 vector
  return outputs;
}
// 将单个输入张量广播到所有进程，并收集结果到一个张量中返回
at::Tensor all_gather_into_tensor(
    const at::Tensor& input,        // 输入张量
    int64_t group_size,             // 进程组大小
    std::string group_name) {       // 进程组名称
  std::vector<at::Tensor> inputs{input};   // 将输入张量放入向量中
  return all_gather_into_tensor_coalesced(
      inputs, group_size, std::move(group_name))[0];   // 调用聚合操作函数并返回第一个张量
}

// 将输入张量的数据按照指定的规则减少并分散到所有进程
// 输出结果存储在给定的输出张量中
at::Tensor& all_gather_into_tensor_out(
    at::Tensor& input,              // 输入张量
    int64_t group_size,             // 进程组大小
    const std::string& group_name,  // 进程组名称
    at::Tensor& output) {           // 输出张量
  c10d::AllgatherOptions opts;      // 初始化 Allgather 选项

  auto group = c10d::resolve_process_group(group_name);  // 解析进程组
  auto work = group->_allgather_base(output, input, opts);  // 执行所有聚合操作的基本方法
  c10d::RankLocal<WorkRegistry>::get().register_work(output, work);  // 注册工作到工作注册表中
  return output;                    // 返回输出张量的引用
}

// 分配用于减少分散操作的输出张量
at::Tensor allocate_reduce_scatter_output(
    const at::Tensor& input,        // 输入张量
    const int64_t group_size) {     // 进程组大小
  auto output_size = input.sizes().vec();   // 获取输入张量的大小
  if (output_size[0] % group_size != 0) {   // 检查第一个维度是否能被进程组大小整除
    LOG(WARNING) << "The first dimension of the reduce_scatter input ("
                 << output_size[0] << ") is not divisible by the group size ("
                 << group_size << ").";     // 若不能，输出警告日志
  }
  output_size[0] /= group_size;     // 计算输出张量的第一个维度大小
  return at::empty(
      output_size,                  // 创建空的输出张量，与输入张量相同类型和设备
      at::TensorOptions().dtype(input.dtype()).device(input.device()));
}

// 将输入张量向量按照指定规则减少并分散到所有进程
std::vector<at::Tensor> reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor> inputs, // 输入张量向量
    std::string reduce_op,          // 减少操作类型
    int64_t group_size,             // 进程组大小
    std::string group_name) {       // 进程组名称
  c10d::ReduceScatterOptions opts;  // 初始化 ReduceScatter 选项
  opts.reduceOp = to_reduce_op(reduce_op);  // 设置减少操作类型

  std::vector<at::Tensor> outputs;  // 存储输出张量的向量
  outputs.reserve(inputs.size());   // 预留输入张量向量大小的空间
  for (const auto& tensor : inputs) {
    outputs.push_back(allocate_reduce_scatter_output(tensor, group_size));  // 分配减少分散操作的输出张量
  }

  auto group = c10d::resolve_process_group(group_name);  // 解析进程组
  auto work = group->reduce_scatter_tensor_coalesced(outputs, inputs, opts);  // 执行减少分散操作
  for (const auto& tensor : outputs) {
    c10d::RankLocal<WorkRegistry>::get().register_work(tensor, work);  // 注册工作到工作注册表中
  }
  return outputs;  // 返回输出张量的向量
}

// 将单个输入张量按照指定规则减少并分散到所有进程，并返回结果张量
at::Tensor reduce_scatter_tensor(
    const at::Tensor& input,        // 输入张量
    std::string reduce_op,          // 减少操作类型
    int64_t group_size,             // 进程组大小
    std::string group_name) {       // 进程组名称
  std::vector<at::Tensor> inputs{input};   // 将输入张量放入向量中
  return reduce_scatter_tensor_coalesced(
      inputs, std::move(reduce_op), group_size, std::move(group_name))[0];  // 调用减少分散操作并返回第一个张量
}

// 将单个输入张量广播到所有进程，按照指定的划分大小进行分割
// 并返回每个进程分配的输入张量
at::Tensor all_to_all_single(
    const at::Tensor& input,                  // 输入张量
    std::vector<int64_t> output_split_sizes,  // 输出张量的分割大小
    std::vector<int64_t> input_split_sizes,   // 输入张量的分割大小
    std::string group_name) {                 // 进程组名称
    // 定义函数，将输入张量分组全局交换，并返回输出张量
    std::vector<int64_t> output_sizes = input.sizes().vec();
    // 获取输入张量的维度大小，并用其构建输出张量的大小
    output_sizes[0] = std::accumulate(
        output_split_sizes.begin(), output_split_sizes.end(), int64_t(0));
    // 计算输出张量的第一个维度大小，作为所有分割大小的累加和
    auto output = input.new_empty(output_sizes);
    // 根据计算出的大小创建空的输出张量
    
    auto group = c10d::resolve_process_group(group_name);
    // 解析给定的进程组名称，返回相应的进程组对象
    auto work = group->alltoall_base(
        output,
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<at::Tensor&>(input),
        output_split_sizes,
        input_split_sizes);
    // 在进程组上执行基于全局交换的分组全局交换操作，并返回工作对象
    
    c10d::RankLocal<WorkRegistry>::get().register_work(output, work);
    // 在本地注册输出张量和对应的工作对象
    return output;
    // 返回执行全局交换后的输出张量
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
// 在给定的分布式组中广播输入张量数据，使用指定的根源进程(src)和组名称(group_name)
at::Tensor& broadcast_(at::Tensor& input, int64_t src, std::string group_name) {
  c10d::BroadcastOptions opts;
  opts.rootRank = src;
  std::vector<at::Tensor> inputs{input};

  auto group = c10d::resolve_process_group(group_name);
  auto work = group->broadcast(inputs, opts);
  c10d::RankLocal<WorkRegistry>::get().register_work(input, work);
  return input;
}

// 创建输入张量的副本，以保证内存格式为连续的，并对其进行广播
at::Tensor broadcast(
    const at::Tensor& input,
    int64_t src,
    std::string group_name) {
  auto output = input.clone(at::MemoryFormat::Contiguous);
  return broadcast_(output, src, std::move(group_name));
}

// 等待给定的张量对应的操作完成
at::Tensor wait_tensor(const at::Tensor& tensor) {
  auto work = c10d::RankLocal<WorkRegistry>::get().pop_work(tensor);
  if (work != nullptr) {
    work->wait();
  }
  return tensor;
}

} // namespace

}

namespace {
// 执行单向传播操作的自动微分函数，用于在分布式设置中进行all-to-all通信
class AllToAllSingle : public torch::autograd::Function<AllToAllSingle> {
 public:
  // 前向传播函数，实现all-to-all单向传播操作
  static torch::autograd::Variable forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      std::vector<int64_t> output_split_sizes,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      std::vector<int64_t> input_split_sizes,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      std::string group_name) {
    // 保存上下文中的数据用于反向传播
    ctx->saved_data["output_split_sizes"] = input_split_sizes;
    ctx->saved_data["input_split_sizes"] = output_split_sizes;
    ctx->saved_data["group_name"] = group_name;

    // 调用C10调度器执行对应的all-to-all单向传播函数
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("_c10d_functional::all_to_all_single", "")
        .typed<decltype(all_to_all_single)>()
        .call(input, output_split_sizes, input_split_sizes, group_name);
  }

  // 反向传播函数，实现对all-to-all单向传播操作的反向求导
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_out_list) {
    // 从上下文中恢复保存的数据
    const std::vector<int64_t>& output_split_sizes =
        ctx->saved_data["output_split_sizes"].toIntVector();
    const std::vector<int64_t>& input_split_sizes =
        ctx->saved_data["input_split_sizes"].toIntVector();
    const std::string& group_name = ctx->saved_data["group_name"].toStringRef();

    // 确保梯度列表长度正确
    DCHECK(grad_out_list.size() == 1);
    auto grad_out = grad_out_list[0].contiguous();

    // 调用C10调度器执行对应的all-to-all单向传播反向传播函数
    auto out =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("_c10d_functional::all_to_all_single", "")
            .typed<decltype(all_to_all_single)>()
            .call(grad_out, output_split_sizes, input_split_sizes, group_name);

    // 执行显式等待操作以避免CUDA流问题
    // TODO: 跟踪等待中的活动CUDA流
    out = c10::Dispatcher::singleton()
              .findSchemaOrThrow("_c10d_functional::wait_tensor", "")
              .typed<decltype(wait_tensor)>()
              .call(out);

    // 返回梯度列表
    return {out, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};
// 调用 AllToAllSingle 的静态方法 apply，将输入张量按照给定参数进行分组传输
at::Tensor all_to_all_single_autograd(
    const at::Tensor& input,
    const std::vector<int64_t>& output_split_sizes,
    const std::vector<int64_t>& input_split_sizes,
    const std::string& group_name) {
  return AllToAllSingle::apply(
      input, output_split_sizes, input_split_sizes, group_name);
}

// 定义 ReduceScatterTensor 类，继承自 torch::autograd::Function
class ReduceScatterTensor
    : public torch::autograd::Function<ReduceScatterTensor> {
 public:
  // 前向传播方法，用于执行张量的 reduce scatter 操作
  static torch::autograd::Variable forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const std::string& reduce_op,
      int64_t group_size,
      const std::string& group_name) {
    // 检查 reduce_op 是否为 "sum"，仅支持求和操作
    TORCH_CHECK(reduce_op == "sum", "Only sum reduce op is supported");

    // 保存组大小和组名称到 AutogradContext 中
    ctx->saved_data["group_size"] = group_size;
    ctx->saved_data["group_name"] = group_name;

    // 调用 reduce_scatter_tensor 函数，执行 reduce scatter 操作
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("_c10d_functional::reduce_scatter_tensor", "")
        .typed<decltype(reduce_scatter_tensor)>()
        .call(input, reduce_op, group_size, group_name);
  }

  // 反向传播方法，用于计算梯度
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_out_list) {
    // 从 AutogradContext 中获取保存的组大小和组名称
    const int64_t group_size = ctx->saved_data["group_size"].toInt();
    const std::string& group_name = ctx->saved_data["group_name"].toStringRef();

    // 检查梯度列表的大小是否为 1
    DCHECK(grad_out_list.size() == 1);
    auto grad_out = grad_out_list[0];

    // 调用 all_gather_into_tensor 函数，将梯度数据进行 gather 操作
    auto out =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("_c10d_functional::all_gather_into_tensor", "")
            .typed<decltype(all_gather_into_tensor)>()
            .call(grad_out, group_size, group_name);

    // 执行显式等待，避免 CUDA 流问题
    // TODO: 跟踪活跃的 CUDA 流以进行等待操作
    out = c10::Dispatcher::singleton()
              .findSchemaOrThrow("_c10d_functional::wait_tensor", "")
              .typed<decltype(wait_tensor)>()
              .call(out);

    // 返回梯度列表，包括一个输出张量和其余三个空张量（对于四个参数的函数）
    return {
        out,
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
    };
  }
};

// 调用 ReduceScatterTensor 的静态方法 apply，执行 reduce scatter 操作
at::Tensor reduce_scatter_tensor_autograd(
    const at::Tensor& input,
    const std::string& reduce_op,
    int64_t group_size,
    const std::string& group_name) {
  return ReduceScatterTensor::apply(input, reduce_op, group_size, group_name);
}

// 定义 AllGatherIntoTensor 类，继承自 torch::autograd::Function
class AllGatherIntoTensor
    : public torch::autograd::Function<AllGatherIntoTensor> {
 public:
  // 前向传播方法，用于执行张量的 all gather 操作
  static torch::autograd::Variable forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      int64_t group_size,
      const std::string& group_name) {
    // 将组大小和组名称保存到 AutogradContext 中
    ctx->saved_data["group_size"] = group_size;
    ctx->saved_data["group_name"] = group_name;
    // 返回全局唯一的 c10::Dispatcher 对象的实例，并调用 findSchemaOrThrow 方法查找指定函数的模式，这里是 "_c10d_functional::all_gather_into_tensor"，返回对应的函数指针类型。
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("_c10d_functional::all_gather_into_tensor", "")
        .typed<decltype(all_gather_into_tensor)>()
        .call(input, group_size, group_name);
  }

  // 定义静态方法 backward，用于计算梯度反向传播
  static torch::autograd::variable_list backward(
      // 接受 AutogradContext 指针 ctx 和梯度列表 grad_out_list 作为参数
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_out_list) {
    // 从 ctx 的 saved_data 中获取 "group_size" 对应的整数值
    const int64_t group_size = ctx->saved_data["group_size"].toInt();
    // 从 ctx 的 saved_data 中获取 "group_name" 对应的字符串引用
    const std::string& group_name = ctx->saved_data["group_name"].toStringRef();

    // 使用 DCHECK 断言确保 grad_out_list 的大小为 1
    DCHECK(grad_out_list.size() == 1);
    // 获取 grad_out_list 中的第一个梯度张量
    auto grad_out = grad_out_list[0];

    // 调用 c10::Dispatcher 的 singleton 方法返回全局唯一的实例，查找指定函数的模式 "_c10d_functional::reduce_scatter_tensor"，并返回对应函数指针类型。
    auto out =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("_c10d_functional::reduce_scatter_tensor", "")
            .typed<decltype(reduce_scatter_tensor)>()
            .call(grad_out, "sum", group_size, group_name);

    // 执行显式等待以避免 CUDA 流问题
    // TODO: 在 wait 中跟踪活动的 CUDA 流
    out = c10::Dispatcher::singleton()
              .findSchemaOrThrow("_c10d_functional::wait_tensor", "")
              .typed<decltype(wait_tensor)>()
              .call(out);

    // 返回包含三个张量的变量列表，其中第一个是 out，后两个是空张量
    return {
        out,
        at::Tensor(),
        at::Tensor(),
    };
  }
namespace {
// 定义一个匿名命名空间，用于限定函数和变量的作用域

// 在指定通信组中进行数据收集和分发操作
at::Tensor shard_dim_alltoall(
    const at::Tensor& input,
    int64_t gather_dim,
    int64_t shard_dim,
    const std::string& group_name) {
  // 解析指定名称的进程组
  auto group = c10d::resolve_process_group(group_name);
  // 获取进程组的大小
  auto group_size = group->getSize();
  
  // 获取输入张量的维度大小
  std::vector<int64_t> output_sizes = input.sizes().vec();
  
  // 检查输入张量在指定维度上是否可以被进程组大小整除
  if (output_sizes[shard_dim] % group_size != 0) {
    LOG(WARNING) << "The first dimension of the shard_dim_alltoall input ("
                 << output_sizes[shard_dim]
                 << ") is not divisible by the group size (" << group_size
                 << ").";
  }
  
  // 调整输出张量在指定维度上的大小
  output_sizes[shard_dim] = output_sizes[shard_dim] / group_size;
  
  // 准备输入张量的向量
  std::vector<at::Tensor> inputs;
  inputs.reserve(group_size);
  auto length = output_sizes[shard_dim];
  
  // 分割输入张量并存储到输入向量中
  for (int i = 0; i < group_size; i++) {
    inputs.push_back(input.narrow(shard_dim, i * length, length).contiguous());
  }
  
  // 分配输出张量的向量
  std::vector<at::Tensor> outputs;
  outputs.reserve(group_size);
  
  // 创建空的输出张量并存储到输出向量中
  for (int i = 0; i < group_size; i++) {
    outputs.push_back(input.new_empty(output_sizes).contiguous());
  }
  
  // 执行所有到所有的通信操作
  auto work = group->alltoall(outputs, inputs);
  
  // 等待通信操作完成
  work->wait();
  
  // 返回在指定维度上连接所有输出张量的结果
  return at::cat(outputs, gather_dim);
}
} // namespace

// DTensor comm op registry
// 注册 DTensor 通信操作
TORCH_LIBRARY(_dtensor, m) {
  // 定义 shard_dim_alltoall 函数的 Torch 库接口
  m.def(
      "shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, str group_name) -> Tensor",
      // 使用 CompositeExplicitAutograd 分发键分发函数
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::shard_dim_alltoall),
      // 添加 pt2_compliant_tag 标签
      {at::Tag::pt2_compliant_tag});
}
```