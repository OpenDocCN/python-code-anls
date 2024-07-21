# `.\pytorch\torch\csrc\distributed\c10d\comm.cpp`

```py
// 包含了 Torch 分布式通信库的头文件
#include <torch/csrc/distributed/c10d/comm.hpp>

// 包含了标准库的双端队列头文件
#include <deque>

// 包含了 ATen 核心的函数头文件
#include <ATen/core/functional.h>
// 包含了 C10 的工具类中的范围操作头文件
#include <c10/util/irange.h>
// 包含了 Torch 分布式通信库中的 Reducer 类的头文件
#include <torch/csrc/distributed/c10d/reducer.hpp>
// 包含了 Torch 实用工具中的张量展平函数头文件
#include <torch/csrc/utils/tensor_flatten.h>

// 定义了 c10d 命名空间
namespace c10d {
// 匿名命名空间用于实现局部类和函数
namespace {

// 定义了一个用于广播操作的工作类
class BroadcastWork {
 public:
  // 构造函数，初始化广播操作
  BroadcastWork(
      const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
      std::vector<at::Tensor> bucket_tensors,
      int root_rank = 0)
      : bucket_tensors_(std::move(bucket_tensors)),
        flat_tensor_({torch::utils::flatten_dense_tensors(bucket_tensors_)}) {
    // 设置广播选项
    BroadcastOptions broadcastOptions;
    broadcastOptions.rootRank = root_rank;
    // 执行广播操作并保存返回的工作对象
    work_ = process_group->broadcast(flat_tensor_, broadcastOptions);
  }

  // 完成广播操作
  void finish() {
    // 等待广播操作完成
    work_->wait();

    // 将广播操作的输出复制回原始张量
    auto output_tensors = torch::utils::unflatten_dense_tensors(
        flat_tensor_.front(), bucket_tensors_);
    // 内部断言确保输出张量的大小与原始张量一致
    TORCH_INTERNAL_ASSERT(output_tensors.size() == bucket_tensors_.size());
    for (const auto i : c10::irange(output_tensors.size())) {
      // 如果输出张量非空，则进行复制，避免形状不匹配的问题
      if (output_tensors[i].numel() != 0) {
        bucket_tensors_[i].copy_(output_tensors[i], /*non_blocking=*/true);
      }
    }
  }

 protected:
  // 需要广播的张量列表，保证它们位于同一设备且具有相同的数据类型
  std::vector<at::Tensor> bucket_tensors_;

  // 包含单个展平张量的向量，该张量包含 bucket_tensors_ 中张量的内容
  // 必须存储在向量中，因为 c10d::ProcessGroup::broadcast 接受向量作为参数
  std::vector<at::Tensor> flat_tensor_;

 private:
  // 在构造时启动的广播工作对象
  c10::intrusive_ptr<c10d::Work> work_;
};

} // namespace

// 将多个张量广播到进程组中的所有进程
void broadcast_coalesced(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank) {
  // 将张量按照最大缓冲区大小分组为桶
  // 此函数支持多设备，因此张量可以分布在多个设备上，可以包含 CPU 和 CUDA 张量的混合
  auto [buckets, _] =
      compute_bucket_assignment_by_size(tensors.vec(), {buffer_size});

  // 通过索引查找输入张量列表中的张量
  const auto lookup = [&tensors](size_t index) { return tensors[index]; };

  // 维护最多 2 个正在进行中的广播操作，以避免分配过多内存（如果指定的张量非常大）
  std::deque<BroadcastWork> in_flight;
  constexpr auto max_in_flight = 2;
  for (const auto& bucket : buckets) {
    # 如果当前正在处理的任务数量达到最大允许的并行处理数，则执行以下操作
    if (in_flight.size() >= max_in_flight) {
      # 完成最先进入处理队列的任务，并将其移出队列
      in_flight.front().finish();
      in_flight.pop_front();
    }
    
    # 将新的任务添加到处理队列末尾，直到所有任务处理完毕
    in_flight.emplace_back(process_group, c10::fmap(bucket, lookup), rank);
    }
    
    # 当前没有正在处理的任务时，执行以下操作，直到处理队列为空
    while (!in_flight.empty()) {
      # 完成最先进入处理队列的任务，并将其移出队列
      in_flight.front().finish();
      in_flight.pop_front();
    }
}

# 定义 GradBucket 类的成员函数 getGradients()，返回一个存储张量的向量
std::vector<at::Tensor> GradBucket::getGradients() const {
  # 创建一个空的张量向量 per_parameter_tensors
  std::vector<at::Tensor> per_parameter_tensors;
  # 获取参数偏移量的数量
  size_t num_parameters = offsets_.size();
  # 预留空间以容纳参数数量的张量
  per_parameter_tensors.reserve(num_parameters);
  # 遍历每个参数的索引 i，使用 offsets_ 和 lengths_ 创建张量并添加到 per_parameter_tensors 中
  for (const auto i : c10::irange(num_parameters)) {
    per_parameter_tensors.push_back(
        buffer_.slice(0, offsets_[i], offsets_[i] + lengths_[i])
            .view(sizes_vec_[i]));
  }
  # 返回存储所有参数张量的向量
  return per_parameter_tensors;
}

# 进入命名空间 detail
namespace detail {

# 定义函数 parseCppCommHookResult，解析 c10::IValue 类型的结果并返回一个张量
at::Tensor parseCppCommHookResult(const c10::IValue& result) {
  # 如果结果是 PyObject 类型，则提取其中的张量并返回第一个张量
  if (result.isPyObject()) {
    std::vector<at::Tensor> tensors =
        result.toPyObjectHolder()->extractTensors();
    return tensors[0];
  }
  # 如果结果是 Tensor 或者 TensorList 类型，则返回第一个张量
  TORCH_INTERNAL_ASSERT(
      result.isTensor() || result.isTensorList(),
      "expected the hook result is either a Tensor or a TensorList found ",
      result.tagKind());

  if (result.isTensor()) {
    return result.toTensor();
  }

  return result.toTensorVector()[0];
}

} // namespace detail

} // namespace c10d
```