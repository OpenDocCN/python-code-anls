# `.\pytorch\torch\csrc\distributed\c10d\reducer.hpp`

```
#pragma once

#include <c10/core/ScalarType.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue_inl.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/default_comm_hooks.hpp>
#include <torch/csrc/distributed/c10d/reducer_timer.hpp>
#ifndef _WIN32
#include <torch/csrc/distributed/autograd/context/context.h>
#endif

namespace c10d {

// 默认第一个桶的大小为 1MB
constexpr int kDefaultFirstBucketBytes = int(1024 * 1024);
// 桶的大小上限为 25MB
constexpr int kDefaultBucketBytesCap = int(25 * 1024 * 1024);
// 每隔 kDDPRuntimeLoggingSampleRate 次迭代收集一次运行时统计信息
constexpr int kDDPRuntimeLoggingSampleRate = 100;

// 前向声明
class Logger;

// 用于单个桶的本地累加器类型
struct BucketAccumulator {
  // 存储索引
  std::vector<size_t> indices;
  // 当前大小
  size_t size = 0;
  // 大小限制
  size_t size_limit = 0;
};
// 定义一个名为 Reducer 的类，用于管理分布式数据并行模型的参数归约
class Reducer {
 public:
  // 构造函数，接收模型参数、桶索引、每个桶大小限制、进程组、稀疏梯度期望、
  // 桶字节容量限制、是否查找未使用参数、是否以梯度视图作为桶视图、参数名字典、
  // 第一个桶的字节容量限制等参数进行初始化
  explicit Reducer(
      std::vector<at::Tensor> params,
      std::vector<std::vector<size_t>> bucket_indices,
      const std::vector<size_t>& per_bucket_size_limits,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group,
      std::vector<bool> expect_sparse_gradients,
      int64_t bucket_bytes_cap,
      bool find_unused_parameters,
      bool gradient_as_bucket_view,
      std::unordered_map<size_t, std::string> param_names,
      int64_t first_bucket_bytes_cap);

  // 析构函数，抛出异常时不使用 noexcept
  ~Reducer() noexcept(false);

  // 重新初始化桶分配，接收桶索引的列表，对桶内变量的设备和维度进行验证
  void initialize_buckets(std::vector<std::vector<size_t>> bucket_indices);

  // 自动求导钩子函数，记录索引
  void autograd_hook(size_t index);

  // 准备进行反向传播，接收前向计算的输出作为参数
  void prepare_for_backward(const std::vector<at::Tensor>& outputs);

  // 在 DistributedDataParallel 的 forward 函数开始时调用，记录前向传播开始时间
  void prepare_for_forward();

  // 返回梯度就绪的相对时间（单位纳秒），相对于调用 prepare_for_backward 的时间
  // 返回值是单个模型副本的参数向量
  std::vector<int64_t> get_backward_stats() const {
    return backward_stats_;
  }

  // 注册一个通信钩子到 Reducer，钩子类型为 CommHookInterface，支持 Python 和 CPP 钩子
  // 只能在调用 backward 之前调用一次，不能与 register_builtin_comm_hook 同时调用
  void register_comm_hook(std::unique_ptr<CommHookInterface> iface);

  // 注册一个内置的 C++ 通信钩子到 Reducer
  // 只能在调用 backward 之前调用一次，不能与 register_comm_hook 同时调用
  void register_builtin_comm_hook(c10d::BuiltinCommHookType comm_hook_type);

  // 通知 Reducer 在反向传播时优化器已经运行，不需要从桶中复制梯度
  void set_optimizer_in_backward() {
  // 将 optim_in_backward_ 设置为 true
  optim_in_backward_ = true;
};

// 使用 GradBucket 实例运行 allreduce 或已安装的通信钩子。
c10::intrusive_ptr<c10::ivalue::Future> run_comm_hook(
    GradBucket& grad_bucket);

// 运行默认的 allreduce 钩子。
c10::intrusive_ptr<c10::ivalue::Future> run_allreduce_hook(
    GradBucket& grad_bucket);

// 按照 buckets_ 的顺序返回梯度桶。这是跨进程进行梯度归约时的顺序。
// 如果 return_zero_tensors=true，则返回与原张量相同形状的零张量而不是真实张量。
std::vector<c10d::GradBucket> get_grad_buckets(
    bool return_zero_tensors = true) const;

// 根据 rebuilt_params_ 和 rebuilt_param_indices_ 重新构建桶，
// 根据反向传播中张量接收梯度的时间顺序。
// TODO: 这个函数会进行广播通信调用，并且可以与下一个 forward() 调用重叠，
// 因此它可以是异步的。对于 find_unused_parameters = true 的情况，
// 我们可能会多次重建桶，因此参数索引顺序可能更频繁地改变。
// 对于 find_unused_parameters = false 的情况，桶只会重建一次，
// 性能成本可以忽略不计。如果桶已重建，则返回 true。
bool rebuild_buckets();

// 设置稀疏元数据。
void setSparseMetadata(std::map<std::string, at::Tensor>& metadata);

// 安装应在反向传播结束时等待的 futures。
// 当前仅由用户定义的自定义缓冲区归约钩子使用，但可以泛化为需要等待的任何用户发起的 futures。
void install_futures(c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futs);

// 如果应重新构建桶，则返回 true，否则返回 false。
// 我们仅在第一次迭代后重建桶一次，并且如果 find_unused_parameters_ 为 true，则从不重建它们。
inline bool should_rebuild_buckets() const {
#ifndef _WIN32
  // 如果不是在 Windows 平台编译，进行静态断言，确保 GradCallback 与 DistAutogradContext::GradCallback 相同
  static_assert(
      std::is_same_v<
          GradCallback,
          torch::distributed::autograd::DistAutogradContext::GradCallback>);
#endif

// 为特定变量执行梯度回调函数
void runGradCallbackForVariable(at::Tensor& variable, GradCallback&& cb);

// 此函数在 `initialize_buckets()` 内部调用。它初始化 `bucket_views_in` 和 `bucket_views_out`，
// 每个变量的梯度视图都与桶的扁平化 `gradients` 张量相关联。视图作为 `copy_()` 每个梯度数据进出扁平化 `gradients` 张量的入口点。
void initialize_bucket_views(Bucket& bucket);

// 此函数在 `finalize_backward` 中调用，仅在注册了 DDP 通信钩子的情况下执行，重新创建 `bucket_views_out`，
// 使用 `future_work` 的结果填充。
void populate_bucket_views_out(Bucket& bucket, at::Tensor& tensor);

// 如果 gradient_as_bucket_view_ 为 false，在所有reduce桶后，将桶结果复制回梯度。
void copy_bucket_to_grad(
    at::Tensor& variable,
    Reducer::Bucket& bucket,
    size_t intra_bucket_index,
    bool global_unused);

// 在将梯度复制到桶之前，检查梯度和桶视图的布局。
void check_grad_layout(const at::Tensor& grad, const at::Tensor& bucket_view);

// 一个桶包含要减少的 [1..N] 个梯度，这些梯度具有相同的 dtype 和 device。
// 在减少之前将梯度合并在一起可以减少开销并/或更快地完成时间。
// 合并要求组成部分的梯度具有相同的 dtype 和 device，并且结果扁平化张量使用该公共 dtype 和 device。
// 扁平化张量在相应梯度计算时（由 autograd 钩子触发）填充，并且桶按照跨进程保持一致的预定顺序进行减少。
struct Bucket {
  // 桶中梯度扁平化为 1 维张量
  at::Tensor gradients;

  // 对 `gradients` 张量中每个单独梯度的视图
  // 每个视图都是用与梯度预期布局匹配的布局（大小和步幅）创建的
  // `bucket_views_in[i].copy_(grad)` 和 `grad.copy_(bucket_views_out[i])`
  // 分别提供了将梯度数据复制进/出 `gradients` 的便捷方式。
  // 我们保持 `bucket_views_in` 和 `bucket_views_out`，因为注册 DDP 通信钩子可能会使用钩子的 `future_work` 重新初始化 `bucket_views_out`，
  // 但我们仍然需要桶原始扁平化梯度的单独视图，以复制梯度数据。
  std::vector<at::Tensor> bucket_views_in;
  std::vector<at::Tensor> bucket_views_out;

  // 存储在此桶中的梯度的变量
  // 我们在这里使用引用计数张量，以便可以轻松地展开
    // bucket's flattened `gradients` tensor into the participating variables
    // after reduction has completed.
    // 存储每个变量参与的梯度张量，以便在完成减少操作后使用。

    std::vector<at::Tensor> variables;
    // 存储参与的变量列表，每个变量对应一个张量。

    // Per-variable offset/length into the flattened `gradients` tensor and
    // the corresponding `GradBucket` instance for communication hooks
    // 每个变量在扁平化的`gradients`张量中的偏移量/长度，
    // 以及用于通信钩子的对应`GradBucket`实例。

    std::vector<size_t> offsets;
    // 每个变量在扁平化梯度张量中的偏移量。

    std::vector<size_t> lengths;
    // 每个变量在扁平化梯度张量中的长度。

    // Per-variable sizes slicing into the bucket's `gradients` tensor
    // 每个变量在桶的`gradients`张量中的切片大小。
    std::vector<c10::IntArrayRef> sizes_vec;

    // Number of gradients left to be computed before the bucket is ready to
    // be reduced
    // 桶准备好进行减少操作之前剩余的梯度数量。
    size_t pending;

    // Global indices of participating variables in the bucket
    // 桶中参与变量的全局索引。
    std::vector<size_t> variable_indices;

    // Future work handle for DDP communication hook
    // If no hook is registered, a temporary vanilla allreduce hook is used.
    // DDP通信钩子的未来工作处理。
    // 如果没有注册钩子，则使用临时的基本全局归约钩子。
    c10::intrusive_ptr<at::ivalue::Future> future_work;

    // If this bucket should expect a single sparse gradient
    // If `true`, then this implies that `bucket.variables.size() == 1`.
    // 如果此桶应该期望一个稀疏梯度，
    // 如果为`true`，则意味着`bucket.variables.size() == 1`。
    bool expect_sparse_gradient = false;

    // Sparse indices tensor
    // 稀疏索引张量
    std::optional<at::Tensor> sparse_tensor_indices = c10::nullopt;

    // TODO(@pietern)
    // Memory copies from gradient tensors into the bucket are potentially
    // done on different CUDA streams. We record an event for every copy
    // so that we can synchronize with them prior to kicking off the reduction.
    // 梯度张量到桶的内存复制可能在不同的CUDA流上完成。
    // 我们为每次复制记录一个事件，以便在开始减少操作之前与它们同步。
    // std::vector<at::cuda::CUDAEvent> events;
  };

  std::vector<Bucket> buckets_;

  // A variable locator locates a particular variable in the reducer's buckets
  // 变量定位器在减少器的桶中定位特定的变量。

  struct VariableLocator {
    // Index of the bucket containing the variable in the `buckets_` vector
    // `buckets_`向量中包含变量的桶的索引。
    size_t bucket_index;

    // Index of the variable in the bucket, which may be used consistently
    // across `bucket_views_in`, `bucket_views_out`, `variables`, `offsets`,
    // `lengths`, `sizes_vec`, and `variable_indices` in `Bucket`
    // 变量在桶中的索引，可能会一致地在`Bucket`中的`bucket_views_in`、
    // `bucket_views_out`、`variables`、`offsets`、`lengths`、`sizes_vec`和`variable_indices`中使用。
    size_t intra_bucket_index;

    VariableLocator() = default;
  // 定义一个结构体 `VariableLocator`，包含两个成员变量：`bucket_index` 和 `intra_bucket_index`
  VariableLocator(size_t bucket_index_, size_t intra_bucket_index_)
      : bucket_index(bucket_index_),
        intra_bucket_index(intra_bucket_index_) {}
};

// 存储变量位置信息的向量
std::vector<VariableLocator> variable_locators_;

// 记录训练中用于同步梯度的迭代次数
long num_iterations_;

// 记录反向传播调用的不同迭代次数。与 `num_iterations_` 不同，例如在多次前向传播后进行反向传播时
long num_bwd_calls_;

// 标记在一个独立的反向传播过程中是否已经调用了第一个自动求导钩子
bool first_autograd_hook_called_;

// 记录已准备好进行通信调用（如 allReduce 或通信钩子调用）的桶（bucket）数量
int num_buckets_ready_;

// 计时信息
int64_t backward_compute_start_time_ = -1;
std::unique_ptr<Timer> timer_;

// 收集每个梯度准备就绪的相对时间戳，用于执行自动求导时推导桶准备就绪的时间线或理想的桶分配/顺序
std::vector<int64_t> backward_stats_;

// 判断是否应该收集运行时统计信息
bool should_collect_runtime_stats();

// 记录前向计算开始的时间
void record_forward_compute_start_time();

// 记录反向计算开始的时间
void record_backward_compute_start_time();

// 记录反向计算结束的时间
void record_backward_compute_end_time();

// 记录反向通信开始的时间
void record_backward_comm_start_time();

// 记录反向通信结束的时间
void record_backward_comm_end_time();

// 获取分布式数据并行（DDP）运行时日志记录的采样率
int get_ddp_runtime_logging_sample_rate();
int ddp_runtime_logging_sample_rate_ = kDDPRuntimeLoggingSampleRate;

// 标记是否是多设备模块
bool is_multi_device_module_ = false;

// 以下变量用于帮助构建动态桶顺序
// 标记是否已经重建过桶顺序
bool has_rebuilt_bucket_;
// 重建后的参数张量向量
std::vector<at::Tensor> rebuilt_params_;
// 重建后的参数索引向量
std::vector<int64_t> rebuilt_param_indices_;
// 桶的字节容量上限
const int64_t bucket_bytes_cap_;
#ifndef _WIN32
// 定义一个名为 RpcContext 的结构体，仅在非 Windows 系统下有效
struct RpcContext {
    using ContextPtr = torch::distributed::autograd::ContextPtr;
    // 使用 shared_ptr 来持有上下文实例
    ContextPtr context_ptr_holder;
    // 原子指针用于存储上下文指针
    std::atomic<ContextPtr::element_type*> context_ptr{nullptr};

    // 设置新的上下文指针
    void set(ContextPtr&& new_context_ptr);
};
// RpcContext 结构体的实例化对象
RpcContext rpc_context_;
#endif

// 包含在前向传递中安排的所有归约操作的工作句柄和张量的结构体
struct ForwardPassAllreduceWork {
    // 归约操作的工作句柄
    c10::intrusive_ptr<c10d::Work> workHandle;
    // 结果张量
    at::Tensor resultTensor;
    // 是否应该除以初始 world_size 或剩余的 DDP 排名数
};

// 等效于 take_tensors，但返回用于桶分配的张量列表中的索引
// 还考虑设备放置，并且不允许桶跨设备
// 当 tensor_indices 为空时，分配给桶的张量索引为 i
TORCH_API std::tuple<std::vector<std::vector<size_t>>, std::vector<size_t>>
compute_bucket_assignment_by_size(
    const std::vector<at::Tensor>& tensors,
    const std::vector<size_t>& bucket_size,
    const std::vector<bool>& expect_sparse_gradient = {},
    const std::vector<int64_t>& tensor_indices = {},
    const std::optional<std::weak_ptr<c10d::Logger>>& logger = {});

// 验证所有进程中的模型与 rank 0 的模型在参数数量、匹配的 dtype/size/layout 上是否一致
TORCH_API void verify_params_across_processes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<at::Tensor>& params,
    const std::optional<std::weak_ptr<c10d::Logger>>& logger);
} // namespace c10d
```