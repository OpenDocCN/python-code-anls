# `.\pytorch\torch\csrc\distributed\c10d\reducer.cpp`

```py
// 引入 Torch 分布式相关的头文件
#include <torch/csrc/distributed/c10d/reducer.hpp>

// 引入 Torch 分布式工具相关的头文件和默认通信钩子
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/default_comm_hooks.hpp>

// 引入 C++ 标准库头文件
#include <functional>

// 引入 C10 核心库相关头文件，包括设备管理、标量类型、流管理等
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>

// 引入 Torch 自动求导引擎相关的头文件，包括引擎、函数钩子、梯度累积等
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/utils/grad_layout_contract.h>
#include <torch/csrc/autograd/utils/lambda_post_hook.h>

// 引入 Torch 分布式通信相关的头文件，包括通信接口、日志记录等
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>

// 引入 C++ 标准库中的实用工具
#include <utility>

// 进入 c10d 命名空间
namespace c10d {

// 匿名命名空间，用于隐藏实现细节
namespace {

// 定义未设置分割因子的常量
constexpr int kUnsetDivFactor = -1;

// 使用宏封装 TORCH_CHECK 函数，用于带有 DDP 日志的条件检查
#define REDUCER_CHECK(cond, logger_, ...)             \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {               \
    if (!logger_.expired()) {                         \
      logger_.lock()->set_error_and_log(__VA_ARGS__); \
    }                                                 \
    TORCH_CHECK(false, ##__VA_ARGS__);                \
  }

} // namespace

// 定义 TimerRegistry 注册表，用于设备类型和计时器类型的映射
C10_DEFINE_TYPED_REGISTRY(
    TimerRegistry,
    c10::DeviceType,
    Timer,
    std::unique_ptr,
    c10::Device);

// 匿名命名空间，用于隐藏实现细节
namespace {

// 定义 CpuTimer 类，继承自 Timer 接口，用于 CPU 计时
class CpuTimer : public Timer {
 public:
  explicit CpuTimer(c10::Device /* unused */) {}

  // 计算事件开始和结束之间的时间差
  std::optional<int64_t> measureDifference(Event start, Event end) override {
    int64_t start_time = getTimeRef(start);
    int64_t end_time = getTimeRef(end);
    // 如果结束时间早于开始时间，则返回无效值
    if (end_time < start_time) {
      return c10::nullopt;
    }
    return end_time - start_time;
  }
};

// 注册 CpuTimer 类到 TimerRegistry 注册表中，关联到 CPU 设备类型
C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kCPU, CpuTimer);

// 提取给定 IValue 结果中的张量，并以向量形式返回
std::vector<at::Tensor> extractTensors(const c10::IValue& result) {
  // 如果结果是 Python 对象，则从中提取张量
  if (result.isPyObject()) {
    return result.toPyObjectHolder()->extractTensors();
  }
  // 否则，确保结果是张量或张量列表，并返回对应的张量向量
  TORCH_INTERNAL_ASSERT(
      result.isTensor() || result.isTensorList(),
      "expected the hook result is either a Tensor or a TensorList found ",
      result.tagKind());

  if (result.isTensor()) {
    return {result.toTensor()};
  }

  return result.toTensorVector();
}

} // namespace

// 构造函数 Reducer 的定义，初始化各种参数和选项
Reducer::Reducer(
    std::vector<at::Tensor> params,
    std::vector<std::vector<size_t>> bucket_indices,
    const std::vector<size_t>& per_bucket_size_limits,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    std::vector<bool> expect_sparse_gradients,
    int64_t bucket_bytes_cap,
    bool find_unused_parameters,
    ```
    // 初始化 Reducer 对象的构造函数，接受一系列参数并进行初始化
    bool gradient_as_bucket_view,
    // param_names 是一个映射，将参数索引映射到其名称
    std::unordered_map<size_t, std::string> param_names,
    // first_bucket_bytes_cap 指定第一个桶的字节容量限制
    int64_t first_bucket_bytes_cap)
    : params_(std::move(params)),  // 移动构造参数列表
      process_group_(std::move(process_group)),  // 移动构造进程组对象
      expect_sparse_gradients_(std::move(expect_sparse_gradients)),  // 移动构造期望的稀疏梯度设置
      expect_autograd_hooks_(false),  // 不期望自动求导钩子
      require_finalize_(false),  // 不需要最终化
      next_bucket_(0),  // 下一个桶的索引
      has_marked_unused_parameters_(false),  // 尚未标记未使用的参数
      find_unused_parameters_(find_unused_parameters),  // 移动构造查找未使用参数设置
      gradient_as_bucket_view_(gradient_as_bucket_view),  // 梯度作为桶视图
      local_used_map_reduced_(false),  // 本地使用映射未减少
      num_iterations_(0),  // 迭代次数初始化为0
      num_bwd_calls_(0),  // 反向传播调用次数初始化为0
      first_autograd_hook_called_(false),  // 尚未调用首个自动求导钩子
      num_buckets_ready_(0),  // 就绪的桶数量为0
      has_rebuilt_bucket_(false),  // 尚未重建桶
      bucket_bytes_cap_(bucket_bytes_cap),  // 桶的字节容量限制
      div_factor_(kUnsetDivFactor),  // 除法因子未设置
      static_graph_(false),  // 静态图设为假
      comm_hook_(nullptr),  // 通信钩子为空指针
      ddp_debug_level_(debug_level()),  // DDP 调试级别初始化
      param_names_(std::move(param_names)),  // 移动构造参数名称映射
      first_bucket_bytes_cap_(first_bucket_bytes_cap) {  // 第一个桶的字节容量限制初始化
  // 记录 API 使用情况
  C10_LOG_API_USAGE_ONCE("torch.distributed.ddp.reducer");
  // 断言参数列表非空
  TORCH_INTERNAL_ASSERT(!params_.empty(), "Expected at least one parameter.");

  // 如果 DDP 调试级别不为关闭状态，记录初始化信息
  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    LOG(INFO) << "Reducer initialized with bucket_bytes_cap: "
              << bucket_bytes_cap_
              << " first_bucket_bytes_cap: " << first_bucket_bytes_cap;
  }

  // 检查模块是否为多设备模块
  {
    std::set<int> unique_devices;
    for (const auto& v : params_) {
      auto device_idx = int(v.device().index());
      if (unique_devices.find(device_idx) == unique_devices.end()) {
        unique_devices.insert(device_idx);
        if (unique_devices.size() > 1) {
          is_multi_device_module_ = true;
          break;
        }
      }
    }
  }

  // 对于 CUDA，仅为单设备模块记录事件
  c10::Device device = params_[0].device();
  if (!(device.is_cuda() && is_multi_device_module_)) {
    timer_ = TimerRegistry()->Create(device.type(), device);
  }

  // 如果未指定 `expect_sparse_gradients`，初始化为不期望稀疏梯度
  if (expect_sparse_gradients_.empty()) {
    expect_sparse_gradients_ = std::vector<bool>(params_.size(), false);
  }
  TORCH_INTERNAL_ASSERT(expect_sparse_gradients_.size() == params_.size());

  // 初始化变量分桶
  // 可在捕获运行时信息后重新初始化
  {
    std::lock_guard<std::mutex> lock(mutex_);
    initialize_buckets(std::move(bucket_indices));
  }

  // 所有变量预期都已设置其 `grad_fn` 为梯度累积函数（因为它们是自动求导图中的叶子节点）
  // 存储这些函数的指针以便检查是否在自动求导过程中使用
  {
    const auto variable_count = params_.size();
    grad_accumulators_.resize(variable_count);
    // ...
  }
}
    // 对于参数列表中的每一个变量索引，使用范围迭代器进行遍历
    for (const auto variable_index : c10::irange(variable_count)) {
        // 获取当前变量的引用
        auto& variable = params_[variable_index];

        // 梯度累加器函数是惰性初始化的，仅在第一次使用时才会初始化。
        // 因此，我们可以通过其在自动求导图中的存在来判断该参数是否在迭代中有所参与。
        auto grad_accumulator = torch::autograd::impl::grad_accumulator(variable);
#ifndef _WIN32
      // 如果不是在 Windows 平台下编译，使用 torch 的分布式自动求导上下文
      using torch::distributed::autograd::ThreadLocalDistAutogradContext;
#endif

      // 在梯度累积器执行后执行的钩子
      hooks_.emplace_back(
          // 添加一个后处理钩子到梯度累积器
          grad_accumulator->add_post_hook(
              // 使用 LambdaPostHook 包装的 Lambda 函数
              std::make_unique<torch::autograd::utils::LambdaPostHook>(
                  [this, variable_index](
                      const torch::autograd::variable_list& outputs,
                      const torch::autograd::variable_list& /* unused */) {
#ifndef _WIN32
                    // 如果不是在 Windows 平台下，设置 RPC 上下文
                    this->rpc_context_.set(
                        ThreadLocalDistAutogradContext::getContextPtr());
#endif
                    // 调用自动求导钩子处理函数
                    this->autograd_hook(variable_index);
                    return outputs;
                  },
                  // 编译节点参数的 Lambda 函数
                  [=](torch::autograd::CompiledNodeArgs& args) {
                    // 如果启用了编译自动求导，将后处理钩子设置为无操作
                    // Make post_hook an noop if compiled_autograds is enabled.
                  })),
          // 将钩子添加到梯度累积器
          grad_accumulator);

      // 将原始函数指针映射到参数索引
      // 当自动求导图遍历时使用，用于检查是否为某些参数计算了梯度（当 find_unused_parameters=True 时）
      if (find_unused_parameters_) {
        // 将梯度累积器映射到变量索引，用于参数重复前检查
        gradAccToVariableMap_[grad_accumulator.get()] = variable_index;
      }

      // 记录梯度钩子触发次数
      numGradHooksTriggeredMap_[variable_index] = 0;

      // 梯度累积器作为弱引用存储在变量的自动求导元数据中，因此需要在此处保持其有效
      REDUCER_CHECK(
          // 检查是否尝试为变量注册重复的梯度累积器
          grad_accumulators_[variable_index] == nullptr,
          logger_,
          c10::str(
              "Reducer tried to register duplicate grad accumulator for variable ",
              variable_index));

      // 移动梯度累积器到梯度累积器数组中
      grad_accumulators_[variable_index] = std::move(grad_accumulator);
    }
  }

  // 初始化反向传播统计向量
  {
    // 获取参数数量
    const auto variable_count = params_.size();
    // 调整反向传播统计向量大小
    backward_stats_.resize(variable_count);
  }

  // 查看注释 [Skip allreducing local_used_map_dev]
  if (find_unused_parameters_) {
    // 初始化本地使用映射
    initialize_local_used_map();
  }
}

// 注释 [Skip allreducing local_used_map_dev]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// 如果 find_unused_parameters_ 设置为 false，则无需对 local_used_map_dev_ 进行全局归约，
// 因为所有参数都将被归约。因此，如果 find_unused_parameters_ 为 false，我们可以避免为
// local_used_map 和 local_used_map_dev_ 分配内存。

// 注释 [DDP Communication Hook]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// 如果未注册 DDP 通信钩子，Reducer 只需调用 allreduce 来归约桶。
// 如果注册了通信钩子，它会调用该钩子并使用 future work handle。
// 如果注册了通信钩子，Reducer 还会跳过将梯度除以全局大小的步骤。
// 析构函数，用于销毁 Reducer 对象时调用
Reducer::~Reducer() noexcept(false) {
    // 移除自动求导钩子
    remove_autograd_hooks();
}

// 动态图模式下是否查找未使用的参数
bool Reducer::dynamic_graph_find_unused() {
    return !static_graph_ && find_unused_parameters_;
}

// 静态图模式下的第一次反向传播迭代
bool Reducer::static_graph_first_iteration() {
    return static_graph_ && num_bwd_calls_ == 1;
}

// 静态图模式下的非第一次反向传播迭代
bool Reducer::static_graph_after_first_iteration() {
    return static_graph_ && num_bwd_calls_ > 1;
}

// 查询是否使用了 DDP 静态图模式
bool Reducer::ddp_graph_static() {
    // 使用互斥锁保护共享变量 ddp_graph_static_
    std::lock_guard<std::mutex> lock(mutex_);
    return ddp_graph_static_;
}

// 初始化本地使用映射
void Reducer::initialize_local_used_map() {
    const auto variable_count = params_.size();
    at::TensorOptions options;
    options = options.dtype(at::kInt);

    // 创建一个全零的 Tensor，表示本地使用映射
    // 注意：即使 local_used_map_dev_ 可能在 CUDA 上，也不会钉住内存。
    // 参见 Note [local_used_map_ -> local_used_map_dev copying]
    local_used_map_ = at::zeros({static_cast<long>(variable_count)}, options);

    // 将 local_used_map_dev_ 放置在与副本参数相同的设备上
    // 某些后端（如 NCCL）可能不支持 CPU Tensor，因此应与参数位于相同设备上
    // 对于 MTIA 的分布后端，不支持 int32 全局归约，因此必须放置在 CPU 上
    options = options.device(
        (params_[0].is_mtia()) ? c10::Device(c10::DeviceType::CPU)
                               : params_[0].device());
    local_used_map_dev_ = at::empty({static_cast<long>(variable_count)}, options);
}

// 检查梯度的布局
void Reducer::check_grad_layout(
    const at::Tensor& grad,
    // 确保梯度类型与桶视图类型匹配，或者在使用混合精度训练时，匹配混合精度类型。
    auto type = mixed_precision_param_dtype_
        ? *mixed_precision_param_dtype_
        : bucket_view.options().dtype().toScalarType();
    
    // 检查梯度张量的数据类型是否与预期的类型相匹配，如果不匹配则输出警告信息。
    REDUCER_CHECK(
        grad.options().dtype().toScalarType() == type,
        logger_,
        c10::str(
            "Expected ", type, ", got ", grad.options().dtype().toScalarType()));
    
    // 断言梯度张量和桶视图张量在同一个设备上。
    TORCH_INTERNAL_ASSERT(grad.device() == bucket_view.device());
    
    // 断言梯度张量和桶视图张量的元素数量相等。
    TORCH_INTERNAL_ASSERT(grad.numel() == bucket_view.numel());
    
    // AccumulateGrad 操作不一定遵循梯度布局约定。
    // 违反约定会降低性能，不会导致数值错误。这里的警告有助于诊断DDP性能不佳的问题。
    if (grad.strides() != bucket_view.strides()) {
      TORCH_WARN_ONCE(
          "Grad strides do not match bucket view strides. "
          "This may indicate grad was not created according to the "
          "gradient layout contract, or that the param's strides "
          "changed since DDP was constructed.  This is not an error, "
          "but may impair performance.\n"
          "grad.sizes() = ",
          grad.sizes(),
          ", strides() = ",
          grad.strides(),
          "\n",
          "bucket_view.sizes() = ",
          bucket_view.sizes(),
          ", strides() = ",
          bucket_view.strides());
    }
    
    // 如果不允许梯度作为桶视图，断言梯度张量不是桶视图张量的别名。
    if (!gradient_as_bucket_view_) {
      TORCH_INTERNAL_ASSERT(!grad.is_alias_of(bucket_view));
    }
// 标记指定变量已准备好的密集操作函数
void Reducer::mark_variable_ready_dense(size_t variable_index) {
  // 获取变量在变量定位器中的索引信息
  const auto& bucket_index = variable_locators_[variable_index];
  // 根据变量在桶索引中的索引获取相应的桶
  auto& bucket = buckets_[bucket_index.bucket_index];
  // 获取桶中指定索引的变量
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];
  // 获取桶视图中指定索引的视图
  auto& bucket_view = bucket.bucket_views_in[bucket_index.intra_bucket_index];

  // 将梯度张量的内容复制到桶的扁平化梯度张量的对应部分
  // 如果梯度未设置，则假定它未在当前反向传播过程中计算，并将其应该持有的桶部分清零
  runGradCallbackForVariable(variable, [&](auto& grad) {
    // 检查梯度 grad 是否已定义
    if (grad.defined()) {
      // 检查梯度 grad 的布局是否符合要求
      this->check_grad_layout(grad, bucket_view);
      // 当 gradient_as_bucket_view_ 为 false，或者即使为 true，用户在每次迭代后可能将 grad 设置为 None。
      // 在这些情况下，grad 和 bucket_view 指向不同的存储，因此需要将 grad 复制到 bucket_view。
      // 如果 gradient_as_bucket_view_ 设置为 true，则让 grad 指向 bucket_view。
      // 如果 grad 在之前的迭代中已经设置为 buckets 的视图，则无需复制。
      if (!grad.is_alias_of(bucket_view)) {
        // 如果没有设置通信钩子
        if (comm_hook_ == nullptr) {
          // 创建一个标量张量，用于除法操作
          auto wrapped =
              at::native::wrapped_scalar_tensor(double(1.) / div_factor_);
          // 如果 grad 不需要梯度
          if (!grad.requires_grad()) {
            // 记录函数操作，用于反向传播追踪
            RECORD_FUNCTION(
                "torch::distributed::reducer::mul_out",
                std::vector<c10::IValue>({bucket_view}))
            // 将 grad 乘以 wrapped，并将结果写入 bucket_view
            at::mul_out(bucket_view, grad, wrapped);
          } else {
            // 如果 DDP 运行时 create_graph=True，梯度本身需要梯度以计算高阶导数。
            // 然而，DDP 目前不会同步这些梯度（参见链接）
            C10_LOG_EVERY_N(WARNING, 1000)
                << "Using DistributedDataParallel with create_graph=True "
                << " is not well-supported. The higher-order gradient will "
                << " not be synchronized across ranks, and backpropagation "
                << " through all_reduce operations will not occur. If you require "
                << " DDP to work with higher-order gradients for your use case, "
                << " please ping https://github.com/pytorch/pytorch/issues/63929";
            // 将 grad 乘以 wrapped，并将结果保存在 div_result 中
            auto div_result = at::mul(grad, wrapped);
            // 记录函数操作，用于反向传播追踪
            RECORD_FUNCTION(
                "torch::distributed::reducer::copy_",
                std::vector<c10::IValue>({bucket_view}))
            // 将 div_result 的值复制到 bucket_view
            bucket_view.copy_(div_result);
          }
        } else {
          // 记录函数操作，用于反向传播追踪
          RECORD_FUNCTION(
              "torch::distributed::reducer::copy_",
              std::vector<c10::IValue>({bucket_view}))
          // 将 grad 的值复制到 bucket_view
          bucket_view.copy_(grad);
        }

        // 如果设置了 gradient_as_bucket_view_
        if (gradient_as_bucket_view_) {
          // 让 grad 指向 bucket_view 的缓冲区
          grad = bucket_view;
          // grad 已修改，需要写回
          return true;
        }
      } else {
        // 如果 grad 和 bucket_view 指向相同的存储，无需复制
        if (comm_hook_ == nullptr) {
          // 在不设置通信钩子的情况下，将 bucket_view 中的值除以 div_factor_
          bucket_view.div_(div_factor_);
        }
      }
    } else {
      // 如果梯度未定义。当 find_unused_parameters=True 时，确保它没有被标记为局部使用，
      // 否则我们会对 .grad 参数进行全局归约，而不是不触及参数的 .grad 字段。
      if (this->dynamic_graph_find_unused() ||
          this->static_graph_first_iteration()) {
        // 检查本地使用映射中的变量索引是否为零，如果不为零，则报错。
        REDUCER_CHECK(
            local_used_map_[variable_index].item<int>() == 0,
            logger_,
            "Encountered gradient which is undefined, but still allreduced by "
            "DDP reducer. This indicates a bug in DDP implementation, please "
            "report a bug with a repro to PyTorch.");
      }
      // 将 bucket_view 清零。
      bucket_view.zero_();
    }
    // 梯度未被修改，不需要写回。
    return false;
  });
}

void Reducer::mark_variable_ready_sparse(size_t variable_index) {
  // 根据变量索引获取其在变量定位器中的索引
  const auto& bucket_index = variable_locators_[variable_index];
  // 根据桶索引获取对应的桶
  auto& bucket = buckets_[bucket_index.bucket_index];
  // 根据变量在桶内的索引获取变量本身
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];

  // 为变量调用梯度回调函数，并处理其梯度
  runGradCallbackForVariable(variable, [&](auto& grad) {
    // 检查稀疏梯度是否已定义
    REDUCER_CHECK(
        grad.defined(), logger_, "Expected sparse gradient to be defined.");
    // 检查梯度选项是否为稀疏布局
    REDUCER_CHECK(
        grad.options().layout() == c10::kSparse,
        logger_,
        "Expected variable to have sparse gradient.");

    // 复制稀疏元数据的索引
    if (sparse_metadata_) {
      grad = grad.coalesce();  // 合并稀疏梯度
      REDUCER_CHECK(
          !param_names_.empty(), logger_, "No parameter names were found");
      std::string& param_name = param_names_[variable_index];
      auto iter = sparse_metadata_->find(param_name);
      REDUCER_CHECK(
          iter != sparse_metadata_->end(),
          logger_,
          "param: " + param_name + " not found in sparse metadata");
      // 将稀疏张量索引存储到桶结构体中
      bucket.sparse_tensor_indices =
          iter->second.to(at::kLong).unsqueeze(0).to(grad.device());
      auto indices = at::searchsorted(
          bucket.sparse_tensor_indices.value(), grad.indices(), false, false);
      // 使用稀疏元数据中的索引创建稀疏 COO 张量
      grad = at::sparse_coo_tensor(indices, grad.values(), grad.sizes());
    }

    // 稀疏张量不能像密集张量那样在单个减少操作中分组在一起。
    // 因此，桶结构体中的 `offsets` 和 `lengths` 向量为空，并且没有预先存在的累积张量。
    // 直接将稀疏张量分配给 `gradients` 字段。
    bucket.gradients = grad;
    // 如果未注册 DDP 通信钩子，则 allreduce 只对值求和，需要额外进行除法操作。
    if (comm_hook_ == nullptr) {
      bucket.gradients.div_(div_factor_);
    }
    // 梯度被原地修改，需要将其写回。
    return true;
  });
}

std::vector<c10d::GradBucket> Reducer::get_grad_buckets(
    bool return_zero_tensors) const {
  std::lock_guard<std::mutex> lock(mutex_);
  // 获取梯度桶的列表，保留足够的空间
  std::vector<c10d::GradBucket> gradBuckets;
  gradBuckets.reserve(buckets_.size());
  // 遍历所有桶，为每个桶创建 GradBucket 对象并加入到 gradBuckets 中
  for (const auto i : c10::irange(buckets_.size())) {
    auto& bucket = buckets_[i];
    auto variables_for_bucket = get_variables_for_bucket(i, bucket);
    gradBuckets.emplace_back(
        i,
        buckets_.size(),
        return_zero_tensors ? at::zeros_like(bucket.gradients)
                            : bucket.gradients,
        bucket.offsets,
        bucket.lengths,
        bucket.sizes_vec,
        variables_for_bucket,
        c10::nullopt);
  }
  return gradBuckets;  // 返回所有梯度桶
}

void Reducer::set_forward_pass_work_handle(
    c10::intrusive_ptr<c10d::Work> forwardPassWorkHandle,
    // 使用 std::lock_guard 对互斥量 mutex_ 进行加锁，确保线程安全操作
    std::lock_guard<std::mutex> lock(mutex_);
    // 将 forwardPassWorkHandle 参数的值移动到 forwardPassWorkHandle_.workHandle 中
    forwardPassWorkHandle_.workHandle = std::move(forwardPassWorkHandle);
    // 将 useStaticWorldSize 参数的值赋给 forwardPassWorkHandle_.useStaticWorldSize
    forwardPassWorkHandle_.useStaticWorldSize = useStaticWorldSize;
}

// 返回在设备上的局部使用映射，使用互斥锁确保线程安全
at::Tensor Reducer::get_local_used_map_on_device() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_used_map_dev_;
}

// 将重建参数推送到所有索引，使用互斥锁确保线程安全
void Reducer::push_rebuilt_params_for_all_indices() {
  std::lock_guard<std::mutex> lock(mutex_);
  // 如果不需要重建桶或者已经存在重建的参数索引，直接返回
  if (!should_rebuild_buckets() || !rebuilt_param_indices_.empty()) {
    return;
  }
  const auto variable_count = params_.size();
  // 遍历参数列表，对每个索引调用push_rebuilt_params方法
  for (const auto variable_index : c10::irange(variable_count)) {
    push_rebuilt_params(variable_index);
  }
}

// 将指定索引处的参数添加到重建参数列表中
void Reducer::push_rebuilt_params(const size_t& index) {
  rebuilt_params_.push_back(params_[index]);
  rebuilt_param_indices_.push_back(static_cast<int64_t>(index));
}

// 设置分割因子，根据条件等待前向传播中的全局约减
void Reducer::set_divide_factor() {
  // 如果分割因子未设置
  if (div_factor_ == kUnsetDivFactor) {
    // 获取当前参与进程数作为初始的分割因子
    div_factor_ = process_group_->getSize();
    auto& workHandle = forwardPassWorkHandle_.workHandle;
    if (workHandle && !forwardPassWorkHandle_.useStaticWorldSize) {
      workHandle->wait();
      // 从future中提取结果
      auto results = extractTensors(workHandle->getFuture()->value());

      // 断言结果不为空
      TORCH_INTERNAL_ASSERT(!results.empty());
      at::Tensor& res = results.front();
      // 更新分割因子为结果中的整数值
      div_factor_ = res.item().to<int>();
    }
  }
}

// 在训练之前将梯度转换为指定的混合精度数据类型
void Reducer::set_mixed_precision_param_dtype(c10::ScalarType dtype) {
  mixed_precision_param_dtype_ = dtype;
  // 遍历桶列表，将每个桶中的梯度转换为指定的数据类型
  for (auto& bucket : buckets_) {
    bucket.gradients = bucket.gradients.to(dtype);
  }
}

// 延迟所有约减操作，仅在static_graph_=true且num_iterations_==1时调用
void Reducer::delay_all_reduce() {
  std::lock_guard<std::mutex> lock(this->mutex_);

  // 如果需要收集运行时统计信息，记录后向计算结束时间和后向通信开始时间
  if (should_collect_runtime_stats()) {
    record_backward_compute_end_time();
    record_backward_comm_start_time();
  }

  // 启动所有约减操作
  all_reduce_local_used_map();

  // 准备设置未使用的参数集合，如果是静态图，第一次迭代后unused_parameters_将不再改变
  unused_parameters_.clear();

  require_finalize_ = true;
  // 复制所有梯度到桶中
  for (const auto variable_index : c10::irange(params_.size())) {
    // 设置未使用的参数集合
    if (numGradHooksTriggeredMap_[variable_index] == 0) {
      unused_parameters_.push_back(variable_index);
    }
    set_divide_factor();
    // 如果期望稀疏梯度，则标记变量为稀疏梯度已准备好
    if (expect_sparse_gradients_[variable_index]) {
      mark_variable_ready_sparse(variable_index);
    } else {
      mark_variable_ready_dense(variable_index);
  }
}

// 避免静态图在不同排名上将某些参数标记为未使用时造成混淆，
// 我们记录每个排名上未使用的参数名称，以提升调试能力，
// 当 TORCH_DISTRIBUTED_DEBUG 设置为 INFO 或 DETAIL 时。
if (ddp_debug_level_ != c10d::DebugLevel::Off) {
  // 构建一个用于输出的字符串流
  std::ostringstream unused_params_stream;

  // 遍历未使用参数列表
  for (const auto& unused_index : unused_parameters_) {
    // 查找未使用参数的名称
    auto param_name = param_names_.find(unused_index);
    // 在调试模式中，应该能找到未使用参数的名称，否则报错
    TORCH_INTERNAL_ASSERT(
        param_name != param_names_.end(),
        "Expected to find parameter name from unused parameters map in debug mode.");
    // 添加参数名称和索引到输出流中
    unused_params_stream << "{" << param_name->second << "," << unused_index
                         << "}";
  }

  // 如果存在未使用的参数，则每个排名打印出所有检测到的未使用参数
  if (!unused_parameters_.empty()) {
    LOG(INFO) << "[Rank " << process_group_->getRank() << "]: "
              << "Parameter(s) (in the format of {param_name, index}): "
              << unused_params_stream.str()
              << " is(are) unused during first iteration. Since"
              << " static_graph=True is enabled for DDP, we expect"
              << " this set of unused parameters to remain consistent"
              << " on this rank throughout the training.";
  }
}

// 对所有桶启动全部的归约操作
for (auto& bucket : buckets_) {
  all_reduce_bucket(bucket);
}

// 完成反向传播的后续操作
finalize_backward();
}

// 设置日志记录器，使用给定的日志器
void Reducer::set_logger(std::weak_ptr<c10d::Logger> logger) {
  logger_ = std::move(logger);
}

// 在模型参数的梯度被累积到其梯度张量后调用 `autograd_hook` 函数。
// 该函数只能从自动求导线程调用。
void Reducer::autograd_hook(size_t index) {
  // 使用互斥锁保护共享资源
  std::lock_guard<std::mutex> lock(this->mutex_);
  
  // 如果尚未调用过第一个 autograd hook，则标记为已调用，并增加计数器
  if (!first_autograd_hook_called_) {
    first_autograd_hook_called_ = true;
    num_bwd_calls_++;
  }

  // 查看注释 [Skip allreducing local_used_map_dev]
  // 如果动态图中发现未使用的变量或者是静态图的第一次迭代
  if (dynamic_graph_find_unused() || static_graph_first_iteration()) {
    // 如果进入此分支，则表示该参数在本次迭代中已被使用。
    // 我们希望在 local_used_map_ 中标记它。在 no_sync 会话期间，同一个变量可以被多次设置，
    // 这是允许的，只要在 no_sync 会话期间至少使用一次即可标记为已使用。
    // 只有在梯度被定义时才将其标记为局部使用。否则，钩子可能会在梯度未定义的情况下触发，
    // 例如在计算损失时没有使用所有输出时。在这种情况下，我们不希望标记它为局部使用，以确保不会触及参数的 .grad 字段。
    auto& variable = get_param_from_index(index);
    runGradCallbackForVariable(variable, [&](auto& grad) {
      if (grad.defined()) {
        local_used_map_[static_cast<int64_t>(index)] = 1;
      }
      // 梯度永远不会被修改。
      return false;
    });
  }

  // 如果是静态图的第一次迭代，增加该参数对应的梯度钩子触发计数并返回
  if (static_graph_first_iteration()) {
    numGradHooksTriggeredMap_[index] += 1;
    return;
  }

  // 如果不期望调用 autograd hook，则直接返回
  // 这可能发生在用户希望在减少梯度之前累积梯度的情况下。
  if (!expect_autograd_hooks_) {
    return;
  }

  // 将参数索引添加到梯度准备就绪的顺序索引中
  grad_ready_order_indices_.push_back(static_cast<int64_t>(index));

  // 如果 `find_unused_parameters_` 为真，则可能有模型参数在计算模型输出时未使用，
  // 它们不会成为自动求导图的一部分，也不会接收梯度。这些参数在 `prepare_for_backward` 函数中被发现，
  // 并将它们的索引存储在 `unused_parameters_` 向量中。
  if (!has_marked_unused_parameters_) {
    has_marked_unused_parameters_ = true;
    for (const auto& unused_index : unused_parameters_) {
      mark_variable_ready(unused_index);
  // 重新构建桶(bucket)，只有在以下情况下才需要重新构建：
  // 1) 第一次重新构建桶
  // 2) static_graph_ 为 true 或者 find_unused_parameters_ 为 false
  // 3) 这个反向传播需要运行全局归约(allreduce)
  // 在这里，我们按照梯度到达的顺序，将张量及其参数索引倒入 rebuilt_params_ 和 rebuilt_param_indices_，
  // 在 finalize_backward() 结束时，桶将基于 rebuilt_params_ 和 rebuilt_param_indices_ 重新构建，
  // 然后进行广播和初始化。
  // 如果是静态图(static graph)，在第一次迭代后，根据 numGradHooksTriggeredMap_ 检查变量是否准备好进行通信。
  if (static_graph_after_first_iteration()) {
    REDUCER_CHECK(
        numGradHooksTriggeredMapPerIteration_[index] > 0,
        logger_,
        "Your training graph has changed in this iteration, ",
        "e.g., one parameter is unused in first iteration, but ",
        "then got used in the second iteration. this is not ",
        "compatible with static_graph set to True.");
    // 如果 numGradHooksTriggeredMapPerIteration_[index] 减至 0，表明变量已准备好通信
    if (--numGradHooksTriggeredMapPerIteration_[index] == 0) {
      // 如果需要重新构建桶，推送重建后的参数
      if (should_rebuild_buckets()) {
        push_rebuilt_params(index);
      }
      // 最后标记原始调用此函数的变量为准备就绪
      mark_variable_ready(index);
    }
  } else {
    // 如果需要重新构建桶，推送重建后的参数
    if (should_rebuild_buckets()) {
      push_rebuilt_params(index);
    }
    // 最后标记原始调用此函数的变量为准备就绪
    mark_variable_ready(index);
  }
// See Note [Skip allreducing local_used_map_dev]
// H2D from local_used_map_ to local_used_map_dev_
if (local_used_map_dev_.is_cuda() || local_used_map_dev_.is_privateuseone()) {
    // Note [local_used_map_ -> local_used_map_dev copying]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // We do async H2D to avoid the blocking overhead. The async copy and
    // allreduce respect the current stream, so will be sequenced
    // correctly.
    //
    // Correct sequencing with respect to host operations is also
    // essential. The H2D copy_ is stream ordered, while the host's
    // changes to local_used_map_ are host ordered. If a large backlog of
    // cuda/privateuseone-stream work pushes the copy_ far into the future, and
    // if no blocking calls occur between now and finalize_backward()** such
    // that finalize_backward() re-zeroes local_used_map_ on the host
    // before the stream executes the copy_, copy_ will read those zeros
    // instead of the values we thought we told it to read here. Copying
    // local_used_map_ to a pinned temporary (which the pinned caching
    // allocator should supply asynchronously) avoids this nasty, rare
    // race condition.
    //
    // ** In the hoped-for case where all params are used, DDP itself
    // won't do any blocking work between now and the re-zeroing, so the
    // danger is real.
    //
    // Defensively ensures local_used_map_tmp is distinct from
    // local_used_map_
    // 创建一个和 local_used_map_ 类型、布局、设备相同的空张量 local_used_map_tmp
    auto local_used_map_tmp = at::native::empty_like(
        local_used_map_,
        c10::optTypeMetaToScalarType(local_used_map_.options().dtype_opt()),
        local_used_map_.options().layout_opt(),
        local_used_map_.options().device_opt(),
        true /* pinned_memory */);
    // 用断言确保 local_used_map_tmp 是固定在内存中的
    TORCH_INTERNAL_ASSERT(local_used_map_tmp.is_pinned());
    // 用断言确保 local_used_map_tmp 的数据指针与 local_used_map_ 的数据指针不同
    TORCH_INTERNAL_ASSERT(
        local_used_map_tmp.data_ptr() != local_used_map_.data_ptr());
    // 将 local_used_map_ 的数据复制到 local_used_map_tmp 中
    local_used_map_tmp.copy_(local_used_map_);
    // 将 local_used_map_tmp 的数据异步复制到 local_used_map_dev_ 中
    local_used_map_dev_.copy_(local_used_map_tmp, true);
} else if (local_used_map_dev_.is_mtia()) {
    // MTIA 可能会在未来有特殊的逻辑，因此这里创建一个新的 if 分支处理 MTIA
    // 目前的实现与 CUDA/privateuseone 的类似，但不包括固定内存步骤。
    // 创建一个和 local_used_map_ 类型、布局、设备相同的空张量 local_used_map_tmp
    auto local_used_map_tmp = at::native::empty_like(
        local_used_map_,
        c10::optTypeMetaToScalarType(local_used_map_.options().dtype_opt()),
        local_used_map_.options().layout_opt(),
        local_used_map_.options().device_opt());
    // 将 local_used_map_ 的数据复制到 local_used_map_tmp 中
    local_used_map_tmp.copy_(local_used_map_);
    // 将 local_used_map_tmp 的数据复制到 local_used_map_dev_ 中
    local_used_map_dev_.copy_(local_used_map_tmp, true);
} else {
    # 将 local_used_map_ 的内容复制到 local_used_map_dev_ 中，并确保复制操作是原位的
    local_used_map_dev_.copy_(local_used_map_, true);
  }
  # 创建包含 local_used_map_dev_ 的单元素向量 temp_local_used_map_dev_vec_
  std::vector<at::Tensor> temp_local_used_map_dev_vec_ = {local_used_map_dev_};
  # 使用 process_group_ 执行全局归约操作，将 temp_local_used_map_dev_vec_ 中的数据进行归约，结果保存在 local_used_work_ 中
  local_used_work_ = process_group_->allreduce(temp_local_used_map_dev_vec_);
}

# 结束了 Reducer 类中的一个成员函数定义。

at::Tensor& Reducer::get_param_from_index(size_t index) {
  # 获取给定索引对应的变量位置信息
  const auto& bucket_index = variable_locators_[index];
  # 获取对应的桶对象
  auto& bucket = buckets_[bucket_index.bucket_index];
  # 由于 `runGradCallbackForVariable()` 不接受 const 张量，不能简单地通过 `bucket.variables[variable_index]` 直接访问变量。
  # 获取桶中指定位置的变量引用
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];
  # 返回变量引用
  return variable;
}

# Reducer 类中的一个成员函数，用于检查并报告指定索引的变量是否被标记两次。

void Reducer::checkAndRaiseMarkedTwiceError(size_t index) {
  # 检查对应索引的变量是否已经被标记两次
  bool marked_twice =
      perIterationReadyParams_.find(index) != perIterationReadyParams_.end();

  # 如果变量被标记两次
  if (marked_twice) {
    # 报告被标记两次的参数索引。在调试模式下，还报告完全限定的参数名。
    auto param_name = param_names_.find(index);
    const bool found_param_name = param_name != param_names_.end();
    # 在调试模式下，必须能够找到参数名
    TORCH_INTERNAL_ASSERT(
        ddp_debug_level_ == c10d::DebugLevel::Off || found_param_name,
        "Expected to find parameter name in debug mode.");

    # 构建参数信息字符串
    std::string paramInfo = c10::str(
        "Parameter at index ",
        index,
        found_param_name ? c10::str(" with name ", param_name->second) : "",
        " has been marked as ready twice. This means that multiple autograd engine ",
        " hooks have fired for this particular parameter during this iteration.");

    # 如果在调试模式下未找到参数名，补充额外信息
    if (!found_param_name) {
      paramInfo += c10::str(
          " You can set the environment variable TORCH_DISTRIBUTED_DEBUG to either",
          " INFO or DETAIL to print parameter names for further debugging.");
    }

    # 构建常见错误信息
    std::string common_error = c10::str(
        "Expected to mark a variable ready only once. ",
        "",
        "This error is caused by one of the following reasons: ",
        "1) Use of a module parameter outside the `forward` function. ",
        "Please make sure model parameters are not shared across multiple ",
        "concurrent forward-backward passes. or try to use _set_static_graph() ",
        "as a workaround if this module graph does not change ",
        "during training loop.",
        "2) Reused parameters in multiple reentrant backward passes. For ",
        "example, if you use multiple `checkpoint` functions to wrap the ",
        "same part of your model, it would result in the same set of ",
        "parameters been used by different reentrant backward passes ",
        "multiple times, and hence marking a variable ready multiple times. ",
        "DDP does not support such use cases in default. You can try to ",
        "use _set_static_graph() as a workaround if your module graph ",
        "does not change over iterations.");

    # 将参数信息添加到常见错误信息中
    common_error += c10::str("\n", paramInfo);
    REDUCER_CHECK(
        has_marked_unused_parameters_,  # 检查是否标记了未使用的参数
        logger_,  # 使用的日志记录器对象
        common_error,  # 公共错误处理对象
        "3) Incorrect unused parameter detection. The return value of the ",  # 错误消息的一部分，指示未使用参数检测的错误
        "`forward` function is inspected by the distributed data parallel ",  # 错误消息的一部分，说明分布式数据并行包装器如何检查`forward`函数的返回值
        "wrapper to figure out if any of the module's parameters went ",  # 错误消息的一部分，说明如何检测模块参数是否未使用
        "unused. For unused parameters, DDP would not expect gradients from ",  # 错误消息的一部分，指出对于未使用的参数，分布式数据并行不会期望梯度
        "then. However, if an unused parameter becomes part of the autograd ",  # 错误消息的一部分，描述如果未使用的参数后来成为自动求导图的一部分
        "graph at a later point in time (e.g., in a reentrant backward when ",  # 错误消息的一部分，示例说明在重新进入的反向传播时未使用的参数如何成为自动求导图的一部分
        "using `checkpoint`), the gradient will show up unexpectedly. If all ",  # 错误消息的一部分，说明在使用`checkpoint`时未使用参数可能导致梯度意外出现
        "parameters in the model participate in the backward pass, you can ",  # 错误消息的一部分，建议如果所有模型参数都参与反向传播，可以禁用未使用参数检测
        "disable unused parameter detection by passing the keyword argument ",  # 错误消息的一部分，建议如何通过传递关键字参数禁用未使用参数检测
        "`find_unused_parameters=False` to ",  # 错误消息的一部分，具体指明传递参数`find_unused_parameters=False`来禁用未使用参数检测
        "`torch.nn.parallel.DistributedDataParallel`. If unused parameters ",  # 错误消息的一部分，说明在`torch.nn.parallel.DistributedDataParallel`中如何禁用未使用参数检测
        "in the model do not change over iterations, You can try to use ",  # 错误消息的一部分，建议如果模型中的未使用参数在迭代过程中不发生变化，可以尝试使用某种解决方法
        "_set_static_graph() as a workaround if this module graph does not ",  # 错误消息的一部分，建议作为一种解决方法，在模块图不会在训练循环中发生变化时使用静态图
        "change during training loop.");  # 错误消息的最后一部分，描述模块图在训练循环中不会发生变化

    REDUCER_CHECK(!has_marked_unused_parameters_, logger_, common_error);  # 再次检查未标记未使用参数的情况，并记录错误
  }
}

// 将变量标记为准备就绪，以便进行后向计算
void Reducer::mark_variable_ready(size_t variable_index) {
  // 检查变量索引是否在范围内
  REDUCER_CHECK(
      variable_index < variable_locators_.size(),
      logger_,
      "Out of range variable index.");

  // 检查并引发重复标记错误
  checkAndRaiseMarkedTwiceError(variable_index);

  // 将变量索引插入到每次迭代准备好的参数集合中
  perIterationReadyParams_.insert(variable_index);

  // 记录后向计算开始时间与当前时间差作为后向统计信息
  backward_stats_[variable_index] =
      current_time_in_nanos() - backward_compute_start_time_;

  // 每当标记变量为准备就绪时，需要调用 finalize 函数。
  // 如果在下一次迭代或准备后向计算之前没有调用 finalize，说明出现问题。
  require_finalize_ = true;

  // 获取变量所在的桶索引
  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];

  // 设置划分因子
  set_divide_factor();

  // 根据桶的期望稀疏梯度属性选择标记变量为准备就绪的方法
  if (bucket.expect_sparse_gradient) {
    mark_variable_ready_sparse(variable_index);
  } else {
    mark_variable_ready_dense(variable_index);
  }

  // 检查是否这是该桶的最后一个梯度
  if (--bucket.pending == 0) {
    mark_bucket_ready(bucket_index.bucket_index);
  }

  // 当最后一个桶标记为准备就绪后，运行 finalizer 函数并开始本地使用映射的减少
  if (next_bucket_ == buckets_.size()) {
    // 如果在第一次迭代后静态图表明需要重建桶，则推送重建参数
    if (dynamic_graph_find_unused()) {
      all_reduce_local_used_map();
    }

    // 使用 Torch 的默认引擎队列回调函数，记录后向计算结束时间等
    torch::autograd::Engine::get_default_engine().queue_callback([this] {
      std::lock_guard<std::mutex> lock(this->mutex_);
      // 如果应收集运行时统计信息，则记录后向计算结束时间
      if (should_collect_runtime_stats()) {
        record_backward_compute_end_time();
      }
      // 检查所有桶是否完成并已启动其工作
      TORCH_INTERNAL_ASSERT(next_bucket_ == buckets_.size());
      // 如果静态图在第一次迭代后需要重建桶，则推送重建参数
      if (static_graph_after_first_iteration() && should_rebuild_buckets()) {
        for (const auto& unused_index : unused_parameters_) {
          push_rebuilt_params(unused_index);
        }
      }
      // 最终化后向计算
      this->finalize_backward();
    });
  }
}

// 运行通信钩子函数，返回异步值
c10::intrusive_ptr<c10::ivalue::Future> Reducer::run_comm_hook(
    GradBucket& grad_bucket) {
  // 如果通信钩子为空，则运行全局归约钩子
  if (comm_hook_ == nullptr) {
    return run_allreduce_hook(grad_bucket);
  } else {
    // 否则运行指定的通信钩子
    return comm_hook_->runHook(grad_bucket);
  }
}

// 运行全局归约钩子函数，返回异步值
c10::intrusive_ptr<c10::ivalue::Future> Reducer::run_allreduce_hook(
    GradBucket& grad_bucket) {
  // 创建全局归约钩子对象并运行钩子函数
  _AllReduceBySumCommHook allreduce_hook(process_group_);
  return allreduce_hook.runHook(grad_bucket);
}
// 当前方法用于对指定的 bucket 进行全局归约操作。
void Reducer::all_reduce_bucket(Bucket& bucket) {
  // 获取与当前 bucket 相关的变量列表
  auto variables_for_bucket = get_variables_for_bucket(next_bucket_, bucket);

  // 提取当前 bucket 中的梯度张量
  const auto& tensor = bucket.gradients;

  // 如果梯度张量在 MTIA 上，需要特殊处理
  // MTIA 是一种特殊的内存布局，可能导致 bucket.bucket_views_in 不指向与 bucket.gradients 相同的存储空间，
  // 因此需要显式地将数据复制回 1-D 梯度张量。
  if (tensor.is_mtia()) {
    for (const auto i : c10::irange(bucket.variables.size())) {
      const auto offset = bucket.offsets[i];
      const auto length = bucket.lengths[i];
      if (!bucket.bucket_views_in[i].is_alias_of(tensor)) {
        tensor
            .narrow(
                0, static_cast<int64_t>(offset), static_cast<int64_t>(length))
            .copy_(bucket.bucket_views_in[i].flatten());
      }
    }
  }

  // 创建 GradBucket 对象，用于存储归约操作相关的信息
  GradBucket grad_bucket(
      next_bucket_,
      buckets_.size(),
      tensor,
      bucket.offsets,
      bucket.lengths,
      bucket.sizes_vec,
      variables_for_bucket,
      bucket.sparse_tensor_indices);

  // 运行通信钩子，并将未来的工作记录到 bucket.future_work
  bucket.future_work = run_comm_hook(grad_bucket);
}

// 获取与指定 bucket 相关的变量列表
std::vector<at::Tensor> Reducer::get_variables_for_bucket(
    size_t bucket_index,
    const Bucket& bucket) const {
  // 如果之前已经重建过 bucket 并且缓存中存在对应的映射，则直接返回缓存的变量列表
  if (has_rebuilt_bucket_ &&
      cached_variables_for_bucket_.find(bucket_index) !=
          cached_variables_for_bucket_.end()) {
    return cached_variables_for_bucket_[bucket_index];
  }

  // 否则，根据 bucket 中的变量索引获取实际的模型参数，并构建变量列表
  std::vector<at::Tensor> variables_for_bucket;
  variables_for_bucket.reserve(bucket.variable_indices.size());
  for (const auto& variable_index : bucket.variable_indices) {
    // 使用 variable_locators_ 获取梯度所在的 bucket 索引
    auto& bucket_index_for_variable = variable_locators_[variable_index];
    // 获取实际的模型参数
    auto& variable =
        bucket.variables[bucket_index_for_variable.intra_bucket_index];
    variables_for_bucket.emplace_back(variable);
  }

  // 如果已经重建过 bucket，则将构建的变量列表缓存起来
  if (has_rebuilt_bucket_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        cached_variables_for_bucket_.find(bucket_index) ==
        cached_variables_for_bucket_.end());
    cached_variables_for_bucket_.insert(
        {bucket_index, std::move(variables_for_bucket)});
    return cached_variables_for_bucket_[bucket_index];
  } else {
    return variables_for_bucket;
  }
}

// 当指定索引处的 bucket 准备好进行归约时调用。
  // 确保 bucket_index 大于等于 next_bucket_
  TORCH_INTERNAL_ASSERT(bucket_index >= next_bucket_);

  // 如果 bucket_index 大于 next_bucket_，则跳过当前 bucket 的标记操作
  // 因为 buckets 需要按顺序进行减少
  if (bucket_index > next_bucket_) {
    return;
  }

  // 继续执行，直到我们要么：
  // - 对所有的 bucket 都已经开始了减少操作，或者
  // - 找到一个尚未准备好进行减少操作的 bucket
  for (; next_bucket_ < buckets_.size() && buckets_[next_bucket_].pending == 0;
       next_bucket_++) {
    // 增加已准备好的 bucket 数量
    num_buckets_ready_++;

    // 如果这是第一个准备好的 bucket 并且需要收集运行时统计信息，则记录开始通信时间
    if (num_buckets_ready_ == 1 && should_collect_runtime_stats()) {
      record_backward_comm_start_time();
    }

    // 获取当前 bucket 的引用并进行全部归约操作
    auto& bucket = buckets_[next_bucket_];
    all_reduce_bucket(bucket);
  }
}

// 安装 futures 到 reducer
void Reducer::install_futures(
    c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futs) {
  // 如果尚未安装 futures，则直接移动赋值
  // 否则，追加 futures 到已有列表中
  if (!installed_futures_) {
    installed_futures_ = std::move(futs);
  } else {
    installed_futures_->append(futs);
  }
}

// 初始化 buckets
void Reducer::initialize_buckets(
    std::vector<std::vector<size_t>> bucket_indices) {
  // 如果在 DDP 构造函数内调用 initialize_buckets，则无需关心 rpc context ptr 是否为空
  // 因为 grad 不会被修改
  // 如果在训练循环中调用，例如在 rebuild_buckets() 内部，
  // 由于 grad 可能会被修改并指向 bucket_view，所以需要检查 rpc context ptr 是否为空
#ifndef _WIN32
  using torch::distributed::autograd::ThreadLocalDistAutogradContext;
  this->rpc_context_.set(ThreadLocalDistAutogradContext::getContextPtr());
#endif

  // 确保不在 autograd 执行期间调用 initialize_buckets
  REDUCER_CHECK(
      !expect_autograd_hooks_,
      logger_,
      "`initialize_buckets` must NOT be called during autograd execution.");

  // 清除当前的 bucket 分配
  buckets_.clear();
  variable_locators_.clear();

  // 确保变量定位器的大小与参数列表中的变量数量相同
  variable_locators_.resize(params_.size());

  // 遍历 bucket_indices
  const auto bucket_count = bucket_indices.size();
  buckets_.reserve(bucket_count);
  for (const auto bucket_index : c10::irange(bucket_count)) {
    Bucket bucket;

    // TODO(@pietern): 验证 indices 的有效性
    // 必须非空，唯一，并且在所有 bucket 中唯一
    REDUCER_CHECK(
        !bucket_indices[bucket_index].empty(),
        logger_,
        "Empty bucket specified.");

    // 如果 bucket 中只有一个索引，则该变量期望稀疏梯度
    if (bucket_indices[bucket_index].size() == 1) {
      const auto variable_index = bucket_indices[bucket_index].front();
      bucket.expect_sparse_gradient = expect_sparse_gradients_[variable_index];
      // ...
    } else {
      // 遍历当前桶中的每个变量索引
      for (const auto variable_index : bucket_indices[bucket_index]) {
        // 检查变量是否期望稀疏梯度，如果是则输出错误信息
        REDUCER_CHECK(
            !expect_sparse_gradients_[variable_index],
            logger_,
            "Buckets with more than one variable cannot include variables ",
            "that expect a sparse gradient.");
      }
    }

    if (bucket.expect_sparse_gradient) {
      // 如果当前桶期望稀疏梯度
      const auto variable_index = bucket_indices[bucket_index].front();
      const auto& variable = params_[variable_index];
      // 断言当前桶中只有一个变量
      TORCH_INTERNAL_ASSERT(bucket_indices[bucket_index].size() == 1);
      // 将该变量放入桶的变量集合中
      bucket.variables = {variable};
    }

    // 将参与的变量映射到当前桶中
    size_t intra_bucket_index = 0;
    for (const auto variable_index : bucket_indices[bucket_index]) {
      // 断言变量索引在有效范围内
      TORCH_INTERNAL_ASSERT(
          variable_index < variable_locators_.size(),
          "Out of range variable index specified.");
      // 将变量定位器更新为指向当前桶和内部索引
      variable_locators_[variable_index] =
          VariableLocator(bucket_index, intra_bucket_index++);
    }
    // 将当前桶的变量索引集合移动到桶对象中
    bucket.variable_indices = std::move(bucket_indices[bucket_index]);

    // 将构建好的桶对象添加到桶列表中
    buckets_.push_back(std::move(bucket));
  }
// 结束 Reducer 类的定义

// 初始化 bucket 的视图，根据梯度信息和参数情况创建视图
void Reducer::initialize_bucket_views(Reducer::Bucket& bucket) {
  // 获取当前 bucket 中的梯度信息
  const auto& gradients = bucket.gradients;
  // 遍历 bucket 中的每一个变量
  for (const auto i : c10::irange(bucket.variables.size())) {
    auto& v = bucket.variables[i];
    // 获取当前变量在 bucket 中的偏移量和长度
    const auto offset = bucket.offsets[i];
    const auto length = bucket.lengths[i];

    // TODO(@egienvalue): 在 MTIA 上完全支持视图操作之后，移除特殊情况
    // 一般情况下，在 MTIA 上，由于特殊的内存布局，不支持创建视图张量，
    // 而 aten::view 目前会在 MTIA 上创建一个新的张量。
    if (v.is_non_overlapping_and_dense() && !v.is_mtia()) {
      // 如果参数的内存布局是密集的，匹配其布局，预期 autograd 引擎 (AccumulateGrad)
      // 也会创建与其布局匹配的梯度。
      bucket.bucket_views_in.push_back(
          gradients.as_strided(v.sizes(), v.strides(), offset));
    } else {
      // 回退到 C 风格的连续视图，预期当为非密集参数积累梯度时，AccumulateGrad 也会执行相同操作。
      bucket.bucket_views_in.push_back(
          gradients
              .narrow(
                  0, static_cast<int64_t>(offset), static_cast<int64_t>(length))
              .view(v.sizes()));
    }

    // 默认情况下 `bucket_views_out` 和 `bucket_views_in` 是一样的。
    bucket.bucket_views_out = bucket.bucket_views_in;

    // 如果设置了 gradient_as_bucket_view_ 为 true，则需要处理两种情况：
    // - 在 rebuild_buckets 过程中调用 initialize_bucket_views，如果梯度已经在
    //   前一次迭代中被定义/计算过，则需要将旧梯度复制到新的 bucket_view 中，并
    //   使 grad 指向新的 bucket_view。
    // - 在构造过程中调用 initialize_bucket_views，此时梯度未定义，因此不应该
    //   让 grad 指向 bucket_view，因为全局未使用的参数的梯度应保持未定义状态。
    if (gradient_as_bucket_view_) {
      auto& bucket_view = bucket.bucket_views_in.back();
      runGradCallbackForVariable(v, [&](auto& grad) {
        if (grad.defined() && !grad.is_alias_of(bucket_view)) {
          bucket_view.copy_(grad);
          grad = bucket_view;
          // 修改了 grad 并需要写回
          return true;
        }
        // 没有修改 grad，不需要写回
        return false;
      });
    }
  }
}

// 清空 bucket_views_out 并为每个变量的 tensor 填充 bucket_views_out
void Reducer::populate_bucket_views_out(
    Reducer::Bucket& bucket,
    at::Tensor& tensor) {
  bucket.bucket_views_out.clear();
  // 遍历 bucket 中的每一个变量
  for (const auto i : c10::irange(bucket.variables.size())) {
    const auto& v = bucket.variables[i];
    const auto offset = bucket.offsets[i];
    const auto length = bucket.lengths[i];
    // TODO(@egienvalue): remove special case after view ops are fully
    // supported on MTIA.
    // In general, on MTIA, due to the special memory layout, it doesn't
    // support as_strided which creates a view tensor and aten::view will
    // create a new tensor on MTIA for now.
    // 获取当前参数在 bucket 中的偏移量和长度

    if (v.is_non_overlapping_and_dense() && !v.is_mtia()) {
      // If the param's memory is dense, match its layout, anticipating
      // the autograd engine (AccumulateGrad) will also create gradients
      // matching its layout.
      // 如果参数的内存布局是密集的且不在 MTIA 上，则匹配其布局，
      // 预期 autograd 引擎（AccumulateGrad）也会创建匹配其布局的梯度。
      bucket.bucket_views_out.push_back(
          tensor.as_strided(v.sizes(), v.strides(), offset));
      // 将基于给定大小、步长和偏移量创建的张量视图添加到 bucket 的输出视图列表中
    } else {
      // Fall back to a C-style contiguous view, again anticipating
      // AccumulateGrad will do the same when stashing grads for non-dense
      // params.
      // 否则，回退到 C 风格的连续视图，再次预期 AccumulateGrad 在为非密集参数
      // 存储梯度时也会采用相同的方式。
      bucket.bucket_views_out.push_back(
          tensor
              .narrow(
                  0, static_cast<int64_t>(offset), static_cast<int64_t>(length))
              .view(v.sizes()));
      // 将基于给定偏移量和长度进行窄化（narrow）并视图（view）操作后的张量添加到
      // bucket 的输出视图列表中
    }
// 使用互斥锁保护临界区，确保线程安全
void Reducer::prepare_for_forward() {
  std::lock_guard<std::mutex> lock(mutex_);
  // 增加迭代计数器
  num_iterations_++;
  // 如果需要收集运行时统计信息
  if (should_collect_runtime_stats()) {
    // 记录前向计算开始时间
    record_forward_compute_start_time();
  }
}

// 重置桶计数器及相关状态
void Reducer::reset_bucket_counting() {
  // 下一个桶的索引置零
  next_bucket_ = 0;
  // 在每次迭代的反向计算开始时重置 num_buckets_ready_
  num_buckets_ready_ = 0;

  // 遍历所有桶，设置每个桶的待处理变量数目
  for (auto& bucket : buckets_) {
    bucket.pending = bucket.variables.size();
  }

  // 如果使用静态图，则将 numGradHooksTriggeredMap_ 的值赋给 numGradHooksTriggeredMapPerIteration_
  if (static_graph_) {
    numGradHooksTriggeredMapPerIteration_ = numGradHooksTriggeredMap_;
  }
}

// 从指定输出开始遍历自动求导图
void Reducer::search_unused_parameters(
    const std::vector<torch::autograd::Variable>& outputs) {
  // 记录函数调用，用于自动求导图搜索的追踪
  RECORD_FUNCTION(
      "torch.distributed.ddp.reducer::search_unused_parameters",
      std::vector<c10::IValue>());

  // 存放已经访问过的自动求导节点
  std::unordered_set<torch::autograd::Node*> seen;
  // 存放待访问的自动求导节点
  std::vector<torch::autograd::Node*> queue;

  // 使用所有输出的梯度函数初始化队列
  for (const auto& output : outputs) {
    const auto& grad_fn = output.grad_fn();
    if (grad_fn) {
      queue.push_back(grad_fn.get());
    }
  }

  // 从指定输出开始遍历自动求导图
  while (!queue.empty()) {
    auto fn = queue.back();
    queue.pop_back();
    // 遍历当前节点的下一个边
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        // 将新发现的节点加入队列，确保不重复访问
        const bool was_inserted = seen.insert(next_ptr).second;
        if (was_inserted) {
          queue.push_back(next_ptr);
        }
      }
    }
  }

  // 查找未在自动求导图中出现的累加器函数
  for (const auto& it : gradAccToVariableMap_) {
    // 如果累加器函数不在自动求导图中
    if (seen.count(it.first) == 0) {
      // 如果调试级别为详细，则记录未使用的参数信息
      if (ddp_debug_level_ == c10d::DebugLevel::Detail) {
        const auto param_info = param_names_.find(it.second);
        TORCH_INTERNAL_ASSERT(
            param_info != param_names_.end(),
            "Did not find variable index ",
            it.second,
            " in DDP parameter name mapping!");
        const auto param_name = param_info->second;
        LOG(INFO) << "[Rank " << process_group_->getRank() << "]: "
                  << "Parameter " << param_name << " at index " << it.second
                  << " is marked as unused.";
      }
      // 将未使用的参数索引添加到列表中
      unused_parameters_.push_back(it.second);
    }
  }

  // 如果所有参数均已使用，则警告用户可能存在不必要的性能损失
  if (unused_parameters_.empty()) {
    TORCH_WARN_ONCE(
        "find_unused_parameters=True was specified in DDP constructor, "
        "but did not find any unused parameters in the forward pass. This flag "
        "results in an extra traversal of the autograd graph every iteration, "
        " which can adversely affect performance. If your model indeed never "
        "has any unused parameters in the forward pass, consider turning this "
        "flag off. Note that this warning may be a false positive if your model "
        "has flow control causing later iterations to have unused parameters.");
  }


    // 如果在 DDP 构造函数中指定了 find_unused_parameters=True，但在前向传播中未发现任何未使用的参数，
    // 则发出警告。这个标志会导致每次迭代都额外遍历自动求导图，可能对性能产生不利影响。
    // 如果你的模型确实在前向传播中从未有未使用的参数，考虑关闭这个标志。
    // 注意，如果你的模型有流控制，导致后续迭代有未使用的参数，这个警告可能是误报的。
  }


  if (!static_graph_ && ddp_graph_static_) {


    // 如果当前不是静态图且之前设置为动态分布式数据并行 (DDP) 图，
    // 进入条件判断，用于检查图是否保持静态状态。
    if (num_iterations_ > 1) {


      // 如果迭代次数大于1，继续进行下面的操作。
      // 当前图仍然是静态的，如果未使用的参数集合没有改变。
      ddp_graph_static_ =
          prev_iteration_unused_parameters_ == unused_parameters_;


      // 如果当前图不再是静态的，
      // 记录日志指示图不再是静态的状态。日志记录器确保此操作仅执行一次，以避免额外开销。
      if (!ddp_graph_static_) {
        logger_.lock()->log_if_graph_static(false);
      }
    }


    // 更新上一迭代的未使用参数集合，用于下一次迭代的比较。
    prev_iteration_unused_parameters_ = unused_parameters_;
  }
}

// 准备反向传播，初始化各种参数和状态
void Reducer::prepare_for_backward(
    const std::vector<torch::autograd::Variable>& outputs) {
  // 使用互斥锁保护多线程操作
  std::lock_guard<std::mutex> lock(mutex_);

  // 记录反向传播开始时间
  backward_compute_start_time_ = current_time_in_nanos();
  // 如果需要收集运行时统计信息，则记录反向传播开始时间
  if (should_collect_runtime_stats()) {
    record_backward_compute_start_time();
  }

  // 重置计数器和状态
  expect_autograd_hooks_ = true;
  // 清空梯度准备就绪的顺序索引，因为在下一次迭代中可能会改变
  grad_ready_order_indices_.clear();

  // 重置桶计数
  reset_bucket_counting();

  // 重置未使用参数的计数
  has_marked_unused_parameters_ = false;
  // 清空每次迭代中准备就绪的参数列表
  perIterationReadyParams_.clear();

  // 如果动态图未设置，搜索图形以检测未使用的参数
  // 当静态图设置后，未使用的参数将被检测并在第一次迭代后不再改变
  // 如果 static_graph_ = false 并且 find_unused_parameters_ = false，
  // 我们假设所有变量都会调用 autograd 钩子，不必搜索图形以查找这些钩子的存在
  if (dynamic_graph_find_unused()) {
    unused_parameters_.clear();
    search_unused_parameters(outputs);
  }
}

// 将桶中的梯度复制到变量的梯度中
void Reducer::copy_bucket_to_grad(
    at::Tensor& variable,
    Reducer::Bucket& bucket,
    size_t intra_bucket_index,
    bool global_unused) {
  // 获取桶视图
  const auto& bucket_view = bucket.bucket_views_out[intra_bucket_index];
  // 运行梯度回调函数
  runGradCallbackForVariable(variable, [&](auto& grad) {
    // 如果参数在全局上未使用，则保持其梯度不变
    if (!global_unused) {
      if (!grad.defined()) {
        // 根据“梯度布局协定”创建梯度
        grad = torch::autograd::utils::clone_obey_contract(bucket_view, variable);
      } else {
        grad.copy_(bucket_view);
      }
      // 梯度已经修改，需要写回
      return true;
    }
    // 梯度未修改
    return false;
  });
}

// 获取当前迭代中未标记的参数名列表
std::vector<std::string> Reducer::getUnmarkedParamsForIteration() {
  std::vector<std::string> unMarkedParamNames;
  for (const auto& it : param_names_) {
    if (perIterationReadyParams_.find(it.first) ==
        perIterationReadyParams_.end()) {
      unMarkedParamNames.push_back(it.second);
    }
  }
  return unMarkedParamNames;
}

// 获取当前迭代中未标记的参数索引列表
std::vector<size_t> Reducer::getUnmarkedParamIndicesForIteration() {
  std::vector<size_t> unmarked_param_indices;
  const auto variable_count = params_.size();
  for (const auto variable_index : c10::irange(variable_count)) {
    if (perIterationReadyParams_.find(variable_index) ==
        perIterationReadyParams_.end()) {
      unmarked_param_indices.push_back(variable_index);
    }
  }
  return unmarked_param_indices;
}

// 对于一个或多个包含稠密张量的桶，需要进行最终化处理
void Reducer::finalize_bucket_dense(Bucket& bucket) {
  // 遍历桶中的每个变量索引
  for (const auto intra_bucket_index : c10::irange(bucket.variables.size())) {
    // 获取对应于给定索引的 bucket 变量的引用
    auto& variable = bucket.variables[intra_bucket_index];

    // 初始化一个标志位，用于表示全局是否未使用
    bool global_unused = false;
    // 查看是否需要跳过 allreducing local_used_map_dev
    if (static_graph_ || find_unused_parameters_) {
      // 确定该参数是否在全局范围内被使用
      //
      // 如果该变量在局部被使用，它也会在全局范围内被使用，因此我们无需等待归约。
      // 否则，我们会延迟等待归约的完成，直到遇到一个在局部未使用的变量。
      // 然后我们会延迟同步点，即 local_used_work_->wait() 的意思是如果我们根本没有未使用的参数，
      // 我们可以完全跳过等待工作的完成，并对所有参数都使用的模型造成可忽略的性能开销。
      // 这种延迟等待的方式可以最小化对大多数所有参数都始终被使用的模型的性能影响。
      // 然后，只有在确实有一个在局部未使用的参数时，我们才需要支付额外的开销，因为我们需要检查其是否也在全局范围内未使用。
      int64_t variable_index =
          static_cast<int64_t>(bucket.variable_indices[intra_bucket_index]);
      // 注意：global_unused 可能还没有全局性。因为我们懒惰地等待归约的完成，它只有在我们到达下面这一点时才会真正变得全局化，
      // 那时我们等待归约工作，进行 D2H 复制，并使用真正的全局共识更新 global_unused，即 local_used_map_reduced_ 为 true 时。
      global_unused = local_used_map_[variable_index].item<int>() == 0;
      if (global_unused && !local_used_map_reduced_) {
        // 等待 local_used_map 的归约完成
        local_used_work_->wait();
        // 从 local_used_map_dev_ 拷贝到 local_used_map_
        // 如果 local_used_map_dev_ 是 cuda，则为阻塞复制
        local_used_map_.copy_(local_used_map_dev_);

        // 再次检查 global_unused 是否为 0
        global_unused = local_used_map_[variable_index].item<int>() == 0;
        local_used_map_reduced_ = true;
      }
    }

    // 如果不是以 bucket 视图形式处理梯度
    if (!gradient_as_bucket_view_) {
      if (optim_in_backward_) {
        // 如果优化器已经运行，则提前返回
        runGradCallbackForVariable(variable, [&](auto& grad) { return true; });
      } else {
        // 记录函数调用，用于跟踪
        RECORD_FUNCTION(
            "torch.distributed.ddp.reducer::copy_bucket_to_grad",
            std::vector<c10::IValue>({variable}));
        // 将 bucket 数据复制到梯度中
        copy_bucket_to_grad(
            variable, bucket, intra_bucket_index, global_unused);
      }
    }
    } else {
      // 获取当前子桶的输出视图
      const auto& bucket_view_out = bucket.bucket_views_out[intra_bucket_index];
      // 获取当前子桶的输入视图，并允许修改
      auto& bucket_view_in = bucket.bucket_views_in[intra_bucket_index];
      // 如果注册了通信钩子，`bucket_view_out` 将存储在新分配的张量中的全约简结果，
      // 因此我们将 `bucket_view_out` 复制回 `bucket_view_in` 以便这个梯度。
      if (!bucket_view_in.is_alias_of(bucket_view_out)) {
        bucket_view_in.copy_(bucket_view_out);
      }
      // 为变量运行梯度回调函数
      runGradCallbackForVariable(variable, [&](auto& grad) {
        if (optim_in_backward_) {
          // 如果优化器已经运行，则提前返回
          return true;
        }
        // 如果参数在全局上未使用，则保持其梯度不变
        if (!global_unused) {
          // 如果全局使用但局部未使用，则让 grad 指向 bucket_view_in
          if (!grad.defined()) {
            grad = bucket_view_in;
          } else {
            // 如果 grad 不是 bucket_view_in 的别名，则报错
            if (!grad.is_alias_of(bucket_view_in)) {
              REDUCER_CHECK(
                  false,
                  logger_,
                  "Detected at least one parameter gradient is not the "
                  "expected DDP bucket view with gradient_as_bucket_view=True. "
                  "This may happen (for example) if multiple allreduce hooks "
                  "were registered onto the same parameter. If you hit this error, "
                  "please file an issue with a minimal repro.");
            }
          }
          // grad 已修改，需要写回
          return true;
        }
        // grad 未被修改
        return false;
      });
    }
  // 不再期望此函数返回后触发自动求导钩子
  TORCH_INTERNAL_ASSERT(expect_autograd_hooks_);
  expect_autograd_hooks_ = false;
  // 为下一次迭代重置状态
  first_autograd_hook_called_ = false;

  // 不再要求此函数返回后调用 finalize
  TORCH_INTERNAL_ASSERT(require_finalize_);
  require_finalize_ = false;

  // 等待异步归约完成，并展开 bucket 的扁平化梯度张量
  for (auto& bucket : buckets_) {
    // 参见注释 [DDP Communication Hook]
    TORCH_INTERNAL_ASSERT(
        bucket.future_work,
        "Expected bucket.future_work not to be null. "
        "This may indicate that communication hook was not properly installed.");
    bucket.future_work->wait();
    auto future_result = comm_hook_ == nullptr
        ? detail::parseCppCommHookResult(bucket.future_work->value())
        : comm_hook_->parseHookResult(bucket.future_work->value());
    if (bucket.expect_sparse_gradient) {
      // 如果设置了稀疏元数据，则 bucket 应包含稀疏梯度张量的索引
      if (sparse_metadata_) {
        REDUCER_CHECK(
            bucket.sparse_tensor_indices.value().numel() ==
                bucket.gradients.sizes()[0],
            logger_,
            "Sparse metadata and gradient size mismatch");
        auto sparse_result = at::sparse_coo_tensor(
            bucket.sparse_tensor_indices.value(),
            future_result,
            bucket.gradients.sizes());
        bucket.gradients.copy_(sparse_result);
      } else {
        bucket.gradients.copy_(future_result);
      }
    } else {
      // 仅使用 future_result 重新初始化 bucket_views_out，遵循 initialize_buckets 中的逻辑
      populate_bucket_views_out(bucket, future_result);
    }

    // 重置所有归约除法因子，因为在 DDP join 模式下可能会更改
    div_factor_ = kUnsetDivFactor;

    if (!bucket.expect_sparse_gradient) {
      // 不需要完成稀疏 bucket，因为稀疏梯度和 bucket 本质上指向同一存储。
      // 因此，一旦完成 allreduce，稀疏梯度就会自动更新。
      finalize_bucket_dense(bucket);
    }
  }

  if (installed_futures_ != c10::nullopt) {
    // 等待所有已安装的 futures 完成
    c10::collectAll(*installed_futures_)->wait();
    installed_futures_ = c10::nullopt;
  }

  // 参见注释 [Skip allreducing local_used_maps_dev]
  if (dynamic_graph_find_unused() || static_graph_first_iteration()) {
    // 由于延迟等待，当前迭代的归约可能仍在进行中，而下一次迭代的归约已经启动。
    // 对于这种情况，我们希望显式等待以确保归约在启动下一次之前完成。
    // 否则，前一个归约可能会干扰，写入设备端内存并覆盖
    // 如果 local_used_map_reduced_ 为假，则等待 local_used_work_ 完成
    if (!local_used_map_reduced_) {
      local_used_work_->wait();
    }
  }

  // 如果动态图中存在未使用的参数，则执行以下操作
  if (dynamic_graph_find_unused()) {
    // 重置未使用参数的计数
    // 参见注释 [local_used_map_ -> local_used_map_dev copying]
    local_used_map_.fill_(0);
    local_used_map_reduced_ = false;
  }

  // 如果需要收集运行时统计信息，则记录反向通信结束时间
  if (should_collect_runtime_stats()) {
    record_backward_comm_end_time();
  }

  // 重置稀疏元数据
  sparse_metadata_.reset();
}

void Reducer::runGradCallbackForVariable(
    at::Tensor& variable,
    GradCallback&& cb) {
#ifdef _WIN32
  // 如果在 Windows 平台，直接调用回调函数处理变量的梯度
  cb(variable.mutable_grad());
#else
  // 在非 Windows 平台，获取当前的 RPC 上下文指针
  auto context_ptr = rpc_context_.context_ptr.load();
  // 如果上下文指针为空，也直接调用回调函数处理变量的梯度
  if (context_ptr == nullptr) {
    cb(variable.mutable_grad());
  } else {
    // 在分布式自动求导情况下，通过上下文指针调用回调函数处理变量的梯度
    context_ptr->runGradCallbackForVariable(variable, std::move(cb));
  }
#endif
}

#ifndef _WIN32
void Reducer::RpcContext::set(ContextPtr&& new_context_ptr) {
  // 设置新的上下文指针 'new_context_ptr'，即使它是 nullptr 也应该设置
  const auto new_context_raw_ptr = new_context_ptr.get();
  // 使用原子操作更新上下文指针，以避免多线程数据竞争
  if (context_ptr.exchange(new_context_raw_ptr) != new_context_raw_ptr) {
    // 如果是首次设置上下文指针，则将共享指针 'context_ptr_holder' 设置为 'new_context_ptr'
    // 所有调用点应该使用同一个上下文指针
    context_ptr_holder = std::move(new_context_ptr);
  }
}
#endif

void Reducer::sync_bucket_indices(
    std::vector<std::vector<size_t>>& bucket_indices) {
  auto num_buckets = bucket_indices.size();
  std::vector<size_t> bucket_sizes;
  bucket_sizes.reserve(num_buckets);
  int64_t total_size = 0;
  for (const auto i : c10::irange(num_buckets)) {
    auto bucket_size = bucket_indices.at(i).size();
    bucket_sizes.push_back(bucket_size);
    total_size += static_cast<int64_t>(bucket_size);
  }

  at::TensorOptions options;
  options = options.dtype(at::kInt);
  options = options.device(params_[0].device());

  // 创建一个大小为 total_size + 1 的整数类型的空张量 'indices_tensor'
  // 将所有的索引值组合到 'indices_tensor' 中，并在最后一个位置放置 'num_buckets'
  auto indices_tensor = at::empty({total_size + 1}, at::kInt);
  auto indices_accessor = indices_tensor.accessor<int, 1>();
  auto indices_accessor_Index = 0;
  for (const auto i : c10::irange(num_buckets)) {
    const auto& bucket_size = bucket_indices.at(i).size();
    for (const auto j : c10::irange(bucket_size)) {
      indices_accessor[indices_accessor_Index++] =
          static_cast<int>(bucket_indices[i][j]);
    }
  }
  indices_accessor[indices_accessor_Index] = static_cast<int>(num_buckets);

  // 将 CPU 上的张量 'indices_tensor' 复制到设备上的张量 'indices_tensor_device'
  // 进程组可能是 NCCL，只能广播设备上的张量
  auto indices_tensor_device = at::empty({total_size + 1}, options);
  indices_tensor_device.copy_(indices_tensor, /*non_blocking=*/true);
  std::vector<at::Tensor> indices_tensor_list = {indices_tensor_device};
  process_group_->broadcast(indices_tensor_list)->wait();
  indices_tensor.copy_(indices_tensor_list.front(), /*non_blocking=*/false);

  // 从 rank 0 接收 'num_buckets' 后更新它
  num_buckets = indices_accessor[indices_accessor_Index];

  // 创建一个大小为 'num_buckets' 的整数类型的空张量 'bucket_sizes_tensor'
  // 广播 'bucket_sizes' 张量
  auto bucket_sizes_tensor = at::empty({(int64_t)num_buckets}, at::kInt);
  auto bucket_sizes_accessor = bucket_sizes_tensor.accessor<int, 1>();
  for (const auto i : c10::irange(num_buckets)) {
  // 对于排名不为 0 的情况，本地的桶大小 bucket_sizes.size() 可能小于广播的 num_buckets
  bucket_sizes_accessor[i] =
      bucket_sizes.at(std::min(i, (bucket_sizes.size() - 1)));
}

// 在设备上创建一个形状为 {num_buckets} 的空张量
auto bucket_sizes_tensor_device = at::empty({(int64_t)num_buckets}, options);
// 将 bucket_sizes_tensor 的数据拷贝到设备张量上，非阻塞操作
bucket_sizes_tensor_device.copy_(bucket_sizes_tensor, /*non_blocking=*/true);

// 创建一个张量列表，包含设备上的 bucket_sizes_tensor_device
std::vector<at::Tensor> bucket_sizes_tensor_list = {
    bucket_sizes_tensor_device};
// 使用 process_group_ 对象进行广播，等待广播完成
process_group_->broadcast(bucket_sizes_tensor_list)->wait();
// 将接收到的广播数据拷贝回 bucket_sizes_tensor，阻塞操作
bucket_sizes_tensor.copy_(
    bucket_sizes_tensor_list.front(), /*non_blocking=*/false);

// 清空 bucket_indices，然后使用从排名 0 接收到的 num_buckets、bucket_sizes_tensor 和 indices_tensor 更新 bucket_indices
bucket_indices.clear();
bucket_indices.reserve(num_buckets);
indices_accessor_Index = 0;
for (const auto i : c10::irange(num_buckets)) {
  // 获取当前桶的大小
  const auto& bucket_size = bucket_sizes_accessor[static_cast<int64_t>(i)];
  // 创建一个容量为 bucket_size 的桶
  std::vector<size_t> bucket;
  bucket.reserve(bucket_size);
  // 填充当前桶的索引数据
  for (const auto j : c10::irange(bucket_size)) {
    (void)j;  // 确保 j 被使用，避免编译器警告
    bucket.push_back(indices_accessor[indices_accessor_Index++]);
  }
  // 将当前桶添加到 bucket_indices 中
  bucket_indices.emplace_back(std::move(bucket));
}
  // 确保上一个反向传播的缩减操作已经完成。如果用户的模型存在未使用的参数，
  // 例如，此处将会引发错误，建议使用 find_unused_parameters=True 来代替下面的尺寸不匹配异常。
  std::lock_guard<std::mutex> lock(mutex_);
  // 确保之前的缩减操作已经完成
  ensure_prior_reduction_finished();
  // 如果不需要重建桶或者重建参数为空，则返回 false
  if (!should_rebuild_buckets() || rebuilt_params_.empty()) {
    return false;
  }

  // 断言重建后的参数张量数量与重建后的参数索引数量相同
  TORCH_INTERNAL_ASSERT(
      rebuilt_params_.size() == rebuilt_param_indices_.size(),
      c10::str(
          "rebuilt parameter tensors size is not same as rebuilt parameter indices size: ",
          rebuilt_params_.size(),
          " versus ",
          rebuilt_param_indices_.size()));
  // 断言模型参数数量与重建后的参数索引数量相同
  TORCH_INTERNAL_ASSERT(
      params_.size() == rebuilt_param_indices_.size(),
      c10::str(
          "rebuilt parameter indices size is not same as original model parameters size.",
          "Original model param size is: ",
          params_.size(),
          " versus rebuilt params size of: ",
          rebuilt_param_indices_.size()));

  // 设置桶的大小限制
  std::vector<size_t> bucket_size_limits;
  bucket_size_limits.push_back(first_bucket_bytes_cap_);
  bucket_size_limits.push_back(bucket_bytes_cap_);
  
  // 检查是否将最后一个桶的大小设置为较小值
  auto ddp_set_last_bucket_as_small =
      (getCvarString({"DDP_SET_LAST_BUCKET_CAP"}, "N/A") == "1");

  // 如果设置了将最后一个桶的大小设置为较小值，则反转桶的顺序
  if (ddp_set_last_bucket_as_small) {
    // 反转重建后的参数和重建后的参数索引的顺序
    std::reverse(rebuilt_params_.begin(), rebuilt_params_.end());
    std::reverse(rebuilt_param_indices_.begin(), rebuilt_param_indices_.end());
  }

  // 计算基于大小的桶分配
  auto [rebuilt_bucket_indices, per_bucket_size_limits] =
      compute_bucket_assignment_by_size(
          rebuilt_params_,
          bucket_size_limits,
          expect_sparse_gradients_,
          rebuilt_param_indices_,
          logger_);

  // 如果设置了将最后一个桶的大小设置为较小值，则再次反转桶的顺序
  if (ddp_set_last_bucket_as_small) {
    std::reverse(rebuilt_bucket_indices.begin(), rebuilt_bucket_indices.end());
    std::reverse(per_bucket_size_limits.begin(), per_bucket_size_limits.end());
  }

  // 如果调试级别不是 Off，则断言重建后的桶索引数量与每个桶大小限制数量相同
  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    TORCH_INTERNAL_ASSERT(
        rebuilt_bucket_indices.size() == per_bucket_size_limits.size())
    // 使用 LOG(INFO) 输出重建后的桶索引数量以及每个桶大小限制的信息
    LOG(INFO) << rebuilt_bucket_indices.size()
              << " buckets rebuilt with size limits: "
              << c10::Join(", ", per_bucket_size_limits) << " bytes.";
  }

  // 对于重建后的桶索引，需要在所有进程之间同步。
  // 从默认的第 0 号进程广播新重建的桶索引。
  // 同步完重建的桶索引后，为缩减器初始化桶。
  sync_bucket_indices(rebuilt_bucket_indices);

  // 标记已经完成桶重建
  has_rebuilt_bucket_ = true;

  // 清空已重建的参数和参数索引
  rebuilt_params_.clear();
  rebuilt_param_indices_.clear();

  // 使用重建后的桶索引初始化桶
  initialize_buckets(std::move(rebuilt_bucket_indices));

  // 返回操作成功标志
  return true;
}

// 设置稀疏元数据，使用给定的 metadata 初始化 sparse_metadata_
void Reducer::setSparseMetadata(std::map<std::string, at::Tensor>& metadata) {
  sparse_metadata_ =
      std::make_unique<std::map<std::string, at::Tensor>>(metadata);
}

// 注册自定义的通信钩子接口
// 查看注释 [DDP Communication Hook]
void Reducer::register_comm_hook(std::unique_ptr<CommHookInterface> iface) {
  // 检查是否已经注册过通信钩子，只能注册一次
  REDUCER_CHECK(
      comm_hook_ == nullptr,
      logger_,
      "register_comm_hook or register_builtin_comm_hook can only be called once.");

  // 将传入的通信钩子接口 iface 移动到 comm_hook_
  comm_hook_ = std::move(iface);
}

// 注册内置的通信钩子
// 查看注释 [DDP Communication Hook]
void Reducer::register_builtin_comm_hook(
    c10d::BuiltinCommHookType comm_hook_type) {
  // 检查是否已经注册过通信钩子，只能注册一次
  REDUCER_CHECK(
      comm_hook_ == nullptr,
      logger_,
      "register_builtin_comm_hook or register_comm_hook can only be called once.");

  // 根据给定的 comm_hook_type 创建相应的内置通信钩子并注册
  switch (comm_hook_type) {
    case c10d::BuiltinCommHookType::ALLREDUCE:
      comm_hook_ = std::make_unique<c10d::AllReduceCommHook>(process_group_);
      LOG(INFO) << "Built-in communication hook ALLREDUCE is registered.";
      break;
    case c10d::BuiltinCommHookType::FP16_COMPRESS:
      comm_hook_ = std::make_unique<c10d::FP16CompressCommHook>(process_group_);
      LOG(INFO) << "Built-in communication hook FP16_COMPRESS is registered.";
      break;
    default:
      // 如果提供了未知的内置 DDP comm hook 类型，则发出警告
      TORCH_WARN_ONCE(
          "Unknown built-in DDP comm hook type is provided. No comm hook will be used.");
  }
}

// 确保之前的 reduction 已完成
void Reducer::ensure_prior_reduction_finished() {
  // 检查是否需要完成前一个 reduction
  // 变量 `require_finalize_` 在所有梯度计算完成并开始所有 bucket 的 reduction 之前为 true
  if (require_finalize_) {
    // 收集未标记的参数索引，在调试模式下还会获取参数名称
    auto unmarked_param_indices = getUnmarkedParamIndicesForIteration();
    // 应该有一些未标记的参数索引，否则不会进入此错误分支
    TORCH_INTERNAL_ASSERT(!unmarked_param_indices.empty());

    // 错误信息：预期在开始新的 iteration 前完成前一个 reduction
    std::string kBaseErrorMsg =
        "Expected to have finished reduction in the prior iteration before "
        "starting a new one. "
        ""
        "This error indicates that your module has parameters that were "
        "not used in producing loss. ";
    std::string kOutputsNotUsedInLossErrorMsg =
        "making sure all "
        "`forward` function outputs participate in calculating loss. ";
    std::string kDDPBugErrorMsg =
        "\nIf you already have done the above, then the distributed "
        "data parallel module wasn't able to locate the output tensors in the "
        "return value of your module's `forward` function. "
        "Please include the loss function and the structure of the return "
        "value of `forward` of your module when reporting this issue (e.g. "
        "list, dict, iterable).";
}
    // 如果 static_graph_ 为真，则给 kBaseErrorMsg 添加静态图错误信息
    if (static_graph_) {
      kBaseErrorMsg =
          "Expected to have finished reduction in the prior iteration before "
          "starting a new one. "
          "This error indicates that your training graph has changed "
          "in this iteration, e.g., one parameter is used in first "
          "iteration, but then got unused in the second iteration. "
          "this is not compatible with static_graph set to True.";
    } else if (!find_unused_parameters_) {
      // 如果未启用 find_unused_parameters_，则给 kBaseErrorMsg 添加相关提示信息
      // 提示如何启用未使用参数检测
      kBaseErrorMsg +=
          "You can enable unused parameter detection by passing the "
          "keyword argument `find_unused_parameters=True` to "
          "`torch.nn.parallel.DistributedDataParallel`, and by \n";
      kBaseErrorMsg += kOutputsNotUsedInLossErrorMsg;
      kBaseErrorMsg += kDDPBugErrorMsg;
    } else {
      // 如果启用了 find_unused_parameters_，则给 kBaseErrorMsg 添加相应提示信息
      // 提示使用者可能出现的问题及解决方法
      kBaseErrorMsg +=
          "Since `find_unused_parameters=True` is enabled, this likely "
          " means that not all `forward` outputs participate in computing loss. You can fix this by ";
      kBaseErrorMsg += kOutputsNotUsedInLossErrorMsg;
      kBaseErrorMsg += kDDPBugErrorMsg;
    }

    // 构建未标记参数索引信息字符串
    const std::string unmarked_param_indices_info = c10::str(
        "\n",
        "Parameter indices which did not receive grad for rank ",
        process_group_->getRank(),
        ": ",
        unmarked_param_indices);

    if (ddp_debug_level_ == DebugLevel::Off) {
      // 如果 ddp_debug_level_ 为 Off，则添加关于未标记参数索引信息的提示
      kBaseErrorMsg += unmarked_param_indices_info;
      kBaseErrorMsg +=
          "\n In addition, you can set the environment variable "
          "TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information "
          "about which particular parameters did not receive gradient on this rank "
          "as part of this error";
    } else {
      // 检索没有接收梯度的参数名称集合。
      auto unmarkedParams = getUnmarkedParamsForIteration();
      // 断言未标记参数集合不为空。
      TORCH_INTERNAL_ASSERT(!unmarkedParams.empty());
      // 遍历未标记参数集合，记录日志显示没有在反向传播中接收梯度的参数信息。
      for (const auto& s : unmarkedParams) {
        LOG(INFO) << "[Rank " << process_group_->getRank() << "] "
                  << "Parameter: " << s
                  << " did not get gradient in backwards pass.";
      }
      // 使用逗号连接未标记参数信息。
      const std::string unmarkedParamInfo = c10::Join(", ", unmarkedParams);
      // 在调试模式下，记录参数名称和未使用的索引信息。
      kBaseErrorMsg += c10::str(
          "\n",
          "Parameters which did not receive grad for rank ",
          process_group_->getRank(),
          ": ",
          unmarkedParamInfo);
      // 添加未标记参数索引信息到错误消息。
      kBaseErrorMsg += unmarked_param_indices_info;
    }
    // 使用 REDUCER_CHECK 断言机制，如果条件为假，则记录错误消息。
    REDUCER_CHECK(false, logger_, kBaseErrorMsg);
  }
}

// 设置 DDP 运行时日志采样率
void Reducer::set_ddp_runtime_logging_sample_rate(int sample_rate) {
    // 将参数 sample_rate 赋值给成员变量 ddp_runtime_logging_sample_rate_
    ddp_runtime_logging_sample_rate_ = sample_rate;
}

// 获取 DDP 运行时日志采样率
int Reducer::get_ddp_runtime_logging_sample_rate() {
    // 返回成员变量 ddp_runtime_logging_sample_rate_ 的值
    return ddp_runtime_logging_sample_rate_;
}

// 判断是否应该收集运行时统计信息
bool Reducer::should_collect_runtime_stats() {
    // 如果迭代次数 num_iterations_ 大于 0，并且满足条件
    if (num_iterations_ > 0 &&
        (num_iterations_ <= 10 ||
         num_iterations_ % get_ddp_runtime_logging_sample_rate() == 0)) {
        return true;
    }
    // 否则返回 false
    return false;
}

// 记录前向计算开始时间
void Reducer::record_forward_compute_start_time() {
    // 如果 timer_ 存在，记录前向计算开始时间
    if (timer_) {
        timer_->record(Timer::Event::kForwardStart);
    }
}

// 记录反向计算开始时间
void Reducer::record_backward_compute_start_time() {
    // 如果 timer_ 存在，记录反向计算开始时间
    if (timer_) {
        timer_->record(Timer::Event::kBackwardComputeStart);
    }
}

// 记录反向计算结束时间
void Reducer::record_backward_compute_end_time() {
    // 如果 timer_ 存在，记录反向计算结束时间
    if (timer_) {
        timer_->record(Timer::Event::kBackwardComputeEnd);
    }
}

// 记录反向通信开始时间
void Reducer::record_backward_comm_start_time() {
    // 如果 timer_ 存在，记录反向通信开始时间
    if (timer_) {
        timer_->record(Timer::Event::kBackwardCommStart);
    }
}

// 记录反向通信结束时间
void Reducer::record_backward_comm_end_time() {
    // 如果 timer_ 存在，记录反向通信结束时间
    if (timer_) {
        timer_->record(Timer::Event::kBackwardCommEnd);
    }
}

// 设置静态计算图模式
void Reducer::set_static_graph() {
    // 使用互斥锁保护，确保设置静态计算图的原子操作
    std::lock_guard<std::mutex> lock(mutex_);
    // 断言条件，若不满足则记录错误信息到 logger_
    REDUCER_CHECK(
        num_iterations_ == 0,
        logger_,
        "set_static_graph() should be called before training loop starts "
        "and after DistributedDataParallel is constructed.");
    // 设置 static_graph_ 为 true
    static_graph_ = true;
    // 当 static_graph_ 设置为 true 时，初始化本地使用映射
    // 并在第一次迭代中检测全局未使用的参数
    initialize_local_used_map();
}

namespace {

// Tensors 可能被合并到 buckets 中。buckets 必须包含相同类型和设备的 tensors，
// 因此可以用张量的类型标识符和设备组成的复合键来标识一个 bucket。
struct BucketKey {
    BucketKey(c10::ScalarType type, c10::Device device)
        : type(type), device(device) {}

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const*)
    const c10::ScalarType type;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const*)
    const c10::Device device;

    // 参见 torch/csrc/utils/hash.h 中的分发代码。
    static size_t hash(const BucketKey& key) {
        // 返回类型和设备的哈希值
        return c10::get_hash(key.type, key.device);
    }
};

// 判断两个 BucketKey 是否相等的运算符重载
inline bool operator==(const BucketKey& lhs, const BucketKey& rhs) {
    // 如果类型和设备相同，则相等
    return lhs.type == rhs.type && lhs.device == rhs.device;
}

} // namespace

// 根据大小计算桶的分配
std::tuple<std::vector<std::vector<size_t>>, std::vector<size_t>>
compute_bucket_assignment_by_size(
    const std::vector<at::Tensor>& tensors,
    const std::vector<size_t>& bucket_size_limits,
    const std::vector<bool>& expect_sparse_gradient,
    const std::vector<int64_t>& tensor_indices,
    // 省略部分参数说明
    const std::optional<std::weak_ptr<c10d::Logger>>& logger) {
    // 如果 expect_sparse_gradient 没有指定或者它的元素数量与张量向量相同，则通过
    // 内部断言进行检查。
    TORCH_INTERNAL_ASSERT(
        expect_sparse_gradient.empty() ||
        (tensors.size() == expect_sparse_gradient.size()));
    // 张量向量不能为空
    TORCH_INTERNAL_ASSERT(!tensors.empty());
    // 存储桶索引和它们的大小，因为稍后我们会按最小张量索引对结果索引进行排序，
    // 并且希望保持大小的一致性。
    std::vector<std::tuple<std::vector<size_t>, size_t>> result;
    // 稀疏张量单独放在一个桶中，因此它们没有强制的大小限制。
    size_t kNoSizeLimit = 0;
    result.reserve(tensors.size());

    // 按张量类型和设备保留大小限制向量的迭代器
    // 这样做是为了可以使用每种类型的连续桶限制。
    std::unordered_map<
        BucketKey,
        std::vector<size_t>::const_iterator,
        c10::hash<BucketKey>>
        bucket_size_limit_iterators;

    // 按张量类型和设备保留索引向量和大小累加器
    std::unordered_map<BucketKey, BucketAccumulator, c10::hash<BucketKey>>
        buckets;

    for (const auto i : c10::irange(tensors.size())) {
        const auto& tensor = tensors[i];
        auto msg = std::string("No support for sparse tensors.");
        // 如果存在 logger 值，则使用 REDUCER_CHECK 进行检查，否则使用 TORCH_CHECK
        if (logger.has_value()) {
            REDUCER_CHECK(!tensor.is_sparse(), logger.value(), msg);
        } else {
            TORCH_CHECK(!tensor.is_sparse(), msg);
        }

        // 如果 tensor_indices 为空，则张量 tensors[i] 的索引为 i，否则使用 tensor_indices[i]
        auto tensor_index = i;
        if (!tensor_indices.empty()) {
            tensor_index = tensor_indices[i];
        }
        // 如果我们期望为该张量产生稀疏梯度，则它不能与其他梯度分组在一起，而是单独一个桶。
        if (!expect_sparse_gradient.empty() &&
            expect_sparse_gradient[tensor_index]) {
            result.emplace_back(std::vector<size_t>({tensor_index}), kNoSizeLimit);
            continue;
        }

        auto key = BucketKey(tensor.scalar_type(), tensor.device());
        auto& bucket = buckets[key];
        bucket.indices.push_back(tensor_index);
        bucket.size += tensor.numel() * tensor.element_size();

        // 如果需要，初始化桶大小限制迭代器
        if (bucket_size_limit_iterators.count(key) == 0) {
            bucket_size_limit_iterators[key] = bucket_size_limits.begin();
        }

        auto& bucket_size_limit_iterator = bucket_size_limit_iterators[key];
        const auto bucket_size_limit = *bucket_size_limit_iterator;
        bucket.size_limit = bucket_size_limit;
    }
    // 检查当前桶的大小是否超过了设定的桶大小限制
    if (bucket.size >= bucket_size_limit) {
      // 如果超过了限制，则将当前桶的索引和大小限制加入结果列表中，并重置桶累加器
      result.emplace_back(std::move(bucket.indices), bucket.size_limit);
      bucket = BucketAccumulator();

      // 获取下一个类型/设备的桶大小限制
      auto next = bucket_size_limit_iterator + 1;
      // 如果还有下一个限制值，则更新迭代器
      if (next != bucket_size_limits.end()) {
        bucket_size_limit_iterator = next;
      }
    }
  }

  // 将剩余的非空桶加入结果列表中
  for (auto& it : buckets) {
    auto& bucket = it.second;
    if (!bucket.indices.empty()) {
      result.emplace_back(std::move(bucket.indices), bucket.size_limit);
    }
  }

  // 如果 tensor_indices 为空，按最小张量索引对结果进行排序
  // 否则，假定张量的顺序已经符合梯度准备顺序，无需排序
  if (tensor_indices.empty()) {
    std::sort(
        result.begin(),
        result.end(),
        [](const std::tuple<std::vector<size_t>, size_t>& a,
           const std::tuple<std::vector<size_t>, size_t>& b) {
          auto indices_a = std::get<0>(a);
          auto indices_b = std::get<0>(b);
          // 比较两个桶的最小张量索引，以确定排序顺序
          const auto amin =
              std::min_element(indices_a.begin(), indices_a.end());
          const auto bmin =
              std::min_element(indices_b.begin(), indices_b.end());
          return *amin < *bmin;
        });
  }

  // 将桶索引和大小限制分别存入两个向量中，并作为元组返回
  std::vector<std::vector<size_t>> bucket_indices;
  bucket_indices.reserve(result.size());
  std::vector<size_t> per_bucket_size_limits;
  per_bucket_size_limits.reserve(result.size());
  for (const auto& bucket_indices_with_size : result) {
    bucket_indices.emplace_back(std::get<0>(bucket_indices_with_size));
    per_bucket_size_limits.emplace_back(std::get<1>(bucket_indices_with_size));
  }
  return std::make_tuple(bucket_indices, per_bucket_size_limits);
// 验证模型副本中对应参数在所有进程中具有相同的大小和步幅。
void verify_params_across_processes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,  // 使用的进程组
    const std::vector<at::Tensor>& params,  // 待验证的参数张量列表
    const std::optional<std::weak_ptr<c10d::Logger>>& logger) {  // 可选的日志记录器

  // 首先验证参数数量，避免不一致的输入导致广播崩溃。
  // 参见 https://github.com/pytorch/pytorch/issues/73547

  at::TensorOptions param_size_options;
  param_size_options = param_size_options.dtype(at::kLong);  // 设置参数尺寸选项的数据类型为长整型
  param_size_options = param_size_options.device(params[0].device());  // 将设备选项设置为第一个参数张量的设备

  // 注意：由于 https://github.com/pytorch/pytorch/issues/74114，不使用张量构建 API
  at::Tensor param_size_tensor =
      at::tensor({static_cast<int64_t>(params.size())}, param_size_options);  // 创建参数数量的张量

  // 执行所有进程的全局收集并验证参数大小
  std::vector<std::vector<at::Tensor>> param_size_output_tensors;
  param_size_output_tensors.emplace_back();
  auto world_size = process_group->getSize();  // 获取进程组的大小
  for (C10_UNUSED const auto i : c10::irange(world_size)) {
    param_size_output_tensors.front().emplace_back(
        at::empty_like(param_size_tensor));  // 在输出张量列表中为每个进程创建空张量
  }

  std::vector<at::Tensor> param_size_vec{param_size_tensor};
  process_group->allgather(param_size_output_tensors, param_size_vec)->wait();  // 执行全局收集操作
  auto result_size_tensors = param_size_output_tensors.front();  // 获取结果张量列表
  for (const auto i : c10::irange(world_size)) {
    auto param_size_for_rank = result_size_tensors[i][0].item<int>();  // 获取每个进程的参数数量
    TORCH_CHECK(
        static_cast<size_t>(param_size_for_rank) == params.size(),  // 检查参数数量是否一致
        c10::str(
            "DDP expects same model across all ranks, but Rank ",
            process_group->getRank(),
            " has ",
            params.size(),
            " params, while rank ",
            i,
            " has inconsistent ",
            param_size_for_rank,
            " params."));
  }

  // 继续进行参数形状的验证
  size_t i = 0;
  for (const auto& t : params) {
    i += 2 * t.dim();  // 计算所有参数张量维度和步幅的总数
  }
  at::TensorOptions options;
  options = options.dtype(at::kLong);  // 设置选项的数据类型为长整型
  auto metadata = at::empty({static_cast<long>(i)}, options);  // 创建存储参数元数据的张量

  // 技术上，进程 0 是广播源，因此只有进程 0 需要填充元数据。但保持进程间的工作对齐也没有坏处。
  auto metadata_accessor = metadata.accessor<int64_t, 1>();  // 获取元数据的访问器
  i = 0;
  for (const auto& t : params) {
    for (const auto& sz : t.sizes()) {
      metadata_accessor[static_cast<int64_t>(i++)] = sz;  // 填充参数尺寸
    }
    for (const auto& str : t.strides()) {
      metadata_accessor[static_cast<int64_t>(i++)] = str;  // 填充参数步幅
    }
  }
}

auto metadata_dev = metadata.clone().to(params[0].device());
// 将元数据克隆并转移到指定设备上
std::vector<at::Tensor> vec{metadata_dev};
// 创建包含单个元数据张量的向量
process_group->broadcast(vec)->wait();
// 使用进程组广播元数据向量，并等待广播完成

// Technically, process 0 doesn't need to double-check metadata, because it
// was the source.  But no harm keeping work aligned.
// 在技术上，进程 0 不需要再次检查元数据，因为它是源头。但保持工作一致也无害。
auto control = at::empty({static_cast<long>(i)}, options);
// 创建一个空张量作为控制张量，使用与元数据相同的选项
control.copy_(metadata_dev, /*non_blocking=*/false);
// 将元数据复制到控制张量中，非异步方式
auto control_accessor = control.accessor<int64_t, 1>();
// 创建控制张量的访问器，用于访问控制张量的元素
i = 0;
// 初始化计数器 i 为 0
for (const auto p : c10::irange(params.size())) {
  const auto& t = params[p];
  // 迭代处理参数列表中的每个参数 t
  for (const auto& sz : t.sizes()) {
    // 迭代处理参数 t 的大小
    auto msg = c10::str(
        "[",
        process_group->getRank(),
        "]: params[",
        p,
        "] in this process",
        " with sizes ",
        t.sizes(),
        " appears not to match sizes of the same param in process 0.");
    // 构造消息字符串，指示参数在当前进程中的大小与进程 0 中的相同参数的大小是否匹配
    if (logger.has_value()) {
      REDUCER_CHECK(sz == control_accessor[i++], logger.value(), msg)
    } else {
      TORCH_CHECK(sz == control_accessor[i++], msg)
    }
    // 如果存在日志记录器，则使用 REDUCER_CHECK 进行大小检查；否则使用 TORCH_CHECK
  }
  for (const auto& str : t.strides()) {
    // 迭代处理参数 t 的步长
    auto msg = c10::str(
        "params[",
        p,
        "] in this process",
        " with sizes ",
        t.sizes(),
        " appears not to match strides of the same param in process 0.");
    // 构造消息字符串，指示参数在当前进程中的大小与进程 0 中的相同参数的步长是否匹配
    if (logger.has_value()) {
      REDUCER_CHECK(str == control_accessor[i++], logger.value(), msg)
    } else {
      TORCH_CHECK(str == control_accessor[i++], msg)
    }
    // 如果存在日志记录器，则使用 REDUCER_CHECK 进行步长检查；否则使用 TORCH_CHECK
  }
}
}

// 删除当前 Reducer 对象注册的所有自动求导钩子
void Reducer::remove_autograd_hooks() {
  // 遍历当前对象的钩子列表，逐个删除对应变量的钩子
  // 这是为了使得 DDP 失败后能够恢复。否则，多个 Reducer 实例（来自恢复过程）
  // 会向原始模型添加它们的钩子，这些钩子将尝试调用已删除的 Reducer 对象上的方法。
  for (auto& hook : hooks_) {
    auto& key = hook.first;                 // 获取钩子的键（变量）
    auto& grad_accumulator = hook.second;   // 获取钩子的梯度累加器

    // 断言删除钩子成功，如果失败则抛出异常
    TORCH_INTERNAL_ASSERT(
        grad_accumulator->del_post_hook(key),
        "Reducer attempts to delete a non-existing hook.");
  }
  hooks_.clear();   // 清空钩子列表
}

// 检查是否已经完成最终的归约操作
void Reducer::check_finalized() {
  std::lock_guard<std::mutex> lock(mutex_);  // 加锁，确保线程安全
  ensure_prior_reduction_finished();         // 确保之前的归约操作已完成
}

// 更新进程组对象
void Reducer::update_process_group(
    c10::intrusive_ptr<c10d::ProcessGroup> new_process_group) {
  std::lock_guard<std::mutex> lock(mutex_);  // 加锁，确保线程安全
  process_group_ = std::move(new_process_group);  // 更新进程组对象
}

// 重置 Reducer 对象的状态
void Reducer::reset_state() {
  std::lock_guard<std::mutex> lock(mutex_);  // 加锁，确保线程安全

  // 强制重建存储桶
  has_rebuilt_bucket_ = false;
  rebuilt_params_.clear();
  rebuilt_param_indices_.clear();

  // 确保即使之前的反向传播失败，前向传播仍可运行
  expect_autograd_hooks_ = false;
  require_finalize_ = false;
  first_autograd_hook_called_ = false;

  // 重置全局归约除法因子，在 DDP join 模式下可能会改变
  div_factor_ = kUnsetDivFactor;

  // 重置未使用参数的计数
  // 参见 "Note [local_used_map_ -> local_used_map_dev copying]"
  if (find_unused_parameters_) {
    local_used_map_.zero_();
    local_used_map_reduced_ = false;
  }
}

} // namespace c10d
```