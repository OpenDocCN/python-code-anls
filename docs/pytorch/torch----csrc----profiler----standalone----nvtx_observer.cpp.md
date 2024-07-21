# `.\pytorch\torch\csrc\profiler\standalone\nvtx_observer.cpp`

```
// 包含 Torch 的 NVTX 观察器头文件
#include <torch/csrc/profiler/standalone/nvtx_observer.h>

// 包含 Torch 的基本存根和实用程序头文件
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

// 命名空间 torch::profiler::impl 下的结构定义
namespace torch::profiler::impl {

// NVTXThreadLocalState 结构体继承自 ProfilerStateBase
struct NVTXThreadLocalState : ProfilerStateBase {
  explicit NVTXThreadLocalState(const ProfilerConfig& config)
      : ProfilerStateBase(config) {
    // 在这个上下文中只有 report_input_shapes 有意义
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  ~NVTXThreadLocalState() override = default;

  // 返回当前的性能分析器类型为 NVTX
  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::NVTX;
  }

  // 不执行任何内存使用报告的操作
  void reportMemoryUsage(void*, int64_t, size_t, size_t, c10::Device) override {
  }

  // 获取当前线程的 NVTXThreadLocalState 实例
  static NVTXThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::NVTX);
    return static_cast<NVTXThreadLocalState*>(tls);
  }

  // 从输入的张量中获取操作 ID
  std::pair<at::RecordFunctionHandle, int> getOpIdFromInput(
      const at::Tensor& tensor);

  // 设置生产者张量映射关系
  void setProducerTensorMap(
      at::TensorImpl* tensor,
      at::RecordFunctionHandle op_id,
      int output_nr) {
    producer_tensor_map_[(void*)tensor] =
        std::pair<at::RecordFunctionHandle, int>{op_id, output_nr};
  }

 protected:
  // 映射输出张量地址到唯一操作 ID 和输出索引的哈希表
  std::unordered_map<void*, std::pair<at::RecordFunctionHandle, int>>
      producer_tensor_map_;
};

// 从输入张量中获取操作 ID 的实现
std::pair<at::RecordFunctionHandle, int> NVTXThreadLocalState::getOpIdFromInput(
    const at::Tensor& tensor) {
  std::pair<at::RecordFunctionHandle, int> producer_op_pair(0, -1);
  if (tensor.defined()) {
    at::TensorImpl* ten_addr = tensor.unsafeGetTensorImpl();
    // 检查地址是否已在映射表中
    if (producer_tensor_map_.count((void*)ten_addr) > 0) {
      producer_op_pair = producer_tensor_map_[(void*)ten_addr];
    }
  }
  return producer_op_pair;
}

// 将 c10::List<c10::IValue> 中的操作 ID 列表展开为平面列表
static std::list<std::pair<at::RecordFunctionHandle, int>> flattenOpIdList(
    const c10::List<c10::IValue>& list) {
  std::list<std::pair<at::RecordFunctionHandle, int>> input_op_id_list;
  auto state_ptr = NVTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& input : list) {
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      auto producer_op_pair = state_ptr->getOpIdFromInput(tensor);
      input_op_id_list.push_back(producer_op_pair);
    }
  }
  return input_op_id_list;
}

// 获取输入张量的操作 ID 列表
static std::list<std::pair<at::RecordFunctionHandle, int>> getInputTensorOpIds(
    // 函数声明，接收一个 RecordFunction 对象作为参数，并返回一个 pair 列表
    const at::RecordFunction& fn) {
      // 创建一个包含初始值的 pair 对象，第一个元素是 RecordFunctionHandle 类型，第二个是整数类型
      std::pair<at::RecordFunctionHandle, int> undefined_op_pair(0, -1);
      // 创建一个空的 pair 列表，用于存储输入操作和它们的操作 ID
      std::list<std::pair<at::RecordFunctionHandle, int>> input_producer_ops_;
      // 获取当前线程的 NVTX 状态指针
      auto state_ptr = NVTXThreadLocalState::getTLS();
      // 断言确保 profiler 状态已设置
      TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
      // 迭代处理输入的每一个 IValue
      for (const c10::IValue& input_item : fn.inputs()) {
        // 检查当前输入是否为 Tensor
        if (input_item.isTensor()) {
          // 将 IValue 转换为 Tensor 类型
          const at::Tensor& tensor = input_item.toTensor();
          // 获取该 Tensor 的操作 ID，并将其与 Tensor 对应的 pair 放入列表末尾
          auto producer_pair = state_ptr->getOpIdFromInput(tensor);
          input_producer_ops_.push_back(producer_pair);
        } else {
          // 如果输入不是 Tensor，检查是否为列表类型
          if (input_item.isList()) {
            // 将列表类型的输入展平为操作 ID pair 列表
            std::list<std::pair<at::RecordFunctionHandle, int>> tmp_op_ids =
                flattenOpIdList(input_item.toList());
            // 如果展平后的列表不为空，将其连接到当前操作列表的末尾
            if (!tmp_op_ids.empty()) {
              input_producer_ops_.splice(input_producer_ops_.end(), tmp_op_ids);
            } else {
              // 否则，添加一个表示未定义操作的 pair 到操作列表
              input_producer_ops_.emplace_back(undefined_op_pair);
            }
          } else {
            // 如果既不是 Tensor 也不是列表，则添加一个表示未定义操作的 pair 到操作列表
            input_producer_ops_.emplace_back(undefined_op_pair);
          }
        }
      }
      // 返回收集的输入操作和它们的操作 ID 的列表
      return input_producer_ops_;
    }
} // 结束命名空间 torch::profiler::impl

static void updateOutputTensorTracker(const at::RecordFunction& fn) {
  // 初始化输出张量索引
  int output_nr = 0;
  // 获取当前线程的 NVTX 状态
  auto state_ptr = NVTXThreadLocalState::getTLS();
  // 断言确保状态指针不为空
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  // 遍历函数的输出
  for (const c10::IValue& s_tensor : fn.outputs()) {
    // 检查当前值是否为张量
    if (s_tensor.isTensor()) {
      // 将 IValue 转换为 Tensor
      const at::Tensor& tensor = s_tensor.toTensor();
      // 检查张量是否已定义
      if (tensor.defined()) {
        // 获取张量实现的地址
        auto ten_addr = tensor.unsafeGetTensorImpl();
        // 更新生产者张量映射到状态中
        state_ptr->setProducerTensorMap(ten_addr, fn.handle(), output_nr);
      }
    }
    // 增加输出张量索引
    output_nr++;
  }
}

template <bool report_input_shapes>
std::unique_ptr<at::ObserverContext> enterNVTX(const at::RecordFunction& fn) {
  // 检查是否获取到 NVTX 线程本地状态
  if (NVTXThreadLocalState::getTLS() != nullptr) {
    // 获取输入张量操作 ID
    auto input_op_ids = getInputTensorOpIds(fn);
    // 推入 NVTX 范围
    torch::profiler::impl::cudaStubs()->rangePush(
        // 获取 NVTX 字符串表示
        torch::profiler::impl::getNvtxStr(
            fn.name(),
            fn.seqNr(),
            // 如果需要报告输入形状，则获取输入尺寸
            report_input_shapes ? torch::profiler::impl::inputSizes(fn, true)
                                : std::vector<std::vector<int64_t>>(),
            fn.handle(),
            // 如果需要报告输入形状，则传入输入操作 ID 列表
            report_input_shapes
                ? input_op_ids
                : std::list<std::pair<at::RecordFunctionHandle, int>>())
            .c_str());
  }
  // 返回空指针
  return nullptr;
}

void pushNVTXCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  // 检查 CUDA 是否启用
  TORCH_CHECK(
      torch::profiler::impl::cudaStubs()->enabled(),
      "Can't use NVTX profiler - PyTorch was compiled without CUDA");

  // 推入线程本地调试信息，使用 NVTX 线程本地状态配置
  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<NVTXThreadLocalState>(config));

  // 获取当前线程的 NVTX 状态
  auto state_ptr = NVTXThreadLocalState::getTLS();
  // 断言确保状态指针不为空
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  // 添加线程本地回调函数
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          // 如果需要报告输入形状，则使用带 report_input_shapes 的 enterNVTX
          state_ptr->config().report_input_shapes
              ? &enterNVTX</*report_input_shapes=*/true>
              : &enterNVTX</*report_input_shapes=*/false>,
          [](const at::RecordFunction& fn, at::ObserverContext* ctx) {
            // 弹出 NVTX 范围
            torch::profiler::impl::cudaStubs()->rangePop();
            // 更新输出张量追踪器
            updateOutputTensorTracker(fn);
          })
          .needsInputs(config.report_input_shapes)
          .needsOutputs(config.report_input_shapes)
          .needsIds(true)
          .scopes(scopes));
  // 设置回调句柄到状态中
  state_ptr->setCallbackHandle(handle);
}
```