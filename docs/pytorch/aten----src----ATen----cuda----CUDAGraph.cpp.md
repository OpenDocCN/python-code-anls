# `.\pytorch\aten\src\ATen\cuda\CUDAGraph.cpp`

```py
/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: 为什么我们在PyTorch中需要图捕获和启动绑定？
 *    为什么它们不能存在于用户扩展中，例如？
 *
 * A1: 方便性。
 * A2: 为了确保在重放时的有效数字，一些本地CUDA操作（例如具有CPU状态的RNG操作）
 *     需要从捕获和重放绑定中得到协作（参见CUDAGeneratorImpl.h中的Note [CUDA图安全RNG状态]）。
 *
 *     我们不能期望用户了解这种协作。如果用户在扩展中简单地编写捕获绑定，
 *     他们可能不会正确地与本地操作交互。他们的图在重放时可能会产生无效的数字。
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * 描述了捕获过程中的内存管理。
 */

/**
 * 增加未决事件查询的原子整数，例如在NCCL看门狗中可能发生，
 * 以便在捕获开始前解决它们。注意，在默认捕获模式下，不允许在图捕获期间进行事件查询。
 */
void CUDAGraph::inc_pending_event_queries() {
  pending_event_queries++;
}

/**
 * 减少未决事件查询的原子整数。
 * 强制要求未决事件查询数大于0。
 */
void CUDAGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
    "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

/**
 * 返回当前未决事件查询的数量。
 */
int CUDAGraph::num_pending_event_queries() {
  return pending_event_queries;
}

/**
 * 构造函数：初始化CUDAGraph对象，使用当前CUDA流。
 * 注意，CUDA流可能不支持默认构造。
 */
CUDAGraph::CUDAGraph()
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
}

/**
 * 注册生成器状态
 */
void CUDAGraph::register_generator_state(
    // 将传入的 CUDAGeneratorState 对象使用 std::move() 移动到 captured_generator_states_ 字典中，并初始化对应值为 0
    captured_generator_states_[std::move(state)] = 0;
}

// 将给定的生成器状态注册到当前 CUDA 图中
void CUDAGraph::register_generator_state(const at::Generator& generator) {
  // 将给定的生成器状态转换为 CUDAGeneratorImpl 类型的智能指针
  c10::intrusive_ptr<CUDAGeneratorImpl> cuda_gen =
      dynamic_intrusive_pointer_cast<CUDAGeneratorImpl>(
          generator.getIntrusivePtr());
  // 将当前 CUDA 图注册到 CUDA 生成器实现中
  cuda_gen->register_graph(this);
}

// 开始捕获 CUDA 图
void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/, cudaStreamCaptureMode capture_mode) {
  // 检查当前 CUDA 图实例是否已经拥有捕获的图
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // 默认生成器始终注册
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  // 将当前 CUDA 图注册到生成器中
  gen->register_graph(this);

  // 遍历已捕获的生成器状态，执行捕获前操作
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  // 获取当前 CUDA 流
  auto stream = at::cuda::getCurrentCUDAStream();

  // 检查当前流是否为默认流
  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  // 设置捕获流和当前设备
  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

  // 生成当前捕获序列的 ID
  id_ = capture_sequence_id();

  // 如果 pool 的第一个值或第二个值不为零，则用户提供了共享的内存池
  if (pool.first != 0 || pool.second != 0) {
    // 只能有一个值不为零，用于标识内存池的来源
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    // 设置内存池 ID
    mempool_id_ = pool;
  } else {
    // 用户未要求共享内存池，使用当前捕获序列的 ID 作为内存池 ID 的第一个值
    mempool_id_ = {id_, 0};
  }

  // 在开始 CUDA 流捕获前，先调用 beginAllocateStreamToPool 防止缓存分配器中的无效事件记录问题
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](cudaStream_t stream) {
      cudaStreamCaptureStatus status;
      CaptureId_t stream_capture_id;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
      // 检查流是否处于活跃的捕获状态，并且捕获 ID 与当前捕获序列 ID 相符
      return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive && stream_capture_id == capture_id_;
  });

  // 此时，任何 NCCL 看门狗应该知道我们处于捕获模式，不应再排队新的工作
  // 但仍需等待尚未清理的现有工作
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
    // 通过休眠当前线程来等待一段时间，以允许后续操作完成
    std::this_thread::sleep_for(
      std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // 使用 cudaStreamCaptureModeGlobal 是最保守的选项，用于在捕获期间阻止可能不安全的 CUDA API 调用。
  // 参考 Nvidia CUDA API 文档：https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  // 获取流的捕获信息，包括捕获状态和捕获 ID
  AT_CUDA_CHECK(cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  // 确保流的捕获状态处于活动状态
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  // 内部断言，确保捕获的 ID 大于 0
  TORCH_INTERNAL_ASSERT(id_ > 0);
void CUDAGraph::capture_end() {
  // 获取当前 CUDA 流
  auto stream = at::cuda::getCurrentCUDAStream();

  // 检查当前流是否与开始捕获时的流一致
  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  // 结束 CUDA 流的捕获，生成 CUDA 图
  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  // 结束向内存池分配内存
  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  // 检查生成的图是否有效
  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  // 在典型的图使用中，一些张量（例如用于图 IO 的张量）在重播之间未被释放。
  // 如果 PyTorch 使用 CUDA 11.4+ 工具包编译和运行，有可能分配器后端是 cudaMallocAsync。
  // cudaMallocAsync 通常是图安全的，但如果在重播之间一些张量未被释放，图的内部管理要求我们使用 cudaGraphInstantiateFlagAutoFreeOnLaunch 实例化。
  // 参见 cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
    // 尾部的 NULL, NULL, 0 参数是 CUDA 驱动程序人员建议的，
    // 他们希望通过这些参数移动向前不再报告错误消息（他们更喜欢返回值，或者在捕获的 API 调用内部出错时报告错误）。
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  } else {
    // 使用自动释放标志实例化 CUDA 图
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif

  // 标记已生成图执行器
  has_graph_exec_ = true;

  // 遍历捕获的生成器状态，执行捕获的结尾
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  // 获取 CUDA 图节点数量
  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, NULL, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      // 提示 CUDA 图为空的情况，通常表示图试图在错误的设备或流上捕获。
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  // 检查是否设置了调试路径
  if (!_cuda_graphs_debug) {
    // 现在我们已经将 graph_ 实例化为 graph_exec_，不再需要 graph_。
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    has_graph_ = false;
  } else {
    // 调试路径已设置，不会在调用 debug_dump 前释放 graph_。
    TORCH_WARN("DEBUG: TORCH_CUDAGRAPHS_DEBUG_PATH detected. graph_ will not be freed until debug_dump is called.");
  }
}
// 用于重新执行捕获的 CUDA 图形。在调用前需要确保已成功捕获图形。
void CUDAGraph::replay() {
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  // 捕获流的设备上下文
  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // 对捕获的每个生成器状态执行回放序言
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  
  // 在任意流中重新执行 graph_exec_
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  // 获取 CUDA 驱动程序版本号
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  
  // 如果 CUDA 版本低于 11.4，进行以下工作
  if (version < 11040) {
    // 解决 libcuda.so 中的一个 bug，该 bug 导致连续回放特定拓扑结构的图形时可能会出现损坏
    // （例如内核省略、内部同步忽略）。CUDA 11.4+ 版本已修复此 bug。
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

// 启用调试模式
void CUDAGraph::enable_debug_mode() {
  _cuda_graphs_debug = true;
}

// 打印调试信息到指定路径
void CUDAGraph::debug_dump(const std::string& debug_path) {
  // 检查 CUDA 版本是否支持 CUDA 图形调试或者 ROCM
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11030)|| defined(USE_ROCM)
  if (_cuda_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      // 打印 CUDA 图形的详细信息到文件
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), cudaGraphDebugDotFlagsVerbose));
      // 销毁 CUDA 图形对象
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with torch._C._cuda_enable_graphs_debug_mode");
  }
#else
  // CUDA 图形仅支持 CUDA >= 11.3 或 ROCM >= 5.6
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

// 重置 CUDA 图形对象状态
void CUDAGraph::reset() {
  // 通常情况下应该抛出异常，而不是打印警告信息，
  // 但析构函数调用 reset() 时，至少一个 CI 构建会拒绝编译抛出异常的析构函数。
  //
  // 我们选择在 C++ 析构函数中调用 reset()，在调用失败时打印警告信息而不是抛出异常，
  // 这是我们做出的妥协。
  //
  // 如果捕获过程（包括开始捕获、捕获和结束捕获）的任何阶段失败，
  // 则当前的 CUDA 图形、生成器和分配器可能会处于各种不正常的状态。
  // 如果用户在脚本中捕获了异常，或者在 REPL 或 Jupyter 笔记本中运行，
  // 我们无法轻松地在 reset() 中修复所有这些潜在的错误状态。
  if (has_graph_ || has_graph_exec_) {
    // 如果 notifyCaptureDestroy 可能抛出异常，我们应该如何处理？
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  // 如果存在图形对象，销毁它并标记为未拥有图形
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  // 如果存在图形执行对象，销毁它并标记为未拥有图形执行对象
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
// CUDAGraph 类的析构函数，负责释放资源和重置状态
CUDAGraph::~CUDAGraph() {
  // 遍历 captured_generator_states_ 容器中的每对键值对
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    // 调用 generator_state 对象的 unregister_graph 方法，解除对当前图的注册
    generator_state->unregister_graph(this);
  }
  // 调用 reset 方法，重置 CUDAGraph 对象的状态
  reset();
}

// 在 CUDA 命名空间中结束 at::cuda 命名空间的声明
} // namespace at::cuda
```