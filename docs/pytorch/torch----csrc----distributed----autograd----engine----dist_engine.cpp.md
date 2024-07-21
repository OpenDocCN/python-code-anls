# `.\pytorch\torch\csrc\distributed\autograd\engine\dist_engine.cpp`

```
// 包含 C++ 标准库中的队列实现
#include <queue>

// 包含 PyTorch ATen 库中的并行执行支持
#include <ATen/Parallel.h>

// 包含 PyTorch C10 库中的事件定义
#include <c10/core/Event.h>

// 包含 PyTorch C10 工具库中的死锁检测工具
#include <c10/util/DeadlockDetection.h>

// 包含 PyTorch C10 工具库中的范围迭代器支持
#include <c10/util/irange.h>

// 包含 PyTorch 自动求导模块中的累积梯度函数定义
#include <torch/csrc/autograd/functions/accumulate_grad.h>

// 包含 PyTorch 自动求导模块中的输入缓冲区定义
#include <torch/csrc/autograd/input_buffer.h>

// 包含 PyTorch 分布式自动求导模块中的上下文容器定义
#include <torch/csrc/distributed/autograd/context/container.h>

// 包含 PyTorch 分布式自动求导模块中的分布式引擎定义
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

// 定义命名空间 torch 中的 distributed 自动求导模块
namespace torch {
namespace distributed {
namespace autograd {

// 使用语句别名定义，引用自动求导模块中的累积梯度函数
using torch::autograd::AccumulateGrad;

// 使用语句别名定义，引用自动求导模块中的边缘列表
using torch::autograd::edge_list;

// 使用语句别名定义，引用自动求导模块中的求导引擎
using torch::autograd::Engine;

// 使用语句别名定义，引用自动求导模块中的图根节点
using torch::autograd::GraphRoot;

// 使用语句别名定义，引用自动求导模块中的图任务
using torch::autograd::GraphTask;

// 使用语句别名定义，引用自动求导模块中的图任务保护器
using torch::autograd::GraphTaskGuard;

// 使用语句别名定义，引用自动求导模块中的输入缓冲区
using torch::autograd::InputBuffer;

// 使用语句别名定义，引用自动求导模块中的节点
using torch::autograd::Node;

// 使用语句别名定义，引用自动求导模块中的节点任务
using torch::autograd::NodeTask;

// 使用语句别名定义，引用自动求导模块中的准备队列
using torch::autograd::ReadyQueue;

// 使用语句别名定义，引用自动求导模块中的输出验证函数
using torch::autograd::validate_outputs;

// 使用语句别名定义，引用自动求导模块中的变量列表
using torch::autograd::variable_list;

// 静态常量，指定当前反向传播次数的字符串常量
static constexpr const char* kNumBackwardPasses = "num_current_backward_passes";

// 静态常量，指定当前自动求导上下文数的字符串常量
static constexpr const char* kNumAutogradContexts = "num_autograd_contexts";

// DistAccumulateGradCaptureHook 类，继承自图任务执行信息捕获的梯度捕获钩子
class DistAccumulateGradCaptureHook
    : public GraphTask::ExecInfo::Capture::GradCaptureHook {
 public:
  // 构造函数，初始化 DistAccumulateGradCaptureHook 实例
  DistAccumulateGradCaptureHook(
      std::shared_ptr<AccumulateGrad> accumulateGrad,
      ContextPtr autogradContext)
      : accumulateGrad_(std::move(accumulateGrad)),
        autogradContext_(std::move(autogradContext)) {}

  // 重载运算符，处理梯度捕获逻辑
  at::Tensor operator()(const at::Tensor& grad) override {
    // 线程本地的分布式自动求导上下文保护
    ThreadLocalDistAutogradContext contextGuard{ContextPtr(autogradContext_)};
    
    // 初始化输入梯度列表，包含传入的梯度张量
    variable_list inputGrads = {grad};

    // 遍历所有前置钩子，修改输入梯度
    for (const auto& hook : accumulateGrad_->pre_hooks()) {
      inputGrads = (*hook)(inputGrads);
    }

    // 检查输入梯度是否已定义，如果未定义则继续执行
    if (inputGrads[0].defined()) {
      // 在此时刻，'inputGrads[0]' 具有三个内部引用：
      //   1. 函数内部的 'inputGrads[0]'。
      //   2. 在本地引擎的调用点上的 'graph_task->captured_vars_'。
      //   3. 作为函数节点输入缓冲区 'InputBuffer& inputs' 的一部分。
      autogradContext_->accumulateGrad(
          accumulateGrad_->variable, inputGrads[0], 3 /* num_expected_refs */);
    }

    // 初始化空的输出变量列表
    const variable_list kEmptyOutput;

    // 遍历所有后置钩子，执行钩子函数
    for (const auto& hook : accumulateGrad_->post_hooks()) {
      (*hook)(kEmptyOutput, inputGrads);
    }

    // 返回处理后的输入梯度
    return inputGrads[0];
  }

 private:
  std::shared_ptr<AccumulateGrad> accumulateGrad_;
  ContextPtr autogradContext_;
};

// DistEngine 类的成员函数，全局 CPU 线程函数
void DistEngine::globalCpuThread(
    const std::shared_ptr<ReadyQueue>& ready_queue) {
  // 无限循环，处理全局 CPU 线程任务
  while (true) {


这段代码中，详细注释了各个头文件的包含目的以及类和函数的定义和作用。
    // 从就绪队列中取出一个任务
    NodeTask task = ready_queue->pop();
    // 如果任务是关闭线程的任务
    if (task.isShutdownTask_) {
      // 需要关闭这个线程。
      C10_LOG_API_USAGE_ONCE("torch.autograd.thread_shutdown");
      break;
    }

    // 获取与任务关联的图任务（GraphTask）
    auto graphTask = task.base_.lock();
    // 如果图任务已经过期（nullptr），忽略并继续处理下一个任务
    if (graphTask == nullptr) {
      // GraphTask 已经过期，忽略并继续处理。
      continue;
    }

    // 在JIT线程上启动执行
    at::launch([this,
                graphTask,
                graphRoot = task.fn_,
                variables =
                    InputBuffer::variables(std::move(task.inputs_))]() mutable {
      // 创建变量列表的输入缓冲区
      InputBuffer inputs(variables.size());
      // 将变量移动到输入缓冲区中
      for (const auto i : c10::irange(variables.size())) {
        inputs.add(i, std::move(variables[i]), c10::nullopt, c10::nullopt);
      }
      // 执行图任务，直到就绪队列为空
      execute_graph_task_until_ready_queue_empty(
          /*node_task*/ NodeTask(graphTask, graphRoot, std::move(inputs)),
          /*incrementOutstandingTasks*/ false);
    });
}

DistEngine::DistEngine()
    : initializedContextIds_(),
      engine_(Engine::get_default_engine()),
      global_cpu_ready_queue_(std::make_shared<ReadyQueue>()),
      global_cpu_thread_(
          &DistEngine::globalCpuThread,
          this,
          global_cpu_ready_queue_) {
  // Note [GPU to CPU continuations]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Initialize a single CPU thread to execute continuations from GPU
  // tasks. The multithreaded structure for the distributed engine works
  // well only for CPU tasks. If we have an order of tasks like
  // CPU->GPU->CPU, distributed autograd has no thread to execute the last
  // CPU task on. To fix this, we introduce a global CPU thread to handle
  // such situations and it will be responsible for executing these CPU
  // tasks. The CPU thread has its own ready_queue which is used as the
  // cpu_ready_queue for all GraphTasks for DistEngine. This ensures all GPU
  // to CPU continuations are enqueued on this thread. The global CPU thread
  // simply dequeues tasks from the global queue and calls
  // "execute_graph_task_until_ready_queue_empty" on a JIT thread to execute the
  // appropriate task.
  // 初始化一个单独的 CPU 线程来执行来自 GPU 任务的延续任务。
  // 分布式引擎的多线程结构仅适用于 CPU 任务。如果任务顺序为 CPU->GPU->CPU，
  // 分布式自动求导没有线程来执行最后的 CPU 任务。因此引入全局 CPU 线程来处理这些情况，
  // 并负责执行这些 CPU 任务。CPU 线程有自己的 ready_queue，作为 DistEngine 的所有 GraphTasks 的 cpu_ready_queue 使用。
  // 这确保所有 GPU 到 CPU 的延续任务都排队在此线程上。全局 CPU 线程简单地从全局队列中出队任务，
  // 并在 JIT 线程上调用 "execute_graph_task_until_ready_queue_empty" 来执行适当的任务。
  global_cpu_thread_.detach();
}

DistEngine::~DistEngine() {
  // Ensure we shutdown the CPU thread.
  TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP();
  global_cpu_ready_queue_->pushShutdownTask();
  global_cpu_thread_.join();
  // 确保关闭 CPU 线程。
  // 使用 TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP() 确保在没有 Python 依赖的情况下不持有 GIL（全局解释器锁）。
  // 将 shutdown 任务推入 global_cpu_ready_queue_，然后等待 global_cpu_thread_ 线程结束。
}

DistEngine& DistEngine::getInstance() {
  // Leaky singleton to avoid module destructor race.
  static DistEngine* engine = new DistEngine();
  return *engine;
  // 漏洞单例以避免模块析构竞争。
}

void DistEngine::validateRootsAndRetrieveEdges(
    const variable_list& roots,
    edge_list& rootEdges,
    variable_list& grads) {
  TORCH_CHECK(!roots.empty(), "No tensors provided for gradient computation.");
  TORCH_INTERNAL_ASSERT(rootEdges.empty());
  TORCH_INTERNAL_ASSERT(grads.empty());

  // Verify roots are all scalar and require gradients.
  for (const auto& root : roots) {
    TORCH_CHECK(root.requires_grad(), "requires_grad not set on root");
    TORCH_CHECK(
        root.numel() == 1,
        root.name(),
        " is not a scalar, all roots need to be scalar");
    TORCH_CHECK(
        root.grad_fn(),
        root.name(),
        " does not have a valid gradient function.");

    // Compute the root edges and generate the appropriate gradients.
    rootEdges.push_back(torch::autograd::impl::gradient_edge(root));
    grads.push_back(at::ones_like(root, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  }

  // Validate rootEdges and grads.
  validate_outputs(
      rootEdges, grads, [](const std::string& msg) { return msg; });
  // 验证 rootEdges 和 grads。
}

void DistEngine::computeDependencies(
    const ContextPtr& autogradContext,
    const edge_list& rootEdges,
    const variable_list& grads,
    const std::shared_ptr<Node>& graphRoot,
    edge_list& outputEdges,
    bool retainGraph) {
  TORCH_INTERNAL_ASSERT(graphRoot, "graphRoot is null!");

  // Store root nodes so we can traverse through the graph later
  // e.g., for get_current_graph_task_execution_order
  // 创建一个临时的 Node 指针向量 temp_roots，用于存储根节点，以便后续遍历图
  c10::SmallVector<Node*, 4> temp_roots{rootEdges.size()};
  // 遍历 rootEdges，将每个根节点的 function 指针存入 temp_roots
  for (const auto i : c10::irange(rootEdges.size())) {
    temp_roots[i] = rootEdges[i].function.get();
  }

  // Build the graph task and graph root.
  // NOTE: we don't need to build and pass a cpu_ready_queue to GraphTask
  // as we use execute_graph_task_until_ready_queue_empty, which will build
  // a separate ReadyQueue for each call.
  // 创建一个 GraphTask 对象 graphTask，用于管理和执行图任务
  auto graphTask = std::make_shared<GraphTask>(
      /* keep_graph */ retainGraph,  // 是否保持计算图
      /* create_graph */ false,      // 是否创建计算图
      /* depth */ 0,                 // 图的深度，这里设置为 0
      /* cpu_ready_queue */ global_cpu_ready_queue_,  // 全局 CPU 就绪队列
      /* graph_roots */ temp_roots,  // 图的根节点集合
      /* exit_on_error */ true);     // 出错时是否退出

  // Run BFS to traverse the graph locally. The roots of the graph are
  // GraphRoot and all send functions for this autograd context.
  // 使用 BFS 遍历本地计算图，将 graphRoot 和当前 autograd 上下文的所有 send 函数作为根节点
  std::unordered_set<Node*> seen;
  std::queue<Node*> queue;
  queue.push(static_cast<Node*>(graphRoot.get()));

  auto sendFunctions = autogradContext->sendFunctions();

  // Add all the send functions to the queue as roots.
  // 将所有 send 函数添加到队列中作为根节点
  for (const auto& mapEntry : sendFunctions) {
    // 递增 GraphTask 的 outstanding_tasks_，用于等待本地 autograd 引擎的所有任务
    graphTask->outstanding_tasks_++;
    queue.push(mapEntry.second.get());
  }

  bool will_use_accelerator = false;

  edge_list recvBackwardEdges;
  // Traverse the graph.
  auto& dependencies = graphTask->dependencies_;
  // 开始遍历图
  while (!queue.empty()) {
    auto fn = queue.front();
    queue.pop();

    // 检查是否将使用加速器
    if (!will_use_accelerator) {
      will_use_accelerator = fn->stream().has_value();
    }
  // 遍历每个函数的后继边
  for (const auto& edge : fn->next_edges()) {
    // 如果存在下一个函数
    if (auto nextFn = edge.function.get()) {
      // 增加依赖计数
      dependencies[nextFn] += 1;
      // 尝试将函数插入到集合中，记录插入是否成功
      const bool wasInserted = seen.insert(nextFn).second;
      // 如果函数是第一次见到
      if (wasInserted) {
        // 将函数加入执行队列
        queue.push(nextFn);

        // 如果下一个函数没有后继边
        if (nextFn->next_edges().empty()) {
          // 断言下一个函数是 AccumulateGrad 或 RecvRpcBackward
          TORCH_INTERNAL_ASSERT(
              dynamic_cast<AccumulateGrad*>(nextFn) ||
              dynamic_cast<RecvRpcBackward*>(nextFn));
          
          // 如果是 RecvRpcBackward，将边记录到 recvBackwardEdges
          if (dynamic_cast<RecvRpcBackward*>(nextFn)) {
            recvBackwardEdges.emplace_back(edge);
          }
          
          // 将边记录到 outputEdges，表示这个函数需要被执行
          outputEdges.emplace_back(edge);
        }
      }
    }
  }



  // 如果将使用加速器
  if (will_use_accelerator) {
    // 将当前 CUDA/ROCM 设备的流收集起来，以便在后处理时与 leaf_streams 同步
    graphTask->stash_current_streams();
  }

  // 现在计算需要执行的函数。算法如下：
  // 1. 创建一个虚拟的 GraphRoot，指向此上下文中所有的 'send' 函数和原始的 graphRoot。
  //    使用 outputEdges 和虚拟的 GraphRoot 运行 'init_to_execute'，确保标记
  //    可达的特定 'send' 函数所需要的函数，不一定需要从提供的根节点开始。
  // 2. 对于所有指向 'RecvRpcBackward' 的 outputEdges，将这些函数标记为需要执行。
  //    原因是 'init_to_execute' 会将这些标记为不需要执行。但是 'RecvRpcBackward'
  //    在图中是一个唯一的叶子节点，我们使用它来精确计算需要执行的函数，但与 AccumulateGrad
  //    不同，我们确实需要执行这个函数。
  if (!outputEdges.empty()) {
    // 从所有的 'send' 函数和原始的 graphRoot 开始计算 'needed execution'
    edge_list edges;
    // 对于 sendFunctions 中的每一个条目，创建一个边并添加到 edges 中
    for (const auto& mapEntry : sendFunctions) {
      edges.emplace_back(mapEntry.second, 0);
    }

    // 将原始的 graphRoot 添加为一个边
    edges.emplace_back(graphRoot, 0);

    // 创建一个虚拟的 GraphRoot，并使用它运行 init_to_execute 方法
    GraphRoot dummyRoot(edges, {});
    graphTask->init_to_execute(
        dummyRoot, outputEdges, /*accumulate_grad=*/false, /*min_topo_nr=*/0);

    // 遍历 graphTask 中的每一个执行信息
    for (auto& mapEntry : graphTask->exec_info_) {
      auto& execInfo = mapEntry.second;

      // 如果 execInfo 没有被捕获，则跳过
      if (!execInfo.captures_) {
        continue;
      }

      auto fn = mapEntry.first;

      // 可能有除了 'AccumulateGrad' 外的节点需要被捕获，比如 RecvRPCBackward
      if (auto accumulateGradFn = dynamic_cast<AccumulateGrad*>(fn)) {

        // 遍历所有捕获
        for (auto& capture : *execInfo.captures_) {

          // 注册捕获钩子，这里是唯一支持的捕获钩子使用实例
          // 请参考 NOTE [Deprecated capture hooks] 以获取更多上下文信息
          capture.DO_NOT_USE_DEPRECATED_register_capture_hook(
              std::make_unique<DistAccumulateGradCaptureHook>(
                  std::dynamic_pointer_cast<AccumulateGrad>(
                      accumulateGradFn->shared_from_this()),
                  autogradContext));
        }
      }
    }

    // 将所有 'RecvRPCBackward' 标记为需要执行
    for (const auto& recvBackwardEdge : recvBackwardEdges) {
      graphTask->exec_info_[recvBackwardEdge.function.get()].needed_ = true;
    }
  }

  // 由于 'owner_' 字段不允许并发访问，因此在单线程中设置 graphTask 的所有者
  graphTask->owner_ = torch::autograd::CPU_DEVICE;

  // 让 autograd 上下文接管 GraphTask 的所有权
  autogradContext->setGraphTask(std::move(graphTask));
}

void DistEngine::execute_graph_task_until_ready_queue_empty(
    NodeTask&& node_task,
    bool incrementOutstandingTasks) {
  // 初始化设备线程池
  engine_.initialize_device_threads_pool();

  // 每次调用都创建一个就绪队列，用于遍历从根节点到执行的图任务
  // 这样允许不同线程并发执行相同的 GraphTask
  std::shared_ptr<ReadyQueue> cpu_ready_queue = std::make_shared<ReadyQueue>();

  // 获取图任务的弱引用
  auto graph_task = node_task.base_.lock();

  // 如果图任务为空，记录错误并跳过执行
  if (graph_task == nullptr) {
    LOG(ERROR) << "GraphTask has expired for NodeTask: "
               << node_task.fn_->name() << ", skipping execution.";
    return;
  }

  // 将节点任务推送到 CPU 就绪队列中
  cpu_ready_queue->push(std::move(node_task), incrementOutstandingTasks);

  // 设置当前设备为 CPU
  torch::autograd::set_device(torch::autograd::CPU_DEVICE);

  // 循环直到就绪队列为空
  while (!cpu_ready_queue->empty()) {
    std::shared_ptr<GraphTask> local_graph_task;
    {
      // 在此作用域内执行，因为 NodeTask 在此后不再需要，可以释放其引用
      NodeTask task = cpu_ready_queue->pop();

      // 如果本地图任务为空，则继续下一个循环
      if (!(local_graph_task = task.base_.lock())) {
        continue;
      }

      // 如果任务函数存在且图任务没有错误
      if (task.fn_ && !local_graph_task->has_error_.load()) {
        // 使用线程本地状态保护
        at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);

        try {
          // 使用图任务保护执行函数
          GraphTaskGuard guard(local_graph_task);
          engine_.evaluate_function(
              local_graph_task, task.fn_.get(), task.inputs_, cpu_ready_queue);
        } catch (std::exception& e) {
          // 处理异常，中止当前图任务的执行
          engine_.thread_on_exception(local_graph_task, task.fn_, e);
          // 在错误时中断循环，以便立即停止执行该 GraphTask，
          // 标记完成并返回带有适当错误消息的 Future
          break;
        }
      }
    }
    // 减少未完成任务的计数
    --local_graph_task->outstanding_tasks_;
  }

  // 检查是否完成了执行
  if (graph_task->completed()) {
    // 不需要显式通知所有者线程，因为 'mark_as_completed_and_run_post_processing'
    // 将标记 Future 为完成，这会通知所有者线程任务已完成
    graph_task->mark_as_completed_and_run_post_processing();
  }
}

c10::intrusive_ptr<c10::ivalue::Future> DistEngine::
    runEngineAndAccumulateGradients(
        const ContextPtr& autogradContext,
        const std::shared_ptr<Node>& graphRoot,
        const edge_list& outputEdges,
        bool incrementOutstandingTasks) {
  // 清理先前状态的未完成远程过程调用
  autogradContext->clearOutstandingRpcs();

  // 恢复图任务的上下文
  auto graphTask = autogradContext->retrieveGraphTask();

  // 使用异步方式启动任务
  at::launch([this, graphTask, graphRoot, incrementOutstandingTasks]() {
    execute_graph_task_until_ready_queue_empty(
        /*node_task*/ NodeTask(graphTask, graphRoot, InputBuffer(0)),
        /*incrementOutstandingTasks*/ incrementOutstandingTasks);

// 调用函数 `execute_graph_task_until_ready_queue_empty`，执行图任务直到就绪队列为空，传入参数为一个 `NodeTask` 对象和一个函数指针 `incrementOutstandingTasks`。


  // Use a reference here to avoid refcount bump on futureGrads.
  auto& futureGrads = graphTask->future_result_;

// 创建引用 `futureGrads`，避免对 `graphTask->future_result_` 的引用计数增加。


  // Build a future that waits for the callbacks to execute (since callbacks
  // execute after the original future is completed). This ensures we return a
  // future that waits for all gradient accumulation to finish.
  auto accumulateGradFuture =
      c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

// 创建一个 `accumulateGradFuture` 对象，它是一个 `c10::ivalue::Future`，用于等待回调函数执行完成。这确保我们返回一个等待所有梯度累积完成的未来对象。


  futureGrads->addCallback([autogradContext, outputEdges, accumulateGradFuture](
                               c10::ivalue::Future& futureGrads) {
    if (futureGrads.hasError()) {
      // Don't accumulate gradients if we receive an error.
      // We must add the node information here since DistEngine::execute
      // waits on accumulateGradFuture and will throw an exception once we
      // set the error below.
      std::string errorMsg = c10::str(
          "Error on Node ",
          DistAutogradContainer::getInstance().getWorkerId(),
          ": ",
          futureGrads.tryRetrieveErrorMessage());
      accumulateGradFuture->setError(std::make_exception_ptr(
          c10::ivalue::Future::FutureError(std::move(errorMsg))));
      return;
    }

// 如果 `futureGrads` 有错误，则设置 `accumulateGradFuture` 的错误状态，防止梯度累积。同时构建一个错误信息字符串，用于描述节点上的错误情况。


    try {
      const variable_list& grads = futureGrads.constValue().toTensorVector();
      TORCH_INTERNAL_ASSERT(grads.size() == outputEdges.size());
      accumulateGradFuture->markCompleted(c10::IValue());
    } catch (std::exception& e) {
      accumulateGradFuture->setErrorIfNeeded(std::current_exception());
    }
  });

// 在 `futureGrads` 没有错误时，尝试从 `futureGrads` 获取梯度列表，并检查其与 `outputEdges` 的大小是否一致。如果一致，将 `accumulateGradFuture` 标记为已完成；否则，设置错误状态。


  return accumulateGradFuture;

// 返回 `accumulateGradFuture` 对象，表示梯度累积的未来操作对象。
  std::unique_lock<std::mutex> lock(initializedContextIdsLock_);
  // 获取互斥锁，确保对 initializedContextIds_ 的安全访问
  if (initializedContextIds_.find(autogradContext->contextId()) ==
      initializedContextIds_.end()) {
    // 如果 autograd 上下文 ID 尚未在 initializedContextIds_ 中注册
    edge_list outputEdges;
    // 创建一个空的边列表，用于存储输出边

    // Pass in a dummy graphRoot since all send functions are the roots.
    // 传入一个虚拟的 graphRoot，因为所有发送函数都是根节点
    auto dummyRoot = std::make_shared<GraphRoot>(edge_list(), variable_list());
    // 创建一个虚拟的 GraphRoot 对象，没有实际的边列表和变量列表

    computeDependencies(
        autogradContext, {}, {}, dummyRoot, outputEdges, retainGraph);
    // 调用 computeDependencies 函数计算依赖关系
    // autogradContext: 当前的自动求导上下文
    // dummyRoot: 虚拟的根节点对象
    // outputEdges: 存储计算得到的输出边列表
    // retainGraph: 是否保留计算图

    // Mark the autograd context id as initialized and unlock.
    // 将 autograd 上下文 ID 标记为已初始化，并解锁互斥锁
    initializedContextIds_.insert(autogradContext->contextId());
    lock.unlock();

    // Enqueue the current send function.
    // 将当前的发送函数加入任务队列
    auto graphTask = autogradContext->retrieveGraphTask();

    // Run the autograd engine.
    // 运行自动求导引擎，计算梯度并累积
    auto accumulateGradFuture = runEngineAndAccumulateGradients(
        autogradContext,
        sendFunction,
        outputEdges,
        /*incrementOutstandingTasks=*/false);

    // Build the 'uber' future that waits for everything.
    // 构建一个综合的 Future 对象，等待所有任务完成
    auto callbackFuture =
        c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    // 创建一个 InstrusivePtr 包装的 Future 对象，初始值为 NoneType::get()
    accumulateGradFuture->addCallback(
        [autogradContext,
         callbackFuture](c10::ivalue::Future& accumulateGradFuture) {
          try {
            if (accumulateGradFuture.hasError()) {
              // 在反向传播结束前执行清理操作（在标记未来完成之前）。
              DistEngine::getInstance().cleanupBackwardPass(autogradContext);

              // 在出现错误时跳过后续处理。
              callbackFuture->setError(accumulateGradFuture.exception_ptr());
              return;
            }

            // 等待自动求导引擎完成后的所有远程过程调用。
            auto rpcFuture =
                autogradContext->clearAndWaitForOutstandingRpcsAsync();
            rpcFuture->addCallback([callbackFuture, autogradContext](
                                       c10::ivalue::Future& rpcFuture) {
              try {
                // 在反向传播结束前执行清理操作（在标记未来完成之前）。
                DistEngine::getInstance().cleanupBackwardPass(autogradContext);
              } catch (std::exception& e) {
                callbackFuture->setErrorIfNeeded(std::current_exception());
                return;
              }

              // 最终标记“超级”未来完成。
              if (!rpcFuture.hasError()) {
                callbackFuture->markCompleted(c10::IValue());
              } else {
                callbackFuture->setError(rpcFuture.exception_ptr());
              }
            });
          } catch (std::exception& e) {
            callbackFuture->setErrorIfNeeded(std::current_exception());
          }
        });

    // 返回等待所有异步处理完成的未来。
    return callbackFuture;
  } else {
    lock.unlock();
    auto graphTask = autogradContext->retrieveGraphTask();
    at::launch([this, graphTask, sendFunction]() {
      execute_graph_task_until_ready_queue_empty(
          /*node_task*/ NodeTask(graphTask, sendFunction, InputBuffer(0)),
          /*incrementOutstandingTasks*/ false);
    });
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    fut->markCompleted(c10::IValue());
    return fut;
  }
}

// 执行分布式引擎的操作，根据给定的上下文 ID、根变量列表和保留图的标志
void DistEngine::execute(
    int64_t contextId,
    const variable_list& roots,
    bool retainGraph) {
  // 根据给定的上下文 ID 获取自动求导上下文，如果上下文 ID 无效将抛出异常
  auto autogradContext =
      DistAutogradContainer::getInstance().retrieveContext(contextId);

  // 执行初始预处理
  edge_list rootEdges;
  variable_list grads;
  // 验证根变量列表并获取边缘列表和梯度列表
  validateRootsAndRetrieveEdges(roots, rootEdges, grads);

  // 创建图根节点，初始化为 GraphRoot 类型
  std::shared_ptr<Node> graphRoot =
      std::make_shared<GraphRoot>(rootEdges, grads);
  edge_list outputEdges;

  // 在本地计算依赖关系，从所有根和所有 'send' 函数开始
  {
    std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
    // 检查上下文 ID 是否已经初始化过
    TORCH_INTERNAL_ASSERT(
        initializedContextIds_.find(autogradContext->contextId()) ==
        initializedContextIds_.end());

    // 计算依赖关系，更新输出边缘列表，保留图的状态由 retainGraph 决定
    computeDependencies(
        autogradContext, rootEdges, grads, graphRoot, outputEdges, retainGraph);

    // 标记自动求导上下文 ID 已经初始化
    initializedContextIds_.insert(autogradContext->contextId());
  }

  // 在 autogradContext 上设置后向传播清理的保护
  BackwardPassCleanupGuard guard(autogradContext);

  // 运行引擎并累积梯度，等待操作完成或抛出异常
  runEngineAndAccumulateGradients(autogradContext, graphRoot, outputEdges)
      ->waitAndThrow();

  // 等待所有未完成的远程过程调用（RPC）完成
  autogradContext->clearAndWaitForOutstandingRpcsAsync()->waitAndThrow();
}

// 清理后向传播过程，确保只有 GraphTask 持有梯度的引用
void DistEngine::cleanupBackwardPass(const ContextPtr& autogradContext) {
  // 验证只有 GraphTask 持有梯度的 Future
  const auto& futureGrads =
      autogradContext->retrieveGraphTask()->future_result_;
  TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1);

  // 重置图任务 GraphTask
  autogradContext->resetGraphTask();

  // 清除所有未完成的远程过程调用（RPC）
  autogradContext->clearOutstandingRpcs();

  // 清除上下文 ID，完成自动求导引擎处理
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  initializedContextIds_.erase(autogradContext->contextId());
}

// 返回当前初始化的后向传播过程数量
size_t DistEngine::numBackwardPasses() const {
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  return initializedContextIds_.size();
}
# 定义名为 DistEngine 的类中的成员函数 getDebugInfo，返回一个无序映射（unordered_map），
# 键类型为 string，值类型为 int，用于存储调试信息。
std::unordered_map<std::string, int> DistEngine::getDebugInfo() const {
  # 创建一个空的无序映射 debugInfo，用于存储调试信息
  std::unordered_map<std::string, int> debugInfo;
  # 将键为 kNumBackwardPasses 的值设为 numBackwardPasses() 函数的返回值，
  # 表示反向传播次数。
  debugInfo[kNumBackwardPasses] = numBackwardPasses();
  # 将键为 kNumAutogradContexts 的值设为 DistAutogradContainer 的单例对象
  # 的 numAutogradContexts() 函数的返回值，表示自动求导上下文的数量。
  debugInfo[kNumAutogradContexts] =
      DistAutogradContainer::getInstance().numAutogradContexts();
  # 返回存储了调试信息的 debugInfo 无序映射。
  return debugInfo;
}

# 结束 autograd 命名空间
} // namespace autograd

# 结束 distributed 命名空间
} // namespace distributed

# 结束 torch 命名空间
} // namespace torch
```