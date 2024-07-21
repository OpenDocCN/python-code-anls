# `.\pytorch\torch\csrc\autograd\engine.cpp`

```py
// 引入 Torch 自动求导引擎相关头文件
#include <torch/csrc/autograd/engine.h>

// 引入 Torch 自动求导异常模式相关头文件
#include <torch/csrc/autograd/anomaly_mode.h>
// 引入 Torch 自动求导相关头文件
#include <torch/csrc/autograd/autograd.h>
// 引入 Torch 自动求导函数相关头文件
#include <torch/csrc/autograd/function.h>
// 引入 Torch 自动求导基本操作相关头文件
#include <torch/csrc/autograd/functions/basic_ops.h>
// 引入 Torch 自动求导梯度模式相关头文件
#include <torch/csrc/autograd/grad_mode.h>
// 引入 Torch 自动求导变量相关头文件
#include <torch/csrc/autograd/variable.h>
// 引入 Torch Dynamo 编译自动求导相关头文件
#include <torch/csrc/dynamo/compiled_autograd.h>

// 引入 ATen 设备加速器相关头文件
#include <ATen/DeviceAccelerator.h>
// 引入 ATen 设备保护相关头文件
#include <ATen/DeviceGuard.h>
// 引入 ATen 扩展工具相关头文件
#include <ATen/ExpandUtils.h>
// 引入 ATen 并行操作相关头文件
#include <ATen/Parallel.h>
// 引入 ATen 稀疏 CSR 张量工具相关头文件
#include <ATen/SparseCsrTensorUtils.h>
// 引入 ATen CUDA 钩子接口相关头文件
#include <ATen/detail/CUDAHooksInterface.h>
// 引入 ATen 私有使用 1 钩子接口相关头文件
#include <ATen/detail/PrivateUse1HooksInterface.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则引入 ATen 函数相关头文件，否则引入 ATen 是否 NaN 相关头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/isnan.h>
#endif

// 引入 C10 核心设备保护相关头文件
#include <c10/core/DeviceGuard.h>
// 引入 C10 核心事件相关头文件
#include <c10/core/Event.h>
// 引入 C10 核心流相关头文件
#include <c10/core/Stream.h>
// 引入 C10 核心流保护相关头文件
#include <c10/core/StreamGuard.h>
// 引入 C10 实用工具中止处理相关头文件
#include <c10/util/AbortHandler.h>
// 引入 C10 实用工具异常处理相关头文件
#include <c10/util/Exception.h>
// 引入 C10 实用工具可选值相关头文件
#include <c10/util/Optional.h>
// 引入 C10 实用工具线程本地存储相关头文件
#include <c10/util/ThreadLocal.h>
// 引入 C10 实用工具范围迭代相关头文件
#include <c10/util/irange.h>
// 引入 C10 实用工具线程命名相关头文件
#include <c10/util/thread_name.h>

// 引入标准库头文件
#include <atomic> // 原子操作相关头文件
#include <chrono> // 时间相关头文件
#include <condition_variable> // 条件变量相关头文件
#include <cstdint> // 标准整数类型相关头文件
#include <functional> // 函数对象相关头文件
#include <iostream> // 输入输出流相关头文件
#include <memory> // 内存管理相关头文件
#include <mutex> // 互斥量相关头文件
#include <queue> // 队列相关头文件
#include <sstream> // 字符串流相关头文件
#include <string> // 字符串处理相关头文件
#include <thread> // 线程相关头文件
#include <unordered_set> // 无序集合相关头文件
#include <utility> // 实用工具相关头文件

// Torch 自动求导命名空间
namespace torch::autograd {

// 匿名命名空间，用于定义局部静态变量和函数
namespace {

// 标志当前是否处于错误的自动求导分叉状态，初始化为 false
static bool in_bad_autograd_fork = false;

// 在自动求导引擎的线程池初始化后，被 fork 的子进程调用此函数
static void forked_autograd_child() {
  in_bad_autograd_fork = true;
}

// 在进行不安全的分叉操作（线程池）之前调用此函数，以跟踪错误的自动求导分叉
static void track_bad_autograd_forks() {
  // 如果不是在 Windows 系统下
#if !defined(WIN32)
  static c10::once_flag flag;
  // 保证以下代码只执行一次，注册 forked_autograd_child 函数
  c10::call_once(
      flag, [&] { pthread_atfork(nullptr, nullptr, forked_autograd_child); });
#endif
}

// 内联函数，判断是否应该在 CPU 就绪队列中运行
inline bool should_run_in_cpu_ready_queue(c10::DeviceType device) {
  // 如果设备类型是 CPU、Meta 或 Lazy，则返回 true，否则返回 false
  if (device == c10::kCPU || device == c10::kMeta || device == c10::kLazy) {
    return true;
  } else {
    return false;
  }
}

// 定义原子变量，存储编译自动求导函数的指针
std::atomic<Engine::compiled_autograd_fn> the_compiled_autograd = nullptr;

// 定义编译自动求导函数被污染的标志常量
#define COMPILED_AUTOGRAD_POISON \
  reinterpret_cast<Engine::compiled_autograd_fn>(1)

// 定义原子变量，存储处于反向传播中的线程数
std::atomic<int32_t> num_threads_in_backwards;

// 在编译自动求导线程中检查调试线程计数
struct CompiledAutogradThreadingDebugCheck {
  CompiledAutogradThreadingDebugCheck() {
    num_threads_in_backwards++;
  }
  ~CompiledAutogradThreadingDebugCheck() {
    release();
  }
  // 释放资源
  void release() {
    if (std::exchange(incremented, false)) {
      num_threads_in_backwards--;
    }
  }

 private:
  bool incremented{true}; // 增加计数标志
};

} // namespace

// 引擎创建的线程分配一个 worker_device 指定其处理的设备工作。此变量在以下时刻初始化：
// 1. 对于 CUDA、XLA 设备线程，在它们作为等待其设备上的工作的自旋线程创建时。
// 2. 对于 CPU 线程，在图任务执行之前，因为每个线程需要知道其工作的设备。
// 当向后调用时，使用调用线程来驱动引擎执行。
// 这在处理可重入的向后调用时使用；
// 参见注释 [Reentrant backwards]
static thread_local int worker_device = NO_DEVICE;

// 如果在重入引擎调用堆栈中的所有调用都是命令式向后调用，则此变量为true。
// 此特殊变量仅适用于梯度检查点功能。
static thread_local bool checkpoint_valid = true;

// 当前线程上当前嵌套的向后调用数量。
static thread_local int current_depth = 0;

// 对于所有设备线程（如CUDA、XLA），total_depth表示所有设备线程上所有嵌套向后调用的总深度。
// 对于CPU设备，它是与原始向后调用关联的总深度。
static thread_local int total_depth = 0;

// 当前线程正在执行的GraphTask。
// 这有助于queue_callback()找到要追加最终回调的目标GraphTask。
C10_DEFINE_TLS_static(std::shared_ptr<GraphTask>, tls_current_graph_task);
#define current_graph_task (tls_current_graph_task.get())

// 每个自动求导工作线程都与一个ready队列关联，
// 指定此线程要执行的工作流。这个shared_ptr是指向每个线程的ready_queue的thread_local指针，
// 它应该通过Engine::init_local_ready_queue()在每个对应线程的执行之前进行初始化。
//
// CUDA、XLA线程在整个向后调用通过device_ready_queues_共享，
// 而调用线程专用于处理应在cpu_ready_queue中返回true的设备的工作（尤其是CPU设备）。
// 因此，任何给定的图任务维护其自己的cpu_ready_queue_，您应将工作发送到其中以执行。
//
// 对于可重入的向后调用，如果我们从当前线程生成新线程，因为达到了最大深度，
// 新线程将仅为了性能改进重用与父线程相同的ReadyQueue。
// 参见注释 [Reentrant backwards] 了解更多详情。
C10_DEFINE_TLS_static(std::shared_ptr<ReadyQueue>, tls_local_ready_queue);
#define local_ready_queue (tls_local_ready_queue.get())

// 注释 [Reentrant backwards]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// 要理解可重入向后问题，我们必须注意当今自动求导引擎实现的两个方面：
//
//  1. 当调用Engine::execute()时，您希望阻塞，直到差分完成，以便获取向后传递的最终结果变量。
//
//  2. 引擎通过每个工作队列具有一个单独的工作线程来操作，并且每个工作队列固定在特定的设备上执行操作。
//
// 问题是，假设您在工作线程内部调用backward()。根据属性（1），我们应该阻塞，直到嵌套任务完成。
// 然而，根据属性（2），此工作线程位于
//`
// hook for processing the tasks assigned to it; we better not block,
// because then all of our backward executions (including the one we
// just started) will deadlock!
//
// We maintain a pool of threads waiting for work to do
// When a reentrant backwards call occurs, the current thread blocks
// and a thread from the pool is woken up to complete the blocking tasks and any
// other tasks that would have been assigned to that worker. If there are no
// threads available, a new thread is spawned. The new thread will continue
// processing tasks from the same ReadyQueue as the parent worker
//
// When the GraphTask is finished, the parent worker thread that is waiting on
// the task is notified and the current thread returns to the pool.

// Note [Streaming backwards]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// On CUDA/privateuse1 devices the autograd engine's device operations are run
// on the same stream that ran them in forward. This requires automatically
// syncing the streams so that function A finishes producing its
// output before function B consumes it.
//
// This synchronization occurs when outputs are placed into input buffers.
// The functions corresponding to input buffer positions have metadata
// recording their streams from forward, and during backward this
// data is used to sync the producer's stream with the consumer's.
//
// When a CUDA/privateuse1 function is run either all its inputs were
// accumulated on the stream used to run the function OR the inputs are on
// different devices and the function is responsible for properly acquiring
// them.
//
// User-facing stream semantics of a backward() (or torch.autograd.grad())
// call with respect to surrounding ops are the same as for any other call.
// See "Stream semantics of backward passes" on
// https://pytorch.org/docs/stable/notes/cuda.html
//
// Internally, backward() runs ops (including leaf nodes) on side threads.
// And streams are thread local. So GraphTask achieves the above semantics by
//  1. remembering the current streams on all active CUDA/privateuse1 devices
//     in the user-facing thread (aka, the thread that called execute() to
//     launch the GraphTask)
//  2. remembering the "leaf streams" (streams each backward leaf node ran on)
//  3. during exec_post_processing, for each leaf stream, sync the remembered
//
    : prev_checkpoint_valid_state(checkpoint_valid) {


// 使用成员初始化列表，初始化 prev_checkpoint_valid_state 成员变量为 checkpoint_valid 的值
checkpoint_valid =
    graph_task->can_checkpoint() && prev_checkpoint_valid_state;


这段代码是C++中的成员初始化列表，用于初始化类的成员变量 `prev_checkpoint_valid_state`。在构造函数中，通过 `prev_checkpoint_valid_state(checkpoint_valid)`，将 `prev_checkpoint_valid_state` 初始化为 `checkpoint_valid` 的值。后续的赋值语句 `checkpoint_valid = ...` 可能是在构造函数体内部执行的逻辑，但这部分超出了给定的代码片段。
}

// CheckpointValidGuard 析构函数实现
CheckpointValidGuard::~CheckpointValidGuard() {
  // 恢复 checkpoint_valid 的前一个状态
  checkpoint_valid = prev_checkpoint_valid_state;
}

// ReadyQueue 类的 push 方法实现
auto ReadyQueue::push(NodeTask item, bool incrementOutstandingTasks) -> void {
  {
    // 对 mutex_ 加锁，用于对 heap_ 的写操作
    std::lock_guard<std::mutex> lock(mutex_);
    // 如果需要增加 outstanding tasks，则获取与 item 关联的 GraphTask 对象，并增加其 outstanding_tasks_ 计数
    if (incrementOutstandingTasks) {
      std::shared_ptr<GraphTask> graph_task = item.base_.lock();
      TORCH_INTERNAL_ASSERT(graph_task, "GraphTask is no longer valid!");
      ++graph_task->outstanding_tasks_;
    }
    // 将 item 推入 heap_ 中
    heap_.push(std::move(item));
  }
  // 通知一个等待的线程（如果有）
  not_empty_.notify_one();
}

// ReadyQueue 类的 pushShutdownTask 方法实现
auto ReadyQueue::pushShutdownTask() -> void {
  {
    // 对 mutex_ 加锁，用于对 heap_ 的写操作
    std::lock_guard<std::mutex> lock(mutex_);
    // 将一个 shutdown 任务推入 heap_ 中
    heap_.push(NodeTask({}, nullptr, InputBuffer(0), true));
  }
  // 通知一个等待的线程（如果有）
  not_empty_.notify_one();
}

// ReadyQueue 类的 size 方法实现
size_t ReadyQueue::size() const {
  // 对 mutex_ 加锁，用于访问 heap_ 的操作
  std::unique_lock<std::mutex> lock(mutex_);
  // 返回 heap_ 的大小
  return heap_.size();
}

// ReadyQueue 类的 pop 方法实现
auto ReadyQueue::pop() -> NodeTask {
  // 对 mutex_ 加锁，用于访问 heap_ 的操作
  std::unique_lock<std::mutex> lock(mutex_);
  // 等待直到 heap_ 不为空
  not_empty_.wait(lock, [this] { return !heap_.empty(); });
  // 获取 heap_ 中顶部的任务，并移除
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto task = std::move(const_cast<NodeTask&>(heap_.top()));
  heap_.pop();
  return task;
}

// ReadyQueue 类的 empty 方法实现
bool ReadyQueue::empty() const {
  // 对 mutex_ 加锁，用于访问 heap_ 的操作
  std::unique_lock<std::mutex> lock(mutex_);
  // 检查 heap_ 是否为空
  return heap_.empty();
}

// Engine 类的构造函数实现
Engine::Engine()
    : max_recursion_depth_(MAX_DEPTH), non_reentrant_device_thread_count_(0) {}

// Engine 类的析构函数实现
Engine::~Engine() {
  // 调用 stop 方法进行清理
  stop();
}

// Engine 类的 stop 方法实现
// 如果没有 backward 任务在运行，则向所有 device_ready_queues_ 发送 shutdown 任务
// 即使 readyQueue 应为空，shutdown 任务具有最高优先级
void Engine::stop() {
  if (stopped_) {
    return;
  }
  stopped_ = true;
  // 在某些情况下，autograd 线程在 shutdown 时可能会挂起
  // 不要无限期等待它们的关闭，而是依赖于超时
  auto wait_duration_str = getenv("TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT");
  auto wait_duration = wait_duration_str ? std::atof(wait_duration_str) : 10.0;
  bool noBackward = true;
  for (auto& queue : device_ready_queues_) {
    noBackward = noBackward && queue->empty();
  }
  if (noBackward && wait_duration > 0.0f) {
    for (auto& queue : device_ready_queues_) {
      queue->pushShutdownTask();
    }
    // 在 Windows 上不等待全局线程的终止
    // 因为 CRT 在调用全局对象析构器之前终止 DLL 线程
#if !defined(_WIN32) || defined(C10_USE_MSVC_STATIC_RUNTIME)

    using namespace std::chrono_literals;
    // 设置等待设备线程关闭的截止时间
    auto wait_deadline =
        std::chrono::steady_clock::now() + wait_duration * 1.0s;
    std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
    // 等待直到所有的 non_reentrant_device_thread_count_ 为零
    while (non_reentrant_device_thread_count_.load() != 0) {
      if (non_reentrant_device_thread_condvar_.wait_until(lk, wait_deadline) ==
          std::cv_status::timeout) {
        break;
      }
    }
#endif
  }
  // 否则线程将会泄漏

void Engine::release_workers() {
  // 获取非重入设备线程互斥锁的独占访问权限
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  // 将非重入设备线程计数器置零
  non_reentrant_device_thread_count_.store(0);
  // 通知一个等待中的线程，非重入设备线程条件变量
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::increment_non_reentrant_thread_count() {
  // 获取非重入设备线程互斥锁的独占访问权限
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  // 非重入设备线程计数器加一
  non_reentrant_device_thread_count_.fetch_add(1);
  // 通知一个等待中的线程，非重入设备线程条件变量
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::decrement_non_reentrant_thread_count() {
  // 获取非重入设备线程互斥锁的独占访问权限
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  // 非重入设备线程计数器减一
  non_reentrant_device_thread_count_.fetch_sub(1);
  // 通知一个等待中的线程，非重入设备线程条件变量
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {
  // pthread_setname_np 限制名称长度为16个字符，包括空字符
  std::string thread_name = "pt_autograd_" + std::to_string(device);
  // 设置当前线程的名称
  c10::setThreadName(thread_name);

  // 设置终止处理程序
  c10::set_terminate_handler();
  if (should_increment) {
    // 如果需要增加线程计数，则执行增加操作
    increment_non_reentrant_thread_count();
  }

  // 初始化 ATen 库的线程数目
  at::init_num_threads();

  // 注释 [分配 GPU 给 autograd 线程]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 我们的策略是什么？最初，autograd 引擎仅针对 CUDA 编写。
  // 我们为处理所有 CPU 操作分配一个线程，每个 CUDA 设备一个线程。
  //
  // 但是如果有其他设备呢？有两种可能的策略：
  //
  //  - 我们可以分配线程数为 max(num_cuda_devices, num_xla_devices, ...)
  //    并将 CUDA 设备 0 与 XLA 设备 0 放在一起。
  //  - 我们可以分配线程数为 sum(num_cuda_devices, num_xla_devices, ...)
  //    保持每个设备分离。
  //
  // 我们没有充分理由偏向其中一种，因此我们任意选择了合并设备的策略。
  // 或许另一种方法更好。
  worker_device = device;

  // 使用在线程初始化之前创建的全局就绪队列初始化每个设备线程的线程本地就绪队列
  init_local_ready_queue(ready_queue);

  // 创建一个空的图任务共享指针
  std::shared_ptr<GraphTask> graph_task = nullptr;
  // 执行线程主函数
  thread_main(graph_task);
  if (should_increment) {
    // 如果之前已增加过计数，则在关闭期间减少计数
    decrement_non_reentrant_thread_count();
  }
}

GraphTaskGuard::GraphTaskGuard(std::shared_ptr<GraphTask> graph_task)
    : last_graph_task_(std::move(current_graph_task)) {
  // 将当前图任务指针存储在 last_graph_task_ 中
  current_graph_task = std::move(graph_task);
}
GraphTaskGuard::~GraphTaskGuard() {
  // 恢复当前图任务指针
  restore_current_graph_task();
}

void GraphTaskGuard::restore_current_graph_task() {
  // 恢复上一个存储的图任务指针到 current_graph_task
  current_graph_task = std::move(last_graph_task_);
}

// 当前图任务的 exec_info 用于在节点评估期间修剪不必要的边缘，
// 见 `Node.task_should_compute_output()` 函数。
const std::unordered_map<Node*, GraphTask::ExecInfo>*
// 返回当前图任务的执行信息的指针，如果当前图任务不存在则返回空指针
get_current_graph_task_exec_info() {
  return current_graph_task ? &current_graph_task->exec_info_ : nullptr;
}

// 返回当前图任务中图中节点的指针集合，如果当前图任务不存在则返回空指针
const std::unordered_set<Node*>* get_current_graph_task_nodes_in_graph() {
  return current_graph_task ? &current_graph_task->nodes_in_graph_ : nullptr;
}

// 返回当前图任务的ID，如果当前图任务不存在则返回-1
int get_current_graph_task_id() {
  return current_graph_task ? current_graph_task->id_ : -1;
}

// 返回当前图任务是否保持图的标志，如果当前图任务不存在则默认返回true
bool get_current_graph_task_keep_graph() {
  return current_graph_task ? current_graph_task->keep_graph_ : true;
}

// 将节点添加到当前图任务的执行信息中，标记为需要执行
void add_node_to_current_graph_task_exec_info(Node* fn) {
  current_graph_task->exec_info_[fn].needed_ = true;
}

// 注意：引擎本身不使用此函数的输出结果。
// 返回当前图任务的执行顺序的节点向量
std::vector<Node*> get_current_graph_task_execution_order() {
  std::shared_ptr<GraphTask> task = current_graph_task;
  if (!task) {
    return {};
  }

  // 检查当前是否启用多线程，若启用则抛出异常
  TORCH_CHECK(
      !c10::AutogradState::get_tls_state().get_multithreading_enabled(),
      "get_current_graph_task_execution_order expects the current backward to be "
      "executed with multithreading disabled, e.g. by running:\n\n"
      ">>> with torch.autograd.set_multithreading_enabled(False):\n"
      "...     torch.autograd.grad(...)\n");

  const bool check_exec_info = !task->exec_info_.empty();
  std::vector<Node*> out{};
  // 复制依赖关系，因为后续会进行修改
  std::unordered_map<Node*, int> dependencies = task->dependencies_;

  // 定义比较函数，用于节点优先队列的排序
  auto compare_seq_nr = [](Node* n1, Node* n2) {
    return n1->sequence_nr() < n2->sequence_nr();
  };
  std::priority_queue<Node*, std::vector<Node*>, decltype(compare_seq_nr)> heap(
      compare_seq_nr);

  // 将图根节点添加到优先队列中
  for (Node* ptr : task->graph_roots_) {
    heap.push(ptr);
  }

  // 实现备注:
  // - 虽然有序列号(sequence_nr)，但需要计算依赖关系，因为在积累梯度的情况下不能假设输出的序列号比输入的序列号高
  // - 不需要检查拓扑序号(topological_nr)，因为有执行信息(exec_info)的存在
  while (!heap.empty()) {
    Node* fn = heap.top();
    heap.pop();

    out.push_back(fn);
    // 遍历节点的后继边
    for (const auto& edge : fn->next_edges()) {
      Node* next_ptr = edge.function.get();
      if (!next_ptr) {
        continue;
      }
      // 如果需要检查执行信息，则验证是否应该执行
      if (check_exec_info) {
        auto it = task->exec_info_.find(next_ptr);
        if (it == task->exec_info_.end() || !it->second.should_execute()) {
          continue;
        }
      }
      // 更新依赖关系计数
      auto it = dependencies.find(edge.function.get());
      TORCH_INTERNAL_ASSERT(it != dependencies.end());
      if (--it->second == 0) {
        dependencies.erase(it);
        heap.push(next_ptr);
      }
    }
  }
  return out;
}

// 注意: 图任务并不一定形成一个栈。考虑以下情况:
//
//    +----> Eval1
//  Root
//    +----> Eval2
//
// 一旦执行Root，Eval1和Eval2都将被添加到准备队列中。
// Next, Eval1 is run and this causes the worker to enter thread_main again.
// Then, it pops the next task from the queue, but at this point it is Eval2.
// It enters thread_main once again, but now with graph_task of Eval2, which is
// completely unrelated to that of Eval1 (it's not a recursive call).
// It's all ok and is handled right now, but it should be accounted for
// in case this code is to be changed.
//
// thread_main is used by:
// 1). autograd threads for devices (i.e. CUDA, XLA)
// 2). the caller/owning thread of the backward call on CPU (sync mode)
// 3). Reentrant backward that is invoked by either 1) or 2)
// The exit conditions are different for the above three cases.
// For 1), we are spinning on running the thread_main on device autograd
//         threads throughout the Engine lifetime, thread_main will get
//         terminated during Engine destruction by pushing shutdown tasks
// For 2), the owning thread of the backward call drives the thread_main
//         synchronously until the graph_task of that owning thread is
//         completed and exit the thread_main to continue executing the
//         result of caller's code.
// For 3), the reentrant backward that invokes
//         thread_main, either from 1) or 2), will not spin and will exit as
//         long as graph_task is completed and notify the owning thread as
//         needed.
auto Engine::thread_main(const std::shared_ptr<GraphTask>& graph_task) -> void {
  // When graph_task is nullptr, this is a long running thread that processes
  // tasks (ex: device threads). When graph_task is non-null (ex: reentrant
  // backwards, user thread), this function is expected to exit once that
  // graph_task complete.

  // local_ready_queue should already been initialized when we get into
  // thread_main
  TORCH_INTERNAL_ASSERT(local_ready_queue != nullptr);
  while (graph_task == nullptr || !graph_task->future_result_->completed()) {
    // local_graph_task represents the graph_task we retrieve from the queue.
    // The outer graph_task represents the overall graph_task we need to execute
    // for reentrant execution.
    std::shared_ptr<GraphTask> local_graph_task;
    {
      // Scope this block of execution since NodeTask is not needed after this
      // block and can be deallocated (release any references to grad tensors
      // as part of inputs_).
      // 从本作用域开始，因为在此之后 NodeTask 不再需要，并且可以释放任何与梯度张量有关的引用作为 inputs_ 的一部分。
      NodeTask task = local_ready_queue->pop();
      // This will only work if the worker is running a non backward task
      // TODO Needs to be fixed this to work in all cases
      // 如果任务是关闭任务，则记录 API 使用信息并中断循环执行。
      if (task.isShutdownTask_) {
        C10_LOG_API_USAGE_ONCE("torch.autograd.thread_shutdown");
        break;
      }

      local_graph_task = task.base_.lock();
      // 如果 local_graph_task 为空，则说明函数的 GraphTask 不再有效，跳过进一步执行。
      if (!local_graph_task) {
        // GraphTask for function is no longer valid, skipping further
        // execution.
        continue;
      }

      set_device(worker_device);

      if (task.fn_ && !local_graph_task->has_error_.load()) {
        // Set the ThreadLocalState before calling the function.
        // NB: The ThreadLocalStateGuard doesn't set the grad_mode because
        // GraphTask always saves ThreadLocalState without grad_mode.
        // 在调用函数之前设置线程本地状态。注意：ThreadLocalStateGuard 不设置梯度模式，因为 GraphTask 总是保存不带梯度模式的 ThreadLocalState。
        at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);
        c10::WarningUtils::WarningHandlerGuard warnings_guard(
            &local_graph_task->warning_handler_);

        try {
          // The guard sets the thread_local current_graph_task on construction
          // and restores it on exit. The current_graph_task variable helps
          // queue_callback() to find the target GraphTask to append final
          // callbacks.
          // guard 在构造时设置 thread_local current_graph_task，在退出时恢复。current_graph_task 变量帮助 queue_callback() 找到目标 GraphTask 来附加最终回调。
          GraphTaskGuard guard(local_graph_task);
          NodeGuard ndguard(task.fn_);
          {
            // 记录函数执行的操作，使用函数名创建记录点。
            RECORD_FUNCTION(
                c10::str(
                    "autograd::engine::evaluate_function: ",
                    task.fn_.get()->name()),
                c10::ArrayRef<const c10::IValue>());
            // 调用 evaluate_function 来执行函数的评估，将结果放入 local_graph_task 的 cpu_ready_queue_。
            evaluate_function(
                local_graph_task,
                task.fn_.get(),
                task.inputs_,
                local_graph_task->cpu_ready_queue_);
          }
        } catch (std::exception& e) {
          // See Note [ Persisting PyErr state across autograd engine threads ]
          // 处理异常，参考 "在自动求导引擎线程中持久化 PyErr 状态" 注释。
          thread_on_exception(local_graph_task, task.fn_, e);
        }
      }
    }

    // Decrement the outstanding tasks.
    // 减少未完成的任务计数。
    --local_graph_task->outstanding_tasks_;

    // Check if we've completed execution.
    // 检查是否完成了执行。
    // 检查本地图任务是否已完成
    if (local_graph_task->completed()) {
      // 标记图任务为已完成并执行后处理
      local_graph_task->mark_as_completed_and_run_post_processing();

      // 获取图任务的所有者
      auto base_owner = local_graph_task->owner_;
      // 如果当前工作线程完成了图任务，但是图任务的所有者线程可能在 pop() 上睡眠，如果它没有工作。
      // 因此，我们需要向拥有线程发送一个虚拟的函数任务，只是为了确保它没有睡眠，这样我们可以退出线程主函数。
      // 如果它有工作，它可能会在到达任务之前看到 graph_task->outstanding_tasks_ == 0，但无论如何这都是一个空操作。
      //
      // 注意：如果当前线程就是所有者线程，则不需要执行此操作。
      if (worker_device != base_owner) {
        // 使用 std::atomic_thread_fence(std::memory_order_release) 同步 outstanding_tasks_ 和队列互斥体
        std::atomic_thread_fence(std::memory_order_release);
        // 将一个虚拟的 NodeTask 推入所有者线程的准备队列中，以确保其不会睡眠
        ready_queue_by_index(local_graph_task->cpu_ready_queue_, base_owner)
            ->push(NodeTask(local_graph_task, nullptr, InputBuffer(0)));
      }
    }
}

// Reentrant call will re-use the graph_task's owner thread ready_queue for
// queueing tasks (NOTE: this is not true in the async_mode of the engine).
// While we can create separate ready queue for each new reentrant
// thread, but sharing the same cpu_ready_queue with parent thread is a
// performance improvement and cuda thread still have to do the same thing.
// 初始化可重入线程
void Engine::reentrant_thread_init() {
  // 设置终止处理程序
  c10::set_terminate_handler();
  // 初始化线程数
  at::init_num_threads();
  auto tp_shared = thread_pool_shared_;
  // 循环执行以下操作
  while (true) {
    std::unique_lock<std::mutex> lk(tp_shared->mutex_);
    // 增加工作线程计数
    ++thread_pool_shared_->num_workers_;
    // 等待直到 graphtasks_queue 非空
    tp_shared->work_.wait(
        lk, [&tp_shared] { return !tp_shared->graphtasks_queue_.empty(); });
    // 减少工作线程计数
    --thread_pool_shared_->num_workers_;
    // 获取任务队列的第一个任务
    auto task = tp_shared->graphtasks_queue_.front();
    tp_shared->graphtasks_queue_.pop();
    lk.unlock();
    // 获取任务的 shared_ptr
    std::shared_ptr<GraphTask> graph_task = task.lock();
    if (!graph_task) {
      // 如果 GraphTask 已过期，则跳过可重入执行
      LOG(INFO) << "GraphTask has expired, skipping reentrant execution";
      continue;
    }
    // 设置设备为 graph_task 的 owner
    set_device(graph_task->owner_);
    // 将 local_ready_queue 设置为 graph_task 的 ready_queue_by_index 方法返回的就绪队列
    local_ready_queue =
        ready_queue_by_index(graph_task->cpu_ready_queue_, graph_task->owner_);
    // 设置总深度为 graph_task 的重入深度
    total_depth = graph_task->reentrant_depth_;
    // 执行线程主体
    thread_main(graph_task);
  }
}

// 在线程异常时的处理方法
void Engine::thread_on_exception(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::shared_ptr<GraphTask> graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {
  // 将异常设置到 graph_task 中
  graph_task->set_exception(std::current_exception(), fn);
}

namespace {
// 图任务 ID 的原子计数器
std::atomic<uint64_t> graph_task_id{0};
}

// 构造函数：初始化图任务
GraphTask::GraphTask(
    bool keep_graph,
    bool grad_mode,
    int reentrant_depth,
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    c10::SmallVector<Node*, 4> graph_roots,
    bool exit_on_error)
    : keep_graph_(keep_graph),
      graph_roots_(std::move(graph_roots)),
      owner_(NO_DEVICE),
      reentrant_depth_(reentrant_depth),
      exit_on_error_(exit_on_error),
      cpu_ready_queue_(std::move(cpu_ready_queue)),
      future_result_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))),
      id_(graph_task_id.fetch_add(1, std::memory_order_relaxed)) {
  // 设置线程本地的梯度模式
  thread_locals_.set_grad_mode(grad_mode);
}

// 检查图任务是否完成
bool GraphTask::completed() {
  return outstanding_tasks_.load() == 0 ||
      (exit_on_error_ && has_error_.load());
}

// 标记图任务为已完成并运行后处理逻辑
void GraphTask::mark_as_completed_and_run_post_processing() {
  // 只允许一个线程尝试处理此逻辑
  if (future_completed_.exchange(true)) {
    // 如果未来已标记为完成或正在标记为完成，则添加 wait() 以确保在退出时标记未来为完成
    future_result_->wait();
    return;
  }

  try {
    // 在标记未来为完成之前运行后处理逻辑
    // 在完成操作之前释放锁，以避免在回调期间保持锁定状态。
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 执行后续处理操作
    exec_post_processing();
    
    // 将捕获的变量移动到新的变量向量中
    std::vector<Variable> vars = std::move(captured_vars_);
    
    // 在调用 markCompleted 之前需要解锁，以避免在回调函数调用时保持锁定状态。
    lock.unlock();
    
    // 尝试标记 future_result_ 为已完成，并传递变量向量作为参数
    future_result_->markCompleted(vars);
    } catch (std::exception&) {
      // 如果捕获到异常，设置 future_result_ 的错误状态
      future_result_->setErrorIfNeeded(std::current_exception());
    }
}

// 执行后处理操作的方法，用于图任务的执行
void GraphTask::exec_post_processing() {
    // 如果 not_ready_ 非空，抛出运行时错误，表示某些函数的梯度计算失败
    if (!not_ready_.empty()) {
        throw std::runtime_error("could not compute gradients for some functions");
    }

    // 将当前的图任务设置为线程本地变量 current_graph_task_
    // 因为可能通过现有的最终回调安装更多的回调
    GraphTaskGuard guard(shared_from_this());

    // 在每次迭代期间通过锁定 final_callbacks_lock_ 互斥量以访问 final_callbacks.size()
    // 解锁是必要的，因为回调可以注册更多回调，或者它们可以从其他线程注册
    std::unique_lock<std::mutex> cb_lock(final_callbacks_lock_);

    // 保存非空的 caller_current_streams_ 到 caller_current_streams_filtered
    std::vector<c10::Stream> caller_current_streams_filtered;

    // 参见注释 [Streaming backwards]。
    // 同步 caller_current_stream 和 leaf_streams，因此 final_callbacks 可以使用
    // 任何当前流上的梯度
    if (!leaf_streams.empty()) {
        for (const auto& leaf_stream : leaf_streams) {
            // stash_current_cuda/privateuse1_streams() 为所有已经在 GraphTask 执行之前具有 CUDA/privateuse1 上下文的设备 ID 保存了流。
            // 对于非活动设备，它保存了 c10::nullopt。不期望 GraphTask 的反向传播在任何新设备上运行叶节点，因此保存的流应该足够。
            // 如果 leaf_stream.device_index() 恰好是一个新设备，则在 c10::nullopt 上使用 operator* 应该会抛出错误。
            const auto caller_current_stream =
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                *caller_current_streams_[leaf_stream.device_index()];

            if (caller_current_stream != leaf_stream) {
                auto event = c10::Event{leaf_stream.device_type()};
                event.record(leaf_stream);
                caller_current_stream.wait(event);
            }
        }

        caller_current_streams_filtered.reserve(caller_current_streams_.size());
        for (const auto& opt_stream : caller_current_streams_) {
            if (opt_stream.has_value()) {
                caller_current_streams_filtered.push_back(*opt_stream);
            }
        }
    }

    {
        // final_callbacks 在每个设备的 caller_current_streams 上运行（即用户调用 backward() 时周围的环境流）。
        // 这有两个好处：
        //  1. caller_current_streams 已经与 leaf_streams 同步，因此回调可以安全地访问任何梯度。
        //  2. 回调的结果可以安全地在（用户可见的）caller_current_streams 上使用在 backward() 之后。
        c10::MultiStreamGuard g(caller_current_streams_filtered);

        // 在调用函数之前设置 ThreadLocalState。
        // 注意：ThreadLocalStateGuard 不会设置 grad_mode，因为 GraphTask 总是保存 ThreadLocalState 而不包括 grad_mode。
        at::ThreadLocalStateGuard tls_guard(this->thread_locals_);

        // 警告：不要在这里使用范围-for 循环，因为可能会有更多的回调
    // 在回调调用之间添加，因此迭代器可能会失效。
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (size_t i = 0; i < final_callbacks_.size(); ++i) {
      // 解锁回调锁
      cb_lock.unlock();
      // 调用最终回调函数列表中的回调函数
      final_callbacks_[i]();
      // 再次锁定回调锁，以便进行下一次迭代
      cb_lock.lock();
    }
}

// 设置异常，不触发信号处理
void GraphTask::set_exception_without_signal(const std::shared_ptr<Node>& fn) {
    // 如果错误状态尚未设置，则设置为true
    if (!has_error_.exchange(true)) {
        // 如果异常模式启用且存在节点fn，则打印节点fn的堆栈信息
        if (AnomalyMode::is_enabled() && fn) {
            fn->metadata()->print_stack(fn->name());
        }
    }
}

// 设置异常，包括设置future_completed_标志和设置错误结果
void GraphTask::set_exception(
    std::exception_ptr eptr,
    const std::shared_ptr<Node>& fn) {
    // 调用set_exception_without_signal设置异常
    set_exception_without_signal(fn);
    // 如果future_completed_标志尚未设置，则设置为true，并设置错误结果为eptr
    if (!future_completed_.exchange(true)) {
        future_result_->setError(std::move(eptr));
    }
}

// 调用节点fn的预处理钩子函数
static variable_list call_pre_hooks(Node& fn, variable_list inputs) {
    for (const auto& hook : fn.pre_hooks()) {
        inputs = (*hook)(inputs);
    }
    return inputs;
}

// 调用节点fn的张量预处理钩子函数和保留梯度钩子函数
static variable_list call_tensor_pre_hooks(Node& fn, variable_list inputs) {
    for (const auto& hook : fn.tensor_pre_hooks()) {
        inputs = (*hook)(inputs);
    }
    for (const auto& pair : fn.retains_grad_hooks()) {
        inputs = (*pair.second)(inputs);
    }
    return inputs;
}

// 调用节点fn的后处理钩子函数
static variable_list call_post_hooks(
    Node& fn,
    variable_list outputs,
    const variable_list& inputs) {
    for (const auto& hook : fn.post_hooks()) {
        outputs = (*hook)(outputs, inputs);
    }
    return outputs;
}

// 设置设备编号为device
void set_device(int device) {
    // 注意：必须避免构造CPU设备的保护，因为某些情况下我们编译时使用cuda，
    // 但是对CUDA功能有延迟存根（因此尝试设置CPU_DEVICE将导致错误，因为它仍然会查询GetDevice）。
    // 这里不使用DeviceGuard，因为其析构函数可能在重设设备之前被调用。
    // 这是安全的，因为设备是线程局部的。
    if (device != CPU_DEVICE) {
        for (const auto i : c10::irange(static_cast<size_t>(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES))) {
            auto* impl = c10::impl::device_guard_impl_registry[i].load();
            if (impl && device < impl->deviceCount()) {
                impl->setDevice(at::Device(
                    static_cast<c10::DeviceType>(i),
                    static_cast<c10::DeviceIndex>(device)));
            }
        }
    }
    // 设置worker_device为当前设备编号
    worker_device = device;
}

// 验证输出梯度的有效性
void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error) {
    // 如果梯度数量与边数量不匹配，则抛出错误
    if (grads.size() != edges.size()) {
        std::stringstream ss;
        ss << "invalid number of gradients - expected ";
        ss << edges.size() << ", but got " << grads.size();
        TORCH_CHECK(false, format_error(ss.str()));
    }
    // 遍历每个梯度，验证是否定义，并可能根据元数据进行减少
    for (const auto i : c10::irange(grads.size())) {
        const auto& edge = edges[i];
        if (!edge.is_valid())
            continue;

        const auto& metadata = edge.function->input_metadata(edge.input_nr);
        auto& grad = grads[i];
        if (!grad.defined()) {
            // FIXME: TestJit.test_ge_optimized fails this assertion.
            // 如果梯度未定义，则跳过，打印错误信息可能引起错误
            // std::stringstream ss;
            // ss << "undefined gradient at index " << i;
            // TORCH_CHECK(false, format_error(ss.str()));
            continue;
        }

        // 可能根据元数据减少梯度
        grad = metadata.maybe_reduce(i, std::move(grad), format_error);
    // 检查输入是否为复杂类型（如复数）
    bool input_is_complex =
        isComplexType(c10::typeMetaToScalarType(metadata.options().dtype()));
    // 检查梯度是否为复杂类型（如复数）
    bool grad_is_complex = isComplexType(grad.scalar_type());

    // 检查梯度类型是否为浮点类型，或者输入和梯度类型复杂性是否匹配
    TORCH_CHECK(
        isFloatingType(grad.scalar_type()) ||
        (input_is_complex == grad_is_complex));

    // 如果梯度的数据类型与元数据不匹配，则将梯度转换为元数据指定的数据类型
    if (c10::typeMetaToScalarType(metadata.options().dtype()) !=
        grad.scalar_type()) {
      grad = grad.to(c10::typeMetaToScalarType(metadata.options().dtype()));
    }

    // 如果梯度的数据类型与元数据的数据类型不匹配，则报告错误
    if (grad.dtype() != metadata.dtype()) {
      std::stringstream ss;
      ss << "invalid gradient at index " << i << " - expected dtype ";
      ss << metadata.dtype() << " but got " << grad.dtype();
      TORCH_CHECK(false, format_error(ss.str()));
    }

    // 如果梯度的布局与元数据的布局不匹配，则根据特定条件报告错误
    if (grad.layout() != metadata.layout()) {
      // 当前仅支持 (*, Sparse) 组合，未来可以支持更多布局组合
      if (!grad.is_sparse() &&
          !(grad.layout() == at::kStrided &&
            (at::sparse_csr::is_sparse_compressed(metadata.layout()) ||
             metadata.layout() == at::kSparse))) {
        std::stringstream ss;
        ss << "invalid gradient at index " << i << " - expected layout ";
        ss << metadata.layout() << " but got " << grad.layout();
        TORCH_CHECK(false, format_error(ss.str()));
      }
    }

    // 如果梯度的设备与元数据的设备不匹配，则根据特定条件报告错误
    if (grad.device() != metadata.device()) {
      // 对于特定情况进行快速修复，后续应移除该部分
      if (!(metadata.is_tensor_subclass() ||
            grad.unsafeGetTensorImpl()->is_python_dispatch())) {
        // 如果梯度的维度为0，则将其转换为元数据指定的设备
        if (grad.dim() == 0) {
          grad = grad.to(metadata.device());
        } else {
          std::stringstream ss;
          ss << "invalid gradient at index " << i << " - expected device ";
          ss << metadata.device() << " but got " << grad.device();
          TORCH_CHECK(false, format_error(ss.str()));
        }
      }
    }

    // 断言对于非可微类型的张量不应构建计算图
    TORCH_INTERNAL_ASSERT(isDifferentiableType(grad.scalar_type()));
  // 调用函数并返回变量列表
  static variable_list call_function(
      // 引用图任务的共享指针和节点指针，以及输入缓冲区
      std::shared_ptr<GraphTask>& graph_task,
      Node* func,
      InputBuffer& inputBuffer) {
    // 检查点有效性守卫，确保在函数执行期间图任务处于有效状态
    CheckpointValidGuard cpvguard(graph_task);
    // 获取函数引用
    auto& fn = *func;
    // 调用张量预处理钩子和输入缓冲区的变量，生成输入
    auto inputs =
        call_tensor_pre_hooks(fn, InputBuffer::variables(std::move(inputBuffer)));
    // 调用预处理钩子处理输入
    inputs = call_pre_hooks(fn, std::move(inputs));
    // 如果不需要保留图，释放函数变量
    if (!graph_task->keep_graph_) {
      fn.will_release_variables();
    }

    // 检查函数是否具有后处理钩子
    const auto has_post_hooks = !fn.post_hooks().empty();
    // 初始化输出变量列表
    variable_list outputs;

    // 如果有后处理钩子
    if (has_post_hooks) {
      // 解释：在 functions/accumulate_grad.cpp 中，有一些逻辑用于检查可以直接窃取传入梯度的条件，
      // 而不是进行克隆（省略深拷贝）。其中一个条件是传入梯度的引用计数必须为 1（没有其他引用相同数据）。
      // 在这里存储 inputs_copy 会增加引用计数，因此如果使用后处理钩子，accumulate_grad.cpp 仍然可以在引用计数为 2 时窃取梯度。
      auto inputs_copy = inputs;
      // 调用函数 fn 处理复制后的输入，并得到输出
      outputs = fn(std::move(inputs_copy));
    } else {
      // 否则直接使用原始输入调用函数 fn，并得到输出
      outputs = fn(std::move(inputs));
    }

    // 验证函数的输出是否有效
    validate_outputs(fn.next_edges(), outputs, [&](const std::string& msg) {
      std::ostringstream ss;
      ss << "Function " << fn.name() << " returned an " << msg;
      return ss.str();
    });

    // 如果存在后处理钩子
    if (has_post_hooks) {
      // NOLINTNEXTLINE(bugprone-use-after-move)
      // 调用后处理钩子处理输出，并返回处理后的结果
      return call_post_hooks(fn, std::move(outputs), inputs);
    }
    // 返回最终的输出结果
    return outputs;
  }

  // 执行函数评估
  void Engine::evaluate_function(
      // 引用图任务的共享指针，节点指针，输入缓冲区和 CPU 就绪队列的共享指针
      std::shared_ptr<GraphTask>& graph_task,
      Node* func,
      InputBuffer& inputs,
      const std::shared_ptr<ReadyQueue>& cpu_ready_queue) {
    // 获取函数的执行信息引用
    auto& exec_info_ = graph_task->exec_info_;
    // 如果执行信息不为空
    if (!exec_info_.empty()) {
      // 获取函数对应的执行信息
      auto& fn_info = exec_info_.at(func);
      // 获取输入缓冲区的新输入变量列表
      variable_list new_inputs = inputs.buffer;
      ```
    if (!fn_info.needed_) {
      // 如果不需要执行此函数，直接返回，避免重复调用
      // 设置 needed_ 为 True 表示稍后将调用张量预处理钩子
      //
      // 更多上下文请查看 NOTE [Hooks ordering]。
      new_inputs = call_tensor_pre_hooks(
          *func, InputBuffer::variables(std::move(inputs)));
    }
    if (auto* capture_vec = fn_info.captures_.get()) {
      auto opt_parent_stream = (*func).stream();
      // 锁定 mutex 以写入 graph_task->captured_vars_。
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      // 遍历捕获的变量，并将新的输入赋值给捕获的梯度变量
      for (const auto& capture : *capture_vec) {
        auto& captured_grad = graph_task->captured_vars_[capture.output_idx_];
        captured_grad = new_inputs[capture.input_idx_];
        // NOTE [Deprecated capture hooks]
        // 遍历执行捕获钩子函数对捕获的梯度进行处理
        for (const auto& hook :
             capture.DO_NOT_USE_DEPRECATED_get_capture_hooks()) {
          captured_grad = (*hook)(captured_grad);
        }
        if (opt_parent_stream) {
          // 这里不需要再获取 graph_task->mutex_，因为上面已经持有它
          // 将父流加入到 graph_task 的 leaf_streams 中
          graph_task->leaf_streams.emplace(*opt_parent_stream);
        }
      }
    }
    if (!fn_info.needed_) {
      // 如果不需要执行此函数，直接返回
      return;
    }
  }

  auto outputs = call_function(graph_task, func, inputs);

  auto& fn = *func;
  if (!graph_task->keep_graph_) {
    // 如果不需要保留计算图，释放函数的变量
    fn.release_variables();
  }

  auto num_outputs = outputs.size();
  if (num_outputs == 0) { // 注意：不会获取 mutex
    // 记录 leaf stream（如果适用）
    // 查看 Note [Streaming backwards]
    if (opt_parent_stream) {
      // 锁定 mutex 以写入 graph_task->leaf_streams。
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      graph_task->leaf_streams.emplace(*opt_parent_stream);
    }
    return;
  }

  if (AnomalyMode::is_enabled() && AnomalyMode::should_check_nan()) {
    // 关闭自动梯度模式
    AutoGradMode grad_mode(false);
    // 检查每个输出是否包含 NaN 值
    for (const auto i : c10::irange(num_outputs)) {
      auto& output = outputs[i];
      at::OptionalDeviceGuard guard(device_of(output));
      if (output.defined() && isnan(output)._is_any_true().item<bool>()) {
        std::stringstream ss;
        ss << "Function '" << fn.name() << "' returned nan values in its " << i
           << "th output.";
        throw std::runtime_error(ss.str());
      }
    }
  }

  // 锁定 mutex 以访问 GraphTask 的 dependencies_, not_ready_ 和 cpu_ready_queue_
  std::lock_guard<std::mutex> lock(graph_task->mutex_);
  for (const auto i : c10::irange(num_outputs)) {
    auto& output = outputs[i];
    const auto& next = fn.next_edge(i);

    if (!next.is_valid())
      continue;

    // 检查下一个函数是否准备好计算
    bool is_ready = false;
    auto& dependencies = graph_task->dependencies_;
    auto it = dependencies.find(next.function.get());

    if (it == dependencies.end()) {
      auto name = next.function->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    }
    } else if (--it->second == 0) {
      // Decrease the count of dependencies for the current function
      dependencies.erase(it);
      // Mark the function as ready for execution
      is_ready = true;
    }

    auto& not_ready = graph_task->not_ready_;
    auto not_ready_it = not_ready.find(next.function.get());
    if (not_ready_it == not_ready.end()) {
      // Skip functions that aren't supposed to be executed based on exec_info_
      if (!exec_info_.empty()) {
        auto it = exec_info_.find(next.function.get());
        if (it == exec_info_.end() || !it->second.should_execute()) {
          // Continue to the next function if execution is not required
          continue;
        }
      }
      // Initialize an input buffer for the function
      InputBuffer input_buffer(next.function->num_inputs());

      // Accumulate outputs into the input buffer
      auto opt_next_stream = next.function->stream();
      input_buffer.add(
          next.input_nr, std::move(output), opt_parent_stream, opt_next_stream);

      // If the function is ready for execution, push it into the ready queue
      if (is_ready) {
        auto queue = ready_queue(cpu_ready_queue, input_buffer.device());
        queue->push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
      } else {
        // Store the input buffer in the not_ready map
        not_ready.emplace(next.function.get(), std::move(input_buffer));
      }
    } else {
      // The function already has an associated input buffer
      auto& input_buffer = not_ready_it->second;

      // Accumulate outputs into the existing input buffer
      auto opt_next_stream = next.function->stream();
      input_buffer.add(
          next.input_nr, std::move(output), opt_parent_stream, opt_next_stream);

      // If the function is ready for execution, push it into the ready queue and remove from not_ready
      if (is_ready) {
        auto queue = ready_queue(cpu_ready_queue, input_buffer.device());
        queue->push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
        not_ready.erase(not_ready_it);
      }
    }
  }
}

inline static uint64_t compute_min_topological_nr(const edge_list& outputs) {
  // 计算所有输出中的最小拓扑编号
  if (outputs.empty()) {
    return 0;
  }
  // 初始化最小拓扑编号为 uint64_t 的最大值
  auto min_topo_nr = std::numeric_limits<uint64_t>::max();
  // 遍历输出的边列表
  for (auto& output_edge : outputs) {
    // 获取当前边对应的函数的拓扑编号
    auto topo_nr = output_edge.function->topological_nr();
    // 更新最小拓扑编号为当前边的拓扑编号和当前最小拓扑编号中较小的一个
    min_topo_nr = (min_topo_nr < topo_nr) ? min_topo_nr : topo_nr;
  }
  return min_topo_nr;
}

auto Engine::compute_dependencies(
    Node* root,
    GraphTask& task,
    uint64_t min_topo_nr) -> void {
  // 计算需要梯度的每个函数的依赖数
  std::vector<Node*> queue{root};
  bool will_use_accelerator = false;

  // 队列包含所有将开始传播梯度的节点。
  // 我们不再需要扩展不需要梯度的函数。
  auto& dependencies = task.dependencies_;
  while (!queue.empty()) {
    auto fn = queue.back();
    queue.pop_back();
    // 如果函数的拓扑编号小于最小拓扑编号，则跳过
    if (fn->topological_nr() < min_topo_nr) {
      continue;
    }
    // 如果未来会使用加速器，则设置标志为 true
    if (!will_use_accelerator) {
      will_use_accelerator = fn->stream().has_value();
    }
    // 遍历当前函数的下一条边
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        // 增加下一函数的依赖数
        dependencies[next_ptr] += 1;
        // 如果该函数节点是第一次插入到图中，则将其加入队列
        const bool was_inserted = task.nodes_in_graph_.insert(next_ptr).second;
        if (was_inserted)
          queue.push_back(next_ptr);
      }
    }
  }

  // 如果将使用加速器，则收集当前设备上的流，以便后续处理同步这些流与 leaf_streams。
  if (will_use_accelerator) {
    task.stash_current_streams();
  }
}

auto Engine::execute(
    const edge_list& root_edges,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) -> variable_list {
  // 验证输出，处理输入变量列表
  validate_outputs(
      root_edges,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<variable_list&>(inputs),
      [](const std::string& msg) { return msg; });
  // 如果允许梯度累积并且需要创建图
  if (accumulate_grad && create_graph) {
    // 发出警告，指出使用带有 create_graph=True 的 backward() 可能会导致参数与梯度之间的引用循环，
    // 这可能导致内存泄漏。建议在创建图时使用 autograd.grad 来避免此问题。
    // 如果必须使用此函数，请确保在使用后重置参数的 .grad 字段为 None，以打破循环并避免泄漏。
    TORCH_WARN_ONCE(
        "Using backward() with create_graph=True will create a reference cycle "
        "between the parameter and its gradient which can cause a memory leak. "
        "We recommend using autograd.grad when creating the graph to avoid this. "
        "If you have to use this function, make sure to reset the .grad fields of "
        "your parameters to None after use to break the cycle and avoid the leak.");
  }

  // 允许我们断言没有其他线程在进行反向传播
  CompiledAutogradThreadingDebugCheck _thread_check;
  auto compiled_autograd = the_compiled_autograd.load();
  // 断言编译后的自动微分不是毒药状态
  TORCH_INTERNAL_ASSERT(compiled_autograd != COMPILED_AUTOGRAD_POISON);

  // accumulate_grad 为 true 当且仅当前端调用为 backward()，而不是 grad()。
  // grad() 返回相对于输入的梯度的总和，因此需要输入不为空。
  TORCH_CHECK_VALUE(
      accumulate_grad || !outputs.empty(), "grad requires non-empty inputs.");

  // 如果是首次调用 Engine::execute，则应从 CPU 设备开始，
  // 初始化一个新的线程本地就绪队列在 CPU 上，或者重用已分配的队列（如果已有，即连续的反向传播调用，可重入的反向传播调用），
  // 然后在 GraphTask 中将本地就绪队列进行记忆化
  init_local_ready_queue();
  bool not_reentrant_backward_call = worker_device == NO_DEVICE;

  // 存储根节点，以便稍后通过图进行遍历，例如用于 get_current_graph_task_execution_order
  c10::SmallVector<Node*, 4> temp_roots{root_edges.size()};
  for (const auto i : c10::irange(root_edges.size())) {
    temp_roots[i] = root_edges[i].function.get();
  }

  // 创建 GraphTask 的共享指针，配置其参数
  auto graph_task = std::make_shared<GraphTask>(
      /* keep_graph */ keep_graph,
      /* create_graph */ create_graph,
      /* depth */ not_reentrant_backward_call ? 0 : total_depth + 1,
      /* cpu_ready_queue */ local_ready_queue,
      /* graph_roots */ std::move(temp_roots));

  // 如果只有一个根节点，则跳过创建额外的根节点
  bool skip_dummy_node = root_edges.size() == 1 && compiled_autograd == nullptr;
  auto graph_root = skip_dummy_node
      ? root_edges.at(0).function
      : std::make_shared<GraphRoot>(root_edges, inputs);

  // 计算最小拓扑编号，用于计算所有可执行函数的依赖关系
  auto min_topo_nr = compute_min_topological_nr(outputs);
  // 现在为所有可执行函数计算依赖关系
  compute_dependencies(graph_root.get(), *graph_task, min_topo_nr);

  // 如果输出不为空，则初始化执行的 GraphTask
  if (!outputs.empty()) {
    graph_task->init_to_execute(
        *graph_root, outputs, accumulate_grad, min_topo_nr);
  }

  // 如果 compiled_autograd 不为 nullptr，则执行以下操作
  if (compiled_autograd != nullptr) {
    // 参见 [Note: Compiled Autograd]
    TORCH_CHECK(
        !create_graph, "compiled_autograd does not support create_graph");
    _thread_check.release();
    TORCH_CHECK(
        !AnomalyMode::is_enabled(),
        "compiled_autograd does not support AnomalyMode")
    return (*compiled_autograd)(
        graph_root, *graph_task, accumulate_grad, outputs);

# 调用编译后的自动求导函数指针，执行自动求导计算。

  // Queue the root
  if (skip_dummy_node) {
    InputBuffer input_buffer(root_edges.at(0).function->num_inputs());
    auto input = inputs.at(0);

    const auto input_stream = InputMetadata(input).stream();
    auto opt_next_stream = root_edges.at(0).function->stream();
    input_buffer.add(
        root_edges.at(0).input_nr,
        std::move(input),
        input_stream,
        opt_next_stream);

    execute_with_graph_task(
        graph_task, std::move(graph_root), std::move(input_buffer));
  } else {
    execute_with_graph_task(
        graph_task, std::move(graph_root), InputBuffer(variable_list()));
  }

# 如果需要跳过虚拟节点，创建输入缓冲区并加入输入，然后执行带有图任务的执行。
# 否则，使用变量列表创建空输入缓冲区并执行带有图任务的执行。

  // Avoid a refcount bump for the Future, since we check for refcount in
  // DistEngine (see TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1)
  // in dist_engine.cpp).
  auto& fut = graph_task->future_result_;
  fut->wait();
  graph_task->warning_handler_.replay_warnings();

# 避免对未来结果的引用计数增加，因为我们在 DistEngine 中检查引用计数。
# 参见 dist_engine.cpp 中的 TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1)。

  return fut->value().toTensorVector();

# 返回未来结果的值作为张量向量。
}

// 初始化设备线程池
void Engine::initialize_device_threads_pool() {
  // 检查是否处于不良的自动求导分支中，不支持自动求导线程与基于fork的多进程混合使用
  TORCH_CHECK(
      !in_bad_autograd_fork,
      "Unable to handle autograd's threading in combination with fork-based multiprocessing. "
      "See https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork");
  
  // 仅调用一次，启动设备线程池
  c10::call_once(
      start_device_threads_flag_, &Engine::start_device_threads, this);
}

// 使用图任务执行操作
c10::intrusive_ptr<at::ivalue::Future> Engine::execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {
  // 初始化设备线程池
  initialize_device_threads_pool();
  
  // 锁定图任务的互斥锁
  std::unique_lock<std::mutex> lock(graph_task->mutex_);
  
  // 获取准备就绪的队列，根据输入缓冲区的设备类型选择对应的队列
  auto queue = ready_queue(graph_task->cpu_ready_queue_, input_buffer.device());

  // 如果工作设备为NO_DEVICE，则说明是CPU线程尝试驱动自动求导引擎的图任务，并且不是重入调用
  if (worker_device == NO_DEVICE) {
    // 将工作设备设置为CPU_DEVICE，仅在之前工作设备为NO_DEVICE时设置为CPU，以便检测是否为重入调用
    set_device(CPU_DEVICE);

    // 设置图任务的所有者为当前设备
    graph_task->owner_ = worker_device;

    // 现在图任务的所有非线程安全字段都已填充，可以将其入队
    queue->push(
        NodeTask(graph_task, std::move(graph_root), std::move(input_buffer)));

    // 拥有线程开始驱动引擎执行任何刚刚推送的或稍后从其他工作线程添加的CPU任务
    lock.unlock();
    thread_main(graph_task);
    // 断言图任务的未来结果已完成
    TORCH_INTERNAL_ASSERT(graph_task->future_result_->completed());

    // 在图任务完成后重置工作设备，以确保引擎的初始状态在每次backward()或grad()调用中保持一致
    worker_device = NO_DEVICE;
  } else {
    // 如果工作设备是任何设备（例如CPU、CUDA），则这是来自该设备的重入backward调用
    graph_task->owner_ = worker_device;

    // 现在图任务的所有非线程安全字段都已填充，可以将其入队
    queue->push(
        NodeTask(graph_task, std::move(graph_root), std::move(input_buffer)));

    // 如果当前深度达到最大递归深度
    if (current_depth >= max_recursion_depth_) {
      // 参见注释[Reentrant backwards]
      // 如果达到最大深度，则切换到另一个线程池任务
      add_thread_pool_task(graph_task);
    } else {
      // 在这个代码路径中，需要更新总深度，因为在上面的代码块中（调用 add_thread_pool_task 时）不使用它。
      // 在上述代码路径中，GraphTask.reentrant_depth_ 用于启动另一个线程中的 total_depth。
      ++total_depth;

      // 继续工作，同时等待新的 graph_task 完成！
      ++current_depth;
      lock.unlock();
      // 调用线程主函数执行 graph_task
      thread_main(graph_task);
      --current_depth;
      --total_depth;

      // 由于 'thread_main' 是一个阻塞 autograd 引擎线程的调用，图任务应该已经完成，
      // 相关的 future_result_ 也应该被标记为完成。
      TORCH_INTERNAL_ASSERT(graph_task->future_result_->completed());
    }
  }
  // 当 Future 在 mark_as_completed_and_run_post_processing 中标记为完成时，graph_task_exec_post_processing 就完成了。
  return graph_task->future_result_;
}

// 注意：当 Python 存在时，此基本引擎将被 PythonEngine 覆盖。
// 因为这通常发生在调用 get_default_engine 之前，所以此基本引擎将永远不会被创建。

// 获取基本引擎的静态方法
Engine& Engine::get_base_engine() {
  static Engine engine;
  return engine;
}

// 创建原子变量 engine_stub 并初始化为基本引擎
std::atomic<EngineStub> engine_stub(Engine::get_base_engine);

// 设置默认引擎的存根
void set_default_engine_stub(EngineStub stub) {
  engine_stub.store(stub);
}

// 获取默认引擎的方法
Engine& Engine::get_default_engine() {
  return engine_stub.load()();
}

// 设置编译后自动求导的函数
void Engine::set_compiled_autograd(Engine::compiled_autograd_fn fn) {
  // 如果当前的编译后自动求导函数与新函数相同，则直接返回
  if (the_compiled_autograd.load() == fn) {
    return;
  }
  // 尝试交换当前的编译后自动求导函数并进行检查
  auto prior = the_compiled_autograd.exchange(COMPILED_AUTOGRAD_POISON);
  TORCH_CHECK(
      num_threads_in_backwards.load() == 0 && prior != COMPILED_AUTOGRAD_POISON,
      "compiled_autograd.enable() requires no threads in backwards()");
  // 存储新的编译后自动求导函数
  the_compiled_autograd.store(fn);
}

// 将回调函数加入到队列中
void Engine::queue_callback(std::function<void()> callback) {
  TORCH_CHECK(
      current_graph_task,
      "Final callbacks can only be installed during backward pass.");

  // 使用互斥锁保护对最终回调列表的访问
  std::lock_guard<std::mutex> lock(current_graph_task->final_callbacks_lock_);
  current_graph_task->final_callbacks_.emplace_back(std::move(callback));
}

// 检查检查点是否有效
bool Engine::is_checkpoint_valid() {
  return checkpoint_valid;
}

// 初始化本地准备队列
void Engine::init_local_ready_queue(std::shared_ptr<ReadyQueue> ready_queue) {
  if (ready_queue) {
    // 如果调用方提供了准备队列，则使用其提供的准备队列初始化本地准备队列
    local_ready_queue = std::move(ready_queue);
  } else if (!local_ready_queue) {
    // 否则，如果本地准备队列尚未分配，则分配一个新的准备队列
    local_ready_queue = std::make_shared<ReadyQueue>();
  }
}

// 根据设备返回对应的准备队列
// CPU 的准备队列是针对每个 GraphTask 的，而 CUDA 设备的准备队列是共享的
auto Engine::ready_queue(
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    at::Device device) -> std::shared_ptr<ReadyQueue> {
  bool multithreading_disabled =
      !c10::AutogradState::get_tls_state().get_multithreading_enabled();
  if (multithreading_disabled || should_run_in_cpu_ready_queue(device.type())) {
    // 如果多线程被禁用或者应该在 CPU 准备队列中运行，则返回传入的 CPU 准备队列
    TORCH_INTERNAL_ASSERT(cpu_ready_queue);
    return cpu_ready_queue;
  } else {
    TORCH_INTERNAL_ASSERT(
        0 <= device.index() &&
        device.index() <
            static_cast<c10::DeviceIndex>(device_ready_queues_.size()));
    // 参见注释 [为 autograd 线程分配 GPU]
    return device_ready_queues_.at(device.index());
  }
}

// 根据索引返回对应的准备队列
auto Engine::ready_queue_by_index(
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    int device_index) -> std::shared_ptr<ReadyQueue> {
  if (device_index == CPU_DEVICE) {
    // 如果索引为 CPU_DEVICE，则返回传入的 CPU 准备队列
    TORCH_INTERNAL_ASSERT(cpu_ready_queue);
    return cpu_ready_queue;
  } else {
    # 使用 TORCH_INTERNAL_ASSERT 宏来确保设备索引 device_index 的有效性
    TORCH_INTERNAL_ASSERT(
        0 <= device_index &&
        device_index <
            static_cast<c10::DeviceIndex>(device_ready_queues_.size()));
    # 查看注释 [Allocating GPUs to autograd threads]
    # 注意：如果我们真正为每个设备分配一个 CPU 线程而不是共享一个线程，这个函数可能会变得过时。
    # 返回 device_ready_queues_ 中索引为 device_index 的元素
    return device_ready_queues_.at(device_index);
}

auto Engine::start_device_threads() -> void {
  // 首先，初始化线程池以支持可重入线程
  thread_pool_shared_ = std::make_shared<ThreadPoolShared>();

  // 接着，为每个非 CPU 设备创建特殊线程
  // 参见注释 [分配 GPU 给自动求导线程]
  c10::DeviceIndex num_devices = 0;
  for (const auto& impl_atomic : c10::impl::device_guard_impl_registry) {
    auto* impl = impl_atomic.load();
    // 只记录不运行在 CPU 准备队列上的设备数量
    if (impl && !should_run_in_cpu_ready_queue(impl->type())) {
      num_devices = std::max(num_devices, impl->deviceCount());
    }
  }

  // 如果没有除 CPU 外的设备，无需创建工作线程
  if (num_devices == 0) {
    return;
  }

  // 由于即将创建线程，因此不再允许分叉操作
  track_bad_autograd_forks();

  // 为每个 GPU 设备分配一个线程（但同类型 GPU 放置在一起），并预分配 device_ready_queues_，
  // 以确保在其上进行安全读取操作。
  device_ready_queues_ = std::vector<std::shared_ptr<ReadyQueue>>(num_devices);
  for (auto& queue : device_ready_queues_) {
    queue = std::make_shared<ReadyQueue>();
  }

  for (const auto i : c10::irange(num_devices)) {
    // 启动线程并初始化
    std::thread t(&Engine::thread_init, this, i, device_ready_queues_[i], true);
    t.detach();
  }
  // 等待线程启动
  {
    std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
    while (non_reentrant_device_thread_count_.load() !=
           static_cast<uint32_t>(num_devices)) {
      non_reentrant_device_thread_condvar_.wait(lk);
    }
  }
}

void Engine::add_thread_pool_task(const std::weak_ptr<GraphTask>& graph_task) {
  std::unique_lock<std::mutex> lck(thread_pool_shared_->mutex_);
  // 可能已有一些图任务在 graphtasks_queue_ 中，由其他线程添加，但还没有足够的工作者来处理新任务
  bool create_thread =
      (thread_pool_shared_->num_workers_ <=
       thread_pool_shared_->graphtasks_queue_.size());
  thread_pool_shared_->graphtasks_queue_.push(graph_task);
  // 在创建线程时不需要持有锁
  lck.unlock();
  if (create_thread) {
    // 如果正在创建新线程，不再允许分叉操作
    track_bad_autograd_forks();
    // 启动可重入线程初始化
    std::thread t(&Engine::reentrant_thread_init, this);
    t.detach();
  }
  // 即使创建了新线程，此处仍然有效，因为 wait() 在等待之前会测试条件
  thread_pool_shared_->work_.notify_one();
}

// 记录已为所有创建上下文的设备上的当前流
// 此函数假设加速器设备是可用的。
// 保存当前的流状态，用于稍后恢复
void GraphTask::stash_current_streams() {
  // 获取当前的加速器（硬件加速器），例如 GPU
  const auto accelerator = at::getAccelerator(true).value();
  // 创建一个虚拟的保护器对象，保护当前线程中的设备
  const auto guard = c10::impl::VirtualGuardImpl{accelerator};
  // 获取当前设备数量
  auto num_devices = guard.deviceCount();
  // 调整 caller_current_streams_ 的大小以存储设备的流状态
  caller_current_streams_.resize(num_devices);
  // 如果存在设备
  if (num_devices > 0) {
    // 遍历每个设备
    for (c10::DeviceIndex idx = 0; idx < num_devices; idx++) {
      // 检查设备是否有主要上下文
      if (at::globalContext().getAcceleratorHooksInterface().hasPrimaryContext(
              idx)) {
        // 获取设备上的流，并存储到 caller_current_streams_
        caller_current_streams_[idx] = guard.getStream({accelerator, idx});
      } else {
        // 如果设备没有主要上下文，则流状态置为 null
        caller_current_streams_[idx] = c10::nullopt;
      }
    }
  }
}

// 初始化执行图任务，设置执行所需信息
void GraphTask::init_to_execute(
    Node& graph_root,
    const edge_list& outputs,
    bool accumulate_grad,
    uint64_t min_topo_nr) {
  // 填充 exec_info_，标记需要执行的节点为 true，仅对与 outputs 有路径的节点有效
  // 以下代码通过递归方式填充 exec_info_
  // 实际代码中采用迭代方式执行，详细见后面的编号说明
  //
  // is_needed = {fn: True for fn in outputs}             // (0)
  // seen = {}
  // def compute_is_needed(fn):
  //   for next_edge in fn.next_edges:
  //     child_fn = next_edge.fn
  //     if child_fn in seen and is_needed[child_fn]:     // (1)
  //       is_needed[fn] = true
  //     else:
  //       seen.add(child_fn)
  //       if compute_is_needed(child_fn):
  //         is_needed[fn] = true                         // (2)
  //                                                      // (3) exit for-loop
  //   return is_needed[fn]
  // compute_is_needed(graph_root)
  //
  // 注意：不能用 outputs 初始化 seen，因为两个输出可能在同一路径上，需要继续探索以获取所有需要的节点。
  
  int output_idx = 0;
  // 遍历每个输出边
  for (auto& output_edge : outputs) {
    // (0) 上述 is_needed 对应于 exec_info_[fn].needed_
    // 获取输出节点
    Node* output = output_edge.function.get();
    // 获取与该输出节点相关的执行信息
    auto& info = exec_info_[output];
    // 如果需要累积梯度信息
    if (accumulate_grad) {
      // 如果通过 `.backward()` 调用，直接设置节点的 needed_ 为 true
      info.needed_ = true;
    } else {
      // 否则通过 `.grad()` 调用，设置 exec_info_[fn].captures_ 而不是 needed_
      // 在填充 exec_info_ 的其他部分时，可以将其视为直接设置 needed_ 为 true
      if (!info.captures_) {
        info.captures_ = std::make_unique<std::vector<ExecInfo::Capture>>();
      }
      // 存储输出边的输入编号和输出索引到 captures_
      info.captures_->emplace_back(output_edge.input_nr, output_idx++);
    }
  }
}
  }
  }
  // 调整 captured_vars_ 的大小，以匹配输出索引
  captured_vars_.resize(output_idx);

  // 定义帧结构体，用于存储节点信息
  struct Frame {
    Frame(Node* fn) : fn_(fn) {}
    Node* fn_{};
    size_t next_next_fn_{};

    // 获取下一个需要执行的节点
    Node* get_next_fn() {
      const auto& next = fn_->next_edges();
      auto num_next = next.size();
      while (next_next_fn_ < num_next) {
        auto fn = next[next_next_fn_++].function.get();
        if (fn)
          return fn;
      }
      return nullptr;
    }
  };

  // lambda 函数，用于检查节点是否需要执行
  auto nodeShouldExecute = [this](Node* fn) {
    auto it = exec_info_.find(fn);
    return it != exec_info_.end() && it->second.should_execute();
  };

  // 用于存储帧的栈和已经遍历过的节点集合
  std::vector<Frame> stack;
  std::unordered_set<Node*> seen;
  // 将根节点添加到栈中，并标记其需要执行
  stack.emplace_back(&graph_root);
  exec_info_.emplace(stack.back().fn_, ExecInfo());

  // 开始深度优先搜索执行节点
  while (!stack.empty()) {
    auto& frame = stack.back();
    const auto fn = frame.fn_;

    Node* child_fn = nullptr;
    // 遍历节点的下一个待执行节点，直到所有节点都被遍历过
    while ((child_fn = frame.get_next_fn()) && !seen.emplace(child_fn).second) {
      // (1) 下一个子节点存在且已经被遍历过
      if (nodeShouldExecute(child_fn)) {
        exec_info_[fn].needed_ = true;
      }
    }

    if (child_fn) {
      // (2) 下一个子节点存在但尚未被遍历过
      if (child_fn->topological_nr() < min_topo_nr) {
        // 若子节点在第一个输出之前创建，则该子节点不可能连接到输出
        continue;
      }
      // 将子节点压入栈中，准备继续遍历其子节点
      stack.emplace_back(child_fn);
    } else {
      // (3) `fn` 没有下一个子节点，意味着其 `needed` 属性已经被确定，可以出栈并更新父节点
      stack.pop_back();
      // 若 `fn` 需要执行且栈不为空，则更新其父节点的 `needed` 属性
      if (nodeShouldExecute(fn) && !stack.empty()) {
        exec_info_[stack.back().fn_].needed_ = true;
      }
    }
  }
}

} // namespace torch::autograd


// 关闭 torch::autograd 命名空间
```