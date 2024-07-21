# `.\pytorch\torch\csrc\autograd\engine.h`

```
#pragma once

// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

#include <ATen/Tensor.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>
#include <torch/csrc/autograd/utils/warnings.h>

#include <c10/util/CallOnce.h>

#include <exception>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace torch::autograd {
struct ReadyQueue;
}

namespace torch::autograd {

// Maximum reentrant backward depth before switching to a new thread
// This limit is based on the TSAN's deadlock detector, where it will
// fail if a program hold more than 65 locks in one thread at once.
// As we hold mutex in every of our custom C++ autograd Node, we would
// like to avoid TSAN complains on this when doing reentrant backwards
// For reference, see https://github.com/google/sanitizers/issues/950
static constexpr int MAX_DEPTH = 60;

// 设置设备的函数声明
void set_device(int device);

// 验证输出的有效性，对梯度进行验证并格式化错误消息
TORCH_API void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error);

// NodeTask 结构体，用于表示一个节点任务
struct NodeTask {
  std::weak_ptr<GraphTask> base_; // 指向任务图的弱指针
  std::shared_ptr<Node> fn_; // 指向节点函数的共享指针
  // 用作所有流向此处的梯度的隐式“加法”节点的缓冲区。一旦所有依赖项完成，我们将使用此缓冲区的内容运行函数。
  InputBuffer inputs_;
  // 当工作线程接收到 isShutdownTask = true 的任务时，它将立即退出。引擎在销毁时向每个队列发送关闭任务。
  bool isShutdownTask_;

  // 获取当前任务的重新进入深度
  int getReentrantDepth() const;

  // NodeTask 的构造函数
  NodeTask(
      std::weak_ptr<GraphTask> base,
      std::shared_ptr<Node> fn,
      InputBuffer inputs,
      bool isShutdownTask = false)
      : base_(std::move(base)),
        fn_(std::move(fn)),
        inputs_(std::move(inputs)),
        isShutdownTask_(isShutdownTask) {}
};

// 用于设置和恢复检查点有效性的保护类
class CheckpointValidGuard {
 public:
  explicit CheckpointValidGuard(
      const std::shared_ptr<const GraphTask>& graph_task); // 构造函数
  ~CheckpointValidGuard(); // 析构函数

 private:
  bool prev_checkpoint_valid_state; // 前一个检查点有效性状态
};

// ReadyQueue 结构体，表示准备队列
struct ReadyQueue {
 private:
  // 用于比较 NodeTask 时间顺序的比较器结构
  // 返回 true 表示 t2 应该在队列中 t1 之前（弱顺序）
  // 首先是关闭任务，然后是空的 NodeTask
  struct CompareNodeTaskTime {
      // 比较函数的运算符重载
      bool operator()(const std::shared_ptr<NodeTask>& t1, const std::shared_ptr<NodeTask>& t2) const;
  };
    bool operator()(NodeTask const& t1, NodeTask const& t2) {
      // 自定义的比较运算符，用于比较两个 NodeTask 对象的优先级
      // NOLINTNEXTLINE(bugprone-branch-clone)
      // 如果 t2 是关机任务，优先级高，返回 true
      if (t2.isShutdownTask_) {
        return true;
      } else if (!t1.fn_ || t1.isShutdownTask_) {
        // 如果 t1 没有有效的函数或者 t1 是关机任务，则 t1 优先级低，返回 false
        return false;
      } else if (!t2.fn_) {
        // 如果 t2 没有有效的函数，t2 优先级低，返回 true
        return true;
      } else if (t1.getReentrantDepth() == t2.getReentrantDepth()) {
        // 如果 t1 和 t2 的递归深度相同，比较它们关联函数的序列号，序列号小的优先级高，返回 true
        return t1.fn_->sequence_nr() < t2.fn_->sequence_nr();
      } else {
        // 否则比较它们的递归深度，递归深度小的优先级高，返回 true
        return t1.getReentrantDepth() < t2.getReentrantDepth();
      }
    }
  };

  // 用于通知等待在 ReadyQueue 上的线程，heap_ 中有可用任务
  std::condition_variable not_empty_;
  // 用于保护对 heap_ 的读写操作
  mutable std::mutex mutex_;

  // 使用 CompareNodeTaskTime 比较器的优先队列，存储 NodeTask 对象
  std::priority_queue<NodeTask, std::vector<NodeTask>, CompareNodeTaskTime>
      heap_;

 public:
  // 将任务 item 推入堆中
  // incrementOutstandingTasks 表示是否应该增加与 GraphTask 关联的 outstanding_tasks_
  // 大多数情况下应该为 true，只在某些情况下为 false（见 DistEngine.execute_graph_task_until_ready_queue_empty 的文档）
  void push(NodeTask item, bool incrementOutstandingTasks = true);
  
  // 将关机任务推入堆中
  void pushShutdownTask();
  
  // 从堆中弹出一个任务
  NodeTask pop();
  
  // 检查堆是否为空
  bool empty() const;
  
  // 返回堆中任务的数量
  size_t size() const;
};

// 这是一个结构体定义，表示 Torch 引擎的实例。整个进程生命周期中只应创建一个实例。
// 工作线程创建逻辑和引擎的析构依赖于此结构体。
struct TORCH_API Engine {
  /// 返回静态的 Engine 实例的引用。
  static Engine& get_default_engine();

  static Engine& get_base_engine();

  // compiled_autograd 需要存在于不同的 .so 文件中，以便具有 Python 符号，因此我们增加了一层间接性。
  // 参见 [Note: Compiled Autograd]
  typedef variable_list (*compiled_autograd_fn)(
      const std::shared_ptr<Node>& graph_root,
      GraphTask& graph_task,
      bool accumulate_grad,
      const edge_list& outputs);
  static void set_compiled_autograd(compiled_autograd_fn fn);

  // 构造函数被删除，禁止复制构造 Engine 实例
  Engine(const Engine&) = delete;
  // 移动构造函数被删除，禁止移动构造 Engine 实例
  Engine(Engine&&) = delete;
  // 虚析构函数，用于确保子类对象正确释放资源
  virtual ~Engine();

  // 根据给定的 (Node, 输入编号) 对列表计算图的值，通过跟随 next_edge 引用进行计算。
  virtual variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      bool accumulate_grad,
      const edge_list& outputs = {});

  // 给定预填充的 GraphTask 和 GraphRoot，计算图的反向传播。
  //
  // 注意：此 API 仅应由内部自动求导特定机制使用，并且不应以任何方式向用户公开。
  virtual c10::intrusive_ptr<at::ivalue::Future> execute_with_graph_task(
      const std::shared_ptr<GraphTask>& graph_task,
      std::shared_ptr<Node> graph_root,
      InputBuffer&& input_buffer);

  // 创建一个异常元数据的唯一指针，用于捕获和处理异常信息。
  virtual std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() {
    return std::make_unique<AnomalyMetadata>();
  }

  // 获取默认的 SavedVariableHooks 的唯一指针，用于处理保存的变量钩子。
  virtual std::unique_ptr<SavedVariableHooks> get_default_saved_variable_hooks() {

    return std::make_unique<SavedVariableHooks>();
  }
    // 返回空指针，通常用于表示无效的返回值或异常情况
    return nullptr;
    }
    
    // 将 cpu_ready_queue 传递给 evaluate_function，以便它知道在 NodeTask 就绪后要推送到的正确就绪队列
    void evaluate_function(
        std::shared_ptr<GraphTask>& graph_task,
        Node* func,
        InputBuffer& inputs,
        const std::shared_ptr<ReadyQueue>& cpu_ready_queue);
    
    // 初始化设备线程池
    void initialize_device_threads_pool();
    
    // 线程异常处理函数，在发生异常时调用，传递图任务和函数节点以及异常对象
    virtual void thread_on_exception(
        std::shared_ptr<GraphTask> graph_task,
        const std::shared_ptr<Node>& fn,
        std::exception& e);
    
    // 将回调函数加入队列，以便稍后执行
    void queue_callback(std::function<void()> callback);
    
    // 检查检查点是否有效
    bool is_checkpoint_valid();
    
    // 在 fork 后调用，通知工作线程已终止
    void release_workers();
    
    // 在销毁之前必须由子类调用，避免 vptr 数据竞争
    void stop();
    
    // 初始化用于 autograd 引擎的设备线程
    virtual void thread_init(
        int device,
        const std::shared_ptr<ReadyQueue>& ready_queue,
        bool should_increment = true);
    
    protected:
    Engine();
    void compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr);
    
    // 使用在其他地方创建的就绪队列（例如 thread_init、Engine::execute 等）来初始化线程本地的就绪队列，
    // 如果没有提供 ready_queue，则创建一个新的就绪队列
    void init_local_ready_queue(
        std::shared_ptr<ReadyQueue> ready_queue = nullptr);
    
    // 返回特定设备的就绪队列
    std::shared_ptr<ReadyQueue> ready_queue(
        std::shared_ptr<ReadyQueue> cpu_ready_queue,
        at::Device device);
    
    // 根据设备索引返回就绪队列
    std::shared_ptr<ReadyQueue> ready_queue_by_index(
        std::shared_ptr<ReadyQueue> cpu_ready_queue,
        int device_index);
    
    // 启动设备线程（如 CUDA、XLA 等），不包括 CPU 线程
    void start_device_threads();
    
    // 增加非可重入线程计数
    void increment_non_reentrant_thread_count();
    
    // 减少非可重入线程计数
    void decrement_non_reentrant_thread_count();
    
    // 线程主函数，用于执行图任务
    virtual void thread_main(const std::shared_ptr<GraphTask>& task);
    
    // 初始化可重入线程
    void reentrant_thread_init();
    
    // 向线程池添加任务
    void add_thread_pool_task(const std::weak_ptr<GraphTask>& graph_task);
    
    // 确保 device_ready_queues_ 仅被初始化一次
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    c10::once_flag start_device_threads_flag_;
    
    // 初始化后可以在没有同步的情况下安全地读取 device_ready_queues_
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::vector<std::shared_ptr<ReadyQueue>> device_ready_queues_;
    
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::vector<std::function<void()>> final_callbacks_;
    
    // 保护对 final_callbacks_ 的读写的互斥锁
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::mutex post_callbacks_lock_;
    
    // 允许多少个嵌套可重入调用，直到使用新线程
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    int max_recursion_depth_;
    
    struct ThreadPoolShared {
    // 执行可重入反向任务的线程使用的数据结构。参见注释 [Reentrant backwards]
    // 可用于处理新 GraphTask 的线程数量
    unsigned int num_workers_{0};
    
    // 线程将等待在 work_ 上，以便通过通知来接收 GraphTask
    std::condition_variable work_;
    
    // 用于保护对 graphtasks_queue_ 和 num_workers_ 的读写操作，并在需要时同步创建新线程
    std::mutex mutex_;
    
    // Workers 将处理添加到此队列的 GraphTask。GraphTask 在 Engine::execute 中分配，并在执行期间存在
    std::queue<std::weak_ptr<GraphTask>> graphtasks_queue_;
    
    ThreadPoolShared() = default;
    };
    
    // 在关闭线程完成之前的临时解决方案
    // 我们需要对所有这些对象进行共享所有权，因为在 Engine 关闭时线程可能会泄漏，
    // 因此可能有线程在等待 work_，以使 graphtasks_queue_ 不为空。
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::shared_ptr<ThreadPoolShared> thread_pool_shared_;
    
    private:
    // 非可重入线程的数量
    std::atomic<uint32_t> non_reentrant_device_thread_count_;
    
    // 析构函数将等待非可重入线程完成
    std::condition_variable non_reentrant_device_thread_condvar_;
    std::mutex non_reentrant_device_thread_mutex_;
    
    // 在销毁路径下降到基类之前必须调用 stop()，以避免在 vptr 上的数据竞争。
    // 使用此布尔值来保护是否已调用 stop()，以便在类层次结构的每个析构函数中调用它。
    bool stopped_{false};
};

// 结束 torch::autograd 命名空间

// 允许 python_engine 在加载时覆盖默认的引擎
using EngineStub = Engine& (*)();

// 设置默认引擎的存根函数，允许通过 python_engine 覆盖默认引擎
TORCH_API void set_default_engine_stub(EngineStub stub);

} // namespace torch::autograd
```