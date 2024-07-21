# `.\pytorch\torch\csrc\autograd\graph_task.h`

```
#pragma once
#include <ATen/ThreadLocalState.h>
#include <ATen/core/Tensor.h>
#include <c10/util/ThreadLocal.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/autograd/utils/warnings.h>
#include <vector>

// 定义命名空间 torch::autograd
namespace torch::autograd {

// 使用别名 edge_list 表示 std::vector<Edge>
using edge_list = std::vector<Edge>;

// 定义常量 NO_DEVICE 和 CPU_DEVICE 分别表示无设备和 CPU 设备
static constexpr int NO_DEVICE = -2;
static constexpr int CPU_DEVICE = -1;

// GraphTask 包含执行 backward() 所需的元数据
struct GraphTask : std::enable_shared_from_this<GraphTask> {
  // 记录未完成的任务数，使用原子类型保证多线程安全
  std::atomic<uint64_t> outstanding_tasks_{0};

  // 标志在执行任务时是否出错，当为 true 时通知所有线程停止执行
  std::atomic_bool has_error_{false};

  // 标志 future 是否完成，原子操作确保多线程安全
  std::atomic_bool future_completed_{false};

  // 是否保持图的计算过程，可以不使用同步机制进行读取
  bool keep_graph_;

  // 用于保护 not_ready_, dependencies_, captured_vars_,
  // has_error_, future_result_, cpu_ready_queue_, and leaf_streams 的互斥锁
  std::mutex mutex_;

  // 记录节点到输入缓冲区的映射，无序映射容器
  std::unordered_map<Node*, InputBuffer> not_ready_;

  // 记录节点到依赖计数的映射，无序映射容器
  std::unordered_map<Node*, int> dependencies_;

  // 记录图中存在的节点，无序集合容器
  std::unordered_set<Node*> nodes_in_graph_;

  // 存储图的根节点，使用 c10::SmallVector 容器存储最多四个节点
  c10::SmallVector<Node*, 4> graph_roots_;

  // 执行信息结构体
  struct ExecInfo {
    // 默认构造函数产生 "default" 模式的执行信息
    // 在这种模式下，所有遇到的 next_edges 都应该被执行
    // 当 exec_info 为空时，表示通过 .backward() 执行图，且未传递 inputs 参数
    // 否则，通过 .grad() 执行，或者通过 .backward() 传递了 inputs 参数，exec_info 将非空
    ExecInfo() {}

    // 声明和定义一个位复杂的语义，决定是否执行图的某些路径
    // 仅当 exec_info 不为空时，有需要的 entry 才应该被执行
    bool needed;
  };

  // exec_info 是 GraphTask 的一部分，决定执行哪些路径的语义
  ExecInfo exec_info;
};
    // 定义名为 Capture 的结构体，用于管理捕获的信息
    struct Capture {
      // 禁用复制构造函数
      Capture(const Capture&) = delete;
      // 启用移动构造函数
      Capture(Capture&&) = default;

      // 构造函数，初始化输入索引和输出索引
      Capture(int input_idx, int output_idx)
          : input_idx_(input_idx), output_idx_(output_idx) {}
      
      // 在节点输入中的索引
      int input_idx_;
      // 在图任务输出向量中的索引
      int output_idx_;

      // 梯度捕获钩子，用于替换捕获的梯度
      struct GradCaptureHook {
        virtual ~GradCaptureHook() = default;
        virtual at::Tensor operator()(const at::Tensor& grad) = 0;
      };

      // NOTE [Deprecated capture hooks]
      //
      // 捕获钩子的当前状态是，我们继续支持其在分布式 dist_engine 中的单一用途。
      // 如果其他人需要将其用于其他目的，他们应该提交一个问题报告。
      //
      // 捕获钩子最初创建是因为没有一种方法可以注册到 grad_fn 中的
      // 前/后钩子，以便在传递为 input= 的 Tensor 的 grad_fn 仍然执行时，
      // 它仍然会执行。据我所知，只有 dist_engine 使用了这个钩子。
      //
      // 然而，今天存在其他替代方案，如张量钩子，可以替代最初激发其创建的用法。
      // 此外，捕获钩子在 autograd 提供的钩子类型中是一个异常，在其注册和行为方面，
      // 例如，它是注册到特定 graph_task 而不是图本身的钩子！这使得它很难维护。
      //
      // 清理/从分布式中进行迁移以使用张量钩子将是非常好的，但目前我们只是将此方法标记为
      // 弃用以防止额外的使用。
      //
      // 如果您仍然认为您确实需要捕获钩子，请提交一个问题报告（并标记 autograd）。
      const std::vector<std::unique_ptr<GradCaptureHook>>&
      DO_NOT_USE_DEPRECATED_get_capture_hooks() const {
        return hooks_;
      }

      // 查看 NOTE [deprecated capture hooks]
      void DO_NOT_USE_DEPRECATED_register_capture_hook(
          std::unique_ptr<GradCaptureHook> hook) {
        hooks_.push_back(std::move(hook));
      }

     private:
      // 钩子将按添加的顺序依次调用。
      // 钩子的输入 grad 将是其前一个钩子的输出。
      // 第一个钩子将接收捕获的 grad 作为输入。
      // 最后一个钩子的输出将替换捕获的 grad。
      std::vector<std::unique_ptr<GradCaptureHook>> hooks_;
    };

    // 返回是否应执行的布尔值，基于 needed_ 或 captures_ 的状态
    bool should_execute() const {
      return needed_ || captures_;
    }

    // 是否需要的标志，默认为 false
    bool needed_ = false;
    // 使用 unique_ptr 管理捕获对象的指针，这些对象是捕获的变量列表
    std::unique_ptr<std::vector<Capture>> captures_;
  };
  // exec_info_ 可以在没有同步的情况下安全读取
  std::unordered_map<Node*, ExecInfo> exec_info_;
  // captured_vars_ 是捕获的变量列表，执行 GraphTask 完成后，这些变量将被移出 GraphTask 并且不再有效
  std::vector<Variable> captured_vars_;

  // 注意：在构造函数中适当调用 `thread_locals_.set_grad_mode()` 之前，此字段不应使用
  at::ThreadLocalState thread_locals_ = at::ThreadLocalState();

  // leaf_streams 包含不同设备的流集合
  std::unordered_set<c10::Stream> leaf_streams;

  // 执行 execute() 的设备的每个设备的当前流
  // 这些流将在 exec_post_processing 中与 leaf_streams 同步
  std::vector<std::optional<c10::Stream>> caller_current_streams_;

  // 收集加速器设备的 caller_current_streams_
  void stash_current_streams();

  // 初始化执行的相关信息
  void init_to_execute(
      Node& graph_root,
      const edge_list& outputs,
      bool accumulate_grad,
      uint64_t min_topo_nr);

  // 创建此任务的线程中的 worker_device 的值
  // 参见注释 [Reentrant backwards]
  // 可以在没有同步的情况下安全读取 owner_ 和 reentrant_depth_
  int owner_;
  // 此图任务的父图任务数目
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int reentrant_depth_;

  // 检查是否可以进行检查点操作
  bool can_checkpoint() const {
  // 返回 exec_info_ 是否为空的布尔值
  return exec_info_.empty();
}

// 检查 GraphTask 是否已完成
bool completed();

// 将图任务标记为已完成，并触发后处理
void mark_as_completed_and_run_post_processing();

// 为此 graph_task 设置适当的异常，该异常在运行提供的函数时遇到
void set_exception(std::exception_ptr eptr, const std::shared_ptr<Node>& fn);

// 为此 graph_task 设置适当的异常，但不立即在 'future_result_' 上标记完成。
// 用户需要显式标记 'future_result_' 为完成，使用适当的异常。
void set_exception_without_signal(const std::shared_ptr<Node>& fn);

// 当遇到错误时是否停止此 GraphTask 的执行。
// 如果设置为 true，这将导致 Engine::execute() 在自动求导引擎收到异常时立即抛出异常。
bool exit_on_error_;

// CPU 线程专门用于处理其调用的后向 CPU 工作。
// 因此，每个给定的图任务维护自己的 cpu_ready_queue_，用于发送要完成的工作。
// 我们为每个 GraphTask 缓存 cpu_ready_queue_，以便知道如果我们在设备线程上（如 GPU 上），
// 我们应该推送到哪个准备队列，但是下一个 NodeTask 应在 CPU 上运行。
std::shared_ptr<ReadyQueue> cpu_ready_queue_;

// 表示图任务完成的 Future。在所有任务完成时得到通知。
c10::intrusive_ptr<at::ivalue::Future> future_result_;

// 在执行此 GraphTask 期间安装的最终回调
std::vector<std::function<void()>> final_callbacks_;

// 保护对 final_callbacks_ 的读写的互斥量。故意不重用 mutex_，因为两者保护不同的数据结构。
std::mutex final_callbacks_lock_;

// 延迟警告处理程序
utils::DelayWarningHandler warning_handler_;

// 图任务的唯一标识符
uint64_t id_;

// GraphTask 的构造函数，初始化图根、CPU 准备队列等参数
GraphTask(
    bool keep_graph,
    bool grad_mode,
    int reentrant_depth,
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    c10::SmallVector<Node*, 4> graph_roots,
    bool exit_on_error = false);

// 执行 GraphTask 的后处理
void exec_post_processing();
};

// GraphTaskGuard 类的定义，用于设置和恢复当前的图任务对象
class GraphTaskGuard {
 public:
  // 构造函数，接受一个共享指针参数，用于设置当前的图任务对象
  explicit GraphTaskGuard(std::shared_ptr<GraphTask> graph_task);
  // 析构函数，用于在对象生命周期结束时恢复之前的图任务对象
  ~GraphTaskGuard();

  // 恢复之前保存的当前图任务对象
  void restore_current_graph_task();

 private:
  // 上一个图任务对象的共享指针
  std::shared_ptr<GraphTask> last_graph_task_;
};

// 获取当前图任务的执行信息的函数声明
TORCH_API const std::unordered_map<Node*, GraphTask::ExecInfo>*
get_current_graph_task_exec_info();
// 获取当前图任务中包含的节点集合的函数声明
TORCH_API const std::unordered_set<Node*>*
get_current_graph_task_nodes_in_graph();
// 获取当前图任务是否保持图的标志的函数声明
TORCH_API bool get_current_graph_task_keep_graph();
// 获取当前图任务的节点执行顺序的函数声明
TORCH_API std::vector<Node*> get_current_graph_task_execution_order();
// 获取当前图任务的 ID 的函数声明
TORCH_API int get_current_graph_task_id();
// 将节点添加到当前图任务的执行信息中的函数声明
void add_node_to_current_graph_task_exec_info(Node* fn);

} // namespace torch::autograd


这些注释解释了每个语句或声明在给定的 C++ 代码片段中的作用和功能。
```