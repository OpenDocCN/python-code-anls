# `.\pytorch\torch\csrc\jit\runtime\interpreter.cpp`

```py
// 包含 Torch 的 JIT（即时编译）运行时解释器的头文件

#include <torch/csrc/jit/runtime/interpreter.h>

// 包含 Torch 的并行计算库的头文件
#include <ATen/Parallel.h>

// 包含 Torch 的 IValue 类定义
#include <ATen/core/ivalue.h>

// 包含 Torch 的记录函数库的头文件
#include <ATen/record_function.h>

// 包含 C10 核心库的线程池定义
#include <c10/core/thread_pool.h>

// 包含 C10 宏定义
#include <c10/macros/Macros.h>

// 包含 C10 异常处理的头文件
#include <c10/util/Exception.h>

// 包含 C10 实用工具中的整数范围定义
#include <c10/util/irange.h>

// 包含 Torch 的自动求导边缘和梯度模式的头文件
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/grad_mode.h>

// 包含 Torch 的自动求导性能分析器的头文件
#include <torch/csrc/autograd/profiler.h>

// 包含 Torch 的自动求导变量定义
#include <torch/csrc/autograd/variable.h>

// 包含 Torch 的 JIT 编译单元定义
#include <torch/csrc/jit/api/compilation_unit.h>

// 包含 Torch 的 JIT 函数实现定义
#include <torch/csrc/jit/api/function_impl.h>

// 包含 Torch 的 JIT IR 常量定义
#include <torch/csrc/jit/ir/constants.h>

// 包含 Torch 的 JIT IR 声明
#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch 的 JIT 日志记录功能
#include <torch/csrc/jit/jit_log.h>

// 包含 Torch 移动平台 JIT 提升的原始操作定义
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>

// 包含 Torch 的 JIT 运行时异常信息处理
#include <torch/csrc/jit/runtime/exception_message.h>

// 包含 Torch 的 JIT 图执行器定义
#include <torch/csrc/jit/runtime/graph_executor.h>

// 包含 Torch 的 JIT 运行时指令定义
#include <torch/csrc/jit/runtime/instruction.h>

// 包含 Torch 的 JIT 解释器代码实现
#include <torch/csrc/jit/runtime/interpreter/code_impl.h>

// 包含 Torch 的 JIT 解释器帧定义
#include <torch/csrc/jit/runtime/interpreter/frame.h>

// 包含 Torch 的 JIT 运行时异常处理定义
#include <torch/csrc/jit/runtime/jit_exception.h>

// 包含 Torch 的 JIT 运算符定义
#include <torch/csrc/jit/runtime/operator.h>

// 包含 Torch 的 JIT 运行时性能记录定义
#include <torch/csrc/jit/runtime/profiling_record.h>

// 包含 Torch 的 JIT 脚本性能分析定义
#include <torch/csrc/jit/runtime/script_profile.h>

// 包含 Torch 的 JIT 可变参数函数定义
#include <torch/csrc/jit/runtime/vararg_functions.h>

// 包含 Torch 的 C++ 栈跟踪工具
#include <torch/csrc/utils/cpp_stacktraces.h>

// 包含标准库中的字符串定义
#include <string>

// 如果编译器定义了 USE_RPC，则包含分布式自动求导上下文容器定义
#ifdef USE_RPC
#include <torch/csrc/distributed/autograd/context/container.h>
using torch::distributed::autograd::DistAutogradContainer;
#endif

// 包含标准库的异常处理
#include <exception>

// 包含标准库的智能指针
#include <memory>

// 包含标准库的互斥锁
#include <mutex>

// 包含标准库的输出流定义
#include <ostream>

// 包含标准库的运行时异常处理
#include <stdexcept>

// 包含标准库的类型信息定义
#include <typeinfo>

// 包含标准库的无序映射定义
#include <unordered_map>

// 包含标准库的无序集合定义
#include <unordered_set>

// 包含标准库的实用工具
#include <utility>

// 包含标准库的向量定义
#include <vector>

// 定义 C10 标志变量：是否启用重新抛出捕获的异常
C10_DEFINE_bool(
    torch_jit_enable_rethrow_caught_exception,
    false,
    "enable rethrowing caught exception");

// 定义 C10 标志变量：是否启用扩展堆栈跟踪
C10_DEFINE_bool(
    torch_jit_enable_expanded_stacks,
    false,
    "When true we will attemps to pre-expand node stacks and cache expanded stacks.");

// Torch JIT 命名空间的开始，包含了 JIT 编译器的所有实现
namespace torch::jit {

// 使用 interpreter 命名空间中的 CodeImpl 类
using CodeImpl = interpreter::CodeImpl;

// 在转换为解释器指令之前，对图进行预处理以更接近指令的形式。
// 特别是我们：
// * 计算节点的输入是否是最后一次使用，这样我们可以发出 MOVE 指令而不是 LOAD 指令。
// * 插入丢弃节点用于未使用的任何节点，以创建一个虚拟使用，这将导致解释器释放节点。
//   丢弃节点只是从堆栈中弹出其输入，以确保解释器释放从未使用的节点的引用。
//   当节点的最后一个使用在某些条件运行的控制流中（例如 If 的一侧）时，
//   还会插入丢弃节点，解释器必须在控制流重新汇合后释放节点。
// 输出是：
// * graph - 处理后的图的副本
// * move_flags[n] - 一个布尔列表，每个输入一个，指示这是否是该值的最后一次使用。解释器
// 定义一个函数，用于确定当前执行上下文中张量的类型指针
TensorTypePtr tensorTypeInCurrentExecutionContext(const at::Tensor& t) {
    // 如果张量未定义，返回一个未定义类型的张量类型指针
    if (!t.defined()) {
        return TensorType::get()->withUndefined();
    }
    // 创建一个张量类型指针，基于给定的张量 t
    auto r = TensorType::create(t);
    // 如果梯度模式未启用，返回一个不需要梯度的张量类型指针
    if (!at::GradMode::is_enabled()) {
        return r->withRequiresGrad(false);
    }
    // 否则返回原始创建的张量类型指针
    return r;
}

namespace {
// 获取分布式自动求导上下文的 ID
inline int64_t getDistAutogradContextId() {
#ifdef USE_RPC
    return DistAutogradContainer::currentContextId();
#else
    return 0;
#endif
}
} // namespace

// 线程局部变量，指向解释器状态的指针
thread_local InterpreterStateImpl* tls_int_state_ptr_ = nullptr;

// 用于保护当前解释器状态的结构体
struct TLSCurrentInterpreterGuard {
    TLSCurrentInterpreterGuard(InterpreterStateImpl* state)
        : prev_state_(tls_int_state_ptr_) {
        tls_int_state_ptr_ = state;
    }

    // 析构函数，恢复之前的解释器状态
    ~TLSCurrentInterpreterGuard() {
        tls_int_state_ptr_ = prev_state_;
    }

 private:
    InterpreterStateImpl* prev_state_;
};

// InterpreterStateImpl 结构体，用于计算代码的解释器状态
struct InterpreterStateImpl : c10::intrusive_ptr_target {
    // 构造函数，接受代码和任务启动器
    InterpreterStateImpl(const Code& code, TaskLauncher taskLauncher)
        : taskLauncher_(std::move(taskLauncher)) {
        // 进入第一个代码帧
        enterFrame(code, 0);
    }

 private:
    // 内部帧结构体
    using Frame = torch::jit::interpreter::Frame;

    // 用于警告节点的私有类
    struct WarnedNodes {
     public:
        // 将索引插入警告节点集合中，返回是否成功插入的布尔值
        bool insert(int32_t idx) {
            std::unique_lock<std::mutex> lock(mutex_);
            return warned_nodes_.insert(idx).second;
        }

     private:
        std::mutex mutex_;
        std::unordered_set<int32_t> warned_nodes_;
    };

    WarnedNodes warned_nodes_;

    // 栈的起始位置，用于暂停时重置栈的位置
    int64_t stack_start_ = -1;
    // 未来对象的指针
    c10::intrusive_ptr<Future> future_;
    // 任务启动器
    TaskLauncher taskLauncher_;

    // 存储此解释器运行中所有张量的向量
    std::vector<IValue> registers;

    // 已进入对象的堆栈
    std::vector<IValue> entered_objects;

    // 帧的堆栈
    std::vector<Frame> frames;

    // 获取当前对象的侵入式指针
    c10::intrusive_ptr<InterpreterStateImpl> intrusive_from_this() {
        c10::raw::intrusive_ptr::incref(this);
        return c10::intrusive_ptr<InterpreterStateImpl>::reclaim(this);
    }

    // 进入新的帧，给定代码和基础指针
    void enterFrame(const Code& code, size_t base_pointer) {
        frames.emplace_back(Frame{code.pImpl, 0, base_pointer, c10::nullopt});
  // 调整寄存器列表的大小，增加指定数量的寄存器空间
  registers.resize(registers.size() + code.pImpl->register_size_);
}

void leaveFrame() {
  // 函数调用结束后，缩减寄存器列表的大小，减去最后一个帧的寄存器大小
  registers.resize(registers.size() - frames.back().function->register_size_);
  // 弹出最后一个帧
  frames.pop_back();
}

void callFunction(
    Function& f,
    Stack& stack,
    std::optional<size_t> bailOut = c10::nullopt,
    bool next = true) {
  // 调用函数，并在回调函数中进行新帧的进入及函数调用校验
  bool newFrame = f.call(stack, bailOut, [&](const Code& code) {
    enterFrame(code, stack.size() - code.num_inputs());
    checkAndStartRecordFunction(frames.back(), stack);
  });
  // 如果指定执行下一步操作，更新当前帧的程序计数器
  if (next) {
    (frames.rbegin() + (newFrame ? 1 : 0))->pc++;
  }
}

// 返回相对于寄存器列表末尾的指定寄存器引用，以便在函数调用时引用当前执行函数的寄存器
IValue& reg(size_t reg) {
  return *(registers.end() - reg);
}

void dump(std::ostream& out, const Stack& stack) const {
  // 输出堆栈内容到指定的输出流
  out << "Stack:\n";
  for (const auto& val : stack) {
    out << val;
    out << "\n";
  }
}

class StackSizeDidntChangeGuard {
 public:
  StackSizeDidntChangeGuard(const StackSizeDidntChangeGuard&) = delete;
  StackSizeDidntChangeGuard(StackSizeDidntChangeGuard&&) = delete;
  StackSizeDidntChangeGuard& operator=(const StackSizeDidntChangeGuard&) =
      delete;
  StackSizeDidntChangeGuard& operator=(StackSizeDidntChangeGuard&&) = delete;

  StackSizeDidntChangeGuard(
      const Frame& frame,
      const torch::jit::Stack& stack,
      const Instruction& inst)
      : frame_(frame), stack_(stack), instX_(inst.X) {
    // 用于支持跨平台的 maybe_unused 属性，防止未使用的变量警告
    (void)frame_;
    (void)stack_;
    (void)instX_;
    (void)initialSize_;
  }

  // 调用断言函数，尚未实现具体内容
  void callAssert() const {
// 如果未定义 NDEBUG 宏，则执行以下代码段（用于调试模式）
#ifndef NDEBUG
      // 使用帧对象的函数属性，检查堆栈大小是否符合预期
      frame_.function->assert_stack_size(instX_, initialSize_, stack_.size());
#endif
    }

   private:
    // 引用类型成员变量，表示当前帧和堆栈
    const Frame& frame_;
    const torch::jit::Stack& stack_;
    // 无符号整型成员变量，表示当前指令编号
    std::uint32_t instX_;
    // 大小类型成员变量，记录初始时堆栈大小
    std::size_t initialSize_{stack_.size()};
  };

  // 定义一个结构体 DoNothing，并标记其未使用
  struct C10_UNUSED DoNothing {};

// 如果定义了 __GNUC__ 或者 __clang__ 宏，则定义 JIT_USE_COMPUTED_GOTO 宏
#if defined(__GNUC__) || defined(__clang__)
#define JIT_USE_COMPUTED_GOTO
#endif

// 用于处理解释器内部状态转换的基本操作。
// 通过维护两个本地变量来表示内部解释器状态：
// `frame` 表示当前解释器操作的帧。
// `inst` 表示程序计数器指向的当前指令。
//
// 指令块应始终通过 `INST` 宏声明，指令体应始于 `instGuard()` 声明。
// 同时，块应以 `INST_NEXT`（用于执行下一条指令）或 `INST_DISPATCH`（通过 `instFetch` 跳转到计算位置）结束。
#if defined(JIT_USE_COMPUTED_GOTO)
// 定义 INST 宏，用于标记指令位置，并创建标签
#define INST(NAME) \
  NAME:            \
  label_##NAME
// 在 JIT 使用计算跳转时，定义 INST_DISPATCH 为跳转到指令对应位置的宏
#define INST_DISPATCH goto* dispatch_table[inst.op]
#else
// 如果不使用计算跳转，INST 宏直接为指令名称
#define INST(NAME) NAME
// 在不使用计算跳转时，INST_DISPATCH 为跳出当前指令块的宏
#define INST_DISPATCH break
#endif

// 定义 INST_NEXT 宏，用于获取下一条指令，并根据情况执行跳转
#define INST_NEXT      \
  inst = instFetch(1); \
  INST_DISPATCH

  // 模板函数 runTemplate，根据 EnableProfiling 模板参数确定是否启用性能分析
  template <bool EnableProfiling>
  bool runTemplate(Stack& stack) {
    // 如果从未运行过，则可能在挂起时需要返回堆栈；记录堆栈开始位置以便正确返回堆栈
    if (stack_start_ == -1) {
      // 断言堆栈大小至少符合当前帧函数的输入数目
      TORCH_INTERNAL_ASSERT(stack.size() >= frames.back().function->n_inputs);
      stack_start_ = stack.size() - frames.back().function->n_inputs;
    } else {
      // 在重启时，整个堆栈都属于我们自己，所以不保留任何东西
      stack_start_ = 0;
    }

    // TLS 当前解释器保护
    TLSCurrentInterpreterGuard g(this);
    // 如果当前帧的程序计数器为 0 并且堆栈起始位置为 0，则检查并开始记录函数
    if (frames.back().pc == 0 && stack_start_ == 0) {
      checkAndStartRecordFunction(frames.back(), stack);
    }

#if defined(JIT_USE_COMPUTED_GOTO)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    // 定义静态跳转表，根据操作码跳转到相应标签位置
    static void* dispatch_table[] = {
#define DISPATCH_TABLE_ENTRY(op, _) &&label_##op,
        FORALL_OPCODES(DISPATCH_TABLE_ENTRY)
#undef DISPATCH_TABLE_ENTRY
    };
#endif
    } catch (std::exception& e) {
      // 捕获标准库异常并进行处理
      for (auto it = entered_objects.rbegin(), end = entered_objects.rend();
           it != end;
           ++it) {
        // 获取当前迭代对象的类型的 __exit__ 方法
        auto& f = it->toObject()->type()->getMethod("__exit__");
        // 创建一个新的堆栈
        Stack stack;
        // 将当前迭代对象推入堆栈
        push(stack, *it);
        // 推入三个空的 IValue 对象
        push(stack, IValue());
        push(stack, IValue());
        push(stack, IValue());
        try {
          // 运行获取到的 __exit__ 方法
          f.run(stack);
        } catch (std::exception& _) {
          // 捕获并处理异常，当前未做具体处理
          // TODO(T98048876): Handle `_` correctly.
        }
      }
      // 如果启用了重新抛出已捕获异常标志
      if (FLAGS_torch_jit_enable_rethrow_caught_exception) {
        // 如果存在 future 对象
        if (future_) {
          // 设置 future 对象的错误状态为当前异常
          future_->setError(std::current_exception());
          // 返回 false
          return false;
        }
        // 否则直接重新抛出异常
        throw;
      }
      // 尝试将异常转换为 JITException 指针
      auto* jit_exception = dynamic_cast<JITException*>(&e);
      // Janky af.  See https://github.com/pytorch/pytorch/issues/54612
      // 尝试将异常转换为 NotImplementedError 指针
      auto* not_implemented_error = dynamic_cast<c10::NotImplementedError*>(&e);

      // 定义一个可选的 Python 类名字符串
      std::optional<std::string> python_class_name;
      // 如果异常为 JITException 类型
      if (jit_exception) {
        // 获取 JITException 对象的 Python 类名
        python_class_name = jit_exception->getPythonClassName();
      }
      // 处理错误，传入异常对象、是否为 JITException、是否为 NotImplementedError、Python 类名
      handleError(
          e, (bool)jit_exception, not_implemented_error, python_class_name);
      // 返回 false
      return false;
    }
  }
// 如果未处于性能分析状态，则调用非性能分析版本的 runTemplate()，并返回结果
// 否则，调用启用性能分析版本的 runTemplate()，并返回结果
bool runImpl(Stack& stack) {
    if (!profiling::isProfilingOngoing()) {
        return runTemplate</*EnableProfiling*/ false>(stack);
    } else {
        return runTemplate</*EnableProfiling*/ true>(stack);
    }
}

// 格式化调用栈信息并输出到指定的输出流
void formatStackTrace(std::ostream& out) {
    format_stack_trace(out, callstack());
}

// 处理异常情况，生成相应的错误消息并抛出异常
void handleError(
    const std::exception& e,                      // 异常对象的引用
    bool is_jit_exception,                        // 是否为 JIT 异常的标志
    c10::NotImplementedError* not_implemented_error, // 未实现错误的指针
    std::optional<std::string> python_class_name  // Python 类名的可选参数
) {
    // 根据异常对象创建异常消息对象
    ExceptionMessage msg(e);
    std::ostringstream ss;
    std::string class_name =
        python_class_name ? *python_class_name : "RuntimeError";
    // 输出 TorchScript 解释器中操作失败的通用消息
    ss << "The following operation failed in the TorchScript interpreter.\n";
    // 格式化并输出调用栈信息到字符串流 ss 中
    formatStackTrace(ss);
    // 输出异常类名和异常消息
    ss << class_name << ": " << msg << "\n";
    // 如果存在 future_，则设置 Future 对象的错误状态
    if (future_) {
        future_->setError(std::make_exception_ptr(Future::FutureError(ss.str())));
    } else if (is_jit_exception) {
        // 当创建新的 JITException 时，保存原始异常的消息
        throw JITException(ss.str(), python_class_name, e.what());
    } else if (not_implemented_error) {
        // 抛出未实现错误异常，包含错误消息、回溯信息和调用者信息
        throw c10::NotImplementedError(
            ss.str(),
            not_implemented_error->backtrace(),
            not_implemented_error->caller());
    } else {
        // 如果启用了 C++ 堆栈跟踪，输出异常对象的 what() 信息
        if (get_cpp_stacktraces_enabled()) {
            ss << e.what() << "\n";
        }
        // 抛出标准运行时错误异常，包含完整错误消息
        throw std::runtime_error(ss.str());
    }
}

// 检查并开始记录函数执行的信息
static void checkAndStartRecordFunction(Frame& frame, Stack& stack) {
    // 如果当前帧未记录函数执行信息
    if (!frame.record_function) {
        // 获取当前 TorchScript 函数的步骤回调函数
        auto step_callbacks = at::getStepCallbacksUnlessEmpty(
            at::RecordScope::TORCHSCRIPT_FUNCTION);
        // 如果步骤回调函数可用
        if (C10_UNLIKELY(step_callbacks.has_value())) {
            // 创建记录函数对象，并确保其处于活动状态
            auto rec_fn =
                std::make_unique<at::RecordFunction>(std::move(*step_callbacks));
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rec_fn->isActive());
            // 如果记录函数需要输入参数，则在执行之前记录输入
            if (rec_fn->needsInputs()) {
                rec_fn->before(
                    frame.function->function_name_,
                    last(stack, frame.function->n_inputs));
            } else {
                rec_fn->before(frame.function->function_name_);
            }
            // 将创建的记录函数对象移动到帧中保存
            frame.record_function = std::move(rec_fn);
        }
    }
}
  }

 public:
  // 返回一个存储帧函数的向量，以避免形成字符串的开销，即返回 CodeImpl*
  // 这种方式不够干净，因为会暴露解释器的内部细节。但这样我们可以保留图形/节点和函数，
  // 在 autograd 分析器中的每个事件结束时，可以为每个事件创建模块层次结构字符串。
  // 目前看来，开销并不是非常大。
  // 另一种选择是返回 (string, InlinedCallstackPtrs) 的向量，
  // 字符串将包含函数名和 self 的类型名。
  // 返回的字符串向量的格式：
  // 对于每个帧，对应的模块名、类型和函数名的格式如下：
  // <module-instance-name>(module type)::<function-name>
  // 模块实例名称的特殊键：
  //   - TOP：顶层模块
  //   - SELF：当帧的方法/函数与前一个帧的模块实例相关联时
  //   - INSTANCE_NAME_UNKNOWN：无法确定实例名称
  //   - CALL_FUNCTION：调用自由函数
  std::vector<std::string> moduleHierarchy() const {
    // 初始化存储模块和函数名的向量
    std::vector<std::string> module_function_list;
    // 初始模块层次设置为 "TOP"
    std::string module_hierarchy("TOP");
    // 返回存储模块和函数名的向量
    return module_function_list;
  }

  // 返回调用堆栈的向量
  std::vector<StackEntry> callstack() const {
    // 初始化存储堆栈条目的向量
    std::vector<StackEntry> entries;
    // 遍历每个帧
    for (const auto i : c10::irange(frames.size())) {
      // 获取当前帧
      const Frame& frame = frames[i];
      // 获取前一个函数名
      std::string previous_fn_name = frame.function->function_name_;
      // 获取程序计数器的值
      size_t pc = frame.pc;
      // 对于 CALL 节点，已经提前递增了程序计数器，因此需要减去来报告调用节点
      if (i + 1 < frames.size()) {
        --pc;
      }

      // 获取节点
      Node* node = frame.function->instructions_source_[pc];
      // 如果节点有调用堆栈
      if (node->callstack()) {
        // 遍历调用堆栈的每个元素
        for (const auto& p : (*node->callstack())->vec()) {
          // 将前一个函数名和调用堆栈中的信息添加到堆栈条目中
          entries.emplace_back(StackEntry{previous_fn_name, std::get<1>(p)});
          // 更新前一个函数名为调用堆栈中下一个节点的名称
          previous_fn_name = std::get<0>(p)->name();
        }
      }
      // 将当前节点和源范围添加到堆栈条目中
      entries.emplace_back(StackEntry{previous_fn_name, node->sourceRange()});
    }
    // 返回堆栈条目的向量
    return entries;
  }

  // 获取或创建 Future 对象
  c10::intrusive_ptr<Future> getOrCreateFuture() {
    // 如果 future_ 为空，创建一个新的 Future 对象
    if (!future_) {
      future_ = c10::make_intrusive<Future>(frames.front().function->return_type_);
    }
    // 返回 Future 对象
    return future_;
  }

  // 异步运行，返回 Future 对象
  c10::intrusive_ptr<Future> runAsync(Stack& stack) {
    // 获取或创建 Future 对象
    getOrCreateFuture();
    // 执行异步运行实现
    runImpl(stack);
    // 返回 Future 对象
    return future_;
  }

  // 同步运行，使用堆栈参数
  void run(Stack& stack) {
    // 在调用 runImpl() 之前，必须确保继续完成帧，因此需要检查帧是否为空
    TORCH_INTERNAL_ASSERT(!frames.empty());
    // 获取输出的数量
    const auto num_outputs = frames.front().function->n_outputs;
    // ...
  }
    # 如果 runImpl(stack) 返回 true，则执行以下代码块
    if (runImpl(stack)) {
      # 等待 future_ 的完成
      future_->wait();

      # 如果 num_outputs 等于 1
      if (num_outputs == 1) {
        # 将 future_ 的值压入 stack 中
        push(stack, future_->value());
      } else {
        # 将 future_ 的值转换为元组
        auto tuple = future_->value().toTuple();
        
        # 遍历元组中的每一个元素
        for (const IValue& value : tuple->elements()) {
          # 将每个元素压入 stack 中
          push(stack, value);
        }
      }
    }
  }
};

// 返回当前线程局部状态的调用栈，若存在则返回反转后的调用栈，否则返回空向量
std::vector<StackEntry> currentCallstack() {
  if (tls_int_state_ptr_) {  // 如果线程局部状态指针不为空
    auto cs = tls_int_state_ptr_->callstack();  // 获取调用栈
    std::reverse(cs.begin(), cs.end());  // 反转调用栈
    return cs;  // 返回反转后的调用栈
  }
  return std::vector<StackEntry>();  // 否则返回空向量
}

// 返回当前线程局部状态的模块层次结构，若存在则返回该层次结构，否则返回空向量
std::vector<std::string> currentModuleHierarchy() {
  if (tls_int_state_ptr_) {  // 如果线程局部状态指针不为空
    return tls_int_state_ptr_->moduleHierarchy();  // 返回模块层次结构
  }
  return std::vector<std::string>();  // 否则返回空向量
}

// 重载运算符<<，输出代码对象的图形表示和详细信息
std::ostream& operator<<(std::ostream& out, const Code& code) {
  out << *code.pImpl->graph_ << "\n";  // 输出代码对象的图形表示
  code.pImpl->dump(out);  // 输出代码对象的详细信息
  return out;  // 返回输出流
}

// Code类的构造函数，初始化私有成员指针pImpl
Code::Code(
    const std::shared_ptr<Graph>& graph,
    std::string function_name,
    size_t remaining_bailout_depth)
    : pImpl(new CodeImpl(
          graph,
          std::move(function_name),
          remaining_bailout_depth)) {}

// Code类的构造函数，使用给定的CodeImpl指针初始化pImpl
Code::Code(CodeImpl* codeImpl) : pImpl(codeImpl) {}

// MobileCode类的构造函数，继承自Code，初始化MobileCodeImpl对象
MobileCode::MobileCode(
    const std::shared_ptr<Graph>& graph,
    std::string function_name,
    bool emit_default_input_instructions,
    bool support_default_args_before_out,
    bool emit_promoted_ops,
    size_t remaining_bailout_depth)
    : Code(new interpreter::MobileCodeImpl(
          graph,
          std::move(function_name),
          emit_default_input_instructions,
          support_default_args_before_out,
          emit_promoted_ops,
          remaining_bailout_depth)) {}

// 返回Code对象中梯度执行器的引用向量
const std::vector<GraphExecutor*>& Code::grad_executors() {
  return pImpl->grad_executors();
}

// 返回Code对象中不同图操作执行器的引用向量
const std::vector<GraphExecutor*>& Code::diff_graph_op_executors() {
  return pImpl->diff_graph_op_executors();
}

// 返回Code对象中记录的失败救援点数量
size_t Code::num_bailouts() const {
  return pImpl->type_table_.size();
}

// 请求Code对象中指定索引的失败救援点
void Code::request_bailout(size_t index) {
  pImpl->request_bailout(index);
}

// 返回Code对象中记录的输入数量
size_t Code::num_inputs() const {
  return pImpl->n_inputs;
}

// 返回Code对象中记录的输出数量
size_t Code::num_outputs() const {
  return pImpl->n_outputs;
}

// 返回Code对象中的常量表
const std::vector<c10::IValue>& Code::constant_table() const {
  return pImpl->constant_table();
}

// 返回Code对象中的指令向量
const std::vector<Instruction>& Code::instructions() const {
  return pImpl->instructions();
}

// 返回Code对象中操作名称到指定参数数量的映射表
const std::unordered_map<std::string, size_t>& Code::op_to_num_specified_args()
    const {
  return pImpl->op_to_num_specified_args();
}

// 返回Code对象中指令源的引用向量
const std::vector<Node*>& Code::instructions_source() const {
  return pImpl->instructions_source();
}

// 返回Code对象中类型表
const std::vector<TypePtr>& Code::type_table() const {
  return pImpl->type_table_;
}

// 返回Code对象中寄存器的大小
size_t Code::register_size() const {
  return pImpl->register_size_;
}

// 返回Code对象中的图对象的共享指针
std::shared_ptr<Graph> Code::graph() const {
  return pImpl->preprocess_.graph;
}

// InterpreterState类的构造函数，使用给定的Code对象和任务启动器初始化
InterpreterState::InterpreterState(const Code& code, TaskLauncher taskLauncher)
    : pImpl(c10::make_intrusive<InterpreterStateImpl>(
          code,
          std::move(taskLauncher))) {}

// 运行InterpreterState对象，使用给定的栈作为输入
void InterpreterState::run(Stack& stack) {
  static_cast<InterpreterStateImpl*>(pImpl.get())->run(stack);
}

// 异步运行InterpreterState对象，使用给定的栈作为输入，并返回Future指针
c10::intrusive_ptr<Future> InterpreterState::runAsync(Stack& stack) {
  return static_cast<InterpreterStateImpl*>(pImpl.get())->runAsync(stack);
}
// 返回InterpreterStateImpl对象的getOrCreateFuture()方法返回的Future智能指针
c10::intrusive_ptr<Future> InterpreterState::getFuture() {
    return static_cast<InterpreterStateImpl*>(pImpl.get())->getOrCreateFuture();
}

// InterpreterState类的构造函数，接受一个pImpl_参数作为成员变量pImpl，并移动其所有权
InterpreterState::InterpreterState(
    c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl_)
    : pImpl(std::move(pImpl_)) {}

// InterpreterContinuation类的函数调用运算符重载
void InterpreterContinuation::operator()() {
#ifdef USE_RPC
  // 保存当前的DistAutograd上下文ID，并设置为dist_autograd_context_id_
  auto prev_dist_id = DistAutogradContainer::currentContextId();
  DistAutogradContainer::forceCurrentContextId(dist_autograd_context_id_);
#endif
  // 如果tls_state_不为null，使用ThreadLocalStateGuard保护*tls_state_，然后异步运行state的runAsync方法
  if (tls_state_ != c10::nullopt) {
    at::ThreadLocalStateGuard g(*tls_state_);
    state.runAsync(stack);
  } else {
    // 否则直接异步运行state的runAsync方法
    state.runAsync(stack);
  }
#ifdef USE_RPC
  // 恢复之前保存的DistAutograd上下文ID
  DistAutogradContainer::forceCurrentContextId(prev_dist_id);
#endif
}

} // namespace torch::jit
```