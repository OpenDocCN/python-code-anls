# `.\pytorch\aten\src\ATen\functorch\Interpreter.h`

```py
#pragma once

#include <ATen/functorch/Macros.h>  // 引入 functorch 宏定义
#include <ATen/core/dispatch/Dispatcher.h>  // 引入 ATen 分发器
#include <c10/core/impl/LocalDispatchKeySet.h>  // 引入 c10 本地分发键集
#include <c10/util/Optional.h>  // 引入 c10 可选类型
#include <bitset>  // 引入 bitset 类型
#include <utility>  // 引入实用工具
#include <variant>  // 引入 variant 类型

namespace at::functorch {

// NOTE: [functorch interpreter stack]
//
// functorch 的分发系统使用一个解释器堆栈。
// 历史上我们称之为 "DynamicLayerStack"。
//
// 每个解释器负责一个转换：例如 "VmapInterpreter" 负责解释批处理版本的操作符 (例如 aten::mv)。
//
// 具体来说，每个解释器负责两件事：
//
// 1) process(ophandle, stack)
// 给定一个操作符句柄和一个参数堆栈，解释器负责根据解释器的语义执行操作。
// 例如，对于 VmapInterpreter，这意味着调用批处理规则。
//
// 批处理规则存储在 FuncTorchBatched 键上的内核中，因此 VmapInterpreter 调用批处理规则的方式大致是：
// (A) 排除除批处理键之外的所有分发键，(B) 重新分发以便到达批处理键。
//
// 2) sendToNextInterpreter(ophandle, stack)
// 当 VmapInterpreter 看到 aten::mv 时，它会处理成对 aten::mm 的调用。然后需要将对 aten::mm 的调用发送到解释器堆栈中的下一个解释器。
//
// VmapInterpreter 通过调用 ophandle.callBoxed(stack) 来实现这一点，大多数解释器都会以这种方式实现。

enum class RandomnessType {
    Error,      // 调用随机函数时总是出错
    Same,       // 批次间随机性相同
    Different,  // 批次间随机性不同
    END
};

enum class TransformType {
  Torch,  // 未使用
  Vmap,
  Grad,  // 反向模式自动微分，又称为 vjp
  Jvp,  // 正向模式自动微分
  Functionalize,
};

std::ostream& operator<<(std::ostream& os, const TransformType& t);

// NOTE: [Interpreter "subclassing" design]
//
// 如何实现不同转换 (vmap、grad 等) 的各种解释器？
//
// 访问解释器是 functorch 的热路径，因此我们必须尽可能快地执行此代码。
//
// 因此，我们避免使用虚拟方法，这导致我们的代码看起来有点特别。
//
// `Interpreter` 是解释器的结构体。它包含所有相关信息（解释器类型和元数据）。
// 每个解释器的元数据表示为所有可能元数据的联合体 (std::variant)。
//
// 给定一个解释器，如何获取 "VmapInterpreter"？如果要访问元数据字段（如 batchSize 和 randomness），可能会这样做。
//
// 每种解释器（例如 Vmap）都有一个关联的方便结构体（例如 VmapInterpreterPtr）。
//
// 使用 VmapInterpreterPtr(Interpreter*) 构造方便结构体，然后可以像这样访问 VmapInterpreterPtr 上的方法：
// >>> VmapInterpreterPtr(&interpreter).batchSize()
//
// 最后，Interpreter::process 根据解释器的类型进行切换，并在底层调用其中的 {Transform}Interpreter::processImpl。
// Interpreter::sendToNextInterpreter 也是类似的操作 :)

struct VmapInterpreterMeta {
  explicit VmapInterpreterMeta(c10::SymInt batchSize, RandomnessType randomness) :
    batchSize_(std::move(batchSize)), randomness_(randomness) {}
  c10::SymInt batchSize_;
  RandomnessType randomness_;
};

struct GradInterpreterMeta {
  explicit GradInterpreterMeta(bool prevGradMode): prevGradMode_(prevGradMode) {}
  bool prevGradMode_;
};

struct JvpInterpreterMeta {
  explicit JvpInterpreterMeta(bool prevFwdGradMode) : prevFwdGradMode_(prevFwdGradMode) {}
  bool prevFwdGradMode_;
};

struct FunctionalizeInterpreterMeta {
  explicit FunctionalizeInterpreterMeta(bool functionalizeAddBackViews) :
    functionalizeAddBackViews_(functionalizeAddBackViews) {}
  bool functionalizeAddBackViews_;
};

typedef std::variant<
  int64_t,
  GradInterpreterMeta,
  JvpInterpreterMeta,
  VmapInterpreterMeta,
  FunctionalizeInterpreterMeta
> InterpreterMeta;


struct Interpreter {
  // 工厂函数
  static Interpreter Vmap(int64_t level, c10::SymInt batchSize, RandomnessType randomness) {
    return Interpreter(TransformType::Vmap, level, VmapInterpreterMeta(std::move(batchSize), randomness));
  }
  static Interpreter Grad(int64_t level, bool prevGradMode) {
    return Interpreter(TransformType::Grad, level, GradInterpreterMeta(prevGradMode));
  }
  static Interpreter Jvp(int64_t level, bool prevFwdGradMode) {
    return Interpreter(TransformType::Jvp, level, JvpInterpreterMeta(prevFwdGradMode));
  }
  static Interpreter Functionalize(int64_t level, bool functionalizeAddBackViews) {
    return Interpreter(TransformType::Functionalize, level, FunctionalizeInterpreterMeta(functionalizeAddBackViews));
  }

  // 方法
  TransformType key() const { return type_; }
  int64_t level() const { return level_; }
  const InterpreterMeta& meta() const { return meta_; }

  // 处理函数，根据操作符处理当前堆栈
  void process(const c10::OperatorHandle& op, torch::jit::Stack* stack);
  // 将堆栈内容发送给下一个解释器，根据需要处理梯度特例
  void sendToNextInterpreter(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case);

  // 保存本地分发键集合
  void saveLocalDispatchKeySet(c10::impl::LocalDispatchKeySet keyset) {
    TORCH_INTERNAL_ASSERT(!savedLocalDispatchKeySet_.has_value());
    savedLocalDispatchKeySet_ = keyset;
  }
  // 清除保存的本地分发键集合
  void clearSavedLocalDispatchKeySet() {
    TORCH_INTERNAL_ASSERT(savedLocalDispatchKeySet_.has_value());
    savedLocalDispatchKeySet_ = c10::nullopt;
  }
  // 获取保存的本地分发键集合
  c10::impl::LocalDispatchKeySet getSavedLocalDispatchKeySet() const {
    TORCH_INTERNAL_ASSERT(savedLocalDispatchKeySet_.has_value());
    return *savedLocalDispatchKeySet_;
  }

  // 私有成员变量
private:
  c10::optional<c10::impl::LocalDispatchKeySet> savedLocalDispatchKeySet_;
  TransformType type_;
  int64_t level_;
  InterpreterMeta meta_;

  explicit Interpreter(TransformType type, int64_t level, InterpreterMeta meta) :
    type_(type), level_(level), meta_(std::move(meta)) {}
};
  // 返回当前保存的本地调度键集合
  return *savedLocalDispatchKeySet_;
}

// 如果当前解释器正在进行的变换中，则解释器处于活动状态。
// 例如，在函数 f 中调用 vmap(f)(x)，在 f 函数内部，vmap 对应的解释器是活动的，即使它不在 DynamicLayerStack 上。
bool is_alive() const {
  return *is_alive_;
}
// 返回解释器的活动状态的共享指针
const std::shared_ptr<bool>& is_alive_ptr() const {
  return is_alive_;
}
// 设置解释器的活动状态
void set_is_alive(bool alive) {
  *is_alive_ = alive;
}

// 请不要使用这个构造函数
explicit Interpreter() = default;

private:
// 显式构造函数，初始化解释器的类型、级别、活动状态和元信息
explicit Interpreter(TransformType type, int64_t level, InterpreterMeta meta):
  type_(type), level_(level), is_alive_(std::make_shared<bool>(false)), meta_(std::move(meta)) {}

// 字段
// 解释器的变换类型
TransformType type_{};
// 变换的级别
int64_t level_{};
// 可选的本地调度键集合
optional<c10::impl::LocalDispatchKeySet> savedLocalDispatchKeySet_;
// 活动状态的共享指针
std::shared_ptr<bool> is_alive_;
// 解释器的元信息
InterpreterMeta meta_;
// 结束命名空间 at::functorch

} // namespace at::functorch
```