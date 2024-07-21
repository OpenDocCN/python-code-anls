# `.\pytorch\aten\src\ATen\core\stack.h`

```
#pragma once
// 预处理指令：#pragma once 确保头文件只被包含一次，避免重复定义

#include <type_traits>
// 引入类型特性库

#include <ATen/core/ivalue.h>
// 引入 ATen 库中的 IValue 类定义

#include <c10/util/Deprecated.h>
// 引入 c10 库中的 Deprecated 功能

#include <c10/util/irange.h>
// 引入 c10 库中的 irange 功能

// TODO move this to c10 namespace

namespace torch::jit {
// 定义命名空间 torch::jit

using c10::IValue;
// 使用 c10 命名空间中的 IValue 类
using Stack = std::vector<IValue>;
// 定义别名 Stack 为 std::vector<IValue>

class Operation {
  template <typename F, typename Arg>
  using accepts = std::is_constructible<std::function<void(Arg)>, F&&>;
  // 接受模板：用于检查类型 F 是否可构造为 std::function<void(Arg)>

 public:
  template <typename F,
            std::enable_if_t<accepts<F, Stack*>::value, int> = 0>
  C10_DEPRECATED_MESSAGE("Please use void(Stack&) to register operator instead.")
  // 使用 C10_DEPRECATED_MESSAGE 宏，提醒使用 void(Stack&) 注册操作符
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  // 禁止 linter 检查，因为未使用 std::forward
  // 构造函数模板：接受函数对象 F&&，将其转换为 std::function<void(Stack&)> 类型
  Operation(F&& raw): op_([raw = std::forward<F>(raw)](Stack& stack) {
    raw(&stack);
  }) {}

  template <typename F,
            std::enable_if_t<accepts<F, Stack&>::value &&
                !std::is_same_v<std::decay_t<F>, Operation>, int> = 0>
  // 构造函数模板：接受函数对象 F&&，将其转换为 std::function<void(Stack&)>
  Operation(F&& op): op_(std::forward<F>(op)) {}

  Operation(std::nullptr_t) noexcept {}
  // 空指针构造函数

  explicit operator bool() const noexcept {
    return op_ ? true : false;
  }
  // 显式转换运算符，检查是否存在有效的操作函数对象

  void operator()(Stack& stack) {
    op_(stack);
  }
  // 函数调用运算符重载，调用内部的操作函数对象

  template <typename T>
  T* target() noexcept {
    return op_.target<T>();
  }
  // 返回操作函数对象的目标指针类型 T*

 private:
  std::function<void(Stack&)> op_;
  // 内部私有成员变量：保存操作函数对象
};

// An operation with N inputs and M outputs pops the last N inputs off
// the stack and pushes its M inputs onto the stack
// before: <other stack items> I0, I1, ... IN <- stack.back()
// after: <other stack items> O0, O1, ... OM
// operations are defined this way so that ownership of inputs can be
// transferred to the operation and it can incrementally drop ownership of
// tensors when they become unneeded. For large operations, like 'run an entire
// subgraph', this functionality is very important for minimizing gpu memory
// usage return value is the relative 'offset' to jump to for the next
// operation:
//   pc += 1 + offset
// so a return value of 0 goes to the next instruction

// treat the last N elements of the stack as a list, looking up
// element i
inline IValue& peek(Stack& stack, size_t i, size_t N) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
  // 禁止 linter 检查，因为进行了窄化转换
  return *(stack.end() - N + i);
}
inline IValue& peek(Stack* stack, size_t i, size_t N) {
  // peek 函数的指针版本，调用非指针版本的 peek
  return peek(*stack, i, N);
}
inline const IValue& peek(const Stack& stack, size_t i, size_t N) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
  // 禁止 linter 检查，因为进行了窄化转换
  return *(stack.end() - N + i);
}
inline const IValue& peek(const Stack* stack, size_t i, size_t N) {
  // peek 函数的常量指针版本，调用非常量版本的 peek
  return peek(*stack, i, N);
}
// treat the last N elements of the stack as a list, looking up the
// slice starting at index i and having length len
inline at::ArrayRef<IValue> peekSlice(
    const Stack& stack,
    size_t i,
    size_t len,
    size_t N) {
  // 返回从 stack 中 N 个元素的切片，从索引 i 开始，长度为 len
  return at::ArrayRef<IValue>(stack).slice(stack.size() - N + i, len);
}
inline at::ArrayRef<IValue> last(const Stack& stack, size_t N) {
  // 返回 stack 中最后 N 个元素的切片
  return peekSlice(stack, 0, N, N);
}
inline at::ArrayRef<IValue> last(const Stack* stack, size_t N) {
  // last 函数的指针版本，调用非指针版本的 last
  return last(*stack, N);
}
inline void drop(Stack& stack, size_t n) {
    // 删除栈顶的 n 个元素
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
    stack.erase(stack.end() - n, stack.end());
}

inline void drop(Stack* stack, size_t n) {
    // 使用指针调用删除栈顶的 n 个元素
    drop(*stack, n);
}

inline IValue pop(Stack& stack) {
    // 弹出栈顶的元素并返回
    auto r = std::move(stack.back());
    stack.pop_back();
    return r;
}

inline IValue pop(Stack* stack) {
    // 使用指针调用弹出栈顶的元素并返回
    return pop(*stack);
}

inline std::vector<IValue> pop(Stack& stack, size_t n) {
    // 弹出栈顶的 n 个元素，并返回一个包含这些元素的向量
    std::vector<IValue> result;
    result.reserve(n);
    for (const auto i : c10::irange(n)) {
        result.push_back(std::move(peek(stack, i, n)));
    }
    drop(stack, n);
    return result;
}

// variadic pop:
// int64_t a; at::Tensor b;
// pop(stack, a, b);
// equivalent to:
// b = pop(stack).toTensor();
// a = pop(stack).toInt();
template <typename... Types>
inline void pop(Stack& stack, Types&... args) {
    // 弹出栈顶的多个元素，并将它们转换为指定类型的参数
    size_t i = 0;
    constexpr size_t N = sizeof...(args);
    (void)std::initializer_list<int>{
        (args = std::move(peek(stack, i++, N)).template to<Types>(), 0)...
    };
    drop(stack, N);
}

template <typename... Types>
inline void pop(Stack* stack, Types&... args) {
    // 使用指针调用弹出栈顶的多个元素，并将它们转换为指定类型的参数
    pop(*stack, args...);
}

template <typename Type>
inline void push_one(Stack& stack, Type&& arg) {
    // 将一个元素推入栈中
    stack.emplace_back(std::forward<Type>(arg));
}

inline void push_one(Stack& stack, c10::TensorOptions options) {
    // 将张量选项的各部分推入栈中
    stack.emplace_back(c10::typeMetaToScalarType(options.dtype()));
    stack.emplace_back(options.layout());
    stack.emplace_back(options.device());
    stack.emplace_back(options.pinned_memory());
}

template <typename... Types>
inline void push(Stack& stack, Types&&... args) {
    // 将多个元素推入栈中
    (void)std::initializer_list<int>{(push_one(stack, std::forward<Types>(args)), 0)...};
}

template <typename... Types>
inline void push(Stack* stack, Types&&... args) {
    // 使用指针调用将多个元素推入栈中
    return push(*stack, std::forward<Types>(args)...);
}

template <class T>
inline void push_list_elements(Stack& stack, const c10::List<T>& elements) {
    // 将列表中的元素依次推入栈中
    for (T elem : elements) {
        stack.push_back(std::move(elem));
    }
}

// The packer here is carefully written not to make any unnecessary
// copies.

// pack takes the return values of aten functions pushes them onto the stack
template <typename T>
inline void pack(Stack& stack, T&& v) {
    // 将一个值压入栈中
    stack.emplace_back(std::forward<T>(v));
}

template <typename T>
inline void pack(Stack* stack, T&& v) {
    // 使用指针调用将一个值压入栈中
    pack(*stack, std::forward<T>(v));
}

template <std::size_t remaining, typename... Args>
struct TuplePacker {
    // NB: *Not* a universal reference.
    static void execute(Stack& stack, std::tuple<Args...>&& t) {
        // 将元组中的特定位置的值压入栈中
        pack(stack, std::get<sizeof...(Args) - remaining>(std::move(t)));
        TuplePacker<remaining - 1, Args...>::execute(stack, std::move(t));
    }
};
struct TuplePacker<0, Args...> {
  // 定义一个模板特化，当参数包 Args 为空时执行该结构体
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  // execute 函数用于执行打包操作，这里的参数 stack 表示栈，t 表示要移动的元组
  static void execute(Stack& /*stack*/, std::tuple<Args...>&& /*t*/){};
};

// pack 函数模板，用于将元组打包到栈中
template <typename... Args>
inline void pack(Stack& stack, std::tuple<Args...>&& t) {
  // 使用 TuplePacker 执行打包操作，传入栈和移动后的元组 t
  TuplePacker<sizeof...(Args), Args...>::execute(stack, std::move(t));
}

// 命名空间结束声明，结束了命名空间 torch::jit 的定义
} // namespace torch::jit
```