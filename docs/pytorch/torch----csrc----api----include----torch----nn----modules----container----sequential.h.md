# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\sequential.h`

```py
#pragma once
// 使用 #pragma once 确保头文件只被编译一次

#include <torch/detail/static.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/named_any.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

/// A list of `Module`s that acts as a `Module` itself.
///
/// A `Sequential` is fundamentally a list of `Module`s, each with a `forward()`
/// method. `Sequential` provides a `forward()` method of its own, which accepts
/// any input and forwards it to the first module it stores. It then "chains"
/// outputs to inputs sequentially for each subsequent module, finally returning
/// the output of the last module. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::Sequential seq(
///     torch::nn::Linear(3, 4),
///     torch::nn::BatchNorm1d(4),
///     torch::nn::Dropout(0.5)
///   );
///
///   auto output = seq->forward(torch::ones(3));
///
/// \endrst
///
/// This can conceptually be thought of as the following loop (using Python as
/// pseudocode):
///
/// \rst
/// .. code-block:: python
///
///   def forward(sequential, input):
///     for module in sequential:
///       input = module(input)
///     return input
///
/// \endrst
///
/// Why should you use `Sequential` instead of a simple `std::vector`? The value
/// a `Sequential` provides over manually calling a sequence of modules is that
/// it allows treating the whole container *as a single module*, such that
/// performing a transformation on the `Sequential` applies to each of the
/// modules it stores (which are each a registered submodule of the
/// `Sequential`). For example, calling
/// `.to(torch::kCUDA)` on a `Sequential` will move each module in the list to
/// CUDA memory. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::Sequential seq(
///     torch::nn::Linear(3, 4),
///     torch::nn::BatchNorm1d(4),
///     torch::nn::Dropout(0.5)
///   );
///
///   // Convert all modules to CUDA.
///   seq->to(torch::kCUDA);
///
/// \endrst
///
/// Finally, `Sequential` provides a lightweight container API, such as allowing
/// iteration over submodules, positional access, adding a new module after
/// construction via `push_back`, as well as joining two `Sequential`s via
/// `extend`.
///
/// \rst
/// .. attention::
///   One current limitation of `Sequential` is that all except the first module
///   must accept a single argument. If your modules need to take multiple
///   arguments, you should define them to take and return tuples.
/// \endrst
class Sequential : public torch::nn::CloneableModule<Sequential> {
    // 定义 Sequential 类，它继承自 torch::nn::CloneableModule<Sequential>

public:
    // 公共部分开始

    /// Constructs an empty Sequential.
    Sequential() {}

    /// Constructs a Sequential from a list of modules.
    ///
    /// \param modules A variadic list of modules to add to the Sequential.
    template <typename... Modules>
    explicit Sequential(Modules&&... modules)
        : modules_(std::forward<Modules>(modules)...) {}

    /// Moves the modules from `other` into `this`.
    ///
    /// \param other Another Sequential instance to move from.
    Sequential(Sequential&& other) noexcept
        : modules_(std::move(other.modules_)) {}

    /// Moves the modules from `other` into `this`.
    ///
    /// \param other Another Sequential instance to move from.
    /// \returns Reference to the current Sequential instance.
    Sequential& operator=(Sequential&& other) noexcept {
        modules_ = std::move(other.modules_);
        return *this;
    }

    /// Returns the number of modules in the Sequential container.
    size_t size() const noexcept {
        return modules_.size();
    }

    /// Pushes a module to the back of the Sequential container.
    ///
    /// \param module The module to add.
    void push_back(torch::nn::AnyModule module) {
        modules_.push_back(std::move(module));
    }

    /// Accesses the module at position `index` in the Sequential container.
    ///
    /// \param index The index of the module to access.
    /// \returns Reference to the module at the specified index.
    torch::nn::AnyModule& operator[](size_t index) {
        return modules_[index];
    }

    /// Accesses the module at position `index` in the Sequential container (const version).
    ///
    /// \param index The index of the module to access.
    /// \returns Const reference to the module at the specified index.
    const torch::nn::AnyModule& operator[](size_t index) const {
        return modules_[index];
    }

    /// Applies the `forward()` method of each module in sequence.
    ///
    /// \param input The input tensor or data to forward through the Sequential.
    /// \returns The output tensor or data produced by the last module in the Sequential.
    template <typename T>
    T forward(T input) {
        for (auto& module : modules_) {
            input = module->forward<T>(std::move(input));
        }
        return input;
    }

    // 私有部分开始

private:
    std::vector<torch::nn::AnyModule> modules_; // 存储模块的容器
};

} // namespace nn
} // namespace torch
    // 定义迭代器类型，用于遍历 modules_ 容器
    using Iterator = std::vector<AnyModule>::iterator;
    // 定义常量迭代器类型，用于以只读方式遍历 modules_ 容器
    using ConstIterator = std::vector<AnyModule>::const_iterator;

    // 默认构造函数，使用默认方式初始化 SequentialImpl 对象
    SequentialImpl() = default;

    /// 从可变参数模块列表构造 Sequential
    template <typename... Modules>
    explicit SequentialImpl(Modules&&... modules) {
        // 预留足够的空间以容纳所有传入的模块
        modules_.reserve(sizeof...(Modules));
        // 将所有传入的模块添加到 modules_ 中
        push_back(std::forward<Modules>(modules)...);
    }

    /// 从 OrderedDict 构造 Sequential，其中包含命名的 AnyModule
    explicit SequentialImpl(
        torch::OrderedDict<std::string, AnyModule>&& ordered_dict) {
        // 预留足够的空间以容纳 ordered_dict 的大小
        modules_.reserve(ordered_dict.size());
        // 遍历 ordered_dict，将每个项添加到 modules_ 中
        for (auto& item : ordered_dict) {
            push_back(item.key(), std::move(item.value()));
        }
    }

    /// 从初始化列表构造 Sequential，其中包含命名的 AnyModule
    explicit SequentialImpl(std::initializer_list<NamedAnyModule> named_modules) {
        // 预留足够的空间以容纳 named_modules 的大小
        modules_.reserve(named_modules.size());
        // 遍历 named_modules，将每个命名模块添加到 modules_ 中
        for (const auto& named_module : named_modules) {
            push_back(named_module.name(), named_module.module());
        }
    }

    /// 特殊的克隆函数，用于 Sequential，因为它不使用 reset()
    std::shared_ptr<Module> clone(
        const optional<Device>& device = nullopt) const override {
        // 创建一个新的 SequentialImpl 对象的共享指针
        auto clone = std::make_shared<SequentialImpl>();
        // 遍历当前对象的 modules_ 容器，克隆每个模块并添加到 clone 对象中
        for (const auto& module : modules_) {
            clone->push_back(module.clone(device));
        }
        // 返回克隆后的对象的共享指针
        return clone;
    }

    /// `reset()` 对于 Sequential 是空的，因为它没有自己的参数
    void reset() override {}

    /// 将 Sequential 模块漂亮地打印到给定的流中
    void pretty_print(std::ostream& stream) const override {
        stream << "torch::nn::Sequential";
    }

    /// 将输入输入到第一个模块，然后将输出链式传递到输入，返回最后的输出
    ///
    /// 在 Python 中的概念性实现如下：
    ///
    /// \rst
    /// .. code-block:: python
    ///
    ///   def forward(sequential, input):
    ///     for module in sequential:
    ///       input = module(input)
    ///     return input
    ///
    /// \endrst
    ///
    /// 返回类型被视为第一个模板参数，默认为 Tensor。如果 Sequential 中的最后一个模块返回另一种类型 T，
    /// 则应该使用 forward<T>(inputs) 而不仅仅是 forward(inputs)：
    ///
    /// \rst
    /// .. code-block:: cpp
    ///
    ///   torch::Tensor tensor = sequential1->forward(inputs);
    ///   int integer = sequential2->forward<int>(inputs);
    ///   float value = sequential3->forward<float>(inputs);
    ///
    /// \endrst
    template <typename ReturnType = Tensor, typename... InputTypes>
    ReturnType forward(InputTypes&&... inputs) {
        // 检查 modules_ 是否为空，如果为空则抛出异常
        TORCH_CHECK(!is_empty(), "Cannot call forward() on an empty Sequential");

        // 获取 modules_ 的开始迭代器
        auto iterator = modules_.begin();
    // 从迭代器中获取输入，并对输入进行处理
    auto input = iterator->any_forward(std::forward<InputTypes>(inputs)...);

    // 迭代处理所有的模块，对输入进行连续处理
    for (++iterator; iterator != modules_.end(); ++iterator) {
      input = iterator->any_forward(std::move(input));
    }

    // 检查返回值的类型，并在请求的返回类型不匹配时提供友好的错误消息
    if (auto* return_value = input.template try_get<ReturnType>()) {
      return std::move(*return_value);
    }
    // 抛出错误，指明实际返回值类型和请求的返回类型不匹配
    AT_ERROR(
        "The type of the return value is ",
        c10::demangle(input.type_info().name()),
        ", but you asked for type ",
        c10::demangle(typeid(ReturnType).name()));
  }

  /// 向 `Sequential` 容器中添加一个新的（装箱的）`Module`。
  template <typename ModuleType>
  void push_back(std::shared_ptr<ModuleType> module_ptr) {
    push_back(std::to_string(modules_.size()), std::move(module_ptr));
  }

  /// 向 `Sequential` 容器中添加一个带名称的新的（装箱的）`Module`。
  template <typename ModuleType>
  void push_back(std::string name, std::shared_ptr<ModuleType> module_ptr) {
    push_back(std::move(name), AnyModule(std::move(module_ptr)));
  }

  /// 向 `Sequential` 容器中添加一个新的 `Module`，内部移动或复制到 `shared_ptr`。
  /// 允许传递值类型，并让容器处理装箱。
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  void push_back(M&& module) {
    push_back(std::to_string(modules_.size()), std::forward<M>(module));
  }

  /// 向 `Sequential` 容器中添加一个带名称的新的 `Module`，内部移动或复制到 `shared_ptr`。
  /// 允许传递值类型，并让容器处理装箱。
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  void push_back(std::string name, M&& module) {
    using Type = typename std::remove_reference_t<M>;
    push_back(std::move(name), std::make_shared<Type>(std::forward<M>(module)));
  }

  /// 解包 `ModuleHolder` 的包含模块并添加到 `Sequential` 中。
  template <typename M>
  void push_back(const ModuleHolder<M>& module_holder) {
    push_back(std::to_string(modules_.size()), module_holder);
  }

  /// 解包 `ModuleHolder` 的包含带名称的模块并添加到 `Sequential` 中。
  template <typename M>
  void push_back(std::string name, const ModuleHolder<M>& module_holder) {
    push_back(std::move(name), module_holder.ptr());
  }

  /// 遍历容器并对每个值调用 `push_back()`。
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& module : container) {
      push_back(module);
    }
  }

  /// 向 `Sequential` 中添加一个类型擦除的 `AnyModule`。
  void push_back(AnyModule any_module) {
  // 将新模块添加到容器的末尾，并使用模块数量作为其名称
  void push_back(std::to_string(modules_.size()), std::move(any_module));

  // 将新模块添加到容器的末尾，使用给定的名称，并注册模块
  void push_back(std::string name, AnyModule any_module) {
    modules_.push_back(std::move(any_module));
    const auto index = modules_.size() - 1;
    register_module(std::move(name), modules_[index].ptr());
  }

  /// 返回指向 `Sequential` 开始位置的迭代器
  Iterator begin() {
    return modules_.begin();
  }

  /// 返回指向 `Sequential` 开始位置的常量迭代器
  ConstIterator begin() const {
    return modules_.begin();
  }

  /// 返回指向 `Sequential` 结束位置的迭代器
  Iterator end() {
    return modules_.end();
  }

  /// 返回指向 `Sequential` 结束位置的常量迭代器
  ConstIterator end() const {
    return modules_.end();
  }

  /// 尝试返回给定索引处的模块，并将其作为请求类型返回
  /// 如果索引超出范围或类型不匹配，则抛出异常
  template <typename T>
  T& at(size_t index) {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call Sequential::at with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index].get<T>();
  }

  /// 尝试返回给定索引处的模块，并将其作为请求类型的常量引用返回
  /// 如果索引超出范围或类型不匹配，则抛出异常
  template <typename T>
  const T& at(size_t index) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call Sequential::at with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index].get<T>();
  }

  /// 尝试返回一个 `std::shared_ptr`，其动态类型与给定索引处的模块相同
  /// 如果索引超出范围，则抛出异常
  std::shared_ptr<Module> ptr(size_t index) const {
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index].ptr();
  }

  /// 尝试返回一个 `std::shared_ptr`，其类型与提供的类型相同
  /// 如果索引超出范围或类型不匹配，则抛出异常
  template <typename T>
  std::shared_ptr<T> ptr(size_t index) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call Sequential::ptr with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index].ptr<T>();
  }

  /// 与 `ptr(index)` 相同
  std::shared_ptr<Module> operator[](size_t index) const {
    // 这是唯一可以在没有类型的情况下调用的方法
    return ptr(index);
  }

  /// 返回 `Sequential` 容器的当前大小
  size_t size() const noexcept {
    return modules_.size();
  }

  /// 如果 `Sequential` 容器中没有模块，则返回 true
  bool is_empty() const noexcept {
    return size() == 0;
  }



// 检查容器是否为空，返回布尔值表示是否为空



 private:
  /// Takes a First *and* Second parameter, to avoid ambiguity when a parameter
  /// pack has only one type, in which case the template would be preferred,
  /// even if the other `push_back` functions are better fits (e.g. `unique_ptr`
  /// -> `shared_ptr` overload).
  /// NOTE: We explicitly avoid matching this template with
  /// `push_back(std::string("name"), module)` or `push_back("name", module)`,
  /// since they should be handled by their respective `push_back` functions.
  template <
      typename First,
      typename Second,
      typename... Rest,
      typename = std::enable_if_t<
          !std::is_same_v<First, std::string> &&
          // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
          !std::is_same_v<std::decay_t<First>, std::decay_t<const char (&)[]>>>>
  void push_back(First&& first, Second&& second, Rest&&... rest) {
    push_back(std::forward<First>(first));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `push_back()` a final time (above).
    push_back(std::forward<Second>(second), std::forward<Rest>(rest)...);
  }



// 定义了一个模板函数 `push_back`，接受两个及以上参数，用于推入到容器中。
// 如果参数类型为字符串或者字符数组，不会匹配到这个模板，以免与其他重载冲突。
// 函数通过递归调用自身来处理参数包中的每一个参数，直到参数包为空。



  /// The base case, when the list of modules is empty.
  void push_back() {}



// 当模块列表为空时的基本情况处理函数。



  // Box the AnyModules to give Sequential reference semantics, like the rest of
  // the API. Note that this is not required otherwise, this could just be a
  // `vector<AnyModule>`.
  std::vector<AnyModule> modules_;



// 使用框架将 AnyModule 包装成顺序引用语义，与 API 的其余部分保持一致。
// 注意，否则这只是一个 `vector<AnyModule>`，不需要额外的封装。
};

/// A `ModuleHolder` subclass for `SequentialImpl`.
/// This class inherits from `ModuleHolder` and specifically holds instances of
/// `SequentialImpl`, which encapsulates the sequential module behavior in PyTorch.
/// See the documentation for `SequentialImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
class Sequential : public torch::nn::ModuleHolder<SequentialImpl> {
 public:
  using torch::nn::ModuleHolder<SequentialImpl>::ModuleHolder;

  /// Default constructor for `Sequential`.
  Sequential() : ModuleHolder() {}

  /// Constructor that initializes `Sequential` from a list of named `AnyModule`s.
  /// Enables construction like `Sequential sequential({{"m1", M(1)}, {"m2", M(2)}})`.
  /// \param named_modules A list of named `AnyModule` instances to initialize the sequence.
  Sequential(std::initializer_list<NamedAnyModule> named_modules)
      : ModuleHolder(std::make_shared<SequentialImpl>(named_modules)) {}
};
} // namespace nn
} // namespace torch
```