# `.\pytorch\test\cpp\jit\test_custom_class_registrations.cpp`

```
#include <test/cpp/jit/test_custom_class_registrations.h>

#include <torch/custom_class.h>
#include <torch/script.h>

#include <iostream>
#include <string>
#include <vector>

using namespace torch::jit;

namespace {

// 自定义结构体 DefaultArgs，继承自 torch::CustomClassHolder
struct DefaultArgs : torch::CustomClassHolder {
  int x; // 定义整数成员变量 x

  // 构造函数，默认参数为 3，初始化 x
  DefaultArgs(int64_t start = 3) : x(start) {}

  // 成员函数 increment，增加 x 的值，并返回增加后的值
  int64_t increment(int64_t val = 1) {
    x += val;
    return x;
  }

  // 成员函数 decrement，减少 x 的值，并返回减少后的值
  int64_t decrement(int64_t val = 1) {
    x -= val; // 减少操作符应该是减去而不是加
    return x;
// 定义一个结构体 PickleTester，继承自 torch 的 CustomClassHolder 类
struct PickleTester : torch::CustomClassHolder {
  // 构造函数，接受一个 int64_t 型向量 vals，并将其移动给成员变量 vals
  PickleTester(std::vector<int64_t> vals) : vals(std::move(vals)) {}
  // 成员变量，存储 int64_t 型向量
  std::vector<int64_t> vals;
};

// 线程安全的 Tensor 队列结构
struct TensorQueue : torch::CustomClassHolder {
  // 构造函数，接受一个 Tensor 类型的参数 t，初始化 init_tensor_ 成员变量
  explicit TensorQueue(at::Tensor t) : init_tensor_(t) {}

  // 另一个构造函数，接受一个 c10::Dict<std::string, at::Tensor> 类型的字典 dict
  explicit TensorQueue(c10::Dict<std::string, at::Tensor> dict) {
    // 从字典中读取并初始化 init_tensor_
    init_tensor_ = dict.at(std::string("init_tensor"));
    const std::string key = "queue";
    at::Tensor size_tensor;
    // 从字典中读取 size，计算队列大小
    size_tensor = dict.at(std::string(key + "/size")).cpu();
    const auto* size_tensor_acc = size_tensor.const_data_ptr<int64_t>();
    int64_t queue_size = size_tensor_acc[0];

    // 从字典中读取队列中的每个元素，并加入到队列中
    for (const auto index : c10::irange(queue_size)) {
      at::Tensor val;
      queue_[index] = dict.at(key + "/" + std::to_string(index));
      queue_.push_back(val);  // 这里似乎是个错误，应该是 queue_.push_back(queue_[index]);
    }
  }

  // 序列化方法，将对象转化为字典形式
  c10::Dict<std::string, at::Tensor> serialize() const {
    c10::Dict<std::string, at::Tensor> dict;
    dict.insert(std::string("init_tensor"), init_tensor_);
    const std::string key = "queue";
    dict.insert(
        key + "/size", torch::tensor(static_cast<int64_t>(queue_.size())));
    // 将队列中的每个元素加入到字典中
    for (const auto index : c10::irange(queue_.size())) {
      dict.insert(key + "/" + std::to_string(index), queue_[index]);
    }
    return dict;
  }

  // 将元素推入队列尾部的方法，添加了线程安全的锁保护
  void push(at::Tensor x) {
    std::lock_guard<std::mutex> guard(mutex_);
    queue_.push_back(x);
  }

  // 弹出队列头部元素并返回的方法，如果队列为空则返回 init_tensor_
  // 添加了线程安全的锁保护
  at::Tensor pop() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!queue_.empty()) {
      auto val = queue_.front();
      queue_.pop_front();
      return val;
    } else {
      return init_tensor_;
    }
  }

  // 返回队列头部元素的方法（只读），如果队列为空则返回 init_tensor_
  // 添加了线程安全的锁保护
  at::Tensor top() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!queue_.empty()) {
      auto val = queue_.front();
      return val;
    } else {
      return init_tensor_;
    }
  }

  // 返回队列大小的方法
  int64_t size() {
    return queue_.size();
  }

  // 判断队列是否为空的方法
  bool is_empty() {
    std::lock_guard<std::mutex> guard(mutex_);
    return queue_.empty();
  }

  // 返回队列大小的浮点数表示
  double float_size() {
    return 1. * queue_.size();
  }

  // 克隆队列中所有元素的方法
  std::vector<at::Tensor> clone_queue() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<at::Tensor> ret;
    for (const auto& t : queue_) {
      ret.push_back(t.clone());
    }
    return ret;
  }

  // 获取原始队列中所有元素的方法
  std::vector<at::Tensor> get_raw_queue() {
    std::vector<at::Tensor> raw_queue(queue_.begin(), queue_.end());
    return raw_queue;
  }

  // 对象扁平化方法，返回队列名和原始队列的元组
  std::tuple<std::tuple<std::string, std::vector<at::Tensor>>> __obj_flatten__() {
    return std::tuple(std::tuple("queue", this->get_raw_queue()));
  }

 private:
  // 使用 deque 存储 Tensor 对象的队列
  std::deque<at::Tensor> queue_;
  // 用于线程安全的互斥锁
  std::mutex mutex_;
  // 初始 Tensor 对象，用于空队列返回值
  at::Tensor init_tensor_;
};
// 返回一个张量，该张量是由给定实例的最后一个值和 4 组成的零张量
at::Tensor take_an_instance(const c10::intrusive_ptr<PickleTester>& instance) {
  return torch::zeros({instance->vals.back(), 4});
}

// 结构体 ElementwiseInterpreter 继承自 torch::CustomClassHolder
struct ElementwiseInterpreter : torch::CustomClassHolder {
  // 指令类型定义，包括操作字符串 "op"，输入值名称列表 "inputs"，输出值名称 "output"
  using InstructionType = std::tuple<
      std::string /*op*/,
      std::vector<std::string> /*inputs*/,
      std::string /*output*/>;

  // 默认构造函数，使用 NOLINTNEXTLINE(modernize-use-equals-default) 禁止 Lint 提示
  ElementwiseInterpreter() {}

  // 将指令列表加载到解释器中
  void setInstructions(std::vector<InstructionType> instructions) {
    instructions_ = std::move(instructions);
  }

  // 添加常量到解释器中，常量以名称为键，可以在指令中通过名称引用
  void addConstant(const std::string& name, at::Tensor value) {
    constants_.insert_or_assign(name, std::move(value));
  }

  // 设置函数的位置输入参数的名称列表
  void setInputNames(std::vector<std::string> input_names) {
    input_names_ = std::move(input_names);
  }

  // 设置函数的输出名称，应该匹配指令列表中某个指令的 "output" 字段
  void setOutputName(std::string output_name) {
    output_name_ = std::move(output_name);
  }

  // 调用解释器，接受位置输入列表并返回单个输出张量
  at::Tensor __call__(std::vector<at::Tensor> inputs) {
    // 环境变量，用于保存局部变量
    std::unordered_map<std::string, at::Tensor> environment;

    // 根据指定的名称加载输入
    if (inputs.size() != input_names_.size()) {
      std::stringstream err;
      err << "Expected " << input_names_.size() << " inputs, but got "
          << inputs.size() << "!";
      throw std::runtime_error(err.str());
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      environment[input_names_[i]] = inputs[i];
    }
    for (InstructionType& instr : instructions_) {
      // 遍历指令列表 instructions_

      // Retrieve all input values for this op
      std::vector<at::Tensor> inputs;
      // 为当前操作收集所有输入值
      for (const auto& input_name : std::get<1>(instr)) {
        // 遍历当前指令的输入名称列表

        // Operator output values shadow constants.
        // Imagine all constants are defined in statements at the beginning
        // of a function (a la K&R C). Any definition of an output value must
        // necessarily come after constant definition in textual order. Thus,
        // We look up values in the environment first then the constant table
        // second to implement this shadowing behavior
        // 操作符的输出值会覆盖常量值。
        // 假设所有常量都在函数开头的语句中定义（类似于K&R C风格）。
        // 任何输出值的定义必须在文本顺序中常量定义之后。因此，
        // 我们先在环境中查找值，然后在常量表中查找，以实现这种遮蔽行为
        if (environment.find(input_name) != environment.end()) {
          // 如果环境中存在该输入名称的值，则将其添加到输入向量中
          inputs.push_back(environment.at(input_name));
        } else if (constants_.find(input_name) != constants_.end()) {
          // 否则，如果常量表中存在该输入名称的值，则将其添加到输入向量中
          inputs.push_back(constants_.at(input_name));
        } else {
          // 否则，抛出异常，指令引用了未知的值
          std::stringstream err;
          err << "Instruction referenced unknown value " << input_name << "!";
          throw std::runtime_error(err.str());
        }
      }

      // Run the specified operation
      // 执行指定的操作
      at::Tensor result;
      const auto& op = std::get<0>(instr);
      // 获取当前指令的操作符名称
      if (op == "add") {
        // 如果操作符是 "add"
        if (inputs.size() != 2) {
          // 如果输入数量不等于2，抛出异常
          throw std::runtime_error("Unexpected number of inputs for add op!");
        }
        // 执行加法操作
        result = inputs[0] + inputs[1];
      } else if (op == "mul") {
        // 如果操作符是 "mul"
        if (inputs.size() != 2) {
          // 如果输入数量不等于2，抛出异常
          throw std::runtime_error("Unexpected number of inputs for mul op!");
        }
        // 执行乘法操作
        result = inputs[0] * inputs[1];
      } else {
        // 否则，抛出异常，未知的操作符
        std::stringstream err;
        err << "Unknown operator " << op << "!";
        throw std::runtime_error(err.str());
      }

      // Write back result into environment
      // 将结果写回环境中
      const auto& output_name = std::get<2>(instr);
      // 获取当前指令的输出名称
      environment[output_name] = std::move(result);
      // 将结果保存到环境中
    }

    if (!output_name_) {
      // 如果输出名称未指定，抛出异常
      throw std::runtime_error("Output name not specified!");
    }

    return environment.at(*output_name_);
    // 返回指定输出名称在环境中的值
  }

  // Ser/De infrastructure. See
  // https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html#defining-serialization-deserialization-methods-for-custom-c-classes
  // for more info.

  // This is the type we will use to marshall information on disk during
  // ser/de. It is a simple tuple composed of primitive types and simple
  // collection types like vector, optional, and dict.
  // 这是我们在序列化/反序列化时用于在磁盘上编组信息的类型。
  // 它是一个由基本类型和简单集合类型（如vector、optional和dict）组成的简单元组。
  using SerializationType = std::tuple<
      std::vector<std::string> /*input_names_*/,
      std::optional<std::string> /*output_name_*/,
      c10::Dict<std::string, at::Tensor> /*constants_*/,
      std::vector<InstructionType> /*instructions_*/
      >;

  // This function yields the SerializationType instance for `this`.
  // 此函数生成当前对象的 SerializationType 实例。
  SerializationType __getstate__() const {
    // 返回一个 SerializationType 类型的对象，其中包含 input_names_, output_name_, constants_, instructions_ 这些成员的数据
    return SerializationType{
        input_names_, output_name_, constants_, instructions_};
    }
    
    // 根据给定的 SerializationType 对象状态，创建并返回一个 ElementwiseInterpreter 实例
    static c10::intrusive_ptr<ElementwiseInterpreter> __setstate__(
        SerializationType state) {
      // 创建 ElementwiseInterpreter 的智能指针实例
      auto instance = c10::make_intrusive<ElementwiseInterpreter>();
      // 使用给定的 state 对象初始化 instance 的成员变量
      std::tie(
          instance->input_names_,
          instance->output_name_,
          instance->constants_,
          instance->instructions_) = std::move(state);
      // 返回初始化后的 ElementwiseInterpreter 实例
      return instance;
    }
    
    // 类的成员变量
    std::vector<std::string> input_names_;                       // 输入名称列表
    std::optional<std::string> output_name_;                     // 可选的输出名称
    c10::Dict<std::string, at::Tensor> constants_;               // 字典，存储字符串到张量的常量映射
    std::vector<InstructionType> instructions_;                  // 指令类型的向量，存储指令列表
};

// 定义一个名为 ReLUClass 的结构体，继承自 torch::CustomClassHolder
struct ReLUClass : public torch::CustomClassHolder {
  // 定义成员函数 run，接收一个张量 t 作为参数，返回 t 的 ReLU 函数应用结果
  at::Tensor run(const at::Tensor& t) {
    return t.relu();
  }
};

// 定义一个名为 FlattenWithTensorOp 的结构体，继承自 torch::CustomClassHolder
struct FlattenWithTensorOp : public torch::CustomClassHolder {
  // 构造函数，接收一个张量 t 作为参数，并将其存储在成员变量 t_ 中
  explicit FlattenWithTensorOp(at::Tensor t) : t_(t) {}

  // 返回成员变量 t_
  at::Tensor get() {
    return t_;
  }

  // 实现特殊方法 __obj_flatten__
  std::tuple<std::tuple<std::string, at::Tensor>> __obj_flatten__() {
    // 返回一个元组，包含字符串 "t" 和张量 this->t_ 的 sin 函数结果
    return std::tuple(std::tuple("t", this->t_.sin()));
  }

 private:
  at::Tensor t_;
  ; // 无效的额外分号，可能是误留的
};

// 定义一个名为 ContainsTensor 的结构体，继承自 torch::CustomClassHolder
struct ContainsTensor : public torch::CustomClassHolder {
  // 构造函数，接收一个张量 t 作为参数，并将其存储在成员变量 t_ 中
  explicit ContainsTensor(at::Tensor t) : t_(t) {}

  // 返回成员变量 t_
  at::Tensor get() {
    return t_;
  }

  // 实现特殊方法 __obj_flatten__
  std::tuple<std::tuple<std::string, at::Tensor>> __obj_flatten__() {
    // 返回一个元组，包含字符串 "t" 和张量 this->t_
    return std::tuple(std::tuple("t", this->t_));
  }

  at::Tensor t_;
};

// 结构体定义结束

}

// 函数定义，接收一个 Foo 类的智能指针 foo 和一个张量 x 作为参数，调用 foo 对象的 add_tensor 方法并返回结果张量
at::Tensor takes_foo(c10::intrusive_ptr<Foo> foo, at::Tensor x) {
  return foo->add_tensor(x);
}

// 函数定义，接收一个 Foo 类的智能指针 foo 和一个张量 x 作为参数，调用 foo 对象的 add_tensor 方法多次，将结果存入 vector 并返回
std::vector<at::Tensor> takes_foo_list_return(
    c10::intrusive_ptr<Foo> foo,
    at::Tensor x) {
  std::vector<at::Tensor> result;
  result.reserve(3);
  auto a = foo->add_tensor(x);
  auto b = foo->add_tensor(a);
  auto c = foo->add_tensor(b);
  result.push_back(a);
  result.push_back(b);
  result.push_back(c);
  return result;
}

// 函数定义，接收一个 Foo 类的智能指针 foo 和一个张量 x 作为参数，调用 foo 对象的 add_tensor 方法多次，返回两个结果张量的 tuple
std::tuple<at::Tensor, at::Tensor> takes_foo_tuple_return(
    c10::intrusive_ptr<Foo> foo,
    at::Tensor x) {
  auto a = foo->add_tensor(x);
  auto b = foo->add_tensor(a);
  return std::make_tuple(a, b);
}

// 函数定义，接收一个 TensorQueue 类的智能指针 tq 和一个张量 x 作为参数，调用 tq 对象的 push 方法
void queue_push(c10::intrusive_ptr<TensorQueue> tq, at::Tensor x) {
  tq->push(x);
}

// 函数定义，接收一个 TensorQueue 类的智能指针 tq 作为参数，调用 tq 对象的 pop 方法并返回结果张量
at::Tensor queue_pop(c10::intrusive_ptr<TensorQueue> tq) {
  return tq->pop();
}

// 函数定义，接收一个 TensorQueue 类的智能指针 tq 作为参数，调用 tq 对象的 size 方法并返回结果
int64_t queue_size(c10::intrusive_ptr<TensorQueue> tq) {
  return tq->size();
}

// 定义 TORCH_LIBRARY_FRAGMENT，注册 TorchScriptTesting 库的函数和实现
TORCH_LIBRARY_FRAGMENT(_TorchScriptTesting, m) {
  // 注册抽象 PyStub 函数
  m.impl_abstract_pystub("torch.testing._internal.torchbind_impls");
  // 定义函数 takes_foo_cia 的 TorchScript 绑定
  m.def(
      "takes_foo_cia(__torch__.torch.classes._TorchScriptTesting._Foo foo, Tensor x) -> Tensor");
  // 定义函数 queue_pop 的 TorchScript 绑定
  m.def(
      "queue_pop(__torch__.torch.classes._TorchScriptTesting._TensorQueue foo) -> Tensor");
  // 定义函数 queue_push 的 TorchScript 绑定
  m.def(
      "queue_push(__torch__.torch.classes._TorchScriptTesting._TensorQueue foo, Tensor x) -> ()");
  // 定义函数 queue_size 的 TorchScript 绑定
  m.def(
      "queue_size(__torch__.torch.classes._TorchScriptTesting._TensorQueue foo) -> int");
}

// 定义 TORCH_LIBRARY_IMPL，实现 TorchScriptTesting 库的函数和实现（CPU 版本）
TORCH_LIBRARY_IMPL(_TorchScriptTesting, CPU, m) {
  // 实现函数 takes_foo 的 TorchScript 绑定
  m.impl("takes_foo", takes_foo);
  // 实现函数 takes_foo_list_return 的 TorchScript 绑定
  m.impl("takes_foo_list_return", takes_foo_list_return);
  // 实现函数 takes_foo_tuple_return 的 TorchScript 绑定
  m.impl("takes_foo_tuple_return", takes_foo_tuple_return);
  // 实现函数 queue_push 的 TorchScript 绑定
  m.impl("queue_push", queue_push);
  // 实现函数 queue_pop 的 TorchScript 绑定
  m.impl("queue_pop", queue_pop);
  // 实现函数 queue_size 的 TorchScript 绑定
  m.impl("queue_size", queue_size);
}

// 定义 TORCH_LIBRARY_IMPL，实现 TorchScriptTesting 库的函数和实现（Meta 版本）
TORCH_LIBRARY_IMPL(_TorchScriptTesting, Meta, m) {
  // 实现函数 takes_foo 的 TorchScript 绑定
  m.impl("takes_foo", &takes_foo);
  // 实现函数 takes_foo_list_return 的 TorchScript 绑定
  m.impl("takes_foo_list_return", takes_foo_list_return);
  // 实现函数 takes_foo_tuple_return 的 TorchScript 绑定
  m.impl("takes_foo_tuple_return", takes_foo_tuple_return);
}

// 定义 TORCH_LIBRARY_IMPL，实现 TorchScriptTesting 库的函数和实现（CompositeImplicitAutograd 版本）
TORCH_LIBRARY_IMPL(_TorchScriptTesting, CompositeImplicitAutograd, m) {
  // 实现函数 takes_foo_cia 的 TorchScript 绑定
  m.impl("takes_foo_cia", takes_foo);
}

// 注释：需要实现 BackendSelect，因为这两个操作符没有张量输入。
TORCH_LIBRARY_IMPL(_TorchScriptTesting, BackendSelect, m) {
  // 将函数 `queue_pop` 注册为 TorchScript 中 `_TorchScriptTesting` 模块的实现
  m.impl("queue_pop", queue_pop);
  // 将函数 `queue_size` 注册为 TorchScript 中 `_TorchScriptTesting` 模块的实现
  m.impl("queue_size", queue_size);
}

} // namespace
```