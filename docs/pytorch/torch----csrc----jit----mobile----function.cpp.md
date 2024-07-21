# `.\pytorch\torch\csrc\jit\mobile\function.cpp`

```py
#include <ATen/core/dynamic_type.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/csrc/jit/mobile/prim_ops_registery.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

// 将 OpCode 转换为字符串表示
char const* toString(OpCode op);

namespace mobile {

// Function 类的构造函数，接受一个合格名称作为参数
Function::Function(c10::QualifiedName name) : name_(std::move(name)) {}

// Function 类的构造函数，接受一个合格名称、代码对象和可选的函数模式作为参数
Function::Function(
    c10::QualifiedName name,
    Code code,
    at::optional<c10::FunctionSchema> schema)
    : name_(std::move(name)),
      code_(std::move(code)),
      schema_(std::move(schema)) {}

// 返回函数的合格名称
const c10::QualifiedName& Function::qualname() const {
  return name_;
}

// 向函数的代码对象中追加指令，包括操作码、X、N 和调试句柄
void Function::append_instruction(OpCode op, int X, int N, int64_t dbg_handle) {
  // 检查操作码是否在移动模块中支持，否则抛出异常
  TORCH_CHECK(
      isOpSupportedInMobile(op),
      toString(op),
      " is not supported in mobile module.");
  // 向指令列表中添加新指令
  code_.instructions_.emplace_back(op, X, N);
  // 向调试句柄列表中添加调试句柄
  code_.debug_handles_.emplace_back(dbg_handle);
}

// 向函数的代码对象中追加指令，包括操作码、X 和 N
void Function::append_instruction(OpCode op, int X, int N) {
  // 检查操作码是否在移动模块中支持，否则抛出异常
  TORCH_CHECK(
      isOpSupportedInMobile(op),
      toString(op),
      " is not supported in mobile module.");
  // 向指令列表中添加新指令
  code_.instructions_.emplace_back(op, X, N);
}

// 向函数的代码对象中追加运算符名称、重载名称和可选的指定参数数目
void Function::append_operator(
    const std::string& name,
    const std::string& overload_name,
    const std::optional<int>& num_specified_args) {
  // 保留原始操作符名称在代码对象中
  code_.op_names_.emplace_back(name, overload_name);
  // 向操作符输入大小列表中添加指定参数数目，若无则添加-1
  code_.operator_input_sizes_.emplace_back(num_specified_args.value_or(-1));
}

// 将操作符名称转换为字符串表示
std::string operator_str(const c10::OperatorName& opname) {
  std::string result = opname.name;
  if (!opname.overload_name.empty()) {
    result += "." + opname.overload_name;
  }
  return result;
}

// 初始化函数的操作符列表，可选择是否检查操作符支持情况
bool Function::initialize_operators(bool should_check_operators) {
  if (code_.initialized) {
    return true;
  }
  // 未支持的操作符名称集合
  std::unordered_set<std::string> unsupported_op_names;
  // 调整操作符列表大小以匹配操作符名称列表
  code_.operators_.resize(code_.op_names_.size());
  // 所有操作符是否都受支持的标志位
  bool all_ops_supported = true;
  // 遍历操作符名称列表
  for (unsigned i = 0; i < code_.op_names_.size(); i++) {
    const auto& opname = code_.op_names_[i];
    int num_args = code_.operator_input_sizes_[i];
    // 获取可选的指定参数数目
    std::optional<int> num_specified_args =
        num_args < 0 ? c10::nullopt : std::optional<int>(num_args);
    // 创建操作符函数
    auto func = makeOperatorFunction(opname, num_specified_args);
    // 若操作符函数为空，则将操作符名称添加至不支持的操作符集合，并标记不支持
    if (!func.has_value()) {
      unsupported_op_names.insert(operator_str(opname));
      all_ops_supported = false;
    } else {
      // 否则将操作符函数添加至代码对象的操作符列表
      code_.operators_[i] = *func;
    }
  }
  // 若需要检查操作符支持情况，则执行以下代码块
  if (should_check_operators) {
    # 使用 TORCH_CHECK 宏来检查条件 unsupported_op_names 是否为空，如果不为空则抛出错误信息
    TORCH_CHECK(
        unsupported_op_names.empty(),
        "Following ops cannot be found: [",
        c10::Join(", ", unsupported_op_names),
        "]. Please check if the operator library is included in the build. If built with selected ops, check if these ops are in the list. If you are a Meta employee, please see fburl.com/missing_ops for a fix. Or post it in https://discuss.pytorch.org/c/mobile/");
  }
  # 将 code_.initialized 设置为 all_ops_supported 变量的值
  code_.initialized = all_ops_supported;
  # 返回变量 all_ops_supported 的值作为函数的结果
  return all_ops_supported;
}

void Function::append_constant(const c10::IValue& constant) {
  // 向代码对象的常量列表末尾添加一个常量
  code_.constants_.push_back(constant);
}

void Function::append_type(const at::TypePtr& type) {
  // 向代码对象的类型列表末尾添加一个类型
  code_.types_.push_back(type);
}

void Function::append_function(mobile::Function& function) {
  // 向代码对象的函数列表末尾添加一个函数指针
  code_.functions_.push_back(&function);
}

void Function::set_register_size(size_t size) {
  // 设置代码对象的寄存器大小
  code_.register_size_ = size;
}

int64_t Function::get_debug_handle(size_t pc) const {
  // 获取给定程序计数器位置的调试句柄
  TORCH_CHECK(
      pc < code_.debug_handles_.size(),
      "Module debug info index out of boundary.");
  return code_.debug_handles_[pc];
}

torch::jit::Function& Function::setSchema(c10::FunctionSchema schema) {
  // 设置函数对象的模式
  schema_ = std::move(schema);
  return *this;
}

bool Function::hasSchema() const {
  // 检查函数对象是否有模式
  return schema_.has_value();
}

const c10::FunctionSchema& Function::getSchema() const {
  // 获取函数对象的模式
  return *schema_;
}

void Function::run(Stack& stack) {
  // 初始化操作符并运行函数对象关联的代码
  initialize_operators(/* should_check_operators */ true);
  if (hasSchema()) { // 如果有模式，则检查并规范化输入参数
    getSchema().checkAndNormalizeInputs<c10::DynamicType>(
        stack, std::unordered_map<std::string, IValue>{} /*kwargs*/);
  }
  InterpreterState interp_state(code_);
  interp_state.run(stack);
}

at::IValue Function::operator()(Stack& stack) {
  // 运行函数并返回栈顶元素
  run(stack);
  return stack.front();
}

size_t Function::num_inputs() const {
  // 获取函数模式定义的输入参数数量
  return schema_->arguments().size();
}

bool Function::call(Stack&, c10::function_ref<void(const mobile::Code&)> f) {
  // 初始化操作符并调用函数对象的代码对象
  initialize_operators(true);
  f(code_);
  return true;
}

const Code& Function::get_code() const {
  // 获取函数对象关联的代码对象（常量引用）
  return code_;
}

Code& Function::get_code() {
  // 获取函数对象关联的代码对象（引用）
  return code_;
}

const std::vector<int64_t>& Function::getExceptionDebugHandles() const {
  // 获取解释器异常调试句柄列表
  return getInterpretersExceptionDebugHandles();
}

std::optional<std::function<void(Stack&)>> makeOperatorFunction(
    c10::OperatorName opname,
    std::optional<int> num_specified_args) {
  std::function<void(Stack&)> fn;
  const auto full_name = c10::toString(opname);
  const std::vector<c10::Argument>* pArgs = nullptr;
  bool promoted_op = mobile::hasPrimOpsFn(full_name);
  if (promoted_op) {
    // 如果是提升的操作符，则获取其操作函数
    fn = mobile::getPrimOpsFn(full_name);
  } else {
    std::shared_ptr<Operator> jit_op = findOperatorFor(opname);
    if (jit_op) {
      // 如果可以找到 JIT 操作符，则获取其操作函数，并指定参数列表
      fn = [jit_op](Stack& stack) { jit_op->getOperation()(stack); };
      pArgs = &jit_op->schema().arguments();
    } else {
      auto op = c10::Dispatcher::singleton().findSchema(opname);
      if (op.has_value()) {
        // 如果找到分发器中的操作符模式，则获取其调用函数并指定参数列表
        fn = [op](Stack& stack) { op->callBoxed(&stack); };
        if (op->hasSchema()) {
          pArgs = &op->schema().arguments();
        } else {
          TORCH_CHECK(false, "arguments are missing for operator ", opname);
        }
      } else {
        return c10::nullopt;
      }
    }
  }

  if (!promoted_op) {
    // 对于非提升的操作符，确保参数列表不为空
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(pArgs);
    const auto& args = *pArgs;
    // num_specified_args >= 0 表示参数数量已知
    ```
    // 如果指定了参数数量并且小于实际参数数量，则进行处理以保证向后兼容性。
    if (num_specified_args &&
        num_specified_args.value() < static_cast<int64_t>(args.size())) {
      // 创建一个 lambda 函数 fn，它接收一个 Stack 引用参数，用于处理默认值和输出参数。
      fn = [fn, num_specified_args, &args](Stack& stack) {
        std::vector<IValue> out_args;
        // 下面的逻辑从栈中弹出并临时存储所有的输出参数（如果有的话），
        // 并将它们的默认值推送到栈中。最后，将输出参数推回到栈中。
        for (size_t i = args.size() - 1; i > 0 && args.at(i).is_out(); i--) {
          out_args.push_back(stack.back());
          stack.pop_back();
        }
        // 检查输出参数的数量不超过指定的参数数量，以确保兼容性。
        TORCH_CHECK(
            static_cast<size_t>(num_specified_args.value()) >= out_args.size(),
            "The number of output arguments is: ",
            out_args.size(),
            ", which is more then the number of specified arguments: ",
            num_specified_args.value());
        // 计算需要推送默认值的参数起始索引，并为这些参数推送默认值。
        size_t start_index = num_specified_args.value() - out_args.size();
        for (size_t i = start_index; i < (args.size() - out_args.size()); ++i) {
          // 检查参数是否有默认值，如果没有，则抛出错误信息。
          TORCH_CHECK(
              args[i].default_value().has_value(),
              "Error happened at preparing for default values for the argument. The ",
              i,
              "th argument ",
              args[i].name(),
              " does not have a specified value or default value. ");
          // 将参数的默认值推送到栈中。
          stack.emplace_back(args[i].default_value());
        }
        // 将保存的输出参数逆序推回到栈中。
        stack.insert(stack.end(), out_args.rbegin(), out_args.rend());
        // 调用原始函数 fn 处理栈中的参数。
        fn(stack);
      };
    }
  }
  // 返回处理参数的 lambda 函数 fn。
  return fn;
// 定义成员函数 registerFunc，返回 Function 对象的引用
Function& Function::registerFunc(
    // 函数参数：函数名、指令集合、常量集合、类型集合、寄存器大小
    const std::string& qualified_name,
    const std::vector<Instruction>& instructions,
    const std::vector<c10::IValue>& constants,
    const std::vector<c10::TypePtr>& types,
    const size_t register_size) {
  
  // 静态局部变量，存储 QualifiedName 到 Function 对象的映射
  static std::unordered_map<c10::QualifiedName, Function> upgrader_function_holder;

  // 创建 QualifiedName 对象
  c10::QualifiedName name = c10::QualifiedName(qualified_name);

  // 在映射中查找指定名称的函数
  auto found = upgrader_function_holder.find(name);

  // 如果未找到该函数，则将其注册到映射中
  if (found == upgrader_function_holder.end()) {
    // 将函数对象插入映射中，并获取插入后的迭代器
    auto name_function_pair = upgrader_function_holder.emplace(name, Function(name));
    // 获取插入的 Function 对象的引用
    auto& func = name_function_pair.first->second;
    
    // 将指令集合中的指令添加到函数对象中
    for (auto const& inst : instructions) {
      func.append_instruction(inst.op, inst.X, inst.N);
    }
    
    // 将常量集合中的常量添加到函数对象中
    for (auto const& constant : constants) {
      func.append_constant(constant);
    }
    
    // 将类型集合中的类型添加到函数对象中
    for (auto const& type : types) {
      func.append_type(type);
    }
    
    // 设置函数对象的寄存器大小
    func.set_register_size(register_size);
    
    // 返回插入的函数对象的引用
    return func;
  }
  
  // 如果已经存在该函数，则返回映射中已有的函数对象的引用
  auto& upgrader_function_in_holder = found->second;
  return upgrader_function_in_holder;
}

// 命名空间闭合：mobile
} // namespace mobile

// 命名空间闭合：jit
} // namespace jit

// 命名空间闭合：torch
} // namespace torch
```