# `.\pytorch\torch\csrc\distributed\rpc\script_call.cpp`

```
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace distributed {
namespace rpc {

// 定义内置操作的命名空间前缀
const std::string ScriptCall::BUILTIN_OP_NAMESPACE_("torch.ops.aten.");
// 定义 ATen 操作的前缀
const std::string ScriptCall::ATEN_PREFIX_("aten::");

// 构造函数，接受操作符和栈作为参数
ScriptCall::ScriptCall(
    std::shared_ptr<Operator> op,
    std::vector<at::IValue>&& stack)
    : op_(std::move(op)), stack_(stack), isAsyncExecution_(false) {}

// 构造函数，接受限定名称、栈和异步执行标志作为参数
ScriptCall::ScriptCall(
    const c10::QualifiedName& qualifiedName,
    std::vector<at::IValue>&& stack,
    const bool isAsyncExecution)
    : qualifiedName_(qualifiedName),
      stack_(stack),
      isAsyncExecution_(isAsyncExecution) {}

// 判断是否有操作符
bool ScriptCall::hasOp() const {
  return op_ ? true : false;
}

// 返回操作符的共享指针
std::shared_ptr<Operator> ScriptCall::op() const {
  return *op_;
}

// 判断是否有限定名称
bool ScriptCall::hasQualifiedName() const {
  return qualifiedName_ ? true : false;
}

// 返回限定名称的引用
const c10::QualifiedName& ScriptCall::qualifiedName() const {
  return *qualifiedName_;
}

// 返回栈中的 IValue 向量
const std::vector<at::IValue>& ScriptCall::stack() const {
  return stack_;
}

// 返回栈中的 IValue 向量的可变引用
std::vector<at::IValue>& ScriptCall::stackRef() {
  return stack_;
}

// 将 ScriptCall 对象转换为 IValue 向量
void ScriptCall::toIValues(std::vector<at::IValue>& ivalues) const {
  // 将栈中的值逐个添加到 ivalues 中
  for (auto& value : stack_) {
    ivalues.push_back(value);
  }

  if (hasOp()) {
    // 如果有操作符，向 ivalues 中添加其字符串化的 schema
    TORCH_CHECK(
        !hasQualifiedName(),
        "It is builtin operator call, qualifiedName_ should not be set.");
    ivalues.emplace_back(toString((*op_)->schema()));
    // 插入限定名称
    auto opName = (*op_)->schema().name();
    TORCH_CHECK(
        opName.find("::") == opName.rfind("::") &&
            opName.rfind(ATEN_PREFIX_) == 0,
        "Unexpected operator name ",
        opName);
    // 转换操作符名称格式，例如 aten::add -> torch.ops.aten.add
    opName.replace(0, ATEN_PREFIX_.length(), BUILTIN_OP_NAMESPACE_);
    ivalues.emplace_back(std::move(opName));
  } else if (hasQualifiedName()) {
    // 如果有限定名称，向 ivalues 中添加异步执行标志和限定名称字符串
    ivalues.emplace_back(isAsyncExecution());
    TORCH_CHECK(
        !hasOp(),
        "It is TorchScript function call, operator should not be set.");
    ivalues.emplace_back((*qualifiedName_).qualifiedName());
  } else {
    // 否则抛出内部断言错误
    TORCH_INTERNAL_ASSERT(
        false,
        "Either builtin operator or TorchScript function name should be set.");
  }
}

// 从 IValue 向量创建 ScriptCall 对象的唯一指针
std::unique_ptr<ScriptCall> ScriptCall::fromIValues(
    std::vector<at::IValue>& ivalues) {
  TORCH_INTERNAL_ASSERT(
      ivalues.size() > 1,
      "At least 2 IValues are required to build a ScriptCall.");

  // 向量中最后一个元素始终是限定名称，用于内置操作符和 TorchScript 函数
  const std::string& qualifiedName = ivalues.back().toStringRef();

  if (qualifiedName.rfind(BUILTIN_OP_NAMESPACE_) == 0) {
    // 如果限定名称以内置操作的命名空间前缀开头，则将其视为内置操作符
    ivalues.pop_back();
    // 获取最后一个元素并转换为常量引用的字符串
    const std::string& str_schema = ivalues.back().toStringRef();
    // 使用字符串进行操作匹配，返回操作符对象
    auto op = matchOperator(str_schema);

    // 移除最后一个元素，即 str_schema
    ivalues.pop_back();
    // 创建并返回一个 ScriptCall 对象，包含操作符和剩余的 ivalues
    return std::make_unique<ScriptCall>(op, std::move(ivalues));
  } else {
    // 移除最后一个元素，假定是布尔值用于异步执行
    ivalues.pop_back();
    bool isAsyncExecution = ivalues.back().toBool();
    // 再次移除最后一个元素，此时为异步执行标志
    ivalues.pop_back();
    // 创建并返回一个 ScriptCall 对象，包含限定名称、ivalues 和异步执行标志
    return std::make_unique<ScriptCall>(
        c10::QualifiedName(qualifiedName),
        std::move(ivalues),
        isAsyncExecution);
  }
}

c10::intrusive_ptr<Message> ScriptCall::toMessageImpl() && {
  // 创建一个空的 IValue 列表
  std::vector<IValue> ivalues;
  // 将当前对象的数据转换为 IValue 列表
  toIValues(ivalues);

  // 创建一个空的 Tensor 列表
  std::vector<torch::Tensor> tensor_table;
  // 使用 JIT 序列化 ivalues 到 payload，并将涉及到的 Tensor 放入 tensor_table
  auto payload = jit::pickle(
      c10::ivalue::Tuple::create(std::move(ivalues)), &tensor_table);

  // 创建一个 Message 对象，包含序列化后的 payload 和 tensor_table，指定消息类型为 SCRIPT_CALL
  return c10::make_intrusive<Message>(
      std::move(payload), std::move(tensor_table), MessageType::SCRIPT_CALL);
}

std::unique_ptr<ScriptCall> ScriptCall::fromMessage(const Message& message) {
  // 从消息中获取 payload 的指针和大小
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  // 使用 JIT 反序列化 payload，使用当前的 RpcAgent 的类型解析器和消息中的 tensors
  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());

  // 从反序列化后的 value 中获取元组的元素列表，并根据这些元素创建一个 ScriptCall 对象
  auto values = value.toTupleRef().elements().vec();
  return fromIValues(values);
}

std::shared_ptr<Operator> ScriptCall::matchOperator(
    const std::string& str_schema) {
  // TODO: 这是一个临时解决方案。我们应该传递足够的信息以便确定匹配到一个操作符。

  // 从 schema 解析出符号
  auto schema = torch::jit::parseSchema(str_schema);
  auto symbol = at::Symbol::fromQualString(schema.name());

  // 遍历所有符号对应的操作符，找到与 str_schema 匹配的操作符并返回
  for (auto op : torch::jit::getAllOperatorsFor(symbol)) {
    if (toString(op->schema()) == str_schema) {
      return op;
    }
  }

  // 如果找不到匹配的操作符，抛出错误信息
  TORCH_CHECK(false, "Cannot find matching operator for schema ", str_schema);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```