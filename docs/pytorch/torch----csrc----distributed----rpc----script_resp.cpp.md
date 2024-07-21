# `.\pytorch\torch\csrc\distributed\rpc\script_resp.cpp`

```
# 引入 Torch 分布式 RPC 模块中的 ScriptResp 类的头文件
#include <torch/csrc/distributed/rpc/script_resp.h>

# 引入 Torch 分布式 RPC 模块中的 RPC Agent 类的头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>

# 引入 Torch JIT 序列化模块中的 pickle 函数的头文件
#include <torch/csrc/jit/serialization/pickle.h>

# 引入 Torch JIT 序列化模块中的 unpickler 函数的头文件
#include <torch/csrc/jit/serialization/unpickler.h>

# 定义 Torch 命名空间
namespace torch {
# 定义 Torch 分布式命名空间
namespace distributed {
# 定义 Torch 分布式 RPC 命名空间
namespace rpc {

# ScriptResp 类的构造函数，接受一个移动语义的 IValue 类对象作为参数
ScriptResp::ScriptResp(at::IValue&& value) : value_(value) {}

# 返回当前 ScriptResp 对象中存储的 IValue 类对象的引用
const at::IValue& ScriptResp::value() {
  return value_;
}

# 将当前对象转换为 Message 对象的实现，使用移动语义
c10::intrusive_ptr<Message> ScriptResp::toMessageImpl() && {
  # 创建一个存储 torch::Tensor 的向量
  std::vector<torch::Tensor> tensor_table;
  # 使用 jit::pickle 函数将当前对象的 value_ 序列化为 payload，同时收集 tensor 到 tensor_table 中
  auto payload = jit::pickle(value_, &tensor_table);
  # 创建并返回一个 Message 对象，携带 payload、tensor_table 和消息类型 SCRIPT_RET
  return c10::make_intrusive<Message>(
      std::move(payload), std::move(tensor_table), MessageType::SCRIPT_RET);
}

# 从 Message 对象中构建 ScriptResp 对象的静态方法
std::unique_ptr<ScriptResp> ScriptResp::fromMessage(const Message& message) {
  # 提取消息中的 payload 和大小信息
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  # 使用 jit::unpickle 函数反序列化 payload，并根据 message 中的 tensor 解析器获取类型解析器
  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  # 创建并返回一个包含反序列化结果的 unique_ptr<ScriptResp> 对象
  return std::make_unique<ScriptResp>(std::move(value));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```