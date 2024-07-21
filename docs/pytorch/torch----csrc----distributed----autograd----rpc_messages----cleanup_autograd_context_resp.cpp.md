# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\cleanup_autograd_context_resp.cpp`

```
// 包含头文件 <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>

namespace torch {
namespace distributed {
namespace autograd {

// 定义 CleanupAutogradContextResp 类的成员函数 toMessageImpl()，&& 表示右值引用
c10::intrusive_ptr<rpc::Message> CleanupAutogradContextResp::
    toMessageImpl() && {
  
  // 创建存储 torch::Tensor 的向量 tensors
  std::vector<torch::Tensor> tensors;
  
  // 创建存储 char 类型数据的向量 payload
  std::vector<char> payload;
  
  // 使用 c10::make_intrusive 创建并返回 rpc::Message 智能指针，
  // 参数为 payload 和 tensors，并指定消息类型为 CLEANUP_AUTOGRAD_CONTEXT_RESP
  return c10::make_intrusive<rpc::Message>(
      std::move(payload),
      std::move(tensors),
      rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP);
}

// 定义 CleanupAutogradContextResp 类的静态成员函数 fromMessage，
// 从 rpc::Message 参数中构造并返回 std::unique_ptr<CleanupAutogradContextResp>
std::unique_ptr<CleanupAutogradContextResp> CleanupAutogradContextResp::
    fromMessage(const rpc::Message& message /* unused */) {
  
  // 返回空的 std::unique_ptr<CleanupAutogradContextResp>
  return std::unique_ptr<CleanupAutogradContextResp>();
}

} // namespace autograd
} // namespace distributed
} // namespace torch
```