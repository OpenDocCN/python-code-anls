# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\cleanup_autograd_context_resp.h`

```
// 声明一个空的 CleanupAutogradContextResp 类，继承自 rpc::RpcCommandBase 类，用于响应 CleanupAutogradContextReq 请求。
class CleanupAutogradContextResp : public rpc::RpcCommandBase {
 public:
  // 默认构造函数
  CleanupAutogradContextResp() = default;
  // 虚函数，返回移动语义下的 Message 对象指针，用于序列化
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  // 从 Message 对象构造 CleanupAutogradContextResp 对象的静态方法
  static std::unique_ptr<CleanupAutogradContextResp> fromMessage(
      const rpc::Message& message);
};
```