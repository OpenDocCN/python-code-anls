# `.\pytorch\test\cpp\rpc\test_tensorpipe_serialization.cpp`

```
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <tensorpipe/common/cpu_buffer.h>
#include <tensorpipe/core/message.h>
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

TEST(TensorpipeSerialize, Base) {
  // Sender serializes

  // 创建一个包含1024个整数1的张量
  at::Tensor t1 = torch::ones({1024}, at::ScalarType::Int);
  // 创建一个包含1024个浮点数1.0的张量
  at::Tensor t2 = torch::ones({1024}, at::ScalarType::Float);
  // 将两个张量放入向量中
  std::vector<at::Tensor> tensors{t1, t2};
  // 创建一个包含字符 '1', '2', '3' 的载荷向量
  std::vector<char> payload = {'1', '2', '3'};
  // 为测试创建载荷向量的副本
  std::vector<char> payloadCopy = payload; // for testing
  // 定义消息类型为UNKNOWN
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;
  // 设置消息ID为100
  int64_t mId = 100;
  // 创建发送的RPC消息
  auto sendingRpcMessage =
      c10::make_intrusive<torch::distributed::rpc::Message>(
          std::move(payload), std::move(tensors), mtype);
  // 设置RPC消息的ID
  sendingRpcMessage->setId(mId);
  // 准备用于Tensorpipe的消息和缓冲区
  tensorpipe::Message sendingTpMessage;
  torch::distributed::rpc::TensorpipeWriteBuffers sendingTpBuffers;
  // 序列化RPC消息为Tensorpipe消息和缓冲区
  std::tie(sendingTpMessage, sendingTpBuffers) =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage), {}, {});

  // Mimic receiving message descriptor: recvingTpDescriptor is a copy of
  // sendingTpMessage except for the data pointers which are left null.
  // 模拟接收消息描述符：recvingTpDescriptor 是 sendingTpMessage 的一个副本，但数据指针为空。

  // 创建接收消息的描述符
  tensorpipe::Descriptor recvingTpDescriptor;
  // 复制元数据
  recvingTpDescriptor.metadata = sendingTpMessage.metadata;
  // 保留与发送消息相同数量的负载
  recvingTpDescriptor.payloads.reserve(sendingTpMessage.payloads.size());
  // 复制发送消息的负载到接收消息的描述符中
  for (auto& tpPayload : sendingTpMessage.payloads) {
    tensorpipe::Descriptor::Payload p;
    p.length = tpPayload.length;
    p.metadata = tpPayload.metadata;
    recvingTpDescriptor.payloads.push_back(std::move(p));
  }
  // 断言接收描述符负载数量与发送消息相同
  EXPECT_EQ(
      recvingTpDescriptor.payloads.size(), sendingTpMessage.payloads.size());
  // 保留与发送消息相同数量的张量
  recvingTpDescriptor.tensors.reserve(sendingTpMessage.tensors.size());
  // 复制发送消息的张量到接收消息的描述符中
  for (auto& tpTensor : sendingTpMessage.tensors) {
    tensorpipe::Descriptor::Tensor t;
    t.length = tpTensor.length;
    t.sourceDevice = tpTensor.buffer.device();
    t.targetDevice = tpTensor.targetDevice;
    t.metadata = tpTensor.metadata;
    recvingTpDescriptor.tensors.push_back(std::move(t));
  }
  // 断言接收描述符张量数量与发送消息相同
  EXPECT_EQ(
      recvingTpDescriptor.tensors.size(), sendingTpMessage.tensors.size());

  // Mimic readDescriptor() callback:
  // - Allocate buffers
  // - Fill pointers in tensorpipe message
  // 模拟 readDescriptor() 回调：
  // - 分配缓冲区
  // - 填充Tensorpipe消息中的指针

  // 分配接收消息描述符所需的缓冲区和Tensorpipe读缓冲区
  tensorpipe::Allocation recvingTpAllocation;
  torch::distributed::rpc::TensorpipeReadBuffers recvingTpBuffers;
  std::tie(recvingTpAllocation, recvingTpBuffers) =
      torch::distributed::rpc::tensorpipeAllocate(recvingTpDescriptor, {});

  // Mimic tensorpipe data transfer
  // 模拟Tensorpipe数据传输

  // 断言接收到的负载数量与发送消息中的负载数量相同
  EXPECT_EQ(
      recvingTpAllocation.payloads.size(), sendingTpMessage.payloads.size());
  // 遍历每一个负载并比较
  for (const auto i : c10::irange(recvingTpAllocation.payloads.size())) {
    tensorpipe::Message::Payload& srcPayload = sendingTpMessage.payloads[i];
    tensorpipe::Allocation::Payload& dstPayload =
        recvingTpAllocation.payloads[i];
    if (srcPayload.length) {
      // 如果 srcPayload 的长度不为零，则执行下面的拷贝操作
      // 空向量的 data() 可能返回 nullptr，使用长度来避免拷贝到 nullptr
      memcpy(dstPayload.data, srcPayload.data, srcPayload.length);
    }
  }
  EXPECT_EQ(
      recvingTpAllocation.tensors.size(), sendingTpMessage.tensors.size());
  // 遍历接收端分配的张量列表，进行数据拷贝操作
  for (const auto i : c10::irange(recvingTpAllocation.tensors.size())) {
    // 获取发送端消息中第 i 个张量的引用
    tensorpipe::Message::Tensor& srcTensor = sendingTpMessage.tensors[i];
    // 获取接收端分配的张量中第 i 个张量的引用
    tensorpipe::Allocation::Tensor& dstTensor = recvingTpAllocation.tensors[i];
    // 使用 memcpy 进行内存拷贝，从发送端张量到接收端张量
    memcpy(
        dstTensor.buffer.unwrap<tensorpipe::CpuBuffer>().ptr,
        srcTensor.buffer.unwrap<tensorpipe::CpuBuffer>().ptr,
        srcTensor.length);
  }

  // 模拟 read() 回调：
  // - 反序列化
  // 使用 tensorpipe 进行反序列化，得到接收到的 RPC 消息
  c10::intrusive_ptr<torch::distributed::rpc::Message> recvingRpcMessage =
      torch::distributed::rpc::tensorpipeDeserialize(
          std::move(recvingTpDescriptor), std::move(recvingTpBuffers));

  // 数据准备就绪
  // 检查各项接收到的消息属性是否符合预期
  EXPECT_EQ(mtype, recvingRpcMessage->type());
  EXPECT_EQ(payloadCopy, recvingRpcMessage->payload());
  EXPECT_EQ(mId, recvingRpcMessage->id());
  EXPECT_TRUE(torch::equal(t1, recvingRpcMessage->tensors()[0]));
  EXPECT_TRUE(torch::equal(t2, recvingRpcMessage->tensors()[1]));
TEST(TensorpipeSerialize, RecopySparseTensors) {
  // 定义常量 k1K 为 1024，表示一千
  constexpr size_t k1K = 1024;
  // 创建一个 k1K x k1K 的随机张量 main
  at::Tensor main = torch::randn({k1K, k1K});
  // 从 main 中选择第二行作为 tiny 张量
  at::Tensor tiny = main.select(0, 2); // Select a row in the middle
  // 断言 tiny 张量的元素个数为 k1K
  EXPECT_EQ(tiny.numel(), k1K);
  // 断言 tiny 张量存储的字节数除以元素大小等于 k1K x k1K
  EXPECT_EQ(tiny.storage().nbytes() / tiny.itemsize(), k1K * k1K);

  // 创建包含 main 和 tiny 两个张量的向量 tensors
  std::vector<at::Tensor> tensors{main, tiny};
  // 创建包含字符 '1', '2', '3' 的 payload 向量
  std::vector<char> payload = {'1', '2', '3'};
  // 定义消息类型 mtype 为 UNKNOWN
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;
  // 使用 payload 和 tensors 创建发送消息 sendingRpcMessage
  auto sendingRpcMessage =
      c10::make_intrusive<torch::distributed::rpc::Message>(
          std::move(payload), std::move(tensors), mtype);

  // 创建 tensorpipe 的消息 sendingTpMessage 和 TensorpipeWriteBuffers tpBuffers
  tensorpipe::Message sendingTpMessage;
  torch::distributed::rpc::TensorpipeWriteBuffers tpBuffers;
  // 调用 tensorpipeSerialize 函数将 sendingRpcMessage 序列化为 sendingTpMessage 和 tpBuffers
  std::tie(sendingTpMessage, tpBuffers) =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage), {}, {});

  // 断言 tpBuffers 中张量的数量为 2
  EXPECT_EQ(tpBuffers.tensors.size(), 2);
  // 断言 sendingTpMessage 中张量的数量为 2
  EXPECT_EQ(sendingTpMessage.tensors.size(), 2);
  // 断言 main 和 tpBuffers 中的第一个张量相等
  EXPECT_TRUE(torch::equal(main, tpBuffers.tensors[0]));
  // 断言 tiny 和 tpBuffers 中的第二个张量相等
  EXPECT_TRUE(torch::equal(tiny, tpBuffers.tensors[1]));
  // 测试克隆的存储空间
  // 断言 main 的存储空间数据指针与 sendingTpMessage 中第一个张量的 CpuBuffer 指针相同
  EXPECT_EQ(
      main.storage().data(),
      sendingTpMessage.tensors[0].buffer.unwrap<tensorpipe::CpuBuffer>().ptr);
  // 断言 tiny 的存储空间数据指针与 sendingTpMessage 中第二个张量的 CpuBuffer 指针不同
  EXPECT_NE(
      tiny.storage().data(),
      sendingTpMessage.tensors[1].buffer.unwrap<tensorpipe::CpuBuffer>().ptr);
  // 断言 sendingTpMessage 中第二个张量的长度等于 tiny 的元素大小乘以 k1K
  EXPECT_EQ(tiny.element_size() * k1K, sendingTpMessage.tensors[1].length);
}
// 定义名为 `TEST` 的测试用例，测试 `TensorpipeSerialize` 函数中无删除器的张量序列化
TEST(TensorpipeSerialize, NoDeleterTensors) {
  // 创建两个浮点数向量作为示例数据
  std::vector<float> blob1{.8, .2};
  std::vector<float> blob2{.7, .5, .9};

  // 从向量数据创建 PyTorch 张量 `t1` 和 `t2`
  at::Tensor t1 = torch::from_blob((float*)(blob1.data()), blob1.size());
  at::Tensor t2 = torch::from_blob((float*)(blob2.data()), blob2.size());

  // 将张量 `t1` 和 `t2` 存入向量 `tensors`
  std::vector<at::Tensor> tensors{t1, t2};

  // 创建包含 '1', '2', '3' 的负载数据向量 `payload`
  std::vector<char> payload = {'1', '2', '3'};

  // 创建枚举类型 `MessageType` 实例 `mtype`，值为 `UNKNOWN`
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;

  // 使用 `make_intrusive` 创建共享指针 `sendingRpcMessage`，
  // 包含负载数据 `payload`、张量 `tensors` 和消息类型 `mtype`
  auto sendingRpcMessage =
      c10::make_intrusive<torch::distributed::rpc::Message>(
          std::move(payload), std::move(tensors), mtype);

  // 创建 `tensorpipe::Message` 类型的 `sendingTpMessage` 实例
  tensorpipe::Message sendingTpMessage;

  // 创建 `torch::distributed::rpc::TensorpipeWriteBuffers` 类型的 `tpBuffers` 实例
  torch::distributed::rpc::TensorpipeWriteBuffers tpBuffers;

  // 调用 `tensorpipeSerialize` 函数，将 `sendingRpcMessage` 序列化为 `sendingTpMessage` 和 `tpBuffers`
  std::tie(sendingTpMessage, tpBuffers) =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage), {}, {});

  // 验证 `tpBuffers.copiedTensors` 的长度为 2
  EXPECT_EQ(tpBuffers.copiedTensors.size(), 2);

  // 验证 `sendingTpMessage.tensors` 的长度为 2
  EXPECT_EQ(sendingTpMessage.tensors.size(), 2);

  // 验证 `tpBuffers.copiedTensors[0]` 的大小与 `sendingTpMessage.tensors[0]` 的长度相等
  EXPECT_EQ(
      tpBuffers.copiedTensors[0].size(), sendingTpMessage.tensors[0].length);

  // 验证 `tpBuffers.copiedTensors[1]` 的大小与 `sendingTpMessage.tensors[1]` 的长度相等
  EXPECT_EQ(
      tpBuffers.copiedTensors[1].size(), sendingTpMessage.tensors[1].length);

  // 验证 `tpBuffers.copiedTensors[0]` 的数据指针与 `sendingTpMessage.tensors[0]` 的数据指针相等
  EXPECT_EQ(
      tpBuffers.copiedTensors[0].data(),
      sendingTpMessage.tensors[0].buffer.unwrap<tensorpipe::CpuBuffer>().ptr);

  // 验证 `tpBuffers.copiedTensors[1]` 的数据指针与 `sendingTpMessage.tensors[1]` 的数据指针相等
  EXPECT_EQ(
      tpBuffers.copiedTensors[1].data(),
      sendingTpMessage.tensors[1].buffer.unwrap<tensorpipe::CpuBuffer>().ptr);

  // 验证 `tpBuffers.copiedTensors[0]` 的数据与 `t1` 的存储数据相等
  EXPECT_TRUE(
      memcmp(
          tpBuffers.copiedTensors[0].data(),
          t1.storage().data(),
          sendingTpMessage.tensors[0].length) == 0);

  // 验证 `tpBuffers.copiedTensors[1]` 的数据与 `t2` 的存储数据相等
  EXPECT_TRUE(
      memcmp(
          tpBuffers.copiedTensors[1].data(),
          t2.storage().data(),
          sendingTpMessage.tensors[1].length) == 0);
}
```