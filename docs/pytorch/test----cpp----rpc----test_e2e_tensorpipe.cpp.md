# `.\pytorch\test\cpp\rpc\test_e2e_tensorpipe.cpp`

```
#ifdef USE_TENSORPIPE
// 如果定义了 USE_TENSORPIPE 宏，则编译以下代码

class TestE2ETensorPipe : public TestE2EBase {
 protected:
  // 重写基类方法，用于构建 RPC 代理
  void buildRpcAgent() override {
    // 创建 Gloo 进程组选项对象
    auto options = c10d::ProcessGroupGloo::Options::create();
    // 将服务器地址设为 Gloo 设备
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname(serverAddress));
    // 设置 RPC 超时时间为 30 秒
    float rpcTimeout = 30;

    // 创建 TensorPipe 后端选项对象
    TensorPipeRpcBackendOptions opts(
        /*numWorkerThreads=*/std::max(16U, std::thread::hardware_concurrency()),
        /*transports=*/nullopt,
        /*channels=*/nullopt,
        /*rpc_timeout=*/rpcTimeout,
        /*init_method=*/"unused");

    // 使用选项创建 TensorPipeAgent 实例
    rpcAgent = std::make_shared<TensorPipeAgent>(
        store,
        "worker",
        0,
        numWorkers,
        opts,
        std::unordered_map<std::string, DeviceMap>{},
        std::vector<c10::Device>{},
        std::make_unique<RequestCallbackNoPython>());
  }
};

// 在 C++ 中进行端到端训练循环的测试，以便能够在此测试上运行 LSAN 以捕获内存泄漏。
// 使用 Python 多进程启用 LSAN 曾经存在挑战，目前尚无理想的解决方案。
TEST_F(TestE2ETensorPipe, TestTrainingLoop) {
  // 运行训练循环测试
  runTrainingLoop();
  
  // 确保清理 TensorPipe 内部状态
  auto tensorpipeAgent = std::static_pointer_cast<TensorPipeAgent>(rpcAgent);

  // 关闭 RPC 代理以清理所有 RPC
  tensorpipeAgent->join();
  tensorpipeAgent->shutdown();
  
  // 断言以下条件为真，以确保清理正常进行
  ASSERT_EQ(0, tensorpipeAgent->numPendingResponses());
  ASSERT_EQ(0, tensorpipeAgent->timeoutMapSize());
  ASSERT_EQ(0, tensorpipeAgent->messageIdToTimeoutMapSize());
}

#endif
// 结束 USE_TENSORPIPE 宏的条件编译
```