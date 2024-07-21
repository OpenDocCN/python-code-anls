# `.\pytorch\aten\src\ATen\test\mps_test_allocator.cpp`

```
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/mps/MPSAllocatorInterface.h>

// 定义命名空间 replay
namespace replay {
    // 声明全局变量 callback_action，用于存储回调函数
    std::function<void()> callback_action;

    // 定义类 ReplayBufferCleaner，并实现 MPS 分配器回调接口
    class ReplayBufferCleaner : virtual public at::mps::IMpsAllocatorCallback {
    public:
        // 实现接口函数，处理 MPS 分配器的回调事件
        void executeMPSAllocatorCallback(void* ptr, EventType event) override {
            // 如果事件为 ALLOCATION_FAILED
            if (event == EventType::ALLOCATION_FAILED) {
                // 调用全局回调函数 callback_action
                callback_action();
            }
        }
    };
}

// 在 at::mps 命名空间中注册 MPS 分配器回调，关联到 replay 命名空间下的 ReplayBufferCleaner 类
namespace at::mps {
    REGISTER_MPS_ALLOCATOR_CALLBACK("ReplayBufferCleaner", replay::ReplayBufferCleaner);
}

// 定义测试用例 MPSAllocator，测试 MPS 分配器的回调功能
TEST(MPSAllocator, MPSAllocatorCallbacks) {
    // 断言：如果 MPS 可用，则测试通过
    ASSERT_TRUE(torch::mps::is_available());

    // 定义回放缓冲区的 Tensor 向量
    std::vector<torch::Tensor> replay_buffer;
    
    // 设置 replay::callback_action 为 lambda 表达式，处理回放缓冲区的操作
    replay::callback_action = [&]() {
        // 如果回放缓冲区非空
        if (!replay_buffer.empty()) {
            // 移除回放缓冲区的前 1/10 部分
            replay_buffer.erase(replay_buffer.begin(), replay_buffer.begin() + (replay_buffer.size()/10));
        }
    };
    
    // 定义最大迭代次数
    size_t max_iter = 100000;
    
    // 迭代 max_iter 次
    for (size_t i = 0; i < max_iter; i++) {
        // 创建一个在 MPS 设备上的随机 Tensor
        torch::Tensor new_value = torch::randn({10000, 10000}, at::device(at::kMPS));
        
        // 如果回放缓冲区大小不等于当前迭代次数 i，则跳出循环
        if (replay_buffer.size() != i) {
            break;
        }
        
        // 将新创建的 Tensor 添加到回放缓冲区
        replay_buffer.push_back(new_value);
    }
    
    // 显式调用 synchronize()，等待所有 MPS 流中的 Metal completionHandlers 完成。
    // 注意，MPSAllocator 隐式地会执行这一步，但我们为测试目的显式调用它。
    torch::mps::synchronize();
    
    // 断言：回放缓冲区的大小应小于最大迭代次数 max_iter
    ASSERT_TRUE(replay_buffer.size() < max_iter);
}
```