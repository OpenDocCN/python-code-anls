# `.\pytorch\aten\src\ATen\test\cpu_profiling_allocator_test.cpp`

```py
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <c10/core/CPUAllocator.h> // 引入 C10 核心模块的 CPUAllocator 头文件
#include <c10/mobile/CPUProfilingAllocator.h> // 引入 C10 移动端 CPUProfilingAllocator 头文件
#include <ATen/ATen.h> // 引入 PyTorch ATen 库的头文件
#include <ATen/Context.h> // 引入 PyTorch ATen 上下文相关的头文件

// 定义函数 run_with_control_flow，接收多个张量和布尔值作为参数，执行包含控制流的运算并返回张量
at::Tensor run_with_control_flow(
    at::Tensor input,
    at::Tensor conv_weight,
    at::Tensor linear_weight,
    bool cond,
    std::vector<void*>& pointers,
    bool record = false,
    bool validate = false) {
  
  if (cond) {
    input = input * 2; // 如果 cond 为 true，则将 input 中的每个元素乘以 2
  }
  
  void* input_ptr = input.data_ptr(); // 获取 input 张量的数据指针
  
  auto conv_out = at::conv2d(input, conv_weight); // 使用 conv_weight 对 input 进行二维卷积运算
  void* conv_out_ptr = input.data_ptr(); // 获取 conv_out 张量的数据指针（应为 conv_out）
  
  auto conv_out_flat = conv_out.view({conv_out.size(0), -1}); // 将 conv_out 张量展平为二维张量
  auto output = at::linear(conv_out_flat, linear_weight); // 对展平后的 conv_out 应用线性变换
  
  if (record) {
    pointers.push_back(input_ptr); // 如果 record 为 true，则将 input_ptr 添加到 pointers 容器中
    pointers.push_back(conv_out_ptr); // 将 conv_out_ptr 添加到 pointers 容器中
  }
  
  if (validate) {
    TORCH_CHECK(input_ptr == pointers[0]); // 如果 validate 为 true，则检查 input_ptr 是否与 pointers 中存储的第一个指针相同
    TORCH_CHECK(conv_out_ptr == pointers[1]); // 检查 conv_out_ptr 是否与 pointers 中存储的第二个指针相同
  }
  
  return output; // 返回计算结果张量 output
}

// 定义 CPUAllocationPlanTest 测试类的 with_control_flow 测试用例
TEST(CPUAllocationPlanTest, with_control_flow) {
  at::Tensor a = at::rand({23, 16, 16, 16}); // 创建形状为 [23, 16, 16, 16] 的随机张量 a
  at::Tensor conv_weight = at::rand({16, 16, 3, 3}); // 创建形状为 [16, 16, 3, 3] 的随机张量 conv_weight
  // 输出形状为 [23, 16, 14, 14]
  // 展平后的形状为 [23, 3136]
  at::Tensor linear_weight = at::rand({32, 3136}); // 创建形状为 [32, 3136] 的随机张量 linear_weight
  at::Tensor output, ref_output; // 声明输出张量 output 和参考输出张量 ref_output
  std::vector<void*> pointers; // 声明用于存储指针的容器 pointers

  // 定义 valid_allocation_plan Lambda 函数，用于执行测试
  auto valid_allocation_plan = [&]() {
    c10::AllocationPlan plan; // 创建 AllocationPlan 对象 plan
    {
      c10::WithProfileAllocationsGuard profile_guard(&plan); // 使用 WithProfileAllocationsGuard 对象进行分配记录
      ref_output = run_with_control_flow(
          a, conv_weight, linear_weight, true, pointers); // 调用 run_with_control_flow 函数执行计算
    }
  };

  // NOLINTNEXTLINE 指令，忽略特定的静态代码分析规则
  ASSERT_NO_THROW(valid_allocation_plan()); // 断言不会抛出异常

  // 定义 validate_allocation_plan Lambda 函数，用于执行验证测试
  auto validate_allocation_plan = [&](bool record_mode, bool validation_mode) -> bool {
    c10::AllocationPlan plan; // 创建 AllocationPlan 对象 plan
    {
      c10::WithProfileAllocationsGuard profile_guard(&plan); // 使用 WithProfileAllocationsGuard 对象进行分配记录
      ref_output = run_with_control_flow(
          a, conv_weight, linear_weight, record_mode, pointers); // 调用 run_with_control_flow 函数执行计算
    }
    bool success{true}; // 定义 success 变量，并初始化为 true
    for (uint64_t i = 0; i < 10; ++i) { // 执行 10 次循环
      bool validation_success; // 定义 validation_success 变量
      {
        c10::WithValidateAllocationPlanGuard validation_guard(&plan, &validation_success); // 使用 WithValidateAllocationPlanGuard 进行验证
        output = run_with_control_flow(
            a, conv_weight, linear_weight, validation_mode, pointers); // 调用 run_with_control_flow 函数执行计算
      }
      success = success && validation_success; // 更新 success 变量
    }
    return success; // 返回最终的成功状态
  };

  ASSERT_TRUE(validate_allocation_plan(true, true)); // 断言验证模式为 true 的执行成功
  ASSERT_TRUE(validate_allocation_plan(false, false)); // 断言验证模式为 false 的执行成功

  #ifdef C10_MOBILE
  // 当 record_mode 与 validation_mode 不同时，DefaultMobileCPUAllocator 返回 false
  ASSERT_FALSE(validate_allocation_plan(false, true)); // 断言验证模式与记录模式不同时返回 false
  ASSERT_FALSE(validate_allocation_plan(true, false)); // 断言验证模式与记录模式不同时返回 false
  #else
  ASSERT_TRUE(validate_allocation_plan(false, true)); // 断言验证模式与记录模式相同时返回 true
  ASSERT_TRUE(validate_allocation_plan(true, false)); // 断言验证模式与记录模式相同时返回 true
  #endif
}
TEST(CPUAllocationPlanTest, with_profiling_alloc) {
  // 创建大小为 [23, 16, 16, 16] 的随机张量 a
  at::Tensor a = at::rand({23, 16, 16, 16});
  // 创建大小为 [16, 16, 3, 3] 的随机张量 conv_weight
  at::Tensor conv_weight = at::rand({16, 16, 3, 3});
  // output shape
  // 23, 16, 14, 14
  // 将输出形状展平为 23, 3136
  // 创建大小为 [32, 3136] 的随机张量 linear_weight
  at::Tensor linear_weight = at::rand({32, 3136});
  // 创建 output 和 ref_output 张量
  at::Tensor output, ref_output;
  // 创建空指针向量 pointers
  std::vector<void*> pointers;

  // 定义 lambda 函数 valid_allocation_plan，用于验证分配计划
  auto valid_allocation_plan = [&]() {
    // 创建 AllocationPlan 对象 plan
    c10::AllocationPlan plan;
    {
      // 创建 WithProfileAllocationsGuard 对象 profile_guard，用于分析分配情况
      c10::WithProfileAllocationsGuard profile_guard(&plan);
      // 运行带有控制流的函数 run_with_control_flow，并将结果存入 ref_output
      ref_output = run_with_control_flow(
          a, conv_weight, linear_weight, false, pointers);
    }
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 断言 valid_allocation_plan 可以正常执行
  ASSERT_NO_THROW(valid_allocation_plan());

  // 声明 validate_allocation_plan
  auto validate_allocation_plan =
}

int main(int argc, char* argv[]) {
  // 设置 CPU 分配器的优先级为 100，确保其他分配器不会替代它
  c10::SetCPUAllocator(c10::GetDefaultCPUAllocator(), /*priority*/ 100);

  #ifdef C10_MOBILE
  // 在移动设备上禁用 MKLDNN，因为它通过 raw_allocate 接口分配内存，
  // 要求上下文指针和原始指针相同，这在移动分配器上不成立
  at::globalContext().setUserEnabledMkldnn(false);
  #endif

  // 初始化 Google Test 框架
  ::testing::InitGoogleTest(&argc, argv);
  // 设置随机数种子为 42
  at::manual_seed(42);
  // 运行所有测试
  return RUN_ALL_TESTS();
}
```