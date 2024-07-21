# `.\pytorch\test\edge\test_operator_registration.cpp`

```py
// 包含自定义的头文件 "kernel_runtime_context.h" 和 "operator_registry.h"
#include "kernel_runtime_context.h"
#include "operator_registry.h"

// 包含 Google 测试框架的头文件
#include <gtest/gtest.h>

// 定义命名空间 torch 下的 executor 命名空间
namespace torch {
namespace executor {

// 定义一个测试用例 OperatorRegistrationTest，用于测试 aten::add.out 操作符
TEST(OperatorRegistrationTest, Add) {
    // 创建 EValue 类型的数组 values，长度为 4
    EValue values[4];
    // values[0] 和 values[1] 初始化为形状为 {2, 3} 的全一张量
    values[0] = EValue(at::ones({2, 3}));
    values[1] = EValue(at::ones({2, 3}));
    // values[2] 初始化为整数值 1
    values[2] = EValue(int64_t(1));
    // values[3] 初始化为形状为 {2, 3} 的全零张量
    values[3] = EValue(at::zeros({2, 3}));
    // 断言系统中存在 aten::add.out 的核函数
    ASSERT_TRUE(hasKernelFn("aten::add.out"));
    // 获取并保存 aten::add.out 的核函数
    auto op = getKernelFn("aten::add.out");

    // 创建 EValue* 类型的数组 kernel_values，长度为 4
    EValue* kernel_values[4];
    // 将 values 中的每个元素的地址赋值给 kernel_values 对应位置的元素
    for (size_t i = 0; i < 4; i++) {
        kernel_values[i] = &values[i];
    }
    // 创建 KernelRuntimeContext 类型的对象 context
    KernelRuntimeContext context{};
    // 调用操作符 op 执行计算，传入 context 和 kernel_values
    op(context, kernel_values);
    // 创建预期结果张量 expected，初始化为形状 {2, 3} 的全一张量，并用数值 2 填充
    at::Tensor expected = at::ones({2, 3});
    expected = at::fill(expected, 2);
    // 断言预期结果张量与 kernel_values[3] 转换后的张量相等
    ASSERT_TRUE(expected.equal(kernel_values[3]->toTensor()));
}

// 定义一个测试用例 OperatorRegistrationTest，用于测试 custom::add_3.out 操作符
TEST(OperatorRegistrationTest, CustomAdd3) {
    // 创建 EValue 类型的数组 values，长度为 4
    EValue values[4];
    // values[0]、values[1] 和 values[2] 初始化为形状为 {2, 3} 的全一张量
    values[0] = EValue(at::ones({2, 3}));
    values[1] = EValue(at::ones({2, 3}));
    values[2] = EValue(at::ones({2, 3}));
    // values[3] 初始化为形状为 {2, 3} 的全零张量
    values[3] = EValue(at::zeros({2, 3}));
    // 断言系统中存在 custom::add_3.out 的核函数
    ASSERT_TRUE(hasKernelFn("custom::add_3.out"));
    // 获取并保存 custom::add_3.out 的核函数
    auto op = getKernelFn("custom::add_3.out");

    // 创建 EValue* 类型的数组 kernel_values，长度为 4
    EValue* kernel_values[4];
    // 将 values 中的每个元素的地址赋值给 kernel_values 对应位置的元素
    for (size_t i = 0; i < 4; i++) {
        kernel_values[i] = &values[i];
    }
    // 创建 KernelRuntimeContext 类型的对象 context
    KernelRuntimeContext context{};
    // 调用操作符 op 执行计算，传入 context 和 kernel_values
    op(context, kernel_values);
    // 创建预期结果张量 expected，初始化为形状 {2, 3} 的全一张量，并用数值 3 填充
    at::Tensor expected = at::ones({2, 3});
    expected = at::fill(expected, 3);
    // 断言预期结果张量与 kernel_values[3] 转换后的张量相等
    ASSERT_TRUE(expected.equal(kernel_values[3]->toTensor()));
}

} // namespace executor
} // namespace torch
```