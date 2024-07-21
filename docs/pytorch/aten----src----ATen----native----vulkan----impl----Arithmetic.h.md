# `.\pytorch\aten\src\ATen\native\vulkan\impl\Arithmetic.h`

```
#pragma once
// 预处理指令，确保此头文件仅被编译一次

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName
// 忽略 CLANGTIDY 静态分析工具的 facebook-hte-BadMemberName 错误

#include <ATen/native/vulkan/api/api.h>
// 包含 Vulkan API 头文件，用于与 Vulkan 相关的操作

namespace at {
namespace native {
namespace vulkan {
namespace arithmetic {

enum class OpType : uint32_t {
  ADD,          // 加法运算
  SUB,          // 减法运算
  MUL,          // 乘法运算
  DIV,          // 除法运算
  FLOOR_DIV,    // 向下取整除法运算
  POW,          // 指数运算
};

// 获取指定操作类型的着色器信息
api::ShaderInfo get_shader(const OpType type);

// 记录操作的函数，使用 Vulkan 加速计算
void record_op(
    api::Context* const context,    // Vulkan 计算环境上下文指针
    const api::ShaderInfo& compute_shader,  // 着色器信息，描述计算的具体操作
    vTensor& v_in1,     // Vulkan 张量输入参数1
    vTensor& v_in2,     // Vulkan 张量输入参数2
    vTensor& v_dst,     // Vulkan 张量输出参数
    const float alpha   // 操作的额外参数，例如缩放因子
);

} // namespace arithmetic
} // namespace vulkan
} // namespace native
} // namespace at
```