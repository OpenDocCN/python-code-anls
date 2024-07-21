# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8rmax\neon.c`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录中的LICENSE文件中以BSD风格许可证授权。
 */

#include <assert.h>     // 引入断言库，用于条件检查

#include <arm_neon.h>   // 引入ARM NEON指令集加速库

#include <qnnpack/u8rmax.h>   // 引入特定功能的头文件

// 定义一个函数，使用ARM NEON指令集加速计算输入数组中的最大值
uint8_t pytorch_u8rmax_ukernel__neon(size_t n, const uint8_t* x) {
  assert(n != 0);   // 断言：输入数组长度不为0

  // 如果输入数组长度大于等于16，则使用SIMD加速计算最大值
  if PYTORCH_QNNP_LIKELY(n >= 16) {
    uint8x16_t vmax = vmovq_n_u8(0);    // 初始化一个16个元素的无符号8位整数向量，并全部置0
    do {
      const uint8x16_t vx = vld1q_u8(x);   // 加载地址x处的16个8位整数到向量vx
      x += 16;    // 地址x增加16，移动到下一个16字节的数据
      vmax = vmaxq_u8(vmax, vx);   // 求取vmax和vx中每对元素的最大值，并保存到vmax中
      n -= 16;    // 剩余处理元素数减16
    } while (n >= 16);   // 如果剩余处理元素数仍大于等于16，继续循环处理

    // 处理剩余不足16个元素的情况
    if (n != 0) {
      const size_t x_increment = n - 16;   // 计算偏移量，用于调整地址x
      x = (const uint8_t*)((uintptr_t)x + x_increment);   // 调整地址x，使其指向正确的位置
      const uint8x16_t vx = vld1q_u8(x);   // 加载地址x处的不足16个8位整数到向量vx
      vmax = vmaxq_u8(vmax, vx);   // 求取vmax和vx中每对元素的最大值，并保存到vmax中
    }

    // 通过SIMD指令求取vmax中的最大值
    uint8x8_t vmax8 = vmax_u8(vget_low_u8(vmax), vget_high_u8(vmax));   // 取vmax的低8个元素和高8个元素，分别求出各自的最大值
    const uint8x8_t vmax4 = vpmax_u8(vmax8, vmax8);   // 求取vmax8中的最大值，水平求和
    const uint8x8_t vmax2 = vpmax_u8(vmax4, vmax4);   // 再次求取vmax4中的最大值，水平求和
    const uint8x8_t vmax1 = vpmax_u8(vmax2, vmax2);   // 最后求取vmax2中的最大值，水平求和
    return vget_lane_u8(vmax1, 0);   // 返回vmax1中的第0个元素作为最终结果
  }
  // 如果输入数组长度小于16，则使用标量方式计算最大值
  else {
    uint8x8_t vmax = vmov_n_u8(0);   // 初始化一个8位整数向量，并全部置0
    do {
      const uint8x8_t vx = vld1_dup_u8(x);   // 加载地址x处的一个8位整数到向量vx，并复制到所有元素中
      x += 1;   // 地址x增加1，移动到下一个数据
      vmax = vmax_u8(vmax, vx);   // 求取vmax和vx中每对元素的最大值，并保存到vmax中
    } while (--n != 0);   // 循环处理直到所有元素处理完毕
    return vget_lane_u8(vmax, 0);   // 返回vmax中的第0个元素作为最终结果
  }
}
```