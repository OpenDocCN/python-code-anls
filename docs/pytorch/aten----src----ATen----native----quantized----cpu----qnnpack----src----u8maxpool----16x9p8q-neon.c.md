# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8maxpool\16x9p8q-neon.c`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录下的LICENSE文件中使用BSD风格许可证授权。
 */

#include <assert.h>         // 包含断言相关的头文件

#include <arm_neon.h>       // 包含 ARM NEON 指令集的头文件

#include <qnnpack/u8maxpool.h>   // 引入QNNPACK中定义的无符号8位最大池化函数头文件

void pytorch_u8maxpool_ukernel_16x9p8q__neon(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_u8_clamping_params params[restrict static 1]) {
  assert(n != 0);   // 断言n不为0，确保输入数量大于0
  assert(ks != 0);  // 断言ks不为0，确保内核大小大于0
  assert(kc >= 16); // 断言kc至少为16，确保输入通道数不小于16

  // 使用params中的NEON输出最大值创建一个128位的无符号整数向量
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);
  // 使用params中的NEON输出最小值创建一个128位的无符号整数向量
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);
  
  // 进行循环，处理每个输入通道的数据
  do {
    uint8_t* o = output;  // 初始化输出指针

    // 处理每个通道的数据
    for (size_t k = kc; k > 0; k -= 16) {
      // 加载输入数据向量，每次加载128位（16字节）
      const uint8x16_t vi = vld1q_u8(*input);
      
      // 使用ARM NEON指令进行最大池化操作，计算每个向量元素与输出最大值的最小值
      const uint8x16_t vo = vminq_u8(vi, voutput_max);
      // 将结果存储到输出内存中
      vst1q_u8(o, vo);

      // 更新输入指针，移动到下一个输入向量
      input = (const uint8_t**)((uintptr_t)input + input_increment);
      // 更新输出指针，移动到下一个输出向量
      o += 16;
    }

    // 更新输出指针，移动到下一个输出元素
    output = (uint8_t*)((uintptr_t)o + output_increment);
  } while (--n != 0);  // 继续处理下一个样本，直到处理完所有样本
}
```