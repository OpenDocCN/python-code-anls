# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8zip\x2-neon.c`

```py
#include <arm_neon.h>  // 包含 ARM NEON 指令集的头文件

#include <qnnpack/x8zip.h>  // 包含 x8zip 相关的头文件

void pytorch_qnnp_x8zip_x2__neon(size_t n, const void* input, void* output) {
  const uint8_t* x = input;  // 将输入指针强制转换为 uint8_t 类型的指针 x
  const uint8_t* y = x + n;  // 计算第二个输入指针 y，指向输入指针 x 后 n 个元素
  uint8_t* o = output;  // 设置输出指针 o 指向 output

  if (n >= 8) {  // 如果 n 大于等于 8，执行以下操作
    do {
      uint8x8x2_t vxy;  // 定义一个结构体，包含两个 8 位无符号整数向量
      vxy.val[0] = vld1_u8(x);  // 加载 x 指向的 8 个无符号 8 位整数，存储到 vxy 的第一个向量中
      x += 8;  // 更新 x 指针，指向下一个 8 个元素
      vxy.val[1] = vld1_u8(y);  // 加载 y 指向的 8 个无符号 8 位整数，存储到 vxy 的第二个向量中
      y += 8;  // 更新 y 指针，指向下一个 8 个元素
      vst2_u8(o, vxy);  // 将 vxy 结构体的数据存储到 o 指向的内存地址
      o += 16;  // 更新 o 指针，指向下一个 16 个元素的地址
      n -= 8;  // 更新剩余元素的数量
    } while (n >= 8);  // 当剩余元素数量大于等于 8 时循环执行上述操作

    if (n != 0) {  // 如果剩余元素数量不为 0，执行以下操作
      const size_t address_increment = n - 8;  // 计算地址增量为 n - 8
      uint8x8x2_t vxy;  // 定义一个结构体，包含两个 8 位无符号整数向量
      vxy.val[0] = vld1_u8((const uint8_t*)((uintptr_t)x + address_increment));  // 加载 x 指针加上地址增量后的数据到 vxy 的第一个向量中
      vxy.val[1] = vld1_u8((const uint8_t*)((uintptr_t)y + address_increment));  // 加载 y 指针加上地址增量后的数据到 vxy 的第二个向量中
      vst2_u8((uint8_t*)((uintptr_t)o + address_increment * 2), vxy);  // 将 vxy 结构体的数据存储到 o 指针加上地址增量乘以 2 后的地址
    }
  } else {  // 如果 n 小于 8，执行以下操作
    do {
      const uint8_t vx = *x++;  // 从 x 指针处读取一个 8 位无符号整数并递增指针 x
      const uint8_t vy = *y++;  // 从 y 指针处读取一个 8 位无符号整数并递增指针 y
      o[0] = vx;  // 将 vx 的值存储到 o 指向的第一个位置
      o[1] = vy;  // 将 vy 的值存储到 o 指向的第二个位置
      o += 2;  // 更新 o 指针，指向下一个 2 个元素的地址
    } while (--n != 0);  // 当剩余元素数量不为 0 时继续循环
  }
}
```