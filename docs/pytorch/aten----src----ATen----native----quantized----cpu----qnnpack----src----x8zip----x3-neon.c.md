# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8zip\x3-neon.c`

```
/*
 * 使用 ARM NEON 指令集进行优化的函数，用于将输入数据按照 x, y, z 三个通道进行打包。
 * 如果输入数据量大于等于 8 个字节，则使用向量化处理；否则，使用标量处理。
 */
void pytorch_qnnp_x8zip_x3__neon(size_t n, const void* input, void* output) {
  // 将输入数据的起始地址分别赋值给指针 x, y, z
  const uint8_t* x = input;
  const uint8_t* y = x + n;   // 指向 y 数据的起始地址
  const uint8_t* z = y + n;   // 指向 z 数据的起始地址
  uint8_t* o = output;        // 输出数据的起始地址

  // 如果数据量大于等于 8 个字节，则执行向量化处理
  if (n >= 8) {
    do {
      // 使用 NEON 加载 x, y, z 数据到一个 8x8x3 的数据结构中
      uint8x8x3_t vxyz;
      vxyz.val[0] = vld1_u8(x);  // 加载 x 数据
      x += 8;
      vxyz.val[1] = vld1_u8(y);  // 加载 y 数据
      y += 8;
      vxyz.val[2] = vld1_u8(z);  // 加载 z 数据
      z += 8;
      // 使用 NEON 存储打包好的数据到输出地址 o，并递增 o 指针
      vst3_u8(o, vxyz);
      o += 24;  // 每次处理 8 个字节，输出 24 个字节
      n -= 8;   // 减去已处理的数据量
    } while (n >= 8);

    // 处理剩余不足 8 个字节的情况
    if (n != 0) {
      const size_t address_increment = n - 8;
      uint8x8x3_t vxyz;
      // 加载剩余数据
      vxyz.val[0] = vld1_u8(x + address_increment);
      vxyz.val[1] = vld1_u8(y + address_increment);
      vxyz.val[2] = vld1_u8(z + address_increment);
      // 使用 NEON 存储剩余数据到输出地址，并根据地址增量调整存储位置
      vst3_u8((uint8_t*)((uintptr_t)o + address_increment * 3), vxyz);
    }
  } else {
    // 如果数据量小于 8 个字节，则使用标量处理
    do {
      // 逐个处理 x, y, z 数据，并存储到输出地址 o
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      const uint8_t vz = *z++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o += 3;  // 递增输出地址
    } while (--n != 0);  // 继续直到处理完所有数据
  }
}
```