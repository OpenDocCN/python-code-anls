# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8zip\x4-neon.c`

```
/*
 * 使用 ARM NEON 指令集对输入数据进行处理，将四个连续的 uint8_t 数组打包成一个 uint8x8x4_t 结构。
 * 这个函数假设输入的数据长度 n 至少为 8。
 */
void pytorch_qnnp_x8zip_x4__neon(size_t n, const void* input, void* output) {
    // 将输入指针转换为 uint8_t* 类型，分别指向四个连续的数组 x, y, z, w
    const uint8_t* x = input;
    const uint8_t* y = x + n;
    const uint8_t* z = y + n;
    const uint8_t* w = z + n;
    // 输出指针，指向结果数组 o
    uint8_t* o = output;

    // 如果 n 大于等于 8，则进入以下处理逻辑
    if (n >= 8) {
        do {
            // 定义一个 uint8x8x4_t 类型的变量 vxyzw，用来存储打包后的数据
            uint8x8x4_t vxyzw;
            // 依次加载 x, y, z, w 指向的连续 8 个 uint8_t 数据到 vxyzw 中
            vxyzw.val[0] = vld1_u8(x);
            x += 8;
            vxyzw.val[1] = vld1_u8(y);
            y += 8;
            vxyzw.val[2] = vld1_u8(z);
            z += 8;
            vxyzw.val[3] = vld1_u8(w);
            w += 8;
            // 将 vxyzw 中的四组数据以紧凑形式存储到 o 指向的内存位置
            vst4_u8(o, vxyzw);
            o += 32; // o 指针向后移动 32 个字节，以处理下一组四个数组
            n -= 8; // 减少剩余处理的数据量
        } while (n >= 8); // 当剩余数据量大于等于 8 时循环处理

        // 处理剩余的数据，如果 n 不等于 0
        if (n != 0) {
            // 计算地址偏移量，用于处理最后不足 8 个元素的情况
            const size_t address_increment = n - 8;
            uint8x8x4_t vxyzw;
            // 加载剩余数据到 vxyzw
            vxyzw.val[0] = vld1_u8(x + address_increment);
            vxyzw.val[1] = vld1_u8(y + address_increment);
            vxyzw.val[2] = vld1_u8(z + address_increment);
            vxyzw.val[3] = vld1_u8(w + address_increment);
            // 将数据存储到输出数组的正确位置
            vst4_u8((uint8_t*)((uintptr_t)o + address_increment * 4), vxyzw);
        }
    } else {
        // 如果 n 小于 8，则采用简单循环处理每个元素
        do {
            // 依次将 x, y, z, w 指向的单个元素加载到 o 指向的数组中
            const uint8_t vx = *x++;
            const uint8_t vy = *y++;
            const uint8_t vz = *z++;
            const uint8_t vw = *w++;
            o[0] = vx;
            o[1] = vy;
            o[2] = vz;
            o[3] = vw;
            o += 4; // o 指针向后移动 4 个字节，以处理下一个元素
        } while (--n != 0); // 继续循环，直到处理完所有的数据
    }
}
```