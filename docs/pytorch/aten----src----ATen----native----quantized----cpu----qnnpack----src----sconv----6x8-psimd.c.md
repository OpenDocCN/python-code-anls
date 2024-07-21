# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\sconv\6x8-psimd.c`

```
/*
 * 该函数实现了一个特定形状的卷积操作（6x8），利用PSIMD库进行加速计算。
 * 它将卷积核w与输入数据a的子集进行相乘累加，结果保存在输出数组c中。
 */

void pytorch_sconv_ukernel_6x8__psimd(
    size_t mr,  // 输入数据a的行数（通常为6）
    size_t nr,  // 输入数据a的列数（通常为8）
    size_t kc,  // 卷积核w的列数
    size_t ks,  // 卷积核w的行数
    const float** restrict a,  // 输入数据a的指针数组
    const float* restrict w,   // 卷积核w的指针
    float* restrict c,         // 输出结果数组c的指针
    size_t c_stride,           // 输出结果数组c的行步长
    const struct pytorch_qnnp_fp32_clamping_params
        clamping_params[restrict static 1]) {  // 限幅参数结构体数组
  psimd_f32 vacc0x0123 = psimd_load_f32(w);  // 从卷积核w加载四个单精度浮点数到向量寄存器中
  w += 4;  // 更新卷积核w的指针位置，移动到下一个四个单精度浮点数的位置
  psimd_f32 vacc0x4567 = psimd_load_f32(w);  // 继续加载卷积核w的下一组四个单精度浮点数到向量寄存器中
  w += 4;  // 更新卷积核w的指针位置

  // 复制加载的向量寄存器内容，用于后续计算的中间结果累加
  psimd_f32 vacc1x0123 = vacc0x0123;
  psimd_f32 vacc1x4567 = vacc0x4567;
  psimd_f32 vacc2x0123 = vacc0x0123;
  psimd_f32 vacc2x4567 = vacc0x4567;
  psimd_f32 vacc3x0123 = vacc0x0123;
  psimd_f32 vacc3x4567 = vacc0x4567;
  psimd_f32 vacc4x0123 = vacc0x0123;
  psimd_f32 vacc4x4567 = vacc0x4567;
  psimd_f32 vacc5x0123 = vacc0x0123;
  psimd_f32 vacc5x4567 = vacc0x4567;

  do {
    const float* restrict a0 = *a++;  // 获取输入数据a的第一行，并更新指针
    const float* restrict a1 = *a++;  // 获取输入数据a的第二行，并更新指针
    const float* restrict a2 = *a++;  // 获取输入数据a的第三行，并更新指针
    const float* restrict a3 = *a++;  // 获取输入数据a的第四行，并更新指针
    const float* restrict a4 = *a++;  // 获取输入数据a的第五行，并更新指针
    const float* restrict a5 = *a++;  // 获取输入数据a的第六行，并更新指针

    size_t k = kc;
    do {
      const psimd_f32 va0 = psimd_splat_f32(*a0);  // 将a0指向的单精度浮点数复制到向量寄存器中
      a0 += 1;  // 更新a0指针，移动到下一个单精度浮点数位置
      const psimd_f32 va1 = psimd_splat_f32(*a1);  // 将a1指向的单精度浮点数复制到向量寄存器中
      a1 += 1;  // 更新a1指针，移动到下一个单精度浮点数位置
      const psimd_f32 va2 = psimd_splat_f32(*a2);  // 将a2指向的单精度浮点数复制到向量寄存器中
      a2 += 1;  // 更新a2指针，移动到下一个单精度浮点数位置
      const psimd_f32 va3 = psimd_splat_f32(*a3);  // 将a3指向的单精度浮点数复制到向量寄存器中
      a3 += 1;  // 更新a3指针，移动到下一个单精度浮点数位置
      const psimd_f32 va4 = psimd_splat_f32(*a4);  // 将a4指向的单精度浮点数复制到向量寄存器中
      a4 += 1;  // 更新a4指针，移动到下一个单精度浮点数位置
      const psimd_f32 va5 = psimd_splat_f32(*a5);  // 将a5指向的单精度浮点数复制到向量寄存器中
      a5 += 1;  // 更新a5指针，移动到下一个单精度浮点数位置

      const psimd_f32 vb0123 = psimd_load_f32(w);  // 加载卷积核w的四个单精度浮点数到向量寄存器中
      w += 4;  // 更新卷积核w的指针位置，移动到下一个四个单精度浮点数的位置
      const psimd_f32 vb4567 = psimd_load_f32(w);  // 继续加载卷积核w的下一组四个单精度浮点数到向量寄存器中
      w += 4;  // 更新卷积核w的指针位置

      // 使用向量寄存器中的值计算累加结果
      vacc0x0123 += vb0123 * va0;
      vacc0x4567 += vb4567 * va0;
      vacc1x0123 += vb0123 * va1;
      vacc1x4567 += vb4567 * va1;
      vacc2x0123 += vb0123 * va2;
      vacc2x4567 += vb4567 * va2;
      vacc3x0123 += vb0123 * va3;
      vacc3x4567 += vb4567 * va3;
      vacc4x0123 += vb0123 * va4;
      vacc4x4567 += vb4567 * va4;
      vacc5x0123 += vb0123 * va5;
      vacc5x4567 += vb4567 * va5;
  } while (--k != 0);
  } while (--ks != 0);

  // 从参数中获取最大值，创建 PSIMD 浮点数向量
  const psimd_f32 vmax = psimd_splat_f32(clamping_params->max);
  // 对多个向量进行最小值限制
  vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
  vacc0x4567 = psimd_min_f32(vacc0x4567, vmax);
  vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);
  vacc1x4567 = psimd_min_f32(vacc1x4567, vmax);
  vacc2x0123 = psimd_min_f32(vacc2x0123, vmax);
  vacc2x4567 = psimd_min_f32(vacc2x4567, vmax);
  vacc3x0123 = psimd_min_f32(vacc3x0123, vmax);
  vacc3x4567 = psimd_min_f32(vacc3x4567, vmax);
  vacc4x0123 = psimd_min_f32(vacc4x0123, vmax);
  vacc4x4567 = psimd_min_f32(vacc4x4567, vmax);
  vacc5x0123 = psimd_min_f32(vacc5x0123, vmax);
  vacc5x4567 = psimd_min_f32(vacc5x4567, vmax);

  // 从参数中获取最小值，创建 PSIMD 浮点数向量
  const psimd_f32 vmin = psimd_splat_f32(clamping_params->min);
  // 对多个向量进行最大值限制
  vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
  vacc0x4567 = psimd_max_f32(vacc0x4567, vmin);
  vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);
  vacc1x4567 = psimd_max_f32(vacc1x4567, vmin);
  vacc2x0123 = psimd_max_f32(vacc2x0123, vmin);
  vacc2x4567 = psimd_max_f32(vacc2x4567, vmin);
  vacc3x0123 = psimd_max_f32(vacc3x0123, vmin);
  vacc3x4567 = psimd_max_f32(vacc3x4567, vmin);
  vacc4x0123 = psimd_max_f32(vacc4x0123, vmin);
  vacc4x4567 = psimd_max_f32(vacc4x4567, vmin);
  vacc5x0123 = psimd_max_f32(vacc5x0123, vmin);
  vacc5x4567 = psimd_max_f32(vacc5x4567, vmin);

  // 设置多个指针，用于访问矩阵中的不同行
  float* c0 = c;
  float* c1 = (float*)((uintptr_t)c0 + c_stride);
  // 如果剩余行数小于 2，则第二行指针指向第一行
  if (mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*)((uintptr_t)c1 + c_stride);
  // 如果剩余行数小于等于 2，则第三行指针指向第二行或第一行
  if (mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*)((uintptr_t)c2 + c_stride);
  // 如果剩余行数小于 4，则第四行指针指向第三行或第二行
  if (mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*)((uintptr_t)c3 + c_stride);
  // 如果剩余行数小于等于 4，则第五行指针指向第四行或第三行
  if (mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*)((uintptr_t)c4 + c_stride);
  // 如果剩余行数不等于 6，则第六行指针指向第五行或第四行
  if (mr != 6) {
    c5 = c4;
  }
  // 如果列数为 8，则将计算结果存储到指定地址，更新指针位置
  if (nr == 8) {
    psimd_store_f32(c0, vacc0x0123);
    c0 += 4;
    psimd_store_f32(c1, vacc1x0123);
    c1 += 4;
    psimd_store_f32(c2, vacc2x0123);
    c2 += 4;
    psimd_store_f32(c3, vacc3x0123);
    c3 += 4;
    psimd_store_f32(c4, vacc4x0123);
    c4 += 4;
    psimd_store_f32(c5, vacc5x0123);
    c5 += 4;

    psimd_store_f32(c0, vacc0x4567);
    psimd_store_f32(c1, vacc1x4567);
    psimd_store_f32(c2, vacc2x4567);
    psimd_store_f32(c3, vacc3x4567);
    psimd_store_f32(c4, vacc4x4567);
    psimd_store_f32(c5, vacc5x4567);
  } else {
    // 如果列数大于等于 4，则存储部分计算结果到指定地址，更新指针位置及列数
    if (nr >= 4) {
      psimd_store_f32(c0, vacc0x0123);
      c0 += 4;
      psimd_store_f32(c1, vacc1x0123);
      c1 += 4;
      psimd_store_f32(c2, vacc2x0123);
      c2 += 4;
      psimd_store_f32(c3, vacc3x0123);
      c3 += 4;
      psimd_store_f32(c4, vacc4x0123);
      c4 += 4;
      psimd_store_f32(c5, vacc5x0123);
      c5 += 4;
      // 将未存储的计算结果复制到对应变量，更新列数
      vacc0x0123 = vacc0x4567;
      vacc1x0123 = vacc1x4567;
      vacc2x0123 = vacc2x4567;
      vacc3x0123 = vacc3x4567;
      vacc4x0123 = vacc4x4567;
      vacc5x0123 = vacc5x4567;
      nr -= 4;
    }
    // 如果剩余的寄存器数量大于等于2，则执行以下操作
    if (nr >= 2) {
      // 将vacc0x0123的前两个浮点数存储到c0指向的内存位置，然后移动c0指针两个位置
      psimd_store2_f32(c0, vacc0x0123);
      c0 += 2;
      // 将vacc1x0123的前两个浮点数存储到c1指向的内存位置，然后移动c1指针两个位置
      psimd_store2_f32(c1, vacc1x0123);
      c1 += 2;
      // 将vacc2x0123的前两个浮点数存储到c2指向的内存位置，然后移动c2指针两个位置
      psimd_store2_f32(c2, vacc2x0123);
      c2 += 2;
      // 将vacc3x0123的前两个浮点数存储到c3指向的内存位置，然后移动c3指针两个位置
      psimd_store2_f32(c3, vacc3x0123);
      c3 += 2;
      // 将vacc4x0123的前两个浮点数存储到c4指向的内存位置，然后移动c4指针两个位置
      psimd_store2_f32(c4, vacc4x0123);
      c4 += 2;
      // 将vacc5x0123的前两个浮点数存储到c5指向的内存位置，然后移动c5指针两个位置
      psimd_store2_f32(c5, vacc5x0123);
      c5 += 2;
      // 将vacc0x0123的高位浮点数复制到低位，为下次存储做准备
      vacc0x0123 = psimd_concat_hi_f32(vacc0x0123, vacc0x0123);
      // 同上，操作vacc1x0123到vacc5x0123
      vacc1x0123 = psimd_concat_hi_f32(vacc1x0123, vacc1x0123);
      vacc2x0123 = psimd_concat_hi_f32(vacc2x0123, vacc2x0123);
      vacc3x0123 = psimd_concat_hi_f32(vacc3x0123, vacc3x0123);
      vacc4x0123 = psimd_concat_hi_f32(vacc4x0123, vacc4x0123);
      vacc5x0123 = psimd_concat_hi_f32(vacc5x0123, vacc5x0123);
      // 减少寄存器数量计数器nr的值，因为已处理2个寄存器
      nr -= 2;
    }
    // 如果剩余寄存器数量不为0，则执行以下操作
    if (nr != 0) {
      // 将vacc0x0123的一个浮点数存储到c0指向的内存位置
      psimd_store1_f32(c0, vacc0x0123);
      // 同上，操作c1到c5
      psimd_store1_f32(c1, vacc1x0123);
      psimd_store1_f32(c2, vacc2x0123);
      psimd_store1_f32(c3, vacc3x0123);
      psimd_store1_f32(c4, vacc4x0123);
      psimd_store1_f32(c5, vacc5x0123);
    }
  }
}



# 这行代码表示一个代码块的结束，与之前的“{”对应，用于结束一个函数、循环或者条件语句的定义。
```