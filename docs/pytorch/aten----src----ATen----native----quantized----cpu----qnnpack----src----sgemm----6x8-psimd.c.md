# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\sgemm\6x8-psimd.c`

```py
/*
 * 引入 psimd.h 头文件，其中包含 SIMD 操作的函数和定义
 */
#include <psimd.h>

/*
 * 包含 QNNPACK 中的 sgemm 头文件，定义了矩阵乘法相关函数
 */
#include <qnnpack/sgemm.h>

/*
 * 定义了一个名为 pytorch_sgemm_ukernel_6x8__psimd 的函数，用于执行 6x8 的矩阵乘法运算，
 * 使用 PSIMD（Portable SIMD）指令集进行优化
 */
void pytorch_sgemm_ukernel_6x8__psimd(
    size_t mr,  // 行数 mr
    size_t nr,  // 列数 nr
    size_t k,   // 维度 k
    const float* restrict a,  // 输入矩阵 a，限制对指针的优化限制
    size_t a_stride,  // 矩阵 a 的行步长
    const float* restrict w,  // 权重矩阵 w，限制对指针的优化限制
    float* restrict c,        // 输出矩阵 c，限制对指针的优化限制
    size_t c_stride,          // 矩阵 c 的行步长
    const struct pytorch_qnnp_fp32_clamping_params
        clamping_params[restrict static 1]) {  // 浮点数限制参数
  /*
   * 从 w 加载 psimd_f32 类型的向量，存储在 vacc0x0123 中，并递增 w 指针
   */
  psimd_f32 vacc0x0123 = psimd_load_f32(w);
  w += 4;
  /*
   * 从 w 加载 psimd_f32 类型的向量，存储在 vacc0x4567 中，并递增 w 指针
   */
  psimd_f32 vacc0x4567 = psimd_load_f32(w);
  w += 4;
  /*
   * 复制 vacc0x0123 到 vacc1x0123
   */
  psimd_f32 vacc1x0123 = vacc0x0123;
  /*
   * 复制 vacc0x4567 到 vacc1x4567
   */
  psimd_f32 vacc1x4567 = vacc0x4567;
  /*
   * 复制 vacc0x0123 到 vacc2x0123
   */
  psimd_f32 vacc2x0123 = vacc0x0123;
  /*
   * 复制 vacc0x4567 到 vacc2x4567
   */
  psimd_f32 vacc2x4567 = vacc0x4567;
  /*
   * 复制 vacc0x0123 到 vacc3x0123
   */
  psimd_f32 vacc3x0123 = vacc0x0123;
  /*
   * 复制 vacc0x4567 到 vacc3x4567
   */
  psimd_f32 vacc3x4567 = vacc0x4567;
  /*
   * 复制 vacc0x0123 到 vacc4x0123
   */
  psimd_f32 vacc4x0123 = vacc0x0123;
  /*
   * 复制 vacc0x4567 到 vacc4x4567
   */
  psimd_f32 vacc4x4567 = vacc0x4567;
  /*
   * 复制 vacc0x0123 到 vacc5x0123
   */
  psimd_f32 vacc5x0123 = vacc0x0123;
  /*
   * 复制 vacc0x4567 到 vacc5x4567
   */
  psimd_f32 vacc5x4567 = vacc0x4567;

  /*
   * 初始化指针 a0 指向 a
   */
  const float* a0 = a;
  /*
   * 初始化指针 a1 指向 a0 + a_stride（a 的下一行）
   */
  const float* a1 = (const float*)((uintptr_t)a0 + a_stride);
  /*
   * 如果 mr 小于 2，则将 a1 指向 a0，保持指针在有效范围内
   */
  if (mr < 2) {
    a1 = a0;
  }
  /*
   * 初始化指针 a2 指向 a1 + a_stride（a 的第三行）
   */
  const float* a2 = (const float*)((uintptr_t)a1 + a_stride);
  /*
   * 如果 mr 小于等于 2，则将 a2 指向 a1，保持指针在有效范围内
   */
  if (mr <= 2) {
    a2 = a1;
  }
  /*
   * 初始化指针 a3 指向 a2 + a_stride（a 的第四行）
   */
  const float* a3 = (const float*)((uintptr_t)a2 + a_stride);
  /*
   * 如果 mr 小于 4，则将 a3 指向 a2，保持指针在有效范围内
   */
  if (mr < 4) {
    a3 = a2;
  }
  /*
   * 初始化指针 a4 指向 a3 + a_stride（a 的第五行）
   */
  const float* a4 = (const float*)((uintptr_t)a3 + a_stride);
  /*
   * 如果 mr 小于等于 4，则将 a4 指向 a3，保持指针在有效范围内
   */
  if (mr <= 4) {
    a4 = a3;
  }
  /*
   * 初始化指针 a5 指向 a4 + a_stride（a 的第六行）
   */
  const float* a5 = (const float*)((uintptr_t)a4 + a_stride);
  /*
   * 如果 mr 不等于 6，则将 a5 指向 a4，保持指针在有效范围内
   */
  if (mr != 6) {
    a5 = a4;
  }

  /*
   * 执行矩阵乘法操作，利用 PSIMD 指令集进行优化，重复 mr 次
   */
  do {
    /*
     * 从 a0 加载 psimd_f32 类型的值，并复制给 va0，然后递增 a0 指针
     */
    const psimd_f32 va0 = psimd_splat_f32(*a0);
    a0 += 1;
    /*
     * 从 a1 加载 psimd_f32 类型的值，并复制给 va1，然后递增 a1 指针
     */
    const psimd_f32 va1 = psimd_splat_f32(*a1);
    a1 += 1;
    /*
     * 从 a2 加载 psimd_f32 类型的值，并复制给 va2，然后递增 a2 指针
     */
    const psimd_f32 va2 = psimd_splat_f32(*a2);
    a2 += 1;
    /*
     * 从 a3 加载 psimd_f32 类型的值，并复制给 va3，然后递增 a3 指针
     */
    const psimd_f32 va3 = psimd_splat_f32(*a3);
    a3 += 1;
    /*
     * 从 a4 加载 psimd_f32 类型的值，并复制给 va4，然后递增 a4 指针
     */
    const psimd_f32 va4 = psimd_splat_f32(*a4);
    a4 += 1;
    /*
     * 从 a5 加载 psimd_f32 类型的值，并复制给 va5，然后递增 a5 指针
     */
    const psimd_f32 va5 = psimd_splat_f32(*a5);
    a5 += 1;

    /*
     * 从 w 加载 psimd_f32 类型的向量，并复制给 vb0123，然后递增 w 指针
     */
    const psimd_f32 vb0123 = psimd_load_f32(w);
    w += 4;
    /*
     * 从 w 加载 psimd_f32 类型的向量，并复制给 vb4567，然后递增 w 指针
     */
    const psimd_f32 vb4567 = psimd_load_f32(w);
    w += 4;

    /*
     * 计算矩阵乘法的部分结果，利用 PSIMD 指令集进行优化
     */
    vacc0x0123 += vb0123 * va0;
    vacc0x4567 += vb4567 * va0;
    vacc1
    // 计算向量累加的最大值上限
    vacc5x4567 += vb4567 * va5;
  } while (--k != 0);

  // 使用给定的上限值对累加结果进行截断
  const psimd_f32 vmax = psimd_splat_f32(clamping_params->max);
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

  // 使用给定的下限值对累加结果进行截断
  const psimd_f32 vmin = psimd_splat_f32(clamping_params->min);
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

  // 根据计算矩阵的行数和结果矩阵的列数选择正确的存储指针
  float* c0 = c;
  float* c1 = (float*)((uintptr_t)c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*)((uintptr_t)c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*)((uintptr_t)c2 + c_stride);
  if (mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*)((uintptr_t)c3 + c_stride);
  if (mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*)((uintptr_t)c4 + c_stride);
  if (mr != 6) {
    c5 = c4;
  }

  // 根据结果矩阵的列数选择正确的存储方式
  // 如果列数为8，则直接存储
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
    // 如果列数不足8，则根据列数存储对应的结果
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
      // 更新列数和累加结果
      vacc0x0123 = vacc0x4567;
      vacc1x0123 = vacc1x4567;
      vacc2x0123 = vacc2x4567;
      vacc3x0123 = vacc3x4567;
      vacc4x0123 = vacc4x4567;
      vacc5x0123 = vacc5x4567;
      nr -= 4;
    }
    # 如果剩余可处理的向量数量大于等于2，则执行以下操作
    if (nr >= 2) {
      # 将vacc0x0123的前两个浮点数存储到地址c0，并更新c0指针
      psimd_store2_f32(c0, vacc0x0123);
      c0 += 2;
      # 将vacc1x0123的前两个浮点数存储到地址c1，并更新c1指针
      psimd_store2_f32(c1, vacc1x0123);
      c1 += 2;
      # 将vacc2x0123的前两个浮点数存储到地址c2，并更新c2指针
      psimd_store2_f32(c2, vacc2x0123);
      c2 += 2;
      # 将vacc3x0123的前两个浮点数存储到地址c3，并更新c3指针
      psimd_store2_f32(c3, vacc3x0123);
      c3 += 2;
      # 将vacc4x0123的前两个浮点数存储到地址c4，并更新c4指针
      psimd_store2_f32(c4, vacc4x0123);
      c4 += 2;
      # 将vacc5x0123的前两个浮点数存储到地址c5，并更新c5指针
      psimd_store2_f32(c5, vacc5x0123);
      c5 += 2;
      # 将vacc0x0123的高位浮点数复制到其低位，即复制前两个浮点数到后两个位置
      vacc0x0123 = psimd_concat_hi_f32(vacc0x0123, vacc0x0123);
      # 同上，对vacc1x0123、vacc2x0123、vacc3x0123、vacc4x0123、vacc5x0123执行相同操作
      vacc1x0123 = psimd_concat_hi_f32(vacc1x0123, vacc1x0123);
      vacc2x0123 = psimd_concat_hi_f32(vacc2x0123, vacc2x0123);
      vacc3x0123 = psimd_concat_hi_f32(vacc3x0123, vacc3x0123);
      vacc4x0123 = psimd_concat_hi_f32(vacc4x0123, vacc4x0123);
      vacc5x0123 = psimd_concat_hi_f32(vacc5x0123, vacc5x0123);
      # 减少剩余向量处理数量
      nr -= 2;
    }
    # 如果剩余可处理的向量数量不为0，则执行以下操作
    if (nr != 0) {
      # 将vacc0x0123的一个浮点数存储到地址c0
      psimd_store1_f32(c0, vacc0x0123);
      # 同上，对c1、c2、c3、c4、c5执行相同操作
      psimd_store1_f32(c1, vacc1x0123);
      psimd_store1_f32(c2, vacc2x0123);
      psimd_store1_f32(c3, vacc3x0123);
      psimd_store1_f32(c4, vacc4x0123);
      psimd_store1_f32(c5, vacc5x0123);
    }
  }
}


注释：


# 这行代码表示一个代码块的结束，通常用于结束一个函数、循环或条件语句的定义。
```