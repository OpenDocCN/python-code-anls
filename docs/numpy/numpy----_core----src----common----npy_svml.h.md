# `.\numpy\numpy\_core\src\common\npy_svml.h`

```
// 如果 NPY_SIMD 为真，并且定义了 NPY_HAVE_AVX512_SPR 和 NPY_CAN_LINK_SVML，则声明下列函数
extern void __svml_exps32(const npy_half*, npy_half*, npy_intp);
extern void __svml_exp2s32(const npy_half*, npy_half*, npy_intp);
extern void __svml_logs32(const npy_half*, npy_half*, npy_intp);
extern void __svml_log2s32(const npy_half*, npy_half*, npy_intp);
extern void __svml_log10s32(const npy_half*, npy_half*, npy_intp);
extern void __svml_expm1s32(const npy_half*, npy_half*, npy_intp);
extern void __svml_log1ps32(const npy_half*, npy_half*, npy_intp);
extern void __svml_cbrts32(const npy_half*, npy_half*, npy_intp);
extern void __svml_sins32(const npy_half*, npy_half*, npy_intp);
extern void __svml_coss32(const npy_half*, npy_half*, npy_intp);
extern void __svml_tans32(const npy_half*, npy_half*, npy_intp);
extern void __svml_asins32(const npy_half*, npy_half*, npy_intp);
extern void __svml_acoss32(const npy_half*, npy_half*, npy_intp);
extern void __svml_atans32(const npy_half*, npy_half*, npy_intp);
extern void __svml_atan2s32(const npy_half*, npy_half*, npy_intp);
extern void __svml_sinhs32(const npy_half*, npy_half*, npy_intp);
extern void __svml_coshs32(const npy_half*, npy_half*, npy_intp);
extern void __svml_tanhs32(const npy_half*, npy_half*, npy_intp);
extern void __svml_asinhs32(const npy_half*, npy_half*, npy_intp);
extern void __svml_acoshs32(const npy_half*, npy_half*, npy_intp);
extern void __svml_atanhs32(const npy_half*, npy_half*, npy_intp);
#endif

// 如果 NPY_SIMD 为真，并且定义了 NPY_HAVE_AVX512_SKX 和 NPY_CAN_LINK_SVML，则声明下列函数
extern __m512 __svml_expf16(__m512 x);
extern __m512 __svml_exp2f16(__m512 x);
extern __m512 __svml_logf16(__m512 x);
extern __m512 __svml_log2f16(__m512 x);
extern __m512 __svml_log10f16(__m512 x);
extern __m512 __svml_expm1f16(__m512 x);
extern __m512 __svml_log1pf16(__m512 x);
extern __m512 __svml_cbrtf16(__m512 x);
extern __m512 __svml_sinf16(__m512 x);
extern __m512 __svml_cosf16(__m512 x);
extern __m512 __svml_tanf16(__m512 x);
extern __m512 __svml_asinf16(__m512 x);
extern __m512 __svml_acosf16(__m512 x);
extern __m512 __svml_atanf16(__m512 x);
extern __m512 __svml_atan2f16(__m512 x, __m512 y);
extern __m512 __svml_sinhf16(__m512 x);
extern __m512 __svml_coshf16(__m512 x);
extern __m512 __svml_tanhf16(__m512 x);
extern __m512 __svml_asinhf16(__m512 x);
extern __m512 __svml_acoshf16(__m512 x);
extern __m512 __svml_atanhf16(__m512 x);
extern __m512 __svml_powf16(__m512 x, __m512 y);

extern __m512d __svml_exp8_ha(__m512d x);
extern __m512d __svml_exp28_ha(__m512d x);
extern __m512d __svml_log8_ha(__m512d x);
extern __m512d __svml_log28_ha(__m512d x);
extern __m512d __svml_log108_ha(__m512d x);
extern __m512d __svml_expm18_ha(__m512d x);
extern __m512d __svml_log1p8_ha(__m512d x);
extern __m512d __svml_cbrt8_ha(__m512d x);
extern __m512d __svml_sin8_ha(__m512d x);
extern __m512d __svml_cos8_ha(__m512d x);
extern __m512d __svml_tan8_ha(__m512d x);
extern __m512d __svml_asin8_ha(__m512d x);
# 声明外部函数 __svml_acos8_ha，参数和返回类型为 __m512d
extern __m512d __svml_acos8_ha(__m512d x);
# 声明外部函数 __svml_atan8_ha，参数和返回类型为 __m512d
extern __m512d __svml_atan8_ha(__m512d x);
# 声明外部函数 __svml_atan28_ha，参数为 __m512d x 和 __m512d y，返回类型为 __m512d
extern __m512d __svml_atan28_ha(__m512d x, __m512d y);
# 声明外部函数 __svml_sinh8_ha，参数和返回类型为 __m512d
extern __m512d __svml_sinh8_ha(__m512d x);
# 声明外部函数 __svml_cosh8_ha，参数和返回类型为 __m512d
extern __m512d __svml_cosh8_ha(__m512d x);
# 声明外部函数 __svml_tanh8_ha，参数和返回类型为 __m512d
extern __m512d __svml_tanh8_ha(__m512d x);
# 声明外部函数 __svml_asinh8_ha，参数和返回类型为 __m512d
extern __m512d __svml_asinh8_ha(__m512d x);
# 声明外部函数 __svml_acosh8_ha，参数和返回类型为 __m512d
extern __m512d __svml_acosh8_ha(__m512d x);
# 声明外部函数 __svml_atanh8_ha，参数和返回类型为 __m512d
extern __m512d __svml_atanh8_ha(__m512d x);
# 声明外部函数 __svml_pow8_ha，参数为 __m512d x 和 __m512d y，返回类型为 __m512d
extern __m512d __svml_pow8_ha(__m512d x, __m512d y);
#endif
```