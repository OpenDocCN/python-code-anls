# `.\pytorch\aten\src\ATen\test\vec_test_all_types.cpp`

```
// 包含 ATen 库的向量测试头文件
#include <ATen/test/vec_test_all_types.h>
// 包含 C10 实用工具的范围迭代器头文件
#include <c10/util/irange.h>

// 命名空间开始
namespace {
    // 如果 GTest 支持类型化测试，则定义 Memory 类模板继承自 testing::Test
    template <typename T>
    class Memory : public ::testing::Test {};

    // 定义 Arithmetics 类模板继承自 testing::Test
    template <typename T>
    class Arithmetics : public ::testing::Test {};

    // 定义 Comparison 类模板继承自 testing::Test
    template <typename T>
    class Comparison : public ::testing::Test {};

    // 定义 Bitwise 类模板继承自 testing::Test
    template <typename T>
    class Bitwise : public ::testing::Test {};

    // 定义 MinMax 类模板继承自 testing::Test
    template <typename T>
    class MinMax : public ::testing::Test {};

    // 定义 Nan 类模板继承自 testing::Test
    template <typename T>
    class Nan : public ::testing::Test {};

    // 定义 Interleave 类模板继承自 testing::Test
    template <typename T>
    class Interleave : public ::testing::Test {};

    // 定义 SignManipulation 类模板继承自 testing::Test
    template <typename T>
    class SignManipulation : public ::testing::Test {};

    // 定义 SignManipulationHalfPrecision 类模板继承自 testing::Test
    template <typename T>
    class SignManipulationHalfPrecision : public ::testing::Test {};

    // 定义 Rounding 类模板继承自 testing::Test
    template <typename T>
    class Rounding : public ::testing::Test {};

    // 定义 SqrtAndReciprocal 类模板继承自 testing::Test
    template <typename T>
    class SqrtAndReciprocal : public ::testing::Test {};

    // 定义 SqrtAndReciprocalReal 类模板继承自 testing::Test
    template <typename T>
    class SqrtAndReciprocalReal : public ::testing::Test {};

    // 定义 FractionAndRemainderReal 类模板继承自 testing::Test
    template <typename T>
    class FractionAndRemainderReal : public ::testing::Test {};

    // 定义 Trigonometric 类模板继承自 testing::Test
    template <typename T>
    class Trigonometric : public ::testing::Test {};

    // 定义 ErrorFunctions 类模板继承自 testing::Test
    template <typename T>
    class ErrorFunctions : public ::testing::Test {};

    // 定义 Exponents 类模板继承自 testing::Test
    template <typename T>
    class Exponents : public ::testing::Test {};

    // 定义 Hyperbolic 类模板继承自 testing::Test
    template <typename T>
    class Hyperbolic : public ::testing::Test {};

    // 定义 InverseTrigonometric 类模板继承自 testing::Test
    template <typename T>
    class InverseTrigonometric : public ::testing::Test {};

    // 定义 InverseTrigonometricReal 类模板继承自 testing::Test
    template <typename T>
    class InverseTrigonometricReal : public ::testing::Test {};

    // 定义 LGamma 类模板继承自 testing::Test
    template <typename T>
    class LGamma : public ::testing::Test {};

    // 定义 Logarithm 类模板继承自 testing::Test
    template <typename T>
    class Logarithm : public ::testing::Test {};

    // 定义 LogarithmReals 类模板继承自 testing::Test
    template <typename T>
    class LogarithmReals : public ::testing::Test {};

    // 定义 Pow 类模板继承自 testing::Test
    template <typename T>
    class Pow : public ::testing::Test {};

    // 定义 RangeFactories 类模板继承自 testing::Test
    template <typename T>
    class RangeFactories : public ::testing::Test {};

    // 定义 BitwiseFloatsAdditional 类模板继承自 testing::Test
    template <typename T>
    class BitwiseFloatsAdditional : public ::testing::Test {};

    // 定义 BitwiseFloatsAdditional2 类模板继承自 testing::Test
    template <typename T>
    class BitwiseFloatsAdditional2 : public ::testing::Test {};

    // 定义 RealTests 类模板继承自 testing::Test
    template <typename T>
    class RealTests : public ::testing::Test {};

    // 定义 ComplexTests 类模板继承自 testing::Test
    template <typename T>
    class ComplexTests : public ::testing::Test {};

    // 定义 QuantizationTests 类模板继承自 testing::Test
    template <typename T>
    class QuantizationTests : public ::testing::Test {};

    // 定义 Quantization8BitWithTailTests 类模板继承自 testing::Test
    template <typename T>
    class Quantization8BitWithTailTests : public ::testing::Test {};

    // 定义 FunctionalTests 类模板继承自 testing::Test
    template <typename T>
    class FunctionalTests : public ::testing::Test {};

    // 定义 FunctionalTestsReducedFloat 类模板继承自 testing::Test
    template <typename T>
    class FunctionalTestsReducedFloat : public ::testing::Test {};

    // 定义 InfiniteTests 类模板继承自 testing::Test
    template <typename T>
    class InfiniteTests : public ::testing::Test {};

    // 定义 VecConvertTests 类模板继承自 testing::Test
    template <typename T>
    class VecConvertTests : public ::testing::Test {};

    // 定义 VecMaskTests 类模板继承自 testing::Test
    template <typename T>
    class VecMaskTests : public ::testing::Test {};

    // 使用 RealFloatTestedTypes 作为模板参数，定义 RealFloatTestedTypes 别名
    using RealFloatTestedTypes = ::testing::Types<vfloat, vdouble>;
    // 定义一个模板类型别名 FloatTestedTypes，包含四种类型：vfloat, vdouble, vcomplex, vcomplexDbl
    using FloatTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl>;
    // 定义一个模板类型别名 ALLTestedTypes，包含九种类型：vfloat, vdouble, vcomplex, vlong, vint, vshort, vqint8, vquint8, vqint
    using ALLTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vlong, vint, vshort, vqint8, vquint8, vqint>;
    // 定义一个模板类型别名 QuantTestedTypes，包含三种类型：vqint8, vquint8, vqint
    using QuantTestedTypes = ::testing::Types<vqint8, vquint8, vqint>;
#if (defined(CPU_CAPABILITY_AVX2) ||  defined(CPU_CAPABILITY_AVX512))  && !defined(_MSC_VER)
    // 如果定义了 AVX2 或者 AVX512 CPU能力，并且不是在 MSC 编译器下，则使用 Quantization8BitWithTailTestedTypes 进行测试
    using Quantization8BitWithTailTestedTypes =
        ::testing::Types<vqint8, vquint8>;
#endif

// 使用 RealFloatIntTestedTypes 进行测试，包括 vfloat, vdouble, vlong, vint, vshort 类型
using RealFloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vlong, vint, vshort>;

// 使用 FloatIntTestedTypes 进行测试，包括 vfloat, vdouble, vcomplex, vcomplexDbl, vlong, vint, vshort 类型
using FloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl, vlong, vint, vshort>;

// 使用 ComplexTypes 进行测试，包括 vcomplex, vcomplexDbl 类型
using ComplexTypes = ::testing::Types<vcomplex, vcomplexDbl>;

// 使用 ReducedFloatTestedTypes 进行测试，包括 vBFloat16, vHalf 类型
using ReducedFloatTestedTypes = ::testing::Types<vBFloat16, vHalf>;

// 定义不同类型的测试套件
TYPED_TEST_SUITE(Memory, ALLTestedTypes);  // 内存相关测试套件
TYPED_TEST_SUITE(Arithmetics, FloatIntTestedTypes);  // 算术运算相关测试套件
TYPED_TEST_SUITE(Comparison, RealFloatIntTestedTypes);  // 比较运算相关测试套件
TYPED_TEST_SUITE(Bitwise, FloatIntTestedTypes);  // 位运算相关测试套件
TYPED_TEST_SUITE(MinMax, RealFloatIntTestedTypes);  // 最小值和最大值相关测试套件
TYPED_TEST_SUITE(Nan, RealFloatTestedTypes);  // NaN 相关测试套件
TYPED_TEST_SUITE(Interleave, RealFloatIntTestedTypes);  // 插值相关测试套件
TYPED_TEST_SUITE(SignManipulation, FloatIntTestedTypes);  // 符号操作相关测试套件
TYPED_TEST_SUITE(SignManipulationHalfPrecision, ReducedFloatTestedTypes);  // 半精度符号操作相关测试套件
TYPED_TEST_SUITE(Rounding, RealFloatTestedTypes);  // 四舍五入相关测试套件
TYPED_TEST_SUITE(SqrtAndReciprocal, FloatTestedTypes);  // 平方根和倒数相关测试套件
TYPED_TEST_SUITE(SqrtAndReciprocalReal, RealFloatTestedTypes);  // 实数平方根和倒数相关测试套件
TYPED_TEST_SUITE(FractionAndRemainderReal, RealFloatTestedTypes);  // 实数分数和余数相关测试套件
TYPED_TEST_SUITE(Trigonometric, RealFloatTestedTypes);  // 三角函数相关测试套件
TYPED_TEST_SUITE(ErrorFunctions, RealFloatTestedTypes);  // 错误函数相关测试套件
TYPED_TEST_SUITE(Exponents, RealFloatTestedTypes);  // 指数相关测试套件
TYPED_TEST_SUITE(Hyperbolic, RealFloatTestedTypes);  // 双曲函数相关测试套件
TYPED_TEST_SUITE(InverseTrigonometricReal, RealFloatTestedTypes);  // 实数反三角函数相关测试套件
TYPED_TEST_SUITE(InverseTrigonometric, FloatTestedTypes);  // 反三角函数相关测试套件
TYPED_TEST_SUITE(LGamma, RealFloatTestedTypes);  // Gamma 函数相关测试套件
TYPED_TEST_SUITE(Logarithm, FloatTestedTypes);  // 对数相关测试套件
TYPED_TEST_SUITE(LogarithmReals, RealFloatTestedTypes);  // 实数对数相关测试套件
TYPED_TEST_SUITE(Pow, RealFloatTestedTypes);  // 幂函数相关测试套件
TYPED_TEST_SUITE(RealTests, RealFloatTestedTypes);  // 实数测试套件
TYPED_TEST_SUITE(RangeFactories, FloatIntTestedTypes);  // 范围工厂相关测试套件
TYPED_TEST_SUITE(BitwiseFloatsAdditional, RealFloatTestedTypes);  // 浮点数位运算附加测试套件
TYPED_TEST_SUITE(BitwiseFloatsAdditional2, FloatTestedTypes);  // 浮点数位运算附加测试套件2
TYPED_TEST_SUITE(QuantizationTests, QuantTestedTypes);  // 量化测试套件
TYPED_TEST_SUITE(InfiniteTests, RealFloatTestedTypes);  // 无穷大测试套件

#if (defined(CPU_CAPABILITY_AVX2) ||  defined(CPU_CAPABILITY_AVX512))  && !defined(_MSC_VER)
    // 如果定义了 AVX2 或者 AVX512 CPU能力，并且不是在 MSC 编译器下，则使用 Quantization8BitWithTailTests 进行测试
    TYPED_TEST_SUITE(
        Quantization8BitWithTailTests,
        Quantization8BitWithTailTestedTypes);
#endif

// 功能性测试套件，包括不同的实数和整数类型
TYPED_TEST_SUITE(FunctionalTests, RealFloatIntTestedTypes);

// 功能性测试套件，针对降低精度的浮点数类型
TYPED_TEST_SUITE(FunctionalTestsReducedFloat, ReducedFloatTestedTypes);

// 向量转换相关测试套件，包括不同的实数和整数类型
TYPED_TEST_SUITE(VecConvertTests, RealFloatIntTestedTypes);

// 向量掩码相关测试套件，包括不同的实数和整数类型
TYPED_TEST_SUITE(VecMaskTests, RealFloatIntTestedTypes);
    # 定义一个模板化的测试函数 Memory.UnAlignedLoadStore，使用 TypeParam 和 ValueType<TypeParam>
    TYPED_TEST(Memory, UnAlignedLoadStore) {
        # 使用 vec 作为 TypeParam 类型的别名，VT 作为 TypeParam 类型的值类型别名
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        # 计算向量 vec 的大小乘以值类型 VT 的字节大小，得到缓冲区大小 b_size
        constexpr size_t b_size = vec::size() * sizeof(VT);
        # 定义一个 128 * b_size 大小的 CACHE_ALIGN 对齐的无符号字符数组 ref_storage
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN unsigned char ref_storage[128 * b_size];
        # 定义一个 128 * b_size 大小的 CACHE_ALIGN 对齐的无符号字符数组 storage
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN unsigned char storage[128 * b_size];
        # 使用测试种子 TestSeed() 初始化随机数生成器 generator
        auto seed = TestSeed();
        ValueGen<unsigned char> generator(seed);
        # 对 ref_storage 数组中的每个元素调用 generator.get() 方法来填充数据
        for (auto& x : ref_storage) {
            x = generator.get();
        }
        # 测试计数加载存储操作
#if defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR)
        // 如果编译器支持 VSX 或者 ZVECTOR，执行以下代码块
        for (int i = 1; i < 2 * vec::size(); i++) {
            // 从引用存储中加载向量数据到 v，索引为 i
            vec v = vec::loadu(ref_storage, i);
            // 将 v 中的数据存储到 storage
            v.store(storage);
            // 计算要比较的字节数量，最小为 i * sizeof(VT) 和 b_size 的较小值
            size_t count = std::min(i * sizeof(VT), b_size);
            // 比较 ref_storage 和 storage 的前 count 字节是否相同
            bool cmp = (std::memcmp(ref_storage, storage, count) == 0);
            // 断言比较结果为真，否则输出详细失败信息
            ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nCount: " << i;
            // 如果有测试失败，跳出循环
            if (::testing::Test::HasFailure()) {
                break;
            }
            // 清空 storage 的内容
            std::memset(storage, 0, b_size);
        }
#endif
        // 测试非对齐加载存储
        for (size_t offset = 0; offset < b_size; offset += 1) {
            // 设置指针 p1 和 p2 分别指向 ref_storage 和 storage 的偏移位置
            unsigned char* p1 = ref_storage + offset;
            unsigned char* p2 = storage + offset;
            // 从 p1 和 p2 开始，每次移动 b_size 字节，直到超出 ref_storage 的末尾
            for (; p1 + b_size <= std::end(ref_storage); p1 += b_size, p2 += b_size) {
                // 从 p1 中非对齐加载一个向量 v
                vec v = vec::loadu(p1);
                // 将 v 中的数据存储到 p2
                v.store(p2);
            }
            // 计算写入的字节数
            size_t written = p1 - ref_storage - offset;
            // 比较 ref_storage 和 storage 从 offset 开始，长度为 written 的数据是否相同
            bool cmp = (std::memcmp(ref_storage + offset, storage + offset, written) == 0);
            // 断言比较结果为真，否则输出详细失败信息
            ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nMismatch at unaligned offset: " << offset;
            // 如果有测试失败，跳出循环
            if (::testing::Test::HasFailure()) {
                break;
            }
            // 清空 storage 的内容
            std::memset(storage, 0, sizeof storage);
        }
    }
    TYPED_TEST(SignManipulation, Absolute) {
        // 类型别名，vec 表示 TypeParam
        using vec = TypeParam;
        // 检查是否为复数类型的相对误差
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        // 调用 test_unary 函数进行一元操作测试
        test_unary<vec>(
            NAME_INFO(absolute), RESOLVE_OVERLOAD(local_abs),
            [](vec v) { return v.abs(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, checkRelativeErr),
            RESOLVE_OVERLOAD(filter_int_minimum));
    }
    TYPED_TEST(SignManipulation, Negate) {
        // 类型别名，vec 表示 TypeParam
        using vec = TypeParam;
        // 对 int 和 long 类型的最小值进行负数处理测试
        test_unary<vec>(
            NAME_INFO(negate), std::negate<ValueType<vec>>(),
            [](vec v) { return v.neg(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_int_minimum));
    }
    TYPED_TEST(SignManipulationHalfPrecision, AbsNegate) {
      // 定义枚举类型 SignOpType，包含 ABS 和 NEGATE 两个值
      typedef enum  {
        ABS,
        NEGATE
      } SignOpType;
      // 类型别名，vec 表示 TypeParam，VT 表示 UholdType<TypeParam>，RT 表示 float 类型
      using vec = TypeParam;
      using VT = UholdType<TypeParam>;
      using RT = float; // reference
      float atol = 0.01f; // 绝对容差
      float rtol = 0.01f; // 相对容差

      // lambda 函数 cmp，比较 ref 和 val 的差值是否在容差范围内
      auto cmp = [&](RT ref, VT val) {
        return std::abs(ref - RT(val)) <= atol + rtol * std::abs(val);
      };
#define APPLY_FN_AND_STORE(VEC_TYPE)                            \
      [&](SignOpType op_type, VEC_TYPE& x_fp_vec, void *x_fp) { \
        if (op_type == SignOpType::NEGATE) {                    \
          x_fp_vec.neg().store(x_fp);                           \
        } else {                                                \
          x_fp_vec.abs().store(x_fp);                           \
        }                                                       \
      }

定义宏 `APPLY_FN_AND_STORE`，用于创建一个 lambda 函数，接受操作类型 `op_type`、向量 `x_fp_vec` 和指向数据的指针 `x_fp`。根据 `op_type` 的值，对 `x_fp_vec` 进行取负或取绝对值操作，并将结果存储到 `x_fp` 指向的位置。


      auto apply_fn_and_store_ref = APPLY_FN_AND_STORE(vfloat);
      auto apply_fn_and_store_half = APPLY_FN_AND_STORE(vec);

创建两个函数对象 `apply_fn_and_store_ref` 和 `apply_fn_and_store_half`，分别通过 `APPLY_FN_AND_STORE` 宏定义的 lambda 函数来操作 `vfloat` 和 `vec` 类型的向量。


      auto half_precision_ut = [&](SignOpType op_type) {
        constexpr auto N = vec::size();
        CACHE_ALIGN RT x_fp[N];
        CACHE_ALIGN VT x_hp[N];
        auto seed = TestSeed();
        ValueGen<RT> generator(RT(-1), RT(1), seed);
        for (const auto i : c10::irange(N)) {
            x_fp[i] = generator.get();
            x_hp[i] = VT(x_fp[i]);
        }
        auto x_fp_vec = vfloat::loadu(x_fp);
        apply_fn_and_store_ref(op_type, x_fp_vec, x_fp);
        x_fp_vec = vfloat::loadu(x_fp + vfloat::size());
        apply_fn_and_store_ref(op_type, x_fp_vec, x_fp + vfloat::size());

        auto x_hp_vec = vec::loadu(x_hp);
        apply_fn_and_store_half(op_type, x_hp_vec, x_hp);

        for (int64_t len = 0; len < N; len++) {
            ASSERT_TRUE(cmp(x_fp[len], x_hp[len])) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nabs/negate, Length: " << len << "; fp32: " << x_fp[len] << "; bf16/fp16: " << RT(x_hp[len]);
        }
      };

定义函数 `half_precision_ut`，接受一个 `SignOpType` 类型的参数 `op_type`。在函数内部，首先生成长度为 `N` 的随机浮点数数组 `x_fp` 和对应的半精度浮点数数组 `x_hp`。然后使用 `vfloat::loadu` 和 `vec::loadu` 加载 `x_fp` 和 `x_hp` 到向量对象 `x_fp_vec` 和 `x_hp_vec` 中，并通过 `apply_fn_and_store_ref` 和 `apply_fn_and_store_half` 执行具体的操作（绝对值或取负）。最后，使用断言检查操作后的结果是否一致，若不一致则输出详细的失败信息，包括测试种子和具体数值。


      half_precision_ut(SignOpType::ABS);
      half_precision_ut(SignOpType::NEGATE);

分别调用 `half_precision_ut` 函数两次，以测试绝对值和取负操作对应的情况。


    }
    TYPED_TEST(Rounding, Round) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        UVT case1 = -658.5f;
        UVT exp1 = -658.f;
        UVT case2 = -657.5f;
        UVT exp2 = -658.f;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-1000, 1000}} })
            .addCustom({ {case1},exp1 })
            .addCustom({ {case2},exp2 })
            .setTrialCount(64000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(round),
            RESOLVE_OVERLOAD(at::native::round_impl),
            [](vec v) { return v.round(); },
            test_case);
    }
    TYPED_TEST(Rounding, Ceil) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(ceil),
            RESOLVE_OVERLOAD(std::ceil),
            [](vec v) { return v.ceil(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Rounding, Floor) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(floor),
            RESOLVE_OVERLOAD(std::floor),
            [](vec v) { return v.floor(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }

以下部分代码不需要注释，因为它们属于另外的测试用例定义和调用，与之前的部分功能关联不大。
    # 定义名为 Rounding 的类型测试模板，针对截断操作进行测试
    TYPED_TEST(Rounding, Trunc) {
        # 使用 TypeParam 定义向量类型
        using vec = TypeParam;
        # 调用 test_unary 函数，测试截断操作
        test_unary<vec>(
            # 获取截断操作的名称信息
            NAME_INFO(trunc),
            # 解析 std::trunc 的重载版本
            RESOLVE_OVERLOAD(std::trunc),
            # Lambda 表达式，执行向量 v 的截断操作
            [](vec v) { return v.trunc(); },
            # 创建默认的一元测试用例，使用 TestSeed 初始化
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    
    # 定义名为 SqrtAndReciprocal 的类型测试模板，针对平方根操作进行测试
    TYPED_TEST(SqrtAndReciprocal, Sqrt) {
        # 使用 TypeParam 定义向量类型
        using vec = TypeParam;
        # 调用 test_unary 函数，测试平方根操作
        test_unary<vec>(
            # 获取平方根操作的名称信息
            NAME_INFO(sqrt),
            # 解析 local_sqrt 的重载版本
            RESOLVE_OVERLOAD(local_sqrt),
            # Lambda 表达式，执行向量 v 的平方根操作
            [](vec v) { return v.sqrt(); },
            # 创建默认的一元测试用例，使用 TestSeed 初始化，并设置额外的参数
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    
    # 定义名为 SqrtAndReciprocalReal 的类型测试模板，针对倒数平方根操作进行测试
    TYPED_TEST(SqrtAndReciprocalReal, RSqrt) {
        # 使用 TypeParam 定义向量类型
        using vec = TypeParam;
        # 调用 test_unary 函数，测试倒数平方根操作
        test_unary<vec>(
            # 获取倒数平方根操作的名称信息
            NAME_INFO(rsqrt),
            # 解析 rsqrt<ValueType<vec>> 的重载版本
            rsqrt<ValueType<vec>>,
            # Lambda 表达式，执行向量 v 的倒数平方根操作
            [](vec v) { return v.rsqrt(); },
            # 创建默认的一元测试用例，使用 TestSeed 初始化，并设置额外的参数
            createDefaultUnaryTestCase<vec>(TestSeed()),
            # 解析 filter_zero 的重载版本，用于处理零值情况
            RESOLVE_OVERLOAD(filter_zero));
    }
    
    # 定义名为 SqrtAndReciprocalReal 的类型测试模板，针对倒数操作进行测试
    TYPED_TEST(SqrtAndReciprocalReal, Reciprocal) {
        # 使用 TypeParam 定义向量类型
        using vec = TypeParam;
        # 调用 test_unary 函数，测试倒数操作
        test_unary<vec>(
            # 获取倒数操作的名称信息
            NAME_INFO(reciprocal),
            # 解析 reciprocal<ValueType<vec>> 的重载版本
            reciprocal<ValueType<vec>>,
            # Lambda 表达式，执行向量 v 的倒数操作
            [](vec v) { return v.reciprocal(); },
            # 创建默认的一元测试用例，使用 TestSeed 初始化，并设置额外的参数
            createDefaultUnaryTestCase<vec>(TestSeed()),
            # 解析 filter_zero 的重载版本，用于处理零值情况
            RESOLVE_OVERLOAD(filter_zero));
    }
    
    # 定义名为 FractionAndRemainderReal 的类型测试模板，针对小数部分操作进行测试
    TYPED_TEST(FractionAndRemainderReal, Frac) {
        # 使用 TypeParam 定义向量类型
        using vec = TypeParam;
        # 调用 test_unary 函数，测试小数部分操作
        test_unary<vec>(
            # 获取小数部分操作的名称信息
            NAME_INFO(frac),
            # 解析 frac 的重载版本
            RESOLVE_OVERLOAD(frac),
            # Lambda 表达式，执行向量 v 的小数部分操作
            [](vec v) { return v.frac(); },
            # 创建默认的一元测试用例，使用 TestSeed 初始化，并设置额外的参数
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    
    # 定义名为 FractionAndRemainderReal 的类型测试模板，针对浮点数取余操作进行测试
    TYPED_TEST(FractionAndRemainderReal, Fmod) {
        # 使用 TypeParam 定义向量类型
        using vec = TypeParam;
        # 调用 test_binary 函数，测试浮点数取余操作
        test_binary<vec>(
            # 获取浮点数取余操作的名称信息
            NAME_INFO(fmod),
            # 解析 std::fmod 的重载版本
            RESOLVE_OVERLOAD(std::fmod),
            # Lambda 表达式，执行向量 v0 和 v1 的浮点数取余操作
            [](vec v0, vec v1) { return v0.fmod(v1); },
            # 创建默认的二元测试用例，使用 TestSeed 初始化
            createDefaultBinaryTestCase<vec>(TestSeed()),
            # 解析 filter_fmod 的重载版本，用于处理特定情况
            RESOLVE_OVERLOAD(filter_fmod));
    }
    
    # 定义名为 Trigonometric 的类型测试模板，针对正弦操作进行测试
    TYPED_TEST(Trigonometric, Sin) {
        # 使用 TypeParam 定义向量类型
        using vec = TypeParam;
        # 使用 UvalueType<TypeParam> 定义 UVT 类型
        using UVT = UvalueType<TypeParam>;
        # 创建 TestingCase<vec> 的构建器
        auto test_case = TestingCase<vec>::getBuilder()
            # 添加在特定域内的测试条件，检查 UVT 是否在给定范围内
            .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
            .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
            # 设置试验次数为 8000
            .setTrialCount(8000)
            # 使用 TestSeed 初始化测试种子
            .setTestSeed(TestSeed());
        # 调用 test_unary 函数，测试正弦操作
        test_unary<vec>(
            # 获取正弦操作的名称信息
            NAME_INFO(sin),
            # 解析 std::sin 的重载版本
            RESOLVE_OVERLOAD(std::sin),
            # Lambda 表达式，执行向量 v 的正弦操作
            [](vec v) { return v.sin(); },
            # 使用之前创建的测试用例
            test_case);
    }
    
    # 定义名为 Trigonometric 的类型测试模板，针对余弦操作进行测试
    TYPED_TEST(Trigonometric, Cos) {
        # 使用 TypeParam 定义向量类型
        using vec = TypeParam;
        # 使用 UvalueType<TypeParam> 定义 UVT 类型
        using UVT = UvalueType<TypeParam>;
        # 创建 TestingCase<vec> 的构建器
        auto test_case = TestingCase<vec>::getBuilder()
            # 添加在特定域内的测试条件，检查 UVT 是否在给定范围内
            .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
            .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
            # 设置试验次数为 8000
            .setTrialCount(8000)
            # 使用 TestSeed 初始化测试种子
            .setTestSeed(TestSeed());
        # 调用 test_unary 函数，测试余弦操作
        test_unary<vec>(
            # 获取余弦操作的名称信息
            NAME_INFO(cos),
            # 解析 std::cos 的重载版本
            RESOLVE_OVERLOAD(std::cos),
            # Lambda 表达式，执行向量 v 的余弦操作
            [](vec v) { return v.cos(); },
            # 使用之前创建的测试用例
            test_case);
    }
    // 定义名为 Trigonometric 的类型参数化测试，测试正切函数
    TYPED_TEST(Trigonometric, Tan) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 调用 test_unary 函数，测试一元操作 tan
        test_unary<vec>(
            // 获取 tan 的名称信息
            NAME_INFO(tan),
            // 解析 std::tan 的重载版本
            RESOLVE_OVERLOAD(std::tan),
            // 使用 lambda 表达式调用 vec 类型的 tan 方法
            [](vec v) { return v.tan(); },
            // 创建默认的一元测试用例
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    
    // 定义名为 Hyperbolic 的类型参数化测试，测试双曲正切函数
    TYPED_TEST(Hyperbolic, Tanh) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 调用 test_unary 函数，测试一元操作 tanh
        test_unary<vec>(
            // 获取 tanh 的名称信息
            NAME_INFO(tanH),
            // 解析 std::tanh 的重载版本
            RESOLVE_OVERLOAD(std::tanh),
            // 使用 lambda 表达式调用 vec 类型的 tanh 方法
            [](vec v) { return v.tanh(); },
            // 创建默认的一元测试用例
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    
    // 定义名为 Hyperbolic 的类型参数化测试，测试双曲正弦函数
    TYPED_TEST(Hyperbolic, Sinh) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 使用 UVT 作为 vec 的值类型别名
        using UVT = UvalueType<TypeParam>;
        // 创建测试用例
        auto test_case =
            TestingCase<vec>::getBuilder()
            // 添加值域检查，检查在 {-88, 88} 范围内
            .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
            // 设置试验次数
            .setTrialCount(65536)
            // 设置测试种子
            .setTestSeed(TestSeed());
        // 调用 test_unary 函数，测试一元操作 sinh
        test_unary<vec>(
            // 获取 sinh 的名称信息
            NAME_INFO(sinh),
            // 解析 std::sinh 的重载版本
            RESOLVE_OVERLOAD(std::sinh),
            // 使用 lambda 表达式调用 vec 类型的 sinh 方法
            [](vec v) { return v.sinh(); },
            // 使用指定的测试用例
            test_case);
    }
    
    // 定义名为 Hyperbolic 的类型参数化测试，测试双曲余弦函数
    TYPED_TEST(Hyperbolic, Cosh) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 使用 UVT 作为 vec 的值类型别名
        using UVT = UvalueType<TypeParam>;
        // 创建测试用例
        auto test_case =
            TestingCase<vec>::getBuilder()
            // 添加值域检查，检查在 {-88, 88} 范围内
            .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
            // 设置试验次数
            .setTrialCount(65536)
            // 设置测试种子
            .setTestSeed(TestSeed());
        // 调用 test_unary 函数，测试一元操作 cosh
        test_unary<vec>(
            // 获取 cosh 的名称信息
            NAME_INFO(cosh),
            // 解析 std::cosh 的重载版本
            RESOLVE_OVERLOAD(std::cosh),
            // 使用 lambda 表达式调用 vec 类型的 cosh 方法
            [](vec v) { return v.cosh(); },
            // 使用指定的测试用例
            test_case);
    }
    
    // 定义名为 InverseTrigonometric 的类型参数化测试，测试反正弦函数
    TYPED_TEST(InverseTrigonometric, Asin) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 使用 UVT 作为 vec 的值类型别名
        using UVT = UvalueType<TypeParam>;
        // 检查是否为复数类型
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        // 创建测试用例
        auto test_case =
            TestingCase<vec>::getBuilder()
            // 添加值域检查，检查在 {-10, 10} 范围内，根据是否为复数类型进行相对误差检查
            .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
            // 设置试验次数
            .setTrialCount(125536)
            // 设置测试种子
            .setTestSeed(TestSeed());
        // 调用 test_unary 函数，测试一元操作 asin
        test_unary<vec>(
            // 获取 asin 的名称信息
            NAME_INFO(asin),
            // 解析 local_asin 函数
            RESOLVE_OVERLOAD(local_asin),
            // 使用 lambda 表达式调用 vec 类型的 asin 方法
            [](vec v) { return v.asin(); },
            // 使用指定的测试用例
            test_case);
    }
    
    // 定义名为 InverseTrigonometric 的类型参数化测试，测试反余弦函数
    TYPED_TEST(InverseTrigonometric, ACos) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 使用 UVT 作为 vec 的值类型别名
        using UVT = UvalueType<TypeParam>;
        // 检查是否为复数类型
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        // 创建测试用例
        auto test_case =
            TestingCase<vec>::getBuilder()
            // 添加值域检查，检查在 {-10, 10} 范围内，根据是否为复数类型进行相对误差检查
            .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
            // 设置试验次数
            .setTrialCount(125536)
            // 设置测试种子
            .setTestSeed(TestSeed());
        // 调用 test_unary 函数，测试一元操作 acos
        test_unary<vec>(
            // 获取 acos 的名称信息
            NAME_INFO(acos),
            // 解析 local_acos 函数
            RESOLVE_OVERLOAD(local_acos),
            // 使用 lambda 表达式调用 vec 类型的 acos 方法
            [](vec v) { return v.acos(); },
            // 使用指定的测试用例
            test_case);
    }
    // 在逆三角函数测试中，测试反正切函数
    TYPED_TEST(InverseTrigonometric, ATan) {
        // 检查是否复数类型
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 使用 UVT 作为 TypeParam 的值类型
        using UVT = UvalueType<TypeParam>;
        // 创建测试案例
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-100, 100}}, checkRelativeErr, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        // 进行一元函数测试，名称为 atan
        test_unary<vec>(
            NAME_INFO(atan),
            // 解析 std::atan 函数
            RESOLVE_OVERLOAD(std::atan),
            // 使用 v.atan() 方法
            [](vec v) { return v.atan(); },
            // 使用前面定义的测试案例
            test_case,
            // 解析 filter_zero 函数重载
            RESOLVE_OVERLOAD(filter_zero));
    }
    // 在对数函数测试中，测试自然对数函数
    TYPED_TEST(Logarithm, Log) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 进行一元函数测试，名称为 log
        test_unary<vec>(
            NAME_INFO(log),
            // 解析 std::log 函数
            RESOLVE_OVERLOAD(std::log),
            // 使用 v.log() 方法
            [](const vec& v) { return v.log(); },
            // 创建默认的一元测试案例，使用默认的测试种子
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    // 在对数函数测试中，测试以2为底的对数函数
    TYPED_TEST(LogarithmReals, Log2) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 进行一元函数测试，名称为 log2
        test_unary<vec>(
            NAME_INFO(log2),
            // 解析 local_log2 函数重载
            RESOLVE_OVERLOAD(local_log2),
            // 使用 v.log2() 方法
            [](const vec& v) { return v.log2(); },
            // 创建默认的一元测试案例，使用默认的测试种子
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    // 在对数函数测试中，测试常用对数函数
    TYPED_TEST(Logarithm, Log10) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 进行一元函数测试，名称为 log10
        test_unary<vec>(
            NAME_INFO(log10),
            // 解析 std::log10 函数
            RESOLVE_OVERLOAD(std::log10),
            // 使用 v.log10() 方法
            [](const vec& v) { return v.log10(); },
            // 创建默认的一元测试案例，使用默认的测试种子
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    // 在对数函数测试中，测试 log(1+x) 函数
    TYPED_TEST(LogarithmReals, Log1p) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 使用 UVT 作为 TypeParam 的值类型
        using UVT = UvalueType<TypeParam>;
        // 创建测试案例
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-1, 1000}}, true, getDefaultTolerance<UVT>()})
            .addDomain(CheckWithinDomains<UVT>{ { {1000, 1.e+30}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        // 进行一元函数测试，名称为 log1p
        test_unary<vec>(
            NAME_INFO(log1p),
            // 解析 std::log1p 函数
            RESOLVE_OVERLOAD(std::log1p),
            // 使用 v.log1p() 方法
            [](const vec& v) { return v.log1p(); },
            // 使用前面定义的测试案例
            test_case);
    }
    // 在指数函数测试中，测试自然指数函数
    TYPED_TEST(Exponents, Exp) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 进行一元函数测试，名称为 exp
        test_unary<vec>(
            NAME_INFO(exp),
            // 解析 std::exp 函数
            RESOLVE_OVERLOAD(std::exp),
            // 使用 v.exp() 方法
            [](const vec& v) { return v.exp(); },
            // 创建默认的一元测试案例，使用默认的测试种子
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    // 在指数函数测试中，测试 exp(x)-1 函数
    TYPED_TEST(Exponents, Expm1) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 进行一元函数测试，名称为 expm1
        test_unary<vec>(
            NAME_INFO(expm1),
            // 解析 std::expm1 函数
            RESOLVE_OVERLOAD(std::expm1),
            // 使用 v.expm1() 方法
            [](const vec& v) { return v.expm1(); },
            // 创建默认的一元测试案例，使用默认的测试种子，不检查零点，检查负值
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    // 在误差函数测试中，测试误差函数
    TYPED_TEST(ErrorFunctions, Erf) {
        // 使用 vec 作为 TypeParam 的别名
        using vec = TypeParam;
        // 进行一元函数测试，名称为 erf
        test_unary<vec>(
            NAME_INFO(erf),
            // 解析 std::erf 函数
            RESOLVE_OVERLOAD(std::erf),
            // 使用 v.erf() 方法
            [](const vec& v) { return v.erf(); },
            // 创建默认的一元测试案例，使用默认的测试种子，不检查零点，检查负值
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    // 定义一个名为 ErrorFunctions 的类型模板测试类的特化，测试 std::erfc 函数
    TYPED_TEST(ErrorFunctions, Erfc) {
        // 使用 TypeParam 类型别名来代表向量类型
        using vec = TypeParam;
        // 调用 test_unary 函数测试单目操作，以名称信息 "erfc" 标识，使用 std::erfc 解析函数，
        // lambda 函数以向量 v 作为输入调用 v.erfc() 方法，使用默认的单目测试用例创建。
        test_unary<vec>(
            NAME_INFO(erfc),
            RESOLVE_OVERLOAD(std::erfc),
            [](const vec& v) { return v.erfc(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    
    // 定义 ErrorFunctions 类的另一个特化，测试 calc_erfinv 函数
    TYPED_TEST(ErrorFunctions, Erfinv) {
        // 使用 TypeParam 类型别名来代表向量类型
        using vec = TypeParam;
        // 调用 test_unary 函数测试单目操作，以名称信息 "erfinv" 标识，使用 calc_erfinv 解析函数，
        // lambda 函数以向量 v 作为输入调用 v.erfinv() 方法，使用默认的单目测试用例创建。
        test_unary<vec>(
            NAME_INFO(erfinv),
            RESOLVE_OVERLOAD(calc_erfinv),
            [](const vec& v) { return v.erfinv(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    
    // 定义 Nan 类的特化，测试 IsNan 函数
    TYPED_TEST(Nan, IsNan) {
        // 使用 TypeParam 类型别名来代表向量类型
        using vec = TypeParam;
        // 使用 ValueType<TypeParam> 别名表示向量中的值类型
        using VT = ValueType<TypeParam>;
        // 声明一个缓存对齐的数组 test_vals，其大小为向量的大小 vec::size()
        // NOLINTNEXTLINE 表示禁止特定的代码检查规则，避免使用 C 风格的数组
        CACHE_ALIGN VT test_vals[vec::size()];
        // 声明一个缓存对齐的数组 expected_vals，其大小为向量的大小 vec::size()
        // NOLINTNEXTLINE 表示禁止特定的代码检查规则，避免使用 C 风格的数组
        CACHE_ALIGN VT expected_vals[vec::size()];
        // 计算 vals 的值，其为 2 的 vec::size() 次方
        auto vals = 1 << (vec::size());
        // 对于 c10 命名空间中的范围值 val，执行以下循环
        for (const auto val : c10::irange(vals)) {
          // 遍历向量的大小 vec::size()
          for (int i = 0; i < vec::size(); ++i) {
            // 如果 val 的第 i 位为 1
            if (val & (1 << i)) {
              // 将 test_vals[i] 设置为 std::numeric_limits<VT>::quiet_NaN()
              test_vals[i] = std::numeric_limits<VT>::quiet_NaN();
              // 将 expected_vals[i] 的所有位设置为 1，大小为 sizeof(VT)
              std::memset(static_cast<void*>(&expected_vals[i]), 0xFF, sizeof(VT));
            } else {
              // 否则，将 test_vals[i] 设置为 (VT)0.123
              test_vals[i] = (VT)0.123;
              // 将 expected_vals[i] 的所有位设置为 0，大小为 sizeof(VT)
              std::memset(static_cast<void*>(&expected_vals[i]), 0, sizeof(VT));
            }
          }
          // 从 test_vals 加载到向量 actual，检查是否包含 NaN 值
          vec actual = vec::loadu(test_vals).isnan();
          // 从 expected_vals 加载到向量 expected，预期结果
          vec expected = vec::loadu(expected_vals);
          // 使用 AssertVectorized 类检查向量化的比较结果
          AssertVectorized<vec>(NAME_INFO(isnan), expected, actual).check();
        }
    }
    
    // 定义 LGamma 类的特化，测试 lgamma 函数
    TYPED_TEST(LGamma, LGamma) {
        // 使用 TypeParam 类型别名来代表向量类型
        using vec = TypeParam;
        // 使用 UvalueType<vec> 别名表示向量中的值类型
        using UVT = UvalueType<vec>;
        // 获取默认容差 tolerance
        UVT tolerance = getDefaultTolerance<UVT>();
        // 根据 UVT 类型选择不同的最大正确值 maxCorrect
        // 如果 UVT 为 float，则设置 maxCorrect 为 (UVT)4e+36，否则为 (UVT)2e+305
        UVT maxCorrect = std::is_same_v<UVT, float> ? (UVT)4e+36 : (UVT)2e+305;
        // 创建 TestingCase<vec> 对象 testCase，包含不同的 lgamma 测试域
        TestingCase<vec> testCase = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)0}}, true, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)0, (UVT)1000 }}, true, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)1000, maxCorrect }}, true, tolerance})
            .setTestSeed(TestSeed());
        // 调用 test_unary 函数测试单目操作，以名称信息 "lgamma" 标识，使用 std::lgamma 解析函数，
        // lambda 函数以向量 v 作为输入调用 v.lgamma() 方法，使用 testCase 作为测试用例
        test_unary<vec>(
            NAME_INFO(lgamma),
            RESOLVE_OVERLOAD(std::lgamma),
            [](vec v) { return v.lgamma(); },
            testCase);
    }
    
    // 定义 InverseTrigonometricReal 类的特化，测试 atan2 函数
    TYPED_TEST(InverseTrigonometricReal, ATan2) {
        // 使用 TypeParam 类型别名来代表向量类型
        using vec = TypeParam;
        // 调用 test_binary 函数测试二元操作，以名称信息 "atan2" 标识，使用 std::atan2 解析函数，
        // lambda 函数以向量 v0 和 v1 作为输入调用 v0.atan2(v1) 方法，使用默认的二元测试用例创建。
        test_binary<vec>(
            NAME_INFO(atan2),
            RESOLVE_OVERLOAD(std::atan2),
            [](vec v0, vec v1) {
                return v0.atan2(v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    # 定义类型测试用例 Pow，使用模板参数 TypeParam 作为向量类型
    TYPED_TEST(Pow, Pow) {
        # 使用 TypeParam 作为向量类型的别名 vec
        using vec = TypeParam;
        # 调用通用的二元测试函数 test_binary，测试 std::pow 的功能
        test_binary<vec>(
            # 获取函数名称和参数类型信息，用于测试名称信息
            NAME_INFO(pow),
            # 解析 std::pow 的重载版本
            RESOLVE_OVERLOAD(std::pow),
            # 使用 Lambda 表达式计算 v0 和 v1 的 pow 运算结果
            [](vec v0, vec v1) { return v0.pow(v1); },
            # 创建默认的二元测试用例，禁用特殊情况，启用正常情况
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    # 定义实数测试用例 Hypot，使用模板参数 TypeParam 作为向量类型
    TYPED_TEST(RealTests, Hypot) {
        # 使用 TypeParam 作为向量类型的别名 vec
        using vec = TypeParam;
        # 调用通用的二元测试函数 test_binary，测试 std::hypot 的功能
        test_binary<vec>(
            # 获取函数名称和参数类型信息，用于测试名称信息
            NAME_INFO(hypot),
            # 解析 std::hypot 的重载版本
            RESOLVE_OVERLOAD(std::hypot),
            # 使用 Lambda 表达式计算 v0 和 v1 的 hypot 运算结果
            [](vec v0, vec v1) { return v0.hypot(v1); },
            # 创建默认的二元测试用例，禁用特殊情况，启用正常情况
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    # 定义实数测试用例 NextAfter，使用模板参数 TypeParam 作为向量类型
    TYPED_TEST(RealTests, NextAfter) {
        # 使用 TypeParam 作为向量类型的别名 vec
        using vec = TypeParam;
        # 调用通用的二元测试函数 test_binary，测试 std::nextafter 的功能
        test_binary<vec>(
            # 获取函数名称和参数类型信息，用于测试名称信息
            NAME_INFO(nextafter),
            # 解析 std::nextafter 的重载版本
            RESOLVE_OVERLOAD(std::nextafter),
            # 使用 Lambda 表达式计算 v0 和 v1 的 nextafter 运算结果
            [](vec v0, vec v1) { return v0.nextafter(v1); },
            # 创建默认的二元测试用例，禁用特殊情况，启用正常情况
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    # 定义插入/反插入测试用例 Interleave，使用模板参数 TypeParam 作为向量类型
    TYPED_TEST(Interleave, Interleave) {
        # 使用 TypeParam 作为向量类型的别名 vec
        using vec = TypeParam;
        # 使用 ValueType<TypeParam> 别名 VT，表示向量元素类型
        using VT = ValueType<TypeParam>;
        # 计算数组大小 N，等于向量大小乘以 2
        constexpr auto N = vec::size() * 2LL;
        # 声明缓存对齐的数组 vals，存储随机生成的值
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT vals[N];
        # 声明缓存对齐的数组 interleaved，存储插入后的值
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT interleaved[N];
        # 创建随机数生成器，使用测试种子生成初始值
        auto seed = TestSeed();
        ValueGen<VT> generator(seed);
        # 填充 vals 数组，使用生成器生成的随机值
        for (VT& v : vals) {
            v = generator.get();
        }
        # 将 vals 数组的值复制到 interleaved 数组并进行插入操作
        copy_interleave(vals, interleaved);
        # 从 interleaved 数组加载前半部分数据到向量 a
        auto a = vec::loadu(vals);
        # 从 interleaved 数组加载后半部分数据到向量 b
        auto b = vec::loadu(vals + vec::size());
        # 使用 interleave2 函数插入 a 和 b 的数据，并返回结果
        auto cc = interleave2(a, b);
        # 断言向量化的结果与插入后的 interleaved 数组的前半部分匹配
        AssertVectorized<vec>(NAME_INFO(Interleave FirstHalf), std::get<0>(cc), vec::loadu(interleaved)).check(true);
        # 断言向量化的结果与插入后的 interleaved 数组的后半部分匹配
        AssertVectorized<vec>(NAME_INFO(Interleave SecondHalf), std::get<1>(cc), vec::loadu(interleaved + vec::size())).check(true);
    }
    # 定义插入/反插入测试用例 DeInterleave，使用模板参数 TypeParam 作为向量类型
    TYPED_TEST(Interleave, DeInterleave) {
        # 使用 TypeParam 作为向量类型的别名 vec
        using vec = TypeParam;
        # 使用 ValueType<TypeParam> 别名 VT，表示向量元素类型
        using VT = ValueType<TypeParam>;
        # 计算数组大小 N，等于向量大小乘以 2
        constexpr auto N = vec::size() * 2LL;
        # 声明缓存对齐的数组 vals，存储随机生成的值
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT vals[N];
        # 声明缓存对齐的数组 interleaved，存储插入后的值
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT interleaved[N];
        # 创建随机数生成器，使用测试种子生成初始值
        auto seed = TestSeed();
        ValueGen<VT> generator(seed);
        # 填充 vals 数组，使用生成器生成的随机值
        for (VT& v : vals) {
            v = generator.get();
        }
        # 将 vals 数组的值复制到 interleaved 数组并进行插入操作
        copy_interleave(vals, interleaved);
        # 从 interleaved 数组加载前半部分数据到向量 a
        auto a = vec::loadu(interleaved);
        # 从 interleaved 数组加载后半部分数据到向量 b
        auto b = vec::loadu(interleaved + vec::size());
        # 使用 deinterleave2 函数反插入 a 和 b 的数据，并返回结果
        auto cc = deinterleave2(a, b);
        # 断言向量化的结果与原始 vals 数组的前半部分匹配
        AssertVectorized<vec>(NAME_INFO(DeInterleave FirstHalf), std::get<0>(cc), vec::loadu(vals)).check(true);
        # 断言向量化的结果与原始 vals 数组的后半部分匹配
        AssertVectorized<vec>(NAME_INFO(DeInterleave SecondHalf), std::get<1>(cc), vec::loadu(vals + vec::size())).check(true);
    }
    # 测试加法操作，使用 TypeParam 类型的向量
    TYPED_TEST(Arithmetics, Plus) {
        # 定义类型别名 vec 为 TypeParam
        using vec = TypeParam;
        # 定义 VT 为 TypeParam 类型的值类型
        using VT = ValueType<TypeParam>;
        # 调用 test_binary 函数测试二元操作，测试名称为 plus
        test_binary<vec>(
            NAME_INFO(plus),  # 提供名称信息 plus
            std::plus<VT>(),  # 使用 std::plus 进行加法运算
            # lambda 函数定义，执行向量 v0 和 v1 的加法操作并返回结果
            [](const vec& v0, const vec& v1) -> vec {
                return v0 + v1;
            },
            # 创建默认的二元测试用例，使用 TestSeed 生成随机数种子
            createDefaultBinaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_add_overflow));  # 解析加法溢出的重载函数
    }

    # 测试减法操作，使用 TypeParam 类型的向量
    TYPED_TEST(Arithmetics, Minus) {
        # 定义类型别名 vec 为 TypeParam
        using vec = TypeParam;
        # 定义 VT 为 TypeParam 类型的值类型
        using VT = ValueType<TypeParam>;
        # 调用 test_binary 函数测试二元操作，测试名称为 minus
        test_binary<vec>(
            NAME_INFO(minus),  # 提供名称信息 minus
            std::minus<VT>(),  # 使用 std::minus 进行减法运算
            # lambda 函数定义，执行向量 v0 和 v1 的减法操作并返回结果
            [](const vec& v0, const vec& v1) -> vec {
                return v0 - v1;
            },
            # 创建默认的二元测试用例，使用 TestSeed 生成随机数种子
            createDefaultBinaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_sub_overflow));  # 解析减法溢出的重载函数
    }

    # 测试乘法操作，使用 TypeParam 类型的向量
    TYPED_TEST(Arithmetics, Multiplication) {
        # 定义类型别名 vec 为 TypeParam
        using vec = TypeParam;
        # 调用 test_binary 函数测试二元操作，测试名称为 mult
        test_binary<vec>(
            NAME_INFO(mult),  # 提供名称信息 mult
            RESOLVE_OVERLOAD(local_multiply),  # 解析本地乘法的重载函数
            # lambda 函数定义，执行向量 v0 和 v1 的乘法操作并返回结果
            [](const vec& v0, const vec& v1) { return v0 * v1; },
            # 创建默认的二元测试用例，使用 TestSeed 生成随机数种子，并且不使用特殊测试标志
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true),
            RESOLVE_OVERLOAD(filter_mult_overflow));  # 解析乘法溢出的重载函数
    }

    # 测试除法操作，使用 TypeParam 类型的向量
    TYPED_TEST(Arithmetics, Division) {
        # 定义类型别名 vec 为 TypeParam
        using vec = TypeParam;
        # 创建随机数种子对象 TestSeed
        TestSeed seed;
        # 调用 test_binary 函数测试二元操作，测试名称为 division
        test_binary<vec>(
            NAME_INFO(division),  # 提供名称信息 division
            RESOLVE_OVERLOAD(local_division),  # 解析本地除法的重载函数
            # lambda 函数定义，执行向量 v0 和 v1 的除法操作并返回结果
            [](const vec& v0, const vec& v1) { return v0 / v1; },
            # 创建默认的二元测试用例，使用 TestSeed 生成随机数种子
            createDefaultBinaryTestCase<vec>(seed),
            RESOLVE_OVERLOAD(filter_div_ub));  # 解析除法未定义行为的重载函数
    }

    # 测试按位与操作，使用 TypeParam 类型的向量
    TYPED_TEST(Bitwise, BitAnd) {
        # 定义类型别名 vec 为 TypeParam
        using vec = TypeParam;
        # 调用 test_binary 函数测试二元操作，测试名称为 bit_and
        test_binary<vec>(
            NAME_INFO(bit_and),  # 提供名称信息 bit_and
            RESOLVE_OVERLOAD(local_and),  # 解析本地按位与的重载函数
            # lambda 函数定义，执行向量 v0 和 v1 的按位与操作并返回结果
            [](const vec& v0, const vec& v1) { return v0 & v1; },
            # 创建默认的二元测试用例，使用 TestSeed 生成随机数种子，并且使用特殊测试标志
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }

    # 测试按位或操作，使用 TypeParam 类型的向量
    TYPED_TEST(Bitwise, BitOr) {
        # 定义类型别名 vec 为 TypeParam
        using vec = TypeParam;
        # 调用 test_binary 函数测试二元操作，测试名称为 bit_or
        test_binary<vec>(
            NAME_INFO(bit_or),  # 提供名称信息 bit_or
            RESOLVE_OVERLOAD(local_or),  # 解析本地按位或的重载函数
            # lambda 函数定义，执行向量 v0 和 v1 的按位或操作并返回结果
            [](const vec& v0, const vec& v1) { return v0 | v1; },
            # 创建默认的二元测试用例，使用 TestSeed 生成随机数种子，并且使用特殊测试标志
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }

    # 测试按位异或操作，使用 TypeParam 类型的向量
    TYPED_TEST(Bitwise, BitXor) {
        # 定义类型别名 vec 为 TypeParam
        using vec = TypeParam;
        # 调用 test_binary 函数测试二元操作，测试名称为 bit_xor
        test_binary<vec>(
            NAME_INFO(bit_xor),  # 提供名称信息 bit_xor
            RESOLVE_OVERLOAD(local_xor),  # 解析本地按位异或的重载函数
            # lambda 函数定义，执行向量 v0 和 v1 的按位异或操作并返回结果
            [](const vec& v0, const vec& v1) { return v0 ^ v1; },
            # 创建默认的二元测试用例，使用 TestSeed 生成随机数种子，并且使用特殊测试标志
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }

    # 测试相等比较操作，使用 TypeParam 类型的向量
    TYPED_TEST(Comparison, Equal) {
        # 定义类型别名 vec 为 TypeParam
        using vec = TypeParam;
        # 定义 VT 为 TypeParam 类型的值类型
        using VT = ValueType<TypeParam>;
        # 调用 test_binary 函数测试二元操作，测试名称为 ==
        test_binary<vec>(
            NAME_INFO(== ),  # 提供名称信息 ==
            # lambda 函数定义，使用 std::equal_to 比较操作符，对 v1 和 v2 执行相等比较并返回结果
            [](const VT& v1, const VT& v2) {return func_cmp(std::equal_to<VT>(), v1, v2); },
            # lambda 函数定义，执行向量 v0 和 v1 的相等比较操作并返回结果
            [](const vec& v0, const vec& v1) { return v0 == v1; },
            # 创建默认的二元测试用例，使用 TestSeed 生成随机数种子，并且使用特殊测试标志
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, NotEqual) {
        // 使用 TypeParam 作为向量类型，并且使用 ValueType<TypeParam> 作为值类型
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // 调用 test_binary 函数进行二元测试
        test_binary<vec>(
            // 传递 NAME_INFO(!=) 作为测试名称信息
            NAME_INFO(!= ),
            // 使用 lambda 表达式调用 func_cmp 函数，使用 std::not_equal_to<VT>() 进行比较
            [](const VT& v1, const VT& v2) {return func_cmp(std::not_equal_to<VT>(), v1, v2); },
            // 使用 lambda 表达式检查 v0 是否不等于 v1
            [](const vec& v0, const vec& v1) { return v0 != v1; },
            // 创建默认的二元测试用例，使用 TestSeed() 生成随机种子，确保真实测试运行
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Greater) {
        // 使用 TypeParam 作为向量类型，并且使用 ValueType<TypeParam> 作为值类型
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // 调用 test_binary 函数进行二元测试
        test_binary<vec>(
            // 传递 NAME_INFO(>) 作为测试名称信息
            NAME_INFO(> ),
            // 使用 lambda 表达式调用 func_cmp 函数，使用 std::greater<VT>() 进行比较
            [](const VT& v1, const VT& v2) {return func_cmp(std::greater<VT>(), v1, v2); },
            // 使用 lambda 表达式检查 v0 是否大于 v1
            [](const vec& v0, const vec& v1) { return v0 > v1; },
            // 创建默认的二元测试用例，使用 TestSeed() 生成随机种子，确保真实测试运行
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Less) {
        // 使用 TypeParam 作为向量类型，并且使用 ValueType<TypeParam> 作为值类型
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // 调用 test_binary 函数进行二元测试
        test_binary<vec>(
            // 传递 NAME_INFO(<) 作为测试名称信息
            NAME_INFO(< ),
            // 使用 lambda 表达式调用 func_cmp 函数，使用 std::less<VT>() 进行比较
            [](const VT& v1, const VT& v2) {return func_cmp(std::less<VT>(), v1, v2); },
            // 使用 lambda 表达式检查 v0 是否小于 v1
            [](const vec& v0, const vec& v1) { return v0 < v1; },
            // 创建默认的二元测试用例，使用 TestSeed() 生成随机种子，确保真实测试运行
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, GreaterEqual) {
        // 使用 TypeParam 作为向量类型，并且使用 ValueType<TypeParam> 作为值类型
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // 调用 test_binary 函数进行二元测试
        test_binary<vec>(
            // 传递 NAME_INFO(>=) 作为测试名称信息
            NAME_INFO(>= ),
            // 使用 lambda 表达式调用 func_cmp 函数，使用 std::greater_equal<VT>() 进行比较
            [](const VT& v1, const VT& v2) {return func_cmp(std::greater_equal<VT>(), v1, v2); },
            // 使用 lambda 表达式检查 v0 是否大于等于 v1
            [](const vec& v0, const vec& v1) { return v0 >= v1; },
            // 创建默认的二元测试用例，使用 TestSeed() 生成随机种子，确保真实测试运行
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, LessEqual) {
        // 使用 TypeParam 作为向量类型，并且使用 ValueType<TypeParam> 作为值类型
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // 调用 test_binary 函数进行二元测试
        test_binary<vec>(
            // 传递 NAME_INFO(<=) 作为测试名称信息
            NAME_INFO(<= ),
            // 使用 lambda 表达式调用 func_cmp 函数，使用 std::less_equal<VT>() 进行比较
            [](const VT& v1, const VT& v2) {return func_cmp(std::less_equal<VT>(), v1, v2); },
            // 使用 lambda 表达式检查 v0 是否小于等于 v1
            [](const vec& v0, const vec& v1) { return v0 <= v1; },
            // 创建默认的二元测试用例，使用 TestSeed() 生成随机种子，确保真实测试运行
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(MinMax, Minimum) {
        // 使用 TypeParam 作为向量类型，并且使用 ValueType<TypeParam> 作为值类型
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // 调用 test_binary 函数进行二元测试
        test_binary<vec>(
            // 传递 NAME_INFO(minimum) 作为测试名称信息
            NAME_INFO(minimum),
            // 传递 minimum<VT> 函数作为比较函数
            minimum<VT>,
            // 使用 lambda 表达式调用 minimum(v0, v1) 检查最小值
            [](const vec& v0, const vec& v1) {
                return minimum(v0, v1);
            },
            // 创建默认的二元测试用例，使用 TestSeed() 生成随机种子，确保真实测试运行
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, Maximum) {
        // 使用 TypeParam 作为向量类型，并且使用 ValueType<TypeParam> 作为值类型
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // 调用 test_binary 函数进行二元测试
        test_binary<vec>(
            // 传递 NAME_INFO(maximum) 作为测试名称信息
            NAME_INFO(maximum),
            // 传递 maximum<VT> 函数作为比较函数
            maximum<VT>,
            // 使用 lambda 表达式调用 maximum(v0, v1) 检查最大值
            [](const vec& v0, const vec& v1) {
                return maximum(v0, v1);
            },
            // 创建默认的二元测试用例，使用 TestSeed() 生成随机种子，确保真实测试运行
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    // 定义测试函数模板 MinMax::ClampMin，TypeParam 是测试向量类型，ValueType<TypeParam> 是向量元素类型
    TYPED_TEST(MinMax, ClampMin) {
        using vec = TypeParam;  // 使用 vec 作为 TypeParam 的别名
        using VT = ValueType<TypeParam>;  // 使用 VT 作为 TypeParam 的值类型的别名
        // 调用 test_binary 函数，测试 clamp_min 函数
        test_binary<vec>(
            NAME_INFO(clamp min),  // 测试名称信息
            clamp_min<VT>,  // clamp_min 函数作为测试函数
            [](const vec& v0, const vec& v1) {  // Lambda 表达式，调用 clamp_min(v0, v1)
                return clamp_min(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));  // 创建默认的二元测试用例，并执行
    }
    
    // 定义测试函数模板 MinMax::ClampMax，TypeParam 是测试向量类型，ValueType<TypeParam> 是向量元素类型
    TYPED_TEST(MinMax, ClampMax) {
        using vec = TypeParam;  // 使用 vec 作为 TypeParam 的别名
        using VT = ValueType<TypeParam>;  // 使用 VT 作为 TypeParam 的值类型的别名
        // 调用 test_binary 函数，测试 clamp_max 函数
        test_binary<vec>(
            NAME_INFO(clamp max),  // 测试名称信息
            clamp_max<VT>,  // clamp_max 函数作为测试函数
            [](const vec& v0, const vec& v1) {  // Lambda 表达式，调用 clamp_max(v0, v1)
                return clamp_max(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));  // 创建默认的二元测试用例，并执行
    }
    
    // 定义测试函数模板 MinMax::Clamp，TypeParam 是测试向量类型，ValueType<TypeParam> 是向量元素类型
    TYPED_TEST(MinMax, Clamp) {
        using vec = TypeParam;  // 使用 vec 作为 TypeParam 的别名
        using VT = ValueType<TypeParam>;  // 使用 VT 作为 TypeParam 的值类型的别名
        // 调用 test_ternary 函数，测试 clamp 函数
        test_ternary<vec>(
            NAME_INFO(clamp),  // 测试名称信息
            clamp<VT>,  // clamp 函数作为测试函数
            [](const vec& v0, const vec& v1, const vec& v2) {  // Lambda 表达式，调用 clamp(v0, v1, v2)
                return clamp(v0, v1, v2);
            },
            createDefaultTernaryTestCase<vec>(TestSeed()),  // 创建默认的三元测试用例，并执行
            RESOLVE_OVERLOAD(filter_clamp));  // 使用 RESOLVE_OVERLOAD 宏解析 clamp 的重载
    }
    
    // 定义测试函数模板 BitwiseFloatsAdditional::ZeroMask，TypeParam 是测试向量类型，ValueType<TypeParam> 是向量元素类型
    TYPED_TEST(BitwiseFloatsAdditional, ZeroMask) {
        using vec = TypeParam;  // 使用 vec 作为 TypeParam 的别名
        using VT = ValueType<TypeParam>;  // 使用 VT 作为 TypeParam 的值类型的别名
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT test_vals[vec::size()];  // 声明与 vec::size() 大小相等的缓存对齐的 test_vals 数组
        // 计算 power_sets，表示可能的测试集数量
        auto power_sets = 1 << (vec::size());
        // 遍历 expected，生成测试值
        for (const auto expected : c10::irange(power_sets)) {
            // 根据 expected 生成 test_vals
            for (int i = 0; i < vec::size(); ++i)
            {
                if (expected & (1 << i)) {
                    test_vals[i] = (VT)0;  // 如果 expected 中第 i 位为 1，则设置 test_vals[i] 为 0
                }
                else {
                    test_vals[i] = (VT)0.897;  // 如果 expected 中第 i 位为 0，则设置 test_vals[i] 为 0.897
                }
            }
            // 调用 vec::loadu 加载 test_vals，并计算零掩码
            int actual = vec::loadu(test_vals).zero_mask();
            // 断言，判断预期与实际的零掩码是否相等，如果不等则输出详细的失败信息
            ASSERT_EQ(expected, actual) << "Failure Details:\n"
                << std::hex << "Expected:\n#\t" << expected
                << "\nActual:\n#\t" << actual;
        }
    }
    TYPED_TEST(BitwiseFloatsAdditional, Convert) {
        using vec = TypeParam;  // 使用模板参数 TypeParam 定义别名 vec
        using VT = ValueType<TypeParam>;  // 使用 TypeParam 实例化的模板 ValueType 的类型别名 VT
        using IntVT = at::vec::int_same_size_t<VT>;  // 使用 ValueType<TypeParam> 实例化 at::vec::int_same_size_t 的类型别名 IntVT

        // 验证浮点数到整数的转换
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT input1[vec::size()];  // 声明大小为 vec::size() 的缓存对齐的 VT 类型数组 input1
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT expected_vals1[vec::size()];  // 声明大小为 vec::size() 的缓存对齐的 IntVT 类型数组 expected_vals1
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT actual_vals1[vec::size()];  // 声明大小为 vec::size() 的缓存对齐的 IntVT 类型数组 actual_vals1
        for (int64_t i = 0; i < vec::size(); i++) {
            input1[i] = (VT)i * (VT)2.1 + (VT)0.5;  // 对 input1 数组进行赋值，将浮点数转换为整数
            expected_vals1[i] = static_cast<IntVT>(input1[i]);  // 将 input1 数组中的值转换为 IntVT 类型，并赋给 expected_vals1 数组
        }
        at::vec::convert(input1, actual_vals1, vec::size());  // 调用 convert 函数将 input1 数组中的值转换为 actual_vals1 数组中的整数类型
        auto expected1 = VecType<IntVT>::loadu(expected_vals1);  // 从 expected_vals1 数组加载数据并存入 expected1 变量
        auto actual1 = VecType<IntVT>::loadu(actual_vals1);  // 从 actual_vals1 数组加载数据并存入 actual1 变量
        if (AssertVectorized<VecType<IntVT>>(NAME_INFO(test_convert_to_int), expected1, actual1).check()) {  // 使用 AssertVectorized 检查预期和实际值是否相等
          return;  // 如果相等则返回
        }

        // 验证整数到浮点数的转换
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT input2[vec::size()];  // 声明大小为 vec::size() 的缓存对齐的 IntVT 类型数组 input2
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_vals2[vec::size()];  // 声明大小为 vec::size() 的缓存对齐的 VT 类型数组 expected_vals2
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT actual_vals2[vec::size()];  // 声明大小为 vec::size() 的缓存对齐的 VT 类型数组 actual_vals2
        for (int64_t i = 0; i < vec::size(); i++) {
            input2[i] = (IntVT)i * (IntVT)2 + (IntVT)1;  // 对 input2 数组进行赋值，将整数转换为浮点数
            expected_vals2[i] = (VT)input2[i];  // 将 input2 数组中的值直接赋给 expected_vals2 数组
        }
        at::vec::convert(input2, actual_vals2, vec::size());  // 调用 convert 函数将 input2 数组中的值转换为 actual_vals2 数组中的浮点数类型
        auto expected2 = vec::loadu(expected_vals2);  // 从 expected_vals2 数组加载数据并存入 expected2 变量
        auto actual2 = vec::loadu(actual_vals2);  // 从 actual_vals2 数组加载数据并存入 actual2 变量
        AssertVectorized<vec>(NAME_INFO(test_convert_to_float), expected2, actual2).check();  // 使用 AssertVectorized 检查预期和实际值是否相等
    }
    TYPED_TEST(BitwiseFloatsAdditional, Fmadd) {
        using vec = TypeParam;  // 使用模板参数 TypeParam 定义别名 vec
        using VT = ValueType<TypeParam>;  // 使用 TypeParam 实例化的模板 ValueType 的类型别名 VT

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());  // 设置测试用例

        test_ternary<vec>(
            NAME_INFO(clamp), RESOLVE_OVERLOAD(local_fmadd),
            [](const vec& v0, const vec& v1, const vec& v2) {
                return at::vec::fmadd(v0, v1, v2);  // 调用 fmadd 函数进行向量运算
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));  // 解析过载函数 filter_fmadd
    }
    template<typename vec, typename VT, int64_t mask>
    typename std::enable_if_t<(mask < 0 || mask> 255), void>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()])
    {
    }
    template<typename vec, typename VT, int64_t mask>
    // 假设mask在0到255之间，使用SFINAE技术使这个函数仅在满足条件时可用
    typename std::enable_if_t<(mask >= 0 && mask <= 255), void>
    // 禁止clang-tidy对下一行进行lint检查，理由是避免使用C风格数组
    test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()]) {
        // 生成期望值expected_val
        int64_t m = mask;
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = (m & 0x01) ? b[i] : a[i];
            m = m >> 1;
        }
        // 使用blend进行测试
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::template blend<mask>(vec_a, vec_b);
        auto mask_str = std::string("\nblend mask: ") + std::to_string(mask);
        // 使用AssertVectorized检查向量化操作的正确性，如果通过则返回
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_blend)) + mask_str, expected, actual).check()) return;
        // 递归调用test_blend，减少mask的值
        test_blend<vec, VT, mask - 1>(expected_val, a, b);
    }
    
    template<typename vec, typename VT, int64_t idx, int64_t N>
    // 禁止clang-tidy对下一行进行lint检查，理由是避免使用C风格数组
    std::enable_if_t<(!is_complex<VT>::value && idx == N), bool>
    // 使用blendv函数进行向量混合操作，只有在不是复数且idx等于N时可用
    test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
        using bit_rep = BitType<VT>;
        // 生成期望值expected_val
        for (int64_t i = 0; i < vec::size(); i++) {
            bit_rep hex_mask = 0;
            hex_mask = c10::bit_cast<bit_rep>(mask[i]);
            expected_val[i] = (hex_mask & 0x01) ? b[i] : a[i];
        }
        // 使用blendv进行测试
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto vec_m = vec::loadu(mask);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::blendv(vec_a, vec_b, vec_m);
        auto mask_str = std::string("\nblendv mask: ");
        for (int64_t i = 0; i < vec::size(); i++) {
            mask_str += std::to_string(mask[i]) + " ";
        }
        // 使用AssertVectorized检查向量化操作的正确性，如果通过则返回false
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_blendv)) + mask_str, expected, actual).check()) {
            return false;
        }
        // 返回true表示测试通过
        return true;
    }
    
    template<typename vec, typename VT, int64_t idx, int64_t N>
    // 禁止clang-tidy对下一行进行lint检查，理由是避免使用C风格数组
    std::enable_if_t<(!is_complex<VT>::value && idx != N), bool>
    // 使用blendv函数进行向量混合操作，只有在不是复数且idx不等于N时可用
    test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
        // 混洗mask并进行blendv测试
        VT m = mask[idx];
        if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
        if (m != (VT)0) {
            mask[idx] = (VT)0;
        } else {
            int64_t hex_mask = 0xFFFFFFFFFFFFFFFF;
            std::memcpy(&mask[idx], &hex_mask, sizeof(VT));
        }
        if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
        mask[idx] = m;
        // 返回true表示测试通过
        return true;
    }
    // 初始化混合操作的起始值数组a和b
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    void blend_init(T(&a)[N], T(&b)[N]) {
        // 设置数组a的第一个元素为1.0
        a[0] = (T)1.0;
        // 设置数组b的第一个元素为a[0] + N
        b[0] = a[0] + (T)N;
        // 遍历数组，初始化其余元素
        for (const auto i : c10::irange(1, N)) {
            // 数组a的当前元素为前一元素加1.0
            a[i] = a[i - 1] + (T)(1.0);
            // 数组b的当前元素为前一元素加1.0
            b[i] = b[i - 1] + (T)(1.0);
        }
    }
    // 测试混合操作的辅助功能，使用模板进行类型参数化
    TYPED_TEST(BitwiseFloatsAdditional, Blendv) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT a[vec::size()];  // 对齐缓存，用于存储数组a
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT b[vec::size()];  // 对齐缓存，用于存储数组b
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT mask[vec::size()] = {0};  // 对齐缓存，初始化mask数组为0
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];  // 对齐缓存，用于存储期望的值
        blend_init(a, b);  // 调用初始化函数blend_init初始化数组a和b
        test_blendv<vec, VT, 0, vec::size()>(expected_val, a, b, mask);  // 执行混合测试
    }
    // 测试混合操作的第二个辅助功能，使用模板进行类型参数化
    TYPED_TEST(BitwiseFloatsAdditional2, Blend) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT a[vec::size()];  // 对齐缓存，用于存储数组a
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT b[vec::size()];  // 对齐缓存，用于存储数组b
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];  // 对齐缓存，用于存储期望的值
        blend_init(a, b);  // 调用初始化函数blend_init初始化数组a和b
        constexpr int64_t power_sets = 1LL << (vec::size());
        test_blend<vec, VT, power_sets - 1>(expected_val, a, b);  // 执行混合测试
    }
    // 用于设置测试的模板函数，将预期值expected_val与数组a和b进行比较
    template<typename vec, typename VT>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    void test_set(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], int64_t count){
        if (count < 0) return;  // 如果count小于0，则直接返回
        // 生成期望的值数组expected_val
        for (int64_t i = 0; i < vec::size(); i++) {
            // 如果i小于count，则expected_val[i]为b[i]，否则为a[i]
            expected_val[i] = (i < count) ? b[i] : a[i];
        }
        // 使用set函数进行测试
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::set(vec_a, vec_b, count);

        auto count_str = std::string("\ncount: ") + std::to_string(count);
        // 使用AssertVectorized类检查向量化操作的正确性
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_set)) + count_str, expected, actual).check()) {
          return;
        }
        // 递归调用test_set函数，缩小count的范围（如果count为0，则设为-1）
        test_set<vec, VT>(expected_val, a, b, (count == 0 ? -1 : count / 2));
    }
    TYPED_TEST(BitwiseFloatsAdditional2, Set) {
        // 使用 TypeParam 和 ValueType<TypeParam> 类型别名来定义 vec 和 VT
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // 定义一个缓存对齐的数组 a，长度为 vec::size()
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT a[vec::size()];
        // 定义一个缓存对齐的数组 b，长度为 vec::size()
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT b[vec::size()];
        // 定义一个缓存对齐的数组 expected_val，长度为 vec::size()
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];
        // 调用 blend_init 函数，初始化数组 a 和 b
        blend_init(a, b);
        // 调用 test_set 函数，进行测试设置，传入 expected_val、a、b 和 vec::size()
        test_set<vec, VT>(expected_val, a, b, vec::size());
    }

    template<typename T>
    // 如果 T 不是复数类型，则启用此函数
    std::enable_if_t<!is_complex<T>::value, void>
    arange_init(T& base, T& step) {
        // 设置 base 为 5.0
        base = (T)5.0;
        // 设置 step 为 2.0
        step = (T)2.0;
    }

    template<typename T>
    // 如果 T 是复数类型，则启用此函数
    std::enable_if_t<is_complex<T>::value, void>
    arange_init(T& base, T& step) {
       // 设置 base 为复数 (5.0, 5.0)
       base = T(5.0, 5.0);
       // 设置 step 为复数 (2.0, 3.0)
       step = T(2.0, 3.0);
    }

    TYPED_TEST(RangeFactories, Arange) {
        // 使用 TypeParam 和 ValueType<TypeParam> 类型别名来定义 vec、VT 和 UVT
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        using UVT = UvalueType<TypeParam>;
        // 定义一个缓存对齐的数组 expected_val，长度为 vec::size()
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];
        // 定义变量 base 和 step，它们的类型为 VT
        VT base, step;
        // 调用 arange_init 函数，初始化 base 和 step
        arange_init(base, step);
        // 循环迭代，计算并填充 expected_val 数组
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = base + VT((UVT)i) * step;
        }
        // 将 expected_val 载入到向量化类型 vec 中
        auto expected = vec::loadu(expected_val);
        // 调用 vec::arange 函数，生成实际结果 actual
        auto actual = vec::arange(base, step);
        // 断言向量化操作的结果，检查是否符合预期
        AssertVectorized<vec>(NAME_INFO(test_arange), expected, actual).check();
    }
    TEST(ComplexTests, TestComplexFloatImagRealConj) {
        // 定义一个浮点数数组，表示复数的实部和虚部交替出现
        float aa[] = { 1.5488e-28, 2.5488e-28, 3.5488e-28, 4.5488e-28, 5.5488e-28, 6.5488e-28, 7.5488e-28, 8.5488e-28,
                       9.5488e-28, 10.5488e-28, 11.5488e-28, 12.5488e-28, 13.5488e-28, 14.5488e-28, 15.5488e-28, 16.5488e-28 };
        // 定义期望的复数数组，虚部为0
        float exp[] = { aa[0], 0, aa[2], 0, aa[4], 0, aa[6], 0, aa[8], 0, aa[10], 0, aa[12], 0, aa[14], 0 };
        // 定义另一组期望的复数数组，虚部为0
        float exp3[] = { aa[1], 0, aa[3], 0, aa[5], 0, aa[7], 0, aa[9], 0, aa[11], 0, aa[13], 0, aa[15], 0 };
        // 定义复数的共轭期望数组
        float exp4[] = { 1.5488e-28, -2.5488e-28, 3.5488e-28, -4.5488e-28,
                         5.5488e-28, -6.5488e-28, 7.5488e-28, -8.5488e-28,
                         9.5488e-28, -10.5488e-28, 11.5488e-28, -12.5488e-28,
                         13.5488e-28, -14.5488e-28, 15.5488e-28, -16.5488e-28 };
        
        // 载入 aa 数组到 vcomplex 类型
        auto a = vcomplex::loadu(aa);
        // 计算载入复数的实部
        auto actual1 = a.real();
        // 计算载入复数的虚部
        auto actual3 = a.imag();
        // 计算载入复数的共轭
        auto actual4 = a.conj();
        // 根据实际值和期望值进行断言检查，检查实部是否正确
        AssertVectorized<vcomplex>(NAME_INFO(complex real), expected1, actual1).check();
        // 根据实际值和期望值进行断言检查，检查虚部是否正确
        AssertVectorized<vcomplex>(NAME_INFO(complex imag), expected3, actual3).check();
        // 根据实际值和期望值进行断言检查，检查共轭是否正确
        AssertVectorized<vcomplex>(NAME_INFO(complex conj), expected4, actual4).check();
    }
    
    TEST(ComplexTests, TestComplexConstructor) {
        // 调用 vcomplex 的构造函数，创建实际值
        auto actual1 = vcomplex(1.0);
        // 调用 Complex<float> 的构造函数，创建期望值
        auto expected1 = vcomplex(Complex<float>(1.0));
        // 根据实际值和期望值进行断言检查，检查构造函数是否正确
        AssertVectorized<vcomplex>(NAME_INFO(complex constructor), expected1, actual1).check();
    }
    TYPED_TEST(QuantizationTests, Quantize) {
        // 使用类型 TypeParam 作为向量类型
        using vec = TypeParam;
        // 使用 underlying 作为 vec 中元素的基础类型
        using underlying = ValueType<vec>;
        // 设定测试循环次数
        constexpr int trials = 4000;
        
        // 定义基础类型的最小值和最大值
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        constexpr int min_val = std::numeric_limits<underlying>::min();
        constexpr int max_val = std::numeric_limits<underlying>::max();
        
        // 获取向量中元素的数量
        constexpr int el_count = vfloat::size();
        
        // 定义缓存对齐的浮点数向量数组
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN float unit_float_vec[el_count];
        
        // 定义缓存对齐的基础类型数组
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN underlying expected_qint_vals[vec::size()];
        
        // 定义向量的浮点数返回类型
        typename vec::float_vec_return_type float_ret;
        
        // 生成测试用的随机种子
        auto seed = TestSeed();
        
        // 定义用于生成零点值的整数值生成器
        ValueGen<int> generator_zp(min_val, max_val, seed);
        
        // 定义用于生成缩放因子的浮点数值生成器
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
        
        // 定义用于生成测试值的浮点数值生成器
        float minv = static_cast<float>(static_cast<double>(min_val) * 2.0);
        float maxv = static_cast<float>(static_cast<double>(max_val) * 2.0);
        ValueGen<float> gen(minv, maxv, seed.add(2));
        
        // 开始执行 trials 次测试循环
        for (C10_UNUSED const auto i : c10::irange(trials)) {
            // 获取当前循环的缩放因子和其倒数
            float scale = generator_sc.get();
            float inv_scale = 1.0f / static_cast<float>(scale);
            
            // 获取当前循环的零点值
            auto zero_point_val = generator_zp.get();
            
            // 初始化索引值
            int index = 0;
            
            // 遍历 vec 中的每一个浮点数向量
            for (int j = 0; j < vec::float_num_vecs(); j++) {
                // 生成浮点数值填充 unit_float_vec
                for (auto& v : unit_float_vec) {
                    v = gen.get();
                    // 使用 quantize_val 函数量化当前值，并存储到 expected_qint_vals 中
                    expected_qint_vals[index] = quantize_val<underlying>(scale, zero_point_val, v);
                    index++;
                }
                // 将填充好的 unit_float_vec 转换为 float_ret 的第 j 个元素
                float_ret[j] = vfloat::loadu(unit_float_vec);
            }
            
            // 使用 vec::loadu 加载期望的量化整数值
            auto expected = vec::loadu(expected_qint_vals);
            
            // 执行量化操作，得到实际的量化整数值
            auto actual = vec::quantize(float_ret, scale, zero_point_val, inv_scale);
            
            // 使用 AssertVectorized 检查预期值和实际值是否一致，如果一致则返回
            if (AssertVectorized<vec>(NAME_INFO(Quantize), expected, actual).check()) return;
        } //trials;
    }
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && !defined(_MSC_VER)
    // 如果定义了 CPU_CAPABILITY_AVX2 或 CPU_CAPABILITY_AVX512，并且未定义 _MSC_VER
    // 此测试用例旨在测试 at::vec::QuantizeAvx512 和 at::vec::QuantizeAVX2，
    // 它们不支持 CPU_CAPABILITY_DEFAULT 情况
    
    TYPED_TEST(Quantization8BitWithTailTests, QuantizeTile) {
      // 使用的类型和向量类型参数
      using vec = TypeParam;
      // 底层数据类型
      using underlying = ValueType<vec>;
      // 固定的试验次数
      constexpr int trials = 4000;
      
      // 最小值和最大值
      // NOLINTNEXTLINE(bugprone-signed-char-misuse)
      constexpr int min_val = std::numeric_limits<underlying>::min();
      constexpr int max_val = std::numeric_limits<underlying>::max();
      
      // 向量中元素的数量
      constexpr int el_count = vfloat::size();
      
      // 缓存对齐的浮点数向量
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      CACHE_ALIGN float unit_float_vec[el_count];
      
      // 缓存对齐的期望的 qint 值数组
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      CACHE_ALIGN underlying expected_qint_vals[vec::size()];
      
      // 缓存对齐的实际 qint 值数组
      CACHE_ALIGN underlying actual_qint_vals[vec::size()];
      
      // 瓦片大小，为向量大小减一
      constexpr int tile_size = vec::size() - 1;
      
      // 向量浮点返回类型
      typename vec::float_vec_return_type float_ret;
      
      // 测试的种子
      auto seed = TestSeed();
      
      // 零点生成器
      ValueGen<int> generator_zp(min_val, max_val, seed);
      
      // 缩放因子生成器
      ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
      
      // 值生成器，用于生成浮点数
      float minv = static_cast<float>(static_cast<double>(min_val) * 2.0);
      float maxv = static_cast<float>(static_cast<double>(max_val) * 2.0);
      ValueGen<float> gen(minv, maxv, seed.add(2));
      
      // 对于 trials 次数范围内的循环
      for (C10_UNUSED const auto i : c10::irange(trials)) {
        // 获取缩放因子和其倒数
        float scale = generator_sc.get();
        float inv_scale = 1.0f / static_cast<float>(scale);
        
        // 获取零点值
        auto zero_point_val = generator_zp.get();
        int index = 0;
        
        // 对于每个向量浮点数的向量数量
        for (int j = 0; j < vec::float_num_vecs(); j++) {
          // 生成浮点数值
          for (auto& v : unit_float_vec) {
            v = gen.get();
            // 量化为 qint 值，并存储到期望 qint 数组中
            expected_qint_vals[index] =
                quantize_val<underlying>(scale, zero_point_val, v);
            index++;
          }
          // 将浮点数向量加载到 float_ret 中
          float_ret[j] = vfloat::loadu(unit_float_vec);
        }
        
        // 根据定义的 CPU_CAPABILITY_AVX512 或 CPU_CAPABILITY_AVX2 调用相应的量化函数
#if defined(CPU_CAPABILITY_AVX512)
        at::vec::QuantizeAvx512(
            (float*)float_ret.data(),
            actual_qint_vals,
            tile_size,
            inv_scale,
            zero_point_val);
#endif
#if defined(CPU_CAPABILITY_AVX2)
        at::vec::QuantizeAvx2(
            (float*)float_ret.data(),
            actual_qint_vals,
            tile_size,
            inv_scale,
            zero_point_val);
#endif
        
        // 将最后一个元素置为零
        expected_qint_vals[tile_size] = 0;
        actual_qint_vals[tile_size] = 0;
        
        // 加载期望值和实际值，并比较
        auto expected = vec::loadu(expected_qint_vals);
        auto actual = vec::loadu(actual_qint_vals);
        
        // 如果比较通过，直接返回
        if (AssertVectorized<vec>(NAME_INFO(QuantizeTile), expected, actual)
                .check())
          return;
      } // trials;
    }
#endif
    # 定义一个类型化的测试，用于反量化操作
    TYPED_TEST(QuantizationTests, DeQuantize) {
        # 使用模板参数 TypeParam 作为向量类型的别名
        using vec = TypeParam;
        # 使用模板参数 ValueType<vec> 作为向量元素类型的别名
        using underlying = ValueType<vec>;
        # 根据 underlying 类型是否大于一个字节来确定是否为大型数据类型
        constexpr bool is_large = sizeof(underlying) > 1;
        # 根据数据类型大小选择不同的试验次数
        constexpr int trials = is_large ? 4000 : std::numeric_limits<underlying>::max() / 2;
        # 根据数据类型大小选择不同的最小值
        constexpr int min_val = is_large ? -2190 : std::numeric_limits<underlying>::min();
        # 根据数据类型大小选择不同的最大值
        constexpr int max_val = is_large ? 2199 : std::numeric_limits<underlying>::max();
        # 使用 CACHE_ALIGN 宏定义一个与特定向量大小相等的浮点数数组，用于存储单位指数值
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN float unit_exp_vals[vfloat::size()];
        # 使用 CACHE_ALIGN 宏定义一个与特定向量大小相等的 underlying 数组，用于存储量化整数值
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN underlying qint_vals[vec::size()];
#if  defined(CHECK_DEQUANT_WITH_LOW_PRECISION)
        // 如果定义了宏 CHECK_DEQUANT_WITH_LOW_PRECISION，则输出相对误差为 1.e-3f 的信息
        std::cout << "Dequant will be tested with relative error " << 1.e-3f << std::endl;
#endif
        // 生成测试用的随机种子
        auto seed = TestSeed();
        // 生成整数值的随机数生成器，范围在 min_val 到 max_val 之间，种子为 seed.add(1)
        ValueGen<int> generator(min_val, max_val, seed.add(1));
        // 生成浮点数值的随机数生成器，范围在 1.f 到 15.f 之间，种子为 seed.add(2)
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(2));
        // 对 trials 次数循环
        for (C10_UNUSED const auto i : c10::irange(trials)) {
            // 获取当前浮点数生成器的值作为 scale
            float scale = generator_sc.get();
            // 获取当前整数生成器的值作为 zero_point_val
            int32_t zero_point_val = generator.get();
            // 计算 scale_zp_premul 值，即 -(scale * zero_point_val)
            float scale_zp_premul = -(scale * zero_point_val);
            // 将 scale 转换为向量化浮点数 vf_scale
            vfloat vf_scale = vfloat{ scale };
            // 将 zero_point_val 转换为向量化浮点数 vf_zp
            vfloat vf_zp = vfloat{ static_cast<float>(zero_point_val) };
            // 将 scale_zp_premul 转换为向量化浮点数 vf_scale_zp
            vfloat vf_scale_zp = vfloat{ scale_zp_premul };
            // 生成 qint_vals 数组的值
            for (auto& x : qint_vals) {
                x = generator.get();
            }
            // 计算预期的浮点数结果
            int index = 0;
            auto qint_vec = vec::loadu(qint_vals);
            auto actual_float_ret = qint_vec.dequantize(vf_scale, vf_zp, vf_scale_zp);
            // 对于每个向量化浮点数进行检查
            for (int j = 0; j < vec::float_num_vecs(); j++) {
                // 生成单元表达式的期望值
                for (auto& v : unit_exp_vals) {
                    v = dequantize_val(scale, zero_point_val, qint_vals[index]);
                    index++;
                }
                // 将 unit_exp_vals 转换为向量化浮点数 expected
                vfloat expected = vfloat::loadu(unit_exp_vals);
                // 比较 actual 和 expected 的结果
                const auto& actual = actual_float_ret[j];
#if  defined(CHECK_DEQUANT_WITH_LOW_PRECISION)
                // 如果定义了宏 CHECK_DEQUANT_WITH_LOW_PRECISION，则进行相对误差检查
                if (AssertVectorized<vfloat>(NAME_INFO(DeQuantize), seed, expected, actual).check(false, true, 1.e-3f)) return;
#else
                // 否则，进行默认的向量化值检查
                if (AssertVectorized<vfloat>(NAME_INFO(DeQuantize), seed, expected, actual).check()) return;
#endif
            }
        } //trials;
    }
    // 定义测试函数 QuantizationTests::ReQuantizeFromInt，使用 TypeParam 作为向量类型，ValueType<vec> 作为向量元素类型
    TYPED_TEST(QuantizationTests, ReQuantizeFromInt) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr int trials = 4000;  // 定义循环试验次数
        constexpr int min_val = -65535;  // 定义整数范围下限
        constexpr int max_val = 65535;   // 定义整数范围上限
        constexpr int el_count = vint::size();  // 获取向量中元素的数量

        // 创建 CACHE_ALIGN 类型的 qint32 数组 unit_int_vec，用于存储整数向量
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN c10::qint32 unit_int_vec[el_count];

        // 创建 CACHE_ALIGN 类型的 underlying 数组 expected_qint_vals，用于存储期望的 qint 值
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN underlying expected_qint_vals[vec::size()];

        // 定义 vec::int_vec_return_type 类型的 int_ret，用于存储整数向量返回类型
        typename vec::int_vec_return_type int_ret;

        // 生成随机种子 TestSeed
        auto seed = TestSeed();

        // 创建整数值生成器 generator，生成器范围在 min_val 到 max_val 之间
        ValueGen<int32_t> generator(min_val, max_val, seed);

        // 创建浮点数值生成器 generator_sc，生成器范围在 1.f 到 15.f 之间
        // 使用 seed.add(1) 作为新种子
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));

        // 执行 trials 次循环
        for (C10_UNUSED const auto i : c10::irange(trials)) {
            // 计算 multiplier 为 1 除以 generator_sc.get() 的值，用于缩放
            float multiplier = 1.f / (generator_sc.get());

            // 获取 zero_point_val，即 generator 生成的随机整数值
            auto zero_point_val = generator.get();

            // 初始化 index 为 0，用于索引 expected_qint_vals 数组
            int index = 0;

            // 遍历 vec::float_num_vecs() 次，通常是处理浮点数向量的次数
            for (int j = 0; j < vec::float_num_vecs(); j++) {
                // 为 unit_int_vec 数组生成值
                for (auto& v : unit_int_vec) {
                    // 使用 generator 生成 qint32 值赋给 v
                    v = c10::qint32(generator.get());

                    // 根据 multiplier、zero_point_val、v.val_ 调用 requantize_from_int 函数，计算并存储结果
                    expected_qint_vals[index] = requantize_from_int<underlying>(multiplier, zero_point_val, v.val_);
                    index++;
                }

                // 将 unit_int_vec 中的值加载到 int_ret[j] 中
                int_ret[j] = vqint::loadu(unit_int_vec);
            }

            // 将 expected_qint_vals 中的值加载到 expected 向量中
            auto expected = vec::loadu(expected_qint_vals);

            // 调用 vec::requantize_from_int 函数，计算实际结果并加载到 actual 向量中
            auto actual = vec::requantize_from_int(int_ret, multiplier, zero_point_val);

            // 检查期望值 expected 和实际值 actual 是否匹配，如果匹配则返回
            if (AssertVectorized<vec>(NAME_INFO(ReQuantizeFromInt), seed, expected, actual).check()) {
                return;
            }
        } //trials;
    }
    // 定义一个模板测试类型，测试宽化减法
    TYPED_TEST(QuantizationTests, WideningSubtract) {
        // 使用 TypeParam 定义一个向量类型 vec
        using vec = TypeParam;
        // 使用 ValueType 获取向量元素的基础类型 underlying
        using underlying = ValueType<vec>;
        // 检查 underlying 是否大于一个字节，确定 trials 的值
        constexpr bool is_large = sizeof(underlying) > 1;
        constexpr int trials = is_large ? 4000 : std::numeric_limits<underlying>::max() / 2;
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        // 定义 underlying 的最小值和最大值
        constexpr int min_val = std::numeric_limits<underlying>::min();
        constexpr int max_val = std::numeric_limits<underlying>::max();
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        // 使用 CACHE_ALIGN 定义一个与 SIMD 对齐的 int32_t 数组
        CACHE_ALIGN int32_t unit_exp_vals[vfloat::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        // 使用 CACHE_ALIGN 定义一个与 SIMD 对齐的 underlying 类型数组 qint_vals
        CACHE_ALIGN underlying qint_vals[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        // 使用 CACHE_ALIGN 定义一个与 SIMD 对齐的 underlying 类型数组 qint_b
        CACHE_ALIGN underlying qint_b[vec::size()];
        // 定义 vec 的 int_vec_return_type 类型的 expected_int_ret 变量
        typename vec::int_vec_return_type  expected_int_ret;
        // 创建一个随机种子 seed
        auto seed = TestSeed();
        // 使用 ValueGen 类生成在 [min_val, max_val] 范围内的 underlying 类型随机数
        ValueGen<underlying> generator(min_val, max_val, seed);
        // 开始 trials 次循环测试
        for (C10_UNUSED const auto i : c10::irange(trials)) {
            // 生成 qint_vals 和 qint_b 数组的随机数值
            for (int j = 0; j < vec::size(); j++) {
                qint_vals[j] = generator.get();
                qint_b[j] = generator.get();
                // 如果 underlying 类型为 int，则过滤溢出情况
                if constexpr (std::is_same_v<underlying, int>) {
                    // 调用 filter_sub_overflow 函数，过滤掉溢出的情况
                    filter_sub_overflow(qint_vals[j], qint_b[j]);
                }
            }
            // 初始化索引为 0
            int index = 0;
            // 从 qint_vals 和 qint_b 数组加载向量 qint_vec 和 qint_vec_b
            auto qint_vec = vec::loadu(qint_vals);
            auto qint_vec_b = vec::loadu(qint_b);
            // 调用 qint_vec 的 widening_subtract 方法，计算向量减法
            auto actual_int_ret = qint_vec.widening_subtract(qint_vec_b);
            // 遍历处理每个向量单元
            for (int j = 0; j < vec::float_num_vecs(); j++) {
                // 为 unit_exp_vals 数组赋值，执行 widening_subtract 函数
                for (auto& v : unit_exp_vals) {
                    v = widening_subtract(qint_vals[index], qint_b[index]);
                    index++;
                }
                // 从 unit_exp_vals 加载为 vqint 类型的 expected 向量
                auto expected = vqint::loadu(unit_exp_vals);
                // 获取 actual_int_ret[j] 的引用为 actual
                const auto& actual = actual_int_ret[j];
                // 如果 AssertVectorized<vqint> 检查通过，则返回
                if (AssertVectorized<vqint>(NAME_INFO(WideningSubtract), seed, expected, actual).check()) return;
            }
        } //trials;
    }
    // 定义一个模板测试类型，测试 Relu 函数
    TYPED_TEST(QuantizationTests, Relu) {
        // 使用 TypeParam 定义一个向量类型 vec
        using vec = TypeParam;
        // 使用 ValueType 获取向量元素的基础类型 VT
        using VT = ValueType<TypeParam>;
        // 定义 VT 类型的最小值和最大值
        constexpr VT min_val = std::numeric_limits<VT>::min();
        constexpr VT max_val = std::numeric_limits<VT>::max();
        // 根据 VT 类型大小不同定义 fake_zp 值
        constexpr VT fake_zp = sizeof(VT) > 1 ? static_cast<VT>(65535) : static_cast<VT>(47);
        // 创建一个 TestingCase 实例 test_case，设置测试域和测试种子
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<VT>{ { DomainRange<VT>{min_val, max_val}, DomainRange<VT>{(VT)0, (VT)fake_zp}} })
            .setTestSeed(TestSeed());
        // 调用 test_binary 函数，测试 relu 函数
        test_binary<vec>(
            NAME_INFO(relu),
            RESOLVE_OVERLOAD(relu),
            // lambda 表达式，调用 v0.relu(v1) 计算 relu 函数
            [](const vec& v0, const vec& v1) {
                return v0.relu(v1);
            },
            test_case);
    }
    TYPED_TEST(QuantizationTests, Relu6) {
        // 使用 TypeParam 定义 vec 和 VT 的类型别名
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        
        // 定义 VT 类型的最小和最大值
        constexpr VT min_val = std::numeric_limits<VT>::min();
        constexpr VT max_val = std::numeric_limits<VT>::max();
        
        // 根据 VT 的大小选择不同的虚假零点和临时值
        constexpr VT fake_zp = sizeof(VT) > 1 ? static_cast<VT>(65535) : static_cast<VT>(47);
        constexpr VT temp = sizeof(VT) > 1 ? static_cast<VT>(12345) : static_cast<VT>(32);
        
        // 计算虚假的量化 6 值
        constexpr VT fake_qsix = fake_zp + temp;
        
        // 创建测试用例对象
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<VT>{
                {
                    // 添加 VT 类型的域范围
                    DomainRange<VT>{min_val, max_val},
                    DomainRange<VT>{(VT)0, (VT)fake_zp},
                    DomainRange<VT>{(VT)fake_zp, (VT)fake_qsix}
                }})
            .setTestSeed(TestSeed());
        
        // 执行测试
        test_ternary<vec>(
            NAME_INFO(relu6), // 测试名称为 relu6
            RESOLVE_OVERLOAD(relu6), // 解析并执行 relu6 函数重载
            [](/*const*/ vec& v0, const vec& v1, const vec& v2) {
                return  v0.relu6(v1, v2); // 调用 vec 类型的 relu6 方法
            },
            test_case); // 使用前面定义的测试用例
    }

    TYPED_TEST(FunctionalTests, Map) {
        // 使用 TypeParam 定义 vec 和 VT 的类型别名
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        
        // 定义向量大小和余数
        constexpr auto R = 2LL; // residual
        constexpr auto N = vec::size() + R;
        
        // 声明缓存对齐的变量数组
        CACHE_ALIGN VT x1[N];
        CACHE_ALIGN VT x2[N];
        CACHE_ALIGN VT x3[N];
        CACHE_ALIGN VT x4[N];
        CACHE_ALIGN VT y[N];
        CACHE_ALIGN VT ref_y[N];
        
        // 初始化随机数种子
        auto seed = TestSeed();
        ValueGen<VT> generator(VT(-100), VT(100), seed);
        
        // 生成随机数填充数组
        for (const auto i : c10::irange(N)) {
            x1[i] = generator.get();
            x2[i] = generator.get();
            x3[i] = generator.get();
            x4[i] = generator.get();
        }
        
        // 定义比较函数
        auto cmp = [&](VT* y, VT* ref_y) {
            // 使用 AssertVectorized 类对向量进行检查
            AssertVectorized<vec>(NAME_INFO(Map), vec::loadu(y), vec::loadu(ref_y)).check(true);
            AssertVectorized<vec>(NAME_INFO(Map), vec::loadu(y + vec::size(), R), vec::loadu(ref_y + vec::size(), R)).check(true);
        };
        
        // 测试 map 函数：y = x1
        at::vec::map<VT>([](vec x) { return x; }, y, x1, N);
        for (const auto i : c10::irange(N)) { ref_y[i] = x1[i]; }
        cmp(y, ref_y);
        
        // 测试 map2 函数：y = x1 + x2
        at::vec::map2<VT>([](vec x1, vec x2) { return x1 + x2; }, y, x1, x2, N);
        for (const auto i : c10::irange(N)) { ref_y[i] = x1[i] + x2[i]; }
        cmp(y, ref_y);
        
        // 测试 map3 函数：y = x1 + x2 + x3
        at::vec::map3<VT>([](vec x1, vec x2, vec x3) { return x1 + x2 + x3; }, y, x1, x2, x3, N);
        for (const auto i : c10::irange(N)) { ref_y[i] = x1[i] + x2[i] + x3[i]; }
        cmp(y, ref_y);
        
        // 测试 map4 函数：y = x1 + x2 + x3 + x4
        at::vec::map4<VT>([](vec x1, vec x2, vec x3, vec x4) { return x1 + x2 + x3 + x4; }, y, x1, x2, x3, x4, N);
        for (const auto i : c10::irange(N)) { ref_y[i] = x1[i] + x2[i] + x3[i] + x4[i]; }
        cmp(y, ref_y);
    }
    TEST(HalfConversionTest, HalfFloat) {
      // 创建一个包含100个单精度浮点数的数组
      float f32s[100];
      // 使用范围循环填充数组，每个元素都是其索引加上0.3
      for (const auto i : c10::irange(100)) {
        f32s[i] = i + 0.3;
      }
      // 定义一个16位无符号整数和一个单精度浮点数变量
      uint16_t u16;
      float x;
      // 再次使用范围循环，进行后续处理
      for (const auto i : c10::irange(100)) {
      // 如果定义了CPU能力为AVX2或AVX512，并且不是苹果系统
      #if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
          !defined(__APPLE__)
        // 使用AVX指令集转换单精度浮点数到半精度浮点数
        u16 = at::vec::float2half_scalar(f32s[i]);
        // 将半精度浮点数转换回单精度浮点数
        x = at::vec::half2float_scalar(u16);
      // 否则使用通用的IEEE标准方法进行转换
      #else
        u16 = c10::detail::fp16_ieee_from_fp32_value(f32s[i]);
        x = c10::detail::fp16_ieee_to_fp32_value(u16);
      #endif

        // 断言转换的正确性，如果不正确输出错误信息
        EXPECT_EQ(u16, c10::detail::fp16_ieee_from_fp32_value(f32s[i]))
            << "Test failed for float to uint16 " << f32s[i] << "\n";
        EXPECT_EQ(x, c10::detail::fp16_ieee_to_fp32_value(u16))
            << "Test failed for uint16 to float " << u16 << "\n";
      }
    }

    TYPED_TEST(InfiniteTests, HasInfNan) {
      // 使用类型参数TypeParam和其持有类型UholdType
      using vec = TypeParam;
      using VT = UholdType<TypeParam>;
      // 获取向量的大小
      auto vec_size = vec::size();
      // 创建一个包含20个值的数组
      VT values[20];
      // 使用范围循环填充数组，每个元素都是其索引加上0.3
      for (const auto i : c10::irange(20)) {
        values[i] = i + 0.3;
      }
      // 从TypeParam创建一个向量vec_val
      auto vec_val = vec::loadu(values);
      // 创建一个用于测试的种子
      auto seed = TestSeed();
      // 创建一个整数生成器，并生成一个索引值
      ValueGen<int> generator(int(0), int(vec_size - 1), seed);
      int index = generator.get();
      // 定义NaN的位表示
      int nanBits = 0x7FC00000;
      // 将NaN的位表示转换为浮点数，然后转换为类型VT
      VT v_nan = static_cast<VT>(*(float *)&nanBits);
      // 将特定索引处的值设置为NaN
      values[index] = v_nan;
      // 从修改后的数组创建一个包含NaN值的向量vec_nan
      auto vec_nan = vec::loadu(values);
      // 定义正无穷大的位表示
      int infBits = 0x7F800000;
      // 将正无穷大的位表示转换为浮点数，然后转换为类型VT
      VT v_pinf = static_cast<VT>(*(float *)&infBits);
      // 将特定索引处的值设置为正无穷大
      values[index] = v_pinf;
      // 从修改后的数组创建一个包含正无穷大值的向量vec_pinf
      auto vec_pinf = vec::loadu(values);
      // 定义负无穷大的位表示
      int negInfBits = 0xFF800000;
      // 将负无穷大的位表示转换为浮点数，然后转换为类型VT
      VT v_ninf  = static_cast<VT>(*(float *)&negInfBits);
      // 将特定索引处的值设置为负无穷大
      values[index] = v_ninf;
      // 从修改后的数组创建一个包含负无穷大值的向量vec_ninf
      auto vec_ninf = vec::loadu(values);

      // 断言向量是否包含无穷大或NaN，如果不是则输出错误信息
      ASSERT_TRUE(!(vec_val.has_inf_nan())) << "Test failed for normal value\n";
      ASSERT_TRUE(vec_nan.has_inf_nan()) << "Test failed for NAN\n";
      ASSERT_TRUE(vec_pinf.has_inf_nan()) << "Test failed for positive Infinity\n";
      ASSERT_TRUE(vec_ninf.has_inf_nan()) << "Test failed for negative Infinity\n";
    }

    TYPED_TEST(VecConvertTests, Convert) {
      // 使用类型参数TypeParam
      using vec = TypeParam;
      // 使用TypeParam的值类型作为src_t
      using src_t = ValueType<TypeParam>;
      // 定义常量N，代表向量的大小
      constexpr auto N = vec::size();
    #define TEST_CONVERT_TO(dst_t)                                     \
      do {                                                             \
        // 定义缓存对齐的源类型数组和目标类型数组，每个数组大小为 N
        CACHE_ALIGN src_t x[N];                                        \
        CACHE_ALIGN dst_t y[N];                                        \
        CACHE_ALIGN dst_t ref[N];                                      \
        // 获取用于生成随机数的种子
        auto seed = TestSeed();                                        \
        // 根据目标类型确定低端值，如果目标类型为有符号，则低端值为 src_t(-100)，否则为 0
        auto low = std::is_signed_v<dst_t> ? src_t(-100) : 0;          \
        // 创建值生成器，生成范围在 [low, 100] 内的随机数
        ValueGen<src_t> generator(low, src_t(100), seed);              \
        // 填充源类型数组 x
        for (const auto i : c10::irange(N)) {                          \
          x[i] = generator.get();                                      \
        }                                                              \
        // 将源类型数组 x 转换为目标类型数组 ref
        for (const auto i : c10::irange(N)) {                          \
          ref[i] = static_cast<dst_t>(x[i]);                           \
        }                                                              \
        // 将源类型数组 x 转换为 SIMD 加速的向量类型 x_vec
        auto x_vec = vec::loadu(x);                                    \
        // 调用 SIMD 函数将 x_vec 中的数据转换为目标类型的 SIMD 向量 y_vec
        auto y_vec = at::vec::convert<dst_t>(x_vec);                   \
        // 计算实际转换的元素个数，不超过 SIMD 向量的最大容量
        constexpr int num_dst_elements =                               \
            std::min(N, at::vec::Vectorized<dst_t>::size());           \
        // 将 SIMD 向量 y_vec 中的数据存储回目标类型数组 y 中
        y_vec.store(y, num_dst_elements);                              \
        // 检查转换结果是否与参考数组 ref 相等，如果不相等则输出详细错误信息
        for (const auto i : c10::irange(num_dst_elements)) {           \
          ASSERT_EQ(y[i], ref[i])                                      \
              << "Failure Details:\nTest Seed to reproduce: " << seed  \
              << " x[" << i << "]=" << x[i] << " dst_t=" #dst_t;       \
        }                                                              \
        // 计算需要进行完整元素转换的次数
        constexpr int dst_n = N / num_dst_elements;                    \
        // 将源类型数组 x 通过 SIMD 加速转换为目标类型数组 y，处理全部 N 个元素
        auto y_vec_n = at::vec::convert<dst_t, dst_n, src_t, 1>(       \
            at::vec::VectorizedN<src_t, 1>(x_vec));                    \
        y_vec_n.store(y, N);                                           \
        // 再次检查转换结果是否与参考数组 ref 相等，如果不相等则输出详细错误信息
        for (const auto i : c10::irange(N)) {                          \
          ASSERT_EQ(y[i], ref[i])                                      \
              << "Failure Details:\nTest Seed to reproduce: " << seed  \
              << " x[" << i << "]=" << x[i] << " dst_t=" #dst_t;       \
        }                                                              \
      } while (0)
    
    // 依次对不同的目标类型进行测试转换
    TEST_CONVERT_TO(int8_t);
    TEST_CONVERT_TO(uint8_t);
    TEST_CONVERT_TO(int16_t);
    TEST_CONVERT_TO(uint16_t);
    TEST_CONVERT_TO(int32_t);
    TEST_CONVERT_TO(uint32_t);
    TEST_CONVERT_TO(int64_t);
    TEST_CONVERT_TO(uint64_t);
    TEST_CONVERT_TO(c10::BFloat16);
    TEST_CONVERT_TO(c10::Half);
    TEST_CONVERT_TO(float);
    TEST_CONVERT_TO(double);
    
    #undef TEST_CONVERT_TO
    // 定义一个测试用例 VecMaskTests 的 MaskedLoad 函数
    TYPED_TEST(VecMaskTests, MaskedLoad) {
      // 使用模板参数 TypeParam 定义向量类型 vec 和其值类型 VT
      using vec = TypeParam;
      using VT = ValueType<TypeParam>;
      // 向量大小 N
      constexpr auto N = vec::size();
      // 缓存对齐的数组 x、y、ref，存储向量值
      CACHE_ALIGN VT x[N];
      CACHE_ALIGN VT y[N];
      CACHE_ALIGN VT ref[N];
      // 使用测试种子生成器创建种子 seed
      auto seed = TestSeed();
      // 使用 ValueGen<VT> 生成器生成在范围 [-100, 100] 内的随机数填充数组 x
      ValueGen<VT> generator(VT(-100), VT(100), seed);
      for (const auto i : c10::irange(N)) {
        x[i] = generator.get();
      }
      // 生成向量掩码 vec_mask
      auto vec_mask = generate_vec_mask<VT>(seed);
      // 使用 vec_mask 加载数组 x 的数据到 x_vec
      x_vec.store(y);
      // 根据 vec_mask 判断每个元素是否被掩码，填充 ref 数组
      for (const auto i : c10::irange(N)) {
        if (vec_mask.is_masked(i)) {
          ref[i] = x[i];
        } else {
          ref[i] = 0;
        }
      }
      // 检查 y 和 ref 数组中的元素是否相等，用于验证测试结果
      for (const auto i : c10::irange(N)) {
        ASSERT_EQ(y[i], ref[i])
            << "Failure Details:\nTest Seed to reproduce: " << seed;
      }
    }
    
    // 定义测试用例 VecMaskTests 的 MaskedCheck 函数
    TYPED_TEST(VecMaskTests, MaskedCheck) {
      // 定义值类型 VT
      using VT = ValueType<TypeParam>;
      // 创建向量掩码 vec_mask
      auto vec_mask = create_vec_mask<VT>(0);
      // 断言向量掩码是否全零
      ASSERT_TRUE(vec_mask.all_zero()) << "all_zero check failed";
      // 将 vec_mask 设置为全掩码
      vec_mask = create_vec_mask<VT>(-1);
      // 断言向量掩码是否全掩码
      ASSERT_TRUE(vec_mask.all_masked()) << "all_masked check failed";
      // 将 vec_mask 设置为部分掩码，检查特定索引是否被掩码
      vec_mask = create_vec_mask<VT>(2);
      ASSERT_TRUE(vec_mask.is_masked(1)) << "is_masked(1) check failed";
      ASSERT_TRUE(!vec_mask.is_masked(0)) << "!is_masked(0) check failed";
    }
    
    // 定义测试用例 VecMaskTests 的 ToFrom 函数
    TYPED_TEST(VecMaskTests, ToFrom) {
      // 定义向量类型 vec 和其值类型 VT
      using vec = TypeParam;
      using VT = ValueType<TypeParam>;
      // 向量大小 N
      constexpr auto N = vec::size();
      // 使用 at::vec::VecMask<VT, 1> 创建向量掩码 vec_mask
      auto vec_mask = at::vec::VecMask<VT, 1>::from(1);
      // 断言向量掩码是否全掩码
      ASSERT_TRUE(vec_mask.all_masked()) << "expect all_masked with from(1)";
      // 重新设置向量掩码为全零
      vec_mask = at::vec::VecMask<VT, 1>::from(0);
      // 断言向量掩码是否全零
      ASSERT_TRUE(vec_mask.all_zero()) << "expect all_zero with from(0)";
    
      // 缓存对齐的数组 x、y，存储向量值
      CACHE_ALIGN VT x[N];
      CACHE_ALIGN VT y[N];
      // 使用测试种子生成器创建种子 seed
      auto seed = TestSeed();
      // 使用 ValueGen<VT> 生成器生成在范围 [0, 2] 内的随机数填充数组 x
      ValueGen<VT> generator(VT(0), VT(2), seed);
      for (const auto i : c10::irange(N)) {
        x[i] = generator.get();
      }
      // 将数组 x 的数据加载到 x_vec
      auto x_vec = vec::loadu(x);
      // 使用 at::vec::VecMask<VT, 1> 将 x_vec 转换为向量掩码 vec_mask
      vec_mask = at::vec::VecMask<VT, 1>::template from<VT, 1>(x_vec);
      // 将 vec_mask 转换回向量 y_vec
      auto y_vec = vec_mask.template to<VT, 1>();
      // 将 y_vec 的数据存储到数组 y
      y_vec.store(y);
      // 检查 y 和 x 数组中的元素是否相等，用于验证测试结果
      for (const auto i : c10::irange(N)) {
        ASSERT_EQ(y[i] != 0, x[i] != 0)
            << "Failure Details:\nTest Seed to reproduce: " << seed;
      }
    }
    #define TEST_MASK_CAST(dst_t)                                      \
      do {                                                             \
        // 定义缓存对齐的源类型数组和目标类型数组
        CACHE_ALIGN src_t x[N];                                        \
        CACHE_ALIGN dst_t y[N];                                        \
        // 生成测试用的随机种子
        auto seed = TestSeed();                                        \
        // 生成源类型的向量掩码
        auto vec_mask = generate_vec_mask<src_t>(seed);                \
        // 计算目标类型能容纳的最大元素数量
        constexpr int num_dst_elements =                               \
            std::min(N, at::vec::Vectorized<dst_t>::size());           \
        // 计算每个目标类型元素所需的源类型元素个数
        constexpr int dst_n = N / num_dst_elements;                    \
        // 将向量掩码转换为目标类型的向量掩码
        auto vec_mask_new = vec_mask.template cast<dst_t, dst_n>();    \
        // 将源类型的向量掩码存储到源数组 x 中
        vec_mask.template to<src_t, 1>().store(x);                     \
        // 将目标类型的向量掩码存储到目标数组 y 中
        vec_mask_new.template to<dst_t, dst_n>().store(y, N);          \
        // 验证每个元素在源数组和目标数组中的一致性
        for (const auto i : c10::irange(N)) {                          \
          ASSERT_EQ(y[i], x[i])                                        \
              << "Failure Details:\nTest Seed to reproduce: " << seed; \
        }                                                              \
      } while (0)
    
    // 分别测试不同类型的掩码转换
    TEST_MASK_CAST(int8_t);
    TEST_MASK_CAST(uint8_t);
    TEST_MASK_CAST(int16_t);
    TEST_MASK_CAST(uint16_t);
    TEST_MASK_CAST(int32_t);
    TEST_MASK_CAST(uint32_t);
    TEST_MASK_CAST(int64_t);
    TEST_MASK_CAST(uint64_t);
    TEST_MASK_CAST(c10::BFloat16);
    TEST_MASK_CAST(c10::Half);
    TEST_MASK_CAST(float);
    TEST_MASK_CAST(double);
    #undef TEST_MASK_CAST
#else
#error GTEST does not have TYPED_TEST
#endif
}  // namespace
```