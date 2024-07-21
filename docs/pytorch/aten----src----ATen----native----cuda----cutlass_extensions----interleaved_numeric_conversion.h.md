# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\interleaved_numeric_conversion.h`

```
/*!
    \file
    \brief Boost-like numeric conversion operator for int8 and CUTLASS int4b_t interleaved in a register
*/

#pragma once

#include <cutlass/arch/arch.h>       // CUTLASS architecture definitions
#include <cutlass/array.h>           // CUTLASS Array template
#include <cutlass/half.h>            // CUTLASS half-precision floating point support
#include <cutlass/numeric_types.h>   // CUTLASS numeric type definitions

namespace cutlass {

// This converter is meant to be used with data interleaved in a 32-bit register where the even elements are in the low
// bits and the odd elements are in the high bits of the register. In addition, it assumes elements were originally
// signed and had a bias of 2**(b-1) added (where b is the number of bits in the type) to make all numbers unsigned.
// This converter will uninterleave the data and subtract the bias while converting to the result type.
template<typename T, typename S, int N>
struct FastInterleavedAndBiasedNumericArrayConverter {
};

// Specialization for converting from Array<uint8_t, 4> to Array<half_t, 4>
template<>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, 4> {
    using result_type = Array<half_t, 4>;   // Define result type as Array of half-precision floats with 4 elements
    using source_type = Array<uint8_t, 4>;  // Define source type as Array of uint8_t with 4 elements

    // Device function to perform the conversion
    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        // 定义存储结果的变量 result
        result_type result;
    
        // 将 result 转换为 uint32_t* 类型，以便进行位操作
        uint32_t* h = reinterpret_cast<uint32_t*>(&result);
    
        // 将 source 引用转换为 uint32_t 类型，用于后续指令的操作数
        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);
    
        // 定义用于 PRMT 指令的掩码常量
        static constexpr uint32_t mask_for_elt_01 = 0x5250;
        static constexpr uint32_t mask_for_elt_23 = 0x5351;
        static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
    
        // 使用内联汇编调用 prmt.b32 指令，对 h[0] 进行位重排操作
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
        
        // 使用内联汇编调用 prmt.b32 指令，对 h[1] 进行位重排操作
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
    
        // 使用 fp16 数学指令，将构造的数值减去常量 I8s_TO_F16s_MAGIC_NUM，并存储到 h[0]
        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
        
        // 使用 fp16 数学指令，将构造的数值减去常量 I8s_TO_F16s_MAGIC_NUM，并存储到 h[1]
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
    
        // 返回结果 result
        return result;
    }
    
    CUTLASS_DEVICE
    // 重载操作符，调用 convert 函数将输入参数转换为结果返回
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
        result_type result;
        // 定义结果类型为包含 bfloat16_t 类型的数组，长度为 4
        // 使用 result_type 作为结果的数据类型
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        // 如果在 CUDA 架构中且版本大于等于 800，则执行以下代码块
        // 将 result 转换为 uint32_t* 类型指针
        uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&result);
        // 将 source 转换为 uint32_t 类型并复制给 i8s
        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

        // 定义浮点数基础常量 fp32_base
        static constexpr uint32_t fp32_base = 0x4B000000;
        // 定义存储浮点数中间值的数组 fp32_intermediates，长度为 4
        float fp32_intermediates[4];

        // 使用字节重排 __byte_perm 构造 FP32 数值，bfloat16 没有足够的尾数进行 IADD 技巧
        uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
        fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
        fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7652);
        fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7651);
        fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

        // 减去 fp32_base + 128 以将无符号整数转换为有符号整数
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 4; ++ii) {
            fp32_intermediates[ii] -= 8388736.f;
        }

        // 截断 fp32 表示并打包为 bfloat16 数组
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 2; ++ii) {
            bf16_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0], fp32_intermediates_casted[2 * ii + 1], 0x7632);
        }
#else
        // 在 Ampere 架构之前的老旧架构上禁用此功能，因为它们缺乏用于 bf16 mma 的硬件支持。如果希望在老旧硬件上使用 HMMA，应直接使用 FP16 转换器进行转换。
        result.clear();  // 抑制编译器警告，清空结果对象
        arch::device_breakpoint();  // 调用设备断点函数
#endif
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);  // 调用转换函数并返回结果
    }
};

template<int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, N> {
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using result_type = Array<bfloat16_t, N>;
    using source_type = Array<uint8_t, N>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        using scalar_result_type = typename result_type::Element;  // 定义结果数组的标量元素类型
        using scalar_source_type = typename source_type::Element;  // 定义源数组的标量元素类型
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;  // 创建标量类型的转换器对象

        result_type result;  // 定义结果数组对象
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;  // 定义结果向量数组类型
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;  // 定义源向量数组类型

        vec_result*       result_ptr = reinterpret_cast<vec_result*>(&result);  // 将结果数组转换为向量数组指针
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);  // 将源数组转换为向量数组指针

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i) {
            result_ptr[i] = convert_vector_(source_ptr[i]);  // 对每个向量进行转换并存储结果
        }

        return result;  // 返回转换后的结果数组
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);  // 调用转换函数并返回结果
    }
};

template<>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, 8> {
    using result_type = Array<half_t, 8>;  // 结果数组类型为包含8个half_t类型元素的数组
    using source_type = Array<uint4b_t, 8>;  // 源数组类型为包含8个uint4b_t类型元素的数组

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);  // 调用转换函数并返回结果
    }
};

template<int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, N> {
    static constexpr int VEC_WIDTH = 8;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

    using result_type = Array<half_t, N>;  // 结果数组类型为包含N个half_t类型元素的数组
    using source_type = Array<uint4b_t, N>;  // 源数组类型为包含N个uint4b_t类型元素的数组

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        // 定义模板中的类型别名，用于存储结果元素和源类型元素的标量类型
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;

        // 使用指定的模板参数创建 FastInterleavedAndBiasedNumericArrayConverter 对象
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;

        // 定义结果对象
        result_type result;

        // 定义向量化结果和源的类型别名
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;

        // 将结果对象的指针重新解释为向量化结果对象的指针
        vec_result*       result_ptr = reinterpret_cast<vec_result*>(&result);
        // 将源对象的指针重新解释为向量化源对象的指针
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        // 使用 CUTLASS_PRAGMA_UNROLL 指令实现循环展开，对每个向量化处理单元进行转换操作
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i) {
            // 将向量化的源数据转换为向量化的结果数据，并存储在结果对象中
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }

        // 返回处理后的结果对象
        return result;
    }

    // 定义运算符重载函数，接受源类型对象并返回结果类型对象
    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        // 调用 convert 函数处理源对象并返回结果
        return convert(s);
    }
    };


template<>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint4b_t, 8> {
    // 定义结果类型为包含8个bfloat16_t的数组
    using result_type = Array<bfloat16_t, 8>;
    // 定义源类型为包含8个uint4b_t的数组
    using source_type = Array<uint4b_t, 8>;

    // CUTLASS_DEVICE表示CUDA设备函数
    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        // 定义结果对象
        result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

        // 将结果对象的地址转换为uint32_t指针
        uint32_t*      h          = reinterpret_cast<uint32_t*>(&result);
        // 将源对象的地址转换为uint32_t常量引用
        uint32_t const source_i4s = reinterpret_cast<uint32_t const&>(source);

        // 配置用于转换的常量
        static constexpr uint32_t immLut                 = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t MASK                   = 0x000f000f;
        static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

        // 不需要移位的第一个条目
        uint32_t i4s = source_i4s;
        // 使用内联汇编执行lop3.b32指令
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[0])
                     : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
        
        // 循环处理剩余的元素
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 1; ii < result_type::kElements / 2; ++ii) {
            i4s >>= sizeof_bits<typename source_type::Element>::value;
            // 使用内联汇编执行lop3.b32指令
            asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                         : "=r"(h[ii])
                         : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
        }

        // 定义BF16的偏置和ONE常量
        static constexpr uint32_t BF16_BIAS = 0xC308C308;
        static constexpr uint32_t BF16_ONE  = 0x3F803F80;

        // 构造最终的输出数字
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < result_type::kElements / 2; ++ii) {
            // 对于Ampere+架构，使用bf16 fma执行偏置减法
            asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[ii]) : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
        }
#else
        // 对于低于Ampere架构的设备，禁用此功能因为缺少bf16 mma硬件支持
        arch::device_breakpoint();
        result.clear();  // 抑制编译器警告
#endif
        // 返回转换后的结果
        return result;
    }

    // CUTLASS_DEVICE表示CUDA设备函数
    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        // 调用convert函数进行转换并返回结果
        return convert(s);
    }
};

template<int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint4b_t, N> {
    // 定义向量宽度为8
    static constexpr int VEC_WIDTH = 8;
    // 断言N必须是8的倍数
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

    // 定义结果类型为包含N个bfloat16_t的数组
    using result_type = Array<bfloat16_t, N>;
    // 定义源类型为包含N个uint4b_t的数组
    using source_type = Array<uint4b_t, N>;

    // CUTLASS_DEVICE表示CUDA设备函数
    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        // 定义类型别名，用于标量结果和源类型元素
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;
        
        // 使用指定的类型和宽度创建一个快速交错偏置数值数组转换器对象
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH> convert_vector_;
        
        // 创建结果对象
        result_type result;
        
        // 定义向量结果和源类型的数组别名
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;
        
        // 将结果对象的指针解释为向量结果类型的指针
        vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
        // 将源对象的指针解释为常量向量源类型的指针
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);
        
        // 循环展开，对于每个向量宽度的块，执行向量转换
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i) {
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }
        
        // 返回转换后的结果
        return result;
    }
    
    CUTLASS_DEVICE
    // 调用转换操作符，接受源类型对象并返回转换后的结果
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


注释：


// 结束当前的命名空间 cutlass
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// 声明 cutlass 命名空间结束
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


这段代码是C++的命名空间的结束部分。在C++中，命名空间用于避免命名冲突，这里的 `}` 表示当前命名空间的结束，而 `namespace cutlass` 则声明了 cutlass 命名空间的起始。
```