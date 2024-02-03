# `bitsandbytes\include\Algo-Direct2.h`

```py
#pragma once

// 防止头文件被重复包含


#include "Algo-Direct-Common.h"

// 包含名为"Algo-Direct-Common.h"的头文件


namespace BinSearch {
namespace Details {

// 命名空间的开始


template <typename T, Algos A>
struct AlgoScalarBase<T, A, typename std::enable_if<DirectAux::IsDirect2<A>::value>::type> : DirectAux::DirectInfo<2, T, A>
{
private:
    typedef DirectAux::DirectInfo<2, T, A> base_t;
    static const size_t Offset=2;

// 定义一个模板结构体AlgoScalarBase，继承自DirectAux::DirectInfo<2, T, A>，其中T为类型参数，A为枚举类型参数
// 定义私有成员base_t为DirectAux::DirectInfo<2, T, A>类型，Offset为常量2


public:
    AlgoScalarBase(const T* x, const uint32 n)
        : base_t(x, n)
    {
    }

// AlgoScalarBase的构造函数，接受指向T类型的指针x和uint32类型的n作为参数，调用base_t的构造函数进行初始化


    FORCE_INLINE uint32 scalar(T z) const
    {
        const T* px = base_t::data.xi;
        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        uint32 bidx = base_t::fun_t::f(base_t::data.scaler, base_t::data.cst0, z);
        uint32 iidx = buckets[bidx];
        px += iidx;
        if (z < *px)
            --iidx;
        if (z < *(px+1))
            --iidx;
        return iidx;
    }
};

// 定义AlgoScalarBase的成员函数scalar，接受参数z，根据一系列条件计算并返回iidx


template <InstrSet I, typename T, Algos A>
struct AlgoVecBase<I, T, A, typename std::enable_if<DirectAux::IsDirect2<A>::value>::type> : AlgoScalarBase<T, A>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

    typedef FVec<I, T> fVec;
    typedef IVec<SSE, T> i128;

    struct Constants
    {
        fVec vscaler;
        fVec vcst0;
        IVec<I, T> one;
    };

// 定义一个模板结构体AlgoVecBase，继承自AlgoScalarBase<T, A>，其中I为枚举类型参数，T为类型参数，A为枚举类型参数
// 定义静态常量nElem为InstrFloatTraits<I, T>::vec_t类型大小除以T类型大小
// 定义fVec和i128为FVec<I, T>和IVec<SSE, T>的别名
// 定义内部结构体Constants，包含fVec类型的vscaler和vcst0，以及IVec<I, T>类型的one


private:
    typedef AlgoScalarBase<T, A> base_t;

    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<SSE, float>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const

// 定义私有成员base_t为AlgoScalarBase<T, A>类型
// 定义FORCE_INLINE修饰的resolve函数，接受FVec<SSE, float>类型的vz、IVec<SSE, float>类型的bidx和uint32指针pr作为参数，无返回值
    {
        // 定义一个联合体 U，包含一个 __m128i 类型的向量和一个包含4个 uint32 类型元素的数组
        union U {
            __m128i vec;
            uint32 ui32[4];
        } u;
    
        // 将 base_t::data.buckets 强制转换为 const uint32* 类型
        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        // 将 base_t::data.xi 强制转换为 const float* 类型
        const float *xi = base_t::data.xi;
    
        // 读取索引 t
        // 将 buckets[bidx.get3()] 转换为 double* 类型，并存储到 p3 中
        const double *p3 = reinterpret_cast<const double *>(&xi[(u.ui32[3] = buckets[bidx.get3()])]);
        // 将 buckets[bidx.get2()] 转换为 double* 类型，并存储到 p2 中
        const double *p2 = reinterpret_cast<const double *>(&xi[(u.ui32[2] = buckets[bidx.get2()])]);
        // 将 buckets[bidx.get1()] 转换为 double* 类型，并存储到 p1 中
        const double *p1 = reinterpret_cast<const double *>(&xi[(u.ui32[1] = buckets[bidx.get1()])]);
        // 将 buckets[bidx.get0()] 转换为 double* 类型，并存储到 p0 中
        const double *p0 = reinterpret_cast<const double *>(&xi[(u.ui32[0] = buckets[bidx.get0()])]);
    }
#if 0
        // 读取一对值 ( X(t-1), X(t) )
        __m128 xp3 = _mm_castpd_ps(_mm_load_sd(p3));
        __m128 xp2 = _mm_castpd_ps(_mm_load_sd(p2));
        __m128 xp1 = _mm_castpd_ps(_mm_load_sd(p1));
        __m128 xp0 = _mm_castpd_ps(_mm_load_sd(p0));

        // 构建:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        __m128 h13 = _mm_shuffle_ps(xp1, xp3, (1 << 2) + (1 << 6));
        __m128 h02 = _mm_shuffle_ps(xp0, xp2, (1 << 2) + (1 << 6));
        __m128 u01 = _mm_unpacklo_ps(h02, h13);
        __m128 u23 = _mm_unpackhi_ps(h02, h13);
        __m128 vxm = _mm_shuffle_ps(u01, u23, (0) + (1 << 2) + (0 << 4) + (1 << 6));
        __m128 vxp = _mm_shuffle_ps(u01, u23, (2) + (3 << 2) + (2 << 4) + (3 << 6));
#else
        // 读取两对值 ( X(t-1), X(t) )，并将其转换为单精度浮点数
        __m128 xp23 = _mm_castpd_ps(_mm_set_pd(*p3, *p2));
        __m128 xp01 = _mm_castpd_ps(_mm_set_pd(*p1, *p0));
        // 从两对值中构建新的向量 vxm 和 vxp
        __m128 vxm = _mm_shuffle_ps(xp01, xp23, (0) + (2 << 2) + (0 << 4) + (2 << 6));
        __m128 vxp = _mm_shuffle_ps(xp01, xp23, (1) + (3 << 2) + (1 << 4) + (3 << 6));
#endif
        // 将浮点数向量转换为整数向量
        IVec<SSE, float> i(u.vec);
        // 比较向量 vz 和 vxm，将结果存储在 vlem 中
        IVec<SSE, float> vlem = vz < vxm;
        // 比较向量 vz 和 vxp，将结果存储在 vlep 中
        IVec<SSE, float> vlep = vz < vxp;
        // 将整数向量 i 加上 vlem 和 vlep 的结果，并将结果存储在 i 中
        i = i + vlem + vlep;
        // 将整数向量 i 存储到指定地址 pr 中
        i.store(pr);
    }

    // 强制内联函数
    FORCE_INLINE
        //NO_INLINE
        // 解析函数，接受一个双精度浮点数向量 vz 和一个浮点数整数向量 bidx，以及一个指向 uint32 的指针 pr
        void resolve(const FVec<SSE, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        // 将 base_t::data.buckets 强制转换为 uint32 类型的指针，并赋值给 buckets
        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        // 将 base_t::data.xi 赋值给 xi
        const double *xi = base_t::data.xi;
    
        // 从 buckets 中获取索引 bidx.get1() 对应的值，赋值给 b1
        uint32 b1 = buckets[bidx.get1()];
        // 从 buckets 中获取索引 bidx.get0() 对应的值，赋值给 b0
        uint32 b0 = buckets[bidx.get0()];
    
        // 从 xi 中获取地址为 b1 的值，赋值给 p1
        const double *p1 = &xi[b1];
        // 从 xi 中获取地址为 b0 的值，赋值给 p0
        const double *p0 = &xi[b0];
    
        // 从 p1 和 p0 中加载两个双精度浮点数到寄存器 vx1 和 vx0
        __m128d vx1 = _mm_loadu_pd(p1);
        __m128d vx0 = _mm_loadu_pd(p0);
    
        // 从 vx0 和 vx1 中按指定顺序组合成新的寄存器 vxm 和 vxp
        __m128d vxm = _mm_shuffle_pd(vx0, vx1, 0);
        __m128d vxp = _mm_shuffle_pd(vx0, vx1, 3);
    
        // 创建 SSE 类型的整型向量 i，初始值为 b1 和 b0
        IVec<SSE, double> i(b1, b0);
        // 比较 vz 和 vxm，将比较结果存储在 vlem 中
        IVec<SSE, double> vlem = (vz < vxm);
        // 比较 vz 和 vxp，将比较结果存储在 vlep 中
        IVec<SSE, double> vlep = (vz < vxp);
        // 更新 i 的值，加上 vlem 和 vlep 的结果
        i = i + vlem + vlep;
    
        // 定义一个联合体 u，包含一个 __m128i 类型的向量 vec 和一个 uint32 类型的数组 ui32
        union {
            __m128i vec;
            uint32 ui32[4];
        } u;
        // 将 i 的值赋给 vec
        u.vec = i;
        // 将 u.ui32 中的值分别赋给 pr[0] 和 pr[1]
        pr[0] = u.ui32[0];
        pr[1] = u.ui32[2];
    }
#ifdef USE_AVX
    // 如果定义了使用 AVX 指令集

    FORCE_INLINE
        // 定义一个强制内联函数 resolve，参数为 AVX 浮点向量 vz，AVX 整型向量 bidx，以及指向 uint32 的指针 pr
        void resolve(const FVec<AVX, float>& vz, const IVec<AVX, float>& bidx, uint32 *pr) const
    {
        // 从基类的数据中获取 buckets 和 xi 的指针
        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        const float *xi = base_t::data.xi;

#if 0   // use gather instructions
        // 如果为 0，则使用 gather 指令

        // 创建一个 AVX 整型向量 idxm，并根据 buckets 和 bidx 设置其索引
        IVec<AVX,float> idxm;
        idxm.setidx(buckets, bidx);
        // 创建一个全零的 AVX 整型向量 z
        __m256i z = _mm256_setzero_si256();
        // 创建一个全一的 AVX 整型向量 minusone
        IVec<AVX,float> minusone = _mm256_cmpeq_epi32(z,z);
        // 计算 idxp，即 idxm 减去 minusone
        IVec<AVX,float> idxp = idxm - minusone;

        // 从 xi 中根据 idxm 和 idxp 读取数据，分别存储在 vxm 和 vxp 中
        FVec<AVX, float> vxm = _mm256_i32gather_ps(xi, idxm, sizeof(float));
        FVec<AVX, float> vxp = _mm256_i32gather_ps(xi, idxp, sizeof(float));
        // 将 idxm 存储在 ip 中
        IVec<AVX, float> ip = idxm;

#else // do not use gather instrucions
        // 如果不使用 gather 指令

        union U {
            __m256i vec;
            uint32 ui32[8];
        } u;

        // 读取索引 t，并将其存储在对应的指针 p7 到 p0 中
        const double *p7 = reinterpret_cast<const double *>(&xi[(u.ui32[7] = buckets[bidx.get7()])]);
        const double *p6 = reinterpret_cast<const double *>(&xi[(u.ui32[6] = buckets[bidx.get6()])]);
        const double *p5 = reinterpret_cast<const double *>(&xi[(u.ui32[5] = buckets[bidx.get5()])]);
        const double *p4 = reinterpret_cast<const double *>(&xi[(u.ui32[4] = buckets[bidx.get4()])]);
        const double *p3 = reinterpret_cast<const double *>(&xi[(u.ui32[3] = buckets[bidx.get3()])]);
        const double *p2 = reinterpret_cast<const double *>(&xi[(u.ui32[2] = buckets[bidx.get2()])]);
        const double *p1 = reinterpret_cast<const double *>(&xi[(u.ui32[1] = buckets[bidx.get1()])]);
        const double *p0 = reinterpret_cast<const double *>(&xi[(u.ui32[0] = buckets[bidx.get0()])]);
#if 0 // perform 8 loads in double precision

        // 读取一对值 ( X(t-1), X(t) )
        __m128 xp7 = _mm_castpd_ps(_mm_load_sd(p7));
        __m128 xp6 = _mm_castpd_ps(_mm_load_sd(p6));
        __m128 xp5 = _mm_castpd_ps(_mm_load_sd(p5));
        __m128 xp4 = _mm_castpd_ps(_mm_load_sd(p4));
        __m128 xp3 = _mm_castpd_ps(_mm_load_sd(p3));
        __m128 xp2 = _mm_castpd_ps(_mm_load_sd(p2));
        __m128 xp1 = _mm_castpd_ps(_mm_load_sd(p1));
        __m128 xp0 = _mm_castpd_ps(_mm_load_sd(p0));

        // 构建:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        __m128 h57 = _mm_shuffle_ps(xp5, xp7, (1 << 2) + (1 << 6));  // F- F+ H- H+
        __m128 h46 = _mm_shuffle_ps(xp4, xp6, (1 << 2) + (1 << 6));  // E- E+ G- G+
        __m128 h13 = _mm_shuffle_ps(xp1, xp3, (1 << 2) + (1 << 6));  // B- B+ D- D+
        __m128 h02 = _mm_shuffle_ps(xp0, xp2, (1 << 2) + (1 << 6));  // A- A+ C- C+

        __m128 u01 = _mm_unpacklo_ps(h02, h13);  // A- B- A+ B+
        __m128 u23 = _mm_unpackhi_ps(h02, h13);  // C- D- C+ D+
        __m128 u45 = _mm_unpacklo_ps(h46, h57);  // E- F- E+ F+
        __m128 u67 = _mm_unpackhi_ps(h46, h57);  // G- H- G+ H+

        __m128 abcdm = _mm_shuffle_ps(u01, u23, (0) + (1 << 2) + (0 << 4) + (1 << 6));  // A- B- C- D-
        __m128 abcdp = _mm_shuffle_ps(u01, u23, (2) + (3 << 2) + (2 << 4) + (3 << 6));  // A+ B+ C+ D+
        __m128 efghm = _mm_shuffle_ps(u45, u67, (0) + (1 << 2) + (0 << 4) + (1 << 6));  // E- F- G- H-
        __m128 efghp = _mm_shuffle_ps(u45, u67, (2) + (3 << 2) + (2 << 4) + (3 << 6));  // E+ F+ G+ H+

        // 将 abcdm 和 efghm 插入到 AVX 寄存器中
        FVec<AVX, float> vxp = _mm256_insertf128_ps(_mm256_castps128_ps256(abcdm), efghm, 1);
        // 将 abcdp 和 efghp 插入到 AVX 寄存器中
        FVec<AVX, float> vxm = _mm256_insertf128_ps(_mm256_castps128_ps256(abcdp), efghp, 1);

        // 使用 u.vec 初始化 IVec<AVX, float> 对象 ip
        IVec<AVX, float> ip(u.vec);
#else   // use __mm256_set_pd
// 如果不使用 __mm256_set_pd，则执行以下代码

        // read pairs ( X(t-1), X(t) )
        // 读取一对值（X(t-1), X(t)）
        __m256 x0145 = _mm256_castpd_ps(_mm256_set_pd(*p5, *p4, *p1, *p0)); // { x0(t-1), x0(t), x1(t-1), x1(t), x4(t-1), x4(t), x5(t-1), x5(t) }
        __m256 x2367 = _mm256_castpd_ps(_mm256_set_pd(*p7, *p6, *p3, *p2)); // { x2(t-1), x2(t), x3(t-1), x3(t), x6(t-1), x6(t), x7(t-1), x7(t) }

        // { x0(t-1), x1(t-1), x2(t-1), 3(t-1, x4(t-1), x5(t-1), x6(t-1), xt(t-1) }
        // 从 x0145 和 x2367 中按照指定顺序组合成新的 FVec 对象 vxm
        FVec<AVX, float> vxm = _mm256_shuffle_ps(x0145, x2367, 0 + (2 << 2) + (0 << 4) + (2 << 6) );
        // { x0(t), x1(t), x2(t), 3(t, x4(t), x5(t), x6(t), xt(t) }
        // 从 x0145 和 x2367 中按照指定顺序组合成新的 FVec 对象 vxp
        FVec<AVX, float> vxp = _mm256_shuffle_ps(x0145, x2367, 1 + (3 << 2) + (1 << 4) + (3 << 6) );

        // 将 u.vec 转换为 IVec 对象 ip
        IVec<AVX, float> ip(u.vec);

#endif

#endif

        // 比较 vz 和 vxm，将结果存储在 vlem 中
        IVec<AVX, float> vlem = vz < vxm;
        // 比较 vz 和 vxp，将结果存储在 vlep 中
        IVec<AVX, float> vlep = vz < vxp;
        // 将 ip、vlem 和 vlep 相加，结果存储在 ip 中
        ip = ip + vlem + vlep;

        // 将 ip 中的值存储到 pr 中
        ip.store(pr);
    }



    FORCE_INLINE
        //NO_INLINE
        // 解析函数，接收一个 FVec 对象 vz、一个 IVec 对象 bidx 和一个 uint32 类型指针 pr
        void resolve(const FVec<AVX, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        // 定义一个联合体，包含一个__m256i类型的向量和一个uint64类型的数组
        union {
            __m256i vec;
            uint64 ui64[4];
        } u;

        // 将base_t::data.buckets强制转换为const uint32指针，并赋值给buckets
        const uint32* buckets = reinterpret_cast<const uint32 *>(base_t::data.buckets);
        // 将base_t::data.xi赋值给xi
        const double *xi = base_t::data.xi;

        // 读取索引t
        // 将buckets[bidx.get3()]的值赋给u.ui64[3]，并将结果作为xi的索引，赋值给p3
        const double *p3 = &xi[(u.ui64[3] = buckets[bidx.get3()])];
        const double *p2 = &xi[(u.ui64[2] = buckets[bidx.get2()])];
        const double *p1 = &xi[(u.ui64[1] = buckets[bidx.get1()])];
        const double *p0 = &xi[(u.ui64[0] = buckets[bidx.get0()])];

        // 读取对 ( X(t-1), X(t) ) 的值
        __m128d xp3 = _mm_loadu_pd(p3);
        __m128d xp2 = _mm_loadu_pd(p2);
        __m128d xp1 = _mm_loadu_pd(p1);
        __m128d xp0 = _mm_loadu_pd(p0);

        // 构建:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        // 将xp0和xp2合并为x02，将xp1和xp3合并为x13
        __m256d x02 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xp0), xp2, 1);
        __m256d x13 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xp1), xp3, 1);
        // 将x02和x13按照低位和高位拆分，分别赋值给vxm和vxp
        FVec<AVX, double> vxm = _mm256_unpacklo_pd(x02,x13);
        FVec<AVX, double> vxp = _mm256_unpackhi_pd(x02,x13);
    }
// 创建 AVX 双精度向量，将 u.vec 赋值给 i
IVec<AVX, double> i(u.vec);
// 比较 vz 和 vxm 的每个元素，将比较结果存储在 vlem 中
IVec<AVX, double> vlem = vz < vxm;
// 比较 vz 和 vxp 的每个元素，将比较结果存储在 vlep 中
IVec<AVX, double> vlep = vz < vxp;
// 将 i、vlem 和 vlep 的对应元素相加，结果存储在 i 中
i = i + vlem + vlep;
// 提取 i 中的低 32 位整数，并将结果存储在 pr 中
i.extractLo32s().store(pr);
#endif

public:

// 使用给定的 x 和 n 初始化 AlgoVecBase 对象
AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

// 初始化常量对象 Constants 中的各个成员
void initConstants(Constants& cst) const
{
    // 设置 vscaler 为 base_t::data.scaler
    cst.vscaler.setN(base_t::data.scaler);
    // 设置 vcst0 为 base_t::data.cst0
    cst.vcst0.setN(base_t::data.cst0);
    // 设置 one 为 1
    cst.one.setN(uint32(1));
}

// 对输入的 pz 进行向量化处理，并将结果存储在 pr 中
void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
{
    // 创建 fVec 对象 vz，将 pz 赋值给它
    fVec vz(pz);
    // 调用 resolve 函数，传入 vz、base_t::fun_t::f(cst.vscaler, cst.vcst0, vz) 和 pr 作为参数
    resolve(vz, base_t::fun_t::f(cst.vscaler, cst.vcst0, vz), pr);
}
};
} // namespace Details
} // namespace BinSearch
```