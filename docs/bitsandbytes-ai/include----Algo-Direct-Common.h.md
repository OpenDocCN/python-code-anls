# `bitsandbytes\include\Algo-Direct-Common.h`

```py
#pragma once

#include <algorithm> // 包含算法库
#include <limits> // 包含数值极限库
#include <type_traits> // 包含类型特性库
#include "AAlloc.h" // 包含自定义头文件 AAlloc.h

namespace BinSearch { // 命名空间 BinSearch
namespace Details { // 命名空间 Details

namespace DirectAux { // 命名空间 DirectAux

#define SAFETY_MULTI_PASS true // 定义 SAFETY_MULTI_PASS 为 true

template <typename T> // 模板定义结构体 HResults
struct HResults
{
    HResults(T h, double ratio, size_t n) : H(h), hRatio(ratio), nInc(n) {} // 结构体构造函数
    T H; // 成员变量 H
    double hRatio; // 成员变量 hRatio
    size_t nInc; // 成员变量 nInc
};

#ifdef USE_FMA
template <Algos A> struct IsDirect { static const bool value = (A == Direct) || (A == DirectFMA); }; // 判断是否为 Direct 或 DirectFMA
template <Algos A> struct IsDirect2 { static const bool value = (A == Direct2) || (A == Direct2FMA); }; // 判断是否为 Direct2 或 Direct2FMA
template <Algos A> struct IsDirectCache { static const bool value = (A == DirectCache) || (A == DirectCacheFMA); }; // 判断是否为 DirectCache 或 DirectCacheFMA
#else
template <Algos A> struct IsDirect { static const bool value = (A == Direct); }; // 判断是否为 Direct
template <Algos A> struct IsDirect2 { static const bool value = (A == Direct2); }; // 判断是否为 Direct2
template <Algos A> struct IsDirectCache { static const bool value = (A == DirectCache); }; // 判断是否为 DirectCache
#endif

// general definition
template <Algos A, typename T, typename Enable = void> // 模板定义结构体 BucketElem
struct BucketElem
{
    FORCE_INLINE void set( uint32 b, const T *) // 设置函数 set
    {
        m_b = b; // 设置成员变量 m_b
    }

    FORCE_INLINE uint32 index() const { return m_b; } // 返回索引值

private:
    uint32 m_b; // 成员变量 m_b
};

// specialization for DirectCache methods

template <typename T> struct MatchingIntType; // 模板定义结构体 MatchingIntType
template <> struct MatchingIntType<double> { typedef uint64 type; }; // 特化 MatchingIntType 结构体
template <> struct MatchingIntType<float> { typedef uint32 type; }; // 特化 MatchingIntType 结构体

template <Algos A, typename T> // 模板定义结构体 BucketElem
struct BucketElem<A, T, typename std::enable_if< IsDirectCache<A>::value >::type >
{
    typedef typename MatchingIntType<T>::type I; // 定义类型 I

    void set(uint32 b, const T *xi) // 设置函数 set
    {
        u.u.x = xi[b]; // 设置成员变量 u.u.x
        u.u.b = b; // 设置成员变量 u.u.b
    }

    FORCE_INLINE I index() const { return u.u.b; } // 返回索引值
    FORCE_INLINE T x() const { return u.u.x; } // 返回 x 值

private:
    union { // 联合体
        double dummy; // 双精度浮点数 dummy
        struct // 结构体
        {
            T x; // 成员变量 x
            I b; // 成员变量 b
        } u; // 结构体变量 u
    } u; // 联合体变量 u
};

template <bool UseFMA, unsigned char Gap, typename T> // 模板定义结构体
// 定义 DirectTraits 结构体，包含一些静态方法和模板方法
struct DirectTraits
{
    // 静态方法，检查是否超出 uint32 容量限制
    static void checkH(T scaler, T x0, T xN)
    {
        // 计算差值 Dn
        T Dn = xN - x0;
        // 计算 ifmax
        T ifmax = Dn * scaler;
        // 断言 ifmax 是否小于 uint32 最大值减去 Gap - 1
        myassert((ifmax < std::numeric_limits<uint32>::max() - (Gap - 1)),
            "Problem unfeasible: index size exceeds uint32 capacity:"
            << " D[N] =" << Dn
            << ", H =" << scaler
            << ", H D[n] =" << ifmax << "\n"
        );
    }

    // 强制内联的静态方法，根据 scaler、x0 和 z 计算结果并返回 uint32 类型
    FORCE_INLINE static uint32 f(T scaler, T x0, T z)
    {
        // 计算 tmp
        T tmp = scaler * (z - x0);
#ifdef USE_SSE2
        // 使用 SSE2 指令集计算结果并返回
        return ftoi(FVec1<SSE,T>(tmp));
#else
        // 转换为 uint32 类型并返回
        return static_cast<uint32>(tmp);
#endif
    }

    // 模板方法，根据不同指令集计算结果并返回
    template <InstrSet I>
    FORCE_INLINE static typename FTOITraits<I, T>::vec_t f(const FVec<I, T>& scaler, const FVec<I, T>& x0, const FVec<I, T>& z)
    {
        // 计算结果并返回
        return ftoi(scaler*(z-x0));
    }

    // 静态方法，返回 x0
    static T cst0(T scaler, T x0)
    {
        return x0;
    }
};

// 当使用 FMA 指令集时的特化结构体
#ifdef USE_FMA
template <unsigned char Gap, typename T>
struct DirectTraits<true,Gap,T>
{
    // 定义 FVec1 类型
    typedef FVec1<SSE, T> fVec1;

    // 静态方法，检查是否超出 uint32 容量限制
    static void checkH(T scaler, T H_Times_x0, T xN)
    {
        // 定义联合体 ifmax，用于存储结果
        union {
            typename FVec1<SSE, T>::vec_t v;
            T s;
        } ifmax;
        // 计算结果并存储在 ifmax 中
        ifmax.v = mulSub(fVec1(scaler), fVec1(xN), fVec1(H_Times_x0));
        // 断言结果是否小于 uint32 最大值减去 Gap - 1
        myassert((ifmax.s < std::numeric_limits<uint32>::max() - (Gap - 1)),
            "Problem unfeasible: index size exceeds uint32 capacity:"
            << " H X[0] =" << H_Times_x0
            << ", H =" << scaler
            << ", X[N] =" << xN
            << ", H X[N] - H X[0] =" << ifmax.s << "\n"
        );
    }

    // 强制内联的静态方法，根据 scaler、Hx0 和 xi 计算结果并返回 uint32 类型
    FORCE_INLINE static uint32 f(T scaler, T Hx0, T xi)
    {
        // 计算结果并返回
        return ftoi(mulSub(fVec1(scaler), fVec1(xi), fVec1(Hx0)));
    }

    // 模板方法，根据不同指令集计算结果并返回
    template <InstrSet I>
    FORCE_INLINE static typename FTOITraits<I,T>::vec_t f(const FVec<I,T>& scaler, const FVec<I, T>& H_Times_X0, const FVec<I, T>& z)
    {
        // 计算结果并返回
        return ftoi(mulSub(scaler, z, H_Times_X0));
    }

    // 静态方法，返回 scaler 乘以 x0 的结果
    static T cst0(T scaler, T x0)
    {
        return scaler*x0;
    }
};
#endif

// 定义 DirectInfo 结构体模板，包含模板参数 Gap、T 和 Algos
template <unsigned char Gap, typename T, Algos A>
struct DirectInfo
{
    // 根据 Algos 类型确定是否使用 FMA 指令集
    static const bool UseFMA = (A == DirectFMA) || (A == Direct2FMA) || (A == DirectCacheFMA);
    // 定义函数类型 fun_t、桶元素类型 bucket_t 和对齐向量类型 bucketvec_t
    typedef DirectTraits<UseFMA, Gap, T> fun_t;
    typedef BucketElem<A,T> bucket_t;
    typedef AlignedVec<bucket_t> bucketvec_t;

    // 定义 Data 结构体
    struct Data {
        // 默认构造函数
        Data() : buckets(0), xi(0), scaler(0), cst0(0) {}
        // 构造函数，初始化 Data 对象
        Data( const T *x      // for Direct must persist if xws=NULL
            , uint32 n
            , T H
            , bucket_t *bws   // assumed to gave size nb, as computed below
            , T *xws = NULL   // assumed to have size (n+Gap-1). Optional for Direct, unused for DirectCache, required for DirectGap
            )
            : buckets(bws)
            , scaler(H)
            , cst0(fun_t::cst0(H, x[0]))
        {
            // 断言检查桶指针是否已分配且对齐
            myassert(((bws != NULL) && (isAligned(bws,64))), "bucket pointer not allocated or incorrectly aligned");

            // 计算桶的数量
            uint32 nb = 1 + fun_t::f(H, cst0, x[n-1]);

            const uint32 npad = Gap-1;
            const uint32 n_sz = n + npad;   // size of padded vector

            if (xws) {
                // 断言检查 x 指针是否已分配且对齐
                myassert(isAligned(xws,8), "x pointer not allocated or incorrectly aligned");
                // 在前面用 x[0] 填充 npad 个元素
                std::fill_n(xws, npad, x[0]);
                // 将 x 复制到 xws 中
                std::copy(x, x+n, xws + npad);
                xi = xws;
            }
            else {
                // 如果 Gap>1，则必须提供 X 工作空间
                myassert(Gap==1, "if Gap>1 then X workspace must be provided");
                xi = x;
            }

            // 填充索引
            populateIndex(bws, nb, xi, n_sz, scaler, cst0);
        }

        const bucket_t *buckets;
        const T *xi;
        T scaler;
        T cst0;  // could be x0 or (scaler*x0), depending if we are using FMA or not
    } data;

    // 计算增长步长
    static T growStep(T H)
    {
        T step;
        T P = next(H);
        while ((step = P - H) == 0)
            P = next(P);
        return step;
    }

    // 计算 H 值
    static HResults<T> computeH(const T *px, uint32 nx)
    }
    // 用给定的数据填充索引桶
    static void populateIndex(BucketElem<A, T> *buckets, uint32 index_size, const T *px, uint32 x_size, T scaler, T cst0)
    {
        // 初始化循环变量 i, b, j
        for (uint32 i = x_size-1, b = index_size-1, j=0; ; --i) {
            // 计算当前元素的索引
            uint32 idx = fun_t::f(scaler, cst0, px[i]);
            // 当索引大于当前桶的索引时，将当前元素添加到桶中
            while (b > idx) {  // 在第一次迭代中，j=0，但此条件始终为假
                buckets[b].set( j, px );
                --b;
            }
            // 如果 Gap 等于 1 或者当前桶的索引等于计算得到的索引，则将元素添加到桶中
            if (Gap==1 || b == idx) { // 如果 Gap 等于 1，在编译时已知，检查 b==idx 是多余的
                // 计算 j 的值，指向要检查的第一个 X 元素的索引
                j = i - (Gap-1);
                buckets[b].set(j, px);
                // 如果当前桶索引为 0，则跳出循环
                if (b-- == 0)
                    break;
            }
        }
    }

    // 构造函数，初始化 DirectInfo 对象
    DirectInfo(const Data& d)
        : data(d)
    {
    }

    // 构造函数，根据给定的数据计算 HResults
    DirectInfo(const T* px, const uint32 n)
    {
        // 计算 HResults
        HResults<T> res = computeH(px, n);
#ifdef PAPER_TEST
        // 如果定义了宏PAPER_TEST，则将res结构体中的nInc赋值给nInc变量，将res结构体中的hRatio赋值给hRatio变量
        nInc = res.nInc;
        hRatio = res.hRatio;
#endif
        // 定义一个常量npad，值为Gap-1
        const uint32 npad = Gap-1;
        // 定义一个常量n_sz，值为n加上npad，即n的大小加上npad
        const uint32 n_sz = n + npad;   // size of padded vector

        // 如果npad不为0，则将xi向量调整大小为n_sz
        if (npad)
            xi.resize(n_sz);

        // 将res结构体中的H赋值给H变量
        T H    = res.H;
        // 调用fun_t命名空间中的cst0函数，传入H和px[0]，将返回值赋值给cst0变量
        T cst0 = fun_t::cst0(H, px[0]);
        // 调用fun_t命名空间中的f函数，传入H、cst0和px[n-1]，将返回值赋值给maxIndex变量
        const uint32 maxIndex = fun_t::f(H, cst0, px[n-1]);
        // 调整buckets向量大小为maxIndex + 1
        buckets.resize(maxIndex + 1);

        // 创建一个Data对象data，传入px、n、H、buckets.begin()和（如果npad不为0，则传入xi.begin()，否则传入NULL）
        data = Data(px, n, H, buckets.begin(), (npad? xi.begin(): NULL));
    }

private:
    // 定义一个bucketvec_t类型的buckets向量
    bucketvec_t buckets;
    // 定义一个AlignedVec类型的xi向量，元素类型为T，初始大小为8

#ifdef PAPER_TEST
public:
    // 如果定义了宏PAPER_TEST，则定义一个double类型的hRatio变量和一个size_t类型的nInc变量
    double hRatio;
    size_t nInc;
#endif
};


} // namespace DirectAux
} // namespace Details
} // namespace BinSearch
```