# `bitsandbytes\include\Type.h`

```py
 # 指令用于告诉编译器只包含一次头文件，防止重复包含
// 包含必要的头文件
#include <stddef.h>
#include <vector>
#include <limits>

// 包含自定义头文件 Portable.h
#include "Portable.h"

// 使用 size_t 命名空间
using std::size_t;

// 定义枚举类型 InstrSet 包括 Scalar, SSE, AVX
namespace BinSearch {

enum InstrSet { Scalar, SSE, AVX };

// 定义宏 ALGOENUM 和 Algos 枚举类型
#define ALGOENUM(x, b) x,
enum Algos
    {
#include "AlgoXCodes.h"
    };
#undef ALGOENUM

// 定义命名空间 Details
namespace Details {

    // 模板类 InstrIntTraits 用于不同指令集的整数类型特性
    template <InstrSet I>
    struct InstrIntTraits;

    // 模板类 InstrFloatTraits 用于不同指令集的浮点数类型特性
    template <InstrSet I, typename T>
    struct InstrFloatTraits;

    // 模板类 AlgoScalarBase 用于支持 scalar 方法的算法基类
    template <typename T, Algos A, typename Enable=void>
    struct AlgoScalarBase;

    // 模板类 AlgoVecBase 用于支持 vectorial 方法和相关常量的算法基类
    template <InstrSet I, typename T, Algos A, typename Enable=void>
    struct AlgoVecBase;

    // 模板类 IntTraits 用于不同类型的整数特性
    template <typename T> struct IntTraits;

    // 特化 IntTraits 模板类，为 float 类型定义 itype 为 uint32
    template <> struct IntTraits<float>
    {
        typedef uint32 itype;
    };

    // 特化 IntTraits 模板类，为 double 类型定义 itype 为 uint64
    template <> struct IntTraits<double>
    {
        typedef uint64 itype;
    };

    // 模板类 Body 用于迭代处理
    template <int N>
    struct Body
    {
        // 迭代处理方法 iteration
        template <uint32 D, typename T, typename Expr>
        FORCE_INLINE static void iteration(const Expr& e, uint32 *ri, const T* zi, const typename Expr::Constants& cst)
        {
            // 调用 vectorial 方法处理数据
            e.vectorial(ri, zi, cst);
            // 递归调用 iteration 方法
            Body<N - 1>::template iteration<D>(e, ri + D, zi + D, cst);
        }

    };

    // 特化 Body 模板类，处理迭代结束的情况
    template <>
    struct Body<0>
    {
        // 空的 iteration 方法
        template <uint32 D, typename T, typename Expr, typename H>
        FORCE_INLINE static void iteration(const Expr& e, uint32 *ri, const T* zi, const H&)
        {
        }
    };

    // 模板类 Loop
    template <typename T, typename Algo>
    struct Loop
    {
        // 定义一个名为 algo_type 的别名为 Algo 的类型
        typedef Algo algo_type;
        // 定义一个名为 M 的常量，值为 4
        static const uint32 M = 4;
        // 定义一个名为 D 的常量，值为 algo_type 类型的 nElem 值
        static const uint32 D = algo_type::nElem;

        // 定义一个静态方法 loop，接受一个 algo_type 类型的引用 e，一个 uint32 类型的指针 ri，一个 T 类型的指针 zi，一个 uint32 类型的参数 n
        FORCE_INLINE static void loop(const algo_type& e, uint32 *ri, const T* zi, uint32 n)
        {
            // 定义一个名为 cst 的变量，类型为 algo_type::Constants
            typename algo_type::Constants cst;
            // 调用 e 的 initConstants 方法，传入 cst 变量
            e.initConstants(cst);

            // 初始化 j 为 0
            uint32 j = 0;
            // 当 j + (D*M) 小于等于 n 时，执行循环
            while (j + (D*M) <= n) {
                // 调用 Details::Body<M>::template iteration<D> 方法，传入 e、ri + j、zi + j、cst 参数
                Details::Body<M>::template iteration<D>(e, ri + j, zi + j, cst);
                // 更新 j 的值为 j + (D*M)
                j += (D*M);
            }
            // 当 j + D 小于等于 n 时，执行循环
            while (j + D <= n) {
                // 调用 e 的 vectorial 方法，传入 ri + j、zi + j、cst 参数
                e.vectorial(ri + j, zi + j, cst);
                // 更新 j 的值为 j + D
                j += D;
            }
            // 当 D 大于 1 且 j 小于 n 时，执行循环
            while (D > 1 && j < n) {
                // 将 ri[j] 的值设为 e 的 scalar 方法返回值，传入 zi[j] 参数
                ri[j] = e.scalar(zi[j]);
                // 更新 j 的值为 j + 1
                j += 1;
            }
        }
    };

    // 定义一个模板结构体 _Pipeliner，接受两个 uint32 类型的模板参数 nIterTot 和 nIterLeft
    template <uint32 nIterTot, uint32 nIterLeft>
    struct _Pipeliner
    {
        // 定义一个模板方法 go，接受一个 Expr 类型的引用 e，一个 Data 类型的指针 d
        template <typename Expr, typename Data>
        FORCE_INLINE static void go(const Expr& e, Data* d)
        {
            // 调用 e 的 run 方法，传入 nIterTot - nIterLeft 和 d 参数
            e.template run<nIterTot - nIterLeft>(d);
            // 递归调用 _Pipeliner<nIterTot, nIterLeft - 1>::go 方法，传入 e 和 d 参数
            _Pipeliner<nIterTot, nIterLeft - 1>::go(e, d);
        }
    };

    // 定义一个特化的模板结构体 _Pipeliner，当 nIterLeft 为 0 时
    template <uint32 nIterTot>
    struct _Pipeliner<nIterTot, 0>
    {
        // 定义一个模板方法 go，接受一个 Expr 类型的引用 e，一个 Data 类型的指针 d
        template <typename Expr, typename Data>
        FORCE_INLINE static void go(const Expr& e, Data* d)
        {
            // 空方法体
        }
    };

    // 定义一个模板结构体 Pipeliner，接受一个 uint32 类型的模板参数 nIter
    template <uint32 nIter>
    struct Pipeliner
    {
        // 定义一个模板方法 go，接受一个 Expr 类型的引用 e，一个 Data 类型的指针 d
        FORCE_INLINE static void go(const Expr& e, Data* d)
        {
            // 调用 _Pipeliner<nIter, nIter>::go 方法，传入 e 和 d 参数
            _Pipeliner<nIter, nIter>::go(e, d);
        }
    };
#if 1
    // 定义模板函数，用于判断类型是否完整
    template <class T>
    char is_complete_impl(char (*)[sizeof(T)]);

    // 定义模板函数，用于处理类型不完整的情况
    template <class>
    long is_complete_impl(...);

    // 定义结构体 IsComplete，用于判断类型是否完整
    template <class T>
    struct IsComplete
    {
        // 判断类型是否完整，返回布尔值
        static const bool value = sizeof(is_complete_impl<T>(0)) == sizeof(char);
    };
#else
    // 定义模板函数，用于判断类型是否完整
    template <class T, std::size_t = sizeof(T)>
    std::true_type is_complete_impl(T *);

    // 定义模板函数，用于处理类型不完整的情况
    std::false_type is_complete_impl(...);

    // 定义结构体 IsComplete，用于判断类型是否完整
    template <class T>
    struct IsComplete : decltype(is_complete_impl(std::declval<T*>())) {};
#endif

// 定义结构体 AlgoScalarToVec，用于将标量转换为向量
template <typename T, Algos A>
struct AlgoScalarToVec : AlgoScalarBase<T,A>
{
    typedef AlgoScalarBase<T, A> base_t;

    // 构造函数，根据数据初始化
    AlgoScalarToVec(const typename base_t::Data& d) :  base_t(d) {}
    // 构造函数，根据指针和数量初始化
    AlgoScalarToVec(const T* px, const uint32 n) :  base_t(px, n) {}

    // 定义常量 nElem 为 1
    static const uint32 nElem = 1;

    // 定义结构体 Constants
    struct Constants
    {
    };

    // 初始化常量
    void initConstants(Constants& cst) const
    {
    }

    // 向量化操作，将标量转换为向量
    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        *pr = base_t::scalar(*pz);
    }
};

// 定义条件模板结构体 conditional
template<bool B, class T, class F>
struct conditional { typedef T type; };

// 定义条件模板结构体 conditional 的特化版本
template<class T, class F>
struct conditional<false, T, F> { typedef F type; };

// 定义结构体 CondData，根据条件决定是否包含数据
template <typename T, bool C>
struct CondData
{
    FORCE_INLINE CondData(T x) : v(x) {}
    FORCE_INLINE operator const T&() const { return v;}
private:
    T v;
};

// 定义结构体 CondData 的特化版本，当条件为 false 时
template <typename T>
struct CondData<T,false>
{
    FORCE_INLINE CondData(T) {}
    FORCE_INLINE operator const T() const { return 0;}
};

// 定义结构体 BinAlgoBase，根据条件选择算法类型
template <InstrSet I, typename T, Algos A, bool L=false>
struct BinAlgoBase : Details::conditional< Details::IsComplete<Details::AlgoVecBase<I, T, A>>::value
                                 , Details::AlgoVecBase<I, T, A>
                                 , Details::AlgoScalarToVec<T,A>
                                 >::type
{
    # 定义一个类型别名，根据条件选择使用 AlgoVecBase 还是 AlgoScalarToVec
    typedef typename Details::conditional< Details::IsComplete<Details::AlgoVecBase<I, T, A>>::value
                                 , Details::AlgoVecBase<I, T, A>
                                 , Details::AlgoScalarToVec<T,A>
                                 >::type base_t;

    # 使用给定的指针和长度构造 BinAlgoBase 对象，根据条件选择使用 AlgoVecBase 或 AlgoScalarToVec
    BinAlgoBase(const T* px, const uint32 n) :  base_t(px, n) {}

    # 使用给定的数据构造 BinAlgoBase 对象，根据条件选择使用 AlgoVecBase 或 AlgoScalarToVec
    BinAlgoBase(const typename base_t::Data& d) : base_t(d) {}
};

} // namespace Details

} // namespace BinSearch
```