# `bitsandbytes\include\BinAlgo.h`

```py
#pragma once
// 防止头文件被重复包含

#include "Type.h"
// 包含自定义的 Type.h 头文件

#include <algorithm>
// 包含标准库中的 algorithm 头文件

namespace BinSearch {
// 命名空间 BinSearch

template <InstrSet I, typename T, Algos A, bool L=false, bool R=false>
// 模板类定义，包含指令集、数据类型、算法类型、左边界标志和右边界标志

struct BinAlgo : Details::BinAlgoBase<I,T,A>
// BinAlgo 类继承自 Details 命名空间中的 BinAlgoBase 类

{
    typedef Details::BinAlgoBase<I,T,A> base_t;
    // 定义 base_t 类型为 BinAlgoBase<I,T,A>

    BinAlgo(const T* px, const uint32 n) :  base_t(px, n), x0(px[0]), xN(px[n-1]), N(n) {}
    // 构造函数，初始化 BinAlgo 对象，包括调用基类构造函数和初始化 x0、xN、N

    BinAlgo(const T* px, const uint32 n, const typename base_t::Data& d) : base_t(d), x0(px[0]), xN(px[n-1]), N(n) {}
    // 构造函数，初始化 BinAlgo 对象，包括调用基类构造函数和初始化 x0、xN、N

    FORCE_INLINE
    uint32 scalar(T z) const
    // 标量计算函数，根据输入值 z 返回对应的索引值

    {
        if (!L || z >= x0)
            // 如果左边界标志为 false 或者 z 大于等于 x0
            if (!R || z < xN)
                // 如果右边界标志为 false 或者 z 小于 xN
                return base_t::scalar(z);
                // 返回基类的标量计算结果
            else
                return N;
                // 返回 N
        else
            return std::numeric_limits<uint32>::max();
            // 返回 uint32 类型的最大值
    }

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, uint32 n) const
    // 向量计算函数，根据输入数组 pz 计算对应的索引数组 pr
    {
        // 如果左右条件都为假，则调用Details::Loop<T,base_t>::loop函数
        if (!L && !R) {
            Details::Loop<T,base_t>::loop(*this, pr, pz, n);
        }
        // 如果左右条件有一个为真
        else {
            // 定义常量nElem为base_t::nElem
            const uint32 nElem = base_t::nElem;
            // 定义idealbufsize为256
            const uint32 idealbufsize = 256;
            // 计算bufsize，确保bufsize是idealbufsize的整数倍
            const uint32 bufsize = nElem * (idealbufsize / nElem + ((idealbufsize % nElem) ? 1 : 0));
            // 定义数据缓冲区databuf，结果缓冲区resbuf，索引缓冲区indexbuf
            T databuf[bufsize];
            uint32 resbuf[bufsize];
            uint32 indexbuf[bufsize];
    
            // 设置指针prend指向pr + n
            uint32 *prend = pr + n;
            // 循环直到pr等于prend
            while(pr != prend) {
                // 初始化计数器cnt为0，计算本次循环的迭代次数niter
                uint32 cnt = 0;
                uint32 niter = std::min(bufsize, (uint32)std::distance(pr,prend));
                // 遍历pz数组
                for (uint32 j = 0; j < niter; ++j) {
                    T z = pz[j];
                    // 如果左条件为假或z大于等于x0
                    if (!L || z >= x0)
                        // 如果右条件为假或z小于xN
                        if (!R || z < xN) {
                            // 将z添加到databuf和indexbuf中
                            databuf[cnt] = z;
                            indexbuf[cnt] = j;
                            ++cnt;
                        }
                        else
                            // 如果z不满足右条件，则将pr[j]设置为N
                            pr[j] = N;
                    else
                        // 如果z不满足左条件，则将pr[j]设置为uint32最大值
                        pr[j] = std::numeric_limits<uint32>::max();
                }
                // 调用Details::Loop<T,base_t>::loop函数处理databuf中的数据
                Details::Loop<T,base_t>::loop(*this, resbuf, databuf, cnt);
                // 将处理后的结果写回pr数组中
                for (uint32 j = 0; j < cnt; ++j)
                    pr[indexbuf[j]] = resbuf[j];
                // 更新指针位置
                pr += niter;
                pz += niter;
            }
        }
    }
    
    // 定义条件数据x0，xN，N
    Details::CondData<T,L> x0;
    Details::CondData<T,R> xN;
    Details::CondData<uint32,R> N;
};

} // namespace BinSearch
```