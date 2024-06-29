# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_allocator.h`

```
// 包含必要的头文件和声明命名空间 agg
#ifndef AGG_SPAN_ALLOCATOR_INCLUDED
#define AGG_SPAN_ALLOCATOR_INCLUDED

#include "agg_array.h"

namespace agg
{
    //----------------------------------------------------------span_allocator
    // span_allocator 模板类定义，用于分配颜色元素的连续内存块
    template<class ColorT> class span_allocator
    {
    public:
        // 定义颜色类型为 ColorT
        typedef ColorT color_type;

        //--------------------------------------------------------------------
        // 分配 span_len 个颜色元素的内存块
        AGG_INLINE color_type* allocate(unsigned span_len)
        {
            // 如果需要的 span_len 超过当前 m_span 的容量
            if(span_len > m_span.size())
            {
                // 为了减少重新分配的次数，将 span_len 对齐到 256 个颜色元素
                // 我只是喜欢这个数字，而且看起来也合理。
                //-----------------------
                m_span.resize(((span_len + 255) >> 8) << 8);
            }
            // 返回分配后的内存块起始地址
            return &m_span[0];
        }

        // 返回当前 span 的起始地址
        AGG_INLINE color_type* span()               { return &m_span[0]; }
        // 返回当前 span 的最大长度
        AGG_INLINE unsigned    max_span_len() const { return m_span.size(); }

    private:
        // 使用 pod_array 存储颜色元素的连续内存块
        pod_array<color_type> m_span;
    };
}

#endif
```