# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_bitset_iterator.h`

```
// 声明防止重复包含的预处理指令，确保本文件内容仅被编译一次
#ifndef AGG_BITSET_ITERATOR_INCLUDED
#define AGG_BITSET_ITERATOR_INCLUDED

// 包含基础的 AGG 库头文件
#include "agg_basics.h"

// 声明命名空间 agg，用于封装本文件中的类和函数
namespace agg
{
    
    // 定义位集迭代器类
    class bitset_iterator
    {
    public:
        // 构造函数，初始化位集迭代器
        bitset_iterator(const int8u* bits, unsigned offset = 0) :
            m_bits(bits + (offset >> 3)),  // 初始化 m_bits 指向偏移后的字节位置
            m_mask(0x80 >> (offset & 7))   // 初始化 m_mask 为偏移后的位掩码
        {}

        // 前置递增运算符重载，将迭代器移动到下一个位
        void operator ++ ()
        {
            m_mask >>= 1;  // 右移位掩码，移动到下一个位
            if(m_mask == 0)
            {
                ++m_bits;   // 如果位掩码为0，移动到下一个字节
                m_mask = 0x80;  // 重置位掩码为最高位
            }
        }

        // 返回当前位的值
        unsigned bit() const
        {
            return (*m_bits) & m_mask;  // 返回当前位的值，使用位运算获取
        }

    private:
        const int8u* m_bits;  // 指向位集数据的指针
        int8u        m_mask;  // 当前位的位掩码
    };

}

// 结束命名空间 agg
#endif  // 结束防止重复包含的预处理指令
```