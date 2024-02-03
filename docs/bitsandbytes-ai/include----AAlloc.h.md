# `bitsandbytes\include\AAlloc.h`

```py
#pragma once

#include "Portable.h"

namespace BinSearch {
namespace Details {

// 检查指针是否按照给定对齐方式对齐
template <typename T>
bool isAligned(const T *p, size_t A)
{
    return (reinterpret_cast<size_t>(p) % A) == 0;
}

// 对齐的向量结构模板
template <class T, size_t A=64>
struct AlignedVec
{
    // 默认构造函数，初始化成员变量
    AlignedVec()
        : m_storage(0)
        , m_data(0)
        , m_sz(0)
    {
    }

    // 计算存储空间大小
    static size_t nBytes(size_t sz)
    {
        return sz * sizeof(T) + A;
    }

    // 计算指针偏移量
    static size_t shiftAmt(char *p)
    {
        return A>1? (A - (reinterpret_cast<size_t>(p) % A)) % A: 0;
    }

    // 设置指针和大小
    void setPtr(char *p, size_t sz)
    {
        m_sz = sz;
        m_data = reinterpret_cast<T *>(p + shiftAmt(p));
    }

    // 内部分配
    void resize(size_t sz)
    {
        m_storage = new char[nBytes(sz)];
        setPtr(m_storage, sz);
    }

    // 外部分配
    void set(char *storage, size_t sz)
    {
        setPtr(storage, sz);
    }

    // 析构函数，释放内存
    ~AlignedVec()
    {
        if (m_storage)
            delete [] m_storage;
    }

    // 返回向量大小
    size_t size() const { return m_sz; }
    // 重载下标运算符
    T& operator[](size_t i) { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }
    // 返回起始和结束迭代器
    T* begin()  { return m_data;  }
    T* end()  { return m_data+m_sz; }
    const T* begin() const { return m_data;  }
    const T* end() const { return m_data+m_sz; }
    // 返回第一个和最后一个元素
    T& front() { return m_data[0]; }
    T& back() { return m_data[m_sz-1]; }
    const T& front() const { return m_data[0]; }
    const T& back() const { return m_data[m_sz - 1]; }

private:
    char *m_storage; // 存储空间指针
    T *m_data; // 数据指针
    size_t m_sz; // 大小
};

} // namespace Details
} // namespace BinSearch
```