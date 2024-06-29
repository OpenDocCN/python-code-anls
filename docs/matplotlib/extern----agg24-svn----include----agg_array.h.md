# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_array.h`

```py
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
// 定义防止重复包含的宏
#ifndef AGG_ARRAY_INCLUDED
#define AGG_ARRAY_INCLUDED

// 包含必要的头文件
#include <stddef.h>
#include <string.h>
#include "agg_basics.h"

// 命名空间 agg
namespace agg
{

    //-------------------------------------------------------pod_array_adaptor
    // 模板类 pod_array_adaptor，用于封装数组，提供访问接口
    template<class T> class pod_array_adaptor
    {
    public:
        typedef T value_type;
        // 构造函数，传入数组和大小
        pod_array_adaptor(T* array, unsigned size) : 
            m_array(array), m_size(size) {}

        // 返回数组大小
        unsigned size() const { return m_size; }
        // 重载 [] 运算符，用于访问数组元素
        const T& operator [] (unsigned i) const { return m_array[i]; }
              T& operator [] (unsigned i)       { return m_array[i]; }
        // 访问指定位置的元素
        const T& at(unsigned i) const           { return m_array[i]; }
              T& at(unsigned i)                 { return m_array[i]; }
        // 返回指定位置的元素值
        T  value_at(unsigned i) const           { return m_array[i]; }

    private:
        T*       m_array;
        unsigned m_size;
    };


    //---------------------------------------------------------pod_auto_array
    // 模板类 pod_auto_array，用于封装固定大小的数组，提供访问接口
    template<class T, unsigned Size> class pod_auto_array
    {
    public:
        typedef T value_type;
        typedef pod_auto_array<T, Size> self_type;

        // 默认构造函数
        pod_auto_array() {}
        // 显式构造函数，传入指向数组的指针
        explicit pod_auto_array(const T* c)
        {
            // 复制数组内容到成员数组
            memcpy(m_array, c, sizeof(T) * Size);
        }

        // 赋值运算符重载，用于复制数组内容
        const self_type& operator = (const T* c)
        {
            // 复制数组内容到成员数组
            memcpy(m_array, c, sizeof(T) * Size);
            return *this;
        }

        // 返回数组大小
        static unsigned size() { return Size; }
        // 重载 [] 运算符，用于访问数组元素
        const T& operator [] (unsigned i) const { return m_array[i]; }
              T& operator [] (unsigned i)       { return m_array[i]; }
        // 访问指定位置的元素
        const T& at(unsigned i) const           { return m_array[i]; }
              T& at(unsigned i)                 { return m_array[i]; }
        // 返回指定位置的元素值
        T  value_at(unsigned i) const           { return m_array[i]; }

    private:
        T m_array[Size];
    };


    //--------------------------------------------------------pod_auto_vector
    // 模板类 pod_auto_vector，用于封装固定大小的数组，提供访问接口
    template<class T, unsigned Size> class pod_auto_vector
    {
    //--------------------------------------------------------------pod_vector
    // 简单的模板类，用于存储平凡数据类型（Plain Old Data, POD）的固定大小向量。
    // 数据在内存中是连续存储的。
    //------------------------------------------------------------------------
    template<class T> class pod_vector
    {
    // 公有部分开始

        // 定义类型 T 的 value_type
        typedef T value_type;

        // 析构函数，释放动态分配的内存
        ~pod_vector() { pod_allocator<T>::deallocate(m_array, m_capacity); }

        // 默认构造函数，初始化成员变量
        pod_vector() : m_size(0), m_capacity(0), m_array(0) {}

        // 带参数的构造函数声明
        pod_vector(unsigned cap, unsigned extra_tail=0);

        // 复制构造函数声明
        pod_vector(const pod_vector<T>&);

        // 赋值运算符重载声明
        const pod_vector<T>& operator = (const pod_vector<T>&);

        // 设置新的容量。所有数据将丢失，大小被设置为零。
        void capacity(unsigned cap, unsigned extra_tail=0);

        // 返回当前容量
        unsigned capacity() const { return m_capacity; }

        // 分配 n 个元素。所有数据将丢失，但可以访问的元素范围为 0 到 size-1。
        void allocate(unsigned size, unsigned extra_tail=0);

        // 保持内容不变地调整大小
        void resize(unsigned new_size);

    // 公有部分结束

        // 将 m_array 数组的前 m_size 个元素设置为 0
        void zero()
        {
            memset(m_array, 0, sizeof(T) * m_size);
        }

        // 添加元素 v 到数组末尾，并增加 m_size
        void add(const T& v)         { m_array[m_size++] = v; }

        // 添加元素 v 到数组末尾，并增加 m_size
        void push_back(const T& v)   { m_array[m_size++] = v; }

        // 在指定位置 pos 处插入值为 val 的元素
        void insert_at(unsigned pos, const T& val);

        // 增加 m_size 的大小
        void inc_size(unsigned size) { m_size += size; }

        // 返回当前数组的大小
        unsigned size()      const   { return m_size; }

        // 返回当前数组所占字节数
        unsigned byte_size() const   { return m_size * sizeof(T); }

        // 序列化函数声明，将数组内容写入指定的内存位置 ptr
        void serialize(int8u* ptr) const;

        // 反序列化函数声明，从指定的内存数据 data 中读取数据，并设置字节大小为 byte_size
        void deserialize(const int8u* data, unsigned byte_size);

        // 下标运算符重载，返回数组中下标为 i 的元素的常量引用
        const T& operator [] (unsigned i) const { return m_array[i]; }

        // 下标运算符重载，返回数组中下标为 i 的元素的引用
              T& operator [] (unsigned i)       { return m_array[i]; }

        // 返回数组中下标为 i 的元素的常量引用
        const T& at(unsigned i) const           { return m_array[i]; }

        // 返回数组中下标为 i 的元素的引用
              T& at(unsigned i)                 { return m_array[i]; }

        // 返回数组中下标为 i 的元素的拷贝
        T  value_at(unsigned i) const           { return m_array[i]; }

        // 返回数组的常量指针
        const T* data() const { return m_array; }

        // 返回数组的指针
              T* data()       { return m_array; }

        // 清空数组，设置 m_size 为零
        void remove_all()         { m_size = 0; }

        // 清空数组，设置 m_size 为零
        void clear()              { m_size = 0; }

        // 截断数组，保留前 num 个元素
        void cut_at(unsigned num) { if(num < m_size) m_size = num; }

    // 私有部分开始

    private:
        // 数组当前大小
        unsigned m_size;

        // 数组当前容量
        unsigned m_capacity;

        // 指向动态分配数组的指针
        T*       m_array;
    };

    //------------------------------------------------------------------------

    // 模板类 pod_vector 的 capacity 函数实现
    template<class T> 
    void pod_vector<T>::capacity(unsigned cap, unsigned extra_tail)
    {
        // 重置当前大小为零
        m_size = 0;

        // 如果请求的容量大于当前容量
        if(cap > m_capacity)
        {
            // 释放当前数组
            pod_allocator<T>::deallocate(m_array, m_capacity);

            // 更新容量为请求容量加上额外的尾部空间
            m_capacity = cap + extra_tail;

            // 分配新的数组空间，如果 m_capacity 非零
            m_array = m_capacity ? pod_allocator<T>::allocate(m_capacity) : 0;
        }
    }

    //------------------------------------------------------------------------

    // 模板类 pod_vector 的 allocate 函数实现
    template<class T> 
    void pod_vector<T>::allocate(unsigned size, unsigned extra_tail)
    {
        // 调用 capacity 函数以分配指定大小的空间
        capacity(size, extra_tail);

        // 设置当前大小为给定大小
        m_size = size;
    }

    //------------------------------------------------------------------------

    // 模板类 pod_vector 的 resize 函数实现
    {
        // 如果新大小超过当前大小，则进行以下操作
        if(new_size > m_size)
        {
            // 如果新大小超过当前容量，则执行以下操作
            if(new_size > m_capacity)
            {
                // 使用 POD 分配器分配新大小的内存
                T* data = pod_allocator<T>::allocate(new_size);
                // 将当前数组的数据复制到新分配的内存中
                memcpy(data, m_array, m_size * sizeof(T));
                // 使用 POD 分配器释放当前数组的内存
                pod_allocator<T>::deallocate(m_array, m_capacity);
                // 更新成员变量 m_array 指向新分配的内存
                m_array = data;
            }
        }
        else
        {
            // 如果新大小不超过当前大小，则直接更新成员变量 m_size
            m_size = new_size;
        }
    }
    
    //------------------------------------------------------------------------
    template<class T> pod_vector<T>::pod_vector(unsigned cap, unsigned extra_tail) :
        // 初始化成员变量 m_size 为 0
        m_size(0), 
        // 初始化成员变量 m_capacity 为 cap + extra_tail
        m_capacity(cap + extra_tail), 
        // 使用 POD 分配器分配 m_capacity 大小的内存，并将指针赋给 m_array
        m_array(pod_allocator<T>::allocate(m_capacity)) {}
    
    //------------------------------------------------------------------------
    template<class T> pod_vector<T>::pod_vector(const pod_vector<T>& v) :
        // 复制构造函数，初始化 m_size 为 v 的 m_size
        m_size(v.m_size),
        // 复制构造函数，初始化 m_capacity 为 v 的 m_capacity
        m_capacity(v.m_capacity),
        // 如果 v 的 m_capacity 不为 0，则使用 POD 分配器分配 m_capacity 大小的内存，否则初始化 m_array 为 0
        m_array(v.m_capacity ? pod_allocator<T>::allocate(v.m_capacity) : 0)
    {
        // 将 v 的 m_array 中的数据复制到当前对象的 m_array 中，复制长度为 v.m_size 个 T 类型的元素
        memcpy(m_array, v.m_array, sizeof(T) * v.m_size);
    }
    
    //------------------------------------------------------------------------
    template<class T> const pod_vector<T>& 
    pod_vector<T>::operator = (const pod_vector<T>&v)
    {
        // 分配内存以容纳 v 的大小
        allocate(v.m_size);
        // 如果 v 的大小不为 0，则将 v 的 m_array 中的数据复制到当前对象的 m_array 中
        if(v.m_size) memcpy(m_array, v.m_array, sizeof(T) * v.m_size);
        // 返回当前对象的引用
        return *this;
    }
    
    //------------------------------------------------------------------------
    template<class T> void pod_vector<T>::serialize(int8u* ptr) const
    { 
        // 如果当前大小不为 0，则将当前对象的 m_array 中的数据序列化到 ptr 指向的内存中
        if(m_size) memcpy(ptr, m_array, m_size * sizeof(T)); 
    }
    
    //------------------------------------------------------------------------
    template<class T> 
    void pod_vector<T>::deserialize(const int8u* data, unsigned byte_size)
    {
        // 将 byte_size 转换为 T 类型的元素个数
        byte_size /= sizeof(T);
        // 分配内存以容纳 byte_size 个 T 类型的元素
        allocate(byte_size);
        // 如果 byte_size 不为 0，则将 data 指向的内存中的数据反序列化到当前对象的 m_array 中
        if(byte_size) memcpy(m_array, data, byte_size * sizeof(T));
    }
    
    //------------------------------------------------------------------------
    template<class T> 
    void pod_vector<T>::insert_at(unsigned pos, const T& val)
    {
        // 如果插入位置超过当前大小，则直接将 val 放在末尾
        if(pos >= m_size) 
        {
            m_array[m_size] = val;
        }
        else
        {
            // 在插入位置前移动数据，然后将 val 插入到 pos 处
            memmove(m_array + pos + 1, m_array + pos, (m_size - pos) * sizeof(T));
            m_array[pos] = val;
        }
        // 增加当前大小
        ++m_size;
    }
    // 默认情况下，增量值等于 (1 << S)，即块大小。
    //------------------------------------------------------------------------
    template<class T, unsigned S=6> class pod_bvector
    {
    private:
        // 分配块内存的私有方法声明
        void allocate_block(unsigned nb);
        // 获取数据指针的私有方法声明
        T*   data_ptr();
    
        unsigned        m_size;         // 向量中元素的当前数量
        unsigned        m_num_blocks;   // 当前分配的块数
        unsigned        m_max_blocks;   // 最大允许的块数
        T**             m_blocks;       // 指向块指针数组的指针
        unsigned        m_block_ptr_inc;    // 块指针增量
    };
    
    
    //------------------------------------------------------------------------
    // pod_bvector 类的析构函数定义
    template<class T, unsigned S> pod_bvector<T, S>::~pod_bvector()
    {
        if(m_num_blocks)
        {
            T** blk = m_blocks + m_num_blocks - 1;
            // 释放每个块分配的内存
            while(m_num_blocks--)
            {
                pod_allocator<T>::deallocate(*blk, block_size);
                --blk;
            }
        }
        // 释放块指针数组的内存
        pod_allocator<T*>::deallocate(m_blocks, m_max_blocks);
    }
    
    
    //------------------------------------------------------------------------
    // pod_bvector 类的 free_tail 方法定义
    template<class T, unsigned S> 
    void pod_bvector<T, S>::free_tail(unsigned size)
    {
        if(size < m_size)
        {
            // 计算要保留的块数
            unsigned nb = (size + block_mask) >> block_shift;
            // 释放多余的块
            while(m_num_blocks > nb)
            {
                pod_allocator<T>::deallocate(m_blocks[--m_num_blocks], block_size);
            }
            // 如果没有块被保留，则释放块指针数组的内存
            if(m_num_blocks == 0)
            {
                pod_allocator<T*>::deallocate(m_blocks, m_max_blocks);
                m_blocks = 0;
                m_max_blocks = 0;
            }
            // 更新向量的当前大小
            m_size = size;
        }
    }
    
    
    //------------------------------------------------------------------------
    // pod_bvector 类的默认构造函数定义
    template<class T, unsigned S> pod_bvector<T, S>::pod_bvector() :
        m_size(0),
        m_num_blocks(0),
        m_max_blocks(0),
        m_blocks(0),
        m_block_ptr_inc(block_size)
    {
    }
    
    
    //------------------------------------------------------------------------
    // pod_bvector 类的带参数构造函数定义
    template<class T, unsigned S> 
    pod_bvector<T, S>::pod_bvector(unsigned block_ptr_inc) :
        m_size(0),
        m_num_blocks(0),
        m_max_blocks(0),
        m_blocks(0),
        m_block_ptr_inc(block_ptr_inc)
    {
    }
    
    
    //------------------------------------------------------------------------
    // pod_bvector 类的拷贝构造函数定义
    template<class T, unsigned S> 
    pod_bvector<T, S>::pod_bvector(const pod_bvector<T, S>& v) :
        m_size(v.m_size),
        m_num_blocks(v.m_num_blocks),
        m_max_blocks(v.m_max_blocks),
        // 根据 v 的块数量分配块指针数组的内存，如果 v 没有块则置为 0
        m_blocks(v.m_max_blocks ? 
                 pod_allocator<T*>::allocate(v.m_max_blocks) : 
                 0),
        m_block_ptr_inc(v.m_block_ptr_inc)
    {
        unsigned i;
        // 拷贝每个块的数据到新的块中
        for(i = 0; i < v.m_num_blocks; ++i)
        {
            m_blocks[i] = pod_allocator<T>::allocate(block_size);
            memcpy(m_blocks[i], v.m_blocks[i], block_size * sizeof(T));
        }
    }
    //------------------------------------------------------------------------
    // 赋值运算符重载函数实现，用于将另一个 pod_bvector<T, S> 对象的内容复制到当前对象
    template<class T, unsigned S> 
    const pod_bvector<T, S>& 
    pod_bvector<T, S>::operator = (const pod_bvector<T, S>& v)
    {
        unsigned i;
        // 将当前对象的块数量扩展到与 v 的块数量相同
        for(i = m_num_blocks; i < v.m_num_blocks; ++i)
        {
            allocate_block(i);
        }
        // 复制 v 的每个块的数据到当前对象对应的块
        for(i = 0; i < v.m_num_blocks; ++i)
        {
            memcpy(m_blocks[i], v.m_blocks[i], block_size * sizeof(T));
        }
        // 更新当前对象的元素数量
        m_size = v.m_size;
        return *this;
    }
    
    
    //------------------------------------------------------------------------
    // 分配新的块内存空间，当需要更多块时使用
    template<class T, unsigned S>
    void pod_bvector<T, S>::allocate_block(unsigned nb)
    {
        // 如果需要的块数超过当前分配的最大块数，则进行扩展
        if(nb >= m_max_blocks) 
        {
            // 分配新的块指针数组内存空间
            T** new_blocks = pod_allocator<T*>::allocate(m_max_blocks + m_block_ptr_inc);
    
            // 如果当前有已分配的块，则将其复制到新的块指针数组中
            if(m_blocks)
            {
                memcpy(new_blocks, 
                       m_blocks, 
                       m_num_blocks * sizeof(T*));
    
                // 释放原来的块指针数组内存空间
                pod_allocator<T*>::deallocate(m_blocks, m_max_blocks);
            }
            // 更新当前对象的块指针数组和最大块数
            m_blocks = new_blocks;
            m_max_blocks += m_block_ptr_inc;
        }
        // 分配新的块内存空间，并增加块数量计数
        m_blocks[nb] = pod_allocator<T>::allocate(block_size);
        m_num_blocks++;
    }
    
    
    
    //------------------------------------------------------------------------
    // 返回当前对象数据的指针，确保指向合适的内存块
    template<class T, unsigned S>
    inline T* pod_bvector<T, S>::data_ptr()
    {
        // 计算当前元素所在的块索引
        unsigned nb = m_size >> block_shift;
        // 如果需要的块索引超过当前已分配的块数，则分配新的块内存空间
        if(nb >= m_num_blocks)
        {
            allocate_block(nb);
        }
        // 返回当前元素的指针
        return m_blocks[nb] + (m_size & block_mask);
    }
    
    
    
    //------------------------------------------------------------------------
    // 将给定的值添加到当前对象的末尾
    template<class T, unsigned S> 
    inline void pod_bvector<T, S>::add(const T& val)
    {
        // 将值存储到当前元素的指针所指向的位置，并增加元素数量计数
        *data_ptr() = val;
        ++m_size;
    }
    
    
    //------------------------------------------------------------------------
    // 移除当前对象的最后一个元素
    template<class T, unsigned S> 
    inline void pod_bvector<T, S>::remove_last()
    {
        // 如果当前元素数量大于 0，则减少元素数量计数
        if(m_size) --m_size;
    }
    
    
    //------------------------------------------------------------------------
    // 替换当前对象的最后一个元素为给定的值
    template<class T, unsigned S> 
    void pod_bvector<T, S>::modify_last(const T& val)
    {
        // 移除当前对象的最后一个元素
        remove_last();
        // 添加给定的值作为当前对象的最后一个元素
        add(val);
    }
    
    
    //------------------------------------------------------------------------
    // 分配指定数量的连续元素的块内存空间
    template<class T, unsigned S> 
    int pod_bvector<T, S>::allocate_continuous_block(unsigned num_elements)
    {
        if(num_elements < block_size)
        {
            data_ptr(); // 分配初始块（如果需要）
            unsigned rest = block_size - (m_size & block_mask);
            unsigned index;
            if(num_elements <= rest)
            {
                // 剩余的块大小足够，可以使用它
                //-----------------
                index = m_size;
                m_size += num_elements;
                return index;
            }
    
            // 新块
            //---------------
            m_size += rest;
            data_ptr(); // 分配新块
            index = m_size;
            m_size += num_elements;
            return index;
        }
        return -1; // 无法分配
    }
    
    
    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    unsigned pod_bvector<T, S>::byte_size() const
    {
        return m_size * sizeof(T); // 返回向量中元素占用的字节大小
    }
    
    
    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void pod_bvector<T, S>::serialize(int8u* ptr) const
    {
        unsigned i;
        for(i = 0; i < m_size; i++)
        {
            memcpy(ptr, &(*this)[i], sizeof(T)); // 序列化向量中的元素到指定指针位置
            ptr += sizeof(T);
        }
    }
    
    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void pod_bvector<T, S>::deserialize(const int8u* data, unsigned byte_size)
    {
        remove_all(); // 清空当前向量内容
        byte_size /= sizeof(T); // 计算需要反序列化的元素数量
        for(unsigned i = 0; i < byte_size; ++i)
        {
            T* ptr = data_ptr(); // 获取当前数据块的指针
            memcpy(ptr, data, sizeof(T)); // 将数据从给定指针复制到向量中
            ++m_size; // 更新向量大小
            data += sizeof(T); // 移动数据指针到下一个元素
        }
    }
    
    // 替换或添加从“start”位置开始的一些元素
    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void pod_bvector<T, S>::deserialize(unsigned start, const T& empty_val, 
                                        const int8u* data, unsigned byte_size)
    {
        while(m_size < start)
        {
            add(empty_val); // 在当前向量末尾添加空值，直到达到指定的起始位置
        }
    
        byte_size /= sizeof(T); // 计算需要反序列化的元素数量
        for(unsigned i = 0; i < byte_size; ++i)
        {
            if(start + i < m_size)
            {
                memcpy(&((*this)[start + i]), data, sizeof(T)); // 替换现有位置的元素
            }
            else
            {
                T* ptr = data_ptr(); // 获取当前数据块的指针
                memcpy(ptr, data, sizeof(T)); // 将数据从给定指针复制到向量中
                ++m_size; // 更新向量大小
            }
            data += sizeof(T); // 移动数据指针到下一个元素
        }
    }
    
    
    //---------------------------------------------------------block_allocator
    // 用于任意 POD 数据的分配器。在不同的缓存系统中用于高效的内存分配。
    // 内存以固定大小的块分配（在构造函数中的“block_size”）。如果所需大小超过块大小，则分配器
    // 创建一个块分配器类，用于管理分配和释放内存块
    //------------------------------------------------------------------------
    class block_allocator
    {
        // 内部结构体定义，用于存储每个内存块的数据和大小
        struct block_type
        {
            int8u*   data;     // 指向内存块数据的指针
            unsigned size;     // 内存块的大小
        };

    public:
        // 移除所有已分配的内存块
        void remove_all()
        {
            if(m_num_blocks)
            {
                // 获取最后一个内存块
                block_type* blk = m_blocks + m_num_blocks - 1;
                // 逐个释放所有内存块
                while(m_num_blocks--)
                {
                    pod_allocator<int8u>::deallocate(blk->data, blk->size);
                    --blk;
                }
                // 最后释放存储块信息的数组
                pod_allocator<block_type>::deallocate(m_blocks, m_max_blocks);
            }
            // 重置内部状态变量
            m_num_blocks = 0;
            m_max_blocks = 0;
            m_blocks = 0;
            m_buf_ptr = 0;
            m_rest = 0;
        }

        // 析构函数，用于对象销毁时调用移除所有内存块
        ~block_allocator()
        {
            remove_all();
        }

        // 块分配器的构造函数，初始化相关参数和状态
        block_allocator(unsigned block_size, unsigned block_ptr_inc=256-8) :
            m_block_size(block_size),
            m_block_ptr_inc(block_ptr_inc),
            m_num_blocks(0),
            m_max_blocks(0),
            m_blocks(0),
            m_buf_ptr(0),
            m_rest(0)
        {
        }
       

        // 分配指定大小和对齐方式的内存块
        int8u* allocate(unsigned size, unsigned alignment=1)
        {
            // 如果请求大小为0，直接返回空指针
            if(size == 0) return 0;
            
            // 如果请求的大小小于当前剩余可用大小，则直接从当前内存块分配
            if(size <= m_rest)
            {
                int8u* ptr = m_buf_ptr;
                
                // 如果需要对齐，则进行对齐操作
                if(alignment > 1)
                {
                    unsigned align = 
                        (alignment - unsigned((size_t)ptr) % alignment) % alignment;

                    // 调整分配大小和指针位置
                    size += align;
                    ptr += align;
                    
                    // 如果调整后的大小仍然小于当前剩余可用大小，则直接分配
                    if(size <= m_rest)
                    {
                        m_rest -= size;
                        m_buf_ptr += size;
                        return ptr;
                    }
                    
                    // 否则分配一个新的内存块
                    allocate_block(size);
                    return allocate(size - align, alignment);
                }
                
                // 直接分配当前大小并调整指针位置
                m_rest -= size;
                m_buf_ptr += size;
                return ptr;
            }
            
            // 如果当前剩余空间不足以分配请求的大小，则分配一个新的内存块
            allocate_block(size + alignment - 1);
            return allocate(size, alignment);
        }
    private:
        // 分配一个新的块，大小至少为 size
        void allocate_block(unsigned size)
        {
            // 如果请求大小小于 m_block_size，则使用 m_block_size
            if(size < m_block_size) size = m_block_size;
            // 如果当前块数量超过最大块数限制，则分配更多内存
            if(m_num_blocks >= m_max_blocks) 
            {
                // 分配新的块数组，增加 m_max_blocks + m_block_ptr_inc 大小
                block_type* new_blocks = 
                    pod_allocator<block_type>::allocate(m_max_blocks + m_block_ptr_inc);

                // 如果已有块数组 m_blocks，将其数据复制到新的块数组中
                if(m_blocks)
                {
                    memcpy(new_blocks, 
                           m_blocks, 
                           m_num_blocks * sizeof(block_type));
                    // 释放原来的块数组内存
                    pod_allocator<block_type>::deallocate(m_blocks, m_max_blocks);
                }
                // 更新 m_blocks 指向新分配的块数组，更新最大块数
                m_blocks = new_blocks;
                m_max_blocks += m_block_ptr_inc;
            }

            // 设置新分配的块的大小和数据指针
            m_blocks[m_num_blocks].size = size;
            m_blocks[m_num_blocks].data = 
                m_buf_ptr =
                pod_allocator<int8u>::allocate(size);

            // 更新块计数和剩余大小
            m_num_blocks++;
            m_rest = size;
        }

        // 块大小
        unsigned    m_block_size;
        // 块指针增加量
        unsigned    m_block_ptr_inc;
        // 当前块数量
        unsigned    m_num_blocks;
        // 最大块数量
        unsigned    m_max_blocks;
        // 块数组指针
        block_type* m_blocks;
        // 缓冲区指针
        int8u*      m_buf_ptr;
        // 剩余大小
        unsigned    m_rest;
    };

    //------------------------------------------------------------------------
    // 快速排序阈值
    enum quick_sort_threshold_e
    {
        quick_sort_threshold = 9
    };

    
    //-----------------------------------------------------------swap_elements
    // 交换两个元素
    template<class T> inline void swap_elements(T& a, T& b)
    {
        T temp = a;
        a = b;
        b = temp;
    }


    //--------------------------------------------------------------quick_sort
    // 快速排序函数模板
    template<class Array, class Less>
    void quick_sort(Array& arr, Less less)
    {
        // 如果数组大小小于 2，则直接返回，不需要排序
        if(arr.size() < 2) return;
    
        // 定义指向数组元素的指针 e1 和 e2
        typename Array::value_type* e1;
        typename Array::value_type* e2;
    
        // 定义一个大小为 80 的整型栈数组 stack
        int stack[80];
        // 栈顶指针 top 指向栈数组的起始位置
        int* top = stack; 
        // 定义数组的上界 limit 为数组大小
        int limit = arr.size();
        // 定义数组的下界 base 为 0
        int base = 0;
    
        // 开始快速排序过程
        for(;;)
        {
            // 当前子数组的长度
            int len = limit - base;
    
            // 定义排序过程中的索引 i、j 和中间值 pivot
            int i;
            int j;
            int pivot;
    
            // 如果子数组长度大于设定的快速排序阈值 quick_sort_threshold
            if(len > quick_sort_threshold)
            {
                // 选择 base + len/2 作为中间值 pivot
                pivot = base + len / 2;
                // 将中间值 pivot 与 base 处元素交换
                swap_elements(arr[base], arr[pivot]);
    
                // 初始化 i 和 j
                i = base + 1;
                j = limit - 1;
    
                // 确保 *i <= *base <= *j 
                e1 = &(arr[j]); 
                e2 = &(arr[i]);
                if(less(*e1, *e2)) swap_elements(*e1, *e2);
    
                e1 = &(arr[base]); 
                e2 = &(arr[i]);
                if(less(*e1, *e2)) swap_elements(*e1, *e2);
    
                e1 = &(arr[j]); 
                e2 = &(arr[base]);
                if(less(*e1, *e2)) swap_elements(*e1, *e2);
    
                // 开始快速排序的分区过程
                for(;;)
                {
                    do i++; while( less(arr[i], arr[base]) );
                    do j--; while( less(arr[base], arr[j]) );
    
                    // 当 i > j 时跳出循环
                    if( i > j )
                    {
                        break;
                    }
    
                    // 交换 arr[i] 和 arr[j] 的值
                    swap_elements(arr[i], arr[j]);
                }
    
                // 将 arr[base] 与 arr[j] 的值交换，完成一次分区操作
                swap_elements(arr[base], arr[j]);
    
                // 推入较大子数组的边界值
                if(j - base > limit - i)
                {
                    top[0] = base;
                    top[1] = j;
                    base   = i;
                }
                else
                {
                    top[0] = i;
                    top[1] = limit;
                    limit  = j;
                }
                top += 2;
            }
            else
            {
                // 当子数组长度小于等于快速排序阈值时，执行插入排序
                j = base;
                i = j + 1;
    
                // 开始插入排序过程
                for(; i < limit; j = i, i++)
                {
                    for(; less(*(e1 = &(arr[j + 1])), *(e2 = &(arr[j]))); j--)
                    {
                        // 若 e1 小于 e2，则交换它们的值
                        swap_elements(*e1, *e2);
                        // 若 j 已经是 base，则跳出内层循环
                        if(j == base)
                        {
                            break;
                        }
                    }
                }
                
                // 如果栈非空，弹出栈顶元素继续排序
                if(top > stack)
                {
                    top  -= 2;
                    base  = top[0];
                    limit = top[1];
                }
                else
                {
                    // 栈为空时结束排序过程
                    break;
                }
            }
        }
    }
    
    //------------------------------------------------------remove_duplicates
    // 从已排序的数组中移除重复元素。该函数不截断数组尾部，只返回剩余元素的数量。
    //-----------------------------------------------------------------------
    // 删除数组中重复的元素，并返回新数组的长度，使用给定的相等比较器 'equal'
    template<class Array, class Equal>
    unsigned remove_duplicates(Array& arr, Equal equal)
    {
        // 如果数组长度小于2，直接返回数组长度，因为无重复元素可删除
        if(arr.size() < 2) return arr.size();
    
        unsigned i, j;
        // 遍历数组，i为当前元素索引，j为新数组的索引
        for(i = 1, j = 1; i < arr.size(); i++)
        {
            typename Array::value_type& e = arr[i];
            // 如果当前元素与前一个元素不相等，将当前元素放入新数组中
            if(!equal(e, arr[i - 1]))
            {
                arr[j++] = e;
            }
        }
        // 返回新数组的长度
        return j;
    }
    
    //--------------------------------------------------------invert_container
    // 反转容器中的元素顺序
    template<class Array> void invert_container(Array& arr)
    {
        int i = 0;
        int j = arr.size() - 1;
        // 使用双指针法，将数组首尾元素依次交换直至中间
        while(i < j)
        {
            swap_elements(arr[i++], arr[j--]);
        }
    }
    
    //------------------------------------------------------binary_search_pos
    // 使用二分查找在已排序的数组中寻找插入位置
    template<class Array, class Value, class Less>
    unsigned binary_search_pos(const Array& arr, const Value& val, Less less)
    {
        // 如果数组为空，直接返回插入位置0
        if(arr.size() == 0) return 0;
    
        unsigned beg = 0;
        unsigned end = arr.size() - 1;
    
        // 如果插入值小于数组第一个元素，返回插入位置0
        if(less(val, arr[0])) return 0;
        // 如果插入值大于数组最后一个元素，返回插入位置在数组末尾之后
        if(less(arr[end], val)) return end + 1;
    
        // 二分查找，缩小插入位置范围直至找到正确的位置
        while(end - beg > 1)
        {
            unsigned mid = (end + beg) >> 1;
            if(less(val, arr[mid])) end = mid; 
            else                    beg = mid;
        }
    
        return end;
    }
    
    //----------------------------------------------------------range_adaptor
    // 数组范围适配器，用于访问数组中指定范围的元素
    template<class Array> class range_adaptor
    {
    public:
        typedef typename Array::value_type value_type;
    
        // 构造函数，接受数组、起始索引和大小作为参数
        range_adaptor(Array& array, unsigned start, unsigned size) :
            m_array(array), m_start(start), m_size(size)
        {}
    
        // 返回适配器范围内的元素个数
        unsigned size() const { return m_size; }
        // 访问适配器范围内的元素（const 和 非 const 两个版本）
        const value_type& operator [] (unsigned i) const { return m_array[m_start + i]; }
              value_type& operator [] (unsigned i)       { return m_array[m_start + i]; }
        // 使用at()方法访问适配器范围内的元素（const 和 非 const 两个版本）
        const value_type& at(unsigned i) const           { return m_array[m_start + i]; }
              value_type& at(unsigned i)                 { return m_array[m_start + i]; }
        // 直接返回适配器范围内的元素值，无法修改元素
        value_type  value_at(unsigned i) const           { return m_array[m_start + i]; }
    
    private:
        Array& m_array;     // 引用原始数组
        unsigned m_start;   // 起始索引
        unsigned m_size;    // 适配器范围大小
    };
    
    //---------------------------------------------------------------int_less
    // 整数比较函数，返回a是否小于b
    inline bool int_less(int a, int b) { return a < b; }
    
    //------------------------------------------------------------int_greater
    // 整数比较函数，返回a是否大于b
    inline bool int_greater(int a, int b) { return a > b; }
    
    //----------------------------------------------------------unsigned_less
    // 无符号整数比较函数，返回a是否小于b
    inline bool unsigned_less(unsigned a, unsigned b) { return a < b; }
    
    //-------------------------------------------------------unsigned_greater
    // 无符号整数比较函数，返回a是否大于b
    inline bool unsigned_greater(unsigned a, unsigned b) { return a > b; }
}


注释：

// 结束一个 C/C++ 的预处理器条件指令块



#endif


注释：

// 如果前面有 #ifdef 或 #ifndef，那么结束这个条件指令块
```