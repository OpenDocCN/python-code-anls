# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_scanline_storage_aa.h`

```
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
//
// Adaptation for 32-bit screen coordinates has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------

#ifndef AGG_SCANLINE_STORAGE_AA_INCLUDED
#define AGG_SCANLINE_STORAGE_AA_INCLUDED

#include <string.h>   // Include for string manipulation functions
#include <stdlib.h>   // Include for standard library functions
#include <math.h>     // Include for math functions
#include "agg_array.h"  // Include for the aggregation array template


namespace agg
{

    //----------------------------------------------scanline_cell_storage
    // 模板类定义：扫描线单元存储器
    template<class T> class scanline_cell_storage
    {
        // 内部结构定义：额外跨度
        struct extra_span
        {
            unsigned len;  // 额外跨度的长度
            T*       ptr;  // 指向 T 类型数据的指针
        };
    // 公共成员声明开始，定义了类型别名 value_type 为模板参数 T
    public:
        typedef T value_type;

        //---------------------------------------------------------------
        // 析构函数，清理所有存储的数据
        ~scanline_cell_storage()
        {
            remove_all();
        }

        //---------------------------------------------------------------
        // 默认构造函数，初始化 m_cells 和 m_extra_storage
        scanline_cell_storage() :
            m_cells(128-2),     // 初始化 m_cells，大小为 126
            m_extra_storage()   // 初始化 m_extra_storage，为空
        {}

        // Copying
        //---------------------------------------------------------------
        // 拷贝构造函数，从给定的 scanline_cell_storage<T> 对象 v 拷贝数据
        scanline_cell_storage(const scanline_cell_storage<T>& v) :
            m_cells(v.m_cells),      // 拷贝 m_cells
            m_extra_storage()        // 初始化 m_extra_storage，为空
        {
            copy_extra_storage(v);   // 拷贝额外存储的数据
        }

        //---------------------------------------------------------------
        // 赋值运算符重载，从给定的 scanline_cell_storage<T> 对象 v 赋值数据
        const scanline_cell_storage<T>& 
        operator = (const scanline_cell_storage<T>& v)
        {
            remove_all();           // 清空当前对象的所有数据
            m_cells = v.m_cells;    // 赋值 m_cells
            copy_extra_storage(v);  // 拷贝额外存储的数据
            return *this;           // 返回当前对象的引用
        }

        //---------------------------------------------------------------
        // 清空所有存储的数据
        void remove_all()
        {
            int i;
            // 逆序释放额外存储的每个块
            for(i = m_extra_storage.size()-1; i >= 0; --i)
            {
                pod_allocator<T>::deallocate(m_extra_storage[i].ptr,
                                             m_extra_storage[i].len);
            }
            m_extra_storage.remove_all();   // 清空额外存储列表
            m_cells.remove_all();           // 清空主存储列表
        }

        //---------------------------------------------------------------
        // 添加一组新的数据块到 m_cells 中
        int add_cells(const T* cells, unsigned num_cells)
        {
            // 在 m_cells 中分配连续的空间块
            int idx = m_cells.allocate_continuous_block(num_cells);
            if(idx >= 0)
            {
                T* ptr = &m_cells[idx];     // 获取分配的起始指针
                memcpy(ptr, cells, sizeof(T) * num_cells);   // 复制数据到分配的块中
                return idx;                 // 返回分配的索引
            }
            // 如果分配失败，创建一个额外的存储块并添加
            extra_span s;
            s.len = num_cells;
            s.ptr = pod_allocator<T>::allocate(num_cells);   // 分配新的内存块
            memcpy(s.ptr, cells, sizeof(T) * num_cells);     // 复制数据到新分配的块中
            m_extra_storage.add(s);     // 添加新的额外存储块到列表中
            return -int(m_extra_storage.size());   // 返回负数表示添加到了额外存储中
        }

        //---------------------------------------------------------------
        // 索引运算符重载，返回指定索引处的数据块指针（只读版本）
        const T* operator [] (int idx) const
        {
            if(idx >= 0)
            {
                if((unsigned)idx >= m_cells.size()) return 0;    // 如果索引超出 m_cells 的范围，返回空指针
                return &m_cells[(unsigned)idx];     // 返回 m_cells 中指定索引的数据块指针
            }
            unsigned i = unsigned(-idx - 1);
            if(i >= m_extra_storage.size()) return 0;    // 如果索引超出 m_extra_storage 的范围，返回空指针
            return m_extra_storage[i].ptr;      // 返回 m_extra_storage 中指定索引的额外存储块指针
        }

        //---------------------------------------------------------------
        // 索引运算符重载，返回指定索引处的数据块指针（可修改版本）
        T* operator [] (int idx)
        {
            if(idx >= 0)
            {
                if((unsigned)idx >= m_cells.size()) return 0;    // 如果索引超出 m_cells 的范围，返回空指针
                return &m_cells[(unsigned)idx];     // 返回 m_cells 中指定索引的数据块指针
            }
            unsigned i = unsigned(-idx - 1);
            if(i >= m_extra_storage.size()) return 0;    // 如果索引超出 m_extra_storage 的范围，返回空指针
            return m_extra_storage[i].ptr;      // 返回 m_extra_storage 中指定索引的额外存储块指针
        }
    private:
        // 复制给定对象的额外存储空间数据
        void copy_extra_storage(const scanline_cell_storage<T>& v)
        {
            unsigned i;
            // 遍历源对象的额外存储空间
            for(i = 0; i < v.m_extra_storage.size(); ++i)
            {
                // 获取当前源额外存储空间
                const extra_span& src = v.m_extra_storage[i];
                // 创建目标额外存储空间
                extra_span dst;
                dst.len = src.len;
                // 分配内存并复制数据
                dst.ptr = pod_allocator<T>::allocate(dst.len);
                memcpy(dst.ptr, src.ptr, dst.len * sizeof(T));
                // 将目标额外存储空间添加到当前对象的额外存储空间中
                m_extra_storage.add(dst);
            }
        }

        // 主存储单元
        pod_bvector<T, 12>         m_cells;
        // 额外存储空间向量
        pod_bvector<extra_span, 6> m_extra_storage;
    };






    //-----------------------------------------------scanline_storage_aa
    template<class T> class scanline_storage_aa
    {
    private:
        // 涵盖单元存储
        scanline_cell_storage<T>      m_covers;
        // 跨度数据向量
        pod_bvector<span_data, 10>    m_spans;
        // 扫描线数据向量
        pod_bvector<scanline_data, 8> m_scanlines;
        // 虚拟跨度数据
        span_data     m_fake_span;
        // 虚拟扫描线数据
        scanline_data m_fake_scanline;
        // 最小 X 坐标
        int           m_min_x;
        // 最小 Y 坐标
        int           m_min_y;
        // 最大 X 坐标
        int           m_max_x;
        // 最大 Y 坐标
        int           m_max_y;
        // 当前扫描线索引
        unsigned      m_cur_scanline;
    };


    // 扫描线存储适配器类型定义
    typedef scanline_storage_aa<int8u>  scanline_storage_aa8;  //--------scanline_storage_aa8
    typedef scanline_storage_aa<int16u> scanline_storage_aa16; //--------scanline_storage_aa16
    typedef scanline_storage_aa<int32u> scanline_storage_aa32; //--------scanline_storage_aa32




    //------------------------------------------serialized_scanlines_adaptor_aa
    template<class T> class serialized_scanlines_adaptor_aa
    {
    public:
        //--------------------------------------------------------------------
        // 默认构造函数，初始化成员变量
        serialized_scanlines_adaptor_aa() :
            m_data(0),
            m_end(0),
            m_ptr(0),
            m_dx(0),
            m_dy(0),
            m_min_x(0x7FFFFFFF),
            m_min_y(0x7FFFFFFF),
            m_max_x(-0x7FFFFFFF),
            m_max_y(-0x7FFFFFFF)
        {}

        //--------------------------------------------------------------------
        // 带参数的构造函数，初始化成员变量
        serialized_scanlines_adaptor_aa(const int8u* data, unsigned size,
                                        double dx, double dy) :
            m_data(data),
            m_end(data + size),
            m_ptr(data),
            m_dx(iround(dx)),
            m_dy(iround(dy)),
            m_min_x(0x7FFFFFFF),
            m_min_y(0x7FFFFFFF),
            m_max_x(-0x7FFFFFFF),
            m_max_y(-0x7FFFFFFF)
        {}

        //--------------------------------------------------------------------
        // 初始化函数，用指定参数设置成员变量
        void init(const int8u* data, unsigned size, double dx, double dy)
        {
            m_data  = data;
            m_end   = data + size;
            m_ptr   = data;
            m_dx    = iround(dx);
            m_dy    = iround(dy);
            m_min_x = 0x7FFFFFFF;
            m_min_y = 0x7FFFFFFF;
            m_max_x = -0x7FFFFFFF;
            m_max_y = -0x7FFFFFFF;
        }
    private:
        //--------------------------------------------------------------------
        # 读取一个32位有符号整数
        int read_int32()
        {
            int32 val;
            ((int8u*)&val)[0] = *m_ptr++;  // 将指针当前位置的字节作为最低有效字节，存入整数的第一个字节位置
            ((int8u*)&val)[1] = *m_ptr++;  // 将指针当前位置的字节作为次低有效字节，存入整数的第二个字节位置
            ((int8u*)&val)[2] = *m_ptr++;  // 将指针当前位置的字节作为次高有效字节，存入整数的第三个字节位置
            ((int8u*)&val)[3] = *m_ptr++;  // 将指针当前位置的字节作为最高有效字节，存入整数的第四个字节位置
            return val;  // 返回读取的32位有符号整数
        }

        //--------------------------------------------------------------------
        # 读取一个32位无符号整数
        unsigned read_int32u()
        {
            int32u val;
            ((int8u*)&val)[0] = *m_ptr++;  // 将指针当前位置的字节作为最低有效字节，存入整数的第一个字节位置
            ((int8u*)&val)[1] = *m_ptr++;  // 将指针当前位置的字节作为次低有效字节，存入整数的第二个字节位置
            ((int8u*)&val)[2] = *m_ptr++;  // 将指针当前位置的字节作为次高有效字节，存入整数的第三个字节位置
            ((int8u*)&val)[3] = *m_ptr++;  // 将指针当前位置的字节作为最高有效字节，存入整数的第四个字节位置
            return val;  // 返回读取的32位无符号整数
        }
    // Iterate scanlines interface
    //--------------------------------------------------------------------
    bool rewind_scanlines()
    {
        // 将指针重置为数据起始位置
        m_ptr = m_data;
        // 如果指针位置小于数据结束位置
        if(m_ptr < m_end)
        {
            // 读取并调整最小 x 和 y 坐标值
            m_min_x = read_int32u() + m_dx;
            m_min_y = read_int32u() + m_dy;
            // 读取并调整最大 x 和 y 坐标值
            m_max_x = read_int32u() + m_dx;
            m_max_y = read_int32u() + m_dy;
        }
        // 返回指针是否小于结束位置，用于确定是否还有数据可读
        return m_ptr < m_end;
    }
    
    //--------------------------------------------------------------------
    // 返回最小 x 坐标
    int min_x() const { return m_min_x; }
    // 返回最小 y 坐标
    int min_y() const { return m_min_y; }
    // 返回最大 x 坐标
    int max_x() const { return m_max_x; }
    // 返回最大 y 坐标
    int max_y() const { return m_max_y; }
    
    //--------------------------------------------------------------------
    // 模板方法，处理扫描线数据
    template<class Scanline> bool sweep_scanline(Scanline& sl)
    {
        // 重置扫描线数据
        sl.reset_spans();
        for(;;)
        {
            // 如果指针超过了数据结束位置，则返回 false
            if(m_ptr >= m_end) return false;
    
            // 跳过扫描线字节大小
            read_int32();
            // 读取 y 坐标并调整
            int y = read_int32() + m_dy;
            // 读取扫描线包含的跨度数量
            unsigned num_spans = read_int32();
    
            do
            {
                // 读取 x 坐标并调整
                int x = read_int32() + m_dx;
                // 读取长度信息
                int len = read_int32();
    
                if(len < 0)
                {
                    // 添加跨度数据到扫描线
                    sl.add_span(x, unsigned(-len), *m_ptr);
                    m_ptr += sizeof(T);
                }
                else
                {
                    // 添加单元格数据到扫描线
                    sl.add_cells(x, len, m_ptr);
                    m_ptr += len * sizeof(T);
                }
            }
            while(--num_spans);
    
            // 如果扫描线有跨度数据，则完成该扫描线处理
            if(sl.num_spans())
            {
                sl.finalize(y);
                break;
            }
        }
        // 返回 true 表示还有更多数据可读
        return true;
    }
    
    //--------------------------------------------------------------------
    // 专门为 embedded_scanline 定制的扫描线处理方法
    bool sweep_scanline(embedded_scanline& sl)
    {
        do
        {
            // 如果指针超过了数据结束位置，则返回 false
            if(m_ptr >= m_end) return false;
    
            // 读取字节大小并初始化扫描线
            unsigned byte_size = read_int32u();
            sl.init(m_ptr, m_dx, m_dy);
            // 更新指针位置
            m_ptr += byte_size - sizeof(int32);
        }
        while(sl.num_spans() == 0);
    
        // 返回 true 表示还有更多数据可读
        return true;
    }
    // 定义一个名为 serialized_scanlines_adaptor_aa32 的类型别名，表示它是一个模板化的 serialized_scanlines_adaptor_aa 类型，模板参数为 int32u
    typedef serialized_scanlines_adaptor_aa<int32u> serialized_scanlines_adaptor_aa32; //----serialized_scanlines_adaptor_aa32
}
// 结束一个 C/C++ 的预处理条件指令，对应于上面的 #ifdef 或 #ifndef
#endif
// 结束一个 C/C++ 的条件编译指令，对应于上面的 #if 或 #ifdef 或 #ifndef
```