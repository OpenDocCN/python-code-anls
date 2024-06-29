# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_path_storage.h`

```
    //------------------------------------------------------------------------
    // Anti-Grain Geometry - Version 2.4
    // Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
    //
    // Permission to copy, use, modify, sell and distribute this software 
    // is granted provided this copyright notice appears in all copies. 
    // This software is provided "as is" without express or implied
    // warranty, and with no claim as to its suitability for any purpose.
    //
    //------------------------------------------------------------------------
    // Contact: mcseem@antigrain.com
    //          mcseemagg@yahoo.com
    //          http://www.antigrain.com
    //------------------------------------------------------------------------

    #ifndef AGG_PATH_STORAGE_INCLUDED
    #define AGG_PATH_STORAGE_INCLUDED

    #include <string.h>
    #include <math.h>
    #include "agg_math.h"
    #include "agg_array.h"
    #include "agg_bezier_arc.h"

    namespace agg
    {

        //----------------------------------------------------vertex_block_storage
        // 顶点块存储类模板，用于管理顶点和命令数据的存储
        template<class T, unsigned BlockShift=8, unsigned BlockPool=256>
        class vertex_block_storage
        {
        public:
            // Allocation parameters
            // 分配参数
            enum block_scale_e
            {
                block_shift = BlockShift,    // 块大小的位移量
                block_size  = 1 << block_shift, // 块大小
                block_mask  = block_size - 1,   // 块掩码
                block_pool  = BlockPool       // 块池大小
            };

            typedef T value_type;   // 值类型
            typedef vertex_block_storage<T, BlockShift, BlockPool> self_type; // 类型本身

            ~vertex_block_storage();   // 析构函数
            vertex_block_storage();    // 默认构造函数
            vertex_block_storage(const self_type& v);   // 拷贝构造函数
            const self_type& operator = (const self_type& ps);   // 赋值运算符重载

            void remove_all();  // 移除所有顶点和命令数据
            void free_all();    // 释放所有顶点和命令数据

            void add_vertex(double x, double y, unsigned cmd);   // 添加顶点和命令数据
            void modify_vertex(unsigned idx, double x, double y);   // 修改指定索引的顶点数据
            void modify_vertex(unsigned idx, double x, double y, unsigned cmd);   // 修改指定索引的顶点和命令数据
            void modify_command(unsigned idx, unsigned cmd);   // 修改指定索引的命令数据
            void swap_vertices(unsigned v1, unsigned v2);  // 交换两个顶点的数据

            unsigned last_command() const;  // 获取最后一个命令数据
            unsigned last_vertex(double* x, double* y) const;  // 获取最后一个顶点数据
            unsigned prev_vertex(double* x, double* y) const;  // 获取前一个顶点数据

            double last_x() const;   // 获取最后一个顶点的 x 坐标
            double last_y() const;   // 获取最后一个顶点的 y 坐标

            unsigned total_vertices() const;   // 获取总顶点数
            unsigned vertex(unsigned idx, double* x, double* y) const;   // 获取指定索引的顶点数据
            unsigned command(unsigned idx) const;   // 获取指定索引的命令数据

        private:
            void   allocate_block(unsigned nb);   // 分配块
            int8u* storage_ptrs(T** xy_ptr);   // 存储指针

        private:
            unsigned m_total_vertices;   // 总顶点数
            unsigned m_total_blocks;     // 总块数
            unsigned m_max_blocks;       // 最大块数
            T**      m_coord_blocks;     // 顶点数据块指针数组
            int8u**  m_cmd_blocks;       // 命令数据块指针数组
        };

    }

    #endif // AGG_PATH_STORAGE_INCLUDED



    //------------------------------------------------------------------------
    // Anti-Grain Geometry - Version 2.4
    // Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
    //
    // Permission to copy, use, modify, sell and distribute this software 
    // is granted provided this copyright notice appears in all copies. 
    // This software is provided "as is" without express or implied
    // warranty, and with no claim as to its suitability for any purpose.
    //
    //------------------------------------------------------------------------
    // Contact: mcseem@antigrain.com
    //          mcseemagg@yahoo.com
    //          http://www.antigrain.com
    //------------------------------------------------------------------------

    #ifndef AGG_PATH_STORAGE_INCLUDED
    #define AGG_PATH_STORAGE_INCLUDED

    #include <string.h>
    #include <math.h>
    #include "agg_math.h"
    #include "agg_array.h"
    #include "agg_bezier_arc.h"

    namespace agg
    {

        //----------------------------------------------------vertex_block_storage
        // 顶点块存储类模板，用于管理顶点和命令数据的存储
        template<class T, unsigned BlockShift=8, unsigned BlockPool=256>
        class vertex_block_storage
        {
        public:
            // Allocation parameters
            // 分配参数
            enum block_scale_e
            {
                block_shift = BlockShift,    // 块大小的位移量
                block_size  = 1 << block_shift, // 块大小
                block_mask  = block_size - 1,   // 块掩码
                block_pool  = BlockPool       // 块池大小
            };

            typedef T value_type;   // 值类型
            typedef vertex_block_storage<T, BlockShift, BlockPool> self_type; // 类型本身

            ~vertex_block_storage();   // 析构函数
            vertex_block_storage();    // 默认构造函数
            vertex_block_storage(const self_type& v);   // 拷贝构造函数
            const self_type& operator = (const self_type& ps);   // 赋值运算符重载

            void remove_all();  // 移除所有顶点和命令数据
            void free_all();    // 释放所有顶点和命令数据

            void add_vertex(double x, double y, unsigned cmd);   // 添加顶点和命令数据
            void modify_vertex(unsigned idx, double x, double y);   // 修改指定索引的顶点数据
            void modify_vertex(unsigned idx, double x, double y, unsigned cmd);   // 修改指定索引的顶点和命令数据
            void modify_command(unsigned idx, unsigned cmd);   // 修改指定索引的命令数据
            void swap_vertices(unsigned v1, unsigned v2);  // 交换两个顶点的数据

            unsigned last_command() const;  // 获取最后一个命令数据
            unsigned last_vertex(double* x, double* y) const;  // 获取最后一个顶点数据
            unsigned prev_vertex(double* x, double* y) const;  // 获取前一个顶点数据

            double last_x() const;   // 获取最后一个顶点的 x 坐标
            double last_y() const;   // 获取最后一个顶点的 y 坐标

            unsigned total_vertices() const;   // 获取总顶点数
            unsigned vertex(unsigned idx, double* x, double* y) const;   // 获取指定索引的顶点数据
            unsigned command(unsigned idx) const;   // 获取指定索引的命令数据

        private:
            void   allocate_block(unsigned nb);   // 分配块
            int8u* storage_ptrs(T** xy_ptr);   // 存储指针

        private:
            unsigned m_total_vertices;   // 总顶点数
            unsigned m_total_blocks;     // 总块数
            unsigned m_max_blocks;       // 最大块数
            T**      m_coord_blocks;     // 顶点数据块指针数组
            int8u**  m_cmd_blocks;       // 命令数据块指针数组
        };

    }

    #endif // AGG_PATH_STORAGE_INCLUDED
    {
        // 如果 m_total_blocks 大于 0，则执行以下操作
        if(m_total_blocks)
        {
            // 将 coord_blk 指针指向 m_coord_blocks 数组中最后一个块的位置
            T** coord_blk = m_coord_blocks + m_total_blocks - 1;
            // 循环，处理每个块直到 m_total_blocks 变为 0
            while(m_total_blocks--)
            {
                // 释放 *coord_blk 指向的内存块
                pod_allocator<T>::deallocate(
                    *coord_blk,
                    block_size * 2 + 
                    block_size / (sizeof(T) / sizeof(unsigned char)));
                // 将 coord_blk 指针向前移动一个位置
                --coord_blk;
            }
            // 释放 m_coord_blocks 指向的内存块数组
            pod_allocator<T*>::deallocate(m_coord_blocks, m_max_blocks * 2);
            // 重置所有相关的计数和指针成员变量
            m_total_blocks   = 0;
            m_max_blocks     = 0;
            m_coord_blocks   = 0;
            m_cmd_blocks     = 0;
            m_total_vertices = 0;
        }
    }
    
    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    // vertex_block_storage 类的析构函数定义
    vertex_block_storage<T,S,P>::~vertex_block_storage()
    {
        // 调用 free_all 函数释放所有资源
        free_all();
    }
    
    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    // vertex_block_storage 类的默认构造函数定义
    vertex_block_storage<T,S,P>::vertex_block_storage() :
        // 初始化所有成员变量为 0
        m_total_vertices(0),
        m_total_blocks(0),
        m_max_blocks(0),
        m_coord_blocks(0),
        m_cmd_blocks(0)
    {
    }
    
    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    // vertex_block_storage 类的复制构造函数定义
    vertex_block_storage<T,S,P>::vertex_block_storage(const vertex_block_storage<T,S,P>& v) :
        // 初始化所有成员变量为 0
        m_total_vertices(0),
        m_total_blocks(0),
        m_max_blocks(0),
        m_coord_blocks(0),
        m_cmd_blocks(0)
    {
        // 使用 operator= 进行对象的赋值操作
        *this = v;
    }
    
    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    // vertex_block_storage 类的赋值运算符重载定义
    const vertex_block_storage<T,S,P>& 
    vertex_block_storage<T,S,P>::operator = (const vertex_block_storage<T,S,P>& v)
    {
        // 清空当前对象的所有数据
        remove_all();
        unsigned i;
        // 循环复制 v 中的所有顶点数据到当前对象中
        for(i = 0; i < v.total_vertices(); i++)
        {
            double x, y;
            // 从 v 中获取顶点的坐标和指令
            unsigned cmd = v.vertex(i, &x, &y);
            // 将顶点数据添加到当前对象中
            add_vertex(x, y, cmd);
        }
        return *this;
    }
    
    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    // vertex_block_storage 类的成员函数定义，用于移除所有顶点数据
    inline void vertex_block_storage<T,S,P>::remove_all()
    {
        // 将 m_total_vertices 设置为 0，表示移除所有顶点
        m_total_vertices = 0;
    }
    
    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    // vertex_block_storage 类的成员函数定义，用于添加顶点数据
    inline void vertex_block_storage<T,S,P>::add_vertex(double x, double y, 
                                                        unsigned cmd)
    {
        // 初始化 coord_ptr 指针
        T* coord_ptr = 0;
        // 使用 storage_ptrs 获取 coord_ptr 指向的位置，存储指令值
        *storage_ptrs(&coord_ptr) = (int8u)cmd;
        // 存储 x 和 y 的坐标值到 coord_ptr 指向的位置
        coord_ptr[0] = T(x);
        coord_ptr[1] = T(y);
        // 增加顶点总数计数器
        m_total_vertices++;
    }
    
    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    // 下面的代码块尚未提供完整，无法添加注释
    // 修改顶点的坐标信息
    inline void vertex_block_storage<T,S,P>::modify_vertex(unsigned idx, 
                                                           double x, double y)
    {
        // 计算出顶点在块中的位置
        T* pv = m_coord_blocks[idx >> block_shift] + ((idx & block_mask) << 1);
        // 更新顶点的 x 坐标
        pv[0] = T(x);
        // 更新顶点的 y 坐标
        pv[1] = T(y);
    }
    
    //------------------------------------------------------------------------
    // 修改顶点的坐标信息和命令信息
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::modify_vertex(unsigned idx, 
                                                           double x, double y, 
                                                           unsigned cmd)
    {
        // 计算出顶点所在的块和在块中的偏移量
        unsigned block = idx >> block_shift;
        unsigned offset = idx & block_mask;
        // 获取顶点在块中的地址
        T* pv = m_coord_blocks[block] + (offset << 1);
        // 更新顶点的 x 坐标
        pv[0] = T(x);
        // 更新顶点的 y 坐标
        pv[1] = T(y);
        // 更新顶点的命令信息
        m_cmd_blocks[block][offset] = (int8u)cmd;
    }
    
    //------------------------------------------------------------------------
    // 修改顶点的命令信息
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::modify_command(unsigned idx, 
                                                            unsigned cmd)
    {
        // 更新顶点在命令块中的命令信息
        m_cmd_blocks[idx >> block_shift][idx & block_mask] = (int8u)cmd;
    }
    
    //------------------------------------------------------------------------
    // 交换两个顶点的坐标信息和命令信息
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::swap_vertices(unsigned v1, unsigned v2)
    {
        // 计算两个顶点所在的块和在块中的偏移量
        unsigned b1 = v1 >> block_shift;
        unsigned b2 = v2 >> block_shift;
        unsigned o1 = v1 & block_mask;
        unsigned o2 = v2 & block_mask;
        // 获取两个顶点在块中的地址
        T* pv1 = m_coord_blocks[b1] + (o1 << 1);
        T* pv2 = m_coord_blocks[b2] + (o2 << 1);
        T  val;
        // 交换顶点1和顶点2的 x 坐标
        val = pv1[0]; pv1[0] = pv2[0]; pv2[0] = val;
        // 交换顶点1和顶点2的 y 坐标
        val = pv1[1]; pv1[1] = pv2[1]; pv2[1] = val;
        // 交换顶点1和顶点2的命令信息
        int8u cmd = m_cmd_blocks[b1][o1];
        m_cmd_blocks[b1][o1] = m_cmd_blocks[b2][o2];
        m_cmd_blocks[b2][o2] = cmd;
    }
    
    //------------------------------------------------------------------------
    // 返回最后一个顶点的命令信息
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::last_command() const
    {
        // 如果有顶点存在，返回最后一个顶点的命令信息；否则返回停止命令
        if(m_total_vertices) return command(m_total_vertices - 1);
        return path_cmd_stop;
    }
    
    //------------------------------------------------------------------------
    // 返回最后一个顶点的坐标信息
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::last_vertex(double* x, double* y) const
    {
        // 如果有顶点存在，返回最后一个顶点的坐标信息；否则返回停止命令
        if(m_total_vertices) return vertex(m_total_vertices - 1, x, y);
        return path_cmd_stop;
    }
    
    //------------------------------------------------------------------------
    // 返回倒数第二个顶点的坐标信息
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::prev_vertex(double* x, double* y) const
    {
        // 如果顶点数大于1，返回倒数第二个顶点的坐标信息；否则返回停止命令
        if(m_total_vertices > 1) return vertex(m_total_vertices - 2, x, y);
        return path_cmd_stop;
    }
    //------------------------------------------------------------------------
    // 返回最后一个顶点的 x 坐标
    template<class T, unsigned S, unsigned P>
    inline double vertex_block_storage<T,S,P>::last_x() const
    {
        // 如果总顶点数不为零
        if(m_total_vertices)
        {
            // 计算最后一个顶点的索引
            unsigned idx = m_total_vertices - 1;
            // 返回最后一个顶点的 x 坐标
            return m_coord_blocks[idx >> block_shift][(idx & block_mask) << 1];
        }
        // 总顶点数为零时返回0.0
        return 0.0;
    }
    
    //------------------------------------------------------------------------
    // 返回最后一个顶点的 y 坐标
    template<class T, unsigned S, unsigned P>
    inline double vertex_block_storage<T,S,P>::last_y() const
    {
        // 如果总顶点数不为零
        if(m_total_vertices)
        {
            // 计算最后一个顶点的索引
            unsigned idx = m_total_vertices - 1;
            // 返回最后一个顶点的 y 坐标
            return m_coord_blocks[idx >> block_shift][((idx & block_mask) << 1) + 1];
        }
        // 总顶点数为零时返回0.0
        return 0.0;
    }
    
    //------------------------------------------------------------------------
    // 返回总顶点数
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::total_vertices() const
    {
        return m_total_vertices;
    }
    
    //------------------------------------------------------------------------
    // 返回指定索引的顶点坐标，并通过指针 x 和 y 返回其坐标值，同时返回命令值
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::vertex(unsigned idx, 
                                                        double* x, double* y) const
    {
        // 计算顶点所在块的索引
        unsigned nb = idx >> block_shift;
        // 获取指定顶点在块中的指针
        const T* pv = m_coord_blocks[nb] + ((idx & block_mask) << 1);
        // 返回顶点的 x 和 y 坐标值
        *x = pv[0];
        *y = pv[1];
        // 返回顶点对应的命令值
        return m_cmd_blocks[nb][idx & block_mask];
    }
    
    //------------------------------------------------------------------------
    // 返回指定索引处的命令值
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::command(unsigned idx) const
    {
        // 计算命令所在块的索引
        return m_cmd_blocks[idx >> block_shift][idx & block_mask];
    }
    
    //------------------------------------------------------------------------
    // 分配指定数量的块
    template<class T, unsigned S, unsigned P>
    void vertex_block_storage<T,S,P>::allocate_block(unsigned nb)
    {
        // 检查当前块数量是否超过最大块数，如果超过则进行扩展
        if(nb >= m_max_blocks) 
        {
            // 分配新的块内存给坐标数组
            T** new_coords = 
                pod_allocator<T*>::allocate((m_max_blocks + block_pool) * 2);

            // 分配新的块内存给命令数组，并转换为unsigned char指针数组
            unsigned char** new_cmds = 
                (unsigned char**)(new_coords + m_max_blocks + block_pool);

            // 如果之前已有块数据，将原有坐标和命令数据复制到新的块内存
            if(m_coord_blocks)
            {
                memcpy(new_coords, 
                       m_coord_blocks, 
                       m_max_blocks * sizeof(T*));

                memcpy(new_cmds, 
                       m_cmd_blocks, 
                       m_max_blocks * sizeof(unsigned char*));

                // 释放原有块内存
                pod_allocator<T*>::deallocate(m_coord_blocks, m_max_blocks * 2);
            }

            // 更新坐标和命令数组为新分配的内存块
            m_coord_blocks = new_coords;
            m_cmd_blocks   = new_cmds;
            // 增加最大块数量
            m_max_blocks  += block_pool;
        }

        // 分配新的块内存给坐标数组中的当前块
        m_coord_blocks[nb] = 
            pod_allocator<T>::allocate(block_size * 2 + 
                   block_size / (sizeof(T) / sizeof(unsigned char)));

        // 设置当前块的命令数组指针
        m_cmd_blocks[nb]  = 
            (unsigned char*)(m_coord_blocks[nb] + block_size * 2);

        // 增加总块数计数
        m_total_blocks++;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    int8u* vertex_block_storage<T,S,P>::storage_ptrs(T** xy_ptr)
    {
        // 计算当前顶点数量所在的块索引
        unsigned nb = m_total_vertices >> block_shift;
        
        // 如果当前块索引超过已分配的总块数，则分配新的块
        if(nb >= m_total_blocks)
        {
            allocate_block(nb);
        }

        // 设置返回参数xy_ptr为当前块中的坐标数组指针偏移
        *xy_ptr = m_coord_blocks[nb] + ((m_total_vertices & block_mask) << 1);
        
        // 返回当前块中的命令数组指针偏移
        return m_cmd_blocks[nb] + (m_total_vertices & block_mask);
    }
    // 定义公共部分：类型别名 value_type 设定为 T
    public:
        typedef T value_type;

        // 默认构造函数：初始化所有成员变量为默认值
        poly_plain_adaptor() : 
            m_data(0),                     // m_data 指针初始化为 0
            m_ptr(0),                      // m_ptr 指针初始化为 0
            m_end(0),                      // m_end 指针初始化为 0
            m_closed(false),               // m_closed 布尔值初始化为 false
            m_stop(false)                  // m_stop 布尔值初始化为 false
        {}

        // 带参数的构造函数：使用给定的数据指针、点数和闭合标志来初始化对象
        poly_plain_adaptor(const T* data, unsigned num_points, bool closed) :
            m_data(data),                  // 使用给定的 data 指针初始化 m_data
            m_ptr(data),                   // 将 m_ptr 指针指向 data
            m_end(data + num_points * 2),  // 计算数据的结束位置，并初始化 m_end
            m_closed(closed),              // 使用给定的闭合标志初始化 m_closed
            m_stop(false)                  // m_stop 布尔值初始化为 false
        {}

        // 初始化函数：重新设置对象的数据指针、点数和闭合标志
        void init(const T* data, unsigned num_points, bool closed)
        {
            m_data = data;                 // 使用给定的 data 指针设置 m_data
            m_ptr = data;                  // 将 m_ptr 指针指向 data
            m_end = data + num_points * 2; // 计算数据的结束位置，并设置 m_end
            m_closed = closed;             // 使用给定的闭合标志设置 m_closed
            m_stop = false;                // 将 m_stop 重新设为 false
        }

        // 重置函数：将指针 m_ptr 重新指向 m_data，并将 m_stop 设为 false
        void rewind(unsigned)
        {
            m_ptr = m_data;                // 将 m_ptr 指针重新指向 m_data
            m_stop = false;                // 将 m_stop 设为 false
        }

        // 获取顶点函数：将当前顶点的坐标存入 x 和 y 中，并返回适当的路径命令
        unsigned vertex(double* x, double* y)
        {
            if(m_ptr < m_end)              // 如果当前指针 m_ptr 小于结束指针 m_end
            {
                bool first = m_ptr == m_data;  // 检查是否为第一个顶点
                *x = *m_ptr++;              // 将当前顶点的 x 坐标存入 x 中，并移动指针 m_ptr
                *y = *m_ptr++;              // 将当前顶点的 y 坐标存入 y 中，并再次移动指针 m_ptr
                return first ? path_cmd_move_to : path_cmd_line_to;  // 如果是第一个顶点返回移动命令，否则返回直线命令
            }
            *x = *y = 0.0;                  // 如果没有更多顶点，则将 x 和 y 设置为 0
            if(m_closed && !m_stop)         // 如果设置了闭合且未停止
            {
                m_stop = true;              // 设置停止标志为 true
                return path_cmd_end_poly | path_flags_close;  // 返回结束多边形的命令，并标记为闭合
            }
            return path_cmd_stop;           // 返回停止命令
        }

    private:
        const T* m_data;                    // 指向数据的常量指针
        const T* m_ptr;                     // 指向当前顶点的常量指针
        const T* m_end;                     // 指向数据结束位置的常量指针
        bool     m_closed;                  // 表示多边形是否闭合的布尔值
        bool     m_stop;                    // 表示是否停止获取顶点的布尔值
    };
    private:
        const Container* m_container;  // 指向常量容器的指针，存储顶点数据
        unsigned m_index;              // 当前顶点在容器中的索引
        bool m_closed;                 // 标记路径是否闭合
        bool m_stop;                   // 标记是否停止遍历路径
    };



    //-----------------------------------------poly_container_reverse_adaptor
    template<class Container> class poly_container_reverse_adaptor
    {
    public:
        typedef typename Container::value_type vertex_type;

        poly_container_reverse_adaptor() : 
            m_container(0),                // 初始化容器指针为nullptr
            m_index(-1),                   // 初始化索引为-1
            m_closed(false),               // 初始化路径闭合标志为false
            m_stop(false)                  // 初始化停止标志为false
        {}

        poly_container_reverse_adaptor(Container& data, bool closed) :
            m_container(&data),            // 使用给定的数据容器初始化容器指针
            m_index(-1),                   // 初始化索引为-1
            m_closed(closed),              // 初始化路径闭合标志为给定的闭合状态
            m_stop(false)                  // 初始化停止标志为false
        {}

        void init(Container& data, bool closed)
        {
            m_container = &data;           // 初始化容器指针为给定的数据容器
            m_index = m_container->size() - 1;  // 初始化索引为容器大小减一
            m_closed = closed;             // 初始化路径闭合标志为给定的闭合状态
            m_stop = false;                // 初始化停止标志为false
        }

        void rewind(unsigned)
        {
            m_index = m_container->size() - 1;  // 将索引重置为容器大小减一，即末尾索引
            m_stop = false;                // 重置停止标志为false
        }

        unsigned vertex(double* x, double* y)
        {
            if(m_index >= 0)
            {
                bool first = m_index == int(m_container->size() - 1);  // 判断是否是第一个顶点
                const vertex_type& v = (*m_container)[m_index--];       // 获取当前顶点并递减索引
                *x = v.x;                   // 将顶点的 x 坐标赋给传入的 x 指针
                *y = v.y;                   // 将顶点的 y 坐标赋给传入的 y 指针
                return first ? path_cmd_move_to : path_cmd_line_to;     // 返回操作命令，移动到或者直线到
            }
            *x = *y = 0.0;                  // 当索引小于0时，将 x 和 y 坐标置为0
            if(m_closed && !m_stop)
            {
                m_stop = true;              // 标记已经停止遍历
                return path_cmd_end_poly | path_flags_close;           // 返回多边形结束命令并标记闭合
            }
            return path_cmd_stop;           // 返回停止路径命令
        }

    private:
        Container* m_container;            // 指向容器的指针，存储顶点数据
        int m_index;                       // 当前顶点在容器中的索引
        bool m_closed;                     // 标记路径是否闭合
        bool m_stop;                       // 标记是否停止遍历路径
    };





    //--------------------------------------------------------line_adaptor
    class line_adaptor
    {
    public:
        typedef double value_type;

        line_adaptor() : m_line(m_coord, 2, false) {}   // 使用坐标数组初始化线段适配器
        line_adaptor(double x1, double y1, double x2, double y2) :
            m_line(m_coord, 2, false)                    // 使用坐标数组初始化线段适配器
        {
            m_coord[0] = x1;                            // 初始化第一个顶点的 x 坐标
            m_coord[1] = y1;                            // 初始化第一个顶点的 y 坐标
            m_coord[2] = x2;                            // 初始化第二个顶点的 x 坐标
            m_coord[3] = y2;                            // 初始化第二个顶点的 y 坐标
        }
        
        void init(double x1, double y1, double x2, double y2)
        {
            m_coord[0] = x1;                            // 初始化第一个顶点的 x 坐标
            m_coord[1] = y1;                            // 初始化第一个顶点的 y 坐标
            m_coord[2] = x2;                            // 初始化第二个顶点的 x 坐标
            m_coord[3] = y2;                            // 初始化第二个顶点的 y 坐标
            m_line.rewind(0);                           // 重置线段适配器的状态
        }

        void rewind(unsigned)
        {
            m_line.rewind(0);                           // 重置线段适配器的状态
        }

        unsigned vertex(double* x, double* y)
        {
            return m_line.vertex(x, y);                 // 获取线段适配器的下一个顶点
        }

    private:
        double m_coord[4];                              // 存储线段两个顶点的坐标数组
        poly_plain_adaptor<double> m_line;               // 用坐标数组初始化的线段适配器
    };













    //---------------------------------------------------------------path_base
    // 用于存储带有标志的顶点的容器。
    // 路径由多个以“move_to”分隔的轮廓组成。
    // commands. The path storage can keep and maintain more than one
    // path. 
    // To navigate to the beginning of a particular path, use rewind(path_id);
    // Where path_id is what start_new_path() returns. So, when you call
    // start_new_path() you need to store its return value somewhere else
    // to navigate to the path afterwards.
    //
    // See also: vertex_source concept
    //------------------------------------------------------------------------
    // 定义一个模板类 path_base，使用 VertexContainer 作为顶点容器类型
    template<class VertexContainer> class path_base
    {
    private:
        // 声明私有成员函数，用于判断多边形的方向
        unsigned perceive_polygon_orientation(unsigned start, unsigned end);
        // 声明私有成员函数，用于反转多边形的方向
        void invert_polygon(unsigned start, unsigned end);
    
        // 成员变量，顶点容器
        VertexContainer m_vertices;
        // 迭代器位置
        unsigned m_iterator;
    };
    
    //------------------------------------------------------------------------
    // path_base 模板类的成员函数定义：start_new_path()
    template<class VC> 
    unsigned path_base<VC>::start_new_path()
    {
        // 如果顶点容器的最后一个命令不是停止命令，则添加一个停止命令
        if(!is_stop(m_vertices.last_command()))
        {
            m_vertices.add_vertex(0.0, 0.0, path_cmd_stop);
        }
        // 返回顶点总数作为 path_id
        return m_vertices.total_vertices();
    }
    
    //------------------------------------------------------------------------
    // path_base 模板类的成员函数定义：rel_to_abs()
    template<class VC> 
    inline void path_base<VC>::rel_to_abs(double* x, double* y) const
    {
        // 如果顶点容器中有顶点
        if(m_vertices.total_vertices())
        {
            double x2;
            double y2;
            // 获取最后一个顶点的坐标到 (x2, y2)
            if(is_vertex(m_vertices.last_vertex(&x2, &y2)))
            {
                // 将传入的相对坐标 (*x, *y) 转换为绝对坐标
                *x += x2;
                *y += y2;
            }
        }
    }
    
    //------------------------------------------------------------------------
    // path_base 模板类的成员函数定义：move_to()
    template<class VC> 
    inline void path_base<VC>::move_to(double x, double y)
    {
        // 向顶点容器中添加一个移动到命令，位置为 (x, y)
        m_vertices.add_vertex(x, y, path_cmd_move_to);
    }
    
    //------------------------------------------------------------------------
    // path_base 模板类的成员函数定义：move_rel()
    template<class VC> 
    inline void path_base<VC>::move_rel(double dx, double dy)
    {
        // 将传入的相对坐标 (dx, dy) 转换为绝对坐标
        rel_to_abs(&dx, &dy);
        // 向顶点容器中添加一个移动到命令，位置为 (dx, dy)
        m_vertices.add_vertex(dx, dy, path_cmd_move_to);
    }
    
    //------------------------------------------------------------------------
    // path_base 模板类的成员函数定义：line_to()
    template<class VC> 
    inline void path_base<VC>::line_to(double x, double y)
    {
        // 向顶点容器中添加一个直线到命令，位置为 (x, y)
        m_vertices.add_vertex(x, y, path_cmd_line_to);
    }
    
    //------------------------------------------------------------------------
    // path_base 模板类的成员函数定义：line_rel()
    template<class VC> 
    inline void path_base<VC>::line_rel(double dx, double dy)
    {
        // 将传入的相对坐标 (dx, dy) 转换为绝对坐标
        rel_to_abs(&dx, &dy);
        // 向顶点容器中添加一个直线到命令，位置为 (dx, dy)
        m_vertices.add_vertex(dx, dy, path_cmd_line_to);
    }
    
    //------------------------------------------------------------------------
    // path_base 模板类的成员函数定义：hline_to()
    template<class VC> 
    inline void path_base<VC>::hline_to(double x)
    {
        // 向顶点容器中添加一个水平线到命令，位置的 y 坐标为当前最后一个顶点的 y 坐标
        m_vertices.add_vertex(x, last_y(), path_cmd_line_to);
    }
    
    //------------------------------------------------------------------------
    // path_base 模板类的成员函数定义：hline_rel()
    {
        // 将相对坐标转换为绝对坐标，并添加到顶点列表中
        double dy = 0;
        rel_to_abs(&dx, &dy);
        m_vertices.add_vertex(dx, dy, path_cmd_line_to);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::vline_to(double y)
    {
        // 添加垂直线到顶点列表中，使用当前的 X 坐标和指定的 Y 坐标
        m_vertices.add_vertex(last_x(), y, path_cmd_line_to);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::vline_rel(double dy)
    {
        // 将相对的垂直线坐标转换为绝对坐标，并添加到顶点列表中
        double dx = 0;
        rel_to_abs(&dx, &dy);
        m_vertices.add_vertex(dx, dy, path_cmd_line_to);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::arc_to(double rx, double ry,
                               double angle,
                               bool large_arc_flag,
                               bool sweep_flag,
                               double x, double y)
    {
        // 如果顶点列表中有顶点，并且最后一个命令是顶点
        if(m_vertices.total_vertices() && is_vertex(m_vertices.last_command()))
        {
            const double epsilon = 1e-30;
            double x0 = 0.0;
            double y0 = 0.0;
            m_vertices.last_vertex(&x0, &y0);
    
            rx = fabs(rx);
            ry = fabs(ry);
    
            // 确保半径有效
            //-------------------------
            if(rx < epsilon || ry < epsilon) 
            {
                // 如果半径太小，直接画直线到指定的点
                line_to(x, y);
                return;
            }
    
            if(calc_distance(x0, y0, x, y) < epsilon)
            {
                // 如果起点和终点重合，则不添加椭圆弧段
                return;
            }
    
            // 创建 SVG 格式的贝塞尔曲线椭圆弧对象
            bezier_arc_svg a(x0, y0, rx, ry, angle, large_arc_flag, sweep_flag, x, y);
            if(a.radii_ok())
            {
                // 如果椭圆弧参数有效，将其添加到路径中
                join_path(a);
            }
            else
            {
                // 否则，画直线到指定的点
                line_to(x, y);
            }
        }
        else
        {
            // 如果没有顶点或者最后一个命令不是顶点，则将起点移到指定的点
            move_to(x, y);
        }
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::arc_rel(double rx, double ry,
                                double angle,
                                bool large_arc_flag,
                                bool sweep_flag,
                                double dx, double dy)
    {
        // 将相对的椭圆弧参数转换为绝对坐标，并调用 arc_to 函数处理
        rel_to_abs(&dx, &dy);
        arc_to(rx, ry, angle, large_arc_flag, sweep_flag, dx, dy);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve3(double x_ctrl, double y_ctrl, 
                               double x_to,   double y_to)
    {
        // 添加三次贝塞尔曲线控制点和终点到顶点列表中
        m_vertices.add_vertex(x_ctrl, y_ctrl, path_cmd_curve3);
        m_vertices.add_vertex(x_to,   y_to,   path_cmd_curve3);
    }
    
    //------------------------------------------------------------------------
    // 定义一个模板函数，用于添加一个三次贝塞尔曲线的控制点和终点
    template<class VC> 
    void path_base<VC>::curve3_rel(double dx_ctrl, double dy_ctrl, 
                                   double dx_to,   double dy_to)
    {
        // 将相对坐标转换为绝对坐标
        rel_to_abs(&dx_ctrl, &dy_ctrl);
        rel_to_abs(&dx_to,   &dy_to);
        // 添加控制点和终点到顶点列表中
        m_vertices.add_vertex(dx_ctrl, dy_ctrl, path_cmd_curve3);
        m_vertices.add_vertex(dx_to,   dy_to,   path_cmd_curve3);
    }

    //------------------------------------------------------------------------
    // 定义一个模板函数，用于添加一个三次贝塞尔曲线的终点
    template<class VC> 
    void path_base<VC>::curve3(double x_to, double y_to)
    {
        double x0;
        double y0;
        // 获取最后一个顶点的坐标
        if(is_vertex(m_vertices.last_vertex(&x0, &y0)))
        {
            double x_ctrl;
            double y_ctrl; 
            unsigned cmd = m_vertices.prev_vertex(&x_ctrl, &y_ctrl);
            // 如果前一个顶点是三次贝塞尔曲线的控制点，则计算新的控制点坐标
            if(is_curve(cmd))
            {
                x_ctrl = x0 + x0 - x_ctrl;
                y_ctrl = y0 + y0 - y_ctrl;
            }
            else
            {
                x_ctrl = x0;
                y_ctrl = y0;
            }
            // 添加三次贝塞尔曲线的控制点和终点到顶点列表中
            curve3(x_ctrl, y_ctrl, x_to, y_to);
        }
    }

    //------------------------------------------------------------------------
    // 定义一个模板函数，用于添加一个相对坐标的三次贝塞尔曲线的终点
    template<class VC> 
    void path_base<VC>::curve3_rel(double dx_to, double dy_to)
    {
        // 将相对坐标转换为绝对坐标
        rel_to_abs(&dx_to, &dy_to);
        // 添加三次贝塞尔曲线的终点到顶点列表中
        curve3(dx_to, dy_to);
    }

    //------------------------------------------------------------------------
    // 定义一个模板函数，用于添加一个四次贝塞尔曲线的控制点和终点
    template<class VC> 
    void path_base<VC>::curve4(double x_ctrl1, double y_ctrl1, 
                               double x_ctrl2, double y_ctrl2, 
                               double x_to,    double y_to)
    {
        // 添加两个控制点和一个终点到顶点列表中
        m_vertices.add_vertex(x_ctrl1, y_ctrl1, path_cmd_curve4);
        m_vertices.add_vertex(x_ctrl2, y_ctrl2, path_cmd_curve4);
        m_vertices.add_vertex(x_to,    y_to,    path_cmd_curve4);
    }

    //------------------------------------------------------------------------
    // 定义一个模板函数，用于添加一个相对坐标的四次贝塞尔曲线的控制点和终点
    template<class VC> 
    void path_base<VC>::curve4_rel(double dx_ctrl1, double dy_ctrl1, 
                                   double dx_ctrl2, double dy_ctrl2, 
                                   double dx_to,    double dy_to)
    {
        // 将相对坐标转换为绝对坐标
        rel_to_abs(&dx_ctrl1, &dy_ctrl1);
        rel_to_abs(&dx_ctrl2, &dy_ctrl2);
        rel_to_abs(&dx_to,    &dy_to);
        // 添加两个控制点和一个终点到顶点列表中
        m_vertices.add_vertex(dx_ctrl1, dy_ctrl1, path_cmd_curve4);
        m_vertices.add_vertex(dx_ctrl2, dy_ctrl2, path_cmd_curve4);
        m_vertices.add_vertex(dx_to,    dy_to,    path_cmd_curve4);
    }

    //------------------------------------------------------------------------
    // 定义一个模板函数，用于添加一个四次贝塞尔曲线的控制点和终点
    template<class VC> 
    void path_base<VC>::curve4(double x_ctrl2, double y_ctrl2, 
                               double x_to,    double y_to)
    {
        // 声明变量：x0、y0，用于存储最后一个顶点的坐标
        double x0;
        double y0;
        // 如果最后一个顶点存在
        if(is_vertex(last_vertex(&x0, &y0)))
        {
            // 声明变量：x_ctrl1、y_ctrl1，用于存储控制点1的坐标
            double x_ctrl1;
            double y_ctrl1; 
            // 获取前一个顶点的命令，并存储到cmd中
            unsigned cmd = prev_vertex(&x_ctrl1, &y_ctrl1);
            // 如果前一个命令是曲线命令
            if(is_curve(cmd))
            {
                // 根据对称性计算控制点1的新坐标
                x_ctrl1 = x0 + x0 - x_ctrl1;
                y_ctrl1 = y0 + y0 - y_ctrl1;
            }
            else
            {
                // 否则控制点1与当前顶点相同
                x_ctrl1 = x0;
                y_ctrl1 = y0;
            }
            // 调用curve4函数，传入控制点1的坐标、控制点2的坐标、目标点的坐标
            curve4(x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x_to, y_to);
        }
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve4_rel(double dx_ctrl2, double dy_ctrl2, 
                                   double dx_to,    double dy_to)
    {
        // 将相对坐标转换为绝对坐标
        rel_to_abs(&dx_ctrl2, &dy_ctrl2);
        rel_to_abs(&dx_to,    &dy_to);
        // 调用curve4函数，传入控制点2的坐标、目标点的坐标
        curve4(dx_ctrl2, dy_ctrl2, dx_to, dy_to);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::end_poly(unsigned flags)
    {
        // 如果最后一个顶点是顶点命令，则添加一个终结多边形命令
        if(is_vertex(m_vertices.last_command()))
        {
            m_vertices.add_vertex(0.0, 0.0, path_cmd_end_poly | flags);
        }
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::close_polygon(unsigned flags)
    {
        // 调用end_poly函数，并传入关闭多边形的标志
        end_poly(path_flags_close | flags);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::total_vertices() const
    {
        // 返回顶点总数
        return m_vertices.total_vertices();
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::last_vertex(double* x, double* y) const
    {
        // 获取最后一个顶点的坐标，并返回其命令
        return m_vertices.last_vertex(x, y);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::prev_vertex(double* x, double* y) const
    {
        // 获取前一个顶点的坐标，并返回其命令
        return m_vertices.prev_vertex(x, y);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline double path_base<VC>::last_x() const
    {
        // 返回最后一个顶点的x坐标
        return m_vertices.last_x();
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline double path_base<VC>::last_y() const
    {
        // 返回最后一个顶点的y坐标
        return m_vertices.last_y();
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::vertex(unsigned idx, double* x, double* y) const
    {
        // 获取指定索引的顶点的坐标，并返回其命令
        return m_vertices.vertex(idx, x, y);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::command(unsigned idx) const
    {
        return m_vertices.command(idx);
    }


    // 返回指定索引位置顶点的命令
    return m_vertices.command(idx);



    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::modify_vertex(unsigned idx, double x, double y)
    {
        m_vertices.modify_vertex(idx, x, y);
    }


    // 修改指定索引位置的顶点坐标
    m_vertices.modify_vertex(idx, x, y);



    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::modify_vertex(unsigned idx, double x, double y, unsigned cmd)
    {
        m_vertices.modify_vertex(idx, x, y, cmd);
    }


    // 修改指定索引位置的顶点坐标及命令
    m_vertices.modify_vertex(idx, x, y, cmd);



    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::modify_command(unsigned idx, unsigned cmd)
    {
        m_vertices.modify_command(idx, cmd);
    }


    // 修改指定索引位置的顶点命令
    m_vertices.modify_command(idx, cmd);



    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::rewind(unsigned path_id)
    {
        m_iterator = path_id;
    }


    // 将迭代器设置为指定路径ID
    m_iterator = path_id;



    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::vertex(double* x, double* y)
    {
        if(m_iterator >= m_vertices.total_vertices()) return path_cmd_stop;
        return m_vertices.vertex(m_iterator++, x, y);
    }


    // 获取当前迭代器指向的顶点坐标
    if(m_iterator >= m_vertices.total_vertices()) return path_cmd_stop;
    return m_vertices.vertex(m_iterator++, x, y);



    //------------------------------------------------------------------------
    template<class VC> 
    unsigned path_base<VC>::perceive_polygon_orientation(unsigned start,
                                                         unsigned end)
    {
        // 计算多边形的方向（顺时针或逆时针）
        //---------------------
        unsigned np = end - start;
        double area = 0.0;
        unsigned i;
        for(i = 0; i < np; i++)
        {
            double x1, y1, x2, y2;
            m_vertices.vertex(start + i,            &x1, &y1);
            m_vertices.vertex(start + (i + 1) % np, &x2, &y2);
            area += x1 * y2 - y1 * x2;
        }
        return (area < 0.0) ? path_flags_cw : path_flags_ccw;
    }


    // 感知多边形的方向（顺时针或逆时针）
    //---------------------
    unsigned np = end - start;
    double area = 0.0;
    unsigned i;
    for(i = 0; i < np; i++)
    {
        double x1, y1, x2, y2;
        m_vertices.vertex(start + i,            &x1, &y1);
        m_vertices.vertex(start + (i + 1) % np, &x2, &y2);
        area += x1 * y2 - y1 * x2;
    }
    return (area < 0.0) ? path_flags_cw : path_flags_ccw;



    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::invert_polygon(unsigned start, unsigned end)
    {
        unsigned i;
        unsigned tmp_cmd = m_vertices.command(start);
        
        --end; // Make "end" inclusive

        // Shift all commands to one position
        for(i = start; i < end; i++)
        {
            m_vertices.modify_command(i, m_vertices.command(i + 1));
        }

        // Assign starting command to the ending command
        m_vertices.modify_command(end, tmp_cmd);

        // Reverse the polygon
        while(end > start)
        {
            m_vertices.swap_vertices(start++, end--);
        }
    }


    // 反转多边形顶点顺序
    unsigned i;
    unsigned tmp_cmd = m_vertices.command(start);
    
    --end; // Make "end" inclusive

    // 将所有顶点命令向前移动一位
    for(i = start; i < end; i++)
    {
        m_vertices.modify_command(i, m_vertices.command(i + 1));
    }

    // 将起始顶点命令赋给结束顶点
    m_vertices.modify_command(end, tmp_cmd);

    // 反转多边形顶点顺序
    while(end > start)
    {
        m_vertices.swap_vertices(start++, end--);
    }



    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::invert_polygon(unsigned start)


    // 反转从指定起始索引开始的多边形顶点顺序
    {
        // 跳过起始处所有非顶点
        while(start < m_vertices.total_vertices() && 
              !is_vertex(m_vertices.command(start))) ++start;
    
        // 跳过所有不重要的 move_to
        while(start+1 < m_vertices.total_vertices() && 
              is_move_to(m_vertices.command(start)) &&
              is_move_to(m_vertices.command(start+1))) ++start;
    
        // 找到最后一个顶点
        unsigned end = start + 1;
        while(end < m_vertices.total_vertices() && 
              !is_next_poly(m_vertices.command(end))) ++end;
    
        // 反转多边形顶点顺序
        invert_polygon(start, end);
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    unsigned path_base<VC>::arrange_polygon_orientation(unsigned start, 
                                                        path_flags_e orientation)
    {
        if(orientation == path_flags_none) return start;
        
        // 跳过起始处所有非顶点
        while(start < m_vertices.total_vertices() && 
              !is_vertex(m_vertices.command(start))) ++start;
    
        // 跳过所有不重要的 move_to
        while(start+1 < m_vertices.total_vertices() && 
              is_move_to(m_vertices.command(start)) &&
              is_move_to(m_vertices.command(start+1))) ++start;
    
        // 找到最后一个顶点
        unsigned end = start + 1;
        while(end < m_vertices.total_vertices() && 
              !is_next_poly(m_vertices.command(end))) ++end;
    
        if(end - start > 2)
        {
            // 如果多边形顶点数大于2，则检查方向是否需要调整
            if(perceive_polygon_orientation(start, end) != unsigned(orientation))
            {
                // 反转多边形顶点顺序，并设置方向标志，跳过所有 end_poly
                invert_polygon(start, end);
                unsigned cmd;
                while(end < m_vertices.total_vertices() && 
                      is_end_poly(cmd = m_vertices.command(end)))
                {
                    m_vertices.modify_command(end++, set_orientation(cmd, orientation));
                }
            }
        }
        return end;
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    unsigned path_base<VC>::arrange_orientations(unsigned start, 
                                                 path_flags_e orientation)
    {
        // 如果方向不是 path_flags_none
        if(orientation != path_flags_none)
        {
            while(start < m_vertices.total_vertices())
            {
                // 调整多边形顶点顺序，并更新起始位置
                start = arrange_polygon_orientation(start, orientation);
                // 如果遇到 stop 命令，则跳过
                if(is_stop(m_vertices.command(start)))
                {
                    ++start;
                    break;
                }
            }
        }
        return start;
    }
    
    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::arrange_orientations_all_paths(path_flags_e orientation)
    {
        // 如果传入的方向不是无效标志，则执行以下操作
        if(orientation != path_flags_none)
        {
            // 设置起始索引为0
            unsigned start = 0;
            // 循环直到处理完所有顶点
            while(start < m_vertices.total_vertices())
            {
                // 调用函数按照给定方向重新排列顶点，并返回下一个起始索引
                start = arrange_orientations(start, orientation);
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::flip_x(double x1, double x2)
    {
        // 声明变量
        unsigned i;
        double x, y;
        // 遍历所有顶点
        for(i = 0; i < m_vertices.total_vertices(); i++)
        {
            // 获取顶点的命令和坐标
            unsigned cmd = m_vertices.vertex(i, &x, &y);
            // 如果是顶点命令
            if(is_vertex(cmd))
            {
                // 修改顶点的 x 坐标以实现水平翻转
                m_vertices.modify_vertex(i, x2 - x + x1, y);
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::flip_y(double y1, double y2)
    {
        // 声明变量
        unsigned i;
        double x, y;
        // 遍历所有顶点
        for(i = 0; i < m_vertices.total_vertices(); i++)
        {
            // 获取顶点的命令和坐标
            unsigned cmd = m_vertices.vertex(i, &x, &y);
            // 如果是顶点命令
            if(is_vertex(cmd))
            {
                // 修改顶点的 y 坐标以实现垂直翻转
                m_vertices.modify_vertex(i, x, y2 - y + y1);
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::translate(double dx, double dy, unsigned path_id)
    {
        // 获取顶点总数
        unsigned num_ver = m_vertices.total_vertices();
        // 从指定路径索引开始遍历顶点
        for(; path_id < num_ver; path_id++)
        {
            double x, y;
            // 获取顶点的命令和坐标
            unsigned cmd = m_vertices.vertex(path_id, &x, &y);
            // 如果遇到路径结束命令则退出循环
            if(is_stop(cmd)) break;
            // 如果是顶点命令
            if(is_vertex(cmd))
            {
                // 对顶点坐标进行平移操作
                x += dx;
                y += dy;
                m_vertices.modify_vertex(path_id, x, y);
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::translate_all_paths(double dx, double dy)
    {
        // 声明变量
        unsigned idx;
        // 获取顶点总数
        unsigned num_ver = m_vertices.total_vertices();
        // 遍历所有顶点
        for(idx = 0; idx < num_ver; idx++)
        {
            double x, y;
            // 获取顶点的命令和坐标
            if(is_vertex(m_vertices.vertex(idx, &x, &y)))
            {
                // 对顶点坐标进行平移操作
                x += dx;
                y += dy;
                m_vertices.modify_vertex(idx, x, y);
            }
        }
    }

    //-----------------------------------------------------vertex_stl_storage
    template<class Container> class vertex_stl_storage
    {
        // 这里定义了一个模板类 vertex_stl_storage
    public:
        // 定义类型别名，vertex_type 为容器 Container 中元素的类型，value_type 为 vertex_type 的值类型
        typedef typename Container::value_type vertex_type;
        typedef typename vertex_type::value_type value_type;

        // 清空 m_vertices 容器中所有元素
        void remove_all() { m_vertices.clear(); }
        
        // 清空 m_vertices 容器中所有元素，与 remove_all 功能相同
        void free_all()   { m_vertices.clear(); }

        // 向 m_vertices 容器中添加一个新顶点，使用给定的 x、y 坐标和命令 cmd
        void add_vertex(double x, double y, unsigned cmd)
        {
            // 创建新顶点并添加到 m_vertices 容器末尾
            m_vertices.push_back(vertex_type(value_type(x), 
                                             value_type(y), 
                                             int8u(cmd)));
        }

        // 修改指定索引 idx 处顶点的坐标为给定的 x 和 y 值
        void modify_vertex(unsigned idx, double x, double y)
        {
            // 获取索引 idx 处的顶点引用并更新其坐标值
            vertex_type& v = m_vertices[idx];
            v.x = value_type(x);
            v.y = value_type(y);
        }

        // 修改指定索引 idx 处顶点的坐标和命令为给定的 x、y 和 cmd
        void modify_vertex(unsigned idx, double x, double y, unsigned cmd)
        {
            // 获取索引 idx 处的顶点引用并更新其坐标和命令值
            vertex_type& v = m_vertices[idx];
            v.x   = value_type(x);
            v.y   = value_type(y);
            v.cmd = int8u(cmd);
        }

        // 修改指定索引 idx 处顶点的命令为给定的 cmd
        void modify_command(unsigned idx, unsigned cmd)
        {
            // 更新索引 idx 处顶点的命令值
            m_vertices[idx].cmd = int8u(cmd);
        }

        // 交换索引 v1 和 v2 处顶点在 m_vertices 容器中的位置
        void swap_vertices(unsigned v1, unsigned v2)
        {
            // 交换索引 v1 和 v2 处顶点
            vertex_type t = m_vertices[v1];
            m_vertices[v1] = m_vertices[v2];
            m_vertices[v2] = t;
        }

        // 返回最后一个顶点的命令值，若容器为空则返回 path_cmd_stop
        unsigned last_command() const
        {
            return m_vertices.size() ? 
                m_vertices[m_vertices.size() - 1].cmd : 
                path_cmd_stop;
        }

        // 返回最后一个顶点的坐标，若容器为空则将 x 和 y 设为 0.0，并返回 path_cmd_stop
        unsigned last_vertex(double* x, double* y) const
        {
            if(m_vertices.size() == 0)
            {
                *x = *y = 0.0;
                return path_cmd_stop;
            }
            // 返回最后一个顶点的索引和坐标
            return vertex(m_vertices.size() - 1, x, y);
        }

        // 返回倒数第二个顶点的坐标，若容器中顶点数小于 2，则将 x 和 y 设为 0.0，并返回 path_cmd_stop
        unsigned prev_vertex(double* x, double* y) const
        {
            if(m_vertices.size() < 2)
            {
                *x = *y = 0.0;
                return path_cmd_stop;
            }
            // 返回倒数第二个顶点的索引和坐标
            return vertex(m_vertices.size() - 2, x, y);
        }

        // 返回最后一个顶点的 x 坐标，若容器为空则返回 0.0
        double last_x() const
        {
            return m_vertices.size() ? m_vertices[m_vertices.size() - 1].x : 0.0;
        }

        // 返回最后一个顶点的 y 坐标，若容器为空则返回 0.0
        double last_y() const
        {
            return m_vertices.size() ? m_vertices[m_vertices.size() - 1].y : 0.0;
        }

        // 返回 m_vertices 容器中顶点的总数
        unsigned total_vertices() const
        {
            return m_vertices.size();
        }

        // 返回指定索引 idx 处顶点的坐标和命令，并将 x 和 y 设为对应值
        unsigned vertex(unsigned idx, double* x, double* y) const
        {
            // 获取索引 idx 处顶点的引用并将其坐标赋值给 x 和 y
            const vertex_type& v = m_vertices[idx];
            *x = v.x;
            *y = v.y;
            return v.cmd;
        }

        // 返回指定索引 idx 处顶点的命令
        unsigned command(unsigned idx) const
        {
            return m_vertices[idx].cmd;
        }

    private:
        // 顶点容器，类型为 Container
        Container m_vertices;
    };

    //-----------------------------------------------------------path_storage
    // 使用 vertex_block_storage<double> 作为容器类型的 path_base 类型的声明
    typedef path_base<vertex_block_storage<double> > path_storage;

    // 使用 pod_bvector 作为容器类型的 path_storage 类型的声明示例
    //-----------------------------------------------------------------------
    // 定义一个类型别名 path_storage
    // 使用 path_base 作为模板，其中的模板参数是 vertex_stl_storage，其内部又使用 pod_bvector 来存储 vertex_d 类型的数据
    typedef path_base<vertex_stl_storage<pod_bvector<vertex_d> > > path_storage;
// 结尾的代码块声明了一个名为 stl_path_storage 的 typedef，这是一个特定的路径存储类型
//---------------------------------------------------------------------------
//#include <vector>
//namespace agg
//{
//    typedef path_base<vertex_stl_storage<std::vector<vertex_d> > > stl_path_storage; 
//}
// #endif
```