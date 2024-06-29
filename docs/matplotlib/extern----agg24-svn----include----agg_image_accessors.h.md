# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_image_accessors.h`

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

#ifndef AGG_IMAGE_ACCESSORS_INCLUDED
#define AGG_IMAGE_ACCESSORS_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //-----------------------------------------------------image_accessor_clip
    // 像素访问器类模板，支持剪切功能
    template<class PixFmt> class image_accessor_clip
    {
    public:
        typedef PixFmt   pixfmt_type;  // 像素格式类型
        typedef typename pixfmt_type::color_type color_type;  // 颜色类型
        typedef typename pixfmt_type::order_type order_type;  // 像素顺序类型
        typedef typename pixfmt_type::value_type value_type;  // 像素值类型
        enum pix_width_e { pix_width = pixfmt_type::pix_width };  // 像素宽度枚举

        // 默认构造函数
        image_accessor_clip() {}
        // 带参构造函数，初始化像素格式和背景颜色
        explicit image_accessor_clip(pixfmt_type& pixf, 
                                     const color_type& bk) : 
            m_pixf(&pixf)
        {
            pixfmt_type::make_pix(m_bk_buf, bk);  // 设置背景颜色
        }

        // 关联新的像素格式对象
        void attach(pixfmt_type& pixf)
        {
            m_pixf = &pixf;
        }

        // 设置背景颜色
        void background_color(const color_type& bk)
        {
            pixfmt_type::make_pix(m_bk_buf, bk);
        }

    private:
        // 内部像素指针访问函数，根据坐标判断是否在有效区域内
        AGG_INLINE const int8u* pixel() const
        {
            if(m_y >= 0 && m_y < (int)m_pixf->height() &&
               m_x >= 0 && m_x < (int)m_pixf->width())
            {
                return m_pixf->pix_ptr(m_x, m_y);  // 返回指定坐标的像素指针
            }
            return m_bk_buf;  // 超出边界返回背景颜色
        }

    public:
        // 获取指定位置开始的像素数据，支持剪切功能
        AGG_INLINE const int8u* span(int x, int y, unsigned len)
        {
            m_x = m_x0 = x;
            m_y = y;
            if(y >= 0 && y < (int)m_pixf->height() &&
               x >= 0 && x+(int)len <= (int)m_pixf->width())
            {
                return m_pix_ptr = m_pixf->pix_ptr(x, y);  // 返回指定区域的像素数据指针
            }
            m_pix_ptr = 0;
            return pixel();  // 返回像素区域之外的像素数据或者背景颜色
        }

        // 获取下一个 x 方向的像素数据
        AGG_INLINE const int8u* next_x()
        {
            if(m_pix_ptr) return m_pix_ptr += pix_width;  // 如果当前像素指针有效，则移动到下一个 x 位置
            ++m_x;
            return pixel();  // 返回下一个 x 位置的像素数据或者背景颜色
        }

        // 获取下一个 y 方向的像素数据
        AGG_INLINE const int8u* next_y()
        {
            ++m_y;
            m_x = m_x0;
            if(m_pix_ptr && 
               m_y >= 0 && m_y < (int)m_pixf->height())
            {
                return m_pix_ptr = m_pixf->pix_ptr(m_x, m_y);  // 返回下一个 y 位置的像素数据指针
            }
            m_pix_ptr = 0;
            return pixel();  // 返回下一个 y 位置的像素数据或者背景颜色
        }


这段代码定义了一个像素访问器类模板，支持像素数据的访问和剪切功能，能够处理给定的像素格式。
    private:
        // 指向像素格式的指针
        const pixfmt_type* m_pixf;
        // 背景缓冲区，大小为像素宽度
        int8u              m_bk_buf[pix_width];
        // 当前像素位置和起始位置
        int                m_x, m_x0, m_y;
        // 指向当前像素的指针
        const int8u*       m_pix_ptr;
    };



    //--------------------------------------------------image_accessor_no_clip
    // 无裁剪的图像访问器模板类
    template<class PixFmt> class image_accessor_no_clip
    {
    public:
        typedef PixFmt   pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::order_type order_type;
        typedef typename pixfmt_type::value_type value_type;
        enum pix_width_e { pix_width = pixfmt_type::pix_width };

        // 默认构造函数
        image_accessor_no_clip() {}
        // 构造函数，初始化时指定像素格式对象
        explicit image_accessor_no_clip(pixfmt_type& pixf) : 
            m_pixf(&pixf) 
        {}

        // 关联像素格式对象
        void attach(pixfmt_type& pixf)
        {
            m_pixf = &pixf;
        }

        // 获取指定位置的像素数据指针，无裁剪
        AGG_INLINE const int8u* span(int x, int y, unsigned)
        {
            m_x = x;
            m_y = y;
            return m_pix_ptr = m_pixf->pix_ptr(x, y);
        }

        // 获取下一列像素数据指针，无裁剪
        AGG_INLINE const int8u* next_x()
        {
            return m_pix_ptr += pix_width;
        }

        // 获取下一行像素数据指针，无裁剪
        AGG_INLINE const int8u* next_y()
        {
            ++m_y;
            return m_pix_ptr = m_pixf->pix_ptr(m_x, m_y);
        }

    private:
        // 指向像素格式的指针
        const pixfmt_type* m_pixf;
        // 当前像素位置
        int                m_x, m_y;
        // 指向当前像素的指针
        const int8u*       m_pix_ptr;
    };



    //----------------------------------------------------image_accessor_clone
    // 克隆图像访问器模板类
    template<class PixFmt> class image_accessor_clone
    {
    public:
        typedef PixFmt   pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::order_type order_type;
        typedef typename pixfmt_type::value_type value_type;
        enum pix_width_e { pix_width = pixfmt_type::pix_width };

        // 默认构造函数
        image_accessor_clone() {}
        // 构造函数，初始化时指定像素格式对象
        explicit image_accessor_clone(pixfmt_type& pixf) : 
            m_pixf(&pixf) 
        {}

        // 关联像素格式对象
        void attach(pixfmt_type& pixf)
        {
            m_pixf = &pixf;
        }

    private:
        // 获取当前像素的指针，支持裁剪
        AGG_INLINE const int8u* pixel() const
        {
            int x = m_x;
            int y = m_y;
            // 裁剪X坐标
            if(x < 0) x = 0;
            if(x >= (int)m_pixf->width())  x = m_pixf->width() - 1;
            // 裁剪Y坐标
            if(y < 0) y = 0;
            if(y >= (int)m_pixf->height()) y = m_pixf->height() - 1;
            // 返回裁剪后位置的像素数据指针
            return m_pixf->pix_ptr(x, y);
        }
        // 定义 image_accessor_wrap 类模板，用于封装 PixFmt 类型的像素格式操作，并支持 X 和 Y 轴的包装（wrap）模式
        template<class PixFmt, class WrapX, class WrapY> class image_accessor_wrap
        {
        public:
            // 定义类型别名
            typedef PixFmt   pixfmt_type;
            typedef typename pixfmt_type::color_type color_type;
            typedef typename pixfmt_type::order_type order_type;
            typedef typename pixfmt_type::value_type value_type;
            // 像素宽度枚举常量
            enum pix_width_e { pix_width = pixfmt_type::pix_width };

            // 默认构造函数
            image_accessor_wrap() {}
            // 显式构造函数，接受一个 pixfmt_type 类型的引用，初始化成员变量
            explicit image_accessor_wrap(pixfmt_type& pixf) : 
                m_pixf(&pixf), 
                m_wrap_x(pixf.width()), 
                m_wrap_y(pixf.height())
            {}

            // 将 pixfmt_type 对象附加到当前实例
            void attach(pixfmt_type& pixf)
            {
                m_pixf = &pixf;
            }

            // 返回指向像素数据的指针，使用包装后的 X 和 Y 坐标
            AGG_INLINE const int8u* span(int x, int y, unsigned)
            {
                m_x = x;
                // 获取像素行指针，使用包装后的 Y 坐标
                m_row_ptr = m_pixf->pix_ptr(0, m_wrap_y(y));
                // 返回位于包装后的 X 坐标处的像素数据指针
                return m_row_ptr + m_wrap_x(x) * pix_width;
            }

            // 返回下一个 X 坐标处的像素数据指针
            AGG_INLINE const int8u* next_x()
            {
                int x = ++m_wrap_x;
                return m_row_ptr + x * pix_width;
            }

            // 返回下一个 Y 坐标处的像素数据指针
            AGG_INLINE const int8u* next_y()
            {
                // 更新行指针到下一行的像素数据，使用包装后的 Y 坐标
                m_row_ptr = m_pixf->pix_ptr(0, ++m_wrap_y);
                // 返回位于当前 X 坐标处的像素数据指针
                return m_row_ptr + m_wrap_x(m_x) * pix_width;
            }

        private:
            const pixfmt_type* m_pixf; // 指向 pixfmt_type 对象的常量指针
            const int8u*       m_row_ptr; // 当前行的像素数据指针
            int                m_x; // 当前 X 坐标
            WrapX              m_wrap_x; // X 轴包装模式对象
            WrapY              m_wrap_y; // Y 轴包装模式对象
        };
    //---------------------------------------------------wrap_mode_repeat
    class wrap_mode_repeat
    {
    public:
        // 默认构造函数，无操作
        wrap_mode_repeat() {}

        // 带参构造函数，初始化对象
        wrap_mode_repeat(unsigned size) : 
            m_size(size),  // 初始化成员变量 m_size
            m_add(size * (0x3FFFFFFF / size)),  // 初始化成员变量 m_add
            m_value(0)  // 初始化成员变量 m_value
        {}

        // 重载 () 运算符，实现 repeat 的环绕模式
        AGG_INLINE unsigned operator() (int v)
        { 
            return m_value = (unsigned(v) + m_add) % m_size; 
        }

        // 前置递增运算符重载，实现 m_value 环绕递增
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value >= m_size) m_value = 0;
            return m_value;
        }
    private:
        unsigned m_size;  // 环绕大小
        unsigned m_add;   // 增量值
        unsigned m_value; // 当前值
    };


    //---------------------------------------------------wrap_mode_repeat_pow2
    class wrap_mode_repeat_pow2
    {
    public:
        // 默认构造函数，无操作
        wrap_mode_repeat_pow2() {}

        // 带参构造函数，初始化对象
        wrap_mode_repeat_pow2(unsigned size) : m_value(0)
        {
            // 初始化 m_mask 为最接近且大于 size 的 2 的幂减一的值
            m_mask = 1;
            while(m_mask < size) m_mask = (m_mask << 1) | 1;
            m_mask >>= 1;
        }

        // 重载 () 运算符，实现 pow2 repeat 的环绕模式
        AGG_INLINE unsigned operator() (int v)
        { 
            return m_value = unsigned(v) & m_mask;
        }

        // 前置递增运算符重载，实现 m_value 环绕递增
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value > m_mask) m_value = 0;
            return m_value;
        }
    private:
        unsigned m_mask;  // 掩码，用于取模运算
        unsigned m_value; // 当前值
    };


    //----------------------------------------------wrap_mode_repeat_auto_pow2
    class wrap_mode_repeat_auto_pow2
    {
    public:
        // 默认构造函数，无操作
        wrap_mode_repeat_auto_pow2() {}

        // 带参构造函数，初始化对象
        wrap_mode_repeat_auto_pow2(unsigned size) :
            m_size(size),  // 初始化成员变量 m_size
            m_add(size * (0x3FFFFFFF / size)),  // 初始化成员变量 m_add
            m_mask((m_size & (m_size-1)) ? 0 : m_size-1),  // 初始化成员变量 m_mask
            m_value(0)  // 初始化成员变量 m_value
        {}

        // 重载 () 运算符，根据 m_mask 决定返回 unsigned(v) & m_mask 或 (unsigned(v) + m_add) % m_size
        AGG_INLINE unsigned operator() (int v) 
        { 
            if(m_mask) return m_value = unsigned(v) & m_mask;
            return m_value = (unsigned(v) + m_add) % m_size;
        }

        // 前置递增运算符重载，实现 m_value 环绕递增
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value >= m_size) m_value = 0;
            return m_value;
        }

    private:
        unsigned m_size;  // 环绕大小
        unsigned m_add;   // 增量值
        unsigned m_mask;  // 掩码，用于取模运算或按位与运算
        unsigned m_value; // 当前值
    };


    //-------------------------------------------------------wrap_mode_reflect
    class wrap_mode_reflect
    {
    public:
        // 默认构造函数，无操作
        wrap_mode_reflect() {}

        // 带参构造函数，初始化对象
        wrap_mode_reflect(unsigned size) : 
            m_size(size),  // 初始化成员变量 m_size
            m_size2(size * 2),  // 初始化成员变量 m_size2
            m_add(m_size2 * (0x3FFFFFFF / m_size2)),  // 初始化成员变量 m_add
            m_value(0)  // 初始化成员变量 m_value
        {}

        // 重载 () 运算符，实现 reflect 的环绕模式
        AGG_INLINE unsigned operator() (int v)
        { 
            m_value = (unsigned(v) + m_add) % m_size2;
            if(m_value >= m_size) return m_size2 - m_value - 1;
            return m_value;
        }

        // 前置递增运算符重载，实现 m_value 环绕递增
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value >= m_size2) m_value = 0;
            if(m_value >= m_size) return m_size2 - m_value - 1;
            return m_value;
        }
    // 用于封装包装模式的类，实现反射加倍2的幂次方的包装模式
    class wrap_mode_reflect_pow2
    {
    public:
        // 默认构造函数
        wrap_mode_reflect_pow2() {}
    
        // 构造函数，根据给定的尺寸初始化对象
        wrap_mode_reflect_pow2(unsigned size) : m_value(0)
        {
            // 初始化掩码和尺寸，确保掩码大于等于size的最小2的幂次方
            m_mask = 1;
            m_size = 1;
            while(m_mask < size) 
            {
                m_mask = (m_mask << 1) | 1;
                m_size <<= 1;
            }
        }
    
        // 重载函数调用操作符，返回输入整数的反射加倍2的幂次方模式下的结果
        AGG_INLINE unsigned operator() (int v)
        { 
            // 计算m_value，确保它在0到m_mask之间，超过m_size则返回其反射
            m_value = unsigned(v) & m_mask;
            if(m_value >= m_size) return m_mask - m_value;
            return m_value;
        }
    
        // 重载前置自增操作符，返回加一后的反射加倍2的幂次方模式下的结果
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            m_value &= m_mask;
            if(m_value >= m_size) return m_mask - m_value;
            return m_value;
        }
    
    private:
        unsigned m_size;   // 尺寸
        unsigned m_mask;   // 控制掩码
        unsigned m_value;  // 当前值
    };
    
    
    
    //--------------------------------------------------wrap_mode_reflect_auto_pow2
    // 用于封装包装模式的类，实现自动反射加倍2的幂次方的包装模式
    class wrap_mode_reflect_auto_pow2
    {
    public:
        // 默认构造函数
        wrap_mode_reflect_auto_pow2() {}
    
        // 构造函数，根据给定的尺寸初始化对象，同时计算相关参数
        wrap_mode_reflect_auto_pow2(unsigned size) :
            m_size(size),
            m_size2(size * 2),
            m_add(m_size2 * (0x3FFFFFFF / m_size2)),
            m_mask((m_size2 & (m_size2-1)) ? 0 : m_size2-1),
            m_value(0)
        {}
    
        // 重载函数调用操作符，返回输入整数的自动反射加倍2的幂次方模式下的结果
        AGG_INLINE unsigned operator() (int v) 
        { 
            // 计算m_value，如果m_mask为0，则返回(v + m_add) % m_size2，否则返回v & m_mask
            m_value = m_mask ? unsigned(v) & m_mask : (unsigned(v) + m_add) % m_size2;
            if(m_value >= m_size) return m_size2 - m_value - 1;
            return m_value;            
        }
    
        // 重载前置自增操作符，返回加一后的自动反射加倍2的幂次方模式下的结果
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value >= m_size2) m_value = 0;
            if(m_value >= m_size) return m_size2 - m_value - 1;
            return m_value;
        }
    
    private:
        unsigned m_size;   // 尺寸
        unsigned m_size2;  // 两倍尺寸
        unsigned m_add;    // 增加值
        unsigned m_mask;   // 控制掩码
        unsigned m_value;  // 当前值
    };
}


这行代码结束了一个代码块。在C语言中，`}`字符用于标识一个代码块的结束。


#endif


这行代码是条件预处理指令，用于结束一个条件编译块，它用于C语言中。`#endif`指示条件预处理指令的结束，其前面通常有一个`#ifdef`或`#ifndef`，用于控制代码的编译。
```