# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_gsv_text.cpp`

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

// 包含标准库头文件，用于字符串操作和输入输出
#include <string.h>
#include <stdio.h>

// 包含AGG库的头文件，定义了文本渲染的基本功能
#include "agg_gsv_text.h"
#include "agg_bounding_rect.h"

// 命名空间agg，用于组织AGG库中的类和函数
namespace agg
{
    // 默认的字体数据，用于文本渲染
    int8u gsv_default_font[] = 
    };

    //-------------------------------------------------------------------------
    // 构造函数，初始化文本渲染器的各种参数和状态
    gsv_text::gsv_text() :
      m_x(0.0),                    // 当前文本渲染位置的X坐标
      m_y(0.0),                    // 当前文本渲染位置的Y坐标
      m_start_x(0.0),              // 起始文本渲染位置的X坐标
      m_width(10.0),               // 文本的默认宽度
      m_height(0.0),               // 文本的默认高度
      m_space(0.0),                // 文字间的默认间距
      m_line_space(0.0),           // 行之间的默认间距
      m_text(m_chr),               // 当前文本字符串的指针
      m_text_buf(),                // 文本缓冲区
      m_cur_chr(m_chr),            // 当前字符的指针
      m_font(gsv_default_font),    // 当前使用的字体数据指针，默认为默认字体
      m_loaded_font(),             // 加载的字体数据（未使用）
      m_status(initial),           // 文本渲染器的当前状态
      m_big_endian(false),         // 是否为大端序
      m_flip(false)                // 是否翻转渲染
    {
        m_chr[0] = m_chr[1] = 0;   // 初始化字符缓冲区为零

        int t = 1;
        if(*(char*)&t == 0) m_big_endian = true;  // 检测系统是否为大端序
    }

    //-------------------------------------------------------------------------
    // 设置当前使用的字体数据指针
    void gsv_text::font(const void* font)
    {
        m_font = font;             // 设置当前字体数据指针
        if(m_font == 0) m_font = &m_loaded_font[0];  // 若字体数据为空，则使用加载的字体数据（未使用）
    }

    //-------------------------------------------------------------------------
    // 设置文本的高度和宽度
    void gsv_text::size(double height, double width)
    {
        m_height = height;         // 设置文本的高度
        m_width  = width;          // 设置文本的宽度
    }

    //-------------------------------------------------------------------------
    // 设置文字间的间距
    void gsv_text::space(double space)
    {
        m_space = space;           // 设置文字间的间距
    }

    //-------------------------------------------------------------------------
    // 设置行间的间距
    void gsv_text::line_space(double line_space)
    {
        m_line_space = line_space; // 设置行间的间距
    }

    //-------------------------------------------------------------------------
    // 设置文本的起始绘制点
    void gsv_text::start_point(double x, double y)
    {
        m_x = m_start_x = x;       // 设置文本起始绘制点的X坐标
        m_y = y;                   // 设置文本起始绘制点的Y坐标
        //if(m_flip) m_y += m_height; // 如果需要翻转渲染，则调整起始绘制点的Y坐标
    }

    //-------------------------------------------------------------------------
    // 加载指定文件中的字体数据（未实现的函数，可能用于将外部字体加载到内存）
    void gsv_text::load_font(const char* file)
    {
        // 清空已加载字体的数据
        m_loaded_font.resize(0);
        // 打开文件以二进制只读方式
        FILE* fd = fopen(file, "rb");
        if(fd)
        {
            unsigned len;
    
            // 定位到文件末尾并获取文件长度
            fseek(fd, 0l, SEEK_END);
            len = ftell(fd);
            // 重新定位到文件开头
            fseek(fd, 0l, SEEK_SET);
            if(len > 0)
            {
                // 调整已加载字体容器的大小为文件长度
                m_loaded_font.resize(len);
                // 从文件中读取数据到已加载字体容器
                fread(&m_loaded_font[0], 1, len, fd);
                // 将字体指针指向已加载字体的第一个字节
                m_font = &m_loaded_font[0];
            }
            // 关闭文件
            fclose(fd);
        }
    }
    
    //-------------------------------------------------------------------------
    void gsv_text::text(const char* text)
    {
        // 如果传入的文本为空指针，则设置默认字符并返回
        if(text == 0)
        {
            m_chr[0] = 0;
            m_text = m_chr;
            return;
        }
        // 计算新文本的大小（包括结尾的空字符）
        unsigned new_size = strlen(text) + 1;
        // 如果新文本大小超过当前文本缓冲区大小，则重新调整缓冲区大小
        if(new_size > m_text_buf.size())
        {
            m_text_buf.resize(new_size);
        }
        // 将文本复制到文本缓冲区
        memcpy(&m_text_buf[0], text, new_size);
        // 将文本指针指向文本缓冲区的第一个字符
        m_text = &m_text_buf[0];
    }
    
    //-------------------------------------------------------------------------
    void gsv_text::rewind(unsigned)
    {
        // 重置文本状态为初始状态
        m_status = initial;
        // 如果当前没有加载字体数据，则直接返回
        if(m_font == 0) return;
        
        // 将索引指针设置为字体数据的起始位置
        m_indices = (int8u*)m_font;
        // 从字体数据中获取基础高度
        double base_height = value(m_indices + 4);
        // 调整索引指针到字符索引数据的位置
        m_indices += value(m_indices);
        // 设置字符图形数据的起始位置
        m_glyphs = (int8*)(m_indices + 257*2);
        // 计算当前字符的高度和宽度比例
        m_h = m_height / base_height;
        m_w = (m_width == 0.0) ? m_h : m_width / base_height;
        // 如果需要翻转字符图形，则设置高度为负值
        if(m_flip) m_h = -m_h;
        // 将当前字符指针指向文本的起始位置
        m_cur_chr = m_text;
    }
    
    //-------------------------------------------------------------------------
    unsigned gsv_text::vertex(double* x, double* y)
    {
        // 无符号整数索引
        unsigned idx;
        // 字符的垂直和水平偏移量
        int8 yc, yf;
        // 笔画的水平和垂直距离
        int dx, dy;
        // 退出标志，初始为假
        bool quit = false;
        
        // 当退出标志为假时执行循环
        while(!quit)
        {
            // 根据状态机状态进行不同的操作
            switch(m_status)
            {
            // 初始状态
            case initial:
                // 如果字体为0，则设置退出标志为真并跳出循环
                if(m_font == 0) 
                {
                    quit = true;
                    break;
                }
                // 否则，状态转移到下一个字符状态
                m_status = next_char;
    
            // 下一个字符状态
            case next_char:
                // 如果当前字符指针指向的字符为0，设置退出标志为真并跳出循环
                if(*m_cur_chr == 0) 
                {
                    quit = true;
                    break;
                }
                // 从当前字符指针指向的字符中取得索引，并按位与0xFF
                idx = (*m_cur_chr++) & 0xFF;
                // 如果索引是换行符'\n'
                if(idx == '\n')
                {
                    // 重置m_x为起始横坐标，根据翻转状态调整m_y的纵坐标
                    m_x = m_start_x;
                    m_y -= m_flip ? -m_height - m_line_space : m_height + m_line_space;
                    break;
                }
                // 索引左移一位
                idx <<= 1;
                // 计算背景字形和结束字形的指针位置
                m_bglyph = m_glyphs + value(m_indices + idx);
                m_eglyph = m_glyphs + value(m_indices + idx + 2);
                // 状态转移到开始字形状态
                m_status = start_glyph;
    
            // 开始字形状态
            case start_glyph:
                // 将当前横纵坐标赋值给传入的指针
                *x = m_x;
                *y = m_y;
                // 状态转移到字形状态
                m_status = glyph;
                // 返回路径命令，移动到新坐标
                return path_cmd_move_to;
    
            // 字形状态
            case glyph:
                // 如果背景字形超过或等于结束字形
                if(m_bglyph >= m_eglyph)
                {
                    // 状态转移到下一个字符状态，增加横坐标的间距
                    m_status = next_char;
                    m_x += m_space;
                    break;
                }
                // 从背景字形中取得水平偏移量dx
                dx = int(*m_bglyph++);
                // 从背景字形中取得垂直偏移量yc，并保留最高位作为标志yf
                yf = (yc = *m_bglyph++) & 0x80;
                yc <<= 1; 
                yc >>= 1;
                // 将yc转换为整数dy
                dy = int(yc);
                // 根据偏移量计算新的横纵坐标
                m_x += double(dx) * m_w;
                m_y += double(dy) * m_h;
                // 将当前横纵坐标赋值给传入的指针
                *x = m_x;
                *y = m_y;
                // 返回路径命令，根据yf决定是移动到新坐标还是绘制直线到新坐标
                return yf ? path_cmd_move_to : path_cmd_line_to;
            }
    
        }
        // 循环结束，返回停止路径命令
        return path_cmd_stop;
    }
    
    //-------------------------------------------------------------------------
    // 计算文本宽度
    double gsv_text::text_width()
    {
        // 定义边界框的四个角点坐标
        double x1, y1, x2, y2;
        // 计算当前文本的边界框
        bounding_rect_single(*this, 0, &x1, &y1, &x2, &y2);
        // 返回边界框宽度
        return x2 - x1;
    }
}



# 这行代码表示代码块的结束，} 是闭合代码块的标记，用于结束一个函数、循环、条件语句或其他代码块。
```