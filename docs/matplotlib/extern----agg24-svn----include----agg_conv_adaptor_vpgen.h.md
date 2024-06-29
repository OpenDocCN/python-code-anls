# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_adaptor_vpgen.h`

```
// 包含必要的基础头文件
#include "agg_basics.h"

// 命名空间agg开始
namespace agg
{

    // conv_adaptor_vpgen模板类的声明
    template<class VertexSource, class VPGen> class conv_adaptor_vpgen
    {
    public:
        // 显式构造函数，初始化m_source成员变量
        explicit conv_adaptor_vpgen(VertexSource& source) : m_source(&source) {}

        // 附加新的顶点源
        void attach(VertexSource& source) { m_source = &source; }

        // 返回VPGen对象的引用（可变）
        VPGen& vpgen() { return m_vpgen; }

        // 返回VPGen对象的常量引用
        const VPGen& vpgen() const { return m_vpgen; }

        // 重置顶点生成器和顶点源
        void rewind(unsigned path_id);

        // 获取下一个顶点的坐标，返回顶点索引
        unsigned vertex(double* x, double* y);

    private:
        // 禁止拷贝构造函数和赋值运算符
        conv_adaptor_vpgen(const conv_adaptor_vpgen<VertexSource, VPGen>&);
        const conv_adaptor_vpgen<VertexSource, VPGen>& 
            operator = (const conv_adaptor_vpgen<VertexSource, VPGen>&);

        // 顶点源指针
        VertexSource* m_source;

        // 顶点生成器对象
        VPGen         m_vpgen;

        // 起始坐标
        double        m_start_x;

        // 起始坐标
        double        m_start_y;

        // 多边形标志
        unsigned      m_poly_flags;

        // 顶点数目
        int           m_vertices;
    };



    //------------------------------------------------------------------------
    // conv_adaptor_vpgen模板类的rewind函数定义
    template<class VertexSource, class VPGen>
    void conv_adaptor_vpgen<VertexSource, VPGen>::rewind(unsigned path_id) 
    { 
        // 调用顶点源的rewind函数
        m_source->rewind(path_id);

        // 重置顶点生成器
        m_vpgen.reset();

        // 重置起始坐标为0
        m_start_x    = 0;
        m_start_y    = 0;

        // 重置多边形标志为0
        m_poly_flags = 0;

        // 重置顶点数目为0
        m_vertices   = 0;
    }


    //------------------------------------------------------------------------
    // conv_adaptor_vpgen模板类的vertex函数定义
    template<class VertexSource, class VPGen>
    unsigned conv_adaptor_vpgen<VertexSource, VPGen>::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_stop;
        // 初始化 cmd 变量为 path_cmd_stop，表示路径命令的停止符
    
        for(;;)
        {
            // 无限循环，用于处理路径中的每个顶点命令
            cmd = m_vpgen.vertex(x, y);
            // 调用 m_vpgen 对象的 vertex 方法获取下一个顶点命令，并更新 cmd 变量
    
            if(!is_stop(cmd)) break;
            // 如果当前命令不是停止符，则跳出循环
    
            if(m_poly_flags && !m_vpgen.auto_unclose())
            {
                // 如果存在多边形标志并且不自动取消闭合
                *x = 0.0;
                *y = 0.0;
                // 重置当前点的坐标为 (0.0, 0.0)
                cmd = m_poly_flags;
                // 更新 cmd 变量为多边形标志
                m_poly_flags = 0;
                // 清除多边形标志
                break;
                // 跳出循环
            }
    
            if(m_vertices < 0)
            {
                // 如果顶点数小于 0
                if(m_vertices < -1) 
                {
                    m_vertices = 0;
                    // 重置顶点数为 0
                    return path_cmd_stop;
                    // 返回停止符
                }
                m_vpgen.move_to(m_start_x, m_start_y);
                // 调用 m_vpgen 对象的 move_to 方法，移动到起始点 (m_start_x, m_start_y)
                m_vertices = 1;
                // 设置顶点数为 1
                continue;
                // 继续循环
            }
    
            double tx, ty;
            // 定义临时变量 tx 和 ty，用于存储顶点坐标
            cmd = m_source->vertex(&tx, &ty);
            // 调用 m_source 对象的 vertex 方法获取下一个顶点命令，并更新 cmd 变量
    
            if(is_vertex(cmd))
            {
                // 如果当前命令是顶点命令
                if(is_move_to(cmd)) 
                {
                    // 如果是移动到命令
                    if(m_vpgen.auto_close() && m_vertices > 2)
                    {
                        // 如果自动闭合并且顶点数大于 2
                        m_vpgen.line_to(m_start_x, m_start_y);
                        // 调用 m_vpgen 对象的 line_to 方法，绘制一条线段到起始点 (m_start_x, m_start_y)
                        m_poly_flags = path_cmd_end_poly | path_flags_close;
                        // 设置多边形标志为结束多边形且闭合的标志
                        m_start_x    = tx;
                        m_start_y    = ty;
                        // 更新起始点坐标为当前顶点坐标
                        m_vertices   = -1;
                        // 设置顶点数为 -1，表示结束当前多边形
                        continue;
                        // 继续循环
                    }
                    m_vpgen.move_to(tx, ty);
                    // 否则调用 m_vpgen 对象的 move_to 方法，移动到当前顶点坐标
                    m_start_x  = tx;
                    m_start_y  = ty;
                    // 更新起始点坐标为当前顶点坐标
                    m_vertices = 1;
                    // 设置顶点数为 1
                }
                else 
                {
                    // 如果不是移动到命令，则是直线到命令
                    m_vpgen.line_to(tx, ty);
                    // 调用 m_vpgen 对象的 line_to 方法，绘制一条线段到当前顶点坐标
                    ++m_vertices;
                    // 增加顶点数计数
                }
            }
            else
            {
                // 如果当前命令不是顶点命令
                if(is_end_poly(cmd))
                {
                    // 如果是结束多边形命令
                    m_poly_flags = cmd;
                    // 更新多边形标志为当前命令
                    if(is_closed(cmd) || m_vpgen.auto_close())
                    {
                        // 如果是闭合的多边形或者支持自动闭合
                        if(m_vpgen.auto_close()) m_poly_flags |= path_flags_close;
                        // 如果支持自动闭合，则设置多边形标志为闭合
                        if(m_vertices > 2)
                        {
                            m_vpgen.line_to(m_start_x, m_start_y);
                            // 如果顶点数大于 2，则绘制一条线段到起始点 (m_start_x, m_start_y)
                        }
                        m_vertices = 0;
                        // 重置顶点数为 0
                    }
                }
                else
                {
                    // 如果当前命令是停止符
                    if(m_vpgen.auto_close() && m_vertices > 2)
                    {
                        // 如果支持自动闭合并且顶点数大于 2
                        m_vpgen.line_to(m_start_x, m_start_y);
                        // 调用 m_vpgen 对象的 line_to 方法，绘制一条线段到起始点 (m_start_x, m_start_y)
                        m_poly_flags = path_cmd_end_poly | path_flags_close;
                        // 设置多边形标志为结束多边形且闭合的标志
                        m_vertices   = -2;
                        // 设置顶点数为 -2，表示结束多边形
                        continue;
                        // 继续循环
                    }
                    break;
                    // 否则跳出循环
                }
            }
        }
        return cmd;
        // 返回最后的命令
    }
}



#endif



// 这两行代码分别用于结束一个 C++ 文件中的函数定义和预处理指令的条件编译部分
// } 用于结束之前定义的某个函数或者代码块
// #endif 用于结束之前的 #ifdef 或 #ifndef 预处理指令的条件编译部分
```