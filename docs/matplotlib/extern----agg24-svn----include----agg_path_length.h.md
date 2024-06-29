# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_path_length.h`

```py
#ifndef AGG_PATH_LENGTH_INCLUDED
#define AGG_PATH_LENGTH_INCLUDED

// 如果 AGG_PATH_LENGTH_INCLUDED 宏未定义，则定义它，以避免重复包含该头文件


#include "agg_math.h"

// 包含 agg_math.h 头文件，提供数学计算所需的功能和定义


namespace agg
{
    template<class VertexSource> 
    double path_length(VertexSource& vs, unsigned path_id = 0)
    {
        double len = 0.0;
        double start_x = 0.0;
        double start_y = 0.0;
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;
        bool first = true;

        unsigned cmd;
        vs.rewind(path_id);

// 在 agg 命名空间内定义一个模板函数 path_length，接受一个 VertexSource 类型的引用 vs 和一个默认为 0 的无符号整数 path_id 作为参数

// 初始化用于路径长度计算的变量，包括长度 len，起始点坐标 start_x 和 start_y，前一个点坐标 x1 和 y1，当前点坐标 x2 和 y2，以及一个标志 first 表示是否是第一个顶点

// 声明一个无符号整数 cmd，用于存储顶点源 vs 生成的命令

// 调用顶点源的 rewind 方法，将路径重置到指定的 path_id


        while(!is_stop(cmd = vs.vertex(&x2, &y2)))
        {

// 当顶点源 vs 生成的命令不是停止命令时，执行循环


            if(is_vertex(cmd))
            {
                if(first || is_move_to(cmd))
                {
                    start_x = x2;
                    start_y = y2;
                }
                else
                {
                    len += calc_distance(x1, y1, x2, y2);
                }
                x1 = x2;
                y1 = y2;
                first = false;
            }

// 如果当前命令表示一个顶点：
// - 如果是第一个顶点或者是移动到命令，则更新起始点坐标 start_x 和 start_y
// - 否则，计算前一个点 (x1, y1) 到当前点 (x2, y2) 的距离并累加到长度 len 中，然后更新前一个点坐标为当前点坐标 (x2, y2)


            else
            {
                if(is_close(cmd) && !first)
                {
                    len += calc_distance(x1, y1, start_x, start_y);
                }
            }

// 如果当前命令表示路径关闭，并且不是第一个顶点，则计算前一个点 (x1, y1) 到起始点 (start_x, start_y) 的距离并累加到长度 len 中


        }
        return len;
    }
}

// 循环结束后，返回计算出的路径长度 len


#endif

// 结束 ifndef 指令块，确保头文件内容只被包含一次，避免重复定义错误
```