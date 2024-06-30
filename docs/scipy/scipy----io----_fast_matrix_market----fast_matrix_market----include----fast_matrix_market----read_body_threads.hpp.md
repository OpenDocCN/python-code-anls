# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\read_body_threads.hpp`

```
// 版权声明及许可证声明，指明此代码遵循 BSD 2-Clause 许可证
// SPDX-License-Identifier: BSD-2-Clause
#pragma once

// 包含异步操作和队列的标准库头文件
#include <future>
#include <queue>

// 包含自定义的矩阵市场快速读取头文件
#include "fast_matrix_market.hpp"
// 包含第三方的任务线程池头文件
#include "thirdparty/task_thread_pool.hpp"

// fast_matrix_market 命名空间，用于组织库中的代码
namespace fast_matrix_market {

    // 结构体 line_count_result_s 包含字符串块和行计数对象
    struct line_count_result_s {
        std::string chunk; // 字符串块
        line_counts counts; // 行计数对象

        // 显式构造函数，初始化 chunk 成员
        explicit line_count_result_s(std::string && c): chunk(c) {}
    };

    // line_count_result 是 line_count_result_s 的共享指针类型
    using line_count_result = std::shared_ptr<line_count_result_s>;

    // count_chunk_lines 函数，用于计算字符串块的行数并更新 counts 成员
    inline line_count_result count_chunk_lines(line_count_result lcr) {
        auto [lines, empties] = count_lines(lcr->chunk);

        lcr->counts.file_line = lines; // 更新文件行数
        lcr->counts.element_num = lines - empties; // 更新非空行数
        return lcr;
    }

    // process_chunk 函数模板，根据不同编译格式处理字符串块
    template <typename HANDLER, compile_format FORMAT = compile_all>
#ifdef FMM_NO_VECTOR
                // 如果未定义 FMM_NO_VECTOR，抛出异常，不支持向量矩阵市场文件
                throw no_vector_support("Vector Matrix Market files not supported.");
#else
                // 使用线程池异步处理字符串块
                parse_futures.push(pool.submit([=]() mutable {
                    read_chunk_vector_coordinate(lcr->chunk, header, lc, chunk_handler, options);
                    return lcr;
                }));
#endif
            }

            // 更新 lc 的行数计数器，用于下一个字符串块
            lc.file_line += lcr->counts.file_line;
            lc.element_num += lcr->counts.element_num;
        }

        // 等待所有解析任务完成，若有解析错误则抛出异常
        while (!parse_futures.empty()) {
            parse_futures.front().get(); // 等待第一个异步任务完成
            parse_futures.pop(); // 弹出已完成的异步任务
        }

        return lc; // 返回最终的行数计数器
    }
}
```