# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\write_body_threads.hpp`

```
// 版权声明和许可证信息
// 版权所有 (C) 2022 Adam Lugowski。使用本源代码受 LICENSE.txt 文件中的 BSD 2-clause 许可证约束。
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include <queue> // 引入标准队列容器

#include "fast_matrix_market.hpp" // 引入自定义的 fast_matrix_market 头文件
#include "thirdparty/task_thread_pool.hpp" // 引入第三方的任务线程池头文件

namespace fast_matrix_market {
    /**
     * 并行写入 Matrix Market 文件内容主体部分。
     *
     * 基于块的方法使得可以并行处理。每个块由 FORMATTER 类负责写入。
     * @tparam FORMATTER 实现类，用于写入块数据。
     */
    template <typename FORMATTER>
    void write_body_threads(std::ostream& os,
                            FORMATTER& formatter, const write_options& options = {}) {
        /*
         * 要求：
         * 1. 由格式化器按顺序创建块。
         * 2. 块可以并行计算（即调用 operator()）。
         * 3. 块的结果必须按创建顺序写入。
         *
         * 实际上这是一个流水线，包含一个串行的生产者（块生成器）、并行的工作者和一个串行的消费者（写入器）。
         *
         * 最大的挑战在于按顺序写入所有块结果。
         *
         * 我们采用简单的方法。主线程处理串行的块生成和 I/O 操作，
         * 而线程池执行并行工作。
         */
        std::queue<std::future<std::string>> futures; // 存放未来的任务结果队列
        task_thread_pool::task_thread_pool pool(options.num_threads); // 创建线程池对象

        // 可以同时处理的并发块数量。
        // 太少可能会使工作者线程饥饿（例如由于不均匀的块分割），
        // 太多会增加成本，例如在写入块结果之前在内存中存储块数据。
        const int inflight_count = 2 * (int)pool.get_num_threads();

        // 开始计算任务。
        for (int batch_i = 0; batch_i < inflight_count && formatter.has_next(); ++batch_i) {
            // 可以直接推送块，但是 MSVC 会出现问题。
            futures.push(pool.submit([](auto chunk){ return chunk(); }, formatter.next_chunk(options)));
//            futures.push(pool.submit(formatter.next_chunk(options)));
        }

        // 按顺序写入可用的块。
        while (!futures.empty()) {
            std::string chunk = futures.front().get(); // 获取队列头部的任务结果
            futures.pop(); // 弹出已处理的任务

            // 如果还有下一个块准备好，则开始另一个替换它。
            if (formatter.has_next()) {
                futures.push(pool.submit([](auto chunk){ return chunk(); }, formatter.next_chunk(options)));
            }

            // 将当前块写入输出流。
            os.write(chunk.c_str(), (std::streamsize) chunk.size());
        }
    }
}
```