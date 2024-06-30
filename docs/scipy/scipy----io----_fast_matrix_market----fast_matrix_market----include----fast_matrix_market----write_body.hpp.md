# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\write_body.hpp`

```
// 版权声明，指明此代码版权归Adam Lugowski所有，受BSD 2条款许可证管辖，详见LICENSE.txt文件
// SPDX-License-Identifier: BSD-2-Clause

// 预处理指令，确保此头文件只包含一次
#pragma once

// 包含自定义的fast_matrix_market.hpp头文件
#include "fast_matrix_market.hpp"

// 包含多线程写入功能的头文件
#include "write_body_threads.hpp"

// fast_matrix_market命名空间
namespace fast_matrix_market {

    /**
     * 根据要写入的值的C++类型获取头部字段类型。
     * 当T为整数类型时启用此模板。
     */
    template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    field_type get_field_type([[maybe_unused]] const T* type) {
        return integer;
    }

    /**
     * 根据要写入的值的C++类型获取头部字段类型。
     * 当T为浮点数类型时启用此模板。
     */
    template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
    field_type get_field_type([[maybe_unused]] const T* type) {
        return real;
    }

    /**
     * 根据要写入的值的C++类型获取头部字段类型。
     * 当T为复数类型时启用此模板。
     */
    template <typename T, typename std::enable_if<is_complex<T>::value, int>::type = 0>
    field_type get_field_type([[maybe_unused]] const T* type) {
        return complex;
    }

    /**
     * 根据要写入的值的C++类型获取头部字段类型。
     * 当T为pattern_placeholder_type类型时启用此模板。
     */
    template <typename T, typename std::enable_if<std::is_same<T, pattern_placeholder_type>::value, int>::type = 0>
    field_type get_field_type([[maybe_unused]] const T* type) {
        return pattern;
    }

    /**
     * 顺序写入Matrix Market格式的数据体。
     *
     * 将数据分块计算并顺序写入。
     */
    template <typename FORMATTER>
    void write_body_sequential(std::ostream& os,
                               FORMATTER& formatter, const write_options& options = {}) {

        // 循环直到格式化器没有下一个块为止
        while (formatter.has_next()) {
            // 获取下一个数据块
            std::string chunk = formatter.next_chunk(options)();

            // 将块写入输出流
            os.write(chunk.c_str(), (std::streamsize)chunk.size());
        }
    }

    /**
     * 写入Matrix Market格式的数据体。
     *
     * @tparam FORMATTER 实现块写入的类。
     */
    template <typename FORMATTER>
    void write_body(std::ostream& os,
                    FORMATTER& formatter, const write_options& options = {}) {
        // 如果支持并行且线程数不为1，则使用多线程写入
        if (options.parallel_ok && options.num_threads != 1) {
            write_body_threads(os, formatter, options);
            return;
        }
        // 否则顺序写入
        write_body_sequential(os, formatter, options);
    }

} // namespace fast_matrix_market
```