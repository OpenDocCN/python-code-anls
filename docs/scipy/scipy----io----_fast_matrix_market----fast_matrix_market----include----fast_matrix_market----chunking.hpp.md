# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\include\fast_matrix_market\chunking.hpp`

```
// 版权声明和许可证声明
// 版权所有 (C) 2022 Adam Lugowski。所有权利保留。
// 使用此源代码受 LICENSE.txt 文件中的 BSD 2-Clause 许可证管辖。
// SPDX-License-Identifier: BSD-2-Clause

// 只包含一次头文件的宏定义
#pragma once

// 包含标准输入输出流和字符串处理的头文件
#include <istream>
#include <string>

// 定义命名空间 fast_matrix_market
namespace fast_matrix_market {

    // 内联函数，用于从输入流中读取数据块到给定的 chunk 中
    inline void get_next_chunk(std::string& chunk, std::istream &instream, const read_options &options) {
        // 额外的数据块字节以确保行尾剩余空间
        constexpr size_t chunk_extra = 4096;
        size_t chunk_length = 0;

        // 确保 chunk 有足够的空间
        chunk.resize(options.chunk_size_bytes);

        // 从流中读取数据块
        auto bytes_to_read = chunk.size() > chunk_extra ? (std::streamsize) (chunk.size() - chunk_extra) : 0;
        if (bytes_to_read > 0) {
            instream.read(chunk.data(), bytes_to_read);
            auto num_read = instream.gcount();
            chunk_length = num_read;

            // 测试是否到达流末尾
            if (num_read == 0 || instream.eof() || chunk[chunk_length - 1] == '\n') {
                chunk.resize(chunk_length);
                return;
            }
        }

        // 读取行尾剩余内容并追加到 chunk
        std::string suffix;
        std::getline(instream, suffix);
        if (instream.good()) {
            suffix += "\n";
        }

        if (chunk_length + suffix.size() > chunk.size()) {
            // 行尾剩余内容超出额外空间，必须进行复制
            chunk.resize(chunk_length);
            chunk += suffix;
        } else {
            // 行尾剩余内容可以完全放入 chunk
            std::copy(suffix.begin(), suffix.end(), chunk.begin() + (ptrdiff_t) chunk_length);
            chunk_length += suffix.size();
            chunk.resize(chunk_length);
        }
    }

    // 返回从输入流中获取的数据块的字符串表示
    inline std::string get_next_chunk(std::istream &instream, const read_options &options) {
        // 分配数据块内存空间
        std::string chunk(options.chunk_size_bytes, ' ');
        // 调用 get_next_chunk 函数从输入流中获取数据块
        get_next_chunk(chunk, instream, options);
        // 返回获取的数据块字符串
        return chunk;
    }

    // 模板函数，用于检查迭代器范围内的字符是否全为空格
    template <typename ITER>
    bool is_all_spaces(ITER begin, ITER end) {
        return std::all_of(begin, end, [](char c) { return c == ' ' || c == '\t' || c == '\r'; });
    }

    /**
     * 计算多行字符串中的总行数和空行数。
     */

} // 命名空间 fast_matrix_market 结束
    // 统计给定文本块中的行数和空行数
    inline std::pair<int64_t, int64_t> count_lines(const std::string& chunk) {
        // 初始化行数和空行数
        int64_t num_newlines = 0;
        int64_t num_empty_lines = 0;

        // 设置迭代器起始位置和结束位置
        auto pos = std::cbegin(chunk);
        auto end = std::cend(chunk);
        // 记录当前行的起始位置
        auto line_start = pos;
        
        // 遍历文本块中的每个字符
        for (; pos != end; ++pos) {
            // 遇到换行符时，增加行数，并检查是否为空行
            if (*pos == '\n') {
                ++num_newlines;
                if (is_all_spaces(line_start, pos)) {
                    ++num_empty_lines;
                }
                line_start = pos + 1;  // 更新下一行的起始位置
            }
        }

        // 如果最后一行没有换行符结尾，但可能仍然是空行
        if (line_start != end) {
            if (is_all_spaces(line_start, end)) {
                ++num_empty_lines;
            }
        }

        // 如果文本块中没有换行符，但不为空，则为单行非空内容
        if (num_newlines == 0) {
            if (chunk.empty()) {
                num_empty_lines = 1;  // 如果文本块为空，则为一行空行
            }
            return std::make_pair(1, num_empty_lines);
        }

        // 如果文本块最后一个字符不是换行符，则增加行数
        if (chunk[chunk.size()-1] != '\n') {
            ++num_newlines;
        }

        // 返回统计结果：行数和空行数的 pair
        return std::make_pair(num_newlines, num_empty_lines);
    }
}



# 这行代码表示一个代码块的结束
```