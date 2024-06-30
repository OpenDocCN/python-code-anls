# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\src\_fmm_core.hpp`

```
/**
 * Copyright (C) 2023 Adam Lugowski. All rights reserved.
 * Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#ifdef FMM_SCIPY_PRUNE
#define FMM_NO_VECTOR
#endif

#include <Python.h>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "pystreambuf.h"

#include <fast_matrix_market/types.hpp>
namespace fast_matrix_market {
    // Be able to set unsigned-integer field type. This type is only used by SciPy to represent uint64 values.
    field_type get_field_type([[maybe_unused]] const uint32_t* type);

    field_type get_field_type([[maybe_unused]] const uint64_t* type);
}
#include <fast_matrix_market/fast_matrix_market.hpp>

namespace py = pybind11;
using namespace pybind11::literals;
namespace fmm = fast_matrix_market;

/**
 * A structure that represents an open MatrixMarket file or stream (for reading)
 */
struct read_cursor {
    /**
     * Open a file.
     * @param filename Path to the file to open.
     */
    read_cursor(const std::string& filename): stream_ptr(std::make_shared<std::ifstream>(filename)) {}

    /**
     * Use a Python stream. Needs to be a shared_ptr because this stream object needs to stay alive for the lifetime
     * of this cursor object.
     * @param external Shared pointer to a pystream::istream object.
     */
    read_cursor(std::shared_ptr<pystream::istream>& external): stream_ptr(external) {}

    std::shared_ptr<std::istream> stream_ptr; ///< Shared pointer to the input stream.

    fmm::matrix_market_header header{}; ///< Header information for MatrixMarket format.
    fmm::read_options options{}; ///< Options for reading from MatrixMarket format.

    /**
     * Provides access to the underlying stream.
     * @return Reference to the underlying input stream.
     */
    std::istream& stream() {
        return *stream_ptr;
    }

    /**
     * Finish using the cursor. If a file has been opened it will be closed.
     */
    void close() {
        // If stream is a std::ifstream() then close the file.
        std::ifstream* f = dynamic_cast<std::ifstream*>(stream_ptr.get());
        if (f != nullptr) {
            f->close();
        }

        // Remove this reference to the stream.
        stream_ptr.reset();
    }
};

/**
 * A structure that represents an open MatrixMarket file or stream (for writing)
 */
struct write_cursor {
    /**
     * Open a file.
     * @param filename Path to the file to open.
     */
    write_cursor(const std::string& filename): stream_ptr(std::make_unique<std::ofstream>(filename, std::ios_base::out | std::ios_base::binary)) {}

    /**
     * Use a Python stream. Needs to be a shared_ptr because this stream object needs to stay alive for the lifetime
     * of this cursor object.
     * @param external Shared pointer to a pystream::ostream object.
     */
    write_cursor(std::shared_ptr<pystream::ostream>& external): stream_ptr(external) {}

    std::shared_ptr<std::ostream> stream_ptr; ///< Shared pointer to the output stream.

    fmm::matrix_market_header header{}; ///< Header information for MatrixMarket format.
    fmm::write_options options{}; ///< Options for writing to MatrixMarket format.

    /**
     * Provides access to the underlying stream.
     * @return Reference to the underlying output stream.
     */
    std::ostream& stream() {
        return *stream_ptr;
    }

    /**
     * Finish using the cursor. Flush the backing stream, and if a file has been opened it will be closed.
     */
    // 关闭函数，用于关闭流
    void close() {
        // 将流指针转换为 std::ofstream 指针
        std::ofstream* f = dynamic_cast<std::ofstream*>(stream_ptr.get());
        // 如果指针不为空，则关闭文件
        if (f != nullptr) {
            f->close();
        } else {
            // 否则刷新流
            stream_ptr->flush();
        }

        // 重置流指针，移除对流的引用
        stream_ptr.reset();
    }
};

/**
 * An iterator adapter over py::array_t numpy arrays.
 *
 * This allows using the iterator-based fast_matrix_market methods.
 */
template<typename ARR, typename T>
class py_array_iterator
{
public:
    using value_type = T;
    using difference_type = int64_t;

    // Constructor taking a reference to a py::array_t
    py_array_iterator(ARR& array) : array(array), index(0) {}

    // Constructor taking a reference to a py::array_t and an initial index
    py_array_iterator(ARR& array, int64_t index) : array(array), index(index) {}

    // Copy constructor
    py_array_iterator(const py_array_iterator &rhs) : array(rhs.array), index(rhs.index) {}

    /* Assignment operator */
    // Assignment operator overload
    py_array_iterator& operator=(const py_array_iterator &rhs) {index = rhs.index; return *this;}

    /* Addition assignment */
    // Addition assignment operator overload
    py_array_iterator& operator+=(difference_type rhs) {index += rhs; return *this;}

    /* Subtraction assignment */
    // Subtraction assignment operator overload
    py_array_iterator& operator-=(difference_type rhs) {index -= rhs; return *this;}

    // Dereference operator returning the current element
    T operator*() const {return array(index);}

//    T* operator->() const {return index;}
    // Subscript operator returning the element at index + rhs
    T operator[](difference_type rhs) const {return array(index + rhs);}

    // Prefix increment operator (++it)
    py_array_iterator& operator++() {++index; return *this;}

    // Prefix decrement operator (--it)
    py_array_iterator& operator--() {--index; return *this;}

    // Postfix increment operator (it++)
    py_array_iterator operator++(int) {py_array_iterator tmp(*this); ++index; return tmp;}

    // Postfix decrement operator (it--)
    py_array_iterator operator--(int) {py_array_iterator tmp(*this); --index; return tmp;}

    /* Subtraction operator */
    // Subtraction operator returning the difference in indices
    difference_type operator-(const py_array_iterator& rhs) const {return index-rhs.index;}

    /* Addition operator */
    // Addition operator returning a new iterator with index + rhs
    py_array_iterator operator+(difference_type rhs) const {return py_array_iterator(array, index+rhs);}

    // Subtraction operator returning a new iterator with index - rhs
    py_array_iterator operator-(difference_type rhs) const {return py_array_iterator(array, index-rhs);}

    // Friend function for addition operator with lhs as difference_type
    friend py_array_iterator operator+(difference_type lhs, const py_array_iterator& rhs) {return py_array_iterator(rhs.array, lhs+rhs.index);}

    // Friend function for subtraction operator with lhs as difference_type
    friend py_array_iterator operator-(difference_type lhs, const py_array_iterator& rhs) {return py_array_iterator(rhs.array, lhs-rhs.index);}

    // Equality comparison operator
    bool operator==(const py_array_iterator& rhs) const {return index == rhs.index;}

    // Inequality comparison operator
    bool operator!=(const py_array_iterator& rhs) const {return index != rhs.index;}

    // Greater than comparison operator
    bool operator>(const py_array_iterator& rhs) const {return index > rhs.index;}

    // Less than comparison operator
    bool operator<(const py_array_iterator& rhs) const {return index < rhs.index;}

    // Greater than or equal to comparison operator
    bool operator>=(const py_array_iterator& rhs) const {return index >= rhs.index;}

    // Less than or equal to comparison operator
    bool operator<=(const py_array_iterator& rhs) const {return index <= rhs.index;}

private:
    ARR& array;         // Reference to the py::array_t object
    int64_t index;      // Current index in the array
};


/**
 * Write Python triplets to MatrixMarket.
 */
template <typename IT, typename VT>
void write_body_coo(write_cursor& cursor, const std::tuple<int64_t, int64_t>& shape,
                    py::array_t<IT>& rows, py::array_t<IT>& cols, py::array_t<VT>& data) {
    // Check if the sizes of rows and cols arrays match
    if (rows.size() != cols.size()) {
        // Throw an exception if sizes do not match
        throw std::invalid_argument("len(row) must equal len(col).");
    }
    # 如果行数不等于数据长度，并且数据长度不为零，则抛出无效参数异常
    if (rows.size() != data.size() && data.size() != 0) {
        throw std::invalid_argument("len(row) must equal len(data).");
    }

    # 设置游标的行数、列数和非零元素数
    cursor.header.nrows = std::get<0>(shape);
    cursor.header.ncols = std::get<1>(shape);
    cursor.header.nnz = rows.size();

    # 设置头部信息的对象类型为矩阵
    cursor.header.object = fmm::matrix;
    # 根据条件设置头部信息的字段类型
    cursor.header.field = (data.size() == 0 ? (cursor.header.nnz == 0 ? fmm::real : fmm::pattern) : fmm::get_field_type((const VT*)nullptr));
    # 设置头部信息的格式为坐标格式
    cursor.header.format = fmm::coordinate;

    # 将头部信息写入流
    fmm::write_header(cursor.stream(), cursor.header, cursor.options);

    # 获取行、列、数据的未检查版本
    auto rows_unchecked = rows.unchecked();
    auto cols_unchecked = cols.unchecked();
    auto data_unchecked = data.unchecked();

    # 创建行格式化器
    fmm::line_formatter<IT, VT> lf(cursor.header, cursor.options);
    # 创建三元组格式化器，并初始化迭代器
    auto formatter = fmm::triplet_formatter(lf,
                                            py_array_iterator<decltype(rows_unchecked), IT>(rows_unchecked),
                                            py_array_iterator<decltype(rows_unchecked), IT>(rows_unchecked, rows_unchecked.size()),
                                            py_array_iterator<decltype(cols_unchecked), IT>(cols_unchecked),
                                            py_array_iterator<decltype(cols_unchecked), IT>(cols_unchecked, cols_unchecked.size()),
                                            py_array_iterator<decltype(data_unchecked), VT>(data_unchecked),
                                            py_array_iterator<decltype(data_unchecked), VT>(data_unchecked, data_unchecked.size()));
    # 将格式化后的数据写入流
    fmm::write_body(cursor.stream(), formatter, cursor.options);
    # 关闭游标
    cursor.close();
#ifndef FMM_SCIPY_PRUNE
/**
 * Write Python CSC/CSR to MatrixMarket.
 */
template <typename IT, typename VT>
void write_body_csc(write_cursor& cursor, const std::tuple<int64_t, int64_t>& shape,
                    py::array_t<IT>& indptr, py::array_t<IT>& indices, py::array_t<VT>& data, bool is_csr) {
    // 检查索引和数据数组的长度是否一致，若不一致且数据数组非空则抛出异常
    if (indices.size() != data.size() && data.size() != 0) {
        throw std::invalid_argument("len(indices) must equal len(data).");
    }

    // 设置头部信息：行数、列数、非零元素数目
    cursor.header.nrows = std::get<0>(shape);
    cursor.header.ncols = std::get<1>(shape);
    cursor.header.nnz = indices.size();

    // 检查indptr数组长度是否符合矩阵形状，根据is_csr确定检查行指针（CSR）还是列指针（CSC）
    if ((is_csr && indptr.size() != cursor.header.nrows + 1) ||
        (!is_csr && indptr.size() != cursor.header.ncols + 1)) {
        throw std::invalid_argument("indptr length does not match matrix shape.");
    }

    // 设置头部信息：对象类型为矩阵，字段类型根据数据数组是否为空确定为实数或模式，格式为坐标格式，对称性为一般对称
    cursor.header.object = fmm::matrix;
    cursor.header.field = (data.size() == 0 ? (cursor.header.nnz == 0 ? fmm::real : fmm::pattern) : fmm::get_field_type((const VT*)nullptr));
    cursor.header.format = fmm::coordinate;
    cursor.header.symmetry = fmm::general;

    // 写入头部信息到流中
    fmm::write_header(cursor.stream(), cursor.header, cursor.options);

    // 获取未检查的indptr、indices和data数组，用于格式化数据
    auto indptr_unchecked = indptr.unchecked();
    auto indices_unchecked = indices.unchecked();
    auto data_unchecked = data.unchecked();

    // 创建线格式化器和格式化器对象，根据is_csr选择CSR或CSC格式
    fmm::line_formatter<IT, VT> lf(cursor.header, cursor.options);
    auto formatter = fmm::csc_formatter(lf,
                                        py_array_iterator<decltype(indptr_unchecked), IT>(indptr_unchecked),
                                        py_array_iterator<decltype(indptr_unchecked), IT>(indptr_unchecked, indptr_unchecked.size() - 1),
                                        py_array_iterator<decltype(indices_unchecked), IT>(indices_unchecked),
                                        py_array_iterator<decltype(indices_unchecked), IT>(indices_unchecked, indices_unchecked.size()),
                                        py_array_iterator<decltype(data_unchecked), VT>(data_unchecked),
                                        py_array_iterator<decltype(data_unchecked), VT>(data_unchecked, data_unchecked.size()),
                                        is_csr);
    // 将格式化后的数据写入流
    fmm::write_body(cursor.stream(), formatter, cursor.options);
    // 关闭流
    cursor.close();
}
#endif

// 初始化函数声明
void init_read_array(py::module_ &);
void init_write_array(py::module_ &);
void init_read_coo(py::module_ &);
void init_write_coo_32(py::module_ &);
void init_write_coo_64(py::module_ &);
void init_write_csc_32(py::module_ &);
void init_write_csc_64(py::module_ &);
```