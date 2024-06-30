# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\src\pystreambuf.h`

```
// SPDX-License-Identifier: BSD-3-Clause
/*
Based on https://gist.github.com/asford/544323a5da7dddad2c9174490eb5ed06

Original license text
---------------------

This component utilizes components derived from cctbx, available at
http://cci.lbl.gov/cctbx_sources/boost_adaptbx/python_streambuf.h

*** License agreement ***

cctbx Copyright (c) 2006, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject to
receipt of any required approvals from the U.S. Dept. of Energy).  All
rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes,
patches, or upgrades to the features, functionality or performance of
the source code ("Enhancements") to anyone; however, if you choose to
make your Enhancements available either publicly, or directly to
Lawrence Berkeley National Laboratory, without imposing a separate
written license agreement for such Enhancements, then you hereby grant
the following license: a  non-exclusive, royalty-free perpetual license
to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.

*/

#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>

#include <streambuf>
#include <iostream>

namespace py = pybind11;

/// A stream buffer getting data from and putting data into a Python file object
/** 
 * This class defines a stream buffer that interacts with Python file objects.
 * It facilitates reading data from and writing data to Python streams.
 */
class python_streambuf : public std::streambuf {
public:
    /// Constructor taking a Python file object
    /**
     * Initializes the stream buffer with a Python file object.
     * \param file_ptr A pointer to a Python file object.
     */
    explicit python_streambuf(PyObject *file_ptr) : file(file_ptr) {}

protected:
    /// Read data from the input sequence and write it to the buffer
    /**
     * Reads data from the Python file object and writes it to the buffer.
     * \param s Pointer to the buffer into which the characters are written.
     * \param n Number of characters to extract.
     * \return The number of characters transferred.
     */
    std::streamsize xsgetn(char_type *s, std::streamsize n) override {
        // Call Python read method to get data from the file object
        PyObject *result = PyObject_CallMethod(file, "read", "n", n);
        if (!result) return -1; // Error handling
        const char *data = PyUnicode_AsUTF8(result); // Convert result to UTF-8
        std::streamsize len = std::strlen(data); // Calculate length of data
        std::memcpy(s, data, len); // Copy data to buffer
        Py_DECREF(result); // Release Python object reference
        return len; // Return number of characters transferred
    }

    /// Write up to n characters from the buffer to the output sequence
    /**
     * Writes up to n characters from the buffer to the Python file object.
     * \param s Pointer to the buffer from which characters are written.
     * \param n Number of characters to write.
     * \return The number of characters written.
     */
    std::streamsize xsputn(const char_type *s, std::streamsize n) override {
        // Convert buffer data to Python bytes object
        PyObject *data = PyBytes_FromStringAndSize(s, n);
        if (!data) return -1; // Error handling
        // Call Python write method to write data to the file object
        int result = PyObject_CallMethod(file, "write", "O", data);
        Py_DECREF(data); // Release Python object reference
        return result; // Return number of characters written
    }

private:
    PyObject *file; ///< Python file object pointer
};
    /**
     * Given a C++ function acting on a standard stream, e.g.
     *
     * \code
     * void read_inputs(std::istream& input) {
     *   ...
     *   input >> something >> something_else;
     * }
     * \endcode
     *
     * and given a piece of Python code which creates a file-like object,
     * to be able to pass this file object to that C++ function, e.g.
     *
     * \code
     * import gzip
     * gzip_file_obj = gzip.GzipFile(...)
     * read_inputs(gzip_file_obj)
     * \endcode
     *
     * and have the standard stream pull data from and put data into the Python
     * file object.
     */
    
    /**
     * When Python \c read_inputs() returns, the Python object is able to
     * continue reading or writing where the C++ code left off.
     */
    
    /**
     * Operations in C++ on mere files should be competitively fast compared
     * to the direct use of \c std::fstream.
     */
    
    /**
     * \b Motivation
     *
     * - the standard Python library offer of file-like objects (files,
     *   compressed files and archives, network, ...) is far superior to the
     *   offer of streams in the C++ standard library and Boost C++ libraries.
     *
     * - i/o code involves a fair amount of text processing which is more
     *   efficiently prototyped in Python but then one may need to rewrite
     *   a time-critical part in C++, in as seamless a manner as possible.
     */
    
    /**
     * \b Usage
     *
     * This is 2-step:
     *
     * - a trivial wrapper function
     *
     * \code
     * using boost_adaptbx::python::streambuf;
     * void read_inputs_wrapper(streambuf& input)
     * {
     *   streambuf::istream is(input);
     *   read_inputs(is);
     * }
     *
     * def("read_inputs", read_inputs_wrapper);
     * \endcode
     *
     * which has to be written every time one wants a Python binding for
     * such a C++ function.
     *
     * - the Python side
     *
     * \code
     * from boost.python import streambuf
     * read_inputs(streambuf(python_file_obj=obj, buffer_size=1024))
     * \endcode
     *
     * \c buffer_size is optional. See also: \c default_buffer_size
     */
    
    /**
     * Note: references are to the C++ standard (the numbers between parentheses
     * at the end of references are margin markers).
     */
*/
namespace pystream{

class streambuf : public std::basic_streambuf<char>
{
  private:
    typedef std::basic_streambuf<char> base_t;

  public:
    /* The syntax
        using base_t::char_type;
       would be nicer but Visual Studio C++ 8 chokes on it
    */
    // 定义基类中的类型别名
    typedef base_t::char_type   char_type;
    typedef base_t::int_type    int_type;
    typedef base_t::pos_type    pos_type;
    typedef base_t::off_type    off_type;
    typedef base_t::traits_type traits_type;

    /// The default size of the read and write buffer.
    /** They are respectively used to buffer data read from and data written to
        the Python file object. It can be modified from Python.
    */
    // 默认的读写缓冲区大小
    static inline std::size_t default_buffer_size = 1024;

    /// Construct from a Python file object
    /** if buffer_size is 0 the current default_buffer_size is used.
    */
    // 构造函数，从 Python 文件对象构造
    streambuf(
      py::object& python_file_obj,
      std::size_t buffer_size_=0)
    :
      // 获取 Python 文件对象的 read、write、seek、tell 方法
      py_read (getattr(python_file_obj, "read", py::none())),
      py_write (getattr(python_file_obj, "write", py::none())),
      py_seek (getattr(python_file_obj, "seek", py::none())),
      py_tell (getattr(python_file_obj, "tell", py::none())),
      // 缓冲区大小，默认为 default_buffer_size 或者指定的 buffer_size_
      buffer_size(buffer_size_ != 0 ? buffer_size_ : default_buffer_size),
      // 写入缓冲区的初始化
      write_buffer(0),
      // Python 文件对象中读取缓冲区结束位置的初始化
      pos_of_read_buffer_end_in_py_file(0),
      // Python 文件对象中写入缓冲区结束位置的初始化
      pos_of_write_buffer_end_in_py_file(buffer_size),
      // 最远写指针位置的初始化
      farthest_pptr(0)
    {
      assert(buffer_size != 0);
      /* Some Python file objects (e.g. sys.stdout and sys.stdin)
         have non-functional seek and tell. If so, assign None to
         py_tell and py_seek.
       */
      // 检查 Python 文件对象是否支持 tell 方法，不支持则设置为 None
      if (!py_tell.is_none()) {
        try {
          // 调用 tell 方法尝试获取当前位置
          py_tell();
        }
        catch (py::error_already_set& err) {
          // 如果出现异常，将 tell 和 seek 方法设置为 None
          py_tell = py::none();
          py_seek = py::none();
          // 恢复异常状态并清除异常信息
          err.restore();
          PyErr_Clear();
        }
      }

      // 如果支持 write 方法，则初始化写入缓冲区
      if (!py_write.is_none()) {
        // 为便于调试，创建 C 风格的字符串
        write_buffer = new char[buffer_size + 1];
        write_buffer[buffer_size] = '\0';
        // 设置写入缓冲区的指针范围
        setp(write_buffer, write_buffer + buffer_size);  // 27.5.2.4.5 (5)
        // 记录当前写指针位置
        farthest_pptr = pptr();
      }
      else {
        // 如果不支持 write 方法，首次输出将调用 overflow 方法
        setp(0, 0);
      }

      // 如果支持 tell 方法，则获取当前位置，并初始化读写缓冲区结束位置
      if (!py_tell.is_none()){
        // 获取当前位置并转换为 off_type 类型
        off_type py_pos = py_tell().cast<off_type>();
        // 设置读缓冲区结束位置
        pos_of_read_buffer_end_in_py_file = py_pos;
        // 设置写缓冲区结束位置
        pos_of_write_buffer_end_in_py_file = py_pos;
      }
    }

    /// Mundane destructor freeing the allocated resources
    // 析构函数，释放分配的资源
    virtual ~streambuf() {
      if (write_buffer) delete[] write_buffer;
    }

    /// C.f. C++ standard section 27.5.2.4.3
    /** It is essential to override this virtual function for the stream
        member function readsome to work correctly (c.f. 27.6.1.3, alinea 30)
     */
    /// 返回当前缓冲区内未读取字符的数量（C++ 标准第 27.5.2.4.3 节）
    virtual std::streamsize showmanyc() {
      int_type const failure = traits_type::eof();
      // 获取当前缓冲区的状态
      int_type status = underflow();
      // 如果状态为失败，则返回-1，表示没有未读取的字符
      if (status == failure) return -1;
      // 返回未读取字符的数量
      return egptr() - gptr();
    }

    /// 返回当前输入序列的下一个字符，不移动文件位置（C++ 标准第 27.5.2.4.4 节）
    virtual int_type underflow() {
      int_type const failure = traits_type::eof();
      // 如果 Python 文件对象不存在 'read' 属性，则抛出异常
      if (py_read.is_none()) {
        throw std::invalid_argument(
          "That Python file object has no 'read' attribute");
      }
      // 使用指定大小读取数据到缓冲区
      read_buffer = py_read(buffer_size);
      char *read_buffer_data;
      py::ssize_t py_n_read;
      // 将 Python 字符串转换为 C 字符串和大小
      if (PYBIND11_BYTES_AS_STRING_AND_SIZE(read_buffer.ptr(),
            &read_buffer_data, &py_n_read) == -1) {
        setg(0, 0, 0);
        // 如果读取的数据不是字符串，则抛出异常
        throw std::invalid_argument(
          "The method 'read' of the Python file object "
          "did not return a string.");
      }
      // 记录已读取的数据量
      off_type n_read = (off_type)py_n_read;
      pos_of_read_buffer_end_in_py_file += n_read;
      // 设置缓冲区指针
      setg(read_buffer_data, read_buffer_data, read_buffer_data + n_read);
      // 如果未读取任何数据，则返回失败标志
      if (n_read == 0) return failure;
      // 返回缓冲区中的第一个字符
      return traits_type::to_int_type(read_buffer_data[0]);
    }

    /// 将指定字符写入输出序列，不移动文件位置（C++ 标准第 27.5.2.4.5 节）
    virtual int_type overflow(int_type c=traits_type::eof()) {
      // 如果 Python 文件对象不存在 'write' 属性，则抛出异常
      if (py_write.is_none()) {
        throw std::invalid_argument(
          "That Python file object has no 'write' attribute");
      }
      // 更新最远写入指针位置
      farthest_pptr = (std::max)(farthest_pptr, pptr());
      // 计算已写入的数据量
      off_type n_written = (off_type)(farthest_pptr - pbase());

      // 逐块写入所有未写入的数据，支持32位系统
      // py::bytes 构造函数要求第二个参数的大小 <= sizeof(ssize_t)
      // 在32位系统上可能出现问题，因为 off_type 可能是 long long，而 ssize_t 是 int
      for (off_type offset = 0; offset < n_written; ) {
        off_type chunk_len = std::min((n_written - offset), (off_type)(1 << 25));
        py::bytes chunk(pbase() + offset, (int)chunk_len);
        py_write(chunk);
        offset += chunk_len;
      }

      // 如果指定了字符 c，则将其写入输出序列
      if (!traits_type::eq_int_type(c, traits_type::eof())) {
        char cs = traits_type::to_char_type(c);
        py_write(py::bytes(&cs, 1));
        n_written++;
      }
      // 如果有数据被写入，则更新写入缓冲区结束位置，并重置写入指针
      if (n_written) {
        pos_of_write_buffer_end_in_py_file += n_written;
        setp(pbase(), epptr());
        // ^^^ 27.5.2.4.5 (5)
        farthest_pptr = pptr();
      }
      // 返回写入操作的结果
      return traits_type::eq_int_type(
        c, traits_type::eof()) ? traits_type::not_eof(c) : c;
    }

    /// 更新 Python 文件对象，以反映该流缓冲区的状态
    /** 将写入缓冲区中的所有数据写入 Python 文件对象，并相应地设置后者的 seek 位置
        如果没有写入缓冲区或者它为空，但存在非空的读取缓冲区，则将 Python 文件对象的 seek 位置设置为
        读取缓冲区中的 seek 位置（C++ 标准第 27.5.2.4.2 节）。
    */
    // 虚函数，用于同步缓冲区状态到文件，返回操作结果
    virtual int sync() {
      int result = 0;
      // 更新最远写入位置
      farthest_pptr = (std::max)(farthest_pptr, pptr());
      // 如果存在最远写入位置并且它大于写入位置
      if (farthest_pptr && farthest_pptr > pbase()) {
        // 计算当前写入位置与最远写入位置之间的偏移
        off_type delta = pptr() - farthest_pptr;
        // 调用溢出处理函数
        int_type status = overflow();
        // 如果溢出处理返回 EOF，设置操作结果为 -1
        if (traits_type::eq_int_type(status, traits_type::eof())) result = -1;
        // 如果有 py_seek 函数，则调用它进行定位
        if (!py_seek.is_none()) py_seek(delta, 1);
      }
      // 如果存在读取位置并且它小于等于读取结束位置
      else if (gptr() && gptr() < egptr()) {
        // 如果有 py_seek 函数，则调用它进行定位
        if (!py_seek.is_none()) py_seek(gptr() - egptr(), 1);
      }
      // 返回操作结果
      return result;
    }

    /// C.f. C++ standard section 27.5.2.4.2
    /** 优化实现以检查位置是否在缓冲区内，以避免调用 Python 的 seek 或 tell 函数。
        对于许多需要尽可能减少调用 Python 的开销的应用程序（如可能进行大量回溯的解析器），这一点非常重要。
    */
    virtual
    // 设置偏移位置及其方向，返回新的位置
    pos_type seekoff(off_type off, std::ios_base::seekdir way,
                     std::ios_base::openmode which=  std::ios_base::in
                                                   | std::ios_base::out)
    {
      /* 实际上，“which” 只能是 std::ios_base::in 或 std::ios_base::out，
         因为我们在这里是由于流使用此缓冲区调用了 seekp 或 seekg。这简化了代码
         在一些地方的处理。
      */
      // 失败时的常量值
      int const failure = off_type(-1);

      // 如果 py_seek 未定义，抛出无效参数异常
      if (py_seek.is_none()) {
        throw std::invalid_argument(
          "That Python file object has no 'seek' attribute");
      }

      // 如果需要输入操作并且读取指针为空，则尝试填充读取缓冲区
      if (which == std::ios_base::in && !gptr()) {
        // 如果填充返回 EOF，返回失败常量
        if (traits_type::eq_int_type(underflow(), traits_type::eof())) {
          return failure;
        }
      }

      // 计算 Python seek 的 whence 参数
      int whence;
      switch (way) {
        case std::ios_base::beg:
          whence = 0;
          break;
        case std::ios_base::cur:
          whence = 1;
          break;
        case std::ios_base::end:
          whence = 2;
          break;
        default:
          return failure;
      }

      // 尝试不调用 Python 的偏移函数
      off_type result;
      if (!seekoff_without_calling_python(off, way, which, result)) {
        // 需要调用 Python
        if (which == std::ios_base::out) overflow();
        // 对于当前位置方式，如果是输入，则计算偏移量
        if (way == std::ios_base::cur) {
          if      (which == std::ios_base::in)  off -= egptr() - gptr();
          else if (which == std::ios_base::out) off += pptr() - pbase();
        }
        // 调用 Python 的 seek 函数
        py_seek(off, whence);
        // 获取结果并转换为 off_type 类型
        result = off_type(py_tell().cast<off_type>());
        // 如果是输入，填充缓冲区
        if (which == std::ios_base::in) underflow();
      }
      // 返回最终结果
      return result;
    }

    /// C.f. C++ standard section 27.5.2.4.2
    // 设置指定位置及其方向，返回新的位置
    virtual
    pos_type seekpos(pos_type sp,
                     std::ios_base::openmode which=  std::ios_base::in
                                                   | std::ios_base::out)
    {
      // 使用 streambuf 类的 seekoff 方法重新定位流的位置，返回新的位置
      return streambuf::seekoff(sp, std::ios_base::beg, which);
    }

  private:
    // Python 对象，用于读取、写入、定位和告知操作
    py::object py_read, py_write, py_seek, py_tell;

    // 缓冲区大小
    std::size_t buffer_size;

    /* This is actually a Python bytes object and the actual read buffer is
       its internal data, i.e. an array of characters.
     */
    // 实际上是 Python 的 bytes 对象，读取缓冲区是其内部数据，即字符数组
    py::bytes read_buffer;

    /* A mere array of char's allocated on the heap at construction time and
       de-allocated only at destruction time.
    */
    // 仅仅是在构造时分配在堆上的 char 数组，并且仅在析构时释放
    char *write_buffer;

    // 读取缓冲区结束位置在 Python 文件中的位置
    off_type pos_of_read_buffer_end_in_py_file,
             // 写入缓冲区结束位置在 Python 文件中的位置
             pos_of_write_buffer_end_in_py_file;

    // 缓冲区内最远写入的位置
    // 指针指向缓冲区内部的最大值
    char *farthest_pptr;


    // 不调用 Python 的情况下执行 seekoff 操作
    bool seekoff_without_calling_python(
      // 偏移量、寻找的方式、打开模式、结果
      off_type off,
      std::ios_base::seekdir way,
      std::ios_base::openmode which,
      off_type & result)
    {
      // 缓冲区范围和当前位置
      off_type buf_begin, buf_end, buf_cur, upper_bound;
      off_type pos_of_buffer_end_in_py_file;
      if (which == std::ios_base::in) {
        // 读取缓冲区结束位置在 Python 文件中的位置
        pos_of_buffer_end_in_py_file = pos_of_read_buffer_end_in_py_file;
        // 缓冲区起始位置为 eback() 的强制类型转换后的 std::streamsize
        buf_begin = reinterpret_cast<std::streamsize>(eback());
        // 缓冲区当前位置为 gptr() 的强制类型转换后的 std::streamsize
        buf_cur = reinterpret_cast<std::streamsize>(gptr());
        // 缓冲区结束位置为 egptr() 的强制类型转换后的 std::streamsize
        buf_end = reinterpret_cast<std::streamsize>(egptr());
        // 上限为 buf_end
        upper_bound = buf_end;
      }
      else if (which == std::ios_base::out) {
        // 写入缓冲区结束位置在 Python 文件中的位置
        pos_of_buffer_end_in_py_file = pos_of_write_buffer_end_in_py_file;
        // 缓冲区起始位置为 pbase() 的强制类型转换后的 std::streamsize
        buf_begin = reinterpret_cast<std::streamsize>(pbase());
        // 缓冲区当前位置为 pptr() 的强制类型转换后的 std::streamsize
        buf_cur = reinterpret_cast<std::streamsize>(pptr());
        // 缓冲区结束位置为 epptr() 的强制类型转换后的 std::streamsize
        buf_end = reinterpret_cast<std::streamsize>(epptr());
        // farthest_pptr 为 pptr() 和 farthest_pptr 之间的最大值
        farthest_pptr = (std::max)(farthest_pptr, pptr());
        // 上限为 farthest_pptr 强制类型转换后的 std::streamsize + 1
        upper_bound = reinterpret_cast<std::streamsize>(farthest_pptr) + 1;
      }
      else {
           // 抛出运行时错误，控制流经过不应到达的分支
           throw std::runtime_error(
             "Control flow passes through branch that should be unreachable.");
      }

      // "缓冲区坐标" 中的寻找位置
      off_type buf_sought;
      if (way == std::ios_base::cur) {
        // 如果是相对当前位置寻找，缓冲区寻找位置为 buf_cur + off
        buf_sought = buf_cur + off;
      }
      else if (way == std::ios_base::beg) {
        // 如果是相对开始位置寻找，缓冲区寻找位置为 buf_end + (off - pos_of_buffer_end_in_py_file)
        buf_sought = buf_end + (off - pos_of_buffer_end_in_py_file);
      }
      else if (way == std::ios_base::end) {
        // 如果是相对结束位置寻找，返回假
        return false;
      }
      else {
           // 抛出运行时错误，控制流经过不应到达的分支
           throw std::runtime_error(
             "Control flow passes through branch that should be unreachable.");
      }

      // 如果寻找的位置不在缓冲区内，放弃
      if (buf_sought < buf_begin || buf_sought >= upper_bound) return false;

      // 如果是在奇妙的地方
      if      (which == std::ios_base::in)  gbump(buf_sought - buf_cur);
      else if (which == std::ios_base::out) pbump(buf_sought - buf_cur);

      // 结果为 pos_of_buffer_end_in_py_file + (buf_sought - buf_end)
      result = pos_of_buffer_end_in_py_file + (buf_sought - buf_end);
      return true;
    }

  public:

    class istream : public std::istream
    {
      // 定义一个名为 istream 的类，公开继承自 std::istream
      public:
        // 构造函数，接受一个 streambuf 引用作为参数，并将其传递给 std::istream 的构造函数进行初始化
        istream(streambuf& buf) : std::istream(&buf)
        {
          // 设置异常控制位，指示在出现异常时抛出 std::ios_base::failure 异常
          exceptions(std::ios_base::badbit);
        }
    
        // 析构函数，用于销毁 istream 对象
        ~istream() {
          // 如果 istream 对象状态良好（good），则同步缓冲区
          if (this->good())
            this->sync();
        }
    };
    
    // 定义一个名为 ostream 的类，公开继承自 std::ostream
    class ostream : public std::ostream
    {
      // 定义公共部分
      public:
        // 构造函数，接受一个 streambuf 引用作为参数，并将其传递给 std::ostream 的构造函数进行初始化
        ostream(streambuf& buf) : std::ostream(&buf)
        {
          // 设置异常控制位，指示在出现异常时抛出 std::ios_base::failure 异常
          exceptions(std::ios_base::badbit);
        }
    
        // 析构函数，用于销毁 ostream 对象
        ~ostream() {
          // 如果 ostream 对象状态良好（good），则刷新缓冲区
          if (this->good())
            this->flush();
        }
    };
    // 结构体，封装了 Python 文件对象的输入流
    struct streambuf_capsule
    {
        // 成员变量，包含一个 Python 文件对象的输入流
        streambuf python_streambuf;

        // 构造函数，接受一个 Python 文件对象和一个可选的缓冲区大小参数
        streambuf_capsule(
          py::object& python_file_obj,
          std::size_t buffer_size=0)
        :
          // 初始化基类 streambuf，并传入 Python 文件对象和缓冲区大小
          python_streambuf(python_file_obj, buffer_size)
        {}
    };

    // 结构体，继承自 streambuf_capsule 和 streambuf::ostream，封装了 Python 文件对象的输出流
    struct ostream : private streambuf_capsule, streambuf::ostream
    {
        // 构造函数，接受一个 Python 文件对象和一个可选的缓冲区大小参数
        ostream(
          py::object& python_file_obj,
          std::size_t buffer_size=0)
        :
          // 初始化基类 streambuf_capsule，并传入 Python 文件对象和缓冲区大小
          streambuf_capsule(python_file_obj, buffer_size),
          // 初始化基类 streambuf::ostream，并传入基类的 python_streambuf
          streambuf::ostream(python_streambuf)
        {}

        // 析构函数
        ~ostream()
        {
            // 如果流处于可用状态，则刷新输出流
            if (this->good()){
              this->flush();
            }
        }
    };

    // 结构体，继承自 streambuf_capsule 和 streambuf::istream，封装了 Python 文件对象的输入流
    struct istream : private streambuf_capsule, streambuf::istream
    {
        // 构造函数，接受一个 Python 文件对象和一个可选的缓冲区大小参数
        istream(
          py::object& python_file_obj,
          std::size_t buffer_size=0)
        :
          // 初始化基类 streambuf_capsule，并传入 Python 文件对象和缓冲区大小
          streambuf_capsule(python_file_obj, buffer_size),
          // 初始化基类 streambuf::istream，并传入基类的 python_streambuf
          streambuf::istream(python_streambuf)
        {}

        // 析构函数
        ~istream()
        {
            // 如果流处于可用状态，则同步输入流
            if (this->good()) {
              this->sync();
            }
        }
    };

    // 结束命名空间 pybind11::detail

    };
    // 定义一个公共成员变量 `name`，其值为 _("io.BytesIO")
    static constexpr auto name = _("io.BytesIO");

    // 定义一个类型转换函数 `cast`，接受一个指向 `pystream::ostream` 共享指针的引用 `src`，
    // 以及返回值策略 `policy` 和一个句柄 `parent`。
    // 函数返回一个空值的释放状态。
    static handle cast(std::shared_ptr<pystream::ostream> &src, return_value_policy policy, handle parent) {
        // 返回一个已释放的空值句柄
        return none().release();
    }

    // 转换操作符，将当前对象转换为指向 `pystream::ostream` 共享指针的指针
    operator std::shared_ptr<pystream::ostream>*() { return &value; }

    // 转换操作符，将当前对象转换为指向 `pystream::ostream` 共享指针的引用
    operator std::shared_ptr<pystream::ostream>&() { return value; }

    // 模板定义，使用 `_T` 类型的别名 `cast_op_type`
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
}} // 结束 pybind11::detail 命名空间
```