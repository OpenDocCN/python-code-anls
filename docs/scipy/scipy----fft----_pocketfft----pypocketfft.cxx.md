# `D:\src\scipysrc\scipy\scipy\fft\_pocketfft\pypocketfft.cxx`

```
/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE.md
 */

/*
 *  Python interface.
 *
 *  Copyright (C) 2019 Max-Planck-Society
 *  Copyright (C) 2019 Peter Bell
 *  \author Martin Reinecke
 *  \author Peter Bell
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pocketfft_hdronly.h"

namespace {

using pocketfft::shape_t;              // 引入 shape_t 类型定义
using pocketfft::stride_t;             // 引入 stride_t 类型定义
using std::size_t;                     // 引入标准库的 size_t 类型
using std::ptrdiff_t;                  // 引入标准库的 ptrdiff_t 类型

namespace py = pybind11;               // pybind11 命名空间别名为 py

// Only instantiate long double transforms if they offer more precision
using ldbl_t = typename std::conditional<
  sizeof(long double)==sizeof(double), double, long double>::type;

using c64 = std::complex<float>;       // 复数类型 c64 定义为 std::complex<float>
using c128 = std::complex<double>;     // 复数类型 c128 定义为 std::complex<double>
using clong = std::complex<ldbl_t>;    // 复数类型 clong 定义为 std::complex<ldbl_t>
using f32 = float;                     // 浮点数类型 f32 定义为 float
using f64 = double;                    // 浮点数类型 f64 定义为 double
using flong = ldbl_t;                  // 浮点数类型 flong 定义为 ldbl_t

// 复制 numpy 数组的形状信息到 shape_t 类型
shape_t copy_shape(const py::array &arr)
  {
  shape_t res(size_t(arr.ndim()));     // 创建 shape_t 类型对象 res，大小为数组的维度数
  for (size_t i=0; i<res.size(); ++i)
    res[i] = size_t(arr.shape(int(i))); // 将每个维度的大小复制到 res 中
  return res;                          // 返回复制后的形状信息
  }

// 复制 numpy 数组的步幅信息到 stride_t 类型
stride_t copy_strides(const py::array &arr)
  {
  stride_t res(size_t(arr.ndim()));    // 创建 stride_t 类型对象 res，大小为数组的维度数
  for (size_t i=0; i<res.size(); ++i)
    res[i] = arr.strides(int(i));      // 将每个维度的步幅复制到 res 中
  return res;                          // 返回复制后的步幅信息
  }

// 根据输入数组和 axes 对象生成新的 shape_t 类型，表示轴的顺序
shape_t makeaxes(const py::array &in, const py::object &axes)
  {
  if (axes.is_none())
    {
    shape_t res(size_t(in.ndim()));    // 如果 axes 为 None，则生成默认轴顺序的 shape_t 对象
    for (size_t i=0; i<res.size(); ++i)
      res[i]=i;                        // 默认顺序为 0, 1, 2, ...
    return res;                        // 返回默认轴顺序
    }
  auto tmp=axes.cast<std::vector<ptrdiff_t>>(); // 否则，从 axes 中获取轴顺序并转换为 std::vector<ptrdiff_t>
  auto ndim = in.ndim();               // 获取输入数组的维度数
  if ((tmp.size()>size_t(ndim)) || (tmp.size()==0))
    throw std::runtime_error("bad axes argument"); // 如果轴数量超出维度数或者轴数为 0，则抛出异常
  for (auto& sz: tmp)
    {
    if (sz<0)
      sz += ndim;                      // 处理负索引，转换为正索引
    if ((sz>=ndim) || (sz<0))
      throw std::invalid_argument("axes exceeds dimensionality of output"); // 如果轴超出维度数范围，则抛出异常
    }
  return shape_t(tmp.begin(), tmp.end()); // 返回经过处理的轴顺序信息
  }

// 根据输入数组的数据类型选择合适的函数模板进行调用
#define DISPATCH(arr, T1, T2, T3, func, args) \
  { \
  if (py::isinstance<py::array_t<T1>>(arr)) return func<double> args; \
  if (py::isinstance<py::array_t<T2>>(arr)) return func<float> args;  \
  if (py::isinstance<py::array_t<T3>>(arr)) return func<ldbl_t> args; \
  throw std::runtime_error("unsupported data type"); \
  }

// 根据数据类型 T 选择适当的归一化函数模板
template<typename T> T norm_fct(int inorm, size_t N)
  {
  if (inorm==0) return T(1);           // 如果 inorm 为 0，则返回归一化系数 1
  if (inorm==2) return T(1/ldbl_t(N)); // 如果 inorm 为 2，则返回归一化系数 1/N
  if (inorm==1) return T(1/sqrt(ldbl_t(N))); // 如果 inorm 为 1，则返回归一化系数 1/sqrt(N)
  throw std::invalid_argument("invalid value for inorm (must be 0, 1, or 2)"); // 其他情况抛出异常
  }

// 根据数据类型 T 选择适当的归一化函数模板，考虑到形状和轴信息
template<typename T> T norm_fct(int inorm, const shape_t &shape,
  const shape_t &axes, size_t fct=1, int delta=0)
  {
  if (inorm==0) return T(1);           // 如果 inorm 为 0，则返回归一化系数 1
  size_t N(1);
  for (auto a: axes)
    N *= fct * size_t(int64_t(shape[a])+delta); // 计算归一化系数 N
  return norm_fct<T>(inorm, N);        // 调用前面的归一化函数模板
  }

// 准备输出数组，根据输出对象和维度信息选择合适的类型并创建数组
template<typename T> py::array_t<T> prepare_output(py::object &out_,
  shape_t &dims)
  {
  if (out_.is_none()) return py::array_t<T>(dims); // 如果输出对象为 None，则创建新的数组
  auto tmp = out_.cast<py::array_t<T>>(); // 否则，尝试将输出对象转换为指定类型的数组
  if (!tmp.is(out_)) // 如果转换后的数组对象不是原来的输出对象
    throw std::runtime_error("unexpected data type for output array");
  # 抛出运行时错误异常，指示输出数组的数据类型不符合预期
  return tmp;
  # 返回 tmp 变量的值作为函数的结果
  }
  # 结束函数定义
template<typename T> py::array c2c_internal(const py::array &in,
  const py::object &axes_, bool forward, int inorm, py::object &out_,
  size_t nthreads)
  {
  // 从Python的数组对象和轴参数创建内部表示的轴
  auto axes = makeaxes(in, axes_);
  // 复制输入数组的形状
  auto dims(copy_shape(in));
  // 准备输出数组，以std::complex<T>类型
  auto res = prepare_output<std::complex<T>>(out_, dims);
  // 复制输入数组的步长信息
  auto s_in=copy_strides(in);
  // 复制输出数组的步长信息
  auto s_out=copy_strides(res);
  // 将输入数组解释为std::complex<T>类型的数据
  auto d_in=reinterpret_cast<const std::complex<T> *>(in.data());
  // 将输出数组解释为std::complex<T>类型的可变数据
  auto d_out=reinterpret_cast<std::complex<T> *>(res.mutable_data());
  {
  // 释放全局解释器锁，允许并行计算
  py::gil_scoped_release release;
  // 计算归一化因子
  T fct = norm_fct<T>(inorm, dims, axes);
  // 执行复数到复数的快速傅里叶变换
  pocketfft::c2c(dims, s_in, s_out, axes, forward, d_in, d_out, fct, nthreads);
  }
  // 返回移动语义的输出数组
  return std::move(res);
  }

template<typename T> py::array c2c_sym_internal(const py::array &in,
  const py::object &axes_, bool forward, int inorm, py::object &out_,
  size_t nthreads)
  {
  // 从Python的数组对象和轴参数创建内部表示的轴
  auto axes = makeaxes(in, axes_);
  // 复制输入数组的形状
  auto dims(copy_shape(in));
  // 准备输出数组，以std::complex<T>类型
  auto res = prepare_output<std::complex<T>>(out_, dims);
  // 复制输入数组的步长信息
  auto s_in=copy_strides(in);
  // 复制输出数组的步长信息
  auto s_out=copy_strides(res);
  // 将输入数组解释为T类型的数据
  auto d_in=reinterpret_cast<const T *>(in.data());
  // 将输出数组解释为std::complex<T>类型的可变数据
  auto d_out=reinterpret_cast<std::complex<T> *>(res.mutable_data());
  {
  // 释放全局解释器锁，允许并行计算
  py::gil_scoped_release release;
  // 计算归一化因子
  T fct = norm_fct<T>(inorm, dims, axes);
  // 执行实数到复数的快速傅里叶变换
  pocketfft::r2c(dims, s_in, s_out, axes, forward, d_in, d_out, fct, nthreads);
  // 填充复数输出数组的第二半部分
  using namespace pocketfft::detail;
  ndarr<std::complex<T>> ares(res.mutable_data(), dims, s_out);
  rev_iter iter(ares, axes);
  while(iter.remaining()>0)
    {
    auto v = ares[iter.ofs()];
    ares[iter.rev_ofs()] = conj(v);
    iter.advance();
    }
  }
  // 返回移动语义的输出数组
  return std::move(res);
  }

py::array c2c(const py::array &a, const py::object &axes_, bool forward,
  int inorm, py::object &out_, size_t nthreads)
  {
  // 如果输入数组的数据类型为复数类型
  if (a.dtype().kind() == 'c')
    // 调度到复数到复数的快速傅里叶变换实现
    DISPATCH(a, c128, c64, clong, c2c_internal, (a, axes_, forward,
             inorm, out_, nthreads))

  // 调度到实数到复数的快速傅里叶变换实现
  DISPATCH(a, f64, f32, flong, c2c_sym_internal, (a, axes_, forward,
           inorm, out_, nthreads))
  }

template<typename T> py::array r2c_internal(const py::array &in,
  const py::object &axes_, bool forward, int inorm, py::object &out_,
  size_t nthreads)
  {
  // 从Python的数组对象和轴参数创建内部表示的轴
  auto axes = makeaxes(in, axes_);
  // 复制输入数组的形状，同时创建输出数组的形状
  auto dims_in(copy_shape(in)), dims_out(dims_in);
  // 计算输出数组在指定轴上的长度
  dims_out[axes.back()] = (dims_out[axes.back()]>>1)+1;
  // 准备输出数组，以std::complex<T>类型
  py::array res = prepare_output<std::complex<T>>(out_, dims_out);
  // 复制输入数组的步长信息
  auto s_in=copy_strides(in);
  // 复制输出数组的步长信息
  auto s_out=copy_strides(res);
  // 将输入数组解释为T类型的数据
  auto d_in=reinterpret_cast<const T *>(in.data());
  // 将输出数组解释为std::complex<T>类型的可变数据
  auto d_out=reinterpret_cast<std::complex<T> *>(res.mutable_data());
  {
  // 释放全局解释器锁，允许并行计算
  py::gil_scoped_release release;
  // 计算归一化因子
  T fct = norm_fct<T>(inorm, dims_in, axes);
  // 执行实数到复数的快速傅里叶变换
  pocketfft::r2c(dims_in, s_in, s_out, axes, forward, d_in, d_out, fct,
    nthreads);
  }
  // 返回输出数组
  return res;
  }

py::array r2c(const py::array &in, const py::object &axes_, bool forward,
  int inorm, py::object &out_, size_t nthreads)
  {
  // 调度到实数到复数的快速傅里叶变换实现
  DISPATCH(in, f64, f32, flong, r2c_internal, (in, axes_, forward, inorm, out_,
    nthreads))
  }
// 定义模板函数 r2r_fftpack_internal，接受输入数组 in，轴 axes_，标志 real2hermitian 和 forward，
// 规范化参数 inorm，输出数组对象 out_，以及线程数 nthreads
template<typename T> py::array r2r_fftpack_internal(const py::array &in,
  const py::object &axes_, bool real2hermitian, bool forward, int inorm,
  py::object &out_, size_t nthreads)
  {
  // 调用 makeaxes 函数处理输入数组和轴对象 axes_
  auto axes = makeaxes(in, axes_);
  // 复制输入数组的形状并准备输出数组 res
  auto dims(copy_shape(in));
  py::array res = prepare_output<T>(out_, dims);
  // 复制输入数组的步长 s_in 和输出数组的步长 s_out
  auto s_in=copy_strides(in);
  auto s_out=copy_strides(res);
  // 将输入数组转换为类型 T 的常量指针 d_in 和输出数组的可变数据指针 d_out
  auto d_in=reinterpret_cast<const T *>(in.data());
  auto d_out=reinterpret_cast<T *>(res.mutable_data());
  {
  // 释放全局解释器锁，允许多线程执行
  py::gil_scoped_release release;
  // 根据指定的规范化参数 inorm、数组维度 dims 和轴 axes 计算规范化系数 fct
  T fct = norm_fct<T>(inorm, dims, axes);
  // 调用 pocketfft::r2r_fftpack 函数执行实数到实数的 FFT 变换
  pocketfft::r2r_fftpack(dims, s_in, s_out, axes, real2hermitian, forward,
    d_in, d_out, fct, nthreads);
  }
  // 返回结果数组 res
  return res;
  }

// 定义函数 r2r_fftpack，接受输入数组 in，轴 axes_，标志 real2hermitian 和 forward，
// 规范化参数 inorm，输出数组对象 out_，以及线程数 nthreads
py::array r2r_fftpack(const py::array &in, const py::object &axes_,
  bool real2hermitian, bool forward, int inorm, py::object &out_,
  size_t nthreads)
  {
  // 使用宏 DISPATCH 调度执行 r2r_fftpack_internal 函数的实例化版本，根据输入数组 in 的类型 T
  DISPATCH(in, f64, f32, flong, r2r_fftpack_internal, (in, axes_,
    real2hermitian, forward, inorm, out_, nthreads))
  }

// 定义模板函数 dct_internal，接受输入数组 in，轴 axes_，DCT 类型 type，规范化参数 inorm，
// 输出数组对象 out_，线程数 nthreads，以及是否正交的标志 ortho
template<typename T> py::array dct_internal(const py::array &in,
  const py::object &axes_, int type, int inorm, py::object &out_,
  size_t nthreads, bool ortho)
  {
  // 调用 makeaxes 函数处理输入数组和轴对象 axes_
  auto axes = makeaxes(in, axes_);
  // 复制输入数组的形状并准备输出数组 res
  auto dims(copy_shape(in));
  py::array res = prepare_output<T>(out_, dims);
  // 复制输入数组的步长 s_in 和输出数组的步长 s_out
  auto s_in=copy_strides(in);
  auto s_out=copy_strides(res);
  // 将输入数组转换为类型 T 的常量指针 d_in 和输出数组的可变数据指针 d_out
  auto d_in=reinterpret_cast<const T *>(in.data());
  auto d_out=reinterpret_cast<T *>(res.mutable_data());
  {
  // 释放全局解释器锁，允许多线程执行
  py::gil_scoped_release release;
  // 根据 DCT 类型 type 和正交标志 ortho，计算规范化系数 fct
  T fct = (type==1) ? norm_fct<T>(inorm, dims, axes, 2, -1)
                    : norm_fct<T>(inorm, dims, axes, 2);
  // 调用 pocketfft::dct 函数执行离散余弦变换
  pocketfft::dct(dims, s_in, s_out, axes, type, d_in, d_out, fct, ortho,
    nthreads);
  }
  // 返回结果数组 res
  return res;
  }

// 定义函数 dct，接受输入数组 in，DCT 类型 type，轴 axes_，规范化参数 inorm，
// 输出数组对象 out_，线程数 nthreads，以及正交对象 ortho_obj
py::array dct(const py::array &in, int type, const py::object &axes_,
  int inorm, py::object &out_, size_t nthreads, const py::object & ortho_obj)
  {
  // 根据 inorm 的值确定是否使用正交参数
  bool ortho=inorm==1;
  if (!ortho_obj.is_none())
    ortho=ortho_obj.cast<bool>();

  // 若 DCT 类型 type 不在合法范围内，抛出异常
  if ((type<1) || (type>4)) throw std::invalid_argument("invalid DCT type");
  // 使用宏 DISPATCH 调度执行 dct_internal 函数的实例化版本，根据输入数组 in 的类型 T
  DISPATCH(in, f64, f32, flong, dct_internal, (in, axes_, type, inorm, out_,
    nthreads, ortho))
  }

// 定义模板函数 dst_internal，接受输入数组 in，轴 axes_，DST 类型 type，规范化参数 inorm，
// 输出数组对象 out_，线程数 nthreads，以及是否正交的标志 ortho
template<typename T> py::array dst_internal(const py::array &in,
  const py::object &axes_, int type, int inorm, py::object &out_,
  size_t nthreads, bool ortho)
  {
  // 调用 makeaxes 函数处理输入数组和轴对象 axes_
  auto axes = makeaxes(in, axes_);
  // 复制输入数组的形状并准备输出数组 res
  auto dims(copy_shape(in));
  py::array res = prepare_output<T>(out_, dims);
  // 复制输入数组的步长 s_in 和输出数组的步长 s_out
  auto s_in=copy_strides(in);
  auto s_out=copy_strides(res);
  // 将输入数组转换为类型 T 的常量指针 d_in 和输出数组的可变数据指针 d_out
  auto d_in=reinterpret_cast<const T *>(in.data());
  auto d_out=reinterpret_cast<T *>(res.mutable_data());
  {
  // 释放全局解释器锁，允许多线程执行
  py::gil_scoped_release release;
  // 根据 DST 类型 type 和正交标志 ortho，计算规范化系数 fct
  T fct = (type==1) ? norm_fct<T>(inorm, dims, axes, 2, 1)
                    : norm_fct<T>(inorm, dims, axes, 2);
  // 调用 pocketfft::dst 函数执行离散正弦变换
  pocketfft::dst(dims, s_in, s_out, axes, type, d_in, d_out, fct, ortho,
    nthreads);
  }
  // 返回结果数组 res
  return res;
  }


这段代码是一系列使用 C++ 编写的模板函数和宏定义，用于执行不同类型的离散变换（包括离散傅里叶变换、离散余弦变换和离散正弦变换）。这些函数使用了 Python 的 `pybind11` 库来与 Python 环境交互，可以在多线程环境中运行。
py::array dst(const py::array &in, int type, const py::object &axes_,
  int inorm, py::object &out_, size_t nthreads, const py::object &ortho_obj)
  {
  // 根据输入参数确定是否使用正交变换
  bool ortho=inorm==1;
  // 如果提供了正交对象参数，根据其值确定是否使用正交变换
  if (!ortho_obj.is_none())
    ortho=ortho_obj.cast<bool>();

  // 检查 DST 类型是否有效，如果无效则抛出异常
  if ((type<1) || (type>4)) throw std::invalid_argument("invalid DST type");
  // 调用模板函数 DISPATCH 处理 DST 变换的内部实现
  DISPATCH(in, f64, f32, flong, dst_internal, (in, axes_, type, inorm,
    out_, nthreads, ortho))
  }

template<typename T> py::array c2r_internal(const py::array &in,
  const py::object &axes_, size_t lastsize, bool forward, int inorm,
  py::object &out_, size_t nthreads)
  {
  // 根据输入参数生成轴列表 axes
  auto axes = makeaxes(in, axes_);
  // 获取最后一个轴的索引
  size_t axis = axes.back();
  // 复制输入数组的形状
  shape_t dims_in(copy_shape(in)), dims_out=dims_in;
  // 如果未提供最后大小参数，则根据输入数组计算
  if (lastsize==0) lastsize=2*dims_in[axis]-1;
  // 检查最后大小参数是否正确
  if ((lastsize/2) + 1 != dims_in[axis])
    throw std::invalid_argument("bad lastsize");
  // 调整输出数组的维度，使其适应变换结果
  dims_out[axis] = lastsize;
  // 准备输出数组
  py::array res = prepare_output<T>(out_, dims_out);
  // 复制输入数组的步长
  auto s_in=copy_strides(in);
  // 复制输出数组的步长
  auto s_out=copy_strides(res);
  // 将输入数组数据解释为复数类型 T
  auto d_in=reinterpret_cast<const std::complex<T> *>(in.data());
  // 将输出数组数据解释为类型 T
  auto d_out=reinterpret_cast<T *>(res.mutable_data());
  {
  // 释放 GIL，允许多线程执行
  py::gil_scoped_release release;
  // 计算规范化因子 fct
  T fct = norm_fct<T>(inorm, dims_out, axes);
  // 调用 pocketfft 库的 c2r 函数执行变换
  pocketfft::c2r(dims_out, s_in, s_out, axes, forward, d_in, d_out, fct,
    nthreads);
  }
  // 返回变换后的结果数组
  return res;
  }

py::array c2r(const py::array &in, const py::object &axes_, size_t lastsize,
  bool forward, int inorm, py::object &out_, size_t nthreads)
  {
  // 调用模板函数 DISPATCH 处理 c2r 变换的内部实现
  DISPATCH(in, c128, c64, clong, c2r_internal, (in, axes_, lastsize, forward,
    inorm, out_, nthreads))
  }

template<typename T> py::array separable_hartley_internal(const py::array &in,
  const py::object &axes_, int inorm, py::object &out_, size_t nthreads)
  {
  // 复制输入数组的形状
  auto dims(copy_shape(in));
  // 准备输出数组
  py::array res = prepare_output<T>(out_, dims);
  // 根据输入参数生成轴列表 axes
  auto axes = makeaxes(in, axes_);
  // 复制输入数组的步长
  auto s_in=copy_strides(in);
  // 复制输出数组的步长
  auto s_out=copy_strides(res);
  // 将输入数组数据解释为类型 T
  auto d_in=reinterpret_cast<const T *>(in.data());
  // 将输出数组数据解释为类型 T
  auto d_out=reinterpret_cast<T *>(res.mutable_data());
  {
  // 释放 GIL，允许多线程执行
  py::gil_scoped_release release;
  // 计算规范化因子 fct
  T fct = norm_fct<T>(inorm, dims, axes);
  // 调用 pocketfft 库的 r2r_separable_hartley 函数执行变换
  pocketfft::r2r_separable_hartley(dims, s_in, s_out, axes, d_in, d_out, fct,
    nthreads);
  }
  // 返回变换后的结果数组
  return res;
  }

py::array separable_hartley(const py::array &in, const py::object &axes_,
  int inorm, py::object &out_, size_t nthreads)
  {
  // 调用模板函数 DISPATCH 处理 separable_hartley 变换的内部实现
  DISPATCH(in, f64, f32, flong, separable_hartley_internal, (in, axes_, inorm,
    out_, nthreads))
  }
// 定义模板函数 genuine_hartley_internal，接受输入数组和参数，执行 Hartley 变换
template<typename T> py::array genuine_hartley_internal(const py::array &in,
  const py::object &axes_, int inorm, py::object &out_, size_t nthreads)
  {
  // 复制输入数组的维度信息
  auto dims(copy_shape(in));
  // 准备输出数组，调用 prepare_output 函数生成结果数组
  py::array res = prepare_output<T>(out_, dims);
  // 创建轴对象，以便在输入数组上执行 Hartley 变换
  auto axes = makeaxes(in, axes_);
  // 复制输入数组的步长信息
  auto s_in=copy_strides(in);
  // 复制输出数组的步长信息
  auto s_out=copy_strides(res);
  // 解释输入数组数据为类型 T 的常量指针
  auto d_in=reinterpret_cast<const T *>(in.data());
  // 解释输出数组数据为类型 T 的可变指针
  auto d_out=reinterpret_cast<T *>(res.mutable_data());
  {
  // 释放全局解释器锁，以便在多线程环境中执行 FFT 计算
  py::gil_scoped_release release;
  // 计算规范化因子，用于 Hartley 变换
  T fct = norm_fct<T>(inorm, dims, axes);
  // 调用 Hartley 变换函数，将输入数据转换为输出数据
  pocketfft::r2r_genuine_hartley(dims, s_in, s_out, axes, d_in, d_out, fct,
    nthreads);
  }
  // 返回执行 Hartley 变换后的结果数组
  return res;
  }

// 定义函数 genuine_hartley，接受输入数组和参数，分派到对应的内部模板函数处理
py::array genuine_hartley(const py::array &in, const py::object &axes_,
  int inorm, py::object &out_, size_t nthreads)
  {
  // 使用 DISPATCH 宏根据输入数组的类型分派到不同的处理函数
  DISPATCH(in, f64, f32, flong, genuine_hartley_internal, (in, axes_, inorm,
    out_, nthreads))
  }

// 导出 good_size 函数至 C-API，用于计算最适合进行 FFT 的长度
PyObject * good_size(PyObject * /*self*/, PyObject * args, PyObject * kwargs)
  {
  // 解析 Python 函数参数，获取目标长度和是否为实数类型
  Py_ssize_t n_ = -1;
  int real = false;
  static const char * keywords[] = {"target", "real", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|p:good_size",
                                   (char **) keywords, &n_, &real))
    return nullptr;

  // 目标长度必须为正数，否则报错
  if (n_<0)
    {
    PyErr_SetString(PyExc_ValueError, "Target length must be positive");
    return nullptr;
    }
  // 检查目标长度是否过大以至于无法执行 FFT 计算
  if ((n_-1) > static_cast<Py_ssize_t>(std::numeric_limits<size_t>::max() / 11))
    {
    PyErr_Format(PyExc_ValueError,
                 "Target length is too large to perform an FFT: %zi", n_);
    return nullptr;
    }
  // 将目标长度转换为 size_t 类型
  const auto n = static_cast<size_t>(n_);
  // 使用 pocketfft::detail 命名空间下的实用工具函数计算最适合的 FFT 长度
  using namespace pocketfft::detail;
  // 返回计算出的最适合的 FFT 长度作为 Python 长整型对象
  return PyLong_FromSize_t(
    real ? util::good_size_real(n) : util::good_size_cmplx(n));
  }

// 导出 prev_good_size 函数至 C-API，用于计算较小的最适合 FFT 长度
PyObject * prev_good_size(PyObject * /*self*/, PyObject * args, PyObject * kwargs)
  {
  // 解析 Python 函数参数，获取目标长度和是否为实数类型
  Py_ssize_t n_ = -1;
  int real = false;
  static const char * keywords[] = {"target", "real", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|p:prev_good_size",
                                   (char **) keywords, &n_, &real))
    return nullptr;

  // 目标长度必须为正数，否则报错
  if (n_<0)
    {
    PyErr_SetString(PyExc_ValueError, "Target length must be positive");
    return nullptr;
    }
  // 检查目标长度是否过大以至于无法执行 FFT 计算
  if ((n_-1) > static_cast<Py_ssize_t>(std::numeric_limits<size_t>::max() / 11))
    {
    PyErr_Format(PyExc_ValueError,
                 "Target length is too large to perform an FFT: %zi", n_);
    return nullptr;
    }
  // 将目标长度转换为 size_t 类型
  const auto n = static_cast<size_t>(n_);
  // 使用 pocketfft::detail 命名空间下的实用工具函数计算较小的最适合 FFT 长度
  using namespace pocketfft::detail;
  // 返回计算出的较小的最适合 FFT 长度作为 Python 长整型对象
  return PyLong_FromSize_t(
    real ? util::prev_good_size_real(n) : util::prev_good_size_cmplx(n));
  }

// 导出 pypocketfft_DS 字符串，提供 pypocketfft 模块的简要描述文档
const char *pypocketfft_DS = R"""(Fast Fourier and Hartley transforms.

This module supports
- single, double, and long double precision
- complex and real-valued transforms
- multi-dimensional transforms

For two- and higher-dimensional transforms the code will use SSE2 and AVX
vector instructions for faster execution if these are supported by the CPU and
were enabled during compilation.
)""";

const char *c2c_DS = R"""(Performs a complex FFT.

Parameters
----------
a : numpy.ndarray (any complex or real type)
    The input data. If its type is real, a more efficient real-to-complex
    transform will be used.
axes : list of integers
    The axes along which the FFT is carried out.
    If not set, all axes will be transformed.
forward : bool
    If `True`, a negative sign is used in the exponent, else a positive one.
inorm : int
    Normalization type
      0 : no normalization
      1 : divide by sqrt(N)
      2 : divide by N
    where N is the product of the lengths of the transformed axes.
out : numpy.ndarray (same shape as `a`, complex type with same accuracy as `a`)
    May be identical to `a`, but if it isn't, it must not overlap with `a`.
    If None, a new array is allocated to store the output.
nthreads : int
    Number of threads to use. If 0, use the system default (typically governed
    by the `OMP_NUM_THREADS` environment variable).

Returns
-------
numpy.ndarray (same shape as `a`, complex type with same accuracy as `a`)
    The transformed data.
)""";

const char *r2c_DS = R"""(Performs an FFT whose input is strictly real.

Parameters
----------
a : numpy.ndarray (any real type)
    The input data
axes : list of integers
    The axes along which the FFT is carried out.
    If not set, all axes will be transformed in ascending order.
forward : bool
    If `True`, a negative sign is used in the exponent, else a positive one.
inorm : int
    Normalization type
      0 : no normalization
      1 : divide by sqrt(N)
      2 : divide by N
    where N is the product of the lengths of the transformed input axes.
out : numpy.ndarray (complex type with same accuracy as `a`)
    For the required shape, see the `Returns` section.
    Must not overlap with `a`.
    If None, a new array is allocated to store the output.
nthreads : int
    Number of threads to use. If 0, use the system default (typically governed
    by the `OMP_NUM_THREADS` environment variable).

Returns
-------
numpy.ndarray (complex type with same accuracy as `a`)
    The transformed data. The shape is identical to that of the input array,
    except for the axis that was transformed last. If the length of that axis
    was n on input, it is n//2+1 on output.
)""";

const char *c2r_DS = R"""(Performs an FFT whose output is strictly real.

Parameters
----------
a : numpy.ndarray (any complex type)
    The input data
axes : list of integers
    The axes along which the FFT is carried out.
    If not set, all axes will be transformed in ascending order.
lastsize : the output size of the last axis to be transformed.
    If the corresponding input axis has size n, this can be 2*n-2 or 2*n-1.
forward : bool
    If `True`, a negative sign is used in the exponent, else a positive one.
inorm : int
    Normalization type
      0 : no normalization
      1 : divide by sqrt(N)
      2 : divide by N
    where N is the product of the lengths of the transformed axes.
out : numpy.ndarray (real type with same accuracy as `a`)
    For the required shape, see the `Returns` section.
    Must not overlap with `a`.
    If None, a new array is allocated to store the output.
nthreads : int
    Number of threads to use. If 0, use the system default (typically governed
    by the `OMP_NUM_THREADS` environment variable).

Returns
-------
numpy.ndarray (real type with same accuracy as `a`)
    The transformed data.
)""";
    # 正规化类型说明
    # 0：无正规化
    # 1：除以 sqrt(N)，其中 N 是转换输出轴长度的乘积
    # 2：除以 N，其中 N 是转换输出轴长度的乘积
    # 其中，N 是指转换后输出轴长度的乘积。
out : numpy.ndarray (real type with same accuracy as `a`)
    // 输出参数，表示函数返回的 numpy 数组，数据类型与 `a` 相同，精度也相同。
    // 查看 `Returns` 部分获取所需的形状。
    // 不得与 `a` 重叠。如果为 None，则分配新数组来存储输出。
nthreads : int
    // 线程数，用于指定使用的线程数。如果为 0，则使用系统默认值（通常由 `OMP_NUM_THREADS` 环境变量控制）。

Returns
-------
numpy.ndarray (real type with same accuracy as `a`)
    // 变换后的数据。形状与输入数组相同，除了最后一个被变换的轴，其大小为 `lastsize`。
)""";
sum of real and imaginary parts of the result is stored in the output
array. For a single transformed axis, this is identical to `separable_hartley`,
but when transforming multiple axes, the results are different.

Parameters
----------
a : numpy.ndarray (any real type)
    The input data
axes : list of integers
    The axes along which the transform is carried out.
    If not set, all axes will be transformed.
inorm : int
    Normalization type
      0 : no normalization
      1 : divide by sqrt(N)
      2 : divide by N
    where N is the product of the lengths of the transformed axes.
out : numpy.ndarray (same shape and data type as `a`)
    May be identical to `a`, but if it isn't, it must not overlap with `a`.
    If None, a new array is allocated to store the output.
nthreads : int
    Number of threads to use. If 0, use the system default (typically governed
    by the `OMP_NUM_THREADS` environment variable).

Returns
-------
numpy.ndarray (same shape and data type as `a`)
    The transformed data
)""";

const char *dct_DS = R"""(Performs a discrete cosine transform.

Parameters
----------
a : numpy.ndarray (any real type)
    The input data
type : integer
    the type of DCT. Must be in [1; 4].
axes : list of integers
    The axes along which the transform is carried out.
    If not set, all axes will be transformed.
inorm : int
    Normalization type
      0 : no normalization
      1 : divide by sqrt(N)
      2 : divide by N
    where N is the product of n_i for every transformed axis i.
    n_i is 2*(<axis_length>-1 for type 1 and 2*<axis length>
    for types 2, 3, 4.
    Making the transform orthogonal involves the following additional steps
    for every 1D sub-transform:
      Type 1 : multiply first and last input value by sqrt(2)
               divide first and last output value by sqrt(2)
      Type 2 : divide first output value by sqrt(2)
      Type 3 : multiply first input value by sqrt(2)
      Type 4 : nothing
out : numpy.ndarray (same shape and data type as `a`)
    May be identical to `a`, but if it isn't, it must not overlap with `a`.
    If None, a new array is allocated to store the output.
nthreads : int
    Number of threads to use. If 0, use the system default (typically governed
    by the `OMP_NUM_THREADS` environment variable).
ortho: bool
    Orthogonalize transform (defaults to ``inorm=1``)

Returns
-------
numpy.ndarray (same shape and data type as `a`)
    The transformed data
)""";

const char *dst_DS = R"""(Performs a discrete sine transform.

Parameters
----------
a : numpy.ndarray (any real type)
    The input data
type : integer
    the type of DST. Must be in [1; 4].
axes : list of integers
    The axes along which the transform is carried out.
    If not set, all axes will be transformed.
inorm : int
    Normalization type
      0 : no normalization
      1 : divide by sqrt(N)
      2 : divide by N
    where N is the product of n_i for every transformed axis i.

Returns
-------
numpy.ndarray (same shape and data type as `a`)
    The transformed data
)""";
    n_i is 2*(<axis_length>+1 for type 1 and 2*<axis length>
    for types 2, 3, 4.
    Making the transform orthogonal involves the following additional steps
    for every 1D sub-transform:
      Type 1 : nothing
      Type 2 : divide first output value by sqrt(2)
      Type 3 : multiply first input value by sqrt(2)
      Type 4 : nothing



# 定义了变量 n_i 的计算方式，取决于不同的类型（type）和轴长度（axis_length）：
# - 对于类型 1，n_i 是 2*(<axis_length>+1)
# - 对于类型 2、3、4，n_i 是 2*<axis length>
# 在使变换正交化的过程中，对每个一维子变换需要额外进行以下步骤：
# - 类型 1：无操作
# - 类型 2：将第一个输出值除以 sqrt(2)
# - 类型 3：将第一个输入值乘以 sqrt(2)
# - 类型 4：无操作
out : numpy.ndarray (same shape and data type as `a`)
    May be identical to `a`, but if it isn't, it must not overlap with `a`.
    If None, a new array is allocated to store the output.
nthreads : int
    Number of threads to use. If 0, use the system default (typically governed
    by the `OMP_NUM_THREADS` environment variable).
ortho: bool
    Orthogonalize transform (defaults to ``inorm=1``)



Returns
-------
numpy.ndarray (same shape and data type as `a`)
    The transformed data
)""";

const char * good_size_DS = R"""(Returns a good length to pad an FFT to.

Parameters
----------
target : int
    Minimum transform length
real : bool, optional
    True if either input or output of FFT should be fully real.

Returns
-------
out : int
    The smallest fast size >= n

)""";


const char * prev_good_size_DS = R"""(Returns the largest FFT length less than target length.

Parameters
----------
target : int
    Maximum transform length
real : bool, optional
    True if either input or output of FFT should be fully real.

Returns
-------
out : int
    The largest fast length <= n

)""";



} // unnamed namespace

PYBIND11_MODULE(pypocketfft, m)
{
    using namespace pybind11::literals;

    auto None = py::none();

    // 设置模块文档字符串为 pypocketfft_DS
    m.doc() = pypocketfft_DS;
    
    // 定义 c2c 函数，接受参数和文档字符串 c2c_DS
    m.def("c2c", c2c, c2c_DS, "a"_a, "axes"_a=None, "forward"_a=true,
        "inorm"_a=0, "out"_a=None, "nthreads"_a=1);
    
    // 定义 r2c 函数，接受参数和文档字符串 r2c_DS
    m.def("r2c", r2c, r2c_DS, "a"_a, "axes"_a=None, "forward"_a=true,
        "inorm"_a=0, "out"_a=None, "nthreads"_a=1);
    
    // 定义 c2r 函数，接受参数和文档字符串 c2r_DS
    m.def("c2r", c2r, c2r_DS, "a"_a, "axes"_a=None, "lastsize"_a=0,
        "forward"_a=true, "inorm"_a=0, "out"_a=None, "nthreads"_a=1);
    
    // 定义 r2r_fftpack 函数，接受参数和文档字符串 r2r_fftpack_DS
    m.def("r2r_fftpack", r2r_fftpack, r2r_fftpack_DS, "a"_a, "axes"_a,
        "real2hermitian"_a, "forward"_a, "inorm"_a=0, "out"_a=None, "nthreads"_a=1);
    
    // 定义 separable_hartley 函数，接受参数和文档字符串 separable_hartley_DS
    m.def("separable_hartley", separable_hartley, separable_hartley_DS, "a"_a,
        "axes"_a=None, "inorm"_a=0, "out"_a=None, "nthreads"_a=1);
    
    // 定义 genuine_hartley 函数，接受参数和文档字符串 genuine_hartley_DS
    m.def("genuine_hartley", genuine_hartley, genuine_hartley_DS, "a"_a,
        "axes"_a=None, "inorm"_a=0, "out"_a=None, "nthreads"_a=1);
    
    // 定义 dct 函数，接受参数和文档字符串 dct_DS
    m.def("dct", dct, dct_DS, "a"_a, "type"_a, "axes"_a=None, "inorm"_a=0,
        "out"_a=None, "nthreads"_a=1, "ortho"_a=None);
    
    // 定义 dst 函数，接受参数和文档字符串 dst_DS
    m.def("dst", dst, dst_DS, "a"_a, "type"_a, "axes"_a=None, "inorm"_a=0,
        "out"_a=None, "nthreads"_a=1, "ortho"_a=None);

    // 定义 good_size_meth 函数，接受参数和文档字符串 good_size_DS
    static PyMethodDef good_size_meth[] =
        {{"good_size", (PyCFunction)good_size,
          METH_VARARGS | METH_KEYWORDS, good_size_DS}, {0}};
    
    // 将 good_size_meth 函数添加到模块中
    PyModule_AddFunctions(m.ptr(), good_size_meth);

    // 定义 prev_good_size_meth 函数，接受参数和文档字符串 prev_good_size_DS
    static PyMethodDef prev_good_size_meth[] =
        {{"prev_good_size", (PyCFunction)prev_good_size,
          METH_VARARGS | METH_KEYWORDS, prev_good_size_DS}, {0}};
    
    // 将 prev_good_size_meth 函数添加到模块中
    PyModule_AddFunctions(m.ptr(), prev_good_size_meth);
}
```