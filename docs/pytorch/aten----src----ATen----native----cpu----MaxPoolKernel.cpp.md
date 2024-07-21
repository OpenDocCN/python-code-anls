# `.\pytorch\aten\src\ATen\native\cpu\MaxPoolKernel.cpp`

```
// 定义宏以仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含适应性池化的头文件
#include <ATen/native/AdaptivePooling.h>
// 包含张量相关的核心头文件
#include <ATen/core/Tensor.h>

// 包含分发机制相关的头文件
#include <ATen/Dispatch.h>
// 包含并行处理相关的头文件
#include <ATen/Parallel.h>
// 包含矢量化相关的头文件
#include <ATen/cpu/vec/vec.h>
// 包含矢量化相关的功能头文件
#include <ATen/cpu/vec/functional.h>
// 包含池化相关的头文件
#include <ATen/native/Pool.h>
// 包含CPU工具函数的头文件
#include <ATen/native/cpu/utils.h>
// 包含C++ 10中的范围工具函数的头文件
#include <c10/util/irange.h>
// 包含操作数数学类型的头文件
#include <ATen/OpMathType.h>
// 包含减少操作工具函数的头文件
#include <ATen/native/ReduceOpsUtils.h>

namespace at::native {

namespace {

// 检查标量是否为NaN的模板函数
template <typename scalar_t>
bool is_nan(scalar_t v) {
  // 如果标量类型是整数或者是无符号字符类型，则返回false
  if (std::is_integral<scalar_t>::value || std::is_same<scalar_t, unsigned char>::value) {
    return false;
  }
  // 否则使用std::isnan()函数判断标量是否为NaN
  return std::isnan(v);
}

// 检查矢量中每个元素是否为NaN的模板函数
template <typename scalar_t>
vec::Vectorized<scalar_t> is_nan_vec(vec::Vectorized<scalar_t> vec) {
  // 直接调用矢量化类型的isnan()方法检查每个元素是否为NaN
  return vec.isnan();
}

// 对于unsigned char类型，重载的矢量化检查NaN的函数
template <>
vec::Vectorized<unsigned char> is_nan_vec<unsigned char>(vec::Vectorized<unsigned char> vec) {
  // 创建一个全为false的矢量对象，因为unsigned char类型不涉及NaN检查
  Vectorized<unsigned char> ret(false);
  return ret;
}

// 对于signed char类型，重载的矢量化检查NaN的函数
template <>
vec::Vectorized<signed char> is_nan_vec<signed char>(vec::Vectorized<signed char> vec) {
  // 创建一个全为false的矢量对象，因为signed char类型不涉及NaN检查
  Vectorized<signed char> ret(false);
  return ret;
}

// 对于short类型，重载的矢量化检查NaN的函数
template <>
vec::Vectorized<short> is_nan_vec<short>(vec::Vectorized<short> vec) {
  // 创建一个全为false的矢量对象，因为short类型不涉及NaN检查
  Vectorized<short> ret(false);
  return ret;
}

// 对于int类型，重载的矢量化检查NaN的函数
template <>
vec::Vectorized<int> is_nan_vec<int>(vec::Vectorized<int> vec) {
  // 创建一个全为false的矢量对象，因为int类型不涉及NaN检查
  Vectorized<int> ret(false);
  return ret;
}

// 对于int64_t类型，重载的矢量化检查NaN的函数
template <>
vec::Vectorized<int64_t> is_nan_vec<int64_t>(vec::Vectorized<int64_t> vec) {
  // 创建一个全为false的矢量对象，因为int64_t类型不涉及NaN检查
  Vectorized<int64_t> ret(false);
  return ret;
}

// 计算内部操作的模板函数，根据不同的数据类型执行不同的计算逻辑
template <typename scalar_t, typename opmath_t>
inline
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
compute_internal(
  const scalar_t* input_data,         // 输入数据的指针
  scalar_t* out_data,                 // 输出数据的指针
  opmath_t* max_ptr,                  // 最大值指针
  vec::int_same_size_t<opmath_t>* index_ptr,  // 索引指针
  int64_t* ind,                       // 索引数组
  int64_t input_depth, int64_t input_height, int64_t input_width, int64_t channels,  // 输入深度、高度、宽度和通道数
  int64_t n,                          // n参数
  int64_t len,                        // len参数
  int64_t size,                       // size参数
  int64_t id0, int64_t id1,           // id0和id1参数
  int64_t ih0, int64_t ih1,           // ih0和ih1参数
  int64_t iw0, int64_t iw1,           // iw0和iw1参数
  int64_t dilationD,                  // dilationD参数
  int64_t dilationH,                  // dilationH参数
  int64_t dilationW) {                // dilationW参数
  using Vec = vec::Vectorized<scalar_t>;       // 使用矢量化的标量类型Vec
  using integer_t = vec::int_same_size_t<opmath_t>;  // 使用矢量化的整数类型integer_t
  using iVec = vec::Vectorized<integer_t>;   // 使用矢量化的整数矢量类型iVec
  // Pass I: init out lane
  iVec index0_vec = iVec(id0 * input_height * input_width + ih0 * input_width + iw0);  // 初始化索引矢量

  scalar_t min_value = lower_bound<scalar_t>();  // 设置最小值
  Vec out_vec = Vec(min_value);                 // 使用最小值初始化输出矢量
  int64_t d1 = 0;
  for (; d1 < len; d1 += Vec::size()) {         // 循环处理长度为Vec::size()的部分
    index0_vec.store(index_ptr + d1);           // 存储索引到指定位置
    out_vec.store(out_data + d1);               // 存储输出到指定位置
  }
  for (; d1 < size; d1++) {                      // 处理剩余的部分
    ind[d1] = ih0 * input_width + iw0;          // 设置索引值
    out_data[d1] = min_value;                   // 设置输出值为最小值
  }
  // Pass II: compute local max
  for (int64_t id = id0; id < id1; id += dilationD) {  // 处理第二阶段的本地最大值计算
    // 遍历输入的高度和宽度范围，以给定的 dilationH 和 dilationW 递增步长进行遍历
    for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
      for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
        // 计算当前位置在输入数据中的偏移量
        const scalar_t* in = input_data + (n * input_depth * input_height * input_width +
            id * input_height * input_width + ih * input_width + iw) * channels;

        int64_t d2 = 0;
        // 使用 SIMD 向量化方式处理数据，每次处理 Vec::size() 个元素
        for (; d2 < len; d2 += Vec::size()) {
          // 创建包含当前索引的向量
          iVec index_vec = iVec(id * input_height * input_width + ih * input_width + iw);
          // 从内存中加载输入数据到向量
          Vec val_vec = Vec::loadu(in + d2);
          // 从内存中加载当前最大值的索引到向量
          iVec maxindex_vec = iVec::loadu(index_ptr + d2);
          // 从内存中加载当前最大值到向量
          Vec maxval_vec = Vec::loadu(out_data + d2);

          // 创建掩码向量，用于判断哪些位置需要更新
          // true 表示当前值大于最大值或者当前值为 NaN，false 表示不更新
          Vec mask = (val_vec > maxval_vec) | is_nan_vec(val_vec);
          // 将掩码向量转换为整数类型的向量
          iVec imask = vec::cast<integer_t>(mask);
          // 根据掩码更新输出向量中的值
          Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
          // 根据掩码更新索引向量中的值
          iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

          // 将处理后的向量数据存回内存
          out_vec.store(out_data + d2);
          ind_vec.store(index_ptr + d2);
        }
        // 处理剩余不足一个向量大小的元素
        for (; d2 < size; d2++) {
          // 计算当前元素在输入数据中的索引
          int64_t index = id * input_height * input_width + ih * input_width + iw;
          // 获取当前元素的值
          scalar_t val = in[d2];
          // 获取当前元素对应的最大值索引
          int64_t maxindex = ind[d2];
          // 获取当前元素对应的最大值
          scalar_t maxval = out_data[d2];

          // 判断当前值是否大于最大值或者是否为 NaN
          bool mask = (val > maxval) || is_nan(static_cast<double>(val));
          // 根据条件更新输出数据和索引数据
          out_data[d2] = mask ? val : maxval;
          ind[d2] = mask ? index : maxindex;
        }
      }
    }
}

// 当 scalar_t 不等于 opmath_t 时执行计算的模板函数
template <typename scalar_t, typename opmath_t>
inline
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
compute_internal(
  const scalar_t* input_data,  // 输入数据数组的指针，类型为 scalar_t
  scalar_t* out_data,          // 输出数据数组的指针，类型为 scalar_t
  opmath_t* max_ptr,           // 最大值数据数组的指针，类型为 opmath_t
  vec::int_same_size_t<opmath_t>* index_ptr,  // 索引数据数组的指针，类型为 vec::int_same_size_t<opmath_t>
  int64_t* ind,                // 索引数组的指针，类型为 int64_t
  int64_t input_depth, int64_t input_height, int64_t input_width, int64_t channels,  // 输入数据的深度、高度、宽度和通道数
  int64_t n,                   // 参数 n
  int64_t len,                 // 参数 len
  int64_t size,                // 参数 size
  int64_t id0, int64_t id1,    // 参数 id0 和 id1
  int64_t ih0, int64_t ih1,    // 参数 ih0 和 ih1
  int64_t iw0, int64_t iw1,    // 参数 iw0 和 iw1
  int64_t dilationD,           // 参数 dilationD
  int64_t dilationH,           // 参数 dilationH
  int64_t dilationW) {         // 参数 dilationW
  using Vec = vec::Vectorized<scalar_t>;  // 使用 scalar_t 类型的向量化操作 Vec
  using fVec = vec::Vectorized<opmath_t>;  // 使用 opmath_t 类型的向量化操作 fVec
  using iVec = vec::Vectorized<int32_t>;   // 使用 int32_t 类型的向量化操作 iVec
  // Pass I: init out lane
  iVec index0_vec = iVec(id0 * input_height * input_width + ih0 * input_width + iw0);  // 计算初始索引值并存储于 index0_vec 中
  fVec out_vec = fVec(-std::numeric_limits<opmath_t>::infinity());  // 初始化 out_vec 为 opmath_t 类型的负无穷大值
  int64_t d1 = 0;
  for (; d1 < len; d1 += fVec::size()) {
    index0_vec.store(index_ptr + d1);  // 将 index0_vec 存储到 index_ptr + d1 处
    out_vec.store(max_ptr + d1);       // 将 out_vec 存储到 max_ptr + d1 处
  }
  for (; d1 < size; d1++) {
    ind[d1] = ih0 * input_width + iw0;  // 设置 ind 数组的值为 ih0 * input_width + iw0
    max_ptr[d1] = -std::numeric_limits<opmath_t>::infinity();  // 初始化 max_ptr 数组的值为 opmath_t 类型的负无穷大值
  }
  // Pass II: compute local max
  for (int64_t id = id0; id < id1; id += dilationD) {
    // 循环遍历输出张量的深度方向上的每个位置
    for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
      // 循环遍历输出张量的高度方向上的每个位置
      for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
        // 计算输入张量中当前位置对应的指针
        const scalar_t* in = input_data + (n * input_depth * input_height * input_width +
            id * input_height * input_width + ih * input_width + iw) * channels;

        // 初始化向量化计算需要的变量
        int64_t d2 = 0;
        // 向量化计算主循环，处理每个向量化操作的元素
        for (; d2 < len; d2 += Vec::size()) {
          // 创建输入张量当前位置的索引向量
          iVec index_ivec = iVec(id * input_height * input_width + ih * input_width + iw);
          // 加载输入数据向量
          Vec val_bvec = Vec::loadu(in + d2);
          // 将加载的数据向量转换为浮点数向量
          auto [val_fvec0, val_fvec1] = convert_to_float<scalar_t>(val_bvec);

          // 加载当前位置处的最大值索引向量和最大值向量
          iVec maxindex_ivec0 = iVec::loadu(index_ptr + d2);
          iVec maxindex_ivec1 = iVec::loadu(index_ptr + d2 + iVec::size());
          fVec maxval_fvec0 = fVec::loadu(max_ptr + d2);
          fVec maxval_fvec1 = fVec::loadu(max_ptr + d2 + fVec::size());

          // 比较当前值向量和最大值向量，生成掩码向量
          // true 表示当前值大于最大值或者为 NaN，false 表示反之
          fVec mask0 = (val_fvec0 > maxval_fvec0) | is_nan_vec(val_fvec0);
          fVec mask1 = (val_fvec1 > maxval_fvec1) | is_nan_vec(val_fvec1);
          // 将掩码向量转换为整数向量
          iVec imask0 = vec::cast<int32_t>(mask0);
          iVec imask1 = vec::cast<int32_t>(mask1);

          // 使用掩码更新最大值向量
          fVec max_fvec0 = fVec::blendv(maxval_fvec0, val_fvec0, mask0);
          fVec max_fvec1 = fVec::blendv(maxval_fvec1, val_fvec1, mask1);
          // 使用掩码更新最大值索引向量
          iVec ind_vec0 = iVec::blendv(maxindex_ivec0, index_ivec, imask0);
          iVec ind_vec1 = iVec::blendv(maxindex_ivec1, index_ivec, imask1);

          // 将更新后的最大值向量存储回数组
          max_fvec0.store(max_ptr + d2);
          max_fvec1.store(max_ptr + d2 + fVec::size());
          // out_vec.store(out + d2);
          // 将更新后的最大值索引向量存储回数组
          ind_vec0.store(index_ptr + d2);
          ind_vec1.store(index_ptr + d2 + iVec::size());
        }
        // 处理剩余的非向量化计算元素
        for (; d2 < size; d2++) {
          // 计算当前位置在输入数组中的索引
          int64_t index = id * input_height * input_width + ih * input_width + iw;
          // 获取当前位置的值
          opmath_t val = opmath_t(in[d2]);
          // 获取当前位置的最大值和最大值索引
          int64_t maxindex = ind[d2];
          opmath_t maxval = max_ptr[d2];

          // 比较当前值和最大值，生成掩码
          bool mask = (val > maxval) || std::isnan(val);
          // 根据掩码更新最大值和最大值索引数组
          max_ptr[d2] = mask ? val : maxval;
          ind[d2] = mask ? index : maxindex;
        }
      }
    }
  }
  // 将最大值数组从浮点数类型转换为 bfloat16/half 类型
  int64_t d3 = 0;
  // 向量化转换主循环
  for (; d3 < len; d3 += Vec::size()) {
    // 加载当前位置的最大值向量
    fVec max_fvec0 = fVec::loadu(max_ptr + d3);
    fVec max_fvec1 = fVec::loadu(max_ptr + d3 + fVec::size());
    // 将浮点数向量转换为目标类型向量
    Vec max_bvec = convert_from_float<scalar_t>(max_fvec0, max_fvec1);
    // 将转换后的向量存储回输出数组
    max_bvec.store(out_data + d3);
  }
  // 处理剩余的非向量化转换元素
  for (; d3 < size; d3++) {
    // 将浮点数值转换为目标类型，并存储回输出数组
    out_data[d3] = scalar_t(max_ptr[d3]);
  }
  }



template <typename scalar_t, bool is_3d>
void cpu_max_pool(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef kWHD,
    IntArrayRef dWHD,
    IntArrayRef padWHD,
    IntArrayRef dilWHD) {

函数定义，实现了一个 CPU 上的最大池化操作。参数包括输出张量 `output_`，索引张量 `indices_`，输入张量 `input_`，以及池化核大小 `kWHD`、步长 `dWHD`、填充 `padWHD`、膨胀率 `dilWHD`。


  size_t dims =  is_3d ? 3 : 2;

根据 `is_3d` 的值确定张量的维度数（2D 或 3D）。


  TORCH_CHECK(kWHD.size() == dims && dWHD.size() == dims && padWHD.size() == dims && dilWHD.size() == dims,
              "max pooling 2d/3d are not matched");

检查池化操作的核大小、步长、填充和膨胀率的维度是否与张量维度匹配，否则抛出错误信息。


  int kW = kWHD[0];
  int kH = kWHD[1];
  int dW = dWHD[0];
  int dH = dWHD[1];
  int padW = padWHD[0];
  int padH = padWHD[1];
  int dilationW = dilWHD[0];
  int dilationH = dilWHD[1];

提取池化操作的宽度、高度、步长、填充和膨胀率的具体数值。


  int kD = is_3d ? kWHD[dims - 1] : 1;
  int dD = is_3d ? dWHD[dims - 1] : 1;
  int padD = is_3d ? padWHD[dims - 1] : 0;
  int dilationD = is_3d ? dilWHD[dims - 1] : 1;

如果是 3D 池化，提取深度方向的池化核大小、步长、填充和膨胀率。


  auto input = input_.contiguous();
  auto output = output_.contiguous();
  auto indices = indices_.contiguous();

将输入、输出和索引张量转换为连续内存的张量。


  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

获取输入、输出和索引张量的数据指针。


  int64_t ndim = input.ndimension();

获取输入张量的维度数。


  // treat batch size and channels as one dimension
  //
  // MaxPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // MaxPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW

说明在池化操作中将批次大小和通道数视为一个维度。


  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
  } else {
    channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  }

根据是否是 3D 池化和输入张量的维度数，计算通道数。


  int64_t input_depth = is_3d ? input.size(-3) : 1;
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = is_3d ? output.size(-3) : 1;
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

计算输入和输出张量的深度、高度和宽度。


  using opmath_t = at::opmath_type<scalar_t>;

定义操作数类型 `opmath_t`。


  // parallel on dim N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {

并行循环处理通道和批次维度的操作。
    // 外层循环遍历输出张量的通道维度
    for (int64_t c = begin; c < end; c++) {
      // 计算当前通道在输入张量中的起始指针位置
      const scalar_t* input_ptr = input_data + c * input_depth * input_height * input_width;
      // 计算当前通道在输出张量中的起始指针位置
      scalar_t* output_ptr = output_data + c * output_depth * output_height * output_width;
      // 计算当前通道在索引张量中的起始指针位置
      int64_t* indices_ptr = indices_data + c * output_depth * output_height * output_width;

      // 遍历输出张量的深度维度
      for (int64_t od = 0; od < output_depth; od++) {
        // 计算输入张量中对应深度维度的起始和结束索引
        int64_t id0 = od * dD - padD;
        int64_t id1 = std::min(id0 + (kD - 1) * dilationD + 1, input_depth);
        while(id0 < 0) { id0 += dilationD; }

        // 遍历输出张量的高度维度
        for (int64_t oh = 0; oh < output_height; oh++) {
          // 计算输入张量中对应高度维度的起始和结束索引
          int64_t ih0 = oh * dH - padH;
          int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
          while(ih0 < 0) { ih0 += dilationH; }

          // 遍历输出张量的宽度维度
          for (int64_t ow = 0; ow < output_width; ow++) {
            // 计算输入张量中对应宽度维度的起始和结束索引
            int64_t iw0 = ow * dW - padW;
            int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
            while(iw0 < 0) { iw0 += dilationW; }

            // 计算局部最大值的位置索引
            int64_t maxindex = id0 * input_height * input_width + ih0 * input_width + iw0;
            opmath_t maxval;
            // 初始化局部最大值
            if (std::numeric_limits<opmath_t>::has_infinity) {
              maxval = -std::numeric_limits<opmath_t>::infinity();
            } else {
              maxval = std::numeric_limits<opmath_t>::min();
            }

            // 遍历当前窗口内的所有元素，找出局部最大值及其位置
            for (int64_t id = id0; id < id1; id += dilationD) {
              for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
                for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
                  int64_t index = id * input_height * input_width + ih * input_width + iw;
                  opmath_t val = input_ptr[index];
                  // 更新局部最大值及其位置
                  if ((val > maxval) || is_nan(static_cast<double>(val))) {
                    maxval = val;
                    maxindex = index;
                  }
                }
              }
            }

            // 将输出张量中的当前位置设置为局部最大值，并将最大值的位置存储到索引张量中
            int64_t i = od * output_height * output_width + oh * output_width + ow;
            output_ptr[i] = scalar_t(maxval);
            indices_ptr[i] = maxindex;
          }
        }
      }
    }
  });

  // 如果输出张量不是连续的，复制数据以保证连续性
  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
  // 如果索引张量不是连续的，复制数据以保证连续性
  if (!indices_.is_contiguous()) {
    indices_.copy_(indices);
  }
# 结束 CPU 最大池化函数的定义，该函数根据通道最后格式操作张量
template <typename scalar_t, bool is_3d>
void cpu_max_pool_channels_last(
    const Tensor& output_,               // 输出张量的引用
    const Tensor& indices_,              // 索引张量的引用
    const Tensor& input_,                // 输入张量的引用
    IntArrayRef kWHD,                    // 卷积核宽度、高度、深度
    IntArrayRef dWHD,                    // 步幅宽度、高度、深度
    IntArrayRef padWHD,                  // 填充宽度、高度、深度
    IntArrayRef dilWHD) {                // 膨胀率宽度、高度、深度
  size_t dims =  is_3d ? 3 : 2;          // 维度数，2D 或 3D
  TORCH_CHECK(kWHD.size() == dims && dWHD.size() == dims && padWHD.size() == dims && dilWHD.size() == dims,
              "max pooling 2d/3d are not matched");  // 检查卷积参数的维度是否匹配
  int64_t ndim = input_.ndimension();    // 输入张量的维度数
  // MaxPool2d: NHWC
  // MaxPool3d: NDHWC
  if (is_3d) {
    TORCH_CHECK(ndim == 5, "max pooling 3d with channels last format supports tensors with 5 dims");  // 检查是否为支持的张量维度
  } else {
    TORCH_CHECK(ndim == 4, "max pooling 2d with channels last format supports tensors with 4 dims");  // 检查是否为支持的张量维度
  }

  int kW = kWHD[0];                      // 卷积核宽度
  int kH = kWHD[1];                      // 卷积核高度
  int dW = dWHD[0];                      // 步幅宽度
  int dH = dWHD[1];                      // 步幅高度
  int padW = padWHD[0];                  // 填充宽度
  int padH = padWHD[1];                  // 填充高度
  int dilationW = dilWHD[0];             // 膨胀率宽度
  int dilationH = dilWHD[1];             // 膨胀率高度

  int kD = is_3d ? kWHD[dims - 1] : 1;   // 卷积核深度
  int dD = is_3d ? dWHD[dims - 1] : 1;   // 步幅深度
  int padD = is_3d ? padWHD[dims - 1] : 0;  // 填充深度
  int dilationD = is_3d ? dilWHD[dims - 1] : 1;  // 膨胀率深度

  auto memory_format = is_3d ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;  // 内存布局格式
  auto input = input_.contiguous(memory_format);  // 使输入张量连续，并采用指定的内存布局
  auto output = output_.contiguous(memory_format);  // 使输出张量连续，并采用指定的内存布局
  auto indices = indices_.contiguous(memory_format);  // 使索引张量连续，并采用指定的内存布局

  auto input_data = input.const_data_ptr<scalar_t>();  // 获取输入张量的数据指针
  auto output_data = output.data_ptr<scalar_t>();      // 获取输出张量的数据指针
  auto indices_data = indices.data_ptr<int64_t>();     // 获取索引张量的数据指针

  int64_t nbatch = input.size(0);         // 批次大小
  int64_t channels = input.size(1);       // 通道数
  int64_t input_depth = is_3d ? input.size(-3) : 1;  // 输入深度
  int64_t input_height = input.size(-2);  // 输入高度
  int64_t input_width = input.size(-1);   // 输入宽度
  int64_t output_depth = is_3d ? output.size(-3) : 1;  // 输出深度
  int64_t output_height = output.size(-2);  // 输出高度
  int64_t output_width = output.size(-1);  // 输出宽度

  using opmath_t = at::opmath_type<scalar_t>;  // 操作数的类型
  using Vec = vec::Vectorized<scalar_t>;       // 向量化类型
  using integer_t = vec::int_same_size_t<opmath_t>;  // 与 scalar_t 相同大小的整数类型
  // for the convenience of vectorization, use integer of the same size of scalar_t,
  //   e.g. int32_t for float, int64_t for double
  // need to make sure doesn't overflow
  TORCH_CHECK(input_depth * input_height * input_width <= std::numeric_limits<integer_t>::max());  // 检查是否会导致整数溢出

  // parallel on dim N, {D}, H, W
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);  // 初始化数据索引

    int64_t size = channels;              // 通道数大小
    int64_t len = size - (size % Vec::size());  // 向量化处理的长度
    // temp buffer holding index with integer_t
    auto index_buffer = std::make_unique<integer_t []>(len);  // 创建整数类型的索引缓冲区
    integer_t * index_ptr = index_buffer.get();  // 获取索引缓冲区的指针
    // temp buffer holding max value with opmath_t
    std::unique_ptr<opmath_t []> max_arr;   // 创建操作数类型的最大值缓冲区
    opmath_t* max_ptr = nullptr;            // 最大值缓冲区的指针
    // 如果 scalar_t 和 opmath_t 不是相同类型
    if (!std::is_same<scalar_t, opmath_t>::value) {
      // 创建一个大小为 size 的 opmath_t 类型的动态数组 max_arr
      max_arr = std::make_unique<opmath_t[]>(size);
      // 获取 max_arr 的指针，并赋给 max_ptr
      max_ptr = max_arr.get();
    }

    // 循环遍历从 begin 到 end 的索引 i
    for (int64_t i = begin; i < end; i++) {
      // 计算 id0, ih0, iw0 的初始值
      int64_t id0 = od * dD - padD;
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      // 计算 id1, ih1, iw1 的值，限制在输入张量的范围内
      int64_t id1 = std::min(id0 + (kD - 1) * dilationD + 1, input_depth);
      int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
      int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
      // 将 id0, ih0, iw0 调整为非负数
      while(id0 < 0) { id0 += dilationD; }
      while(ih0 < 0) { ih0 += dilationH; }
      while(iw0 < 0) { iw0 += dilationW; }

      // 计算输出指针 out 和索引指针 ind 的位置
      scalar_t* out = output_data + i * channels;
      int64_t* ind = indices_data + i * channels;

      // 调用 compute_internal 函数进行计算
      compute_internal(input_data, out, max_ptr, index_ptr, ind, input_depth, input_height, input_width, channels,
                        n, len, size, id0, id1, ih0, ih1, iw0, iw1,
                        dilationD, dilationH, dilationW);

      // 将 ind 转换为 integer_t 类型，存储在 index_buffer 中
      vec::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

      // 移动到下一个输出索引位置
      data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  // 如果输出张量 output_ 不是按照给定的内存格式（memory_format）连续存储，则复制到 output
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  // 如果索引张量 indices_ 不是按照给定的内存格式（memory_format）连续存储，则复制到 indices
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_max_pool_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  // 将输入张量转为连续存储
  auto grad_output = grad_output_.contiguous();
  auto indices = indices_.contiguous();
  auto grad_input = grad_input_.contiguous();

  // 获取各张量的数据指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 处理批次大小和通道数作为一个维度
  //
  // MaxPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // MaxPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW
  int64_t ndim = grad_output.ndimension();
  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  } else {
    channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  }
  int64_t input_depth = is_3d ? grad_input.size(-3) : 1;

  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = is_3d ? grad_output.size(-3) : 1;
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // 在 N 和 C 的维度上并行处理
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + c * input_depth * input_height * input_width;
      const scalar_t* grad_output_ptr = grad_output_data + c * output_depth * output_height * output_width;
      const int64_t * indices_ptr = indices_data + c * output_depth * output_height * output_width;

      // 遍历输出的深度、高度、宽度
      for (int64_t od = 0; od < output_depth; od++) {
        for (int64_t oh = 0; oh < output_height; oh++) {
          for (int64_t ow = 0; ow < output_width; ow++) {
            // 获取最大值的位置索引
            int64_t index = od * output_height * output_width + oh * output_width + ow;
            int64_t maxindex = indices_ptr[index];
            if (maxindex != -1) {
              // 更新梯度
              grad_input_ptr[maxindex] += grad_output_ptr[index];
            }
          }
        }
      }
    }
  });

  // 如果输入张量不是连续的，则拷贝数据到原张量中
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_max_pool_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  int64_t ndim = grad_output_.ndimension();
  if (is_3d) {
    // 检查是否支持通道最后格式的 MaxPool3d 反向传播
    TORCH_CHECK(ndim == 5, "MaxPool3d backward with channels last format supports tensors with 5 dims.");
  } else {
    // 检查输入张量的维度是否为4，如果不是则抛出错误信息
    TORCH_CHECK(ndim == 4, "MaxPool2d backward with channels last format supports tensors with 4 dims.");
  }
  // 根据是否为3D张量选择对应的内存布局方式
  auto memory_format = is_3d ? at::MemoryFormat::ChannelsLast3d
                             : at::MemoryFormat::ChannelsLast;
  // 将梯度输入张量转换为指定的内存布局方式，并保证其连续
  auto grad_input = grad_input_.contiguous(memory_format);
  // 将梯度输出张量转换为指定的内存布局方式，并保证其连续
  auto grad_output = grad_output_.contiguous(memory_format);
  // 将索引张量转换为指定的内存布局方式，并保证其连续
  auto indices = indices_.contiguous(memory_format);

  // 获取可修改的梯度输入数据指针
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  // 获取常量梯度输出数据指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  // 获取常量索引数据指针
  auto indices_data = indices.const_data_ptr<int64_t>();

  // 确定张量的维度信息，这里的维度命名遵循NHWC或者NDHWC格式
  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_depth = is_3d ? grad_input.size(2) : 1;
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = is_3d ? grad_output.size(2) : 1;
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // 在维度N上并行处理
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    // 循环处理每个batch中的数据
    for (const auto n : c10::irange(begin, end)) {
      // 计算当前batch的梯度输入指针位置
      scalar_t* grad_input_ptr = grad_input_data + n * input_depth * input_height * input_width * channels;
      // 计算当前batch的梯度输出指针位置
      const scalar_t* grad_output_ptr = grad_output_data + n * output_depth * output_height * output_width * channels;
      // 计算当前batch的索引指针位置
      const int64_t* indices_ptr = indices_data + n * output_depth * output_height * output_width * channels;

      // 遍历输出的深度、高度和宽度
      for (int64_t od = 0; od < output_depth; od++) {
        for (int64_t oh = 0; oh < output_height; oh++) {
          for (int64_t ow = 0; ow < output_width; ow++) {
            // 计算梯度输出和索引的偏移量，以访问当前位置的数据
            const scalar_t* gout = grad_output_ptr + (od * output_height * output_width + oh * output_width + ow) * channels;
            const int64_t* ind = indices_ptr + (od * output_height * output_width + oh * output_width + ow) * channels;

            // 在通道维度上遍历，将梯度输出加到对应的梯度输入位置上
            for (int64_t c = 0; c < channels; c++) {
              int64_t maxindex = ind[c];
              // 如果索引不为-1，则在对应位置累加梯度输出值
              if (maxindex != -1) {
                grad_input_ptr[maxindex * channels + c] += gout[c];
              }
            }
          }
        }
      }
    }
  });

  // 如果梯度输入张量不是指定的内存布局方式，则复制转换后的数据
  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void max_pool2d_kernel_impl(
    const Tensor& output,  // 输出张量，用于存储池化后的结果
    const Tensor& indices,  // 索引张量，用于存储池化过程中的索引位置
    const Tensor& input,    // 输入张量，即进行池化操作的原始数据
    int kW, int kH,         // 池化窗口的宽度和高度
    int dW, int dH,         // 池化窗口的水平和垂直步长
    int padW, int padH,     // 水平和垂直填充的大小
    int dilationW, int dilationH) {  // 水平和垂直方向的扩展率
  switch (input.suggest_memory_format()) {  // 根据输入张量的内存格式选择不同的池化方法
    case at::MemoryFormat::Contiguous: {    // 如果是连续内存格式
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool2d", [&] {
        cpu_max_pool<scalar_t, /*is 3d*/false>(output, indices, input, {kW, kH}, {dW, dH}, {padW, padH}, {dilationW, dilationH});
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {  // 如果是通道最后的内存格式
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool2d_channels_last", [&] {
        cpu_max_pool_channels_last<scalar_t, false>(output, indices, input, {kW, kH}, {dW, dH}, {padW, padH}, {dilationW, dilationH});
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");  // 不支持的内存格式异常处理
  }
}

void max_pool3d_kernel_impl(
    Tensor& output,  // 输出张量，用于存储池化后的结果
    Tensor& indices,  // 索引张量，用于存储池化过程中的索引位置
    const Tensor& input,  // 输入张量，即进行池化操作的原始数据
    int kW, int kH, int kD,  // 池化窗口的宽度、高度和深度
    int dW, int dH, int dD,  // 水平、垂直和深度方向的步长
    int padW, int padH, int padD,  // 水平、垂直和深度方向的填充大小
    int dilationW, int dilationH, int dilationD) {  // 水平、垂直和深度方向的扩展率
  if (input.ndimension() == 4) {  // 如果输入张量是四维的
    Tensor input_cl_check = input.unsqueeze(0);  // 在第0维度上增加一个维度，检查是否通道最后的内存格式
    // align with cuda:
    // work around buggy behavior of suggest_memory_format here where
    // suggested format of unsqueezed tensor is contiguous while it is
    // really only contiguous in ChannelsLast3d
    if ((!input_cl_check.is_contiguous()) &&
                     input_cl_check.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      TORCH_CHECK(output.ndimension() == 4 && indices.ndimension() == 4);  // 确保输出和索引张量也是四维的
      DimVector out_sizes(output.sizes().begin(), output.sizes().end());  // 获取输出张量的尺寸
      out_sizes.insert(out_sizes.begin(), 1);  // 在第0维度上插入1
      output.resize_(out_sizes, at::MemoryFormat::ChannelsLast3d);  // 调整输出张量的尺寸和内存格式为通道最后的三维
      DimVector indices_sizes(indices.sizes().begin(), indices.sizes().end());  // 获取索引张量的尺寸
      indices_sizes.insert(indices_sizes.begin(), 1);  // 在第0维度上插入1
      indices.resize_(indices_sizes, at::MemoryFormat::ChannelsLast3d);  // 调整索引张量的尺寸和内存格式为通道最后的三维
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool3d_channels_last", [&] {
        cpu_max_pool_channels_last<scalar_t, /*is 3d*/true>(output, indices, input_cl_check,
          {kW, kH, kD}, {dW, dH, dD}, {padW, padH, padD}, {dilationW, dilationH, dilationD});  // 调用通道最后的三维池化函数
      });
      output.squeeze_(0);  // 去除第0维度
      indices.squeeze_(0);  // 去除第0维度
      return;
    }
  }
  switch (input.suggest_memory_format()) {  // 根据输入张量的内存格式选择不同的池化方法
    case at::MemoryFormat::Contiguous: {    // 如果是连续内存格式
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool3d", [&] {
        cpu_max_pool<scalar_t, /*is 3d*/true>(output, indices, input,
            {kW, kH, kD}, {dW, dH, dD}, {padW, padH, padD}, {dilationW, dilationH, dilationD});  // 调用三维池化函数
      });
      break;
    }
    // 处理内存格式为 ChannelsLast3d 的情况
    case at::MemoryFormat::ChannelsLast3d: {
      // 使用宏展开，处理所有数据类型，包括 BFloat16 和 Half，执行 max_pool3d_channels_last 函数
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool3d_channels_last", [&] {
        // 在 CPU 上执行 channels_last 形式的最大池化操作，输出结果存储在 output 和 indices 中
        cpu_max_pool_channels_last<scalar_t, true>(output, indices, input,
          {kW, kH, kD}, {dW, dH, dD}, {padW, padH, padD}, {dilationW, dilationH, dilationD});
      });
      break; // 结束当前 case 分支
    }
    default:
      // 如果内存格式不是 ChannelsLast3d 或连续存储，则抛出错误
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
} // 匿名命名空间结束

void max_pool2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  // 根据输出梯度推荐的内存格式选择执行不同的操作
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 使用宏，根据浮点数类型和操作名称执行池化层反向传播
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool2d_backward", [&] {
        cpu_max_pool_backward<scalar_t, /*is 3d*/ false>(grad_input, grad_output, indices);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // 使用宏，根据浮点数类型和操作名称执行通道最后内存格式的池化层反向传播
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool2d_backward_channels_last", [&] {
        cpu_max_pool_backward_channels_last<scalar_t, /*is 3d*/ false>(grad_input, grad_output, indices);
      });
      break;
    }
    default:
      // 报错：不支持的内存格式，仅支持 ChannelsLast 和 Contiguous
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void max_pool3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  if (grad_output.ndimension() == 4) {
    // 创建一个维度为 5 的张量，用于检查是否符合 ChannelsLast3d 的内存格式
    Tensor grad_output_cl_check = grad_output.unsqueeze(0);
    // 对齐 cuda 的行为:
    // 解决 suggest_memory_format 的错误行为，其中建议的张量格式为 Contiguous，实际上仅在 ChannelsLast3d 中是连续的
    if ((!grad_output_cl_check.is_contiguous()) &&
                     grad_output_cl_check.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      // 检查张量维度是否为 4
      TORCH_CHECK(grad_input.ndimension() == 4 && indices.ndimension() == 4);
      // 将 grad_input 调整为 ChannelsLast3d 的内存格式
      DimVector sizes(grad_input.sizes().begin(), grad_input.sizes().end());
      sizes.insert(sizes.begin(), 1);
      grad_input.resize_(sizes, at::MemoryFormat::ChannelsLast3d);
      // 将 indices 调整为 ChannelsLast3d 的内存格式
      auto _indices = indices.unsqueeze(0).contiguous(at::MemoryFormat::ChannelsLast3d);
      // 使用宏，根据浮点数类型和操作名称执行通道最后内存格式的 3D 池化层反向传播
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool3d_backward_channels_last", [&] {
        cpu_max_pool_backward_channels_last<scalar_t, /*is_3d*/ true>(grad_input, grad_output_cl_check, _indices);
      });
      // 恢复 grad_input 的维度为 4
      grad_input.squeeze_(0);
      return;
    }
  }
  // 根据输出梯度推荐的内存格式选择执行不同的操作
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 使用宏，根据浮点数类型和操作名称执行池化层反向传播
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool3d_backward", [&] {
        cpu_max_pool_backward<scalar_t, /*is 3d*/ true>(grad_input, grad_output, indices);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      // 使用宏，根据浮点数类型和操作名称执行通道最后内存格式的 3D 池化层反向传播
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool3d_backward_channels_last", [&] {
        cpu_max_pool_backward_channels_last<scalar_t, /*is 3d*/ true>(grad_input, grad_output, indices);
      });
      break;
    }
    default:
      // 报错：不支持的内存格式，仅支持 ChannelsLast3d, Contiguous
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}
# 注册 max_pool2d_kernel 函数的分发器，将其实现函数指针注册到调度表中
REGISTER_DISPATCH(max_pool2d_kernel, &max_pool2d_kernel_impl);
# 注册 max_pool2d_backward_kernel 函数的分发器，将其实现函数指针注册到调度表中
REGISTER_DISPATCH(max_pool2d_backward_kernel, &max_pool2d_backward_kernel_impl);
# 注册 max_pool3d_kernel 函数的分发器，将其实现函数指针注册到调度表中
REGISTER_DISPATCH(max_pool3d_kernel, &max_pool3d_kernel_impl);
# 注册 max_pool3d_backward_kernel 函数的分发器，将其实现函数指针注册到调度表中
REGISTER_DISPATCH(max_pool3d_backward_kernel, &max_pool3d_backward_kernel_impl);
# 结束 at::native 命名空间
} // at::native
```