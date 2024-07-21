# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\include\pack_block_sparse.h`

```
/*
 * 版权所有（c）Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 本源代码使用 BSD 风格许可证，可在根目录下的 LICENSE 文件中找到。
 */

#pragma once  // 只包含一次的头文件保护

#include <cassert>  // 断言库，用于调试时的条件检查
#include <cstdint>  // 标准整数类型
#include <memory>   // 内存管理库
#include <vector>   // 动态数组容器

#ifndef _WIN32
#include <qnnpack/AlignedAllocator.h>  // 自定义对齐分配器
#endif

#include <pytorch_qnnpack.h>  // PyTorch QNNPACK 头文件
#include <qnnpack/common.h>   // QNNPACK 公共定义
#include <qnnpack/math.h>     // QNNPACK 数学函数

#ifdef QNNPACK_BCSRMATRIX_DEBUG
#include <iostream>  // 标准输出流，用于调试信息打印
#endif // QNNPACK_BCSRMATRIX_DEBUG

namespace qnnpack {

template <typename T>
struct OwnedOrBorrowedVector {
  using VECTOR_T =
#ifndef _WIN32
      std::vector<T, AlignedAllocator<T, 16>>;  // 对齐分配的向量容器
#else
      std::vector<T>;  // 普通的向量容器
#endif

  VECTOR_T owned_vec_data_;         // 拥有数据的向量
  std::tuple<T*, uint32_t> borrowed_tuple_data_;  // 借用数据的元组
  bool owned;                       // 标识是否拥有数据

  VECTOR_T& vector() {              // 获取向量的引用
    assert(owned);                  // 断言确保数据是拥有的
    return owned_vec_data_;         // 返回拥有的向量
  }

  uint32_t size() const {           // 获取数据大小
    if (owned) {                    // 如果拥有数据
      return owned_vec_data_.size();  // 返回拥有的向量大小
    } else {                        // 否则
      return std::get<1>(borrowed_tuple_data_);  // 返回借用数据的大小
    }
  }

  const T* data() const {           // 获取数据指针
    if (owned) {                    // 如果拥有数据
      return owned_vec_data_.data();  // 返回拥有的向量数据指针
    } else {                        // 否则
      return std::get<0>(borrowed_tuple_data_);  // 返回借用数据的指针
    }
  }

  const T& operator[](int i) const {  // 重载下标操作符
    return data()[i];               // 返回指定下标的数据
  }

  OwnedOrBorrowedVector() : owned(true) {}  // 默认构造函数，拥有数据

  OwnedOrBorrowedVector(T* data_ptr, const uint32_t size)
      : borrowed_tuple_data_(std::tuple<T*, uint32_t>(data_ptr, size)),
        owned(false) {}             // 带参数的构造函数，借用数据
};

struct BCSRMatrix {
  OwnedOrBorrowedVector<uint8_t> values;  // 存储稀疏矩阵值的数据结构
  uint32_t col_block_size;    // 列块大小，输入特征块大小
  uint32_t row_block_size;    // 行块大小，输出特征块大小
  enum pytorch_qnnp_sparse_matrix_indices_dtype indices_dtype;  // 稀疏矩阵索引数据类型
  virtual ~BCSRMatrix() = default;  // 虚析构函数

  // 获取列索引数据指针的纯虚函数
  virtual const void* col_indices_data_ptr() const = 0;

  // 获取行值数据指针的纯虚函数
  virtual const void* row_values_data_ptr() const = 0;

#ifdef QNNPACK_BCSRMATRIX_DEBUG
  // 打印调试信息的纯虚函数
  virtual void print() const = 0;
#endif // QNNPACK_BCSRMATRIX_DEBUG

  /*
   * 从 BCSR 格式解包到 Dense 格式
   * - 每个值和零点通过减去 128 转换为 int8_t 类型
   * - num_rows 和 num_cols 是密集权重张量的维度
   * - dst 应能够容纳 num_rows * num_cols 个元素
   * - zero_points 应能够容纳 num_rows 个零点
   */
  virtual void unpack(
      int8_t* dst,
      const int64_t num_rows,
      const int64_t num_cols,
      const uint8_t* zero_points) const = 0;

  // 获取最大索引的纯虚函数
  virtual uint32_t max_index() const = 0;
};

template <typename INDICES_DTYPE>
struct TypedBCSRMatrix : BCSRMatrix {
  OwnedOrBorrowedVector<INDICES_DTYPE> col_indices;  // 列索引向量，存储非零元素的列索引
  OwnedOrBorrowedVector<INDICES_DTYPE> row_values;   // 行值向量，存储行索引到非零元素的映射
  TypedBCSRMatrix();  // 默认构造函数声明
  const void* col_indices_data_ptr() const override;  // 返回列索引数据的指针，覆盖基类函数
  const void* row_values_data_ptr() const override;   // 返回行值数据的指针，覆盖基类函数
#ifdef QNNPACK_BCSRMATRIX_DEBUG
  void print() const override;  // 调试时打印函数，覆盖基类函数
#endif // QNNPACK_BCSRMATRIX_DEBUG
  void unpack(
      int8_t* dst,
      const int64_t num_rows,
      const int64_t num_cols,
      const uint8_t* zero_points) const override;  // 解包函数，将矩阵解包为密集格式
  uint32_t max_index() const override;  // 返回最大索引值，覆盖基类函数

  ~TypedBCSRMatrix() override = default;  // 默认析构函数声明
};

template <typename INDICES_DTYPE>
std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    const uint8_t* a,
    const size_t N,
    const size_t K,
    const uint32_t row_block_size,
    const uint32_t col_block_size,
    const uint8_t* zero_points) {
  assert(K > 0);  // 断言确保 K 大于 0
  std::unique_ptr<TypedBCSRMatrix<INDICES_DTYPE>> bcsr_mat =
      std::make_unique<TypedBCSRMatrix<INDICES_DTYPE>>();  // 创建 TypedBCSRMatrix 对象的智能指针
  auto& row_values = bcsr_mat->row_values.vector();  // 获取行值向量的引用
  auto& col_indices = bcsr_mat->col_indices.vector();  // 获取列索引向量的引用
  auto& values = bcsr_mat->values.vector();  // 获取值向量的引用

  const uint32_t num_row_blocks = (N + row_block_size - 1) / row_block_size;  // 计算行块数
  const uint32_t num_col_blocks = (K + col_block_size - 1) / col_block_size;  // 计算列块数

  row_values.reserve(num_row_blocks);  // 预留行值向量的空间
  uint32_t num_nnz_blocks{0};  // 非零块计数初始化为零
  row_values.push_back(num_nnz_blocks);  // 在行值向量中添加一个零计数块
  for (uint32_t i = 0; i < num_row_blocks; ++i) {  // 循环遍历行块
    for (uint32_t j = 0; j < num_col_blocks; ++j) {  // 在每个行块内循环遍历列块
      bool block_zero{true};  // 块是否全零的标志，默认为真
      for (uint32_t ib = 0; ib < row_block_size; ++ib) {  // 循环遍历行块内的行
        uint32_t row_index = i * row_block_size + ib;  // 计算行索引
        if PYTORCH_QNNP_UNLIKELY(row_index >= N) {  // 如果行索引超出范围，跳出循环
          break;
        }
        for (uint32_t jb = 0; jb < col_block_size; ++jb) {  // 在每个行内循环遍历列
          uint32_t col_index = j * col_block_size + jb;  // 计算列索引
          if PYTORCH_QNNP_UNLIKELY(col_index >= K) {  // 如果列索引超出范围，跳转到块扫描结束标签
            goto block_scanned;
          }
          if (*(a + row_index * K + col_index) != zero_points[row_index]) {  // 检查非零元素是否等于零点
            block_zero = false;  // 如果非零元素不等于零点，设置块非零标志为假
            goto block_scanned;  // 跳转到块扫描结束标签
          }
        }
      }
      block_scanned:;  // 块扫描结束标签
      if (!block_zero) {  // 如果块不全为零
        row_values.push_back(num_nnz_blocks);  // 在行值向量中添加非零块计数
        for (uint32_t ib = 0; ib < row_block_size; ++ib) {  // 再次遍历块内的行
          uint32_t row_index = i * row_block_size + ib;  // 计算行索引
          if PYTORCH_QNNP_UNLIKELY(row_index >= N) {  // 如果行索引超出范围，跳出循环
            break;
          }
          for (uint32_t jb = 0; jb < col_block_size; ++jb) {  // 在每个行内再次循环遍历列
            uint32_t col_index = j * col_block_size + jb;  // 计算列索引
            if PYTORCH_QNNP_UNLIKELY(col_index >= K) {  // 如果列索引超出范围，跳出循环
              break;
            }
            values.push_back(*(a + row_index * K + col_index));  // 将非零值添加到值向量中
            col_indices.push_back(col_index);  // 将列索引添加到列索引向量中
          }
        }
        ++num_nnz_blocks;  // 非零块计数加一
      }
    }
  }
block_scanned:
  // 如果 block_zero 为假，则执行以下代码块
  if (!block_zero) {
    // 将列索引 j 添加到 col_indices 中
    col_indices.push_back(j);
    // 非零块计数加一
    num_nnz_blocks++;
    // 遍历行块的每一行
    for (uint32_t ib = 0; ib < row_block_size; ++ib) {
      // 计算当前行的索引
      uint32_t row_index = i * row_block_size + ib;
      // 如果行索引超出 N 的范围，则填充零点值直至行块的末尾
      if PYTORCH_QNNP_UNLIKELY(row_index >= N) {
        for (; row_index < (num_row_blocks * row_block_size); row_index++) {
          for (uint32_t jb = 0; jb < col_block_size; ++jb) {
            values.push_back(zero_points[N-1]);
          }
        }
        break;
      }
      // 否则，遍历当前列块的每一列
      for (uint32_t jb = 0; jb < col_block_size; ++jb) {
        // 计算当前列的索引
        uint32_t col_index = j * col_block_size + jb;
        // 如果列索引超出 K 的范围，则使用行索引对应的零点值填充
        if PYTORCH_QNNP_UNLIKELY(col_index >= K) {
          values.push_back(zero_points[row_index]);
        } else {
          // 否则，从输入数组 a 中获取值并添加到 values 中
          uint8_t val = *(a + row_index * K + col_index);
          values.push_back(val);
        }
      }
    }
  }
}
// 将非零块数目添加到 row_values 中
row_values.push_back(num_nnz_blocks);
// 设置 bcsr_mat 的行块大小和列块大小
bcsr_mat->row_block_size = row_block_size;
bcsr_mat->col_block_size = col_block_size;
// 返回 bcsr_mat 智能指针
return bcsr_mat;
}

// 生成块压缩稀疏行（BCSR）矩阵的模板函数
template <typename INDICES_DTYPE>
std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    INDICES_DTYPE* col_indices,
    INDICES_DTYPE* row_values,
    uint8_t* values,
    const int64_t col_indices_size,
    const int64_t row_values_size,
    const int64_t values_size,
    const int64_t row_block_size,
    const int64_t col_block_size) {
  // 创建 INDICES_DTYPE 类型的 BCSRMatrix 智能指针
  std::unique_ptr<TypedBCSRMatrix<INDICES_DTYPE>> bcsr_mat =
      std::make_unique<TypedBCSRMatrix<INDICES_DTYPE>>();
  // 初始化 bcsr_mat 的 col_indices、row_values 和 values 字段
  bcsr_mat->col_indices =
      OwnedOrBorrowedVector<INDICES_DTYPE>(col_indices, col_indices_size);
  bcsr_mat->row_values =
      OwnedOrBorrowedVector<INDICES_DTYPE>(row_values, row_values_size);
  bcsr_mat->values = OwnedOrBorrowedVector<uint8_t>(values, values_size);
  // 设置 bcsr_mat 的行块大小和列块大小
  bcsr_mat->row_block_size = row_block_size;
  bcsr_mat->col_block_size = col_block_size;
  // 返回 bcsr_mat 智能指针
  return bcsr_mat;
}

// 对于不同的 INDICES_DTYPE，定义其对应的稀疏矩阵索引数据类型
template <typename INDICES_DTYPE>
struct IndicesDtypeEnumTrait {
  // 当 INDICES_DTYPE 为 uint32_t 时，dtype 为 pytorch_qnnp_sparse_matrix_indices_dtype_uint32_t
  static_assert(
      sizeof(INDICES_DTYPE) == 0,
      "Invalid dtype for IndicesDtypeEnumTrait");
};

template <>
struct IndicesDtypeEnumTrait<uint32_t> {
  const static pytorch_qnnp_sparse_matrix_indices_dtype dtype =
      pytorch_qnnp_sparse_matrix_indices_dtype_uint32_t;
};

template <>
struct IndicesDtypeEnumTrait<uint16_t> {
  const static pytorch_qnnp_sparse_matrix_indices_dtype dtype =
      pytorch_qnnp_sparse_matrix_indices_dtype_uint16_t;
};

template <>
struct IndicesDtypeEnumTrait<uint8_t> {
  const static pytorch_qnnp_sparse_matrix_indices_dtype dtype =
      pytorch_qnnp_sparse_matrix_indices_dtype_uint8_t;
};

// TypedBCSRMatrix 类的模板构造函数
template <typename INDICES_DTYPE>
TypedBCSRMatrix<INDICES_DTYPE>::TypedBCSRMatrix() {
  // 设置 indices_dtype 为 INDICES_DTYPE 对应的稀疏矩阵索引数据类型
  indices_dtype = IndicesDtypeEnumTrait<INDICES_DTYPE>::dtype;
}

// 返回 col_indices 数据的指针
template <typename INDICES_DTYPE>
const void* TypedBCSRMatrix<INDICES_DTYPE>::col_indices_data_ptr() const {
  return static_cast<const void*>(col_indices.data());
}

template <typename INDICES_DTYPE>
// 返回行值数据的常指针
const void* TypedBCSRMatrix<INDICES_DTYPE>::row_values_data_ptr() const {
  return static_cast<const void*>(row_values.data());
}

#ifdef QNNPACK_BCSRMATRIX_DEBUG
// 打印稀疏矩阵的调试信息，仅在 QNNPACK_BCSRMATRIX_DEBUG 宏定义时有效
template <typename INDICES_DTYPE>
void TypedBCSRMatrix<INDICES_DTYPE>::print() const {
  std::cout << "row block size:" << row_block_size << std::endl; // 打印行块大小
  std::cout << "col block size:" << col_block_size << std::endl; // 打印列块大小
  std::cout << "row ptr\n"; // 打印行指针信息
  std::cout
      << "indices dtype: uint"
      << static_cast<
             std::underlying_type_t<pytorch_qnnp_sparse_matrix_indices_dtype>>(
             indices_dtype)
      << "_t" << std::endl; // 打印索引数据类型
  for (uint32_t i = 0; i < row_values.size(); i++) {
    std::cout << (uint32_t)row_values[i] << ", "; // 打印行值数组内容
  }
  std::cout << std::endl;
  std::cout << "col indices\n"; // 打印列索引信息
  for (uint32_t i = 0; i < col_indices.size(); i++) {
    std::cout << (uint32_t)col_indices[i] << ", "; // 打印列索引数组内容
  }
  std::cout << std::endl;
  std::cout << "Actual values\n"; // 打印实际数值信息
  for (uint32_t i = 0; i < values.size(); i++) {
    std::cout << (uint32_t)values[i] << ", "; // 打印实际数值数组内容
  }
  std::cout << std::endl;
}
#endif // QNNPACK_BCSRMATRIX_DEBUG

// 将稀疏矩阵解包到目标数组中
template <typename INDICES_DTYPE>
void TypedBCSRMatrix<INDICES_DTYPE>::unpack(
    int8_t* dst,
    const int64_t num_rows,
    const int64_t num_cols,
    const uint8_t* zero_points) const {
  for (int64_t i = 0; i < num_rows; i++) {
    // 使用零点偏移量填充目标数组的每一行
    memset(
        dst + i * num_cols,
        static_cast<int8_t>(static_cast<int16_t>(zero_points[i]) - 128),
        num_cols * sizeof(int8_t));
  }

  const int64_t num_block_rows = static_cast<int64_t>(row_values.size()) - 1;
  const int64_t block_size = (int64_t)row_block_size * col_block_size;
  int64_t weight_values_num = 0;
  for (int64_t block_row_num = 0; block_row_num < num_block_rows;
       block_row_num++) {
    const int64_t num_blocks_in_current_block_row =
        row_values[block_row_num + 1] - row_values[block_row_num];
    for (int64_t k = 0; k < num_blocks_in_current_block_row;
         k++) { // 遍历当前行中的每个块
      const int64_t block_start_row_num = block_row_num * row_block_size;
      const int64_t block_start_col_num =
          (int64_t)(col_indices[weight_values_num / block_size]) *
          col_block_size;
      for (int64_t l = 0; l < block_size;
           l++) { // 遍历块中的每个值
        const int64_t row_num = block_start_row_num + l / col_block_size;
        const int64_t col_num = block_start_col_num + l % col_block_size;
        if (row_num < num_rows && col_num < num_cols) {
          // 将稀疏矩阵中的值解包到目标数组中
          dst[row_num * num_cols + col_num] = static_cast<int8_t>(
              static_cast<int16_t>(values[weight_values_num]) - 128);
        }
        weight_values_num++;
      }
    }
  }
}
/**
 * Compute the maximum index present in a TypedBCSRMatrix object.
 * This function retrieves the maximum of the largest element in
 * row_values and col_indices arrays of the TypedBCSRMatrix.
 */
uint32_t TypedBCSRMatrix<INDICES_DTYPE>::max_index() const {
  // Compute the maximum index value using std::max_element on row_values and col_indices arrays
  return static_cast<uint32_t>(std::max(
      *std::max_element(
          row_values.data(), row_values.data() + row_values.size()),
      *std::max_element(
          col_indices.data(), col_indices.data() + col_indices.size())));
}

/**
 * Macro to dispatch operations based on the indices data type of a BCSRMatrix.
 * The macro creates a lambda function that switches on bcsr->indices_dtype and
 * executes a provided dispatch_body with the appropriate TypedBCSRMatrix instance.
 * It handles uint32_t, uint16_t, and uint8_t types, and asserts on invalid types.
 */
#define QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(bcsr_, dispatch_body)        \
  [&bcsr = bcsr_]() {                                                          \
    switch (bcsr->indices_dtype) {                                             \
      case pytorch_qnnp_sparse_matrix_indices_dtype_uint32_t: {                \
        using INDICES_DTYPE = uint32_t;                                        \
        // Cast bcsr to TypedBCSRMatrix<INDICES_DTYPE> and execute dispatch_body
        const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>* typed_bcsr =            \
            static_cast<const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>*>(       \
                bcsr.get());                                                   \
        return [&typed_bcsr]() dispatch_body();                                \
      }                                                                        \
      case pytorch_qnnp_sparse_matrix_indices_dtype_uint16_t: {                \
        using INDICES_DTYPE = uint16_t;                                        \
        // Cast bcsr to TypedBCSRMatrix<INDICES_DTYPE> and execute dispatch_body
        const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>* typed_bcsr =            \
            static_cast<const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>*>(       \
                bcsr.get());                                                   \
        return [&typed_bcsr]() dispatch_body();                                \
      }                                                                        \
      case pytorch_qnnp_sparse_matrix_indices_dtype_uint8_t: {                 \
        using INDICES_DTYPE = uint8_t;                                         \
        // Cast bcsr to TypedBCSRMatrix<INDICES_DTYPE> and execute dispatch_body
        const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>* typed_bcsr =            \
            static_cast<const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>*>(       \
                bcsr.get());                                                   \
        return [&typed_bcsr]() dispatch_body();                                \
      }                                                                        \
      case pytorch_qnnp_sparse_matrix_indices_dtype_invalid: {                 \
        // Assertion for invalid indices_dtype
        assert(false);                                                         \
      }                                                                        \
    }                                                                          \
    /* Throw exception to avoid the following errors: */
    /* 抛出异常，指示在 QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE 中的无效索引数据类型 */
    /* 抛出 std::invalid_argument 异常，指示函数返回路径中存在无法处理的无效情况 */
    throw std::invalid_argument(
        "Invalid indices dtype in QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE");
} // namespace qnnpack
```