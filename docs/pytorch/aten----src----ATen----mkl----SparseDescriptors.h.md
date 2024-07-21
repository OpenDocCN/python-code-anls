# `.\pytorch\aten\src\ATen\mkl\SparseDescriptors.h`

```
    sparse_matrix_t raw_descriptor;



    // 定义一个稀疏矩阵的原始描述符



    TORCH_MKLSPARSE_CHECK(mkl_sparse_d_create_csr(
        &raw_descriptor,
        SPARSE_INDEX_BASE_ZERO,
        rows,
        cols,
        crow_indices_ptr,
        crow_indices_->size(0) - 1,
        col_indices_ptr,
        values_ptr));



    // 使用 MKL 函数创建 CSR 格式的稀疏矩阵描述符
    // 函数参数依次为：
    // - raw_descriptor：用于存储创建后的稀疏矩阵描述符的指针
    // - SPARSE_INDEX_BASE_ZERO：索引基础从零开始
    // - rows：矩阵的行数
    // - cols：矩阵的列数
    // - crow_indices_ptr：行偏移数组的指针
    // - crow_indices_->size(0) - 1：行偏移数组的长度（减一）
    // - col_indices_ptr：列索引数组的指针
    // - values_ptr：值数组的指针



    this->descriptor_.reset(raw_descriptor);



    // 将创建的稀疏矩阵描述符赋值给类成员变量 descriptor_



    // No need to free resources explicitly here, as it's managed by unique_ptr
  }
};

} // namespace at::mkl::sparse



// 结束 at::mkl::sparse 命名空间
    # 如果输入的稀疏矩阵布局为 kSparseBsr
    if (input.layout() == kSparseBsr) {
      # 断言调试模式下：值张量为三维，行索引张量为一维，列索引张量为一维
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          values.dim() == 3 && crow_indices.dim() == 1 &&
          col_indices.dim() == 1);
      # 检查值张量的最后两个维度是否相等，MKL Sparse 不支持非方块块的矩阵
      TORCH_CHECK(
          values.size(-1) == values.size(-2),
          "MKL Sparse doesn't support matrices with non-square blocks.");
      # 将值张量的最后一个维度作为块大小，进行类型转换为 mkl_int，并命名为 block_size
      auto block_size = mkl_int_cast(values.size(-1), "block_size");
      # 使用稀疏行块（BSR）格式创建稀疏矩阵描述符
      create_bsr<scalar_t>(
          &raw_descriptor,
          SPARSE_INDEX_BASE_ZERO,    // 索引从零开始
          SPARSE_LAYOUT_ROW_MAJOR,   // 行主序布局
          rows / block_size,         // 稀疏矩阵行数除以块大小
          cols / block_size,         // 稀疏矩阵列数除以块大小
          block_size,                // 块大小
          crow_indices_ptr,          // 行索引指针
          crow_indices_ptr + 1,      // 下一个行索引指针
          col_indices_ptr,           // 列索引指针
          values_ptr);               // 值指针
    } else {
      # 使用稀疏压缩行（CSR）格式创建稀疏矩阵描述符
      create_csr<scalar_t>(
          &raw_descriptor,
          SPARSE_INDEX_BASE_ZERO,    // 索引从零开始
          rows,                      // 稀疏矩阵行数
          cols,                      // 稀疏矩阵列数
          crow_indices_ptr,          // 行索引指针
          crow_indices_ptr + 1,      // 下一个行索引指针
          col_indices_ptr,           // 列索引指针
          values_ptr);               // 值指针
    }

    # 将原始描述符作为智能指针包装，并赋给成员变量 descriptor_
    descriptor_.reset(raw_descriptor);
  }

  # 默认构造函数，初始化稀疏矩阵描述符为空
  MklSparseCsrDescriptor() {
    sparse_matrix_t raw_descriptor = nullptr;
    descriptor_.reset(raw_descriptor);
  }

 private:
  # 存储行索引的张量，可能由外部拥有
  c10::MaybeOwned<Tensor> crow_indices_;
  # 存储列索引的张量，可能由外部拥有
  c10::MaybeOwned<Tensor> col_indices_;
  # 存储值的张量，可能由外部拥有
  c10::MaybeOwned<Tensor> values_;
};

// 结束命名空间 at::mkl::sparse
} // namespace at::mkl::sparse
```