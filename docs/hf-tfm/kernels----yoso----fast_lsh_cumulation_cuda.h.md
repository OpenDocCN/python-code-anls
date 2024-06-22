# `.\transformers\kernels\yoso\fast_lsh_cumulation_cuda.h`

```py
# CUDA 核函数，用于计算快速哈希的第一版本
__global__ void fast_hash_ver1_cuda_kernel(
  int *mask,        // [batch_size, num_vector]，用于指示哪些向量要计算哈希码
  float *vector,    // [batch_size, num_vector, vector_dim]，输入的向量数据
  int *Dmat,        // [3, num_part, vector_dim]，哈希函数参数
  int *hash_code,   // [batch_size, num_vector, num_hash_f]，存储计算得到的哈希码
  int batch_size,   // 批量处理的样本数量
  int num_vector,   // 向量数量
  int vector_dim,   // 向量维度
  int num_part,     // 哈希函数参数
  int num_hash_f,   // 哈希函数数量
  int hash_code_len // 哈希码长度
);

# CUDA 核函数，LSH 累积的第一步
__global__ void lsh_cumulation_ver1_step1_cuda_kernel(
  int *key_mask,           // [batch_size, num_key]，关键字的掩码
  int *key_hash_code,      // [batch_size, num_key, num_hash_f]，关键字的哈希码
  float *value,            // [batch_size, num_key, value_dim]，关键字对应的值
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, value_dim]，哈希表中的值
  int batch_size,          // 批量处理的样本数量
  int num_hash_f,          // 哈希函数数量
  int hashtable_capacity,  // 哈希表容量
  int num_key,             // 关键字数量
  int value_dim,           // 值的维度
  int offset_warp          // 偏移 warp 的数量
);

# CUDA 核函数，LSH 累积的第二步
__global__ void lsh_cumulation_ver1_step2_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]，查询的掩码
  int *query_hash_code,    // [batch_size, num_query, num_hash_f]，查询的哈希码
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, value_dim]，哈希表中的值
  float *cumulation_value, // [batch_size, num_query, value_dim]，累积的值
  int batch_size,          // 批量处理的样本数量
  int num_hash_f,          // 哈希函数数量
  int hashtable_capacity,  // 哈希表容量
  int num_query,           // 查询数量
  int value_dim,           // 值的维度
  int offset_warp          // 偏移 warp 的数量
);

# CUDA 核函数，带权重的 LSH 累积的第一步
__global__ void lsh_weighted_cumulation_ver1_step1_cuda_kernel(
  int *key_mask,            // [batch_size, num_key]，关键字的掩码
  int *key_hash_code,       // [batch_size, num_key, num_hash_f]，关键字的哈希码
  float *key_weight,        // [batch_size, num_key, weight_dim]，关键字的权重
  float *value,             // [batch_size, num_key, value_dim]，关键字对应的值
  float *hashtable_value,   // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]，哈希表中的值
  int batch_size,           // 批量处理的样本数量
  int num_hash_f,           // 哈希函数数量
  int hashtable_capacity,   // 哈希表容量
  int num_key,              // 关键字数量
  int value_dim,            // 值的维度
  int weight_dim,           // 权重的维度
  int offset_warp,          // 偏移 warp 的数量
  int weight_idx            // 权重索引
);

# CUDA 核函数，带权重的 LSH 累积的第二步
__global__ void lsh_weighted_cumulation_ver1_step2_cuda_kernel(
  int *query_mask,          // [batch_size, num_query]，查询的掩码
  int *query_hash_code,     // [batch_size, num_query, num_hash_f]，查询的哈希码
  float *query_weight,      // [batch_size, num_query, weight_dim]，查询的权重
  float *hashtable_value,   // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]，哈希表中的值
  float *cumulation_value,  // [batch_size, num_query, value_dim]，累积的值
  int batch_size,           // 批量处理的样本数量
  int num_hash_f,           // 哈希函数数量
  int hashtable_capacity,   // 哈希表容量
  int num_query,            // 查询数量
  int value_dim,            // 值的维度
  int weight_dim,           // 权重的维度
  int offset_warp,          // 偏移 warp 的数量
  int weight_idx            // 权重索引
);

# CUDA 核函数，计数排序的第一步
__global__ void count_sort_step1_cuda_kernel(
  int *key_mask,         // [batch_size, num_key]，关键字的掩码
  int *key_hash_code,    // [batch_size, num_key, num_hash_f]，关键字的哈希码
  int *count_sort_table, // [batch_size, num_hash_f, hashtable_capacity]，计数排序表
  int batch_size,        // 批量处理的样本数量
  int num_hash_f,        // 哈希函数数量
  int hashtable_capacity,// 哈希表容量
  int num_key            // 关键字数量
);

# CUDA 核函数，计数排序的第二步
__global__ void count_sort_step2_cuda_kernel(
  int *count_sort_table,  // [batch_size, num_hash_f, hashtable_capacity]，计数排序表
  int batch_size,         // 批量处理的样本数量
  int num_hash_f,         // 哈希函数数量
// CUDA 核函数：执行计数排序的第三步
__global__ void count_sort_step3_cuda_kernel(
  int *key_mask,          // 关键字掩码，形状为 [batch_size, num_key]
  int *key_hash_code,     // 关键字哈希码，形状为 [batch_size, num_key, num_hash_f]
  int *count_sort_table,  // 计数排序表，形状为 [batch_size, num_hash_f, hashtable_capacity]
  int *key_sorted_idxes,  // 排序后的关键字索引，形状为 [batch_size, num_hash_f, num_key]
  int batch_size,         // 批次大小
  int num_hash_f,         // 哈希函数数量
  int hashtable_capacity, // 哈希表容量
  int num_key             // 关键字数量
);

// CUDA 核函数：提取查询信息
__global__ void extract_query_info_cuda_kernel(
  int *query_mask,       // 查询掩码，形状为 [batch_size, num_query]
  int *query_hash_code,  // 查询哈希码，形状为 [batch_size, num_query, num_hash_f]
  int *count_sort_table, // 计数排序表，形状为 [batch_size, num_hash_f, hashtable_capacity]
  int *query_info,       // 查询信息，形状为 [batch_size, num_query, 2, num_hash_f]
  int batch_size,        // 批次大小
  int num_hash_f,        // 哈希函数数量
  int hashtable_capacity,// 哈希表容量
  int num_query          // 查询数量
);

// CUDA 核函数：LSH 加权累积版本 2 的第二步
__global__ void lsh_weighted_cumulation_ver2_step2_cuda_kernel(
  int *query_mask,         // 查询掩码，形状为 [batch_size, num_query]
  int *query_info,         // 查询信息，形状为 [batch_size, num_query, 2, num_hash_f]
  int *key_sorted_idxes,   // 排序后的关键字索引，形状为 [batch_size, num_hash_f, num_key]
  float *query_weight,     // 查询权重，形状为 [batch_size, num_query, weight_dim]
  float *key_weight,       // 关键字权重，形状为 [batch_size, num_key, weight_dim]
  float *value,            // 值，形状为 [batch_size, num_key, value_dim]
  float *cumulation_value, // 累积值，形状为 [batch_size, num_query, value_dim]
  int batch_size,          // 批次大小
  int num_hash_f,          // 哈希函数数量
  int num_query,           // 查询数量
  int num_key,             // 关键字数量
  int value_dim,           // 值维度
  int weight_dim           // 权重维度
);

// CUDA 核函数：LSH 加权累积版本 3 的第二步
__global__ void lsh_weighted_cumulation_ver3_step2_cuda_kernel(
  int *query_sorted_idxes,   // 排序后的查询索引，形状为 [batch_size, num_hash_f, num_query]
  int *key_mask,             // 关键字掩码，形状为 [batch_size, num_key]
  int *key_info,             // 关键字信息，形状为 [batch_size, num_key, 2, num_hash_f]
  float *query_weight,       // 查询权重，形状为 [batch_size, num_query, weight_dim]
  float *key_weight,         // 关键字权重，形状为 [batch_size, num_key, weight_dim]
  float *value,              // 值，形状为 [batch_size, num_key, value_dim]
  float *cumulation_value,   // 累积值，形状为 [batch_size, num_query, value_dim]
  int batch_size,            // 批次大小
  int num_hash_f,            // 哈希函数数量
  int num_query,             // 查询数量
  int num_key,               // 关键字数量
  int value_dim,             // 值维度
  int weight_dim             // 权重维度
);

// CUDA 核函数：LSH 加权累积版本 4 的第二步
__global__ void lsh_weighted_cumulation_ver4_step2_cuda_kernel(
  int *query_sorted_idxes,   // 排序后的查询索引，形状为 [batch_size, num_hash_f, num_query]
  int *key_mask,             // 关键字掩码，形状为 [batch_size, num_key]
  int *key_info,             // 关键字信息，形状为 [batch_size, num_key, 2, num_hash_f]
  float *query_weight,       // 查询权重，形状为 [batch_size, num_query, weight_dim]
  float *key_weight,         // 关键字权重，形状为 [batch_size, num_key, weight_dim]
  float *value,              // 值，形状为 [batch_size, num_key, value_dim]
  float *cumulation_value,   // 累积值，形状为 [batch_size, num_query, value_dim]
  int batch_size,            // 批次大小
  int num_hash_f,            // 哈希函数数量
  int num_query,             // 查询数量
  int num_key,               // 关键字数量
  int value_dim,             // 值维度
  int weight_dim             // 权重维度
);
```