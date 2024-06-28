# `.\kernels\yoso\fast_lsh_cumulation_cuda.h`

```py
__global__ void fast_hash_ver1_cuda_kernel(
  int *mask,        // [batch_size, num_vector]，用于存储掩码数据的整数指针
  float *vector,    // [batch_size, num_vector, vector_dim]，存储向量数据的浮点数指针
  int *Dmat,        // [3, num_part, vector_dim]，存储分割矩阵数据的整数指针
  int *hash_code,   // [batch_size, num_vector, num_hash_f]，存储哈希码数据的整数指针
  int batch_size,   // 批处理大小，整数参数
  int num_vector,   // 向量数量，整数参数
  int vector_dim,   // 向量维度，整数参数
  int num_part,     // 分割数，整数参数
  int num_hash_f,   // 哈希函数数量，整数参数
  int hash_code_len // 哈希码长度，整数参数
);

__global__ void lsh_cumulation_ver1_step1_cuda_kernel(
  int *key_mask,           // [batch_size, num_key]，用于存储键掩码数据的整数指针
  int *key_hash_code,      // [batch_size, num_key, num_hash_f]，存储键哈希码数据的整数指针
  float *value,            // [batch_size, num_key, value_dim]，存储值数据的浮点数指针
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, value_dim]，哈希表值的浮点数指针
  int batch_size,          // 批处理大小，整数参数
  int num_hash_f,          // 哈希函数数量，整数参数
  int hashtable_capacity,  // 哈希表容量，整数参数
  int num_key,             // 键数量，整数参数
  int value_dim,           // 值维度，整数参数
  int offset_warp          // 偏移量（warp），整数参数
);

__global__ void lsh_cumulation_ver1_step2_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]，用于存储查询掩码数据的整数指针
  int *query_hash_code,    // [batch_size, num_query, num_hash_f]，存储查询哈希码数据的整数指针
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, value_dim]，哈希表值的浮点数指针
  float *cumulation_value, // [batch_size, num_query, value_dim]，累积值的浮点数指针
  int batch_size,          // 批处理大小，整数参数
  int num_hash_f,          // 哈希函数数量，整数参数
  int hashtable_capacity,  // 哈希表容量，整数参数
  int num_query,           // 查询数量，整数参数
  int value_dim,           // 值维度，整数参数
  int offset_warp          // 偏移量（warp），整数参数
);

__global__ void lsh_weighted_cumulation_ver1_step1_cuda_kernel(
  int *key_mask,            // [batch_size, num_key]，用于存储键掩码数据的整数指针
  int *key_hash_code,       // [batch_size, num_key, num_hash_f]，存储键哈希码数据的整数指针
  float *key_weight,        // [batch_size, num_key, weight_dim]，存储键权重数据的浮点数指针
  float *value,             // [batch_size, num_key, value_dim]，存储值数据的浮点数指针
  float *hashtable_value,   // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]，哈希表值的浮点数指针
  int batch_size,           // 批处理大小，整数参数
  int num_hash_f,           // 哈希函数数量，整数参数
  int hashtable_capacity,   // 哈希表容量，整数参数
  int num_key,              // 键数量，整数参数
  int value_dim,            // 值维度，整数参数
  int weight_dim,           // 权重维度，整数参数
  int offset_warp,          // 偏移量（warp），整数参数
  int weight_idx            // 权重索引，整数参数
);

__global__ void lsh_weighted_cumulation_ver1_step2_cuda_kernel(
  int *query_mask,          // [batch_size, num_query]，用于存储查询掩码数据的整数指针
  int *query_hash_code,     // [batch_size, num_query, num_hash_f]，存储查询哈希码数据的整数指针
  float *query_weight,      // [batch_size, num_query, weight_dim]，存储查询权重数据的浮点数指针
  float *hashtable_value,   // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]，哈希表值的浮点数指针
  float *cumulation_value,  // [batch_size, num_query, value_dim]，累积值的浮点数指针
  int batch_size,           // 批处理大小，整数参数
  int num_hash_f,           // 哈希函数数量，整数参数
  int hashtable_capacity,   // 哈希表容量，整数参数
  int num_query,            // 查询数量，整数参数
  int value_dim,            // 值维度，整数参数
  int weight_dim,           // 权重维度，整数参数
  int offset_warp,          // 偏移量（warp），整数参数
  int weight_idx            // 权重索引，整数参数
);

__global__ void count_sort_step1_cuda_kernel(
  int *key_mask,         // [batch_size, num_key]，用于存储键掩码数据的整数指针
  int *key_hash_code,    // [batch_size, num_key, num_hash_f]，存储键哈希码数据的整数指针
  int *count_sort_table, // [batch_size, num_hash_f, hashtable_capacity]，计数排序表的整数指针
  int batch_size,        // 批处理大小，整数参数
  int num_hash_f,        // 哈希函数数量，整数参数
  int hashtable_capacity,// 哈希表容量，整数参数
  int num_key            // 键数量，整数参数
);

__global__ void count_sort_step2_cuda_kernel(
  int *count_sort_table,  // [batch_size, num_hash_f, hashtable_capacity]，计数排序表的整数指针
  int batch_size,         // 批处理大小，整数参数
  int num_hash_f,         // 哈希函数数量，整数参数
  int hashtable_capacity  // 哈希表容量，整数参数
);
__global__ void count_sort_step3_cuda_kernel(
  int *key_mask,          // 输入：表示批次中每个关键字的掩码数组 [batch_size, num_key]
  int *key_hash_code,     // 输入：表示批次中每个关键字的哈希码数组 [batch_size, num_key, num_hash_f]
  int *count_sort_table,  // 输入/输出：计数排序表格，用于存储排序后的关键字索引 [batch_size, num_hash_f, hashtable_capacity]
  int *key_sorted_idxes,  // 输出：存储排序后的关键字索引 [batch_size, num_hash_f, num_key]
  int batch_size,         // 输入：批次大小
  int num_hash_f,         // 输入：哈希函数数量
  int hashtable_capacity, // 输入：哈希表容量
  int num_key             // 输入：每个批次中的关键字数量
);

__global__ void extract_query_info_cuda_kernel(
  int *query_mask,       // 输入：表示批次中每个查询的掩码数组 [batch_size, num_query]
  int *query_hash_code,  // 输入：表示批次中每个查询的哈希码数组 [batch_size, num_query, num_hash_f]
  int *count_sort_table, // 输入：计数排序表格，用于存储排序后的关键字索引 [batch_size, num_hash_f, hashtable_capacity]
  int *query_info,       // 输出：存储查询信息，包括关键字索引和哈希函数索引 [batch_size, num_query, 2, num_hash_f]
  int batch_size,        // 输入：批次大小
  int num_hash_f,        // 输入：哈希函数数量
  int hashtable_capacity,// 输入：哈希表容量
  int num_query          // 输入：每个批次中的查询数量
);

__global__ void lsh_weighted_cumulation_ver2_step2_cuda_kernel(
  int *query_mask,         // 输入：表示批次中每个查询的掩码数组 [batch_size, num_query]
  int *query_info,         // 输入：存储查询信息，包括关键字索引和哈希函数索引 [batch_size, num_query, 2, num_hash_f]
  int *key_sorted_idxes,   // 输入：存储排序后的关键字索引 [batch_size, num_hash_f, num_key]
  float *query_weight,     // 输入：查询的权重数组 [batch_size, num_query, weight_dim]
  float *key_weight,       // 输入：关键字的权重数组 [batch_size, num_key, weight_dim]
  float *value,            // 输入：关键字对应的值数组 [batch_size, num_key, value_dim]
  float *cumulation_value, // 输出：累积后的值数组 [batch_size, num_query, value_dim]
  int batch_size,          // 输入：批次大小
  int num_hash_f,          // 输入：哈希函数数量
  int num_query,           // 输入：每个批次中的查询数量
  int num_key,             // 输入：每个批次中的关键字数量
  int value_dim,           // 输入：值的维度
  int weight_dim           // 输入：权重的维度
);

__global__ void lsh_weighted_cumulation_ver3_step2_cuda_kernel(
  int *query_sorted_idxes,   // 输入：存储排序后的查询索引 [batch_size, num_hash_f, num_query]
  int *key_mask,             // 输入：表示批次中每个关键字的掩码数组 [batch_size, num_key]
  int *key_info,             // 输入：关键字的信息数组，包括索引和哈希函数索引 [batch_size, num_key, 2, num_hash_f]
  float *query_weight,       // 输入：查询的权重数组 [batch_size, num_query, weight_dim]
  float *key_weight,         // 输入：关键字的权重数组 [batch_size, num_key, weight_dim]
  float *value,              // 输入：关键字对应的值数组 [batch_size, num_key, value_dim]
  float *cumulation_value,   // 输出：累积后的值数组 [batch_size, num_query, value_dim]
  int batch_size,            // 输入：批次大小
  int num_hash_f,            // 输入：哈希函数数量
  int num_query,             // 输入：每个批次中的查询数量
  int num_key,               // 输入：每个批次中的关键字数量
  int value_dim,             // 输入：值的维度
  int weight_dim             // 输入：权重的维度
);

__global__ void lsh_weighted_cumulation_ver4_step2_cuda_kernel(
  int *query_sorted_idxes,   // 输入：存储排序后的查询索引 [batch_size, num_hash_f, num_query]
  int *key_mask,             // 输入：表示批次中每个关键字的掩码数组 [batch_size, num_key]
  int *key_info,             // 输入：关键字的信息数组，包括索引和哈希函数索引 [batch_size, num_key, 2, num_hash_f]
  float *query_weight,       // 输入：查询的权重数组 [batch_size, num_query, weight_dim]
  float *key_weight,         // 输入：关键字的权重数组 [batch_size, num_key, weight_dim]
  float *value,              // 输入：关键字对应的值数组 [batch_size, num_key, value_dim]
  float *cumulation_value,   // 输出：累积后的值数组 [batch_size, num_query, value_dim]
  int batch_size,            // 输入：批次大小
  int num_hash_f,            // 输入：哈希函数数量
  int num_query,             // 输入：每个批次中的查询数量
  int num_key,               // 输入：每个批次中的关键字数量
  int value_dim,             // 输入：值的维度
  int weight_dim             // 输入：权重的维度
);
```