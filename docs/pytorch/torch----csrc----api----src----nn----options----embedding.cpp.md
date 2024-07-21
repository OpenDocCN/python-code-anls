# `.\pytorch\torch\csrc\api\src\nn\options\embedding.cpp`

```py
#include <torch/nn/options/embedding.h>  // 包含了 torch 库中的 embedding 相关选项头文件

namespace torch {  // 进入 torch 命名空间

namespace nn {  // 进入 nn 命名空间

// EmbeddingOptions 类的构造函数，接受两个参数：num_embeddings 和 embedding_dim
EmbeddingOptions::EmbeddingOptions(
    int64_t num_embeddings,
    int64_t embedding_dim)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {}  // 初始化成员变量 num_embeddings_ 和 embedding_dim_

// EmbeddingBagOptions 类的构造函数，接受两个参数：num_embeddings 和 embedding_dim
EmbeddingBagOptions::EmbeddingBagOptions(
    int64_t num_embeddings,
    int64_t embedding_dim)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {}  // 初始化成员变量 num_embeddings_ 和 embedding_dim_

} // namespace nn

} // namespace torch
```