# `.\pytorch\torch\csrc\api\src\nn\options\linear.cpp`

```py
#include <torch/nn/options/linear.h>

namespace torch {
namespace nn {

// LinearOptions 类的构造函数，接受输入特征数和输出特征数作为参数
LinearOptions::LinearOptions(int64_t in_features, int64_t out_features)
    : in_features_(in_features), out_features_(out_features) {}

// BilinearOptions 类的构造函数，接受两个输入特征数和输出特征数作为参数
BilinearOptions::BilinearOptions(
    int64_t in1_features,
    int64_t in2_features,
    int64_t out_features)
    : in1_features_(in1_features),
      in2_features_(in2_features),
      out_features_(out_features) {}

// UnflattenOptions 类的构造函数，接受维度和大小向量作为参数
UnflattenOptions::UnflattenOptions(int64_t dim, std::vector<int64_t> sizes)
    : dim_(dim), sizes_(std::move(sizes)) {}

// UnflattenOptions 类的构造函数，接受维度名称和命名形状映射作为参数
UnflattenOptions::UnflattenOptions(const char* dimname, namedshape_t namedshape)
    : dim_(0),
      dimname_(std::string(dimname)),
      namedshape_(std::move(namedshape)) {}

// UnflattenOptions 类的构造函数，接受维度名称和命名形状映射作为参数
UnflattenOptions::UnflattenOptions(std::string dimname, namedshape_t namedshape)
    : dim_(0),
      dimname_(std::move(dimname)),
      namedshape_(std::move(namedshape)) {}

} // namespace nn
} // namespace torch
```