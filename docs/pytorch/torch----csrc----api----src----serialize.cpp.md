# `.\pytorch\torch\csrc\api\src\serialize.cpp`

```py
# 包含 Torch 序列化模块中的 pickle 相关头文件
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/serialize.h>

# 包含标准库中的 vector 头文件
#include <vector>

# 定义 torch 命名空间
namespace torch {

# 定义函数 pickle_save，接受一个 at::IValue 类型的参数 ivalue，并返回一个包含序列化数据的字符向量
std::vector<char> pickle_save(const at::IValue& ivalue) {
  # 调用 jit 命名空间中的 pickle_save 函数，将传入的 ivalue 序列化并返回序列化后的数据
  return jit::pickle_save(ivalue);
}

# 定义函数 pickle_load，接受一个 const std::vector<char>& 类型的参数 data，并返回一个 torch::IValue 类型的对象
torch::IValue pickle_load(const std::vector<char>& data) {
  # 调用 jit 命名空间中的 pickle_load 函数，从传入的 data 中反序列化数据并返回反序列化后的 torch::IValue 对象
  return jit::pickle_load(data);
}

} // namespace torch
```