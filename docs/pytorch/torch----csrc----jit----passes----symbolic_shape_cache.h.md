# `.\pytorch\torch\csrc\jit\passes\symbolic_shape_cache.h`

```py
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>

namespace torch {
namespace jit {

// 结构体 CanonicalizedSymbolicShape 是对符号形状进行规范化的表示
struct TORCH_API CanonicalizedSymbolicShape {
  // 构造函数，接受原始符号形状和映射表 ss_map
  CanonicalizedSymbolicShape(
      const c10::SymbolicShape& orig_shape,
      std::unordered_map<int64_t, int64_t>& ss_map) {
    // 调用 init 函数进行初始化
    init(orig_shape, ss_map);
  }

  // 构造函数，接受原始符号形状 orig_shape，创建新的空映射表 new_ssmap
  CanonicalizedSymbolicShape(c10::SymbolicShape& orig_shape) {
    std::unordered_map<int64_t, int64_t> new_ssmap;
    // 调用 init 函数进行初始化
    init(orig_shape, new_ssmap);
  }

  // 返回当前对象的哈希值
  size_t hash() const;

  // 将规范化的符号形状转换为符号形状对象，接受逆映射表 inverse_ss_map
  c10::SymbolicShape toSymbolicShape(
      std::unordered_map<int64_t, int64_t>& inverse_ss_map) const;

  // 友元函数，用于比较两个 CanonicalizedSymbolicShape 对象是否相等
  TORCH_API friend bool operator==(
      const CanonicalizedSymbolicShape& a,
      const CanonicalizedSymbolicShape& b);

 private:
  // 可选的值向量，用于存储形状的值
  std::optional<std::vector<int64_t>> values_;

  // 初始化函数，接受原始符号形状 orig_shape 和映射表 ss_map
  void init(
      const c10::SymbolicShape& orig_shape,
      std::unordered_map<int64_t, int64_t>& ss_map);
};

// 形状缓存 API

// 获取缓存的形状函数，接受函数模式 schema 和 SSAInput 参数向量 arg_vec
TORCH_API std::optional<std::vector<at::SymbolicShape>>
get_cached_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec);

// 缓存形状函数，接受函数模式 schema、输入参数 SSAInput 向量 arg_vec 和返回形状向量 ret_vec
TORCH_API void cache_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    const std::vector<at::SymbolicShape>& ret_vec);

// 用于测试代码的函数，清除形状缓存
TORCH_API void clear_shape_cache();

// 获取形状缓存大小的函数
TORCH_API size_t get_shape_cache_size();

} // namespace jit
} // namespace torch
```