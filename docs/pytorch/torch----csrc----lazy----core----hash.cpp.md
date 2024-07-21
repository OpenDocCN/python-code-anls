# `.\pytorch\torch\csrc\lazy\core\hash.cpp`

```py
/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/util.h
 */
#include <iomanip>      // 包含格式化输出的标准库
#include <sstream>      // 包含字符串流的标准库

#include <torch/csrc/lazy/core/hash.h>   // 引入相关的头文件

namespace torch {
namespace lazy {
namespace {

/**
 * 从给定的数据块中加载哈希值
 * @param data 指向数据的指针的指针
 * @param top 数据块的末尾指针
 * @return 加载的哈希值
 */
hash_t LoadHash(const uint8_t** data, const uint8_t* top) {
  std::ptrdiff_t size = top - (*data);
  if (size >= (int)sizeof(hash_t)) {    // 如果数据块大小足够大以容纳一个哈希值
    hash_t v;
    std::memcpy(&v, *data, sizeof(v));  // 将数据块中的内容复制到哈希值变量中
    *data += sizeof(hash_t);            // 移动数据指针
    return v;                           // 返回加载的哈希值
  }
  union {
    hash_t h;
    std::array<uint8_t, sizeof(hash_t)> b;
#ifdef _MSC_VER
    // MSVC (or some versions we use) doesn't support C99 union field init
    // but it initializes the first member of the union.
  } uval = {hash_t(0)};
#else
  } uval = {.h = hash_t(0)};
#endif
  // 使用 memcpy 实现对不支持非对齐访问的平台的兼容性
  std::memcpy(uval.b.data(), *data, size);
  *data += size;
  return uval.h;                        // 返回加载的哈希值
}

} // namespace

/**
 * 计算数据块的哈希值
 * @param data 指向数据块的指针
 * @param n 数据块的大小
 * @param seed 哈希种子值
 * @return 计算得到的哈希值
 */
hash_t HashBlock(const void* data, size_t n, const hash_t& seed) {
  const hash_t m(static_cast<uint64_t>(0xc6a4a7935bd1e995));  // 定义哈希函数的参数
  const int r = 47;

  const uint8_t* u8_data = reinterpret_cast<const uint8_t*>(data);  // 将数据块转换为字节流
  const uint8_t* top = u8_data + n;        // 计算数据块的末尾位置
  hash_t h(seed ^ ((uint64_t)n * m));      // 初始化哈希值
  while (u8_data < top) {
    hash_t k = LoadHash(&u8_data, top);    // 加载当前数据块片段的哈希值
    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }
  h ^= h >> r;
  h *= m;
  h ^= h >> r;
  return h;                               // 返回计算得到的最终哈希值
}

/**
 * 计算数据块的哈希值
 * @param data 指向数据块的指针
 * @param size 数据块的大小
 * @return 计算得到的哈希值
 */
hash_t DataHash(const void* data, size_t size) {
  return HashBlock(
      data, size, hash_t(static_cast<uint64_t>(0xc2b2ae3d27d4eb4f)));  // 调用哈希块函数计算哈希值
}

/**
 * 标准化哈希值的减少操作
 * @param a 哈希值
 * @return 标准化后的哈希值
 */
size_t StdDataHash(const void* data, size_t size) {
  return HashReduce(DataHash(data, size));  // 调用哈希减少函数计算标准化哈希值
}

/**
 * 组合两个哈希值
 * @param a 第一个哈希值
 * @param b 第二个哈希值
 * @return 组合后的哈希值
 */
size_t StdHashCombine(uintmax_t a, uintmax_t b) {
  return a ^
      (b * 0x27d4eb2f165667c5 + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));  // 执行哈希值的组合操作
}

/**
 * 组合两个哈希值
 * @param a 第一个哈希值
 * @param b 第二个哈希值
 * @return 组合后的哈希值
 */
hash_t HashCombine(const hash_t& a, const hash_t& b) {
  static const hash_t kb(101, 0x27d4eb2f165667c5);
  return hash_t(
      a ^ (b * kb + (uint64_t)0x9e3779b97f4a7c15 + (a << 6) + (a >> 2)));  // 执行哈希值的组合操作
}

/**
 * 标准化哈希值的减少操作
 * @param a 哈希值
 * @return 标准化后的哈希值
 */
size_t HashReduce(const hash_t& a) {
  return StdHashCombine(c10::Uint128Low64(a), c10::Uint128High64(a));  // 调用标准化哈希值的减少函数
}

/**
 * 将哈希值转换为十六进制字符串
 * @param a 哈希值
 * @return 哈希值的十六进制字符串表示
 */
std::string HashToString(const hash_t& a) {
  std::stringstream ss;
  ss << std::hex << c10::Uint128High64(a) << std::setfill('0') << std::setw(16)
     << Uint128Low64(a);  // 将哈希值转换为十六进制字符串
  return ss.str();        // 返回转换后的字符串
}

/**
 * 计算给定布尔向量的哈希值
 * @param values 布尔向量
 * @return 计算得到的哈希值
 */
hash_t Hash(const std::vector<bool>& values) {
  // 由于 vector<bool> 可以被优化为 vector<bit>，因此此处不能假设数据块的大小和指针的方法
  hash_t h(static_cast<uint64_t>(0xad2ed1983bbf2e28));  // 初始化哈希值
  static const hash_t h_true(static_cast<uint64_t>(0x74f6b5198daa2b2));  // 真值的哈希常量
  static const hash_t h_false(static_cast<uint64_t>(0xe39f30789cab5382));  // 假值的哈希常量
  for (const auto& b : values) {  // 遍历布尔向量
    if (b) {
      h = HashCombine(h, h_true);  // 如果值为真，则组合真值的哈希值
    } else {
      h = HashCombine(h, h_false);  // 如果值为假，则组合假值的哈希值
    }
  }
  return h;  // 返回计算得到的最终哈希值
}

} // namespace lazy
} // namespace torch
} // 结束 lazy 命名空间
} // 结束 torch 命名空间
```