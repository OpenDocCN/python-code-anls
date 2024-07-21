# `.\pytorch\torch\csrc\jit\tensorexpr\hash_provider.h`

```
#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <utility>

namespace torch {
namespace jit {
namespace tensorexpr {

// 定义结构体 SimplifierHashType，用于表示简化后的哈希值
struct TORCH_API SimplifierHashType {
  SimplifierHashType() = default;
  explicit SimplifierHashType(size_t s) : _h(s) {}

  bool operator==(const SimplifierHashType& other) const;
  bool operator!=(const SimplifierHashType& other) const;
  bool operator<(const SimplifierHashType& other) const;
  bool operator==(const size_t other) const;
  bool operator!=(const size_t other) const;

  size_t _h{0}; // 存储哈希值的成员变量
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {
// 定义哈希函数模板，用于 std::hash<torch::jit::tensorexpr::SimplifierHashType>
template <>
struct hash<torch::jit::tensorexpr::SimplifierHashType> {
  size_t operator()(const torch::jit::tensorexpr::SimplifierHashType& k) const {
    return k._h; // 返回 SimplifierHashType 对象中的哈希值
  }
};

} // namespace std

namespace torch {
namespace jit {
namespace tensorexpr {

// 定义宏 CACHE_GUARD()，用于缓存机制，如果表达式的哈希已经计算过则直接返回
#define CACHE_GUARD()  \
  if (cachedHash(v)) { \
    return;            \
  }

// 前置声明类 Term 和 Polynomial
class Term;
class Polynomial;

/* Expression hasher providing comparable values representing sub-exprs.
 * Uses memoization to avoid excessive recursion. */
// 声明类 HashProvider，继承自 IRVisitor，提供表达式哈希计算服务
class TORCH_API HashProvider : public IRVisitor {
 public:
  // 模板方法 hash，计算给定表达式或语句的哈希值
  template <class T>
  SimplifierHashType hash(T e) {
    e->accept(this); // 调用 IRVisitor 的 accept 方法
    return hashOf(e); // 返回计算得到的哈希值
  }

  // 检查表达式是否已经被缓存过哈希值
  bool cachedHash(ExprPtr e) {
    return exprToHash_.find(e) != exprToHash_.end();
  }
  // 检查语句是否已经被缓存过哈希值
  bool cachedHash(StmtPtr s) {
    return stmtToHash_.find(s) != stmtToHash_.end();
  }

  // 清空表达式和语句的哈希缓存
  void clearCache() {
    exprToHash_.clear();
    stmtToHash_.clear();
  }

  // 重写 IRVisitor 的各种 visit 方法，用于不同类型表达式和语句的处理
  void visit(AddPtr v) override;
  void visit(SubPtr v) override;
  void visit(MulPtr v) override;
  void visit(DivPtr v) override;
  void visit(ModPtr v) override;
  void visit(RoundOffPtr v) override;
  void visit(MaxPtr v) override;
  void visit(MinPtr v) override;
  void visit(AndPtr v) override;
  void visit(OrPtr v) override;
  void visit(XorPtr v) override;
  void visit(LshiftPtr v) override;
  void visit(RshiftPtr v) override;
  void visit(CompareSelectPtr v) override;

// NOLINTNEXTLINE
// 定义宏 IMM_VISIT(Type, Name)，用于访问各种立即数类型的表达式
#define IMM_VISIT(Type, Name)                    \
  void visit(Name##ImmPtr v) override {          \
    CACHE_GUARD();                               \
    putHash(v, hash_combine(#Name, v->value())); \
  }
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_VISIT); // 遍历所有标量类型并应用 IMM_VISIT 宏
// 定义了一系列虚函数 visit，用于访问不同类型的指针对象
void visit(CastPtr v) override;
void visit(VarPtr v) override;
void visit(RampPtr v) override;
void visit(LoadPtr v) override;
void visit(StorePtr v) override;
void visit(BlockPtr v) override;
void visit(ForPtr v) override;
void visit(BroadcastPtr v) override;
void visit(IfThenElsePtr v) override;
void visit(IntrinsicsPtr v) override;
void visit(AllocatePtr v) override;
void visit(FreePtr v) override;
void visit(CondPtr v) override;
void visit(TermPtr v) override;
void visit(PolynomialPtr v) override;
void visit(MaxTermPtr v) override;
void visit(MinTermPtr v) override;

// 模板函数 hash_combine，用于合并多个参数的哈希值
template <typename... Types>
SimplifierHashType hash_combine(const Types&... args) {
  SimplifierHashType seed;
  _hash_combine(seed, args...);
  return seed;
}

private:
// 根据表达式指针 e 计算哈希值
SimplifierHashType hashOf(ExprPtr e) {
  auto it = exprToHash_.find(e);
  if (it != exprToHash_.end()) {
    return it->second;
  }

  // 当查找不到哈希值时，使用 IRPrinter 打印表达式 e 的内容到 stringstream
  std::stringstream ss;
  IRPrinter printer(ss);
  e->accept(&printer);
  SimplifierHashType hash = SimplifierHashType(te_hash(ss.str()));
  // 将计算得到的哈希值存入 exprToHash_ 中
  putHash(std::move(e), hash);

  return hash;
}

// 根据语句指针 s 计算哈希值
SimplifierHashType hashOf(StmtPtr s) {
  auto it = stmtToHash_.find(s);
  if (it != stmtToHash_.end()) {
    return it->second;
  }

  // 当查找不到哈希值时，使用 IRPrinter 打印语句 s 的内容到 stringstream
  std::stringstream ss;
  IRPrinter printer(ss);
  s->accept(&printer);
  SimplifierHashType hash = SimplifierHashType(te_hash(ss.str()));
  // 将计算得到的哈希值存入 stmtToHash_ 中
  putHash(std::move(s), hash);

  return hash;
}

// 不同类型的哈希函数，用于计算不同类型的对象的哈希值
// 具体的哈希计算方式包括将输入值与种子进行异或、位移和加法操作
template <typename T>
void _hash_combine(SimplifierHashType& seed, const T& val) {
  seed._h ^= te_hash(val) + 0x1f752c19 + (seed._h << 7) + (seed._h >> 4);
}

// 特化版本，处理 const char* 类型的哈希计算
void _hash_combine(SimplifierHashType& seed, const char* val) {
  seed._h ^= te_hash(val) + 0x1f752c19 + (seed._h << 7) + (seed._h >> 4);
}

// 特化版本，处理 at::Half 类型的哈希计算
void _hash_combine(SimplifierHashType& seed, const at::Half& val) {
  seed._h ^=
      te_hash((uint16_t)val) + 0x1f752c19 + (seed._h << 7) + (seed._h >> 4);
}

// 特化版本，处理 Dtype 类型的哈希计算
void _hash_combine(SimplifierHashType& seed, const Dtype& val) {
  seed._h ^= te_hash(val.ToCppString()) + 0x1f752c19 + (seed._h << 7) +
      (seed._h >> 4);
}

// 特化版本，处理 ExprPtr 类型的哈希计算
void _hash_combine(SimplifierHashType& seed, ExprPtr e) {
  _hash_combine(seed, hash(std::move(e)));
}

// 模板函数，递归地合并多个参数的哈希值
template <typename T, typename... Types>
void _hash_combine(
    SimplifierHashType& seed,
    const T& val,
    const Types&... args) {
  _hash_combine(seed, val);
  _hash_combine(seed, args...);
}

// 将表达式指针 e 与其哈希值 h 存入 exprToHash_ 中
void putHash(ExprPtr e, SimplifierHashType h) {
  auto res = exprToHash_.emplace(e, h);
  if (res.second == false) {
    // 如果插入失败，抛出运行时错误，表示哈希冲突
    throw std::runtime_error("hash collision");
}
  }
  }
  // 将语句 s 映射到哈希值 h，并存储在 stmtToHash_ 中
  void putHash(StmtPtr s, SimplifierHashType h) {
    // 尝试将 s 和 h 放入 stmtToHash_，如果 s 已存在则抛出运行时错误
    auto res = stmtToHash_.emplace(s, h);
    if (res.second == false) {
      // 如果插入失败，表示存在哈希冲突，抛出异常
      throw std::runtime_error("hash collision");
    }
  }

  // 表达式指针到哈希值的映射
  std::unordered_map<ExprPtr, SimplifierHashType> exprToHash_;
  // 语句指针到哈希值的映射
  std::unordered_map<StmtPtr, SimplifierHashType> stmtToHash_;
  // 管理唯一名称的实例
  UniqueNameManager name_manager_;

  // 计算 SimplifierHashType 类型的哈希值
  size_t te_hash(SimplifierHashType val) {
    return val._h;
  }

  // 计算 int64_t 类型的哈希值
  size_t te_hash(int64_t val) {
    // 添加一些位操作
    size_t h = val ^ 0x647AA4D20C0B;
    size_t h2 = ~h;
    size_t h3 = 0;
    // 反转字节顺序
    for (unsigned int i = 0; i < 64; i += 8) {
      h3 |= ((h2 >> i) & 0xFF) << (64 - i - 8);
    }
    return h3;
  }

  // 计算 int32_t 类型的哈希值
  size_t te_hash(int32_t val) {
    int64_t v2 = val;
    return te_hash(v2);
  }

  // 计算 uint32_t 类型的哈希值
  size_t te_hash(uint32_t val) {
    int64_t v2 = val;
    return te_hash(v2);
  }

  // 计算 uint64_t 类型的哈希值
  size_t te_hash(uint64_t val) {
    int64_t v2 = val;
    return te_hash(v2);
  }

  // 计算 int16_t 类型的哈希值
  size_t te_hash(int16_t val) {
    int64_t v2 = val;
    return te_hash(v2);
  }

  // 计算 std::string 类型的哈希值
  size_t te_hash(std::string val) {
    size_t hash{0};
    int64_t intval{0};
    int64_t s = val.size() - 1;
    while (s >= 0) {
      for (unsigned int i = 0; i < 8; ++i) {
        if (s < 0)
          break;
        int64_t c = val.data()[s];
        intval |= (c << (i * 8));
        s--;
      }
      hash ^= te_hash(intval);
      intval = 0;
    }
    return hash;
  }

  // 计算 double 类型的哈希值
  size_t te_hash(double d) {
    int64_t n;
    // 使用 memcpy 进行类型转换，以便对 double 进行哈希计算
    std::memcpy(&n, &d, sizeof d);
    return te_hash(n);
  }

  // 计算 float 类型的哈希值
  size_t te_hash(float d) {
    int32_t n;
    // 使用 memcpy 进行类型转换，以便对 float 进行哈希计算
    std::memcpy(&n, &d, sizeof d);
    return te_hash(n);
  }

  // 计算 at::Half 类型的哈希值
  size_t te_hash(at::Half d) {
    int16_t n;
    // 使用 memcpy 进行类型转换，以便对 at::Half 进行哈希计算
    std::memcpy(&n, &d, sizeof d);
    return te_hash(n);
  }

  // 计算 at::BFloat16 类型的哈希值
  size_t te_hash(at::BFloat16 d) {
    int16_t n;
    // 使用 memcpy 进行类型转换，以便对 at::BFloat16 进行哈希计算
    std::memcpy(&n, &d, sizeof d);
    return te_hash(n);
  }
};

// 结束 torch 命名空间
} // namespace torch

// 结束 jit 命名空间
} // namespace jit

// 结束 tensorexpr 命名空间
} // namespace tensorexpr
```