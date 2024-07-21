# `.\pytorch\torch\csrc\jit\tensorexpr\bounds_inference.cpp`

```
// 引入 Torch Tensor Expression 库中的头文件
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>

// 引入 Torch Tensor Expression 库中的其他头文件
#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

// 引入 C++ 标准库中的头文件
#include <c10/util/irange.h>
#include <iostream>

// Torch Tensor Expression 命名空间
namespace torch::jit::tensorexpr {

// 使用 analysis 命名空间
using namespace analysis;

// 模板函数：合并 tensor 访问信息
template <typename Container>
BoundsInfo mergeTensorAccesses(
    const Container& accesses,
    const std::unordered_map<VarPtr, BufPtr>& varToBuf,
    bool distinctAccessKinds) {
  // 初始化返回的 BoundsInfo 对象
  BoundsInfo ret;

  // 遍历每个访问对象
  for (auto& access : accesses) {
    // 如果访问类型为输入或输出，则跳过
    if (access->type() == AccessType::Input ||
        access->type() == AccessType::Output) {
      continue;
    }

    // 查找变量对应的缓冲区指针，并断言找到
    auto vtbIt = varToBuf.find(access->var());
    TORCH_INTERNAL_ASSERT(vtbIt != varToBuf.end(), buildErrorMessage());
    BufPtr buf = vtbIt->second;

    // 获取该缓冲区的访问边界信息列表
    std::vector<TensorAccessBoundsInfo>& infos = ret[buf];

    bool added = false;

    // 遍历边界信息列表
    for (auto& TABI : infos) {
      // 确定当前访问的类型（加载或存储）
      TensorAccessKind kind = access->isWrite() ? kStore : kLoad;

      // 如果允许不同访问类型或者访问类型相同，则更新边界信息
      if (!distinctAccessKinds || kind == TABI.kind) {
        // 断言开始和结束边界信息的维度与访问对象相同
        TORCH_INTERNAL_ASSERT(
            TABI.start.size() == access->bounds().size(), buildErrorMessage());
        TORCH_INTERNAL_ASSERT(
            TABI.stop.size() == access->bounds().size(), buildErrorMessage());

        // 对每一维进行简化并更新边界信息
        for (size_t i = 0; i < TABI.start.size(); ++i) {
          TABI.start[i] = IRSimplifier::simplify(
              alloc<Min>(TABI.start[i], access->bounds()[i].start, true));
          TABI.stop[i] = IRSimplifier::simplify(
              alloc<Max>(TABI.stop[i], access->bounds()[i].end, true));
          added = true;

          // 如果访问类型不同，则将边界信息类型设置为 mutate
          if (kind != TABI.kind) {
            TABI.kind = kMutate;
          }
        }
      }
    }

    // 如果未添加新的边界信息，则创建新的边界信息对象并加入列表
    if (!added) {
      TensorAccessBoundsInfo info;
      info.kind = access->isWrite() ? kStore : kLoad;

      for (auto& b : access->bounds()) {
        info.start.push_back(b.start);
        info.stop.push_back(b.end);
      }

      infos.push_back(info);
    }
  }

  // 返回合并后的 BoundsInfo 对象
  return ret;
}

// 静态函数：获取语句中的所有缓冲区
static std::unordered_map<VarPtr, BufPtr> getAllBufs(StmtPtr s) {
  // 初始化变量到缓冲区的映射表
  std::unordered_map<VarPtr, BufPtr> varToBuf;

  // 查找语句中的所有缓冲区并填充映射表
  auto bufs = NodeFinder<Buf>::find(s);
  for (const auto& b : bufs) {
    varToBuf[b->base_handle()] = b;
  }

  // 返回填充后的映射表
  return varToBuf;
}

// 静态函数：获取表达式中的所有缓冲区
static std::unordered_map<VarPtr, BufPtr> getAllBufs(ExprPtr e) {
  // 初始化变量到缓冲区的映射表
  std::unordered_map<VarPtr, BufPtr> varToBuf;

  // 查找表达式中的所有缓冲区并填充映射表
  auto bufs = NodeFinder<Buf>::find(e);
  for (const auto& b : bufs) {
    varToBuf[b->base_handle()] = b;
  }

  // 返回填充后的映射表
  return varToBuf;
}

// 推断语句中的访问边界信息
BoundsInfo inferBounds(StmtPtr s, bool distinctAccessKinds) {
  // 获取语句中的所有缓冲区
  auto varToBuf = getAllBufs(s);

  // 创建内存依赖检查器对象并访问语句
  MemDependencyChecker checker;
  s->accept(&checker);

  // 返回合并后的 tensor 访问边界信息
  return mergeTensorAccesses(
      checker.getHistory(), varToBuf, distinctAccessKinds);
}
// 返回基于语句 `s` 的推断边界信息，合并张量访问，可能考虑不同的访问类型
BoundsInfo getInferredBounds(
    MemDependencyChecker& analyzer, // 内存依赖分析器的引用
    StmtPtr s,                      // 语句指针
    bool distinctAccessKinds) {     // 指示是否考虑不同的访问类型
  return mergeTensorAccesses(
      analyzer.accessesWithin(s),  // 获取在语句 `s` 内部的访问信息
      getAllBufs(s),               // 获取语句 `s` 中所有缓冲区的信息
      distinctAccessKinds);        // 是否考虑不同的访问类型进行合并
}

// 返回基于表达式 `e` 的推断边界信息，合并张量访问，可能考虑不同的访问类型
BoundsInfo getInferredBounds(
    MemDependencyChecker& analyzer, // 内存依赖分析器的引用
    ExprPtr e,                      // 表达式指针
    bool distinctAccessKinds) {     // 指示是否考虑不同的访问类型
  return mergeTensorAccesses(
      analyzer.accessesWithin(e),  // 获取在表达式 `e` 内部的访问信息
      getAllBufs(e),               // 获取表达式 `e` 中所有缓冲区的信息
      distinctAccessKinds);        // 是否考虑不同的访问类型进行合并
}

// 打印边界信息到标准错误流
void printBoundsInfo(const BoundsInfo& v) {
  std::cerr << "Access vector {\n";  // 输出访问向量的起始标记
  for (auto& pair : v) {
    std::cerr << *pair.first << " in [";  // 输出缓冲区名称及其访问信息的起始标记
    bool first = true;
    for (auto& b : pair.second) {
      if (!first) {
        std::cerr << ", ";  // 输出分隔符
      }
      std::cerr << ((b.kind == kLoad) ? "LOAD" : "STORE") << "(";  // 输出访问类型
      int i = 0;
      if (b.start.empty()) {
        std::cerr << "0";  // 如果起始位置为空，输出默认值
      }
      for (auto& s : b.start) {
        if (i != 0) {
          std::cerr << ", ";  // 输出起始位置之间的分隔符
        }
        std::cerr << *s;  // 输出起始位置表达式
        i++;
      }
      std::cerr << "; ";
      i = 0;
      if (b.stop.empty()) {
        std::cerr << "0";  // 如果结束位置为空，输出默认值
      }
      for (auto& s : b.stop) {
        if (i != 0) {
          std::cerr << ", ";  // 输出结束位置之间的分隔符
        }
        std::cerr << *s;  // 输出结束位置表达式
        i++;
      }
      std::cerr << ")";  // 输出访问信息的结束标记
      first = false;
    }
    std::cerr << "]\n";  // 输出缓冲区的访问信息的结束标记
  }
  std::cerr << "}\n";  // 输出访问向量的结束标记
}

// 获取边界信息的尺寸范围
std::vector<ExprPtr> getBoundExtents(
    const std::vector<TensorAccessBoundsInfo>& infos) {
  std::vector<ExprPtr> starts;  // 起始位置表达式向量
  std::vector<ExprPtr> stops;   // 结束位置表达式向量

  // 查找临时缓冲区的安全大小，确定所有边界的外部范围
  for (const TensorAccessBoundsInfo& p : infos) {
    for (const auto i : c10::irange(p.start.size())) {
      if (starts.size() <= i) {
        starts.push_back(p.start[i]);  // 添加起始位置表达式
      } else {
        starts[i] =
            IRSimplifier::simplify(alloc<Min>(starts[i], p.start[i], true));  // 简化并更新起始位置表达式
      }

      if (stops.size() <= i) {
        stops.push_back(p.stop[i]);  // 添加结束位置表达式
      } else {
        stops[i] =
            IRSimplifier::simplify(alloc<Max>(stops[i], p.stop[i], true));  // 简化并更新结束位置表达式
      }
    }
  }

  std::vector<ExprPtr> extents;  // 尺寸范围表达式向量
  for (size_t i = 0; i < starts.size(); ++i) {
    ExprPtr dim = IRSimplifier::simplify(
        alloc<Add>(alloc<Sub>(stops[i], starts[i]), immLike(stops[i], 1)));  // 计算尺寸范围表达式
    extents.push_back(dim);  // 添加尺寸范围表达式到向量
  }

  return extents;  // 返回尺寸范围表达式向量
}

// 将边界信息转换为无序集合的静态函数
using BoundSet = std::unordered_set<Bound, BoundHash>;

static BoundSet convertBounds(
    const std::vector<TensorAccessBoundsInfo>& bounds,
    TensorAccessKind filter = kMutate) {
  BoundSet ret;  // 结果边界集合
  for (auto& TABI : bounds) {
    if (filter == kMutate || TABI.kind == filter) {
      for (size_t i = 0; i < TABI.start.size(); ++i) {
        ret.insert(Bound(TABI.start[i], TABI.stop[i]));  // 插入边界信息到集合中
      }
    }
  }
  return ret;  // 返回结果边界集合
}

// 将边界信息转换为无序集合的静态函数，针对特定缓冲区和访问类型进行过滤
static BoundSet convertBounds(
    BoundsInfo& bounds,
    BufPtr buf,
    TensorAccessKind filter = kMutate) {
  auto it = bounds.find(buf);  // 在边界信息中查找特定缓冲区
  if (it == bounds.end()) {
    return BoundSet();  // 如果未找到，返回空的边界集合
  }

  return convertBounds(it->second, filter);  // 调用上述函数进行边界信息转换
}
}

// 根据内存依赖分析器、两个语句 A 和 B 获取潜在的危险类型
HazardKind getPotentialHazards(
    MemDependencyChecker& analyzer,
    StmtPtr A,
    StmtPtr B) {
  // 获取语句 A 的推断边界信息
  BoundsInfo aBounds = getInferredBounds(analyzer, A, true);
  // 获取语句 B 的推断边界信息
  BoundsInfo bBounds = getInferredBounds(analyzer, B, true);

  // 遍历语句 B 的边界信息
  for (auto& pair : bBounds) {
    // 获取缓冲区指针
    BufPtr buf = pair.first;
    // 如果语句 A 中不存在该缓冲区，则跳过
    if (aBounds.find(buf) == aBounds.end()) {
      continue;
    }

    // 转换语句 A 中该缓冲区的写操作边界信息
    auto aWrites = convertBounds(aBounds, buf, kStore);
    // 转换语句 A 中该缓冲区的读操作边界信息
    auto aReads = convertBounds(aBounds, buf, kLoad);

    // 转换语句 B 中该缓冲区的写操作边界信息
    auto bWrites = convertBounds(pair.second, kStore);
    // 转换语句 B 中该缓冲区的读操作边界信息
    auto bReads = convertBounds(pair.second, kLoad);

    // 首先检查 RAW 依赖
    for (auto& bR : bReads) {
      for (auto& aW : aWrites) {
        // 如果读写边界存在重叠，则返回读后写危险类型
        if (boundOverlap(bR, aW) != OverlapKind::NoOverlap) {
          return HazardKind::ReadAfterWrite;
        }
      }
    }

    // 然后检查 WAR 依赖
    for (auto& bW : bWrites) {
      for (auto& aR : aReads) {
        // 如果写读边界存在重叠，则返回写后读危险类型
        if (boundOverlap(bW, aR) != OverlapKind::NoOverlap) {
          return HazardKind::WriteAfterRead;
        }
      }
    }

    // 最后检查 WAW 依赖
    for (auto& bW : bWrites) {
      for (auto& aW : aWrites) {
        // 如果写写边界存在重叠，则返回写后写危险类型
        if (boundOverlap(bW, aW) != OverlapKind::NoOverlap) {
          return HazardKind::WriteAfterWrite;
        }
      }
    }
  }

  // 如果没有依赖冲突，则返回无依赖危险类型
  return HazardKind::NoDependency;
}

// 获取张量访问边界信息的索引边界
static IndexBounds getIndexBounds(const TensorAccessBoundsInfo& tabi) {
  // 断言开始和停止的大小相同，如果不是则抛出内部错误
  TORCH_INTERNAL_ASSERT(
      tabi.start.size() == tabi.stop.size(), buildErrorMessage());
  IndexBounds ret(tabi.start.size());
  // 如果开始索引为空，则直接返回边界信息
  if (tabi.start.empty()) {
    return ret;
  }
  // 遍历开始和停止索引，构建边界信息
  for (size_t i = 0; i < tabi.start.size(); ++i) {
    ret[i] = Bound(tabi.start[i], tabi.stop[i]);
  }
  return ret;
}

// 获取多个张量访问边界信息的索引边界
static std::vector<IndexBounds> getIndexBounds(
    const std::vector<TensorAccessBoundsInfo>& vTABI,
    TensorAccessKind filter = kMutate) {
  std::vector<IndexBounds> bounds;
  // 遍历所有张量访问边界信息
  for (auto& TABI : vTABI) {
    // 如果过滤器为 kMutate 或者与过滤器匹配，则获取索引边界并添加到列表中
    if (filter == kMutate || TABI.kind == filter) {
      bounds.push_back(getIndexBounds(TABI));
    }
  }
  return bounds;
}

// 检查两个边界信息是否存在冲突重叠
static bool hasConflictingOverlap(
    const BoundsInfo& aBounds,
    const BoundsInfo& bBounds,
    TensorAccessKind aFilter = kMutate,
    TensorAccessKind bFilter = kMutate) {
  // 定义索引边界信息的哈希映射
  using IndexBoundsInfo = std::unordered_map<BufPtr, std::vector<IndexBounds>>;
  // 创建语句 A 的索引边界信息哈希映射
  IndexBoundsInfo aIndexBoundsInfo;
  for (auto& aBound : aBounds) {
    // 将语句 A 的每个缓冲区的索引边界信息添加到哈希映射中
    aIndexBoundsInfo[aBound.first] = getIndexBounds(aBound.second, aFilter);
  }
  // 创建语句 B 的索引边界信息哈希映射
  IndexBoundsInfo bIndexBoundsInfo;
  for (auto& bBound : bBounds) {
    // 将语句 B 的每个缓冲区的索引边界信息添加到哈希映射中
    bIndexBoundsInfo[bBound.first] = getIndexBounds(bBound.second, bFilter);
  }

  // 遍历语句 A 的边界信息
  for (auto& aBound : aBounds) {
    // 查找语句 B 是否存在相同的缓冲区
    auto bIt = bBounds.find(aBound.first);
    if (bIt == bBounds.end()) {
      continue;
    }
    // 获取语句 A 和语句 B 的索引边界信息
    auto aIndexBounds = aIndexBoundsInfo[aBound.first];
    auto bIndexBounds = bIndexBoundsInfo[bIt->first];
    auto aTABIs = aBound.second;
    auto bTABIs = bIt->second;
    # 外层循环遍历 aTABIs 容器中的元素，size_t 是 C++ 中表示无符号整数的类型
    for (size_t i = 0; i < aTABIs.size(); ++i) {
      # 内层循环遍历 bTABIs 容器中的元素，size_t 是 C++ 中表示无符号整数的类型
      for (size_t j = 0; j < bTABIs.size(); ++j) {
        # 获取当前外层循环中的 aTABI 元素
        auto aTABI = aTABIs[i];
        # 获取当前内层循环中的 bTABI 元素
        auto bTABI = bTABIs[j];
        # 如果 aTABI 和 bTABI 的 kind 属性都为 kLoad，则跳过当前循环，继续下一次迭代
        if (aTABI.kind == kLoad && bTABI.kind == kLoad) {
          continue;
        }
        # 调用 overlaps 函数，判断 aIndexBounds[i] 和 bIndexBounds[j] 是否有重叠
        auto overlap = overlaps(aIndexBounds[i], bIndexBounds[j]);
        # 如果存在重叠，则返回 true，表示找到重叠
        if (overlap != OverlapKind::NoOverlap) {
          return true;
        }
      }
    }
  }
  # 如果所有循环结束仍未找到重叠，返回 false，表示没有重叠
  return false;
}

bool hasConflictingOverlap(
    analysis::MemDependencyChecker& analyzer,
    StmtPtr A,
    StmtPtr B) {
  // 获取语句 A 的推断边界信息
  BoundsInfo aBounds = getInferredBounds(analyzer, A, true);
  // 获取语句 B 的推断边界信息
  BoundsInfo bBounds = getInferredBounds(analyzer, B, true);
  // 检查语句 A 和语句 B 是否有冲突的重叠
  return hasConflictingOverlap(aBounds, bBounds);
}

bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    StorePtr S1,
    StorePtr S2) {
  // 获取存储指针 S1 的推断边界信息
  BoundsInfo s1Bounds = getInferredBounds(analyzer, S1, true);
  // 获取存储指针 S2 的推断边界信息
  BoundsInfo s2Bounds = getInferredBounds(analyzer, S2, true);
  // 检查存储指针 S1 和 S2 是否有冲突的重叠，用于存储和存储之间的情况
  return hasConflictingOverlap(s1Bounds, s2Bounds, kStore, kStore);
}

bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    StorePtr S,
    LoadPtr L) {
  // 获取存储指针 S 的推断边界信息
  BoundsInfo sBounds = getInferredBounds(analyzer, S, true);
  // 获取加载指针 L 的推断边界信息
  BoundsInfo lBounds = getInferredBounds(analyzer, L, true);
  // 检查存储指针 S 和加载指针 L 是否有冲突的重叠，用于存储和加载之间的情况
  return hasConflictingOverlap(sBounds, lBounds, kStore, kLoad);
}

} // namespace torch::jit::tensorexpr
```