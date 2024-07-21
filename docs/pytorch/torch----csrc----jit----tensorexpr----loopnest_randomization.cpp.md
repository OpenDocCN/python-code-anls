# `.\pytorch\torch\csrc\jit\tensorexpr\loopnest_randomization.cpp`

```py
// 包含 C++ 标准库的头文件
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// 包含 Torch Tensor Expression 库的特定头文件
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/jit_opt_limit.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/loopnest_randomization.h>

// Torch Tensor Expression 命名空间
namespace torch::jit::tensorexpr {

// Tensor Expression 中随机化辅助函数命名空间
namespace randomization_helper {

// 返回最大的变换次数，受环境变量 PYTORCH_JIT_OPT_LIMIT 控制
static int64_t max_transformations(int n_max_transforms) {
  if (!JIT_OPT_ALLOWED) {
    return n_max_transforms;
  }
  int max_transforms = 1;
  while (JIT_OPT_ALLOWED && max_transforms < n_max_transforms) {
    max_transforms++;
  }
  return max_transforms;
}

// 获取所有完全嵌套的循环嵌套结构
static std::vector<std::vector<ForPtr>> GetAllPerfectlyNestedLoopNests(
    std::vector<ForPtr> loops) {
  std::vector<std::vector<ForPtr>> all_nested_loops;
  std::vector<ForPtr> nested_loops;
  if (loops.empty()) {
    return all_nested_loops;
  }
  nested_loops.push_back(loops[0]);
  for (size_t i = 1; i < loops.size(); i++) {
    auto last_loop = nested_loops.back();
    auto next_loop = loops[i];
    if (last_loop->body()->nstmts() == 1 &&
        last_loop->body()->front() == next_loop) {
      nested_loops.push_back(next_loop);
    } else {
      if (nested_loops.size() > 1) {
        all_nested_loops.push_back(nested_loops);
      }
      nested_loops.clear();
      nested_loops.push_back(next_loop);
    }
  }
  return all_nested_loops;
}

// 随机选择指定数量的对象
template <typename T>
std::tuple<std::vector<T>, std::vector<int>> select_n_randomly(
    std::vector<T>& objects,
    int n,
    std::default_random_engine& random_engine) {
  std::vector<int> indices(objects.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), random_engine);

  std::vector<T> selected_objects;
  std::vector<int> selected_indices;
  if (static_cast<int>(indices.size()) < n) {
    return std::make_tuple(selected_objects, selected_indices);
  }
  for (int i = 0; i < n; i++) {
    int index = indices[i];
    selected_indices.push_back(index);
    selected_objects.push_back(objects[index]);
  }
  return std::make_tuple(selected_objects, selected_indices);
}

// 在循环中查找有效的因子
static int find_factor(ForPtr loop) {
  ExprPtr loop_stop = loop->stop();
  auto loop_imm = intValue(loop_stop);
  if (loop_imm) {
    int loop_bound = *loop_imm;
    int factor = rand() % (loop_bound - 1) + 1;
    return factor;
  }
  return -1;
}

} // namespace randomization_helper
} // namespace torch::jit::tensorexpr
static void printHistory(int index, std::string message) {
  // 构造带有索引的消息字符串，用于调试和记录历史信息
  message = "Random Transform Sequence - Transformations[" +
      std::to_string(index) + "] = " + message;
  // 调试输出消息
  GRAPH_DEBUG(message);
}

template <typename T>
std::string join(std::vector<T> indices, char sep = ',') {
  // 将向量中的元素连接成一个字符串，每个元素用指定的分隔符隔开
  std::string s = "";
  for (const auto& index : indices) {
    s += std::to_string(index) + sep;
  }
  return s;
}

static std::string join(std::vector<std::string> indices, char sep = ',') {
  // 将字符串向量中的元素连接成一个字符串，每个元素用指定的分隔符隔开
  std::string s = "";
  for (const auto& index : indices) {
    s += index + sep;
  }
  return s;
}

template <typename T>
std::string indexOf(const std::vector<T>& objects, const T& object) {
  // 返回对象在向量中的索引位置的字符串表示
  return std::to_string(std::distance(
      objects.begin(), std::find(objects.begin(), objects.end(), object)));
}

} // namespace randomization_helper

void loopnestRandomization(int64_t seed, LoopNest& l) {
  // 用于确定性测试随机化基础设施的辅助函数
  // 当种子值为1时，执行预设的循环变换，以便测试接口
  if (seed == 1) {
    // 简化循环嵌套结构
    l.simplify();
    return;
  }
  
  std::default_random_engine random_engine(seed);
  std::srand(seed);
  // 设置允许的最大变换次数，超过此数可能难以跟踪和调试。选择20作为最大次数。
  int max_allowed_transformations = 20;
  // 随机生成变换次数，取值范围在0到max_allowed_transformations之间
  int n_transforms = randomization_helper::max_transformations(
      std::rand() % max_allowed_transformations);
  // 初始化空消息字符串
  std::string message = "";
  // clang-format off
  // 变换列表：
  //
  //     StmtPtr simplify();
  //     bool computeInline(BufPtr b);
  //     void inlineIntermediateBufs(bool allow_duplicated_work);
  //     bool optimizeConditionals();
  //     static void splitWithTail(ForPtr f, int factor);
  //     static void splitWithMask(ForPtr f, int factor);
  //     static std::vector<ForPtr> distributeLoop(ForPtr loop, const std::unordered_set<StmtPtr>& pivots);
  //     static std::vector<ForPtr> distributeLoop(ForPtr loop);
  //     static std::vector<ForPtr> distributeLoopAndParents(ForPtr loop);
  //     static std::vector<ForPtr> distributeLoopOverInnerLoops(ForPtr loop);
  //     static std::vector<ForPtr> distributeLoopAndParentsOverInnerLoops(ForPtr loop);
  //     static bool fuseLoops(const std::vector<ForPtr>& loops, ForPtr* fused);
  //     static void reorderAxis(ForPtr a, ForPtr b);
  //     static std::vector<ForPtr> reorder(const std::vector<ForPtr>& loops, const std::vector<size_t>& permutation);
  //     ForPtr tile(ForPtr x, ForPtr y, int x_factor, int y_factor);
  //     static void fullUnroll(ForPtr f);
  //     static bool normalize(ForPtr f);
  //     static bool flatten(const std::vector<ForPtr>& f, ForPtr* flattened);
  //     static void compressBuffer(BufPtr buf, StmtPtr stmt);
  //     static void compressAllBuffers(StmtPtr stmt);
  //     static void sliceHead(ForPtr f, int factor, ForPtr* head, ForPtr* tail);
  //     static void sliceHead(ForPtr f, int factor);
  //     static void sliceTail(ForPtr f, int factor, ForPtr* head, ForPtr* tail);
  //     static void sliceTail(ForPtr f, int factor);
  //     static AccessResult cacheAccesses(BufPtr producer, const std::string& name, StmtPtr consumer);
  //     static void computeAt(StmtPtr s, ForPtr at);
  //     static bool rfactor(StmtPtr s, ForPtr outer_reduction_for);
  //     static bool vectorize(ForPtr);
  //     void vectorizeInnerLoops();
  //     void eliminateDeadStores();
  //     void prepareForCodegen();
  // clang-format on
  // 枚举不同的变换类型
  enum TransformKind {
    SIMPLIFY = 0,                   // 简化语句
    COMPUTE_INLINE,                 // 内联计算
    INLINE_ALL,                     // 内联所有中间缓冲区
    OPT_COND,                       // 优化条件语句
    SPLIT_TAIL,                     // 尾部分割
    SPLIT_MASK,                     // 掩码分割
    DIST1,                          // 循环分布类型1
    DIST2,                          // 循环分布类型2
    DIST3,                          // 循环分布类型3
    DIST4,                          // 循环分布类型4
    DIST5,                          // 循环分布类型5
    FUSE_LOOPS,                     // 循环融合
    REORDER_AXIS,                   // 重新排序轴
    REORDER,                        // 重新排序循环
    TILE,                           // 切片
    FULL_UNROLL,                    // 完全展开
    NORMALIZE,                      // 标准化
    FLATTEN,                        // 展平
    COMPRESS_BUFFER,                // 压缩缓冲区
    COMPRESS_ALL_BUFFERS,           // 压缩所有缓冲区
    SLICE_HEAD,                     // 切片头部
    SLICE_TAIL,                     // 切片尾部
    CACHE_ACCESSES,                 // 缓存访问
    COMPUTE_AT,                     // 计算位置
    RFACTOR,                        // 重构因子
    VECTORIZE,                      // 向量化
    VECTORIZE_INNER_LOOPS,          // 向量化内部循环
    ELIMINATE_DEAD_STORES,          // 消除死存储
    MAX_TRANSFORM,
  };


// 定义一个名为 MAX_TRANSFORM 的常量
MAX_TRANSFORM,



  bool can_inline = true;


// 声明一个布尔变量 can_inline，并初始化为 true
bool can_inline = true;



  try {
    }
  } catch (...) {
    std::cout << "EXCEPTION THROWN!\n";
    std::cout << "SEED: " << seed << "\n";
    throw std::runtime_error("Random test failed");
  }


// 尝试执行空的代码块
try {
    // 空代码块，无操作
}
// 捕获任何异常
catch (...) {
    // 输出异常信息到标准输出流
    std::cout << "EXCEPTION THROWN!\n";
    // 输出 seed 变量的值到标准输出流
    std::cout << "SEED: " << seed << "\n";
    // 抛出 std::runtime_error 异常并提供错误消息 "Random test failed"
    throw std::runtime_error("Random test failed");
}



  message = "End of transformations;\n";
  randomization_helper::printHistory(n_transforms, message);
  return;


// 将字符串赋值给 message 变量，表示变换结束
message = "End of transformations;\n";
// 调用 randomization_helper 命名空间中的 printHistory 函数，
// 打印变换历史记录，传递变换数量 n_transforms 和消息 message
randomization_helper::printHistory(n_transforms, message);
// 函数返回，结束执行
return;
}

// 结束命名空间 torch::jit::tensorexpr
} // namespace torch::jit::tensorexpr
```