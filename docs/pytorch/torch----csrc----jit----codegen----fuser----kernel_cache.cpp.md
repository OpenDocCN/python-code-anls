# `.\pytorch\torch\csrc\jit\codegen\fuser\kernel_cache.cpp`

```py
namespace torch {
namespace jit {
namespace fuser {

struct KernelCacheImpl {
  // 互斥量，用于保护数据结构，确保线程安全
  std::mutex mutex_;
  // 记录已存储的内核数量
  int64_t kernel_counter{0};

  // 映射：融合键到内核规范的映射
  std::unordered_map<int64_t, KernelSpec> specMap_;

  // 映射：漂亮打印的图形字符串到融合键的映射
  // 用于检查图形是否已经缓存于specMap_
  std::unordered_map<std::string, int64_t> graphToKey_;
};

// 获取内核缓存的实例
static KernelCacheImpl& getKernelCache() {
  // 使用静态局部变量确保单例模式，只初始化一次
  static KernelCacheImpl cache;
  return cache;
}

// 调试函数：返回缓存中已存储的内核规范数量
int64_t debugNumCachedKernelSpecs() {
  auto& cache = getKernelCache();
  // 加锁保护缓存访问
  std::lock_guard<std::mutex> guard{cache.mutex_};
  return cache.specMap_.size();
}

// 函数：对图形进行规范化以便缓存
std::shared_ptr<Graph> normalizeGraphForCache(
    const std::shared_ptr<Graph>& graph) {
  // 规范化图形并擦除形状信息
  auto result = Canonicalize(graph, /*keep_unique_names=*/false);
  EraseShapeInformation(result);
  return result;
}

// TODO: 根据历史字符串键查找，然后适当发出键以便将来更快查找
// 前提条件：图形已通过normalizeGraphForCache规范化
int64_t store(std::shared_ptr<Graph> graph) {
  auto& cache = getKernelCache();
  // 获取图形的字符串表示
  std::string repr = graph->toString(false);

  // 加锁保护缓存访问
  std::lock_guard<std::mutex> guard{cache.mutex_};
  // 分配新的键并将图形存储到缓存中
  const auto key = cache.kernel_counter++;
  cache.specMap_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(key),
      std::forward_as_tuple(key, graph));
  // 记录图形字符串到键的映射
  cache.graphToKey_.emplace(std::move(repr), key);
  return key;
}

// XXX: 不获取互斥量
static at::optional<KernelSpec*> nolock_retrieve(
    KernelCacheImpl& cache,
    const int64_t key) {
  // 在不加锁情况下从缓存中检索内核规范指针
  auto it = cache.specMap_.find(key);
  if (it == cache.specMap_.end())
    return at::nullopt;
  return &(it->second);
}

// 函数：获取指定键的内核规范指针
at::optional<KernelSpec*> retrieve(const int64_t key) {
  auto& cache = getKernelCache();
  // 加锁保护缓存访问
  std::lock_guard<std::mutex> guard{cache.mutex_};
  return nolock_retrieve(cache, key);
}

// 前提条件：图形已通过normalizeGraphForCache规范化
// 函数：查找具有指定图形的内核规范指针
at::optional<KernelSpec*> lookupGraph(std::shared_ptr<Graph> graph) {
  auto& cache = getKernelCache();
  // 获取图形的字符串表示
  std::string repr = graph->toString(false);

  // 加锁保护缓存访问
  std::lock_guard<std::mutex> guard{cache.mutex_};
  // 查找图形字符串对应的键，然后获取其内核规范指针
  auto it = cache.graphToKey_.find(repr);
  if (it == cache.graphToKey_.end())
    return at::nullopt;
  return nolock_retrieve(cache, it->second);
}

} // namespace fuser
} // namespace jit
} // namespace torch
```