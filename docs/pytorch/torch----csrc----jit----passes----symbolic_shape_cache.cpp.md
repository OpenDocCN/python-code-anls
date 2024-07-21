# `.\pytorch\torch\csrc\jit\passes\symbolic_shape_cache.cpp`

```
// 包含 Torch 库的符号形状分析和形状缓存相关头文件
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_cache.h>
#include <torch/csrc/lazy/core/cache.h>

#include <utility>  // 包含用于泛型编程的实用工具

// SHAPE CACHING CODE
namespace torch {
namespace jit {
namespace {

// 定义使用于形状缓存的类型和结构
using CanonicalArg = std::variant<CanonicalizedSymbolicShape, IValue>;
using CanonicalArgVec = std::vector<CanonicalArg>;
using CanonicalRet = std::vector<CanonicalizedSymbolicShape>;
using ShapeCacheKey = std::tuple<c10::OperatorName, CanonicalArgVec>;

// 将 SSAInput 向量规范化为 CanonicalArg 向量
CanonicalArgVec cannonicalizeVec(
    const std::vector<SSAInput>& arg_vec,
    std::unordered_map<int64_t, int64_t>& ss_map,
    bool deep_copy = true) {
  CanonicalArgVec canonical_args;
  canonical_args.reserve(arg_vec.size());
  for (auto& arg : arg_vec) {
    if (const IValue* iv = std::get_if<IValue>(&arg)) {
      if (deep_copy) {
        canonical_args.emplace_back(iv->deepcopy());  // 深拷贝 IValue
      } else {
        canonical_args.emplace_back(*iv);  // 直接拷贝 IValue
      }
    } else {
      auto& ss = std::get<at::SymbolicShape>(arg);
      canonical_args.emplace_back(CanonicalizedSymbolicShape(ss, ss_map));  // 使用 SymbolicShape 构造 CanonicalizedSymbolicShape
    }
  }
  return canonical_args;
}

// 将 SymbolicShape 向量规范化为 CanonicalizedSymbolicShape 向量
std::vector<CanonicalizedSymbolicShape> cannonicalizeVec(
    const std::vector<at::SymbolicShape>& ret_vec,
    std::unordered_map<int64_t, int64_t>& ss_map) {
  std::vector<CanonicalizedSymbolicShape> canonical_rets;
  canonical_rets.reserve(ret_vec.size());
  for (auto& ss : ret_vec) {
    canonical_rets.emplace_back(ss, ss_map);  // 使用 SymbolicShape 构造 CanonicalizedSymbolicShape
  }
  return canonical_rets;
}

// 自定义的哈希函数对象，用于计算 ShapeCacheKey 的哈希值
struct ArgumentsHasher {
  size_t operator()(const ShapeCacheKey& cacheKey) const {
    // TODO: 忽略在形状函数中未使用的参数（初始时不需要）
    auto& op_name = std::get<0>(cacheKey);
    auto& arg_vec = std::get<1>(cacheKey);

    size_t hash_val = c10::hash<c10::OperatorName>()(op_name);  // 哈希操作符名

    hash_val = at::hash_combine(std::hash<size_t>{}(arg_vec.size()), hash_val);  // 结合参数向量大小

    // 遍历参数向量并计算哈希值
    for (const CanonicalArg& arg : arg_vec) {
      size_t cur_arg = 0;
      if (const IValue* ival = std::get_if<IValue>(&arg)) {
        // IValue 不会哈希 List（与 Python 类似），因此需要自定义列表哈希
        if (ival->isList()) {
          TORCH_INTERNAL_ASSERT(ival->isIntList(), "Unexpected Args in List");
          cur_arg = ival->toListRef().size();
          for (const IValue& elem_ival : ival->toListRef()) {
            cur_arg = at::hash_combine(cur_arg, IValue::hash(elem_ival));
          }
        } else {
          cur_arg = IValue::hash(ival);
        }
      } else {
        cur_arg = std::get<CanonicalizedSymbolicShape>(arg).hash();  // 计算 CanonicalizedSymbolicShape 的哈希值
      }
      hash_val = at::hash_combine(hash_val, cur_arg);  // 结合当前参数的哈希值
    }
    return hash_val;
  }
};

// 定义形状缓存类型
using ShapeCache = lazy::Cache<
    ShapeCacheKey,
    std::vector<CanonicalizedSymbolicShape>,
    ArgumentsHasher>;

// 定义形状缓存的大小常量
constexpr size_t kShapeCacheSize = 1024;
ShapeCache shapeCache(kShapeCacheSize);  // 创建形状缓存对象

// 获取形状缓存的键
ShapeCacheKey get_cache_key(
    const FunctionSchema* schema,
    // 接受参数：一个常量引用向量 arg_vec，一个整型到整型的无序映射 ss_map，一个布尔型标志 deep_copy，默认为 true
    const std::vector<SSAInput>& arg_vec,
    // 修改参数：整型到整型的无序映射 ss_map
    std::unordered_map<int64_t, int64_t>& ss_map,
    // 修改参数：深度拷贝标志，默认为 true
    bool deep_copy = true) {
  // 调用函数 cannonicalizeVec，对 arg_vec 和 ss_map 进行规范化处理，得到 CanonicalArgVec 类型的结果 canonical_args
  CanonicalArgVec canonical_args = cannonicalizeVec(arg_vec, ss_map, deep_copy);
  // 返回一个包含两个元素的元组：第一个元素是 schema 对象的操作名称，第二个元素是 canonical_args
  return std::make_tuple(schema->operator_name(), canonical_args);
}

} // namespace

// 缓存函数形状信息
TORCH_API void cache_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    const std::vector<at::SymbolicShape>& ret_vec) {
  // TODO: compare perf using std::vector<std::tuple<int64_t, int64_t>>
  
  // 创建空的无序映射表，用于存储形状映射关系
  auto ss_map = std::unordered_map<int64_t, int64_t>();

  // 获取缓存键值
  auto cache_key = get_cache_key(schema, arg_vec, ss_map, /* deep_copy */ true);

  // 规范化返回值向量，并存储为共享指针
  auto can_ret_vec = std::make_shared<std::vector<CanonicalizedSymbolicShape>>(
      cannonicalizeVec(ret_vec, ss_map));

  // 将缓存键和规范化的返回值向量添加到缓存中
  shapeCache.Add(std::move(cache_key), std::move(can_ret_vec));
}

// 从缓存中获取函数形状信息
TORCH_API std::optional<std::vector<at::SymbolicShape>>
get_cached_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec) {
  // TODO: compare perf using std::vector<std::tuple<int64_t, int64_t>> for both
  // ss_map and inverse_ss_map

  // 创建空的无序映射表，用于存储形状映射关系
  auto ss_map = std::unordered_map<int64_t, int64_t>();

  // 获取缓存键值
  auto cache_key =
      get_cache_key(schema, arg_vec, ss_map, /* deep_copy */ false);

  // 从缓存中获取已缓存的返回值向量
  auto cached_ret_vec = shapeCache.Get(cache_key);

  // 如果未找到缓存结果，则返回空值
  if (cached_ret_vec == nullptr) {
    return c10::nullopt;
  }

  // 反规范化返回值向量
  auto inverse_ss_map = std::unordered_map<int64_t, int64_t>();
  for (auto& ss_val : ss_map) {
    inverse_ss_map[ss_val.second] = ss_val.first;
  }

  // 创建目标返回值向量，进行反规范化操作
  std::vector<at::SymbolicShape> ret_vec;
  for (auto& css : *cached_ret_vec) {
    ret_vec.emplace_back(css.toSymbolicShape(inverse_ss_map));
  }
  return ret_vec;
}

// 用于测试访问缓存的函数，清除缓存
TORCH_API void clear_shape_cache() {
  shapeCache.Clear();
}

// 获取形状缓存的大小
TORCH_API size_t get_shape_cache_size() {
  return shapeCache.Numel();
}

// 初始化规范化符号形状对象
void CanonicalizedSymbolicShape::init(
    const c10::SymbolicShape& orig_shape,
    std::unordered_map<int64_t, int64_t>& ss_map) {
  // 获取形状的尺寸信息
  auto sizes = orig_shape.sizes();

  // 如果尺寸信息不存在，则置空
  if (!sizes) {
    values_ = c10::nullopt;
    return;
  }

  // 初始化存储尺寸值的向量
  values_ = std::vector<int64_t>();

  // 当前符号索引，用于处理动态形状
  int64_t cur_symbolic_index = -static_cast<int64_t>(ss_map.size()) - 1;

  // 遍历每个形状信息，根据静态或动态性进行处理
  for (auto& cur_shape : *sizes) {
    if (cur_shape.is_static()) {
      values_->push_back(cur_shape.static_size());
    } else {
      // 检查是否已存在映射关系
      auto it = ss_map.find(cur_shape.value());

      // 若不存在则添加新的符号索引映射
      if (it == ss_map.end()) {
        values_->push_back(cur_symbolic_index);
        ss_map.insert({cur_shape.value(), cur_symbolic_index});
        cur_symbolic_index--;
      } else {
        // 否则使用已存在的符号索引映射
        values_->push_back(it->second);
      }
    }
  }
}

// 将规范化符号形状对象转换为符号形状对象
c10::SymbolicShape CanonicalizedSymbolicShape::toSymbolicShape(
    std::unordered_map<int64_t, int64_t>& inverse_ss_map) const {
  // 若值不存在，则返回空符号形状对象
  if (!values_.has_value()) {
    return c10::SymbolicShape();
  }

  // 创建存储形状符号的向量
  std::vector<at::ShapeSymbol> sizes;

  // 遍历每个形状值，进行静态大小或符号映射转换
  for (long long cur_val : *values_) {
    if (cur_val >= 0) {
      sizes.push_back(at::ShapeSymbol::fromStaticSize(cur_val));
      continue;
    }
    auto res = inverse_ss_map.find(cur_val);
    if (res != inverse_ss_map.end()) {
      sizes.push_back(at::ShapeSymbol::fromStaticSize(res->second));
    }
  }
  return c10::SymbolicShape(std::move(sizes));
}
    } else {
      // 如果当前值不在映射中，创建一个新的形状符号
      auto new_symbol = at::ShapeSymbol::newSymbol();
      // 将当前值和新符号的值插入到逆映射中
      inverse_ss_map.insert({cur_val, new_symbol.value()});
      // 将新符号添加到大小列表中
      sizes.push_back(new_symbol);
    }
  }
  // 返回移动后的符号形状对象
  return c10::SymbolicShape(std::move(sizes));
}

// 定义类 CanonicalizedSymbolicShape 的成员函数 hash()
size_t CanonicalizedSymbolicShape::hash() const {
    // 如果成员变量 values_ 没有值
    if (!values_.has_value()) {
        // 返回一个随机值，用于避免哈希冲突
        return 0x8cc80c80;
    }
    // 否则，使用 c10::hash 对 values_.value() 中的 std::vector<int64_t> 进行哈希计算并返回结果
    return c10::hash<std::vector<int64_t>>()(values_.value());
}

// 定义全局的操作符重载函数 operator==，用于比较两个 CanonicalizedSymbolicShape 对象是否相等
bool operator==(
    const CanonicalizedSymbolicShape& a,
    const CanonicalizedSymbolicShape& b) {
    // 直接比较两个对象的 values_ 成员变量是否相等
    return a.values_ == b.values_;
};

// 命名空间 jit 结束
} // namespace jit

// 命名空间 torch 结束
} // namespace torch
```