# `.\pytorch\aten\src\ATen\core\tensor_type.cpp`

```
namespace c10 {

namespace {

// 我们的目标是仅标记可能在维度间有重叠的情况。我们希望对于扩展的张量和置换的张量返回false，
// 对于这些张量来说，维度折叠是安全的。
bool possible_cross_dimension_overlap(c10::IntArrayRef sizes, c10::IntArrayRef strides) {
  // 获取维度数量
  int n_dim = static_cast<int>(sizes.size());
  // 创建一个包含维度索引的向量，并且逆序排列
  std::vector<size_t> stride_indices(n_dim);
  std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);

  // 根据步长的升序对索引进行排序
  for (int i = 1; i < n_dim; i++) {
    auto c = i;
    for (int j = i - 1; j >= 0; j--) {
      if (strides[stride_indices[j]] > strides[stride_indices[c]]) {
        std::swap(stride_indices[j], stride_indices[c]);
        c = j;
      }
    }
  }

  // 遍历维度，检查内存重叠的保守性
  for (const auto i : c10::irange(1, n_dim)) {
    if (i != 0) {
      if (sizes[stride_indices[i]] != 1 && strides[stride_indices[i]] < sizes[stride_indices[i-1]] * strides[stride_indices[i-1]]) {
        return true;
      }
    }
  }
  // 没有检测到重叠，返回false
  return false;
}

}

// 返回静态变量TensorType::create的引用
const TensorTypePtr& TensorType::get() {
  static auto value = TensorType::create(
      {}, {}, SymbolicShape(), VaryingShape<Stride>{}, {});
  return value;
}

// 返回静态变量ListType::create(TensorType::get())的引用
ListTypePtr ListType::ofTensors() {
  static auto value = ListType::create(TensorType::get());
  return value;
}

// 合并两个VaryingShape<T>对象，返回合并后的结果
template <typename T>
VaryingShape<T> VaryingShape<T>::merge(const VaryingShape<T>& other) const {
  // 如果其中一个对象没有维度信息，则返回空对象
  if (!dims_ || !other.dims_ || dims_->size() != other.dims_->size()) {
    return VaryingShape<T>();
  }
  // 合并两个对象的维度信息
  ListOfOptionalElements dims;
  for (size_t i = 0, n = dims_->size(); i < n; i++) {
    dims.push_back(merge_primitive((*dims_)[i], (*other.dims_)[i]));
  }
  return VaryingShape<T>(std::move(dims));
}

// 重载操作符<<，用于将VaryingShape<T>对象输出到流中
template <typename T>
std::ostream& operator<<(std::ostream& out, const VaryingShape<T>& vs) {
  out << "(";
  if (!vs.size()) {
    out << "*)";
    return out;
  }

  for (size_t i = 0; i < vs.size(); i++) {
    if (i > 0) {
      out << ", ";
    }
    if (vs[i].has_value()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      out << vs[i].value();
    } else {
      out << "*";
    }
  }
  out << ")";
  return out;
}

// 重载操作符<<，用于将SymbolicShape对象输出到流中
std::ostream& operator<<(
    std::ostream& os,
    const SymbolicShape& ss) {
  // TODO: Unranked SymbolicShape printing is ambiguous with that of
  // dynamic-shaped vector.
  // 如果rank为0，则输出(*)
  if(!ss.rank()) {
    os << "(*)";
    return os;
  }

  auto sizes = ss.sizes().value();

  os << "(";
  // 输出每个维度的大小或者*号（如果是动态形状）
  for (size_t i = 0; i < ss.rank().value(); i++) {
    if (i > 0) {
      os << ", ";
    }
    if(sizes[i].is_static()) {
      os << sizes[i];
    } else {
      os << "*";
    }
  }
  os << ")";
  return os;
}
    }
  }
  os << ")";

  

# 结束所有的代码块，对输出流 os 输出一个右括号



  return os;

  

# 返回输出流 os，表示函数执行完毕并返回流对象
}

// 重载运算符<<，用于输出ShapeSymbol对象到流中
std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s) {
    // 如果ShapeSymbol的值大于等于0，则直接输出其值
    if (s.value_ >= 0) {
        os << s.value_;
    } else {  // 否则输出带有SS(...)格式的值
        os << "SS(" << s.value_ << ')';
    }
    return os;  // 返回输出流
}

// 重载运算符<<，用于输出Stride对象到流中
std::ostream& operator<<(std::ostream& os, const Stride& s) {
    os << "{";  // 输出左大括号
    // 如果stride_index_有值，则输出其值；否则输出 *
    if (s.stride_index_.has_value()) {
        os << *s.stride_index_;
    } else {
        os << "*";
    }
    os << ":";  // 输出冒号
    // 如果stride_有值，则输出其值；否则输出 *
    if (s.stride_.has_value()) {
        os << *s.stride_;
    } else {
        os << "*";
    }
    os << '}';  // 输出右大括号
    return os;  // 返回输出流
}

// 计算Tensor类型的Stride属性
VaryingShape<Stride> TensorType::computeStrideProps(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    bool tensor_contiguity) {
  int n_dim = static_cast<int>(sizes.size());
  std::vector<size_t> stride_indices(n_dim);
  // 将has_overlap默认设置为false，因为我们只在以下情况下计算重叠：
  // 1. 输入的sizes/strides格式检查失败；
  // 2. tensor_contiguity未设置。
  bool has_overlap = false;

  // 将strides按升序排序
  // 示例：
  //  在排序之前
  //  Idx:     [0,   1,  2,  3]
  //  sizes:   [8,   1, 10, 16]
  //  Strides: [160, 1, 16,  1]
  //  排序之后
  //  Idx:     [1,  3,  2,   0]
  //  sizes:   [1, 16, 10,   8]
  //  Strides: [1,  1, 16, 160]
  //
  // 下面的逻辑遵循TensorIterator在其逻辑中使用的方式：
  //   1. Fast_set_up是快速识别通道最后和连续格式的捷径，这是下面逻辑的情况。
  //   2. 在更一般的情况下，它尽力保留排列。
  if (is_channels_last_strides_2d(sizes, strides) || is_channels_last_strides_3d(sizes, strides)) {
    // case 1.a. 通道最后的快捷方式
    std::iota(stride_indices.rbegin() + 1, stride_indices.rend() - 1, 2);
    stride_indices[0] = 1;
    stride_indices[n_dim - 1] = 0;
  } else if (is_contiguous_strides(sizes, strides)) {
    // case 1.b. 连续的快捷方式
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
  } else {
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
    // case 2.
    //
    // 对于广播维度，其中stride为0，我们必须坚持在急切计算中TensorIterator的行为，
    // 在那里他们引入了一个模糊的比较结果，通过尽力保留排列。
    // 更多细节，请参见注释：[计算输出步幅]
    auto should_swap = [&](size_t a, size_t b) {
      if (strides[a] == 0 || strides[b] == 0) {
        return 0;
      } else if (strides[a] < strides[b]) {
        return -1;
      } else if (strides[a] > strides[b]) {
        return 1;
      } else { // strides[a] == strides[b]
        if (sizes[a] > sizes[b]) {
          return 1;
        }
      }
      return 0;
    };
    // 外层循环遍历维度索引，从第二个维度开始到倒数第二个维度
    for (int i = 1; i < n_dim; i++) {
      // 记录当前维度索引
      int dim1 = i;
      // 内层循环从当前维度索引往前遍历到第一个维度索引
      for (int dim0 = i - 1; dim0 >= 0; dim0--) {
        // 调用 should_swap 函数比较两个维度索引的应交换关系
        int comparison = should_swap(stride_indices[dim0], stride_indices[dim1]);
        // 如果应交换
        if (comparison > 0) {
          // 交换两个维度索引
          std::swap(stride_indices[dim0], stride_indices[dim1]);
          // 更新 dim1 为当前维度索引
          dim1 = dim0;
        }
        // 如果不应交换，则退出内层循环
        else if (comparison < 0) {
          break;
        }
      }
    }
    // 如果 tensor_contiguity 为假
    if (!tensor_contiguity) {
      // 根据 sizes 和 strides 计算是否可能有跨维度重叠
      has_overlap = possible_cross_dimension_overlap(sizes, strides);
    }
  }

  // 存储每个维度索引的属性列表
  std::vector<Stride> stride_properties;
  // 预留足够空间以存储所有维度索引的属性
  stride_properties.reserve(stride_indices.size());
  // 遍历所有维度索引
  for (size_t i = 0; i < stride_indices.size(); i++) {
    // 默认为 tensor_contiguity 的值
    bool contiguous_ = tensor_contiguity;
    // 如果不是连续的
    if (!contiguous_) {
      // 如果没有内存重叠
      if (!has_overlap) {
        // 对于第一个维度索引，预期其步长为 1
        if (i == 0) {
          contiguous_ = strides[stride_indices[i]] == 1;
        } else {
          // 对于其他维度索引，判断步长是否为 1 或者符合连续性条件
          contiguous_ = strides[stride_indices[i]] == 1 ||
              (strides[stride_indices[i]] != 0 &&
               strides[stride_indices[i]] ==
                   strides[stride_indices[i - 1]] * sizes[stride_indices[i - 1]]);
        }
      } else {
        // 如果有内存重叠，则不是连续的
        contiguous_ = false;
      }
    }
    // 将每个维度索引的属性加入 stride_properties 中
    stride_properties.emplace_back(stride_indices[i], contiguous_, strides[stride_indices[i]]);
  }

  // 返回带有维度索引属性的 VaryingShape 对象
  return VaryingShape<Stride>{stride_properties};
}

TensorTypePtr TensorType::create(const at::Tensor& t) {
  VaryingShape<bool> contiguity;  // 定义布尔型变量，用于存储张量的连续性信息
  VaryingShape<size_t> stride_indices;  // 定义变量，存储张量的步长索引信息
  VaryingShape<int64_t> strides;  // 定义变量，存储张量的步长信息
  VaryingShape<int64_t> sizes;  // 定义变量，存储张量的尺寸信息
  if (t.layout() == at::kStrided && !t.is_nested()) {  // 检查张量是否是步进布局且非嵌套
    sizes = VaryingShape<int64_t>{t.sizes().vec()};  // 获取张量的尺寸信息，并存入变量
    strides = VaryingShape<int64_t>{t.strides().vec()};  // 获取张量的步长信息，并存入变量
    return TensorType::create(
        t.scalar_type(), t.device(), sizes, strides, t.requires_grad(), false, t.is_contiguous());  // 调用另一个create函数创建TensorTypePtr对象，传递张量的相关信息
  }

  return TensorType::create(
      t.scalar_type(),
      t.device(),
      SymbolicShape(),
      VaryingShape<Stride>{},
      t.requires_grad(),
      false);
}

TensorTypePtr TensorType::create(
    std::optional<at::ScalarType> scalar_type,
    std::optional<Device> device,
    const VaryingShape<int64_t>& sizes,
    const VaryingShape<int64_t>& strides,
    std::optional<bool> requires_grad,
    std::optional<bool> undefined, bool tensor_contiguity) {
  if(strides.concrete_sizes() && strides.concrete_sizes().has_value()){
    // 处理步长已设置的情况
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    TORCH_INTERNAL_ASSERT(sizes.concrete_sizes()->size() == strides.concrete_sizes()->size());  // 断言：尺寸和步长的具体大小相等
    auto sprops = strides.concrete_sizes().has_value()
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      ? computeStrideProps(*sizes.concrete_sizes(), *strides.concrete_sizes(), tensor_contiguity)  // 如果步长已设置，则计算步长属性
      : VaryingShape<Stride>();
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto symbol_sizes = SymbolicShape(*sizes.concrete_sizes());  // 创建具有符号尺寸的SymbolicShape对象
    return TensorType::create(
      scalar_type, device, symbol_sizes, sprops, requires_grad, undefined);  // 调用另一个create函数创建TensorTypePtr对象，传递相关参数
  } else {
    // 步长全部为空，但仍有与秩数相等的步长数
    TORCH_INTERNAL_ASSERT(sizes.sizes() && sizes.size());  // 断言：尺寸存在且尺寸数量大于零
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto symbol_sizes = SymbolicShape(*sizes.sizes());  // 创建具有符号尺寸的SymbolicShape对象
    return TensorType::create(
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      scalar_type, device, symbol_sizes, VaryingShape<Stride>(*sizes.size()), requires_grad, undefined);  // 调用另一个create函数创建TensorTypePtr对象，传递相关参数
  }
}

TensorTypePtr TensorType::create(
    std::optional<at::ScalarType> scalar_type,
    std::optional<Device> device,
    const SymbolicShape& sizes,
    const VaryingShape<Stride>& strides,
    std::optional<bool> requires_grad,
    std::optional<bool> undefined) {
  auto pt = TensorTypePtr(new TensorType(
      scalar_type, device, sizes, strides, requires_grad, undefined));  // 使用给定的参数创建新的TensorType对象，并赋值给指针pt
  return pt;  // 返回指向新创建对象的指针
}

TensorTypePtr TensorType::create(
    std::optional<at::ScalarType> scalar_type,
    std::optional<Device> device,
    std::optional<size_t> dim,
    std::optional<bool> requires_grad) {
  return TensorType::create(
      scalar_type,
      device,
      SymbolicShape(dim),  // 创建具有符号尺寸的SymbolicShape对象
      VaryingShape<Stride>(dim),  // 创建具有变化步长的VaryingShape对象
      requires_grad);
}

std::string TensorType::str() const {
  return "Tensor";  // 返回字符串 "Tensor"
}
// 初始化 ShapeSymbol 类的静态原子计数为 1
std::atomic<size_t> ShapeSymbol::num_symbols{1};

// 实例化 VaryingShape 模板类，分别使用 c10::ShapeSymbol、bool、size_t、int64_t、c10::Stride 作为模板参数
template struct VaryingShape<c10::ShapeSymbol>;
template struct VaryingShape<bool>;
template struct VaryingShape<size_t>;
template struct VaryingShape<int64_t>;
template struct VaryingShape<c10::Stride>;

// 返回当前 TensorType 对象的 sizes，若 sizes_ 的 rank 为 0，则返回一个默认构造的 VaryingShape<int64_t> 对象
VaryingShape<int64_t> TensorType::sizes() const {
  if (!sizes_.rank()) {
    return VaryingShape<int64_t>();
  }
  // 使用 fmap 函数对 sizes_ 的 sizes() 返回的每个 ShapeSymbol 进行处理，将其转换为 int64_t 或者 c10::nullopt
  return VaryingShape<int64_t>(
      fmap(*sizes_.sizes(), [](ShapeSymbol ss) {
        // 将符号形状转换为未知数
        return ss.is_static()
            ? std::optional<int64_t>(ss.static_size())
            : c10::nullopt;
      }));
}

// 合并当前 TensorType 对象与另一个对象 other，根据 merge_sizes 决定是否合并 symbolic_sizes
TensorTypePtr TensorType::merge(const TensorType& other, bool merge_sizes) const {
  // 合并标量类型、设备、步幅属性、是否需要梯度、是否未定义
  auto scalar_type = merge_primitive(scalarType(), other.scalarType());
  auto dev = merge_primitive(device(), other.device());
  auto sprops = stride_properties().merge(other.stride_properties());
  auto gr = merge_primitive(requiresGrad(), other.requiresGrad());
  auto undef = merge_primitive(undefined(), other.undefined());
  // 创建一个新的 TensorType 对象，合并后的属性作为参数传入
  return TensorType::create(
      scalar_type,
      dev,
      merge_sizes ? symbolic_sizes().merge(other.symbolic_sizes())
                  : symbolic_sizes(),
      sprops,
      gr,
      undef);
}

// 检查可选类型 a 是否为空或者与 b 相等
template <typename T>
bool is_null_or_equal(std::optional<T> a, c10::IntArrayRef b) {
  return !a.has_value() || a.value() == b;
}

// 匹配当前 TensorType 对象与给定的 at::Tensor t 是否相符
bool TensorType::matchTensor(const at::Tensor& t) {
  // 判断当前 TensorType 是否为未定义，若未定义则检查 t 是否也未定义
  bool undef = undefined().value_or(!t.defined());
  if (undef != !t.defined()) {
    // 当满足以下条件时，认为不匹配：
    // - undefined().has_value() == true
    // - undefined().value() != !t.defined()
    return false;
  } else if (!t.defined()) {
    // 当满足以下条件时，认为匹配：
    // - t 未定义
    // - undefined() 为 null 或者 undefined().value() == true
    return true;
  }
  // 此处已知 t 已定义，比较所有其他属性
  bool rg = at::GradMode::is_enabled() && t.requires_grad();
  bool matched_strides = (!stride_properties().size()) ||
      (!t.has_storage() && !stride_properties().isComplete()) ||
      stride_properties() ==
          computeStrideProps(t.sizes(), t.strides(), t.is_contiguous());
  return scalarType().value_or(t.scalar_type()) == t.scalar_type()
    && device().value_or(t.device()) == t.device()
    && requiresGrad().value_or(rg) == rg
    && matched_strides
    && is_null_or_equal(sizes().concrete_sizes(), t.sizes());
}

// 检查当前 TensorType 对象是否与另一个 Type 对象 rhs 相等
bool TensorType::equals(const c10::Type& rhs) const {
  // 若类型不同，则不相等
  if (rhs.kind() != kind()) {
    return false;
  }
  auto rt = rhs.expect<TensorType>();

  // 比较标量类型、sizes、步幅属性、设备、是否需要梯度、是否未定义
  return scalar_type_ == rt->scalarType() && sizes() == rt->sizes() &&
      stride_properties() == rt->stride_properties() &&
      device() == rt->device() && requiresGrad() == rt->requiresGrad() &&
      undefined() == rt->undefined();
}
// 返回当前张量类型对象的步幅
VaryingShape<int64_t> TensorType::strides() const {
  // 如果步幅大小未定义，则返回一个空的 VaryingShape 对象
  if (!strides_.size().has_value()) {
    return VaryingShape<int64_t>();
  }
  // 创建一个包含大小为 strides_.size() 的可选整数向量
  std::vector<std::optional<int64_t>> ss(*strides_.size());
  // 遍历每一个步幅
  for (size_t i = 0; i < *strides_.size(); i++) {
    // 如果当前步幅未定义，则继续下一个循环
    if (!strides_[i].has_value()) {
      continue;
    }
    // 取出当前步幅值
    auto s = *strides_[i];
    // 如果步幅索引和步幅值都有定义，则将步幅值存入对应索引的位置
    if (s.stride_index_.has_value() && s.stride_.has_value()) {
      ss[*s.stride_index_] = *s.stride_;
    }
  }
  // 返回一个带有步幅信息的 VaryingShape 对象
  return VaryingShape<int64_t>(std::move(ss));
}

// 张量类型的构造函数，初始化各种属性
TensorType::TensorType(
    std::optional<at::ScalarType> scalar_type,
    std::optional<Device> device,
    SymbolicShape sizes,
    VaryingShape<Stride> strides,
    std::optional<bool> requires_grad,
    std::optional<bool> undefined)
    : SharedType(TypeKind::TensorType),
      scalar_type_(scalar_type),
      device_(device),
      sizes_(std::move(sizes)),
      strides_(std::move(strides)),
      requires_grad_(requires_grad),
      undefined_(undefined) {}

// 创建连续张量的静态方法
TensorTypePtr TensorType::createContiguous(
    at::ScalarType scalar_type,
    at::Device device,
    at::IntArrayRef sizes) {
  // 计算连续张量的步幅
  auto strides = contiguousStridesOf(sizes);
  // 断言步幅和尺寸的大小相同
  TORCH_INTERNAL_ASSERT(strides.size() == sizes.size());
  // 使用给定的参数创建一个新的张量类型对象
  return create(
      scalar_type,
      device,
      VaryingShape<int64_t>(sizes),
      VaryingShape<int64_t>(strides),
      c10::nullopt);
}

// 返回张量类型对象的符号尺寸
const SymbolicShape& TensorType::symbolic_sizes() const {
  return sizes_;
}

// 判断当前张量类型是否是指定类型的子类型
bool TensorType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  if (auto rhs_p = rhs.cast<TensorType>()) {
    // 如果指针相同，直接返回 true 避免计算合并
    if (this == rhs_p.get()) {
      return true;
    }
    // 判断当前类型是否可以合并为指定类型
    return *merge(*rhs_p) == *rhs_p;
  }
  // 调用基类的判断方法
  return Type::isSubtypeOfExt(rhs, why_not);
}
```