# `.\pytorch\c10\core\SymFloat.cpp`

```
namespace c10 {

# 进入c10命名空间，定义了SymFloat类的成员函数和静态函数


SymNode SymFloat::toSymNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

# 将SymFloat对象转换为SymNode对象的实现。要求当前对象必须是符号化的。


SymNode SymFloat::wrap_node(const SymNode& base) const {
  if (is_symbolic()) {
    return toSymNodeImpl();
  } else {
    return base->wrap_float(as_float_unchecked());
  }
}

# 根据基础节点包装SymFloat对象，如果当前对象是符号化的则直接转换为SymNode，否则使用基础节点将当前浮点数包装为SymNode。


static std::array<SymNode, 2> normalize_symfloats(
    const SymFloat& a_,
    const SymFloat& b_) {
  SymNode a, b;
  if (a_.is_symbolic())
    a = a_.toSymNodeImpl();
  if (b_.is_symbolic())
    b = b_.toSymNodeImpl();

  SymNodeImpl* common = a ? a.get() : b.get();
  if (!a) {
    a = common->wrap_float(a_.as_float_unchecked());
  }
  if (!b) {
    b = common->wrap_float(b_.as_float_unchecked());
  }
  return {std::move(a), std::move(b)};
}

# 标准化两个SymFloat对象为SymNode数组。如果其中一个对象是符号化的，则转换为对应的SymNode对象；否则，将非符号化对象包装为SymNode对象，以便后续操作。


SymFloat SymFloat::operator+(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ + sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->add(res[1]));
}

# SymFloat对象的加法操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数加法；否则，将它们标准化为SymNode数组，然后执行加法操作。


SymFloat SymFloat::operator-(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ - sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->sub(res[1]));
}

# SymFloat对象的减法操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数减法；否则，将它们标准化为SymNode数组，然后执行减法操作。


SymFloat SymFloat::operator*(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ * sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->mul(res[1]));
}

# SymFloat对象的乘法操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数乘法；否则，将它们标准化为SymNode数组，然后执行乘法操作。


SymFloat SymFloat::operator/(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ / sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->truediv(res[1]));
}

# SymFloat对象的除法操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数除法；否则，将它们标准化为SymNode数组，然后执行除法操作。


SymBool SymFloat::sym_eq(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ == sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->eq(res[1]);
}

# SymFloat对象的相等比较操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数相等比较；否则，将它们标准化为SymNode数组，然后执行相等比较。


SymBool SymFloat::sym_ne(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ != sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->ne(res[1]);
}

# SymFloat对象的不等比较操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数不等比较；否则，将它们标准化为SymNode数组，然后执行不等比较。


SymBool SymFloat::sym_lt(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ < sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->lt(res[1]);
}

# SymFloat对象的小于比较操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数小于比较；否则，将它们标准化为SymNode数组，然后执行小于比较。


SymBool SymFloat::sym_le(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ <= sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->le(res[1]);
}

# SymFloat对象的小于等于比较操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数小于等于比较；否则，将它们标准化为SymNode数组，然后执行小于等于比较。


SymBool SymFloat::sym_gt(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ > sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->gt(res[1]);
}

# SymFloat对象的大于比较操作符重载实现。如果两个对象都不是符号化的，则直接执行浮点数大于比较；否则，将它们标准化为SymNode数组，然后执行大于比较。
// 比较当前 SymFloat 对象和给定 SymFloat 对象 sci 的大小关系，返回比较结果
SymBool SymFloat::sym_ge(const SymFloat& sci) const {
  // 如果当前对象和 sci 都不是符号化的
  if (!is_symbolic() && !sci.is_symbolic()) {
    // 直接比较两个实际数据的大小并返回结果
    return data_ >= sci.data_;
  }
  // 将当前对象和 sci 进行标准化处理，获取标准化后的结果数组
  auto res = normalize_symfloats(*this, sci);
  // 调用标准化后的结果数组中第一个元素的 ge 方法，返回比较结果
  return res[0]->ge(res[1]);
}

// 返回当前 SymFloat 对象和给定 SymFloat 对象 sci 的较小值
SymFloat SymFloat::min(const SymFloat& sci) const {
  // 如果当前对象和 sci 都不是符号化的
  if (!is_symbolic() && !sci.is_symbolic()) {
    // 返回两个实际数据的较小值
    return std::min(data_, sci.data_);
  }
  // 将当前对象和 sci 进行标准化处理，获取标准化后的结果数组
  auto res = normalize_symfloats(*this, sci);
  // 调用标准化后的结果数组中第一个元素的 sym_min 方法，返回结果
  return SymFloat(res[0]->sym_min(res[1]));
}

// 返回当前 SymFloat 对象和给定 SymFloat 对象 sci 的较大值
SymFloat SymFloat::max(const SymFloat& sci) const {
  // 如果当前对象和 sci 都不是符号化的
  if (!is_symbolic() && !sci.is_symbolic()) {
    // 返回两个实际数据的较大值
    return std::max(data_, sci.data_);
  }
  // 将当前对象和 sci 进行标准化处理，获取标准化后的结果数组
  auto res = normalize_symfloats(*this, sci);
  // 调用标准化后的结果数组中第一个元素的 sym_max 方法，返回结果
  return SymFloat(res[0]->sym_max(res[1]));
}

// 输出 SymFloat 对象 s 到输出流 os 中
std::ostream& operator<<(std::ostream& os, const SymFloat& s) {
  // 如果对象 s 是符号化的
  if (s.is_symbolic()) {
    // 将符号化对象转换为 SymNodeImpl，并输出其字符串表示
    os << s.toSymNodeImpl()->str();
  } else {
    // 直接输出对象 s 的未检查浮点数值
    os << s.as_float_unchecked();
  }
  return os;
}

// 返回当前 SymFloat 对象的平方根
SymFloat SymFloat::sqrt() const {
  // 如果当前对象不是符号化的
  if (!is_symbolic()) {
    // 返回当前对象数据的平方根作为 SymFloat 对象
    return SymFloat(std::sqrt(data_));
  }
  // 创建 SymFloat 对象表示 -0.5
  auto other = SymFloat(-0.5);
  // 将当前对象和 other 进行标准化处理，获取标准化后的结果数组
  auto res = normalize_symfloats(*this, other);
  // 调用标准化后的结果数组中第一个元素的 pow 方法，计算当前对象的 other 次方，返回结果
  return SymFloat(res[0]->pow(res[1]));
}

// 如果当前 SymFloat 对象不是符号化的，则返回其数据值
double SymFloat::guard_float(const char* file, int64_t line) const {
  // 如果当前对象不是符号化的
  if (!is_symbolic()) {
    // 直接返回当前对象的数据值
    return data_;
  }
  // 将当前对象转换为 SymNodeImpl，并调用其 guard_float 方法，返回结果
  SymNode a = toSymNodeImpl();
  return a->guard_float(file, line);
}

// 如果当前 SymFloat 对象是符号化的，则返回 true；否则返回其 SymNodeImpl 对象的 has_hint 方法的结果
bool SymFloat::has_hint() const {
  // 如果当前对象不是符号化的
  if (!is_symbolic()) {
    // 返回 true，表示不是符号化的
    return true;
  }
  // 调用当前对象的 toSymNodeImpl 方法获取 SymNodeImpl 对象，并调用其 has_hint 方法返回结果
  return toSymNodeImpl()->has_hint();
}
```