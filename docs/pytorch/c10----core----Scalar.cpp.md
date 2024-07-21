# `.\pytorch\c10\core\Scalar.cpp`

```py
namespace c10 {

Scalar Scalar::operator-() const {
  // 检查是否为布尔类型，布尔类型不支持取负操作
  TORCH_CHECK(
      !isBoolean(),
      "torch boolean negative, the `-` operator, is not supported.");
  // 如果是浮点数类型
  if (isFloatingPoint()) {
    // 检查是否为符号类型，不支持符号类型的浮点数取负操作
    TORCH_CHECK(!isSymbolic(), "NYI negate symbolic float");
    // 返回当前浮点数取负后的 Scalar 对象
    return Scalar(-v.d);
  } else if (isComplex()) { // 如果是复数类型
    // 内部断言，确保不是符号类型
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    // 返回当前复数取共轭后的 Scalar 对象
    return Scalar(-v.z);
  } else if (isIntegral(false)) { // 如果是整数类型（可能是有符号或无符号）
    // 检查是否为符号类型，不支持符号类型的整数取负操作
    TORCH_CHECK(!isSymbolic(), "NYI negate symbolic int");
    // 返回当前整数取负后的 Scalar 对象
    return Scalar(-v.i);
  }
  // 如果以上条件都不满足，抛出未知的 ivalue 标签异常
  TORCH_INTERNAL_ASSERT(false, "unknown ivalue tag ", static_cast<int>(tag));
}

Scalar Scalar::conj() const {
  // 如果是复数类型
  if (isComplex()) {
    // 内部断言，确保不是符号类型
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    // 返回当前复数取共轭后的 Scalar 对象
    return Scalar(std::conj(v.z));
  } else {
    // 如果不是复数类型，返回当前 Scalar 对象
    return *this;
  }
}

Scalar Scalar::log() const {
  // 如果是复数类型
  if (isComplex()) {
    // 内部断言，确保不是符号类型
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    // 返回当前复数取对数后的 Scalar 对象
    return std::log(v.z);
  } else if (isFloatingPoint()) { // 如果是浮点数类型
    // 检查是否为符号类型，不支持符号类型的浮点数取对数操作
    TORCH_CHECK(!isSymbolic(), "NYI log symbolic float");
    // 返回当前浮点数取对数后的 Scalar 对象
    return std::log(v.d);
  } else if (isIntegral(false)) { // 如果是整数类型（可能是有符号或无符号）
    // 检查是否为符号类型，不支持符号类型的整数取对数操作
    TORCH_CHECK(!isSymbolic(), "NYI log symbolic int");
    // 返回当前整数取对数后的 Scalar 对象
    return std::log(v.i);
  }
  // 如果以上条件都不满足，抛出未知的 ivalue 标签异常
  TORCH_INTERNAL_ASSERT(false, "unknown ivalue tag ", static_cast<int>(tag));
}

} // namespace c10
```