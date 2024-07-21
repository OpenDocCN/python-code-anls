# `.\pytorch\torch\csrc\jit\tensorexpr\expr.cpp`

```py
// 引入TensorExpr库中的表达式头文件
#include <torch/csrc/jit/tensorexpr/expr.h>

// 引入TensorExpr库中的IR相关头文件
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

// 定义torch::jit::tensorexpr命名空间
namespace torch::jit::tensorexpr {

// 实现ExprHandle类的加法运算符重载，返回相加的表达式
ExprHandle ExprHandle::operator+(const ExprHandle& other) const {
  return Add::make(*this, other);
}

// 实现ExprHandle类的减法运算符重载，返回相减的表达式
ExprHandle ExprHandle::operator-(const ExprHandle& other) const {
  return Sub::make(*this, other);
}

// 实现ExprHandle类的乘法运算符重载，返回相乘的表达式
ExprHandle ExprHandle::operator*(const ExprHandle& other) const {
  return Mul::make(*this, other);
}

// 实现ExprHandle类的除法运算符重载，返回相除的表达式
ExprHandle ExprHandle::operator/(const ExprHandle& other) const {
  return Div::make(*this, other);
}

// 实现ExprHandle类的取模运算符重载，返回取模的表达式
ExprHandle ExprHandle::operator%(const ExprHandle& other) const {
  return Mod::make(*this, other);
}

// 实现ExprHandle类的等于比较运算符重载，返回等于比较的表达式
ExprHandle ExprHandle::operator==(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kEQ);
}

// 实现ExprHandle类的不等于比较运算符重载，返回不等于比较的表达式
ExprHandle ExprHandle::operator!=(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kNE);
}

// 实现ExprHandle类的大于比较运算符重载，返回大于比较的表达式
ExprHandle ExprHandle::operator>(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGT);
}

// 实现ExprHandle类的大于等于比较运算符重载，返回大于等于比较的表达式
ExprHandle ExprHandle::operator>=(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGE);
}

// 实现ExprHandle类的小于比较运算符重载，返回小于比较的表达式
ExprHandle ExprHandle::operator<(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLT);
}

// 实现ExprHandle类的小于等于比较运算符重载，返回小于等于比较的表达式
ExprHandle ExprHandle::operator<=(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLE);
}

// 实现ExprHandle类的逻辑与运算符重载，如果操作数不是整数类型则抛出异常，返回条件选择表达式
ExprHandle ExprHandle::operator&&(const ExprHandle& other) const {
  if (!this->node()->dtype().is_integral()) {
    throw unsupported_dtype();
  }
  return IfThenElse::make(
      *this, other, ExprHandle(getImmediateByType(other.dtype(), 0)));
}

// 实现ExprHandle类的逻辑或运算符重载，如果操作数不是整数类型则抛出异常，返回条件选择表达式
ExprHandle ExprHandle::operator||(const ExprHandle& other) const {
  if (!this->node()->dtype().is_integral()) {
    throw unsupported_dtype();
  }
  return IfThenElse::make(
      *this, ExprHandle(getImmediateByType(other.dtype(), 1)), other);
}

// 实现ExprHandle类的位与运算符重载，返回位与的表达式
ExprHandle ExprHandle::operator&(const ExprHandle& other) const {
  return And::make(*this, other);
}

// 实现ExprHandle类的位或运算符重载，返回位或的表达式
ExprHandle ExprHandle::operator|(const ExprHandle& other) const {
  return Or::make(*this, other);
}

// 实现ExprHandle类的位异或运算符重载，返回位异或的表达式
ExprHandle ExprHandle::operator^(const ExprHandle& other) const {
  return Xor::make(*this, other);
}

// 实现ExprHandle类的左移运算符重载，返回左移的表达式
ExprHandle ExprHandle::operator<<(const ExprHandle& other) const {
  return Lshift::make(*this, other);
}

// 实现ExprHandle类的右移运算符重载，返回右移的表达式
ExprHandle ExprHandle::operator>>(const ExprHandle& other) const {
  return Rshift::make(*this, other);
}

// 宏定义IMM_EXPR_DECLARE用于声明标量类型的构造函数，生成具体的ExprHandle对象
#define IMM_EXPR_DECLARE(Type, Name) \
  ExprHandle::ExprHandle(Type v) : ExprHandle(Name##Imm::make(v)) {}
// 扩展宏定义，包括Bool、Half、BFloat16等类型
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_EXPR_DECLARE);
// 取消宏定义IMM_EXPR_DECLARE，结束对标量类型的构造函数声明
#undef IMM_EXPR_DECLARE

// 定义sin函数，返回对给定表达式的正弦计算结果
ExprHandle sin(const ExprHandle& v) {
  return Intrinsics::make(kSin, v);
}

// 定义cos函数，返回对给定表达式的余弦计算结果
ExprHandle cos(const ExprHandle& v) {
  return Intrinsics::make(kCos, v);
}

} // 结束torch::jit::tensorexpr命名空间
// 计算输入表达式的正切值
ExprHandle tan(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建正切操作的表达式
  return Intrinsics::make(kTan, v);
}

// 计算输入表达式的反正弦值
ExprHandle asin(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建反正弦操作的表达式
  return Intrinsics::make(kAsin, v);
}

// 计算输入表达式的反余弦值
ExprHandle acos(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建反余弦操作的表达式
  return Intrinsics::make(kAcos, v);
}

// 计算输入表达式的反正切值
ExprHandle atan(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建反正切操作的表达式
  return Intrinsics::make(kAtan, v);
}

// 计算输入表达式的双曲正弦值
ExprHandle sinh(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建双曲正弦操作的表达式
  return Intrinsics::make(kSinh, v);
}

// 计算输入表达式的双曲余弦值
ExprHandle cosh(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建双曲余弦操作的表达式
  return Intrinsics::make(kCosh, v);
}

// 计算输入表达式的双曲正切值
ExprHandle tanh(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建双曲正切操作的表达式
  return Intrinsics::make(kTanh, v);
}

// 计算输入表达式的 Sigmoid 函数值
ExprHandle sigmoid(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建 Sigmoid 操作的表达式
  return Intrinsics::make(kSigmoid, v);
}

// 计算输入表达式的指数函数值
ExprHandle exp(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建指数函数操作的表达式
  return Intrinsics::make(kExp, v);
}

// 计算输入表达式的 expm1 函数值（即 exp(v) - 1）
ExprHandle expm1(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建 expm1 函数操作的表达式
  return Intrinsics::make(kExpm1, v);
}

// 计算输入表达式的绝对值
ExprHandle abs(const ExprHandle& v) {
  // 使用 Intrinsics::make 创建绝对值操作的表达式
  return Intrinsics::make(kAbs, v);
}

// 使用 Eigen 版本的快速双曲正切函数，详细信息参见链接：
// https://bitbucket.org/eigen/eigen/src/94875feeeeb9abe5509b314197da1991ba2070f5/Eigen/src/Core/MathFunctionsImpl.h#lines-26
ExprHandle fast_tanh(const ExprHandle& v) {
  // TODO: 使用专用绑定变量确保 v 不会被多次评估。将输入表达式限制在 [-9, 9] 范围内
  ExprHandle plus_9 = FloatImm::make(9.0f);
  ExprHandle minus_9 = FloatImm::make(-9.0f);
  ExprHandle v1 = Min::make(v, plus_9, false);
  v1 = Max::make(v1, minus_9, false);

  // 计算分子的系数
  ExprHandle alpha_1 = FloatImm::make(4.89352455891786e-03f);
  ExprHandle alpha_3 = FloatImm::make(6.37261928875436e-04f);
  ExprHandle alpha_5 = FloatImm::make(1.48572235717979e-05f);
  ExprHandle alpha_7 = FloatImm::make(5.12229709037114e-08f);
  ExprHandle alpha_9 = FloatImm::make(-8.60467152213735e-11f);
  ExprHandle alpha_11 = FloatImm::make(2.00018790482477e-13f);
  ExprHandle alpha_13 = FloatImm::make(-2.76076847742355e-16f);

  // 计算分母的系数
  ExprHandle beta_0 = FloatImm::make(4.89352518554385e-03f);
  ExprHandle beta_2 = FloatImm::make(2.26843463243900e-03f);
  ExprHandle beta_4 = FloatImm::make(1.18534705686654e-04f);
  ExprHandle beta_6 = FloatImm::make(1.19825839466702e-06f);

  // 计算分子
  ExprHandle v2 = v1 * v1;
  ExprHandle p = v2 * alpha_13 + alpha_11;
  p = v2 * p + alpha_9;
  p = v2 * p + alpha_7;
  p = v2 * p + alpha_5;
  p = v2 * p + alpha_3;
  p = v2 * p + alpha_1;
  p = v1 * p;

  // 计算分母
  ExprHandle q = v2 * beta_6 + beta_4;
  q = v2 * q + beta_2;
  q = v2 * q + beta_0;

  // 计算最终结果
  ExprHandle result = p / q;
  return result;
}
// 定义一个函数 fast_sigmoid，计算快速 sigmoid 函数的值
ExprHandle fast_sigmoid(const ExprHandle& x) {
  // sigmoid(x) = (tanh(x / 2) + 1) / 2
  // 创建常量表达式：1.0
  ExprHandle one_v = FloatImm::make(1.f);
  // 创建常量表达式：0.5
  ExprHandle half_v = FloatImm::make(0.5f);
  // 创建常量表达式：0.0
  ExprHandle zero_v = FloatImm::make(0.0f);
  // 将输入值 x 除以 2
  ExprHandle x2 = x * half_v;
  // 调用快速 tanh 函数 fast_tanh 计算 tanh(x / 2) 的值
  ExprHandle y{fast_tanh(x2)};
  // 计算 sigmoid(x) = (tanh(x / 2) + 1) / 2
  ExprHandle z = (y + one_v) * half_v;
  // 快速 tanh 函数的精度不高
  // 但是客户端依赖 sigmoid 返回概率值
  // 因此将其限制在 (0, 1) 范围内
  return Min::make(
      one_v,
      Max::make(zero_v, z, /* propagate_nans= */ false),
      /* propagate_nans= */ false);
}

// 定义一个函数 fast_log，计算快速对数函数的值
ExprHandle fast_log(const ExprHandle& v) {
  // 此实现取自 sleef：
  // https://github.com/shibatch/sleef/blob/master/src/libm/sleefsp.c#L1131
  // 生成系数使用的工具：
  // https://github.com/shibatch/sleef/blob/master/src/gencoef/gencoef.txt

  // 定义函数 ilogb2kf，计算 v 的指数部分
  auto ilogb2kf = [](ExprHandle x) {
    // 取整数表示的 v 的指数部分并调整范围
    auto y = (bitcast<int32_t>(x) >> IntImm::make(23)) & IntImm::make(0xff);
    return y - IntImm::make(0x7f);
  };

  // 定义函数 ldexp3kf，将 x 扩展 e 次幂
  auto ldexp3kf = [](ExprHandle x, ExprHandle e) {
    return bitcast<float>(bitcast<int32_t>(x) + (e << IntImm::make(23)));
  };

  // 计算 e 为 v 除以 0.75 的 ilogb2kf 结果
  auto e = ilogb2kf(v * FloatImm::make(1.0 / 0.75));
  // 计算 m 为 v 除以 2^e 的 ldexp3kf 结果
  auto m = ldexp3kf(v, IntImm::make(-1) * e);
  // 创建常量表达式：1.0
  auto one = FloatImm::make(1.0f);
  // 计算 x = (m - 1) / (m + 1)
  auto x = (m - one) / (m + one);
  // 计算 x 的平方
  auto x2 = x * x;

  // 定义 mlaf 函数，计算 x * y + z
  auto mlaf = [](ExprHandle x, ExprHandle y, float z) {
    return x * y + FloatImm::make(z);
  };

  // 计算 t 的多项式函数
  auto t = FloatImm::make(0.2392828464508056640625);
  t = mlaf(t, x2, 0.28518211841583251953125);
  t = mlaf(t, x2, 0.400005877017974853515625);
  t = mlaf(t, x2, 0.666666686534881591796875);
  t = mlaf(t, x2, 2.0);
  // 计算 x = x * t + 0.693147180559945286226764 * e
  x = x * t + FloatImm::make(0.693147180559945286226764) * e;

  // 创建常量表达式：0.0
  auto zero = FloatImm::make(0);
  // 创建常量表达式：NaN
  auto nan = FloatImm::make(std::numeric_limits<float>::quiet_NaN());
  // 创建常量表达式：负无穷
  auto neg_inf = FloatImm::make(-std::numeric_limits<float>::infinity());
  // 如果 v 小于等于 0，则返回 NaN
  x = CompareSelect::make(v, zero, nan, x, kLT);
  // 如果 v 等于 0，则返回负无穷
  x = CompareSelect::make(v, zero, neg_inf, x, kEQ);
  return x;
}

// 定义一个函数 log_vml，计算 VML（Vector Math Library）的对数
ExprHandle log_vml(const ExprHandle& v) {
  // 定义 mlaf 函数，计算 x * y + z
  auto mlaf = [](ExprHandle x, ExprHandle y, float z) {
    return x * y + FloatImm::make(z);
  };

  // 将 v 转换为整数表示
  auto in = bitcast<int32_t>(v);
  // 计算 a = in - 0x3f2aaaab
  auto a = in - IntImm::make(0x3f2aaaab);
  // 计算 e = a >> 23，并转换为浮点数
  auto e = cast<float>(a >> IntImm::make(23));

  // 计算 x = (a & 0x7fffff) + 0x3f2aaaab
  auto x = (a & IntImm::make(0x7fffff)) + IntImm::make(0x3f2aaaab);
  // 将 x 转换为浮点数，并减去 1.0
  x = bitcast<float>(x) - 1.0f;

  // 计算 t 的多项式函数
  auto t = FloatImm::make(-0.12891686f);
  t = mlaf(x, t, 0.139844373f);
  t = mlaf(x, t, -0.121842608f);
  t = mlaf(x, t, 0.140058696f);
  t = mlaf(x, t, -0.16680488f);
  t = mlaf(x, t, 0.200104058f);
  t = mlaf(x, t, -0.249997973f);
  t = mlaf(x, t, 0.333332151f);
  t = mlaf(x, t, -0.5f);
  t = x * t;
  // 计算 t = x * t + x
  t = x * t + x;

  // 计算 z = e * 1.42860677e-06 + t
  z = e * FloatImm::make(1.42860677e-06f) + t;
  // 计算 z = e * 0.693145752 + z
  z = e * FloatImm::make(0.693145752f) + z;

  // 如果 in + 0x800000 > 0x1000000，则返回 log(v)，否则返回 z
  return CompareSelect::make(
      IntImm::make(0x1000000),
      in + IntImm::make(0x800000),
      log(v),
      z,
      kGT,
      kUnlikely);
}
ExprHandle log(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示对数函数的表达式
  return Intrinsics::make(kLog, v);
}

ExprHandle log2(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示以2为底的对数函数的表达式
  return Intrinsics::make(kLog2, v);
}

ExprHandle log10(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示以10为底的对数函数的表达式
  return Intrinsics::make(kLog10, v);
}

ExprHandle log1p(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示 log(1 + v) 函数的表达式
  return Intrinsics::make(kLog1p, v);
}

ExprHandle erf(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示误差函数 erf(v) 的表达式
  return Intrinsics::make(kErf, v);
}

ExprHandle erfc(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示余误差函数 erfc(v) 的表达式
  return Intrinsics::make(kErfc, v);
}

ExprHandle sqrt(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示平方根函数 sqrt(v) 的表达式
  return Intrinsics::make(kSqrt, v);
}

ExprHandle rsqrt(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示倒数平方根函数 rsqrt(v) 的表达式
  return Intrinsics::make(kRsqrt, v);
}

ExprHandle ceil(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示向上取整函数 ceil(v) 的表达式
  return Intrinsics::make(kCeil, v);
}

ExprHandle floor(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示向下取整函数 floor(v) 的表达式
  return Intrinsics::make(kFloor, v);
}

ExprHandle round(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示四舍五入函数 round(v) 的表达式
  return Intrinsics::make(kRound, v);
}

ExprHandle trunc(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示截断函数 trunc(v) 的表达式
  return Intrinsics::make(kTrunc, v);
}

ExprHandle frac(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示小数部分函数 frac(v) 的表达式
  return Intrinsics::make(kFrac, v);
}

ExprHandle lgamma(const ExprHandle& v) {
  // 使用 Intrinsics::make 函数创建一个表示伽玛对数函数 lgamma(v) 的表达式
  return Intrinsics::make(kLgamma, v);
}

ExprHandle atan2(const ExprHandle& v1, const ExprHandle& v2) {
  // 使用 Intrinsics::make 函数创建一个表示反正切函数 atan2(v1, v2) 的表达式
  return Intrinsics::make(kAtan2, v1, v2);
}

ExprHandle pow(const ExprHandle& v1, const ExprHandle& v2) {
  // 使用 Intrinsics::make 函数创建一个表示幂函数 pow(v1, v2) 的表达式
  return Intrinsics::make(kPow, v1, v2);
}

ExprHandle fmod(const ExprHandle& v1, const ExprHandle& v2) {
  // 使用 Intrinsics::make 函数创建一个表示取余函数 fmod(v1, v2) 的表达式
  return Intrinsics::make(kFmod, v1, v2);
}

ExprHandle remainder(const ExprHandle& v1, const ExprHandle& v2) {
  // 使用 Intrinsics::make 函数创建一个表示求余函数 remainder(v1, v2) 的表达式
  return Intrinsics::make(kRemainder, v1, v2);
}

ExprHandle isnan(const ExprHandle& v1) {
  // 使用 Intrinsics::make 函数创建一个表示判断是否为 NaN 的表达式
  return Intrinsics::make(kIsNan, v1);
}

ExprHandle ifThenElse(
    const ExprHandle& c,
    const ExprHandle& t,
    const ExprHandle& f) {
  // 使用 IfThenElse::make 函数创建一个表示条件表达式的表达式
  return IfThenElse::make(c, t, f);
}

std::vector<ExprPtr> make_contiguous_strides(
    const std::vector<ExprHandle>& dims) {
  std::vector<ExprPtr> strides;

  if (!dims.empty()) {
    strides.resize(dims.size());
    auto si = immLike(dims[0], 1);
    // NOLINTNEXTLINE
    for (int i = dims.size() - 1; i >= 0; --i) {
      // NOLINTNEXTLINE
      // 创建并存储每个维度的步长表达式，si 初始为1
      strides[i] = si;
      // 更新 si，si 乘以当前维度的表达式 dims[i]
      si = alloc<Mul>(si, dims[i].node());
    }
  }
  return strides;
}

std::vector<ExprPtr> make_channels_last_strides(
    const std::vector<ExprHandle>& dims) {
  std::vector<ExprPtr> strides;
  TORCH_INTERNAL_ASSERT(
      dims.size() == 4 || dims.size() == 3, "got size:", dims.size());
  if (dims.size() == 4) {
    strides.resize(dims.size());
    ExprHandle handle = ExprHandle(immLike(dims[0], 1));
    // dims:               n   c    h  w
    // strides(nhwc):  w*c*h   1  w*c  c
    // 计算 channels_last 布局的步长
    strides[1] = handle.node();
    handle = handle * dims[1];
    strides[3] = handle.node();
    handle = handle * dims[3];
    strides[2] = handle.node();
    handle = handle * dims[2];
    strides[0] = handle.node();
  }
  if (dims.size() == 3) {
    strides.resize(dims.size());
    ExprHandle handle = ExprHandle(immLike(dims[0], 1));
    // dims:              n   c    l
    // 计算 3D 数据的 channels_last 布局的步长
    strides[2] = handle.node();
    handle = handle * dims[2];
    strides[1] = handle.node();
    handle = handle * dims[1];
    strides[0] = handle.node();
  }
  return strides;
}
    // 计算数组的步幅（strides）数组
    // 计算 strides 数组的第一个元素，其计算方式为 handle.node()
    strides[1] = handle.node();
    // 将 handle 乘以 dims[1]，更新 handle 的值
    handle = handle * dims[1];
    // 计算 strides 数组的第二个元素，其计算方式为 handle.node()
    strides[2] = handle.node();
    // 将 handle 乘以 dims[2]，更新 handle 的值
    handle = handle * dims[2];
    // 计算 strides 数组的第三个元素，其计算方式为 handle.node()
    strides[0] = handle.node();
  }
  // 返回计算得到的 strides 数组
  return strides;
}

Buf::Buf(
    VarPtr var,                                // 构造函数，接受变量指针
    std::vector<ExprPtr> dims,                 // 构造函数，接受维度表达式的向量
    Dtype dtype,                               // 构造函数，接受数据类型
    ExprPtr initializer,                       // 构造函数，接受初始化表达式
    std::optional<std::vector<ExprPtr>> strides, // 构造函数，接受可选的步长表达式向量
    ExprPtr qscale,                            // 构造函数，接受量化比例表达式
    ExprPtr qzero)                             // 构造函数，接受量化零点表达式
    : ExprNodeBase(dtype, kPrimitive),         // 调用基类 ExprNodeBase 的构造函数
      base_handle_(var),                      // 初始化成员变量 base_handle_ 为 var
      dims_(std::move(dims)),                 // 初始化成员变量 dims_ 为 dims
      strides_(                                 // 初始化成员变量 strides_
          strides                                // 如果 strides 存在则使用它
              ? *strides                         // 否则调用 make_contiguous_strides 函数
              : make_contiguous_strides(ExprVectorToExprHandleVector(dims_))),
      initializer_(std::move(initializer)),   // 初始化成员变量 initializer_ 为 initializer
      qscale_(std::move(qscale)),             // 初始化成员变量 qscale_ 为 qscale
      qzero_(std::move(qzero)) {              // 初始化成员变量 qzero_ 为 qzero
  TORCH_CHECK(var);                           // 断言，确保 var 不为空
}

BufHandle Buf::make(const std::vector<ExprHandle>& dims, Dtype dtype) {
  return Buf::make("", dims, dtype);          // 调用重载函数 make，传递空字符串作为 name_hint
}

BufHandle Buf::make(
    const std::string& name_hint,             // 创建缓冲区时使用的名称提示
    const std::vector<ExprHandle>& dims,      // 维度表达式向量
    const std::vector<ExprHandle>& strides,   // 步长表达式向量
    Dtype dtype) {                            // 数据类型
  return BufHandle(alloc<Buf>(                // 返回 BufHandle，分配新的 Buf 对象
      name_hint,                              // 传递名称提示
      ExprHandleVectorToExprVector(dims),     // 转换维度表达式向量到表达式指针向量
      dtype,                                  // 传递数据类型
      nullptr,                                // 初始化为 nullptr
      ExprHandleVectorToExprVector(strides))); // 转换步长表达式向量到表达式指针向量
}

BufHandle Buf::make(
    const std::string& name_hint,             // 创建缓冲区时使用的名称提示
    const std::vector<ExprHandle>& dims,      // 维度表达式向量
    Dtype dtype,                              // 数据类型
    std::optional<ExprHandle> initializer,    // 可选的初始化表达式
    std::optional<std::vector<ExprHandle>> strides, // 可选的步长表达式向量
    std::optional<ExprHandle> qscale,         // 可选的量化比例表达式
    std::optional<ExprHandle> qzero) {        // 可选的量化零点表达式
  std::optional<std::vector<ExprPtr>> opt_strides; // 声明可选的步长表达式向量
  if (strides) {                              // 如果 strides 存在
    opt_strides = ExprHandleVectorToExprVector(*strides); // 转换为表达式指针向量
  }
  return BufHandle(alloc<Buf>(                // 返回 BufHandle，分配新的 Buf 对象
      name_hint,                              // 传递名称提示
      ExprHandleVectorToExprVector(dims),     // 转换维度表达式向量到表达式指针向量
      dtype,                                  // 传递数据类型
      initializer ? initializer->node() : nullptr, // 如果 initializer 存在则传递其节点，否则传递 nullptr
      opt_strides,                            // 传递可选的步长表达式向量
      qscale ? qscale->node() : nullptr,      // 如果 qscale 存在则传递其节点，否则传递 nullptr
      qzero ? qzero->node() : nullptr));      // 如果 qzero 存在则传递其节点，否则传递 nullptr
}

bool Buf::is_contiguous(at::MemoryFormat memory_format) const {
  auto ndims = dims_.size();                  // 获取维度数量
  std::vector<int64_t> dim_order(ndims);      // 声明维度顺序向量
  if (memory_format == at::MemoryFormat::ChannelsLast) { // 如果内存格式为 ChannelsLast
    if (dims_.size() != 4)                    // 如果维度数量不为 4
      return false;                           // 返回 false
    dim_order = {1, 3, 2, 0};                 // 设置维度顺序为特定顺序
  } else if (memory_format == at::MemoryFormat::ChannelsLast3d) { // 如果内存格式为 ChannelsLast3d
    if (dims_.size() != 5)                    // 如果维度数量不为 5
      return false;                           // 返回 false
    dim_order = {1, 4, 3, 2, 0};              // 设置维度顺序为特定顺序
  } else {                                    // 否则
    if (dims_.empty()) {                      // 如果维度为空（标量张量）
      TORCH_CHECK(strides_.empty());          // 断言，确保步长为空
      return true;                            // 返回 true，与 kernel.cpp 中的 isContiguous 逻辑对齐
    }
    for (size_t i = 0; i < ndims; i++) {      // 遍历维度
      dim_order[i] = ndims - i - 1;           // 设置维度顺序为反向顺序
    }
  }

  bool res = is_stride_one(dim_order[0]);      // 检查第一个维度是否为单位步长
  if (!res)                                   // 如果不是单位步长
    return false;                             // 返回 false

  for (size_t i = 1; i < ndims; i++) {         // 遍历剩余维度
    auto cur_dim = dim_order[i];              // 获取当前维度
    auto pre_dim = dim_order[i - 1];          // 获取前一维度
    res &= is_cont_with(cur_dim, pre_dim);    // 检查当前维度是否与前一维度连续
    if (!res)                                 // 如果不连续
      return false;                           // 返回 false
  }

  return true;                                // 默认情况下返回 true
}

std::vector<ExprHandle> BufHandle::dims() const {
  return ExprVectorToExprHandleVector(node()->dims()); // 获取节点的维度表达式向量并返回
}

bool Buf::is_cont_with(int cur_dim, int adjacent_dim) const {
  auto is_cont_fn = [](ExprPtr adjacent_dim,
                       ExprPtr adjacent_stride,
                       ExprPtr cur_stride) {
    // For static shape
    // 检查当前步长是否等于邻接维度的维度乘以邻接维度的步长
    bool res = exprEquals(
        cur_stride,
        (ExprHandle(adjacent_dim) * ExprHandle(adjacent_stride)).node());
    // 如果相等，则返回 true
    if (res)
      return res;

    // 对于符号形状（symbolic shape）

    // 尝试将当前步长表示为乘法节点
    auto mul_node = to<Mul>(cur_stride);
    // 如果不能转换为乘法节点，则返回 false
    if (!mul_node) {
      return false;
    }

    // 获取乘法节点的左右子表达式
    auto lhs_ = mul_node->lhs();
    auto rhs_ = mul_node->rhs();

    bool same_stride = false;
    // 检查左子表达式是否与邻接维度相等，或者邻接维度是否等于左子表达式
    auto same_dim = exprEquals(lhs_, adjacent_dim) || (adjacent_dim == lhs_);
    if (same_dim) {
      // 如果左子表达式是维度，右子表达式是步长
      same_stride =
          exprEquals(rhs_, adjacent_stride) || (adjacent_stride == rhs_);
    } else {
      // 如果左子表达式是步长，右子表达式是维度
      same_dim = exprEquals(rhs_, adjacent_dim) || (adjacent_dim == rhs_);
      same_stride =
          exprEquals(lhs_, adjacent_stride) || (adjacent_stride == lhs_);
    }

    // 返回维度和步长都相等的结果
    return same_dim && same_stride;
  };
  // 调用 is_cont_fn 函数检查邻接维度的相关参数是否连续
  return is_cont_fn(
      dims_[adjacent_dim], strides_[adjacent_dim], strides_[cur_dim]);
}

// 判断当前维度的步长是否为1
bool Buf::is_stride_one(int cur_dim) const {
  // 检查步长数组中的当前维度是否等于1
  return exprEquals(strides_[cur_dim], alloc<LongImm>(1));
}

// 将标量表达式转换为向量表达式
ExprHandle expr_to_vec(ExprHandle v, int lanes) {
  // 如果向量宽度为1，则直接返回原始表达式
  if (lanes == 1) {
    return v;
  } else {
    // 否则将表达式广播为指定宽度的向量
    return Broadcast::make(v, lanes);
  }
}

// 命名空间结束标记，结束 torch::jit::tensorexpr 命名空间
} // namespace torch::jit::tensorexpr
```